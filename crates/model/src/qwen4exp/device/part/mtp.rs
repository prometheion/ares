//! A part that is the model's MTP block (section 4 of the specification):
//! one layer of the attention kind with a mixture of experts of its own, as a
//! model of one layer whose residual comes from elsewhere and whose output
//! head is the model's.
//!
//! Such a part runs steps of its own, of up to [`MAX_BATCH`] rows, next to the
//! steps of the part that holds the model's head, on the same device and
//! stream. Besides what every part does it has:
//!
//! - **The input.** A row starts from a residual (the trunk's after its last
//!   layer, or the block's own of the link before) and the embedding of the
//!   token it is paired with ([`Part::mtp_input`]):
//!
//!   ```text
//!   e     = fc_embedding (rms(embedding) * norm_embedding)
//!   hn[c] = rms(residual[c]) * norm_hidden[c]
//!   R[c]  = fc_hidden hn[c] + e
//!   ```
//!
//!   The host decodes the embedding, as it does for the model's first layer,
//!   and stages it once per stream, so that the sum is one launch.
//! - **Rows that only write.** A cell of the block the model has passed gets
//!   its keys and values and the indexer's key and reads nothing
//!   ([`Part::enqueue_keys`]): the first half of the layer up to the write,
//!   without the queries, the attention and the mixture of experts.
//! - **A state that can be put back.** The links of a chain behind the first
//!   are provisional: the indexer's carried keys of a sequence are saved before
//!   them and restored behind them ([`Part::save_slot`], [`Part::restore_slot`]);
//!   keys and values of their cells are overwritten when the model has passed.

use ares_core::Ledger;
use ares_cuda::{Device, DeviceBuffer, DevicePtr, Kernels, MAX_BATCH, PinnedBuffer, Stream};
use ares_runtime::ModelError;

use super::{Frame, MTP_KV, Options, Part, PartShape, Role, StepEnv, invalid};
use crate::qwen4exp::Config;
use crate::qwen4exp::device::load::{CoreW, DeviceWeights, Mat};
use crate::qwen4exp::weights::MtpWeights;

/// The buffers of an MTP block's input, for [`MAX_BATCH`] rows.
pub(super) struct State {
    /// The embeddings of the paired tokens, once per stream, as the host
    /// stages them.
    emb_host: PinnedBuffer,
    scratch: DeviceBuffer,
    /// Offsets in the scratch: the staged embeddings (`rows * D` f32), the
    /// normed values (`rows * D`), the projected embeddings (`rows * D`), the
    /// input of the block (`rows * D`), and quantized activations of
    /// [`MAX_BATCH`] rows of `N` values.
    emb: u64,
    normed: u64,
    e: u64,
    input: u64,
    act: u64,
    /// The state of every sequence slot as it was saved.
    saved: DeviceBuffer,
}

impl Part {
    /// Builds the part of an MTP block: `c` is the block as a model of one
    /// layer (`reference::mtp_config`), `w` its weights, and `output` the
    /// model's output head on `device` with its description
    /// ([`Part::output`] of the part that holds it, which must outlive this
    /// one). `opts.max_rows` must be at most [`MAX_BATCH`]: the block runs the
    /// decode kernels.
    pub(crate) fn open_mtp(
        c: &Config,
        w: &MtpWeights<'_>,
        opts: &Options,
        index: usize,
        output: (Mat, DevicePtr),
        device: &Device,
        ledger: &mut Ledger,
    ) -> Result<Self, ModelError> {
        if c.n_layer != 1 || !c.is_attention(0) || c.ngram.is_some() {
            return Err(invalid("an MTP block is one attention layer without the n-gram block"));
        }
        if opts.max_rows > MAX_BATCH {
            return Err(invalid(format!(
                "an MTP block runs steps of up to {MAX_BATCH} rows, not {}",
                opts.max_rows
            )));
        }
        // The block's K and V are in 4 bits, whatever the model's.
        let opts = &Options { kv: MTP_KV, prune: None, ..opts.clone() };
        let shape = PartShape::whole(c, 0..1, true);
        let f = Frame::new(c, opts, &shape, device.info().multiprocessors)?;
        let weights = DeviceWeights::load_block(device, ledger, c, w, output)?;
        let mut part = Self::assemble(
            c,
            opts,
            index,
            Role::Whole,
            true,
            f,
            weights,
            device,
            ledger,
            false,
            "qwen4exp MTP block",
        )?;
        let rows = u64::from(MAX_BATCH);
        let row = c.wide() as u64 * 4;
        let (emb, normed, e, input) = (0, rows * row, 2 * rows * row, 3 * rows * row);
        let act = 4 * rows * row;
        let len = act + Kernels::quantized_activation_bytes(c.n_embd as u32, MAX_BATCH);
        let saved = part.f.slot_bytes * u64::from(opts.max_slots);
        ledger.reserve("qwen4exp MTP block input", len + saved)?;
        part.mtp = Some(State {
            emb_host: PinnedBuffer::new(device, (rows * row) as usize)?,
            scratch: DeviceBuffer::new(device, len)?,
            emb,
            normed,
            e,
            input,
            act,
            saved: DeviceBuffer::new(device, saved)?,
        });
        Ok(part)
    }

    /// The model's output head on this part's device and its description, for
    /// an MTP block on the same device; `None` for a part without the head.
    pub(crate) fn output(&self) -> Option<(Mat, DevicePtr)> {
        if self.role == Role::Worker { None } else { self.w.output() }
    }

    /// Enqueues the input of an MTP block for `rows` rows and returns where
    /// it is (`rows * D` f32, valid until the next call): what
    /// [`StepEnv::inbound`] of the block's step names. `residual` addresses
    /// the rows' residuals on this device (`rows * D` f32) and `embeddings`
    /// holds the embeddings of the paired tokens (`rows * N` values).
    ///
    /// # Safety
    ///
    /// `residual` must address `rows * D` f32 values on this part's device,
    /// 16-byte aligned, that stay as they are until the launches have run.
    /// The device must have run the block's last step: its host buffers are
    /// written again here.
    pub unsafe fn mtp_input(
        &mut self,
        stream: &Stream,
        residual: DevicePtr,
        embeddings: &[f32],
        rows: u32,
    ) -> Result<DevicePtr, ModelError> {
        let (n, hc, wide, eps) = (self.n, self.hc, self.wide, self.eps);
        let input = self.w.mtp_in.ok_or_else(|| invalid("the part is no MTP block"))?;
        let block = self.mtp.as_mut().ok_or_else(|| invalid("the part is no MTP block"))?;
        if rows == 0 || rows > MAX_BATCH || embeddings.len() != (rows * n) as usize {
            return Err(invalid(format!(
                "the input of an MTP block for {rows} rows from {} embedding values",
                embeddings.len()
            )));
        }
        let at = |offset: u64| block.scratch.ptr().offset(offset);
        let values = (rows * wide) as usize;
        // SAFETY: the pinned buffer and every section of the scratch hold
        // `MAX_BATCH` rows of `D` values; the copies of the step before have
        // run (the caller's guarantee).
        unsafe {
            let staged = block.emb_host.as_mut_ptr().cast::<f32>();
            for (row, x) in embeddings.chunks_exact(n as usize).enumerate() {
                for stream_no in 0..hc as usize {
                    let to = staged.add((row * hc as usize + stream_no) * n as usize);
                    std::ptr::copy_nonoverlapping(x.as_ptr(), to, n as usize);
                }
            }
            stream.copy_from_host(at(block.emb), block.emb_host.as_ptr(), values * 4)?;
            // e, once per stream.
            self.kernels.rms_norm_f32(
                stream,
                at(block.emb),
                self.w.at(input.norm_embedding),
                at(block.normed),
                n,
                rows * hc,
                eps,
            )?;
            let project = |m: &Mat, from: u64, to: u64| -> Result<(), ModelError> {
                let mut done = 0;
                while done < rows * hc {
                    let batch = (rows * hc - done).min(MAX_BATCH);
                    let x = at(from + u64::from(done * n) * 4);
                    self.kernels.quantize_activations(stream, x, at(block.act), n, batch)?;
                    self.kernels.matvec(
                        stream,
                        m.ty,
                        self.w.mat(m),
                        at(block.act),
                        at(to + u64::from(done * n) * 4),
                        m.n_in,
                        m.n_out,
                        batch,
                    )?;
                    done += batch;
                }
                Ok(())
            };
            project(&input.fc_embedding, block.normed, block.e)?;
            // The residual, every stream normed by itself, then each stream
            // through its projection.
            self.q4.stream_norm(
                stream,
                residual,
                self.w.at(input.norm_hidden),
                at(block.normed),
                rows,
                hc,
                n,
                eps,
            )?;
            project(&input.fc_hidden, block.normed, block.input)?;
            self.kernels.add_f32(stream, at(block.input), at(block.e), rows * wide)?;
        }
        Ok(at(block.input))
    }

    /// Enqueues the first half of attention layer `i` for the rows of the
    /// step up to their keys: the read of the layer's mixer, keys and values
    /// into the pages, the indexer's keys. The rows read nothing and the
    /// layer's mixture of experts does not run. Between [`Part::begin`] and
    /// [`Part::end_keys`].
    pub fn enqueue_keys(&mut self, env: &StepEnv<'_>, i: usize) -> Result<(), ModelError> {
        let layer =
            *self.w.layers.get(i).ok_or_else(|| {
                invalid(format!("layer {i} of a part of {}", self.w.layers.len()))
            })?;
        let (CoreW::Attention(a), Some((mixer, _))) = (&layer.core, layer.mixers) else {
            return Err(invalid("only an attention layer with its mixers writes keys alone"));
        };
        let rows = env.record.dims().n_rows;
        let s = self.f.s;
        let write = self.take_write();
        self.read(env.stream, &mixer, write, Some(self.at(s.inj_a)), self.at(s.m), None, rows)?;
        self.attend_rows(env, i, a, true)
    }

    /// Ends a step whose rows only wrote keys ([`Part::enqueue_keys`]): no
    /// residual is written back and no head runs. What writing the keys
    /// reported is read with the next step that runs the whole layer
    /// ([`Part::end`]): the host does not wait between steps of this kind.
    pub fn end_keys(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if env.segments.iter().any(|s| !s.direct) {
            return Err(invalid("rows that only write keys are certain"));
        }
        self.pending = None;
        Ok(())
    }

    /// Enqueues a copy of the state of sequence slot `slot` (what the part
    /// carries of a sequence besides its pages) to where
    /// [`Part::restore_slot`] finds it.
    pub fn save_slot(&self, stream: &Stream, slot: u32) -> Result<(), ModelError> {
        let (state, saved, len) = self.slot_copy(slot)?;
        // SAFETY: both ranges are one slot of buffers of `max_slots` slots.
        unsafe { stream.copy(saved, state, len)? };
        Ok(())
    }

    /// Enqueues putting back the state of `slot` as [`Part::save_slot`] left
    /// it.
    pub fn restore_slot(&self, stream: &Stream, slot: u32) -> Result<(), ModelError> {
        let (state, saved, len) = self.slot_copy(slot)?;
        // SAFETY: as in `save_slot`.
        unsafe { stream.copy(state, saved, len)? };
        Ok(())
    }

    fn slot_copy(&self, slot: u32) -> Result<(DevicePtr, DevicePtr, usize), ModelError> {
        let block = self.mtp.as_ref().ok_or_else(|| invalid("the part is no MTP block"))?;
        if slot >= self.opts.max_slots {
            return Err(invalid(format!("slot {slot} of {}", self.opts.max_slots)));
        }
        let at = u64::from(slot) * self.f.slot_bytes;
        Ok((self.state.ptr().offset(at), block.saved.ptr().offset(at), self.f.slot_bytes as usize))
    }
}
