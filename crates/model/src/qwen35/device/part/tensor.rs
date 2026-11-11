//! The step of a tensor split.
//!
//! Every device holds a slice of each half of a block (the mixer, the
//! feed-forward part) and computes a partial result; the residual stream of
//! every device needs the sum of all partials before the next half can start.
//! A step therefore is a chain of *merges*: one kernel per device that adds
//! the partials to the residual stream and normalises it for the next half
//! (`kernels/layer_merge.cuh`), with the half's own kernels in between.
//!
//! Partials cross the link as f16 (as f32 in the exact mode, which is a
//! reference: [`Options::wire`]), in one of two ways:
//!
//! - **A few rows** (decode, draft windows) are pushed: the merge kernel
//!   writes its rows straight into the other device's memory and waits for
//!   the other's. One launch per half, nothing else between the devices.
//! - **Many rows** (a prompt) go through the slots of the mesh: the partial
//!   is posted, the copy engine carries it, and the merge waits for the
//!   other devices' parcels. A long step is cut into groups of rows, and the
//!   groups take each half one after the other. While a group's partial is on
//!   the link, the device computes the next group, so the link and the
//!   processors work at the same time; the stream only waits when the other
//!   device is behind.
//!
//! The order within a half is: merge of the group (what the half before left
//! of it), the half's kernels for the group, post of its partial. The groups
//! pass a half in row order, which is the order the recurrent state and the
//! K/V rows of a sequence need.

use ares_cuda::{DevicePtr, MergeNorm, Precision, Wire, partial_wire_bytes};
use ares_runtime::ModelError;

use super::{Exchange, Group, Normed, Options, Part, Phase, Prepared, StepEnv, invalid};
use crate::qwen35::device::load::{BlockW, Mat, MixerW};

/// A half of a block: what a device computes between two merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Half {
    Mixer,
    Ffn,
}

/// The norm weight of a half and the matrices that read the normalised rows
/// (a half with fewer than four names one twice).
type Readers<'b> = (DevicePtr, [&'b Mat; 4]);

impl Options {
    /// How partial results cross between the devices of a tensor split: as
    /// f16 where the products round anyway, exactly where they are exact.
    /// Measured on Qwen3.5-0.8B against the CPU reference, f16 moves the
    /// worst logit of a pick from 3.4e-3 to 5.5e-3 with exact products and
    /// is not visible next to the 1e-2 to 4e-2 of 8-bit activations.
    pub fn wire(&self) -> Wire {
        match self.precision {
            Precision::Fast => Wire::F16,
            Precision::Exact => Wire::F32,
        }
    }
}

fn paths<'a>(env: &StepEnv<'a>) -> Result<&'a dyn Exchange, ModelError> {
    env.exchange.ok_or_else(|| invalid("a tensor split needs the paths between devices"))
}

impl Part {
    /// Enqueues the blocks of a step under a tensor split, from the
    /// embedding to the residual stream the head reads.
    pub(super) fn enqueue_tensor(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let groups = std::mem::take(&mut self.groups);
        let result = self.tensor_step(env, &groups);
        (self.group, self.row_base, self.normed) = (0, 0, Normed::No);
        self.groups = groups;
        result
    }

    /// The step as group `g` of `groups` sees it (the whole step when it is
    /// not cut), and that group made the one being enqueued.
    fn enter<'a>(&mut self, env: &'a StepEnv<'a>, groups: &'a [Group], g: usize) -> StepEnv<'a> {
        let Some(group) = groups.get(g) else {
            (self.group, self.row_base) = (0, 0);
            return StepEnv {
                stream: env.stream,
                step: env.step,
                dims: env.dims,
                segments: env.segments,
                tables: env.tables,
                pages: env.pages,
                exchange: env.exchange,
                replay: env.replay,
            };
        };
        (self.group, self.row_base) = (g, group.first_row);
        StepEnv {
            stream: env.stream,
            step: self.group_records.ptr().offset(group.record),
            dims: group.dims,
            segments: &group.segments,
            tables: env.tables,
            pages: env.pages,
            exchange: env.exchange,
            replay: env.replay,
        }
    }

    fn tensor_step(&mut self, env: &StepEnv<'_>, groups: &[Group]) -> Result<(), ModelError> {
        let (stream, dims) = (env.stream, env.dims);
        let bytes = dims.n_rows as usize * self.d as usize * 4;
        let (x, p) = (self.at(self.s.x), self.at(self.s.p));
        // The residual stream starts as the sum of the embedding, which one
        // device holds, and nothing from the others.
        // SAFETY: `x` and `p` hold the step's rows; the table is in the arena.
        unsafe {
            stream.fill(x, 0, bytes)?;
            match &self.w.embedding {
                Some(table) => self.layers.embed_gather(
                    stream,
                    env.step,
                    dims,
                    self.w.mat(table),
                    table.ty,
                    table.n_out,
                    table.n_in,
                    p,
                )?,
                None => stream.fill(p, 0, bytes)?,
            }
        }
        self.mark(stream, Phase::Layers)?;
        self.enqueue_rope(env)?;

        let count = groups.len().max(1);
        for g in 0..count {
            let group = self.enter(env, groups, g);
            self.post(&group, g as u32)?;
        }
        for i in 0..self.w.blocks.len() {
            let block: BlockW = self.w.blocks[i].clone();
            for half in [Half::Mixer, Half::Ffn] {
                let readers = self.readers(&block, half);
                for g in 0..count {
                    let group = self.enter(env, groups, g);
                    self.collect(&group, g as u32, readers)?;
                    self.half(&group, i, &block, half)?;
                    self.post(&group, g as u32)?;
                }
            }
        }
        for g in 0..count {
            let group = self.enter(env, groups, g);
            self.collect(&group, g as u32, None)?;
        }
        Ok(())
    }

    /// Who reads the normalised residual stream in `half` of `block`, or
    /// `None` when this device holds nothing of that half.
    fn readers<'b>(&self, block: &'b BlockW, half: Half) -> Option<Readers<'b>> {
        match half {
            Half::Mixer => match &block.mixer {
                MixerW::Recurrent(r) => {
                    Some((self.w.at(block.attn_norm), [&r.qkv, &r.z, &r.beta, &r.alpha]))
                }
                MixerW::Attention(a) => {
                    Some((self.w.at(block.attn_norm), [&a.q, &a.k, &a.v, &a.v]))
                }
                MixerW::Empty => None,
            },
            Half::Ffn => block
                .ffn
                .as_ref()
                .map(|f| (self.w.at(block.post_norm), [&f.gate, &f.up, &f.up, &f.up])),
        }
    }

    /// This device's partial result of `half` of block `i` for the rows of
    /// the group, into `p`.
    fn half(
        &mut self,
        env: &StepEnv<'_>,
        i: usize,
        block: &BlockW,
        half: Half,
    ) -> Result<(), ModelError> {
        match half {
            Half::Mixer => match &block.mixer {
                MixerW::Recurrent(r) => {
                    let at = self.recurrent_at[i].ok_or_else(|| invalid("recurrent state"))?;
                    self.recurrent(env, block, r, at)
                }
                MixerW::Attention(a) => {
                    let at = self.attention_at[i].ok_or_else(|| invalid("attention pool"))?;
                    self.attend(env, block, a, at)
                }
                MixerW::Empty => self.zero_partial(env),
            },
            Half::Ffn => match &block.ffn {
                Some(ffn) => self.ffn(env, block, ffn),
                None => self.zero_partial(env),
            },
        }
    }

    /// Whether a sum of `n` rows is pushed by its merge kernel.
    fn pushes(&self, exchange: &dyn Exchange, n: u32) -> bool {
        n <= self.push_rows
            && partial_wire_bytes(self.opts.wire(), n, self.d) <= exchange.push_bytes()
    }

    /// Sends the partial result `p` of the group's rows on its way to the
    /// other devices. The stream does not wait for it.
    fn post(&mut self, env: &StepEnv<'_>, slot: u32) -> Result<(), ModelError> {
        let n = env.dims.n_rows;
        let exchange = paths(env)?;
        if self.pushes(exchange, n) {
            // The merge kernel sends the rows itself.
            return Ok(());
        }
        let wire = exchange.outbox(slot)?;
        // SAFETY: `p` holds the group's rows and the outbox the wire form of
        // the most rows of a group (`Qwen35::links`).
        unsafe {
            self.layers.partial_to_wire(
                env.stream,
                self.opts.wire(),
                self.rows(self.s.p, self.d),
                wire,
                n,
                self.d,
            )?;
        }
        exchange.post(slot, partial_wire_bytes(self.opts.wire(), n, self.d))?;
        if self.parcels.len() <= slot as usize {
            self.parcels.resize(slot as usize + 1, DevicePtr::NULL);
        }
        self.parcels[slot as usize] = wire;
        self.mark(env.stream, Phase::Exchange)
    }

    /// Adds the sum of all devices' partial results of the group's rows to
    /// the residual stream and, for the half that `readers` belong to,
    /// normalises it: as the activations the half's products read (quantized,
    /// f16 or both), or as rows where they are prepared launch by launch.
    fn collect(
        &mut self,
        env: &StepEnv<'_>,
        slot: u32,
        readers: Option<Readers<'_>>,
    ) -> Result<(), ModelError> {
        let (stream, n) = (env.stream, env.dims.n_rows);
        let exchange = paths(env)?;
        let (x, p) = (self.rows(self.s.x, self.d), self.rows(self.s.p, self.d));
        let (norm, normed) = match readers {
            None => (MergeNorm::None, Normed::No),
            Some((weight, mats)) => {
                let forms = self.forms(&mats, n)?;
                let (act_q, act_h) = (self.at(self.s.act_q), self.at(self.s.act_h));
                let prepared = Prepared { int8: forms.int8, f16: forms.f16 };
                match (forms.batched, forms.int8, forms.f16) {
                    // The decode kernels in several launches quantize the rows
                    // launch by launch (`Part::prepare`), from the rows.
                    (true, ..) | (false, false, false) => {
                        let out = self.rows(self.s.xn, self.d);
                        (MergeNorm::Rows { weight, out }, Normed::Rows)
                    }
                    (false, true, false) => {
                        (MergeNorm::Quantized { weight, act: act_q }, Normed::Prepared(prepared))
                    }
                    (false, false, true) => {
                        (MergeNorm::F16 { weight, act: act_h }, Normed::Prepared(prepared))
                    }
                    (false, true, true) => (
                        MergeNorm::Both { weight, quantized: act_q, f16: act_h },
                        Normed::Prepared(prepared),
                    ),
                }
            }
        };
        if self.pushes(exchange, n) {
            let push = exchange.push()?;
            // SAFETY: `p` and `x` hold the group's rows; the norm's buffers
            // are the scratch of the step, sized for its most rows; the other
            // device enqueues the same merge (the parts run the same step).
            unsafe {
                self.layers.merge_push(
                    stream,
                    self.opts.wire(),
                    p,
                    x,
                    push,
                    self.index == 0,
                    n,
                    self.d,
                    norm,
                    self.eps,
                )?;
            }
        } else {
            let own = self
                .parcels
                .get(slot as usize)
                .copied()
                .filter(|p| *p != DevicePtr::NULL)
                .ok_or_else(|| invalid("a merge without a posted partial result"))?;
            let peers = exchange.arrive(slot)?;
            self.mark(stream, Phase::Wait)?;
            // SAFETY: `x` holds the group's rows; the parcels are the slot's
            // regions of the mesh, which hold the wire form of the group
            // (posted by every device with the same row count); the norm's
            // buffers as above.
            unsafe {
                self.layers.merge_wire(
                    stream,
                    self.opts.wire(),
                    x,
                    own,
                    peers,
                    exchange.slot_bytes(),
                    self.n_devices as u32 - 1,
                    self.index as u32,
                    n,
                    self.d,
                    norm,
                    self.eps,
                )?;
            }
            exchange.consumed(slot)?;
        }
        self.normed = normed;
        self.mark(stream, Phase::Exchange)?;
        self.mark_merge();
        Ok(())
    }
}
