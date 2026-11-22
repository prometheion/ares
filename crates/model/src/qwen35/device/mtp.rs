//! The 27B's MTP block on its device: the next-token-prediction block of
//! `qwen35` (multi-token prediction) as the layer of the shared MTP driver
//! (`blocks/mtp.rs`), which runs its chains, its picks, its carry and its
//! following of the model.
//!
//! What the block computes is in `reference/mtp.rs`:
//!
//! ```text
//! x      = eh_proj [enorm(e); hnorm(h)]        e: the token's embedding, h: the hidden state before
//! x      = x + attention(attn_norm(x))          over the block's own keys and values
//! x      = x + ffn(post_norm(x))
//! out    = shared_head_norm(x)                  the next link's h
//! logits = draft head @ out                     the head view's rows of the output head
//! ```
//!
//! The model gives the driver its hidden state where it is tapped: the final
//! hidden state of every row of a step (`output_norm` of the residual stream
//! after the last block, in the step's `xn`), on the device that holds the
//! block. A pair of the token at position `p` and the hidden state of `p - 1`
//! is the block's cell `p`.
//!
//! The block is whole on one device, the one with the head view: the device
//! that embeds under a tensor split or alone, the last one under a layer
//! split, which holds the head and the final hidden states. Its head is the
//! head view's first rows of the output head there, so it takes no weights
//! twice, and the device keeps the rows of the embedding table for the tokens
//! the chain can pick (`blocks/rows.rs`). A link of the chain works in a
//! scratch of the block's own, sized for the sequences that can draft at once;
//! rows that only write keys use the step's scratch, which is free at the start
//! of a step (only the hidden states of the last step are kept there, in `xn`,
//! until they are followed).

use std::collections::VecDeque;
use std::sync::Arc;

use ares_core::Ledger;
use ares_cuda::{
    AttendBuffers, AttentionShape, AttnRow, DeviceBuffer, DevicePtr, ENTRY_CELLS, KeptRows,
    Kernels, LayerKernels, Plan, Planner, PrefillPath, Segment, Stream,
};
use ares_runtime::ModelError;

use super::super::load::{DeviceWeights, Mat, MixerW, MtpW};
use super::kv;
use super::{
    AttentionAt, MTP_KV, Options, Part, Phase, Prepared, StepEnv, TAILS_PER_SLOT, geometry, invalid,
};
pub use crate::blocks::mtp::{Drafted, MtpChain, MtpStep, MtpTimes};
use crate::blocks::mtp::{Driver, DriverSetup, LinkLogits, MtpLayer, Rows};
use crate::blocks::rows::HostTable;
use crate::qwen35::Config;

/// What the MTP block takes on its device, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MtpBytes {
    /// The block's weights (the draft head is rows of the output head and
    /// counts nothing).
    pub weights: u64,
    /// Its K and V per page of 128 cells.
    pub page: u64,
    /// Its exact K/V entries and the page table they need, when its formats
    /// have a tile mean.
    pub exact: u64,
    /// The hidden states kept from step to step, for all sequence slots.
    pub carry: u64,
    /// The chain's scratch, the sampler's included.
    pub scratch: u64,
    /// The records of a step: links, plans, rows.
    pub records: u64,
    /// What the step's own scratch grew by for the block.
    pub step_scratch: u64,
}

/// What the block needs to know of the part it is in.
#[derive(Clone, Copy, Debug)]
pub(super) struct Setup {
    pub max_slots: u32,
    pub pages: u32,
    pub table_stride: u32,
    /// The attention blocks keep exact K/V entries, so the step's page table
    /// names exact slots that the block's layer must not read.
    pub main_exact: bool,
}

/// What the block adds to the sizes a part plans its memory from.
#[derive(Clone, Copy, Debug)]
pub(super) struct MtpSizes {
    pub at: AttentionAt,
    /// Bytes of the block's pool.
    pub pool: u64,
    /// Values of all KV heads of a row: the step's K and V buffers hold them.
    pub kv_values: u64,
    /// Bytes of partial attention states a link's plan may need.
    pub attn_state: u64,
    /// Bytes of prepared activations the chain needs, quantized and f16.
    pub act_q: u64,
    pub act_h: u64,
}

/// The sizes for the MTP block of `w`, whose pool starts at byte `pool` of
/// the pages; `None` on a device without it.
pub(super) fn sizes(
    c: &Config,
    w: &DeviceWeights,
    max_slots: u32,
    pages: u32,
    target_blocks: u32,
    pool: u64,
) -> Option<MtpSizes> {
    let mtp = w.mtp.as_ref()?;
    let MixerW::Attention(a) = &mtp.block.mixer else { return None };
    let geo = geometry(c, MTP_KV, a.n_kv);
    let rows = max_slots;
    let widest = mtp.block.ffn.as_ref().map_or(0, |f| f.gate.n_out).max(a.out.n_in).max(a.q.n_in);
    let slots = 20 * u64::from(target_blocks) / u64::from(a.n_kv) + 64;
    Some(MtpSizes {
        at: AttentionAt { geo, pool, exact: 0 },
        pool: u64::from(pages) * geo.converted_entry_bytes(),
        kv_values: u64::from(a.n_kv) * u64::from(geo.head_dim),
        attn_state: slots * u64::from(a.n_head) * (u64::from(geo.head_dim) + 1) * 4,
        act_q: Kernels::quantized_activation_bytes(widest, rows),
        act_h: Kernels::prefill_activation_bytes(PrefillPath::F16, widest, rows),
    })
}

/// Bytes reserved per link and sequence for the attention plan of a link.
const PLAN_BYTES: u64 = 32 * 1024;

/// Bytes of an [`AttnRow`]: four words.
const ATTN_ROW_BYTES: u64 = 16;

/// Offsets of the scratch of a link, for the sequences that draft at once.
#[derive(Clone, Copy, Debug, Default)]
struct Scratch {
    x: u64,
    xn: u64,
    p: u64,
    qg: u64,
    k_raw: u64,
    v: u64,
    q: u64,
    gate: u64,
    k: u64,
    o: u64,
    g1: u64,
    g2: u64,
    rope: u64,
    len: u64,
}

/// The 27B's MTP layer: its weights, its K/V group and the scratch of a
/// link.
#[derive(Debug)]
pub(super) struct Layer {
    pub(super) at: AttentionAt,
    /// The K/V group of the block's layer: its exact pool and table.
    pub(super) kv: kv::Group,
    w: MtpW,
    scratch: DeviceBuffer,
    s: Scratch,
    /// The attention plans of the links laid out for the step, in their order.
    plans: VecDeque<Plan>,
    step_scratch: u64,
}

impl Layer {
    fn at(&self, offset: u64) -> DevicePtr {
        self.scratch.ptr().offset(offset)
    }

    /// Where the block's exact entries are: null when its formats have none.
    fn exact_pool(&self) -> DevicePtr {
        if self.kv.has_exact() { self.kv.exact().offset(self.at.exact) } else { DevicePtr::NULL }
    }
}

/// The driver and the layer of a part's MTP block.
#[derive(Debug)]
pub(super) struct Mtp {
    pub(super) driver: Driver,
    pub(super) layer: Layer,
}

impl Mtp {
    /// Reserves and allocates what the block needs besides its weights and
    /// its pool; `table` is the embedding table on the host, for the tokens
    /// whose rows the device does not keep.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open(
        c: &Config,
        device: &ares_cuda::Device,
        ledger: &mut Ledger,
        w: &DeviceWeights,
        opts: &Options,
        setup: Setup,
        sizes: Option<MtpSizes>,
        step_scratch: u64,
        table: Option<Arc<HostTable>>,
    ) -> Result<Option<Self>, ModelError> {
        let Setup { max_slots, pages, table_stride, main_exact } = setup;
        let (Some(sizes), Some(mtp)) = (sizes, w.mtp.as_ref()) else { return Ok(None) };
        let MixerW::Attention(a) = &mtp.block.mixer else {
            return Err(invalid("the MTP block is an attention block"));
        };
        let ffn = mtp.block.ffn.as_ref().ok_or_else(|| invalid("the MTP block has no ffn"))?;
        let Some((shape, kept)) = w.embedding.and_then(|e| Some((e.table, e.kept?))) else {
            return Err(invalid("the MTP block's device keeps no rows of the embedding table"));
        };
        if kept.rows.n_out < shape.n_vocab && table.is_none() {
            return Err(invalid("the MTP block's device has no embedding table on the host"));
        }
        let d = c.n_embd as u64;
        let rows = u64::from(max_slots);
        let (hq, hkv) = (u64::from(a.n_head) * u64::from(sizes.at.geo.head_dim), sizes.kv_values);
        let inner = u64::from(ffn.gate.n_out);
        let mut s = Scratch::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        s.x = take(rows * d * 4);
        s.xn = take(rows * d * 4);
        s.p = take(rows * d * 4);
        s.qg = take(rows * 2 * hq * 4);
        s.k_raw = take(rows * hkv * 4);
        s.v = take(rows * hkv * 4);
        s.q = take(rows * hq * 4);
        s.gate = take(rows * hq * 4);
        s.k = take(rows * hkv * 4);
        s.o = take(rows * hq * 4);
        s.g1 = take(rows * inner * 4);
        s.g2 = take(rows * inner * 4);
        s.rope = take(rows * (c.attention.rope.dim as u64 / 2) * 8);
        s.len = len;
        ledger.reserve("qwen35 MTP block: link scratch", s.len)?;
        let scratch = DeviceBuffer::new(device, s.len)?;
        let sizing =
            kv::Sizing { slots: max_slots, rows: opts.max_rows, tails: max_slots * TAILS_PER_SLOT };
        let exact_slots = kv::slots_for(MTP_KV.0, MTP_KV.1, sizing);
        let group = kv::Group::new(
            device,
            ledger,
            "qwen35 MTP block",
            vec![sizes.at],
            MTP_KV,
            exact_slots,
            pages,
            max_slots,
            table_stride,
            exact_slots > 0 || main_exact,
        )?;
        let kept_rows = KeptRows {
            rows: w.mat(&kept.rows),
            tokens: kept.tokens.map_or(DevicePtr::NULL, |at| w.at(at)),
            count: kept.rows.n_out,
        };
        let layer_bytes = |rows: u32, links: bool| {
            u64::from(rows) * if links { PLAN_BYTES } else { ATTN_ROW_BYTES }
        };
        let driver = Driver::open(
            device,
            ledger,
            DriverSetup {
                label: "qwen35 MTP block",
                width: c.n_embd as u32,
                max_slots,
                max_rows: opts.max_rows,
                window_rows: opts.window_rows,
                head_rows: mtp.head.n_out,
                head_tokens: w.head_tokens.map_or(DevicePtr::NULL, |at| w.at(at)),
                table: shape,
                kept: kept_rows,
                host: table,
            },
            opts.max_rows,
            true,
            layer_bytes,
        )?;
        Ok(Some(Self {
            driver,
            layer: Layer {
                at: sizes.at,
                kv: group,
                w: mtp.clone(),
                scratch,
                s,
                plans: VecDeque::new(),
                step_scratch,
            },
        }))
    }
}

/// The layer with the part around it, as the driver drives it.
struct PartLayer<'a> {
    part: &'a mut Part,
    layer: &'a mut Layer,
}

impl MtpLayer for PartLayer<'_> {
    fn lag(&self) -> u32 {
        0
    }

    fn keys_inputs(&self) -> Option<(DevicePtr, DevicePtr)> {
        Some((self.part.at(self.part.s.p), self.part.at(self.part.s.x)))
    }

    fn bytes(
        &mut self,
        segments: &[Segment],
        links: bool,
        out: &mut [u8],
    ) -> Result<usize, ModelError> {
        let (part, layer) = (&*self.part, &mut *self.layer);
        let full = || invalid("the MTP block's step records do not fit their buffer");
        if !links {
            // The rows' places in the pages, for the append of their keys and
            // values.
            let n: u32 = segments.iter().map(|s| s.n_rows).sum();
            let mut rows = vec![AttnRow::default(); n as usize];
            for s in segments {
                for i in 0..s.n_rows {
                    rows[(s.row0 + i) as usize] = AttnRow {
                        table: s.slot * part.table_stride,
                        pos: s.pos0 as u32 + i,
                        slot_base: 0,
                        n_slots: 0,
                    };
                }
            }
            let bytes = AttnRow::to_bytes(&rows);
            out.get_mut(..bytes.len()).ok_or_else(full)?.copy_from_slice(&bytes);
            return Ok(bytes.len());
        }
        // A link's attention plan: each row the only row of its step, over
        // the block's cells up to its own.
        let geo = layer.at.geo;
        let mut planner = Planner::new(geo, part.plan_config, segments.len() as u32);
        for (row, segment) in segments.iter().enumerate() {
            let table = segment.slot * part.table_stride;
            let need = (segment.pos0 as u32 + 1).div_ceil(ENTRY_CELLS) as usize;
            let entries = &layer.kv.entries(&part.page_tab)[table as usize..table as usize + need];
            planner.dense(table, entries, row as u32, 1, segment.pos0 as u32);
        }
        let plan = planner.finish();
        if plan.state_bytes(&geo) > part.s.attn_state_bytes {
            return Err(invalid("a link's attention plan needs more partial states"));
        }
        let bytes = plan.layout().bytes as usize;
        plan.write(out.get_mut(..bytes).ok_or_else(full)?);
        layer.plans.push_back(plan);
        Ok(bytes)
    }

    fn keys(&mut self, rows: &Rows<'_>) -> Result<(), ModelError> {
        let (part, layer) = (&mut *self.part, &*self.layer);
        let (stream, dims) = (rows.stream, rows.record.dims());
        let (n, d, eps) = (dims.n_rows, part.d, part.eps);
        let w = &layer.w;
        let MixerW::Attention(a) = &w.block.mixer else { return Err(invalid("MTP attention")) };
        let attention = part.attention.clone().ok_or_else(|| invalid("attention kernels"))?;
        let geo = layer.at.geo;
        let s = part.s;
        let (x, xn, p) = (part.at(s.x), part.at(s.xn), part.at(s.p));
        let (k_raw, k, v) = (part.at(s.w2), part.at(s.k), part.at(s.v));
        // Only the keys are wanted of what prepares attention's inputs: the
        // queries it computes next to them come from, and go to, memory
        // nothing reads.
        let shape = AttentionShape { n_head: a.n_kv, n_kv_head: a.n_kv, head_dim: geo.head_dim };
        // SAFETY: at the start of a step the step's scratch is free but for
        // the hidden states the driver gathered into `x` and the embeddings
        // in `p`; its buffers hold the step's rows of `d` values and of all
        // KV heads of the block (`Part::open` sized them for it). Weights,
        // tables and pool as in `Part::attend`.
        unsafe {
            part.kernels.rms_norm_f32(stream, p, part.w.at(w.enorm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&w.proj_e], Prepared::default())?;
            part.product(stream, &w.proj_e, n, p)?;
            part.kernels.rms_norm_f32(stream, x, part.w.at(w.hnorm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&w.proj_h], Prepared::default())?;
            part.product(stream, &w.proj_h, n, x)?;
            part.kernels.add_f32(stream, x, p, n * d)?;
            part.kernels.rms_norm_f32(stream, x, part.w.at(w.block.attn_norm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&a.k, &a.v], Prepared::default())?;
            part.product(stream, &a.k, n, k_raw)?;
            part.product(stream, &a.v, n, v)?;
            part.layers.rope_table(stream, rows.step, dims, part.at(s.rope), part.rope)?;
            part.layers.attn_prep(
                stream,
                rows.step,
                dims,
                shape,
                part.rope.dim,
                eps,
                part.at(s.w1),
                k_raw,
                part.w.at(a.q_norm),
                part.w.at(a.k_norm),
                part.at(s.rope),
                part.at(s.q),
                None,
                k,
            )?;
            layer.kv.append(
                &attention,
                stream,
                &layer.at,
                rows.bytes,
                k,
                v,
                rows.tables,
                rows.pages,
                part.at(s.status),
                rows.replay,
                0,
                n,
            )?;
        }
        Ok(())
    }

    fn link(
        &mut self,
        rows: &Rows<'_>,
        out: DevicePtr,
        logits: DevicePtr,
    ) -> Result<LinkLogits, ModelError> {
        let (part, layer) = (&mut *self.part, &mut *self.layer);
        let (stream, dims) = (rows.stream, rows.record.dims());
        let (n, d, eps) = (dims.n_rows, part.d, part.eps);
        let plan = layer.plans.pop_front().ok_or_else(|| invalid("a link without its plan"))?;
        let w = &layer.w;
        let MixerW::Attention(a) = &w.block.mixer else { return Err(invalid("MTP attention")) };
        let ffn = w.block.ffn.as_ref().ok_or_else(|| invalid("MTP ffn"))?;
        let attention = part.attention.clone().ok_or_else(|| invalid("attention kernels"))?;
        let geo = layer.at.geo;
        let s = layer.s;
        let (e, h) = (rows.e, rows.h);
        let (x, xn, p) = (layer.at(s.x), layer.at(s.xn), layer.at(s.p));
        let (tables, exact) = (layer.kv.table(rows.tables), layer.exact_pool());
        let pool = rows.pages.offset(layer.at.pool);
        let shape = AttentionShape { n_head: a.n_head, n_kv_head: a.n_kv, head_dim: geo.head_dim };
        let inner = ffn.gate.n_out;
        // SAFETY: the link's scratch holds `max_slots` rows of every buffer
        // and `n` is at most that; the records hold what `bytes` wrote for
        // this link; weights, tables and pool as in `Part::attend`.
        unsafe {
            let plan = plan.on_device(rows.bytes);
            // x = proj_e e' + proj_h h'
            part.kernels.rms_norm_f32(stream, e, part.w.at(w.enorm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&w.proj_e], Prepared::default())?;
            part.product(stream, &w.proj_e, n, x)?;
            part.kernels.rms_norm_f32(stream, h, part.w.at(w.hnorm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&w.proj_h], Prepared::default())?;
            part.product(stream, &w.proj_h, n, p)?;
            part.kernels.add_f32(stream, x, p, n * d)?;
            // Attention over the block's own keys and values.
            part.kernels.rms_norm_f32(stream, x, part.w.at(w.block.attn_norm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&a.q, &a.k, &a.v], Prepared::default())?;
            part.product(stream, &a.q, n, layer.at(s.qg))?;
            part.product(stream, &a.k, n, layer.at(s.k_raw))?;
            part.product(stream, &a.v, n, layer.at(s.v))?;
            part.layers.rope_table(stream, rows.step, dims, layer.at(s.rope), part.rope)?;
            part.layers.attn_prep(
                stream,
                rows.step,
                dims,
                shape,
                part.rope.dim,
                eps,
                layer.at(s.qg),
                layer.at(s.k_raw),
                part.w.at(a.q_norm),
                part.w.at(a.k_norm),
                layer.at(s.rope),
                layer.at(s.q),
                Some(layer.at(s.gate)),
                layer.at(s.k),
            )?;
            attention.append(
                stream,
                &geo,
                plan.rows(),
                n,
                layer.at(s.k),
                layer.at(s.v),
                tables,
                exact,
                pool,
                part.at(part.s.status),
            )?;
            attention.attend(
                stream,
                &geo,
                &plan,
                &AttendBuffers {
                    tables,
                    exact_pool: exact,
                    converted_pool: pool,
                    q: layer.at(s.q),
                    gate: Some(layer.at(s.gate)),
                    out: layer.at(s.o),
                    scratch: part.at(part.s.attn_state),
                },
                1.0 / (part.attn_head_dim as f32).sqrt(),
            )?;
            part.prepare(stream, layer.at(s.o), a.out.n_in, n, &[&a.out], Prepared::default())?;
            part.product(stream, &a.out, n, p)?;
            part.kernels.add_f32(stream, x, p, n * d)?;
            // The feed-forward part.
            part.kernels.rms_norm_f32(stream, x, part.w.at(w.block.post_norm), xn, d, n, eps)?;
            part.prepare(stream, xn, d, n, &[&ffn.gate, &ffn.up], Prepared::default())?;
            part.product(stream, &ffn.gate, n, layer.at(s.g1))?;
            part.product(stream, &ffn.up, n, layer.at(s.g2))?;
            part.swiglu(stream, layer.at(s.g1), layer.at(s.g2), inner, n, &ffn.down)?;
            part.product(stream, &ffn.down, n, p)?;
            part.kernels.add_f32(stream, x, p, n * d)?;
            // The output and the draft head.
            part.kernels.rms_norm_f32(stream, x, part.w.at(w.head_norm), out, d, n, eps)?;
            part.prepare(stream, out, d, n, &[&w.head], Prepared::default())?;
            part.product(stream, &w.head, n, logits)?;
        }
        Ok(LinkLogits { ptr: logits, stride: w.head.n_out })
    }

    fn forget_links(&mut self) {
        // The plans of links a draft bar ended were never taken by `link`.
        self.layer.plans.clear();
    }

    fn save(&mut self, _stream: &Stream, _slots: &[u32]) -> Result<(), ModelError> {
        // The block keeps no state per sequence besides its cells.
        Ok(())
    }

    fn restore(&mut self, _stream: &Stream, _slots: &[u32]) -> Result<(), ModelError> {
        Ok(())
    }

    fn writes_converted(&self, slot: u32, cell: u32) -> bool {
        self.layer.kv.writes_converted(slot, cell, &self.part.page_tab)
    }

    fn settle(&mut self, slot: u32, cells: u32) {
        self.layer.kv.settled(slot, cells, &self.part.page_tab);
    }
}

/// What a closure over the MTP block gets: the driver, the layer with the
/// part around it, and the part's layer kernels.
type WithMtp<'a, 'b> = (&'a mut Driver, &'a mut PartLayer<'b>, &'a LayerKernels);

impl Part {
    /// Runs `f` with the MTP driver and the layer with the part around it;
    /// `None` without the block.
    fn with_mtp<T>(
        &mut self,
        f: impl FnOnce(WithMtp<'_, '_>) -> Result<T, ModelError>,
    ) -> Result<Option<T>, ModelError> {
        let Some(mut m) = self.mtp.take() else { return Ok(None) };
        let layers = self.layers.clone();
        let result =
            f((&mut m.driver, &mut PartLayer { part: self, layer: &mut m.layer }, &layers));
        self.mtp = Some(m);
        result.map(Some)
    }

    /// Whether this part holds the model's MTP block.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// The most drafts the block makes per sequence in a step.
    pub fn mtp_links(&self) -> u32 {
        self.mtp.as_ref().map_or(0, |m| m.driver.max_links())
    }

    /// What the block takes on this device.
    pub fn mtp_bytes(&self) -> Option<MtpBytes> {
        let m = self.mtp.as_ref()?;
        let w = &m.layer.w;
        let MixerW::Attention(a) = &w.block.mixer else { return None };
        let ffn = w.block.ffn.as_ref()?;
        let weights =
            [&w.proj_e, &w.proj_h, &a.q, &a.k, &a.v, &a.out, &ffn.gate, &ffn.up, &ffn.down]
                .iter()
                .map(|x| Mat::bytes(x))
                .sum::<u64>()
                + 5 * u64::from(self.d) * 4;
        let (carry, scratch, records) = m.driver.bytes();
        Some(MtpBytes {
            weights,
            page: m.layer.at.geo.converted_entry_bytes(),
            exact: m.layer.kv.bytes(),
            carry,
            scratch: scratch + m.layer.scratch.len(),
            records,
            step_scratch: m.layer.step_scratch,
        })
    }

    /// Device time of the MTP block so far. Includes the last step only
    /// after the next one was enqueued or [`Part::draft_ms`] was called.
    pub fn mtp_times(&self) -> MtpTimes {
        self.mtp.as_ref().map_or_else(MtpTimes::default, |m| m.driver.times())
    }

    /// Device milliseconds of the drafters of the step that ran last: the
    /// MTP block's chains and the DFlash drafter's pass (0 when it drafted
    /// nothing). The step must have ended.
    pub fn draft_ms(&mut self) -> Result<f64, ModelError> {
        let pass = match &mut self.dflash {
            Some(d) => d.pass_ms()?,
            None => 0.0,
        };
        let chain = match &mut self.mtp {
            Some(m) => m.driver.chain_ms()?,
            None => 0.0,
        };
        Ok(chain + pass)
    }

    /// Tells the part what its MTP block drafts in the step that is enqueued
    /// next. Without this call a step drafts nothing.
    pub fn mtp_step(&mut self, step: MtpStep) -> Result<(), ModelError> {
        let m = self.mtp.as_mut().ok_or_else(|| invalid("this part holds no MTP block"))?;
        m.driver.draft_step(step)
    }

    /// Where the step enqueued last left its drafts, the MTP block's and
    /// the DFlash drafter's in one list; `None` when it drafted nothing.
    pub fn drafted(&self) -> Option<Drafted> {
        let mtp = self.mtp.as_ref().map_or(0, |m| m.driver.drafted());
        let dflash = self.dflash.as_ref().map_or(0, |d| d.drafted());
        let tokens = mtp.max(dflash);
        (tokens > 0).then(|| Drafted { ptr: self.draft_list(), tokens })
    }

    /// The step's list of drafts, which every drafter of the part writes at
    /// the places of its windows: the MTP block's when the part has one.
    pub fn draft_list(&self) -> DevicePtr {
        match (&self.mtp, &self.dflash) {
            (Some(m), _) => m.driver.list(),
            (None, Some(d)) => d.own_list(),
            (None, None) => DevicePtr::NULL,
        }
    }

    /// The probability the drafter gave each draft of the step that ran last,
    /// in the order of [`Part::drafted`], appended to `out`; NaN for a link
    /// that was not made. The step has ended.
    pub fn draft_probs(&self, out: &mut Vec<f32>) {
        let Some(drafted) = self.drafted() else { return };
        let from = out.len();
        out.resize(from + drafted.tokens as usize, f32::NAN);
        if let Some(m) = &self.mtp {
            m.driver.probs(&mut out[from..]);
        }
        if let Some(d) = self.dflash.as_ref().filter(|d| d.drafted() > 0) {
            for s in d.last_pass() {
                for j in s.place..s.place + s.wanted {
                    out[from + j as usize] = d.prob(j);
                }
            }
        }
    }

    /// The block's exact K/V entries (its cells are in them while their tile is
    /// exact: see [`MTP_KV`]), for tests that read the cells it wrote.
    pub fn mtp_exact_entries(&self) -> Option<&DeviceBuffer> {
        self.mtp.as_ref().and_then(|m| m.layer.kv.exact_buffer())
    }

    /// The hidden states kept per sequence slot (`d` f32 each, and a row of
    /// zeros behind them), for tests that put a hidden state on the device.
    pub fn mtp_carry(&self) -> Option<&DeviceBuffer> {
        self.mtp.as_ref().map(|m| m.driver.carry())
    }

    /// The chain's scratch and, in it, the offsets of the block's output of
    /// link `link` (a row of `d` values per chain), of the logits of the last
    /// link that ran (a row of the draft head per chain) and of the picks of
    /// link `link` ([`ares_sampler::SamplePick`] records): for tests.
    pub fn mtp_scratch(&self, link: u32) -> Option<(&DeviceBuffer, [u64; 3])> {
        self.mtp.as_ref().map(|m| m.driver.scratch(link))
    }

    /// The tokens of the columns of this part's logits (a u32 each) when a
    /// draft vocabulary reorders the output head; null when column `i` is
    /// the token of the part's first row plus `i`.
    pub fn head_tokens(&self) -> DevicePtr {
        self.w.head_tokens.map_or(DevicePtr::NULL, |at| self.w.at(at))
    }

    /// The rows of the draft head.
    pub fn draft_head_rows(&self) -> u32 {
        self.mtp.as_ref().map_or(0, |m| m.driver.head_rows())
    }

    /// Step 1 of the MTP block at the start of a step, before its copies of
    /// the prefix cache: the rows of the last step that count (the outcome
    /// of its windows is what [`Part::commit_windows`] was told), from the
    /// final hidden states that step left in `xn`. A step follows first
    /// itself when this was not called.
    pub fn mtp_follow(
        &mut self,
        stream: &Stream,
        tables: DevicePtr,
        pages: DevicePtr,
    ) -> Result<(), ModelError> {
        let tap = self.at(self.s.xn);
        let outcomes = std::mem::take(&mut self.outcomes);
        let accepted = |slot: u32, _row0: u32| {
            outcomes.iter().find(|(s, _)| *s == slot).map(|(_, accepted)| *accepted)
        };
        self.with_mtp(|(driver, layer, layers)| {
            driver.follow(layer, layers, stream, &accepted, tap, tables, pages)
        })?;
        Ok(())
    }

    /// Before the model's blocks: the block follows the last step if that
    /// was not done, writes the newest cells of the sequences that do not
    /// draft and enqueues the chains. `only` is a step that only drafts.
    pub(super) fn mtp_before(&mut self, env: &StepEnv<'_>, only: bool) -> Result<(), ModelError> {
        if self.mtp.is_none() {
            return Ok(());
        }
        self.mtp_follow(env.stream, env.tables, env.pages)?;
        let rotary = self.rotary_of_step();
        self.with_mtp(|(driver, layer, layers)| {
            driver.set_rotary(rotary);
            driver.step(
                layer,
                layers,
                env.stream,
                env.segments,
                env.tokens,
                env.replay,
                env.tables,
                env.pages,
                only,
            )
        })?;
        self.mark(env.stream, Phase::Other)
    }

    /// After the model's blocks and its head: the final hidden state of the
    /// step's rows, which the block follows at the start of the next step.
    pub(super) fn mtp_after(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.mtp.is_none() {
            return Ok(());
        }
        let (x, xn) = (self.at(self.s.x), self.at(self.s.xn));
        // SAFETY: `x` holds the residual stream of the step's rows after
        // the last block, `xn` has room for as many rows.
        unsafe {
            self.kernels.rms_norm_f32(
                env.stream,
                x,
                self.w.at(self.w.output_norm),
                xn,
                self.d,
                env.dims.n_rows,
                self.eps,
            )?;
        }
        self.mark(env.stream, Phase::Other)
    }

    /// For tests of the block alone: a step of the model whose final hidden
    /// states are `hidden` (`d` values per row of `env`), which the block
    /// follows at the start of the next step. The block does what it does in
    /// such a step first: it follows the step before, writes the newest cells
    /// and makes the chains of [`Part::mtp_step`].
    pub fn enqueue_mtp_step(
        &mut self,
        env: &StepEnv<'_>,
        hidden: &[f32],
    ) -> Result<(), ModelError> {
        if self.mtp.is_none() {
            return Err(invalid("no MTP block"));
        }
        if hidden.len() != env.dims.n_rows as usize * self.d as usize {
            return Err(invalid("one hidden state per row of the step"));
        }
        // The exact entries and the table of the block's layer are made ready
        // for the step, as `Part::enqueue` does in its first stage.
        self.prepare_kv(env)?;
        self.mtp_before(env, false)?;
        let bytes: Vec<u8> = hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
        env.stream.synchronize()?;
        self.scratch.write(self.s.xn, &bytes)?;
        Ok(())
    }

    /// Adds the MTP block's events of the last step to its times.
    pub(super) fn mtp_flush(&mut self) -> Result<(), ModelError> {
        match &mut self.mtp {
            Some(m) => m.driver.chain_ms().map(|_| ()),
            None => Ok(()),
        }
    }
}
