//! The memory of a part: how its scratch, its inputs (the inbox), its
//! pools and its exchanges are laid out for the rows it takes ([`Frame`]),
//! computed before anything is allocated so that the placement and the
//! server can size a part without opening it.

use std::ops::Range;

use ares_core::GgmlType;
use ares_cuda::{
    CacheFormat, CacheGeometry, DevicePtr, IndexerShape, Kernels, KeyFormat, ListHandOver,
    MAX_BATCH, NgramShape, PlanConfig, Q4Kernels, RecurrentShape,
};
use ares_quant::act;
use ares_runtime::ModelError;
use ares_runtime::expert_store::cpu::act_bytes as cpu_act_bytes;

use super::super::load::{LayerShare, Role};
use super::{AttentionAt, MAX_PIECES, Options, RecurrentAt, SCORE_ROWS, wide};
use crate::blocks::slots::{SlotLayout, TAILS_PER_SLOT};
use crate::blocks::{invalid, kv, recurrent};
use crate::qwen4exp::Config;

/// What a part is and holds: its role and, per layer from `first` on, its
/// share.
#[derive(Clone, Debug)]
pub(crate) struct PartShape {
    pub role: Role,
    pub first: usize,
    pub shares: Vec<LayerShare>,
    pub head: bool,
}

impl PartShape {
    /// Whole layers `layers`, as a split by layers gives them to a device.
    pub(crate) fn whole(c: &Config, layers: Range<usize>, head: bool) -> Self {
        Self {
            role: Role::Whole,
            first: layers.start,
            shares: layers.map(|l| LayerShare::whole(c, l)).collect(),
            head,
        }
    }
}

/// Offsets of the step scratch.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Scratch {
    pub(super) r: u64,
    /// The second buffer of the residual: a read that writes goes from one
    /// to the other.
    pub(super) r2: u64,
    /// The partial sums between the two launches of a read.
    pub(super) partials: u64,
    /// The router's record before it is cut between two devices.
    pub(super) rec_full: u64,
    pub(super) zeros: u64,
    pub(super) xn: u64,
    pub(super) lo: u64,
    pub(super) down: u64,
    pub(super) gate: u64,
    pub(super) inj_a: u64,
    pub(super) inj_f: u64,
    pub(super) m: u64,
    pub(super) p: u64,
    pub(super) out: u64,
    // Recurrent and attention: two wide buffers and the smaller ones.
    pub(super) w1: u64,
    pub(super) w2: u64,
    pub(super) alpha: u64,
    pub(super) beta: u64,
    pub(super) q: u64,
    pub(super) kv: u64,
    pub(super) o: u64,
    pub(super) k: u64,
    pub(super) v: u64,
    pub(super) agate: u64,
    pub(super) iq: u64,
    pub(super) raw: u64,
    pub(super) rope: u64,
    pub(super) scores: u64,
    pub(super) counts: u64,
    pub(super) act: u64,
    pub(super) attn_state: u64,
    pub(super) status: u64,
    // Mixture of experts.
    pub(super) logits: u64,
    pub(super) hidden: u64,
    pub(super) sg: u64,
    pub(super) su: u64,
    pub(super) ysh: u64,
    pub(super) s: u64,
    // n-gram block.
    pub(super) emb: u64,
    pub(super) key: u64,
    pub(super) value: u64,
    pub(super) gated: u64,
    pub(super) nrm: u64,
    // Head.
    pub(super) picked: u64,
    pub(super) act_p: u64,
    pub(super) head: u64,
    // Steps wider than the decode kernels take ([`wide`]): the f16 form of the mixed input
    // and of the outputs that are quantized, the addresses of the experts of the rounds,
    // the plan, the hidden values and the f16 form of the mixed input for the experts.
    pub(super) mixed_f16: u64,
    /// The mixed input of a half block as the mixer pushed it, f32, before it is
    /// prepared (on a worker).
    pub(super) m_raw: u64,
    pub(super) out_f16: u64,
    /// The f16 form of the input of a float product ([`wide::float_f16_bytes`]).
    pub(super) float_f16: u64,
    pub(super) addresses: u64,
    pub(super) plan_rows: u64,
    pub(super) hidden_rows: u64,
    pub(super) experts_f16: u64,
    /// The sums of the experts the CPU computed for rows of a wide step, f32, for steps
    /// of up to [`Options::cpu_rows`] rows ([`wide`]).
    pub(super) cpu_sums: u64,
    /// The bytes of `zeros`, which a step on lent VRAM clears first.
    pub(super) zeros_len: u64,
    /// Steps on lent VRAM: where the converted entries of a layer's pages in
    /// host memory are copied before its attention reads them, and the bytes
    /// for all of them (0 when no page is far or no step is lent for).
    pub(super) far_stage: u64,
    pub(super) far_stage_len: u64,
    pub(super) len: u64,
}

/// VRAM lent by the expert slots in pieces (a piece of each pool a loan takes): per piece,
/// where it starts in the layout of the scratch ([`Frame::new_in`]) and where it lies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pieces(pub Vec<(u64, DevicePtr)>);

impl Pieces {
    /// Where the scratch's offset `offset` lies: in the last piece that starts at or before
    /// it.
    pub fn address(&self, offset: u64) -> DevicePtr {
        let at = self.0.iter().rposition(|(start, _)| *start <= offset).unwrap_or(0);
        let (start, base) = self.0[at];
        base.offset(offset - start)
    }
}

/// Offsets of what the host sends with a step.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Inbox {
    pub(super) record: u64,
    pub(super) record_cap: u64,
    /// The rows of the step's tokens in the table's form, as the host staged
    /// them (`blocks/rows.rs`), and the gather's output: their values, on the
    /// device only.
    pub(super) rows: u64,
    pub(super) x: u64,
    pub(super) plan: u64,
    pub(super) plan_cap: u64,
    pub(super) hands: u64,
    pub(super) ngram: u64,
    /// The record of the step before, with its accepted counts.
    pub(super) commit: u64,
    /// The bytes the host writes (everything but `x`), and all.
    pub(super) host_len: u64,
    pub(super) len: u64,
}

/// What the layers of a part keep and where, before the scratch: the slot of
/// a sequence (each layer's recurrent state or its indexer's carry, and the
/// n-gram block's history), the pools of K/V and keys, and what a draft window
/// keeps of the indexer's inputs.
struct Layers {
    slot: SlotLayout,
    pool_bytes: u64,
    raw_bytes: u64,
    rec_layout: recurrent::Layout,
    recurrent: Vec<Option<RecurrentAt>>,
    attention: Vec<Option<AttentionAt>>,
    ngram_history: Option<u64>,
    layer_slot: Vec<Range<u64>>,
    any_recurrent: bool,
    any_attention: bool,
    any_ngram: bool,
}

impl Layers {
    /// The layers of a part of `shape` (one that holds the streams when
    /// `streams`), with `exact_slots` exact entries of K and V per layer.
    fn of(c: &Config, opts: &Options, shape: &PartShape, streams: bool, exact_slots: u32) -> Self {
        let window = u64::from(opts.window_rows);
        let geo = geometry(c, opts.kv);
        let ix = indexer_shape(c);
        let ng = ngram_shape(c);
        let pages = u64::from(opts.pages());
        // The converted entries of K and V of the pages in VRAM; the indexer's
        // keys of every page.
        let near = u64::from(opts.near_pages());
        let r = &c.recurrent;
        let id = c.indexer.key_dim as u64;
        let mut exact_at = 0u64;
        let (mut slot, mut pool_bytes, mut raw_bytes) = (SlotLayout::new(), 0u64, 0u64);
        let mut rec_layout = recurrent::Layout::new(opts.state, opts.window_rows);
        let mut recurrent = Vec::new();
        let mut attention = Vec::new();
        let mut ngram_history = None;
        let (mut any_recurrent, mut any_attention, mut any_ngram) = (false, false, false);
        let mut keep = |bytes: u64| {
            let at = raw_bytes;
            raw_bytes = (raw_bytes + window * bytes).next_multiple_of(64);
            at
        };
        let mut layer_slot = Vec::with_capacity(shape.shares.len());
        for (i, share) in shape.shares.iter().enumerate() {
            let layer = shape.first + i;
            let heads = u64::from(share.heads.end.saturating_sub(share.heads.start));
            let (mut r_at, mut a_at) = (None, None);
            let slot_start = slot.bytes();
            if heads == 0 {
                // The layer's heads are all on the peer.
            } else if c.is_attention(layer) {
                any_attention = true;
                let kv_pool = pool_bytes;
                let key_pool = (kv_pool + near * geo.converted_entry_bytes()).next_multiple_of(64);
                pool_bytes = (key_pool + pages * ix.key_slot_bytes(opts.keys)).next_multiple_of(64);
                let carry = slot.take(ix.carry_len() * 4);
                let exact = exact_at;
                exact_at += u64::from(exact_slots) * geo.exact_entry_bytes();
                let kept_raw = keep(id * 4);
                a_at = Some(AttentionAt { kv_pool, key_pool, carry, kept_raw, exact });
            } else {
                any_recurrent = true;
                // The part's key heads with their value heads: a narrower
                // layer.
                let n_k = heads as u32;
                let shape = RecurrentShape {
                    n_k_heads: n_k,
                    n_v_heads: n_k * (r.n_v_heads / r.n_k_heads) as u32,
                    head_dim: r.head_dim as u32,
                    conv_kernel: r.conv_kernel as u32,
                };
                r_at = Some(rec_layout.layer(shape, &mut slot));
            }
            if streams && c.ngram.as_ref().is_some_and(|g| g.layer == layer) {
                any_ngram = true;
                let history = ng.as_ref().map_or(0, NgramShape::history_len);
                ngram_history = Some(slot.take(history * 4));
            }
            recurrent.push(r_at);
            attention.push(a_at);
            layer_slot.push(slot_start..slot.bytes());
        }
        Self {
            slot,
            pool_bytes,
            raw_bytes,
            rec_layout,
            recurrent,
            attention,
            ngram_history,
            layer_slot,
            any_recurrent,
            any_attention,
            any_ngram,
        }
    }
}

/// The scratch of a step as it is carved buffer by buffer, every buffer from
/// the start of a line of the devices' caches ([`Kernels::ACTIVATION_LINE`],
/// as the 27B's scratch: the tile kernels read the f16 activations fastest
/// from there), in `pieces` when it is laid out in VRAM the expert slots lend:
/// a buffer that the end of a piece would cut goes to the start of the next,
/// and what each piece holds is counted. A piece starts at a slot of its
/// pool, and a slot of a real file's experts is a multiple of 256 bytes.
struct PieceCarve {
    len: u64,
    /// The pieces as ranges of the layout's offsets, and what each holds.
    starts: Vec<u64>,
    ends: Vec<u64>,
    held: Vec<u64>,
    fits: bool,
}

impl PieceCarve {
    fn new(pieces: &[u64]) -> Self {
        let starts: Vec<u64> = pieces
            .iter()
            .scan(0u64, |at, bytes| {
                let start = *at;
                *at += bytes / 256 * 256;
                Some(start)
            })
            .collect();
        let ends: Vec<u64> =
            starts.iter().zip(pieces).map(|(start, bytes)| start + bytes / 256 * 256).collect();
        Self { len: 0, starts, ends, held: vec![0; pieces.len()], fits: true }
    }

    /// The offset of the next buffer, of `bytes` bytes (at least 16).
    fn take(&mut self, bytes: u64) -> u64 {
        let bytes = bytes.max(16);
        let mut at = self.len;
        if !self.ends.is_empty() {
            let ends = &self.ends;
            let mut k = ends.iter().position(|end| at < *end).unwrap_or(ends.len());
            while k < ends.len() && at + bytes > ends[k] {
                k += 1;
                at = at.max(ends[k - 1]);
            }
            match self.held.get_mut(k) {
                Some(held) => *held = (*held).max(at + bytes - self.starts[k]),
                None => self.fits = false,
            }
        }
        self.len = (at + bytes).next_multiple_of(Kernels::ACTIVATION_LINE);
        at
    }

    /// The bytes the scratch takes of each piece; `None` without pieces or when
    /// it does not fit in them.
    fn lent(self) -> Option<Vec<u64>> {
        (!self.held.is_empty() && self.fits).then_some(self.held)
    }
}

impl Inbox {
    /// What the host sends with a step of `opts.max_rows` rows: the record,
    /// the attention plan with its hands (a part with attention layers), the
    /// n-gram rows (a part with the block) and the record of the step before,
    /// the staged rows of the table and the rows they decode to.
    fn lay_out(
        opts: &Options,
        n: u64,
        geo: &CacheGeometry,
        ix: &IndexerShape,
        ng: Option<&NgramShape>,
        any_attention: bool,
        any_ngram: bool,
    ) -> Self {
        let (att, gram) = (u64::from(any_attention), u64::from(any_ngram));
        let rows = u64::from(opts.max_rows);
        let mut inbox = Inbox::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        // A record: header, a segment per slot, a token and four positions
        // per row, the output rows.
        inbox.record_cap = (64 + 32 * u64::from(opts.max_slots) + 28 * rows).next_multiple_of(64);
        inbox.record = take(inbox.record_cap);
        // A plan: the planner's dense plans are a few dozen bytes per row and
        // item; a plan of device-made lists has `MAX_PIECES` items per row
        // and KV head and three words per range.
        let listed = rows
            * (u64::from(MAX_PIECES) * u64::from(geo.kv_heads) * 48
                + u64::from(ix.max_ranges(0)) * 12
                + 32);
        inbox.plan_cap = att * (listed.max(64 << 10) + 96 * rows + 256).next_multiple_of(64);
        inbox.plan = take(inbox.plan_cap);
        inbox.hands = take(att * rows * ListHandOver::BYTES as u64);
        inbox.ngram = take(gram * rows * ng.map_or(0, NgramShape::fetched_row_bytes));
        inbox.commit = take(inbox.record_cap);
        // The staged rows of a table of any type, at most four bytes a value,
        // and the rows they decode to, which only the device writes.
        inbox.rows = take(rows * n * 4);
        inbox.x = take(rows * n * 4);
        inbox.host_len = inbox.x;
        inbox.len = len;
        inbox
    }
}

/// Every size and offset of a part: what [`Part::open`] allocates and what a
/// placement counts.
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub(super) slot_bytes: u64,
    /// Per layer of the part, the bytes of a slot that hold its state (its recurrent state,
    /// its indexer's carried keys, the n-gram history at its layer): what a save inside a
    /// pass copies of the layer as the pass goes past the position.
    pub(super) layer_slot: Vec<Range<u64>>,
    /// The exact slots of the K/V group, and the bytes it takes besides the
    /// pages on the device and in page-locked host memory.
    pub(super) exact_slots: u32,
    pub(super) exact_device: u64,
    pub(super) exact_host: u64,
    /// Bytes of partial attention states the scratch holds for a plan.
    pub(super) attn_state_bytes: u64,
    pub(super) recurrent: Vec<Option<RecurrentAt>>,
    pub(super) attention: Vec<Option<AttentionAt>>,
    pub(super) ngram_history: Option<u64>,
    pub(super) pool_bytes: u64,
    /// The recurrent layers' extents and kept inputs.
    pub(super) rec: recurrent::Layout,
    /// Bytes of the indexer's kept inputs of a window.
    pub(super) raw_bytes: u64,
    pub(super) s: Scratch,
    /// The scratch the part keeps for good: `s` itself, or the layout of a
    /// step of [`Options::own_rows`] rows when wider steps run on lent VRAM.
    pub(super) own: Scratch,
    /// Whether steps wider than the own scratch run on lent VRAM.
    pub(super) lends: bool,
    /// Laid out in pieces ([`Frame::new_in`]): what each piece holds, or `None` when the
    /// scratch does not fit in them.
    pub(super) lent_pieces: Option<Vec<u64>>,
    pub(super) inbox: Inbox,
    pub(super) record_stride: u64,
    /// Bytes from one layer's input of the mixture of experts to the next,
    /// on the device (the whole activations) and on the host (what the CPU
    /// reads of them).
    pub(super) act_stride: u64,
    pub(super) act_host_stride: u64,
    pub(super) geo: CacheGeometry,
    pub(super) ix: IndexerShape,
    /// How the indexer's pooled keys are stored.
    pub(super) keys: KeyFormat,
    pub(super) ng: Option<NgramShape>,
    pub(super) plan_config: PlanConfig,
    /// The flags of the exchange with a peer and the bytes of the exchange
    /// buffer (flags, then a partial output); 0 for a whole part.
    pub(super) x_flags: u32,
    pub(super) x_bytes: u64,
    /// Bytes of the table that says which experts the peer owns.
    pub(super) owner_bytes: u64,
    /// Bytes of page-locked memory per layer for the address tables of the rounds of a
    /// wide step; 0 when steps have at most [`MAX_BATCH`] rows.
    pub(super) addr_stage: u64,
    /// The layers after a layer whose picks the part predicts (0 for none), the bytes of
    /// one list of predicted picks, and of the logits of the routers ahead for the rows of a
    /// decode step.
    pub(super) ahead_layers: u32,
    pub(super) ahead_list_bytes: u64,
    pub(super) ahead_logits_bytes: u64,
    /// In wide steps (when the part predicts): the bytes of the f16 form of a step's rows,
    /// of their logits and of their picks by the next layer's router; and, per layer, the
    /// bytes of the picks' ids the host gets. 0 without wide steps.
    pub(super) wide_input_bytes: u64,
    pub(super) wide_logits_bytes: u64,
    pub(super) wide_record_bytes: u64,
    pub(super) wide_ids_bytes: u64,
    /// Page-locked bytes for the CPU's part of a wide step: the rows' activations the CPU
    /// reads and the sums it gives back, for steps of up to [`Options::cpu_rows`] rows; 0
    /// without wide steps.
    pub(super) wide_cpu_host: u64,
}

/// The bytes of the quantized form of `rows` rows of `n_in` activations.
pub(super) fn act_bytes(n_in: u64, rows: u64) -> u64 {
    act::Layout::of(n_in * rows).bytes
}

/// The K/V geometry of an attention layer: every KV head on the layer's
/// device.
pub(crate) fn geometry(c: &Config, kv: (CacheFormat, CacheFormat)) -> CacheGeometry {
    let a = &c.attention;
    CacheGeometry {
        k: kv.0,
        v: kv.1,
        head_dim: a.head_dim as u32,
        kv_heads: a.n_head_kv as u32,
        kv_total: a.n_head_kv as u32,
        first_kv_head: 0,
        q_per_kv: (a.n_head / a.n_head_kv) as u32,
    }
}

/// The layer of the K/V group that the attention layer at `at` is.
pub(super) fn kv_layer(f: &Frame, at: &AttentionAt) -> kv::Layer {
    kv::Layer { geo: f.geo, pool: at.kv_pool, exact: at.exact }
}

pub(super) fn indexer_shape(c: &Config) -> IndexerShape {
    IndexerShape {
        n_head: c.indexer.n_head as u32,
        key_dim: c.indexer.key_dim as u32,
        block: c.indexer.block as u32,
        top_k: c.indexer.top_k as u32,
        rope_dim: c.attention.rope.dim as u32,
    }
}

pub(in super::super) fn ngram_shape(c: &Config) -> Option<NgramShape> {
    c.ngram.as_ref().map(|g| NgramShape {
        n_stream: c.n_stream as u32,
        n_embd: c.n_embd as u32,
        heads: g.n_heads() as u32,
        row_dim: g.row_dim as u32,
        ty: g.table_type,
        conv_kernel: g.conv_kernel as u32,
        dilation: g.ngram_size as u32,
    })
}

impl Frame {
    /// The sizes of a part of `shape` on a device with `multiprocessors`
    /// multiprocessors (the attention plans are cut by it). With
    /// [`Options::own_rows`] below `max_rows`, the part's own scratch is laid out
    /// for that many rows and `s` is the layout of a step on lent VRAM.
    pub(crate) fn new(
        c: &Config,
        opts: &Options,
        shape: &PartShape,
        multiprocessors: u32,
    ) -> Result<Self, ModelError> {
        Self::new_in(c, opts, shape, multiprocessors, &[])
    }

    /// [`Frame::new`] whose scratch for steps on lent VRAM lies in pieces of at most
    /// `pieces` bytes each, in their order (the pieces a loan of several pools of expert
    /// slots is made of): no buffer is cut by the end of a piece. Empty: one piece as large
    /// as it needs. [`Frame::lent_pieces`] says what each piece holds, and whether the
    /// scratch fits.
    pub(crate) fn new_in(
        c: &Config,
        opts: &Options,
        shape: &PartShape,
        multiprocessors: u32,
        pieces: &[u64],
    ) -> Result<Self, ModelError> {
        let own = opts.own_rows();
        let lends = own < opts.max_rows;
        let mut frame =
            Self::widest(c, opts, shape, multiprocessors, lends, if lends { pieces } else { &[] })?;
        if lends {
            let small = Options { max_rows: own, own_rows: 0, ..opts.clone() };
            frame.own = Self::widest(c, &small, shape, multiprocessors, false, &[])?.s;
            frame.lends = true;
        }
        Ok(frame)
    }

    /// The sizes of a part whose scratch holds a step of `opts.max_rows` rows. With
    /// `overlay` (the scratch of a step on lent VRAM, which verifies no drafts) the buffers
    /// of the read of a stream mixer, of the recurrent or attention half and of the n-gram
    /// block share their place: a layer uses them one after the other.
    ///
    /// The scratch lies in `pieces` as [`Frame::new_in`] says.
    pub(super) fn widest(
        c: &Config,
        opts: &Options,
        shape: &PartShape,
        multiprocessors: u32,
        overlay: bool,
        pieces: &[u64],
    ) -> Result<Self, ModelError> {
        let head = shape.head;
        let streams = shape.role != Role::Worker;
        if opts.max_rows == 0 || u64::from(opts.max_rows) * c.wide() as u64 >= 1 << 32 {
            return Err(invalid(format!("steps of up to {} rows", opts.max_rows)));
        }
        if opts.logit_rows == 0 || opts.logit_rows > opts.max_rows || opts.max_slots == 0 {
            return Err(invalid(
                "a part needs a logit row (at most the rows of a step) and a slot",
            ));
        }
        if opts.window_rows > opts.max_rows {
            return Err(invalid("a draft window has at most the rows of a step"));
        }
        let rows = u64::from(opts.max_rows);
        let (n, hc) = (c.n_embd as u64, c.n_stream as u64);
        let (wide, rank) = (c.wide() as u64, c.mixer_rank as u64);
        let geo = geometry(c, opts.kv);
        let ix = indexer_shape(c);
        let ng = ngram_shape(c);
        let plan_config = PlanConfig::for_multiprocessors(multiprocessors);
        let (pages, near) = (u64::from(opts.pages()), u64::from(opts.near_pages()));
        // The exact slots of the K/V group, when its formats have a tile mean,
        // with the tails the prefix cache may hold.
        let sizing = kv::Sizing {
            slots: opts.max_slots,
            rows: opts.max_rows,
            tails: opts.max_slots * TAILS_PER_SLOT,
        };
        let exact_slots = kv::slots_for(opts.kv.0, opts.kv.1, sizing);

        let r = &c.recurrent;
        let (cd, kd, vd, nv) =
            (r.conv_dim() as u64, r.key_dim() as u64, r.value_dim() as u64, r.n_v_heads as u64);
        let (hq, hkv) = c.attention_dims();
        let (hq, hkv) = (hq as u64, hkv as u64);
        let iq = (c.indexer.n_head * c.indexer.key_dim) as u64;
        let id = c.indexer.key_dim as u64;

        let Layers {
            slot,
            pool_bytes,
            raw_bytes,
            rec_layout,
            recurrent,
            attention,
            ngram_history,
            layer_slot,
            any_recurrent,
            any_attention,
            any_ngram,
        } = Layers::of(c, opts, shape, streams, exact_slots);

        // The scratch of a step.
        let mut s = Scratch::default();
        let mut carve = PieceCarve::new(pieces);
        macro_rules! take {
            ($bytes:expr $(,)?) => {
                carve.take($bytes)
            };
        }
        s.r = take!(rows * wide * 4);
        s.r2 = take!(u64::from(streams) * rows * wide * 4);
        s.partials = take!(
            Q4Kernels::read_partials(opts.max_rows, c.n_stream as u32, c.mixer_rank as u32) * 4,
        );
        s.zeros = take!(rows * hc * 4);
        s.zeros_len = rows * hc * 4;
        s.inj_a = take!(rows * hc * 4);
        s.inj_f = take!(rows * hc * 4);
        s.m = take!(act_bytes(n, rows));
        s.p = take!(rows * n * 4);
        s.out = take!(rows * n * 4);
        let (rec, att) = (u64::from(any_recurrent), u64::from(any_attention));
        s.rope = take!(rows * att * (c.attention.rope.dim as u64 / 2) * 8);
        let shared_ff = c.experts.shared_ff as u64;
        s.act = take!(act_bytes((rec * vd).max(att * hq).max(shared_ff), rows));
        // Three groups of buffers that a layer uses one after the other: the read of a
        // stream mixer's, the recurrent or attention half's, and the n-gram block's (which
        // runs before the read of its layer). On lent VRAM (`overlay`) they lie in
        // the same place: a step on a loan verifies no drafts, whose windows keep the
        // n-gram block's `nrm` until their commit. Elsewhere they lie one after the other.
        let shared_at = carve.len;
        let mut shared_end = carve.len;
        s.xn = take!(act_bytes(wide, rows));
        s.lo = take!(act_bytes(rank, rows));
        s.down = take!(rows * rank * 4);
        s.gate = take!(rows * wide * 4);
        if overlay {
            shared_end = shared_end.max(carve.len);
            carve.len = shared_at;
        }
        s.w1 = take!(rows * (rec * cd).max(att * 2 * hq) * 4);
        s.w2 = take!(rows * (rec * vd).max(att * hkv) * 4);
        s.alpha = take!(rows * rec * nv * 4);
        s.beta = take!(rows * rec * nv * 4);
        s.q = take!(rows * (rec * kd).max(att * hq) * 4);
        s.kv = take!(rows * rec * (kd + vd) * 4);
        s.o = take!(rows * (rec * vd).max(att * hq) * 4);
        s.k = take!(rows * att * hkv * 4);
        s.v = take!(rows * att * hkv * 4);
        s.agate = take!(rows * att * hq * 4);
        s.iq = take!(rows * att * iq * 4);
        s.raw = take!(rows * att * id * 4);
        // A row's block scores take 4 bytes per block of the longest sequence
        // (256 KiB a row at 262 144 cells): a step scores and selects its rows
        // `SCORE_ROWS` at a time, so the scratch does not grow with the rows.
        let max_blocks = u64::from(opts.cells.div_ceil(ix.block)).max(1);
        s.scores = take!(rows.min(u64::from(SCORE_ROWS)) * att * max_blocks * 4);
        s.counts = take!(rows * 16);
        // The most partial states a plan makes: the planner's bound for dense
        // rows (`PlanConfig::state_slots_bound`, as the 27B: a row that reads its
        // prefix with another sequence, or past its first tiles with a tile
        // mean, has several pieces; no row more than its sequence has entries,
        // which is the smaller bound for short sequences) and the pieces of
        // device-made lists (`build_plan`).
        let alone = opts.window_rows.max(opts.max_slots);
        let mean = opts.kv.0.has_mean() || opts.kv.1.has_mean();
        let dense = plan_config
            .state_slots_bound(geo.kv_heads, geo.q_per_kv, rows as u32, alone, opts.max_slots, mean)
            .min((rows + u64::from(alone)) * PlanConfig::row_slots_bound(opts.cells));
        let state_slots = dense.max(list_state_slots(plan_config, geo.kv_heads, rows, alone));
        // A state is, per query head, the weighted V and the log of the sum.
        let state = c.attention.n_head as u64 * (u64::from(geo.head_dim) + 1) * 4;
        let attn_state_bytes = att * state_slots * state;
        s.attn_state = take!(attn_state_bytes);
        let gram = u64::from(any_ngram);
        if overlay {
            shared_end = shared_end.max(carve.len);
            carve.len = shared_at;
        }
        s.emb = take!(gram * act_bytes(n, rows));
        s.key = take!(gram * rows * wide * 4);
        s.value = take!(gram * rows * n * 4);
        s.gated = take!(gram * rows * wide * 4);
        s.nrm = take!(gram * rows * wide * 4);
        carve.len = carve.len.max(shared_end);
        s.status = take!(16);
        let expert_shape = ares_cuda::ExpertShape {
            n_embd: n as u32,
            n_ff: c.experts.ff as u32,
            n_experts: c.experts.count as u32,
            top_k: c.experts.used as u32,
            gate_ty: GgmlType::IQ2_S,
            down_ty: GgmlType::Q2_0,
        };
        // The records of the decode kernels have 8 rows; a wide step's record is the ids and
        // the weights of the router's picks.
        let decode_rows = opts.max_rows.min(MAX_BATCH);
        let record_bytes = expert_shape
            .record_bytes(decode_rows)
            .max(wide::record_bytes(&expert_shape, opts.max_rows));
        s.rec_full = take!(record_bytes);
        let ahead_layers = if shape.role == Role::Worker {
            0
        } else {
            opts.predict.min(2).min(shape.shares.len().saturating_sub(1) as u32)
        };
        // Wide steps predict the next layer's picks as well.
        let wide_ahead = u64::from(ahead_layers > 0 && opts.max_rows > MAX_BATCH);
        s.logits = take!(expert_shape.logits_bytes(opts.max_rows));
        s.hidden = take!(expert_shape.hidden_bytes(decode_rows));
        s.sg = take!(rows * shared_ff * 4);
        s.su = take!(rows * shared_ff * 4);
        s.ysh = take!(rows * n * 4);
        s.s = take!(rows * 4);
        let logit_rows = u64::from(opts.logit_rows);
        let heads = u64::from(head);
        s.picked = take!(heads * logit_rows * n * 4);
        s.act_p = take!(heads * act_bytes(n, logit_rows));
        s.head = take!(heads * logit_rows * c.n_vocab as u64 * 4);
        if opts.max_rows > MAX_BATCH {
            let f16 = |width: u64| {
                Kernels::prefill_activation_bytes(
                    ares_cuda::PrefillPath::F16,
                    width as u32,
                    opts.max_rows,
                )
            };
            s.mixed_f16 = take!(f16(n));
            s.m_raw = take!(rows * n * 4);
            s.out_f16 = take!(f16((rec * vd).max(att * hq).max(shared_ff)));
            s.float_f16 = take!(wide::float_f16_bytes(streams, wide, n, opts.max_rows));
            s.addresses = take!(wide::ROUNDS * 8 * c.experts.count as u64);
            s.plan_rows = take!(wide::plan_bytes(&expert_shape, opts.max_rows));
            s.hidden_rows = take!(expert_shape.rows_hidden_bytes(opts.max_rows));
            s.experts_f16 = take!(expert_shape.rows_activation_bytes(opts.max_rows));
            s.cpu_sums = take!(u64::from(opts.wide_cpu_rows()) * n * 4);
        }
        // A wide step on lent VRAM: every row of a prompt reads the cells of its
        // own pick, so a layer's K and V beyond VRAM are copied to VRAM once for
        // all rows, behind the step's append, and read there.
        if opts.max_rows > opts.own_rows() && any_attention && near < pages {
            s.far_stage_len = (pages - near) * geo.converted_entry_bytes();
            s.far_stage = take!(s.far_stage_len);
        }
        s.len = carve.len;
        let lent_pieces = carve.lent();

        // A flag per block of a read, and one for a route record.
        let x_flags = Q4Kernels::read_flags(opts.max_rows, c.n_embd as u32) + 16;

        // What the host sends with a step.
        let inbox = Inbox::lay_out(opts, n, &geo, &ix, ng.as_ref(), any_attention, any_ngram);

        let geos = vec![geo; attention.iter().flatten().count()];
        let (exact_device, exact_host) =
            kv::group_bytes(&geos, exact_slots, u64::from(opts.table_entries()));
        Ok(Self {
            slot_bytes: slot.bytes().max(64),
            layer_slot,
            exact_slots,
            exact_device,
            exact_host,
            attn_state_bytes,
            recurrent,
            attention,
            ngram_history,
            pool_bytes: pool_bytes.max(64),
            rec: rec_layout,
            raw_bytes,
            s,
            own: s,
            lends: false,
            lent_pieces,
            inbox,
            record_stride: record_bytes.next_multiple_of(64),
            act_stride: act_bytes(n, u64::from(decode_rows)).next_multiple_of(64),
            act_host_stride: (cpu_act_bytes(n as usize, decode_rows as usize) as u64)
                .next_multiple_of(64),
            geo,
            ix,
            keys: opts.keys,
            ng,
            plan_config,
            x_flags,
            x_bytes: if shape.role == Role::Whole {
                0
            } else {
                (u64::from(x_flags) * 4).next_multiple_of(64) + rows * n * 4
            },
            owner_bytes: if shape.role == Role::Mixer {
                (shape.shares.len() as u64 * c.experts.count as u64).next_multiple_of(64)
            } else {
                0
            },
            addr_stage: if opts.max_rows > MAX_BATCH {
                wide::ROUNDS * 8 * c.experts.count as u64
            } else {
                0
            },
            ahead_layers,
            ahead_list_bytes: expert_shape.predict_list_bytes(decode_rows),
            ahead_logits_bytes: u64::from(ahead_layers) * expert_shape.logits_bytes(decode_rows),
            wide_input_bytes: wide_ahead
                * Kernels::prefill_activation_bytes(
                    ares_cuda::PrefillPath::F16,
                    n as u32,
                    opts.max_rows,
                ),
            wide_logits_bytes: wide_ahead * expert_shape.logits_bytes(opts.max_rows),
            wide_record_bytes: wide_ahead * wide::record_bytes(&expert_shape, opts.max_rows),
            wide_ids_bytes: wide_ahead
                * (4 * u64::from(expert_shape.top_k) * u64::from(opts.max_rows))
                    .next_multiple_of(64),
            wide_cpu_host: if opts.max_rows > MAX_BATCH {
                wide::cpu_host_bytes(n, opts.wide_cpu_rows())
            } else {
                0
            },
        })
    }

    /// Laid out in pieces ([`Frame::new_in`]): the bytes the scratch of a step on lent VRAM
    /// takes of each piece, without the pieces it does not reach; `None` when it does not fit
    /// in them or no step is wider than the part's own scratch.
    pub(crate) fn lent_pieces(&self) -> Option<Vec<u64>> {
        let mut pieces = self.lent_pieces.clone().filter(|_| self.lends)?;
        while pieces.last() == Some(&0) {
            pieces.pop();
        }
        Some(pieces)
    }

    /// Where the input of a half block lies in the scratch of a step on lent VRAM.
    pub(crate) fn lent_raw(&self) -> u64 {
        self.s.m_raw
    }

    /// Bytes of the lists of predicted picks of every layer of the part, on the device and
    /// in page-locked memory.
    pub(super) fn predict_lists_bytes(&self) -> u64 {
        u64::from(self.ahead_layers) * self.ahead_list_bytes * self.recurrent.len() as u64
    }

    /// Bytes of recurrent state per sequence slot.
    pub(crate) fn state_bytes_per_slot(&self) -> u64 {
        self.slot_bytes
    }

    /// Bytes of the K/V pools and key pools.
    pub(crate) fn pool_bytes(&self) -> u64 {
        self.pool_bytes
    }

    /// The bytes a page takes in the part's attention layers: the converted
    /// entries of K and V, which lie in host memory for a page beyond VRAM,
    /// and the indexer's keys, which are in VRAM for every page.
    pub(crate) fn page_parts(&self) -> (u64, u64) {
        let layers = self.attention.iter().flatten().count() as u64;
        (layers * self.geo.converted_entry_bytes(), layers * self.ix.key_slot_bytes(self.keys))
    }

    /// Bytes of everything a step needs besides weights, state and pools:
    /// scratch, kept window inputs, route records, the step's inputs, tables.
    pub(crate) fn step_bytes(&self, opts: &Options) -> u64 {
        self.own.len
            + self.rec.kept_bytes()
            + self.raw_bytes
            + (self.record_stride + self.act_stride) * self.recurrent.len() as u64
            + self.inbox.len
            + self.tables_bytes(opts)
            + self.x_bytes
            + self.owner_bytes
            + self.exact_device
            + self.predict_bytes()
    }

    /// Bytes of the predictions on the device, as `predict::State::open` allocates them.
    pub(super) fn predict_bytes(&self) -> u64 {
        if self.ahead_layers == 0 {
            return 0;
        }
        self.ahead_logits_bytes.next_multiple_of(256)
            + self.predict_lists_bytes().max(64).next_multiple_of(256)
            + self.wide_input_bytes.next_multiple_of(256)
            + self.wide_logits_bytes.next_multiple_of(256)
            + self.wide_record_bytes.next_multiple_of(256)
    }

    /// Bytes of page-locked host memory the part holds: the route records
    /// and rows of `layers` layers, the step's inputs, the mirror of the
    /// tables and the status word.
    pub(crate) fn pinned_bytes(&self, opts: &Options, layers: usize) -> u64 {
        (self.record_stride + self.act_host_stride + self.addr_stage) * layers as u64
            + (u64::from(self.ahead_layers) * self.ahead_list_bytes + self.wide_ids_bytes)
                * layers as u64
            + self.inbox.host_len
            + self.tables_bytes(opts)
            + self.exact_host
            + self.wide_cpu_host
            + 16
    }

    pub(super) fn tables_bytes(&self, opts: &Options) -> u64 {
        (2 * 4 * u64::from(opts.table_entries())).next_multiple_of(64)
    }
}

/// The most partial states a plan of device-made lists takes (`Part::build_plan`, a
/// step with a row past the dense limit): a row of a prompt has the step's pieces of
/// converted entries (the plan's blocks over the rows, at least one) and one of exact
/// entries, a row alone (a generated row, a row of a draft window; at most `alone` in a
/// step) the pieces it has alone and one of exact entries.
fn list_state_slots(config: PlanConfig, kv_heads: u32, rows: u64, alone: u32) -> u64 {
    let budget = u64::from(config.target_blocks / kv_heads.max(1));
    let alone_pieces = budget.clamp(1, u64::from(MAX_PIECES));
    budget.max(rows) + rows + u64::from(alone) * (alone_pieces + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every step of device-made lists stays in [`list_state_slots`]: `prompt` rows of
    /// prompts with the step's pieces and `alone` rows with their own, each with a piece
    /// of exact entries (the counts of `Part::build_plan`).
    #[test]
    fn device_made_lists_stay_in_their_bound_of_states() {
        for sms in [28, 34, 46, 170] {
            let config = PlanConfig::for_multiprocessors(sms);
            for kv_heads in [1, 2, 4] {
                for (rows, most_alone) in [(8u32, 8u32), (64, 8), (512, 4), (4096, 8)] {
                    let bound = list_state_slots(config, kv_heads, u64::from(rows), most_alone);
                    let alone_pieces = (config.target_blocks / kv_heads).clamp(1, MAX_PIECES);
                    for alone in 0..=most_alone.min(rows) {
                        let prompt = rows - alone;
                        let step = (config.target_blocks / (prompt.max(1) * kv_heads))
                            .clamp(1, MAX_PIECES);
                        let slots = u64::from(prompt * (step + 1) + alone * (alone_pieces + 1));
                        assert!(
                            slots <= bound,
                            "{sms} {kv_heads} {rows} {alone}: {slots} > {bound}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_carve_starts_every_buffer_on_a_line_and_in_one_piece() {
        let line = Kernels::ACTIVATION_LINE;
        for pieces in [vec![], vec![3000, 5000, 100_000], vec![1 << 20, 4096]] {
            let mut carve = PieceCarve::new(&pieces);
            let ends: Vec<u64> = carve.ends.clone();
            let starts: Vec<u64> = carve.starts.clone();
            for bytes in [1, 16, 80, 130, 700, 2049, 64, 3, 1000, 1500] {
                let at = carve.take(bytes);
                assert_eq!(at % line, 0, "{pieces:?}: a buffer of {bytes} at {at}");
                let inside = starts.iter().zip(&ends).any(|(s, e)| at >= *s && at + bytes <= *e);
                assert!(pieces.is_empty() || inside || !carve.fits, "{pieces:?}: {at}+{bytes}");
            }
        }
    }
}
