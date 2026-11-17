//! The DFlash drafter on the device of the model it drafts for.
//!
//! The drafter is a kind of the runtime, not a part of one model: any model
//! with a DFlash file gets it, and gives it only what is its own.
//!
//! - **Taps.** The model calls [`Drafter::tap`] for each target layer with the
//!   residual stream of the step's rows before that layer, in the model's
//!   form: the 27B's rows `x` (one stream), Flash-Next's four
//!   hyper-connection streams, of which the drafter takes the mean (what it
//!   was trained on). The drafter keeps no taps: each one goes at once
//!   through its share of the fusion `fc`, and the sum waits for the rest.
//! - **Inject**, after the model's last layer ([`Drafter::inject`]): the
//!   fused rows are normed and every layer's key and value of them go into
//!   the ring of the row's sequence at the row's position. Every row of a step
//!   is injected, the draft rows of a window included: the ring has room for
//!   `window + rows of a block` cells, so a rejected row's cell lies at a
//!   position that the next pass does not read and the row that is accepted
//!   there later overwrites it.
//! - **Pass** ([`Drafter::pass_layers`], the model's head, then
//!   [`Drafter::pass_select`]): the block of the newest token and mask tokens
//!   for each sequence that drafts, the layers, the final norm; the model
//!   multiplies the rows by its draft head (the model's own head view); then
//!   the best candidates per position, the selector and the greedy walk,
//!   whose drafts go straight into the step record's token array and into the
//!   list the host reads.
//!
//! The products are the weight kernels of [`Kernels`] over the matrices as the
//! file stores them, with their activations in 8 bits (the decode kernels up
//! to 8 rows, the tile kernels above).

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    DFLASH_HEAD_DIM, DFLASH_MAX_BLOCK, DFLASH_MAX_RANK, DFLASH_MAX_TOP_K, DeviceBuffer, DevicePtr,
    DflashKernels, DflashSelect, DflashSeq, Event, Kernels, MAX_BATCH, PinnedBuffer, PrefillPath,
    Ring, Selector, Stream,
};
use ares_runtime::ModelError;

use super::config::{Config, Matrix, Vector, Weights};

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

/// A matrix in the drafter's arena, as the file stores it.
#[derive(Clone, Copy, Debug)]
struct DMat {
    ty: GgmlType,
    n_in: u32,
    n_out: u32,
    at: u64,
}

#[derive(Clone, Debug)]
struct DLayer {
    attn_norm: u64,
    attn_conv_proj: DMat,
    attn_conv_base: u64,
    q: DMat,
    k: DMat,
    v: DMat,
    q_norm: u64,
    k_norm: u64,
    out: DMat,
    ffn_norm: u64,
    ffn_conv_proj: DMat,
    ffn_conv_base: u64,
    gate: DMat,
    up: DMat,
    down: DMat,
}

/// Where the drafter's tensors are in its arena.
#[derive(Clone, Debug)]
struct DWeights {
    /// The fusion, one matrix per tap (the columns of `fc` that read it).
    fc: Vec<DMat>,
    fc_out_norm: u64,
    layers: Vec<DLayer>,
    output_norm: u64,
    selector_hidden: DMat,
    successor: u64,
    predecessor: u64,
    table_ty: GgmlType,
    table_row_bytes: u32,
}

/// Offsets of the scratch.
#[derive(Clone, Copy, Debug, Default)]
struct Scratch {
    // The pass, `pass_rows` rows.
    x: u64,
    xn: u64,
    xc: u64,
    dynamic: u64,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    a: u64,
    ffn_in: u64,
    g1: u64,
    g2: u64,
    h: u64,
    logits: u64,
    cols: u64,
    vals: u64,
    gate: u64,
    lattice: u64,
    drafts: u64,
    probs: u64,
    // The inject, `inject_rows` rows.
    fused: u64,
    y: u64,
    part: u64,
    ik: u64,
    iv: u64,
    // The activations of a product, for the most rows of either.
    act: u64,
    len: u64,
}

/// The ring of one slot on the host's side: which positions it holds cells of.
#[derive(Clone, Copy, Debug, Default)]
struct SlotRing {
    /// The first position injected since the sequence took the slot (or since
    /// it resumed somewhere else), and one past the last one.
    from: u32,
    end: u32,
}

/// A sequence that drafts in a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassSeq {
    pub slot: u32,
    /// The position of the newest token, the first row of the block.
    pub position: u32,
    pub newest: u32,
    /// The row of the step record that takes the first draft, or `u32::MAX`.
    pub record_row: u32,
    /// The drafts the window takes, at most a block's.
    pub wanted: u32,
    /// Where its drafts go in the list the host reads.
    pub place: u32,
}

/// What the drafter takes on its device, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrafterBytes {
    pub weights: u64,
    pub ring: u64,
    pub scratch: u64,
    pub records: u64,
}

impl DrafterBytes {
    pub fn total(&self) -> u64 {
        self.weights + self.ring + self.scratch + self.records
    }
}

/// How the drafter is sized on its device.
#[derive(Clone, Copy, Debug)]
pub struct DrafterSetup {
    /// Sequence slots, each with a ring.
    pub max_slots: u32,
    /// The most rows of a step: what one inject takes.
    pub max_rows: u32,
    /// The most rows of a window: the ring keeps this many cells past the
    /// window so that a rejected row is never read.
    pub window_rows: u32,
    /// The rows of the model's draft head, which the logits of a pass hold.
    pub head_rows: u32,
}

/// The sizes of the drafter on a device, before anything is allocated: what a
/// placement counts in its VRAM plan.
pub fn bytes_of(c: &Config, w: &Weights<'_>, setup: &DrafterSetup) -> DrafterBytes {
    let (weights, ring, cells) = arena_and_ring(c, w, setup);
    let _ = cells;
    let s = scratch(c, setup);
    DrafterBytes { weights, ring, scratch: s.len, records: record_bytes(c, setup) }
}

fn ring_cells(c: &Config, setup: &DrafterSetup) -> u32 {
    (c.window + c.block.max(setup.window_rows as usize)) as u32
}

fn arena_and_ring(c: &Config, w: &Weights<'_>, setup: &DrafterSetup) -> (u64, u64, u32) {
    let cells = ring_cells(c, setup);
    let kv = (c.n_head_kv * c.head_dim) as u32;
    let ring = u64::from(setup.max_slots) * Ring::slot_bytes(c.n_layer as u32, cells, kv);
    // Every tensor at a multiple of 256 bytes, the fusion in one part per tap.
    let round = |bytes: usize| (bytes as u64).next_multiple_of(256);
    let taps = c.target_layers.len();
    let matrices: u64 = w.matrices().iter().skip(1).map(|m| round(m.data.len())).sum::<u64>()
        + taps as u64 * round(w.fc.data.len() / taps);
    let vectors: u64 = w
        .layers
        .iter()
        .flat_map(|l| {
            [&l.attn_norm, &l.attn_conv_base, &l.q_norm, &l.k_norm, &l.ffn_norm, &l.ffn_conv_base]
        })
        .chain([&w.fc_out_norm, &w.output_norm])
        .map(|v| round(v.data.len()))
        .sum();
    (matrices + vectors, ring, cells)
}

fn record_bytes(c: &Config, setup: &DrafterSetup) -> u64 {
    let pass_rows = u64::from(setup.max_slots) * c.block as u64;
    // Per row of a step its slot and position; per pass row its position and
    // its input (the embedding); per sequence two small records.
    (u64::from(setup.max_rows) * 8
        + pass_rows * 4
        + pass_rows * c.n_embd as u64 * 4
        + u64::from(setup.max_slots) * 32
        + 1024)
        .next_multiple_of(256)
}

fn scratch(c: &Config, setup: &DrafterSetup) -> Scratch {
    let (d, hd) = (c.n_embd as u64, c.head_dim as u64);
    let pass = u64::from(setup.max_slots) * c.block as u64;
    let drafted = u64::from(setup.max_slots) * c.drafts() as u64;
    let inject = u64::from(setup.max_rows);
    let (qd, kv, ff) = (c.n_head as u64 * hd, c.n_head_kv as u64 * hd, c.n_ff as u64);
    let k = c.top_k as u64;
    let mut s = Scratch::default();
    let mut len = 0u64;
    let mut take = |bytes: u64| {
        let at = len;
        len = (len + bytes.max(16)).next_multiple_of(256);
        at
    };
    s.x = take(pass * d * 4);
    s.xn = take(pass * d * 4);
    s.xc = take(pass * d * 4);
    s.dynamic = take(pass * c.conv_values() as u64 * 4);
    s.q = take(pass * qd * 4);
    s.k = take(pass * kv * 4);
    s.v = take(pass * kv * 4);
    s.o = take(pass * qd * 4);
    s.a = take(pass * d * 4);
    s.ffn_in = take(pass * d * 4);
    s.g1 = take(pass * ff * 4);
    s.g2 = take(pass * ff * 4);
    s.h = take(drafted * d * 4);
    s.logits = take(drafted * u64::from(setup.head_rows) * 4);
    s.cols = take(drafted * k * 4);
    s.vals = take(drafted * k * 4);
    s.gate = take(drafted * c.selector_rank as u64 * 4);
    s.lattice = take(drafted * (k + k * k) * 4);
    s.drafts = take(drafted * 4);
    s.probs = take(drafted * 4);
    s.fused = take(inject * d * 4);
    s.y = take(inject * d * 4);
    s.part = take(inject * d * 4);
    s.ik = take(inject * kv * 4);
    s.iv = take(inject * kv * 4);
    let widest = [d, ff, qd, (c.tap_values() as u64)].into_iter().max().unwrap_or(d) as u32;
    let rows = pass.max(inject) as u32;
    let act = Kernels::quantized_activation_bytes(widest, MAX_BATCH)
        .max(Kernels::prefill_activation_bytes(PrefillPath::Int8, widest, rows))
        .max(Kernels::prefill_activation_bytes(PrefillPath::F16, widest, rows));
    s.act = take(act);
    s.len = len;
    s
}

/// The DFlash drafter on one device.
#[derive(Debug)]
pub struct Drafter {
    c: Config,
    kernels: DflashKernels,
    w: DWeights,
    arena: DeviceBuffer,
    ring: DeviceBuffer,
    cells: u32,
    scratch: DeviceBuffer,
    s: Scratch,
    host: PinnedBuffer,
    records: DeviceBuffer,
    /// Where the records of the inject and of the pass are in `records`.
    inject_slots: u64,
    inject_positions: u64,
    pass_positions: u64,
    pass_x: u64,
    pass_seqs: u64,
    pass_select: u64,
    setup: DrafterSetup,
    rings: Vec<SlotRing>,
    /// The embedding of the mask token.
    mask_row: Vec<f32>,
    /// Rows of the step being enqueued, and the taps it has had (a bit per
    /// tap: a step cut into groups of rows gives each tap once per group).
    inject_rows: u32,
    taps: u64,
    /// The sequences of the pass being enqueued.
    pass: Vec<PassSeq>,
    /// The drafts' shares of the last pass, copied to the host.
    probs_host: PinnedBuffer,
    /// Drafts of another drafter on their way into the list.
    put_host: PinnedBuffer,
    drafted: u32,
    clock: [Event; 2],
    clock_pending: bool,
    last_ms: f64,
    /// The layers a pass runs: all but in tests ([`Drafter::run_layers`]).
    layers_run: usize,
}

impl Drafter {
    /// Uploads the drafter's weights and reserves its ring and scratch on the
    /// device, in `ledger` first: a drafter that does not fit is refused with
    /// what holds the device's memory. `mask_row` is the model's embedding of
    /// the mask token.
    pub fn open(
        c: &Config,
        w: &Weights<'_>,
        device: &ares_cuda::Device,
        ledger: &mut Ledger,
        setup: DrafterSetup,
        mask_row: Vec<f32>,
    ) -> Result<Self, ModelError> {
        if c.head_dim as u32 != DFLASH_HEAD_DIM
            || c.block as u32 > DFLASH_MAX_BLOCK
            || c.top_k as u32 > DFLASH_MAX_TOP_K
            || c.selector_rank as u32 > DFLASH_MAX_RANK
        {
            return Err(invalid(format!(
                "a DFlash drafter with heads of {}, blocks of {}, {} candidates and a selector of \
                 rank {}: the kernels take heads of {DFLASH_HEAD_DIM}, blocks up to \
                 {DFLASH_MAX_BLOCK}, up to {DFLASH_MAX_TOP_K} candidates and a rank up to \
                 {DFLASH_MAX_RANK}",
                c.head_dim, c.block, c.top_k, c.selector_rank
            )));
        }
        if mask_row.len() != c.n_embd {
            return Err(invalid("the mask token's row has another width than the drafter"));
        }
        let (weights, ring_bytes, cells) = arena_and_ring(c, w, &setup);
        let s = scratch(c, &setup);
        let records = record_bytes(c, &setup);
        ledger.reserve("dflash drafter: weights", weights)?;
        ledger.reserve("dflash drafter: keys and values of the last window", ring_bytes)?;
        ledger.reserve("dflash drafter: scratch", s.len)?;
        ledger.reserve("dflash drafter: step records", records)?;

        // The arena, laid out on the host and uploaded in one copy per tensor.
        let arena = DeviceBuffer::new(device, weights)?;
        let mut end = 0u64;
        let mut put = |bytes: &[u8]| -> Result<u64, ModelError> {
            let at = end;
            arena.write(at, bytes)?;
            end = (at + bytes.len() as u64).next_multiple_of(256);
            Ok(at)
        };
        let mut mat = |m: &Matrix<'_>| -> Result<DMat, ModelError> {
            Ok(DMat { ty: m.ty, n_in: m.n_in as u32, n_out: m.n_out as u32, at: put(m.data)? })
        };
        // The fusion in one matrix per tap: the columns of each tap are whole
        // blocks of every row.
        let fc = &w.fc;
        let row_bytes = fc.data.len() / fc.n_out;
        let taps = c.target_layers.len();
        let seg = row_bytes / taps;
        if seg * taps != row_bytes || !c.n_embd.is_multiple_of(fc.ty.block_values() as usize) {
            return Err(invalid("the fusion's rows do not split into whole blocks per tap"));
        }
        let mut fc_parts = Vec::with_capacity(taps);
        for t in 0..taps {
            let mut bytes = Vec::with_capacity(seg * fc.n_out);
            for r in 0..fc.n_out {
                bytes.extend_from_slice(
                    &fc.data[r * row_bytes + t * seg..r * row_bytes + (t + 1) * seg],
                );
            }
            let part =
                Matrix { name: format!("fc tap {t}"), n_in: c.n_embd, data: &bytes, ..fc.clone() };
            fc_parts.push(mat(&part)?);
        }
        let selector_hidden = mat(&w.selector_hidden)?;
        let mut layers = Vec::with_capacity(c.n_layer);
        for l in &w.layers {
            layers.push(DLayer {
                attn_conv_proj: mat(&l.attn_conv_proj)?,
                q: mat(&l.q)?,
                k: mat(&l.k)?,
                v: mat(&l.v)?,
                out: mat(&l.out)?,
                ffn_conv_proj: mat(&l.ffn_conv_proj)?,
                gate: mat(&l.gate)?,
                up: mat(&l.up)?,
                down: mat(&l.down)?,
                attn_norm: 0,
                attn_conv_base: 0,
                q_norm: 0,
                k_norm: 0,
                ffn_norm: 0,
                ffn_conv_base: 0,
            });
        }
        drop(mat);
        let mut vector = |v: &Vector<'_>| put(&v.data);
        for (dl, l) in layers.iter_mut().zip(&w.layers) {
            dl.attn_norm = vector(&l.attn_norm)?;
            dl.attn_conv_base = vector(&l.attn_conv_base)?;
            dl.q_norm = vector(&l.q_norm)?;
            dl.k_norm = vector(&l.k_norm)?;
            dl.ffn_norm = vector(&l.ffn_norm)?;
            dl.ffn_conv_base = vector(&l.ffn_conv_base)?;
        }
        let fc_out_norm = vector(&w.fc_out_norm)?;
        let output_norm = vector(&w.output_norm)?;
        let successor = put(w.selector_successor.data)?;
        let predecessor = put(w.selector_predecessor.data)?;
        let table = &w.selector_successor;
        if w.selector_predecessor.ty != table.ty {
            return Err(invalid("the selector's two tables are of different types"));
        }
        let dw = DWeights {
            fc: fc_parts,
            fc_out_norm,
            layers,
            output_norm,
            selector_hidden,
            successor,
            predecessor,
            table_ty: table.ty,
            table_row_bytes: (table.data.len() / table.n_out) as u32,
        };
        let ring = DeviceBuffer::new(device, ring_bytes)?;
        let scratch_buf = DeviceBuffer::new(device, s.len)?;
        // Records: the inject's rows, then the pass's.
        let pass_rows = u64::from(setup.max_slots) * c.block as u64;
        let inject_slots = 0;
        let inject_positions = (u64::from(setup.max_rows) * 4).next_multiple_of(256);
        let pass_positions =
            inject_positions + (u64::from(setup.max_rows) * 4).next_multiple_of(256);
        let pass_x = pass_positions + (pass_rows * 4).next_multiple_of(256);
        let pass_seqs = pass_x + (pass_rows * c.n_embd as u64 * 4).next_multiple_of(256);
        let pass_select = pass_seqs
            + (u64::from(setup.max_slots) * DflashSeq::BYTES as u64).next_multiple_of(256);
        if pass_select + u64::from(setup.max_slots) * DflashSelect::BYTES as u64 > records {
            return Err(invalid("the drafter's records do not fit their buffer"));
        }
        let drafted = setup.max_slots as usize * c.drafts();
        Ok(Self {
            c: c.clone(),
            kernels: DflashKernels::load(device)?,
            w: dw,
            arena,
            ring,
            cells,
            scratch: scratch_buf,
            s,
            host: PinnedBuffer::new(device, records as usize)?,
            records: DeviceBuffer::new(device, records)?,
            inject_slots,
            inject_positions,
            pass_positions,
            pass_x,
            pass_seqs,
            pass_select,
            setup,
            rings: vec![SlotRing::default(); setup.max_slots as usize],
            mask_row,
            inject_rows: 0,
            taps: 0,
            pass: Vec::new(),
            probs_host: PinnedBuffer::new(device, drafted * 4)?,
            put_host: PinnedBuffer::new(device, drafted * 4)?,
            drafted: 0,
            clock: [Event::timed(device)?, Event::timed(device)?],
            clock_pending: false,
            last_ms: 0.0,
            layers_run: c.n_layer,
        })
    }

    pub fn config(&self) -> &Config {
        &self.c
    }

    /// What the drafter takes on its device.
    pub fn bytes(&self) -> DrafterBytes {
        DrafterBytes {
            weights: self.arena.len(),
            ring: self.ring.len(),
            scratch: self.scratch.len(),
            records: self.records.len(),
        }
    }

    fn at(&self, offset: u64) -> DevicePtr {
        self.scratch.ptr().offset(offset)
    }

    fn rec(&self, offset: u64) -> DevicePtr {
        self.records.ptr().offset(offset)
    }

    fn weight(&self, offset: u64) -> DevicePtr {
        self.arena.ptr().offset(offset)
    }

    /// `y = m x` for `rows` rows of activations at `x`.
    fn product(
        &self,
        kernels: &Kernels,
        stream: &Stream,
        m: &DMat,
        x: DevicePtr,
        rows: u32,
        y: DevicePtr,
    ) -> Result<(), ModelError> {
        self.product_at(kernels, stream, m.ty, self.weight(m.at), m.n_in, m.n_out, x, rows, y)
    }

    /// `y = w x` for a matrix `w` of type `ty` as the file stores it, `rows`
    /// rows of activations at `x`: the decode kernels up to 8 rows, the tile
    /// kernels above.
    #[allow(clippy::too_many_arguments)]
    fn product_at(
        &self,
        kernels: &Kernels,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        n_in: u32,
        n_out: u32,
        x: DevicePtr,
        rows: u32,
        y: DevicePtr,
    ) -> Result<(), ModelError> {
        let act = self.at(self.s.act);
        // SAFETY: `x` holds `rows` rows of `n_in` f32 and `y` of `n_out`, the
        // activations' scratch the bytes of the widest input at the most rows;
        // the matrix is on the device as the file stores it.
        unsafe {
            if rows <= MAX_BATCH {
                kernels.quantize_activations(stream, x, act, n_in, rows)?;
                kernels.matvec(stream, ty, w, act, y, n_in, n_out, rows)?;
            } else {
                // The integer path where the type has one; the f16 path (exact weights, f16
                // rows) for the floats and the types without.
                let path = if ty.is_quantized() && kernels.has_prefill_path(ty, PrefillPath::Int8) {
                    PrefillPath::Int8
                } else {
                    PrefillPath::F16
                };
                kernels.prepare_prefill_activations(stream, path, x, act, n_in, rows)?;
                kernels.matmul(stream, ty, path, w, act, y, n_in, n_out, rows)?;
            }
        }
        Ok(())
    }

    /// The model's draft head over the rows [`Drafter::pass_layers`] returned,
    /// for a model whose head lies as the file stores it: `w` of type `ty`, a
    /// row of `n_embd` values per column of the logits.
    pub fn head_product(
        &self,
        kernels: &Kernels,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        h: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        let d = self.c.n_embd as u32;
        self.product_at(kernels, stream, ty, w, d, self.setup.head_rows, h, rows, self.logits())
    }

    /// The start of a step of the model: its rows, as `(slot, position)` in
    /// the order of the step, whose taps follow. Every row is injected; of a
    /// sequence with more rows in the step than its ring holds cells, only the
    /// last that many.
    pub fn begin_step(&mut self, stream: &Stream, rows: &[(u32, u32)]) -> Result<(), ModelError> {
        let n = rows.len();
        if n > self.setup.max_rows as usize {
            return Err(invalid(format!(
                "a step of {n} rows; the drafter takes {}",
                self.setup.max_rows
            )));
        }
        // The last position of each slot in the step.
        let mut last: Vec<(u32, u32)> = Vec::new();
        for &(slot, position) in rows {
            if slot >= self.setup.max_slots {
                return Err(invalid(format!("slot {slot} of {}", self.setup.max_slots)));
            }
            match last.iter_mut().find(|(s, _)| *s == slot) {
                Some((_, p)) => *p = (*p).max(position),
                None => last.push((slot, position)),
            }
        }
        // The host's view of each ring: a segment that does not continue the
        // cells the ring holds (a new sequence, a sequence that resumed from the
        // prefix cache) starts it again. A window's rows start at most a window
        // before its end: that is a draft window going on after a rejection.
        let back = self.back();
        for &(slot, position) in rows {
            let ring = &mut self.rings[slot as usize];
            if !(position <= ring.end && position + back >= ring.end) {
                *ring = SlotRing { from: position, end: position };
            }
            ring.end = position + 1;
        }
        let cells = self.cells;
        let words = |at: u64, values: &mut dyn Iterator<Item = u32>, host: &mut PinnedBuffer| {
            let mut at = at as usize;
            for v in values {
                host[at..at + 4].copy_from_slice(&v.to_le_bytes());
                at += 4;
            }
        };
        let skip = |slot: u32, position: u32| {
            let end = last.iter().find(|(s, _)| *s == slot).map_or(0, |(_, p)| *p);
            // Two rows of a step at the same cell: only the later one is
            // written.
            if position + cells <= end { u32::MAX } else { slot }
        };
        let (slots_at, positions_at) = (self.inject_slots, self.inject_positions);
        words(slots_at, &mut rows.iter().map(|&(s, p)| skip(s, p)), &mut self.host);
        words(positions_at, &mut rows.iter().map(|&(_, p)| p), &mut self.host);
        // SAFETY: both ranges are in the pinned buffer and in the device's
        // records; the pinned bytes stay unchanged until the step has run (the
        // pass of a step that only drafts writes other bytes).
        unsafe {
            stream.copy_from_host(
                self.rec(slots_at),
                self.host.as_ptr().add(slots_at as usize),
                n * 4,
            )?;
            stream.copy_from_host(
                self.rec(positions_at),
                self.host.as_ptr().add(positions_at as usize),
                n * 4,
            )?;
        }
        self.inject_rows = n as u32;
        self.taps = 0;
        Ok(())
    }

    /// The tap of target layer number `tap` (its place in the file's list):
    /// rows `first .. first + rows` of the step, the mean of `streams` streams
    /// at `src` (`stream_pitch` f32 between streams, `row_pitch` between rows;
    /// one stream with rows of `n_embd` is read as it is).
    #[allow(clippy::too_many_arguments)]
    pub fn tap(
        &mut self,
        kernels: &Kernels,
        stream: &Stream,
        tap: usize,
        src: DevicePtr,
        streams: u32,
        stream_pitch: u64,
        row_pitch: u32,
        first: u32,
        rows: u32,
    ) -> Result<(), ModelError> {
        let d = self.c.n_embd as u32;
        if tap >= self.w.fc.len() || first + rows > self.inject_rows {
            return Err(invalid(format!(
                "tap {tap} of rows {first}..{} in a step of {} rows",
                first + rows,
                self.inject_rows
            )));
        }
        let row_off = |base: u64| self.at(base).offset(u64::from(first) * u64::from(d) * 4);
        let input = if streams == 1 && row_pitch == d {
            src
        } else {
            // SAFETY: the stream rows are the caller's; `y` holds the rows.
            unsafe {
                self.kernels.tap(
                    stream,
                    src,
                    streams,
                    stream_pitch,
                    row_pitch,
                    row_off(self.s.y),
                    d,
                    d,
                    rows,
                )?;
            }
            row_off(self.s.y)
        };
        let fc = self.w.fc[tap];
        if tap == 0 {
            self.product(kernels, stream, &fc, input, rows, row_off(self.s.fused))?;
        } else {
            let part = row_off(self.s.part);
            self.product(kernels, stream, &fc, input, rows, part)?;
            // SAFETY: both hold `rows * d` f32.
            unsafe { kernels.add_f32(stream, row_off(self.s.fused), part, rows * d)? };
        }
        self.taps |= 1 << tap;
        Ok(())
    }

    /// After the model's last layer: the fused rows into every layer's ring.
    pub fn inject(&mut self, kernels: &Kernels, stream: &Stream) -> Result<(), ModelError> {
        let n = self.inject_rows;
        if n == 0 {
            return Ok(());
        }
        if self.taps.count_ones() as usize != self.w.fc.len() {
            return Err(invalid(format!(
                "the step gave {} taps of the {} the drafter reads",
                self.taps.count_ones(),
                self.w.fc.len()
            )));
        }
        let c = &self.c;
        let d = c.n_embd as u32;
        let (fused, g) = (self.at(self.s.fused), self.at(self.s.y));
        let (k, v) = (self.at(self.s.ik), self.at(self.s.iv));
        let ring = self.ring_view();
        // SAFETY: the scratch holds the step's rows of every buffer used; the
        // records the rows' slots and positions (`begin_step`).
        unsafe {
            kernels.rms_norm_f32(stream, fused, self.weight(self.w.fc_out_norm), g, d, n, c.eps)?;
        }
        for (l, layer) in self.w.layers.iter().enumerate() {
            self.product(kernels, stream, &layer.k, g, n, k)?;
            self.product(kernels, stream, &layer.v, g, n, v)?;
            // SAFETY: as above.
            unsafe {
                self.kernels.head_norm_rope(
                    stream,
                    k,
                    Some(self.weight(layer.k_norm)),
                    c.n_head_kv as u32,
                    c.head_dim as u32,
                    self.rec(self.inject_positions),
                    c.rope_base,
                    c.eps,
                    n,
                )?;
                self.kernels.ring_write(
                    stream,
                    k,
                    v,
                    &ring,
                    self.rec(self.inject_slots),
                    self.rec(self.inject_positions),
                    l as u32,
                    n,
                )?;
            }
        }
        self.inject_rows = 0;
        Ok(())
    }

    /// How far before the end of its ring a sequence's rows may start and
    /// still go on from it: the rows of a window, after a rejection.
    fn back(&self) -> u32 {
        self.setup.window_rows.max(self.c.block as u32)
    }

    fn ring_view(&self) -> Ring {
        Ring {
            ptr: self.ring.ptr(),
            layers: self.c.n_layer as u32,
            cells: self.cells,
            kv: (self.c.n_head_kv * self.c.head_dim) as u32,
        }
    }

    /// The first part of a pass: the blocks of `seqs` through the layers and
    /// the final norm. `embed` is the model's embedding of a token, as f32.
    /// Returns where the rows that make drafts are (`block - 1` per sequence,
    /// rows of `n_embd`) and how many: the model multiplies them by its draft
    /// head into [`Drafter::logits`].
    pub fn pass_layers(
        &mut self,
        kernels: &Kernels,
        stream: &Stream,
        seqs: &[PassSeq],
        embed: &dyn Fn(u32) -> Vec<f32>,
    ) -> Result<(DevicePtr, u32), ModelError> {
        let c = self.c.clone();
        let (b, d) = (c.block as u32, c.n_embd as u32);
        if seqs.len() > self.setup.max_slots as usize {
            return Err(invalid("more sequences draft than the drafter has slots"));
        }
        if let Some(bad) =
            seqs.iter().find(|s| s.wanted as usize > c.drafts() || s.slot >= self.setup.max_slots)
        {
            return Err(invalid(format!("a pass of {bad:?}; a block makes {} drafts", c.drafts())));
        }
        self.clock[0].record(stream)?;
        self.pass = seqs.to_vec();
        let n = seqs.len() as u32;
        let rows = n * b;
        // The records: positions, inputs, sequences.
        let back = self.back();
        {
            let host: &mut [u8] = &mut self.host;
            let mut at = self.pass_positions as usize;
            for s in seqs {
                for r in 0..b {
                    host[at..at + 4].copy_from_slice(&(s.position + r).to_le_bytes());
                    at += 4;
                }
            }
            let mut at = self.pass_x as usize;
            for s in seqs {
                let row = embed(s.newest);
                if row.len() != c.n_embd {
                    return Err(invalid("the model's embedding row has another width"));
                }
                for r in 0..b {
                    let values = if r == 0 { &row } else { &self.mask_row };
                    for v in values {
                        host[at..at + 4].copy_from_slice(&v.to_le_bytes());
                        at += 4;
                    }
                }
            }
            let mut at = self.pass_seqs as usize;
            for s in seqs {
                let ring = self.rings[s.slot as usize];
                // A ring that does not reach the newest position, or reaches
                // further than a window past it, holds nothing of this
                // sequence's last cells.
                let holds = s.position <= ring.end && s.position + back >= ring.end;
                let from = if holds { ring.from } else { s.position };
                let seq = DflashSeq { slot: s.slot, position: s.position, from };
                host[at..at + DflashSeq::BYTES].copy_from_slice(&seq.to_bytes());
                at += DflashSeq::BYTES;
            }
            let mut at = self.pass_select as usize;
            for s in seqs {
                let sel = DflashSelect {
                    newest: s.newest,
                    record_row: s.record_row,
                    wanted: s.wanted,
                    place: s.place,
                };
                host[at..at + DflashSelect::BYTES].copy_from_slice(&sel.to_bytes());
                at += DflashSelect::BYTES;
            }
        }
        let from = self.pass_positions;
        let end = self.pass_select + u64::from(n) * DflashSelect::BYTES as u64;
        // SAFETY: the range is inside both buffers; the pinned bytes stay
        // unchanged until the step has run.
        unsafe {
            stream.copy_from_host(
                self.rec(from),
                self.host.as_ptr().add(from as usize),
                (end - from) as usize,
            )?;
            stream.copy(
                self.at(self.s.x),
                self.rec(self.pass_x),
                rows as usize * d as usize * 4,
            )?;
        }
        let ring = self.ring_view();
        let s = self.s;
        let (x, xn, xc, dynamic) = (self.at(s.x), self.at(s.xn), self.at(s.xc), self.at(s.dynamic));
        let (q, k, v, o, a) =
            (self.at(s.q), self.at(s.k), self.at(s.v), self.at(s.o), self.at(s.a));
        let (ffn_in, g1, g2) = (self.at(s.ffn_in), self.at(s.g1), self.at(s.g2));
        let positions = self.rec(self.pass_positions);
        let group = c.conv_group as u32;
        let kernel = c.conv_kernel as u32;
        for (l, layer) in self.w.layers.iter().enumerate().take(self.layers_run) {
            // SAFETY: every buffer holds `rows` rows of its width (sized for
            // every slot drafting at once); the arena holds the weights.
            unsafe {
                kernels.rms_norm_f32(
                    stream,
                    x,
                    self.weight(layer.attn_norm),
                    xn,
                    d,
                    rows,
                    c.eps,
                )?;
                self.product(kernels, stream, &layer.attn_conv_proj, xn, rows, dynamic)?;
                let base = self.weight(layer.attn_conv_base);
                self.kernels.conv(stream, xn, dynamic, base, xc, d, group, kernel, 0, b, rows)?;
                self.product(kernels, stream, &layer.q, xc, rows, q)?;
                self.product(kernels, stream, &layer.k, xc, rows, k)?;
                self.product(kernels, stream, &layer.v, xc, rows, v)?;
                let (hd, base_r) = (c.head_dim as u32, c.rope_base);
                let qn = Some(self.weight(layer.q_norm));
                let kn = Some(self.weight(layer.k_norm));
                self.kernels.head_norm_rope(
                    stream,
                    q,
                    qn,
                    c.n_head as u32,
                    hd,
                    positions,
                    base_r,
                    c.eps,
                    rows,
                )?;
                self.kernels.head_norm_rope(
                    stream,
                    k,
                    kn,
                    c.n_head_kv as u32,
                    hd,
                    positions,
                    base_r,
                    c.eps,
                    rows,
                )?;
                self.kernels.attend(
                    stream,
                    q,
                    k,
                    v,
                    &ring,
                    o,
                    self.rec(self.pass_seqs),
                    l as u32,
                    c.window as u32,
                    b,
                    c.n_head as u32,
                    c.n_head_kv as u32,
                    n,
                )?;
                self.product(kernels, stream, &layer.out, o, rows, xn)?;
                self.kernels.conv(stream, xn, dynamic, base, a, d, group, kernel, 1, b, rows)?;
                // The feed-forward part reads x + attention.
                stream.copy(ffn_in, x, rows as usize * d as usize * 4)?;
                kernels.add_f32(stream, ffn_in, a, rows * d)?;
                kernels.rms_norm_f32(
                    stream,
                    ffn_in,
                    self.weight(layer.ffn_norm),
                    xn,
                    d,
                    rows,
                    c.eps,
                )?;
                self.product(kernels, stream, &layer.ffn_conv_proj, xn, rows, dynamic)?;
                let fbase = self.weight(layer.ffn_conv_base);
                self.kernels.conv(stream, xn, dynamic, fbase, xc, d, group, kernel, 0, b, rows)?;
                self.product(kernels, stream, &layer.gate, xc, rows, g1)?;
                self.product(kernels, stream, &layer.up, xc, rows, g2)?;
                let ff = c.n_ff as u32;
                self.kernels.swiglu(stream, g1, g2, g1, rows * ff)?;
                self.product(kernels, stream, &layer.down, g1, rows, xn)?;
                self.kernels.conv(stream, xn, dynamic, fbase, a, d, group, kernel, 1, b, rows)?;
                stream.copy(x, ffn_in, rows as usize * d as usize * 4)?;
                kernels.add_f32(stream, x, a, rows * d)?;
            }
        }
        // The final norm of the rows that make drafts: rows 1 .. block of
        // each block, gathered.
        let drafted = c.drafts() as u32;
        let h = self.at(s.h);
        for i in 0..n {
            // SAFETY: as above; `h` holds `drafted` rows per sequence.
            unsafe {
                let src = x.offset((u64::from(i) * u64::from(b) + 1) * u64::from(d) * 4);
                let dst = h.offset(u64::from(i) * u64::from(drafted) * u64::from(d) * 4);
                kernels.rms_norm_f32(
                    stream,
                    src,
                    self.weight(self.w.output_norm),
                    dst,
                    d,
                    drafted,
                    c.eps,
                )?;
            }
        }
        Ok((h, n * drafted))
    }

    /// Where the model leaves the draft head's logits of the rows
    /// [`Drafter::pass_layers`] returned, `head_rows` f32 per row.
    pub fn logits(&self) -> DevicePtr {
        self.at(self.s.logits)
    }

    /// The second part of a pass, after the model's head: the candidates, the
    /// selector and the walk. The drafts go into `record_tokens` (the step
    /// record's token array on this device, at each sequence's `record_row`)
    /// and into `drafts` at each sequence's `place`; `head_tokens` is the token
    /// of each column of the head (null: column `i` is token `i`).
    pub fn pass_select(
        &mut self,
        kernels: &Kernels,
        stream: &Stream,
        head_tokens: DevicePtr,
        record_tokens: DevicePtr,
        drafts: DevicePtr,
    ) -> Result<(), ModelError> {
        let c = self.c.clone();
        let n = self.pass.len() as u32;
        let drafted = c.drafts() as u32;
        let rows = n * drafted;
        let s = self.s;
        let k = c.top_k as u32;
        // SAFETY: the logits hold `rows` rows of `head_rows`; the other
        // buffers are sized for every slot.
        unsafe {
            self.kernels.top(
                stream,
                self.at(s.logits),
                self.setup.head_rows,
                self.setup.head_rows,
                k,
                self.at(s.cols),
                self.at(s.vals),
                rows,
            )?;
        }
        self.product(
            kernels,
            stream,
            &self.w.selector_hidden,
            self.at(s.h),
            rows,
            self.at(s.gate),
        )?;
        let selector = Selector {
            successor: self.weight(self.w.successor),
            predecessor: self.weight(self.w.predecessor),
            ty: self.w.table_ty.id(),
            row_bytes: self.w.table_row_bytes,
            rank: c.selector_rank as u32,
            top_k: k,
        };
        // SAFETY: as above; the record's tokens and the list are the caller's.
        unsafe {
            self.kernels.select(
                stream,
                self.at(s.cols),
                self.at(s.vals),
                self.at(s.gate),
                head_tokens,
                &selector,
                drafted,
                self.rec(self.pass_select),
                self.at(s.lattice),
                record_tokens,
                drafts,
                self.at(s.probs),
                n,
            )?;
            // The shares, to where the host reads them after the step.
            let total = self.pass.iter().map(|p| p.place + p.wanted).max().unwrap_or(0);
            stream.copy_to_host(
                self.probs_host.as_mut_ptr(),
                self.at(s.probs),
                total as usize * 4,
            )?;
            self.drafted = total;
        }
        self.clock[1].record(stream)?;
        self.clock_pending = true;
        Ok(())
    }

    /// The share of its softmax the drafter gave the draft at `place` of the
    /// list, for the last pass. The step has ended.
    pub fn prob(&self, place: u32) -> f32 {
        if place >= self.drafted {
            return f32::NAN;
        }
        let at = place as usize * 4;
        let b = &self.probs_host[at..at + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// The drafts of the last pass in the list: one past the last place.
    pub fn drafted(&self) -> u32 {
        self.drafted
    }

    /// A list of drafts of the drafter's own, for a model whose drafts have no
    /// other list: a draft per draft row of every slot.
    pub fn own_list(&self) -> DevicePtr {
        self.at(self.s.drafts)
    }

    /// Puts drafts another drafter of the model made into the list at
    /// `place`, so that a step with windows of both has one list. After the
    /// pass of the step.
    pub fn put_drafts(
        &mut self,
        stream: &Stream,
        place: u32,
        drafts: &[u32],
    ) -> Result<(), ModelError> {
        let at = place as usize * 4;
        let end = at + drafts.len() * 4;
        if end > self.put_host.len() || end > self.s.probs as usize - self.s.drafts as usize {
            return Err(invalid("drafts past the drafter's list"));
        }
        for (i, t) in drafts.iter().enumerate() {
            self.put_host[at + i * 4..at + i * 4 + 4].copy_from_slice(&t.to_le_bytes());
        }
        // SAFETY: the range is inside the pinned buffer and the list; the
        // pinned bytes stay until the step has run.
        unsafe {
            stream.copy_from_host(
                self.at(self.s.drafts).offset(at as u64),
                self.put_host.as_ptr().add(at),
                drafts.len() * 4,
            )?;
        }
        self.drafted = self.drafted.max(place + drafts.len() as u32);
        Ok(())
    }

    /// The sequences of the last pass.
    pub fn last_pass(&self) -> &[PassSeq] {
        &self.pass
    }

    /// Clears the pass, so that a step without one reports none.
    pub fn no_pass(&mut self) {
        self.pass.clear();
        self.drafted = 0;
    }

    /// Device milliseconds of the last pass (0 when there was none). The step
    /// has ended.
    pub fn pass_ms(&mut self) -> Result<f64, ModelError> {
        if std::mem::take(&mut self.clock_pending) {
            self.last_ms = f64::from(self.clock[1].elapsed_ms_since(&self.clock[0])?);
        } else if self.pass.is_empty() {
            self.last_ms = 0.0;
        }
        Ok(self.last_ms)
    }

    /// For tests: the scratch, and in it the lattice of the last pass (per
    /// sequence and position, `k` tokens as f32 then `k * k` scores), the final
    /// hidden rows that made drafts and the logits of the head.
    pub fn lattice(&self) -> (&DeviceBuffer, u64, u64, u64) {
        (&self.scratch, self.s.lattice, self.s.h, self.s.logits)
    }

    /// For tests: the ring's cells of `slot`, layer `layer`: the keys then the
    /// values of every cell, f16.
    pub fn ring_of(&self, slot: u32, layer: u32) -> (&DeviceBuffer, u64, u64) {
        let ring = self.ring_view();
        let bytes = u64::from(ring.cells) * u64::from(ring.kv) * 2;
        let at = ((u64::from(slot) * u64::from(ring.layers) + u64::from(layer)) * 2) * bytes;
        (&self.ring, at, bytes)
    }

    /// For tests: where a buffer of the pass is in the scratch (`x`, `q`, `k`,
    /// `v`, `o`, `a`, `ffn_in`, `g1`, `dynamic`, `xc`, `xn`), as it is when
    /// the pass has run: the last layer's.
    pub fn pass_buffer(&self, name: &str) -> Option<u64> {
        let s = &self.s;
        Some(match name {
            "x" => s.x,
            "q" => s.q,
            "k" => s.k,
            "v" => s.v,
            "o" => s.o,
            "a" => s.a,
            "ffn_in" => s.ffn_in,
            "g1" => s.g1,
            "dynamic" => s.dynamic,
            "xc" => s.xc,
            "xn" => s.xn,
            _ => return None,
        })
    }

    /// For tests: passes run only the first `n` layers, as
    /// [`Reference::layers`](super::Reference) does.
    pub fn run_layers(&mut self, n: usize) {
        self.layers_run = n.min(self.c.n_layer);
    }

    /// The cells per slot of the ring.
    pub fn ring_cells(&self) -> u32 {
        self.cells
    }
}
