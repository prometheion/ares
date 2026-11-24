//! A prompt read layer by layer: the part's side of a pass.
//!
//! A step of more rows than the part's own scratch holds reads each expert of a layer once
//! for its rows, so the longer the step the fewer times a prompt reads the file and uploads
//! the experts. A *pass* takes that to a whole span of a prompt (tens of thousands of rows)
//! without a scratch of that many rows: the rows go through one layer in *micro-batches*
//! that the own scratch holds, each up to the layer's router (the read of the stream mixer,
//! the recurrent or attention half, the second read, the router and the shared expert),
//! then the layer's routed experts run once for all rows of the span, then the next layer.
//! The kernels are the wide steps' (`wide.rs`).
//!
//! What a row keeps from one layer to the next lies in VRAM the expert slots lend for the
//! pass, in *chunks* of whole micro-batches (the many-row kernels index their input and
//! output by row, so each buffer of a chunk is one range): the residual streams, the output
//! of the layer's mixture of experts with its inject values (which the next read writes to
//! the streams), the experts' f16 input and the router's picks, and the plan of the
//! many-row kernels; the hidden values of the experts of one chunk are shared by the chunks.
//! About 56 KiB a row on Flash-Next, against the 0.5 to 0.7 MiB of a wide step's scratch.
//! Where the K and V of the context lie partly in host memory (`blocks/far.rs`), the loan
//! also holds a copy of one layer's far slots, which every micro-batch's attention reads in
//! place of the far pool ([`Part::pass_stage`]): a row reads the cells of its own pick, so
//! reading host memory would cross the link once per row.
//!
//! While a micro-batch runs, the scratch's residual, MoE output and inject values are its
//! rows of the chunk ([`Part::at`] maps those ranges), the router writes its picks into the
//! chunk's record (`ExpertKernels::pick_rows_into`), and its experts' input is copied into
//! the chunk's. The output of a micro-batch's last half block stays pending between layers,
//! one per micro-batch. The records, the attention plans and the rows of the n-gram table go
//! to the inbox per micro-batch and layer, from a ring of page-locked buffers.
//!
//! The rounds of a layer's experts (`Part::experts_round`) run, in a pass, over every chunk:
//! the first makes each chunk's plan, each round computes the experts of its table for the
//! rows of every chunk.

use std::ops::Range;

use ares_cuda::{
    DevicePtr, Event, ExpertRowKernels, ExpertShape, Kernels, PinnedBuffer, PrefillPath,
};
use ares_runtime::ModelError;

use super::{Part, Pending, Phase, Role, StepEnv, StepPlan, invalid, wide};
use crate::qwen4exp::config::Config;

/// The most rows of a chunk: what the hidden values of the experts of one chunk are sized
/// for (12.6 KiB a row on Flash-Next), and how many times a round reads its experts' weights
/// in a long pass (once per chunk).
pub const CHUNK_ROWS: u32 = 8192;

/// The most micro-batches of a chunk.
const CHUNK_MICRO: u32 = 16;

/// The most rows of a chunk of a pass in micro-batches of up to `own` rows: [`CHUNK_ROWS`],
/// or sixteen micro-batches where those are fewer rows.
pub fn chunk_rows(own: u32) -> u32 {
    CHUNK_ROWS.min(CHUNK_MICRO * own.max(1))
}

/// Page-locked buffers the inbox of a micro-batch is sent from, in turn.
const RING: usize = 4;

/// The fewest rows of a micro-batch: more than the decode kernels take, so that every
/// micro-batch runs on the many-row kernels.
const MIN_MICRO: u32 = 2 * ares_cuda::MAX_BATCH;

/// The micro-batches of a pass over `rows` rows on parts whose own scratch holds `own` rows:
/// the first row and the rows of each, `own` rows each but for the last two, which share
/// what is left when the last would be narrower than [`MIN_MICRO`].
pub fn micro_batches(rows: u32, own: u32) -> Vec<(u32, u32)> {
    let own = own.max(MIN_MICRO);
    let mut sizes = vec![own; (rows / own) as usize];
    let left = rows % own;
    if left > 0 {
        sizes.push(left);
    }
    let n = sizes.len();
    if n >= 2 && sizes[n - 1] < MIN_MICRO {
        let both = sizes[n - 2] + sizes[n - 1];
        sizes[n - 2] = both - both / 2;
        sizes[n - 1] = both / 2;
    }
    sizes
        .iter()
        .scan(0, |at, rows| {
            let first = *at;
            *at += rows;
            Some((first, *rows))
        })
        .collect()
}

/// [`micro_batches`] of a pass whose rows are cut at `cuts` (rows from its first, ascending,
/// each at least [`MIN_MICRO`] rows from the next and from the ends): no micro-batch crosses a
/// cut, so the pass is at a cut in every layer between two of its micro-batches.
pub fn micro_batches_at(rows: u32, own: u32, cuts: &[u32]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut start = 0;
    for &cut in cuts.iter().chain(std::iter::once(&rows)) {
        if cut <= start || cut > rows {
            continue;
        }
        out.extend(micro_batches(cut - start, own).into_iter().map(|(f, n)| (f + start, n)));
        start = cut;
    }
    out
}

/// The chunks of a pass: consecutive micro-batches of at most `cap` rows together, at least
/// one each; the range of the micro-batches of each.
pub fn chunks_of(micro: &[(u32, u32)], cap: u32) -> Vec<Range<usize>> {
    let mut chunks = Vec::new();
    let (mut start, mut rows) = (0, 0);
    for (m, (_, n)) in micro.iter().enumerate() {
        if m > start && rows + n > cap {
            chunks.push(start..m);
            (start, rows) = (m, 0);
        }
        rows += n;
    }
    if start < micro.len() {
        chunks.push(start..micro.len());
    }
    chunks
}

/// What a pass keeps of its rows on a part, by the part's role.
#[derive(Clone, Copy, Debug)]
pub struct PassSizes {
    /// Whether the part holds the residual streams (all but a worker do).
    streams: bool,
    /// Values of a row of the streams, of a stream, and the streams.
    wide: u64,
    n: u64,
    hc: u64,
    shape: ExpertShape,
    /// The bytes of one layer's converted entry of a page beyond VRAM on the part, 0 when
    /// none is (`blocks/far.rs`): the copy of a layer's far slots takes this per slot.
    far_entry: u64,
}

/// The buffers of a chunk, in the order [`PassSizes::chunk`] gives their bytes.
const BUFFERS: usize = 6;

impl PassSizes {
    /// The sizes on a part of `role` of the model `c`.
    pub(crate) fn of(c: &Config, role: Role) -> Self {
        Self {
            streams: role != Role::Worker,
            wide: c.wide() as u64,
            n: c.n_embd as u64,
            hc: c.n_stream as u64,
            shape: ExpertShape {
                n_embd: c.n_embd as u32,
                n_ff: c.experts.ff as u32,
                n_experts: c.experts.count as u32,
                top_k: c.experts.used as u32,
                gate_ty: ares_core::GgmlType::IQ2_S,
                down_ty: ares_core::GgmlType::Q2_0,
            },
            far_entry: 0,
        }
    }

    /// The sizes of a part whose attention layers keep the K and V of the pages beyond VRAM
    /// in host memory, `entry` bytes per page and layer ([`crate::blocks::far::FarPool`]).
    pub(crate) fn with_far(self, entry: u64) -> Self {
        Self { far_entry: entry, ..self }
    }

    /// The bytes of the buffers of a chunk of `rows` rows: the residual streams, the output
    /// of the mixture of experts, its inject values, the experts' f16 input, the record of
    /// the picks and the plan of the many-row kernels (the streams and the inject values
    /// only on a part that holds the streams).
    fn chunk(&self, rows: u32) -> [u64; BUFFERS] {
        let r = u64::from(rows);
        let streams = u64::from(self.streams);
        [
            streams * r * self.wide * 4,
            r * self.n * 4,
            streams * r * self.hc * 4,
            self.shape.rows_activation_bytes(rows),
            wide::record_bytes(&self.shape, rows),
            wide::plan_bytes(&self.shape, rows),
        ]
    }

    /// The bytes of the hidden values of the experts of a chunk of `rows` rows.
    fn hidden(&self, rows: u32) -> u64 {
        self.shape.rows_hidden_bytes(rows)
    }
}

/// Where the buffers of a pass lie in the layout of its loan, and what each piece of the
/// loan holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassLayout {
    /// Per chunk, the offset of each of its buffers.
    chunks: Vec<[u64; BUFFERS]>,
    /// The rows of each chunk.
    rows: Vec<u32>,
    hidden: u64,
    /// The copy of one layer's far slots: its offset and its slots.
    far_stage: Option<(u64, u32)>,
    /// The bytes each piece holds, without the pieces it does not reach.
    pub held: Vec<u64>,
}

impl PassLayout {
    /// The layout of the buffers of a pass whose chunks have `rows` rows each, in `pieces`
    /// (the bytes each piece of a loan may hold, in the order of the loan's pieces): no
    /// buffer is cut by the end of a piece, each starts at a multiple of 256 bytes. Empty
    /// `pieces`: one piece as large as it needs. `None` when the buffers do not fit. With
    /// `far_slots` on a part whose K and V lie partly in host memory, the copy of a layer's
    /// first `far_slots` far slots comes first (in the first piece, the largest pool's).
    pub fn new(sizes: &PassSizes, rows: &[u32], pieces: &[u64], far_slots: u32) -> Option<Self> {
        let stage = sizes.far_entry * u64::from(far_slots);
        let mut buffers: Vec<u64> = Vec::with_capacity(rows.len() * BUFFERS + 2);
        if stage > 0 {
            buffers.push(stage);
        }
        buffers.extend(rows.iter().flat_map(|r| sizes.chunk(*r)));
        buffers.push(sizes.hidden(rows.iter().copied().max().unwrap_or(0)));
        let (offsets, held) = lay_out_in(pieces, &buffers)?;
        let (far_stage, offsets) = match stage > 0 {
            true => (Some((offsets[0], far_slots)), &offsets[1..]),
            false => (None, &offsets[..]),
        };
        let chunks = offsets.as_chunks::<BUFFERS>().0.to_vec();
        Some(Self { chunks, rows: rows.to_vec(), hidden: *offsets.last()?, far_stage, held })
    }

    /// Where the pieces start in the layout: piece `k` where the pieces before it end, as
    /// [`PassLayout::new`] laid them out.
    pub fn starts(pieces: &[u64]) -> Vec<u64> {
        pieces
            .iter()
            .scan(0u64, |at, bytes| {
                let start = *at;
                *at += bytes / 256 * 256;
                Some(start)
            })
            .collect()
    }

    /// The addresses of the buffers, with `address` mapping an offset of the layout to
    /// where it lies.
    pub fn place(&self, address: impl Fn(u64) -> DevicePtr) -> PassAt {
        let mut first = 0;
        let chunks = self
            .chunks
            .iter()
            .zip(&self.rows)
            .map(|(at, rows)| {
                let chunk = ChunkAt {
                    first,
                    rows: *rows,
                    r: address(at[0]),
                    out: address(at[1]),
                    inj: address(at[2]),
                    xf16: address(at[3]),
                    record: address(at[4]),
                    plan: address(at[5]),
                };
                first += rows;
                chunk
            })
            .collect();
        let far_stage = self.far_stage.map(|(at, slots)| (address(at), slots));
        PassAt { chunks, hidden: address(self.hidden), far_stage }
    }
}

/// Lays `buffers` (their bytes) out one after the other in `pieces` (the bytes of each, in
/// order), at multiples of 256 bytes, none cut by the end of a piece. Returns each buffer's
/// offset in the layout (piece `k` starts where the ones before it end, each rounded down to
/// 256 bytes) and the bytes each piece holds (without the pieces after the last used one).
/// Empty `pieces`: one piece without an end. `None` when the buffers do not fit.
fn lay_out_in(pieces: &[u64], buffers: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    let starts = PassLayout::starts(pieces);
    let ends: Vec<u64> =
        starts.iter().zip(pieces).map(|(start, bytes)| start + bytes / 256 * 256).collect();
    let mut held = vec![0u64; pieces.len()];
    let mut offsets = Vec::with_capacity(buffers.len());
    let mut len = 0u64;
    for bytes in buffers {
        let bytes = (*bytes).max(16);
        let mut at = len;
        if !ends.is_empty() {
            let mut k = ends.iter().position(|end| at < *end)?;
            while at + bytes > ends[k] {
                k += 1;
                at = at.max(*ends.get(k - 1)?);
                ends.get(k)?;
            }
            held[k] = held[k].max(at + bytes - starts[k]);
        }
        offsets.push(at);
        len = (at + bytes).next_multiple_of(256);
    }
    let used = held.iter().rposition(|h| *h > 0).map_or(0, |k| k + 1);
    held.truncate(used);
    Some((offsets, held))
}

/// Where the buffers of one chunk of a pass lie on a part.
#[derive(Clone, Copy, Debug)]
pub struct ChunkAt {
    /// The first row of the span the chunk holds, and its rows.
    pub first: u32,
    pub rows: u32,
    pub r: DevicePtr,
    pub out: DevicePtr,
    pub inj: DevicePtr,
    pub xf16: DevicePtr,
    pub record: DevicePtr,
    pub plan: DevicePtr,
}

/// Where a pass keeps its rows on a part.
#[derive(Clone, Debug)]
pub struct PassAt {
    pub chunks: Vec<ChunkAt>,
    /// The hidden values of the experts of a chunk, which the chunks share.
    pub hidden: DevicePtr,
    /// The copy of one layer's far slots, and its slots ([`Part::pass_stage`]).
    pub far_stage: Option<(DevicePtr, u32)>,
}

/// The attention plan of a micro-batch, made when the pass starts.
pub struct MicroPlan(Option<StepPlan>);

impl std::fmt::Debug for MicroPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicroPlan").field("attention", &self.0.is_some()).finish()
    }
}

/// A pass on a part, between its first micro-batch and its end.
pub(super) struct State {
    at: PassAt,
    /// Per micro-batch: its chunk, its first row in the chunk and its rows.
    micro: Vec<(usize, u32, u32)>,
    /// The output of each micro-batch's last half block, which its next read writes.
    pending: Vec<Option<Pending>>,
    /// The scratch ranges that are the rows of the running micro-batch: (offset, bytes,
    /// where).
    pub(super) redirects: Vec<(u64, u64, DevicePtr)>,
    /// The micro-batch that runs, while one does.
    running: Option<usize>,
    /// Page-locked buffers the inbox is sent from, each with the event behind its copy.
    ring: Vec<(PinnedBuffer, Event)>,
    ring_at: usize,
    /// The buffer of the ring the running micro-batch was sent from.
    pub(super) host: Option<usize>,
    /// The tile of each chunk's plan, once the first round of a layer has made it.
    tiles: Vec<u32>,
    /// Whether the pass runs: between [`Part::pass_open`] (or [`Part::pass_resume`]) and
    /// [`Part::pass_pause`] (or [`Part::pass_close`]); the steps that run while it is paused
    /// are steps of their own.
    active: bool,
    /// The layer (of the part) whose far slots the copy in the loan holds, and up to which
    /// slot it is up to date ([`Part::pass_stage`]).
    staged: Option<(usize, u32)>,
}

impl Part {
    /// Whether the part can read prompts layer by layer: its own scratch takes steps wider
    /// than the decode kernels, so a micro-batch runs on the many-row kernels.
    pub fn takes_passes(&self) -> bool {
        self.wide_state.is_some() && self.opts.own_rows() >= MIN_MICRO
    }

    /// The attention plan of a micro-batch of a pass (`env` is the micro-batch as a step of
    /// its own), on a part with attention layers.
    pub fn pass_plan(&self, env: &StepEnv<'_>) -> Result<MicroPlan, ModelError> {
        self.check(env)?;
        Ok(MicroPlan(self.attention.is_some().then(|| self.build_plan(env)).transpose()?))
    }

    /// Starts a pass of the prompt rows `segments` (one segment; `replay` its rows that are a
    /// replay, `pools` the pools of K/V and keys) whose rows lie at `at`, in the micro-batches
    /// `micro` (first row in the span and rows each), which fill the chunks in order: the
    /// scratch is the own one, the K/V group's entries of every cell the pass reaches are in
    /// place (as a step's are), a sequence that starts in its slot starts from zero state,
    /// and what the step before predicted from the scratch is done with it.
    pub fn pass_open(
        &mut self,
        stream: &ares_cuda::Stream,
        segments: &[ares_cuda::Segment],
        replay: &[Range<u32>],
        pools: DevicePtr,
        at: PassAt,
        micro: &[(u32, u32)],
    ) -> Result<(), ModelError> {
        if !self.takes_passes() {
            return Err(invalid("the part reads no prompt layer by layer"));
        }
        self.lay_out(stream, self.opts.own_rows())?;
        let status = self.at(self.f.s.status);
        if let Some(kv) = &mut self.kv {
            let reaches = crate::blocks::kv::reaches(segments, replay);
            let (tables, kernels) = (self.tables.ptr(), self.attention.as_ref());
            kv.prepare(kernels, stream, &reaches, &self.entries, tables, pools, status, true)?;
        }
        if let Some(marks) = &mut self.marks {
            marks.phases.clear();
            marks.on = !marks.profile.paused.load(std::sync::atomic::Ordering::Relaxed);
        }
        self.mark(stream, Phase::Other)?;
        self.predictions_done(stream)?;
        self.state.clear_starting(stream, segments)?;
        let mut placed = Vec::with_capacity(micro.len());
        for &(first, rows) in micro {
            let chunk = at
                .chunks
                .iter()
                .position(|c| c.first <= first && first + rows <= c.first + c.rows)
                .ok_or_else(|| invalid(format!("no chunk holds rows {first}..+{rows}")))?;
            if rows > self.opts.own_rows() || rows <= ares_cuda::MAX_BATCH {
                return Err(invalid(format!("a micro-batch of {rows} rows")));
            }
            placed.push((chunk, first - at.chunks[chunk].first, rows));
        }
        let ring = match self.pass.take() {
            Some(old) => old.ring,
            None => (0..RING)
                .map(|_| {
                    Ok((
                        PinnedBuffer::new(&self.device, self.f.inbox.host_len as usize)?,
                        Event::new(&self.device)?,
                    ))
                })
                .collect::<Result<_, ModelError>>()?,
        };
        let tiles = vec![0; at.chunks.len()];
        self.pass = Some(State {
            at,
            pending: vec![None; placed.len()],
            micro: placed,
            redirects: Vec::new(),
            running: None,
            ring,
            ring_at: 0,
            host: None,
            tiles,
            active: true,
            staged: None,
        });
        self.pending = None;
        self.flipped = false;
        Ok(())
    }

    /// Ends the pass: the scratch is the part's own again. The ring of buffers stays for the
    /// next pass.
    pub fn pass_close(&mut self) {
        if let Some(state) = &mut self.pass {
            state.redirects.clear();
            state.running = None;
            state.host = None;
            state.pending.fill(None);
            state.micro.clear();
            state.at.chunks.clear();
            state.at.far_stage = None;
            state.active = false;
            state.staged = None;
        }
        self.pending = None;
    }

    /// Copies the residual streams of `rows` rows at `from` (the kept rows of a pass's last
    /// micro-batch) into the own scratch, where [`Part::residual`] finds them: what the next
    /// step's start reads of this one (the MTP block follows its rows) does not lie in the
    /// pass's loan, which may go back before.
    pub fn keep_residual(
        &mut self,
        stream: &ares_cuda::Stream,
        from: DevicePtr,
        rows: u32,
    ) -> Result<DevicePtr, ModelError> {
        if let Some(pass) = &mut self.pass {
            pass.redirects.clear();
        }
        self.flipped = false;
        let to = self.r();
        // SAFETY: the own scratch holds the residual of the most rows of a micro-batch, and
        // `from` the caller's rows.
        unsafe { stream.copy(to, from, (u64::from(rows) * u64::from(self.wide) * 4) as usize)? };
        Ok(to)
    }

    /// Gives the output of micro-batch `m`'s last half block that is still to be written
    /// (or, on a worker, sent) the exchange number `no`: a pass numbers its exchanges with
    /// the peer by its pieces, in the order they run, and the pending output of a layer
    /// crosses in the piece of the next layer that writes it.
    pub fn pass_renumber(&mut self, m: usize, no: u32) {
        if let Some(Some(Pending::Moe { exchange: Some(at), .. })) =
            self.pass.as_mut().and_then(|p| p.pending.get_mut(m))
        {
            *at = no;
        }
    }

    /// The bytes of a sequence slot that hold the state of layer `i` of the part, which a
    /// save inside a pass copies as the pass goes past its position.
    pub fn layer_slot(&self, i: usize) -> Range<u64> {
        self.f.layer_slot.get(i).cloned().unwrap_or(0..0)
    }

    /// Whether a pass runs on the part (it is open and not paused).
    pub fn in_pass(&self) -> bool {
        self.pass.as_ref().is_some_and(|p| p.active && !p.micro.is_empty())
    }

    /// The pass stops between two micro-batches for steps of their own, which run on the own
    /// scratch as any step does; what it keeps stays where it is.
    pub fn pass_pause(&mut self) {
        if let Some(state) = &mut self.pass {
            state.active = false;
        }
        self.pending = None;
    }

    /// The paused pass goes on: after [`Part::pass_pause`] and the steps between, the scratch
    /// is the own one again (what a pass's micro-batches run in), and what the steps between
    /// predicted from it is done with it.
    pub fn pass_resume(&mut self, stream: &ares_cuda::Stream) -> Result<(), ModelError> {
        if !self.pass.as_ref().is_some_and(|p| !p.micro.is_empty()) {
            return Err(invalid("no pass is open to go on with"));
        }
        self.lay_out(stream, self.opts.own_rows())?;
        if let Some(marks) = &mut self.marks {
            marks.phases.clear();
            marks.on = !marks.profile.paused.load(std::sync::atomic::Ordering::Relaxed);
        }
        self.mark(stream, Phase::Other)?;
        self.predictions_done(stream)?;
        if let Some(state) = &mut self.pass {
            state.active = true;
        }
        self.pending = None;
        Ok(())
    }

    /// The copy of layer `i`'s K and V beyond VRAM that the running micro-batch of a pass
    /// (`env`) reads in place of the far pool, brought up to date behind its append: `None`
    /// when no pass runs, its loan keeps no copy, or the micro-batch's rows reach no far
    /// page. The first micro-batch of a layer copies every far slot its rows reach (the
    /// cached cells before the pass's and its own); each later one the slots from the first
    /// page its own rows wrote on (the last page of the one before holds cells of both) and
    /// those the copy did not hold yet. Every micro-batch runs on the same layer before the
    /// next layer begins, and the steps between two pieces of a paused pass write no cell of
    /// its sequence, so the copy holds what the far pool holds for the pass's cells.
    pub(super) fn pass_stage(
        &mut self,
        env: &StepEnv<'_>,
        i: usize,
        layer: &crate::blocks::kv::Layer,
    ) -> Result<Option<DevicePtr>, ModelError> {
        let Some(state) = self.pass.as_ref().filter(|p| p.active && p.running.is_some()) else {
            return Ok(None);
        };
        let Some((to, room)) = state.at.far_stage else { return Ok(None) };
        let staged = state.staged;
        let reach = self.far_reach(env);
        if reach == 0 {
            return Ok(None);
        }
        if reach > room {
            return Err(invalid(format!(
                "a micro-batch of a pass reaches far slot {reach}; its loan copies {room}"
            )));
        }
        let from = match staged {
            Some((layer_i, upto)) if layer_i == i => self.far_written(env).min(upto),
            _ => 0,
        };
        if from < reach {
            self.group()?.copy_far(env.stream, layer, from..reach, to)?;
        }
        let upto = staged.filter(|(layer_i, _)| *layer_i == i).map_or(0, |(_, upto)| upto);
        if let Some(state) = &mut self.pass {
            state.staged = Some((i, reach.max(upto)));
        }
        Ok(Some(to))
    }

    /// The residual streams of micro-batch `m`'s rows: where they are once what was enqueued
    /// so far has run.
    pub fn pass_residual(&self, m: usize) -> Result<DevicePtr, ModelError> {
        let state = self.pass.as_ref().ok_or_else(|| invalid("no pass is open"))?;
        let &(chunk, row0, _) =
            state.micro.get(m).ok_or_else(|| invalid(format!("no micro-batch {m}")))?;
        Ok(state.at.chunks[chunk].r.offset(u64::from(row0) * u64::from(self.wide) * 4))
    }

    /// Micro-batch `m` of the pass (`env`: its rows as a step of their own, `plan` its
    /// attention plan) through layer `i` of the part, up to the layer's router and shared
    /// expert: its record (and before an attention layer its plan) goes to the inbox, at the
    /// model's first layer its rows start from their embeddings, and its kept rows stand in
    /// for the scratch's residual, MoE output and inject values.
    pub fn pass_layer(
        &mut self,
        env: &StepEnv<'_>,
        i: usize,
        m: usize,
        plan: &mut MicroPlan,
    ) -> Result<(), ModelError> {
        self.pass_enter(env, m, plan, self.first && i == 0)?;
        if self.attention.is_some() && self.f.attention.get(i).is_some_and(Option::is_some) {
            let dims = env.record.dims();
            // SAFETY: the record of the micro-batch is in the inbox, the rotary table in
            // the own scratch, which holds its rows.
            unsafe {
                self.layers.rope_table(
                    env.stream,
                    self.inbox_at(self.f.inbox.record),
                    dims,
                    self.at(self.f.s.rope),
                    self.rope,
                )?;
            }
        }
        let result = self.enqueue_layer(env, i);
        self.pass_leave(env.stream, m, plan)?;
        result
    }

    /// Ends micro-batch `m` after the part's last layer: writes the last layer's output to
    /// its streams, or on the part with the head computes the logits of its rows that are
    /// wanted (`env` says which); returns how many.
    pub fn pass_end(
        &mut self,
        env: &StepEnv<'_>,
        m: usize,
        plan: &mut MicroPlan,
    ) -> Result<u32, ModelError> {
        self.pass_enter(env, m, plan, false)?;
        let result = self.end(env);
        self.pass_leave(env.stream, m, plan)?;
        result
    }

    /// Makes micro-batch `m` the running one: sends its inbox (with the embeddings of its
    /// tokens when `embed`) and points the scratch's kept ranges at its rows.
    fn pass_enter(
        &mut self,
        env: &StepEnv<'_>,
        m: usize,
        plan: &mut MicroPlan,
        embed: bool,
    ) -> Result<(), ModelError> {
        self.check(env)?;
        let stream = env.stream;
        let dims = env.record.dims();
        let n = dims.n_rows;
        let (wide, n_embd, hc) = (u64::from(self.wide), u64::from(self.n), u64::from(self.hc));
        let s = self.f.s;
        let ib = self.f.inbox;
        let state = self.pass.as_mut().ok_or_else(|| invalid("no pass is open"))?;
        let &(chunk, row0, rows) =
            state.micro.get(m).ok_or_else(|| invalid(format!("no micro-batch {m}")))?;
        if rows != n {
            return Err(invalid(format!("micro-batch {m} has {rows} rows, its step {n}")));
        }
        let c = state.at.chunks[chunk];
        let row = u64::from(row0);
        state.redirects.clear();
        if self.role != Role::Worker {
            state.redirects.push((s.r, u64::from(n) * wide * 4, c.r.offset(row * wide * 4)));
            state.redirects.push((s.inj_f, u64::from(n) * hc * 4, c.inj.offset(row * hc * 4)));
        }
        state.redirects.push((s.out, u64::from(n) * n_embd * 4, c.out.offset(row * n_embd * 4)));
        state.running = Some(m);
        // The next buffer of the ring, once the device has read what it held.
        let k = state.ring_at;
        state.ring_at = (k + 1) % state.ring.len();
        state.host = Some(k);
        state.ring[k].1.synchronize()?;
        let host = state.ring[k].0.as_mut_ptr();
        self.pending = state.pending[m].take();
        self.flipped = false;
        std::mem::swap(&mut self.plan, &mut plan.0);
        // The record, and the plan with its hands where the part attends.
        let record = env.record.as_bytes();
        if record.len() as u64 > ib.record_cap {
            return Err(invalid(format!("a step record of {} bytes", record.len())));
        }
        // SAFETY: the pinned buffer has the sections of `Inbox`; the device has read what
        // the buffer held before (its event).
        unsafe {
            std::ptr::copy_nonoverlapping(
                record.as_ptr(),
                host.add(ib.record as usize),
                record.len(),
            );
        }
        let mut sent = ib.record + record.len() as u64;
        if let Some(step_plan) = &self.plan {
            let layout = step_plan.plan.layout();
            if layout.bytes > ib.plan_cap {
                return Err(invalid(format!("an attention plan of {} bytes", layout.bytes)));
            }
            // SAFETY: as above; the plan section holds `plan_cap` bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(host.add(ib.plan as usize), layout.bytes as usize)
            };
            step_plan.plan.write(bytes);
            sent = ib.plan + layout.bytes;
            if let Some(hands) = &step_plan.hands {
                let bytes = ares_cuda::ListHandOver::to_bytes(hands);
                // SAFETY: as above.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        host.add(ib.hands as usize),
                        bytes.len(),
                    );
                }
                sent = ib.hands + bytes.len() as u64;
            }
        }
        // SAFETY: the inbox holds the sections; the pinned buffer stays unchanged until its
        // event says the copy has run.
        unsafe {
            stream.copy_from_host(
                self.inbox_at(ib.record),
                host.add(ib.record as usize).cast_const(),
                (sent - ib.record) as usize,
            )?;
        }
        if embed {
            self.pass_embed(env, host)?;
        }
        Ok(())
    }

    /// The rows of the running micro-batch start from the embeddings of their tokens, staged
    /// in the ring's buffer at `host`, and the rows another source computed for it
    /// ([`Part::splice_rows`]: an image's), in every stream (as [`Part::begin`] starts a step).
    fn pass_embed(&mut self, env: &StepEnv<'_>, host: *mut u8) -> Result<(), ModelError> {
        let stream = env.stream;
        let dims = env.record.dims();
        let n = dims.n_rows;
        let ib = self.f.inbox;
        let table =
            self.token_rows.clone().ok_or_else(|| invalid("the part that embeds has no rows"))?;
        if env.tokens.len() != n as usize {
            return Err(invalid(format!("{} tokens for {n} rows", env.tokens.len())));
        }
        // SAFETY: the rows section of the pinned buffer ends where the device's own section
        // begins; the buffer's copy before has run.
        let section = unsafe {
            std::slice::from_raw_parts_mut(
                host.add(ib.rows as usize),
                (ib.host_len - ib.rows) as usize,
            )
        };
        let bytes = table.stage(env.tokens, section)?;
        let r = self.r();
        // SAFETY: the inbox and the kept rows hold the micro-batch's rows; the spliced rows lie
        // in the runtime's image stage, which holds them while the step runs (a record that
        // goes on with a paused pass stages them again).
        unsafe {
            let (at, x) = (self.inbox_at(ib.rows), self.inbox_at(ib.x));
            stream.copy_from_host(at, host.add(ib.rows as usize).cast_const(), bytes)?;
            let source = crate::blocks::rows::source(&table.shape(), ares_cuda::KeptRows::NONE, at);
            self.layers.embed_gather(stream, self.inbox_at(ib.record), dims, &source, x)?;
            // Rows another source computed (an image's) go over theirs.
            crate::blocks::rows::splice(stream, x, self.n, &std::mem::take(&mut self.spliced))?;
            stream.fill(r, 0, (n * self.wide * 4) as usize)?;
            self.q4.stream_write(stream, r, x, self.at(self.f.s.zeros), n, self.hc, self.n)?;
        }
        self.mark(stream, Phase::Exchange)
    }

    /// Micro-batch `m` stops running: its pending output and its plan are put aside, and
    /// the buffer of the ring it was sent from is free once the copies enqueued on `stream`
    /// so far (the n-gram rows of its layer among them) have run.
    fn pass_leave(
        &mut self,
        stream: &ares_cuda::Stream,
        m: usize,
        plan: &mut MicroPlan,
    ) -> Result<(), ModelError> {
        std::mem::swap(&mut self.plan, &mut plan.0);
        if let Some(state) = &mut self.pass {
            state.pending[m] = self.pending.take();
            state.redirects.clear();
            state.running = None;
            if let Some(k) = state.host.take() {
                state.ring[k].1.record(stream)?;
            }
        }
        Ok(())
    }

    /// The page-locked buffer the running micro-batch's inbox is sent from, while a pass
    /// runs one; the inbox's own otherwise.
    pub(super) fn staging_host(&self) -> *mut u8 {
        match self.pass.as_ref().and_then(|p| p.host.map(|k| &p.ring[k].0)) {
            Some(buffer) => buffer.as_ptr().cast_mut(),
            None => self.inbox_host.as_ptr().cast_mut(),
        }
    }

    /// In a pass: where the router of the running micro-batch writes its picks, as the
    /// record of its chunk, its first row there and the chunk's rows.
    pub(super) fn pass_record(&self) -> Option<(DevicePtr, u32, u32)> {
        let state = self.pass.as_ref()?;
        let (chunk, row0, _) = state.micro[state.running?];
        let c = state.at.chunks[chunk];
        Some((c.record, row0, c.rows))
    }

    /// In a pass, while a micro-batch runs: its first row among the pass's rows, and the
    /// pass's rows.
    pub(super) fn pass_rows(&self) -> Option<(u32, u32)> {
        let state = self.pass.as_ref()?;
        let (chunk, row0, _) = state.micro[state.running?];
        let all = state.at.chunks.iter().map(|c| c.rows).sum();
        Some((state.at.chunks[chunk].first + row0, all))
    }

    /// While a pass runs: where the outputs of its rows lie, per chunk its first row among
    /// the pass's, its rows and its output.
    pub(super) fn pass_outputs(&self) -> Option<Vec<(u32, u32, DevicePtr)>> {
        let state = self.pass.as_ref().filter(|p| p.active)?;
        Some(state.at.chunks.iter().map(|c| (c.first, c.rows, c.out)).collect())
    }

    /// In a pass: copies the experts' f16 input of the running micro-batch's `rows` rows,
    /// prepared at `prepared` in the form of a step of its rows, into its chunk's.
    ///
    /// # Safety
    ///
    /// `prepared` must hold the f16 form of `rows` rows of `n_embd` values, written by what
    /// is enqueued on `stream`.
    pub(super) unsafe fn pass_keep_input(
        &self,
        stream: &ares_cuda::Stream,
        prepared: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        let state = self.pass.as_ref().ok_or_else(|| invalid("no pass is open"))?;
        let m = state.running.ok_or_else(|| invalid("no micro-batch runs"))?;
        let (chunk, row0, _) = state.micro[m];
        let c = state.at.chunks[chunk];
        let (values, scales) = f16_sections(self.n, rows);
        let (chunk_values, _) = f16_sections(self.n, c.rows);
        let pitch = values / u64::from(rows);
        // SAFETY: the chunk's input holds its rows in the same form, the caller the rest.
        unsafe {
            stream.copy(c.xf16.offset(u64::from(row0) * pitch), prepared, values as usize)?;
            stream.copy(
                c.xf16.offset(chunk_values.next_multiple_of(16) + 4 * u64::from(row0)),
                prepared.offset(scales),
                4 * rows as usize,
            )?;
        }
        Ok(())
    }

    /// In a pass: a round of experts for every chunk, with the table at `table` (see
    /// [`Part::experts_round`]); the first round of a layer makes each chunk's plan, from the
    /// picks the part's router left in the chunk's record or, on a part that does not route,
    /// from `record` (the picks of every row of the span as [`Part::pass_record_to_host`]
    /// leaves them), which it copies there first.
    pub(super) fn pass_round(
        &mut self,
        stream: &ares_cuda::Stream,
        i: usize,
        table: DevicePtr,
        first: bool,
        token: u32,
        record: Option<&[u8]>,
    ) -> Result<(), ModelError> {
        let shape = self.w.layers[i].moe.shape;
        let wide = self.wide_state.as_ref().ok_or_else(|| invalid("no wide state"))?;
        let state = self.pass.as_mut().ok_or_else(|| invalid("no pass is open"))?;
        let k_picks = shape.top_k as usize;
        let rows: usize = state.at.chunks.iter().map(|c| c.rows as usize).sum();
        if let Some(record) = record.filter(|_| first) {
            if record.len() < 16 + 8 * k_picks * rows {
                return Err(invalid("the picks of a pass are short"));
            }
            for c in &state.at.chunks {
                let (at, n) = (c.first as usize * k_picks * 4, c.rows as usize * k_picks * 4);
                // SAFETY: the chunk's record holds its rows' ids and weights after its header;
                // the host's record stays unchanged until this layer is done.
                unsafe {
                    stream.copy_from_host(c.record.offset(16), record[16 + at..].as_ptr(), n)?;
                    stream.copy_from_host(
                        c.record.offset(16 + n as u64),
                        record[16 + 4 * k_picks * rows + at..].as_ptr(),
                        n,
                    )?;
                }
            }
        }
        for (k, c) in state.at.chunks.iter().enumerate() {
            if first {
                state.tiles[k] = ExpertRowKernels::tile_for(&shape, c.rows);
            }
            let tile = state.tiles[k];
            // SAFETY: the chunk's buffers hold its rows: the record its picks (every
            // micro-batch of the layer has routed into it before this round), the input the
            // f16 form of its rows, the plan and the output their sizes; the hidden values
            // are the chunks' to share, one after the other on the stream.
            unsafe {
                if first {
                    wide.kernels
                        .plan_rows(stream, &shape, c.rows, tile, c.record, c.plan, token)?;
                }
                wide.kernels.rows_gate_up(
                    stream,
                    &shape,
                    c.rows,
                    tile,
                    table,
                    c.plan,
                    c.xf16,
                    state.at.hidden,
                )?;
                wide.kernels.rows_down(
                    stream,
                    &shape,
                    c.rows,
                    tile,
                    table,
                    c.plan,
                    state.at.hidden,
                    c.out,
                )?;
            }
        }
        Ok(())
    }

    /// In a pass: copies the picks of every chunk of layer `i` (behind what is enqueued on
    /// `stream`) to `host`, as the record of all rows of the span reads to the host: a header
    /// of 16 bytes, then the ids of the rows in order, then their weights. Records `copied`
    /// behind the copies.
    ///
    /// # Safety
    ///
    /// `host` must be page-locked memory of `16 + 8 * top_k * rows` bytes for the rows of
    /// the pass, untouched until `copied` has passed.
    pub(crate) unsafe fn pass_record_to_host(
        &self,
        i: usize,
        host: *mut u8,
        copied: &Event,
    ) -> Result<(), ModelError> {
        let state = self.pass.as_ref().ok_or_else(|| invalid("no pass is open"))?;
        let k = self.w.layers[i].moe.shape.top_k as u64;
        let rows: u64 = state.at.chunks.iter().map(|c| u64::from(c.rows)).sum();
        let side = &self.side;
        side.wait_event(&self.routed[i])?;
        for c in &state.at.chunks {
            let (first, n) = (u64::from(c.first), u64::from(c.rows));
            // SAFETY: the caller's guarantee; the chunk's record holds its ids after its
            // header and their weights after them.
            unsafe {
                side.copy_to_host(
                    host.add((16 + 4 * k * first) as usize),
                    c.record.offset(16),
                    (4 * k * n) as usize,
                )?;
                side.copy_to_host(
                    host.add((16 + 4 * k * (rows + first)) as usize),
                    c.record.offset(16 + 4 * k * n),
                    (4 * k * n) as usize,
                )?;
            }
        }
        copied.record(side)?;
        Ok(())
    }

    /// Records that layer `i`'s routers have run for every micro-batch enqueued so far (the
    /// event the copy of its picks waits for).
    pub(crate) fn pass_routed(
        &self,
        stream: &ares_cuda::Stream,
        i: usize,
    ) -> Result<(), ModelError> {
        Ok(self.routed[i].record(stream)?)
    }
}

/// The bytes of the values of the f16 form of `rows` rows of `n` values, and where the
/// scales of the rows start (`Kernels::prefill_activation_bytes`).
fn f16_sections(n: u32, rows: u32) -> (u64, u64) {
    let all = Kernels::prefill_activation_bytes(PrefillPath::F16, n, rows);
    let scales = all - 4 * u64::from(rows);
    let line = Kernels::ACTIVATION_LINE;
    let pitch = ((u64::from(n) * 2).div_ceil(line) + 1) | 1;
    (pitch * line * u64::from(rows), scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_batches_cover_the_rows_and_none_is_too_narrow() {
        for (rows, own) in [(1000, 256), (1024, 256), (1030, 256), (513, 512), (40000, 512)] {
            let micro = micro_batches(rows, own);
            let mut at = 0;
            for &(first, n) in &micro {
                assert_eq!(first, at);
                assert!(n > ares_cuda::MAX_BATCH && n <= own, "{rows}/{own}: {micro:?}");
                at += n;
            }
            assert_eq!(at, rows);
        }
        assert_eq!(micro_batches(1030, 256).last(), Some(&(899, 131)));
    }

    #[test]
    fn chunks_hold_whole_micro_batches_up_to_their_cap() {
        let micro = micro_batches(20000, 512);
        let chunks = chunks_of(&micro, 8192);
        assert_eq!(chunks.first(), Some(&(0..16)));
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), micro.len());
        let one = chunks_of(&micro_batches(100, 512), 64);
        assert_eq!(one, vec![0..1]);
    }

    #[test]
    fn no_buffer_is_cut_by_the_end_of_a_piece() {
        let (offsets, held) = lay_out_in(&[1000, 600, 4096], &[300, 300, 500, 2000]).unwrap();
        assert_eq!(offsets, vec![0, 768, 1280, 1792]);
        assert_eq!(held, vec![300, 300, 2512]);
        assert!(lay_out_in(&[1000], &[300, 900]).is_none());
        let (offsets, held) = lay_out_in(&[], &[300, 900]).unwrap();
        assert_eq!(offsets, vec![0, 512]);
        assert!(held.is_empty());
    }

    #[test]
    fn the_copy_of_far_slots_comes_first_in_the_loan_of_a_pass() {
        let sizes = PassSizes {
            streams: true,
            wide: 4 * 1024,
            n: 1024,
            hc: 4,
            shape: ExpertShape {
                n_embd: 1024,
                n_ff: 512,
                n_experts: 64,
                top_k: 8,
                gate_ty: ares_core::GgmlType::IQ2_S,
                down_ty: ares_core::GgmlType::Q2_0,
            },
            far_entry: 0,
        };
        let chunks = [1024, 512];
        let plain = PassLayout::new(&sizes, &chunks, &[], 100).unwrap();
        assert_eq!(plain.far_stage, None, "no far pool: no copy");
        let far = sizes.with_far(1 << 20);
        let with = PassLayout::new(&far, &chunks, &[], 100).unwrap();
        assert_eq!(with.far_stage, Some((0, 100)));
        assert_eq!(with.chunks[0][0], 100 << 20, "the chunks after the copy");
        let total = |l: &PassLayout| l.hidden + sizes.hidden(1024);
        assert_eq!(total(&with) - total(&plain), 100 << 20);
        // In pieces: the copy takes the first piece; what it cannot hold does not fit.
        let pieces = [(100 << 20) + total(&plain), 1 << 20];
        assert!(PassLayout::new(&far, &chunks, &pieces, 100).is_some());
        assert!(PassLayout::new(&far, &chunks, &[(99 << 20), 1 << 40], 100).is_some());
        assert!(PassLayout::new(&far, &chunks, &[(99 << 20)], 100).is_none());
    }
}
