//! The model's draft block (section 4 of the specification) as a device
//! thread of the runtime drives it: a part of one layer on the device that
//! holds the output head ([`Part::open_block`]), which runs steps of its own
//! in front of every step of the model.
//!
//! Cell `i` of the block pairs the model's residual after its last layer at
//! cell `i` with the token at cell `i + 1`, so the block is always one token
//! behind the model, and what it does in a step is decided by what the model
//! did in the step before:
//!
//! 1. **Following.** The rows of the model's last step that count (all of a
//!    slice of a prompt, the accepted rows of a window, which the record of
//!    this step tells) give the block the cells in front of the last one: each
//!    pairs a row's residual, which is still in the model part's scratch, with
//!    the next row's token. They only write keys and values. The residual of
//!    the last row that counts is copied to the sequence's *carry*.
//! 2. **The newest cell.** The first row of a sequence in this step is certain
//!    and names the token the carry is paired with. For a sequence that drafts
//!    this is link 0 of its chain, which runs the whole block and picks; for
//!    any other it writes keys and values only.
//! 3. **The chain.** Link `j` starts from the block's own residual of link
//!    `j - 1` and its pick. These cells are provisional: the keys of the
//!    indexer the sequence carries are saved behind link 0 and put back behind
//!    the last link, and the keys and values are overwritten when the model
//!    has passed the cells (step 1 of a later step).
//!
//! Every link ends with a wait of the host: the pick is read back, because the
//! embedding of the next link's token is decoded on the host, as are the
//! embeddings and the rows of the n-gram table of the rows that verify the
//! drafts. While it waits the thread serves the expert store, which brings the
//! experts the block's router picked like those of any layer (the block is
//! layer 0 of the store).
//!
//! The block's pages are its own memory, addressed through the page tables
//! the conductor gives the model: page `p` of a sequence holds the same cells
//! of the model's attention layers and of the block.

use std::sync::Arc;
use std::time::Instant;

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, MAX_BATCH, PinnedBuffer, SampleBatch, SamplePlan,
    SamplerKernels, Segment, StepRecord, Stream,
};
use ares_runtime::ModelError;
use ares_sampler::{SamplePick, SampleRow};

use super::board::Board;
use super::part::{ExpertAt, Options, Part, StepEnv};
use super::runtime::{Embedding, Flow};
use crate::qwen4exp::Config;
use crate::qwen4exp::device::load::Mat;
use crate::qwen4exp::reference::block_config;
use crate::qwen4exp::weights::DraftWeights;

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

/// What the model's last step left of a sequence for the block to follow.
#[derive(Clone, Debug)]
struct Tail {
    slot: u32,
    pos0: u32,
    row0: u32,
    tokens: Vec<u32>,
    direct: bool,
    /// The drafts of the block among the rows: 0 for rows that took none.
    links: u32,
}

/// A window of the step that takes the block's drafts.
#[derive(Clone, Debug)]
pub(super) struct Chain {
    /// The segment of the step that verifies the drafts: its first row is the
    /// sequence's newest token, the others take the drafts.
    pub segment: usize,
    /// How the rows that verify the drafts are sampled, one per draft: link
    /// `j` and row `j` of the window pick the same position.
    pub rows: Vec<SampleRow>,
}

/// Where a row of a block step takes its residual from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// The carry of the row's sequence.
    Carry,
    /// A row of the model's last step.
    Model(u32),
    /// A row of the block's own last step.
    Link(u32),
}

/// One row of a step of the block.
#[derive(Clone, Copy, Debug)]
struct Row {
    slot: u32,
    cell: u32,
    token: u32,
    source: Source,
}

/// What the block took and did, for reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DraftReport {
    /// The device part that holds the block.
    pub index: usize,
    /// Bytes of its dense weights on the device.
    pub weights: u64,
    /// Bytes of its keys, values and indexer keys for all pages.
    pub pools: u64,
    /// Bytes of its scratch, carries and tables.
    pub buffers: u64,
}

/// Device and host time of the block, added up over steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DraftTimes {
    /// Steps of the model in which a chain ran.
    pub chain_steps: u64,
    /// Links of all chains of those steps (a link of several sequences counts
    /// once).
    pub links: u64,
    /// Milliseconds of the chains, from the first link's enqueue to the last
    /// pick on the host.
    pub chain_ms: f64,
    /// Rows that only wrote keys, and the milliseconds the host took to
    /// enqueue them (the device runs them while the model's step is enqueued).
    pub key_rows: u64,
    pub key_ms: f64,
    /// Per link of a chain: how often its draft was verified (every draft
    /// before it had been accepted) and how often it was accepted then.
    pub offered: [u64; MAX_BATCH as usize],
    pub accepted: [u64; MAX_BATCH as usize],
}

/// A chain the block made: for tests and measurements that compare its
/// drafts with the reference's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Drafted {
    /// The sequence slot.
    pub slot: u32,
    /// The cell of the model the chain's first draft is a guess for: one
    /// behind the newest token's.
    pub position: u32,
    /// The newest token, which link 0 is paired with.
    pub token: u32,
    /// The drafts, link after link.
    pub drafts: Vec<u32>,
}

/// What a draft block tells whoever watches it.
#[derive(Debug, Default)]
pub(super) struct Watch {
    pub times: DraftTimes,
    /// The newest chains, oldest first, at most [`Watch::KEPT`].
    pub recent: std::collections::VecDeque<Drafted>,
}

impl Watch {
    const KEPT: usize = 1024;
}

pub(super) struct DraftBlock {
    part: Part,
    embedding: Embedding,
    sampler: SamplerKernels,
    /// Keys, values and indexer keys of the block's cells, for every page.
    pools: DeviceBuffer,
    /// Per sequence slot the residual its next cell starts from (`D` f32).
    carry: DeviceBuffer,
    /// The residuals of the rows of a block step, gathered.
    staged: DeviceBuffer,
    /// The sampler's rows, picks, scratch and exchange.
    work: DeviceBuffer,
    picks_at: u64,
    scratch_at: u64,
    exchange_at: u64,
    /// The host's side of the rows, the picks and the drafts.
    io: PinnedBuffer,
    picks_host: usize,
    drafts_host: usize,
    /// The drafts of the step, for the runtime to read with the picks.
    drafts: DeviceBuffer,
    n_drafts: u32,
    picked: Event,
    /// Recorded behind a step whose rows only wrote keys, and whether the
    /// device may still be running such a step: the host writes the block's
    /// step inputs again only when it has run.
    wrote: Event,
    writing: bool,
    tails: Vec<Tail>,
    serial: u32,
    n: usize,
    wide: u64,
    n_vocab: u32,
    max_slots: u32,
    last_chain_ms: f64,
    times: DraftTimes,
    watch: Arc<std::sync::Mutex<Watch>>,
    x: Vec<f32>,
    report: DraftReport,
}

impl std::fmt::Debug for DraftBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DraftBlock").field("report", &self.report).finish_non_exhaustive()
    }
}

impl DraftBlock {
    /// Builds the block on `device`, next to the model part `index` that
    /// holds the output head `output`. `opts` are the model part's: the block
    /// has the same sequences, pages and formats, and steps of up to
    /// [`MAX_BATCH`] certain rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open(
        c: &Config,
        w: &DraftWeights<'_>,
        opts: &Options,
        index: usize,
        output: (Mat, DevicePtr),
        embedding: Embedding,
        watch: Arc<std::sync::Mutex<Watch>>,
        device: &Device,
        ledger: &mut Ledger,
    ) -> Result<Self, ModelError> {
        let block = block_config(c);
        let opts = Options {
            max_rows: MAX_BATCH,
            logit_rows: MAX_BATCH,
            window_rows: 0,
            host_logits: false,
            profile: None,
            ..opts.clone()
        };
        let before = ledger.reserved();
        let part = Part::open_block(&block, w, &opts, index, output, device, ledger)?;
        let weights = part.weight_bytes();
        let pool_bytes = part.pool_bytes();
        ledger.reserve("qwen4exp draft block K, V and keys", pool_bytes)?;
        let pools = DeviceBuffer::new(device, pool_bytes.max(64))?;
        let wide = c.wide() as u64;
        let rows = u64::from(MAX_BATCH);
        let carry_bytes = u64::from(opts.max_slots) * wide * 4;
        let staged_bytes = rows * wide * 4;
        let picks_at = (rows * SampleRow::BYTES as u64).next_multiple_of(64);
        let scratch_at = (picks_at + rows * SamplePick::BYTES as u64).next_multiple_of(64);
        let scratch_bytes = SamplerKernels::scratch_bytes(MAX_BATCH);
        let exchange_at = (scratch_at + scratch_bytes).next_multiple_of(64);
        let work_bytes = exchange_at + u64::from(SamplerKernels::exchange_values(MAX_BATCH, 1)) * 4;
        let drafts_bytes = rows * 4;
        ledger.reserve(
            "qwen4exp draft block carries and picks",
            carry_bytes + staged_bytes + work_bytes + drafts_bytes,
        )?;
        let carry = DeviceBuffer::new(device, carry_bytes)?;
        carry.write(0, &vec![0u8; carry_bytes as usize])?;
        let work = DeviceBuffer::new(device, work_bytes)?;
        work.write(scratch_at, &vec![0u8; scratch_bytes as usize])?;
        let picks_host = (rows as usize * SampleRow::BYTES).next_multiple_of(64);
        let drafts_host = (picks_host + rows as usize * SamplePick::BYTES).next_multiple_of(64);
        let report = DraftReport {
            index,
            weights,
            pools: pool_bytes,
            buffers: ledger.reserved() - before - weights - pool_bytes,
        };
        Ok(Self {
            embedding,
            sampler: SamplerKernels::load(device)?,
            pools,
            carry,
            staged: DeviceBuffer::new(device, staged_bytes)?,
            work,
            picks_at,
            scratch_at,
            exchange_at,
            io: PinnedBuffer::new(device, drafts_host + rows as usize * 4)?,
            picks_host,
            drafts_host,
            drafts: DeviceBuffer::new(device, drafts_bytes)?,
            n_drafts: 0,
            picked: Event::new(device)?,
            wrote: Event::new(device)?,
            writing: false,
            tails: Vec::new(),
            serial: 0,
            n: c.n_embd,
            wide,
            n_vocab: c.n_vocab as u32,
            max_slots: opts.max_slots,
            last_chain_ms: 0.0,
            times: DraftTimes::default(),
            watch,
            x: Vec::new(),
            report,
            part,
        })
    }

    /// Where the block leaves its route records.
    pub(super) fn board(&self) -> Arc<Board> {
        self.part.board()
    }

    /// Tells the block where its experts are: layer 0 of the store.
    pub(super) fn set_experts(&mut self, at: ExpertAt) -> Result<(), ModelError> {
        self.part.set_experts(vec![at])
    }

    pub(super) fn report(&self) -> DraftReport {
        self.report
    }

    /// Entries `first..` of the page table of `slot` are now `entries`: the
    /// block reads the tables the model reads.
    pub(super) fn update_table(
        &mut self,
        stream: &Stream,
        slot: u32,
        first: u32,
        entries: impl IntoIterator<Item = u32>,
    ) -> Result<(), ModelError> {
        self.part.update_table(stream, slot, first, entries)
    }

    /// The pool freed these pages: the block gives back what it kept for them.
    pub(super) fn release_pages(&mut self, pages: &[u32]) {
        self.part.release_pages(pages.iter().copied());
    }

    /// Step 1: the cells the model's last step has passed. `accepted` says how
    /// many rows of the window of a slot that started at a row count (`None`
    /// when the record names no outcome for it), and `residual` is where the
    /// model's part left the residual of that step's rows (`D` f32 a row).
    /// Before the tables of the new step are written.
    pub(super) fn follow(
        &mut self,
        stream: &Stream,
        flow: &Flow,
        accepted: &dyn Fn(u32, u32) -> Option<u32>,
        residual: DevicePtr,
    ) -> Result<(), ModelError> {
        let begun = Instant::now();
        let tails = std::mem::take(&mut self.tails);
        let mut rows: Vec<Row> = Vec::new();
        for tail in &tails {
            let n = tail.tokens.len() as u32;
            let counted = if tail.direct {
                n
            } else {
                accepted(tail.slot, tail.row0)
                    .ok_or_else(|| {
                        invalid(format!("no outcome for the window of slot {}", tail.slot))
                    })?
                    .min(n)
            };
            if counted == 0 {
                continue;
            }
            // Draft `j` was verified when every draft before it was accepted.
            for link in 0..tail.links.min(counted) as usize {
                self.times.offered[link] += 1;
                self.times.accepted[link] += u64::from(link as u32 + 1 < counted);
            }
            for r in 0..counted - 1 {
                rows.push(Row {
                    slot: tail.slot,
                    cell: tail.pos0 + r,
                    token: tail.tokens[r as usize + 1],
                    source: Source::Model(tail.row0 + r),
                });
            }
            // The residual of the last row that counts waits for its token.
            let last = u64::from(tail.row0 + counted - 1);
            // SAFETY: a row of the model part's residual, which holds the rows
            // of its last step until its next step begins, and a slot of the
            // carries.
            unsafe {
                stream.copy(
                    self.carry.ptr().offset(u64::from(tail.slot) * self.wide * 4),
                    residual.offset(last * self.wide * 4),
                    (self.wide * 4) as usize,
                )?;
            }
        }
        self.times.key_rows += rows.len() as u64;
        // Rows of a sequence stay together and in order; a step takes
        // `MAX_BATCH` of them.
        for chunk in rows.chunks(MAX_BATCH as usize) {
            self.step(stream, flow, chunk, residual, None)?;
        }
        self.times.key_ms += begun.elapsed().as_secs_f64() * 1e3;
        self.watch.lock().unwrap_or_else(|e| e.into_inner()).times = self.times;
        Ok(())
    }

    /// Steps 2 and 3: the newest cell of every sequence of the step, and the
    /// chains. `tokens` are the tokens of the step's rows; the rows that take
    /// the block's drafts get them. After the tables of the step are written.
    pub(super) fn draft(
        &mut self,
        stream: &Stream,
        flow: &Flow,
        segments: &[Segment],
        tokens: &mut [u32],
        chains: &[Chain],
    ) -> Result<(), ModelError> {
        self.n_drafts = 0;
        self.last_chain_ms = 0.0;
        let drafting = |i: usize| chains.iter().any(|c| c.segment == i);
        // The newest cells of the sequences that do not draft.
        let begun = Instant::now();
        let quiet: Vec<Row> = segments
            .iter()
            .enumerate()
            .filter(|(i, s)| s.pos0 > 0 && !drafting(*i))
            .map(|(_, s)| Row {
                slot: s.slot,
                cell: s.pos0 as u32 - 1,
                token: tokens[s.row0 as usize],
                source: Source::Carry,
            })
            .collect();
        self.times.key_rows += quiet.len() as u64;
        for chunk in quiet.chunks(MAX_BATCH as usize) {
            self.step(stream, flow, chunk, DevicePtr::NULL, None)?;
        }
        self.times.key_ms += begun.elapsed().as_secs_f64() * 1e3;
        if chains.is_empty() {
            return Ok(());
        }
        if chains.len() > MAX_BATCH as usize {
            return Err(invalid(format!("{} chains in a step", chains.len())));
        }

        // The chains, longest first, so that the rows of a link are the first
        // rows of the link before.
        let begun = Instant::now();
        let mut order: Vec<usize> = (0..chains.len()).collect();
        order.sort_by_key(|&c| std::cmp::Reverse(chains[c].rows.len()));
        let n_links = chains[order[0]].rows.len();
        let mut newest: Vec<u32> = Vec::with_capacity(order.len());
        for &c in &order {
            let segment = segments
                .get(chains[c].segment)
                .ok_or_else(|| invalid(format!("the step has no segment {}", chains[c].segment)))?;
            if segment.n_rows as usize != chains[c].rows.len() + 1 {
                return Err(invalid(format!(
                    "segment {segment:?} does not verify {} drafts",
                    chains[c].rows.len()
                )));
            }
            newest.push(tokens[segment.row0 as usize]);
        }
        for j in 0..n_links {
            let live: Vec<usize> =
                order.iter().copied().filter(|&c| chains[c].rows.len() > j).collect();
            let rows: Vec<Row> = live
                .iter()
                .enumerate()
                .map(|(row, &c)| {
                    let s = &segments[chains[c].segment];
                    Row {
                        slot: s.slot,
                        // A window at the start of a sequence has no cell in
                        // front of it (the conductor's probes are such).
                        cell: (s.pos0 as u32).saturating_sub(1) + j as u32,
                        token: newest[row],
                        source: if j == 0 { Source::Carry } else { Source::Link(row as u32) },
                    }
                })
                .collect();
            let params: Vec<SampleRow> = live.iter().map(|&c| chains[c].rows[j]).collect();
            let picks = self
                .step(stream, flow, &rows, DevicePtr::NULL, Some(&params))?
                .ok_or_else(|| invalid("a link without picks"))?;
            if j == 0 {
                // Link 0 is the cell of a certain token; what follows is not.
                for row in &rows {
                    self.part.save_slot(stream, row.slot)?;
                }
            }
            for (row, &c) in live.iter().enumerate() {
                let s = &segments[chains[c].segment];
                tokens[(s.row0 + 1) as usize + j] = picks[row];
                newest[row] = picks[row];
            }
        }
        for &c in &order {
            self.part.restore_slot(stream, segments[chains[c].segment].slot)?;
        }

        // The drafts in the order of the step's segments, for the runtime.
        let mut by_segment: Vec<&Chain> = chains.iter().collect();
        by_segment.sort_by_key(|c| c.segment);
        let mut n = 0usize;
        let mut watch = self.watch.lock().unwrap_or_else(|e| e.into_inner());
        for chain in by_segment {
            let s = &segments[chain.segment];
            let rows = (s.row0 + 1) as usize..(s.row0 + s.n_rows) as usize;
            if watch.recent.len() == Watch::KEPT {
                watch.recent.pop_front();
            }
            watch.recent.push_back(Drafted {
                slot: s.slot,
                position: s.pos0 as u32 + 1,
                token: tokens[s.row0 as usize],
                drafts: tokens[rows].to_vec(),
            });
            for j in 0..chain.rows.len() {
                if n >= MAX_BATCH as usize {
                    return Err(invalid("more drafts than a step of the block has rows"));
                }
                let token = tokens[(s.row0 + 1) as usize + j];
                // SAFETY: the drafts' section of the pinned buffer holds
                // `MAX_BATCH` words.
                unsafe {
                    let to = self.io.as_mut_ptr().add(self.drafts_host + 4 * n).cast::<u32>();
                    to.write_unaligned(token);
                }
                n += 1;
            }
        }
        // SAFETY: the section is not written again before the next step's
        // chains, which run after this step has ended.
        unsafe {
            stream.copy_from_host(
                self.drafts.ptr(),
                self.io.as_ptr().add(self.drafts_host),
                4 * n,
            )?;
        }
        self.n_drafts = n as u32;
        self.last_chain_ms = begun.elapsed().as_secs_f64() * 1e3;
        self.times.chain_steps += 1;
        self.times.links += n_links as u64;
        self.times.chain_ms += self.last_chain_ms;
        watch.times = self.times;
        Ok(())
    }

    /// Remembers the rows of the model's step that is enqueued now: the next
    /// step follows them.
    pub(super) fn keep(&mut self, segments: &[Segment], tokens: &[u32], chains: &[Chain]) {
        self.tails.clear();
        self.tails.extend(segments.iter().enumerate().map(|(i, s)| Tail {
            slot: s.slot,
            pos0: s.pos0.max(0) as u32,
            row0: s.row0,
            tokens: tokens[s.row0 as usize..(s.row0 + s.n_rows) as usize].to_vec(),
            direct: s.direct,
            links: chains.iter().find(|c| c.segment == i).map_or(0, |c| c.rows.len() as u32),
        }));
    }

    /// Where the step left the drafts and how many.
    pub(super) fn drafted(&self) -> Option<(DevicePtr, u32)> {
        (self.n_drafts > 0).then(|| (self.drafts.ptr(), self.n_drafts))
    }

    /// Milliseconds the chains of the last step took.
    pub(super) fn chain_ms(&self) -> f64 {
        self.last_chain_ms
    }

    /// One step of the block for `rows` (at most [`MAX_BATCH`], the rows of a
    /// sequence consecutive and in the order of their cells). With `sample`
    /// the rows run the whole block and pick, and the picks are returned once
    /// the device has made them; without, they write keys only and nothing is
    /// waited for.
    fn step(
        &mut self,
        stream: &Stream,
        flow: &Flow,
        rows: &[Row],
        model: DevicePtr,
        sample: Option<&[SampleRow]>,
    ) -> Result<Option<Vec<u32>>, ModelError> {
        let n = rows.len() as u32;
        if n == 0 || n > MAX_BATCH {
            return Err(invalid(format!("a step of the draft block of {n} rows")));
        }
        let row_bytes = self.wide * 4;
        // The inputs of a step are staged in host memory the device copies
        // from when it gets there: a step that nobody waited for has to have
        // run before they are written again.
        if std::mem::take(&mut self.writing) {
            let started = Instant::now();
            while !self.wrote.is_done()? {
                flow.check(started, "the draft block wrote keys")?;
                flow.pump()?;
                std::hint::spin_loop();
            }
        }
        // The residuals, gathered; rows of the model that follow each other
        // are one copy.
        let link = self.part.residual();
        let mut at = 0usize;
        while at < rows.len() {
            let mut run = 1;
            let from = match rows[at].source {
                Source::Carry => {
                    if rows[at].slot >= self.max_slots {
                        return Err(invalid(format!("slot {} of the carries", rows[at].slot)));
                    }
                    self.carry.ptr().offset(u64::from(rows[at].slot) * row_bytes)
                }
                Source::Link(row) => link.offset(u64::from(row) * row_bytes),
                Source::Model(row) => {
                    while at + run < rows.len()
                        && rows[at + run].source == Source::Model(row + run as u32)
                    {
                        run += 1;
                    }
                    model.offset(u64::from(row) * row_bytes)
                }
            };
            // SAFETY: the staged rows hold `MAX_BATCH` rows; the sources are
            // rows of the carries, of the block part's residual of its last
            // step and of the model part's of its last step (the caller's).
            unsafe {
                stream.copy(
                    self.staged.ptr().offset(at as u64 * row_bytes),
                    from,
                    run * row_bytes as usize,
                )?;
            }
            at += run;
        }
        self.x.resize(rows.len() * self.n, 0.0);
        for (row, x) in rows.iter().zip(self.x.chunks_exact_mut(self.n)) {
            self.embedding.decode(row.token, x)?;
        }
        // SAFETY: the staged rows were enqueued above and are not written
        // again before the next step of the block.
        let input = unsafe { self.part.block_input(stream, self.staged.ptr(), &self.x, n)? };

        // The rows of a sequence are one segment.
        let mut segments: Vec<Segment> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            match segments.last_mut() {
                Some(s) if s.slot == row.slot && s.pos0 as u32 + s.n_rows == row.cell => {
                    s.n_rows += 1;
                }
                _ => segments.push(Segment {
                    row0: i as u32,
                    n_rows: 1,
                    slot: row.slot,
                    pos0: row.cell as i32,
                    direct: true,
                }),
            }
        }
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        let out_rows: Vec<u32> = if sample.is_some() { (0..n).collect() } else { Vec::new() };
        let record = StepRecord::new(&segments, &tokens, None, &out_rows)?;
        // The token marks the block's route records: never 0, and never the
        // same for two steps that follow each other.
        self.serial = self.serial.wrapping_add(1).max(1);
        let env = StepEnv {
            stream,
            record: &record,
            segments: &segments,
            token: self.serial,
            x: &[],
            ngram: None,
            inbound: Some(input),
            xbase: 0,
            pools: self.pools.ptr(),
        };
        self.part.begin(&env)?;
        let Some(params) = sample else {
            self.part.enqueue_keys(&env, 0)?;
            self.part.end_keys(&env)?;
            self.wrote.record(stream)?;
            self.writing = true;
            return Ok(None);
        };
        if params.len() != rows.len() {
            return Err(invalid("a sampler row for every row of a link"));
        }
        self.part.enqueue_layer(&env, 0)?;
        flow.lock_feed().draft_routed(n);
        self.part.end(&env)?;
        // SAFETY: the sections of the pinned buffer and of the work buffer
        // hold `MAX_BATCH` rows; the logits are those of the `n` rows the
        // block part just enqueued; the scratch was cleared when the block
        // was opened and only the sampler writes it.
        unsafe {
            for (i, row) in params.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    row.to_bytes().as_ptr(),
                    self.io.as_mut_ptr().add(i * SampleRow::BYTES),
                    SampleRow::BYTES,
                );
            }
            stream.copy_from_host(
                self.work.ptr(),
                self.io.as_ptr(),
                params.len() * SampleRow::BYTES,
            )?;
            let batch = SampleBatch {
                logits: self.part.head_logits(),
                stride: self.n_vocab,
                n_rows: n,
                capacity: MAX_BATCH,
                n_local: self.n_vocab,
                tokens: DevicePtr::NULL,
                first: 0,
                inverse: DevicePtr::NULL,
                vocab: 0,
                params: self.work.ptr(),
                entries: DevicePtr::NULL,
                masks: DevicePtr::NULL,
                mask_words: 0,
                out: self.work.ptr().offset(self.picks_at),
                logprobs: DevicePtr::NULL,
                scratch: self.work.ptr().offset(self.scratch_at),
                exchange: self.work.ptr().offset(self.exchange_at),
                devices: 1,
                rank: 0,
            };
            self.sampler.sample(stream, &batch, SamplePlan::of(params), |_, _| {
                Ok::<(), ares_cuda::Error>(())
            })?;
            stream.copy_to_host(
                self.io.as_mut_ptr().add(self.picks_host),
                self.work.ptr().offset(self.picks_at),
                rows.len() * SamplePick::BYTES,
            )?;
        }
        self.picked.record(stream)?;
        // The host goes on when the picks are here; the store has brought
        // what the block's router asked for by then.
        let started = Instant::now();
        loop {
            if self.picked.is_done()? && flow.lock_feed().draft_followed() {
                break;
            }
            flow.check(started, "the draft block made its picks")?;
            flow.pump()?;
            std::hint::spin_loop();
        }
        let mut picks = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            // SAFETY: the copy of the picks has run.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.io.as_ptr().add(self.picks_host + i * SamplePick::BYTES),
                    SamplePick::BYTES,
                )
            };
            let pick = SamplePick::from_bytes(bytes);
            if pick.status != 0 || pick.token >= self.n_vocab {
                return Err(invalid(format!(
                    "the draft block's row {i}: the sampler reported status {:#x} and token {}",
                    pick.status, pick.token
                )));
            }
            picks.push(pick.token);
        }
        Ok(Some(picks))
    }
}
