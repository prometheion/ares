//! The MTP driver: one for every model that drafts with an MTP block (`mtp`,
//! multi-token prediction: one more layer of the model's own kind).
//!
//! The model gives the driver its MTP layer ([`MtpLayer`]: its launch plan
//! for a link and for rows that only write keys and values) and where its
//! hidden state is tapped (the rows of a step's last hidden state, a row of
//! `width` f32 per row). Everything else is here and the same for every
//! model:
//!
//! - **Pairs and cells.** The block reads a token and the model's hidden state
//!   of the position before it. The pair of the token at position `p` is
//!   written to the block's cell `p - lag` ([`MtpLayer::lag`]: 0 for the 27B,
//!   1 for Flash-Next, whose cell `i` pairs residual `i` with token `i + 1`).
//! - **The carry.** Per sequence slot the hidden state of the sequence's last
//!   row that counts, which the next pair starts from. A slot past the
//!   sequence slots holds zeros: a sequence's first pair has no hidden state
//!   before it.
//! - **Following the model** ([`Driver::follow`]). At the start of a step the
//!   rows of the model's last step that count (every row of a prompt slice or
//!   a single row, the accepted rows of a draft window, which the record of
//!   this step tells) give the block the pairs behind their first: they only
//!   write keys and values. The hidden state of the last row that counts
//!   becomes the carry. A replay's cells are cached, the block's with them:
//!   it only moves the carry. Nothing the block writes is ever undone: the
//!   rows of a window that were rejected are never followed.
//! - **The newest cell** ([`Driver::step`]). The first row of a sequence in a
//!   step names the token the carry is paired with. For a sequence that
//!   drafts this is link 0 of its chain; for any other it only writes keys
//!   and values.
//! - **The chain.** Link `j` pairs the pick of link `j - 1` with the block's
//!   own output of that link. Each pick is made on the device by the sampler,
//!   from the draft head (the head view's rows, see `blocks/head.rs`), with
//!   the settings and the noise of the row that verifies it, and written
//!   straight into the token array of the step record, into the next link's
//!   record (whose rows of the embedding table the device keeps: the head
//!   view's tokens) and into the step's list of drafts. The host reads every
//!   link's picks for the probability the block gave each draft; it waits for
//!   them only where the step's draft bars may end a chain before the next
//!   link, and a link that is not made costs nothing (its rows of the verify
//!   window get the last draft again, and the host is told they are none).
//!   The cells of links after the first are **provisional**: per-sequence
//!   state the layer keeps beside its cells is saved behind link 0 and put
//!   back behind the last link ([`MtpLayer::save`]), and the keys and values
//!   are written again when the model has passed the cells.
//!
//! What a step uploads for the block is written on the host and sent with one
//! copy per phase (following, and the newest cells with the chains); the
//! layer adds its own bytes for each launch ([`MtpLayer::bytes`]: an
//! attention plan, the rows of a K/V append).

use std::ops::Range;
use std::sync::Arc;

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, ENTRY_CELLS, EmbedRows, Event, KeptRows, LayerKernels,
    MAX_BATCH, PinnedBuffer, SampleBatch, SamplePlan, SamplerKernels, Segment, StepRecord, Stream,
};
use ares_runtime::{DraftChances, ModelError, draft_chance};
use ares_sampler::{SamplePick, SampleRow};

use super::invalid;
use super::rows::{HostTable, TableShape, source};

/// A sequence that drafts with the block in the coming step.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpChain {
    /// The segment of the step that verifies the drafts: its first row is
    /// the sequence's newest token, its other rows take the drafts.
    pub segment: u32,
    /// The newest token.
    pub token: u32,
    /// How the rows that verify the drafts are sampled, one per draft: link
    /// `j` of the chain and row `j` of the window pick the same position.
    pub rows: Vec<SampleRow>,
    /// The reach the chain needs for each link to be made (the step record's
    /// draft bars); empty for a chain that makes every link.
    pub needs: Vec<f32>,
    /// Where its drafts go in the step's list of drafts, which the drafts of
    /// every drafter of the part share in the order of the record's windows.
    pub place: u32,
}

/// What the block drafts in the coming step.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpStep {
    /// The token array of the step record on the block's device: the chain
    /// writes the drafts into it.
    pub tokens: DevicePtr,
    pub chains: Vec<MtpChain>,
    /// The chance of a draft by its probability, which the reach of a chain
    /// is made of; `None` when the chains make every link.
    pub chances: Option<DraftChances>,
}

/// Where a step left the tokens its drafters drafted: `tokens` u32 on the
/// device, at the places of their windows. `u32::MAX` is a draft that could
/// not be picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drafted {
    pub ptr: DevicePtr,
    pub tokens: u32,
}

/// Device time of the block, what it wrote and how its drafts fared, added
/// up over steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MtpTimes {
    /// Steps in which a chain ran.
    pub steps: u64,
    /// Links of all chains of those steps (a link of several sequences
    /// counts once).
    pub links: u64,
    /// Milliseconds of the chains on the device.
    pub chain_ms: f64,
    /// Rows that only wrote keys and values (following the model and the
    /// newest cells of sequences that did not draft).
    pub key_rows: u64,
    /// Per link of a chain: how often its draft was verified (every draft
    /// before it had been accepted) and how often it was accepted then.
    pub offered: [u64; MAX_BATCH as usize],
    pub accepted: [u64; MAX_BATCH as usize],
}

/// A launch of the layer: rows of the block, each a pair of a token and the
/// hidden state before it, as one step.
#[derive(Debug)]
pub(crate) struct Rows<'a> {
    pub stream: &'a Stream,
    /// The rows' segments: a sequence's rows are consecutive cells.
    pub segments: &'a [Segment],
    /// The record of the rows (positions are the cells), on the host and on
    /// the device. The host's tokens of a link after the first are none: the
    /// device writes them.
    pub record: &'a StepRecord,
    pub step: DevicePtr,
    /// The rows of the launch that read their cells and write nothing.
    pub replay: &'a [Range<u32>],
    /// The rows' token embeddings, `n_embd` f32 each, and the hidden states
    /// the pairs start from, `width` f32 each.
    pub e: DevicePtr,
    pub h: DevicePtr,
    /// What [`MtpLayer::bytes`] wrote for this launch, on the device.
    pub bytes: DevicePtr,
    /// The step's page tables and pages.
    pub tables: DevicePtr,
    pub pages: DevicePtr,
}

/// The logits a link left: a row of `stride` f32 per row, column `i` the
/// `i`-th row of the draft head.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinkLogits {
    pub ptr: DevicePtr,
    pub stride: u32,
}

/// What a model gives the driver: its MTP layer.
pub(crate) trait MtpLayer {
    /// Positions between a pair's token and the block's cell of the pair.
    fn lag(&self) -> u32;
    /// Where the embeddings and the hidden states of rows that only write
    /// keys go (`keys_rows` rows each), when the layer has room for them in
    /// its own scratch; `None` and the driver keeps them.
    fn keys_inputs(&self) -> Option<(DevicePtr, DevicePtr)>;
    /// Writes what a launch of the layer reads besides the record (an
    /// attention plan, the rows of an append) into `out`, before any launch
    /// of the phase is enqueued: `links` says whether the rows run the whole
    /// layer. Returns the bytes written.
    fn bytes(
        &mut self,
        segments: &[Segment],
        links: bool,
        out: &mut [u8],
    ) -> Result<usize, ModelError>;
    /// The keys and values of the rows' cells (and any state the layer keeps
    /// per cell): the layer up to its write, nothing read.
    fn keys(&mut self, rows: &Rows<'_>) -> Result<(), ModelError>;
    /// A link: the whole layer on the rows, its final norm and the draft
    /// head. Writes each row's output, the next link's hidden state, to `out`
    /// (`width` f32 a row) and leaves the logits where it says; `logits` has
    /// room for them in the driver's scratch.
    fn link(
        &mut self,
        rows: &Rows<'_>,
        out: DevicePtr,
        logits: DevicePtr,
    ) -> Result<LinkLogits, ModelError>;
    /// Keeps the per-sequence state of `slots` that links change, behind the
    /// first link; [`MtpLayer::restore`] puts it back behind the last.
    fn save(&mut self, stream: &Stream, slots: &[u32]) -> Result<(), ModelError>;
    fn restore(&mut self, stream: &Stream, slots: &[u32]) -> Result<(), ModelError>;
    /// Whether a row of `slot` that writes cell `cell` would write into a tile
    /// the block's K/V group holds converted (a cached page another sequence
    /// went past): such a row reads and writes nothing, as a replay.
    fn writes_converted(&self, slot: u32, cell: u32) -> bool;
    /// The first `cells` cells of `slot` are settled in the block's K/V group:
    /// the pages they fill are full.
    fn settle(&mut self, slot: u32, cells: u32);
}

/// What the driver needs to know of its device and its model.
#[derive(Clone, Debug)]
pub(crate) struct DriverSetup {
    /// For the ledger's entries.
    pub label: &'static str,
    /// Values of the hidden state a pair starts from.
    pub width: u32,
    pub max_slots: u32,
    /// The most rows of a model step, which following may give.
    pub max_rows: u32,
    /// The most rows of a draft window: the chain makes one link fewer.
    pub window_rows: u32,
    /// The rows of the draft head (the head view's), and the tokens of its
    /// columns on the device (null: column `i` is token `i`).
    pub head_rows: u32,
    pub head_tokens: DevicePtr,
    /// The token embedding table, the rows of it the device keeps (the head
    /// view's tokens), and the table on the host for the rest.
    pub table: TableShape,
    pub kept: KeptRows,
    pub host: Option<Arc<HostTable>>,
}

/// What the model's last step left of a sequence for the block to follow.
#[derive(Clone, Debug)]
struct Tail {
    slot: u32,
    pos0: u32,
    row0: u32,
    tokens: Vec<u32>,
    direct: bool,
    replay: bool,
    /// The drafts of the block among the rows: 0 for rows that took none.
    links: u32,
}

/// One row of a launch that only writes keys, before it is laid out.
#[derive(Clone, Copy, Debug)]
struct KeyRow {
    slot: u32,
    /// The position of the pair's token.
    pair: u32,
    token: u32,
    /// The row of the hidden states it starts from: of the model's last step,
    /// or of the carries (a slot; `max_slots` for zeros).
    hidden: u32,
}

/// A launch that only writes keys, as laid out in the upload.
#[derive(Debug)]
struct KeyLaunch {
    segments: Vec<Segment>,
    record: StepRecord,
    replay: Vec<Range<u32>>,
    at_record: u64,
    at_staged: u64,
    at_index: u64,
    at_bytes: u64,
}

/// A link of a step's chains, as its kernels need it.
#[derive(Debug)]
struct Link {
    /// Rows: the chains that are at least this long.
    n: u32,
    /// The rows of the next link.
    n_next: u32,
    segments: Vec<Segment>,
    record: StepRecord,
    /// Offsets in the upload: the record, its token array, the sampler rows,
    /// the rows of the step record the picks verify, the places in the list,
    /// the layer's bytes.
    at_record: u64,
    at_tokens: u64,
    at_params: u64,
    at_record_rows: u64,
    at_draft_at: u64,
    at_bytes: u64,
    sample: SamplePlan,
}

/// The events around one step's chains.
#[derive(Debug)]
struct Clock {
    chain: [Event; 2],
    pending: bool,
    links: u64,
    last_chain_ms: f64,
}

/// Offsets of the driver's scratch.
#[derive(Clone, Copy, Debug, Default)]
struct Scratch {
    /// The chains' embeddings and hidden states, and the same for rows that
    /// only write keys when the layer keeps none of its own.
    e: u64,
    h: u64,
    keys_e: u64,
    keys_h: u64,
    /// Per link the output of every chain: the next link's hidden state.
    out: u64,
    logits: u64,
    /// Per link the picks of every chain.
    picks: u64,
    drafts: u64,
    sampler: u64,
    exchange: u64,
    len: u64,
}

/// The MTP driver of a device part.
#[derive(Debug)]
pub(crate) struct Driver {
    sampler: SamplerKernels,
    setup: DriverSetup,
    /// Per slot the hidden state its next pair starts from, and a row of
    /// zeros behind them.
    carry: DeviceBuffer,
    scratch: DeviceBuffer,
    s: Scratch,
    /// What a step uploads, in two halves: following (`0..half`) and the
    /// step's own rows (`half..`).
    host: PinnedBuffer,
    records: DeviceBuffer,
    half: u64,
    keys_rows: u32,
    max_links: u32,
    pending: Option<MtpStep>,
    tails: Vec<Tail>,
    /// The newest cells (slot, pair) that a step that only drafted wrote as
    /// link 0: the model's step that verifies the drafts does not write them
    /// again.
    newest_done: Vec<(u32, u32)>,
    drafted: u32,
    /// The picks of every link of the step, as the device copies them out,
    /// and the event behind the newest copy.
    picks_host: PinnedBuffer,
    picked: Event,
    /// The links the chains of the step made, and where the list has the
    /// draft of each row of each link: `(link, row, place)`.
    made: u32,
    layout: Vec<(u32, u32, u32)>,
    /// The links each sequence's chain made, by its slot, until the model's
    /// step that verifies them is kept.
    chain_links: Vec<(u32, u32)>,
    clock: Clock,
    times: MtpTimes,
}

fn words(buf: &mut [u8], at: u64, values: impl IntoIterator<Item = u32>) -> u64 {
    let mut end = at as usize;
    for v in values {
        buf[end..end + 4].copy_from_slice(&v.to_le_bytes());
        end += 4;
    }
    (end as u64).next_multiple_of(16)
}

/// Bytes a record of `rows` rows in `segments` segments takes, at most.
fn record_bytes(rows: u64, segments: u64) -> u64 {
    (64 + 32 * segments + 28 * rows).next_multiple_of(64)
}

impl Driver {
    /// Reserves and allocates what the driver needs on `device`. `layer`
    /// says the rows of a launch that only writes keys and how many bytes of
    /// its own a launch of `rows` rows may need (`layer_bytes`).
    pub fn open(
        device: &Device,
        ledger: &mut Ledger,
        setup: DriverSetup,
        keys_rows: u32,
        keys_inputs: bool,
        layer_bytes: impl Fn(u32, bool) -> u64,
    ) -> Result<Self, ModelError> {
        let chains = u64::from(setup.max_slots);
        let links = u64::from(setup.window_rows.saturating_sub(1).max(1));
        let (width, n_embd) = (u64::from(setup.width), u64::from(setup.table.n_embd));
        let keys = u64::from(keys_rows.max(1));
        let mut s = Scratch::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        s.e = take(chains * n_embd * 4);
        s.h = take(chains * width * 4);
        if !keys_inputs {
            s.keys_e = take(keys * n_embd * 4);
            s.keys_h = take(keys * width * 4);
        }
        s.out = take(links * chains * width * 4);
        s.logits = take(chains * u64::from(setup.head_rows) * 4);
        s.picks = take(links * chains * SamplePick::BYTES as u64);
        s.drafts = take(links * chains * 4);
        s.sampler = take(SamplerKernels::scratch_bytes(setup.max_slots));
        s.exchange = take(u64::from(SamplerKernels::exchange_values(setup.max_slots, 1)) * 4);
        s.len = len;
        let carry_bytes = (chains + 1) * width * 4;

        // Following: every row of a step, in launches of `keys` rows, each a
        // record, staged rows, an index and the layer's bytes.
        let row_bytes = setup.table.row_bytes();
        let follow_rows = u64::from(setup.max_rows) + chains;
        let launches = follow_rows.div_ceil(keys) + 1;
        let per_launch = record_bytes(keys, keys.min(chains + 1)) + 64 * 4;
        let follow = launches * (per_launch + layer_bytes(keys_rows, false).next_multiple_of(64))
            + follow_rows * (row_bytes + 4)
            + 256;
        // The step: the newest cells of the sequences that do not draft, as
        // above, then per link a record, the settings, two index arrays and
        // the layer's bytes; then the carry index of link 0 and its staged
        // rows.
        let quiet = record_bytes(chains, chains) + chains * (row_bytes + 4) + 64 * 4;
        let per_link = record_bytes(chains, chains)
            + chains * (SampleRow::BYTES as u64 + 8)
            + 64 * 3
            + layer_bytes(setup.max_slots, true).next_multiple_of(64);
        let step = quiet
            + layer_bytes(setup.max_slots, false).next_multiple_of(64)
            + links * per_link
            + chains * (4 + row_bytes)
            + 256;
        let half = follow.next_multiple_of(64);
        let upload = half + step.next_multiple_of(64);
        let label = setup.label;
        ledger.reserve(format!("{label}: hidden states kept"), carry_bytes)?;
        ledger.reserve(format!("{label}: chain scratch"), s.len)?;
        ledger.reserve(format!("{label}: step records"), upload)?;
        let carry = DeviceBuffer::new(device, carry_bytes)?;
        carry.write(0, &vec![0u8; carry_bytes as usize])?;
        let scratch = DeviceBuffer::new(device, s.len)?;
        scratch.write(
            s.sampler,
            &vec![0u8; SamplerKernels::scratch_bytes(setup.max_slots) as usize],
        )?;
        Ok(Self {
            sampler: SamplerKernels::load(device)?,
            carry,
            scratch,
            s,
            host: PinnedBuffer::new(device, upload as usize)?,
            records: DeviceBuffer::new(device, upload)?,
            half,
            keys_rows: keys_rows.max(1),
            max_links: links as u32,
            pending: None,
            tails: Vec::new(),
            newest_done: Vec::new(),
            drafted: 0,
            picks_host: PinnedBuffer::new(device, (links * chains) as usize * SamplePick::BYTES)?,
            picked: Event::new(device)?,
            made: 0,
            layout: Vec::new(),
            chain_links: Vec::new(),
            clock: Clock {
                chain: [Event::timed(device)?, Event::timed(device)?],
                pending: false,
                links: 0,
                last_chain_ms: 0.0,
            },
            times: MtpTimes::default(),
            setup,
        })
    }

    fn at(&self, offset: u64) -> DevicePtr {
        self.scratch.ptr().offset(offset)
    }

    fn rec(&self, offset: u64) -> DevicePtr {
        self.records.ptr().offset(offset)
    }

    /// The most drafts the block makes per sequence in a step.
    pub fn max_links(&self) -> u32 {
        self.max_links
    }

    /// The rows of the draft head.
    pub fn head_rows(&self) -> u32 {
        self.setup.head_rows
    }

    /// The hidden states kept per sequence slot (`width` f32 each), which a
    /// checkpoint keeps of the block.
    pub fn carry(&self) -> &DeviceBuffer {
        &self.carry
    }

    /// Bytes of the carries, the scratch and the step records.
    pub fn bytes(&self) -> (u64, u64, u64) {
        (self.carry.len(), self.scratch.len(), self.records.len())
    }

    /// The device time of the block so far; the last step's chains count
    /// once the next step was enqueued or [`Driver::chain_ms`] was asked.
    pub fn times(&self) -> MtpTimes {
        self.times
    }

    /// The step's list of drafts, which every drafter of the part writes at
    /// the places of its windows.
    pub fn list(&self) -> DevicePtr {
        self.at(self.s.drafts)
    }

    /// The drafts the step enqueued last made, at most: 0 when it made none.
    pub fn drafted(&self) -> u32 {
        self.drafted
    }

    /// The chain scratch and, in it, the offsets of the output of link
    /// `link` (a row of `width` values per chain), of the logits of the last
    /// link that ran (a row of the draft head per chain) and of the picks of
    /// link `link` ([`SamplePick`] records): for tests.
    pub fn scratch(&self, link: u32) -> (&DeviceBuffer, [u64; 3]) {
        let rows = u64::from(self.setup.max_slots);
        let width = u64::from(self.setup.width);
        (
            &self.scratch,
            [
                self.s.out + u64::from(link) * rows * width * 4,
                self.s.logits,
                self.s.picks + u64::from(link) * rows * SamplePick::BYTES as u64,
            ],
        )
    }

    /// Tells the driver what the block drafts in the step that is enqueued
    /// next. Without this call a step drafts nothing.
    pub fn draft_step(&mut self, step: MtpStep) -> Result<(), ModelError> {
        if step.chains.len() > self.setup.max_slots as usize
            || step
                .chains
                .iter()
                .any(|c| c.rows.is_empty() || c.rows.len() > self.max_links as usize)
        {
            return Err(invalid(format!(
                "{} sequences draft, the longest {} tokens; the MTP block takes {} sequences of 1 \
                 to {}",
                step.chains.len(),
                step.chains.iter().map(|c| c.rows.len()).max().unwrap_or(0),
                self.setup.max_slots,
                self.max_links
            )));
        }
        self.pending = Some(step);
        Ok(())
    }

    /// Adds the events of the last step's chains to the times. The step has
    /// run.
    fn flush_clock(&mut self) -> Result<(), ModelError> {
        if std::mem::take(&mut self.clock.pending) {
            let clock = &mut self.clock;
            clock.last_chain_ms = f64::from(clock.chain[1].elapsed_ms_since(&clock.chain[0])?);
            self.times.steps += 1;
            self.times.chain_ms += clock.last_chain_ms;
            self.times.links += std::mem::take(&mut clock.links);
        }
        Ok(())
    }

    /// Device milliseconds of the chains of the step that ran last (0 when it
    /// drafted nothing). The step has ended.
    pub fn chain_ms(&mut self) -> Result<f64, ModelError> {
        self.flush_clock()?;
        Ok(self.clock.last_chain_ms)
    }

    /// How sure link `link` was of the pick of its row `row`. The copy of the
    /// link's picks has run.
    fn confidence(&self, link: u32, row: u32) -> f32 {
        let at = (link as usize * self.setup.max_slots as usize + row as usize) * SamplePick::BYTES;
        // SAFETY: the buffer holds a record for every row of every link, and
        // the device is not writing this one any more.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.picks_host.as_ptr().add(at), SamplePick::BYTES)
        };
        SamplePick::from_bytes(bytes).confidence()
    }

    /// The probability the block gave each draft of the step that ran last,
    /// written into `out` at the drafts' places; NaN stays where a link was
    /// not made. The step has ended.
    pub fn probs(&self, out: &mut [f32]) {
        if self.drafted == 0 {
            return;
        }
        for &(link, row, place) in self.layout.iter().filter(|(link, ..)| *link < self.made) {
            if let Some(p) = out.get_mut(place as usize) {
                *p = self.confidence(link, row);
            }
        }
    }

    /// Where a gather finds the rows of the tokens of a launch whose rows the
    /// host staged at `staged` (null for none).
    fn rows(&self, staged: DevicePtr) -> EmbedRows {
        source(&self.setup.table, self.setup.kept, staged)
    }

    /// Writes the record, the staged rows, the hidden-state index and the
    /// layer's bytes of `rows` into the upload from `end` on, as launches of
    /// at most the layer's rows; a sequence's rows stay consecutive.
    fn lay_out_keys(
        &mut self,
        layer: &mut dyn MtpLayer,
        rows: &[KeyRow],
        replay: &[bool],
        mut end: u64,
        limit: u64,
    ) -> Result<(Vec<KeyLaunch>, u64), ModelError> {
        let full = || invalid("the MTP block's step records do not fit their buffer");
        let size = self.setup.table.row_bytes() as usize;
        let lag = layer.lag();
        let mut launches = Vec::new();
        let per = self.keys_rows as usize;
        for (chunk, flags) in rows.chunks(per).zip(replay.chunks(per)) {
            let mut segments: Vec<Segment> = Vec::new();
            let mut replay_rows = Vec::new();
            for (i, (row, &replays)) in chunk.iter().zip(flags).enumerate() {
                let cell = row.pair - lag;
                match segments.last_mut() {
                    Some(s) if s.slot == row.slot && s.pos0 as u32 + s.n_rows == cell => {
                        s.n_rows += 1;
                    }
                    _ => segments.push(Segment {
                        row0: i as u32,
                        n_rows: 1,
                        slot: row.slot,
                        pos0: cell as i32,
                        direct: true,
                    }),
                }
                if replays {
                    replay_rows.push(i as u32..i as u32 + 1);
                }
            }
            let tokens: Vec<u32> = chunk.iter().map(|r| r.token).collect();
            let mut record = StepRecord::new(&segments, &tokens, None, &[])?;
            for (i, s) in segments.iter().enumerate() {
                if replay_rows.iter().any(|r| r.contains(&s.row0)) {
                    record.mark_replay(i)?;
                }
            }
            let bytes = record.as_bytes();
            let at_record = end;
            let at_staged = (at_record + bytes.len() as u64).next_multiple_of(16);
            let at_index = (at_staged + (chunk.len() * size) as u64).next_multiple_of(16);
            let at_bytes = (at_index + 4 * chunk.len() as u64).next_multiple_of(16);
            if at_bytes > limit {
                return Err(full());
            }
            let buf: &mut [u8] = &mut self.host;
            buf[at_record as usize..at_record as usize + bytes.len()].copy_from_slice(bytes);
            if let Some(host) = &self.setup.host {
                let stage = &mut buf[at_staged as usize..at_index as usize];
                host.stage(&tokens, stage)?;
            }
            words(buf, at_index, chunk.iter().map(|r| r.hidden));
            let written =
                layer.bytes(&segments, false, &mut buf[at_bytes as usize..limit as usize])?;
            end = (at_bytes + written as u64).next_multiple_of(64);
            launches.push(KeyLaunch {
                segments,
                record,
                replay: replay_rows,
                at_record,
                at_staged,
                at_index,
                at_bytes,
            });
        }
        Ok((launches, end))
    }

    /// Enqueues launches that only write keys, laid out by
    /// [`Driver::lay_out_keys`]; their hidden states are rows of `hidden`.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_keys(
        &self,
        layer: &mut dyn MtpLayer,
        layers: &LayerKernels,
        stream: &Stream,
        launches: &[KeyLaunch],
        hidden: DevicePtr,
        tables: DevicePtr,
        pages: DevicePtr,
    ) -> Result<(), ModelError> {
        let (e, h) =
            layer.keys_inputs().unwrap_or_else(|| (self.at(self.s.keys_e), self.at(self.s.keys_h)));
        for launch in launches {
            let dims = launch.record.dims();
            let step = self.rec(launch.at_record);
            let staged = if self.setup.host.is_some() {
                self.rec(launch.at_staged)
            } else {
                DevicePtr::NULL
            };
            // SAFETY: the records hold what `lay_out_keys` wrote for this
            // launch, the inputs `keys_rows` rows, the carries and `hidden`
            // every row the index names.
            unsafe {
                layers.embed_gather(stream, step, dims, &self.rows(staged), e)?;
                self.sampler.gather_hidden(
                    stream,
                    h,
                    hidden,
                    self.rec(launch.at_index),
                    dims.n_rows,
                    self.setup.width,
                )?;
            }
            layer.keys(&Rows {
                stream,
                segments: &launch.segments,
                record: &launch.record,
                step,
                replay: &launch.replay,
                e,
                h,
                bytes: self.rec(launch.at_bytes),
                tables,
                pages,
            })?;
        }
        Ok(())
    }

    /// Step 1, at the start of a step, before its copies of the prefix cache
    /// and its tables: the pairs behind the first of every row of the model's
    /// last step that counts, and the carries. `accepted` says how many rows
    /// of the window of a slot that started at a row count (`None` when the
    /// record names no outcome for it); `tap` is where the model left the
    /// hidden states of that step's rows (`width` f32 each), which are still
    /// there.
    #[allow(clippy::too_many_arguments)]
    pub fn follow(
        &mut self,
        layer: &mut dyn MtpLayer,
        layers: &LayerKernels,
        stream: &Stream,
        accepted: &dyn Fn(u32, u32) -> Option<u32>,
        tap: DevicePtr,
        tables: DevicePtr,
        pages: DevicePtr,
    ) -> Result<(), ModelError> {
        let tails = std::mem::take(&mut self.tails);
        let lag = layer.lag();
        let width = u64::from(self.setup.width) * 4;
        let mut rows: Vec<KeyRow> = Vec::new();
        // The sequences that end at the end of a page while the block lags:
        // the block's last cell of the page waits for a token that may never
        // come.
        let mut at_page_end: Vec<(u32, u32)> = Vec::new();
        let mut carries: Vec<(u32, u32)> = Vec::new();
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
            for link in 0..tail.links.min(counted).min(MAX_BATCH) as usize {
                self.times.offered[link] += 1;
                self.times.accepted[link] += u64::from(link as u32 + 1 < counted);
            }
            let cells = tail.pos0 + counted;
            if lag > 0 && cells.is_multiple_of(ENTRY_CELLS) {
                at_page_end.push((tail.slot, cells));
            }
            // The cells of a replay are cached, the block's with them.
            if !tail.replay {
                rows.extend((1..counted).map(|r| KeyRow {
                    slot: tail.slot,
                    pair: tail.pos0 + r,
                    token: tail.tokens[r as usize],
                    hidden: tail.row0 + r - 1,
                }));
            }
            carries.push((tail.slot, tail.row0 + counted - 1));
        }
        self.times.key_rows += rows.len() as u64;
        if !rows.is_empty() {
            let converted: Vec<bool> =
                rows.iter().map(|row| layer.writes_converted(row.slot, row.pair - lag)).collect();
            let (launches, end) = self.lay_out_keys(layer, &rows, &converted, 0, self.half)?;
            // SAFETY: the first half of the pinned buffer is not written again
            // before the next step, which is enqueued after this one has run.
            unsafe { stream.copy_from_host(self.records.ptr(), self.host.as_ptr(), end as usize)? };
            self.enqueue_keys(layer, layers, stream, &launches, tap, tables, pages)?;
        }
        // Behind those rows the carries move on.
        for (slot, row) in carries {
            // SAFETY: a row of the model's hidden states of its last step,
            // which are kept until this step begins, and a slot of the
            // carries.
            unsafe {
                stream.copy(
                    self.carry.ptr().offset(u64::from(slot) * width),
                    tap.offset(u64::from(row) * width),
                    width as usize,
                )?;
            }
        }
        // The pages those rows fill are settled, the last cell of a page whose
        // sequence stops at its end with them.
        for (slot, cells) in at_page_end {
            layer.settle(slot, cells);
        }
        Ok(())
    }

    /// Steps 2 and 3, after the step's tables are written: the newest cell of
    /// every sequence of the step that does not draft (and whose chain did
    /// not write it in a step that only drafted), and the chains of
    /// [`Driver::draft_step`]. `tokens` are the tokens of the step's rows,
    /// `replay` its rows that are a replay. `only` is a step that only drafts:
    /// the model's step after it verifies the drafts and finds the newest
    /// cells of these chains written. Unless `only`, the step's rows are kept
    /// for the next step to follow.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        layer: &mut dyn MtpLayer,
        layers: &LayerKernels,
        stream: &Stream,
        segments: &[Segment],
        tokens: &[u32],
        replay: &[Range<u32>],
        tables: DevicePtr,
        pages: DevicePtr,
        only: bool,
    ) -> Result<(), ModelError> {
        self.flush_clock()?;
        self.clock.last_chain_ms = 0.0;
        let step = self.pending.take().unwrap_or(MtpStep {
            tokens: DevicePtr::NULL,
            chains: Vec::new(),
            chances: None,
        });
        let lag = layer.lag();
        let done = std::mem::take(&mut self.newest_done);
        let drafting = |i: usize| step.chains.iter().any(|c| c.segment as usize == i);
        let zero = self.setup.max_slots;
        // The newest cells of the sequences that do not draft.
        let quiet: Vec<KeyRow> = segments
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                s.pos0 as u32 >= lag
                    && !drafting(*i)
                    && !done.contains(&(s.slot, s.pos0 as u32))
                    && !replay.iter().any(|r| r.contains(&s.row0))
            })
            .map(|(_, s)| KeyRow {
                slot: s.slot,
                pair: s.pos0 as u32,
                token: tokens[s.row0 as usize],
                hidden: if s.pos0 == 0 { zero } else { s.slot },
            })
            .collect();
        self.times.key_rows += quiet.len() as u64;
        let mut end = self.half;
        let limit = self.host.len() as u64;
        let mut quiet_launches = Vec::new();
        if !quiet.is_empty() {
            let converted: Vec<bool> =
                quiet.iter().map(|row| layer.writes_converted(row.slot, row.pair - lag)).collect();
            let (launches, after) = self.lay_out_keys(layer, &quiet, &converted, end, limit)?;
            quiet_launches = launches;
            end = after;
        }
        let links = self.lay_out_chains(layer, segments, &step, &mut end, limit)?;
        if !only {
            self.keep(segments, tokens, replay);
        }
        for chain in &step.chains {
            let s = &segments[chain.segment as usize];
            if only {
                self.newest_done.push((s.slot, s.pos0 as u32));
            }
        }
        // SAFETY: the second half of the pinned buffer is not written again
        // before the next step's records, which are written after this step
        // has run; the device buffer has its size.
        unsafe {
            stream.copy_from_host(
                self.rec(self.half),
                self.host.as_ptr().add(self.half as usize),
                (end - self.half) as usize,
            )?;
        }
        if !quiet_launches.is_empty() {
            let carry = self.carry.ptr();
            self.enqueue_keys(layer, layers, stream, &quiet_launches, carry, tables, pages)?;
        }
        if let Some((links, carry_index, newest)) = links {
            self.chains(layer, layers, stream, &step, &links, carry_index, newest, tables, pages)?;
        }
        Ok(())
    }

    /// Remembers the rows of the model's step that is enqueued now: the next
    /// step follows them.
    fn keep(&mut self, segments: &[Segment], tokens: &[u32], replay: &[Range<u32>]) {
        let made = std::mem::take(&mut self.chain_links);
        self.tails.clear();
        self.tails.extend(segments.iter().map(|s| {
            Tail {
                slot: s.slot,
                pos0: s.pos0.max(0) as u32,
                row0: s.row0,
                tokens: tokens[s.row0 as usize..(s.row0 + s.n_rows) as usize].to_vec(),
                direct: s.direct,
                replay: replay.iter().any(|r| r.contains(&s.row0)),
                // The drafts of the chain that the window verifies.
                links: made
                    .iter()
                    .find(|(slot, _)| *slot == s.slot)
                    .map_or(0, |(_, made)| (*made).min(s.n_rows.saturating_sub(1))),
            }
        }));
    }

    /// The drafts the step that ran last made, written at their places in
    /// `out` (`u32::MAX` stays where a link was not made). The step has
    /// ended.
    pub fn picked(&self, out: &mut [u32]) {
        for &(link, row, place) in self.layout.iter().filter(|(link, ..)| *link < self.made) {
            let at =
                (link as usize * self.setup.max_slots as usize + row as usize) * SamplePick::BYTES;
            // SAFETY: the buffer holds a record for every row of every link,
            // and the device is not writing it any more.
            let bytes = unsafe {
                std::slice::from_raw_parts(self.picks_host.as_ptr().add(at), SamplePick::BYTES)
            };
            if let Some(token) = out.get_mut(place as usize) {
                *token = SamplePick::from_bytes(bytes).token;
            }
        }
    }

    /// Lays out the records of the step's chains, longest first, so that the
    /// chains of a link are the first rows of the link before. Returns the
    /// links, where the carry index of link 0 is and where its staged rows
    /// are; `None` without chains.
    #[allow(clippy::type_complexity)]
    fn lay_out_chains(
        &mut self,
        layer: &mut dyn MtpLayer,
        segments: &[Segment],
        step: &MtpStep,
        end: &mut u64,
        limit: u64,
    ) -> Result<Option<(Vec<Link>, u64, DevicePtr)>, ModelError> {
        self.drafted = 0;
        self.made = 0;
        self.layout.clear();
        if step.chains.is_empty() {
            return Ok(None);
        }
        let full = || invalid("the MTP block's step records do not fit their buffer");
        let lag = layer.lag();
        let mut first = Vec::with_capacity(step.chains.len());
        let mut total = 0u32;
        for chain in &step.chains {
            let segment = segments
                .get(chain.segment as usize)
                .ok_or_else(|| invalid(format!("the step has no segment {}", chain.segment)))?;
            if segment.direct || segment.n_rows as usize != chain.rows.len() + 1 {
                return Err(invalid(format!(
                    "segment {segment:?} does not verify {} drafts",
                    chain.rows.len()
                )));
            }
            first.push(chain.place);
            total = total.max(chain.place + chain.rows.len() as u32);
        }
        let mut order: Vec<usize> = (0..step.chains.len()).collect();
        order.sort_by_key(|&c| std::cmp::Reverse(step.chains[c].rows.len()));
        let n_links = order.first().map_or(0, |&c| step.chains[c].rows.len());
        let mut links = Vec::with_capacity(n_links);
        for j in 0..n_links {
            let chains: Vec<&MtpChain> = order
                .iter()
                .map(|&c| &step.chains[c])
                .take_while(|chain| chain.rows.len() > j)
                .collect();
            let n = chains.len() as u32;
            let n_next =
                order.iter().filter(|&&c| step.chains[c].rows.len() > j + 1).count() as u32;
            let link_segments: Vec<Segment> = chains
                .iter()
                .enumerate()
                .map(|(row, chain)| {
                    let s = segments[chain.segment as usize];
                    // A window at the start of a sequence has no cell in front
                    // of it with a lag (the conductor's probes are such).
                    Segment {
                        row0: row as u32,
                        n_rows: 1,
                        slot: s.slot,
                        pos0: ((s.pos0 as u32 + j as u32).saturating_sub(lag)) as i32,
                        direct: true,
                    }
                })
                .collect();
            // Link 0 reads the newest tokens; the others get theirs on the
            // device from the picks of the link before.
            let tokens: Vec<u32> =
                chains.iter().map(|c| if j == 0 { c.token } else { 0 }).collect();
            // Every row of a link has its logits.
            let wanted: Vec<u32> = (0..n).collect();
            let record = StepRecord::new(&link_segments, &tokens, None, &wanted)?;
            let bytes = record.as_bytes();
            let at_record = *end;
            let at_params = (at_record + bytes.len() as u64).next_multiple_of(16);
            let at_record_rows = at_params + u64::from(n) * SampleRow::BYTES as u64;
            if at_record_rows + 64 + u64::from(n) * 8 > limit {
                return Err(full());
            }
            let buf: &mut [u8] = &mut self.host;
            buf[at_record as usize..at_record as usize + bytes.len()].copy_from_slice(bytes);
            let off_tokens =
                u32::from_le_bytes(bytes[20..24].try_into().expect("the record's header"));
            let mut at = at_params;
            for chain in &chains {
                buf[at as usize..at as usize + SampleRow::BYTES]
                    .copy_from_slice(&chain.rows[j].to_bytes());
                at += SampleRow::BYTES as u64;
            }
            let rows_of = chains.iter().map(|c| segments[c.segment as usize].row0 + 1 + j as u32);
            let at_draft_at = words(buf, at_record_rows, rows_of);
            let places = order.iter().take(n as usize).map(|&c| first[c] + j as u32);
            let at_bytes = words(buf, at_draft_at, places);
            let written =
                layer.bytes(&link_segments, true, &mut buf[at_bytes as usize..limit as usize])?;
            *end = (at_bytes + written as u64).next_multiple_of(64);
            links.push(Link {
                n,
                n_next,
                segments: link_segments,
                at_record,
                at_tokens: at_record + u64::from(off_tokens),
                at_params,
                at_record_rows,
                at_draft_at,
                at_bytes,
                sample: SamplePlan::of(chains.iter().map(|c| &c.rows[j])),
                record,
            });
        }
        // The carries link 0 starts from, and the rows of its newest tokens,
        // in the order of its rows; the host stages them when the device does
        // not keep every row.
        let carry_index = *end;
        let size = self.setup.table.row_bytes();
        let staged_at = (carry_index + 4 * order.len() as u64).next_multiple_of(16);
        if staged_at + order.len() as u64 * size > limit {
            return Err(full());
        }
        let zero = self.setup.max_slots;
        let starts = order.iter().map(|&c| {
            let s = &segments[step.chains[c].segment as usize];
            if s.pos0 == 0 { zero } else { s.slot }
        });
        words(&mut self.host, carry_index, starts);
        let newest = match &self.setup.host {
            Some(host) => {
                let tokens: Vec<u32> = order.iter().map(|&c| step.chains[c].token).collect();
                let stage = &mut self.host
                    [staged_at as usize..(staged_at + tokens.len() as u64 * size) as usize];
                host.stage(&tokens, stage)?;
                *end = staged_at + tokens.len() as u64 * size;
                self.rec(staged_at)
            }
            None => {
                *end = staged_at;
                DevicePtr::NULL
            }
        };
        self.drafted = total;
        self.made = links.len() as u32;
        Ok(Some((links, carry_index, newest)))
    }

    /// Enqueues the chains laid out by [`Driver::lay_out_chains`].
    #[allow(clippy::too_many_arguments)]
    fn chains(
        &mut self,
        layer: &mut dyn MtpLayer,
        layers: &LayerKernels,
        stream: &Stream,
        step: &MtpStep,
        links: &[Link],
        carry_index: u64,
        newest: DevicePtr,
        tables: DevicePtr,
        pages: DevicePtr,
    ) -> Result<(), ModelError> {
        self.clock.chain[0].record(stream)?;
        self.clock.pending = true;
        let s = self.s;
        let rows = u64::from(self.setup.max_slots);
        let width = u64::from(self.setup.width);
        let mut order: Vec<usize> = (0..step.chains.len()).collect();
        order.sort_by_key(|&c| std::cmp::Reverse(step.chains[c].rows.len()));
        let slots: Vec<u32> = links[0].segments.iter().map(|s| s.slot).collect();
        // SAFETY: the carries hold a row for every slot and the zeros, the
        // chain's scratch a row of `width` values per chain.
        unsafe {
            self.sampler.gather_hidden(
                stream,
                self.at(self.s.h),
                self.carry.ptr(),
                self.rec(carry_index),
                links[0].n,
                self.setup.width,
            )?;
        }
        // The reach of each chain, in the order of the links' rows: the
        // product of the chances of its drafts, as far as the host has read
        // their picks.
        let mut reach = vec![1.0f64; order.len()];
        let lowest = step.chances.map(|c| f64::from(c.iter().copied().fold(1.0, f32::min)));
        let pick_bytes = SamplePick::BYTES;
        let mut read = 0;
        let out_of = |link: usize| s.out + link as u64 * rows * width * 4;
        for (j, link) in links.iter().enumerate() {
            let step_ptr = self.rec(link.at_record);
            let staged = if j == 0 { newest } else { DevicePtr::NULL };
            let (e, h) = (
                self.at(self.s.e),
                if j == 0 { self.at(self.s.h) } else { self.at(out_of(j - 1)) },
            );
            let picks =
                self.at(self.s.picks + (j * self.setup.max_slots as usize * pick_bytes) as u64);
            // SAFETY: the records hold what `lay_out_chains` wrote for this
            // link; the scratch holds `max_slots` rows of every buffer.
            unsafe {
                layers.embed_gather(stream, step_ptr, link.record.dims(), &self.rows(staged), e)?
            };
            let logits = layer.link(
                &Rows {
                    stream,
                    segments: &link.segments,
                    record: &link.record,
                    step: step_ptr,
                    replay: &[],
                    e,
                    h,
                    bytes: self.rec(link.at_bytes),
                    tables,
                    pages,
                },
                self.at(out_of(j)),
                self.at(self.s.logits),
            )?;
            if j == 0 {
                // Link 0 is the cell of a certain token; what follows is not.
                layer.save(stream, &slots)?;
            }
            let next_tokens =
                links.get(j + 1).map_or(DevicePtr::NULL, |next| self.rec(next.at_tokens));
            let batch = SampleBatch {
                logits: logits.ptr,
                stride: logits.stride,
                n_rows: link.n,
                capacity: self.setup.max_slots,
                n_local: logits.stride,
                tokens: self.setup.head_tokens,
                first: 0,
                // A draft is picked from the logits as they are: no
                // corrections, masks or probabilities (`SampleRow::for_draft`).
                inverse: DevicePtr::NULL,
                vocab: 0,
                params: self.rec(link.at_params),
                entries: DevicePtr::NULL,
                masks: DevicePtr::NULL,
                mask_words: 0,
                out: picks,
                logprobs: DevicePtr::NULL,
                scratch: self.at(self.s.sampler),
                exchange: self.at(self.s.exchange),
                devices: 1,
                rank: 0,
            };
            // SAFETY: the logits hold a row of the draft head per row of the
            // link; the scratch of the sampler was cleared when the driver
            // opened and only the sampler writes it; the copy reads what the
            // sampler wrote, into a buffer with room for every link.
            unsafe {
                self.sampler
                    .sample(stream, &batch, link.sample, |_, _| Ok::<(), ares_cuda::Error>(()))?;
                self.sampler.draft_put(
                    stream,
                    picks,
                    link.n,
                    next_tokens,
                    link.n_next,
                    step.tokens,
                    self.rec(link.at_record_rows),
                    self.at(self.s.drafts),
                    self.rec(link.at_draft_at),
                )?;
                let at = j * self.setup.max_slots as usize * pick_bytes;
                stream.copy_to_host(
                    self.picks_host.as_mut_ptr().add(at),
                    picks,
                    link.n as usize * pick_bytes,
                )?;
            }
            self.layout.extend(
                (0..link.n)
                    .map(|row| (j as u32, row, step.chains[order[row as usize]].place + j as u32)),
            );
            let (Some(next), Some(chances), Some(lowest)) =
                (links.get(j + 1), &step.chances, lowest)
            else {
                continue;
            };
            // The next link is made while a chain that has it reaches its
            // bar. Whether one can miss it is known without the picks when
            // even the least chance for every link not read yet leaves the
            // reach above the bar: then nothing is waited for.
            let need = |row: usize| {
                f64::from(step.chains[order[row]].needs.get(j + 1).copied().unwrap_or(0.0))
            };
            let least = lowest.powi((j + 1 - read) as i32);
            if (0..next.n as usize).any(|row| reach[row] * least >= need(row)) {
                continue;
            }
            self.picked.record(stream)?;
            self.picked.synchronize()?;
            for (k, link) in links.iter().enumerate().take(j + 1).skip(read) {
                for (row, reach) in reach.iter_mut().enumerate().take(link.n as usize) {
                    *reach *=
                        f64::from(draft_chance(chances, self.confidence(k as u32, row as u32)));
                }
            }
            read = j + 1;
            if (0..next.n as usize).all(|row| reach[row] < need(row)) {
                self.made = j as u32 + 1;
                break;
            }
        }
        // The rows of the links that were not made take the last draft of
        // their chain again: a token the device has the row of. The host is
        // told that they are no drafts.
        let last = self.made as usize - 1;
        for dead in &links[self.made as usize..] {
            // SAFETY: as above; the picks of link `last` hold a record for
            // each of the chains that are longer.
            unsafe {
                self.sampler.draft_put(
                    stream,
                    self.at(
                        self.s.picks + (last * self.setup.max_slots as usize * pick_bytes) as u64
                    ),
                    dead.n,
                    DevicePtr::NULL,
                    0,
                    step.tokens,
                    self.rec(dead.at_record_rows),
                    self.at(self.s.drafts),
                    self.rec(dead.at_draft_at),
                )?;
            }
        }
        // What the links changed of the sequences' state is put back.
        layer.restore(stream, &slots)?;
        self.chain_links.clear();
        for (row, &c) in order.iter().enumerate() {
            let slot = links[0].segments[row].slot;
            self.chain_links.push((slot, (step.chains[c].rows.len() as u32).min(self.made)));
        }
        self.clock.links = u64::from(self.made);
        self.clock.chain[1].record(stream)?;
        Ok(())
    }
}
