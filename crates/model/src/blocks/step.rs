//! What a part reads from the runtime's step record before it enqueues the
//! step, and how the drafters of a part hand their drafts out, the same way
//! for every model.
//!
//! - [`StepInputs`] decodes a record once: the rows' tokens, the segments,
//!   the rows that are a replay, the rotary positions where they are not the
//!   rows' cells, and the windows whose drafts a drafter of the model makes
//!   (the MTP block's chains, the DFlash drafter's passes), with one set of
//!   checks: a drafter the model does not have, a token past the vocabulary.
//! - [`table_updates`] and [`accepted`] read the page tables' updates and the
//!   outcome of the windows of the step before.
//! - [`Drafters`] is what the runtime reads of a step's drafts on the part
//!   that holds the drafters: one list, its length, the drafts' probabilities
//!   and the drafters' device time.

use std::ops::Range;

use ares_cuda::{CacheFormat, DevicePtr, Segment};
use ares_runtime::{Drafter as ModelDrafter, DrafterKind, ModelDrafts, ModelError, RecordView};

use super::kv;
use super::mtp::{Driver, MtpChain};
use crate::dflash::{Drafter, PassSeq};

/// The ids the record names a model's drafters by: their places in the
/// model's list of drafters plus one, the MTP block first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DrafterIds {
    pub mtp: Option<u32>,
    pub dflash: Option<u32>,
}

impl DrafterIds {
    /// The ids of a model that drafts with an MTP block (`mtp`) and with a
    /// DFlash drafter (`dflash`).
    pub(crate) fn new(mtp: bool, dflash: bool) -> Self {
        Self { mtp: mtp.then_some(1), dflash: dflash.then_some(1 + u32::from(mtp)) }
    }
}

/// The drafters of a model whose windows have `window_rows` rows, in the
/// order of their ids: its MTP block when it drafts with one (`mtp`), its
/// DFlash drafter when it has one that makes `dflash` drafts a pass. Each
/// makes at most a window's drafts, the newest token aside.
pub(crate) fn drafters(mtp: bool, dflash: Option<u32>, window_rows: u32) -> Vec<ModelDrafter> {
    let most = window_rows.saturating_sub(1);
    let mtp = mtp.then_some(ModelDrafter { kind: DrafterKind::Mtp, drafts: most });
    let dflash =
        dflash.map(|drafts| ModelDrafter { kind: DrafterKind::Dflash, drafts: drafts.min(most) });
    mtp.into_iter().chain(dflash).collect()
}

/// What a step record says a part's step is, read once and kept to reuse
/// the memory.
#[derive(Debug, Default)]
pub(crate) struct StepInputs {
    /// The rows' tokens: 0 for the rows that take an MTP chain's drafts, which
    /// have no token yet.
    pub tokens: Vec<u32>,
    pub segments: Vec<Segment>,
    /// The rows that are a replay.
    pub replay: Vec<Range<u32>>,
    /// The rows' rotary positions where they are not their cells (a sequence
    /// with images); empty when every row is at its cell.
    pub rotary: Vec<[i32; 4]>,
    /// The windows whose drafts the model's MTP block makes, with their newest
    /// token and the settings of the rows that verify the drafts.
    pub chains: Vec<MtpChain>,
    /// The windows whose drafts the model's DFlash drafter makes.
    pub passes: Vec<PassSeq>,
}

impl StepInputs {
    /// Reads `record` for a model whose drafters the record names by `ids`
    /// and whose vocabulary has `n_vocab` tokens. Every window's drafts have a
    /// place in the step's one list of drafts, in the order of the windows.
    pub(crate) fn read(
        &mut self,
        record: &RecordView<'_>,
        ids: DrafterIds,
        n_vocab: u32,
    ) -> Result<(), ModelError> {
        let rows = record.rows();
        self.segments.clear();
        self.segments.extend((0..record.segments()).map(|i| record.segment(i)));
        self.replay.clear();
        for (i, segment) in self.segments.iter().enumerate() {
            if record.replay(i as u32) {
                self.replay.push(segment.row0..segment.row0 + segment.n_rows);
            }
        }
        // The windows whose drafts a drafter of the model makes, each at its
        // place in the step's list of drafts: the MTP block's chains, with
        // their newest token and the settings of the rows that verify the
        // drafts (all rows of a window are wanted, so the sampler rows of a
        // window are consecutive; `wanted` counts the wanted rows before it),
        // and the DFlash drafter's passes.
        self.chains.clear();
        self.passes.clear();
        let (mut bar, mut place) = (0, 0);
        for (i, segment) in self.segments.iter().enumerate() {
            let Some(drafter) = record.drafter(i as u32) else { continue };
            if ids == DrafterIds::default() {
                return Err(ModelError::Invalid("the model drafts nothing itself".into()));
            }
            if Some(drafter) != ids.mtp && Some(drafter) != ids.dflash {
                return Err(ModelError::Invalid(format!("the model has no drafter {drafter}")));
            }
            if segment.n_rows < 2 {
                continue;
            }
            let links = segment.n_rows - 1;
            // The bars of the window's links, when the record has them.
            let needs = if record.draft_bars() >= bar + links {
                (bar..bar + links).map(|k| record.draft_need(k)).collect()
            } else {
                Vec::new()
            };
            bar += links;
            if Some(drafter) == ids.mtp {
                let wanted: u32 = (0..record.runs())
                    .map(|r| record.run(r))
                    .map(|(first, count)| segment.row0.clamp(first, first + count) - first)
                    .sum();
                self.chains.push(MtpChain {
                    segment: i as u32,
                    token: record.row(segment.row0).token,
                    rows: (0..links).map(|j| record.sampler(wanted + j).for_draft()).collect(),
                    needs,
                    place,
                });
            } else {
                self.passes.push(PassSeq {
                    slot: segment.slot,
                    position: segment.pos0 as u32,
                    newest: record.row(segment.row0).token,
                    record_row: segment.row0 + 1,
                    wanted: links,
                    place,
                });
            }
            place += links;
        }
        // The rows that take an MTP chain's drafts have no token yet.
        self.tokens.clear();
        self.tokens.extend((0..rows).map(|i| record.row(i).token));
        for chain in &self.chains {
            let s = &self.segments[chain.segment as usize];
            self.tokens[(s.row0 + 1) as usize..(s.row0 + s.n_rows) as usize].fill(0);
        }
        if let Some(token) = self.tokens.iter().find(|&&token| token >= n_vocab) {
            return Err(ModelError::Invalid(format!("token {token} of a vocabulary of {n_vocab}")));
        }
        self.rotary.clear();
        let at_cells = (0..rows).all(|i| {
            let p = record.row(i).position;
            record.rotary(i) == [p, p, p, 0]
        });
        if !at_cells {
            self.rotary.extend((0..rows).map(|i| record.rotary(i).map(|p| p as i32)));
        }
        Ok(())
    }
}

/// An error when a segment of a step lies outside `max_slots` sequence slots
/// of `cells` cells each, or a draft window has more rows than a K/V group of
/// the formats `kv` lets a window have ([`kv::Group::window_limit`]).
pub(crate) fn check_segments(
    segments: &[Segment],
    max_slots: u32,
    cells: u64,
    kv: (CacheFormat, CacheFormat),
) -> Result<(), ModelError> {
    for s in segments {
        let end = i64::from(s.pos0) + i64::from(s.n_rows);
        if s.slot >= max_slots || s.pos0 < 0 || end > cells as i64 {
            return Err(ModelError::Invalid(format!(
                "segment {s:?} is outside {max_slots} slots of {cells} cells"
            )));
        }
    }
    if let Some(limit) = kv::Group::window_limit(kv.0, kv.1)
        && let Some(s) = segments.iter().find(|s| !s.direct && s.n_rows > limit)
    {
        return Err(ModelError::Invalid(format!(
            "a draft window of {} rows; with K/V as {:?} and {:?} a window has at most {limit} \
             rows (a tile that leaves the exact cells inside it must have been full before it)",
            s.n_rows, kv.0, kv.1
        )));
    }
    Ok(())
}

/// The page-table updates of `record`: each one's slot, its first entry, and
/// the range of its entries in the record (`record.entry(k)`).
pub(crate) fn table_updates<'r>(
    record: &'r RecordView<'_>,
) -> impl Iterator<Item = (u32, u32, Range<u32>)> + 'r {
    let mut taken = 0;
    (0..record.updates()).map(move |i| {
        let update = record.update(i);
        let entries = taken..taken + update.count;
        taken += update.count;
        (update.slot, update.first, entries)
    })
}

/// How many rows of the window of `slot` whose first row was `row0` in the
/// step before the record's commits accepted; `None` when it says nothing of
/// that window.
pub(crate) fn accepted(record: &RecordView<'_>, slot: u32, row0: u32) -> Option<u32> {
    (0..record.commits())
        .map(|i| record.commit(i))
        .find(|c| c.slot == slot && c.row0 == row0)
        .map(|c| c.accepted)
}

/// The drafters of a part, as far as it holds them: the model's MTP block's
/// driver and its DFlash drafter. They write a step's drafts into one list,
/// each at the places of its windows: the MTP block's list when the part has
/// the block, else the DFlash drafter's own.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Drafters<'a> {
    pub mtp: Option<&'a Driver>,
    pub dflash: Option<&'a Drafter>,
}

impl Drafters<'_> {
    /// The step's list of drafts; null without a drafter.
    pub(crate) fn list(&self) -> DevicePtr {
        match (self.mtp, self.dflash) {
            (Some(mtp), _) => mtp.list(),
            (None, Some(dflash)) => dflash.own_list(),
            (None, None) => DevicePtr::NULL,
        }
    }

    /// Where the step enqueued last left its drafts and how many; `None` when
    /// it drafted nothing.
    pub(crate) fn drafted(&self) -> Option<ModelDrafts> {
        let mtp = self.mtp.map_or(0, Driver::drafted);
        let dflash = self.dflash.map_or(0, Drafter::drafted);
        let tokens = mtp.max(dflash);
        (tokens > 0).then(|| ModelDrafts { ptr: self.list(), tokens })
    }

    /// The probability the drafter gave each draft of the step that ran last,
    /// in the order of [`Drafters::drafted`], appended to `out`; NaN for a
    /// link that was not made. The step has ended.
    pub(crate) fn probs(&self, out: &mut Vec<f32>) {
        let Some(drafted) = self.drafted() else { return };
        let from = out.len();
        out.resize(from + drafted.tokens as usize, f32::NAN);
        if let Some(mtp) = self.mtp {
            mtp.probs(&mut out[from..]);
        }
        if let Some(dflash) = self.dflash.filter(|d| d.drafted() > 0) {
            for s in dflash.last_pass() {
                for j in s.place..s.place + s.wanted {
                    out[from + j as usize] = dflash.prob(j);
                }
            }
        }
    }
}

/// Device milliseconds of the drafters of the step that ran last: the MTP
/// block's chains and the DFlash drafter's pass. The step must have ended. A
/// time that cannot be read counts 0: the drafter's price stays what it was
/// measured as before, which is not worth failing the step for.
pub(crate) fn draft_ms(mtp: Option<&mut Driver>, dflash: Option<&mut Drafter>) -> f64 {
    let chain = mtp.and_then(|m| m.chain_ms().ok()).unwrap_or(0.0);
    let pass = dflash.and_then(|d| d.pass_ms().ok()).unwrap_or(0.0);
    chain + pass
}
