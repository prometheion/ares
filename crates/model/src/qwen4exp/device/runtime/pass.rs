//! A prompt read layer by layer, as the runtime drives it: the conductor's step of one
//! prompt wider than the parts' own scratch runs as a *pass* (`part/pass.rs` has the part's
//! side and what it keeps of a row).
//!
//! - **The loan.** The first part to begin the step lends on every device the VRAM the pass
//!   keeps its rows in, sized for the step's rows (`Flow::lend_pass`): a piece of each pool
//!   of expert slots, as the lent steps' scratch is. A pass that follows one that has ended
//!   takes the loan over when its rows fit in it; any other step gives it back before it
//!   begins.
//! - **The order.** On each part, layer by layer: every micro-batch through the layer up to
//!   its router and shared expert, each as a step of its own rows (its record and attention
//!   plan, made when the pass starts), then the layer's routed experts for all rows of the
//!   step (`Misses`, the rounds of a wide step, with the picks of every row), then the next
//!   layer. After the last layer, every micro-batch's output is written to its streams, and
//!   the part with the head computes the logits of the rows that are wanted.
//! - **The MTP block** follows the step's rows micro-batch by micro-batch after the trunk,
//!   as it would follow steps of those rows; the last micro-batch's residual is copied to
//!   the part's own scratch, where the next step's start follows it, so the loan is free.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use ares_cuda::{Segment, StepRecord};
use ares_runtime::{Logits, ModelError, StepContext};

use super::super::misses::{Misses, Progress, SpanPicks};
use super::super::part::{
    MicroPlan, Part, PassAt, PassLayout, PassSizes, StepEnv, chunk_rows, chunks_of, micro_batches,
};
use super::{Feed, Flow, Qwen4ExpDevice, Routing, invalid, model_error};

/// The prompt rows a pass reads: its sequence's slot, its first position and its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PassKey {
    pub slot: u32,
    pub pos0: i32,
    pub rows: u32,
}

/// The loan of a pass on every device.
pub(super) struct PassLoan {
    key: PassKey,
    /// Per device: the bytes of each piece lent (in the order of the store's pieces), and
    /// where the pass keeps its rows.
    held: Vec<Vec<u64>>,
    at: Vec<PassAt>,
    /// The pass has read its rows: the loan may serve the next pass or go back.
    done: bool,
}

/// The rows of a pass of `rows` rows on parts whose own scratch holds `own` rows: its
/// micro-batches and the rows of each of its chunks.
fn shape_of(rows: u32, own: u32) -> (Vec<(u32, u32)>, Vec<u32>) {
    let micro = micro_batches(rows, own);
    let chunks = chunks_of(&micro, chunk_rows(own))
        .into_iter()
        .map(|range| micro[range].iter().map(|(_, n)| n).sum())
        .collect();
    (micro, chunks)
}

/// Whether a pass of `rows` rows on parts whose own scratch holds `own` rows keeps them in
/// what each device can lend (`lendable`, per device the bytes of each piece), for parts that
/// keep `sizes`.
pub(super) fn pass_fits(sizes: &[PassSizes], lendable: &[Vec<u64>], own: u32, rows: u32) -> bool {
    let (_, chunks) = shape_of(rows, own);
    sizes.iter().zip(lendable).all(|(s, l)| PassLayout::new(s, &chunks, l).is_some())
}

/// The most rows up to `asked` (and above `own`) that `fits`, which holds for fewer rows
/// wherever it holds for more; `own` when no more rows fit.
pub(super) fn pass_span(own: u32, asked: u32, fits: impl Fn(u32) -> bool) -> u32 {
    if asked <= own {
        return own;
    }
    // `lo` fits (or is the own rows), `hi` does not.
    let (mut lo, mut hi) = (own, asked + 1);
    if fits(asked) {
        return asked;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

impl Flow {
    /// Before a step begins (by the first part to begin it): the loan of the pass the step
    /// is (`pass`), or of none. A pass that is not done keeps its loan; a new pass takes the
    /// loan of the one before when its rows fit in it, else that loan goes back and the
    /// devices lend anew; a step that is no pass gives a done pass's loan back.
    pub(super) fn lend_pass(
        &self,
        feed: &mut Feed,
        pass: Option<PassKey>,
    ) -> Result<(), ModelError> {
        let mut current = self.pass.lock().unwrap_or_else(|e| e.into_inner());
        if let (Some(loan), Some(key)) = (current.as_ref(), pass)
            && loan.key == key
            && !loan.done
        {
            return Ok(());
        }
        if let (Some(loan), Some(key)) = (current.as_mut(), pass)
            && loan.done
            && let Some(at) = self.place_pass(feed, key, &loan.held, false)?
        {
            loan.key = key;
            loan.at = at;
            loan.done = false;
            return Ok(());
        }
        if current.as_ref().is_some_and(|loan| loan.done || pass.is_some()) {
            for device in 0..self.pass_sizes.len() {
                feed.store.return_scratch(device).map_err(|e| invalid(e.to_string()))?;
            }
            *current = None;
        }
        let Some(key) = pass else { return Ok(()) };
        let lendable: Vec<Vec<u64>> =
            (0..self.pass_sizes.len()).map(|d| feed.store.lendable_pieces(d)).collect();
        let at = self.place_pass(feed, key, &lendable, true)?.ok_or_else(|| {
            invalid(format!("the expert slots cannot lend for a prompt of {} rows", key.rows))
        })?;
        let held = (0..self.pass_sizes.len())
            .map(|d| {
                let (_, chunks) = shape_of(key.rows, self.own_rows);
                PassLayout::new(&self.pass_sizes[d], &chunks, &lendable[d])
                    .map(|layout| layout.held)
                    .unwrap_or_default()
            })
            .collect();
        *current = Some(PassLoan { key, held, at, done: false });
        Ok(())
    }

    /// Lays the rows of the pass `key` out in `pieces` (per device, the bytes of each piece
    /// of its loan as the store orders them) and, with `lend`, lends them; returns where
    /// each part keeps the rows, or `None` when they do not fit.
    fn place_pass(
        &self,
        feed: &mut Feed,
        key: PassKey,
        pieces: &[Vec<u64>],
        lend: bool,
    ) -> Result<Option<Vec<PassAt>>, ModelError> {
        let (_, chunks) = shape_of(key.rows, self.own_rows);
        let mut layouts = Vec::with_capacity(self.pass_sizes.len());
        for (sizes, pieces) in self.pass_sizes.iter().zip(pieces) {
            match PassLayout::new(sizes, &chunks, pieces) {
                Some(layout) => layouts.push(layout),
                None => return Ok(None),
            }
        }
        let started = Instant::now();
        let mut placed = Vec::with_capacity(layouts.len());
        for (device, layout) in layouts.iter().enumerate() {
            if lend {
                while feed
                    .store
                    .lend_pieces(device, &layout.held)
                    .map_err(|e| invalid(e.to_string()))?
                    .is_none()
                {
                    self.check(started, "the expert slots lent VRAM for a pass")?;
                    feed.store.poll().map_err(|e| invalid(e.to_string()))?;
                    std::hint::spin_loop();
                }
                while !feed.store.scratch_ready(device).map_err(|e| invalid(e.to_string()))? {
                    self.check(started, "the expert slots lent VRAM for a pass")?;
                    feed.store.poll().map_err(|e| invalid(e.to_string()))?;
                    std::hint::spin_loop();
                }
            }
            // Piece `k` of the layout starts where the pieces before it end; its loan is
            // the last slots of its pool.
            let loans = feed.store.scratch_loans(device).unwrap_or_default();
            let starts = PassLayout::starts(&pieces[device]);
            let mut map = Vec::with_capacity(loans.len());
            let mut loans = loans.iter();
            for (k, bytes) in pieces[device].iter().enumerate() {
                let lent = if lend { layout.held.get(k).copied().unwrap_or(0) } else { *bytes };
                if lent == 0 {
                    continue;
                }
                let loan =
                    loans.next().ok_or_else(|| invalid("a piece of a pass without its loan"))?;
                let (base, _) = feed.store.device(device).pool(loan.pool);
                map.push((starts[k], base.offset(u64::from(loan.first) * loan.stride)));
            }
            let pieces = super::Pieces(map);
            placed.push(layout.place(|offset| pieces.address(offset)));
        }
        Ok(Some(placed))
    }

    /// Where part `index` keeps the rows of the pass of the step being begun.
    fn pass_at(&self, index: usize, key: PassKey) -> Result<PassAt, ModelError> {
        let current = self.pass.lock().unwrap_or_else(|e| e.into_inner());
        let loan = current
            .as_ref()
            .filter(|l| l.key == key)
            .ok_or_else(|| invalid("no loan for the pass"))?;
        loan.at
            .get(index)
            .cloned()
            .ok_or_else(|| invalid(format!("no rows of the pass on part {index}")))
    }

    /// The pass of the step has read its rows (by the last part to end the step).
    pub(super) fn pass_done(&self) {
        if let Some(loan) = self.pass.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            loan.done = true;
        }
    }

    /// Whether a pass holds a loan.
    pub(super) fn has_pass_loan(&self) -> bool {
        self.pass.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
}

/// A micro-batch of a pass as a step of its own.
struct Micro {
    segments: [Segment; 1],
    tokens: Vec<u32>,
    record: StepRecord,
    replay: Vec<std::ops::Range<u32>>,
    plan: MicroPlan,
    /// Its first row in the step, and its rows.
    first: u32,
    rows: u32,
}

/// A pass on one device part: its micro-batches and where it is.
pub(super) struct PassRun {
    key: PassKey,
    micro: Vec<Micro>,
    /// The step of all the rows, for the rounds of the experts, with its segments and tokens.
    span: StepRecord,
    segments: Vec<Segment>,
    tokens: Vec<u32>,
    /// The picks of a layer of all rows, as the host reads them.
    picks: Arc<SpanPicks>,
    /// The rows of the n-gram table of all rows, read when the pass starts (a step that
    /// runs while the pass is paused asks for its own).
    fetched: Option<Vec<u8>>,
    /// Where the pass goes on: the layer of the part and the micro-batch of it.
    layer: usize,
    next: usize,
}

impl PassRun {
    /// The pass reads the rows of `key`.
    pub(super) fn is(&self, key: PassKey) -> bool {
        self.key == key
    }
}

impl Qwen4ExpDevice {
    /// The pass the step is, if it is one: a step of one prompt wider than the parts' own
    /// scratch, on a model that reads such prompts layer by layer.
    pub(super) fn pass_of(&self, step: &StepContext<'_>) -> Result<Option<PassKey>, ModelError> {
        let record = &step.record;
        let rows = record.rows();
        if self.flow.pass_rows == 0 || rows <= self.flow.own_rows || record.drafts_only() {
            return Ok(None);
        }
        if record.segments() != 1 || record.drafter(0).is_some() {
            return Err(invalid(format!(
                "a step of {rows} rows in {} segments; a step wider than {} rows reads one prompt",
                record.segments(),
                self.flow.own_rows
            )));
        }
        let segment = record.segment(0);
        if !segment.direct || rows > self.flow.pass_rows {
            return Err(invalid(format!(
                "a step of {rows} rows; prompts are read layer by layer in steps of up to {}",
                self.flow.pass_rows
            )));
        }
        Ok(Some(PassKey { slot: segment.slot, pos0: segment.pos0, rows }))
    }

    /// Whether the step goes on with the pass this part paused (the same rows).
    pub(super) fn resumes(&self, key: Option<PassKey>) -> bool {
        matches!((&self.pass_run, key), (Some(run), Some(key)) if run.is(key))
    }

    /// The record says the paused pass is given up: what this part keeps of it goes, and
    /// its loan with the next step that is no pass.
    pub(super) fn give_up_pass(&mut self) {
        if self.pass_run.take().is_some() {
            self.part.pass_close();
        }
        self.flow.pass_done();
    }

    /// Runs the step as the pass `key` (the step's segments, tokens and wanted rows are in
    /// `self`), after everything a step does before its rows: from its start, or where it
    /// was paused. Pauses it between two micro-batches when the engine is asked something
    /// or the record's time for it has passed (after one micro-batch at least).
    pub(super) fn enqueue_pass(
        &mut self,
        step: &StepContext<'_>,
        key: PassKey,
        token: u32,
    ) -> Result<Option<Logits>, ModelError> {
        let begun = Instant::now();
        let asked = step.asks.load(Ordering::Acquire);
        let budget = step.record.pause_after();
        let stream = step.stream;
        let mut run = match self.pass_run.take().filter(|run| run.is(key)) {
            Some(run) => {
                self.part.pass_resume(stream)?;
                run
            }
            None => {
                let at = self.flow.pass_at(self.index, key)?;
                self.open_pass(step, key, at)?
            }
        };
        let n_layers = self.part.n_layers();
        let ngram_at = self.part.ngram_layer();
        let row_bytes = run.fetched.as_ref().map(|f| f.len() / key.rows as usize);
        let mut ran = 0usize;
        while run.layer < n_layers {
            let i = run.layer;
            while run.next < run.micro.len() {
                let pause = (budget > 0 && begun.elapsed().as_millis() >= u128::from(budget))
                    || step.asks.load(Ordering::Acquire) != asked;
                if ran > 0 && pause {
                    self.part.pass_pause();
                    self.pass_run = Some(run);
                    self.paused = true;
                    return Ok(None);
                }
                let m = run.next;
                let micro = &mut run.micro[m];
                let ngram = match (&run.fetched, row_bytes) {
                    (Some(rows), Some(each)) if ngram_at == Some(i) => {
                        let start = micro.first as usize * each;
                        Some(&rows[start..start + micro.rows as usize * each])
                    }
                    _ => None,
                };
                let env = StepEnv {
                    stream,
                    record: &micro.record,
                    segments: &micro.segments,
                    token,
                    tokens: &micro.tokens,
                    ngram,
                    inbound: None,
                    xbase: 0,
                    pools: step.pages,
                    replay: &micro.replay,
                };
                self.part.pass_layer(&env, i, m, &mut micro.plan)?;
                self.flow.pump()?;
                run.next += 1;
                ran += 1;
            }
            // Every micro-batch has routed: the picks of all rows go to the host, and the
            // layer's experts run for all of them.
            self.part.pass_routed(stream, i)?;
            {
                let (host, copied) = run.picks.target();
                // SAFETY: the buffer holds the picks of the pass's rows and the host reads it
                // only after `copied`, until the next layer's copies are enqueued (after this
                // layer is done).
                unsafe { self.part.pass_record_to_host(i, host, copied)? };
            }
            self.flow.enqueued[self.index].store(i + 1, Ordering::Release);
            let env = StepEnv {
                stream,
                record: &run.span,
                segments: &run.segments,
                token,
                tokens: &run.tokens,
                ngram: None,
                inbound: None,
                xbase: 0,
                pools: step.pages,
                replay: &self.replay,
            };
            pass_experts(
                &mut self.part,
                &self.flow,
                self.index,
                &env,
                i,
                &self.routing,
                &run.picks,
            )?;
            self.flow.pump()?;
            run.layer += 1;
            run.next = 0;
        }
        // The last layer's output into every micro-batch's streams, and the logits of the
        // rows that are wanted.
        let mut n_out = 0;
        for (m, micro) in run.micro.iter_mut().enumerate() {
            let env = StepEnv {
                stream,
                record: &micro.record,
                segments: &micro.segments,
                token,
                tokens: &micro.tokens,
                ngram: None,
                inbound: None,
                xbase: 0,
                pools: step.pages,
                replay: &micro.replay,
            };
            n_out += self.part.pass_end(&env, m, &mut micro.plan)?;
        }
        // The MTP block follows the rows micro-batch by micro-batch, as it follows steps;
        // the next step follows the last one, whose residual is copied to the own scratch.
        let last = run.micro.len() - 1;
        if let Some(block) = &mut self.mtp {
            let direct = |_: u32, _: u32| None;
            for (m, micro) in run.micro.iter().enumerate() {
                if m > 0 {
                    block.follow(stream, &direct, self.part.pass_residual(m - 1)?)?;
                }
                block.step(stream, &micro.segments, &micro.tokens, &micro.replay, None, false)?;
            }
        }
        if self.mtp.is_some() {
            let residual = self.part.pass_residual(last)?;
            self.residual =
                Some(self.part.keep_residual(stream, residual, run.micro[last].rows)?);
        }
        self.part.pass_close();
        self.flow.pass_completed.store(true, Ordering::Release);
        Ok((n_out > 0 && self.part.has_head()).then(|| Logits {
            ptr: self.part.head_logits(),
            stride: self.n_vocab,
            first_token: 0,
        }))
    }

    /// Opens the pass `key` on the part, its rows kept at `at`: the micro-batches as steps of
    /// their own, with their attention plans.
    fn open_pass(
        &mut self,
        step: &StepContext<'_>,
        key: PassKey,
        at: PassAt,
    ) -> Result<PassRun, ModelError> {
        let stream = step.stream;
        let micro = micro_batches(key.rows, self.flow.own_rows);
        self.part.pass_open(stream, &self.segments, at, &micro)?;
        let replay = !self.replay.is_empty();
        let segment = self.segments[0];
        let mut steps = Vec::with_capacity(micro.len());
        for &(first, rows) in &micro {
            let segments =
                [Segment { row0: 0, n_rows: rows, pos0: segment.pos0 + first as i32, ..segment }];
            let tokens = self.tokens[first as usize..(first + rows) as usize].to_vec();
            let wanted: Vec<u32> = self
                .out_rows
                .iter()
                .filter(|r| (first..first + rows).contains(r))
                .map(|r| r - first)
                .collect();
            let mut record = StepRecord::new(&segments, &tokens, None, &wanted)?;
            if replay {
                record.mark_replay(0)?;
            }
            let replay = if replay { vec![0..rows] } else { Vec::new() };
            let env = StepEnv {
                stream,
                record: &record,
                segments: &segments,
                token: 0,
                tokens: &tokens,
                ngram: None,
                inbound: None,
                xbase: 0,
                pools: step.pages,
                replay: &replay,
            };
            let plan = self.part.pass_plan(&env)?;
            steps.push(Micro { segments, tokens, record, replay, plan, first, rows });
        }
        let wanted_elsewhere = self.out_rows.iter().any(|r| *r < micro[micro.len() - 1].0);
        if wanted_elsewhere {
            return Err(invalid("a prompt read layer by layer wants logits only of its last rows"));
        }
        let mut span = StepRecord::new(&self.segments, &self.tokens, None, &self.out_rows)?;
        if replay {
            span.mark_replay(0)?;
        }
        let top_k = self.flow.top_k;
        let picks = match self.picks.take() {
            Some(picks) if picks.holds(key.rows, top_k) => picks,
            _ => {
                Arc::new(SpanPicks::new(self.part.device(), key.rows, top_k).map_err(model_error)?)
            }
        };
        self.picks = Some(picks.clone());
        // The rows of the n-gram table of every row, before a step between the pieces asks
        // for its own.
        let fetched = match self.part.ngram_layer() {
            Some(_) => Some(self.flow.await_rows()?),
            None => None,
        };
        Ok(PassRun {
            key,
            micro: steps,
            span,
            segments: self.segments.clone(),
            tokens: self.tokens.clone(),
            picks,
            fetched,
            layer: 0,
            next: 0,
        })
    }
}

/// The routed experts of layer `layer` of `part` (the `index`-th) for all rows of a pass,
/// whose picks reach the host in `picks`: [`Misses`] over the rows, serving the store until
/// they are computed.
fn pass_experts(
    part: &mut Part,
    flow: &Flow,
    index: usize,
    env: &StepEnv<'_>,
    layer: usize,
    routing: &Routing,
    picks: &Arc<SpanPicks>,
) -> Result<(), ModelError> {
    let owned = |expert: u32| routing.owned.as_ref().is_none_or(|o| o[layer][expert as usize]);
    let (mut started, mut seen) = (Instant::now(), flow.progress());
    let mut misses = {
        let feed = flow.lock_feed();
        Misses::begin_pass(part, env, &feed, index, layer, picks.clone(), &owned)
            .map_err(model_error)?
    };
    loop {
        let progress = {
            let mut feed = flow.lock_feed();
            misses.advance(part, env, &mut feed, true, &owned).map_err(model_error)?
        };
        if progress == Progress::Done {
            return Ok(());
        }
        let progress = flow.progress();
        if progress != seen {
            (started, seen) = (Instant::now(), progress);
        }
        flow.check(started, "the experts of a layer of a pass were brought")?;
        flow.pump()?;
        std::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_spans_the_most_rows_that_fit() {
        assert_eq!(pass_span(512, 32768, |rows| rows <= 40000), 32768);
        assert_eq!(pass_span(512, 32768, |rows| rows <= 20000), 20000);
        assert_eq!(pass_span(512, 32768, |_| false), 512);
        assert_eq!(pass_span(512, 256, |_| true), 512);
    }
}
