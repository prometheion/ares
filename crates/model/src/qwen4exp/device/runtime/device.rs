//! One device's part as the runtime drives it: the [`DeviceModel`] that turns
//! the runtime's step into the part's, its layers enqueued through the
//! [`Flow`] the device threads share.

use std::sync::Arc;

use ares_cuda::{DevicePtr, Event, MAX_BATCH, StepRecord, Stream};
use ares_runtime::{
    DeviceModel, DeviceTransfer, Logits, ModelDrafts, ModelError, Pick, StepContext, TowerJob,
    TransferContext,
};

use super::super::flow::{Flow, Routing, Turn, wide_layer};
use super::super::invalid;
use super::super::misses::SpanPicks;
use super::super::mtp::MtpBlock;
use super::super::part::{PAGE_ENTRY, Part, StepEnv};
use super::super::resume::Resume;
use super::Qwen4Exp;
use super::pass::PassRun;
use crate::blocks::mtp::MtpStep;
use crate::blocks::rows::{self, HostTable};
use crate::blocks::step::{self, DrafterIds, Drafters, StepInputs};
use crate::qwen3vl::TowerAt;

/// A part behind the runtime's interface.
pub(super) struct Qwen4ExpDevice {
    /// The image tower, on the part that runs it.
    pub(super) tower: Option<TowerAt>,
    pub(super) index: usize,
    pub(super) devices: usize,
    /// Whether the devices hold a range of layers each (and hand the residual
    /// over) rather than share every layer.
    pub(super) by_layers: bool,
    pub(super) n_vocab: u32,
    /// Values of one row of the residual streams.
    pub(super) wide: u32,
    /// Exchanges between a mixer and its worker per step.
    pub(super) per_step: u32,
    /// Whether this part keeps the sequences the n-gram table is read for.
    pub(super) drives_ngram: bool,
    pub(super) routing: Routing,
    pub(super) part: Part,
    pub(super) flow: Arc<Flow>,
    /// The step being enqueued as the record says it, kept to reuse the
    /// memory, and the ids the record names the model's drafters by.
    pub(super) inputs: StepInputs,
    pub(super) ids: DrafterIds,
    pub(super) out_rows: Vec<u32>,
    pub(super) accepted: Vec<u32>,
    /// The draft windows of the step before, as (slot, first row), which the
    /// record of this step says the outcome of.
    pub(super) windows: Vec<(u32, u32)>,
    /// The step that ended last: its segments as (slot, direct, rows, first
    /// row), and its wanted rows.
    pub(super) shape: Vec<(u32, bool, u32, u32)>,
    /// The model's MTP block, on the part that holds the output head.
    pub(super) mtp: Option<MtpBlock>,
    /// The embedding rows, for the DFlash drafter's newest tokens, on the part
    /// that holds it.
    pub(super) dflash_embedding: Option<Arc<HostTable>>,
    /// Where the part left the residual of the rows of its last step.
    pub(super) residual: Option<DevicePtr>,
    /// The host's copy of the picks of a pass's layer, kept for the next pass.
    pub(super) picks: Option<Arc<SpanPicks>>,
    /// The pass this part paused, and whether the step that ended last was paused.
    pub(super) pass_run: Option<PassRun>,
    pub(super) paused: bool,
    /// The step being run only drafts.
    pub(super) only_drafts: bool,
    /// The checkpoints of the prefix cache on this device.
    pub(super) resume: Resume,
    /// The first page whose K and V are in host memory, when some are, and
    /// which pages the tables name and the pool has not freed: the far pools
    /// keep the pieces of those locked (every device keeps the same books,
    /// any of them may begin a step for the store).
    pub(super) near: Option<u32>,
    pub(super) live: Vec<bool>,
}

impl Qwen4ExpDevice {
    /// The part of device `index` of `model`, opened: its routing, the flow the
    /// device threads share, its checkpoints, and what of the image tower and
    /// the drafters it holds.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        model: &Qwen4Exp,
        index: usize,
        part: Part,
        flow: Arc<Flow>,
        routing: Routing,
        resume: Resume,
        tower: Option<TowerAt>,
        mtp: Option<MtpBlock>,
        dflash_embedding: Option<Arc<HostTable>>,
    ) -> Self {
        let (c, o) = (&model.config, &model.opts.part);
        let near = o.near_pages();
        Self {
            tower,
            index,
            devices: model.placement.n_devices(),
            by_layers: model.placement.tensor.is_none(),
            n_vocab: c.n_vocab as u32,
            wide: c.wide() as u32,
            // A pass numbers the exchanges of its pieces from the step's base: a block of
            // eight per layer and micro-batch, and one layer more for its end.
            per_step: 8
                * (c.n_layer as u32 + 1)
                * (flow.pass_rows.div_ceil(o.own_rows().max(1)) + 2),
            mtp,
            inputs: StepInputs::default(),
            ids: DrafterIds::new(model.mtp.is_some(), model.dflash.is_some()),
            dflash_embedding,
            residual: None,
            picks: None,
            pass_run: None,
            paused: false,
            only_drafts: false,
            drives_ngram: part.ngram_layer().is_some(),
            routing,
            part,
            flow,
            out_rows: Vec::new(),
            accepted: Vec::new(),
            windows: Vec::new(),
            shape: Vec::new(),
            resume,
            near: (near < o.pages()).then_some(near),
            live: Vec::new(),
        }
    }

    /// The drafters of this part: the step's one list is the MTP block's when
    /// the device has it (the DFlash drafter writes its drafts into it), else
    /// the DFlash drafter's.
    fn drafters(&self) -> Drafters<'_> {
        Drafters { mtp: self.mtp.as_ref().map(MtpBlock::driver), dflash: self.part.dflash() }
    }

    /// Page `page` is in use until the pool frees it.
    fn mark_live(&mut self, page: u32) {
        if self.near.is_none() {
            return;
        }
        if self.live.len() <= page as usize {
            self.live.resize(page as usize + 1, false);
        }
        self.live[page as usize] = true;
    }

    /// The pages the tables name after the step's updates, less those the
    /// pool freed: the far pools keep the pieces of the far ones locked.
    fn update_live(&mut self, record: &ares_runtime::RecordView<'_>) {
        if self.near.is_none() {
            return;
        }
        for i in 0..record.freed() {
            if let Some(live) = self.live.get_mut(record.freed_page(i) as usize) {
                *live = false;
            }
        }
        for k in 0..record.entries() {
            self.mark_live(record.entry(k) & !PAGE_ENTRY);
        }
    }
}

impl std::fmt::Debug for Qwen4ExpDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4ExpDevice").field("index", &self.index).finish_non_exhaustive()
    }
}

impl Qwen4ExpDevice {
    /// The stream is idle: nothing of a failed step waits for the host any
    /// more.
    fn settle(&self, stream: &Stream) {
        self.flow.drain(|| stream.is_idle().unwrap_or(true));
    }

    fn enqueue_step(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        let record = &step.record;
        let stream = step.stream;
        let rows = record.rows();
        if rows == 0 {
            return Err(invalid("a step without rows"));
        }
        self.inputs.read(record, self.ids, self.n_vocab)?;
        // The rows' rotary positions where they are not their cells (a
        // sequence with images): the step's record and the MTP block's take
        // them.
        if let Some(block) = &mut self.mtp {
            block.set_rotary(self.inputs.rotary.clone());
        }
        self.out_rows.clear();
        for run in 0..record.runs() {
            let (first, count) = record.run(run);
            self.out_rows.extend(first..first + count);
        }
        // The step begins for the store: the MTP block routes in it before
        // any layer of the model. The number of the step marks its route
        // records and its exchanges: a kernel waits for a flag that is behind
        // the number, and the flags start at 0, so no step may be numbered 0
        // (the conductor's first warm-up record is).
        let number = u64::from(record.step()) + 1;
        // The windows whose rows past the first verify drafts; a step that
        // only drafts runs none of the model's rows and is no step of the
        // store.
        let windows: Vec<(u32, u32)> =
            self.inputs.segments.iter().filter(|s| !s.direct).map(|s| (s.row0, s.n_rows)).collect();
        let worth: Option<Vec<f32>> = record
            .row_worth(0)
            .map(|_| (0..rows).map(|row| record.row_worth(row).unwrap_or(0.0)).collect());
        self.only_drafts = record.drafts_only();
        let windows = (!self.only_drafts).then_some((windows.as_slice(), worth.as_deref()));
        // A prompt wider than the parts' own scratch is read layer by layer; a record may
        // give up the pass a step before paused, or go on with it.
        self.paused = false;
        if record.gives_up_paused() {
            self.give_up_pass();
        }
        let pass = self.pass_of(step)?;
        self.update_live(record);
        let far_live =
            self.near.map_or(&[][..], |near| self.live.get(near as usize..).unwrap_or(&[]));
        self.flow.begin(number, rows, windows, far_live, self.index, stream, pass.as_ref())?;
        // What this part keeps of a pass that has ended, or that another pass of the same
        // rows has taken the loan of, goes; a paused pass goes on.
        self.forget_ended_pass();
        let resumes = self.resumes(pass.as_ref());
        self.part.set_lent(self.flow.lent(self.index));

        // The draft windows of the step before: how many rows count.
        if self.part.has_deferred() {
            self.accepted.clear();
            for &(slot, row0) in &self.windows {
                let accepted = step::accepted(record, slot, row0)
                    .ok_or_else(|| invalid(format!("no outcome for the window of slot {slot}")))?;
                self.accepted.push(accepted);
            }
            self.part.enqueue_commit(stream, &self.accepted)?;
            if self.drives_ngram {
                self.flow.lock_ngram().settle(&self.accepted)?;
            }
        }

        // The MTP block follows the rows of the last step that count, in
        // the tables of that step.
        if let (Some(block), Some(residual)) = (&mut self.mtp, self.residual.take()) {
            let outcome = |slot: u32, row0: u32| step::accepted(record, slot, row0);
            block.follow(stream, &outcome, residual)?;
        }

        // The pages the pool freed give back what the K/V groups keep for them.
        let freed: Vec<u32> = (0..record.freed()).map(|i| record.freed_page(i)).collect();
        self.part.release_pages(freed.iter().copied());
        if let Some(block) = &mut self.mtp {
            block.release_pages(&freed);
        }

        // The page tables as the conductor made them.
        for (slot, first, entries) in step::table_updates(record) {
            if self.part.has_attention() {
                let entries = entries.clone().map(|k| record.entry(k));
                self.part.update_table(stream, slot, first, entries)?;
            }
            if let Some(block) = &mut self.mtp {
                block.update_table(stream, slot, first, entries.map(|k| record.entry(k)))?;
            }
        }

        // The copies of the prefix cache, behind the commit of the step before
        // and the MTP block's following of it: the saves, the page copies,
        // the restores.
        self.resume.copies(
            stream,
            step.pages,
            record,
            &mut self.part,
            self.mtp.as_mut(),
            &self.flow,
        )?;
        // The places of the saves inside a pass are locked here, where a step's saves lock
        // theirs: not between the pass's micro-batches (`Resume::hold`).
        for i in 0..record.inner_saves() {
            self.resume.hold(record.inner_save(i).place)?;
        }

        // A step that only drafts: the chains, and nothing of the model. It
        // leaves no window for the next record to commit.
        if record.drafts_only() {
            let tokens = step.record_on_device.offset(record.tokens_offset());
            if let Some(block) = &mut self.mtp {
                let chains = std::mem::take(&mut self.inputs.chains);
                let step = MtpStep { tokens, chains, chances: record.draft_chances() };
                block.step(
                    stream,
                    &self.inputs.segments,
                    &self.inputs.tokens,
                    &self.inputs.replay,
                    Some(step),
                    true,
                )?;
            }
            // The DFlash drafter's passes, into the step's one list: the MTP
            // block's when it is on this device too.
            if !self.inputs.passes.is_empty() && self.part.dflash().is_some() {
                let table = self
                    .dflash_embedding
                    .as_ref()
                    .ok_or_else(|| invalid("the DFlash drafter is on another part"))?;
                let n = self.part.n_embd() as usize;
                let embed = |token: u32| {
                    let mut out = vec![0f32; n];
                    // A token past the table is zeros, as the model's rows are.
                    let _ = table.decode(token, &mut out);
                    out
                };
                let list = self.mtp.as_ref().map(MtpBlock::list);
                self.part.dflash_pass(stream, tokens, &self.inputs.passes, &embed, list)?;
            } else {
                self.part.dflash_no_pass();
            }
            self.windows.clear();
            return Ok(None);
        }
        self.part.dflash_no_pass();

        // The newest cell of every sequence in the MTP block. Its chains run
        // in a step that only drafts: the rows that verify them have their
        // tokens, which every device thread needs.
        if !self.inputs.chains.is_empty() {
            return Err(invalid("the MTP block's chains run in a step that only drafts"));
        }
        // (A pass's rows are followed by the block after the pass, micro-batch by
        // micro-batch.)
        if let Some(block) = self.mtp.as_mut().filter(|_| pass.is_none()) {
            block.step(
                stream,
                &self.inputs.segments,
                &self.inputs.tokens,
                &self.inputs.replay,
                None,
                false,
            )?;
        }

        // The rows of the n-gram table, and where the sequences are now (a pass that goes on
        // read its rows when it started).
        if self.drives_ngram && !resumes {
            let mut ngram = self.flow.lock_ngram();
            let mut parts = Vec::with_capacity(self.inputs.segments.len());
            for s in &self.inputs.segments {
                ngram.begin_sequence(s.slot, s.pos0 as u32)?;
                let rows = s.row0 as usize..(s.row0 + s.n_rows) as usize;
                parts.push((s.slot, &self.inputs.tokens[rows]));
            }
            ngram.ask(&parts)?;
            for (slot, tokens) in parts {
                let direct = self.inputs.segments.iter().any(|s| s.slot == slot && s.direct);
                if direct {
                    ngram.advance(slot, tokens);
                } else {
                    ngram.hold(slot, tokens);
                }
            }
        }
        self.windows.clear();
        self.windows
            .extend(self.inputs.segments.iter().filter(|s| !s.direct).map(|s| (s.slot, s.row0)));
        self.shape.clear();
        self.shape
            .extend(self.inputs.segments.iter().map(|s| (s.slot, s.direct, s.n_rows, s.row0)));
        if let Some(key) = pass {
            return self.enqueue_pass(step, key, number as u32);
        }

        // The residual streams start from the embeddings, which the part that
        // embeds stages and gathers itself (`blocks/rows.rs`).
        // The rows of the step's image cells, copied over the gathered ones by
        // the part that embeds.
        self.part.splice_rows(rows::image_splices(record, step.image_stage, self.part.n_embd()));
        let rotary = (!self.inputs.rotary.is_empty()).then_some(self.inputs.rotary.as_slice());
        let mut host =
            StepRecord::new(&self.inputs.segments, &self.inputs.tokens, rotary, &self.out_rows)?;
        for (i, s) in self.inputs.segments.iter().enumerate() {
            if self.inputs.replay.iter().any(|r| r.contains(&s.row0)) {
                host.mark_replay(i)?;
            }
        }
        let (segments, tokens, replay) =
            (&self.inputs.segments, &self.inputs.tokens, &self.inputs.replay);

        // The step: every layer, with the host serving the store behind it.
        let (token, xbase) =
            (number as u32, (number as u32).wrapping_mul(self.per_step).wrapping_add(1));
        let inbound = if self.by_layers && self.index > 0 {
            Some(step.links.receive(self.index - 1, stream)?)
        } else {
            None
        };
        macro_rules! env {
            ($fetched:expr, $inbound:expr) => {
                StepEnv {
                    stream,
                    record: &host,
                    segments,
                    token,
                    tokens,
                    ngram: $fetched,
                    inbound: $inbound,
                    xbase,
                    pools: step.pages,
                    replay,
                }
            };
        }
        self.part.begin(&env!(None, inbound))?;
        self.part.dflash_begin(&env!(None, None))?;
        if inbound.is_some() {
            step.links.release(self.index - 1, stream)?;
        }
        let mut fetched: Option<Vec<u8>> = None;
        let ngram_at = self.part.ngram_layer();
        for i in 0..self.part.n_layers() {
            if ngram_at == Some(i) && fetched.is_none() {
                fetched = Some(self.flow.await_rows()?);
            }
            let env = env!(fetched.as_deref(), None);
            self.flow.enqueue_layer(&mut self.part, self.index, &env, i)?;
            if rows > MAX_BATCH {
                let part = &mut self.part;
                let (index, routing) = (self.index, &self.routing);
                let turn = Turn { part, index, env: &env, routing, pass: false };
                wide_layer(&self.flow, i, &mut [turn])?;
            }
            self.flow.pump()?;
        }
        let n_out = self.part.end(&env!(fetched.as_deref(), None))?;
        self.part.dflash_inject(stream)?;
        if self.mtp.is_some() {
            self.residual = Some(self.part.residual());
        }
        if self.by_layers && self.index + 1 < self.devices {
            step.links.send(self.index, stream, self.part.residual(), rows * self.wide)?;
        }
        Ok((n_out > 0 && self.part.has_head()).then(|| Logits {
            ptr: self.part.head_logits(),
            stride: self.n_vocab,
            first_token: 0,
        }))
    }

    fn wait_step(&mut self, end: &Event) -> Result<(), ModelError> {
        let flow = &self.flow;
        flow.wait_for(end, "a device ran the step")?;
        // Every route record of this part is with the store.
        flow.follow(self.index)?;
        // The MTP block's chains of a step that only drafts made their drafts
        // on the device; the round after it verifies them, or the first of
        // them, and the rows of the n-gram table its rows need are read now,
        // while the conductor decides.
        if let Some(block) = self.mtp.as_mut().filter(|_| self.only_drafts) {
            let chains = block.chains();
            let next: Vec<(u32, &[u32])> =
                chains.iter().map(|(slot, tokens)| (*slot, tokens.as_slice())).collect();
            // A failed read ahead costs nothing but the read of the step.
            let _ = flow.lock_ngram().expect(&next);
        }
        flow.finish(!self.only_drafts)
    }
}

impl DeviceModel for Qwen4ExpDevice {
    fn logit_tokens(&self) -> DevicePtr {
        self.part.head_tokens()
    }

    fn logits_per_row(&self) -> u32 {
        if self.part.has_head() { self.n_vocab } else { 0 }
    }

    fn tower(&mut self, stream: &Stream, job: &mut TowerJob) -> Result<(), ModelError> {
        match &mut self.tower {
            Some(tower) => tower.run(stream, job),
            None => Err(invalid("this part has no image tower")),
        }
    }

    fn enqueue(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        let result = self.enqueue_step(step);
        if result.is_err() {
            self.flow.fail();
            self.settle(step.stream);
        }
        result
    }

    fn wait(&mut self, end: &Event) -> Result<(), ModelError> {
        let result = self.wait_step(end);
        if result.is_err() {
            self.flow.fail();
            self.flow.drain(|| end.is_done().unwrap_or(true));
        }
        result
    }

    fn picked(&mut self, picks: &[Pick]) {
        // Every sequence most likely goes on with the token it just picked:
        // the rows of the table for that step are asked for now, before the
        // conductor has decided anything.
        let one_each = picks.len() == self.shape.len()
            && self
                .shape
                .iter()
                .zip(&self.out_rows)
                .all(|(&(_, direct, rows, row0), &out)| direct && out == row0 + rows - 1);
        if !one_each {
            return;
        }
        let next: Vec<(u32, [u32; 1])> =
            self.shape.iter().zip(picks).map(|(&(slot, ..), pick)| (slot, [pick.token])).collect();
        let next: Vec<(u32, &[u32])> =
            next.iter().map(|(slot, token)| (*slot, token.as_slice())).collect();
        // A failed read ahead costs nothing but the read of the step.
        let _ = self.flow.lock_ngram().expect(&next);
    }

    fn model_drafts(&self) -> Option<ModelDrafts> {
        self.drafters().drafted()
    }

    fn transfer(
        &mut self,
        cx: &TransferContext<'_>,
        item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        // A page that comes in is in use from now on, before a step's tables
        // name it: the piece of the far pool it lies in is not given back.
        if item.kind == ares_runtime::TransferKind::PageIn {
            self.mark_live(item.object);
        }
        self.resume.transfer(cx.stream, cx.pages, item, &mut self.part, self.mtp.as_mut())
    }

    fn model_draft_ms(&mut self) -> f64 {
        step::draft_ms(self.mtp.as_mut().map(MtpBlock::driver_mut), self.part.dflash_mut())
    }

    fn model_draft_probs(&mut self, out: &mut Vec<f32>) {
        self.drafters().probs(out);
    }

    fn paused(&self) -> bool {
        self.paused
    }

    fn rows_not_exact(&mut self, out: &mut Vec<u32>) {
        // The store left rows that verify drafts without an expert only the
        // file held; every part reports what the store knows when it has
        // ended, and the last to end knows all of it. A step that only
        // drafted has no rows.
        if self.only_drafts {
            return;
        }
        let left = self.flow.lock_feed().store.rows_left();
        out.extend((0..32).filter(|row| left >> row & 1 != 0));
    }
}
