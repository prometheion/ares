//! The step record: written for the parts of a step, run on the devices, and
//! the paused step that goes on.

use super::*;

impl Conductor {
    /// Hands the record of `used` bytes to the devices and waits for them to
    /// run it: the one round trip of every step (a step of rows, the step that
    /// only drafts, a step of the warm-up), which the watchdog sees
    /// ([`Heartbeat`]). Returns what the step took in milliseconds: the
    /// executor's own time when it knows it, else the host's from the dispatch
    /// to the end of the wait.
    pub(super) fn run_on_devices(&mut self, used: usize) -> Result<f64, ExecError> {
        let t_dispatch = Instant::now();
        self.step_no += 1;
        let meta = StepMeta { step: self.step_no, record_bytes: used, after_idle: self.after_idle };
        if let (false, Some(done)) = (self.after_idle, self.last_done_ns) {
            self.stats.record(Phase::Turnaround, self.clock.now_ns().saturating_sub(done));
        }
        self.after_idle = false;
        self.heartbeat.begin();
        let dispatched = self.exec.dispatch(&meta);
        let t_wait = Instant::now();
        let collected = dispatched
            .and_then(|()| self.exec.collect(self.config.conductor_spin, &mut self.output));
        self.heartbeat.end();
        let woke = collected?;
        let t_back = Instant::now();
        self.stats.record(Phase::Dispatch, (t_wait - t_dispatch).as_nanos() as u64);
        self.stats.record(Phase::Wait, (t_back - t_wait).as_nanos() as u64);
        self.stats.counters.conductor_wakes.record(woke);
        self.last_done_ns = self.output.devices.iter().map(|d| d.finished_at_ns).max();
        let step_ms = (t_back - t_dispatch).as_secs_f64() * 1e3;
        Ok(self.output.step_ms.take().unwrap_or(step_ms))
    }

    /// Takes places for the checkpoints due inside the rows of the planned parts.
    /// The latest first: when the memory holds fewer places than are due, the
    /// checkpoints nearest the step's end are the ones a following turn resumes
    /// from (after steps cut at each, a later save would push an earlier out).
    pub(super) fn reserve_inner(&mut self) {
        self.inner.clear();
        let Some(cache) = &mut self.cache else { return };
        for &(slot, position) in self.inner_due.iter().rev() {
            if !self.parts.iter().any(|p| p.slot == slot) {
                continue;
            }
            let (place, dropped) = cache.reserve();
            self.stats.counters.checkpoints_dropped += u64::from(dropped);
            if let Some(place) = place {
                self.inner.push(crate::InnerSave { slot, place, position: position as u32 });
            }
        }
        self.inner.sort_unstable_by_key(|save| (save.slot, save.position));
    }

    /// Gives back the places taken for checkpoints inside a step that will not run.
    pub(super) fn release_inner(&mut self, saves: Vec<crate::InnerSave>) {
        if let Some(cache) = &mut self.cache {
            for save in saves {
                cache.unreserve(save.place);
            }
        }
    }

    /// Runs the step whose record (`used` bytes) is written, for `turn`, with
    /// `generating` sequences generating and `chain_ms` milliseconds of the
    /// drafters' step before it; `observe` lets its time teach the prices. A
    /// step the model paused keeps its part for later and commits nothing.
    pub(super) fn run_record(
        &mut self,
        began: Instant,
        used: usize,
        turn: Turn,
        generating: u32,
        chain_ms: f64,
        observe: bool,
    ) {
        // What the step carries: every row and the cells it reads.
        let load = self.parts.iter().fold(Load::default(), |load, part| {
            load.plus(load_of(self.span, part.from, part.rows))
        });
        // The devices.
        let step_ms = match self.run_on_devices(used) {
            Ok(ms) => ms,
            Err(error) => return self.fail(error.to_string()),
        };
        // A step the model paused, or one that went on where it was paused, did
        // part of its rows' work.
        if observe && !self.output.paused {
            self.prices.observe(load, step_ms);
        }
        // The round took its chains and its rows.
        let step_ms = step_ms + chain_ms;
        let waiting = self.prompt_rows.iter().any(|rows| *rows > 0);
        if let Turn::Piece { .. } = turn {
            // The generating sequences that rode along got a token each.
            let riders = self.parts.iter().filter(|p| matches!(p.kind, PartKind::Window)).count();
            let riding = if generating > 0 { riders as f64 / f64::from(generating) } else { 0.0 };
            // A request in line waits for a place that a generating one gives back.
            let crowded = !self.waiting.is_empty();
            self.share.prompts_ran(step_ms, riding, generating, crowded);
        }
        let last_done = self.last_done_ns.unwrap_or(0);
        for (device, times) in self.output.devices.iter().enumerate() {
            let notice = if times.finished_at_ns == last_done {
                self.clock.now_ns().saturating_sub(last_done)
            } else {
                0
            };
            self.stats.record_device(device, times, notice);
        }
        if self.output.paused {
            // The model stopped between two pieces of the step: its rows run again
            // later, alone, and go on there. What the record carried for the step
            // before (its windows' commits) has been done.
            if self.parts.len() != 1 {
                return self.fail("the model paused a step of several sequences".into());
            }
            let part = self.parts[0];
            self.paused = Some(part);
            self.hold_images(part);
            self.paused_inner = std::mem::take(&mut self.inner);
            self.paused_due = self.inner_due.iter().filter(|d| d.0 == part.slot).copied().collect();
            self.commits.clear();
            self.stats.counters.steps_paused += 1;
            self.stats.record(Phase::Step, began.elapsed().as_nanos() as u64);
            self.stats.counters.steps += 1;
            return;
        }

        // The paused step has run to its end (or another step ran while it waits).
        if self.paused.is_none() {
            self.paused_images.clear();
        }
        let t_commit = Instant::now();
        self.commit();
        self.stats.record(Phase::Commit, t_commit.elapsed().as_nanos() as u64);
        if turn == Turn::Round {
            // What the round gave each generating sequence teaches the share
            // the rate of rounds.
            let windows = self.parts.iter().filter(|p| matches!(p.kind, PartKind::Window)).count();
            let tokens = f64::from(self.round_tokens) / windows.max(1) as f64;
            self.share.round_ran(step_ms, tokens, generating, waiting);
        }

        let t_events = Instant::now();
        self.deliver();
        self.stats.record(Phase::Events, t_events.elapsed().as_nanos() as u64);
        self.stats.record(Phase::Step, began.elapsed().as_nanos() as u64);
        self.stats.counters.steps += 1;
    }

    /// Runs the step the model paused again, alone, for at most `budget_ms`
    /// milliseconds before the model may pause it again (0: as long as it takes):
    /// the same rows, whose cells are claimed already; the model goes on where it
    /// stopped.
    pub(super) fn resume(&mut self, began: Instant, budget_ms: u32) {
        let Some(part) = self.paused.take() else { return };
        if !self.slots[part.slot as usize].as_ref().is_some_and(|a| a.seq == part.seq) {
            self.paused_images.clear();
            self.give_up = true;
            let saves = std::mem::take(&mut self.paused_inner);
            self.release_inner(saves);
            return;
        }
        // The checkpoints inside its rows were named when it began; the model keeps them.
        self.inner = std::mem::take(&mut self.paused_inner);
        self.inner_due.extend(std::mem::take(&mut self.paused_due));
        let t_write = Instant::now();
        self.parts.clear();
        self.parts.push(part);
        if let Some(cache) = &mut self.cache {
            cache.settle_leaving(&mut self.pool, &mut *self.exec);
        }
        if let Err(message) = self.make_room() {
            return self.fail(message);
        }
        if self.parts.is_empty() {
            return;
        }
        let used = match self.write_record(RecordKind::Resume { pause_ms: budget_ms }) {
            Ok(used) => used,
            Err(message) => return self.fail(message),
        };
        self.stats.record(Phase::Write, t_write.elapsed().as_nanos() as u64);
        let generating = self.generating_besides(Some(part.slot as usize));
        let turn = Turn::Piece { rows: part.rows };
        self.run_record(began, used, turn, generating, 0.0, false);
    }

    /// Writes the step record for the parts in `self.parts`, of the `kind`
    /// it is (see [`RecordKind`]). Returns its length in bytes. The commit of
    /// the step before goes with whichever comes first.
    pub(super) fn write_record(&mut self, kind: RecordKind) -> Result<usize, String> {
        let full = |e: record::RecordFull| format!("{e}");
        let drafts_only = kind == RecordKind::DraftsOnly;
        let resuming = matches!(kind, RecordKind::Resume { .. });
        let (mut segments, mut rows) = (0u32, 0u32);
        for part in self.parts.iter().filter(|p| !drafts_only || self.takes_model_drafts(p)) {
            segments += 1;
            rows += part.rows;
        }
        let buffer = self.exec.record();
        let mut writer =
            RecordWriter::new(buffer, self.step_no + 1, segments, rows).map_err(full)?;
        match kind {
            RecordKind::DraftsOnly => writer.drafts_only(),
            RecordKind::Round { pause_ms } | RecordKind::Resume { pause_ms } => {
                writer.pause_after(pause_ms)
            }
        }
        if std::mem::take(&mut self.give_up) {
            writer.give_up_paused();
        }
        writer.asked(self.asked);
        if !resuming && !drafts_only {
            writer.inner_saves(&self.inner).map_err(full)?;
        }
        self.runs.clear();
        self.samplers.clear();
        self.step_rows.clear();
        self.updates.clear();
        self.bars.clear();
        self.worth.clear();
        self.image_runs.clear();
        self.staged.clear();
        let mut stage_at = 0u32;
        let mut row_index = 0u32;
        let counters = &mut self.stats.counters;
        let window = self.spec.window.max(1) as usize;
        let mut has_prompt = false;
        let mut has_generation = false;
        // The drafter whose table of chances the bars are read with: the
        // first that ends chains by them (a model has one MTP block at most).
        let mut bars_of: Option<usize> = None;
        for part in &self.parts {
            let own = if drafts_only {
                self.slots[part.slot as usize].as_ref().and_then(|a| part.model_drafter(a))
            } else {
                None
            };
            if drafts_only && own.is_none() {
                continue;
            }
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            // The position of the first row.
            let first_cell = part.from as u32;
            match part.kind {
                PartKind::Prefill { completes } => {
                    has_prompt = true;
                    if resuming {
                        // Counted when the step was first written.
                    } else if part.replay {
                        counters.replay_rows += u64::from(part.rows);
                    } else {
                        counters.prefill_rows += u64::from(part.rows);
                    }
                    // The rows of a prompt are certain: recurrent state
                    // advances with them.
                    writer.segment(part.slot, first_cell, part.rows, true).map_err(full)?;
                    if part.replay {
                        writer.replay().map_err(full)?;
                    }
                    for k in 0..part.rows {
                        let position = first_cell + k;
                        writer
                            .row_at(
                                Row {
                                    slot: part.slot,
                                    position,
                                    token: a.tokens[position as usize],
                                    context: self
                                        .spec
                                        .reduce(a.states[position as usize], position),
                                },
                                a.spans.rotary(position),
                            )
                            .map_err(full)?;
                    }
                    // The rows of the images among them come from the store,
                    // through the stage.
                    if let Some(first) = a.spans.from(first_cell) {
                        let end = first_cell + part.rows;
                        for span in a.spans.spans()[first..].iter() {
                            if span.first >= end {
                                break;
                            }
                            let (lo, hi) = (span.first.max(first_cell), span.end().min(end));
                            // A paused step's own rows first: the store may have let
                            // them go meanwhile.
                            let held =
                                self.paused_images.iter().find(|(n, _)| *n == span.image.name);
                            let rows = match held {
                                Some((_, rows)) => Some(rows.clone()),
                                None => self.images.get(&span.image.name),
                            };
                            let Some(rows) = rows else {
                                return Err(format!(
                                    "the rows of an image of request {} are not in the store",
                                    a.id
                                ));
                            };
                            self.image_runs.push(ImageRun {
                                row: row_index + (lo - first_cell),
                                rows: hi - lo,
                                stage: stage_at,
                            });
                            self.staged.push((rows, lo - span.first, hi - lo));
                            stage_at += hi - lo;
                        }
                    }
                    if completes {
                        self.runs.push((row_index + part.rows - 1, 1));
                        let row = a.sampler.row(
                            first_cell + part.rows - 1,
                            &a.tokens,
                            a.prompt_len as usize,
                            &[],
                            0,
                            &mut self.step_rows,
                        );
                        self.samplers.push(row);
                    }
                }
                PartKind::Window | PartKind::Score { .. } => {
                    has_generation = !drafts_only;
                    if !drafts_only {
                        counters.window_rows += 1;
                        counters.draft_rows += u64::from(part.rows - 1);
                    }
                    let newest = a.tokens.len() - 1;
                    // A window with drafts leaves recurrent state alone until
                    // the next step commits the rows that were accepted.
                    writer
                        .segment(part.slot, first_cell, part.rows, part.rows == 1)
                        .map_err(full)?;
                    if let Some(d) = own {
                        // The draft rows' tokens below are placeholders: the
                        // model's drafter makes them.
                        writer.model_drafts(d as u32 + 1).map_err(full)?;
                        let links = part.rows as usize - 1;
                        let by_bars = self.drafters[d].kind.ends_chains_by_bars();
                        let needs: &[f32] = if by_bars { &a.model_needs } else { &[] };
                        self.bars.extend(needs.iter().take(links));
                        self.bars.resize(self.bars.len() + links - links.min(needs.len()), 0.0);
                        if by_bars {
                            bars_of.get_or_insert(d);
                        }
                    }
                    // A link of the chain picks with the sampler row of the
                    // row that verifies it, made as the first of the window:
                    // without a mask for the drafts before it, which are not
                    // known yet. The round verifies them with their masks.
                    let known: &[u32] = if own.is_some() { &[] } else { &a.draft };
                    for k in 0..part.rows as usize {
                        let row = a.sampler.row(
                            first_cell + k as u32,
                            &a.tokens,
                            a.prompt_len as usize,
                            known,
                            k.min(known.len()),
                            &mut self.step_rows,
                        );
                        self.samplers.push(row);
                    }
                    writer
                        .row_at(
                            Row {
                                slot: part.slot,
                                position: first_cell,
                                token: a.tokens[newest],
                                context: self.spec.reduce(a.states[newest], first_cell),
                            },
                            a.spans.rotary(first_cell),
                        )
                        .map_err(full)?;
                    // The contexts of the drafts follow from the drafts
                    // themselves, as they will once they are accepted.
                    self.scratch_tokens.clear();
                    self.scratch_tokens
                        .extend_from_slice(&a.tokens[a.tokens.len().saturating_sub(window)..]);
                    let mut state = a.states[newest];
                    for (k, &draft) in a.draft.iter().enumerate() {
                        self.scratch_tokens.push(draft);
                        state = self.spec.advance(state, &self.scratch_tokens);
                        let position = first_cell + 1 + k as u32;
                        writer
                            .row_at(
                                Row {
                                    slot: part.slot,
                                    position,
                                    token: draft,
                                    context: self.spec.reduce(state, position),
                                },
                                a.spans.rotary(position),
                            )
                            .map_err(full)?;
                    }
                    // Every row of a window is picked, in one run with the
                    // windows before it.
                    match self.runs.last_mut() {
                        Some((first, count)) if *first + *count == row_index => *count += part.rows,
                        _ => self.runs.push((row_index, part.rows)),
                    }
                    // What each draft row is worth: the chance that the
                    // drafts up to its own are right, times what a token
                    // costs.
                    // Before the step has a price nothing is known of it.
                    if !drafts_only && part.rows > 1 && self.drafting.token_ms > 0.0 {
                        let chances = if a.model_drafter.is_some() {
                            &a.model_chances
                        } else {
                            &a.ngram_chances
                        };
                        self.worth.resize(row_index as usize + 1, 0.0);
                        let mut reach = 1.0;
                        for k in 0..part.rows as usize - 1 {
                            reach *= chances.get(k).copied().unwrap_or(0.0);
                            self.worth.push((reach * self.drafting.token_ms) as f32);
                        }
                    }
                }
            }
            row_index += part.rows;
        }
        if !drafts_only {
            counters.rows += u64::from(row_index);
        }
        if has_prompt && has_generation {
            counters.mixed_steps += 1;
        }
        writer.runs(&self.runs).map_err(full)?;
        writer.corrections(&self.step_rows.entries).map_err(full)?;
        writer.masks(&self.step_rows.masks, self.mask_words).map_err(full)?;
        writer.samplers(self.samplers.iter().copied()).map_err(full)?;
        writer.commits(&self.commits).map_err(full)?;
        if drafts_only {
            // The devices commit these with this record: the round's must not
            // carry them again.
            self.commits.clear();
        }
        // Page-table changes since the devices last saw the tables.
        for part in &self.parts {
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let dirty = a.cells.take_dirty();
            if !dirty.is_empty() {
                self.updates.push((part.slot, dirty));
            }
        }
        let (slots, model) = (&self.slots, &self.model);
        writer
            .updates(
                self.updates.iter().map(|(slot, range)| TableUpdate {
                    slot: *slot,
                    first: range.start as u32,
                    count: range.len() as u32,
                }),
                self.updates.iter().flat_map(|(slot, range)| {
                    let a = slots[*slot as usize].as_ref().expect("running");
                    a.cells.pages()[range.clone()].iter().map(|page| model.page_entry(page.0))
                }),
            )
            .map_err(full)?;
        self.freed.clear();
        self.pool.take_freed(record::FREED_PER_RECORD, &mut self.freed);
        writer.freed(&self.freed).map_err(full)?;
        if let Some(d) = bars_of {
            writer.draft_bars(&self.drafting.confidences[d].chances(), &self.bars).map_err(full)?;
        }
        if !self.worth.is_empty() {
            self.worth.resize(row_index as usize, 0.0);
            writer.row_worth(&self.worth).map_err(full)?;
        }
        if !self.image_runs.is_empty() {
            writer.images(&self.image_runs).map_err(full)?;
        }
        if let Some(cache) = &mut self.cache {
            cache.write_copies(&mut writer).map_err(full)?;
        }
        let used = writer.finish().map_err(full)?;
        self.fill_stage();
        Ok(used)
    }
}
