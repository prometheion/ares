//! What a round asks for: the pieces of prompt and the windows, with the
//! drafts each verifies (the policy is `ares_sched::Drafting`); the step that
//! only drafts, and the drafts of it that the round verifies.

use super::*;

impl Conductor {
    /// Decides how many drafts each generating sequence asks for, and
    /// fills `requests` for the planner. A round of generation asks for the
    /// windows (`prompts` false), a piece of prompt for the prompts and the
    /// newest token of each generating sequence, without drafts.
    ///
    /// The drafts of the windows are chosen together, by the rate of the
    /// whole step ([`ares_sched::choose`]): each sequence offers what the
    /// n-gram drafter proposes (with its confidence in each token) and
    /// what the model's MTP block is likely to get right at each
    /// position (measured for this sequence, from what the engine saw where it
    /// has none yet), and the step takes the rows that pay, from the drafter
    /// that pays, by the price of a step of that many rows.
    pub(super) fn want(&mut self, turn: Turn) {
        self.requests.clear();
        let round = turn == Turn::Round;
        let span = u64::from(self.span);
        let generating = self.generating_besides(None);
        // Every draft row needs its logits, like the one row every sequence
        // may need: drafts share what the model gives beyond those, and the
        // rows of a step that verifies drafts are bounded by the model.
        let spare_rows = self
            .model
            .logit_rows()
            .saturating_sub(self.active as u32)
            .min(self.model.draft_rows().saturating_sub(generating));
        let drafting = round && !self.drafters.is_empty() && self.config.window_rows > 1;
        let passes = self.drafting.passes();
        // The windows that offer nothing are rows of the step all the same:
        // the price of the step counts them in.
        let (mut quiet, mut quiet_cells) = (0u32, 0u64);
        let paused = self.paused.map(|p| p.slot as usize);
        for (slot, a) in self.slots.iter_mut().enumerate() {
            let Some(a) = a else { continue };
            // The paused step's sequence runs only that step, alone.
            if Some(slot) == paused {
                continue;
            }
            if !a.generating() {
                // Prompts wait for their piece, and for the rows of an image
                // the tower has not computed yet.
                let ready = a.ready_rows(a.state_pos(), a.prefill_left(), &self.images);
                if !round && ready > 0 {
                    self.requests.push(Request {
                        seq: SeqId(slot as u32),
                        want: Want::Prefill { tokens: ready },
                        waited_steps: a.waited_steps,
                        priority: a.priority,
                    });
                }
                continue;
            }
            a.draft.clear();
            a.model_drafter = None;
            a.ngram.clear();
            a.ngram_chances.clear();
            a.model_chances.clear();
            a.offered.iter_mut().for_each(Vec::clear);
            if drafting && a.draft_allowed {
                // A window never needs rows past the end of the request: the
                // tokens still wanted, and the room left in the context.
                let wanted = a.max_tokens.map_or(u32::MAX, |m| m - a.completion);
                let room = self.context_length.saturating_sub(a.tokens.len() as u32);
                let cap = self
                    .config
                    .window_rows
                    .saturating_sub(1)
                    .min(wanted.saturating_sub(1))
                    .min(room)
                    // A sequence whose next pick depends on the picks before
                    // it by a state only a pick can advance verifies no drafts.
                    .min(if a.sampler.allows_drafts() { u32::MAX } else { 0 })
                    as usize;
                // The window ends where a checkpoint is due, so that the
                // state is exactly there when all of it is accepted.
                let cap = match &self.cache {
                    Some(cache) => {
                        let from = a.cells.len();
                        let to = from + 1 + cap as u64;
                        match cache.capture(a.prompt_len, &a.boundaries, a.checkpoint_at, from, to)
                        {
                            Some(at) => cap.min((at - from - 1) as usize),
                            None => cap,
                        }
                    }
                    None => cap,
                };
                if cap > 0 {
                    // The n-gram drafter first: its proposal costs nothing, and it
                    // learns from what is right whether it is taken or not.
                    a.drafter.draft(&self.corpus, cap, &mut a.ngram, &mut a.ngram_chances);
                    // A draft that a rule of the request forbids cannot be
                    // picked: it and the ones after it are not offered.
                    a.sampler.cut_drafts(
                        &a.tokens,
                        a.prompt_len as usize,
                        &mut a.ngram,
                        &mut a.ngram_chances,
                    );
                    // The model's drafters know no rule: a request with one
                    // asks for them while the rule leaves its next token
                    // free. The round verifies their drafts with their masks
                    // and corrections, as an n-gram draft's.
                    let free = a.sampler.allows_model_drafts(&a.tokens, a.prompt_len as usize);
                    for (d, drafter) in self.drafters.iter().enumerate() {
                        let own = if free { cap.min(drafter.drafts as usize) } else { 0 };
                        let chances = &mut a.offered[d];
                        a.model_acceptance[d].rates(
                            &self.drafting.engines[d],
                            self.config.draft_prior,
                            own,
                            chances,
                        );
                        // A draft counts when its link is made and it is right.
                        for (chance, (_, made)) in chances.iter_mut().zip(&passes[d]) {
                            *chance *= made;
                        }
                    }
                }
            }
            if a.ngram_chances.is_empty() && a.offered.iter().all(Vec::is_empty) {
                quiet += 1;
                quiet_cells += a.cells.len().min(span);
            }
            self.requests.push(Request {
                seq: SeqId(slot as u32),
                want: Want::Window { drafts: 0 },
                waited_steps: a.waited_steps,
                priority: a.priority,
            });
        }
        if !drafting {
            return;
        }
        // The offers of every generating sequence that has any: the n-gram drafter's,
        // then each model drafter's under its id.
        let mut offers: Vec<Vec<Offer<'_>>> = Vec::new();
        let mut asking: Vec<usize> = Vec::new();
        for (slot, a) in self.slots.iter().enumerate() {
            let Some(a) = a.as_ref().filter(|a| a.generating() && a.draft_allowed) else {
                continue;
            };
            if a.ngram_chances.is_empty() && a.offered.iter().all(Vec::is_empty) {
                continue;
            }
            offers.push(
                std::iter::once(Offer { drafter: NGRAM, chances: &a.ngram_chances })
                    .chain(
                        a.offered
                            .iter()
                            .enumerate()
                            .map(|(d, chances)| Offer { drafter: d + 1, chances }),
                    )
                    .collect(),
            );
            asking.push(slot);
        }
        let wishes: Vec<Wish<'_>> = asking
            .iter()
            .zip(&offers)
            .map(|(slot, offers)| {
                let a = self.slots[*slot].as_ref().expect("asked");
                Wish { reads: a.cells.len().min(span), offers }
            })
            .collect();
        // The prices a window is tried at: a size that has not run for long
        // costs no more than the nearest smaller one that has, so it is taken
        // again when it could pay, and the step measures it.
        let step = |load: Load| {
            self.prices.trial_ms(Load { rows: load.rows + quiet, cells: load.cells + quiet_cells })
        };
        let (prior, window_rows) = (self.config.draft_prior, self.config.window_rows);
        let choices =
            self.drafting.choose_windows(&wishes, &step, spare_rows, &passes, prior, window_rows);
        drop(wishes);
        let mut asked = vec![false; self.drafters.len()];
        for (slot, choice) in asking.iter().zip(choices) {
            let a = self.slots[*slot].as_mut().expect("asked");
            let Choice { drafter, drafts } = choice;
            let drafts = drafts as usize;
            match drafter {
                Some(NGRAM) => a.draft.extend_from_slice(&a.ngram[..drafts]),
                Some(id) => {
                    let d = id - 1;
                    a.model_drafter = Some(d);
                    asked[d] = true;
                    a.draft.resize(drafts, 0);
                    a.model_chances.clear();
                    a.model_chances.extend_from_slice(&a.offered[d]);
                    // The first link is the window's reason: it has no bar.
                    a.model_needs.clear();
                    a.model_needs.push(0.0);
                    let needs = &self.drafting.needs[d];
                    a.model_needs.extend(needs.iter().take(drafts.saturating_sub(1)));
                }
                None => {}
            }
            for (d, chances) in a.offered.iter().enumerate() {
                if a.model_drafter != Some(d) && !chances.is_empty() {
                    a.model_acceptance[d].fade(UNASKED_SEQUENCE);
                }
            }
            if let Some(request) = self.requests.iter_mut().find(|r| r.seq.0 == *slot as u32) {
                request.want = Want::Window { drafts: drafts as u32 };
            }
        }
        self.drafting.fade_unasked(&asked);
    }

    /// Whether `part` is a window whose drafts one of the model's drafters
    /// makes.
    pub(super) fn takes_model_drafts(&self, part: &StepPart) -> bool {
        self.slots[part.slot as usize].as_ref().and_then(|a| part.model_drafter(a)).is_some()
    }

    /// The step that only drafts: the model's drafters make the drafts of the
    /// windows that take them, as many as were asked (or as far as their bars
    /// let a chain go), and nothing else runs. The drafts come back with what
    /// the drafter said of each, and the round then verifies as many of each
    /// window as pay ([`Conductor::verify_lengths`]). Returns what the step
    /// took, in milliseconds: the price of the drafts.
    pub(super) fn chains_first(&mut self) -> Result<f64, String> {
        let used = self.write_record(RecordKind::DraftsOnly)?;
        let step_ms = self.run_on_devices(used).map_err(|error| error.to_string())?;

        // The drafts of every chain, in the order of the windows: those of the
        // links that were made, and what the drafter said of them.
        let reports = self.output.draft_probs.len() == self.output.drafts.len();
        let mut at = 0usize;
        let mut longest = vec![0usize; self.drafters.len()];
        for part in &self.parts {
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let Some(d) = part.model_drafter(a) else {
                continue;
            };
            let asked = part.rows as usize - 1;
            let end = (at + asked).min(self.output.drafts.len());
            let drafts = &self.output.drafts[at..end];
            let made = drafts.iter().take_while(|draft| **draft != u32::MAX).count();
            a.draft.clear();
            a.draft.extend_from_slice(&drafts[..made]);
            a.model_probs.clear();
            if reports {
                a.model_probs.extend_from_slice(&self.output.draft_probs[at..at + made]);
            }
            at = end;
            longest[d] = longest[d].max(made);
            let counters = &mut self.stats.counters;
            counters.model_links_asked += asked as u64;
            counters.model_links_cut += (asked - made) as u64;
        }
        if longest.iter().any(|links| *links > 0) {
            let counters = &mut self.stats.counters;
            counters.model_draft_steps += 1;
            counters.model_draft_links += *longest.iter().max().expect("a drafter") as u64;
            counters.model_draft_us += (step_ms * 1e3) as u64;
        }
        // The step's time is each drafter's that ran in it: a step of two
        // drafters is rare (two sequences that chose differently) and prices
        // both a little high.
        for (chain, links) in self.drafting.chains.iter_mut().zip(longest) {
            if links > 0 {
                chain.observe(links as u32, step_ms);
            }
        }
        self.verify_lengths();
        Ok(step_ms)
    }

    /// How many drafts of each chain of the model's own drafter the round
    /// verifies, now that they are made: each draft is worth the chance that
    /// it and the ones before it are accepted, read from the probability the
    /// drafter gave it ([`ares_sched::Confidence`]; by its position where the
    /// drafter says nothing), and the round takes as many as raise its rate
    /// ([`ares_sched::choose`], with the chains paid for). The rows of the
    /// drafts it does not take leave the step, and their cells the table.
    pub(super) fn verify_lengths(&mut self) {
        let span = u64::from(self.span);
        // The rows of the round that are not such windows are its rows all
        // the same.
        let (mut fixed, mut fixed_cells) = (0u32, 0u64);
        let mut asking: Vec<usize> = Vec::new();
        let mut chances: Vec<Vec<f64>> = Vec::new();
        let mut drafters: Vec<usize> = Vec::new();
        for (index, part) in self.parts.iter().enumerate() {
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let Some(d) = part.model_drafter(a) else {
                fixed += part.rows;
                fixed_cells += load_of(self.span, part.from, part.rows).cells;
                continue;
            };
            let mut own = Vec::with_capacity(a.draft.len());
            if a.model_probs.len() == a.draft.len() {
                own.extend(a.model_probs.iter().map(|p| self.drafting.confidences[d].chance(*p)));
            } else {
                a.model_acceptance[d].rates(
                    &self.drafting.engines[d],
                    self.config.draft_prior,
                    a.draft.len(),
                    &mut own,
                );
            }
            // A draft that a rule of the request forbids cannot be picked: it
            // and the ones after it are not verified.
            a.sampler.cut_drafts(&a.tokens, a.prompt_len as usize, &mut a.draft, &mut own);
            a.model_probs.truncate(a.draft.len());
            asking.push(index);
            chances.push(own);
            drafters.push(d + 1);
        }
        let offers: Vec<[Offer<'_>; 1]> = chances
            .iter()
            .zip(&drafters)
            .map(|(own, drafter)| [Offer { drafter: *drafter, chances: own }])
            .collect();
        let wishes: Vec<Wish<'_>> = asking
            .iter()
            .zip(&offers)
            .map(|(index, offers)| Wish { reads: self.parts[*index].from.min(span), offers })
            .collect();
        let step = |load: Load| {
            self.prices.trial_ms(Load { rows: load.rows + fixed, cells: load.cells + fixed_cells })
        };
        let spare_rows = self.model.draft_rows().saturating_sub(self.parts.len() as u32);
        let choices = self.drafting.shorten(&wishes, &step, spare_rows);
        drop(wishes);
        for ((index, choice), own) in asking.into_iter().zip(choices).zip(chances.iter()) {
            let part = &mut self.parts[index];
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let taken = (choice.drafts as usize).min(a.draft.len());
            // The chance of each draft the round verifies, for what its row
            // is worth.
            a.model_chances.clear();
            a.model_chances.extend(own.iter().take(taken).copied());
            let dropped = part.rows as usize - 1 - taken;
            self.stats.counters.model_drafts_unverified += (a.draft.len() - taken) as u64;
            a.draft.truncate(taken);
            a.model_probs.truncate(taken);
            part.rows = 1 + taken as u32;
            let keep = a.cells.len() - dropped as u64;
            a.cells.truncate(&mut self.pool, keep);
        }
    }
}
