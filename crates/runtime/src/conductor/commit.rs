//! What a step came to: the picks of every row taken, the drafts that were
//! accepted, the prompt cache told, and the text sent.

use super::*;

impl Conductor {
    /// Accepts or rejects what the devices brought back, sequence by
    /// sequence.
    pub(super) fn commit(&mut self) {
        let mut cursor = 0usize;
        let mut row0 = 0u32;
        self.commits.clear();
        self.round_tokens = 0;
        for part in &self.parts {
            let first_row = row0;
            row0 += part.rows;
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let before = a.completion;
            // Where the recurrent state is after the rows.
            let mut reached = part.from + u64::from(part.rows);
            match part.kind {
                PartKind::Prefill { completes } => {
                    if part.replay {
                        a.replay_left -= part.rows;
                        a.replayed += part.rows;
                    } else {
                        a.computed += part.rows;
                    }
                    // The rows that ran are certain: their tokens are text
                    // now, for every sequence to draft from. A request that
                    // came back computes rows whose tokens are there already.
                    for at in a.drafter.len()..a.cells.len() as usize {
                        a.drafter.push(&mut self.corpus, a.tokens[at]);
                    }
                    if completes {
                        let pick = self.output.picks[cursor];
                        let probs = self.output.logprobs.get(cursor..=cursor).unwrap_or(&[]);
                        a.sampler.picked(&[pick], probs);
                        cursor += 1;
                        a.accept(
                            &*self.model,
                            &self.spec,
                            self.context_length,
                            &mut self.corpus,
                            &[pick.token],
                        );
                    }
                }
                PartKind::Window | PartKind::Score { .. } => {
                    let picks = &self.output.picks[cursor..cursor + part.rows as usize];
                    let probs_at = cursor;
                    cursor += part.rows as usize;
                    // Draft `i` is accepted when it equals what the model
                    // picked after the row before it, and only if all the
                    // drafts before it were.
                    // Every draft of the window is known: an n-gram draft's, and of
                    // the model's own those its chain made that the round
                    // took. Their rows carry the masks the drafts before them
                    // make, so every pick is the request's.
                    let drafts = part.rows as usize - 1;
                    // A row the model left without what it would have waited
                    // for, and every row of the window after it, has no pick
                    // that counts: the rows before it test the drafts, and the
                    // last of them gives the token after the accepted ones.
                    let exact = self
                        .output
                        .inexact
                        .iter()
                        .find(|row| (first_row + 1..first_row + part.rows).contains(row))
                        .map_or(part.rows, |row| row - first_row)
                        as usize;
                    if exact < part.rows as usize {
                        self.stats.counters.windows_left += 1;
                    }
                    let made = drafts.min(a.draft.len()).min(exact);
                    let probs: &[f32] =
                        if a.model_drafter.is_some() { &a.model_probs } else { &[] };
                    let right = (0..made).take_while(|&i| picks[i].token == a.draft[i]).count();
                    let accepted = right.min(exact - 1);
                    if drafts > 0 {
                        let source = a.model_drafter.map_or(0, |d| self.drafters[d].kind.source());
                        if let Some(d) = a.model_drafter {
                            // A draft after the first rejected one was not
                            // tested.
                            let tested = made.min(right + 1);
                            a.model_acceptance[d].observe(tested, right.min(tested));
                            self.drafting.engines[d].observe(tested, right.min(tested));
                            // What its probability said of each draft that
                            // was tested.
                            let counters = &mut self.stats.counters;
                            for (i, p) in probs.iter().enumerate().take(tested) {
                                if i > right {
                                    break;
                                }
                                self.drafting.confidences[d].observe(*p, i < right);
                                let bin = ares_sched::Confidence::bin(*p);
                                counters.model_tested_by_p[bin] += 1;
                                counters.model_accepted_by_p[bin] += u64::from(i < right);
                            }
                        } else {
                            // The corpus learns what matches like this one
                            // are worth.
                            a.drafter.observe(&mut self.corpus, made, right);
                        }
                        // The request's own, counted as the engine's totals are:
                        // the rows of drafts its windows carried.
                        a.drafts.proposed += drafts as u32;
                        a.drafts.accepted += accepted as u32;
                        a.drafts.windows += 1;
                        let counters = &mut self.stats.counters;
                        counters.draft_lengths[source][made.min(DRAFT_POSITIONS)] += 1;
                        counters.drafts_accepted += accepted as u64;
                        counters.drafts_rejected += (made - accepted) as u64;
                        // Positions up to the first rejected one were tested;
                        // a draft no exact row tested was no draft.
                        for i in 0..made.min(right + 1) {
                            let at = i.min(counters.drafts_tested_at[source].len() - 1);
                            counters.drafts_tested_at[source][at] += 1;
                            counters.drafts_accepted_at[source][at] += u64::from(i < right);
                        }
                    }
                    // A proposal of the n-gram drafter that was not taken (the model's
                    // drafts were, or none) learns from the tokens that are now
                    // known: the first of them is known in every window.
                    if (a.model_drafter.is_some() || drafts == 0) && !a.ngram.is_empty() {
                        let known = a.ngram.len().min(accepted + 1);
                        let right =
                            (0..known).take_while(|&i| picks[i].token == a.ngram[i]).count();
                        a.drafter.observe(&mut self.corpus, known, right);
                    }
                    self.scratch_tokens.clear();
                    self.scratch_tokens.extend(picks[..=accepted].iter().map(|p| p.token));
                    let probs =
                        self.output.logprobs.get(probs_at..probs_at + accepted + 1).unwrap_or(&[]);
                    a.sampler.picked(&picks[..=accepted], probs);
                    a.accept(
                        &*self.model,
                        &self.spec,
                        self.context_length,
                        &mut self.corpus,
                        &self.scratch_tokens,
                    );
                    // A request that ends inside the accepted rows keeps
                    // those that came before its end: its state and its cells
                    // stop there.
                    let consumed = (a.completion - before) as usize;
                    // The devices commit the newest token's row and the
                    // accepted drafts to recurrent state with the next step.
                    if drafts > 0 {
                        self.commits.push(Commit {
                            row0: first_row,
                            rows: part.rows,
                            slot: part.slot,
                            accepted: consumed as u32,
                        });
                    }
                    // The newest token's row and the accepted drafts keep
                    // their cells; the rejected rows are cut from the table.
                    let keep = a.cells.len() - u64::from(part.rows) + consumed as u64;
                    a.cells.truncate(&mut self.pool, keep);
                    reached = part.from + consumed as u64;
                }
            }
            let produced = a.completion - before;
            if let PartKind::Window = part.kind {
                self.round_tokens += produced;
            }
            if produced > 0 {
                self.stats.counters.tokens += u64::from(produced);
                self.admission.generated(a.seq, produced);
            }
            // The cache: the pages that are full now enter the index, and a
            // checkpoint is taken where the rows ended on a position that is
            // due one (the rows were cut to end there).
            if let Some(cache) = &mut self.cache {
                let settled = a.cells.len();
                cache.index.publish(
                    &mut self.pool,
                    &mut a.cells,
                    &mut a.chain,
                    &a.names,
                    settled,
                    Priority::NORMAL,
                );
                cache.note(a.owner, &a.chain, settled, false);
                // The checkpoints the model saved inside the rows, in the order of their
                // positions, then the one at their end. A save inside that was due and
                // found no place is passed over (the later ones are named all the same).
                let cells = u64::from(self.pool.cells_per_page());
                let mut k = 0;
                while k < self.inner.len() {
                    let save = self.inner[k];
                    let at = u64::from(save.position);
                    if save.slot != part.slot || at <= part.from || at >= reached {
                        k += 1;
                        continue;
                    }
                    self.inner.remove(k);
                    let page = (at / cells) as usize;
                    let root = cache.at_root(&a.boundaries, at);
                    let hash = a.chain.hashes()[page - 1];
                    let saved = cache.attach(save.place, hash, a.owner, root);
                    self.stats.counters.checkpoints_saved += u64::from(saved.taken);
                    self.stats.counters.checkpoints_dropped += u64::from(saved.dropped);
                    a.checkpoint_at = at;
                }
                let (last, mut from) = (a.checkpoint_at, part.from.max(a.checkpoint_at));
                while let Some(at) = cache.capture(a.prompt_len, &a.boundaries, last, from, reached)
                {
                    if at < reached {
                        // Due inside the rows: a save planned there that found no place is
                        // passed over; anything else ends the search, as a step that was not
                        // cut there takes no checkpoint at its end.
                        if !self.inner_due.contains(&(part.slot, at)) {
                            break;
                        }
                        from = at;
                        continue;
                    }
                    let page = (at / cells) as usize;
                    let root = cache.at_root(&a.boundaries, at);
                    let hash = a.chain.hashes()[page - 1];
                    let saved = cache.save(part.slot, hash, a.owner, root);
                    self.stats.counters.checkpoints_saved += u64::from(saved.taken);
                    self.stats.counters.checkpoints_dropped += u64::from(saved.dropped);
                    a.checkpoint_at = at;
                    break;
                }
            }
        }
        // A place taken for a checkpoint inside the rows that the commit did not name.
        let left = std::mem::take(&mut self.inner);
        self.release_inner(left);
    }

    /// Sends the text of the step to the sinks, ends the requests that are
    /// done and the ones whose sink gave up.
    pub(super) fn deliver(&mut self) {
        for i in 0..self.parts.len() {
            let slot = self.parts[i].slot;
            let a = self.slots[slot as usize].as_mut().expect("running");
            // A rule that failed in the middle of the reply: the picks of
            // this step were made without it, and the text is not sent.
            if let Some(message) = a.sampler.failure() {
                a.emitted.clear();
                let _ = a.sink.send(Event::Error(EngineError::Internal(message)));
                self.retire(slot, FinishReason::Cancelled);
                continue;
            }
            // How far the prompt has got comes first: the step that ends it
            // reports it whole before its first text.
            let mut listening = match self.parts[i].kind {
                PartKind::Prefill { .. } => a.report_progress(0.0),
                _ => true,
            };
            // The probabilities of the tokens whose text goes out now go out
            // first (a token that prints nothing goes out with the text
            // before it).
            if listening && a.sampler.wants_probs() {
                let probs = a.sampler.text_sent(a.emitted.len());
                if !probs.is_empty() {
                    listening = a.sink.send(Event::Logprobs(probs));
                }
            }
            if listening && !a.emitted.is_empty() {
                listening = a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            match a.finished {
                Some(finish) => {
                    if listening {
                        a.sink.send(Event::Done { finish, usage: a.usage() });
                    }
                    self.retire(slot, finish);
                }
                // The sink refused: nobody wants the rest.
                None if !listening => self.retire(slot, FinishReason::Cancelled),
                None => {}
            }
        }
    }
}
