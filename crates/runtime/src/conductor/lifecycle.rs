//! A request's way through the conductor: it is queued, starts in a slot, gives
//! way and comes back, and ends.

use super::*;

impl Conductor {
    /// A name for the admission accounting that no request the engine holds
    /// has.
    pub(super) fn fresh_seq(&mut self) -> SeqId {
        loop {
            self.next_seq = self.next_seq.wrapping_add(1);
            let seq = SeqId(self.next_seq);
            let taken = |a: &Active| a.seq == seq;
            if !self.slots.iter().flatten().any(taken) && !self.waiting.iter().any(taken) {
                return seq;
            }
        }
    }

    /// Puts a new request in line. It starts at the next pass of the
    /// admission accounting in which it fits.
    pub(super) fn enqueue(&mut self, submission: Submission) {
        let Submission {
            id,
            prompt,
            boundaries,
            images,
            max_tokens,
            stop,
            sampling,
            draft,
            filter,
            priority,
            displace,
            sink,
        } = submission;
        let refuse = |error: EngineError| {
            let _ = sink.send(Event::Error(error));
        };
        if let Some(why) = &self.failed {
            return refuse(EngineError::Internal(why.clone()));
        }
        if self.by_id.contains_key(&id) {
            return refuse(EngineError::InvalidRequest(format!("request id {id} is in use")));
        }
        if !images.is_empty() && self.vision.is_none() {
            return refuse(EngineError::InvalidRequest("the model reads no images".into()));
        }
        if !images.is_empty() && self.stage_rows == 0 {
            return refuse(EngineError::Internal(
                "the executor has no image stage for the rows of images".into(),
            ));
        }
        if displace {
            // The line is full: the worst request in it of a worse priority
            // that has not started (the latest of them) is refused in this
            // one's place. A request that ran has sent its client text, and
            // stays.
            let worst = self
                .waiting
                .iter()
                .enumerate()
                .filter(|(_, a)| !a.ran && a.priority > priority)
                .max_by_key(|(_, a)| (a.priority, a.stop.submitted()))
                .map(|(i, _)| i);
            let Some(i) = worst else { return refuse(EngineError::Overloaded) };
            let _ = self.waiting[i].sink.send(Event::Error(EngineError::Overloaded));
            self.dismiss(i, FinishReason::Cancelled);
            self.stats.counters.requests_displaced += 1;
        }
        // The names of the cells: an image's by the image and its row.
        let mut names: Vec<u64> = prompt.iter().map(|t| u64::from(*t)).collect();
        for span in &images {
            for j in 0..span.image.rows() {
                if let Some(name) = names.get_mut((span.first + j) as usize) {
                    *name = cell_name(&span.image.name, j);
                }
            }
        }
        let spans = Spans::new(images);
        // A request without a seed gets one of its own.
        let seed = (self.clock.now_ns() ^ id.0.rotate_left(32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let sampler = match RequestSampler::new(
            &sampling,
            &*self.model,
            &mut self.sampling_context,
            seed,
            filter,
        ) {
            Ok(sampler) => sampler,
            Err(error) => return refuse(error),
        };
        let prompt_len = prompt.len() as u32;
        let seq = self.fresh_seq();
        let room = self.context_length.saturating_sub(prompt_len);
        // What the cache has for the prompt now. A request whose first
        // missing page another sequence is computing is held back.
        let (chain, cached, idle, waits_for, brought_back) = match &mut self.cache {
            Some(cache) => {
                let chain = cache.chain(&names);
                // What RAM has of it comes back now, before the step is
                // planned, so that the accounting sees the pages it took.
                let probe = cache.probe(&mut self.pool, Some(&mut *self.exec), &chain, &names);
                let waits = probe.found.computing.map(|(hash, _)| hash);
                (chain, probe.resume.cached_cells as u32, probe.idle, waits, probe.brought > 0)
            }
            None => {
                (Chain::new(ChainHash::root(&[]), self.config.cells_per_page), 0, 0, None, false)
            }
        };
        let arrival = Arrival {
            seq,
            prompt_cells: prompt_len,
            cached_cells: cached,
            idle_cached_pages: idle,
            max_new_tokens: max_tokens.map_or(room, |max| max.min(room)),
            priority,
        };
        if self.admission.enqueue(arrival).is_err() {
            // The one refusal: the whole pool cannot hold the prompt and the
            // first token after it.
            return refuse(EngineError::ContextOverflow {
                prompt_tokens: prompt_len,
                limit: self.pool.capacity().saturating_mul(self.pool.cells_per_page()),
            });
        }
        let mut states = Vec::with_capacity(prompt.len() + 64);
        let mut state = self.spec.start();
        for i in 0..prompt.len() {
            state = self.spec.advance(state, &prompt[..=i]);
            states.push(state);
        }
        let mut tokens = Vec::with_capacity(prompt.len() + 64);
        tokens.extend_from_slice(&prompt);
        if waits_for.is_some() {
            self.admission.hold(seq);
            self.stats.counters.requests_held += 1;
        }
        self.waiting.push(Active {
            id,
            seq,
            sink,
            cells: Sequence::new(),
            prompt_len,
            boundaries,
            known: prompt_len,
            chain,
            owner: Owner(u64::from(seq.0)),
            ran: false,
            replay_end: 0,
            replay_left: 0,
            checkpoint_at: 0,
            waits_for,
            was_held: waits_for.is_some(),
            brought_back,
            looked_at: self.cache.as_ref().map_or(0, |cache| cache.evictions),
            cached_prompt: 0,
            replayed: 0,
            computed: 0,
            tokens,
            names,
            spans,
            states,
            drafter: NgramDrafter::new(),
            model_acceptance: (0..self.drafters.len())
                .map(|_| Acceptance::new(self.config.window_rows as usize, 0.95))
                .collect(),
            draft_allowed: draft,
            sampler,
            stop,
            max_tokens,
            completion: 0,
            waited_steps: 0,
            priority,
            counted_waiting: false,
            draft: Vec::with_capacity(self.config.window_rows as usize),
            model_drafter: None,
            ngram: Vec::with_capacity(self.config.window_rows as usize),
            ngram_chances: Vec::with_capacity(self.config.window_rows as usize),
            offered: (0..self.drafters.len()).map(|_| Vec::new()).collect(),
            model_chances: Vec::with_capacity(self.config.window_rows as usize),
            model_needs: Vec::with_capacity(self.config.window_rows as usize),
            model_probs: Vec::with_capacity(self.config.window_rows as usize),
            pending: Vec::new(),
            emitted: String::new(),
            finished: None,
        });
        self.by_id.insert(id, Place::Waiting);
    }

    /// Gives the request with the name `seq` a slot on the devices: it runs
    /// from the next plan on.
    pub(super) fn start(&mut self, seq: SeqId) {
        let Some(i) = self.waiting.iter().position(|a| a.seq == seq) else { return };
        // What the cache has for it now, which may be more or less than when
        // it was put in line.
        let probe = match &mut self.cache {
            Some(cache) => {
                let a = &self.waiting[i];
                Some(cache.probe(&mut self.pool, None, &a.chain, &a.names[..a.known as usize]))
            }
            None => None,
        };
        if let Some(probe) = &probe
            && let Some((hash, _)) = probe.found.computing
            && !self.waiting[i].ran
        {
            // A sequence that started a moment ago computes the page this
            // prompt misses first. The accounting let the request in, and
            // only sequences that run announce pages, so it goes back in
            // line and waits for the page.
            let a = &mut self.waiting[i];
            a.waits_for = Some(hash);
            if !a.was_held {
                a.was_held = true;
                self.stats.counters.requests_held += 1;
            }
            let arrival = Arrival {
                seq,
                prompt_cells: a.prompt_len,
                cached_cells: probe.resume.cached_cells as u32,
                idle_cached_pages: probe.idle,
                max_new_tokens: a.max_new_tokens(self.context_length),
                priority: a.priority,
            };
            self.admission.remove(seq);
            // It passed the one refusal when it arrived.
            let _ = self.admission.enqueue(arrival);
            self.admission.hold(seq);
            return;
        }
        let Some(slot) = self.free_slots.pop() else {
            // The accounting never starts more requests than there are
            // slots, so this is a bookkeeping bug and not load.
            let a = self.waiting.swap_remove(i);
            let _ = a.sink.send(Event::Error(EngineError::Internal(
                "a request was started without a free slot".into(),
            )));
            self.by_id.remove(&a.id);
            self.admission.remove(seq);
            return;
        };
        let mut a = self.waiting.swap_remove(i);
        a.stop.started();
        if let (Some(cache), Some(probe)) = (&mut self.cache, probe) {
            let counters = &mut self.stats.counters;
            let Resume { cached_cells: cached, restore, replay, .. } = probe.resume;
            a.cells = Sequence::from_shared(&mut self.pool, &probe.found.table(), cached);
            a.chain = cache.chain_after(&probe.found, &a.names[..a.known as usize]);
            // The pages this sequence will fill are its to announce.
            cache.index.announce(a.owner, a.chain.unpublished());
            a.replay_end = cached as u32;
            a.replay_left = (replay.end - replay.start) as u32;
            a.checkpoint_at = 0;
            if let Some(id) = restore {
                cache.restore(slot, id);
                a.checkpoint_at = replay.start;
                counters.checkpoints_restored += 1;
            }
            if cached > 0 {
                counters.prefix_hits += 1;
                counters.cached_cells += cached;
            }
            if !a.ran {
                a.cached_prompt = cached.min(u64::from(a.prompt_len)) as u32;
            }
        }
        a.ran = true;
        a.waits_for = None;
        self.by_id.insert(a.id, Place::Slot(slot));
        self.slots[slot as usize] = Some(a);
        self.active += 1;
    }

    /// Lets what a sequence computed stay in the prefix cache, as it ends,
    /// gives way or is cancelled: the pages that are full and not published
    /// yet, the partly used last page as a tail, and the live state as the
    /// checkpoint of that tail (or of the last full page, when the cells end
    /// on a page boundary), unless with one place it would push out the
    /// request's own anchor ([`Cache::save_end`]). The copy of the state runs
    /// with the next record, before anything of it writes a slot.
    pub(super) fn keep(&mut self, a: &mut Active, slot: u32) {
        let Some(cache) = &mut self.cache else { return };
        cache.index.withdraw(a.owner);
        // A sequence that leaves before the record that restores its state
        // never had that state in its slot.
        let restored = cache.cancel_restore(slot);
        let cells = a.cells.len();
        let per_page = u64::from(self.pool.cells_per_page());
        cache.index.publish(
            &mut self.pool,
            &mut a.cells,
            &mut a.chain,
            &a.names,
            cells,
            Priority::NORMAL,
        );
        cache.note(a.owner, &a.chain, cells, true);
        let full = (cells / per_page) as usize;
        let rest = &a.names[full * per_page as usize..cells as usize];
        let target = if rest.is_empty() {
            full.checked_sub(1).map(|page| a.chain.hashes()[page])
        } else {
            let prior = a.chain.before(full);
            let page = a.cells.pages()[full];
            if let Some(evicted) =
                cache.index.insert_tail(&mut self.pool, prior, rest, page, Priority::NORMAL)
            {
                self.stats.counters.checkpoints_dropped += u64::from(cache.forget(evicted));
            }
            Some(PrefixIndex::tail_hash(prior, rest))
        };
        if let Some(hash) = target
            && !restored
            && a.state_pos() == cells
        {
            let saved = cache.save_end(slot, hash, a.owner);
            self.stats.counters.checkpoints_saved += u64::from(saved.taken);
            self.stats.counters.checkpoints_dropped += u64::from(saved.dropped);
        }
    }

    /// Takes the running request with the name `seq` off the devices: it
    /// lets go of its pages and its slot and waits at the front of the line.
    /// What it computed stays in the prefix cache (when the model has one), so
    /// it comes back as a prefix hit; without one it computes everything it has
    /// read and written again, as a prompt made of all its tokens.
    pub(super) fn suspend(&mut self, seq: SeqId) {
        let Some(slot) = self.slots.iter().position(|a| a.as_ref().is_some_and(|a| a.seq == seq))
        else {
            return;
        };
        let Some(mut a) = self.slots[slot].take() else { return };
        self.keep(&mut a, slot as u32);
        std::mem::take(&mut a.cells).release(&mut self.pool);
        a.known = a.tokens.len() as u32;
        a.replay_left = 0;
        // What it computed may leave the devices while it waits: it may
        // bring it back from RAM once more.
        a.brought_back = false;
        a.draft.clear();
        a.ngram.clear();
        a.model_drafter = None;
        a.waited_steps = 0;
        // What it needs to come back: the rows the cache does not hold, which
        // are claimed again step by step, as a prompt's are.
        let (rows, idle) = match &mut self.cache {
            Some(cache) => {
                a.looked_at = cache.evictions;
                let names = &a.names[..a.known as usize];
                let probe = cache.probe(&mut self.pool, None, &a.chain, names);
                (
                    probe.resume.compute.end.saturating_sub(probe.resume.compute.start) as u32,
                    probe.idle,
                )
            }
            None => (a.known, 0),
        };
        self.admission.set_restore(seq, 0, idle);
        self.admission.set_recompute(seq, rows);
        self.by_id.insert(a.id, Place::Waiting);
        self.waiting.push(a);
        self.free_slots.push(slot as u32);
        self.active -= 1;
        self.stats.counters.requests_suspended += 1;
    }

    /// Ends the request in `slot`: tells its sink (if it still listens),
    /// leaves what it computed in the prefix cache, gives its pages and its
    /// slot back.
    pub(super) fn retire(&mut self, slot: u32, finish: FinishReason) {
        self.retire_with(slot, finish, true);
    }

    /// [`Conductor::retire`], with or without leaving the request's pages and
    /// state in the cache: not when the engine is going down or a device
    /// failed, and the state is not there.
    pub(super) fn retire_with(&mut self, slot: u32, finish: FinishReason, keep: bool) {
        // A sequence in the middle of a paused step has a state that is neither
        // before nor after its rows: nothing of it is kept, and the model is told
        // to give the step up.
        let keep = match self.paused {
            Some(p) if p.slot == slot => {
                self.paused = None;
                self.paused_images.clear();
                self.give_up = true;
                let saves = std::mem::take(&mut self.paused_inner);
                self.release_inner(saves);
                false
            }
            _ => keep,
        };
        let Some(mut a) = self.slots[slot as usize].take() else { return };
        self.by_id.remove(&a.id);
        if !a.sink.is_closed() {
            if !a.emitted.is_empty() {
                a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            a.sink.send(Event::Done { finish, usage: a.usage() });
        }
        if keep {
            self.keep(&mut a, slot);
        } else if let Some(cache) = &mut self.cache {
            cache.index.withdraw(a.owner);
        }
        self.stats.report(RequestReport {
            id: a.id,
            prompt_tokens: a.prompt_len,
            cached_tokens: a.cached_prompt,
            replayed_rows: a.replayed,
            computed_rows: a.computed,
        });
        a.cells.release(&mut self.pool);
        self.free_slots.push(slot);
        self.active -= 1;
        match finish {
            FinishReason::Cancelled => {
                self.admission.remove(a.seq);
                self.stats.counters.requests_cancelled += 1;
            }
            _ => {
                // What it produced of what it was allowed teaches the
                // accounting what to expect of the others.
                self.admission.finished(a.seq);
                self.stats.counters.requests_finished += 1;
            }
        }
    }

    /// Ends the waiting request at index `i` of the line.
    pub(super) fn dismiss(&mut self, i: usize, finish: FinishReason) {
        let mut a = self.waiting.swap_remove(i);
        self.by_id.remove(&a.id);
        self.admission.remove(a.seq);
        if !a.sink.is_closed() {
            if !a.emitted.is_empty() {
                a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            a.sink.send(Event::Done { finish, usage: a.usage() });
        }
        a.cells.release(&mut self.pool);
        self.stats.counters.requests_cancelled += 1;
    }

    pub(super) fn finish_everything(&mut self, finish: FinishReason) {
        for slot in 0..self.slots.len() as u32 {
            self.retire_with(slot, finish, false);
        }
        while !self.waiting.is_empty() {
            self.dismiss(self.waiting.len() - 1, finish);
        }
    }

    /// Ends every request with an error, after the devices failed. The
    /// engine accepts no more requests.
    pub(super) fn fail(&mut self, message: String) {
        let error = EngineError::Internal(message.clone());
        for slot in 0..self.slots.len() as u32 {
            if let Some(a) = &self.slots[slot as usize] {
                let _ = a.sink.send(Event::Error(error.clone()));
                self.retire_with(slot, FinishReason::Cancelled, false);
            }
        }
        while let Some(a) = self.waiting.last() {
            let _ = a.sink.send(Event::Error(error.clone()));
            self.dismiss(self.waiting.len() - 1, FinishReason::Cancelled);
        }
        let _ = self.failure.set(message.clone());
        self.failed = Some(message);
    }
}
