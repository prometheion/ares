//! The cache's side of the level on SSD: the pump's rounds that bring runs
//! back from the drive and write what leaves RAM, and the drain at the end.

use super::*;

impl Cache {
    /// Whether anything waits to move or is moving: the conductor comes back
    /// to [`Cache::pump`] soon while it does.
    pub(crate) fn busy(&self) -> bool {
        !self.leaving.is_empty()
            || !self.states_out.is_empty()
            || self.ssd.as_ref().is_some_and(Ssd::busy)
    }

    /// What the level on SSD has done, when there is one.
    pub(crate) fn ssd_counters(&mut self) -> SsdCounters {
        self.collect();
        self.ssd.as_ref().map_or_else(SsdCounters::default, Ssd::counters)
    }

    /// A buffer for a checkpoint on its way out of its place, which is added
    /// to `batch`: when the batch so far holds every one, it is carried out
    /// first, and when the writer holds every one, the loop waits for it
    /// briefly. This is the one place where the step loop waits for the
    /// drive, besides a page that would be lost otherwise.
    pub(in crate::cache) fn state_room(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
    ) -> Option<Vec<TransitPart>> {
        let begun = Instant::now();
        let patience = if self.draining { DRAIN_LIMIT } else { PATIENCE };
        loop {
            if let Some(parts) = self.ssd.as_mut()?.state_buffer() {
                return Some(parts);
            }
            if notes.iter().any(|note| matches!(note, Note::StateOut(_) | Note::StateIn { .. })) {
                self.carry_out(pool, exec, std::mem::take(batch), std::mem::take(notes));
                continue;
            }
            if begun.elapsed() >= patience || !self.wait_for_writer() {
                return None;
            }
        }
    }

    /// A checkpoint that was due on the drive found no buffer: it is not
    /// written.
    pub(in crate::cache) fn state_not_written(&mut self, out: StateOut) {
        if let Some(ssd) = &mut self.ssd {
            ssd.unqueue(Kind::State, out.hash);
            ssd.counters.states_skipped += 1;
        }
    }

    /// Waits a moment for the writer to finish a record, which gives its
    /// buffer back: [`PATIENCE`] while the engine runs, as long as the drive
    /// takes when it ends. `false` when it holds none or does not finish one
    /// in time.
    pub(in crate::cache) fn wait_for_writer(&mut self) -> bool {
        let patience = if self.draining { DRAIN_LIMIT } else { PATIENCE };
        let Some(ssd) = &mut self.ssd else { return false };
        if ssd.writing == 0 {
            return false;
        }
        match ssd.from_writer.recv_timeout(patience) {
            Ok(done) => {
                self.written(done);
                true
            }
            Err(_) => false,
        }
    }

    /// The writer has finished a record: a page's buffers go back to its
    /// entry in RAM, which the drive has now too, and a checkpoint's buffer
    /// to the others.
    pub(super) fn written(&mut self, done: WriteDone) {
        let Some(ssd) = &mut self.ssd else { return };
        ssd.writing -= 1;
        ssd.queued.remove(&(done.kind, done.hash));
        ssd.counters.write_ns += done.ns;
        match (done.bytes, done.kind) {
            (None, _) => ssd.counters.write_errors += 1,
            (Some(0), _) => {}
            (Some(bytes), Kind::Page) => {
                ssd.counters.pages_written += 1;
                ssd.counters.bytes_written += bytes;
            }
            (Some(bytes), Kind::State) => {
                ssd.counters.states_written += 1;
                ssd.counters.bytes_written += bytes;
            }
        }
        match done.kind {
            Kind::State => ssd.free_states.push(done.parts),
            Kind::Page => {
                let Some(ram) = &mut self.in_ram else { return };
                let ok = done.bytes.is_some();
                let away = ram.get(done.hash).is_some_and(|entry| entry.parts.is_none());
                if !away {
                    ram.recycle(done.parts);
                    return;
                }
                ram.give(done.hash, done.parts);
                // A page that cannot be written is not tried again.
                ram.update(done.hash, |entry| match ok {
                    true => entry.stored = true,
                    false => entry.worth = false,
                });
            }
        }
    }

    /// Takes what the writer and the readers have finished.
    pub(in crate::cache) fn collect(&mut self) {
        loop {
            let Some(ssd) = &mut self.ssd else { return };
            let Ok(done) = ssd.from_writer.try_recv() else { break };
            self.written(done);
        }
        loop {
            let Some(ssd) = &mut self.ssd else { return };
            let Ok(done) = ssd.from_readers.try_recv() else { break };
            ssd.counters.read_ns += done.ns;
            let Some(job) = ssd.jobs.iter_mut().find(|job| job.id == done.job) else {
                self.recycle(done.parts, done.index == STATE);
                continue;
            };
            job.outstanding -= 1;
            let state = done.index == STATE;
            if done.ok && !job.failed {
                ssd.counters.bytes_read += done.bytes;
                if state {
                    ssd.counters.states_read += 1;
                    job.state_parts = Some(done.parts);
                } else {
                    ssd.counters.pages_read += 1;
                    job.ready.insert(done.index, done.parts);
                }
                continue;
            }
            // A record that is damaged or gone: the store forgets it, and the
            // run ends before it.
            if !done.ok {
                let (kind, at) = if state {
                    (Kind::State, job.state.unwrap_or(0))
                } else {
                    (Kind::Page, done.index)
                };
                ssd.shared.store().forget(kind, job.pages[at].0);
            }
            job.failed = true;
            self.recycle(done.parts, state);
        }
    }

    /// A buffer that came back from a read nobody needs.
    pub(in crate::cache) fn recycle(&mut self, parts: Vec<TransitPart>, state: bool) {
        match (state, &mut self.ssd, &mut self.in_ram) {
            (true, Some(ssd), _) => ssd.free_states.push(parts),
            (false, _, Some(ram)) => ram.recycle(parts),
            _ => {}
        }
    }

    /// Hands the readers what the runs need next, as far as buffers go and
    /// up to [`READ_AHEAD`] pages in all.
    pub(in crate::cache) fn ask(&mut self) {
        let Some(readers) = self.ssd.as_ref().and_then(|ssd| ssd.to_readers.clone()) else {
            return;
        };
        let reading = |ssd: &Ssd| -> usize {
            ssd.jobs.iter().map(|job| job.outstanding + job.ready.len()).sum()
        };
        let jobs = self.ssd.as_ref().map_or(0, |ssd| ssd.jobs.len());
        for at in 0..jobs {
            let ssd = self.ssd.as_mut().expect("checked above");
            let job = &ssd.jobs[at];
            if job.failed {
                continue;
            }
            if let (Some(state), false) = (job.state, job.state_asked) {
                let hash = job.pages[state].0;
                let located = ssd.shared.store().locate(Kind::State, hash);
                match (located, ssd.state_buffer()) {
                    (Some(located), Some(parts)) => {
                        let job = &mut ssd.jobs[at];
                        job.state_asked = true;
                        job.outstanding += 1;
                        let _ = readers.send(ReadJob { job: job.id, index: STATE, located, parts });
                    }
                    // The limit took the checkpoint since the run was planned.
                    (None, parts) => {
                        ssd.free_states.extend(parts);
                        ssd.jobs[at].failed = true;
                        continue;
                    }
                    // No buffer now: the pages wait for the checkpoint.
                    (Some(_), None) => {}
                }
            }
            loop {
                let ssd = self.ssd.as_mut().expect("checked above");
                let job = &ssd.jobs[at];
                if job.asked >= job.pages.len() || job.failed {
                    break;
                }
                if reading(ssd) >= READ_AHEAD {
                    return;
                }
                let (index, hash) = (job.asked, job.pages[job.asked].0);
                let Some(parts) = self.ram_buffer() else { return };
                let ssd = self.ssd.as_mut().expect("checked above");
                let located = ssd.shared.store().locate(Kind::Page, hash);
                let Some(located) = located else {
                    // The limit took the page since the run was planned.
                    ssd.jobs[at].failed = true;
                    if let Some(ram) = &mut self.in_ram {
                        ram.recycle(parts);
                    }
                    break;
                };
                let job = &mut ssd.jobs[at];
                job.asked += 1;
                job.outstanding += 1;
                let _ = readers.send(ReadJob { job: job.id, index, located, parts });
            }
        }
    }

    /// Moves what is due between the devices, the host and the drive:
    /// what left the devices into RAM, checkpoints that leave RAM to the
    /// drive, and the pages and the checkpoint of the runs that come back
    /// from the drive, which enter the index here. Then RAM writes what only
    /// it holds when it is too full. Called between steps.
    pub(crate) fn pump(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        self.collect();
        match self.ssd.is_some() {
            true => self.round(pool, exec),
            false => self.settle_leaving(pool, exec),
        }
        self.clean_ahead();
        self.settle_leaving(pool, exec);
    }

    /// One round of [`Cache::pump`] with a level on SSD: what the readers
    /// need next, one batch of transfers, and what it came to.
    pub(in crate::cache) fn round(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        self.ask();
        let per_page = u64::from(self.cells_per_page);
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        // What left the devices and what leaves RAM, first: the pages of the
        // pool and the places they leave may be taken by what comes back.
        self.push_leaving(pool, exec, &mut batch, &mut notes);
        let settle = u64::from(self.durable.as_ref().map_or(0, |durable| durable.settle_cells));
        let jobs = self.ssd.as_ref().map_or(0, |ssd| ssd.jobs.len());
        for at in 0..jobs {
            loop {
                let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                let job = &mut ssd.jobs[at];
                let in_batch =
                    |note: &&Note| matches!(note, Note::PageIn { job: id, .. } if *id == job.id);
                let next = job.done + notes.iter().filter(in_batch).count();
                if job.failed || !job.ready.contains_key(&next) {
                    break;
                }
                let with_state = job.state == Some(next);
                if with_state && job.state_parts.is_none() {
                    break;
                }
                let mut held = std::mem::take(&mut job.held);
                let grown = self.take_page(pool, exec, &mut held, per_page, &mut batch, &mut notes);
                let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                let job = &mut ssd.jobs[at];
                job.held = held;
                if !grown {
                    job.failed = true;
                    break;
                }
                let page = *job.held.pages().last().expect("a page was appended");
                let id = job.id;
                if with_state {
                    let parts = job.state_parts.take().expect("checked above");
                    match self.new_place().0 {
                        Some(place) => {
                            // The checkpoint that had the place leaves it
                            // for the drive before this one comes in.
                            self.push_leaving(pool, exec, &mut batch, &mut notes);
                            batch.push(Transfer {
                                kind: TransferKind::StateIn,
                                object: place,
                                keep_form: false,
                                tail: false,
                                parts,
                                ok: false,
                            });
                            notes.push(Note::StateIn { job: id });
                        }
                        // No place for it: the run ends before this page.
                        None => {
                            let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                            ssd.free_states.push(parts);
                            ssd.jobs[at].failed = true;
                            break;
                        }
                    }
                }
                let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                let job = &mut ssd.jobs[at];
                let parts = job.ready.remove(&next).expect("checked above");
                let depth = job.pages[next].2;
                let behind = (job.pages.len() - 1 - next) as u64 * per_page;
                let keep_form = keeps_form(depth, behind, settle, per_page);
                batch.push(Transfer {
                    kind: TransferKind::PageIn,
                    object: page.0,
                    keep_form,
                    tail: false,
                    parts,
                    ok: false,
                });
                notes.push(Note::PageIn { job: id, index: next });
            }
        }
        self.carry_out(pool, exec, batch, notes);

        // Runs that are whole, or cut: the index learns that nobody computes
        // the rest, and the pages are the index's alone.
        let mut at = 0;
        loop {
            let Some(ssd) = &mut self.ssd else { return };
            let Some(job) = ssd.jobs.get(at) else { break };
            if !job.over() || job.outstanding > 0 {
                at += 1;
                continue;
            }
            let mut job = ssd.jobs.remove(at).expect("it is there");
            ssd.counters.imports_cut += u64::from(job.done < job.pages.len());
            self.index.withdraw(job.owner);
            std::mem::take(&mut job.held).release(pool);
            for parts in std::mem::take(&mut job.ready).into_values() {
                self.recycle(parts, false);
            }
            if let Some(parts) = job.state_parts.take() {
                self.recycle(parts, true);
            }
        }
    }

    /// A page of a run from the drive came into pool page `object` (`ok`),
    /// with the checkpoint of the run when `state` says one came with it: it
    /// enters the index, and its buffer stays in RAM as a copy.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::cache) fn page_in(
        &mut self,
        pool: &mut PagePool,
        id: u64,
        index: usize,
        object: u32,
        parts: Vec<TransitPart>,
        ok: bool,
        state: Option<(u64, u32, bool)>,
    ) {
        let Some(ssd) = &mut self.ssd else { return };
        let Some(job) = ssd.jobs.iter_mut().find(|job| job.id == id) else {
            self.recycle(parts, false);
            return;
        };
        let (hash, parent, depth) = job.pages[index];
        let with_state = job.state == Some(index);
        let came = ok && !job.failed;
        if came {
            self.index.insert(pool, parent, hash, PageId(object));
            job.done += 1;
        } else {
            job.failed = true;
        }
        let owner = job.owner;
        if let (true, Some((_, place, state_ok))) = (with_state, state)
            && !(came && state_ok && self.hang(hash, place, owner, false))
        {
            self.free(place);
        }
        match (came, &mut self.in_ram) {
            (true, Some(ram)) if ram.get(hash).is_none() => {
                let mut entry = Entry::new(parent, depth, None);
                entry.on_cards = true;
                entry.stored = true;
                entry.worth = true;
                ram.insert(hash, entry, parts);
            }
            _ => self.recycle(parts, false),
        }
    }

    /// Writes everything worth writing that the drive does not have, when
    /// the engine ends: the checkpoints first (a few large records, which
    /// pass through a few buffers), then the pages, what only RAM holds the
    /// coldest first, then the pages of the devices. Waits for the writer,
    /// at most for [`DRAIN_LIMIT`], and stops the threads.
    pub(crate) fn drain(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        self.settle_leaving(pool, exec);
        let Some(ssd) = &mut self.ssd else { return };
        // Runs that come back are for requests that are gone.
        for job in &mut ssd.jobs {
            job.failed = true;
        }
        self.draining = true;
        let begun = Instant::now();
        let order = self.in_ram.as_ref().map_or_else(Vec::new, |ram| ram.in_order());
        let cards = self.index.cached_pages();
        let left_behind = self.left_behind();
        // Checkpoints: of RAM, and of the devices' pages.
        for hash in &order {
            let due = self.in_ram.as_ref().and_then(|ram| ram.get(*hash)).is_some_and(|entry| {
                entry.parts.is_some() && entry.checkpoint.is_some_and(|h| h.worth && !h.stored)
            });
            if due {
                self.clean(Item::Checkpoint, *hash);
                self.settle_leaving(pool, exec);
            }
        }
        for page in &cards {
            if let Some(id) = page.checkpoint
                && !left_behind.contains(&page.hash)
                && let Some(mut out) =
                    self.state_out(id.0 as u32, page.hash, page.parent, page.depth)
            {
                out.keep = true;
                self.ssd.as_mut().expect("checked above").queue(Kind::State, page.hash);
                self.states_out.push(out);
                self.settle_leaving(pool, exec);
            }
        }
        // Pages: what only RAM holds, from its buffers.
        for hash in &order {
            let due = self.in_ram.as_ref().and_then(|ram| ram.get(*hash)).is_some_and(|entry| {
                entry.worth && !entry.stored && entry.tail.is_none() && entry.parts.is_some()
            });
            if due {
                self.clean(Item::Page, *hash);
            }
        }
        // And the devices' pages, through buffers of RAM, a batch at a time.
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        for page in &cards {
            let worth = self.worth.contains(&page.hash) || self.deep(page.depth);
            let ssd = self.ssd.as_ref().expect("checked above");
            let in_ram = self.in_ram.as_ref().is_some_and(|ram| ram.get(page.hash).is_some());
            if !worth || in_ram || ssd.has_or_queued(Kind::Page, page.hash) {
                continue;
            }
            let room = self.page_room(pool, exec, &mut batch, &mut notes, true);
            let Some(parts) = room else { break };
            self.ssd.as_mut().expect("checked above").queue(Kind::Page, page.hash);
            batch.push(Transfer {
                kind: TransferKind::PageOut,
                object: page.page.0,
                keep_form: false,
                tail: false,
                parts,
                ok: false,
            });
            notes.push(Note::CardsOut { hash: page.hash, parent: page.parent, depth: page.depth });
            if batch.len() as u64 >= RESERVE {
                self.carry_out(pool, exec, std::mem::take(&mut batch), std::mem::take(&mut notes));
            }
            if begun.elapsed() > DRAIN_LIMIT {
                break;
            }
        }
        self.carry_out(pool, exec, batch, notes);
        // The writer finishes what it has.
        while self.ssd.as_ref().is_some_and(|ssd| ssd.writing > 0) && begun.elapsed() < DRAIN_LIMIT
        {
            self.wait_for_writer();
        }
        let Some(ssd) = &mut self.ssd else { return };
        for job in &mut ssd.jobs {
            std::mem::take(&mut job.held).release(pool);
        }
        ssd.jobs.clear();
        ssd.close();
    }
}
