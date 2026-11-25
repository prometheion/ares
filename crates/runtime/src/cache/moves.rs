//! What moves between the devices, RAM and the drive: pages that leave the
//! devices, the room RAM keeps, the trim to the allowance, a run that comes
//! back from RAM, and the batches of transfers with what they came to.

use super::*;

impl Cache {
    /// A page or tail that the index gave up outside of [`Cache::evict`]
    /// (one tail too many) leaves the devices. Returns whether a checkpoint
    /// went with it, which happens only without RAM below the devices.
    pub(crate) fn forget(&mut self, evicted: Evicted) -> bool {
        self.leave_cards(evicted, &HashSet::new())
    }

    /// Gives up `pages` cached pages from the ends of chains. They go to
    /// RAM, with their checkpoints, before the next record.
    pub(crate) fn evict(&mut self, pool: &mut PagePool, pages: u32) -> Evictions {
        let left_behind = self.left_behind();
        let mut given = Evictions::default();
        for _ in 0..pages {
            let Some(evicted) = self.index.evict(pool) else { break };
            given.pages += 1;
            given.checkpoints += u32::from(self.leave_cards(evicted, &left_behind));
        }
        if given.pages > 0 {
            self.evictions += 1;
        }
        given
    }

    /// What the index gave up leaves the devices: into RAM, where a copy of
    /// it may lie already. Returns whether a checkpoint went with it, which
    /// happens only without RAM below the devices.
    pub(super) fn leave_cards(
        &mut self,
        evicted: Evicted,
        left_behind: &HashSet<ChainHash>,
    ) -> bool {
        let Evicted { page, checkpoint, root, hash, parent, depth, tail, .. } = evicted;
        let full = tail.is_none();
        let deep = full && self.deep(depth);
        let worth = full && (self.worth.remove(&hash) || deep);
        let stored = self.ssd.as_ref().is_some_and(|ssd| ssd.has_or_queued(Kind::State, hash));
        let held = checkpoint.map(|id| Held {
            place: id.0 as u32,
            worth: deep && !left_behind.contains(&hash),
            stored,
            root,
        });
        let Some(ram) = &mut self.in_ram else {
            // Nothing below the devices: the checkpoint goes with its page.
            if let Some(held) = held {
                self.give_up(held.place, None);
                return true;
            }
            return false;
        };
        match ram.get(hash) {
            // RAM has the page already: nothing to copy.
            Some(entry) => {
                let spare = held.filter(|_| entry.checkpoint.is_some());
                ram.left_the_cards(hash, |entry| {
                    entry.worth |= worth;
                    if entry.checkpoint.is_none() {
                        entry.checkpoint = held;
                    }
                });
                if let Some(held) = spare {
                    self.give_up(held.place, None);
                }
            }
            None => {
                let leaving = Leaving { page, hash, parent, depth, tail, checkpoint: held, worth };
                self.leaving.push(leaving);
            }
        }
        false
    }

    /// Tells the cache that the sequence of `owner` holds `settled` cells
    /// whose full pages are published under the names of `chain`, and with
    /// `last` that it ends here. Once it is long enough for the drive, all
    /// its pages are worth writing when they leave RAM.
    pub(crate) fn note(&mut self, owner: Owner, chain: &Chain, settled: u64, last: bool) {
        let Some(ssd) = &self.ssd else { return };
        if settled < ssd.min_cells() {
            if last {
                self.cursors.remove(&owner);
            }
            return;
        }
        let full = ((settled / u64::from(self.cells_per_page)) as usize).min(chain.hashes().len());
        let from = self.cursors.get(&owner).copied().unwrap_or(0).min(full);
        for hash in &chain.hashes()[from..full] {
            match &mut self.in_ram {
                Some(ram) if ram.get(*hash).is_some() => {
                    ram.update(*hash, |entry| entry.worth = true);
                }
                _ => {
                    self.worth.insert(*hash);
                }
            }
        }
        match last {
            true => self.cursors.remove(&owner),
            false => self.cursors.insert(owner, full),
        };
    }

    /// Moves what has left the devices into RAM, and checkpoints that leave
    /// RAM out of their places for the drive: before anything writes the
    /// pool or the places again. Call it before every record.
    pub(crate) fn settle_leaving(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        if self.leaving.is_empty() && self.states_out.is_empty() {
            return;
        }
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        self.push_leaving(pool, exec, &mut batch, &mut notes);
        self.carry_out(pool, exec, batch, notes);
    }

    /// Adds the copies of what is leaving to `batch`, in front of anything
    /// that is added after them. When what it needs is held by transfers of
    /// the batch, the batch so far is carried out first: the order of the
    /// transfers stays what it was.
    pub(super) fn push_leaving(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
    ) {
        for leaving in std::mem::take(&mut self.leaving) {
            // A page whose K and V lie in host memory ranks as prompt cache, the
            // lowest user of the RAM: it goes into RAM when there is room for it
            // and is dropped otherwise. Nothing waits for the drive to make room
            // for it (hundreds of them leave at once after a long request, a
            // wait for the writer each, while the next request waits).
            let precious = leaving.worth
                && !self.is_far(leaving.page)
                && self
                    .ssd
                    .as_ref()
                    .is_some_and(|ssd| !ssd.has_or_queued(Kind::Page, leaving.hash));
            match self.page_room(pool, exec, batch, notes, precious) {
                Some(parts) => {
                    batch.push(Transfer {
                        kind: TransferKind::PageOut,
                        object: leaving.page.0,
                        keep_form: false,
                        parts,
                        ok: false,
                    });
                    notes.push(Note::ToRam(leaving));
                }
                None => self.lost(leaving),
            }
        }
        // Checkpoints out of their places, for the drive. Making room for
        // the pages above may have added some.
        for out in std::mem::take(&mut self.states_out) {
            match self.state_room(pool, exec, batch, notes) {
                Some(parts) => {
                    batch.push(Transfer {
                        kind: TransferKind::StateOut,
                        object: out.place,
                        keep_form: false,
                        parts,
                        ok: false,
                    });
                    notes.push(Note::StateOut(out));
                }
                None => self.state_not_written(out),
            }
        }
    }

    /// A page that left the devices found no room in RAM: it is gone, and so
    /// is its checkpoint, which goes to the drive when it is due there.
    pub(super) fn lost(&mut self, leaving: Leaving) {
        if let Some(ram) = &mut self.in_ram
            && leaving.tail.is_none()
        {
            ram.counters.pages_dropped += 1;
        }
        if let Some(held) = leaving.checkpoint {
            let out = self.state_out_of(&held, leaving.hash, leaving.parent, leaving.depth);
            self.give_up(held.place, out);
        }
    }

    /// A buffer for a page in RAM: one that holds nothing, a new one when the
    /// allowance has room, or one that something which goes for nothing gave
    /// up. Never waits.
    pub(super) fn ram_buffer(&mut self) -> Option<Vec<crate::exec::TransitPart>> {
        let mut dropped = 0;
        loop {
            let ram = self.in_ram.as_mut()?;
            if let Some(parts) = ram.spare() {
                return Some(parts);
            }
            // The floor of pages is the level's whatever else the allowance
            // holds; and the room in the limit that saves of the next record
            // will lock is theirs.
            let (bytes, page) = (ram.bytes(), ram.page_bytes());
            let room = self.held() + page <= self.allowance() || bytes + page <= self.page_floor;
            let unbooked = self.books.as_ref().map_or(u64::MAX, PinnedRam::unbooked);
            if room
                && unbooked >= page + self.pending_locks()
                && let Ok(parts) = self.in_ram.as_mut()?.make()
            {
                return Some(parts);
            }
            if !self.give_up_free(&mut dropped) {
                return None;
            }
        }
    }

    /// [`Cache::ram_buffer`] for a page that is added to `batch`. A
    /// `precious` page (one that would be lost otherwise) waits for room when
    /// there is no other way: the buffers in the batch so far go into RAM
    /// first, and the writer writes what only RAM holds, briefly.
    pub(super) fn page_room(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
        precious: bool,
    ) -> Option<Vec<crate::exec::TransitPart>> {
        loop {
            if let Some(parts) = self.ram_buffer() {
                return Some(parts);
            }
            if !precious || self.ssd.is_none() {
                return None;
            }
            if !batch.is_empty() {
                self.carry_out(pool, exec, std::mem::take(batch), std::mem::take(notes));
                continue;
            }
            self.clean_ahead();
            if !self.wait_for_writer() {
                return None;
            }
        }
    }

    /// Gives what only RAM holds to the writer, the coldest first, until it
    /// is no more than the allowance less room for [`RESERVE`] pages and one
    /// checkpoint: what comes next then finds room without waiting for the
    /// drive.
    pub(super) fn clean_ahead(&mut self) {
        if self.ssd.is_none() {
            return;
        }
        let Some(ram) = &self.in_ram else { return };
        let reserve = RESERVE * ram.page_bytes() + self.bytes;
        let room = self.allowance().saturating_sub(reserve);
        // Every turn takes one thing out of the dirty part of the order.
        for _ in 0..=ram.len() * 2 {
            let Some(ram) = &self.in_ram else { return };
            let index = self.index.checkpoints() as u64 * self.bytes;
            if ram.dirty_bytes(self.bytes) + index <= room {
                return;
            }
            let Some((_, item, hash)) = ram.first_from(Class::Dirty) else { return };
            self.clean(item, hash);
        }
    }

    /// Writes the page or the checkpoint `hash` that only RAM holds, which
    /// stays in RAM: what the writer gives back is on the drive too.
    pub(super) fn clean(&mut self, item: Item, hash: ChainHash) {
        let (Some(ram), Some(ssd)) = (&mut self.in_ram, &mut self.ssd) else { return };
        let Some(entry) = ram.get(hash) else { return };
        let (parent, depth) = (entry.parent, entry.depth);
        match item {
            Item::Page => {
                if ssd.has_or_queued(Kind::Page, hash) {
                    ram.update(hash, |entry| entry.stored = true);
                    return;
                }
                let Some(parts) = ram.take(hash) else { return };
                // A text on the drive is found from its first page on: what
                // comes before it is worth writing too.
                match ram.get(parent).map(|entry| entry.worth) {
                    Some(false) => ram.update(parent, |entry| entry.worth = true),
                    Some(true) => {}
                    None => {
                        if self.index.page(parent).is_some() {
                            self.worth.insert(parent);
                        }
                    }
                }
                ssd.write(Kind::Page, hash, parent, depth, parts);
            }
            Item::Checkpoint => {
                let Some(held) = entry.checkpoint else { return };
                if ssd.has_or_queued(Kind::State, hash) {
                    ram.update(hash, |entry| {
                        if let Some(held) = &mut entry.checkpoint {
                            held.stored = true;
                        }
                    });
                    return;
                }
                ssd.queue(Kind::State, hash);
                // It counts as on the drive from now on: it is copied out
                // before its place can be written again, and a write that
                // fails costs the drive this checkpoint and nothing else.
                ram.update(hash, |entry| {
                    if let Some(held) = &mut entry.checkpoint {
                        held.stored = true;
                    }
                });
                let out = StateOut { place: held.place, hash, parent, depth, keep: true };
                self.states_out.push(out);
            }
        }
    }

    /// Gives up what the allowance of the books does not hold any more: what
    /// goes for nothing first, then what only RAM holds, which is written
    /// first (its memory goes when the writer is done), then the coldest
    /// checkpoints of the index. Returns the checkpoints that went. Call it
    /// between steps only, after [`Cache::pump`]: when the allowance fell,
    /// also while nothing runs.
    pub(crate) fn trim(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) -> u32 {
        if self.books.is_none() {
            return 0;
        }
        self.settle_leaving(pool, exec);
        self.settle();
        let target = self.allowance();
        // Far pages that only the index holds are the prompt cache's lowest:
        // they leave the devices first, and their memory goes back to the
        // system once the pages of a piece are all free.
        self.count_far(pool);
        while self.held() > target && self.far_held > 0 && self.evict_far(pool) {}
        self.settle_leaving(pool, exec);
        let mut dropped = 0;
        while self.held() > target && self.give_up_free(&mut dropped) {}
        // What is being written goes once it is: it counts as gone.
        let mut going = self.in_ram.as_ref().map_or(0, RamLevel::away_bytes);
        while self.held().saturating_sub(going) > target {
            let next = self.in_ram.as_ref().and_then(|ram| ram.first_from(Class::Dirty));
            match next {
                Some((_, Item::Page, hash)) => {
                    going += self.in_ram.as_ref().map_or(0, RamLevel::page_bytes);
                    self.clean(Item::Page, hash);
                }
                Some((_, Item::Checkpoint, hash)) => {
                    dropped += self.ram_give_up(Item::Checkpoint, hash);
                }
                None if self.give_up_index_checkpoint(false) => dropped += 1,
                None => break,
            }
        }
        // The checkpoints that went are copied out of their places for the
        // drive before their places can be called unused.
        self.settle_leaving(pool, exec);
        // Nothing runs: a freed place that no copy of the next record uses
        // is unused already, and the others wait for that record.
        let (saves, restores) = (&self.saves, &self.restores);
        let (waiting, unused): (Vec<u32>, Vec<u32>) = std::mem::take(&mut self.settling)
            .into_iter()
            .partition(|place| saves.iter().chain(restores).any(|copy| copy.place == *place));
        for place in unused {
            if self.free_places.contains(&place) {
                self.mark(place, false);
            }
        }
        self.settling = waiting;
        dropped
    }

    /// The owner a run that comes back from RAM or the drive is announced to
    /// the index under: one no sequence has.
    pub(super) fn import_owner(&mut self) -> Owner {
        self.next_import += 1;
        Owner(u64::MAX - self.next_import)
    }

    /// A page of the pool for `cells` more cells of `held`, a run that comes
    /// back: when the pool has none free, a cached page nobody holds leaves
    /// the devices, and its copy into RAM goes into `batch` before what comes
    /// in. `false` when there is none either.
    pub(super) fn take_page(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        held: &mut Sequence,
        cells: u64,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
    ) -> bool {
        if held.append(pool, cells).is_ok() {
            return true;
        }
        self.evict(pool, 1);
        self.push_leaving(pool, exec, batch, notes);
        held.append(pool, cells).is_ok()
    }

    /// Hangs the checkpoint in `place` on the page or tail named `hash` for
    /// `owner`, kept longest with `root`; the checkpoint it replaces there is
    /// released. `false` when the index has no such entry: the place holds
    /// nothing it keeps.
    pub(super) fn hang(&mut self, hash: ChainHash, place: u32, owner: Owner, root: bool) -> bool {
        match self.index.attach_checkpoint(hash, CheckpointId(u64::from(place)), owner) {
            Ok(before) => {
                if root {
                    self.index.mark_root(hash);
                }
                self.placed.insert(place, hash);
                if let Some(before) = before {
                    self.release(before);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Copies what RAM holds after the `found` pages of the index for a
    /// prompt of `tokens` (named in `chain`) back to the devices, where it
    /// enters the index. For a model with recurrent state the run ends at
    /// the deepest checkpoint it has, unless a tail with the state at its
    /// end follows or the drive goes on to a deeper one. Returns the pages
    /// and tails that came.
    pub(super) fn import_from_ram(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        chain: &Chain,
        tokens: &[u64],
        found: &PrefixMatch,
    ) -> u32 {
        let Some(ram) = &self.in_ram else { return 0 };
        let per_page = self.cells_per_page as usize;
        let looked = &tokens[..tokens.len() - 1];
        let have = found.pages.len();
        let total = (looked.len() / per_page).min(chain.hashes().len());
        let names = &chain.hashes()[have.min(total)..total];
        let run = run_of(&self.index, names, |hash| ram.ready(hash));
        // A tail where the run ends, unless the index has a better one.
        let rest = &looked[(have + run) * per_page..];
        let tail = (found.tail.is_none() || run > 0)
            .then(|| {
                let tails = ram.tails_under(chain.before(have + run)).map(|(hash, entry)| {
                    ((hash, entry.checkpoint.is_some()), entry.tail.as_deref().unwrap_or(&[]))
                });
                best_tail(tails, rest)
            })
            .flatten()
            .map(|((hash, checkpoint), shared, whole)| (shared, whole, checkpoint, hash));
        let (run, tail) = if self.recurrent {
            let deepest = deepest(names, run, |hash| {
                ram.get(hash).is_some_and(|entry| entry.checkpoint.is_some())
            });
            let drive = self.ssd.as_ref().is_some_and(|ssd| ssd.reaches_state(&names[run..]));
            match tail {
                Some((_, true, true, hash)) => (run, Some(hash)),
                _ if drive => (run, None),
                _ => (deepest.map_or(0, |at| at + 1), None),
            }
        } else {
            (run, tail.map(|(.., hash)| hash))
        };
        if run == 0 && tail.is_none() {
            return 0;
        }
        // What the index matched is held, so that room made for the run does
        // not take it; the run is taken out of RAM's order first, so that
        // room made for pages that leave the devices does not take it either.
        let kept = Sequence::from_shared(pool, &found.table(), found.cells(per_page as u32));
        let run = run_pages(chain, have, run);
        let tail = tail.map(|hash| {
            let entry = self.in_ram.as_ref().and_then(|ram| ram.get(hash)).expect("listed above");
            (hash, entry.parent, entry.tail.as_ref().map_or(0, Vec::len))
        });
        let mut taken: Vec<(ChainHash, ChainHash, u32, Vec<crate::exec::TransitPart>)> = Vec::new();
        let ram = self.in_ram.as_mut().expect("checked above");
        for (hash, parent, cells) in run
            .iter()
            .map(|(hash, parent, _)| (*hash, *parent, per_page as u32))
            .chain(tail.map(|(hash, parent, cells)| (hash, parent, cells as u32)))
        {
            match ram.take(hash) {
                Some(parts) => taken.push((hash, parent, cells, parts)),
                None => break,
            }
        }
        let owner = self.import_owner();
        let settle = u64::from(self.durable.as_ref().map_or(0, |durable| durable.settle_cells));
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        let mut held = Sequence::new();
        let mut left = Vec::new();
        let count = taken.len();
        for (i, (hash, parent, cells, parts)) in taken.into_iter().enumerate() {
            if !self.take_page(pool, exec, &mut held, u64::from(cells), &mut batch, &mut notes) {
                left.push((hash, parts));
                continue;
            }
            let page = *held.pages().last().expect("a page was appended");
            // A tail keeps the form it has, as the newest cells of a run do.
            let tail_page = tail.is_some() && i + 1 == count;
            let behind = (count - 1 - i) as u64 * per_page as u64;
            let depth = run.get(i).map_or(0, |(_, _, depth)| *depth);
            let keep_form = tail_page || keeps_form(depth, behind, settle, per_page as u64);
            batch.push(Transfer {
                kind: TransferKind::PageIn,
                object: page.0,
                keep_form,
                parts,
                ok: false,
            });
            notes.push(Note::FromRam { hash, parent, owner });
        }
        if let Some(ram) = &mut self.in_ram {
            for (hash, parts) in left {
                ram.give(hash, parts);
            }
        }
        let read = self.in_ram.as_ref().map_or(0, |ram| ram.counters.pages_read);
        self.carry_out(pool, exec, batch, notes);
        held.release(pool);
        kept.release(pool);
        (self.in_ram.as_ref().map_or(0, |ram| ram.counters.pages_read) - read) as u32
    }

    /// Carries out `batch` and what it came to.
    pub(super) fn carry_out(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        mut batch: Vec<Transfer>,
        notes: Vec<Note>,
    ) {
        if batch.is_empty() {
            return;
        }
        let begun = Instant::now();
        if exec.transfer(&mut batch).is_err() {
            for transfer in &mut batch {
                transfer.ok = false;
            }
        }
        // Once for the cache, whichever level the batch is for.
        self.transfers.batches += 1;
        self.transfers.ns += begun.elapsed().as_nanos() as u64;
        self.finish(pool, batch, notes);
    }

    /// What the transfers of a batch came to, in their order.
    pub(super) fn finish(&mut self, pool: &mut PagePool, batch: Vec<Transfer>, notes: Vec<Note>) {
        let mut state_in: Option<(u64, u32, bool)> = None;
        for (transfer, note) in batch.into_iter().zip(notes) {
            let Transfer { object, parts, ok, .. } = transfer;
            match note {
                Note::ToRam(leaving) => {
                    let Some(ram) = &mut self.in_ram else { continue };
                    if !ok {
                        ram.recycle(parts);
                        self.lost(leaving);
                        continue;
                    }
                    let Leaving { hash, parent, depth, tail, checkpoint, worth, .. } = leaving;
                    let stored =
                        self.ssd.as_ref().is_some_and(|ssd| ssd.has_or_queued(Kind::Page, hash));
                    let mut entry = Entry::new(parent, depth, tail);
                    entry.checkpoint = checkpoint;
                    entry.worth = worth;
                    entry.stored = stored;
                    ram.counters.pages_in += 1;
                    if ram.get(hash).is_some() {
                        // A copy came in meanwhile: this one is not needed,
                        // and the checkpoint goes on the one that is there.
                        ram.recycle(parts);
                        let spare = checkpoint.filter(|_| {
                            ram.get(hash).is_some_and(|entry| entry.checkpoint.is_some())
                        });
                        ram.update(hash, |entry| {
                            if entry.checkpoint.is_none() {
                                entry.checkpoint = checkpoint;
                            }
                        });
                        if let Some(held) = spare {
                            self.give_up(held.place, None);
                        }
                        continue;
                    }
                    ram.insert(hash, entry, parts);
                }
                Note::FromRam { hash, parent, owner } => {
                    let Some(ram) = &mut self.in_ram else { continue };
                    ram.give(hash, parts);
                    // A run from RAM enters the index in order, and ends at
                    // a page that did not come: what follows it would hang
                    // on nothing.
                    let follows = parent == self.root || self.index.page(parent).is_some();
                    if !ok || !follows {
                        continue;
                    }
                    let tail = ram.get(hash).and_then(|entry| entry.tail.clone());
                    let mut held = None;
                    ram.update(hash, |entry| {
                        entry.on_cards = true;
                        held = entry.checkpoint.take();
                    });
                    ram.counters.pages_read += 1;
                    match tail {
                        None => {
                            self.index.insert(pool, parent, hash, PageId(object));
                        }
                        Some(tokens) => {
                            let page = PageId(object);
                            let dropped = self.index.insert_tail(
                                pool,
                                parent,
                                &tokens,
                                page,
                                Priority::NORMAL,
                            );
                            if let Some(evicted) = dropped {
                                self.leave_cards(evicted, &HashSet::new());
                            }
                        }
                    }
                    if let Some(held) = held
                        && !self.hang(hash, held.place, owner, held.root)
                    {
                        self.give_up(held.place, None);
                    }
                }
                Note::CardsOut { hash, parent, depth } => {
                    let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                    match ok {
                        true => ssd.write(Kind::Page, hash, parent, depth, parts),
                        false => {
                            ssd.unqueue(Kind::Page, hash);
                            if let Some(ram) = &mut self.in_ram {
                                ram.recycle(parts);
                            }
                        }
                    }
                }
                Note::StateOut(out) => {
                    let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                    match ok {
                        true => ssd.write(Kind::State, out.hash, out.parent, out.depth, parts),
                        false => {
                            ssd.unqueue(Kind::State, out.hash);
                            ssd.free_states.push(parts);
                            ssd.counters.states_skipped += 1;
                        }
                    }
                }
                Note::StateIn { job } => {
                    state_in = Some((job, object, ok));
                    if let Some(ssd) = &mut self.ssd {
                        ssd.free_states.push(parts);
                    }
                }
                Note::PageIn { job, index } => {
                    let state = state_in.take_if(|(of, ..)| *of == job);
                    self.page_in(pool, job, index, object, parts, ok, state);
                }
            }
        }
    }
}
