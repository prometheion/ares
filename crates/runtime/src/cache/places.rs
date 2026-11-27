//! The checkpoint places: a new place with room made for it, what goes for
//! nothing first, the saves, restores and copies of the next record.

use super::*;

impl Cache {
    /// The place of a new checkpoint, with room made for it, and how many
    /// checkpoints that cost. What goes for nothing makes room first; then,
    /// when the checkpoints are as many as the allowance holds, those of
    /// pages that left the devices go before those of the index.
    pub(super) fn new_place(&mut self) -> (Option<u32>, u32) {
        let mut dropped = 0;
        while self.held() + self.bytes > self.allowance() && self.give_up_free(&mut dropped) {}
        let capacity = self.policy_now().capacity();
        dropped += self.make_room_for_one();
        // No room for even one checkpoint now: none is taken.
        if capacity == 0 {
            return (None, dropped);
        }
        let place = self.free_place(&mut dropped);
        if let Some(place) = place {
            self.mark(place, true);
        }
        (place, dropped)
    }

    /// Gives checkpoints up until the index and RAM hold one fewer than the
    /// budget allows, those of pages that left the devices before those of the
    /// index; returns how many went.
    pub(super) fn make_room_for_one(&mut self) -> u32 {
        let capacity = self.policy_now().capacity();
        let in_ram = |cache: &Self| cache.in_ram.as_ref().map_or(0, |ram| ram.checkpoints());
        let mut dropped = 0;
        while self.index.checkpoints() as u64 + in_ram(self) + 1 > capacity as u64 {
            if !self.give_up_ram_checkpoint() && !self.give_up_index_checkpoint(false) {
                break;
            }
            dropped += 1;
        }
        dropped
    }

    /// A place for a checkpoint whose memory the limit of the books holds: a
    /// free place whose memory is locked already if there is one, else one
    /// whose first use the limit has room for, made by giving up what goes
    /// for nothing. The devices lock a place when a copy of a step first uses
    /// it, and a lock beyond the limit would fail that step: pages in RAM
    /// share the limit with the places, so this is not given by the number
    /// of places alone. `None` when there is no such place. `dropped` counts
    /// the checkpoints that went to make room.
    pub(super) fn free_place(&mut self, dropped: &mut u32) -> Option<u32> {
        let Some(books) = self.books.clone() else {
            return self.free_places.pop().or_else(|| self.unused_place());
        };
        loop {
            let locked = self.free_places.iter().rposition(|place| books.place_lock(*place) == 0);
            if let Some(at) = locked {
                return Some(self.free_places.remove(at));
            }
            let place = self
                .free_places
                .last()
                .copied()
                .or_else(|| (self.next_place < self.capacity).then_some(self.next_place))?;
            let mut places = self.pending_places();
            places.push(place);
            if books.unbooked() >= books.places_lock(&places) {
                return match self.free_places.last() == Some(&place) {
                    true => self.free_places.pop(),
                    false => self.unused_place(),
                };
            }
            if !self.give_up_free(dropped) {
                return None;
            }
        }
    }

    /// The bytes the devices will lock for the saves of the next record and
    /// the saves inside a step reserved for it: the chunks of the places they
    /// use whose memory is not locked yet, each once. Nothing else may book
    /// that room of the limit before the record has run.
    pub(super) fn pending_locks(&self) -> u64 {
        let Some(books) = &self.books else { return 0 };
        books.places_lock(&self.pending_places())
    }

    /// The places of the next record's saves and of the saves inside a step
    /// reserved for it.
    pub(super) fn pending_places(&self) -> Vec<u32> {
        self.saves.iter().map(|copy| copy.place).chain(self.reserved.iter().copied()).collect()
    }

    /// A place never used so far, if there is one.
    pub(super) fn unused_place(&mut self) -> Option<u32> {
        (self.next_place < self.capacity).then(|| {
            self.next_place += 1;
            self.next_place - 1
        })
    }

    /// Gives one thing up that is worth little: a buffer that holds nothing,
    /// a page or checkpoint in RAM that the devices or the drive hold too or
    /// that is not worth writing, or a checkpoint of the index that later
    /// requests have left behind (written all the same when the drive keeps
    /// nothing deeper of its text). `false` when there is nothing of the
    /// kind. `dropped` counts the checkpoints.
    pub(super) fn give_up_free(&mut self, dropped: &mut u32) -> bool {
        if let Some(ram) = &mut self.in_ram {
            if ram.free_bytes() > 0 {
                ram.shed();
                return true;
            }
            if let Some((Class::Copy | Class::Free, item, hash)) = ram.first() {
                *dropped += self.ram_give_up(item, hash);
                return true;
            }
        }
        if self.give_up_index_checkpoint(true) {
            *dropped += 1;
            return true;
        }
        false
    }

    /// Gives up the page or the checkpoint `hash` in RAM, the checkpoint
    /// with its page; what is due on the drive goes there. Returns the
    /// checkpoints that went.
    pub(super) fn ram_give_up(&mut self, item: Item, hash: ChainHash) -> u32 {
        let writes = self.ssd.is_some();
        let Some(ram) = &mut self.in_ram else { return 0 };
        let (held, parent, depth) = match item {
            Item::Page => {
                let Some(entry) = ram.remove(hash) else { return 0 };
                let unwritten = !entry.on_cards && entry.worth && !entry.stored;
                if unwritten && entry.tail.is_none() && writes {
                    ram.counters.pages_dropped += 1;
                }
                let depth = entry.stored_depth();
                if let Some(parts) = entry.parts {
                    ram.recycle(parts);
                }
                (entry.checkpoint, entry.parent, depth)
            }
            Item::Checkpoint => {
                let mut held = None;
                ram.update(hash, |entry| held = entry.checkpoint.take());
                let entry = ram.get(hash).expect("the entry is there");
                (held, entry.parent, entry.stored_depth())
            }
        };
        let Some(held) = held else { return 0 };
        let out = self.state_out_of(&held, hash, parent, depth);
        self.give_up(held.place, out);
        1
    }

    /// Gives up the checkpoint that goes first of those that hang on pages
    /// in RAM. `false` when there is none.
    pub(super) fn give_up_ram_checkpoint(&mut self) -> bool {
        let Some(hash) = self.in_ram.as_ref().and_then(RamLevel::first_checkpoint) else {
            return false;
        };
        self.ram_give_up(Item::Checkpoint, hash) > 0
    }

    /// Gives up the checkpoint of the index that its ranking drops first, or,
    /// with `left_behind`, only one that later requests have left behind. It
    /// goes to the drive unless later requests left it behind at deeper page
    /// boundaries ([`Cache::left_behind`]). `false` when there is none.
    pub(super) fn give_up_index_checkpoint(&mut self, left_behind: bool) -> bool {
        if self.index.checkpoints() == 0 {
            return false;
        }
        let Some(first) = self.index.checkpoint_ranks(&self.policy).first().copied() else {
            return false;
        };
        if left_behind && !first.aged {
            return false;
        }
        let behind = self.left_behind().contains(&first.hash);
        let Ok(Some(id)) = self.index.set_checkpoint(first.hash, None) else { return false };
        let out = match self.index.page(first.hash) {
            Some((_, parent, depth)) if !behind => {
                self.state_out(id.0 as u32, first.hash, parent, depth)
            }
            _ => None,
        };
        self.give_up(id.0 as u32, out);
        true
    }

    /// The checkpoints of the index at page boundaries that later requests
    /// have left behind at deeper page boundaries. Only a level on SSD asks:
    /// such a checkpoint is not written, since the drive keeps no tails and a
    /// prompt resumes there from the deepest checkpoint at a page boundary.
    pub(super) fn left_behind(&self) -> HashSet<ChainHash> {
        if self.ssd.is_none() || self.index.checkpoints() == 0 {
            return HashSet::new();
        }
        self.index.left_behind_on_pages(&self.policy).into_iter().collect()
    }

    /// A page at `depth` lies deep enough for its text to be written.
    pub(super) fn deep(&self, depth: u32) -> bool {
        self.ssd
            .as_ref()
            .is_some_and(|ssd| u64::from(depth) * u64::from(self.cells_per_page) >= ssd.min_cells())
    }

    /// The checkpoint in `place`, at the end of page `hash`, goes to the
    /// drive when it leaves its place: there is a level on SSD, it lies deep
    /// enough, and the drive does not have it.
    pub(super) fn state_out(
        &self,
        place: u32,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
    ) -> Option<StateOut> {
        let ssd = self.ssd.as_ref()?;
        let due = self.deep(depth) && !ssd.has_or_queued(Kind::State, hash);
        due.then_some(StateOut { place, hash, parent, depth, keep: false })
    }

    pub(super) fn state_out_of(
        &self,
        held: &Held,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
    ) -> Option<StateOut> {
        (held.worth && !held.stored).then(|| self.state_out(held.place, hash, parent, depth))?
    }

    /// The checkpoint in `place` is given up: `out` says it goes to the
    /// drive first. The place is free once nothing reads it.
    pub(super) fn give_up(&mut self, place: u32, out: Option<StateOut>) {
        self.placed.remove(&place);
        // A save of it that still waits for its record has nothing to save,
        // and the level on SSD nothing to write from the place. Unless a
        // restore of that record reads the place: the save runs before it, so
        // the restore gets the state the checkpoint was taken of (a request's
        // end state that the next turn of its conversation starts from, given
        // up for the places of that turn's saves inside its first step).
        let unsaved = self.saves.iter().any(|copy| copy.place == place);
        let restored = self.restores.iter().any(|copy| copy.place == place);
        if !restored {
            self.saves.retain(|copy| copy.place != place);
        }
        if let (Some(ssd), Some(out)) = (&mut self.ssd, out) {
            if unsaved {
                ssd.counters.states_skipped += 1;
            } else {
                ssd.queue(Kind::State, out.hash);
                self.states_out.push(out);
            }
        }
        if restored {
            self.deferred.push(place);
        } else {
            self.free(place);
        }
    }

    /// Gives a checkpoint's place back once nothing reads it, writing the
    /// checkpoint to the drive first when it is due there.
    pub(crate) fn release(&mut self, id: CheckpointId) {
        let place = id.0 as u32;
        let out = self
            .placed
            .get(&place)
            .and_then(|hash| Some((*hash, self.index.page(*hash)?)))
            .and_then(|(hash, (_, parent, depth))| self.state_out(place, hash, parent, depth));
        self.give_up(place, out);
    }

    /// Place `place` holds no checkpoint any more and no restore waits for
    /// it: it can be taken again, and it is unused once the record that was
    /// written last has run (a copy of it may still use the memory).
    pub(super) fn free(&mut self, place: u32) {
        self.free_places.push(place);
        self.settling.push(place);
    }

    /// A place for a checkpoint that a step saves inside its rows
    /// ([`crate::InnerSave`]), before the step: what it holds is named at the
    /// commit ([`Cache::attach`]), or the place goes back ([`Cache::unreserve`]).
    /// `None` without a model that keeps recurrent state or without room.
    pub(crate) fn reserve(&mut self) -> (Option<u32>, u32) {
        if !self.recurrent || self.capacity == 0 {
            return (None, 0);
        }
        let (place, dropped) = self.new_place();
        self.reserved.extend(place);
        (place, dropped)
    }

    /// The checkpoint a step saved inside its rows into `place` (reserved by
    /// [`Cache::reserve`]) is of the page named `hash`: as [`Cache::save`],
    /// with the copy done already.
    pub(crate) fn attach(
        &mut self,
        place: u32,
        hash: ChainHash,
        owner: Owner,
        root: bool,
    ) -> Saved {
        self.reserved.retain(|p| *p != place);
        if self.index.checkpoint(hash).is_some() {
            if root {
                self.index.mark_root(hash);
            }
            self.free(place);
            return Saved::default();
        }
        // The places were taken before the step, when the index did not hold the saves
        // inside it before this one: room is made now, as a save at a step's end makes it,
        // so that a later save pushes an earlier one out as it would after a cut.
        let dropped = self.make_room_for_one();
        let taken = self.hang(hash, place, owner, root);
        if !taken {
            self.free(place);
        }
        Saved { taken, dropped }
    }

    /// A place reserved for a save inside a step that holds nothing after all.
    pub(crate) fn unreserve(&mut self, place: u32) {
        self.reserved.retain(|p| *p != place);
        self.free(place);
    }

    /// Takes a checkpoint of the state in `slot`, which is at the end of the
    /// page named `hash` or of the tail so named, unless the index has one
    /// there already. With `root`, it is at the end of a system prompt or a
    /// tool list, and kept longest. The copy runs with the next record.
    pub(crate) fn save(&mut self, slot: u32, hash: ChainHash, owner: Owner, root: bool) -> Saved {
        if !self.recurrent || self.capacity == 0 {
            return Saved::default();
        }
        if self.index.checkpoint(hash).is_some() {
            // A root of this prompt that another text keeps as a plain one.
            if root {
                self.index.mark_root(hash);
            }
            return Saved::default();
        }
        let (place, dropped) = self.new_place();
        let Some(place) = place else { return Saved { taken: false, dropped } };
        let taken = self.hang(hash, place, owner, root);
        match taken {
            true => self.saves.push(StateCopy { kind: StateCopyKind::Save, slot, place }),
            false => self.free(place),
        }
        Saved { taken, dropped }
    }

    /// Takes the checkpoint of a request's live state at its end, as
    /// [`Cache::save`], unless the budget holds one checkpoint besides the
    /// roots and that is a fresh one of the same request (its anchor). The
    /// end state lies less than a page and the reply's rows after the anchor,
    /// and the next turn of a conversation often parts from it inside the
    /// reply: Claude Code sends back a call of its Edit tool with an argument
    /// the model never wrote, Qwen's templates leave out the thinking of an
    /// answer once a new question follows, and a client asks the same prompt
    /// again. With the end state in the one place such a turn replays every
    /// row it shares after the roots; with the anchor, the reply's rows.
    ///
    /// The roots count: they go after every other checkpoint, so once they
    /// hold all places but one, the conversations have that one place, as
    /// with a budget of one.
    pub(crate) fn save_end(&mut self, slot: u32, hash: ChainHash, owner: Owner) -> Saved {
        let in_ram = self.in_ram.as_ref().map_or(0, RamLevel::checkpoints);
        if in_ram == 0 && self.index.checkpoints() > 0 {
            let ranks = self.index.checkpoint_ranks(&self.policy);
            let roots = ranks.iter().filter(|rank| rank.root).count();
            let mut others = ranks.iter().filter(|rank| !rank.root);
            let one = self.policy_now().capacity() <= roots + 1;
            if let (true, Some(only), None) = (one, others.next(), others.next())
                && only.by == Some(owner)
                && !only.aged
            {
                return Saved::default();
            }
        }
        self.save(slot, hash, owner, false)
    }

    /// Queues the restore of `id` into `slot` for the next record.
    pub(crate) fn restore(&mut self, slot: u32, id: CheckpointId) {
        self.restores.push(StateCopy { kind: StateCopyKind::Restore, slot, place: id.0 as u32 });
    }

    /// Drops the restore queued for `slot`: the sequence it was for left the
    /// devices before the record that carries it was written, and the slot
    /// may be taken by another. Returns whether there was one.
    pub(crate) fn cancel_restore(&mut self, slot: u32) -> bool {
        let before = self.restores.len();
        self.restores.retain(|copy| copy.slot != slot);
        if self.restores.len() == before {
            return false;
        }
        // Places that were waiting for it are free if nothing else reads them,
        // and a save into one that was kept for the restore has nothing to save.
        let restores = &self.restores;
        let (free, waiting): (Vec<u32>, Vec<u32>) = std::mem::take(&mut self.deferred)
            .into_iter()
            .partition(|place| !restores.iter().any(|copy| copy.place == *place));
        self.saves.retain(|copy| !free.contains(&copy.place));
        self.settling.extend(&free);
        self.free_places.extend(free);
        self.deferred = waiting;
        true
    }

    /// Queues the copy of a shared page for the next record.
    pub(crate) fn copy_page(&mut self, from: PageId, to: PageId, cells: u32) {
        self.page_copies.push(PageCopy { from: from.0, to: to.0, cells });
    }
}
