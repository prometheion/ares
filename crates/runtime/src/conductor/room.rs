//! Room in the RAM limit for the K and V beyond VRAM that the pages in use
//! need, made before a step is written (see "Host memory" in
//! `docs/dev/architecture/host-memory.md`): the tier of experts gives what its
//! allowance no longer holds (the model does that), the prompt cache gives up
//! what it holds, and when nothing more can go one request ends with an error
//! that says so, while the others go on.

use ares_core::units::Bytes;
use ares_state::PageId;

use super::*;

impl Conductor {
    /// Makes room for the K and V beyond VRAM of the pages in use once the
    /// planned step (`self.parts`) has run, before its record is written. A
    /// request that finds none ends ([`EngineError::NoMemory`]) and leaves the
    /// step. `Err` only when the model failed.
    pub(super) fn make_room(&mut self) -> Result<(), String> {
        let Some(far) = self.model.far_pages() else { return Ok(()) };
        loop {
            let live: Vec<bool> =
                (0..self.pool.capacity()).map(|p| self.pool.refcount(PageId(p)) > 0).collect();
            let short = self.model.make_room(&live).map_err(|e| e.to_string())?;
            if short == 0 {
                return Ok(());
            }
            if let Some(cache) = &mut self.cache {
                let (dropped, gone) = cache.give_way(&mut self.pool, &mut *self.exec);
                self.stats.counters.checkpoints_dropped += u64::from(dropped);
                if gone {
                    continue;
                }
            }
            // Nothing more can go: a request ends, the one that holds the most
            // pages beyond VRAM of the step's prompts, else of the step, else of
            // all that run.
            let parts = &self.parts;
            let victim = (self.slots.iter().enumerate())
                .filter_map(|(slot, a)| {
                    let a = a.as_ref()?;
                    let part = parts.iter().find(|p| p.slot as usize == slot);
                    let prompt = part.is_some_and(|p| matches!(p.kind, PartKind::Prefill { .. }));
                    let pages = a.cells.pages().iter().filter(|p| p.0 >= far.first).count();
                    Some((prompt, part.is_some(), pages, slot as u32))
                })
                .max();
            let Some((.., slot)) = victim else {
                return Err(format!(
                    "the RAM limit has no room for {} more of K and V beyond VRAM and no request \
                     holds any: {}",
                    Bytes(short),
                    self.exec.ram().map_or_else(String::new, |ram| ram.who_holds())
                ));
            };
            let message = format!(
                "the engine's RAM limit (--ram) had no room for {} more of this request's K and V \
                 beyond VRAM, with the prompt cache given up and the tier of experts as small as \
                 it goes ({}); the request ended and the model serves on: send it again later, \
                 shorter, or start the server with more --ram",
                Bytes(short),
                self.exec.ram().map_or_else(String::new, |ram| ram.who_holds())
            );
            self.end_for_memory(slot, message);
        }
    }

    /// Whether the planned step is one the model takes: a step wider than its
    /// own rows borrows memory for its time, which only a prompt read alone
    /// gets, and none while another step is paused (the planner keeps to that;
    /// a model that reads such a step layer by layer refuses any other, and its
    /// refusal on the devices would fail the engine). A step that is not ends
    /// its requests with an error, and the engine goes on.
    pub(super) fn step_is_taken(&mut self) -> bool {
        let rows: u32 = self.parts.iter().map(|p| p.rows).sum();
        let own = self.model.own_rows().min(self.max_rows);
        let paused = self.paused.is_some_and(|p| self.parts.iter().any(|q| q.slot != p.slot));
        if rows <= own || (self.parts.len() == 1 && !paused) {
            return true;
        }
        let message = format!(
            "a step of {rows} rows for {} sequences{} was planned, wider than the {own} rows \
             of a step that holds more than one prompt: the requests of the step end",
            self.parts.len(),
            if paused { " beside a paused step" } else { "" }
        );
        for slot in self.parts.iter().map(|p| p.slot).collect::<Vec<_>>() {
            if let Some(a) = &self.slots[slot as usize] {
                let _ = a.sink.send(Event::Error(EngineError::Internal(message.clone())));
            }
            self.retire_with(slot, FinishReason::Cancelled, false);
        }
        let saves = std::mem::take(&mut self.inner);
        self.release_inner(saves);
        self.inner_due.clear();
        self.parts.clear();
        false
    }

    /// Ends the request in `slot` with `message`, for lack of memory: it leaves
    /// the planned step and keeps nothing in the cache.
    fn end_for_memory(&mut self, slot: u32, message: String) {
        if let Some(a) = &self.slots[slot as usize] {
            let _ = a.sink.send(Event::Error(EngineError::NoMemory(message)));
        }
        self.stats.counters.requests_no_memory += 1;
        self.retire_with(slot, FinishReason::Cancelled, false);
        self.parts.retain(|p| p.slot != slot);
        let (gone, kept): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.inner).into_iter().partition(|s| s.slot == slot);
        self.inner = kept;
        self.release_inner(gone);
        self.inner_due.retain(|d| d.0 != slot);
    }
}
