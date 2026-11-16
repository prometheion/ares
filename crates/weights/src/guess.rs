//! Guesses that wait for a read.

use crate::expert::{ExpertHeat, SlotSet, Start, Swap, Want};

/// One guess on the list. The derived order is the order they are read in:
/// the nearest step and layer first (it is needed soonest), then the most
/// likely expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Waiting {
    step: u64,
    layer: usize,
    rank: usize,
    expert: u32,
}

/// A read that [`GuessQueue::start`] started: read `swap.load` of `layer`
/// into the slot of `swap`, then commit it in that layer's [`SlotSet`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Started {
    pub layer: usize,
    pub swap: Swap,
}

/// The guesses that no I/O lane has had time for yet.
///
/// A prediction names a handful of experts for a layer that has not run, and
/// the reads for them go behind the demand reads. If each guess took its slot
/// when it was made, a list of guesses waiting for a lane would already have
/// evicted as many cached experts, and the ones that go stale (the layer ran
/// before the guess came to the front) would have evicted them for nothing.
/// So a guess waits here without touching any tier, and takes its slot at the
/// moment a lane is free to read it ([`GuessQueue::start`]); a guess that went
/// stale never takes one.
///
/// Guesses are read in the order the layers will need them: the layers of the
/// current step in order, then those of the next step, and within a layer the
/// likeliest first. The list is plain data for one thread, like the tiers it
/// works on.
#[derive(Clone, Debug, Default)]
pub struct GuessQueue {
    waiting: Vec<Waiting>,
}

impl GuessQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of guesses waiting.
    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    /// Records that `experts` are guessed for `layer` in step `step`, most
    /// likely first. The prediction comes from a router some layers before
    /// `layer`, or, for the first layers of the next step, from the token
    /// table.
    ///
    /// The guessed experts are protected in the layer's tier
    /// ([`SlotSet::predicted`]). Those that are not there wait for a read. An
    /// expert guessed twice (from two routers, or by the table and a router)
    /// waits once, at its better rank.
    pub fn guess(&mut self, sets: &mut [SlotSet], layer: usize, experts: &[u32], step: u64) {
        sets[layer].predicted(experts, step);
        for (rank, expert) in experts.iter().enumerate() {
            if sets[layer].contains(*expert) {
                continue;
            }
            let guess = Waiting { step, layer, rank, expert: *expert };
            match self.waiting.iter().position(|w| w.layer == layer && w.expert == *expert) {
                Some(at) if self.waiting[at].step == step && self.waiting[at].rank <= rank => {}
                Some(at) => {
                    self.waiting.remove(at);
                    self.insert(guess);
                }
                None => self.insert(guess),
            }
        }
    }

    fn insert(&mut self, guess: Waiting) {
        let at = self.waiting.partition_point(|w| *w < guess);
        self.waiting.insert(at, guess);
    }

    /// Drops the guesses for `layer`: its router has run, so what it did not
    /// pick was guessed wrong, and what it did pick is a demand read now.
    pub fn forget_layer(&mut self, layer: usize) {
        self.waiting.retain(|w| w.layer != layer);
    }

    /// Starts the read of the first waiting guess that can have a slot, for
    /// an I/O lane that has nothing more urgent to do. Call it when a lane
    /// is free and no demand read waits, until it returns `None`.
    ///
    /// Guesses of earlier steps are dropped. Guesses whose expert is in the
    /// tier, or on its way, are dropped too (another path brought it). A
    /// guess that no slot can be taken for (every slot of its layer is
    /// loading or needed by the step) stays and is tried again.
    pub fn start(&mut self, sets: &mut [SlotSet], heat: &ExpertHeat, step: u64) -> Option<Started> {
        self.start_where(sets, heat, step, &mut |_, _| true)
    }

    /// [`GuessQueue::start`] for the guesses `allowed(layer, expert)` lets start now; the
    /// others wait as they are (a reader that keeps one kind of guess within a bound of its
    /// own).
    pub fn start_where(
        &mut self,
        sets: &mut [SlotSet],
        heat: &ExpertHeat,
        step: u64,
        allowed: &mut dyn FnMut(usize, u32) -> bool,
    ) -> Option<Started> {
        self.waiting.retain(|w| w.step >= step);
        let mut at = 0;
        while at < self.waiting.len() {
            let w = self.waiting[at];
            if !allowed(w.layer, w.expert) {
                at += 1;
                continue;
            }
            match sets[w.layer].start_read(w.expert, Want::Guess, step, heat.layer(w.layer)) {
                Start::Slot(swap) => {
                    self.waiting.remove(at);
                    return Some(Started { layer: w.layer, swap });
                }
                Start::Present | Start::Loading => {
                    self.waiting.remove(at);
                }
                Start::NoRoom => at += 1,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPERTS: usize = 8;

    /// Two layers of eight experts, each with a tier of `slots` slots that
    /// hold the experts `0..slots` (read at step 0).
    fn tiers(slots: usize) -> (Vec<SlotSet>, ExpertHeat) {
        let mut heat = ExpertHeat::new(2, EXPERTS, 0.9);
        let mut sets = vec![SlotSet::new(EXPERTS, slots), SlotSet::new(EXPERTS, slots)];
        for (layer, set) in sets.iter_mut().enumerate() {
            for expert in 0..slots as u32 {
                heat.record(layer, &[expert, expert]);
                if let Start::Slot(swap) =
                    set.start_read(expert, Want::Demand, 0, heat.layer(layer))
                {
                    set.commit(&swap);
                }
            }
        }
        (sets, heat)
    }

    #[test]
    fn a_guess_takes_no_slot_until_a_lane_is_free_to_read_it() {
        let (mut sets, heat) = tiers(3);
        let mut queue = GuessQueue::new();
        queue.guess(&mut sets, 1, &[5, 6, 7], 1);
        assert_eq!(queue.len(), 3);
        assert!((0..3).all(|e| sets[1].contains(e)), "nothing was evicted by making the guesses");

        let started = queue.start(&mut sets, &heat, 1).unwrap();
        assert_eq!((started.layer, started.swap.load), (1, 5), "the most likely first");
        assert!(started.swap.evict.is_some());
        assert_eq!(queue.len(), 2);
        sets[1].commit(&started.swap);
        assert!(sets[1].contains(5));
    }

    #[test]
    fn the_nearest_layer_and_the_most_likely_expert_are_read_first() {
        let (mut sets, heat) = tiers(2);
        let mut queue = GuessQueue::new();
        queue.guess(&mut sets, 1, &[5, 4], 1);
        queue.guess(&mut sets, 0, &[7, 6], 1);
        // Expert 6 was guessed again, with a better rank; it moves up and
        // waits once.
        queue.guess(&mut sets, 0, &[6], 1);
        assert_eq!(queue.len(), 4);
        let mut order = Vec::new();
        while let Some(started) = queue.start(&mut sets, &heat, 1) {
            sets[started.layer].commit(&started.swap);
            order.push((started.layer, started.swap.load));
        }
        assert_eq!(order, [(0, 6), (0, 7), (1, 5), (1, 4)]);
    }

    #[test]
    fn a_guess_that_is_not_allowed_now_waits_and_lets_the_next_start() {
        let (mut sets, heat) = tiers(2);
        let mut queue = GuessQueue::new();
        queue.guess(&mut sets, 0, &[5, 6], 1);
        queue.guess(&mut sets, 1, &[7], 1);
        let started = queue.start_where(&mut sets, &heat, 1, &mut |layer, _| layer == 1).unwrap();
        assert_eq!((started.layer, started.swap.load), (1, 7));
        sets[1].commit(&started.swap);
        assert!(queue.start_where(&mut sets, &heat, 1, &mut |layer, _| layer == 1).is_none());
        assert_eq!(queue.len(), 2, "the guesses of layer 0 wait");
        assert!(sets[0].contains(0) && sets[0].contains(1), "and have taken nothing");
        let started = queue.start(&mut sets, &heat, 1).unwrap();
        assert_eq!((started.layer, started.swap.load), (0, 5));
    }

    #[test]
    fn guesses_for_a_layer_that_has_run_or_a_step_that_is_over_are_dropped() {
        let (mut sets, heat) = tiers(2);
        let mut queue = GuessQueue::new();
        queue.guess(&mut sets, 0, &[5], 1);
        queue.guess(&mut sets, 1, &[6, 7], 1);
        queue.forget_layer(1);
        assert_eq!(queue.len(), 1);
        // Step 2 has begun: the guess of step 1 never takes a slot.
        assert_eq!(queue.start(&mut sets, &heat, 2), None);
        assert!(queue.is_empty());
        assert!(sets[0].contains(0) && sets[0].contains(1), "nobody was evicted for it");
    }

    #[test]
    fn guessed_experts_in_the_tier_are_protected_and_not_queued() {
        let (mut sets, heat) = tiers(2);
        let mut queue = GuessQueue::new();
        // Expert 0 is in the tier and is guessed for step 1; expert 4 is not.
        queue.guess(&mut sets, 0, &[0, 4], 1);
        assert_eq!(queue.len(), 1);
        let started = queue.start(&mut sets, &heat, 1).unwrap();
        assert_eq!(started.swap.evict, Some(1), "expert 0 is needed, expert 1 gives way");
    }

    #[test]
    fn a_guess_without_room_waits_and_one_that_arrived_by_another_way_is_dropped() {
        let (mut sets, heat) = tiers(2);
        let mut queue = GuessQueue::new();
        queue.guess(&mut sets, 0, &[0, 1, 4], 1);
        queue.guess(&mut sets, 1, &[5], 1);
        // Layer 0's slots both hold experts this step needs: expert 4 finds
        // no room, but the guess for layer 1 can go ahead.
        let started = queue.start(&mut sets, &heat, 1).unwrap();
        assert_eq!(started.layer, 1);
        assert_eq!(queue.len(), 1, "the guess for layer 0 still waits");
        assert_eq!(queue.start(&mut sets, &heat, 1), None);

        // Expert 4 is demanded and read by the model before the guess is tried.
        sets[0].routed(&[4], 1);
        let Start::Slot(swap) = sets[0].start_read(4, Want::Demand, 1, heat.layer(0)) else {
            panic!("a demand read finds room")
        };
        sets[0].commit(&swap);
        assert_eq!(queue.start(&mut sets, &heat, 1), None);
        assert!(queue.is_empty());
    }
}
