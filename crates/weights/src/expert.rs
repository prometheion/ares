//! Heat of experts and the election of who holds a slot.

/// How often each expert of each layer was routed to, recent picks counting
/// more.
///
/// Routing is local in time (an expert used now was very likely used in the
/// last few dozen tokens), so a decayed count predicts the next tokens well
/// and needs no model of the text.
#[derive(Clone, Debug)]
pub struct ExpertHeat {
    n_experts: usize,
    heat: Vec<f32>,
    decay: f32,
}

impl ExpertHeat {
    /// Heat for `n_layers` layers of `n_experts` experts. Every call of
    /// [`ExpertHeat::decay`] multiplies all heat by `decay` (0.9 with a call
    /// every 32 tokens forgets a topic in a few hundred tokens).
    pub fn new(n_layers: usize, n_experts: usize, decay: f32) -> Self {
        assert!((0.0..=1.0).contains(&decay), "decay must be in [0, 1]");
        Self { n_experts, heat: vec![0.0; n_layers * n_experts], decay }
    }

    pub fn n_layers(&self) -> usize {
        self.heat.len().checked_div(self.n_experts).unwrap_or(0)
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Starts a layer from a routing profile: `counts[e]` picks of expert
    /// `e`, scaled so the layer's total is `total`. A profile recorded once
    /// for a model fills the tiers sensibly before the first token.
    pub fn seed(&mut self, layer: usize, counts: &[u32], total: f32) {
        let sum: u64 = counts.iter().map(|c| u64::from(*c)).sum();
        let scale = if sum == 0 { 0.0 } else { total / sum as f32 };
        for (heat, count) in self.layer_mut(layer).iter_mut().zip(counts) {
            *heat = *count as f32 * scale;
        }
    }

    /// Records the experts the router picked in `layer`, for one token or
    /// for many.
    pub fn record(&mut self, layer: usize, experts: &[u32]) {
        let heat = self.layer_mut(layer);
        for expert in experts {
            heat[*expert as usize] += 1.0;
        }
    }

    /// Records the picks of many rows at once: `counts[e]` picks of expert
    /// `e` in `layer`, each counting `weight`. A prefill pass has the routing
    /// of a whole prompt per layer in this form; weighting the prompt's chunks
    /// by their distance from its end
    /// ([`recency_weight`](crate::recency_weight)) makes the tiers that are
    /// elected before the reply follow what the end of the prompt used.
    pub fn record_counts(&mut self, layer: usize, counts: &[u32], weight: f32) {
        for (heat, count) in self.layer_mut(layer).iter_mut().zip(counts) {
            *heat += *count as f32 * weight;
        }
    }

    /// Ages all heat by one step.
    pub fn decay(&mut self) {
        for heat in &mut self.heat {
            *heat *= self.decay;
        }
    }

    /// The heat of each expert of `layer`.
    pub fn layer(&self, layer: usize) -> &[f32] {
        &self.heat[layer * self.n_experts..][..self.n_experts]
    }

    /// The heat of each expert of `layer`, to set it: a tier below another
    /// keeps its own copy of the heat in which the experts the upper tier
    /// holds are worth nothing, so that they give way first.
    pub fn layer_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.heat[layer * self.n_experts..][..self.n_experts]
    }
}

/// When a hotter expert replaces a colder one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapRule {
    /// A swap needs `heat_in > margin * heat_out + bonus`. The margin and the
    /// bonus keep two experts of similar heat from trading places back and
    /// forth, which would cost an upload each time for nothing.
    pub margin: f32,
    pub bonus: f32,
    /// The most swaps one election starts, which bounds the upload traffic
    /// that competes with inference.
    pub max_swaps: usize,
}

impl Default for SwapRule {
    fn default() -> Self {
        Self { margin: 1.25, bonus: 1.0, max_swaps: 8 }
    }
}

/// A change of who holds a slot. The owner of the tier moves the data and
/// then calls [`SlotSet::commit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Swap {
    pub slot: u32,
    /// The expert that left the slot; `None` when the slot was empty.
    pub evict: Option<u32>,
    /// The expert whose data must be brought into the slot.
    pub load: u32,
}

/// Why a read is started, which decides what it may evict. See
/// [`SlotSet::start_read`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// The router picked the expert and the model waits for it.
    Demand,
    /// A prediction or the token table says the expert will be needed. It may
    /// turn out wrong.
    Guess,
    /// A guess that is read ahead only into room nobody uses: a free slot, or
    /// one whose member is worth nothing (`heat` gives it no more than
    /// [`SPARE_WORTH`]: an expert never used, a guess nobody used, a copy the
    /// tier above holds as well). It never takes the place of a member that a
    /// later step may need.
    Spare,
}

/// The most a member may be worth for a [`Want::Spare`] read to take its slot.
pub const SPARE_WORTH: f32 = 1e-6;

/// What [`SlotSet::start_read`] decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// The expert is in a slot already: there is nothing to read.
    Present,
    /// A read of the expert is on its way. A guess needs nothing more; a
    /// demand read should promote the one that is running.
    Loading,
    /// No slot may be taken for it now (all are loading, or hold experts
    /// this step still needs and this read may not evict them). A guess
    /// waits for a later try; a demand read goes to a staging buffer and the
    /// expert is not kept.
    NoRoom,
    /// Read the expert into this slot, then [`SlotSet::commit`] the swap, or
    /// [`SlotSet::abort`] it if the read failed.
    Slot(Swap),
}

/// A slot handed back by [`SlotSet::give_back`]. Its memory is the caller's
/// to free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Given {
    pub slot: u32,
    /// The expert that was in the slot and is gone; `None` for an empty slot.
    pub evicted: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// Given back: the slot's memory belongs to someone else.
    Gone,
    /// Given back and lent: [`SlotSet::take`] does not take it again, only
    /// [`SlotSet::end_loan`] does.
    Lent,
    Empty,
    /// The expert's data is on its way; nobody may use the slot.
    Loading(u32),
    Holds(u32),
}

const NO_SLOT: u32 = u32::MAX;

/// One tier of one layer: a fixed number of slots and the experts in them.
///
/// Who holds a slot is decided in two ways, and a tier may use both:
///
/// - **Elections.** [`SlotSet::elect`] looks at the heat of all experts at
///   once, between steps, and starts the swaps that put the hottest experts
///   in the slots. This keeps the tier in VRAM, fills any tier before the
///   first token, and fills a tier that is idle.
/// - **Reads.** [`SlotSet::start_read`] decides for one expert when its read
///   starts: a free slot, or else the coldest member that this step does not
///   need. This is how a tier in RAM in front of the SSD is fed, and a tier
///   in VRAM can be fed the same way (an upload on a miss).
///
/// They are one type because they share what makes a tier work: the slots,
/// the two-phase swap, the heat that says who is cold, and the slots'
/// states. A second type for the tier in RAM would copy all of that, and an
/// election is still what fills it before the first reply token. What the
/// reads add is small: which experts the current step still needs
/// ([`SlotSet::routed`], [`SlotSet::predicted`]), which members arrived as a
/// guess and were not used, and slots that can be given back and taken again
/// ([`SlotSet::give_back`], [`SlotSet::take`]). A tier in VRAM that is only
/// elected never marks anything, and behaves as it always did.
///
/// A swap takes effect in two steps. [`SlotSet::elect`] and
/// [`SlotSet::start_read`] remove the evicted expert at once, so no kernel
/// reads a slot that is being overwritten, and [`SlotSet::commit`] makes the
/// new expert visible once its data is in place. Between the two the new
/// expert is served from the tier below, as before.
///
/// # A tier fed by reads
///
/// What the owner calls, and when it knows to call it, for one layer:
///
/// - The router of the layer has run for a step: [`SlotSet::routed`] with
///   what it picked, then [`SlotSet::start_read`] with [`Want::Demand`] for
///   each pick that is not [`SlotSet::contains`]ed.
/// - The predictor (or the token table) names experts for this layer before
///   its router has run: [`SlotSet::predicted`]. The ones that are in the
///   tier stay; the others wait in a [`GuessQueue`](crate::GuessQueue),
///   which calls [`SlotSet::start_read`] with [`Want::Guess`] only when an
///   I/O lane is free.
/// - A read has arrived: [`SlotSet::commit`].
///
/// ```
/// use ares_weights::{ExpertHeat, SlotSet, Start, Want};
///
/// let mut heat = ExpertHeat::new(1, 16, 0.9);
/// let mut tier = SlotSet::new(16, 2);
///
/// // Step 7: the router picked experts 3 and 9, and neither is in the tier.
/// let routed = [3, 9];
/// heat.record(0, &routed);
/// tier.routed(&routed, 7);
/// for expert in routed {
///     let Start::Slot(swap) = tier.start_read(expert, Want::Demand, 7, heat.layer(0)) else {
///         unreachable!("two free slots");
///     };
///     // Read the expert into the slot, wait for it, then:
///     tier.commit(&swap);
/// }
/// assert!(tier.contains(3) && tier.contains(9));
///
/// // Step 8 needs expert 5. Both members were picked before, but only expert
/// // 3 is still needed (it is predicted again), so expert 9 gives way.
/// tier.predicted(&[3], 8);
/// heat.record(0, &[5]);
/// let Start::Slot(swap) = tier.start_read(5, Want::Demand, 8, heat.layer(0)) else {
///     unreachable!("a member is cold");
/// };
/// assert_eq!(swap.evict, Some(9));
/// ```
#[derive(Clone, Debug)]
pub struct SlotSet {
    /// Expert -> slot, or `NO_SLOT`. Only experts whose data is in place.
    slot_of: Vec<u32>,
    /// Expert -> slot, or `NO_SLOT`. Only experts whose data is on its way.
    loading_in: Vec<u32>,
    /// Expert -> the first step that no longer needs it (0: never needed).
    protected_until: Vec<u64>,
    /// Expert -> the first step whose router did not pick it ([`SlotSet::routed`]; 0:
    /// never picked). Only read where [`SlotSet::keep_picks`] was set.
    picked_until: Vec<u64>,
    /// Whether a read never evicts an expert the running step picked
    /// ([`SlotSet::keep_picks`]).
    keeps_picks: bool,
    slots: Vec<Slot>,
    /// Per slot: the expert in it came as a guess and nobody has used it.
    unused_guess: Vec<bool>,
    /// Per slot: the expert in it is on loan ([`SlotSet::lend`]).
    loaned: Vec<bool>,
    /// Expert -> how many copies out of its slot are running
    /// ([`SlotSet::hold`]).
    held: Vec<u32>,
}

impl SlotSet {
    /// An empty tier of `n_slots` slots for a layer of `n_experts` experts.
    pub fn new(n_experts: usize, n_slots: usize) -> Self {
        Self {
            slot_of: vec![NO_SLOT; n_experts],
            loading_in: vec![NO_SLOT; n_experts],
            protected_until: vec![0; n_experts],
            picked_until: vec![0; n_experts],
            keeps_picks: false,
            slots: vec![Slot::Empty; n_slots],
            unused_guess: vec![false; n_slots],
            loaned: vec![false; n_slots],
            held: vec![0; n_experts],
        }
    }

    /// Makes every read keep the experts the running step picked ([`SlotSet::routed`]),
    /// however urgent: a demand read that finds nothing else to evict gets
    /// [`Start::NoRoom`]. For a tier whose slots the step's kernels read while it runs (VRAM,
    /// where a slot's entry is taken when the layer is routed): a layer with fewer slots than
    /// its step's picks serves the rest elsewhere instead of evicting what a kernel is about to
    /// read.
    pub fn keep_picks(&mut self) {
        self.keeps_picks = true;
    }

    /// The slots a read for step `step` can take without evicting an expert the step picked
    /// ([`SlotSet::routed`]): the empty ones and those of members that no copy holds and the
    /// step did not pick.
    pub fn room(&self, step: u64) -> usize {
        self.slots
            .iter()
            .filter(|s| match s {
                Slot::Empty => true,
                Slot::Holds(e) => self.held[*e as usize] == 0 && !self.picked(*e, step),
                _ => false,
            })
            .count()
    }

    fn picked(&self, expert: u32, step: u64) -> bool {
        self.picked_until[expert as usize] > step
    }

    /// What the members a read for step `step` can take the slot of (those
    /// [`SlotSet::room`] counts) are worth ([`SlotSet::start_read`]'s
    /// worth), in the order demand reads evict them: what a read that must
    /// not evict anything worth more than itself would take, one after the
    /// other.
    pub fn room_worths(&self, step: u64, heat: &[f32]) -> Vec<f32> {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");
        let mut members: Vec<(usize, u32)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s {
                Slot::Holds(e) if self.held[*e as usize] == 0 && !self.picked(*e, step) => {
                    Some((slot, *e))
                }
                _ => None,
            })
            .collect();
        members.sort_by(|a, b| self.eviction_order(*a, *b, step, heat));
        members.into_iter().map(|(slot, e)| self.worth(slot, e, heat)).collect()
    }

    /// The number of slots that hold nothing and wait for nothing.
    pub fn n_free(&self) -> usize {
        self.slots.iter().filter(|s| **s == Slot::Empty).count()
    }

    /// Whether a read of `expert` is on its way into a slot.
    pub fn is_loading(&self, expert: u32) -> bool {
        self.loading_in.get(expert as usize).is_some_and(|slot| *slot != NO_SLOT)
    }

    /// Keeps `expert` in its slot until [`SlotSet::release`]: its bytes are
    /// being copied out of the slot (to the tier above), so nothing may be
    /// read into the slot meanwhile. No read, election or
    /// [`SlotSet::give_back`] evicts a held member. Holds count: every hold
    /// needs its release.
    ///
    /// # Panics
    ///
    /// Panics when the expert is not in a slot.
    pub fn hold(&mut self, expert: u32) {
        assert!(self.contains(expert), "only a member can be held");
        self.held[expert as usize] += 1;
    }

    /// Ends one [`SlotSet::hold`] of `expert`.
    ///
    /// # Panics
    ///
    /// Panics when the expert is not held.
    pub fn release(&mut self, expert: u32) {
        let held = &mut self.held[expert as usize];
        assert!(*held > 0, "the expert is not held");
        *held -= 1;
    }

    /// The slots the tier has: those not given back with
    /// [`SlotSet::give_back`]. Slot numbers are stable, so after slots were
    /// given back the numbers in use have gaps.
    pub fn n_slots(&self) -> usize {
        self.slots.iter().filter(|s| !matches!(s, Slot::Gone | Slot::Lent)).count()
    }

    /// Keeps `slot`, which was given back ([`SlotSet::give_back`]), from being
    /// taken again until [`SlotSet::end_loan`]: its memory is lent for longer
    /// than the caller's next [`SlotSet::take`]. False when the slot was not
    /// given back.
    pub fn lend(&mut self, slot: u32) -> bool {
        match self.slots.get_mut(slot as usize) {
            Some(state @ (Slot::Gone | Slot::Lent)) => {
                *state = Slot::Lent;
                true
            }
            _ => false,
        }
    }

    /// Ends the loan of `slot` ([`SlotSet::lend`]): it is given back as before
    /// and the next [`SlotSet::take`] may take it.
    pub fn end_loan(&mut self, slot: u32) {
        if let Some(state @ Slot::Lent) = self.slots.get_mut(slot as usize) {
            *state = Slot::Gone;
        }
    }

    /// The slot that holds `expert`, if its data is in place.
    pub fn slot(&self, expert: u32) -> Option<u32> {
        match self.slot_of.get(expert as usize) {
            Some(&slot) if slot != NO_SLOT => Some(slot),
            _ => None,
        }
    }

    pub fn contains(&self, expert: u32) -> bool {
        self.slot(expert).is_some()
    }

    /// The number of experts whose data is in place.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| matches!(s, Slot::Holds(_))).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The expert -> slot map as kernels read it: the slot, or -1.
    pub fn write_slot_map(&self, out: &mut [i32]) {
        for (out, slot) in out.iter_mut().zip(&self.slot_of) {
            *out = if *slot == NO_SLOT { -1 } else { *slot as i32 };
        }
    }

    /// Sets `out[e]` to 1 for every expert whose data is in place or on its
    /// way into a slot, and leaves the others as they are: a pick of such an
    /// expert needs no read from the file. The router's pruning reads this, a
    /// byte per expert over every tier (`ares_cuda::PruneLayer::cached`).
    pub fn mark_present(&self, out: &mut [u8]) {
        for ((out, slot), loading) in out.iter_mut().zip(&self.slot_of).zip(&self.loading_in) {
            if *slot != NO_SLOT || *loading != NO_SLOT {
                *out = 1;
            }
        }
    }

    /// How much a member is worth when it competes for its slot: its heat,
    /// or nothing when it came as a guess and nobody used it, or is on loan.
    fn worth(&self, slot: usize, expert: u32, heat: &[f32]) -> f32 {
        if self.unused_guess[slot] || self.loaned[slot] { 0.0 } else { heat[expert as usize] }
    }

    /// Puts `expert`, which is in a slot or on its way into one, on loan: it
    /// serves the steps that pick it, and its slot is the first a read takes
    /// (it is worth nothing, however hot), also when a step uses it again.
    /// For an expert a step missed that is uploaded for that step without
    /// being worth a member's place: the slots of such experts are passed on
    /// from miss to miss, and the members stay. [`SlotSet::keep_loan`] makes
    /// it a member.
    ///
    /// # Panics
    ///
    /// Panics when the expert is in no slot.
    pub fn lend(&mut self, expert: u32) {
        let slot = match (self.slot_of[expert as usize], self.loading_in[expert as usize]) {
            (NO_SLOT, NO_SLOT) => panic!("only an expert in a slot can be lent"),
            (NO_SLOT, loading) => loading,
            (held, _) => held,
        };
        self.loaned[slot as usize] = true;
    }

    /// The coldest member that is worth something (not on loan, not a guess
    /// nobody used), that no copy holds and that step `step` did not pick,
    /// with its heat: the member a newcomer has to beat.
    pub fn coldest_member(&self, step: u64, heat: &[f32]) -> Option<(u32, f32)> {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s {
                Slot::Holds(e)
                    if self.held[*e as usize] == 0
                        && !self.picked(*e, step)
                        && !self.loaned[slot]
                        && !self.unused_guess[slot] =>
                {
                    Some((*e, heat[*e as usize]))
                }
                _ => None,
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
    }

    /// Whether `expert` is in a slot on loan.
    pub fn is_loaned(&self, expert: u32) -> bool {
        self.slot(expert).is_some_and(|slot| self.loaned[slot as usize])
    }

    /// The members on loan.
    pub fn loans(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.iter().zip(&self.loaned).filter_map(|(s, loaned)| match s {
            Slot::Holds(e) if *loaned => Some(*e),
            _ => None,
        })
    }

    /// Makes `expert`, a member on loan, a member like any other, and puts
    /// `instead`, a member, on loan in its place: a promotion that moves no
    /// bytes.
    ///
    /// # Panics
    ///
    /// Panics when `expert` is not on loan or `instead` is in no slot.
    pub fn keep_loan(&mut self, expert: u32, instead: u32) {
        assert!(self.is_loaned(expert), "only a member on loan can be kept");
        let (kept, lent) = (self.slot_of[expert as usize], self.slot_of[instead as usize]);
        assert_ne!(lent, NO_SLOT, "only a member can go on loan");
        self.loaned[kept as usize] = false;
        self.loaned[lent as usize] = true;
    }

    /// Re-decides who holds the slots from `heat` and returns the swaps to
    /// carry out. Empty slots are filled with the hottest experts that have
    /// any heat; then hot outsiders replace cold members as `rule` allows.
    ///
    /// A member that came as a guess and was not used competes with a worth
    /// of 0. An election looks at the heat only, not at what the current
    /// step needs: run it between steps, before the next step's experts are
    /// named with [`SlotSet::predicted`].
    pub fn elect(&mut self, heat: &[f32], rule: &SwapRule) -> Vec<Swap> {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");

        // Outsiders, hottest first; ties go to the lower number so elections
        // are reproducible.
        let mut outsiders: Vec<u32> = (0..heat.len() as u32)
            .filter(|e| {
                self.slot_of[*e as usize] == NO_SLOT && self.loading_in[*e as usize] == NO_SLOT
            })
            .collect();
        outsiders.sort_by(|a, b| heat[*b as usize].total_cmp(&heat[*a as usize]).then(a.cmp(b)));
        let mut outsiders = outsiders.into_iter().peekable();

        let mut swaps = Vec::new();
        for slot in 0..self.slots.len() {
            if self.slots[slot] != Slot::Empty {
                continue;
            }
            match outsiders.peek() {
                Some(&expert) if heat[expert as usize] > 0.0 => {
                    outsiders.next();
                    self.begin(slot, expert, false);
                    swaps.push(Swap { slot: slot as u32, evict: None, load: expert });
                }
                _ => break,
            }
        }

        // Members, coldest first.
        let mut members: Vec<(u32, u32)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s {
                Slot::Holds(e) if self.held[*e as usize] == 0 => Some((slot as u32, *e)),
                _ => None,
            })
            .collect();
        members.sort_by(|a, b| {
            let (worth_a, worth_b) =
                (self.worth(a.0 as usize, a.1, heat), self.worth(b.0 as usize, b.1, heat));
            worth_a.total_cmp(&worth_b).then(b.1.cmp(&a.1))
        });

        for (slot, member) in members.into_iter().take(rule.max_swaps) {
            let Some(&candidate) = outsiders.peek() else { break };
            let (heat_in, heat_out) =
                (heat[candidate as usize], self.worth(slot as usize, member, heat));
            if heat_in <= rule.margin * heat_out + rule.bonus {
                break;
            }
            outsiders.next();
            self.slot_of[member as usize] = NO_SLOT;
            self.begin(slot as usize, candidate, false);
            swaps.push(Swap { slot, evict: Some(member), load: candidate });
        }
        swaps
    }

    /// The slot starts to wait for `expert`'s data.
    fn begin(&mut self, slot: usize, expert: u32, guess: bool) {
        self.slots[slot] = Slot::Loading(expert);
        self.loading_in[expert as usize] = slot as u32;
        self.unused_guess[slot] = guess;
        self.loaned[slot] = false;
    }

    /// Says that the router picked `experts` in this layer in step `step`:
    /// they are needed until the step is over, and a member that came as a
    /// guess has now been used. Call it with the router's output, before
    /// starting the demand reads for it.
    ///
    /// `step` is any number that grows from step to step.
    pub fn routed(&mut self, experts: &[u32], step: u64) {
        self.predicted(experts, step);
        for expert in experts {
            let picked = &mut self.picked_until[*expert as usize];
            *picked = (*picked).max(step.saturating_add(1));
            let slot = match (self.slot_of[*expert as usize], self.loading_in[*expert as usize]) {
                (NO_SLOT, NO_SLOT) => continue,
                (NO_SLOT, loading) => loading,
                (held, _) => held,
            };
            self.unused_guess[slot as usize] = false;
        }
    }

    /// Says that `experts` are predicted for this layer in step `step`
    /// (before its router has run): they are needed until the step is over.
    /// A guess never evicts them, and a demand read only when nothing else is
    /// left.
    ///
    /// Call it for every predicted expert, in the tier or not: one that is
    /// read later is protected when it arrives. Protection ends with the
    /// step, so there is nothing to undo, and a prediction for the next step
    /// (the token table, made when a token is sampled) does not protect
    /// anything of the step that is ending.
    pub fn predicted(&mut self, experts: &[u32], step: u64) {
        let until = step.saturating_add(1);
        for expert in experts {
            let protected = &mut self.protected_until[*expert as usize];
            *protected = (*protected).max(until);
        }
    }

    /// Says that no step needs `expert` in this tier any more: its data has
    /// been copied to the tier above, so the copy here may give way like any
    /// other member. It ends what [`SlotSet::routed`] and
    /// [`SlotSet::predicted`] said about the expert.
    pub fn unprotect(&mut self, expert: u32) {
        self.protected_until[expert as usize] = 0;
        self.picked_until[expert as usize] = 0;
    }

    fn needed(&self, expert: u32, step: u64) -> bool {
        self.protected_until[expert as usize] > step
    }

    /// Starts the read of `expert` for step `step`: finds the slot it will
    /// be read into.
    ///
    /// `heat` is the layer's heat (what [`ExpertHeat::layer`] returns). The
    /// slot is chosen now, when the read starts and not earlier: a guess
    /// that waits for a free I/O lane has taken nothing from the tier.
    ///
    /// - A free slot is used first.
    /// - Otherwise the victim is the coldest member that is in a slot, is
    ///   not loading and is not needed by this step ([`SlotSet::routed`],
    ///   [`SlotSet::predicted`]). A member that came as a guess and was
    ///   never used counts as worth nothing, however hot the expert is.
    ///   Equal heat gives way in favour of the lower expert number.
    /// - A guess never evicts an expert this step needs: [`Start::NoRoom`]. A
    ///   [`Want::Spare`] guess evicts only a member worth nothing.
    /// - A demand read evicts one only when every other member is needed
    ///   too, and then the coldest of them: the expert it waits for is more
    ///   urgent than a prediction. Where [`SlotSet::keep_picks`] was set it
    ///   never evicts one the step picked, and gets [`Start::NoRoom`] instead.
    ///
    /// A demand read marks the expert as needed and used, like
    /// [`SlotSet::routed`], and turns a guess that is already on its way
    /// into one the model waits for. The evicted expert is gone at once, as
    /// in an election.
    ///
    /// A read that went to a staging buffer ([`Start::NoRoom`], or one made
    /// before the tier had room) is admitted afterwards the same way: call
    /// this once the data is there, copy it into the slot, and commit.
    pub fn start_read(&mut self, expert: u32, want: Want, step: u64, heat: &[f32]) -> Start {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");
        if want == Want::Demand {
            self.routed(&[expert], step);
        }
        if self.slot_of[expert as usize] != NO_SLOT {
            return Start::Present;
        }
        if self.loading_in[expert as usize] != NO_SLOT {
            return Start::Loading;
        }

        let guess = want != Want::Demand;
        if let Some(slot) = self.slots.iter().position(|s| *s == Slot::Empty) {
            self.begin(slot, expert, guess);
            return Start::Slot(Swap { slot: slot as u32, evict: None, load: expert });
        }
        let victim = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s {
                Slot::Holds(e)
                    if self.held[*e as usize] == 0
                        && !(self.keeps_picks && self.picked(*e, step)) =>
                {
                    Some((slot, *e))
                }
                _ => None,
            })
            .min_by(|a, b| self.eviction_order(*a, *b, step, heat));
        let Some((slot, victim)) = victim else { return Start::NoRoom };
        if guess && self.needed(victim, step) {
            return Start::NoRoom;
        }
        if want == Want::Spare && self.worth(slot, victim, heat) > SPARE_WORTH {
            return Start::NoRoom;
        }
        self.slot_of[victim as usize] = NO_SLOT;
        self.begin(slot, expert, guess);
        Start::Slot(Swap { slot: slot as u32, evict: Some(victim), load: expert })
    }

    /// The member a demand read would evict now, with what it is worth
    /// ([`SlotSet::start_read`]'s order) and whether step `step` still needs
    /// it. `None` when a slot is empty or no member may go: for a caller that
    /// brings an expert in only if it is worth more than what leaves.
    pub fn victim(&self, step: u64, heat: &[f32]) -> Option<(u32, f32, bool)> {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");
        if self.slots.contains(&Slot::Empty) {
            return None;
        }
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s {
                Slot::Holds(e)
                    if self.held[*e as usize] == 0
                        && !(self.keeps_picks && self.picked(*e, step)) =>
                {
                    Some((slot, *e))
                }
                _ => None,
            })
            .min_by(|a, b| self.eviction_order(*a, *b, step, heat))
            .map(|(slot, expert)| {
                (expert, self.worth(slot, expert, heat), self.needed(expert, step))
            })
    }

    /// The first of two members to go: not needed before needed, then the
    /// lower worth, then the higher expert number.
    fn eviction_order(
        &self,
        a: (usize, u32),
        b: (usize, u32),
        step: u64,
        heat: &[f32],
    ) -> std::cmp::Ordering {
        let (needed_a, needed_b) = (self.needed(a.1, step), self.needed(b.1, step));
        needed_a
            .cmp(&needed_b)
            .then_with(|| self.worth(a.0, a.1, heat).total_cmp(&self.worth(b.0, b.1, heat)))
            .then(b.1.cmp(&a.1))
    }

    /// Hands back up to `n` slots, because the memory broker asked for the
    /// RAM: empty slots first, then the coldest members in the order
    /// [`SlotSet::start_read`] would evict them (those the step still needs
    /// last). The experts in the slots are gone at once.
    ///
    /// Slots that are loading are kept, since a read is writing into their
    /// memory: when fewer than `n` come back, ask again after those reads
    /// have arrived (or were cancelled and aborted). The caller frees the
    /// memory of each [`Given`] slot, then reports it to the broker.
    pub fn give_back(&mut self, n: usize, step: u64, heat: &[f32]) -> Vec<Given> {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");
        let mut given = Vec::new();
        // The highest numbers go first, so the slots that stay are the low
        // ones, the ones that `take` brings back last.
        for slot in (0..self.slots.len()).rev() {
            if given.len() == n {
                return given;
            }
            if self.slots[slot] == Slot::Empty {
                self.slots[slot] = Slot::Gone;
                given.push(Given { slot: slot as u32, evicted: None });
            }
        }
        let mut members: Vec<(usize, u32)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s {
                Slot::Holds(e) if self.held[*e as usize] == 0 => Some((slot, *e)),
                _ => None,
            })
            .collect();
        members.sort_by(|a, b| self.eviction_order(*a, *b, step, heat));
        for (slot, expert) in members.into_iter().take(n - given.len()) {
            self.slot_of[expert as usize] = NO_SLOT;
            self.slots[slot] = Slot::Gone;
            self.unused_guess[slot] = false;
            self.loaned[slot] = false;
            given.push(Given { slot: slot as u32, evicted: Some(expert) });
        }
        given
    }

    /// Takes `n` slots again, or for the first time: the memory for them is
    /// there. Slots that were given back return first, lowest number first;
    /// then new numbers are added. They are empty; the slot numbers come
    /// back so the caller knows whose memory is whose.
    pub fn take(&mut self, n: usize) -> Vec<u32> {
        let mut taken = Vec::with_capacity(n);
        for (slot, state) in self.slots.iter_mut().enumerate() {
            if taken.len() == n {
                return taken;
            }
            if *state == Slot::Gone {
                *state = Slot::Empty;
                taken.push(slot as u32);
            }
        }
        while taken.len() < n {
            taken.push(self.slots.len() as u32);
            self.slots.push(Slot::Empty);
            self.unused_guess.push(false);
            self.loaned.push(false);
        }
        taken
    }

    /// Makes the expert of a finished swap visible in its slot. An expert
    /// that came as a guess stays marked as unused until
    /// [`SlotSet::routed`] says otherwise.
    ///
    /// # Panics
    ///
    /// Panics when the slot is not waiting for that expert, which means the
    /// swap is not one this set issued or was committed twice.
    pub fn commit(&mut self, swap: &Swap) {
        let slot = &mut self.slots[swap.slot as usize];
        assert_eq!(
            *slot,
            Slot::Loading(swap.load),
            "slot {} is not loading this expert",
            swap.slot
        );
        *slot = Slot::Holds(swap.load);
        self.loading_in[swap.load as usize] = NO_SLOT;
        self.slot_of[swap.load as usize] = swap.slot;
    }

    /// Gives up a swap whose data could not be brought in: the slot is empty
    /// again and a later election or read may fill it.
    pub fn abort(&mut self, swap: &Swap) {
        let slot = &mut self.slots[swap.slot as usize];
        assert_eq!(
            *slot,
            Slot::Loading(swap.load),
            "slot {} is not loading this expert",
            swap.slot
        );
        *slot = Slot::Empty;
        self.loading_in[swap.load as usize] = NO_SLOT;
        self.unused_guess[swap.slot as usize] = false;
        self.loaned[swap.slot as usize] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tier_that_keeps_picks_never_evicts_one_for_a_demand_read() {
        let heat = [1.0f32; 8];
        let mut set = SlotSet::new(8, 2);
        set.keep_picks();
        for expert in [0, 1] {
            let Start::Slot(swap) = set.start_read(expert, Want::Demand, 1, &heat) else {
                panic!("a free slot")
            };
            set.commit(&swap);
        }
        // Step 2 picks both members: no room for a third, and a demand read waits.
        set.routed(&[0, 1], 2);
        assert_eq!(set.room(2), 0);
        assert_eq!(set.start_read(5, Want::Demand, 2, &heat), Start::NoRoom);
        // In step 3 nothing is picked: both slots are room again.
        assert_eq!(set.room(3), 2);
        assert!(matches!(set.start_read(5, Want::Demand, 3, &heat), Start::Slot(_)));
        // A tier that does not keep picks evicts the coldest needed member, as before.
        let mut open = SlotSet::new(8, 1);
        let Start::Slot(swap) = open.start_read(0, Want::Demand, 1, &heat) else { panic!() };
        open.commit(&swap);
        open.routed(&[0], 2);
        assert!(matches!(open.start_read(5, Want::Demand, 2, &heat), Start::Slot(_)));
    }

    #[test]
    fn a_spare_read_takes_only_room_nobody_uses() {
        let mut heat = ExpertHeat::new(1, 8, 0.9);
        let mut set = SlotSet::new(8, 2);
        heat.record(0, &[0, 0]);
        for expert in [0, 1] {
            let Start::Slot(swap) = set.start_read(expert, Want::Demand, 1, heat.layer(0)) else {
                panic!("two free slots")
            };
            set.commit(&swap);
        }
        // Expert 0 is worth something, expert 1 nothing (never counted in the heat).
        let Start::Slot(swap) = set.start_read(5, Want::Spare, 3, heat.layer(0)) else {
            panic!("expert 1 is worth nothing")
        };
        assert_eq!(swap.evict, Some(1));
        set.commit(&swap);
        // A guess nobody used is worth nothing too; once a step has used it, both members
        // are worth something and a spare read finds no room where a guess does.
        heat.record(0, &[5]);
        set.routed(&[5], 3);
        assert_eq!(set.start_read(6, Want::Spare, 4, heat.layer(0)), Start::NoRoom);
        assert!(matches!(set.start_read(6, Want::Guess, 4, heat.layer(0)), Start::Slot(_)));
    }

    #[test]
    fn a_member_on_loan_gives_way_first_and_keeping_it_lends_another() {
        let heat = [0.0, 9.0, 5.0, 7.0, 8.0];
        let mut set = SlotSet::new(5, 3);
        for expert in [1, 2, 3] {
            let Start::Slot(swap) = set.start_read(expert, Want::Demand, 1, &heat) else {
                unreachable!("free slots");
            };
            set.commit(&swap);
        }
        // The hottest member goes on loan: a read takes its slot first.
        set.lend(1);
        assert!(set.is_loaned(1) && !set.is_loaned(2));
        assert_eq!(set.room_worths(2, &heat), [0.0, 5.0, 7.0]);
        let Start::Slot(swap) = set.start_read(4, Want::Demand, 2, &heat) else {
            unreachable!("a member on loan");
        };
        assert_eq!(swap.evict, Some(1));
        set.lend(4);
        set.commit(&swap);
        // A step that uses it leaves it on loan.
        set.routed(&[4], 3);
        assert!(set.is_loaned(4));
        assert_eq!(set.room_worths(4, &heat), [0.0, 5.0, 7.0]);
        assert_eq!(set.coldest_member(4, &heat), Some((2, 5.0)), "a member on loan is none");
        // Kept, it is a member, and the member named goes on loan in its place.
        set.keep_loan(4, 2);
        assert!(!set.is_loaned(4) && set.is_loaned(2));
        assert_eq!(set.coldest_member(4, &heat), Some((3, 7.0)));
        assert_eq!(set.loans().collect::<Vec<_>>(), [2]);
        let Start::Slot(swap) = set.start_read(1, Want::Demand, 4, &heat) else {
            unreachable!("a member on loan");
        };
        assert_eq!(swap.evict, Some(2));
        set.commit(&swap);
        assert!(!set.is_loaned(1), "a new member is not on loan");
    }

    #[test]
    fn the_victim_is_the_member_a_demand_read_would_evict() {
        let heat = [0.0, 5.0, 2.0, 9.0];
        let mut set = SlotSet::new(4, 2);
        assert_eq!(set.victim(1, &heat), None, "a slot is empty");
        for expert in [1, 2] {
            let Start::Slot(swap) = set.start_read(expert, Want::Demand, 1, &heat) else {
                unreachable!("free slots");
            };
            set.commit(&swap);
        }
        // Step 1 routed both: the colder one is a demand read's last resort.
        assert_eq!(set.victim(1, &heat), Some((2, 2.0, true)));
        // A later step needs neither.
        assert_eq!(set.victim(2, &heat), Some((2, 2.0, false)));
        // A member whose bytes are being copied out does not go.
        set.hold(2);
        assert_eq!(set.victim(2, &heat), Some((1, 5.0, false)));
        set.release(2);
        let Start::Slot(swap) = set.start_read(3, Want::Demand, 2, &heat) else {
            unreachable!("expert 2 can go");
        };
        assert_eq!(swap.evict, Some(2));
    }

    #[test]
    fn a_held_member_is_never_evicted() {
        let heat = [0.0, 5.0, 9.0, 7.0];
        let mut set = SlotSet::new(4, 2);
        for expert in [0, 1] {
            let Start::Slot(swap) = set.start_read(expert, Want::Demand, 1, &heat) else {
                unreachable!("free slots");
            };
            set.commit(&swap);
        }
        assert_eq!(set.n_free(), 0);
        // Expert 0 is the coldest, but its bytes are being copied out.
        set.hold(0);
        let Start::Slot(swap) = set.start_read(2, Want::Demand, 5, &heat) else {
            unreachable!("expert 1 can go");
        };
        assert_eq!(swap.evict, Some(1));
        assert!(set.is_loading(2) && !set.is_loading(0));
        set.commit(&swap);
        assert!(set.elect(&heat, &SwapRule { margin: 1.0, bonus: 0.0, max_swaps: 8 }).is_empty());
        assert_eq!(set.give_back(2, 9, &heat), [Given { slot: 1, evicted: Some(2) }]);
        set.release(0);
        assert_eq!(set.give_back(1, 9, &heat), [Given { slot: 0, evicted: Some(0) }]);
    }

    #[test]
    fn an_expert_that_went_to_the_tier_above_gives_way_to_a_guess() {
        let heat = [5.0, 5.0, 5.0];
        let mut set = SlotSet::new(3, 1);
        set.predicted(&[0], 4);
        let Start::Slot(swap) = set.start_read(0, Want::Guess, 4, &heat) else {
            unreachable!("a free slot");
        };
        set.commit(&swap);
        assert_eq!(
            set.start_read(1, Want::Guess, 4, &heat),
            Start::NoRoom,
            "the step needs expert 0"
        );
        set.unprotect(0);
        let Start::Slot(swap) = set.start_read(1, Want::Guess, 4, &heat) else {
            unreachable!("expert 0 is no longer needed here");
        };
        assert_eq!(swap.evict, Some(0));
    }

    fn elect_and_commit(set: &mut SlotSet, heat: &[f32], rule: &SwapRule) -> Vec<Swap> {
        let swaps = set.elect(heat, rule);
        for swap in &swaps {
            set.commit(swap);
        }
        swaps
    }

    #[test]
    fn a_profile_fills_the_slots_with_the_hottest_experts() {
        let mut heat = ExpertHeat::new(2, 6, 0.9);
        heat.seed(1, &[5, 0, 40, 1, 30, 0], 512.0);
        assert_eq!(heat.layer(0), [0.0; 6], "other layers are untouched");
        assert!((heat.layer(1).iter().sum::<f32>() - 512.0).abs() < 1e-3);

        let mut set = SlotSet::new(6, 3);
        let swaps = set.elect(heat.layer(1), &SwapRule::default());
        assert_eq!(swaps.iter().map(|s| s.load).collect::<Vec<_>>(), [2, 4, 0]);
        assert!(set.is_empty(), "nothing is usable before its data arrived");
        for swap in &swaps {
            set.commit(swap);
        }
        assert_eq!(
            (set.slot(2), set.slot(4), set.slot(0), set.slot(3)),
            (Some(0), Some(1), Some(2), None)
        );

        // Experts nobody ever picked do not take slots.
        let mut sparse = SlotSet::new(6, 3);
        assert_eq!(sparse.elect(&[0.0, 3.0, 0.0, 0.0, 0.0, 0.0], &SwapRule::default()).len(), 1);
    }

    #[test]
    fn a_swap_hides_the_old_expert_at_once_and_shows_the_new_one_on_commit() {
        let mut set = SlotSet::new(4, 1);
        elect_and_commit(&mut set, &[9.0, 0.0, 0.0, 0.0], &SwapRule::default());
        assert!(set.contains(0));

        let swaps = set.elect(&[1.0, 50.0, 0.0, 0.0], &SwapRule::default());
        assert_eq!(swaps, [Swap { slot: 0, evict: Some(0), load: 1 }]);
        assert!(!set.contains(0) && !set.contains(1), "the slot is in transit");
        // A second election while the data is on its way starts nothing.
        assert!(set.elect(&[1.0, 50.0, 60.0, 0.0], &SwapRule::default()).is_empty());
        set.commit(&swaps[0]);
        assert_eq!(set.slot(1), Some(0));

        // A swap that failed leaves the slot free for the next election.
        let swaps = set.elect(&[0.0, 1.0, 90.0, 0.0], &SwapRule::default());
        set.abort(&swaps[0]);
        assert!(set.is_empty());
        assert_eq!(set.elect(&[0.0, 1.0, 90.0, 0.0], &SwapRule::default())[0].load, 2);
    }

    #[test]
    fn similar_heat_does_not_trade_places() {
        let rule = SwapRule { margin: 1.25, bonus: 1.0, max_swaps: 8 };
        let mut set = SlotSet::new(4, 2);
        elect_and_commit(&mut set, &[10.0, 9.0, 0.0, 0.0], &rule);
        // 12 is hotter than 9, but not by the margin (1.25 * 9 + 1 = 12.25).
        assert!(set.elect(&[10.0, 9.0, 12.0, 0.0], &rule).is_empty());
        let swaps = set.elect(&[10.0, 9.0, 12.5, 0.0], &rule);
        assert_eq!(swaps, [Swap { slot: 1, evict: Some(1), load: 2 }]);
    }

    #[test]
    fn an_election_starts_no_more_swaps_than_allowed() {
        let rule = SwapRule { max_swaps: 2, ..SwapRule::default() };
        let mut set = SlotSet::new(8, 4);
        elect_and_commit(&mut set, &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], &rule);
        let moved = [0.0, 0.0, 0.0, 0.0, 9.0, 8.0, 7.0, 6.0];
        assert_eq!(elect_and_commit(&mut set, &moved, &rule).len(), 2);
        assert_eq!(elect_and_commit(&mut set, &moved, &rule).len(), 2);
        assert!(elect_and_commit(&mut set, &moved, &rule).is_empty());
        assert!((4..8).all(|e| set.contains(e)));
    }

    #[test]
    fn maps_and_presence_follow_membership() {
        let mut set = SlotSet::new(4, 2);
        elect_and_commit(&mut set, &[0.0, 5.0, 0.0, 7.0], &SwapRule::default());
        let mut map = [0i32; 4];
        set.write_slot_map(&mut map);
        assert_eq!(map, [-1, 1, -1, 0]);
        // Presence adds to what another tier marked (expert 2 here).
        let mut present = [0u8, 0, 1, 0];
        set.mark_present(&mut present);
        assert_eq!(present, [0, 1, 1, 1]);
    }

    /// The premise of the dynamic cache, on a synthetic workload: routing
    /// concentrates on a group of experts and the group changes over time (a
    /// conversation moving between topics). Re-electing from decayed heat
    /// must serve clearly more picks from the tier than a tier filled once.
    #[test]
    fn re_election_follows_a_workload_that_moves() {
        const EXPERTS: usize = 64;
        const SLOTS: usize = 16;
        let rule = SwapRule::default();
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut random = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        // 8 picks per token: mostly from the 12 experts of the current
        // topic, sometimes any expert.
        let mut picks = |topic: u64| -> Vec<u32> {
            (0..8)
                .map(|_| {
                    if random(10) < 8 {
                        ((topic * 12 + random(12)) % EXPERTS as u64) as u32
                    } else {
                        random(EXPERTS as u64) as u32
                    }
                })
                .collect()
        };

        // The topic changes abruptly every 500 tokens. Halving the heat at
        // every election (every 32 tokens) lets the new topic overtake the
        // old one within two elections; a slower decay follows more slowly.
        let mut heat = ExpertHeat::new(1, EXPERTS, 0.5);
        let mut fixed = SlotSet::new(EXPERTS, SLOTS);
        let mut dynamic = SlotSet::new(EXPERTS, SLOTS);
        // Both start from the same profile of the first topic.
        for _ in 0..200 {
            heat.record(0, &picks(0));
        }
        elect_and_commit(&mut fixed, heat.layer(0), &rule);
        elect_and_commit(&mut dynamic, heat.layer(0), &rule);

        let (mut fixed_hits, mut dynamic_hits, mut total) = (0u32, 0u32, 0u32);
        for token in 0..4000u64 {
            let routed = picks(token / 500);
            for expert in &routed {
                total += 1;
                fixed_hits += u32::from(fixed.contains(*expert));
                dynamic_hits += u32::from(dynamic.contains(*expert));
            }
            heat.record(0, &routed);
            if token % 32 == 31 {
                heat.decay();
                elect_and_commit(&mut dynamic, heat.layer(0), &rule);
            }
        }
        let (fixed_rate, dynamic_rate) =
            (f64::from(fixed_hits) / f64::from(total), f64::from(dynamic_hits) / f64::from(total));
        assert!(dynamic_rate > 0.65, "dynamic tier served {dynamic_rate:.2} of the picks");
        assert!(
            dynamic_rate > fixed_rate + 0.3,
            "dynamic {dynamic_rate:.2}, fixed {fixed_rate:.2}"
        );
        eprintln!("served from the tier: fixed {fixed_rate:.3}, dynamic {dynamic_rate:.3}");
    }

    /// Reads one expert into the tier the way the owner would.
    fn read(set: &mut SlotSet, expert: u32, want: Want, step: u64, heat: &[f32]) -> Option<Swap> {
        match set.start_read(expert, want, step, heat) {
            Start::Slot(swap) => {
                set.commit(&swap);
                Some(swap)
            }
            _ => None,
        }
    }

    #[test]
    fn a_read_takes_a_free_slot_and_then_the_coldest_member_the_step_does_not_need() {
        //                 expert: 0    1    2    3    4    5
        let heat = [0.0, 5.0, 0.0, 2.0, 1.0, 0.0];
        let mut set = SlotSet::new(6, 2);
        let first = read(&mut set, 1, Want::Demand, 1, &heat).unwrap();
        let second = read(&mut set, 3, Want::Demand, 1, &heat).unwrap();
        assert_eq!(
            (first, second),
            (Swap { slot: 0, evict: None, load: 1 }, Swap { slot: 1, evict: None, load: 3 }),
            "free slots first, lowest number first"
        );

        // Step 2 needs expert 4. Both members were needed in step 1 only.
        let Start::Slot(swap) = set.start_read(4, Want::Demand, 2, &heat) else {
            panic!("a member is cold")
        };
        assert_eq!(swap, Swap { slot: 1, evict: Some(3), load: 4 }, "the colder member goes");
        assert!(!set.contains(3) && !set.contains(4), "the slot is in transit");
        assert_eq!(set.start_read(4, Want::Demand, 2, &heat), Start::Loading);
        set.commit(&swap);
        assert_eq!(set.start_read(4, Want::Demand, 2, &heat), Start::Present);
        assert_eq!((set.slot(1), set.slot(4)), (Some(0), Some(1)));
    }

    #[test]
    fn a_guess_never_evicts_what_the_step_needs_and_a_demand_read_only_as_a_last_resort() {
        //                 expert: 0    1    2    3    4
        let heat = [0.0, 0.5, 9.0, 2.0, 0.0];
        let mut set = SlotSet::new(5, 3);
        for expert in [1, 2, 3] {
            read(&mut set, expert, Want::Demand, 1, &heat).unwrap();
        }

        // Step 2: expert 1 is cold but predicted for this step; 2 is hot and 3
        // lukewarm, and neither is needed. A guess takes the colder of the
        // two that are not needed, not the coldest of all.
        set.predicted(&[1], 2);
        let Start::Slot(swap) = set.start_read(4, Want::Guess, 2, &heat) else { panic!("room") };
        assert_eq!(swap.evict, Some(3));

        // On a tier where everything is needed, a guess finds no room.
        let mut set = SlotSet::new(5, 3);
        for expert in [1, 2, 3] {
            read(&mut set, expert, Want::Demand, 1, &heat).unwrap();
        }
        set.predicted(&[1, 2, 3], 2);
        assert_eq!(set.start_read(4, Want::Guess, 2, &heat), Start::NoRoom);
        // A demand read evicts the coldest of the needed ones.
        let Start::Slot(swap) = set.start_read(4, Want::Demand, 2, &heat) else {
            panic!("a demand read may evict as a last resort")
        };
        assert_eq!(swap.evict, Some(1));
    }

    #[test]
    fn protection_ends_with_the_step() {
        let heat = [0.0, 1.0, 5.0];
        let mut set = SlotSet::new(3, 1);
        read(&mut set, 1, Want::Demand, 3, &heat).unwrap();
        // Predicted for step 4, so it survives a guess in step 4 and is
        // fair game from step 5.
        set.predicted(&[1], 4);
        assert_eq!(set.start_read(2, Want::Guess, 3, &heat), Start::NoRoom);
        assert_eq!(set.start_read(2, Want::Guess, 4, &heat), Start::NoRoom);
        assert!(matches!(set.start_read(2, Want::Guess, 5, &heat), Start::Slot(_)));
    }

    #[test]
    fn a_guess_nobody_used_is_worth_nothing_and_a_used_one_is_worth_its_heat() {
        //                 expert: 0    1    2    3    4
        let heat = [0.0, 3.0, 8.0, 0.0, 6.0];
        let build = || {
            let mut set = SlotSet::new(5, 2);
            read(&mut set, 1, Want::Demand, 1, &heat).unwrap();
            read(&mut set, 2, Want::Guess, 1, &heat).unwrap();
            set
        };

        // Expert 2 is the hotter, but it came as a guess and no step used it.
        let mut unused = build();
        let Start::Slot(swap) = unused.start_read(4, Want::Demand, 2, &heat) else { panic!() };
        assert_eq!(swap.evict, Some(2));

        // The router picked it in step 1 after it had arrived: it competes by heat.
        let mut used = build();
        used.routed(&[2], 1);
        let Start::Slot(swap) = used.start_read(4, Want::Demand, 2, &heat) else { panic!() };
        assert_eq!(swap.evict, Some(1));

        // A demand read of a guess that is still on its way makes it used too.
        let mut set = SlotSet::new(5, 2);
        let Start::Slot(on_its_way) = set.start_read(2, Want::Guess, 1, &heat) else { panic!() };
        read(&mut set, 1, Want::Demand, 1, &heat).unwrap();
        assert_eq!(set.start_read(2, Want::Demand, 2, &heat), Start::Loading);
        set.commit(&on_its_way);
        let Start::Slot(swap) = set.start_read(4, Want::Demand, 3, &heat) else { panic!() };
        assert_eq!(swap.evict, Some(1));
    }

    #[test]
    fn a_slot_that_is_loading_is_nobodys_victim_and_a_failed_read_frees_it() {
        let heat = [0.0, 1.0, 2.0];
        let mut set = SlotSet::new(3, 1);
        let Start::Slot(swap) = set.start_read(1, Want::Guess, 1, &heat) else { panic!() };
        assert_eq!(set.start_read(1, Want::Guess, 1, &heat), Start::Loading);
        assert_eq!(set.start_read(2, Want::Demand, 1, &heat), Start::NoRoom);
        set.abort(&swap);
        assert!(matches!(set.start_read(2, Want::Demand, 1, &heat), Start::Slot(_)));
        // A tier with no slots has no room, whatever is asked.
        let mut none = SlotSet::new(3, 0);
        assert_eq!(none.start_read(1, Want::Demand, 1, &heat), Start::NoRoom);
    }

    #[test]
    fn slots_are_given_back_empty_first_then_the_coldest_and_the_needed_last() {
        //                 expert: 0    1    2    3
        let heat = [0.0, 9.0, 1.0, 4.0];
        let mut set = SlotSet::new(4, 5);
        read(&mut set, 1, Want::Demand, 1, &heat).unwrap(); // slot 0
        read(&mut set, 2, Want::Demand, 1, &heat).unwrap(); // slot 1
        let Start::Slot(loading) = set.start_read(3, Want::Demand, 1, &heat) else { panic!() };
        assert_eq!(loading.slot, 2);
        // Slots 3 and 4 are empty. Step 2 needs the cold expert 2 only.
        set.routed(&[2], 2);

        let given = set.give_back(2, 2, &heat);
        assert_eq!(
            given,
            [Given { slot: 4, evicted: None }, Given { slot: 3, evicted: None }],
            "empty slots go first, the highest numbers first"
        );
        assert_eq!(set.n_slots(), 3);
        // Expert 1 is hot but the step does not need it; expert 2 is cold but needed.
        assert_eq!(set.give_back(1, 2, &heat), [Given { slot: 0, evicted: Some(1) }]);
        assert!(!set.contains(1), "the expert is gone at once");
        // The slot that is loading stays, so fewer than asked come back.
        assert_eq!(set.give_back(5, 2, &heat), [Given { slot: 1, evicted: Some(2) }]);
        assert_eq!(set.n_slots(), 1);
        assert!(set.give_back(1, 2, &heat).is_empty());

        // Slots come back lowest number first, then new numbers are added.
        assert_eq!(set.take(2), [0, 1]);
        assert_eq!(set.take(3), [3, 4, 5]);
        assert_eq!(set.take(0), [] as [u32; 0]);
        assert_eq!(set.n_slots(), 6);
        set.commit(&loading);
        assert!(set.contains(3));
        // A taken slot is empty and gets used.
        let Start::Slot(swap) = set.start_read(1, Want::Demand, 3, &heat) else { panic!() };
        assert_eq!(swap, Swap { slot: 0, evict: None, load: 1 });
    }

    #[test]
    fn an_election_skips_slots_given_back_and_competes_with_unused_guesses_as_worth_nothing() {
        //                 expert: 0     1     2     3
        let heat = [10.0, 12.0, 9.0, 0.0];
        let rule = SwapRule::default();
        let mut set = SlotSet::new(4, 3);
        read(&mut set, 0, Want::Demand, 1, &heat).unwrap();
        read(&mut set, 1, Want::Guess, 1, &heat).unwrap();
        assert_eq!(set.give_back(1, 2, &heat), [Given { slot: 2, evicted: None }]);

        // Expert 2 (9) is not hot enough to replace expert 0 (10) ...
        // ... but expert 1 is an unused guess and worth nothing.
        let swaps = set.elect(&heat, &rule);
        assert_eq!(swaps, [Swap { slot: 1, evict: Some(1), load: 2 }]);
        assert_eq!(set.n_slots(), 2, "no election fills a slot that was given back");
    }
}
