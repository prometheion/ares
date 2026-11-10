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

    fn layer_mut(&mut self, layer: usize) -> &mut [f32] {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Empty,
    /// The expert's data is on its way; nobody may use the slot.
    Loading(u32),
    Holds(u32),
}

const NO_SLOT: u32 = u32::MAX;

/// One tier of one layer: a fixed number of slots and the experts in them.
///
/// A swap takes effect in two steps. [`SlotSet::elect`] removes the evicted
/// expert at once, so no kernel reads a slot that is being overwritten, and
/// [`SlotSet::commit`] makes the new expert visible once its data is in
/// place. Between the two the new expert is served from the tier below, as
/// before.
#[derive(Clone, Debug)]
pub struct SlotSet {
    /// Expert -> slot, or `NO_SLOT`. Only experts whose data is in place.
    slot_of: Vec<u32>,
    slots: Vec<Slot>,
}

impl SlotSet {
    /// An empty tier of `n_slots` slots for a layer of `n_experts` experts.
    pub fn new(n_experts: usize, n_slots: usize) -> Self {
        Self { slot_of: vec![NO_SLOT; n_experts], slots: vec![Slot::Empty; n_slots] }
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
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

    /// Writes a bias for the router's scores: 0 for experts in this tier and
    /// `-penalty` for the others.
    ///
    /// Added before the top-k, a large penalty restricts routing to the tier
    /// (hard pruning); the router kernel can exempt its strongest raw picks
    /// from the bias, which keeps most of the quality (protected pruning). A
    /// small penalty only breaks near-ties in favour of experts that are
    /// already there.
    pub fn write_route_bias(&self, out: &mut [f32], penalty: f32) {
        for (out, slot) in out.iter_mut().zip(&self.slot_of) {
            *out = if *slot == NO_SLOT { -penalty } else { 0.0 };
        }
    }

    /// Re-decides who holds the slots from `heat` and returns the swaps to
    /// carry out. Empty slots are filled with the hottest experts that have
    /// any heat; then hot outsiders replace cold members as `rule` allows.
    pub fn elect(&mut self, heat: &[f32], rule: &SwapRule) -> Vec<Swap> {
        assert_eq!(heat.len(), self.slot_of.len(), "one heat value per expert");
        let loading: Vec<u32> = self
            .slots
            .iter()
            .filter_map(|s| match s {
                Slot::Loading(e) => Some(*e),
                _ => None,
            })
            .collect();

        // Outsiders, hottest first; ties go to the lower number so elections
        // are reproducible.
        let mut outsiders: Vec<u32> = (0..heat.len() as u32)
            .filter(|e| self.slot_of[*e as usize] == NO_SLOT && !loading.contains(e))
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
                    self.slots[slot] = Slot::Loading(expert);
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
                Slot::Holds(e) => Some((slot as u32, *e)),
                _ => None,
            })
            .collect();
        members
            .sort_by(|a, b| heat[a.1 as usize].total_cmp(&heat[b.1 as usize]).then(b.1.cmp(&a.1)));

        for (slot, member) in members.into_iter().take(rule.max_swaps) {
            let Some(&candidate) = outsiders.peek() else { break };
            let (heat_in, heat_out) = (heat[candidate as usize], heat[member as usize]);
            if heat_in <= rule.margin * heat_out + rule.bonus {
                break;
            }
            outsiders.next();
            self.slot_of[member as usize] = NO_SLOT;
            self.slots[slot as usize] = Slot::Loading(candidate);
            swaps.push(Swap { slot, evict: Some(member), load: candidate });
        }
        swaps
    }

    /// Makes the expert of a finished swap visible in its slot.
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
        self.slot_of[swap.load as usize] = swap.slot;
    }

    /// Gives up a swap whose data could not be brought in: the slot is empty
    /// again and a later election may fill it.
    pub fn abort(&mut self, swap: &Swap) {
        let slot = &mut self.slots[swap.slot as usize];
        assert_eq!(
            *slot,
            Slot::Loading(swap.load),
            "slot {} is not loading this expert",
            swap.slot
        );
        *slot = Slot::Empty;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn maps_and_route_bias_follow_membership() {
        let mut set = SlotSet::new(4, 2);
        elect_and_commit(&mut set, &[0.0, 5.0, 0.0, 7.0], &SwapRule::default());
        let mut map = [0i32; 4];
        set.write_slot_map(&mut map);
        assert_eq!(map, [-1, 1, -1, 0]);
        let mut bias = [9.0f32; 4];
        set.write_route_bias(&mut bias, 1e30);
        assert_eq!(bias, [-1e30, 0.0, -1e30, 0.0]);
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
}
