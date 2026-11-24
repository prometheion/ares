//! One tier of the expert store: pools of slots and who holds them.

use std::collections::{HashMap, HashSet};

use ares_weights::{LayerNeed, SlotSet, SplitRule, split_slots};

/// Slots of one size in one place (a device, or host memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool {
    /// The device the pool is on; 0 for the tier in host memory.
    pub place: usize,
    /// Bytes from one slot to the next.
    pub stride: u64,
    pub slots: u32,
}

/// A tier: its pools, the slots each layer has in them and what they hold.
///
/// A layer's [`SlotSet`] numbers its slots from 0 in the order the layer got
/// them; `slot` maps that number to the slot of the layer's pool. Layers
/// whose slots have the same size in the same place share a pool, so a slot
/// can pass from one of them to another when their needs change. A layer with
/// a fixed number of slots has a pool of its own, which nothing passes in or
/// out of.
#[derive(Debug)]
pub(crate) struct Tier {
    pub pools: Vec<Pool>,
    pub pool_of: Vec<usize>,
    pub sets: Vec<SlotSet>,
    /// The fewest slots each layer keeps, also when slots pass between layers or go back
    /// to the system.
    pub floor: Vec<usize>,
    slot: Vec<Vec<u32>>,
}

/// What a slot number that a layer gave up for good has behind it: no slot of the pool.
pub(super) const NO_MEMORY: u32 = u32::MAX;

/// What lies in a slot that is vacated and stays ([`Tier::vacate`]): what `layer` holds
/// behind its number `number` lay in slot `from` of the pool and is to lie in slot `to`,
/// which the number names now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Move {
    pub layer: usize,
    pub number: u32,
    pub from: u32,
    pub to: u32,
}

/// What [`Tier::vacate`] did.
#[derive(Debug, Default)]
pub(crate) struct Vacated {
    /// What the slots the layers gave held, as (layer, expert, the slot of the pool it lay
    /// in): evicted, whether the slots were vacated or not.
    pub evicted: Vec<(usize, u32, u32)>,
    /// What moves, with the tier's numbers changed already; `None` when the layers could
    /// not give as many slots: they have taken back what they gave, and nothing is vacated.
    pub moves: Option<Vec<Move>>,
}

impl Tier {
    /// A tier for layers of `n_experts` experts: `layers[l]` is the place and
    /// the slot size of layer `l`, `budget[place]` the bytes the tier may
    /// take there, `floor[l]` the fewest slots layer `l` must have. `heat[l]` is
    /// what is known of each layer's routing; the slots of a pool are divided
    /// by it ([`split_slots`]), evenly when it says nothing.
    ///
    /// `fixed[l]`, where it is `Some`, is the number of slots layer `l` has
    /// whatever the heat says; their bytes come out of the budget of the
    /// layer's place first.
    ///
    /// Returns `None` when a budget does not hold the fixed slots and the
    /// floor of every other layer.
    pub fn plan(
        n_experts: usize,
        layers: &[(usize, u64)],
        budget: &[u64],
        floor: &[usize],
        heat: &[&[f32]],
        fixed: &[Option<usize>],
    ) -> Option<Self> {
        let fixed_of = |layer: usize| fixed.get(layer).copied().flatten();
        let mut pools: Vec<Pool> = Vec::new();
        let mut pool_of = Vec::with_capacity(layers.len());
        // Per pool, the slots it has whatever the heat says.
        let mut pool_fixed: Vec<Option<usize>> = Vec::new();
        for (layer, (place, stride)) in layers.iter().enumerate() {
            let found = match fixed_of(layer) {
                Some(_) => None,
                None => (0..pools.len()).find(|p| {
                    pools[*p].place == *place
                        && pools[*p].stride == *stride
                        && pool_fixed[*p].is_none()
                }),
            };
            pool_of.push(found.unwrap_or_else(|| {
                pools.push(Pool { place: *place, stride: *stride, slots: 0 });
                pool_fixed.push(fixed_of(layer));
                pools.len() - 1
            }));
        }
        let mut budget = budget.to_vec();
        for (pool, slots) in pools.iter().zip(&pool_fixed) {
            if let Some(slots) = slots {
                let left = budget.get_mut(pool.place)?;
                *left = left.checked_sub(pool.stride * *slots as u64)?;
            }
        }
        // A place's bytes go to its pools by the bytes one slot per layer
        // takes in each, so that every layer of the place starts with the
        // same number of slots.
        let mut counts = vec![0usize; layers.len()];
        for (index, pool) in pools.iter_mut().enumerate() {
            let members: Vec<usize> = (0..layers.len()).filter(|l| pool_of[*l] == index).collect();
            if let Some(slots) = pool_fixed[index] {
                for layer in &members {
                    counts[*layer] = slots;
                }
                pool.slots = (slots * members.len()) as u32;
                continue;
            }
            let per_round: u64 = layers
                .iter()
                .enumerate()
                .filter(|(layer, (place, _))| *place == pool.place && fixed_of(*layer).is_none())
                .map(|(_, (_, s))| *s)
                .sum();
            let rounds = budget.get(pool.place).copied().unwrap_or(0) / per_round.max(1);
            let slots = (rounds as usize).min(n_experts) * members.len();
            let floors: Vec<usize> = members.iter().map(|l| floor[*l]).collect();
            if slots < floors.iter().sum() {
                return None;
            }
            let needs: Vec<LayerNeed<'_>> =
                members.iter().map(|l| LayerNeed { heat: heat[*l], expert_bytes: 1 }).collect();
            let split = split_slots(&needs, slots as u64, None, &SplitRule::default());
            for (layer, n) in members.iter().zip(raise_to_floor(split, &floors)) {
                counts[*layer] = n;
            }
            pool.slots = slots as u32;
        }
        let mut next = vec![0u32; pools.len()];
        let slot = counts
            .iter()
            .zip(&pool_of)
            .map(|(n, pool)| {
                let first = next[*pool];
                next[*pool] += *n as u32;
                (first..first + *n as u32).collect()
            })
            .collect();
        let sets = counts.iter().map(|n| SlotSet::new(n_experts, *n)).collect();
        let floor = (0..layers.len()).map(|l| fixed_of(l).unwrap_or(floor[l])).collect();
        Some(Self { pools, pool_of, sets, floor, slot })
    }

    /// The slot of the layer's pool behind the layer's slot number `local`.
    pub fn slot(&self, layer: usize, local: u32) -> u32 {
        self.slot[layer][local as usize]
    }

    /// The slots of the layer's pool behind each of the layer's slot numbers.
    pub fn locals(&self, layer: usize) -> &[u32] {
        &self.slot[layer]
    }

    /// Puts slot `slot` of the layer's pool behind the layer's slot number
    /// `local`. The caller keeps every slot of the pool behind exactly one
    /// number of one layer.
    pub fn remap(&mut self, layer: usize, local: u32, slot: u32) {
        self.slot[layer][local as usize] = slot;
    }

    /// The bytes of all pools at `place`.
    pub fn bytes(&self, place: usize) -> u64 {
        self.pools.iter().filter(|p| p.place == place).map(|p| p.stride * u64::from(p.slots)).sum()
    }

    /// Slots of `layer`.
    pub fn slots(&self, layer: usize) -> usize {
        self.sets[layer].n_slots()
    }

    /// Moves a slot of the pool from layer `from` to layer `to` (both of the
    /// same pool): `local` is the number `from` gave back, and the result is
    /// the number `to` knows it by.
    pub fn move_slot(&mut self, from: usize, local: u32, to: usize) -> u32 {
        debug_assert_eq!(self.pool_of[from], self.pool_of[to]);
        let slot = self.slot[from][local as usize];
        let taken = self.sets[to].take(1)[0];
        if taken as usize == self.slot[to].len() {
            self.slot[to].push(slot);
            // `from` keeps its number given up, with no memory behind it: a slot is behind
            // one number of one layer only. (Until 2026-11-22 the number kept the slot, and
            // a loan of VRAM that came after took the given-up number for the slot's owner:
            // the layer that held the slot kept it inside the loan, and a later loan found
            // fewer slots given than lent, "as many slots were given as are lent".)
            self.slot[from][local as usize] = NO_MEMORY;
        } else {
            // A number `to` had given back earlier: swap the memory behind
            // the two numbers, so `from` keeps one for a later take.
            let other = std::mem::replace(&mut self.slot[to][taken as usize], slot);
            self.slot[from][local as usize] = other;
        }
        taken
    }

    /// `layer` takes `slots` of its pool into use again: slots it holds the
    /// numbers of and had given up. The numbers its set hands out are put in
    /// front of exactly these slots.
    pub fn retake(&mut self, layer: usize, slots: &[u32]) {
        if slots.is_empty() {
            return;
        }
        let wanted: HashSet<u32> = slots.iter().copied().collect();
        let taken = self.sets[layer].take(slots.len());
        let taken_numbers: HashSet<u32> = taken.iter().copied().collect();
        // Numbers that stay given up although one of the slots is behind them.
        let mut behind: Vec<u32> = self
            .locals(layer)
            .iter()
            .enumerate()
            .filter(|(number, slot)| {
                wanted.contains(slot) && !taken_numbers.contains(&(*number as u32))
            })
            .map(|(number, _)| number as u32)
            .collect();
        for number in taken {
            let has = self.slot(layer, number);
            if wanted.contains(&has) {
                continue;
            }
            let other = behind.pop().expect("a slot for every number taken");
            let slot = self.slot(layer, other);
            self.remap(layer, number, slot);
            self.remap(layer, other, has);
        }
    }

    /// Takes `slots` of a pool out of use for the pool's layers `members`, between steps,
    /// without taking them as they lie: every layer gives slots in proportion to what it
    /// has above its floor (`floor[i]` for `members[i]`; the largest remainders round),
    /// each its coldest by `worth(layer)` at `step`, and what lies in `slots` and stays
    /// moves into the slots given outside them, whose numbers change hands. The owner of
    /// the tier moves the bytes and lets the evicted experts go; a slot's number in its
    /// layer never changes, only the memory behind it. The RAM tier vacates the pieces it
    /// gives back to the system (`elastic.rs`), VRAM the slots it lends to a step's scratch
    /// (`scratch.rs`); [`Tier::reopen`] brings them back.
    pub fn vacate<'w>(
        &mut self,
        members: &[usize],
        slots: &[u32],
        floor: &[usize],
        step: u64,
        worth: impl Fn(usize) -> &'w [f32],
    ) -> Vacated {
        let mut vacated = Vacated::default();
        let spare: Vec<usize> =
            members.iter().zip(floor).map(|(l, f)| self.slots(*l).saturating_sub(*f)).collect();
        let total: usize = spare.iter().sum();
        if total < slots.len() {
            return vacated;
        }
        let mut quota: Vec<usize> = spare.iter().map(|s| s * slots.len() / total).collect();
        for _ in quota.iter().sum::<usize>()..slots.len() {
            let most = (0..members.len())
                .filter(|i| spare[*i] > quota[*i])
                .max_by_key(|i| spare[*i] - quota[*i])
                .expect("the layers have the spare slots that were counted");
            quota[most] += 1;
        }
        // Every layer gives its coldest.
        let mut given: Vec<(usize, u32)> = Vec::with_capacity(slots.len());
        for (layer, n) in members.iter().zip(&quota) {
            if *n == 0 {
                continue;
            }
            for slot in self.sets[*layer].give_back(*n, step, worth(*layer)) {
                if let Some(expert) = slot.evicted {
                    vacated.evicted.push((*layer, expert, self.slot(*layer, slot.slot)));
                }
                given.push((*layer, slot.slot));
            }
        }
        if given.len() < slots.len() {
            // A slot was busy after all: the layers take theirs back.
            for layer in members {
                let back: Vec<u32> = given
                    .iter()
                    .filter(|(l, _)| l == layer)
                    .map(|(_, local)| self.slot(*layer, *local))
                    .collect();
                self.retake(*layer, &back);
            }
            return vacated;
        }
        // What lies in the slots and stays moves into the slots given.
        let owner: HashMap<u32, (usize, u32)> = members
            .iter()
            .flat_map(|layer| {
                let locals = self.locals(*layer).iter().enumerate();
                locals.map(move |(local, slot)| (*slot, (*layer, local as u32)))
            })
            .collect();
        let vacating: HashSet<u32> = slots.iter().copied().collect();
        let given: Vec<(usize, u32, u32)> = given
            .into_iter()
            .map(|(layer, local)| (layer, local, self.slot(layer, local)))
            .collect();
        let gone: HashSet<u32> = given.iter().map(|(_, _, slot)| *slot).collect();
        let mut homes = given.iter().filter(|(_, _, slot)| !vacating.contains(slot));
        let mut moves = Vec::new();
        for slot in slots.iter().filter(|slot| !gone.contains(slot)) {
            let (stays, number) = owner[slot];
            let (gives, given_number, home) =
                *homes.next().expect("as many slots were given as are vacated");
            self.remap(stays, number, home);
            self.remap(gives, given_number, *slot);
            moves.push(Move { layer: stays, number, from: *slot, to: home });
        }
        vacated.moves = Some(moves);
        vacated
    }

    /// The vacated `slots` of `pool` go back into use, empty, for the layers that hold
    /// their numbers ([`Tier::vacate`]; a number set aside for a loan comes back as well).
    /// Returns the layers that took slots.
    pub fn reopen(&mut self, pool: usize, slots: &HashSet<u32>) -> Vec<usize> {
        let mut layers = Vec::new();
        for layer in 0..self.sets.len() {
            if self.pool_of[layer] != pool {
                continue;
            }
            let own: Vec<(u32, u32)> = (0..self.locals(layer).len() as u32)
                .map(|local| (local, self.slot(layer, local)))
                .filter(|(_, slot)| slots.contains(slot))
                .collect();
            if own.is_empty() {
                continue;
            }
            for (local, _) in &own {
                self.sets[layer].bring_back(*local);
            }
            let own: Vec<u32> = own.iter().map(|(_, slot)| *slot).collect();
            self.retake(layer, &own);
            layers.push(layer);
        }
        layers
    }
}

/// Raises every count below its floor to it and takes the slots from the
/// largest counts above theirs.
fn raise_to_floor(mut counts: Vec<usize>, floor: &[usize]) -> Vec<usize> {
    let mut owed = 0;
    for (n, floor) in counts.iter_mut().zip(floor) {
        if *n < *floor {
            owed += floor - *n;
            *n = *floor;
        }
    }
    while owed > 0 {
        let Some(largest) =
            (0..counts.len()).filter(|i| counts[*i] > floor[*i]).max_by_key(|i| counts[*i])
        else {
            break;
        };
        counts[largest] -= 1;
        owed -= 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_of_one_size_share_a_pool_and_start_even() {
        let flat = vec![1.0f32; 16];
        let heat: Vec<&[f32]> = vec![&flat; 4];
        // Two sizes on device 0, one layer on device 1.
        let layers = [(0, 100), (0, 60), (0, 100), (1, 100)];
        let tier = Tier::plan(16, &layers, &[1300, 450], &[2; 4], &heat, &[]).expect("fits");
        assert_eq!(tier.pools.len(), 3);
        assert_eq!(tier.pool_of, [0, 1, 0, 2]);
        // 1300 bytes / (100 + 60 + 100) per round = 5 slots per layer.
        assert_eq!((0..4).map(|l| tier.slots(l)).collect::<Vec<_>>(), [5, 5, 5, 4]);
        assert_eq!(tier.bytes(0), 1300);
        assert_eq!(tier.bytes(1), 400);
        assert_eq!((tier.slot(0, 0), tier.slot(2, 0), tier.slot(1, 4)), (0, 5, 4));
        assert!(Tier::plan(16, &layers, &[1300, 150], &[2; 4], &heat, &[]).is_none());
    }

    #[test]
    fn a_profile_moves_slots_to_the_layer_that_needs_them() {
        let narrow: Vec<f32> = (0..16).map(|e| if e < 2 { 50.0 } else { 0.0 }).collect();
        let wide = vec![6.0f32; 16];
        let tier = Tier::plan(16, &[(0, 10), (0, 10)], &[160], &[2; 2], &[&narrow, &wide], &[])
            .expect("fits");
        assert_eq!(tier.slots(0) + tier.slots(1), 16);
        assert!(tier.slots(1) > tier.slots(0), "{} {}", tier.slots(0), tier.slots(1));
        assert!(tier.slots(0) >= 2);
    }

    #[test]
    fn a_layer_with_fixed_slots_has_a_pool_of_its_own_out_of_the_budget() {
        let flat = vec![1.0f32; 16];
        let heat: Vec<&[f32]> = vec![&flat; 3];
        // Layer 0 keeps all 16 of its experts; it has the size of layer 1.
        let layers = [(0, 10), (0, 10), (0, 10)];
        let fixed = [Some(16), None, None];
        let tier = Tier::plan(16, &layers, &[300], &[2; 3], &heat, &fixed).expect("fits");
        assert_eq!(tier.pools.len(), 2);
        assert_eq!(tier.pool_of, [0, 1, 1]);
        assert_eq!((tier.slots(0), tier.slots(1), tier.slots(2)), (16, 7, 7));
        assert_eq!(tier.bytes(0), 160 + 140);
        // The fixed slots come first: what is left must hold the others' floor.
        assert!(Tier::plan(16, &layers, &[190], &[2; 3], &heat, &fixed).is_none());
        assert!(Tier::plan(16, &layers, &[150], &[2; 3], &heat, &fixed).is_none());
    }

    #[test]
    fn every_layer_gets_its_own_floor() {
        let narrow: Vec<f32> = (0..16).map(|e| if e < 2 { 50.0 } else { 0.0 }).collect();
        let wide = vec![6.0f32; 16];
        // The profile wants nearly every slot for layer 1; layer 0's floor of 6 holds.
        let tier = Tier::plan(16, &[(0, 10), (0, 10)], &[160], &[6, 2], &[&wide, &narrow], &[])
            .expect("fits");
        assert!(tier.slots(0) >= 6 && tier.slots(1) >= 2, "{} {}", tier.slots(0), tier.slots(1));
        assert_eq!(tier.floor, [6, 2]);
        assert!(
            Tier::plan(16, &[(0, 10), (0, 10)], &[160], &[9, 8], &[&wide, &narrow], &[]).is_none()
        );
    }

    #[test]
    fn a_slot_passes_between_layers() {
        let flat = vec![1.0f32; 8];
        let mut tier =
            Tier::plan(8, &[(0, 10), (0, 10)], &[80], &[1; 2], &[&flat, &flat], &[]).expect("fits");
        assert_eq!((tier.slots(0), tier.slots(1)), (4, 4));
        let given = tier.sets[0].give_back(1, 0, &flat);
        let taken = tier.move_slot(0, given[0].slot, 1);
        assert_eq!((tier.slots(0), tier.slots(1)), (3, 5));
        assert_eq!(tier.slot(1, taken), 3, "layer 1 now owns layer 0's last slot");
        assert_eq!(tier.slot(0, given[0].slot), NO_MEMORY, "and layer 0's number has none");
        // Every slot of the pool is behind one number of one layer.
        let mut owners = std::collections::HashMap::new();
        for layer in 0..2 {
            for slot in tier.locals(layer).iter().filter(|s| **s != NO_MEMORY) {
                assert!(owners.insert(*slot, layer).is_none(), "slot {slot} has two numbers");
            }
        }
        // It comes back: layer 0 takes its given-up number again, with layer 1's slot.
        let given = tier.sets[1].give_back(1, 0, &flat);
        let back = tier.move_slot(1, given[0].slot, 0);
        assert_ne!(tier.slot(0, back), NO_MEMORY);
        assert_eq!((tier.slots(0), tier.slots(1)), (4, 4));
    }
}
