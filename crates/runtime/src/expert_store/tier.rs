//! One tier of the expert store: pools of slots and who holds them.

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
/// can pass from one of them to another when their needs change.
#[derive(Debug)]
pub(crate) struct Tier {
    pub pools: Vec<Pool>,
    pub pool_of: Vec<usize>,
    pub sets: Vec<SlotSet>,
    slot: Vec<Vec<u32>>,
}

impl Tier {
    /// A tier for layers of `n_experts` experts: `layers[l]` is the place and
    /// the slot size of layer `l`, `budget[place]` the bytes the tier may
    /// take there, `floor` the fewest slots a layer must have. `heat[l]` is
    /// what is known of each layer's routing; the slots of a pool are divided
    /// by it ([`split_slots`]), evenly when it says nothing.
    ///
    /// Returns `None` when a budget does not hold `floor` slots per layer.
    pub fn plan(
        n_experts: usize,
        layers: &[(usize, u64)],
        budget: &[u64],
        floor: usize,
        heat: &[&[f32]],
    ) -> Option<Self> {
        let mut pools: Vec<Pool> = Vec::new();
        let mut pool_of = Vec::with_capacity(layers.len());
        for (place, stride) in layers {
            let found = pools.iter().position(|p| p.place == *place && p.stride == *stride);
            pool_of.push(found.unwrap_or_else(|| {
                pools.push(Pool { place: *place, stride: *stride, slots: 0 });
                pools.len() - 1
            }));
        }
        // A place's bytes go to its pools by the bytes one slot per layer
        // takes in each, so that every layer of the place starts with the
        // same number of slots.
        let mut counts = vec![0usize; layers.len()];
        for (index, pool) in pools.iter_mut().enumerate() {
            let members: Vec<usize> = (0..layers.len()).filter(|l| pool_of[*l] == index).collect();
            let per_round: u64 =
                layers.iter().filter(|(place, _)| *place == pool.place).map(|(_, s)| *s).sum();
            let rounds = budget.get(pool.place).copied().unwrap_or(0) / per_round.max(1);
            let slots = (rounds as usize).min(n_experts) * members.len();
            if slots < floor * members.len() {
                return None;
            }
            let needs: Vec<LayerNeed<'_>> =
                members.iter().map(|l| LayerNeed { heat: heat[*l], expert_bytes: 1 }).collect();
            let split = split_slots(&needs, slots as u64, None, &SplitRule::default());
            for (layer, n) in members.iter().zip(raise_to_floor(split, floor)) {
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
        Some(Self { pools, pool_of, sets, slot })
    }

    /// The slot of the layer's pool behind the layer's slot number `local`.
    pub fn slot(&self, layer: usize, local: u32) -> u32 {
        self.slot[layer][local as usize]
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
        } else {
            // A number `to` had given back earlier: swap the memory behind
            // the two numbers, so `from` keeps one for a later take.
            let other = std::mem::replace(&mut self.slot[to][taken as usize], slot);
            self.slot[from][local as usize] = other;
        }
        taken
    }
}

/// Raises every count below `floor` to it and takes the slots from the
/// largest counts.
fn raise_to_floor(mut counts: Vec<usize>, floor: usize) -> Vec<usize> {
    let mut owed = 0;
    for n in &mut counts {
        if *n < floor {
            owed += floor - *n;
            *n = floor;
        }
    }
    while owed > 0 {
        let Some(largest) =
            (0..counts.len()).filter(|i| counts[*i] > floor).max_by_key(|i| counts[*i])
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
        let tier = Tier::plan(16, &layers, &[1300, 450], 2, &heat).expect("fits");
        assert_eq!(tier.pools.len(), 3);
        assert_eq!(tier.pool_of, [0, 1, 0, 2]);
        // 1300 bytes / (100 + 60 + 100) per round = 5 slots per layer.
        assert_eq!((0..4).map(|l| tier.slots(l)).collect::<Vec<_>>(), [5, 5, 5, 4]);
        assert_eq!(tier.bytes(0), 1300);
        assert_eq!(tier.bytes(1), 400);
        assert_eq!((tier.slot(0, 0), tier.slot(2, 0), tier.slot(1, 4)), (0, 5, 4));
        assert!(Tier::plan(16, &layers, &[1300, 150], 2, &heat).is_none());
    }

    #[test]
    fn a_profile_moves_slots_to_the_layer_that_needs_them() {
        let narrow: Vec<f32> = (0..16).map(|e| if e < 2 { 50.0 } else { 0.0 }).collect();
        let wide = vec![6.0f32; 16];
        let tier = Tier::plan(16, &[(0, 10), (0, 10)], &[160], 2, &[&narrow, &wide]).expect("fits");
        assert_eq!(tier.slots(0) + tier.slots(1), 16);
        assert!(tier.slots(1) > tier.slots(0), "{} {}", tier.slots(0), tier.slots(1));
        assert!(tier.slots(0) >= 2);
    }

    #[test]
    fn a_slot_passes_between_layers() {
        let flat = vec![1.0f32; 8];
        let mut tier = Tier::plan(8, &[(0, 10), (0, 10)], &[80], 1, &[&flat, &flat]).expect("fits");
        assert_eq!((tier.slots(0), tier.slots(1)), (4, 4));
        let given = tier.sets[0].give_back(1, 0, &flat);
        let taken = tier.move_slot(0, given[0].slot, 1);
        assert_eq!((tier.slots(0), tier.slots(1)), (3, 5));
        assert_eq!(tier.slot(1, taken), 3, "layer 1 now owns layer 0's last slot");
    }
}
