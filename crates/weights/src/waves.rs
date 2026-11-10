//! The experts of one layer for many rows at once, brought in in waves.
//!
//! A prompt of a few thousand tokens routes to most experts of every layer,
//! far more than the tier in VRAM holds. They cannot all be on the GPU at
//! once, and they do not need to be: once every row of the prompt has been
//! routed in a layer, the experts are taken in **waves**. A wave is a set of
//! experts that fits the space set aside for it; it is brought in once,
//! computed for all the rows routed to it, and gives way to the next. Every
//! expert is then fetched at most once per prompt, however long the prompt is.
//!
//! This only works when a layer sees the whole prompt before the next layer
//! starts (`ares-sched`'s pass). A prompt that runs through all layers in
//! chunks of rows needs most experts of every layer again for every chunk.
//! [`ExpertWaves`] is the division into waves; reading, uploading and
//! computing belong to the layers that own files and devices.

/// Where an expert a layer needs is when its waves are planned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Place {
    /// In a slot of the GPU tier: nothing to bring in.
    InPlace,
    /// In host memory: an upload away.
    Ram,
    /// On disk only: a read and an upload away.
    Disk,
}

/// One wave: experts to bring in together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wave {
    /// The experts, in the order to bring them in.
    pub experts: Vec<u32>,
    /// How many of them come from disk; they are the last of `experts`.
    pub from_disk: usize,
    /// Rows routed to these experts, counting a row once per expert: the
    /// wave's share of the layer's work.
    pub rows: u64,
}

/// The experts a layer needs for a batch of rows, in the order to work
/// through them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertWaves {
    /// Routed experts that are in the GPU tier already. They need no wave and
    /// can be computed while the first wave is on its way.
    pub in_place: Vec<u32>,
    /// Rows routed to the experts in place, a row once per expert.
    pub in_place_rows: u64,
    /// The other routed experts, each in exactly one wave.
    pub waves: Vec<Wave>,
}

impl ExpertWaves {
    /// Divides the experts of a layer into waves.
    ///
    /// `counts[e]` is the number of rows routed to expert `e` (an expert with
    /// none is left out), `place(e)` says where a routed expert is, and
    /// `slots` is how many experts can be brought in at a time. The number of
    /// waves is the experts to bring in over `slots`, rounded up: at most
    /// `ceil(min(n_experts, rows * k) / slots)` for a router that picks `k`.
    ///
    /// Experts in host memory come before experts on disk, so the early
    /// waves start at once and the reads of the later ones (which the caller
    /// should queue for all waves right away, in this order) have the time of
    /// the earlier waves to arrive. Within each kind the most used come
    /// first, then the lower id, which makes the plan a function of its
    /// input.
    ///
    /// Returns `None` when an expert has to be brought in and `slots` is 0.
    pub fn plan(counts: &[u32], place: impl Fn(u32) -> Place, slots: usize) -> Option<Self> {
        let mut plan = Self::default();
        let mut fetch: Vec<(Place, u32)> = Vec::new();
        for (expert, count) in counts.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let expert = expert as u32;
            match place(expert) {
                Place::InPlace => {
                    plan.in_place.push(expert);
                    plan.in_place_rows += u64::from(*count);
                }
                from => fetch.push((from, expert)),
            }
        }
        if fetch.is_empty() {
            return Some(plan);
        }
        if slots == 0 {
            return None;
        }
        let rows = |expert: u32| counts[expert as usize];
        fetch.sort_by_key(|(from, expert)| (*from, std::cmp::Reverse(rows(*expert)), *expert));
        for wave in fetch.chunks(slots) {
            plan.waves.push(Wave {
                experts: wave.iter().map(|(_, expert)| *expert).collect(),
                from_disk: wave.iter().filter(|(from, _)| *from == Place::Disk).count(),
                rows: wave.iter().map(|(_, expert)| u64::from(rows(*expert))).sum(),
            });
        }
        Some(plan)
    }

    /// The experts brought in over all waves.
    pub fn fetched(&self) -> usize {
        self.waves.iter().map(|wave| wave.experts.len()).sum()
    }

    /// Of those, the ones read from disk.
    pub fn read_from_disk(&self) -> usize {
        self.waves.iter().map(|wave| wave.from_disk).sum()
    }
}

/// The weight of routing that lies `tokens_after` tokens before the end of a
/// prompt, when what the end of the prompt used should count most for the
/// tiers the reply starts with: it halves every `half_life` tokens. The
/// weights of a prompt's chunks go to
/// [`ExpertHeat::record_counts`](crate::ExpertHeat::record_counts).
pub fn recency_weight(tokens_after: u32, half_life: u32) -> f32 {
    if half_life == 0 {
        return if tokens_after == 0 { 1.0 } else { 0.0 };
    }
    0.5f32.powf(tokens_after as f32 / half_life as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExpertHeat, SlotSet, SwapRule};

    #[test]
    fn experts_in_place_need_no_wave_and_unused_ones_are_left_out() {
        //        expert:  0  1  2  3  4  5
        let counts = [3, 0, 9, 5, 0, 1];
        let place = |e| if e == 2 || e == 5 { Place::InPlace } else { Place::Ram };
        let plan = ExpertWaves::plan(&counts, place, 8).unwrap();
        assert_eq!(plan.in_place, [2, 5]);
        assert_eq!(plan.in_place_rows, 10);
        assert_eq!(plan.waves, [Wave { experts: vec![3, 0], from_disk: 0, rows: 8 }]);
        assert_eq!((plan.fetched(), plan.read_from_disk()), (2, 0));

        // Nothing routed, or everything in place: no wave, even without slots.
        assert_eq!(ExpertWaves::plan(&[0; 4], |_| Place::Disk, 0), Some(ExpertWaves::default()));
        let all_there = ExpertWaves::plan(&counts, |_| Place::InPlace, 0).unwrap();
        assert!(all_there.waves.is_empty());
        assert_eq!(all_there.in_place, [0, 2, 3, 5]);
        // An expert to bring in and nowhere to put it.
        assert_eq!(ExpertWaves::plan(&counts, |_| Place::Ram, 0), None);
    }

    #[test]
    fn waves_hold_what_fits_memory_before_disk_and_the_most_used_first() {
        //        expert:  0  1  2  3  4  5  6  7
        let counts = [4, 7, 1, 7, 2, 9, 3, 6];
        // Odd experts are in host memory, even ones on disk.
        let place = |e: u32| if e % 2 == 1 { Place::Ram } else { Place::Disk };
        let plan = ExpertWaves::plan(&counts, place, 3).unwrap();
        assert!(plan.in_place.is_empty());
        assert_eq!(
            plan.waves,
            [
                // From memory by use: 5 (9), then 1 and 3 (7 each, lower id first).
                Wave { experts: vec![5, 1, 3], from_disk: 0, rows: 23 },
                // The last one in memory, then disk by use: 0 (4), 6 (3).
                Wave { experts: vec![7, 0, 6], from_disk: 2, rows: 13 },
                Wave { experts: vec![4, 2], from_disk: 2, rows: 3 },
            ]
        );
        assert_eq!((plan.fetched(), plan.read_from_disk()), (8, 4));
        // Every routed expert is in exactly one wave.
        let mut all: Vec<u32> = plan.waves.iter().flat_map(|w| w.experts.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..8).collect::<Vec<u32>>());
    }

    #[test]
    fn the_number_of_waves_is_the_experts_to_bring_in_over_the_slots() {
        // 512 experts, a router that picks 8: a prompt of `rows` tokens can
        // route to at most min(512, rows * 8) of them.
        let n_experts = 512;
        for (rows, slots) in [(1u32, 64usize), (10, 64), (64, 64), (100, 64), (4096, 64), (4096, 1)]
        {
            let routed = (rows as usize * 8).min(n_experts);
            let counts: Vec<u32> = (0..n_experts).map(|e| u32::from(e < routed)).collect();
            let plan = ExpertWaves::plan(&counts, |_| Place::Ram, slots).unwrap();
            assert_eq!(plan.waves.len(), routed.div_ceil(slots), "{rows} rows, {slots} slots");
            assert!(plan.waves.iter().all(|w| w.experts.len() <= slots));
            assert_eq!(plan.fetched(), routed);
        }
    }

    #[test]
    fn the_end_of_a_prompt_weighs_most() {
        assert_eq!(recency_weight(0, 128), 1.0);
        assert!((recency_weight(128, 128) - 0.5).abs() < 1e-6);
        assert!((recency_weight(256, 128) - 0.25).abs() < 1e-6);
        assert!(recency_weight(4096, 128) < 1e-9);
        assert_eq!((recency_weight(0, 0), recency_weight(1, 0)), (1.0, 0.0));
    }

    /// The premise of the pass, on a synthetic prompt. Tokens route to 8
    /// experts of 256, mostly within the topic of their part of the prompt.
    /// Run layer by layer, a layer fetches each expert it uses once. Run in
    /// chunks of rows through the layer, with the same space for experts
    /// (the tier plus the room for a wave, kept between chunks, least recently
    /// used out first), it fetches most of them again for every chunk.
    #[test]
    fn layer_by_layer_fetches_each_expert_once_and_chunks_of_rows_fetch_them_again() {
        const EXPERTS: usize = 256;
        const TIER: usize = 32;
        const WAVE: usize = 32;
        const TOKENS: usize = 4096;
        const CHUNK: usize = 64;

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        // The topic changes every 512 tokens; a topic has 32 experts of its
        // own, and a fifth of the picks go anywhere.
        let routing: Vec<[u32; 8]> = (0..TOKENS)
            .map(|token| {
                let topic = (token / 512) as u64;
                std::array::from_fn(|_| {
                    if random(5) < 4 {
                        (topic * 32 + random(32)) as u32
                    } else {
                        random(EXPERTS as u64) as u32
                    }
                })
            })
            .collect();
        let counts_of = |tokens: &[[u32; 8]]| {
            let mut counts = vec![0u32; EXPERTS];
            for expert in tokens.iter().flatten() {
                counts[*expert as usize] += 1;
            }
            counts
        };

        // The GPU tier holds the 32 experts a profile of the first topic names.
        let mut heat = ExpertHeat::new(1, EXPERTS, 0.5);
        heat.record_counts(0, &counts_of(&routing[..256]), 1.0);
        let mut tier = SlotSet::new(EXPERTS, TIER);
        for swap in tier.elect(heat.layer(0), &SwapRule::default()) {
            tier.commit(&swap);
        }
        let place = |e: u32| if tier.contains(e) { Place::InPlace } else { Place::Disk };

        // Layer by layer: one plan for the whole prompt.
        let all = counts_of(&routing);
        let distinct = all.iter().filter(|c| **c > 0).count();
        let pass = ExpertWaves::plan(&all, place, WAVE).unwrap();
        assert_eq!(pass.fetched() + pass.in_place.len(), distinct);
        assert_eq!(pass.waves.len(), pass.fetched().div_ceil(WAVE));

        // In chunks of rows: the room for a wave is a cache that keeps what
        // the last chunks used.
        let mut cache: Vec<u32> = Vec::new(); // most recently used last
        let mut fetched_in_chunks = 0;
        for chunk in routing.chunks(CHUNK) {
            let counts = counts_of(chunk);
            let cached = |e: u32| tier.contains(e) || cache.contains(&e);
            let place = |e: u32| if cached(e) { Place::InPlace } else { Place::Disk };
            let plan = ExpertWaves::plan(&counts, place, WAVE).unwrap();
            fetched_in_chunks += plan.fetched();
            // What the chunk used is the most recent now; the oldest go.
            for expert in plan.in_place.iter().chain(plan.waves.iter().flat_map(|w| &w.experts)) {
                cache.retain(|e| e != expert);
                if !tier.contains(*expert) {
                    cache.push(*expert);
                }
            }
            let excess = cache.len().saturating_sub(WAVE);
            cache.drain(..excess);
        }

        let per_expert = fetched_in_chunks as f64 / pass.fetched() as f64;
        eprintln!(
            "{distinct} experts used, {} in the tier; fetched layer by layer {} (in {} waves), \
             in chunks of {CHUNK} rows {fetched_in_chunks} ({per_expert:.1} times each)",
            pass.in_place.len(),
            pass.fetched(),
            pass.waves.len()
        );
        assert!(pass.fetched() <= EXPERTS - pass.in_place.len());
        assert!(per_expert > 5.0, "chunks of rows fetched each expert {per_expert:.1} times");

        // Before the reply starts, the tier is elected from the prompt's own
        // routing with its end weighing most: it then holds experts of the
        // last topic, which the profile of the first did not name.
        let mut heat = ExpertHeat::new(1, EXPERTS, 0.5);
        for (i, chunk) in routing.chunks(CHUNK).enumerate() {
            let after = (TOKENS - (i + 1) * CHUNK) as u32;
            heat.record_counts(0, &counts_of(chunk), recency_weight(after, 128));
        }
        let mut for_reply = SlotSet::new(EXPERTS, TIER);
        for swap in for_reply.elect(heat.layer(0), &SwapRule::default()) {
            for_reply.commit(&swap);
        }
        let last = counts_of(&routing[TOKENS - 256..]);
        let served = |set: &SlotSet| -> u32 {
            (0..EXPERTS as u32).filter(|e| set.contains(*e)).map(|e| last[e as usize]).sum()
        };
        let total: u32 = last.iter().sum();
        assert!(served(&for_reply) * 2 > total, "the new tier serves the end of the prompt");
        assert!(served(&for_reply) > 3 * served(&tier), "the tier of the first topic does not");
    }
}
