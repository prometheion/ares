//! Item 2: slots per layer by need.
//!
//! The tiers of the 16 layers of the synthetic workload share one budget of
//! slots. Every tier is re-elected every 32 tokens from heat that halves each
//! time, with the swaps free, as in the existing premise of the dynamic tier;
//! what differs is how many slots each layer has. Divided equally, each has
//! the same. Divided by need, the split is made again every 256 tokens from
//! the same heat ([`split_slots`], starting from the split in force), and the
//! layers hand slots to each other with [`SlotSet::give_back`] and
//! [`SlotSet::take`].

use ares_weights::{ExpertHeat, LayerNeed, SlotSet, SplitRule, SwapRule, split_slots};

use crate::cache::{EXPERTS, LAYERS, workload};

const FIRST_PASS_COUNTED: usize = 1000;

struct Outcome {
    /// Per layer, the share of picks the tier served.
    rate: Vec<f64>,
    /// Slots that changed layers, summed over all divisions.
    moved: usize,
    slots: Vec<usize>,
}

impl Outcome {
    fn mean(&self, layers: std::ops::Range<usize>) -> f64 {
        self.rate[layers.clone()].iter().sum::<f64>() / layers.len() as f64
    }
}

/// Hands slots from the layers that lose them to the layers that gain them.
fn resize(tiers: &mut [SlotSet], from: &[usize], to: &[usize], heat: &ExpertHeat) {
    for (layer, tier) in tiers.iter_mut().enumerate() {
        if to[layer] > from[layer] {
            tier.take(to[layer] - from[layer]);
        } else if to[layer] < from[layer] {
            let given = tier.give_back(from[layer] - to[layer], 0, heat.layer(layer));
            assert_eq!(given.len(), from[layer] - to[layer], "no slot is loading");
        }
    }
}

fn simulate(
    work: &crate::synthetic::Workload,
    per_layer: usize,
    by_need: Option<SplitRule>,
) -> Outcome {
    let budget = (per_layer * LAYERS) as u64;
    let mut heat = ExpertHeat::new(LAYERS, EXPERTS, 0.5);
    let mut slots = vec![per_layer; LAYERS];
    let mut tiers: Vec<SlotSet> = slots.iter().map(|n| SlotSet::new(EXPERTS, *n)).collect();
    let (mut hits, mut counted, mut moved) = (vec![0usize; LAYERS], 0usize, 0usize);
    for pass in 0..work.len() {
        for (layer, tier) in tiers.iter().enumerate() {
            let routed = work.routed(pass, layer);
            if pass >= FIRST_PASS_COUNTED {
                hits[layer] += routed.iter().filter(|e| tier.contains(**e)).count();
            }
            heat.record(layer, routed);
        }
        if pass >= FIRST_PASS_COUNTED {
            counted += work.shape.k;
        }
        if pass % 32 == 31 {
            heat.decay();
            if let Some(rule) = by_need.filter(|_| pass % 256 == 255) {
                let needs: Vec<LayerNeed> = (0..LAYERS)
                    .map(|l| LayerNeed { heat: heat.layer(l), expert_bytes: 1 })
                    .collect();
                let next = split_slots(&needs, budget, Some(&slots), &rule);
                moved += next.iter().zip(&slots).map(|(a, b)| a.abs_diff(*b)).sum::<usize>() / 2;
                resize(&mut tiers, &slots, &next, &heat);
                slots = next;
            }
            for (layer, tier) in tiers.iter_mut().enumerate() {
                for swap in tier.elect(heat.layer(layer), &SwapRule::default()) {
                    tier.commit(&swap);
                }
            }
        }
    }
    Outcome { rate: hits.iter().map(|h| *h as f64 / counted as f64).collect(), moved, slots }
}

/// The premise: layers differ in how much a slot gains (the first ones route
/// flatly, a topic's experts are few in the others), and dividing the slots by
/// the gain of the next one serves more picks than dividing them equally,
/// most of all in the layers that route flatly, which a predictor reaches
/// last. The margin and the cap on moves keep the layers from trading slots.
#[test]
fn slots_by_need_serve_more_picks_than_an_equal_split() {
    let work = workload(12_000);
    for per_layer in [12, 16, 24, 32, 48] {
        let equal = simulate(&work, per_layer, None);
        let need = simulate(&work, per_layer, Some(SplitRule::default()));
        eprintln!(
            "{per_layer} slots per layer: equal {:.3}, by need {:.3}; first four layers {:.3} and {:.3}; \
             {} slots moved\n  slots {:?}",
            equal.mean(0..LAYERS),
            need.mean(0..LAYERS),
            equal.mean(0..4),
            need.mean(0..4),
            need.moved,
            need.slots,
        );
    }

    // A quarter of the experts in the tier.
    let equal = simulate(&work, 32, None);
    let need = simulate(&work, 32, Some(SplitRule::default()));
    assert_eq!(need.slots.iter().sum::<usize>(), 32 * LAYERS);
    assert!(
        need.mean(0..LAYERS) > equal.mean(0..LAYERS) + 0.01,
        "{:.3} against {:.3}",
        need.mean(0..LAYERS),
        equal.mean(0..LAYERS)
    );
    assert!(need.mean(0..4) > equal.mean(0..4) + 0.1, "the flat layers gain the most");
    assert!(need.slots[..4].iter().all(|s| *s > 32), "they get more than an equal share");
    assert!(need.slots[4..].iter().all(|s| *s < 32), "the others give it up");

    // The same budget with the margin and the cap off: the split is as
    // good but changes layers for most of the run.
    let always = SplitRule { margin: 1.0, max_moves: usize::MAX };
    let churn = simulate(&work, 32, Some(always));
    eprintln!(
        "margin and cap: {} slots moved, {:.3}; without: {} slots moved, {:.3}",
        need.moved,
        need.mean(0..LAYERS),
        churn.moved,
        churn.mean(0..LAYERS)
    );
    assert!(need.moved * 3 < churn.moved);
    assert!(churn.mean(0..LAYERS) < need.mean(0..LAYERS) + 0.005);
}

/// The share of picks served in each 1000 passes of 6000, when the budget of
/// 512 slots is 436 from pass 2048 to pass 4095 (the broker took 15 % back and
/// gave them again) or stays 512. Also returns, for each change of the budget,
/// how many slots changed hands in all and how many of them only because the
/// profile moved.
fn run_with_a_broker(
    work: &crate::synthetic::Workload,
    broker_asks: bool,
) -> (Vec<f64>, Vec<(usize, usize)>) {
    let rule = SplitRule::default();
    let mut heat = ExpertHeat::new(LAYERS, EXPERTS, 0.5);
    let mut slots = vec![32usize; LAYERS];
    let mut tiers: Vec<SlotSet> = slots.iter().map(|n| SlotSet::new(EXPERTS, *n)).collect();
    let (mut rates, mut changes) = (Vec::new(), Vec::new());
    let mut counted = (0usize, 0usize);
    for pass in 0..6000 {
        for (layer, tier) in tiers.iter().enumerate() {
            let routed = work.routed(pass, layer);
            counted.0 += routed.iter().filter(|e| tier.contains(**e)).count();
            counted.1 += routed.len();
            heat.record(layer, routed);
        }
        if pass % 32 == 31 {
            heat.decay();
            let budget = if broker_asks && (2048..4096).contains(&pass) { 436 } else { 512 };
            // A new budget is acted on at once; the profile is looked at every 256 passes.
            let changed = budget != slots.iter().sum::<usize>() as u64;
            if changed || pass % 256 == 255 {
                let needs: Vec<LayerNeed> = (0..LAYERS)
                    .map(|l| LayerNeed { heat: heat.layer(l), expert_bytes: 1 })
                    .collect();
                let next = split_slots(&needs, budget, Some(&slots), &rule);
                let all = next.iter().zip(&slots).map(|(a, b)| a.abs_diff(*b)).sum::<usize>();
                let asked = (budget as usize).abs_diff(slots.iter().sum::<usize>());
                if changed {
                    changes.push((asked, (all - asked) / 2));
                }
                resize(&mut tiers, &slots, &next, &heat);
                slots = next;
            }
            for (layer, tier) in tiers.iter_mut().enumerate() {
                for swap in tier.elect(heat.layer(layer), &SwapRule::default()) {
                    tier.commit(&swap);
                }
            }
            assert_eq!(
                tiers.iter().map(SlotSet::n_slots).sum::<usize>(),
                slots.iter().sum::<usize>()
            );
        }
        if pass % 1000 == 999 {
            rates.push(counted.0 as f64 / counted.1 as f64);
            counted = (0, 0);
        }
    }
    (rates, changes)
}

/// The broker asks for memory back and later gives it again: the split is made
/// again for the new budget, the layers hand over exactly those slots, and
/// nothing is shuffled between layers on the way.
#[test]
fn a_budget_that_shrinks_and_grows_moves_the_slots_it_must() {
    let work = workload(6000);
    let (steady, none) = run_with_a_broker(&work, false);
    let (asked, changes) = run_with_a_broker(&work, true);
    eprintln!(
        "served per 1000 passes: steady {steady:.3?}, budget cut {asked:.3?}; changes {changes:?}"
    );
    assert!(none.is_empty());
    // Two changes: 76 slots away, 76 slots back, and at most the cap of slots
    // moved between layers for the sake of the profile.
    assert_eq!(changes.len(), 2);
    for (asked, shuffled) in changes {
        assert_eq!(asked, 76);
        assert!(
            shuffled <= SplitRule::default().max_moves,
            "{shuffled} slots moved between layers"
        );
    }
    // While the budget is 15 % smaller fewer picks are served, and afterwards
    // as many as without the cut (the topics differ from one thousand to the
    // next, so the passes are compared with themselves).
    assert!(asked[2] < steady[2] - 0.005 && asked[3] < steady[3] - 0.005, "{steady:?} {asked:?}");
    assert!((asked[5] - steady[5]).abs() < 0.005, "{steady:?} {asked:?}");
}
