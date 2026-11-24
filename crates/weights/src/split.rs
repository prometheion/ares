//! How many slots each layer gets.

/// What one layer says about how much it needs slots.
#[derive(Clone, Copy, Debug)]
pub struct LayerNeed<'a> {
    /// The heat of each expert of the layer ([`ExpertHeat::layer`](crate::ExpertHeat::layer)).
    pub heat: &'a [f32],
    /// The bytes one slot of this layer takes of the tier's memory. For a
    /// tier that reads from the SSD that is the size of a read buffer (the
    /// expert widened to whole blocks), not the bare expert. The layers of a
    /// model usually have the same size; the tier's budget may also be given
    /// in slots by setting this to 1.
    pub expert_bytes: u64,
}

/// How slots are divided over the layers, and what keeps them from moving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitRule {
    /// A slot moves from one layer to another only if the gain per byte of
    /// the slot it would get exceeds `margin` times that of the slot it would
    /// lose (1 or more). It keeps layers with a similar need from trading a
    /// slot back and forth, each time at the price of an eviction and a read.
    pub margin: f32,
    /// The most slots one call moves between layers for the sake of a changed
    /// profile. (A changed budget is met in full, whatever this says.)
    pub max_moves: usize,
}

impl Default for SplitRule {
    fn default() -> Self {
        Self { margin: 1.25, max_moves: 8 }
    }
}

/// The share of a layer's picks that its `n`-th hottest expert serves, for
/// `n` from 1 on: non-increasing. A layer with no heat at all has told
/// nothing and counts as flat.
fn gains(heat: &[f32]) -> Vec<f64> {
    let total: f64 = heat.iter().map(|h| f64::from(*h)).sum();
    let mut gains: Vec<f64> = if total <= 0.0 {
        vec![1.0 / heat.len().max(1) as f64; heat.len()]
    } else {
        heat.iter().map(|h| f64::from(*h) / total).collect()
    };
    gains.sort_by(|a, b| b.total_cmp(a));
    gains
}

/// Divides a tier's memory over the layers, so that the slots go where the
/// next one gains most.
///
/// A layer's slots hold its hottest experts, so what a slot gains is the
/// share of the layer's picks that its expert serves: the heat of the
/// hottest expert divided by the layer's total for the first slot, the
/// second hottest for the second, and so on. Every layer routes the same
/// number of experts per token, so these shares are comparable between
/// layers. A layer with no heat yet counts as flat: each expert serves an
/// equal share. The split gives each next slot to the layer where that share per
/// byte is largest. For equal sizes that maximises the picks the tiers serve
/// (the gains of a layer only fall, so greedy is optimal); with different
/// sizes it is within the last slot of it.
///
/// A layer that routes to a few experts is served well by a few slots, and
/// its further slots gain next to nothing; a layer that routes flatly gains
/// the same from each of its slots. So an equal split gives the peaked
/// layers slots they do not need and the flat ones (the first layers of a
/// model) too few, and this split moves them over.
///
/// `budget` is the tier's bytes. `current` is the split in force, or `None`
/// for the first time. The result has one number of slots per layer, at most
/// the layer's experts, and costs at most `budget` bytes.
///
/// # Not moving slots back and forth
///
/// The profile moves slowly and the budget changes as the books of host memory say, and
/// every slot that changes layers costs an eviction and a read. So the
/// split is not computed again from scratch, but changed from `current`:
///
/// 1. When the budget fell, slots are taken away one by one, each from the
///    layer where it gains least; when it grew, given one by one, each to the
///    layer where it gains most. With a steady profile, a budget that changes
///    by `n` slots changes exactly `n`, and no slot changes layers besides.
/// 2. Then a slot moves from the layer where it gains least to the layer
///    where one more would gain most, only if that gain exceeds the loss by
///    `rule.margin`, and at most `rule.max_moves` times. A profile that
///    drifts a little moves nothing; one that changed a lot is followed over
///    a few calls.
///
/// Calling it again with its own result and the same inputs returns the same
/// split.
///
/// # Panics
///
/// Panics when `current` does not have one number per layer, a layer has
/// `expert_bytes` of 0, or `rule.margin` is below 1.
pub fn split_slots(
    layers: &[LayerNeed<'_>],
    budget: u64,
    current: Option<&[usize]>,
    rule: &SplitRule,
) -> Vec<usize> {
    assert!(rule.margin >= 1.0, "margin must be at least 1");
    assert!(layers.iter().all(|l| l.expert_bytes > 0), "a slot takes memory");
    let bytes: Vec<u64> = layers.iter().map(|l| l.expert_bytes).collect();
    let gains: Vec<Vec<f64>> = layers.iter().map(|l| gains(l.heat)).collect();
    let mut slots: Vec<usize> = match current {
        Some(current) => {
            assert_eq!(current.len(), layers.len(), "one number of slots per layer");
            current.iter().zip(&gains).map(|(slots, gains)| (*slots).min(gains.len())).collect()
        }
        None => vec![0; layers.len()],
    };
    let mut used: u64 = slots.iter().zip(&bytes).map(|(s, b)| *s as u64 * b).sum();

    // What the next slot of a layer gains per byte, and what its last one does.
    let next =
        |slots: &[usize], l: usize| gains[l].get(slots[l]).map(|gain| gain / bytes[l] as f64);
    let last =
        |slots: &[usize], l: usize| slots[l].checked_sub(1).map(|n| gains[l][n] / bytes[l] as f64);
    // The layer with the least valuable slot. At equal value the one with
    // more slots gives it up (so an even split stays even), then the higher
    // layer (so the lower, usually flatter layers keep theirs).
    let least = |slots: &[usize]| {
        (0..layers.len()).filter_map(|l| last(slots, l).map(|value| (l, value))).min_by(|a, b| {
            a.1.total_cmp(&b.1).then(slots[b.0].cmp(&slots[a.0])).then(b.0.cmp(&a.0))
        })
    };
    // The layer where a slot of at most `room` bytes would gain most. At
    // equal value the one with fewer slots, then the lower layer.
    let most = |slots: &[usize], room: u64, skip: Option<usize>| {
        (0..layers.len())
            .filter(|l| Some(*l) != skip && bytes[*l] <= room)
            .filter_map(|l| next(slots, l).map(|value| (l, value)))
            .max_by(|a, b| {
                a.1.total_cmp(&b.1).then(slots[b.0].cmp(&slots[a.0])).then(b.0.cmp(&a.0))
            })
    };

    while used > budget {
        let (layer, _) = least(&slots).expect("a slot is used, so a layer has one");
        slots[layer] -= 1;
        used -= bytes[layer];
    }
    while let Some((layer, _)) = most(&slots, budget - used, None) {
        slots[layer] += 1;
        used += bytes[layer];
    }
    for _ in 0..rule.max_moves {
        let Some((from, loss)) = least(&slots) else { break };
        let Some((to, gain)) = most(&slots, budget - used + bytes[from], Some(from)) else {
            break;
        };
        if gain <= f64::from(rule.margin) * loss {
            break;
        }
        slots[from] -= 1;
        slots[to] += 1;
        used = used - bytes[from] + bytes[to];
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer of `n` experts where `hot` experts have `hot_heat` each and
    /// the others 1.
    fn profile(n: usize, hot: usize, hot_heat: f32) -> Vec<f32> {
        (0..n).map(|e| if e < hot { hot_heat } else { 1.0 }).collect()
    }

    fn need(heat: &[f32]) -> LayerNeed<'_> {
        LayerNeed { heat, expert_bytes: 1 }
    }

    #[test]
    fn a_layer_that_routes_flatly_gets_the_slots_a_peaked_one_does_not_need() {
        // Layer 0 picks the same four experts almost always; layer 1 any.
        let peaked = profile(32, 4, 400.0);
        let flat = vec![10.0; 32];
        let layers = [need(&peaked), need(&flat)];
        let slots = split_slots(&layers, 16, None, &SplitRule::default());
        assert_eq!(slots, [4, 12], "four slots serve the peaked layer, the flat one gets the rest");
        // With room to spare, the peaked layer's tail gets slots too, but only
        // after the flat layer has all of its experts.
        assert_eq!(split_slots(&layers, 40, None, &SplitRule::default()), [8, 32]);
        assert_eq!(split_slots(&layers, 64, None, &SplitRule::default()), [32, 32]);
    }

    #[test]
    fn without_heat_the_split_is_equal_and_bytes_not_slots_are_divided() {
        let nothing = vec![0.0; 16];
        let layers = [need(&nothing), need(&nothing), need(&nothing)];
        assert_eq!(split_slots(&layers, 10, None, &SplitRule::default()), [4, 3, 3]);
        assert_eq!(split_slots(&[], 10, None, &SplitRule::default()), [] as [usize; 0]);
        assert_eq!(split_slots(&layers, 0, None, &SplitRule::default()), [0, 0, 0]);

        // The second layer's slots take twice the memory: at equal gains the
        // cheaper slots are filled first, and the budget is never exceeded.
        let layers = [
            LayerNeed { heat: &nothing, expert_bytes: 1 },
            LayerNeed { heat: &nothing, expert_bytes: 2 },
        ];
        let slots = split_slots(&layers, 21, None, &SplitRule::default());
        assert_eq!(slots, [16, 2]);
        assert!(slots[0] + 2 * slots[1] <= 21);
    }

    #[test]
    fn a_changed_budget_moves_exactly_the_slots_it_must() {
        let peaked = profile(32, 4, 400.0);
        let medium = profile(32, 10, 40.0);
        let flat = vec![10.0; 32];
        let layers = [need(&peaked), need(&medium), need(&flat)];
        let rule = SplitRule::default();
        let first = split_slots(&layers, 30, None, &rule);

        let less = split_slots(&layers, 26, Some(&first), &rule);
        assert_eq!(less.iter().sum::<usize>(), 26);
        assert!(
            less.iter().zip(&first).all(|(l, f)| l <= f),
            "slots only left: {first:?} {less:?}"
        );
        let more = split_slots(&layers, 30, Some(&less), &rule);
        assert_eq!(more, first, "taking the slots back restores the split");

        let grown = split_slots(&layers, 34, Some(&first), &rule);
        assert!(grown.iter().zip(&first).all(|(g, f)| g >= f));
        assert_eq!(grown, split_slots(&layers, 34, None, &rule));
    }

    #[test]
    fn a_profile_that_drifts_moves_nothing_and_one_that_changes_is_followed_in_steps() {
        let rule = SplitRule { max_moves: 3, ..SplitRule::default() };
        let (a, b) = (profile(64, 6, 300.0), vec![10.0; 64]);
        let layers = [need(&a), need(&b)];
        let split = split_slots(&layers, 24, None, &rule);
        assert_eq!(split_slots(&layers, 24, Some(&split), &rule), split, "a fixed point");

        // Noise of 5 % in every expert's heat, in both directions.
        let noisy = |heat: &[f32], salt: u32| -> Vec<f32> {
            heat.iter()
                .enumerate()
                .map(|(e, h)| h * if (e as u32 * 7 + salt).is_multiple_of(2) { 1.05 } else { 0.95 })
                .collect()
        };
        for salt in 0..4 {
            let (a, b) = (noisy(&a, salt), noisy(&b, salt + 1));
            let moved = split_slots(&[need(&a), need(&b)], 24, Some(&split), &rule);
            assert_eq!(moved, split, "noise {salt} moved a slot");
        }

        // The layers swap roles: the first is flat now and the second peaked.
        let swapped = [need(&b), need(&a)];
        let target = split_slots(&swapped, 24, None, &rule);
        assert_ne!(target, split);
        let mut now = split.clone();
        let mut calls = 0;
        loop {
            let next = split_slots(&swapped, 24, Some(&now), &rule);
            let moved: usize =
                next.iter().zip(&now).map(|(n, c)| n.abs_diff(*c)).sum::<usize>() / 2;
            assert!(moved <= 3, "{moved} slots moved in one call");
            if next == now {
                break;
            }
            now = next;
            calls += 1;
            assert!(calls < 10, "the split does not settle");
        }
        assert_eq!(now, target);
        assert!(calls > 1, "the move takes more than one call");
    }

    #[test]
    fn layers_with_a_similar_need_do_not_trade_a_slot() {
        // The next slot of the second layer gains 3 % more than the last slot of
        // the first (a layer of 31 experts against one of 32, both flat).
        let (first, second) = (vec![1.0; 32], vec![1.0; 31]);
        let layers = [need(&first), need(&second)];
        let rule = SplitRule::default();
        assert_eq!(split_slots(&layers, 8, Some(&[4, 4]), &rule), [4, 4]);
        // The margin is what holds them: without it everything moves over.
        let no_margin = SplitRule { margin: 1.0, ..rule };
        assert_eq!(split_slots(&layers, 8, Some(&[4, 4]), &no_margin), [0, 8]);
        // A first split is not held back by it.
        assert_eq!(split_slots(&layers, 8, None, &rule), [0, 8]);
    }

    #[test]
    fn a_layer_never_gets_more_slots_than_experts() {
        let small = vec![5.0; 4];
        let big = vec![1.0; 64];
        let slots = split_slots(&[need(&small), need(&big)], 100, None, &SplitRule::default());
        assert_eq!(slots, [4, 64]);
        // A split in force from a model with more experts is cut to fit.
        let cut =
            split_slots(&[need(&small), need(&big)], 100, Some(&[9, 0]), &SplitRule::default());
        assert_eq!(cut, [4, 64]);
    }
}
