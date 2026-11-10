//! Item 3: a table from the last tokens to the experts of the first layers.
//!
//! The first four layers of the synthetic workload route by token (see
//! [`crate::synthetic`]): no router before them can predict their experts, and
//! over a text each of them uses its experts about equally, so a tier filled
//! from a profile serves little of them. The table is built from the routing
//! of the first 7000 passes and tried on the next 7000, which the table has
//! not seen (the same topics come up again, with other words in other orders).
//!
//! A guess for layer `j` made when a token is sampled has `j` layers of compute
//! to arrive in, plus the embedding. Reads that take a layer and a half are in
//! time from the second layer on, so the numbers asserted are those of the
//! layers 1 to 3; the numbers printed are for all four.

use ares_weights::{ExpertHeat, SlotSet, SwapRule, TableBuilder, TableConfig, TokenTable};

use crate::cache::{EXPERTS, workload};

const FIRST_LAYERS: usize = 4;
const TRAIN: std::ops::Range<usize> = 0..7000;
const TEST: std::ops::Range<usize> = 7000..14_000;

/// What the first layers are served from in the test passes.
#[derive(Debug, Default)]
struct Served {
    /// Routed experts, per layer.
    routed: [usize; FIRST_LAYERS],
    /// Of them, in the tier.
    in_tier: [usize; FIRST_LAYERS],
    /// Of them, in the tier or named by the table.
    in_tier_or_table: [usize; FIRST_LAYERS],
    /// Experts the table named that were not in the tier: reads to make.
    reads: usize,
    /// Of them, experts the layer then routed to.
    useful: usize,
    /// Test passes whose token (or pair) has an entry.
    found: usize,
}

impl Served {
    fn rate(&self, served: &[usize; FIRST_LAYERS], layers: std::ops::Range<usize>) -> f64 {
        layers.clone().map(|l| served[l]).sum::<usize>() as f64
            / layers.map(|l| self.routed[l]).sum::<usize>() as f64
    }
}

#[test]
fn the_token_table_serves_the_layers_the_router_prediction_cannot_reach() {
    let work = workload(14_000);
    let routed_of =
        |pass: usize| -> Vec<&[u32]> { (0..FIRST_LAYERS).map(|l| work.routed(pass, l)).collect() };

    // The tier of each of the first layers: 32 slots (a quarter of the
    // experts) filled once from the heat of the recording, as a profile would
    // fill them.
    let mut heat = ExpertHeat::new(FIRST_LAYERS, EXPERTS, 1.0);
    for pass in TRAIN {
        for layer in 0..FIRST_LAYERS {
            heat.record(layer, work.routed(pass, layer));
        }
    }
    let tiers: Vec<SlotSet> = (0..FIRST_LAYERS)
        .map(|layer| {
            let mut set = SlotSet::new(EXPERTS, 32);
            for swap in set.elect(heat.layer(layer), &SwapRule::default()) {
                set.commit(&swap);
            }
            set
        })
        .collect();

    let build = |config: TableConfig| {
        let mut builder = TableBuilder::new(config, EXPERTS);
        for pass in TRAIN {
            builder.record(work.previous(pass), work.tokens[pass], &routed_of(pass));
        }
        // Through the form it is stored in, as a runtime would get it.
        TokenTable::from_bytes(&builder.build().to_bytes()).unwrap()
    };
    // Only experts that make up at least `min_share` of an entry's guesses are
    // read.
    let try_table = |table: &TokenTable, min_share: f32| {
        let mut served = Served::default();
        for pass in TEST {
            let entry = table.lookup(work.previous(pass), work.tokens[pass]);
            served.found += usize::from(entry.is_some());
            for (layer, tier) in tiers.iter().enumerate() {
                let guessed: Vec<u32> = entry
                    .iter()
                    .flat_map(|e| e.layer(layer).filter(|(_, s)| *s >= min_share))
                    .map(|(expert, _)| expert)
                    .collect();
                let routed = work.routed(pass, layer);
                for expert in routed {
                    let in_tier = tier.contains(*expert);
                    served.routed[layer] += 1;
                    served.in_tier[layer] += usize::from(in_tier);
                    served.in_tier_or_table[layer] +=
                        usize::from(in_tier || guessed.contains(expert));
                }
                for expert in guessed.iter().filter(|e| !tier.contains(**e)) {
                    served.reads += 1;
                    served.useful += usize::from(routed.contains(expert));
                }
            }
        }
        served
    };
    let per_token = |n: usize| n as f64 / TEST.len() as f64;

    let full = build(TableConfig::new(FIRST_LAYERS, 8));
    let small = build(TableConfig { capacity: 256, ..TableConfig::new(FIRST_LAYERS, 8) });
    let confident = try_table(&full, 0.5);
    let all = try_table(&full, 0.0);
    let few = try_table(&small, 0.0);
    for (name, table, served) in [("full", &full, &all), ("256 entries", &small, &few)] {
        eprintln!(
            "{name}: {} entries, {} bytes; tokens known {:.3}; first layers served: tier {:.3}, tier and table {:.3} \
             (layers 1 to 3: {:.3} and {:.3}); reads {:.2} per token, {:.2} useful",
            table.len(),
            table.byte_size(),
            per_token(served.found),
            served.rate(&served.in_tier, 0..4),
            served.rate(&served.in_tier_or_table, 0..4),
            served.rate(&served.in_tier, 1..4),
            served.rate(&served.in_tier_or_table, 1..4),
            per_token(served.reads),
            per_token(served.useful),
        );
    }
    eprintln!(
        "guessing only experts that made up half of an entry: reads {:.2} per token, {:.2} useful, served {:.3}",
        per_token(confident.reads),
        per_token(confident.useful),
        confident.rate(&confident.in_tier_or_table, 1..4)
    );

    // Against no table: the tier serves about two fifths of the experts of
    // layers 1 to 3, and with the table's guesses read in time three quarters.
    let tier = all.rate(&all.in_tier, 1..4);
    let with_table = all.rate(&all.in_tier_or_table, 1..4);
    assert!(tier < 0.45, "the tier alone serves {tier:.3}");
    assert!(with_table > tier + 0.25, "{with_table:.3} against {tier:.3}");
    // The experts the table adds are mostly the right ones.
    assert!(all.useful * 10 > all.reads * 5, "{} of {} reads were useful", all.useful, all.reads);

    // Bounded: 256 entries are 256 entries and cost 104 bytes each, and serve less
    // than the table of every token that repeats but more than none.
    assert_eq!(small.len(), 256);
    assert_eq!(small.byte_size(), 16 + 256 * 104);
    assert!(full.len() > small.len() && full.byte_size() == 16 + full.len() * 104);
    let (few_rate, full_rate) =
        (few.rate(&few.in_tier_or_table, 1..4), all.rate(&all.in_tier_or_table, 1..4));
    assert!(few_rate > tier + 0.15 && few_rate < full_rate - 0.01, "{few_rate:.3} {full_rate:.3}");

    // The stored shares let a runtime read fewer guesses for most of the benefit.
    assert!(confident.reads * 5 < all.reads * 4, "at least a fifth fewer reads");
    let precision = |s: &Served| s.useful as f64 / s.reads as f64;
    assert!(precision(&confident) > precision(&all) + 0.05);
    let gain = confident.rate(&confident.in_tier_or_table, 1..4) - tier;
    assert!(gain > 0.75 * (with_table - tier), "{gain:.3} of {:.3}", with_table - tier);
}
