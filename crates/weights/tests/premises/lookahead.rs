//! Item 4: the lookahead distance as a measured setting.
//!
//! The machine is the one of the cache premise ([`crate::cache`]) with its SSD
//! changed: how long a read takes, and how many can run at once. A runtime
//! starts at the distance of two layers, reads for it, and meanwhile asks the
//! predictor for every distance and compares each prediction with what the
//! layer then routed. From those, the time of a layer and the time from a
//! guess to its data, [`Lookahead`] says which distance to read for. The
//! test then runs the whole tier at each distance and checks what it chose
//! against what waited least.

use ares_weights::{Lookahead, LookaheadRule};

use crate::cache::{Arm, Params, run, workload};

/// (layers one read takes, reads at once)
const MACHINES: [(f64, usize); 6] =
    [(0.8, 16), (1.2, 16), (2.0, 16), (2.5, 16), (3.0, 24), (4.0, 32)];

#[test]
fn the_distance_follows_what_the_machine_measures() {
    let work = workload(6000);
    let mut chosen_total = 0.0;
    let mut two_total = 0.0;
    for (read, lanes) in MACHINES {
        let waits: Vec<f64> = (1..=4)
            .map(|distance| {
                let params = Params { read, lanes, distance, ..Params::default() };
                run(&work, Arm::Late, params).wait_per_token()
            })
            .collect();
        let best = 1 + waits.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;

        // What the runtime sees while it reads for two layers ahead.
        let started = run(&work, Arm::Late, Params { read, lanes, ..Params::default() });
        let mut lookahead = Lookahead::new(LookaheadRule::default());
        assert_eq!(lookahead.distance(), 2, "before anything is measured");
        for latency in &started.guess_latency {
            lookahead.observe_read(*latency);
        }
        for (distance, hits, wanted) in &started.predictions {
            lookahead.observe_recall(*distance, *hits, *wanted);
        }
        // A layer takes one layer of time, by the units of the machine.
        for _ in 0..started.tokens * 12 {
            lookahead.observe_layer(1.0);
        }
        let chosen = lookahead.distance();
        eprintln!(
            "read {read} layers, {lanes} lanes: waiting per token at 1 to 4 layers ahead {:?}; \
             coverage {:?}; chosen {chosen}, best {best}",
            waits.iter().map(|w| (w * 10.0).round() / 10.0).collect::<Vec<_>>(),
            (1..=4)
                .map(|d| (lookahead.coverage(d).unwrap() * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
        );
        assert_eq!(chosen, best, "read {read}, lanes {lanes}: waits {waits:?}");
        chosen_total += waits[chosen - 1];
        two_total += waits[1];
        assert!(waits[chosen - 1] <= waits[1], "{waits:?}");
    }
    // Over all six machines, reading for the distance that was measured waits
    // less than reading two layers ahead everywhere.
    assert!(chosen_total < 0.95 * two_total, "{chosen_total} against {two_total}");
}
