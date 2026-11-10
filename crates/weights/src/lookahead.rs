//! How far ahead of the layers the experts are read.

/// How the distance is chosen from the measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookaheadRule {
    /// The farthest distance considered, in layers. The predictor has to
    /// produce a prediction for every distance up to this one (the engines
    /// that do this make the predictions for the next two layers in one
    /// operation), because their hit rates are what the choice rests on.
    pub max_distance: usize,
    /// The distance used until the read time, the layer time and the hit
    /// rate of every distance have been measured. Two layers: on the one
    /// machine where this was reported (Run My AI's test laptop) a layer
    /// computes faster than one expert comes from the SSD, and two layers of
    /// lead beat one and three.
    pub start: usize,
    /// A longer distance replaces a shorter one only if the wait it leaves
    /// is smaller by this share (0.05 is 5 %). A longer distance keeps more
    /// guessed experts in flight and in the tier for longer, and wastes more
    /// reads when wrong, which the wait does not show; and it keeps the
    /// choice from flipping between two distances that are about equal.
    pub tolerance: f64,
    /// How much of the weight of an observation is left after the next one
    /// (above 0, up to 1), for each of the measurements. 0.998 remembers
    /// about the last 500 of each: a few seconds of layers, and enough of them
    /// that a hit rate of one layer in two is known to a few percent.
    pub decay: f64,
}

impl Default for LookaheadRule {
    fn default() -> Self {
        Self { max_distance: 4, start: 2, tolerance: 0.05, decay: 0.998 }
    }
}

/// The least number of observed layers before a hit rate is believed.
const MIN_LAYERS: f64 = 32.0;

/// A decayed mean.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    sum: f64,
    weight: f64,
}

impl Mean {
    fn add(&mut self, value: f64, decay: f64) {
        self.sum = self.sum * decay + value;
        self.weight = self.weight * decay + 1.0;
    }

    fn value(&self) -> Option<f64> {
        (self.weight > 0.0).then(|| self.sum / self.weight)
    }
}

/// Layers whose reads were predicted, and picks that were, decayed.
#[derive(Clone, Copy, Debug, Default)]
struct Hits {
    /// Layers that needed reads.
    layers: f64,
    /// Of them, layers where every expert had been predicted.
    covered: f64,
    /// Experts that needed reads.
    wanted: f64,
    /// Of them, experts that had been predicted.
    hits: f64,
}

/// Which distance to predict for: a prediction made when a layer's router
/// has run is read for the layer that many layers later.
///
/// A longer distance gives the read more time to arrive, and a worse
/// prediction: the router of a later layer is predicted from an earlier
/// hidden state, and the further it is, the more the state has changed. Which
/// distance is best depends on the machine and on the model, so it is
/// measured. Three things are:
///
/// - **Read time**: how long an expert takes from the moment it is guessed to
///   the moment its data is there ([`Lookahead::observe_read`]). That is the
///   wait for a free I/O lane and the read itself; on an idle SSD it is the
///   latency of the read, and when the SSD is busy it is what the guess has to
///   hide.
/// - **Layer time**: how long a layer takes from its router output to the
///   next layer's, when nothing waits for a read
///   ([`Lookahead::observe_layer`]). It is the time a read has to hide behind
///   for each layer of distance. A layer that waited for an expert is left
///   out: its time says how long the read took, not how long the layer
///   computes.
/// - **Hit rate** of each distance ([`Lookahead::observe_recall`]): of the
///   experts a layer routed that were not in the tier when the prediction was
///   made, which are the ones a prediction exists to read, how many had been
///   predicted from that many layers earlier.
///
/// The runtime gets the last one without reading anything: the predictor
/// is asked for every distance up to `max_distance`, each prediction is held
/// until its layer has routed and compared with the router's picks, and only
/// the guesses of the chosen distance are read.
///
/// From them follows what a layer that needs reads waits
/// ([`Lookahead::stall`]). The experts of a layer are read together and the
/// layer needs all of them, so it waits for the last one. If every expert
/// was predicted `d` layers ahead, the reads had `d` layers of compute to
/// hide behind and the layer waits for what is left of the slowest. If one was
/// not, a demand read starts when the router has run and the layer waits for
/// all of it. The distance with the least wait is chosen
/// ([`Lookahead::distance`]).
#[derive(Clone, Debug)]
pub struct Lookahead {
    rule: LookaheadRule,
    read_ms: Mean,
    layer_ms: Mean,
    /// Index `d - 1` for distance `d`.
    hits: Vec<Hits>,
}

impl Lookahead {
    pub fn new(rule: LookaheadRule) -> Self {
        assert!(rule.max_distance >= 1, "at least one distance");
        assert!(rule.decay > 0.0 && rule.decay <= 1.0, "decay must be in (0, 1]");
        assert!(rule.tolerance >= 0.0, "tolerance must not be negative");
        Self {
            rule,
            read_ms: Mean::default(),
            layer_ms: Mean::default(),
            hits: vec![Hits::default(); rule.max_distance],
        }
    }

    /// Records that a guessed expert took `ms` milliseconds from being
    /// guessed to being there. Not a number, or a negative one, is ignored.
    pub fn observe_read(&mut self, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            self.read_ms.add(ms, self.rule.decay);
        }
    }

    /// Records that a layer took `ms` milliseconds from its router output to
    /// the next layer's, without waiting for a read.
    pub fn observe_layer(&mut self, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            self.layer_ms.add(ms, self.rule.decay);
        }
    }

    /// Records one layer: of the `wanted` experts it routed that were not in
    /// the tier when the prediction was made, `hits` had been predicted for
    /// it from `distance` layers earlier. A layer that needed no reads
    /// (`wanted` of 0) says nothing and is ignored, as are distances outside
    /// `1..=max_distance`.
    pub fn observe_recall(&mut self, distance: usize, hits: u32, wanted: u32) {
        let Some(seen) = distance.checked_sub(1).and_then(|d| self.hits.get_mut(d)) else {
            return;
        };
        if wanted == 0 {
            return;
        }
        let hits = hits.min(wanted);
        let decay = self.rule.decay;
        seen.layers = seen.layers * decay + 1.0;
        seen.covered = seen.covered * decay + f64::from(u8::from(hits == wanted));
        seen.wanted = seen.wanted * decay + f64::from(wanted);
        seen.hits = seen.hits * decay + f64::from(hits);
    }

    /// The share of the experts that needed reads that were predicted from
    /// `distance` layers earlier, or `None` before enough layers were seen.
    /// The decision rests on [`Lookahead::coverage`]; this is the number
    /// predictors are usually judged by.
    pub fn recall(&self, distance: usize) -> Option<f64> {
        let seen = self.seen(distance)?;
        Some(seen.hits / seen.wanted)
    }

    /// The share of the layers that needed reads in which every expert that
    /// needed one was predicted from `distance` layers earlier, or `None`
    /// before enough layers were seen.
    pub fn coverage(&self, distance: usize) -> Option<f64> {
        let seen = self.seen(distance)?;
        Some(seen.covered / seen.layers)
    }

    fn seen(&self, distance: usize) -> Option<&Hits> {
        let seen = self.hits.get(distance.checked_sub(1)?)?;
        (seen.layers >= MIN_LAYERS).then_some(seen)
    }

    /// What a layer that needs reads waits, in milliseconds on average, when
    /// predictions are read `distance` layers ahead: the share of layers that
    /// were covered times what is left of the read after `distance` layers,
    /// plus the share that were not times the whole read. `None` before the
    /// read time, the layer time and the coverage of this distance have been
    /// seen.
    ///
    /// The model supposes that a layer's reads run side by side, which they do
    /// while the SSD has lanes to spare. A layer that needs more reads than the
    /// SSD can run at once waits longer than this says, for every distance
    /// alike.
    pub fn stall(&self, distance: usize) -> Option<f64> {
        let (read, layer) = (self.read_ms.value()?, self.layer_ms.value()?);
        let covered = self.coverage(distance)?;
        let left = (read - distance as f64 * layer).max(0.0);
        Some(covered * left + (1.0 - covered) * read)
    }

    /// The distance to read for: the one with the least [`Lookahead::stall`],
    /// where a longer distance must beat the best shorter one by
    /// `rule.tolerance`. Until everything is measured for every distance up
    /// to `max_distance`, it is `rule.start`.
    pub fn distance(&self) -> usize {
        let start = self.rule.start.clamp(1, self.rule.max_distance);
        let mut best: Option<(usize, f64)> = None;
        for distance in 1..=self.rule.max_distance {
            let Some(stall) = self.stall(distance) else { return start };
            match best {
                Some((_, least)) if stall >= least * (1.0 - self.rule.tolerance) => {}
                _ => best = Some((distance, stall)),
            }
        }
        best.map_or(start, |(distance, _)| distance)
    }
}

impl Default for Lookahead {
    fn default() -> Self {
        Self::new(LookaheadRule::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `layers` layers of which a share `coverage` were covered,
    /// spread evenly, as distance `distance`.
    fn observe_coverage(lookahead: &mut Lookahead, distance: usize, coverage: f64, layers: u32) {
        let mut owed = 0.0;
        for _ in 0..layers {
            owed += coverage;
            if owed >= 1.0 {
                owed -= 1.0;
                lookahead.observe_recall(distance, 3, 3);
            } else {
                lookahead.observe_recall(distance, 2, 3);
            }
        }
    }

    /// A lookahead that has seen a read of `read` ms, layers of `layer` ms
    /// and the coverages `coverages` (index 0 is distance 1), each over 1000
    /// layers.
    fn measured(read: f64, layer: f64, coverages: &[f64]) -> Lookahead {
        // Nothing is forgotten, so the shares are the ones given.
        let rule =
            LookaheadRule { max_distance: coverages.len(), decay: 1.0, ..Default::default() };
        let mut lookahead = Lookahead::new(rule);
        lookahead.observe_read(read);
        lookahead.observe_layer(layer);
        for (i, coverage) in coverages.iter().enumerate() {
            observe_coverage(&mut lookahead, i + 1, *coverage, 1000);
        }
        lookahead
    }

    #[test]
    fn two_layers_are_used_until_everything_is_measured() {
        let mut lookahead = Lookahead::default();
        assert_eq!(lookahead.distance(), 2);
        lookahead.observe_read(3.0);
        lookahead.observe_layer(2.0);
        assert_eq!(lookahead.distance(), 2, "no hit rates yet");
        for distance in 1..=3 {
            observe_coverage(&mut lookahead, distance, 0.8, 100);
        }
        assert_eq!(lookahead.stall(4), None);
        assert_eq!(lookahead.distance(), 2, "distance 4 is missing");
        // Too few layers to believe a hit rate, and layers that needed
        // nothing say nothing.
        observe_coverage(&mut lookahead, 4, 0.8, 10);
        lookahead.observe_recall(4, 0, 0);
        assert_eq!((lookahead.coverage(4), lookahead.recall(4)), (None, None));
        observe_coverage(&mut lookahead, 4, 0.8, 100);
        assert!((lookahead.coverage(4).unwrap() - 0.8).abs() < 0.05);
        assert!(lookahead.stall(4).is_some());

        // The start is held within the distances that exist.
        let one = Lookahead::new(LookaheadRule { max_distance: 1, ..Default::default() });
        assert_eq!(one.distance(), 1);
    }

    #[test]
    fn a_covered_layer_waits_for_what_is_left_of_its_reads_and_another_for_all_of_them() {
        // A read takes 3 ms, a layer 2 ms; in 70 % of the layers every expert
        // was predicted one layer ahead, in 80 % two layers ahead.
        let lookahead = measured(3.0, 2.0, &[0.7, 0.8]);
        // One layer ahead the reads have 2 ms of the 3 they need.
        let one = lookahead.stall(1).unwrap();
        assert!((one - (0.7 * 1.0 + 0.3 * 3.0)).abs() < 0.02, "{one}");
        // Two layers ahead they are there in time; only the other layers wait.
        let two = lookahead.stall(2).unwrap();
        assert!((two - 0.2 * 3.0).abs() < 0.02, "{two}");
        assert_eq!(lookahead.distance(), 2);

        // The hit rate of experts is kept as well.
        let mut lookahead = Lookahead::default();
        for _ in 0..100 {
            lookahead.observe_recall(1, 3, 4);
        }
        assert!((lookahead.recall(1).unwrap() - 0.75).abs() < 1e-9);
        assert_eq!(lookahead.coverage(1), Some(0.0));
    }

    #[test]
    fn the_distance_follows_the_read_time_the_layer_time_and_the_hit_rates() {
        let falling = [0.80, 0.75, 0.70, 0.65];
        // (read ms, layer ms, coverages, distance)
        let cases: [(f64, f64, &[f64], usize); 6] = [
            // The read is faster than a layer: one layer of lead is enough
            // and the coverage only falls from there.
            (1.0, 2.0, &falling, 1),
            // The read takes longer than a layer: two layers.
            (3.0, 2.0, &falling, 2),
            // Much longer: three.
            (5.0, 2.0, &falling, 3),
            // So long that no distance hides it: every layer of lead still
            // hides some of it, so the farthest.
            (20.0, 2.0, &falling, 4),
            // The same slow read, but the coverage collapses with distance:
            // better a late read of the right experts than an early one of
            // the wrong.
            (3.0, 2.0, &[0.80, 0.40, 0.20, 0.10], 1),
            // Two distances that are equal within the tolerance: the shorter.
            (2.0, 2.0, &[0.900, 0.902, 0.90, 0.90], 1),
        ];
        for (read, layer, coverages, expected) in cases {
            let lookahead = measured(read, layer, coverages);
            assert_eq!(
                lookahead.distance(),
                expected,
                "read {read} ms, layer {layer} ms, coverages {coverages:?}: stalls {:?}",
                (1..=4).map(|d| lookahead.stall(d).unwrap()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn it_follows_a_machine_that_changes_and_ignores_nonsense() {
        let mut lookahead = Lookahead::default();
        let mut jitter = 0u32;
        let mut feed = |lookahead: &mut Lookahead, read: f64, layer: f64| {
            for i in 0..3000u32 {
                jitter = jitter.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = 0.9 + 0.2 * f64::from(jitter >> 16) / 65536.0;
                lookahead.observe_read(read * noise);
                lookahead.observe_layer(layer * noise);
                for (d, coverage) in [0.80, 0.75, 0.70, 0.65].into_iter().enumerate() {
                    // About `coverage` of the layers are covered.
                    let covered = (f64::from(i * 7 % 20) + 0.5) / 20.0 < coverage;
                    lookahead.observe_recall(d + 1, if covered { 3 } else { 2 }, 3);
                }
                lookahead.observe_read(f64::NAN);
                lookahead.observe_layer(-1.0);
                lookahead.observe_recall(0, 3, 3);
                lookahead.observe_recall(9, 3, 3);
            }
        };
        feed(&mut lookahead, 1.0, 2.0);
        assert_eq!(lookahead.distance(), 1, "fast SSD");
        // The SSD is busy with something else: reads take five times as long.
        feed(&mut lookahead, 5.0, 2.0);
        assert_eq!(lookahead.distance(), 3);
        // Layers got slower (a longer context): the same reads are hidden by two.
        feed(&mut lookahead, 5.0, 3.0);
        assert_eq!(lookahead.distance(), 2);
    }
}
