//! A long prompt through a model with routed experts, layer by layer.
//!
//! An ordinary step takes its rows through every layer. For the prompt rows
//! of a model whose experts are not all on the GPU that is the wrong order:
//! a few thousand tokens route to most experts of every layer, and a prompt
//! that runs through the model in chunks of rows needs them again for every
//! chunk, uploaded from host memory or read from disk each time.
//!
//! A **pass** turns the order around. All rows of the prompt run up to the
//! router of the first layer with experts; then that layer's experts are
//! computed for all rows, in waves that fit the room for them
//! (`ares_weights::ExpertWaves`), each expert fetched once; then all rows
//! continue to the next router, and so on. The rows still go through in
//! chunks, because a step's scratch memory holds a limited number of rows, and
//! the hidden state of every row is kept between the pieces.
//!
//! [`Pass`] is that order as a sequence of pieces. Running them, keeping the
//! hidden states, and giving the generating sequences their steps between the
//! pieces is the conductor's part.

use std::ops::Range;

use crate::timing::StepTimeModel;

/// A place in a model at which the computation of a row can stop and go on
/// later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Point {
    /// Before layer `n`. For `n` equal to the number of layers it is the end
    /// of the model, where the logits of a prompt's last row are made.
    Start(u32),
    /// Inside a layer with experts: its router has run for the row, its
    /// experts have not.
    Routed(u32),
    /// Inside the same layer: the outputs of the row's experts are there.
    AfterExperts(u32),
}

impl Point {
    /// The layer the point is in or before.
    pub fn layer(self) -> u32 {
        match self {
            Point::Start(layer) | Point::Routed(layer) | Point::AfterExperts(layer) => layer,
        }
    }

    /// The points of a model in the order a row passes them.
    fn rank(self) -> (u32, u8) {
        match self {
            Point::Start(layer) => (layer, 0),
            Point::Routed(layer) => (layer, 1),
            Point::AfterExperts(layer) => (layer, 2),
        }
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

/// One piece of a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassStep {
    /// Compute the rows `rows` of the pass from `from` to `to`. The rows of a
    /// stretch come in order, so recurrent state and attention over earlier
    /// rows find what the rows before left.
    Rows { from: Point, to: Point, rows: Range<u32> },
    /// Every row of the pass has been routed in `layer`: compute its experts
    /// for all rows, wave by wave.
    Experts { layer: u32 },
}

/// The pieces of a pass over `rows` prompt rows, in order.
///
/// The rows are numbered from 0; which sequence and position a row is (a pass
/// can carry the prompts of several sequences, which then share the fetched
/// experts) is the caller's bookkeeping.
#[derive(Clone, Debug)]
pub struct Pass {
    /// The stretches all rows run through one after the other. Between two of
    /// them lie the experts of the layer the first ends in.
    stretches: Vec<(Point, Point)>,
    rows: u32,
    /// The stretch the next rows belong to, and the first row of it that has
    /// not been handed out.
    stretch: usize,
    next_row: u32,
    /// The layer whose experts come next, when a stretch has just ended at
    /// its router.
    experts: Option<u32>,
}

impl Pass {
    /// A pass over `rows` rows of a model with `has_experts.len()` layers,
    /// of which those marked have routed experts.
    pub fn new(has_experts: &[bool], rows: u32) -> Self {
        let n_layers = has_experts.len() as u32;
        let mut stretches = Vec::new();
        let mut from = Point::Start(0);
        for (layer, _) in has_experts.iter().enumerate().filter(|(_, experts)| **experts) {
            stretches.push((from, Point::Routed(layer as u32)));
            from = Point::AfterExperts(layer as u32);
        }
        stretches.push((from, Point::Start(n_layers)));
        Self { stretches, rows, stretch: 0, next_row: 0, experts: None }
    }

    /// The rows of the pass.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Whether every piece has been handed out.
    pub fn is_done(&self) -> bool {
        self.experts.is_none() && (self.rows == 0 || self.stretch == self.stretches.len())
    }

    /// The stretch the next rows belong to and how many stretches there are:
    /// one more than the model has layers with experts. Step times differ
    /// from stretch to stretch (they cover different numbers of layers), so
    /// this is what a time model of the pass is kept per.
    pub fn stretch(&self) -> (usize, usize) {
        (self.stretch, self.stretches.len())
    }

    /// The next piece: the experts of a layer, or up to `max_rows` rows of the
    /// current stretch. `None` when the pass is done.
    ///
    /// # Panics
    ///
    /// Panics when `max_rows` is 0.
    pub fn next_step(&mut self, max_rows: u32) -> Option<PassStep> {
        assert!(max_rows > 0, "a piece of a pass needs room for a row");
        if let Some(layer) = self.experts.take() {
            return Some(PassStep::Experts { layer });
        }
        if self.is_done() {
            return None;
        }
        let (from, to) = self.stretches[self.stretch];
        let rows = self.next_row..self.rows.min(self.next_row.saturating_add(max_rows));
        self.next_row = rows.end;
        if rows.end == self.rows {
            self.stretch += 1;
            self.next_row = 0;
            if let Point::Routed(layer) = to {
                self.experts = Some(layer);
            }
        }
        Some(PassStep::Rows { from, to, rows })
    }
}

/// Whether the `tokens` prompt rows a sequence still needs are run as a pass.
///
/// They are when the model has experts that are not all on the GPU
/// (`experts_are_fetched`) and the rows do not fit into one step: then
/// ordinary steps would fetch each layer's experts once per step, and a pass
/// fetches them once. A prompt that fits into one step is one chunk either
/// way, and a model whose weights are all on the GPU gains nothing from the
/// other order, so both stay in ordinary steps, where prompt rows share the
/// step with generating sequences.
pub fn runs_as_pass(tokens: u32, max_rows: u32, experts_are_fetched: bool) -> bool {
    experts_are_fetched && tokens > max_rows
}

/// How many rows a piece of a pass may carry between two steps of the
/// generating sequences and decisions, when those may run `slowdown` times
/// slower than they would alone (2.0: half speed while a prompt is read).
///
/// `steps` is the time model of ordinary steps and `window_rows` the rows the
/// next one carries; `stretch` is the time model of the pieces of the
/// stretch the pass is in. The pieces between two ordinary steps may take
/// `slowdown - 1` times the ordinary step. With nobody generating
/// (`window_rows` 0) there is no limit. At least one row is always allowed,
/// or the pass would never finish, and before either model has measured
/// anything the answer is `fallback`.
pub fn pass_rows_between(
    steps: &StepTimeModel,
    window_rows: u32,
    stretch: &StepTimeModel,
    slowdown: f64,
    fallback: u32,
) -> u32 {
    if window_rows == 0 {
        return u32::MAX;
    }
    let (Some(ordinary), Some((base, per_row))) =
        (steps.predict(window_rows), stretch.base_and_per_row())
    else {
        return fallback.max(1);
    };
    if per_row <= 0.0 {
        return u32::MAX;
    }
    let rows = ((slowdown - 1.0).max(0.0) * ordinary - base) / per_row;
    if rows >= f64::from(u32::MAX) { u32::MAX } else { (rows.max(0.0) as u32).max(1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs a pass with pieces of the sizes `max_rows` yields and checks every
    /// rule of the order on the way. Returns the pieces.
    fn run(has_experts: &[bool], rows: u32, mut max_rows: impl FnMut() -> u32) -> Vec<PassStep> {
        let n_layers = has_experts.len() as u32;
        let mut pass = Pass::new(has_experts, rows);
        assert_eq!(pass.rows(), rows);
        // Where each row is.
        let mut at = vec![Point::Start(0); rows as usize];
        let mut steps = Vec::new();
        while let Some(step) = pass.next_step(max_rows()) {
            match &step {
                PassStep::Rows { from, to, rows: range } => {
                    assert!(from < to, "{from:?} to {to:?}");
                    assert!(!range.is_empty());
                    // The rows before these have done the stretch, the rows
                    // after them have not: chunks come in order.
                    for (row, place) in at.iter().enumerate() {
                        let expected = if (row as u32) < range.start { to } else { from };
                        assert_eq!(place, expected, "row {row} before {step:?}");
                    }
                    for place in &mut at[range.start as usize..range.end as usize] {
                        *place = *to;
                    }
                }
                PassStep::Experts { layer } => {
                    assert!(has_experts[*layer as usize]);
                    assert!(
                        at.iter().all(|place| *place == Point::Routed(*layer)),
                        "the experts of layer {layer} need every row routed"
                    );
                    at.fill(Point::AfterExperts(*layer));
                }
            }
            steps.push(step);
        }
        assert!(pass.is_done());
        assert!(pass.next_step(1).is_none(), "a finished pass stays finished");
        assert!(
            at.iter().all(|place| *place == Point::Start(n_layers)),
            "every row reached the end"
        );
        steps
    }

    #[test]
    fn rows_go_to_a_router_together_then_the_experts_then_on() {
        // Four layers, the two in the middle with experts; 5 rows, 2 at a time.
        let steps = run(&[false, true, true, false], 5, || 2);
        let rows = |from, to, rows: Range<u32>| PassStep::Rows { from, to, rows };
        let (s0, r1, a1, r2, a2, end) = (
            Point::Start(0),
            Point::Routed(1),
            Point::AfterExperts(1),
            Point::Routed(2),
            Point::AfterExperts(2),
            Point::Start(4),
        );
        assert_eq!(
            steps,
            [
                rows(s0, r1, 0..2),
                rows(s0, r1, 2..4),
                rows(s0, r1, 4..5),
                PassStep::Experts { layer: 1 },
                rows(a1, r2, 0..2),
                rows(a1, r2, 2..4),
                rows(a1, r2, 4..5),
                PassStep::Experts { layer: 2 },
                rows(a2, end, 0..2),
                rows(a2, end, 2..4),
                rows(a2, end, 4..5),
            ]
        );
    }

    #[test]
    fn every_shape_of_model_and_piece_keeps_the_order() {
        let models: [&[bool]; 6] = [
            &[false],
            &[true],
            &[false, false, false],
            &[true, true, true, true],
            &[true, false, false, true],
            &[false, true, false, true, false, false, true, false],
        ];
        for has_experts in models {
            let n_experts = has_experts.iter().filter(|e| **e).count();
            for rows in [1u32, 2, 7, 64, 65, 1000] {
                for max in [1u32, 3, 64, 1000, u32::MAX] {
                    let steps = run(has_experts, rows, || max);
                    let experts = steps.iter().filter(|s| matches!(s, PassStep::Experts { .. }));
                    assert_eq!(experts.count(), n_experts, "{has_experts:?}");
                    let chunks = rows.div_ceil(max) as usize;
                    assert_eq!(steps.len(), (n_experts + 1) * chunks + n_experts);
                }
                // Pieces of changing size, as a time budget gives them.
                let mut sizes = [5u32, 1, 17, 2, 64].into_iter().cycle();
                run(has_experts, rows, || sizes.next().unwrap());
            }
        }
    }

    #[test]
    fn a_model_without_experts_is_one_stretch_and_no_rows_are_no_pieces() {
        let steps = run(&[false; 3], 10, || 4);
        let whole = |rows| PassStep::Rows { from: Point::Start(0), to: Point::Start(3), rows };
        assert_eq!(steps, [whole(0..4), whole(4..8), whole(8..10)]);

        let mut empty = Pass::new(&[true, true], 0);
        assert!(empty.is_done());
        assert_eq!(empty.next_step(8), None);
    }

    #[test]
    fn the_stretch_counts_up_as_the_pass_goes() {
        let mut pass = Pass::new(&[true, false, true], 3);
        assert_eq!(pass.stretch(), (0, 3));
        assert!(matches!(pass.next_step(2), Some(PassStep::Rows { .. })));
        assert_eq!(pass.stretch(), (0, 3), "the first stretch is not through yet");
        assert!(matches!(pass.next_step(2), Some(PassStep::Rows { .. })));
        assert_eq!(pass.stretch(), (1, 3));
        assert!(!pass.is_done());
        assert_eq!(pass.next_step(2), Some(PassStep::Experts { layer: 0 }));
        assert_eq!(pass.stretch(), (1, 3));
    }

    #[test]
    fn points_are_ordered_as_a_row_passes_them() {
        let order = [
            Point::Start(0),
            Point::Routed(0),
            Point::AfterExperts(0),
            Point::Start(1),
            Point::Start(2),
            Point::Routed(2),
            Point::AfterExperts(2),
            Point::Start(3),
        ];
        assert!(order.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(order.map(Point::layer), [0, 0, 0, 1, 2, 2, 2, 3]);
    }

    #[test]
    fn a_pass_is_for_prompts_longer_than_a_step_when_experts_are_fetched() {
        assert!(runs_as_pass(4096, 1024, true));
        assert!(runs_as_pass(1025, 1024, true));
        assert!(!runs_as_pass(1024, 1024, true), "one step holds it: one chunk either way");
        assert!(!runs_as_pass(100_000, 1024, false), "all weights are on the GPU");
    }

    #[test]
    fn the_pieces_between_two_steps_fit_the_allowed_slowdown() {
        // Ordinary steps: 20 ms plus 0.5 ms per row. Pieces of the stretch:
        // 2 ms plus 0.05 ms per row.
        let mut steps = StepTimeModel::default();
        let mut stretch = StepTimeModel::default();
        for rows in [1u32, 4, 8, 2, 6] {
            steps.observe(rows, 20.0 + 0.5 * f64::from(rows));
        }
        for rows in [100u32, 400, 800, 200] {
            stretch.observe(rows, 2.0 + 0.05 * f64::from(rows));
        }
        // 4 generating rows take 22 ms. At half speed the pass gets 22 ms
        // between two of their steps: (22 - 2) / 0.05 = 400 rows.
        let rows = pass_rows_between(&steps, 4, &stretch, 2.0, 64);
        assert!((399..=401).contains(&rows), "{rows}");
        // A tenth slower only: 2.2 ms, of which 2 are the piece's fixed cost.
        let rows = pass_rows_between(&steps, 4, &stretch, 1.1, 64);
        assert!((3..=5).contains(&rows), "{rows}");
        // No slowdown allowed still lets the pass move.
        assert_eq!(pass_rows_between(&steps, 4, &stretch, 1.0, 64), 1);
        // Nobody generating: no limit.
        assert_eq!(pass_rows_between(&steps, 0, &stretch, 2.0, 64), u32::MAX);
        // Nothing measured yet: the fallback.
        let unknown = StepTimeModel::default();
        assert_eq!(pass_rows_between(&unknown, 4, &stretch, 2.0, 64), 64);
        assert_eq!(pass_rows_between(&steps, 4, &unknown, 2.0, 0), 1);
    }
}
