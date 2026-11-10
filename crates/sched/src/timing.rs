//! What a step costs.

/// What a step carries, as far as its cost goes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Load {
    /// The rows of the step.
    pub rows: u32,
    /// The cells the rows read through attention, summed over the rows: for
    /// dense attention a row at position `p` reads `p` cells. See
    /// [`Load::chunk`].
    pub cells: u64,
}

impl Load {
    /// The load of `rows` consecutive rows of one sequence that start at
    /// position `context`: row `i` reads the `context + i` cells before it.
    pub fn chunk(context: u64, rows: u32) -> Self {
        let n = u64::from(rows);
        Self { rows, cells: n * context + n * n.saturating_sub(1) / 2 }
    }

    /// The load of two parts of one step together.
    pub fn plus(self, other: Load) -> Self {
        Self { rows: self.rows + other.rows, cells: self.cells + other.cells }
    }
}

/// The fitted cost of a step in milliseconds:
/// `base_ms + per_row_ms * rows + per_cell_ms * cells`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCost {
    /// What a step costs whatever it carries: one read of the weights.
    pub base_ms: f64,
    /// What a row adds that does not depend on its context.
    pub per_row_ms: f64,
    /// What a row adds per cell it reads: attention.
    pub per_cell_ms: f64,
}

impl StepCost {
    /// The expected time of a step that carries `load`.
    pub fn ms(&self, load: Load) -> f64 {
        self.base_ms + self.per_row_ms * f64::from(load.rows) + self.per_cell_ms * load.cells as f64
    }

    /// What one more row at position `context` adds to a step. This is the
    /// `per_row_ms` to give [`best_draft_len`](crate::best_draft_len) for a
    /// sequence that deep.
    pub fn row_ms(&self, context: u64) -> f64 {
        self.per_row_ms + self.per_cell_ms * context as f64
    }

    /// How many consecutive rows of a sequence at position `context` fit in
    /// `extra_ms`. The rows of a chunk read each other, so the cost of `x`
    /// rows is a quadratic in `x`, `per_cell / 2 * x^2 + (per_row + per_cell
    /// * (context - 1/2)) * x`, and this is its positive root.
    pub fn rows_within(&self, extra_ms: f64, context: u64) -> u32 {
        if extra_ms.is_nan() || extra_ms <= 0.0 {
            return 0;
        }
        let a = self.per_cell_ms / 2.0;
        let b = self.per_row_ms + self.per_cell_ms * (context as f64 - 0.5);
        let rows = if a > 0.0 {
            // The form that does not cancel when `a` is tiny against `b`.
            2.0 * extra_ms / (b + (b * b + 4.0 * a * extra_ms).sqrt())
        } else if b > 0.0 {
            extra_ms / b
        } else {
            f64::INFINITY
        };
        if rows >= f64::from(u32::MAX) { u32::MAX } else { rows as u32 }
    }
}

/// A running estimate of step time as `base + per_row * rows + per_cell *
/// cells` milliseconds.
///
/// Decode is bound by reading the weights once per step, whatever the number
/// of rows; every row adds compute on top, and attention adds to each row what
/// its context costs. So a chunk of a prompt at 100 000 tokens costs more than
/// the same chunk at 1 000, and a long chunk costs a quadratic in its length
/// (SGLang fits that quadratic to a warm-up; here the same three numbers are
/// fitted to the steps that ran). The fit forgets old steps gradually, because
/// the numbers move with what else the machine is doing.
///
/// [`StepTimeModel::cost`] is the fit with the context term.
/// [`StepTimeModel::base_and_per_row`] is the line over rows alone, in which
/// the context of the observed steps is folded into both numbers; it is what a
/// caller without positions at hand uses.
#[derive(Clone, Debug)]
pub struct StepTimeModel {
    decay: f64,
    weight: f64,
    sum_x: f64,
    sum_c: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_xc: f64,
    sum_cc: f64,
    sum_xy: f64,
    sum_cy: f64,
}

impl Default for StepTimeModel {
    fn default() -> Self {
        Self::new(0.98)
    }
}

/// A variance below this share of the squared mean is no variation at all.
const FLAT: f64 = 1e-6;

impl StepTimeModel {
    /// A model that weighs each step `decay` times the step after it
    /// (0 < `decay` <= 1; 0.98 remembers about the last 50 steps).
    pub fn new(decay: f64) -> Self {
        assert!(decay > 0.0 && decay <= 1.0, "decay must be in (0, 1]");
        Self {
            decay,
            weight: 0.0,
            sum_x: 0.0,
            sum_c: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_xc: 0.0,
            sum_cc: 0.0,
            sum_xy: 0.0,
            sum_cy: 0.0,
        }
    }

    /// Records that a step of `rows` rows took `ms` milliseconds, for a caller
    /// that does not know what the rows read.
    pub fn observe(&mut self, rows: u32, ms: f64) {
        self.observe_load(Load { rows, cells: 0 }, ms);
    }

    /// Records that a step that carried `load` took `ms` milliseconds.
    pub fn observe_load(&mut self, load: Load, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        let x = f64::from(load.rows);
        let c = load.cells as f64;
        for sum in [
            &mut self.weight,
            &mut self.sum_x,
            &mut self.sum_c,
            &mut self.sum_y,
            &mut self.sum_xx,
            &mut self.sum_xc,
            &mut self.sum_cc,
            &mut self.sum_xy,
            &mut self.sum_cy,
        ] {
            *sum *= self.decay;
        }
        self.weight += 1.0;
        self.sum_x += x;
        self.sum_c += c;
        self.sum_y += ms;
        self.sum_xx += x * x;
        self.sum_xc += x * c;
        self.sum_cc += c * c;
        self.sum_xy += x * ms;
        self.sum_cy += c * ms;
    }

    /// `(base, per_row)` in milliseconds, or `None` before any step was seen.
    ///
    /// While all observed steps had (nearly) the same number of rows the
    /// slope is unknown; then the whole time is attributed to the rows, which
    /// overestimates the cost of adding rows and keeps decisions cautious
    /// until a differently sized step has run.
    pub fn base_and_per_row(&self) -> Option<(f64, f64)> {
        if self.weight <= 0.0 {
            return None;
        }
        let mean_x = self.sum_x / self.weight;
        let mean_y = self.sum_y / self.weight;
        let var_x = self.sum_xx / self.weight - mean_x * mean_x;
        if var_x < FLAT * mean_x.max(1.0).powi(2) {
            return Some((0.0, mean_y / mean_x.max(1.0)));
        }
        let cov = self.sum_xy / self.weight - mean_x * mean_y;
        let per_row = (cov / var_x).max(0.0);
        let base = (mean_y - per_row * mean_x).max(0.0);
        Some((base, per_row))
    }

    /// The expected time of a step of `rows` rows in milliseconds, at the
    /// contexts of the steps that were observed.
    pub fn predict(&self, rows: u32) -> Option<f64> {
        let (base, per_row) = self.base_and_per_row()?;
        Some(base + per_row * f64::from(rows))
    }

    /// The fitted cost with its context term, or `None` before any step was
    /// seen.
    ///
    /// No number is ever negative. Where the steps seen so far cannot tell
    /// two terms apart, the simpler model is returned:
    ///
    /// - cells that did not vary, or varied only in step with the rows (one
    ///   sequence at one depth), give the line of
    ///   [`StepTimeModel::base_and_per_row`] with `per_cell_ms` 0: the context
    ///   is inside the other two numbers, which is right for steps like the
    ///   ones seen and unknown for others;
    /// - rows that did not vary while cells did give `base_ms` 0 and
    ///   everything that is not attention on the rows, the same caution as in
    ///   the line.
    pub fn cost(&self) -> Option<StepCost> {
        let (line_base, line_per_row) = self.base_and_per_row()?;
        let line = StepCost { base_ms: line_base, per_row_ms: line_per_row, per_cell_ms: 0.0 };
        let w = self.weight;
        let (mean_x, mean_c, mean_y) = (self.sum_x / w, self.sum_c / w, self.sum_y / w);
        let var_x = self.sum_xx / w - mean_x * mean_x;
        let var_c = self.sum_cc / w - mean_c * mean_c;
        if var_c < FLAT * mean_c.max(1.0).powi(2) {
            return Some(line);
        }
        let cov_cy = self.sum_cy / w - mean_c * mean_y;
        if var_x < FLAT * mean_x.max(1.0).powi(2) {
            let per_cell = (cov_cy / var_c).max(0.0);
            let rest = (mean_y - per_cell * mean_c).max(0.0);
            return Some(StepCost {
                base_ms: 0.0,
                per_row_ms: rest / mean_x.max(1.0),
                per_cell_ms: per_cell,
            });
        }
        let cov_xc = self.sum_xc / w - mean_x * mean_c;
        let cov_xy = self.sum_xy / w - mean_x * mean_y;
        let det = var_x * var_c - cov_xc * cov_xc;
        // Rows and cells that move together leave one direction unknown.
        if det < 1e-4 * var_x * var_c {
            return Some(line);
        }
        let per_row = (cov_xy * var_c - cov_cy * cov_xc) / det;
        let per_cell = (cov_cy * var_x - cov_xy * cov_xc) / det;
        if per_cell <= 0.0 {
            return Some(line);
        }
        if per_row < 0.0 {
            // All of the slope is context: the fit over cells alone.
            let per_cell = (cov_cy / var_c).max(0.0);
            return Some(StepCost {
                base_ms: (mean_y - per_cell * mean_c).max(0.0),
                per_row_ms: 0.0,
                per_cell_ms: per_cell,
            });
        }
        let base = (mean_y - per_row * mean_x - per_cell * mean_c).max(0.0);
        Some(StepCost { base_ms: base, per_row_ms: per_row, per_cell_ms: per_cell })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Step times measured with llama.cpp on the target machine (Qwen3.8-27B,
    /// tensor split over two GPUs, 16k tokens of context).
    const MEASURED: [(u32, f64); 8] = [
        (1, 33.5),
        (3, 36.9),
        (8, 50.6),
        (16, 57.7),
        (32, 71.9),
        (64, 104.5),
        (128, 177.8),
        (256, 325.8),
    ];

    #[test]
    fn a_line_fits_measured_step_times() {
        let mut model = StepTimeModel::new(1.0);
        for (rows, ms) in MEASURED {
            model.observe(rows, ms);
        }
        let (base, per_row) = model.base_and_per_row().unwrap();
        assert!((30.0..45.0).contains(&base), "base {base}");
        assert!((1.0..1.3).contains(&per_row), "per row {per_row}");
        for (rows, ms) in MEASURED {
            let predicted = model.predict(rows).unwrap();
            assert!((predicted - ms).abs() / ms < 0.2, "{rows} rows: {predicted} vs {ms}");
        }
    }

    #[test]
    fn follows_a_change_and_ignores_nonsense() {
        let mut model = StepTimeModel::new(0.9);
        assert_eq!(model.predict(4), None);
        for i in 0..200u32 {
            let rows = 1 + i % 8;
            model.observe(rows, 10.0 + 2.0 * f64::from(rows));
        }
        assert!((model.predict(4).unwrap() - 18.0).abs() < 0.5);
        // The context got deeper: every step is 20 ms slower.
        for i in 0..200u32 {
            let rows = 1 + i % 8;
            model.observe(rows, 30.0 + 2.0 * f64::from(rows));
            model.observe(rows, f64::NAN);
            model.observe(rows, -5.0);
        }
        assert!((model.predict(4).unwrap() - 38.0).abs() < 0.5);
    }

    #[test]
    fn one_step_size_gives_a_cautious_slope() {
        let mut model = StepTimeModel::default();
        for _ in 0..10 {
            model.observe(2, 40.0);
        }
        let (base, per_row) = model.base_and_per_row().unwrap();
        assert_eq!(base, 0.0);
        assert!((per_row - 20.0).abs() < 1e-9, "{per_row}");
    }

    /// A machine whose steps cost 30 ms, 1.1 ms per row and 4e-6 ms per cell
    /// read (a row at 100 000 tokens costs 0.4 ms of attention).
    const TRUE: StepCost = StepCost { base_ms: 30.0, per_row_ms: 1.1, per_cell_ms: 4e-6 };

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// A step like the conductor composes them: a few generating sequences at
    /// their depths and sometimes a chunk of a prompt.
    fn random_step(rng: &mut Rng) -> Load {
        let mut load = Load::default();
        for _ in 0..rng.below(6) {
            load = load.plus(Load::chunk(rng.below(120_000), 1 + rng.below(4) as u32));
        }
        if rng.below(3) == 0 {
            load = load.plus(Load::chunk(rng.below(200_000), 1 + rng.below(256) as u32));
        }
        if load.rows == 0 { Load::chunk(rng.below(1000), 1) } else { load }
    }

    #[test]
    fn the_fit_recovers_what_rows_and_context_cost() {
        let mut rng = Rng(0x5eed);
        let mut model = StepTimeModel::new(1.0);
        for _ in 0..400 {
            let load = random_step(&mut rng);
            // Two percent of noise, as timers and other load give.
            let noise = 1.0 + (rng.below(401) as f64 - 200.0) / 10_000.0;
            model.observe_load(load, TRUE.ms(load) * noise);
        }
        let cost = model.cost().unwrap();
        assert!((cost.base_ms - 30.0).abs() < 1.5, "{cost:?}");
        assert!((cost.per_row_ms - 1.1).abs() < 0.1, "{cost:?}");
        assert!((cost.per_cell_ms - 4e-6).abs() < 2e-7, "{cost:?}");

        // The same 64 rows of a prompt, shallow and deep. A model without the
        // context term gives both the same time and is wrong for at least one
        // of them by a fifth.
        let shallow = Load::chunk(1_000, 64);
        let deep = Load::chunk(200_000, 64);
        for load in [shallow, deep] {
            let (predicted, real) = (cost.ms(load), TRUE.ms(load));
            assert!((predicted - real).abs() / real < 0.03, "{load:?}: {predicted} vs {real}");
        }
        let line = model.predict(64).unwrap();
        let worst = [shallow, deep]
            .iter()
            .map(|l| (line - TRUE.ms(*l)).abs() / TRUE.ms(*l))
            .fold(0.0, f64::max);
        assert!(worst > 0.2, "the line over rows alone was off by only {worst}");
    }

    #[test]
    fn a_chunk_of_a_prompt_costs_a_quadratic_in_its_length() {
        // SGLang's form: reading l tokens from nothing costs a l^2 + b l + c,
        // so a chunk of x rows after L costs a x^2 + (2 a L + b) x. Here a is
        // per_cell / 2 and b is per_row - per_cell / 2.
        let (a, b) = (TRUE.per_cell_ms / 2.0, TRUE.per_row_ms - TRUE.per_cell_ms / 2.0);
        for (context, rows) in [(0u64, 1u32), (0, 4096), (50_000, 512), (250_000, 2048)] {
            let (l, x) = (context as f64, f64::from(rows));
            let quadratic = a * x * x + (2.0 * a * l + b) * x;
            let chunk = TRUE.ms(Load::chunk(context, rows)) - TRUE.base_ms;
            assert!(
                (chunk - quadratic).abs() < 1e-9 * quadratic.max(1.0),
                "{chunk} vs {quadratic}"
            );
        }
        // Two chunks in a row cost what one chunk of both costs.
        let whole = Load::chunk(1000, 300);
        assert_eq!(Load::chunk(1000, 100).plus(Load::chunk(1100, 200)), whole);
    }

    #[test]
    fn rows_within_is_the_inverse_of_the_cost_of_a_chunk() {
        for context in [0u64, 3_000, 100_000, 260_000] {
            for extra in [0.5, 7.0, 36.4, 900.0] {
                let rows = TRUE.rows_within(extra, context);
                let cost_of = |rows: u32| TRUE.ms(Load::chunk(context, rows)) - TRUE.base_ms;
                assert!(cost_of(rows) <= extra * (1.0 + 1e-9), "{rows} rows at {context}");
                assert!(cost_of(rows + 1) > extra, "{rows} rows at {context} leave room");
            }
        }
        assert_eq!(TRUE.rows_within(0.0, 5), 0);
        assert_eq!(TRUE.rows_within(f64::NAN, 5), 0);
        // Without a context term the answer is the old division.
        let line = StepCost { base_ms: 33.0, per_row_ms: 1.14, per_cell_ms: 0.0 };
        assert_eq!(line.rows_within(36.42, 123_456), 31);
        let free = StepCost { base_ms: 33.0, per_row_ms: 0.0, per_cell_ms: 0.0 };
        assert_eq!(free.rows_within(1.0, 0), u32::MAX);
    }

    #[test]
    fn without_contexts_the_cost_is_the_line() {
        let mut model = StepTimeModel::new(1.0);
        assert_eq!(model.cost(), None);
        for (rows, ms) in MEASURED {
            model.observe(rows, ms);
        }
        let (base, per_row) = model.base_and_per_row().unwrap();
        assert_eq!(
            model.cost(),
            Some(StepCost { base_ms: base, per_row_ms: per_row, per_cell_ms: 0.0 })
        );
    }

    #[test]
    fn steps_that_cannot_tell_rows_from_context_give_the_simpler_model() {
        // One sequence at one depth, windows of 1 to 8 rows: cells are the
        // rows times the depth (nearly), so the two terms cannot be told
        // apart. The line is right for such steps and nothing is negative.
        let mut model = StepTimeModel::new(1.0);
        for i in 0..64u32 {
            let load = Load::chunk(40_000, 1 + i % 8);
            model.observe_load(load, TRUE.ms(load));
        }
        let cost = model.cost().unwrap();
        assert_eq!(cost.per_cell_ms, 0.0, "{cost:?}");
        let seen = Load::chunk(40_000, 5);
        assert!((cost.ms(seen) - TRUE.ms(seen)).abs() < 0.05, "{cost:?}");

        // One row per step, at growing depth: rows never vary, so the base
        // cannot be told from the row. Attention is found; the rest goes on
        // the row, as in the line.
        let mut model = StepTimeModel::new(1.0);
        for i in 0..64u64 {
            let load = Load::chunk(i * 2_000, 1);
            model.observe_load(load, TRUE.ms(load));
        }
        let cost = model.cost().unwrap();
        assert_eq!(cost.base_ms, 0.0);
        assert!((cost.per_cell_ms - 4e-6).abs() < 1e-9, "{cost:?}");
        assert!((cost.per_row_ms - 31.1).abs() < 1e-6, "{cost:?}");

        // Noise that would fit a negative context term gives none.
        let mut model = StepTimeModel::new(1.0);
        let mut rng = Rng(7);
        for _ in 0..200 {
            let load = random_step(&mut rng);
            let ms = 30.0 + 1.1 * f64::from(load.rows) - 1e-7 * load.cells as f64;
            model.observe_load(load, ms.max(0.0));
        }
        let cost = model.cost().unwrap();
        assert!(cost.per_cell_ms >= 0.0 && cost.per_row_ms >= 0.0 && cost.base_ms >= 0.0);
    }

    #[test]
    fn the_context_term_follows_a_change() {
        let mut rng = Rng(99);
        let mut model = StepTimeModel::new(0.97);
        for _ in 0..600 {
            let load = random_step(&mut rng);
            model.observe_load(load, TRUE.ms(load));
        }
        assert!((model.cost().unwrap().per_cell_ms - 4e-6).abs() < 1e-8);
        // Another kernel takes over (pages moved to RAM, say): attention costs
        // three times as much per cell.
        let slow = StepCost { per_cell_ms: 12e-6, ..TRUE };
        for _ in 0..600 {
            let load = random_step(&mut rng);
            model.observe_load(load, slow.ms(load));
        }
        let cost = model.cost().unwrap();
        assert!((cost.per_cell_ms - 12e-6).abs() < 2e-7, "{cost:?}");
        assert!((cost.row_ms(100_000) - 2.3).abs() < 0.05, "{cost:?}");
    }
}
