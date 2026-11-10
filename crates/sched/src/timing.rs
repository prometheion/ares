//! What a step costs.

/// A running estimate of step time as `base + per_row * rows` milliseconds.
///
/// Decode is bound by reading the weights once per step, whatever the number
/// of rows, and every row adds compute on top; measured step times are close
/// to a line over the range that matters. The fit forgets old steps
/// gradually, because the line moves with context depth and with what else
/// the machine is doing.
#[derive(Clone, Debug)]
pub struct StepTimeModel {
    decay: f64,
    weight: f64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_xy: f64,
}

impl Default for StepTimeModel {
    fn default() -> Self {
        Self::new(0.98)
    }
}

impl StepTimeModel {
    /// A model that weighs each step `decay` times the step after it
    /// (0 < `decay` <= 1; 0.98 remembers about the last 50 steps).
    pub fn new(decay: f64) -> Self {
        assert!(decay > 0.0 && decay <= 1.0, "decay must be in (0, 1]");
        Self { decay, weight: 0.0, sum_x: 0.0, sum_y: 0.0, sum_xx: 0.0, sum_xy: 0.0 }
    }

    /// Records that a step of `rows` rows took `ms` milliseconds.
    pub fn observe(&mut self, rows: u32, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        let x = f64::from(rows);
        for sum in
            [&mut self.weight, &mut self.sum_x, &mut self.sum_y, &mut self.sum_xx, &mut self.sum_xy]
        {
            *sum *= self.decay;
        }
        self.weight += 1.0;
        self.sum_x += x;
        self.sum_y += ms;
        self.sum_xx += x * x;
        self.sum_xy += x * ms;
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
        if var_x < 1e-6 * mean_x.max(1.0).powi(2) {
            return Some((0.0, mean_y / mean_x.max(1.0)));
        }
        let cov = self.sum_xy / self.weight - mean_x * mean_y;
        let per_row = (cov / var_x).max(0.0);
        let base = (mean_y - per_row * mean_x).max(0.0);
        Some((base, per_row))
    }

    /// The expected time of a step of `rows` rows in milliseconds.
    pub fn predict(&self, rows: u32) -> Option<f64> {
        let (base, per_row) = self.base_and_per_row()?;
        Some(base + per_row * f64::from(rows))
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
}
