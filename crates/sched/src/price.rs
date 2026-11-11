//! What a step costs, as a table.
//!
//! The weight kernels of a device take their rows in launches of a fixed
//! width, and the price of a step is flat inside a launch and steps up at its
//! edge: the 27B on an RTX 4070 and an RTX 3060 reads its weights in 25.4 ms
//! for 8 rows and in 31.0 for 16, and a ninth row is a second launch for a
//! share of the matrices. A line `base + per_row * rows` cannot say that, and
//! the choice of how many draft rows to verify turns on exactly that edge. So
//! the price of a step is a table by number of rows, measured on the devices
//! when the model is loaded ([`PriceTable::measured`]) and corrected by every
//! step that runs, plus a term for the context the rows read.
//!
//! Between two measured row counts the price is interpolated, below the
//! smallest it is the smallest, above the largest it goes on with the slope of
//! the last two. A row count that has never been measured is therefore
//! priced from its neighbours, and the first step of that size corrects it.

use std::collections::BTreeMap;

use crate::timing::{Load, StepTimeModel};

/// How much of a new measurement replaces what the table holds for its row
/// count. A step takes about as long as the last one did, with the noise of
/// the machine it runs on; a fifth remembers the last ten or so.
const MOVE: f64 = 0.2;

/// A price by a count (rows of a step, links of a chain), measured at some
/// counts and interpolated between them.
#[derive(Clone, Debug, Default)]
pub struct Curve {
    points: BTreeMap<u32, f64>,
}

impl Curve {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether anything was measured.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The measured counts and their prices.
    pub fn points(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.points.iter().map(|(n, ms)| (*n, *ms))
    }

    /// Sets the price of `n` to `ms`, whatever it was.
    pub fn set(&mut self, n: u32, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            self.points.insert(n, ms);
        }
    }

    /// Moves the price of `n` towards `ms`; a count that was never measured
    /// takes it.
    pub fn observe(&mut self, n: u32, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        self.points.entry(n).and_modify(|have| *have += MOVE * (ms - *have)).or_insert(ms);
    }

    /// The price of `n`, or `None` before anything was measured.
    pub fn ms(&self, n: u32) -> Option<f64> {
        let below = self.points.range(..=n).next_back().map(|(&a, &p)| (a, p));
        let above = self.points.range(n..).next().map(|(&a, &p)| (a, p));
        match (below, above) {
            (None, None) => None,
            // Below the smallest count.
            (None, Some((_, p))) => Some(p),
            (Some((a, pa)), Some((b, pb))) => {
                Some(if a == b { pa } else { pa + (pb - pa) * f64::from(n - a) / f64::from(b - a) })
            }
            // Above the largest: the slope of the last two.
            (Some((b, pb)), None) => {
                let slope = self
                    .points
                    .range(..b)
                    .next_back()
                    .map_or(0.0, |(&a, &pa)| ((pb - pa) / f64::from(b - a)).max(0.0));
                Some(pb + slope * f64::from(n - b))
            }
        }
    }
}

/// The price of a step: by its rows, and by the cells its rows read.
#[derive(Clone, Debug)]
pub struct PriceTable {
    rows: Curve,
    /// The context term: the fit of cost per cell over the steps that ran.
    context: StepTimeModel,
}

impl Default for PriceTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PriceTable {
    pub fn new() -> Self {
        Self { rows: Curve::new(), context: StepTimeModel::new(0.98) }
    }

    /// The price of a step of `rows` rows measured with next to no context
    /// (the first step of a model is a measurement of this kind). Replaces
    /// what the table had.
    pub fn measured(&mut self, rows: u32, ms: f64) {
        self.rows.set(rows, ms);
    }

    /// Milliseconds a cell that a row reads adds: what attention costs per
    /// cell, from the steps that ran (0 until they tell).
    pub fn per_cell_ms(&self) -> f64 {
        self.context.cost().map_or(0.0, |cost| cost.per_cell_ms)
    }

    /// Whether the table holds a measurement.
    pub fn is_ready(&self) -> bool {
        !self.rows.is_empty()
    }

    /// The measured steps: rows and milliseconds with next to no context.
    pub fn points(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.rows.points()
    }

    /// A step that carried `load` took `ms` milliseconds.
    pub fn observe(&mut self, load: Load, ms: f64) {
        if !ms.is_finite() || ms < 0.0 || load.rows == 0 {
            return;
        }
        // The rows' own price is what is left when the context is taken out.
        let own = ms - self.per_cell_ms() * load.cells as f64;
        self.rows.observe(load.rows, own.max(0.0));
        // The context term is fitted to the steps as they were: the fit has
        // rows and cells as separate terms, and tells them apart wherever
        // steps differ in both (a prompt's piece against a window deep in its
        // context).
        self.context.observe_load(load, ms);
    }

    /// The price of a step that carries `load`, or `None` before anything was
    /// measured.
    pub fn ms(&self, load: Load) -> Option<f64> {
        Some(self.rows.ms(load.rows)? + self.per_cell_ms() * load.cells as f64)
    }

    /// The price of `rows` rows with no context, or `None` before anything
    /// was measured.
    pub fn rows_ms(&self, rows: u32) -> Option<f64> {
        self.rows.ms(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 27B on the two cards: a pass over the weights per launch of 8 rows.
    fn weights_ms(rows: u32) -> f64 {
        let launches = f64::from(rows.div_ceil(8));
        23.4 + 2.0 * (launches - 1.0) + 0.15 * f64::from(rows)
    }

    #[test]
    fn a_table_keeps_the_step_at_the_edge_of_a_launch_that_a_line_loses() {
        let mut table = PriceTable::new();
        for rows in 1..=24 {
            table.measured(rows, weights_ms(rows));
        }
        let at = |rows: u32| table.rows_ms(rows).unwrap();
        // Flat inside a launch except for the rows, a step at the ninth row.
        assert!((at(9) - at(8) - (2.0 + 0.15)).abs() < 1e-9);
        assert!((at(16) - at(9) - 7.0 * 0.15).abs() < 1e-9);
        // A line through the same points puts the ninth row between the two.
        let (first, last) = (at(1), at(24));
        let line = |rows: u32| first + (last - first) * f64::from(rows - 1) / 23.0;
        assert!((line(9) - at(9)).abs() > 0.5, "the line is off by {}", line(9) - at(9));
    }

    #[test]
    fn counts_between_two_measurements_are_interpolated_and_the_ends_are_extended() {
        let mut curve = Curve::new();
        assert_eq!(curve.ms(5), None);
        curve.set(8, 25.0);
        assert_eq!(curve.ms(1), Some(25.0), "one point is flat");
        assert_eq!(curve.ms(100), Some(25.0));
        curve.set(16, 31.0);
        curve.set(64, 70.0);
        assert_eq!(curve.ms(4), Some(25.0), "below the smallest it is the smallest");
        assert_eq!(curve.ms(12), Some(28.0));
        assert_eq!(curve.ms(16), Some(31.0));
        // Above the largest, the slope of the last two: 39 ms over 48 rows.
        assert!((curve.ms(112).unwrap() - (70.0 + 39.0)).abs() < 1e-9);
    }

    #[test]
    fn running_steps_correct_a_row_count_and_leave_the_others() {
        let mut table = PriceTable::new();
        for rows in [1, 8, 9, 16] {
            table.measured(rows, weights_ms(rows));
        }
        let before = table.rows_ms(8).unwrap();
        // The cards are slower than they were at load: every step of 8 rows
        // takes 4 ms more.
        for _ in 0..40 {
            table.observe(Load::chunk(0, 8), before + 4.0);
        }
        assert!((table.rows_ms(8).unwrap() - (before + 4.0)).abs() < 0.05);
        assert_eq!(table.rows_ms(9), Some(weights_ms(9)));
        // A row count nobody measured is priced from its neighbours until a
        // step of that size runs.
        assert!(table.rows_ms(12).unwrap() > weights_ms(9));
        table.observe(Load::chunk(0, 12), 40.0);
        assert_eq!(table.rows_ms(12), Some(40.0));
    }

    #[test]
    fn the_context_a_row_reads_is_a_term_of_its_own() {
        let mut table = PriceTable::new();
        for rows in [1, 8, 64] {
            table.measured(rows, 30.0 + 1.0 * f64::from(rows));
        }
        let truth = |load: Load| 30.0 + load.rows as f64 + 4e-6 * load.cells as f64;
        // Steps of several sizes at several depths.
        for context in [0u64, 10_000, 50_000, 120_000, 200_000] {
            for rows in [1u32, 8, 16, 64] {
                let load = Load::chunk(context, rows);
                for _ in 0..12 {
                    table.observe(load, truth(load));
                }
            }
        }
        assert!((table.per_cell_ms() - 4e-6).abs() < 4e-7, "{}", table.per_cell_ms());
        let deep = Load::chunk(150_000, 8);
        let got = table.ms(deep).unwrap();
        assert!((got - truth(deep)).abs() / truth(deep) < 0.05, "{got} against {}", truth(deep));
    }

    #[test]
    fn nonsense_is_not_a_measurement() {
        let mut table = PriceTable::new();
        table.observe(Load::chunk(0, 4), f64::NAN);
        table.observe(Load::chunk(0, 4), -3.0);
        table.observe(Load { rows: 0, cells: 0 }, 5.0);
        assert!(!table.is_ready());
        assert_eq!(table.ms(Load::chunk(0, 4)), None);
    }
}
