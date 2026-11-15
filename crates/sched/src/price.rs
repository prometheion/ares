//! What a step costs, as a table.
//!
//! The weight kernels of a device take their rows in launches of a fixed
//! width, and the price of a step is flat inside a launch and steps up at its
//! edge: the 27B on an RTX 4070 and an RTX 3060 reads its weights in 25.4 ms
//! for 8 rows and in 31.0 for 16, and a ninth row is a second launch for a
//! share of the matrices. A line `base + per_row * rows` cannot say that, and
//! the choice of how many draft rows to verify turns on exactly that edge. So
//! the price of a step is a table by number of rows ([`PriceTable`]), plus a
//! term for the context the rows read.
//!
//! The table is kept true by the steps that run, and it is built so that no
//! size can be priced out for good ([`Curve`]):
//!
//! - **A size's price is the second smallest of its last five measurements**
//!   (the smallest of up to three). Other work on the machine only ever adds
//!   to a step, so one late step, or three in five, moves nothing; a size
//!   that really got slower is followed after four steps, one that got faster
//!   after two.
//! - **A size that has not run for [`OLD`] measurements of the table is old.**
//!   What it cost then says little about now (tiers that warmed up, a machine
//!   that was busy), so its price is held between the prices of the nearest
//!   sizes on either side that are current (fewer rows do not cost more, more
//!   rows do not cost less), and whoever chooses a window is told the price
//!   to *try* it at ([`Curve::trial_ms`]): that of the nearest smaller size
//!   that is current. An old size is therefore chosen again as soon as its
//!   rows could pay at all, the step that runs measures it, and that
//!   measurement stands beside the old price, the smaller of the two being
//!   the price. It is tried once more, so that one late step among the two
//!   does not waste the turn.
//!
//! Together: a price that is too high because of late steps needs four of
//! them in five; a price that is too high for any reason (a first measurement
//! on cold tiers, a busy stretch) is tried again after [`OLD`] measurements
//! and corrected by one step; and trying costs at most two steps per size and
//! [`OLD`] steps, which also bring the tokens their rows verify.
//!
//! Between two measured row counts the price is interpolated, below the
//! smallest it is the smallest, above the largest it goes on with the slope of
//! the last two. A row count that has never been measured is therefore
//! priced from its neighbours, and the first step of that size corrects it.
//! The measurements at load ([`PriceTable::measured`]) are the first entries:
//! they give every size a price before a request runs, taken together under
//! the same conditions, and are corrected like any other.

use std::collections::BTreeMap;

use crate::timing::{Load, StepTimeModel};

/// How many measurements of a count are kept.
const KEPT: usize = 5;

/// After this many measurements of a curve without one of its own, a count's
/// price is old: held between its neighbours', and tried again by whoever
/// chooses.
pub const OLD: u64 = 256;

/// What was measured of one count.
#[derive(Clone, Debug)]
struct Point {
    /// The last measurements, oldest first.
    samples: [f64; KEPT],
    len: usize,
    /// The measurements of the curve before the newest of this count.
    at: u64,
    /// The count was old and has been measured once since: it is tried once
    /// more.
    again: bool,
}

impl Point {
    fn of(ms: f64, at: u64) -> Self {
        let mut samples = [0.0; KEPT];
        samples[0] = ms;
        Self { samples, len: 1, at, again: false }
    }

    fn push(&mut self, ms: f64) {
        if self.len == KEPT {
            self.samples.copy_within(1.., 0);
            self.len -= 1;
        }
        self.samples[self.len] = ms;
        self.len += 1;
    }

    /// The second smallest of the measurements kept, the smallest of up to
    /// three.
    fn price(&self) -> f64 {
        let mut sorted = self.samples;
        let sorted = &mut sorted[..self.len];
        sorted.sort_by(f64::total_cmp);
        sorted[usize::from(self.len > 3)]
    }
}

/// A price by a count (rows of a step, links of a chain), measured at some
/// counts and interpolated between them, that follows the measurements and
/// gives up on none of its counts (see the module).
#[derive(Clone, Debug, Default)]
pub struct Curve {
    points: BTreeMap<u32, Point>,
    /// The measurements taken.
    seen: u64,
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
        self.points.iter().map(|(n, point)| (*n, self.price_of(*n, point)))
    }

    fn is_old(&self, point: &Point) -> bool {
        self.seen - point.at > OLD
    }

    /// Whether `n` is tried again at [`Curve::trial_ms`]: its price is old,
    /// or it was never measured, or it was old and has been measured once.
    pub fn is_due(&self, n: u32) -> bool {
        self.points.get(&n).is_none_or(|point| point.again || self.is_old(point))
    }

    /// The price of the nearest count below `n` that is current.
    fn current_below(&self, n: u32) -> Option<f64> {
        let mut below = self.points.range(..n).rev();
        below.find(|(_, point)| !self.is_old(point)).map(|(_, point)| point.price())
    }

    /// The price of a measured count: its own while it is current, and held
    /// between the nearest current counts on either side when it is old.
    fn price_of(&self, n: u32, point: &Point) -> f64 {
        let price = point.price();
        if !self.is_old(point) {
            return price;
        }
        let mut above = self.points.range(n + 1..);
        let above = above.find(|(_, point)| !self.is_old(point)).map(|(_, point)| point.price());
        let floor = self.current_below(n).unwrap_or(0.0);
        price.min(above.unwrap_or(f64::INFINITY)).max(floor)
    }

    /// Sets the price of `n` to `ms`, whatever it was: a measurement that is
    /// known to be clean (taken at load, with nothing else running).
    pub fn set(&mut self, n: u32, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            self.points.insert(n, Point::of(ms, self.seen));
        }
    }

    /// A measurement of `n`.
    pub fn observe(&mut self, n: u32, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        let seen = self.seen;
        self.seen += 1;
        let old = self.points.get(&n).filter(|point| self.is_old(point));
        let old = old.map(|point| self.price_of(n, point));
        match (self.points.get_mut(&n), old) {
            (None, _) => {
                self.points.insert(n, Point::of(ms, seen));
            }
            // Its old price stands as one measurement beside the new one, and
            // the smaller of the two is the price: a late step cannot raise
            // it, and a machine that got faster lowers it at once.
            (Some(point), Some(old)) => {
                *point = Point { again: true, ..Point::of(old, seen) };
                point.push(ms);
            }
            (Some(point), None) => {
                point.push(ms);
                point.at = seen;
                point.again = false;
            }
        }
    }

    /// The price of `n`, or `None` before anything was measured.
    pub fn ms(&self, n: u32) -> Option<f64> {
        let price = |(&a, point): (&u32, &Point)| (a, self.price_of(a, point));
        let below = self.points.range(..=n).next_back().map(price);
        let above = self.points.range(n..).next().map(price);
        match (below, above) {
            (None, None) => None,
            // Below the smallest count.
            (None, Some((_, p))) => Some(p),
            (Some((a, pa)), Some((b, pb))) => {
                Some(if a == b { pa } else { pa + (pb - pa) * f64::from(n - a) / f64::from(b - a) })
            }
            // Above the largest: the slope of the last two.
            (Some((b, pb)), None) => {
                let before = self.points.range(..b).next_back().map(price);
                let slope = before.map_or(0.0, |(a, pa)| ((pb - pa) / f64::from(b - a)).max(0.0));
                Some(pb + slope * f64::from(n - b))
            }
        }
    }

    /// The price to try `n` at: its price, and for a count that is due
    /// ([`Curve::is_due`]) no more than the price of the nearest smaller
    /// count that is current. Whoever chooses by this price takes such a
    /// count whenever it could pay at all, and the step that runs measures
    /// it.
    pub fn trial_ms(&self, n: u32) -> Option<f64> {
        let price = self.ms(n)?;
        if !self.is_due(n) {
            return Some(price);
        }
        Some(self.current_below(n).map_or(price, |below| price.min(below)))
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

    /// The price of a step of `rows` rows measured at load, with next to no
    /// context and nothing else running. Replaces what the table had.
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

    /// The price to try a step that carries `load` at: [`PriceTable::ms`],
    /// with the rows at [`Curve::trial_ms`]. For the choice of a window.
    pub fn trial_ms(&self, load: Load) -> Option<f64> {
        Some(self.rows.trial_ms(load.rows)? + self.per_cell_ms() * load.cells as f64)
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
    fn late_steps_do_not_move_a_price_and_a_real_change_is_followed() {
        let mut curve = Curve::new();
        curve.set(4, 25.0);
        for _ in 0..8 {
            curve.observe(4, 25.0);
        }
        // One step in which the machine did something else, a hundred times
        // too long: nothing moves. Nor with a second and a third in five.
        curve.observe(4, 2500.0);
        assert_eq!(curve.ms(4), Some(25.0));
        curve.observe(4, 25.0);
        curve.observe(4, 900.0);
        curve.observe(4, 700.0);
        assert_eq!(curve.ms(4), Some(25.0));
        for _ in 0..5 {
            curve.observe(4, 25.0);
        }
        // Steps that take 30 ms from now on are the price after four of them.
        for _ in 0..3 {
            curve.observe(4, 30.0);
        }
        assert_eq!(curve.ms(4), Some(25.0), "three do not say yet");
        curve.observe(4, 30.0);
        assert_eq!(curve.ms(4), Some(30.0));
        // And steps that got faster after two.
        curve.observe(4, 20.0);
        assert_eq!(curve.ms(4), Some(30.0));
        curve.observe(4, 20.0);
        assert_eq!(curve.ms(4), Some(20.0));
    }

    #[test]
    fn a_count_that_does_not_run_is_tried_again_at_the_price_of_one_that_does() {
        // Measured at load on cold tiers: 35.5 ms for a row, 49.4 for two,
        // 93 for three.
        let mut curve = Curve::new();
        for (rows, ms) in [(1, 35.5), (2, 49.4), (3, 93.0)] {
            curve.set(rows, ms);
        }
        // Only steps of one row run, and warm they take 22 ms.
        for _ in 0..OLD {
            curve.observe(1, 22.0);
        }
        // The other counts are current for a while: they are tried at their
        // prices.
        assert_eq!(curve.ms(1), Some(22.0));
        assert_eq!((curve.ms(2), curve.trial_ms(2)), (Some(49.4), Some(49.4)));
        assert!(!curve.is_due(2));
        // One more and they are old: tried at the price of the row that runs.
        curve.observe(1, 22.0);
        assert!(curve.is_due(2) && curve.is_due(3) && !curve.is_due(1));
        assert_eq!((curve.ms(2), curve.ms(3)), (Some(49.4), Some(93.0)));
        assert_eq!((curve.trial_ms(2), curve.trial_ms(3)), (Some(22.0), Some(22.0)));
        // The step that tries two rows measures 27 ms: that is its price, and
        // three rows are tried at it.
        curve.observe(2, 27.0);
        assert_eq!((curve.ms(2), curve.trial_ms(2)), (Some(27.0), Some(22.0)));
        curve.observe(2, 27.5);
        assert_eq!((curve.ms(2), curve.trial_ms(2)), (Some(27.0), Some(27.0)));
        assert_eq!(curve.trial_ms(3), Some(27.0));
        // A trial that happens to be late does not raise an old price, and
        // the count is tried once more.
        curve.observe(3, 4000.0);
        assert_eq!((curve.ms(3), curve.trial_ms(3)), (Some(93.0), Some(27.0)));
        curve.observe(3, 31.0);
        assert_eq!((curve.ms(3), curve.trial_ms(3)), (Some(31.0), Some(31.0)));
    }

    #[test]
    fn an_old_price_is_held_between_its_current_neighbours() {
        let mut curve = Curve::new();
        for (rows, ms) in [(1, 20.0), (2, 900.0), (4, 5.0), (8, 30.0)] {
            curve.set(rows, ms);
        }
        for _ in 0..=OLD / 2 {
            curve.observe(1, 20.0);
            curve.observe(8, 30.0);
        }
        // Two rows do not cost more than eight, four not less than one.
        assert_eq!((curve.ms(2), curve.ms(4)), (Some(30.0), Some(20.0)));
        assert_eq!(curve.ms(3), Some(25.0), "between them, interpolated");
        assert_eq!(
            curve.points().collect::<Vec<_>>(),
            [(1, 20.0), (2, 30.0), (4, 20.0), (8, 30.0)]
        );
    }

    #[test]
    fn no_count_stays_priced_out() {
        // A count whose price is a hundred times too high for whatever reason
        // (here: four late steps in a row) and is not chosen any more.
        let mut curve = Curve::new();
        for rows in 1..=4 {
            curve.set(rows, 25.0 + f64::from(rows));
        }
        for _ in 0..4 {
            curve.observe(4, 2900.0);
        }
        assert_eq!(curve.ms(4), Some(2900.0));
        // Whoever chooses takes four rows when they cost less than 40 ms, and
        // one row otherwise. The steps cost what they really cost.
        let mut chosen_at = None;
        for step in 0..2 * OLD {
            if curve.trial_ms(4).unwrap() < 40.0 {
                chosen_at.get_or_insert(step);
                curve.observe(4, 29.0);
            } else {
                curve.observe(1, 26.0);
            }
        }
        // It is tried once it is old, and from then on it is chosen at its price.
        assert_eq!(chosen_at, Some(OLD));
        assert_eq!((curve.ms(4), curve.trial_ms(4)), (Some(29.0), Some(29.0)));
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
