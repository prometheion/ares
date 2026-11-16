//! The rows of a step that verify drafts, as the miss path of the store sees
//! them: a row that verifies a draft waits for an expert that has to be read
//! from the file only while it is worth the wait.
//!
//! A window's first row is its sequence's newest token, which the step must
//! give whatever it costs. Every row after it verifies a draft, which the
//! next step can verify again. Row `k` of a window counts only when the drafts
//! before it are right, so what it is worth is that chance times what a token
//! costs, which the step says (the record's row worth). An expert in no tier
//! but the file that only such rows picked costs the layer a read, unless the
//! layer waits for one of the file already; it is read while the rows it would
//! leave behind (the lowest row that picked it in each window, and the rows of
//! that window after it) are worth more than that, and the read is charged to
//! them. Otherwise it is not read for them: it adds nothing to their sums (the
//! CPU claims it and delivers zero), and those rows and every row of their
//! window after them are *left*: their results are not what the rows would
//! give alone, and whoever reads the step accepts no draft from the first left
//! row of a window on ([`super::ExpertStore::rows_left`]). The rows before it
//! stay exact, because a row depends only on the rows before it. A step that
//! says nothing of its rows' worth leaves none.
//!
//! An expert whose rows are all left needs no work at all, wherever it is.

/// The windows of the running step and the rows left without an expert.
#[derive(Clone, Debug, Default)]
pub(super) struct Windows {
    /// Per window, its rows as a mask (a bit per row of the step).
    windows: Vec<u32>,
    /// The rows past the first of every window.
    drafts: u32,
    /// The rows left without an expert, and every row of their window after
    /// them.
    left: u32,
    /// What each row is still worth in milliseconds, the waits it was charged
    /// taken off; `None` when the step said nothing of it.
    worth: Option<[f32; 32]>,
}

/// The mask of `rows` rows from row `first`; rows past the 32nd are none.
fn mask(first: u32, rows: u32) -> u32 {
    let end = (first + rows).min(32);
    let below = |n: u32| if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
    below(end) & !below(first.min(32))
}

impl Windows {
    /// The windows of a new step, as (first row, rows), and what each row of
    /// the step is worth in milliseconds.
    pub(super) fn set(&mut self, windows: &[(u32, u32)], worth: Option<&[f32]>) {
        self.windows.clear();
        self.windows.extend(windows.iter().map(|&(first, rows)| mask(first, rows)));
        self.drafts =
            windows.iter().fold(0, |drafts, &(first, rows)| drafts | mask(first + 1, rows - 1));
        self.left = 0;
        self.worth = worth.map(|worth| {
            std::array::from_fn(|row| worth.get(row).copied().unwrap_or(0.0).max(0.0))
        });
    }

    /// The rows a miss that `rows` picked would leave behind: in each window,
    /// the lowest of them that is not left and every row after it that is not.
    fn stake_rows(&self, rows: u32) -> u32 {
        self.windows.iter().fold(0, |stake, window| {
            let hit = rows & window & !self.left;
            if hit == 0 {
                return stake;
            }
            stake | (window & !self.left & !((1u32 << hit.trailing_zeros()) - 1))
        })
    }

    /// What leaving a miss that `rows` picked would lose, in milliseconds;
    /// without a word of the step's worth, more than any read costs.
    pub(super) fn stake(&self, rows: u32) -> f32 {
        let Some(worth) = &self.worth else { return f32::INFINITY };
        let at = self.stake_rows(rows);
        (0..32).filter(|row| at >> row & 1 != 0).map(|row| worth[row as usize]).sum()
    }

    /// The rows of the miss that `rows` picked pay `cost` milliseconds of
    /// waiting, in proportion to what they are worth.
    pub(super) fn charge(&mut self, rows: u32, cost: f32) {
        let stake = self.stake(rows);
        let at = self.stake_rows(rows);
        let Some(worth) = &mut self.worth else { return };
        if !stake.is_finite() || stake <= 0.0 {
            return;
        }
        let keep = (1.0 - cost / stake).max(0.0);
        for (row, value) in worth.iter_mut().enumerate() {
            if at >> row & 1 != 0 {
                *value *= keep;
            }
        }
    }

    /// Whether a miss that `rows` picked may be left without a read: every
    /// one of them verifies a draft or is left already.
    pub(super) fn may_leave(&self, rows: u32) -> bool {
        rows != 0 && rows & !(self.drafts | self.left) == 0
    }

    /// Whether every one of `rows` is left already.
    pub(super) fn all_left(&self, rows: u32) -> bool {
        rows != 0 && rows & !self.left == 0
    }

    /// Leaves `rows` without an expert: each of them, and every row of its
    /// window after it.
    pub(super) fn leave(&mut self, rows: u32) {
        for window in &self.windows {
            let hit = rows & window;
            if hit != 0 {
                self.left |= window & !((1u32 << hit.trailing_zeros()) - 1);
            }
        }
    }

    /// The rows left so far, a bit per row of the step.
    pub(super) fn left(&self) -> u32 {
        self.left
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draft_row_and_the_rows_after_it_in_its_window_are_left() {
        let mut w = Windows::default();
        // A sequence without drafts at row 0, a window of four rows at 1..5
        // and one of three at 5..8.
        w.set(&[(1, 4), (5, 3)], None);
        // Row 0 and the first rows of the windows are never left.
        assert!(!w.may_leave(0b1) && !w.may_leave(0b10) && !w.may_leave(0b10_0000));
        assert!(!w.may_leave(0b110), "a first row with a draft row");
        assert!(w.may_leave(0b100) && w.may_leave(0b1100_0100));
        w.leave(0b1000);
        assert_eq!(w.left(), 0b1_1000, "row 3 and row 4 of its window");
        // A row left already may take anything from now on.
        assert!(w.all_left(0b1_0000) && !w.all_left(0b1_0100));
        assert!(w.may_leave(0b1_0100));
        w.leave(0b100_0000);
        assert_eq!(w.left(), 0b1101_1000);
        // A new step starts with nothing left.
        w.set(&[(0, 2)], None);
        assert_eq!(w.left(), 0);
        assert!(w.may_leave(0b10) && !w.may_leave(0b100), "no window there");
    }

    #[test]
    fn a_read_is_worth_what_the_rows_it_keeps_are_worth_and_they_pay_for_it() {
        let mut w = Windows::default();
        // Without a word of the rows' worth nothing is worth leaving.
        w.set(&[(0, 4)], None);
        assert!(w.stake(0b100).is_infinite());
        // A window of four rows whose drafts are worth 20, 10 and 5 ms.
        w.set(&[(0, 4)], Some(&[0.0, 20.0, 10.0, 5.0]));
        // Leaving row 2 loses rows 2 and 3; leaving row 1 all three.
        assert_eq!(w.stake(0b100), 15.0);
        assert_eq!(w.stake(0b110), 35.0);
        assert_eq!(w.stake(0b1000), 5.0);
        // A read of 5 ms for rows 2 and 3 is paid by them: a third of their
        // worth is gone, row 1 keeps all of its own.
        w.charge(0b100, 5.0);
        assert!((w.stake(0b100) - 10.0).abs() < 1e-6);
        assert!((w.stake(0b10) - 30.0).abs() < 1e-6);
        // A left row is worth nothing any more.
        w.leave(0b100);
        assert_eq!(w.stake(0b1000), 0.0);
        assert!((w.stake(0b10) - 20.0).abs() < 1e-6);
    }
}
