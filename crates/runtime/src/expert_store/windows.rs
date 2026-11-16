//! The rows of a step that verify drafts, as the miss path of the store sees
//! them: a row that verifies a draft never waits for an expert that has to be
//! read from the file.
//!
//! A window's first row is its sequence's newest token, which the step must
//! give whatever it costs. Every row after it verifies a draft, which the
//! next step can verify again. So an expert that is in no tier but the file,
//! and that only such rows picked, is not read for them: it adds nothing to
//! their sums (the CPU claims it and delivers zero), and those rows and every
//! row of their window after them are *left*: their results are not what the
//! rows would give alone, and whoever reads the step accepts no draft from the
//! first left row of a window on ([`super::ExpertStore::rows_left`]). The rows
//! before it stay exact, because a row depends only on the rows before it.
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
}

/// The mask of `rows` rows from row `first`; rows past the 32nd are none.
fn mask(first: u32, rows: u32) -> u32 {
    let end = (first + rows).min(32);
    let below = |n: u32| if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
    below(end) & !below(first.min(32))
}

impl Windows {
    /// The windows of a new step, as (first row, rows).
    pub(super) fn set(&mut self, windows: &[(u32, u32)]) {
        self.windows.clear();
        self.windows.extend(windows.iter().map(|&(first, rows)| mask(first, rows)));
        self.drafts =
            windows.iter().fold(0, |drafts, &(first, rows)| drafts | mask(first + 1, rows - 1));
        self.left = 0;
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
        w.set(&[(1, 4), (5, 3)]);
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
        w.set(&[(0, 2)]);
        assert_eq!(w.left(), 0);
        assert!(w.may_leave(0b10) && !w.may_leave(0b100), "no window there");
    }
}
