//! How prompts and generation take turns on the devices.
//!
//! A prompt is read in steps of its own, between rounds of generation, and
//! never as rows of a round. A round that verifies drafts keeps the inputs of
//! its rows for the commit that follows, so it has room for the rows of the
//! windows and no more; and a prompt's rows are cheapest where many of them
//! share one pass over the weights, which a round next to a draft window
//! cannot give them. So the two alternate, and what is chosen is how long each
//! one runs (after TensorFold's lanes, which came to the same: the long
//! prompt's step "owes" the generating sequences the time they would have had).
//!
//! The one setting is the *slowdown* `s`: while prompts and generation both
//! wait, the generating sequences get `1 / s` of the time (2: half). A step
//! for prompts that takes `t` milliseconds therefore leaves `t / (s - 1)`
//! milliseconds of rounds owed before the next step for prompts, and a round
//! pays back what it takes. The generating sequences do not stand still while
//! a prompt is read: each rides along in the prompt's step with the row of its
//! newest token (no drafts), so they get a token per step for the rows they
//! cost.
//!
//! Two rules keep the first token of a short prompt close to what the devices
//! need for it:
//!
//! - A step for prompts takes at most one *piece* of rows, where the model's
//!   prompt kernels reach their rate (a smaller piece pays the pass over the
//!   weights again and reads the prompt slower; a larger one stops the
//!   generating sequences for longer). A prompt with no more rows left than
//!   a piece runs whole.
//! - A prompt that a piece completes does not wait for the debt of a long
//!   prompt to be paid, as long as the debt is no more than one whole piece
//!   leaves. Its own debt is added, so over time the share holds, and a burst
//!   of short prompts is read in one step (the planner serves the shortest
//!   first) and then pays once.
//!
//! Priorities (see [`Admission`](crate::Admission)): a prompt that has
//! started and outranks every sequence that generates owes them nothing and
//! is read in pieces at once, the generating sequences riding along; the
//! planner gives it the rows of the piece first. Prompts that do not outrank
//! them, of any priority, take turns with them as above.
//!
//! A prompt that is read alone (nobody generates, no other prompt waits)
//! takes the widest step there is, which may borrow memory for its time (the
//! expert slots of a mixture of experts lend it VRAM). Once another request
//! waits, the steps of prompt shrink to a piece, which the caller bounds in
//! time ([`Pending::piece_rows`]): the request that waits gets its step after
//! the one in flight and one piece at most.

/// What the next step is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    /// Generation: the sequences that generate verify their drafts. No prompt
    /// rows.
    Round,
    /// Prompts: up to `rows` rows of started prompts in a step of their own,
    /// with the newest token of every generating sequence riding along (the
    /// step has `rows` rows of prompt and a row for each of those).
    Piece { rows: u32 },
}

/// What waits for the next step.
#[derive(Clone, Copy, Debug)]
pub struct Pending<'a> {
    /// The sequences that generate.
    pub generating: u32,
    /// The rows still to read of each prompt that has started.
    pub prompts: &'a [u32],
    /// The rows of a piece of prompt, the newest tokens of the generating
    /// sequences included: what a step for prompts takes while another
    /// request waits.
    pub piece_rows: u32,
    /// The most rows of a step: what a prompt read alone takes.
    pub max_rows: u32,
    /// A prompt that has started has a better priority than every sequence
    /// that generates: it is not held back by what is owed to them.
    pub ahead: bool,
}

/// The debt of the prompts to the generating sequences.
#[derive(Clone, Debug)]
pub struct Share {
    /// Milliseconds of rounds a step of prompts leaves owed, per millisecond
    /// it takes: `1 / (s - 1)`.
    per_ms: f64,
    owed_ms: f64,
}

impl Share {
    /// For a slowdown of `slowdown` (above 1: 2 gives generation half of the
    /// time while prompts wait).
    pub fn new(slowdown: f64) -> Self {
        assert!(slowdown > 1.0 && slowdown.is_finite(), "a slowdown above 1, got {slowdown}");
        Self { per_ms: 1.0 / (slowdown - 1.0), owed_ms: 0.0 }
    }

    /// Milliseconds of rounds that are owed now.
    pub fn owed_ms(&self) -> f64 {
        self.owed_ms
    }

    /// What the next step is. `price` is the time of a step of that many rows
    /// in milliseconds.
    pub fn next(&self, pending: &Pending<'_>, price: impl Fn(u32) -> f64) -> Turn {
        let piece = pending.piece_rows.clamp(1, pending.max_rows.max(1));
        // The rows of prompt that fit with the rows of the sequences that
        // ride along: a step of one row more than a piece runs its last row
        // as a step of its own.
        let rows = piece.saturating_sub(pending.generating).max(1);
        let started = pending.prompts.iter().filter(|&&left| left > 0).count();
        if started == 0 {
            return Turn::Round;
        }
        if pending.generating == 0 {
            // Alone, the widest step; next to another prompt, a piece, which
            // the planner gives the shortest prompt first.
            return Turn::Piece { rows: if started == 1 { pending.max_rows } else { piece } };
        }
        if self.owed_ms <= 0.0 || pending.ahead {
            return Turn::Piece { rows };
        }
        let completes = pending.prompts.iter().any(|&left| left > 0 && left <= rows);
        // What one whole piece leaves owed: a short prompt may go ahead of
        // that, and of no more.
        if completes && self.owed_ms <= self.per_ms * price(piece) {
            return Turn::Piece { rows };
        }
        Turn::Round
    }

    /// Books a step that took `ms` milliseconds. `generating` is the number of
    /// sequences that generated while it ran and `waiting` whether a prompt
    /// was waiting for its turn: nobody to pay means no debt, and rounds that
    /// nobody waited for are no credit.
    pub fn spent(&mut self, turn: Turn, ms: f64, generating: u32, waiting: bool) {
        if generating == 0 || !waiting {
            self.owed_ms = 0.0;
            return;
        }
        self.owed_ms = match turn {
            Turn::Piece { .. } => self.owed_ms + ms * self.per_ms,
            // A round that overshoots leaves a credit of up to its own time
            // (rounds are only run while something is owed), which the next
            // debt starts from: rounds come in lumps, and the share holds over
            // time whatever the size of the lump.
            Turn::Round => self.owed_ms - ms,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step of a model whose weights take 33 ms to read and every row 1.1.
    fn price(rows: u32) -> f64 {
        33.0 + 1.1 * f64::from(rows)
    }

    fn pending(generating: u32, prompts: &[u32]) -> Pending<'_> {
        Pending { generating, prompts, piece_rows: 256, max_rows: 512, ahead: false }
    }

    /// Plays decoders and prompts against each other until the prompts are
    /// read, a round taking `round_ms`. Returns the milliseconds the prompts
    /// took, the rounds and the pieces.
    fn play(share: &mut Share, prompts: &mut [u32], round_ms: f64) -> (f64, u32, u32) {
        let (mut time, mut rounds, mut pieces) = (0.0, 0, 0);
        while prompts.iter().any(|&p| p > 0) {
            match share.next(&pending(1, prompts), price) {
                Turn::Round => {
                    share.spent(Turn::Round, round_ms, 1, true);
                    time += round_ms;
                    rounds += 1;
                }
                turn @ Turn::Piece { rows } => {
                    let mut left = rows;
                    for p in prompts.iter_mut() {
                        let take = (*p).min(left);
                        *p -= take;
                        left -= take;
                    }
                    let ms = price(rows - left);
                    share.spent(turn, ms, 1, true);
                    time += ms;
                    pieces += 1;
                }
            }
        }
        (time, rounds, pieces)
    }

    #[test]
    fn nobody_generating_gives_a_prompt_the_widest_step() {
        let share = Share::new(2.0);
        assert_eq!(share.next(&pending(0, &[2000]), price), Turn::Piece { rows: 512 });
        assert_eq!(share.next(&pending(0, &[2000, 0]), price), Turn::Piece { rows: 512 });
        assert_eq!(share.next(&pending(0, &[]), price), Turn::Round);
        assert_eq!(share.next(&pending(3, &[0]), price), Turn::Round);
    }

    #[test]
    fn a_prompt_that_waits_next_to_a_long_one_makes_the_steps_pieces() {
        let share = Share::new(2.0);
        assert_eq!(share.next(&pending(0, &[20_000, 40]), price), Turn::Piece { rows: 256 });
    }

    #[test]
    fn a_prompt_goes_first_when_nothing_is_owed_and_a_long_one_in_pieces() {
        let share = Share::new(2.0);
        // The two sequences that generate ride along: 254 rows of prompt.
        assert_eq!(share.next(&pending(2, &[40]), price), Turn::Piece { rows: 254 });
        assert_eq!(share.next(&pending(2, &[5000]), price), Turn::Piece { rows: 254 });
    }

    #[test]
    fn a_piece_leaves_the_time_it_took_owed_at_half_speed_and_a_round_pays_it_back() {
        let mut share = Share::new(2.0);
        let piece = Turn::Piece { rows: 254 };
        share.spent(piece, 300.0, 2, true);
        assert_eq!(share.owed_ms(), 300.0);
        // The next piece waits for the rounds: 300 ms of them.
        let long = pending(2, &[5000]);
        assert_eq!(share.next(&long, price), Turn::Round);
        share.spent(Turn::Round, 100.0, 2, true);
        assert_eq!(share.next(&long, price), Turn::Round);
        share.spent(Turn::Round, 150.0, 2, true);
        share.spent(Turn::Round, 60.0, 2, true);
        assert_eq!(share.owed_ms(), -10.0, "the round that overshot is a credit");
        assert_eq!(share.next(&long, price), piece);

        // A slowdown of 3 owes half as much, one of 1.5 twice as much.
        let (mut third, mut half) = (Share::new(3.0), Share::new(1.5));
        third.spent(piece, 300.0, 1, true);
        half.spent(piece, 300.0, 1, true);
        assert_eq!((third.owed_ms(), half.owed_ms()), (150.0, 600.0));
    }

    #[test]
    fn a_short_prompt_goes_ahead_of_the_debt_of_one_piece_and_no_more() {
        let mut share = Share::new(2.0);
        let piece = Turn::Piece { rows: 254 };
        // One whole piece: the debt is its price (33 + 1.1 * 256 ms).
        share.spent(piece, price(256), 2, true);
        let both = pending(2, &[5000, 60]);
        assert_eq!(share.next(&both, price), piece, "the short prompt jumps the line");
        // Its own piece adds to the debt: the next short prompt waits.
        share.spent(piece, price(60), 2, true);
        assert_eq!(share.next(&pending(2, &[5000, 60]), price), Turn::Round);
        // And a long prompt alone always waits for the debt.
        let mut share = Share::new(2.0);
        share.spent(piece, price(256), 2, true);
        assert_eq!(share.next(&pending(2, &[5000]), price), Turn::Round);
    }

    #[test]
    fn a_prompt_that_outranks_every_generating_sequence_does_not_wait_for_the_debt() {
        let mut share = Share::new(2.0);
        let piece = Turn::Piece { rows: 254 };
        share.spent(piece, 900.0, 2, true);
        let long = pending(2, &[5000]);
        assert_eq!(share.next(&long, price), Turn::Round);
        let ahead = Pending { ahead: true, ..long };
        assert_eq!(share.next(&ahead, price), piece);
        // Its pieces are owed all the same, to the prompts that come after it.
        share.spent(piece, 300.0, 2, true);
        assert_eq!(share.owed_ms(), 1200.0);
    }

    #[test]
    fn rounds_that_no_prompt_waited_for_are_no_credit() {
        let mut share = Share::new(2.0);
        for _ in 0..100 {
            share.spent(Turn::Round, 35.0, 2, false);
        }
        assert_eq!(share.owed_ms(), 0.0);
        // A prompt that comes then is not read ahead of the debt it makes.
        share.spent(Turn::Piece { rows: 254 }, 300.0, 2, true);
        assert_eq!(share.next(&pending(2, &[5000]), price), Turn::Round);
    }

    #[test]
    fn nobody_generating_means_nobody_is_owed() {
        let mut share = Share::new(2.0);
        share.spent(Turn::Piece { rows: 254 }, 300.0, 2, true);
        share.spent(Turn::Piece { rows: 512 }, 600.0, 0, true);
        assert_eq!(share.owed_ms(), 0.0);
    }

    #[test]
    fn a_long_prompt_next_to_generation_takes_the_time_the_slowdown_says() {
        for (slowdown, expected_share) in [(2.0, 0.5), (3.0, 2.0 / 3.0), (1.25, 0.2)] {
            let mut share = Share::new(slowdown);
            let mut prompts = [8192u32];
            let (time, rounds, pieces) = play(&mut share, &mut prompts, 35.0);
            let prompt_time = time - f64::from(rounds) * 35.0;
            let got = prompt_time / time;
            // Pieces of 255 rows (one is the newest token of the sequence that
            // generates), 33 of them; the last round or two may be missing at
            // the end.
            assert_eq!(pieces, 33);
            assert!(
                (got - expected_share).abs() < 0.03,
                "slowdown {slowdown}: the prompt took {got:.3} of the time, not {expected_share}"
            );
        }
    }

    #[test]
    fn a_short_prompt_next_to_a_long_one_is_read_within_one_piece_of_waiting() {
        // A 6000-row prompt is being read; a 50-row prompt arrives after
        // three pieces, with the generating sequence owed one piece's time.
        let mut share = Share::new(2.0);
        let mut prompts = [6000u32 - 3 * 256, 50];
        share.spent(Turn::Piece { rows: 256 }, price(256), 1, true);
        let mut waited = 0.0;
        loop {
            match share.next(&pending(1, &prompts), price) {
                Turn::Round => {
                    share.spent(Turn::Round, 35.0, 1, true);
                    waited += 35.0;
                }
                Turn::Piece { rows } => {
                    // The shortest prompt goes first within a piece.
                    let take = prompts[1].min(rows);
                    prompts[1] -= take;
                    waited += price(take);
                    break;
                }
            }
        }
        assert_eq!(prompts[1], 0, "the short prompt was read whole");
        assert!(waited < 100.0, "it waited {waited} ms");
    }
}
