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
//! The one setting is the *floor*: the rate, in tokens per second, that every
//! sequence that generates keeps while prompts are read. What generation
//! gives up beyond it goes to the prompts. The share measures what rounds give
//! a generating sequence (`r` tokens per millisecond of rounds, the mean of
//! the sequences over the last rounds), and a step for prompts that takes `t`
//! milliseconds leaves owed the rounds that bring the rate of the sequences
//! over both back to the floor `f`:
//!
//! ```text
//! rounds = (f * t - riding) / (r - f)
//! ```
//!
//! where `riding` is the token each sequence got in the step itself (see
//! below). A round pays back what it takes. So the split follows what the
//! devices do: a model, a context or a text whose rounds give more tokens
//! keeps its floor in less time and leaves the prompts more of it. Two bounds
//! hold it: the rounds owed are never fewer than none (a prompt never waits for
//! rounds a sequence does not need) and never more than the step's own time
//! (generation gets at most half of the time while prompts wait, which is
//! what a floor the rounds cannot reach leaves: a slow model, many sequences;
//! and what is owed while no round has been measured yet).
//!
//! The generating sequences do not stand still while a prompt is read: each
//! rides along in the prompt's step with the row of its newest token (no
//! drafts), so they get a token per step for the rows they cost.
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

/// How many rounds of the last ones the measured rate of rounds stands for,
/// about: each round weighs this much less than the one after it.
const RATE_KEEPS: f64 = 0.95;

/// The debt of the prompts to the generating sequences.
#[derive(Clone, Debug)]
pub struct Share {
    /// Tokens per millisecond that every generating sequence keeps while
    /// prompts are read.
    floor: f64,
    /// What the last rounds gave a generating sequence, tokens and
    /// milliseconds, each round weighing [`RATE_KEEPS`] of the one after it.
    tokens: f64,
    ms: f64,
    owed_ms: f64,
}

impl Share {
    /// For a floor of `floor` tokens per second (0: generation keeps only the
    /// token each sequence gets in a step of prompt).
    pub fn new(floor: f64) -> Self {
        assert!(floor >= 0.0 && floor.is_finite(), "a floor of 0 or more, got {floor}");
        Self { floor: floor / 1e3, tokens: 0.0, ms: 0.0, owed_ms: 0.0 }
    }

    /// Milliseconds of rounds that are owed now.
    pub fn owed_ms(&self) -> f64 {
        self.owed_ms
    }

    /// What the last rounds gave a generating sequence, in tokens per
    /// millisecond; `None` before the first round.
    pub fn rate(&self) -> Option<f64> {
        (self.ms > 0.0).then(|| self.tokens / self.ms)
    }

    /// The milliseconds of rounds that work for prompts of `ms` milliseconds
    /// leaves owed, in which each generating sequence got `riding` tokens:
    /// what brings their rate over both back to the floor, and at most `ms`
    /// (half of the time).
    pub fn rounds_for(&self, ms: f64, riding: f64) -> f64 {
        let need = self.floor * ms - riding;
        if need <= 0.0 {
            return 0.0;
        }
        match self.rate() {
            Some(rate) if rate > self.floor => (need / (rate - self.floor)).min(ms),
            _ => ms,
        }
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
        if completes && self.owed_ms <= self.rounds_for(price(piece), 1.0) {
            return Turn::Piece { rows };
        }
        Turn::Round
    }

    /// Books work for prompts that took `ms` milliseconds (a step for
    /// prompts, or the image tower's units for one), while `generating`
    /// sequences generated, each of which got `riding` tokens meanwhile (the
    /// newest token that rides along in a piece): nobody to pay means no debt.
    pub fn prompts_ran(&mut self, ms: f64, riding: f64, generating: u32) {
        if generating == 0 {
            self.owed_ms = 0.0;
            return;
        }
        self.owed_ms += self.rounds_for(ms, riding);
    }

    /// Books a round that took `ms` milliseconds (its drafts' step included)
    /// and gave each of the `generating` sequences `tokens` tokens, as the
    /// mean over them. `waiting` says whether a prompt was waiting for its
    /// turn: rounds that nobody waited for are no credit. Every round teaches
    /// the rate of rounds, whoever waited.
    pub fn round_ran(&mut self, ms: f64, tokens: f64, generating: u32, waiting: bool) {
        if generating > 0 && ms > 0.0 && tokens.is_finite() {
            self.tokens = self.tokens * RATE_KEEPS + tokens;
            self.ms = self.ms * RATE_KEEPS + ms;
        }
        if generating == 0 || !waiting {
            self.owed_ms = 0.0;
            return;
        }
        // A round that overshoots leaves a credit of up to its own time
        // (rounds are only run while something is owed), which the next debt
        // starts from: rounds come in lumps, and the share holds over time
        // whatever the size of the lump.
        self.owed_ms -= ms;
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

    /// A share whose rounds have given a generating sequence 2.31 tokens in
    /// 33 ms: 0.07 tokens per millisecond.
    fn measured(floor: f64) -> Share {
        let mut share = Share::new(floor);
        for _ in 0..200 {
            share.round_ran(33.0, 2.31, 1, false);
        }
        share
    }

    /// Plays one generating sequence and prompts against each other until the
    /// prompts are read, a round taking 33 ms and giving `tokens` tokens.
    /// Returns the milliseconds the prompts took, the tokens the sequence got
    /// meanwhile, and the pieces.
    fn play(share: &mut Share, prompts: &mut [u32], tokens: f64) -> (f64, f64, u32) {
        let (mut time, mut got, mut pieces) = (0.0, 0.0, 0);
        while prompts.iter().any(|&p| p > 0) {
            match share.next(&pending(1, prompts), price) {
                Turn::Round => {
                    share.round_ran(33.0, tokens, 1, true);
                    time += 33.0;
                    got += tokens;
                }
                Turn::Piece { rows } => {
                    let mut left = rows;
                    for p in prompts.iter_mut() {
                        let take = (*p).min(left);
                        *p -= take;
                        left -= take;
                    }
                    let ms = price(rows - left + 1);
                    share.prompts_ran(ms, 1.0, 1);
                    time += ms;
                    got += 1.0;
                    pieces += 1;
                }
            }
        }
        (time, got, pieces)
    }

    #[test]
    fn nobody_generating_gives_a_prompt_the_widest_step() {
        let share = Share::new(20.0);
        assert_eq!(share.next(&pending(0, &[2000]), price), Turn::Piece { rows: 512 });
        assert_eq!(share.next(&pending(0, &[2000, 0]), price), Turn::Piece { rows: 512 });
        assert_eq!(share.next(&pending(0, &[]), price), Turn::Round);
        assert_eq!(share.next(&pending(3, &[0]), price), Turn::Round);
    }

    #[test]
    fn a_prompt_that_waits_next_to_a_long_one_makes_the_steps_pieces() {
        let share = Share::new(20.0);
        assert_eq!(share.next(&pending(0, &[20_000, 40]), price), Turn::Piece { rows: 256 });
    }

    #[test]
    fn a_prompt_goes_first_when_nothing_is_owed_and_a_long_one_in_pieces() {
        let share = Share::new(20.0);
        // The two sequences that generate ride along: 254 rows of prompt.
        assert_eq!(share.next(&pending(2, &[40]), price), Turn::Piece { rows: 254 });
        assert_eq!(share.next(&pending(2, &[5000]), price), Turn::Piece { rows: 254 });
    }

    #[test]
    fn a_piece_leaves_owed_the_rounds_that_bring_generation_back_to_its_floor() {
        // 0.07 tokens a millisecond in rounds against a floor of 0.02: a piece
        // of 300 ms in which the sequence got its newest token needs 5 tokens
        // more, which rounds bring in 100 ms beyond the floor's own share.
        let mut share = measured(20.0);
        assert!((share.rate().unwrap() - 0.07).abs() < 1e-9);
        share.prompts_ran(300.0, 1.0, 1);
        assert!((share.owed_ms() - 100.0).abs() < 1e-6, "{}", share.owed_ms());
        let long = pending(1, &[5000]);
        assert_eq!(share.next(&long, price), Turn::Round);
        share.round_ran(60.0, 4.2, 1, true);
        assert_eq!(share.next(&long, price), Turn::Round);
        share.round_ran(50.0, 3.5, 1, true);
        assert!((share.owed_ms() + 10.0).abs() < 1e-6, "the round that overshot is a credit");
        assert_eq!(share.next(&long, price), Turn::Piece { rows: 255 });

        // A higher floor owes more; a floor the rounds cannot reach, half.
        let mut thirty = measured(30.0);
        thirty.prompts_ran(300.0, 1.0, 1);
        assert!((thirty.owed_ms() - 200.0).abs() < 1e-6, "{}", thirty.owed_ms());
        let mut unreachable = measured(80.0);
        unreachable.prompts_ran(300.0, 1.0, 1);
        assert_eq!(unreachable.owed_ms(), 300.0);
        // So does a piece before any round was measured.
        let mut fresh = Share::new(20.0);
        fresh.prompts_ran(300.0, 1.0, 1);
        assert_eq!(fresh.owed_ms(), 300.0);
    }

    #[test]
    fn a_floor_of_nothing_or_one_the_riding_tokens_keep_owes_nothing() {
        let mut none = measured(0.0);
        none.prompts_ran(300.0, 1.0, 2);
        assert_eq!(none.owed_ms(), 0.0);
        assert_eq!(none.next(&pending(1, &[5000]), price), Turn::Piece { rows: 255 });
        // One token in a piece of 100 ms is 10 tokens a second.
        let mut low = measured(10.0);
        low.prompts_ran(100.0, 1.0, 1);
        assert_eq!(low.owed_ms(), 0.0);
    }

    #[test]
    fn a_short_prompt_goes_ahead_of_the_debt_of_one_piece_and_no_more() {
        let mut share = measured(20.0);
        let piece = Turn::Piece { rows: 254 };
        // One whole piece's debt.
        share.prompts_ran(price(256), 1.0, 2);
        let both = pending(2, &[5000, 60]);
        assert_eq!(share.next(&both, price), piece, "the short prompt jumps the line");
        // Its own debt is added: the next short prompt waits.
        share.prompts_ran(price(60), 1.0, 2);
        assert_eq!(share.next(&pending(2, &[5000, 60]), price), Turn::Round);
        // And a long prompt alone always waits for the debt.
        let mut share = measured(20.0);
        share.prompts_ran(price(256), 1.0, 2);
        assert_eq!(share.next(&pending(2, &[5000]), price), Turn::Round);
    }

    #[test]
    fn a_prompt_that_outranks_every_generating_sequence_does_not_wait_for_the_debt() {
        let mut share = Share::new(20.0);
        share.prompts_ran(900.0, 1.0, 2);
        let long = pending(2, &[5000]);
        assert_eq!(share.next(&long, price), Turn::Round);
        let ahead = Pending { ahead: true, ..long };
        assert_eq!(share.next(&ahead, price), Turn::Piece { rows: 254 });
        // Its pieces are owed all the same, to the prompts that come after it.
        share.prompts_ran(300.0, 1.0, 2);
        assert_eq!(share.owed_ms(), 1200.0);
    }

    #[test]
    fn rounds_that_no_prompt_waited_for_are_no_credit() {
        let mut share = measured(20.0);
        for _ in 0..100 {
            share.round_ran(35.0, 2.0, 2, false);
        }
        assert_eq!(share.owed_ms(), 0.0);
        // A prompt that comes then is not read ahead of the debt it makes.
        share.prompts_ran(300.0, 1.0, 2);
        assert_eq!(share.next(&pending(2, &[5000]), price), Turn::Round);
    }

    #[test]
    fn nobody_generating_means_nobody_is_owed() {
        let mut share = Share::new(20.0);
        share.prompts_ran(300.0, 1.0, 2);
        share.prompts_ran(600.0, 0.0, 0);
        assert_eq!(share.owed_ms(), 0.0);
        share.prompts_ran(300.0, 1.0, 2);
        share.round_ran(30.0, 2.0, 0, true);
        assert_eq!(share.owed_ms(), 0.0);
    }

    #[test]
    fn a_long_prompt_next_to_generation_leaves_it_its_floor() {
        // Rounds of 33 ms that give 2.31 tokens: 70 tokens a second alone.
        for (floor, expected) in [(20.0, 20.0), (30.0, 30.0), (50.0, 36.0), (80.0, 36.0)] {
            let mut share = measured(floor);
            let mut prompts = [8192u32];
            let (time, got, pieces) = play(&mut share, &mut prompts, 2.31);
            let rate = got / time * 1e3;
            // Pieces of 255 rows (one is the newest token of the sequence that
            // generates); the last rounds may be missing at the end. A floor
            // above 35 tokens a second (half of 70) leaves generation half of the time.
            assert_eq!(pieces, 33);
            assert!(
                (rate - expected).abs() < expected * 0.05,
                "floor {floor}: the sequence got {rate:.1} tokens a second, not {expected}"
            );
        }
    }

    #[test]
    fn a_short_prompt_next_to_a_long_one_is_read_within_one_piece_of_waiting() {
        // A 6000-row prompt is being read; a 50-row prompt arrives after
        // three pieces, with the generating sequence owed one piece's rounds.
        let mut share = measured(20.0);
        let mut prompts = [6000u32 - 3 * 256, 50];
        share.prompts_ran(price(256), 1.0, 1);
        let mut waited = 0.0;
        loop {
            match share.next(&pending(1, &prompts), price) {
                Turn::Round => {
                    share.round_ran(35.0, 2.0, 1, true);
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
