//! How many draft tokens to verify.
//!
//! Verifying drafts is a bet. Each draft adds a row to the step and takes
//! time to produce; it pays off only if it and all drafts before it are
//! accepted. The best length is the one with the highest expected tokens per
//! millisecond, which depends on how well the drafter is doing right now and
//! on what a row costs right now. Both are measured, so the length adapts by
//! itself: long drafts while text is being copied, short or none where the
//! drafter keeps missing.
//!
//! The same number decides between drafters (a draft head of the model, an
//! n-gram lookup): compute the best choice for each and take the higher rate.
//! It also covers fairness between sequences, because the rate is taken over
//! the whole step: a draft that slows the other sequences in the batch counts
//! against itself.

/// Measured acceptance of one drafter for one sequence, per draft position.
///
/// Position `i` holds the probability that draft `i` is accepted given that
/// the drafts before it were. Later positions are usually worse, and whether
/// a position was reached at all depends on the earlier ones, which is why
/// the probabilities are conditional.
#[derive(Clone, Debug)]
pub struct Acceptance {
    rate: Vec<f64>,
    /// How much one observation moves a rate.
    step: f64,
}

impl Acceptance {
    /// Starts every one of `max_len` positions at `prior`. An optimistic
    /// prior makes a new drafter get tried; its real rates take over within a
    /// few steps.
    pub fn new(max_len: usize, prior: f64) -> Self {
        Self { rate: vec![prior.clamp(0.0, 1.0); max_len], step: 0.1 }
    }

    /// Records the outcome of a verified draft: `proposed` tokens were
    /// drafted and the first `accepted` of them were right.
    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        let accepted = accepted.min(proposed);
        for (i, rate) in self.rate.iter_mut().enumerate().take(proposed) {
            if i < accepted {
                *rate += self.step * (1.0 - *rate);
            } else if i == accepted {
                // The first rejected draft. Positions after it were never
                // tested and say nothing.
                *rate -= self.step * *rate;
            }
        }
    }

    /// The conditional acceptance probability of each position.
    pub fn rates(&self) -> &[f64] {
        &self.rate
    }
}

/// What producing drafts costs, in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftCost {
    /// Once per step in which the drafter runs (catching up with the
    /// sequence, for a draft head).
    pub fixed_ms: f64,
    /// Per draft token (0 for a lookup).
    pub per_token_ms: f64,
}

/// The outcome of [`best_draft_len`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftChoice {
    /// How many drafts to verify; 0 means not to draft.
    pub len: usize,
    /// Expected tokens per millisecond of the whole step with that choice.
    pub tokens_per_ms: f64,
}

/// The draft length with the highest expected tokens per millisecond.
///
/// - `accept[i]`: probability that draft `i` is accepted given the earlier
///   ones were, from [`Acceptance::rates`] or from the drafter's own
///   confidence in each proposed token. Its length bounds the draft.
/// - `base_ms`, `per_row_ms`: the step cost, from
///   [`StepTimeModel`](crate::StepTimeModel).
/// - `other_rows`, `other_tokens`: the rows the rest of the step has anyway
///   (this sequence's newest token included) and the tokens they are
///   expected to yield. With these the rate is that of the whole step.
pub fn best_draft_len(
    accept: &[f64],
    base_ms: f64,
    per_row_ms: f64,
    other_rows: u32,
    other_tokens: f64,
    cost: DraftCost,
) -> DraftChoice {
    let plain_ms = base_ms + per_row_ms * f64::from(other_rows);
    let mut best = DraftChoice { len: 0, tokens_per_ms: rate(other_tokens, plain_ms) };
    let mut tokens = other_tokens;
    let mut reach = 1.0;
    for (i, p) in accept.iter().enumerate() {
        reach *= p.clamp(0.0, 1.0);
        tokens += reach;
        let len = i + 1;
        let ms = plain_ms + cost.fixed_ms + (per_row_ms + cost.per_token_ms) * len as f64;
        let candidate = rate(tokens, ms);
        if candidate > best.tokens_per_ms {
            best = DraftChoice { len, tokens_per_ms: candidate };
        }
    }
    best
}

fn rate(tokens: f64, ms: f64) -> f64 {
    if ms > 0.0 { tokens / ms } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREE: DraftCost = DraftCost { fixed_ms: 0.0, per_token_ms: 0.0 };

    #[test]
    fn copying_text_drafts_long_and_guessing_does_not_draft() {
        // A step costs 33 ms plus 1.1 ms per row; the sequence is alone.
        let copy = [0.97; 16];
        let choice = best_draft_len(&copy, 33.0, 1.1, 1, 1.0, FREE);
        assert_eq!(choice.len, 16, "nearly free rows that are nearly always right");
        assert!(choice.tokens_per_ms > 5.0 * (1.0 / 34.1));

        // A drafter that is right one time in ten, at 2.5 ms per draft token
        // plus 1.2 ms to catch up: not worth running.
        let poor = [0.1; 4];
        let cost = DraftCost { fixed_ms: 1.2, per_token_ms: 2.5 };
        assert_eq!(best_draft_len(&poor, 33.0, 1.1, 1, 1.0, cost).len, 0);
    }

    #[test]
    fn the_length_stops_where_drafts_stop_paying() {
        // Good for two tokens, then poor: the cut is after the second.
        let accept = [0.9, 0.8, 0.2, 0.2, 0.2];
        let cost = DraftCost { fixed_ms: 1.2, per_token_ms: 2.5 };
        assert_eq!(best_draft_len(&accept, 33.0, 1.1, 1, 1.0, cost).len, 2);
        // The same drafts verified for free go on for as long as they add
        // anything.
        assert_eq!(best_draft_len(&accept, 33.0, 0.0, 1, 1.0, FREE).len, 5);
    }

    #[test]
    fn rows_that_slow_other_sequences_count_against_the_draft() {
        // Rows are expensive here (5 ms). Alone, two drafts at 0.7 pay off.
        let accept = [0.7, 0.7];
        assert_eq!(best_draft_len(&accept, 10.0, 5.0, 1, 1.0, FREE).len, 2);
        // Next to 7 other generating sequences the step is already long, and
        // every row delays 8 tokens: drafting loses.
        assert_eq!(best_draft_len(&accept, 10.0, 5.0, 8, 8.0, FREE).len, 0);
        // With cheap rows the others barely notice, and drafting wins again.
        assert_eq!(best_draft_len(&accept, 10.0, 0.2, 8, 8.0, FREE).len, 2);
    }

    #[test]
    fn acceptance_follows_what_is_observed() {
        let mut stats = Acceptance::new(3, 0.5);
        for _ in 0..60 {
            stats.observe(3, 1);
        }
        let rates = stats.rates();
        assert!(rates[0] > 0.99, "{rates:?}");
        assert!(rates[1] < 0.01, "{rates:?}");
        assert_eq!(rates[2], 0.5, "a position that is never reached keeps its prior");

        // All accepted moves every proposed position up; more accepted than
        // proposed is clamped.
        stats.observe(2, 5);
        assert!(stats.rates()[1] > 0.09);
        assert_eq!(stats.rates()[2], 0.5);
    }
}
