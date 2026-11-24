//! What a drafter's own probability says about a draft.
//!
//! A drafter that picks from a distribution knows how sure it was of each
//! pick: the probability it gave the token. How often a draft of probability
//! `p` is then accepted is not `p` (the drafter is a smaller model, its
//! probability is taken over a part of the vocabulary, the verifying row may
//! sample), so it is measured: [`Confidence`] counts, by probability, how the
//! drafts that were verified came out, and [`Confidence::chance`] is the
//! chance of a draft given its probability. Before anything is measured the
//! chance is the probability itself, and each range of probabilities moves
//! from there to what was observed, as [`Acceptance`](crate::Acceptance) does
//! by position.
//!
//! The chance is what a chain is cut by. The rule is the one the windows are
//! chosen by, tokens per millisecond of the whole step: one more link of a
//! chain is worth the chance that its draft and every draft before it are
//! accepted, and costs its time; it is made when that raises the rate of the
//! step. The chances of the drafts already made are known from their
//! probabilities, so their product, the *reach* of the chain, is known when
//! the next link is decided; the next draft's own chance is not, and stands
//! at the average over all drafts ([`Confidence::mean`]). So link `j` is made
//! when `reach * mean >= rate * ms(j)`, which [`need`] turns into a bar on
//! the reach. Nothing in it is a number of its own: the rate and the price
//! are the step's, the chances are measured.

use crate::draft::BELIEF;

/// The ranges the probabilities are counted in: `[i / BINS, (i + 1) / BINS)`.
pub const BINS: usize = 16;

/// How the drafts of a drafter came out, by the probability it gave them.
#[derive(Clone, Debug)]
pub struct Confidence {
    tested: [f64; BINS],
    accepted: [f64; BINS],
    /// What an observation leaves of the counts of its range.
    keep: f64,
}

impl Confidence {
    /// An account that forgets: each observation multiplies the counts of
    /// its range by `keep` (1: never).
    pub fn new(keep: f64) -> Self {
        Self { tested: [0.0; BINS], accepted: [0.0; BINS], keep }
    }

    /// The range of `p`. A probability that is not a number is in the lowest.
    pub fn bin(p: f32) -> usize {
        if p.is_nan() { 0 } else { ((p.clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1) }
    }

    /// A draft of probability `p` was verified (every draft before it had
    /// been accepted) and was accepted or not.
    pub fn observe(&mut self, p: f32, accepted: bool) {
        let bin = Self::bin(p);
        self.tested[bin] = self.tested[bin] * self.keep + 1.0;
        self.accepted[bin] = self.accepted[bin] * self.keep + f64::from(u8::from(accepted));
    }

    /// The chance of a draft in range `bin`: what was observed, from the
    /// middle of the range where nothing was.
    fn of(&self, bin: usize) -> f64 {
        let belief = (bin as f64 + 0.5) / BINS as f64;
        (self.accepted[bin] + BELIEF * belief) / (self.tested[bin] + BELIEF)
    }

    /// The chance that a draft of probability `p` is accepted, given that
    /// the drafts before it were.
    pub fn chance(&self, p: f32) -> f64 {
        self.of(Self::bin(p))
    }

    /// The chance of every range, for a drafter that cuts its own chain.
    pub fn chances(&self) -> [f32; BINS] {
        std::array::from_fn(|bin| self.of(bin) as f32)
    }

    /// The chance of a draft whose probability is not known yet: the average
    /// over the drafts that were verified, and `start` before there were any.
    pub fn mean(&self, start: f64) -> f64 {
        let tested: f64 = self.tested.iter().sum();
        let accepted: f64 = self.accepted.iter().sum();
        (accepted + BELIEF * start) / (tested + BELIEF)
    }

    /// How often a draft leaves its chain above a bar: the share of the
    /// drafts whose chance is at least `chance`, among those that were
    /// verified and among those that were accepted. 1 where there were none.
    /// The first says how often the link behind a draft is made, the second
    /// how often it is made behind a draft that was right.
    pub fn share_at_least(&self, chance: f64) -> (f64, f64) {
        let share = |counts: &[f64; BINS]| {
            let all: f64 = counts.iter().sum();
            let above: f64 =
                (0..BINS).filter(|bin| self.of(*bin) >= chance).map(|b| counts[b]).sum();
            if all > 0.0 { above / all } else { 1.0 }
        };
        (share(&self.tested), share(&self.accepted))
    }
}

/// The reach a chain needs for one more link to pay: the link takes `link_ms`
/// in a step that makes `rate` tokens per millisecond, and its draft is
/// accepted with the chance `mean` when the chain got that far. A chain whose
/// reach (the product of the chances of its drafts so far) is below it ends.
/// Above 1 the link cannot pay whatever came before.
pub fn need(rate: f64, link_ms: f64, mean: f64) -> f64 {
    if mean > 0.0 { rate * link_ms / mean } else { f64::INFINITY }
}

/// How many links of a chain are made: link 0 always, and link `j` while the
/// reach after the links before it is at least `needs[j]`. `chances` are the
/// chances of the drafts in order ([`Confidence::chance`] of their
/// probabilities); the chain has `needs.len()` links at most. This is the
/// rule a drafter applies as it goes; here for whoever knows all the
/// chances at once.
pub fn links_made(chances: impl IntoIterator<Item = f64>, needs: &[f32]) -> usize {
    let mut reach = 1.0;
    let mut made = 0;
    for (j, chance) in chances.into_iter().enumerate().take(needs.len()) {
        if j > 0 && reach < f64::from(needs[j]) {
            break;
        }
        made += 1;
        reach *= chance;
    }
    made
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chance_starts_at_the_probability_and_follows_what_is_observed() {
        let mut seen = Confidence::new(1.0);
        // Nothing measured: the middle of the range.
        assert_eq!(seen.chance(0.97), 15.5 / 16.0);
        assert_eq!(seen.chance(0.30), 4.5 / 16.0);
        assert_eq!(seen.mean(0.6), 0.6);
        // Drafts the drafter gave 0.3 are accepted three times in four, and
        // drafts it was sure of nine times in ten.
        for i in 0..400 {
            seen.observe(0.30, i % 4 != 0);
            seen.observe(0.97, i % 10 != 0);
        }
        assert!((seen.chance(0.30) - 0.75).abs() < 0.01, "{}", seen.chance(0.30));
        assert!((seen.chance(0.97) - 0.90).abs() < 0.01, "{}", seen.chance(0.97));
        // A range nothing fell into keeps its start; the table is the same numbers.
        assert_eq!(seen.chance(0.55), 8.5 / 16.0);
        assert_eq!(seen.chances()[4], seen.chance(0.30) as f32);
        assert!((seen.mean(0.6) - 0.825).abs() < 0.01, "{}", seen.mean(0.6));
        // Half of the drafts have a chance of 0.8 and more, and 360 of the
        // 660 that were accepted; all have one of 0.7.
        assert_eq!(Confidence::new(1.0).share_at_least(0.5), (1.0, 1.0));
        assert_eq!(seen.share_at_least(0.8), (0.5, 360.0 / 660.0));
        assert_eq!(seen.share_at_least(0.7), (1.0, 1.0));
        // The ends, and a probability that is none.
        assert_eq!((Confidence::bin(0.0), Confidence::bin(1.0), Confidence::bin(7.0)), (0, 15, 15));
        assert_eq!((Confidence::bin(-1.0), Confidence::bin(f32::NAN)), (0, 0));
    }

    #[test]
    fn counts_forget_so_the_chances_follow_the_text() {
        let mut seen = Confidence::new(0.98);
        for _ in 0..300 {
            seen.observe(0.9, true);
        }
        assert!(seen.chance(0.9) > 0.95);
        for _ in 0..300 {
            seen.observe(0.9, false);
        }
        assert!(seen.chance(0.9) < 0.05, "{}", seen.chance(0.9));
    }

    #[test]
    fn a_chain_ends_where_one_more_link_does_not_pay() {
        // A step that makes 0.08 tokens per millisecond, a link of 1.5 ms,
        // drafts that are accepted 0.75 of the time on average: a link pays
        // while the chain reaches it with a chance of 0.16.
        let bar = need(0.08, 1.5, 0.75);
        assert!((bar - 0.16).abs() < 1e-12);
        let needs = [0.0, bar as f32, bar as f32, bar as f32, bar as f32];
        // Sure of every draft: the whole chain.
        assert_eq!(links_made([0.95; 5], &needs), 5);
        // Unsure of the first: 0.15 is below the bar, the chain is one link.
        assert_eq!(links_made([0.15, 0.95, 0.95, 0.95, 0.95], &needs), 1);
        // The reach is a product: 0.9 * 0.5 * 0.5 is 0.225 and carries a
        // fourth link, and with 0.6 after it 0.135 does not carry a fifth.
        assert_eq!(links_made([0.9, 0.5, 0.5, 0.6, 0.9], &needs), 4);
        // A dear link (15 ms) pays only behind drafts that are almost
        // certain, and one that costs more than a token never.
        let dear = need(0.045, 15.0, 0.75) as f32;
        assert_eq!(links_made([0.95, 0.9, 0.95], &[0.0, dear, dear]), 2);
        assert_eq!(links_made([0.85, 0.95, 0.95], &[0.0, dear, dear]), 1);
        let never = need(0.045, 30.0, 0.75) as f32;
        assert!(never > 1.0);
        assert_eq!(links_made([1.0, 1.0, 1.0], &[0.0, never, never]), 1);
        // The first link is not the chain's to refuse, and no chain is
        // longer than it was asked to be.
        assert_eq!(links_made([0.0, 0.0], &[9.0, 9.0]), 1);
        assert_eq!(links_made([1.0; 9], &[0.0, 0.0]), 2);
        assert_eq!(need(0.08, 1.5, 0.0), f64::INFINITY);
    }
}
