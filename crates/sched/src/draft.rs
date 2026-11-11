//! How well drafts are accepted.
//!
//! Verifying drafts is a bet. A draft adds a row to the step and, for a
//! drafter that runs on the devices, a link of its chain to the time; it pays
//! off only if it and every draft before it are accepted. How likely that is
//! depends on the drafter, on the position in the draft, and on the text, and
//! it is measured while the engine runs: every verified draft says how far it
//! got ([`Acceptance::observe`]).
//!
//! Position `i` of a draft is accepted *given* that the positions before it
//! were: a draft that failed at position 2 says nothing about position 3,
//! which was never tested. So a rate is a conditional probability, and the
//! chance that a draft of `n` tokens is accepted whole is the product of the
//! first `n` rates. Choosing how many drafts to verify, and which drafter's,
//! is [`choose`](crate::choose).
//!
//! A sequence that is new has no data. It starts from what the engine has
//! seen of the same drafter at the same position, and that from the belief the
//! engine starts with; each position moves from there to what was measured,
//! as the measurements outweigh the belief ([`BELIEF`] observations are worth
//! as much as the belief). The counts of a position that was tested forget
//! gradually, so a sequence whose text changes from new writing to copying is
//! followed within a few steps; and the counts of a drafter that is not asked
//! fade ([`Acceptance::fade`]), so that a position nothing tests for a long
//! time is not stuck at the last bad luck.

/// How many observations of a position are worth as much as the belief it
/// starts from.
pub const BELIEF: f64 = 2.0;

/// How a drafter's drafts have been accepted, per position.
#[derive(Clone, Debug)]
pub struct Acceptance {
    tested: Vec<f64>,
    accepted: Vec<f64>,
    /// What an observation leaves of the counts of the positions it tested.
    keep: f64,
}

impl Acceptance {
    /// An account of `positions` positions that forgets: each observation
    /// multiplies the counts of the positions it tested by `keep` (1: never;
    /// 0.9 remembers the last ten or so).
    pub fn new(positions: usize, keep: f64) -> Self {
        Self { tested: vec![0.0; positions], accepted: vec![0.0; positions], keep }
    }

    /// The positions the account has.
    pub fn positions(&self) -> usize {
        self.tested.len()
    }

    /// Records the outcome of a verified draft: `proposed` tokens were
    /// drafted and the first `accepted` of them were right. Positions up to
    /// the first wrong one were tested; the ones after it were not.
    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        let accepted = accepted.min(proposed);
        for position in 0..proposed.min(accepted + 1).min(self.tested.len()) {
            self.tested[position] = self.tested[position] * self.keep + 1.0;
            self.accepted[position] =
                self.accepted[position] * self.keep + f64::from(u8::from(position < accepted));
        }
    }

    /// A step went by in which the drafter was not asked: what was measured
    /// of it fades by `keep` towards the belief. Counts that nothing tests
    /// would otherwise stay as they are for ever, and a drafter that had a
    /// bad run (random text, a language it is poor at) would never be tried
    /// again, though the text it is poor at is long gone; faded, its rate
    /// climbs back until a draft is worth a try, and a try that fails is
    /// counted again.
    pub fn fade(&mut self, keep: f64) {
        for count in self.tested.iter_mut().chain(&mut self.accepted) {
            *count *= keep;
        }
    }

    /// How likely position `position` is accepted given the ones before it
    /// were, from what was measured and `belief` for what was not.
    pub fn rate(&self, position: usize, belief: f64) -> f64 {
        match (self.tested.get(position), self.accepted.get(position)) {
            (Some(tested), Some(accepted)) => (accepted + BELIEF * belief) / (tested + BELIEF),
            _ => belief,
        }
    }

    /// The rates of the first `n` positions, for a sequence whose belief at
    /// each position is what `engine` has seen there, and `start` where the
    /// engine has seen nothing. Replaces the contents of `out`.
    pub fn rates(&self, engine: &Acceptance, start: f64, n: usize, out: &mut Vec<f64>) {
        out.clear();
        out.extend((0..n).map(|position| self.rate(position, engine.rate(position, start))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_follows_what_is_observed_and_starts_from_its_belief() {
        let mut stats = Acceptance::new(3, 1.0);
        assert_eq!(stats.rate(0, 0.6), 0.6, "nothing seen: the belief");
        for _ in 0..60 {
            stats.observe(3, 1);
        }
        // Position 0 was accepted every time, position 1 rejected every time,
        // position 2 was never reached.
        assert!(stats.rate(0, 0.6) > 0.95, "{}", stats.rate(0, 0.6));
        assert!(stats.rate(1, 0.6) < 0.05, "{}", stats.rate(1, 0.6));
        assert_eq!(stats.rate(2, 0.6), 0.6, "a position that is never reached keeps its belief");
        // A few observations are worth as much as the belief and no more.
        let mut few = Acceptance::new(1, 1.0);
        for _ in 0..2 {
            few.observe(1, 0);
        }
        assert!((few.rate(0, 0.8) - 0.4).abs() < 1e-9);
        // More accepted than proposed is clamped; positions past the account are the belief.
        few.observe(2, 5);
        assert_eq!(few.rate(7, 0.3), 0.3);
    }

    #[test]
    fn counts_forget_so_a_sequence_follows_its_text() {
        let mut stats = Acceptance::new(2, 0.95);
        for _ in 0..80 {
            stats.observe(2, 2);
        }
        assert!(stats.rate(0, 0.5) > 0.9, "{}", stats.rate(0, 0.5));
        // The text changes: nothing is accepted any more.
        for _ in 0..40 {
            stats.observe(2, 0);
        }
        assert!(stats.rate(0, 0.5) < 0.2, "{}", stats.rate(0, 0.5));
    }

    #[test]
    fn a_drafter_that_is_not_asked_is_tried_again_once_its_bad_run_has_faded() {
        use crate::{Offer, Prices, Wish, choose};
        // The engine saw thirty drafts fail at their first position.
        let mut engine = Acceptance::new(8, 0.999);
        for _ in 0..30 {
            engine.observe(1, 0);
        }
        let mut own = Acceptance::new(8, 0.95);
        let price = |load: crate::Load| Some(25.0 + 0.15 * f64::from(load.rows));
        let chain = |_: usize, links: u32| 1.5 * f64::from(links);
        let mut rates = Vec::new();
        let drafts = |own: &Acceptance, engine: &Acceptance, rates: &mut Vec<f64>| {
            own.rates(engine, 0.6, 7, rates);
            let offers = [Offer { drafter: 1, chances: rates }];
            let prices = Prices { step: &price, chain: &chain, spare_rows: 7 };
            choose(&[Wish { reads: 100, offers: &offers }], &prices)[0].drafts
        };
        assert_eq!(drafts(&own, &engine, &mut rates), 0, "not worth a row and a link");
        // Rounds in which it is not asked: without fading this is for ever.
        let mut rounds = 0;
        while drafts(&own, &engine, &mut rates) == 0 {
            engine.fade(0.999);
            own.fade(0.995);
            rounds += 1;
            assert!(rounds < 2000, "the drafter is never tried again");
        }
        assert!(rounds > 20, "not at once: {rounds} rounds");
        // And a try that fails is counted, so the drafter is left alone again.
        own.observe(1, 0);
        engine.observe(1, 0);
        own.rates(&engine, 0.6, 7, &mut rates);
        assert!(rates[0] < 0.3, "{rates:?}");
    }

    #[test]
    fn a_drafter_that_is_never_right_is_tried_a_few_times_and_then_hardly_at_all() {
        use crate::{Load, Offer, Prices, Wish, choose};
        // A step of 3 ms and 0.03 ms a row, a link of 0.3 ms, exact; the
        // drafts are never accepted, and the engine has seen nothing before.
        let mut engine = Acceptance::new(8, 0.999);
        let mut own = Acceptance::new(8, 0.95);
        let price = |load: Load| Some(3.0 + 0.03 * f64::from(load.rows));
        let chain = |_: usize, links: u32| 0.3 * f64::from(links);
        let (mut rates, mut tries, mut early) = (Vec::new(), 0, 0);
        for round in 0..400 {
            own.rates(&engine, 0.6, 7, &mut rates);
            let offers = [Offer { drafter: 1, chances: &rates }];
            let prices = Prices { step: &price, chain: &chain, spare_rows: 7 };
            let drafts =
                choose(&[Wish { reads: 100, offers: &offers }], &prices)[0].drafts as usize;
            if drafts > 0 {
                tries += 1;
                early += usize::from(round < 100);
                own.observe(drafts, 0);
                engine.observe(drafts, 0);
            } else {
                own.fade(0.995);
                engine.fade(0.999);
            }
        }
        // It is learnt in a few windows, and the retries that follow are a few
        // in hundreds of rounds.
        assert!((2..=20).contains(&early), "{early} tries in the first 100 rounds");
        assert!(tries - early < 15, "{} tries in the 300 rounds after", tries - early);
    }

    #[test]
    fn a_new_sequence_starts_from_what_the_engine_has_seen_at_each_position() {
        let mut engine = Acceptance::new(4, 0.999);
        for _ in 0..200 {
            engine.observe(4, 2);
        }
        // The engine saw positions 0 and 1 accepted, 2 rejected, 3 never.
        let fresh = Acceptance::new(4, 0.95);
        let mut rates = Vec::new();
        fresh.rates(&engine, 0.6, 4, &mut rates);
        assert!(rates[0] > 0.95 && rates[1] > 0.95 && rates[2] < 0.1, "{rates:?}");
        assert_eq!(rates[3], 0.6);
        // After a few steps of its own the sequence is its own.
        let mut own = Acceptance::new(4, 0.95);
        for _ in 0..30 {
            own.observe(4, 0);
        }
        own.rates(&engine, 0.6, 2, &mut rates);
        assert!(rates[0] < 0.2 && rates.len() == 2, "{rates:?}");
    }
}
