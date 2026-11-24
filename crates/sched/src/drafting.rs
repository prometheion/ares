//! The draft policy of the rounds: what the model's drafters are worth and
//! cost, which windows a round verifies with whose drafts, and how far each
//! chain of links goes.
//!
//! The pieces are elsewhere in this crate: [`Acceptance`] and [`Confidence`]
//! say how often drafts come true, [`Curve`] what a chain of links takes on
//! the devices, [`choose`] picks the windows that raise a round's rate and
//! [`need`] sets the bar a link must pass. [`Drafting`] ties them together
//! for the runtime, which keeps what belongs to a sequence (its own
//! acceptance, its offers, the rules of its request) and asks this for the
//! rest.

use crate::{Acceptance, Choice, Confidence, Curve, Load, Prices, Wish, choose, need, rate};

/// The drafters as the choice of a window and the step record number them:
/// the n-gram drafter, which costs nothing to ask, is 0; the model's drafters
/// follow from 1 in the model's order, each with the device time its drafts
/// cost.
pub const NGRAM: usize = 0;

/// What a round leaves of the measured acceptance of a model's drafter for a
/// sequence that was offered it and did not take it, and of the engine's: a
/// drafter that nobody asks measures nothing, and what it measured when it
/// was poor would keep it out for ever. The sequence's fades quickly (it is
/// the text that changed, usually), the engine's slowly.
pub const UNASKED_SEQUENCE: f64 = 0.995;
const UNASKED_ENGINE: f64 = 0.999;

/// What the engine knows of the model's drafters (the n-gram drafter, which
/// costs nothing, is none of them), and the bars and the worth of a token
/// the last round set.
#[derive(Clone, Debug)]
pub struct Drafting {
    /// Per model drafter: how its drafts have been accepted by every
    /// sequence, per position (where a new sequence's estimate starts), and
    /// by the probability it gave them (what a chain is cut by).
    pub engines: Vec<Acceptance>,
    pub confidences: Vec<Confidence>,
    /// Per model drafter: what a chain of so many links takes, from the
    /// milliseconds of the step that only drafts.
    pub chains: Vec<Curve>,
    /// Per model drafter, the bar of each link after the first, as the last
    /// round set them.
    pub needs: Vec<Vec<f32>>,
    /// What a token costs in milliseconds at the rate the last round was
    /// chosen for; 0 before any.
    pub token_ms: f64,
}

impl Drafting {
    /// The policy for `drafters` model drafters whose windows have at most
    /// `window_rows` rows.
    pub fn new(drafters: usize, window_rows: u32) -> Self {
        Self {
            engines: (0..drafters).map(|_| Acceptance::new(window_rows as usize, 0.999)).collect(),
            confidences: (0..drafters).map(|_| Confidence::new(0.999)).collect(),
            chains: (0..drafters).map(|_| Curve::new()).collect(),
            needs: vec![Vec::new(); drafters],
            token_ms: 0.0,
        }
    }

    /// Per model drafter, how often the draft before each link leaves its
    /// chain above the link's bar, by the bars of the round before: among
    /// all drafts (how often the link is made, which is what asking for it
    /// costs) and among the drafts that are accepted (how often it is made
    /// when its row counts). The first link has no bar.
    pub fn passes(&self) -> Vec<Vec<(f64, f64)>> {
        self.needs
            .iter()
            .zip(&self.confidences)
            .map(|(needs, confidence)| {
                std::iter::once((1.0, 1.0))
                    .chain(needs.iter().map(|need| confidence.share_at_least((*need).into())))
                    .collect()
            })
            .collect()
    }

    /// The windows of a round: for each wish, whose drafts and how many, at
    /// the `step` prices, with `spare_rows` rows of drafts at most and the
    /// chains of the model's drafters paid for as `passes` says they go
    /// ([`Drafting::passes`]). Sets the worth of a token at the round's rate
    /// and the bars of the links of the next round's chains, whose windows
    /// have at most `window_rows` rows; `prior` is where a drafter's mean
    /// acceptance starts.
    pub fn choose_windows(
        &mut self,
        wishes: &[Wish<'_>],
        step: &dyn Fn(Load) -> Option<f64>,
        spare_rows: u32,
        passes: &[Vec<(f64, f64)>],
        prior: f64,
        window_rows: u32,
    ) -> Vec<Choice> {
        let chain = |drafter: usize, links: u32| match drafter {
            NGRAM => 0.0,
            d => expected_chain_ms(&self.chains[d - 1], &passes[d - 1], links),
        };
        let prices = Prices { step, chain: &chain, spare_rows };
        let choices = choose(wishes, &prices);
        // Where a chain of a drafter that makes its links one after the other
        // ends: one more link is made while the chance that its draft counts
        // (the reach of the chain so far, known to the drafter from its
        // probabilities, times the chance of a draft it has not made yet) is
        // worth the link's time at the rate this step is expected to have.
        let rate = rate(wishes, &prices, &choices);
        self.token_ms = if rate > 0.0 { 1.0 / rate } else { 0.0 };
        for (d, needs) in self.needs.iter_mut().enumerate() {
            let curve = &self.chains[d];
            let mean = self.confidences[d].mean(prior);
            let upto = |links: u32| curve.ms(links).unwrap_or(0.0);
            needs.clear();
            needs.extend(
                (1..window_rows).map(|link| need(rate, upto(link + 1) - upto(link), mean) as f32),
            );
        }
        choices
    }

    /// How many of the drafts a model's drafter has made for each wish the
    /// round verifies, now that its chains are paid for: as many as raise
    /// the round's rate at the `step` prices, with `spare_rows` rows of
    /// drafts at most. The worth of a token follows the rate.
    pub fn shorten(
        &mut self,
        wishes: &[Wish<'_>],
        step: &dyn Fn(Load) -> Option<f64>,
        spare_rows: u32,
    ) -> Vec<Choice> {
        let chain = |_: usize, _: u32| 0.0;
        let prices = Prices { step, chain: &chain, spare_rows };
        let choices = choose(wishes, &prices);
        let rate = rate(wishes, &prices, &choices);
        if rate > 0.0 {
            self.token_ms = 1.0 / rate;
        }
        choices
    }

    /// The model's drafters that no sequence asked in a round (`asked[d]`
    /// is false) fade towards what nothing was measured ([`UNASKED_SEQUENCE`]
    /// is the sequences' share of it).
    pub fn fade_unasked(&mut self, asked: &[bool]) {
        for (engine, asked) in self.engines.iter_mut().zip(asked) {
            if !asked {
                engine.fade(UNASKED_ENGINE);
            }
        }
    }
}

/// What a chain of `links` links of a model's drafter is expected to take:
/// link `j` costs what the curve says a chain of `j + 1` links takes over one
/// of `j`, and is made as often as the draft before it leaves the chain above
/// its bar (`passes[j]`; a drafter that knows its probabilities ends a chain
/// there). The price of the whole chain for a drafter that never ends one
/// early.
fn expected_chain_ms(chain: &Curve, passes: &[(f64, f64)], links: u32) -> f64 {
    let (mut ms, mut reach, mut before) = (0.0, 1.0, 0.0);
    for j in 0..links {
        let Some(upto) = chain.trial_ms(j + 1) else { return 0.0 };
        reach *= passes.get(j as usize).map_or(1.0, |(made, _)| *made);
        ms += reach * (upto - before);
        before = upto;
    }
    ms
}
