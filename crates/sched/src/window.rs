//! Which drafter's drafts go into a step's windows, and how many.
//!
//! The rule is the one number every part of speculation is measured in:
//! tokens per millisecond of the whole step. A draft row is worth the chance
//! that it and every draft before it are accepted; it costs what the step
//! costs with it, and for a drafter that runs on the devices what its chain
//! costs, which is paid once for all the windows that use the drafter, however
//! many sequences they belong to. Nothing in the rule is a number: the price
//! of a step by its rows is the table of [`PriceTable`](crate::PriceTable), the
//! chances are the acceptance measured per drafter, position and sequence
//! ([`Acceptance`](crate::Acceptance)), the lookup's chances are its own
//! confidence in each token.
//!
//! The choice is made for the whole step together, because it is a choice of
//! rows: the rows of all windows are ordered by the chance that they are
//! accepted, and the step takes the first `m` of them, for the `m` at which
//! the rate is highest. The price of a step is a table with edges, so the
//! rate is not a smooth function of `m` and every `m` is tried. Which drafter
//! a sequence uses is tried by changing one sequence's drafter at a time and
//! keeping the change when the rate of the step rises, since drafters that run
//! on the devices share a chain.

// A tree of drafts would be chosen the same way: a row's chance is the product
// of the chances along the path from the newest token to it, the rows are
// ranked by that, and the step takes the first `m`. A chain is the tree whose
// every row has the row before it for its parent; what changes is that an
// offer then names a parent for each of its rows, and that the rows taken
// must include the parent of every row taken (which ranking by path chance
// gives, since a child is never likelier than its parent).

use crate::timing::Load;

/// What one drafter offers one sequence: the chance that each of its drafts,
/// in order, is accepted given that the ones before it were.
#[derive(Clone, Copy, Debug)]
pub struct Offer<'a> {
    /// The drafter, as the caller numbers them. A drafter that costs nothing
    /// to ask (the lookup) has a chain that costs nothing, see
    /// [`Prices::chain`].
    pub drafter: usize,
    pub chances: &'a [f64],
}

/// A generating sequence in the step.
#[derive(Clone, Copy, Debug)]
pub struct Wish<'a> {
    /// The cells each row of its window reads.
    pub reads: u64,
    /// What drafters offer it. None: it does not draft in this step.
    pub offers: &'a [Offer<'a>],
}

/// What a step costs, which the caller knows.
pub struct Prices<'a> {
    /// The time of a step that carries `load` in milliseconds, or `None`
    /// before anything was measured.
    pub step: &'a dyn Fn(Load) -> Option<f64>,
    /// The time on the devices of the longest chain of `drafter` in a step,
    /// with that many links, in milliseconds.
    pub chain: &'a dyn Fn(usize, u32) -> f64,
    /// The most rows of drafts the step can take besides the newest token of
    /// every window.
    pub spare_rows: u32,
}

impl std::fmt::Debug for Prices<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prices").field("spare_rows", &self.spare_rows).finish_non_exhaustive()
    }
}

/// What a sequence's window gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    /// The drafter whose drafts are verified, or `None` for a window of the
    /// newest token alone.
    pub drafter: Option<usize>,
    /// How many of its drafts.
    pub drafts: u32,
}

const NONE: Choice = Choice { drafter: None, drafts: 0 };

/// A rate must be this much higher to count as higher.
const EPSILON: f64 = 1e-9;

/// The choice for every sequence of the step, in the order of `wishes`.
pub fn choose(wishes: &[Wish<'_>], prices: &Prices<'_>) -> Vec<Choice> {
    let n = wishes.len();
    let priced = (prices.step)(Load { rows: n as u32, cells: 0 }).is_some();
    if !priced {
        return unpriced(wishes, prices.spare_rows);
    }
    let mut assigned: Vec<Option<usize>> = vec![None; n];
    let mut best = evaluate(wishes, prices, &assigned);
    // One sequence at a time, a few rounds: a sequence that switches to a
    // chain that others use pays little for it, and one that leaves it may
    // make the chain cheaper for the rest.
    for _ in 0..3 {
        let mut changed = false;
        for i in 0..n {
            let current = assigned[i];
            for candidate in wishes[i]
                .offers
                .iter()
                .map(|o| Some(o.drafter))
                .chain([None])
                .filter(|c| *c != current)
            {
                assigned[i] = candidate;
                let tried = evaluate(wishes, prices, &assigned);
                if tried.rate > best.rate * (1.0 + EPSILON) {
                    best = tried;
                    changed = true;
                } else {
                    assigned[i] = best.assigned[i];
                }
            }
            assigned[i] = best.assigned[i];
        }
        if !changed {
            break;
        }
    }
    refine(wishes, prices, &mut best);
    (0..n)
        .map(|i| match (best.assigned[i], best.drafts[i]) {
            (Some(drafter), drafts) if drafts > 0 => Choice { drafter: Some(drafter), drafts },
            _ => NONE,
        })
        .collect()
}

/// The rate a choice is expected to have, in tokens per millisecond of the
/// whole step: what [`choose`] maximised. For whoever decides something else
/// by the same rule (a drafter that ends its chain where one more link would
/// lower this, see [`need`](crate::need)).
pub fn rate(wishes: &[Wish<'_>], prices: &Prices<'_>, choices: &[Choice]) -> f64 {
    let assigned: Vec<Option<usize>> = choices.iter().map(|c| c.drafter).collect();
    let drafts: Vec<u32> = choices.iter().map(|c| c.drafts).collect();
    score(wishes, prices, &assigned, &drafts)
}

/// The rate of the step if each sequence `i` verifies `drafts[i]` drafts of
/// the drafter `assigned[i]`: the tokens it is expected to give over the time
/// of the step and of the chains.
fn score(
    wishes: &[Wish<'_>],
    prices: &Prices<'_>,
    assigned: &[Option<usize>],
    drafts: &[u32],
) -> f64 {
    let n = wishes.len();
    let mut tokens = n as f64;
    let mut rows = n as u32;
    let mut cells = 0u64;
    let mut longest: Vec<u32> = Vec::new();
    for ((wish, drafter), &len) in wishes.iter().zip(assigned).zip(drafts) {
        cells += wish.reads * (1 + u64::from(len));
        rows += len;
        let Some(drafter) = drafter.filter(|_| len > 0) else { continue };
        let Some(offer) = wish.offers.iter().find(|o| o.drafter == drafter) else { continue };
        let mut reach = 1.0;
        for p in offer.chances.iter().take(len as usize) {
            reach *= p.clamp(0.0, 1.0);
            tokens += reach;
        }
        if longest.len() <= drafter {
            longest.resize(drafter + 1, 0);
        }
        longest[drafter] = longest[drafter].max(len);
    }
    let step = (prices.step)(Load { rows, cells }).unwrap_or(f64::INFINITY);
    let chains: f64 = longest
        .iter()
        .enumerate()
        .filter(|(_, l)| **l > 0)
        .map(|(d, l)| (prices.chain)(d, *l))
        .sum();
    let ms = step + chains;
    if ms > 0.0 { tokens / ms } else { f64::INFINITY }
}

/// The rows were picked likeliest first, which is the best order when a row
/// costs the same wherever it goes. Rows differ in what they cost (a window
/// whose sequence is deep in its context reads more cells, and the longest
/// chain of a drafter is paid for by the sequence that has it), so each
/// sequence's length is tried one shorter and one longer while the rate rises.
fn refine(wishes: &[Wish<'_>], prices: &Prices<'_>, best: &mut Tried) {
    let spare = prices.spare_rows;
    for _ in 0..4 {
        let mut changed = false;
        for i in 0..wishes.len() {
            let Some(drafter) = best.assigned[i] else { continue };
            let Some(offer) = wishes[i].offers.iter().find(|o| o.drafter == drafter) else {
                continue;
            };
            for len in [best.drafts[i].saturating_sub(1), best.drafts[i] + 1] {
                let used: u32 = best.drafts.iter().sum::<u32>() - best.drafts[i] + len;
                if len == best.drafts[i] || len as usize > offer.chances.len() || used > spare {
                    continue;
                }
                let before = best.drafts[i];
                best.drafts[i] = len;
                let rate = score(wishes, prices, &best.assigned, &best.drafts);
                if rate > best.rate * (1.0 + EPSILON) {
                    best.rate = rate;
                    changed = true;
                    break;
                }
                best.drafts[i] = before;
            }
        }
        if !changed {
            break;
        }
    }
    for (assigned, &drafts) in best.assigned.iter_mut().zip(&best.drafts) {
        if drafts == 0 {
            *assigned = None;
        }
    }
}

/// Before any step was measured nothing says what a row costs, so each
/// sequence tries the whole of the offer that promises it the most tokens,
/// as far as the rows go, the likeliest rows first.
fn unpriced(wishes: &[Wish<'_>], spare_rows: u32) -> Vec<Choice> {
    let mut picks: Vec<(usize, &Offer<'_>)> = Vec::new();
    for (i, wish) in wishes.iter().enumerate() {
        let promise = |o: &Offer<'_>| {
            let mut reach = 1.0;
            o.chances
                .iter()
                .map(|p| {
                    reach *= p.clamp(0.0, 1.0);
                    reach
                })
                .sum::<f64>()
        };
        if let Some(best) = wish.offers.iter().max_by(|a, b| promise(a).total_cmp(&promise(b))) {
            picks.push((i, best));
        }
    }
    let assigned: Vec<Option<usize>> = (0..wishes.len())
        .map(|i| picks.iter().find(|(j, _)| *j == i).map(|(_, o)| o.drafter))
        .collect();
    let rows = ranked(wishes, &assigned);
    let mut out = vec![NONE; wishes.len()];
    for (_, i, _) in rows.into_iter().take(spare_rows as usize) {
        out[i].drafts += 1;
        out[i].drafter = assigned[i];
    }
    out
}

/// The result of trying one assignment of drafters.
struct Tried {
    rate: f64,
    assigned: Vec<Option<usize>>,
    drafts: Vec<u32>,
}

/// The rows of the windows of the sequences that draft, likeliest first:
/// `(chance the row and every row before it is accepted, sequence, position)`.
fn ranked(wishes: &[Wish<'_>], assigned: &[Option<usize>]) -> Vec<(f64, usize, usize)> {
    let mut rows = Vec::new();
    for (i, wish) in wishes.iter().enumerate() {
        let Some(offer) = assigned[i].and_then(|d| wish.offers.iter().find(|o| o.drafter == d))
        else {
            continue;
        };
        let mut reach = 1.0;
        for (position, p) in offer.chances.iter().enumerate() {
            reach *= p.clamp(0.0, 1.0);
            rows.push((reach, i, position));
        }
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
    rows
}

/// The best number of draft rows for an assignment of drafters, and its rate.
fn evaluate(wishes: &[Wish<'_>], prices: &Prices<'_>, assigned: &[Option<usize>]) -> Tried {
    let n = wishes.len();
    let rows = ranked(wishes, assigned);
    let drafters = assigned.iter().flatten().max().map_or(0, |d| d + 1);
    let mut longest = vec![0u32; drafters];
    let mut drafts = vec![0u32; n];
    let mut cells: u64 = wishes.iter().map(|w| w.reads).sum();
    let mut tokens = n as f64;
    let rate_of = |taken: usize, cells: u64, tokens: f64, longest: &[u32]| {
        let load = Load { rows: (n + taken) as u32, cells };
        let step = (prices.step)(load).unwrap_or(f64::INFINITY);
        let chains: f64 = longest
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > 0)
            .map(|(d, l)| (prices.chain)(d, *l))
            .sum();
        let ms = step + chains;
        if ms > 0.0 { tokens / ms } else { f64::INFINITY }
    };
    let mut best = (rate_of(0, cells, tokens, &longest), 0usize);
    for (taken, (reach, i, _)) in rows.iter().take(prices.spare_rows as usize).enumerate() {
        drafts[*i] += 1;
        cells += wishes[*i].reads;
        tokens += reach;
        let d = assigned[*i].expect("a ranked row belongs to a drafting sequence");
        longest[d] = longest[d].max(drafts[*i]);
        let rate = rate_of(taken + 1, cells, tokens, &longest);
        if rate > best.0 * (1.0 + EPSILON) {
            best = (rate, taken + 1);
        }
    }
    // The counts at the best number of rows.
    drafts.iter_mut().for_each(|d| *d = 0);
    for (_, i, _) in rows.iter().take(best.1) {
        drafts[*i] += 1;
    }
    // A sequence that drafts nothing has no drafter.
    let assigned =
        assigned.iter().zip(&drafts).map(|(a, d)| if *d > 0 { *a } else { None }).collect();
    Tried { rate: best.0, assigned, drafts }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step that costs `base` and `row` per row, with `edge` more from the
    /// ninth row on and again from the seventeenth.
    fn step(base: f64, row: f64, edge: f64) -> impl Fn(Load) -> Option<f64> {
        move |load| {
            let launches = load.rows.div_ceil(8).max(1);
            Some(base + row * f64::from(load.rows) + edge * f64::from(launches - 1))
        }
    }

    const LOOKUP: usize = 0;
    const CHAIN: usize = 1;

    fn free(_: usize, _: u32) -> f64 {
        0.0
    }

    #[test]
    fn a_copying_sequence_takes_a_long_draft_up_to_the_edge_of_a_launch() {
        // 25 ms for the weights, 0.15 ms a row, 6 ms for a second launch.
        let price = step(25.0, 0.15, 6.0);
        let sure = [0.97; 16];
        let offers = [Offer { drafter: LOOKUP, chances: &sure }];
        let wishes = [Wish { reads: 1000, offers: &offers }];
        let prices = Prices { step: &price, chain: &free, spare_rows: 15 };
        let pick = choose(&wishes, &prices);
        // A ninth row costs a second launch: 7 drafts fill the first, and the
        // next 8 are worth 0.97^8 and more, which pays 6 ms.
        assert_eq!(pick[0].drafter, Some(LOOKUP));
        assert_eq!(pick[0].drafts, 15, "{pick:?}");

        // Rows that cost a whole millisecond each are worth less: drafts that are
        // right half of the time stop paying at the fourth.
        let dear = step(25.0, 1.1, 6.0);
        let prices = Prices { step: &dear, chain: &free, spare_rows: 15 };
        let half = [0.55; 16];
        let offers = [Offer { drafter: LOOKUP, chances: &half }];
        let wishes = [Wish { reads: 1000, offers: &offers }];
        assert_eq!(choose(&wishes, &prices)[0].drafts, 4);

        // The edge at the ninth row is not crossed when the second launch
        // costs more than the eight rows after it bring.
        let cliff = step(25.0, 0.15, 14.0);
        let prices = Prices { step: &cliff, chain: &free, spare_rows: 15 };
        let strong = [0.92; 16];
        let offers = [Offer { drafter: LOOKUP, chances: &strong }];
        let wishes = [Wish { reads: 1000, offers: &offers }];
        assert_eq!(choose(&wishes, &prices)[0].drafts, 7);
    }

    #[test]
    fn drafts_that_do_not_pay_are_not_made() {
        let price = step(25.0, 1.1, 0.0);
        let poor = [0.05, 0.02];
        let offers = [Offer { drafter: CHAIN, chances: &poor }];
        let wishes = [Wish { reads: 100, offers: &offers }];
        let chain = |_: usize, links: u32| 1.5 * f64::from(links);
        let prices = Prices { step: &price, chain: &chain, spare_rows: 7 };
        assert_eq!(choose(&wishes, &prices), [NONE]);
    }

    #[test]
    fn the_chain_of_a_drafter_is_paid_once_for_all_the_sequences_that_use_it() {
        // A chain costs 2.5 ms a link, however many sequences use it. One draft
        // of 0.08 does not pay for a link alone; next to a sequence whose
        // three links pay, it rides for the price of its row.
        let price = step(25.0, 0.2, 0.0);
        let chain = |_: usize, links: u32| 2.5 * f64::from(links);
        let strong = [0.9, 0.9, 0.9];
        let faint = [0.08];
        let a = [Offer { drafter: CHAIN, chances: &strong }];
        let b = [Offer { drafter: CHAIN, chances: &faint }];
        let prices = Prices { step: &price, chain: &chain, spare_rows: 30 };
        let alone = choose(&[Wish { reads: 100, offers: &b }], &prices);
        assert_eq!(alone, [NONE]);
        let together =
            choose(&[Wish { reads: 100, offers: &a }, Wish { reads: 100, offers: &b }], &prices);
        assert_eq!(together[0].drafts, 3, "{together:?}");
        assert_eq!(together[1], Choice { drafter: Some(CHAIN), drafts: 1 }, "{together:?}");
    }

    #[test]
    fn a_sequence_with_text_to_copy_leaves_the_chain_to_the_others() {
        let price = step(25.0, 0.2, 0.0);
        let chain = |d: usize, links: u32| if d == CHAIN { f64::from(links) } else { 0.0 };
        let copy = [0.95; 8];
        let guess = [0.55, 0.4, 0.3];
        let a =
            [Offer { drafter: LOOKUP, chances: &copy }, Offer { drafter: CHAIN, chances: &guess }];
        let b = [Offer { drafter: CHAIN, chances: &guess }];
        let wishes = [Wish { reads: 100, offers: &a }, Wish { reads: 100, offers: &b }];
        let prices = Prices { step: &price, chain: &chain, spare_rows: 12 };
        let pick = choose(&wishes, &prices);
        assert_eq!(pick[0], Choice { drafter: Some(LOOKUP), drafts: 8 }, "{pick:?}");
        assert_eq!(pick[1], Choice { drafter: Some(CHAIN), drafts: 1 }, "{pick:?}");
    }

    #[test]
    fn the_rows_that_fit_are_given_to_the_likeliest_drafts_of_any_sequence() {
        let price = step(25.0, 0.1, 0.0);
        let strong = [0.9, 0.9, 0.9, 0.9];
        let weak = [0.6, 0.6, 0.6, 0.6];
        let a = [Offer { drafter: LOOKUP, chances: &weak }];
        let b = [Offer { drafter: LOOKUP, chances: &strong }];
        let wishes = [Wish { reads: 10, offers: &a }, Wish { reads: 10, offers: &b }];
        let prices = Prices { step: &price, chain: &free, spare_rows: 5 };
        let pick = choose(&wishes, &prices);
        // 0.9, 0.81, 0.73, 0.66 of the second, 0.6 of the first: five rows.
        assert_eq!((pick[0].drafts, pick[1].drafts), (1, 4), "{pick:?}");
        let prices = Prices { spare_rows: 2, ..prices };
        let pick = choose(&wishes, &prices);
        assert_eq!((pick[0].drafts, pick[1].drafts), (0, 2), "{pick:?}");
    }

    #[test]
    fn before_a_step_was_measured_each_sequence_tries_its_best_offer_whole() {
        let none = |_: Load| None;
        let a = [0.9, 0.9, 0.9];
        let b = [0.2, 0.2];
        let offers =
            [Offer { drafter: CHAIN, chances: &b }, Offer { drafter: LOOKUP, chances: &a }];
        let wishes = [Wish { reads: 10, offers: &offers }];
        let prices = Prices { step: &none, chain: &free, spare_rows: 10 };
        assert_eq!(choose(&wishes, &prices), [Choice { drafter: Some(LOOKUP), drafts: 3 }]);
        let prices = Prices { spare_rows: 1, ..prices };
        assert_eq!(choose(&wishes, &prices)[0].drafts, 1);
    }

    #[test]
    fn the_rate_of_a_choice_is_its_tokens_over_its_time() {
        let price = step(25.0, 0.5, 0.0);
        let chain = |_: usize, links: u32| 1.5 * f64::from(links);
        let chances = [0.8, 0.5];
        let offers = [Offer { drafter: CHAIN, chances: &chances }];
        let wishes = [Wish { reads: 0, offers: &offers }];
        let prices = Prices { step: &price, chain: &chain, spare_rows: 7 };
        // Two drafts: 1 + 0.8 + 0.4 tokens in 26.5 ms of step and 3 of chain.
        let two = [Choice { drafter: Some(CHAIN), drafts: 2 }];
        assert!((rate(&wishes, &prices, &two) - 2.2 / 29.5).abs() < 1e-12);
        assert!((rate(&wishes, &prices, &[NONE]) - 1.0 / 25.5).abs() < 1e-12);
    }

    #[test]
    fn nothing_offered_is_nothing_drafted() {
        let price = step(25.0, 0.2, 0.0);
        let prices = Prices { step: &price, chain: &free, spare_rows: 8 };
        let wishes = [Wish { reads: 10, offers: &[] }; 3];
        assert_eq!(choose(&wishes, &prices), [NONE; 3]);
        assert!(choose(&[], &prices).is_empty());
    }

    /// The rate of a choice, by the rule.
    fn rate_of(wishes: &[Wish<'_>], picks: &[Choice], prices: &Prices<'_>) -> f64 {
        let mut tokens = wishes.len() as f64;
        let mut rows = wishes.len() as u32;
        let mut cells = 0;
        let mut longest = [0u32; 4];
        for (wish, pick) in wishes.iter().zip(picks) {
            cells += wish.reads * (1 + u64::from(pick.drafts));
            rows += pick.drafts;
            if let Some(d) = pick.drafter {
                let offer = wish.offers.iter().find(|o| o.drafter == d).unwrap();
                let mut reach = 1.0;
                for p in &offer.chances[..pick.drafts as usize] {
                    reach *= p;
                    tokens += reach;
                }
                longest[d] = longest[d].max(pick.drafts);
            }
        }
        let chains: f64 =
            (0..4).map(|d| if longest[d] > 0 { (prices.chain)(d, longest[d]) } else { 0.0 }).sum();
        tokens / ((prices.step)(Load { rows, cells }).unwrap() + chains)
    }

    #[test]
    fn the_choice_is_as_good_as_the_best_of_all_choices() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut worse = 0;
        let mut least: f64 = 1.0;
        let cases = 300;
        for _ in 0..cases {
            let price = step(20.0 + 10.0 * unit(), 0.1 + unit(), 8.0 * unit());
            let chain = |d: usize, links: u32| {
                if d == 0 { 0.0 } else { (0.5 + 2.0 * (d as f64)) * f64::from(links) }
            };
            let lengths = [1 + (unit() * 5.0) as usize, 1 + (unit() * 5.0) as usize];
            let chances: Vec<Vec<f64>> =
                (0..4).map(|k| (0..lengths[k % 2]).map(|_| unit()).collect()).collect();
            let offers: Vec<[Offer<'_>; 2]> = (0..2)
                .map(|i| {
                    [
                        Offer { drafter: 0, chances: &chances[i] },
                        Offer { drafter: 1 + i % 2, chances: &chances[2 + i] },
                    ]
                })
                .collect();
            let wishes: Vec<Wish<'_>> = offers
                .iter()
                .map(|o| Wish { reads: (unit() * 5000.0) as u64, offers: o })
                .collect();
            let prices = Prices { step: &price, chain: &chain, spare_rows: 10 };
            let got = rate_of(&wishes, &choose(&wishes, &prices), &prices);
            // Every way to give each of two sequences a drafter and a length.
            let mut best = rate_of(&wishes, &[NONE, NONE], &prices);
            let options = |w: &Wish<'_>| {
                let mut all = vec![NONE];
                for o in w.offers {
                    for len in 1..=o.chances.len() as u32 {
                        all.push(Choice { drafter: Some(o.drafter), drafts: len });
                    }
                }
                all
            };
            for a in options(&wishes[0]) {
                for b in options(&wishes[1]) {
                    if a.drafts + b.drafts <= prices.spare_rows {
                        best = best.max(rate_of(&wishes, &[a, b], &prices));
                    }
                }
            }
            assert!(got <= best * (1.0 + 1e-9), "the choice cannot beat the best: {got} {best}");
            worse += usize::from(got < best * 0.999);
            least = least.min(got / best);
            assert!(got >= best * 0.95, "{got} against the best {best}");
        }
        eprintln!("RES {worse} of {cases} choices were not the best, the worst at {least:.3}");
        assert!(worse * 20 <= cases, "{worse} of {cases} choices were not the best");
    }
}
