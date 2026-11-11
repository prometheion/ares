//! The reference of the sampler against llama.cpp's samplers, written again
//! here in double precision from llama.cpp's source (`src/llama-sampler.cpp`):
//! for random rows and every setting, the tokens the reference keeps are the
//! tokens llama.cpp's chain keeps, and the draws follow the distribution.

use ares_sampler::{
    Entry, Logprobs, Mirostat, Order, SAMPLE_STATUS_EMPTY, SAMPLE_STATUS_NONFINITE, SampleParams,
    SampleRow, Stage, TokenMask, sample_reference,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit() + 1e-12, self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A row as a model's: most tokens far below a few likely ones.
fn row(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut logits: Vec<f32> = (0..n).map(|_| (rng.normal() * 2.5 - 4.0) as f32).collect();
    for _ in 0..n.min(12) {
        let at = rng.below(n);
        logits[at] = (8.0 + rng.normal() * 3.0) as f32;
    }
    logits
}

fn reference(logits: &[f32], row: &SampleRow) -> ares_sampler::Reference {
    sample_reference(logits, |i| i as u32, row, &[], None)
}

// --- llama.cpp's samplers, in double precision ---------------------------------

#[derive(Clone, Copy)]
struct Cand {
    token: u32,
    logit: f64,
}

fn softmax(c: &[Cand]) -> Vec<f64> {
    let max = c.iter().map(|c| c.logit).fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = c.iter().map(|c| (c.logit - max).exp()).sum();
    c.iter().map(|c| (c.logit - max).exp() / sum).collect()
}

fn sorted(mut c: Vec<Cand>) -> Vec<Cand> {
    c.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap());
    c
}

/// What llama.cpp's chain keeps, and the factor it scaled the logits by.
fn llama_chain(logits: &[f32], p: &SampleParams, xtc_applies: bool) -> (Vec<u32>, f64) {
    let mut c: Vec<Cand> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| Cand { token: i as u32, logit: f64::from(l) })
        .collect();
    c = sorted(c);
    let mut scale = 1.0;
    for stage in p.program() {
        match stage {
            Stage::TopNSigma => {
                if c.len() > 1 {
                    let max = c[0].logit;
                    let mean = c.iter().map(|c| c.logit).sum::<f64>() / c.len() as f64;
                    let var =
                        c.iter().map(|c| (c.logit - mean).powi(2)).sum::<f64>() / c.len() as f64;
                    let limit = max - f64::from(p.top_n_sigma) * var.sqrt();
                    c.retain(|c| c.logit >= limit);
                }
            }
            Stage::TopK => c.truncate(p.top_k as usize),
            Stage::Typical => {
                let probs = softmax(&c);
                let entropy: f64 = probs.iter().map(|p| -p * p.ln()).sum();
                let mut idx: Vec<usize> = (0..c.len()).collect();
                let shifted: Vec<f64> = probs.iter().map(|p| (-p.ln() - entropy).abs()).collect();
                idx.sort_by(|&a, &b| shifted[a].partial_cmp(&shifted[b]).unwrap());
                let mut cum = 0.0;
                let mut keep = idx.len();
                for (i, &j) in idx.iter().enumerate() {
                    cum += probs[j];
                    if cum > f64::from(p.typical_p) {
                        keep = i + 1;
                        break;
                    }
                }
                let kept: Vec<Cand> = idx[..keep].iter().map(|&j| c[j]).collect();
                c = sorted(kept);
            }
            Stage::TopP => {
                let probs = softmax(&c);
                let mut cum = 0.0;
                let mut keep = c.len();
                for (i, p_i) in probs.iter().enumerate() {
                    cum += p_i;
                    if cum >= f64::from(p.top_p) {
                        keep = i + 1;
                        break;
                    }
                }
                c.truncate(keep);
            }
            Stage::MinP => {
                let limit = c[0].logit + f64::from(p.min_p).ln();
                c.retain(|c| c.logit >= limit);
            }
            Stage::Xtc => {
                if xtc_applies && c.len() >= 2 {
                    let probs = softmax(&c);
                    let mut last = 0;
                    for (i, p_i) in probs.iter().enumerate() {
                        if *p_i >= f64::from(p.xtc_threshold) {
                            last = i;
                        } else {
                            break;
                        }
                    }
                    if last > 0 {
                        c.drain(..last);
                    }
                }
            }
            Stage::Temperature => {
                let mut t = f64::from(p.temperature);
                if p.dynatemp_range > 0.0 {
                    if c.len() <= 1 {
                        continue;
                    }
                    let (min_t, max_t) = (
                        (t - f64::from(p.dynatemp_range)).max(0.0),
                        t + f64::from(p.dynatemp_range),
                    );
                    let probs = softmax(&c);
                    let entropy: f64 =
                        probs.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
                    let normalized = entropy / (c.len() as f64).ln();
                    t = min_t + (max_t - min_t) * normalized.powf(f64::from(p.dynatemp_exponent));
                }
                if t <= 0.0 {
                    c.truncate(1);
                } else {
                    for c in &mut c {
                        c.logit /= t;
                    }
                    scale /= t;
                }
            }
            Stage::Mirostat1 | Stage::Mirostat2 => unreachable!("handled by its own tests"),
        }
    }
    let mut tokens: Vec<u32> = c.iter().map(|c| c.token).collect();
    tokens.sort_unstable();
    (tokens, scale)
}

fn kept_tokens(logits: &[f32], row: &SampleRow) -> (Vec<u32>, f32) {
    let r = reference(logits, row);
    let l_max = r.pick.cand_logit[0];
    let tokens = logits
        .iter()
        .enumerate()
        .filter(|(_, l)| r.kept.keeps(l_max, **l))
        .map(|(i, _)| i as u32)
        .collect();
    (tokens, r.kept.inv_t)
}

fn sampled(f: impl FnOnce(&mut SampleParams)) -> SampleParams {
    let mut p = SampleParams { temperature: 1.0, seed: 5, ..SampleParams::GREEDY };
    f(&mut p);
    p
}

fn check(name: &str, p: &SampleParams, rounds: usize) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..rounds {
        let n = 100 + rng.below(1500);
        let logits = row(&mut rng, n);
        let position = round as u32 * 7 + 3;
        let seed = rng.next();
        let p = SampleParams { seed, ..*p };
        let applies = ares_sampler::arith::xtc_draw(seed, position) <= p.xtc_probability;
        let (expected, expected_scale) = llama_chain(&logits, &p, applies);
        let (kept, inv_t) = kept_tokens(&logits, &p.row(position));
        assert_eq!(kept, expected, "{name}: round {round}");
        if p.dynatemp_range > 0.0 || p.temperature != 1.0 {
            let ratio = f64::from(inv_t) / expected_scale;
            assert!(
                (ratio - 1.0).abs() < 2e-5,
                "{name}: round {round}: 1/T {inv_t}, {expected_scale}"
            );
        }
    }
}

#[test]
fn every_cut_keeps_the_tokens_llama_cpp_keeps() {
    check("top-k", &sampled(|p| p.top_k = 17), 300);
    check("top-p", &sampled(|p| p.top_p = 0.9), 300);
    check("min-p", &sampled(|p| p.min_p = 0.07), 300);
    check("typical", &sampled(|p| p.typical_p = 0.8), 300);
    check("top-n-sigma", &sampled(|p| p.top_n_sigma = 1.5), 300);
    check("xtc", &sampled(|p| (p.xtc_probability, p.xtc_threshold) = (1.0, 0.2)), 300);
    check(
        "all of them",
        &sampled(|p| {
            p.top_k = 200;
            p.top_p = 0.97;
            p.min_p = 0.01;
            p.typical_p = 0.99;
            p.top_n_sigma = 4.0;
            p.temperature = 0.8;
        }),
        300,
    );
}

#[test]
fn the_order_of_the_stages_is_the_order_they_apply_in() {
    // The temperature before the cuts changes what they cut.
    let order = |stages: &[Stage]| Order::new(stages).unwrap();
    let cuts = |p: &mut SampleParams| {
        p.top_k = 300;
        p.top_p = 0.9;
        p.min_p = 0.02;
        p.temperature = 1.7;
    };
    check(
        "temperature first",
        &sampled(|p| {
            cuts(p);
            p.order = order(&[Stage::Temperature, Stage::TopK, Stage::TopP, Stage::MinP]);
        }),
        300,
    );
    check(
        "min-p before top-p",
        &sampled(|p| {
            cuts(p);
            p.order = order(&[Stage::MinP, Stage::TopP, Stage::Temperature]);
        }),
        300,
    );
    check(
        "typical after top-p",
        &sampled(|p| {
            cuts(p);
            p.typical_p = 0.7;
            p.order = order(&[Stage::TopP, Stage::Typical, Stage::Temperature, Stage::MinP]);
        }),
        300,
    );
    check(
        "a stage twice",
        &sampled(|p| {
            cuts(p);
            p.order = order(&[
                Stage::TopK,
                Stage::Temperature,
                Stage::TopP,
                Stage::Temperature,
                Stage::MinP,
            ]);
        }),
        300,
    );
    check(
        "xtc then cuts",
        &sampled(|p| {
            cuts(p);
            p.xtc_probability = 1.0;
            p.xtc_threshold = 0.15;
            p.order = order(&[Stage::Xtc, Stage::TopP, Stage::MinP, Stage::Temperature]);
        }),
        300,
    );
    // A stage left out of the order is not applied.
    let p = sampled(|p| {
        cuts(p);
        p.order = order(&[Stage::Temperature]);
    });
    assert_eq!(p.program(), [Stage::Temperature]);
}

#[test]
fn a_dynamic_temperature_follows_the_entropy() {
    check(
        "dynatemp",
        &sampled(|p| {
            p.temperature = 1.0;
            p.dynatemp_range = 0.6;
            p.dynatemp_exponent = 1.5;
        }),
        300,
    );
    // After a cut the entropy is that of the tokens left.
    check(
        "dynatemp after top-k",
        &sampled(|p| {
            p.temperature = 0.9;
            p.dynatemp_range = 0.5;
            p.top_k = 20;
        }),
        300,
    );
    // A temperature of 0 with a range is a temperature between 0 and the range.
    let p = sampled(|p| {
        p.temperature = 0.0;
        p.dynatemp_range = 0.8;
    });
    assert!(p.sampled());
    check("dynatemp from zero", &p, 300);
}

#[test]
fn a_temperature_scales_the_draw() {
    let mut rng = Rng(11);
    let logits: Vec<f32> = (0..5).map(|_| rng.normal() as f32).collect();
    for temperature in [1.0f32, 0.5, 2.0] {
        let p = sampled(|p| p.temperature = temperature);
        let z: f64 = logits.iter().map(|l| (f64::from(*l) / f64::from(temperature)).exp()).sum();
        let mut counts = [0u32; 5];
        let n = 40_000;
        for position in 0..n {
            let r = reference(&logits, &p.row(position));
            assert_eq!(r.pick.status, 0);
            counts[r.pick.token as usize] += 1;
        }
        for (l, count) in logits.iter().zip(counts) {
            let want = (f64::from(*l) / f64::from(temperature)).exp() / z;
            let got = f64::from(count) / f64::from(n);
            assert!((got - want).abs() < 0.012, "T {temperature}: {want:.4} drawn {got:.4}");
        }
    }
}

#[test]
fn xtc_applies_with_its_probability() {
    let mut rng = Rng(3);
    let logits = row(&mut rng, 400);
    let p = sampled(|p| (p.xtc_probability, p.xtc_threshold) = (0.5, 0.1));
    let l_max = logits.iter().copied().fold(f32::MIN, f32::max);
    let best = logits.iter().position(|l| *l == l_max).unwrap() as u32;
    let mut applied = 0;
    let n = 4000;
    for position in 0..n {
        let r = reference(&logits, &p.row(position));
        // The best token is removed whenever the cut applies and a second
        // token passes the threshold; count the rows where it cannot be drawn.
        if !r.kept.keeps(l_max, logits[best as usize]) {
            applied += 1;
        }
    }
    let share = f64::from(applied) / f64::from(n);
    assert!((0.4..0.6).contains(&share), "the cut applied in {share} of the rows");
    // With a probability of 0 or a threshold above a half it never does.
    for p in [
        sampled(|p| p.xtc_threshold = 0.1),
        sampled(|p| (p.xtc_probability, p.xtc_threshold) = (1.0, 0.6)),
    ] {
        assert!((0..50).all(|pos| reference(&logits, &p.row(pos)).kept.keeps(l_max, l_max)));
    }
}

#[test]
fn mirostat_two_keeps_the_tokens_below_its_surprise_and_reports_the_one_it_drew() {
    let mut rng = Rng(8);
    for round in 0..200 {
        let logits = row(&mut rng, 300);
        let mu = 1.0 + 8.0 * rng.unit() as f32;
        let p = sampled(|p| {
            p.mirostat = Some(Mirostat::V2);
            p.mirostat_mu = mu;
            p.temperature = 0.8;
        });
        let r = reference(&logits, &p.row(round));
        let cands: Vec<Cand> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| Cand { token: i as u32, logit: f64::from(l) / 0.8 })
            .collect();
        let cands = sorted(cands);
        let probs = softmax(&cands);
        let mut keep = probs.iter().take_while(|p| -p.log2() <= f64::from(mu)).count().max(1);
        keep = keep.min(cands.len());
        let mut expect: Vec<u32> = cands[..keep].iter().map(|c| c.token).collect();
        expect.sort_unstable();
        let l_max = r.pick.cand_logit[0];
        let got: Vec<u32> = logits
            .iter()
            .enumerate()
            .filter(|(_, l)| r.kept.keeps(l_max, **l))
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(got, expect, "round {round}, mu {mu}");
        // The surprise of the token drawn among the kept ones.
        let kept = softmax(&cands[..keep]);
        let at = cands[..keep].iter().position(|c| c.token == r.pick.token).unwrap();
        assert!(
            (f64::from(r.pick.aux) + kept[at].log2()).abs() < 1e-4,
            "{} {}",
            r.pick.aux,
            -kept[at].log2()
        );
    }
}

#[test]
fn mirostat_one_takes_the_k_of_llama_cpp() {
    let mut rng = Rng(21);
    let n_vocab = 400u32;
    for round in 0..200 {
        let logits = row(&mut rng, n_vocab as usize);
        let mu = 2.0 + 6.0 * rng.unit() as f32;
        let temperature = 0.7;
        let p = sampled(|p| {
            p.mirostat = Some(Mirostat::V1);
            p.mirostat_mu = mu;
            p.temperature = temperature;
            p.vocab = n_vocab;
        });
        let cands = sorted(
            logits
                .iter()
                .enumerate()
                .map(|(i, &l)| Cand {
                    token: i as u32,
                    logit: f64::from(l) / f64::from(temperature),
                })
                .collect(),
        );
        let probs = softmax(&cands);
        let m = 100;
        let (mut sum_tb, mut sum_tt) = (0.0, 0.0);
        for i in 0..(m - 1).min(cands.len() - 1) {
            let t = (((i + 2) as f64) / ((i + 1) as f64)).ln();
            sum_tb += t * (probs[i] / probs[i + 1]).ln();
            sum_tt += t * t;
        }
        let s_hat = sum_tb / sum_tt;
        let eps = s_hat - 1.0;
        let k = ((eps * 2f64.powf(f64::from(mu))) / (1.0 - f64::from(n_vocab).powf(-eps)))
            .powf(1.0 / s_hat);
        let k = (k as usize).max(1);
        let r = reference(&logits, &p.row(round));
        let l_max = r.pick.cand_logit[0];
        let kept = logits.iter().filter(|l| r.kept.keeps(l_max, **l)).count();
        assert_eq!(kept, k, "round {round}: s_hat {s_hat}, k {k}");
    }
}

#[test]
fn corrections_and_the_mask_come_before_everything() {
    let logits = [1.0f32, 3.0, 2.0, -1.0, 0.5];
    let greedy = SampleParams::GREEDY.row(0);
    let pick = |entries: &[Entry], mask: Option<&TokenMask>, row: SampleRow| {
        let row = match mask {
            Some(_) => row.with_mask(0),
            None => row,
        };
        sample_reference(&logits, |i| i as u32, &row, entries, mask.map(TokenMask::words))
    };
    assert_eq!(pick(&[], None, greedy).pick.token, 1);
    // A ban, a penalty that halves a positive logit, a bias that lifts one.
    assert_eq!(
        pick(&[Entry { bias: f32::NEG_INFINITY, ..Entry::new(1) }], None, greedy).pick.token,
        2
    );
    assert_eq!(pick(&[Entry { rep: 4.0, ..Entry::new(1) }], None, greedy).pick.token, 2);
    assert_eq!(pick(&[Entry { bias: 5.0, ..Entry::new(3) }], None, greedy).pick.token, 3);
    // The candidates are of the corrected row.
    let corrected = pick(&[Entry { sub: 10.0, ..Entry::new(1) }], None, greedy).pick;
    assert_eq!(corrected.cand_token[..3], [2, 0, 4]);
    assert_eq!(corrected.cand_logit[4], 3.0 - 10.0);
    // The mask allows only what it names.
    let mask = TokenMask::only(5, [0, 3, 4]);
    let masked = pick(&[], Some(&mask), greedy);
    assert_eq!(masked.pick.token, 0);
    assert_eq!(masked.pick.cand_logit[3], f32::NEG_INFINITY);
    // A mask that allows nothing leaves nothing to pick.
    let none = pick(&[], Some(&TokenMask::none(5)), greedy).pick;
    assert_eq!(none.status, SAMPLE_STATUS_EMPTY);
    let none = pick(&[], Some(&TokenMask::none(5)), sampled(|_| {}).row(3)).pick;
    assert_eq!((none.token, none.status), (u32::MAX, SAMPLE_STATUS_EMPTY));
    // A sampled row draws only from the allowed tokens, whatever the noise.
    let p = sampled(|_| {});
    for position in 0..200 {
        let t = pick(&[], Some(&mask), p.row(position)).pick.token;
        assert!([0, 3, 4].contains(&t), "{t}");
    }
    // Non-finite logits are reported where the mask does not hide them.
    let bad = [f32::NAN, 1.0];
    let r = sample_reference(&bad, |i| i as u32, &greedy, &[], None);
    assert_eq!((r.pick.token, r.pick.status), (1, SAMPLE_STATUS_NONFINITE));
    let hidden = TokenMask::only(2, [1]);
    let r = sample_reference(&bad, |i| i as u32, &greedy.with_mask(0), &[], Some(hidden.words()));
    assert_eq!(r.pick.status, 0, "a masked token is not looked at");
}

#[test]
fn the_probabilities_are_the_softmax_of_the_logits_or_of_what_the_chain_left() {
    let mut rng = Rng(4);
    for round in 0..100 {
        let logits = row(&mut rng, 300);
        let l: Vec<f64> = logits.iter().map(|l| f64::from(*l)).collect();
        let max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = l.iter().map(|l| (l - max).exp()).sum();
        let lp = |x: f64| x - max - z.ln();
        // The model's own distribution, whatever the chain does.
        let raw = sampled(|p| {
            p.top_k = 5;
            p.temperature = 0.3;
            p.logprobs = Logprobs { top: 5, chosen: true, after_chain: false };
        });
        let r = reference(&logits, &raw.row(round));
        let mut order: Vec<usize> = (0..300).collect();
        order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
        assert_eq!(r.logprobs.n, 5);
        for c in 0..5 {
            assert_eq!(r.logprobs.token[c] as usize, order[c]);
            assert!((f64::from(r.logprobs.logprob[c]) - lp(l[order[c]])).abs() < 2e-6, "{c}");
        }
        let t = r.pick.token as usize;
        assert!((f64::from(r.logprobs.chosen) - lp(l[t])).abs() < 2e-6);
        // The distribution the pick is drawn from.
        let post = sampled(|p| {
            p.top_k = 5;
            p.temperature = 0.3;
            p.logprobs = Logprobs { top: 20, chosen: true, after_chain: true };
        });
        let r = reference(&logits, &post.row(round));
        assert_eq!(r.logprobs.n, 5);
        let scaled: Vec<f64> = order[..5].iter().map(|&i| l[i] / 0.3).collect();
        let zk: f64 = scaled.iter().map(|x| (x - scaled[0]).exp()).sum();
        for c in 0..5 {
            assert_eq!(r.logprobs.token[c] as usize, order[c]);
            let want = scaled[c] - scaled[0] - zk.ln();
            assert!((f64::from(r.logprobs.logprob[c]) - want).abs() < 2e-5, "{c}");
        }
        let p_sum: f64 = r.logprobs.logprob[..5].iter().map(|x| f64::from(*x).exp()).sum();
        assert!((p_sum - 1.0).abs() < 1e-5);
        let at = order[..5].iter().position(|&i| i == r.pick.token as usize).unwrap();
        assert!((f64::from(r.logprobs.chosen) - (scaled[at] - scaled[0] - zk.ln())).abs() < 2e-5);
    }
    // A row that takes the most probable token has probability 1 after the chain.
    let greedy = SampleParams {
        logprobs: Logprobs { top: 3, chosen: true, after_chain: true },
        ..SampleParams::GREEDY
    };
    let r = reference(&[0.5, 2.0, 1.0], &greedy.row(0));
    assert_eq!((r.logprobs.n, r.logprobs.token[0], r.logprobs.chosen), (1, 1, 0.0));
}
