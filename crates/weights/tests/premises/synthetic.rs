//! A synthetic routing workload for the premise tests.
//!
//! Nothing in it was measured on a model. It is built to have the properties
//! of real routing that the survey (`docs/reference/moe-offload-survey.md`)
//! reports, so that a technique that depends on them can be shown to work on
//! it, and so that the numbers it gives are not mistaken for numbers of a
//! model:
//!
//! - **Locality in time.** The text stays on one topic for 300 to 900
//!   tokens, and its tokens repeat (a few words are most of it). In the layers
//!   after the first ones, 80 % of a token's picks come from the topic's group
//!   of 8 to 16 experts of that layer, the experts of the group ranked by use;
//!   so an expert used now is very likely used again within a few tokens.
//! - **Topics that change.** There are six topics, each with its own group of
//!   experts in every layer and its own words. The text jumps from one topic
//!   to another at random, and returns to topics it had.
//! - **Early layers that route flatly.** In the first layers a token's picks
//!   do not follow the topic. They follow the token: 35 to 95 % of a token's
//!   picks (it differs from token to token) are fixed by its id, 10 % by the
//!   id before it, and the rest go anywhere, from a popularity that is nearly
//!   even. Over a text every expert of such a
//!   layer is used about equally, and what a token routes to is a function of
//!   it. (That routing depends on the token in the first layers is the premise
//!   of the token table; here it is built in, not found.)
//! - **Skewed popularity.** In every layer some experts are used more than
//!   others outside their topic as well, more so in later layers.

/// A small xorshift generator; the premise tests must give the same numbers on
/// every run.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A fixed hash of a few numbers, to a number in `[0, 1)`.
fn hash_unit(parts: &[u64]) -> f64 {
    let mut x = 0x243f_6a88_85a3_08d3u64;
    for part in parts {
        x ^= part.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
    }
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A distribution over `0..n` where `i` has weight `(i + 1)^-s`.
struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    fn new(n: usize, s: f64) -> Self {
        let mut total = 0.0;
        let mut cdf: Vec<f64> = (0..n)
            .map(|i| {
                total += ((i + 1) as f64).powf(-s);
                total
            })
            .collect();
        for c in &mut cdf {
            *c /= total;
        }
        Self { cdf }
    }

    /// The item at the quantile `u` in `[0, 1)`.
    fn at(&self, u: f64) -> usize {
        self.cdf.partition_point(|c| *c <= u).min(self.cdf.len() - 1)
    }
}

/// How many layers, experts and so on the workload has.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub layers: usize,
    /// The first this many layers route flatly, by token.
    pub early: usize,
    pub experts: usize,
    /// Experts picked per token and layer.
    pub k: usize,
    pub vocab: usize,
    pub topics: usize,
}

/// One layer's popularity of experts, outside any topic.
struct Popularity {
    zipf: Zipf,
    /// Rank -> expert.
    order: Vec<u32>,
}

impl Popularity {
    fn at(&self, u: f64) -> u32 {
        self.order[self.zipf.at(u)]
    }
}

/// The text and the routing of every layer for each token of it.
pub struct Workload {
    pub shape: Shape,
    /// The token fed to the model at each pass.
    pub tokens: Vec<u32>,
    routed: Vec<u32>,
    popularity: Vec<Popularity>,
}

impl Workload {
    pub fn generate(shape: Shape, passes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let Shape { layers, early, experts, k, vocab, topics } = shape;

        // Popularity: nearly even in the early layers, more skewed in later ones.
        let popularity: Vec<Popularity> = (0..layers)
            .map(|l| {
                let s = if l < early {
                    0.3
                } else {
                    0.6 + 0.5 * (l - early) as f64 / (layers - early).max(1) as f64
                };
                let mut order: Vec<u32> = (0..experts as u32).collect();
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.below(i + 1));
                }
                Popularity { zipf: Zipf::new(experts, s), order }
            })
            .collect();

        // Each topic's group of experts in each later layer: 8 to 16 of them,
        // drawn by popularity, in the order of use inside the group.
        let group_size = |l: usize| 8 + (l * 5) % 9;
        let groups: Vec<Vec<Vec<u32>>> = (0..layers)
            .map(|l| {
                (0..topics)
                    .map(|_| {
                        let mut group: Vec<u32> = Vec::new();
                        while group.len() < group_size(l) {
                            let e = popularity[l].at(rng.unit());
                            if !group.contains(&e) {
                                group.push(e);
                            }
                        }
                        group
                    })
                    .collect()
            })
            .collect();
        let in_group = Zipf::new(16, 0.8);

        // Words: 60 function words every topic uses, and 250 of its own.
        let function_words = Zipf::new(60, 1.0);
        let own_words = Zipf::new(250, 1.0);
        let words: Vec<Vec<u32>> = (0..topics)
            .map(|_| (0..250).map(|_| (60 + rng.below(vocab - 60)) as u32).collect())
            .collect();

        let mut tokens = Vec::with_capacity(passes);
        let mut routed = Vec::with_capacity(passes * layers * k);
        let (mut current, mut until) = (0usize, 300 + rng.below(600));
        let mut previous = None::<u32>;
        for pass in 0..passes {
            if pass >= until {
                current = (current + 1 + rng.below(topics - 1)) % topics;
                until = pass + 300 + rng.below(600);
            }
            let token = if rng.unit() < 0.4 {
                function_words.at(rng.unit()) as u32
            } else {
                words[current][own_words.at(rng.unit())]
            };
            // How strongly this token fixes its picks in the early layers:
            // from 35 % to 95 %, by token.
            let fixed = 0.35 + 0.6 * hash_unit(&[u64::from(token), 77]);
            for l in 0..layers {
                let mut picked: Vec<u32> = Vec::with_capacity(k);
                for i in 0..k {
                    let mut expert = if l < early {
                        let r = rng.unit();
                        let from = if r < fixed {
                            Some(u64::from(token))
                        } else if r < fixed + 0.1 {
                            previous.map(u64::from)
                        } else {
                            None
                        };
                        match from {
                            Some(id) => popularity[l].at(hash_unit(&[l as u64, id, i as u64])),
                            None => popularity[l].at(rng.unit()),
                        }
                    } else if rng.unit() < 0.8 {
                        let group = &groups[l][current];
                        group[in_group.at(rng.unit()).min(group.len() - 1)]
                    } else {
                        popularity[l].at(rng.unit())
                    };
                    while picked.contains(&expert) {
                        expert = popularity[l].at(rng.unit());
                    }
                    picked.push(expert);
                }
                routed.extend_from_slice(&picked);
            }
            tokens.push(token);
            previous = Some(token);
        }
        Self { shape, tokens, routed, popularity }
    }

    /// The number of passes (tokens) in the workload.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// The experts the router of `layer` picked at pass `pass`.
    pub fn routed(&self, pass: usize, layer: usize) -> &[u32] {
        let k = self.shape.k;
        &self.routed[(pass * self.shape.layers + layer) * k..][..k]
    }

    /// The token before the one fed at `pass`.
    pub fn previous(&self, pass: usize) -> Option<u32> {
        pass.checked_sub(1).map(|p| self.tokens[p])
    }

    /// An expert of `layer` drawn by its popularity outside any topic: what
    /// a wrong guess of a predictor looks like (plausible, but not it).
    pub fn plausible_expert(&self, layer: usize, rng: &mut Rng) -> u32 {
        self.popularity[layer].at(rng.unit())
    }
}
