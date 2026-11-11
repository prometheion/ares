//! The arithmetic every backend of the sampler repeats bit for bit.
//!
//! Two rules make a device and the host agree on every bit:
//!
//! - Floating point goes through IEEE operations in round-to-nearest only
//!   (`+ - * /`, `sqrt`, conversions). There is no `exp`, `ln` or `pow` of a
//!   library: the exponential and the logarithm below are polynomials of those
//!   operations, in single precision where a pick needs one value per token
//!   and in double precision where a row needs a handful.
//! - Everything that is summed over the tokens of a row is an integer, so the
//!   order in which threads, blocks and devices add does not matter.
//!
//! A backend written in another language mirrors these functions line by line;
//! the tests of this module pin the values it must reproduce.

use std::f64::consts::{LN_2 as LN2_64, LOG2_E as LOG2_E_64};

/// A float as an unsigned number with the same order (minus infinity lowest).
pub fn sortable(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

/// The inverse of [`sortable`].
pub fn unsortable(s: u32) -> f32 {
    f32::from_bits(if s & 0x8000_0000 != 0 { s & 0x7fff_ffff } else { !s })
}

/// A value and its token as one number whose order is "higher value first,
/// lower token on a tie". Zero is below every packed pair and means "none".
pub fn pack(value: f32, token: u32) -> u64 {
    u64::from(sortable(value)) << 32 | u64::from(u32::MAX - token)
}

/// The largest key a token may have and still be picked: the largest finite
/// distance from the best logit. A logit of minus infinity has the key of plus
/// infinity, one above.
pub const KEY_MAX: u32 = 0x7f7f_ffff;

/// The key of a logit: the bits of its distance below the best logit, which
/// order like the distance itself. A smaller key is a larger probability.
pub fn key_of(l_max: f32, logit: f32) -> u32 {
    let d = l_max - logit;
    if d > 0.0 { d.to_bits() } else { 0 }
}

/// `exp(-x)` for `x >= 0` with bit pattern `key`, as a fixed-point number
/// with 40 fractional bits: 2^40 for `x = 0` and 0 from about `x = 28` on.
pub fn weight_fixed(key: u32) -> u64 {
    let d = f32::from_bits(key);
    let t = d * std::f32::consts::LOG2_E;
    // Written this way so that a NaN gives 0 too.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(t < 41.0) {
        return 0;
    }
    let n = t as u32;
    let f = t - n as f32;
    let r = f * std::f32::consts::LN_2;
    let mut p = -1.0f32 / 362_880.0;
    for c in [
        1.0f32 / 40_320.0,
        -1.0 / 5_040.0,
        1.0 / 720.0,
        -1.0 / 120.0,
        1.0 / 24.0,
        -1.0 / 6.0,
        0.5,
        -1.0,
        1.0,
    ] {
        p = p * r + c;
    }
    ((p * 1_099_511_627_776.0) as u64) >> n
}

/// `ln(x)` for a positive normal `x`, in single precision.
pub fn ln_f32(x: f32) -> f32 {
    let b = x.to_bits();
    let mut e = (b >> 23) as i32 - 127;
    let mant = b & 0x007f_ffff;
    let m = if mant > 0x0035_04f3 {
        e += 1;
        f32::from_bits(mant | 0x3f00_0000)
    } else {
        f32::from_bits(mant | 0x3f80_0000)
    };
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let mut p = 1.0f32 / 9.0;
    for c in [1.0f32 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0] {
        p = p * z + c;
    }
    let lm = (2.0 * s) * p;
    e as f32 * std::f32::consts::LN_2 + lm
}

/// Philox 4x32 with 10 rounds: the output words for a counter and a key.
pub fn philox4x32(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let [mut k0, mut k1] = key;
    for _ in 0..10 {
        let p0 = 0xD251_1F53u64 * u64::from(c0);
        let p1 = 0xCD9E_8D57u64 * u64::from(c2);
        let next = [(p1 >> 32) as u32 ^ c1 ^ k0, p1 as u32, (p0 >> 32) as u32 ^ c3 ^ k1, p0 as u32];
        [c0, c1, c2, c3] = next;
        k0 = k0.wrapping_add(0x9E37_79B9);
        k1 = k1.wrapping_add(0xBB67_AE85);
    }
    [c0, c1, c2, c3]
}

/// The random streams of one position, told apart by the counter's third
/// word. Token ids are counters of the first word, so no stream collides.
const STREAM_XTC: u32 = 1;

/// A uniform number in (0, 1) from the first output word of a generator.
fn unit(x: u32) -> f32 {
    ((x >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// The Gumbel noise of `token` at `position` for `seed`.
pub fn gumbel(seed: u64, position: u32, token: u32) -> f32 {
    let x = philox4x32([token, position, 0, 0], [seed as u32, (seed >> 32) as u32])[0];
    let e = -ln_f32(unit(x));
    -ln_f32(e)
}

/// The uniform number in (0, 1) that decides whether the XTC cut applies at
/// `position` for `seed`.
pub fn xtc_draw(seed: u64, position: u32) -> f32 {
    unit(philox4x32([0, position, STREAM_XTC, 0], [seed as u32, (seed >> 32) as u32])[0])
}

// --- double precision --------------------------------------------------------

/// `ln 2` as two doubles whose sum is closer than one: multiples of the first
/// are exact.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);

/// `exp(x)` in double precision for `x` between -700 and 700 (an argument
/// beyond is taken as the nearer limit), by range reduction to `[-ln2/2,
/// ln2/2]` and the Taylor series to the 13th power.
pub fn exp_d(x: f64) -> f64 {
    let x = x.clamp(-700.0, 700.0);
    let k = (x * LOG2_E_64 + 0.5).floor();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = 1.0 / 6_227_020_800.0;
    for c in [
        1.0 / 479_001_600.0,
        1.0 / 39_916_800.0,
        1.0 / 3_628_800.0,
        1.0 / 362_880.0,
        1.0 / 40_320.0,
        1.0 / 5_040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        p = p * r + c;
    }
    p * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

/// `ln(x)` in double precision for a positive normal `x`.
pub fn ln_d(x: f64) -> f64 {
    let b = x.to_bits();
    let mut e = ((b >> 52) & 0x7ff) as i64 - 1023;
    let mant = b & 0x000f_ffff_ffff_ffff;
    // Bring the mantissa into [sqrt(1/2), sqrt(2)).
    let m = if mant > 0x0006_a09e_667f_3bcd {
        e += 1;
        f64::from_bits(mant | 0x3fe0_0000_0000_0000)
    } else {
        f64::from_bits(mant | 0x3ff0_0000_0000_0000)
    };
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    // 2 atanh(s) = 2 s (1 + z/3 + z^2/5 + ... + z^11/23).
    let mut p = 1.0 / 23.0;
    for c in [
        1.0 / 21.0,
        1.0 / 19.0,
        1.0 / 17.0,
        1.0 / 15.0,
        1.0 / 13.0,
        1.0 / 11.0,
        1.0 / 9.0,
        1.0 / 7.0,
        1.0 / 5.0,
        1.0 / 3.0,
        1.0,
    ] {
        p = p * z + c;
    }
    let lm = (2.0 * s) * p;
    e as f64 * LN2_HI + (e as f64 * LN2_LO + lm)
}

/// `x` to the power `y` for `x >= 0` and finite `y`, with `0^y = 0` for
/// `y > 0` and `0^0 = 1`. Defined for results that stay in the range of an
/// `f32`, which is all the sampler asks of it.
pub fn pow_d(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        return if y > 0.0 {
            0.0
        } else if y == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
    }
    exp_d(y * ln_d(x))
}

/// The natural logarithm of a fixed-point sum of weights (40 fractional
/// bits), which must not be zero.
pub fn ln_fixed(sum: u64) -> f64 {
    ln_d(sum as f64) - 40.0 * LN2_64
}

// --- the statistics of a row ---------------------------------------------------

/// Distances above this count as this much in the standard deviation of
/// top-n-sigma, so that the sums of squares fit in 64 bits for any vocabulary.
pub const SIGMA_CLAMP: f32 = 4096.0;

/// The sums over the tokens of a set, all integers.
///
/// With `d` the distance of a token below the best token of the set and
/// `x = d * inv_t` the same scaled by the temperature, and `w = exp(-x)` as a
/// fixed-point number:
///
/// - `count` tokens,
/// - `z` the sum of `w`,
/// - `a` the sum of `w * x`, each term as a fixed-point number of the same
///   scale as `w` (the product is made in single precision),
/// - `q1` the sum of `d` and `q2` the sum of `d * d`, with `d` clamped to
///   [`SIGMA_CLAMP`], as fixed-point numbers with 24 and 16 fractional bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sums {
    pub count: u64,
    pub z: u64,
    pub a: u64,
    pub q1: u64,
    pub q2: u64,
}

impl Sums {
    /// The terms of one token with distance `d` below the best token of the
    /// set, at the scaled distance `x`.
    pub fn of(d: f32, x: f32) -> Sums {
        let w = weight_fixed(x.to_bits());
        let dc = d.min(SIGMA_CLAMP);
        Sums {
            count: 1,
            z: w,
            a: (w as f32 * x) as u64,
            q1: (dc * 16_777_216.0) as u64,
            q2: ((dc * dc) * 65_536.0) as u64,
        }
    }

    pub fn add(&mut self, other: &Sums) {
        self.count += other.count;
        self.z += other.z;
        self.a += other.a;
        self.q1 += other.q1;
        self.q2 += other.q2;
    }

    /// The mean of the scaled distances weighted by their probability, which
    /// is the expectation of `x`.
    pub fn mean_x(&self) -> f32 {
        if self.z == 0 { 0.0 } else { (self.a as f64 / self.z as f64) as f32 }
    }

    /// The entropy of the distribution of the set in nats, from its sums.
    pub fn entropy(&self) -> f64 {
        if self.z == 0 {
            return 0.0;
        }
        ln_fixed(self.z) + self.a as f64 / self.z as f64
    }

    /// The standard deviation of the distances `d` of the set.
    pub fn sigma(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = (self.q1 as f64 / 16_777_216.0) / n;
        let var = (self.q2 as f64 / 65_536.0) / n - mean * mean;
        (if var > 0.0 { var.sqrt() } else { 0.0 }) as f32
    }
}

/// `floor(sum * q / 2^32)` for a fraction `q` of 2^32.
pub fn share(sum: u64, q: u32) -> u64 {
    ((u128::from(sum) * u128::from(q)) >> 32) as u64
}

/// The fraction `p` (between 0 and 1) as a number of 2^32ths, at least 1.
pub fn fraction_q32(p: f32) -> u32 {
    (f64::from(p) * 4_294_967_296.0).round().clamp(1.0, 4_294_967_295.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_gives_the_known_answers() {
        // The known-answer tests of Random123 (kat_vectors, philox4x32 10).
        assert_eq!(
            philox4x32([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn the_fixed_point_weight_is_the_exponential() {
        assert_eq!(weight_fixed(0), 1 << 40);
        let mut worst = 0f64;
        let mut d = 1e-6f32;
        while d < 27.0 {
            let exact = (-f64::from(d)).exp() * (1u64 << 40) as f64;
            let got = weight_fixed(d.to_bits()) as f64;
            // The polynomial's error, beyond the unit the fixed point cuts.
            let error = ((got - exact).abs() - 1.0).max(0.0) / exact;
            worst = worst.max(error);
            d *= 1.01;
        }
        assert!(worst < 1e-6, "relative error {worst}");
        assert_eq!(weight_fixed(f32::INFINITY.to_bits()), 0);
        assert_eq!(weight_fixed(40.0f32.to_bits()), 0);
        // Never above the weight of the best token.
        assert!((0..2000).all(|i| weight_fixed((i as f32 * 1e-9).to_bits()) <= 1 << 40));
    }

    #[test]
    fn the_single_precision_logarithm_is_the_logarithm() {
        let mut x = 5e-8f32;
        while x < 20.0 {
            let exact = f64::from(x).ln();
            let got = f64::from(ln_f32(x));
            assert!((got - exact).abs() < 2e-6 * exact.abs().max(1.0), "ln({x}) = {got}, {exact}");
            x *= 1.003;
        }
    }

    #[test]
    fn the_double_precision_functions_are_accurate_to_the_last_digits() {
        let mut x = -700.0f64;
        while x < 700.0 {
            let (got, exact) = (exp_d(x), x.exp());
            assert!(((got - exact) / exact).abs() < 4e-16, "exp({x}) = {got}, {exact}");
            x += 0.37;
        }
        let mut x = 1e-300f64;
        while x < 1e300 {
            let (got, exact) = (ln_d(x), x.ln());
            assert!(
                (got - exact).abs() <= 4e-16 * exact.abs().max(1.0),
                "ln({x}) = {got}, {exact}"
            );
            x *= 1.7;
        }
        assert_eq!(ln_d(1.0), 0.0);
        assert_eq!(exp_d(0.0), 1.0);
        assert_eq!(pow_d(0.0, 2.0), 0.0);
        assert_eq!(pow_d(0.0, 0.0), 1.0);
        assert!((pow_d(0.5, 3.0) - 0.125).abs() < 1e-15);
        assert!((pow_d(2.0, 0.5) - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn the_logarithm_of_a_fixed_sum_is_the_logarithm_of_the_sum_of_weights() {
        assert_eq!(ln_fixed(1 << 40), 0.0);
        assert!((ln_fixed(3 << 40) - 3f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn the_sums_of_a_set_give_its_entropy_and_deviation() {
        // Two equally likely tokens: entropy ln 2; distances 0 and 0.
        let mut s = Sums::default();
        s.add(&Sums::of(0.0, 0.0));
        s.add(&Sums::of(0.0, 0.0));
        assert!((s.entropy() - std::f64::consts::LN_2).abs() < 1e-6, "{}", s.entropy());
        assert_eq!(s.sigma(), 0.0);
        // Distances 0 and 2: mean 1, deviation 1.
        let mut s = Sums::default();
        s.add(&Sums::of(0.0, 0.0));
        s.add(&Sums::of(2.0, 2.0));
        assert!((f64::from(s.sigma()) - 1.0).abs() < 1e-6);
        let (p0, p1) = (1.0 / (1.0 + (-2f64).exp()), (-2f64).exp() / (1.0 + (-2f64).exp()));
        let h = -(p0 * p0.ln() + p1 * p1.ln());
        assert!((s.entropy() - h).abs() < 1e-5, "{} {h}", s.entropy());
    }

    #[test]
    fn the_draw_of_xtc_is_a_stream_of_its_own() {
        let a = xtc_draw(7, 100);
        assert!(a > 0.0 && a < 1.0);
        assert_eq!(a, xtc_draw(7, 100));
        assert_ne!(a, xtc_draw(8, 100));
        assert_ne!(a, xtc_draw(7, 101));
        // Token 0's noise is not the XTC draw of the same position.
        assert_ne!(gumbel(7, 100, 0).to_bits(), a.to_bits());
    }
}
