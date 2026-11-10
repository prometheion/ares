//! Where the weights of a model go: which device holds which part of which
//! block. Pure bookkeeping, no device is touched here.
//!
//! - **Single**: one device holds everything.
//! - **Tensor split**: every device holds a slice of every block and computes
//!   a partial result of the block's two outputs, which are summed across the
//!   devices. A slice is a narrower block of the same architecture:
//!   - of a recurrent block, some key heads with all value heads that read
//!     them (value head `h` reads key head `h mod n_k_heads`), in units of
//!     two key heads, so that the columns of the output projection a device
//!     holds are whole blocks of 256 values;
//!   - of an attention block, some KV heads with their query heads;
//!   - of the feed-forward part, some of its inner rows, in units of 256.
//!
//!   The share of a device is a parameter. A matrix cannot follow it exactly
//!   (an attention block of four KV heads has five possible cuts), so the
//!   rounding error of one block is carried into the next and the shares hold
//!   over the model.
//! - **Layer split**: each device holds a range of whole blocks, the first one
//!   the embedding table and the last one the output head.

use std::ops::Range;

use ares_runtime::Split;

use crate::qwen35::Config;
use crate::qwen35::weights::{Matrix, Mixer, Weights};

/// Inner rows of the feed-forward part per unit of a tensor split: the block
/// of the quantized types, so that the columns of the down projection a device
/// holds are whole blocks.
pub const FFN_UNIT: u32 = 256;

/// What one device holds of one block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockSlice {
    /// The key heads of a recurrent block or the KV heads of an attention
    /// block that this device computes. May be empty.
    pub heads: Range<u32>,
    /// The inner rows of the feed-forward part that this device computes.
    pub ffn: Range<u32>,
}

/// What one device holds of the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePlan {
    /// Per block of the model its slice here, or `None` when the block is on
    /// another device.
    pub blocks: Vec<Option<BlockSlice>>,
    /// Whether the device holds the token embedding table.
    pub embedding: bool,
    /// The tokens this device computes logits for (rows of the output head).
    pub head: Range<u32>,
}

/// Why a placement cannot be built.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("placement: {0}")]
pub struct PlacementError(pub String);

/// How a model is spread over devices.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// The kind of split.
    pub split: Split,
    /// One plan per device.
    pub devices: Vec<DevicePlan>,
}

/// The cut points of `units` units over devices with the cumulative shares
/// `cumulative` (the last one is 1), with the rounding error of earlier calls
/// carried in `carry` (one per inner boundary).
fn cuts(units: u32, cumulative: &[f64], carry: &mut [f64]) -> Vec<u32> {
    let mut out = vec![0u32];
    for (j, c) in cumulative.iter().enumerate() {
        let cut = if j + 1 == cumulative.len() {
            units
        } else {
            let ideal = c * f64::from(units);
            let previous = *out.last().unwrap_or(&0);
            let cut = ((ideal + carry[j]).round().max(0.0) as u32).clamp(previous, units);
            carry[j] += ideal - f64::from(cut);
            cut
        };
        out.push(cut);
    }
    out
}

fn check_shares(shares: &[f64]) -> Result<Vec<f64>, PlacementError> {
    if shares.is_empty() || shares.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return Err(PlacementError(format!("shares {shares:?} are not usable")));
    }
    let total: f64 = shares.iter().sum();
    if total <= 0.0 {
        return Err(PlacementError("the shares add up to nothing".into()));
    }
    let mut running = 0.0;
    Ok(shares
        .iter()
        .map(|s| {
            running += s / total;
            running
        })
        .collect())
}

fn matrix_bytes(m: &Matrix<'_>) -> u64 {
    m.bytes().len() as u64
}

/// The stored bytes of block `layer`.
fn block_bytes(w: &Weights<'_>, layer: usize) -> u64 {
    let l = &w.layers[layer];
    let ffn = matrix_bytes(&l.block.ffn_gate)
        + matrix_bytes(&l.block.ffn_up)
        + matrix_bytes(&l.block.ffn_down);
    ffn + match &l.mixer {
        Mixer::Recurrent(r) => {
            matrix_bytes(&r.qkv)
                + matrix_bytes(&r.gate)
                + matrix_bytes(&r.beta)
                + matrix_bytes(&r.alpha)
                + matrix_bytes(&r.out)
        }
        Mixer::Attention(a) => {
            matrix_bytes(&a.q) + matrix_bytes(&a.k) + matrix_bytes(&a.v) + matrix_bytes(&a.out)
        }
    }
}

impl Placement {
    /// Everything on one device.
    pub fn single(c: &Config) -> Self {
        let a = &c.attention;
        let r = &c.recurrent;
        let blocks = (0..c.n_layer)
            .map(|layer| {
                let heads = if c.is_attention(layer) { a.n_head_kv } else { r.n_k_heads };
                Some(BlockSlice { heads: 0..heads as u32, ffn: 0..c.n_ff as u32 })
            })
            .collect();
        Self {
            split: Split::Single,
            devices: vec![DevicePlan { blocks, embedding: true, head: 0..c.n_vocab as u32 }],
        }
    }

    /// A tensor split: device `i` holds about `shares[i]` of every matrix
    /// (the shares are normalised to add up to 1). The embedding table is on
    /// the first device; the head is cut by tokens.
    pub fn tensor(c: &Config, shares: &[f64]) -> Result<Self, PlacementError> {
        let cumulative = check_shares(shares)?;
        let n = shares.len();
        if n == 1 {
            return Ok(Self::single(c));
        }
        let (a, r) = (&c.attention, &c.recurrent);
        if !c.n_ff.is_multiple_of(FFN_UNIT as usize) {
            return Err(PlacementError(format!(
                "a feed-forward width of {} is not a multiple of {FFN_UNIT}",
                c.n_ff
            )));
        }
        // Key heads per unit: the heads whose value columns fill a block of 256.
        let key_unit = (256 / r.head_v_dim).max(1) as u32;
        if !r.n_k_heads.is_multiple_of(key_unit as usize) || !256usize.is_multiple_of(r.head_v_dim)
        {
            return Err(PlacementError(format!(
                "{} key heads of {} values cannot be cut into whole blocks of 256",
                r.n_k_heads, r.head_v_dim
            )));
        }
        if !a.head_dim.is_multiple_of(256) {
            return Err(PlacementError(format!(
                "attention heads of {} values are not whole blocks of 256",
                a.head_dim
            )));
        }
        let mut devices: Vec<DevicePlan> = (0..n)
            .map(|i| DevicePlan { blocks: Vec::new(), embedding: i == 0, head: 0..0 })
            .collect();
        let (mut carry_r, mut carry_a, mut carry_f) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for layer in 0..c.n_layer {
            let heads: Vec<u32> = if c.is_attention(layer) {
                cuts(a.n_head_kv as u32, &cumulative, &mut carry_a)
            } else {
                cuts(r.n_k_heads as u32 / key_unit, &cumulative, &mut carry_r)
                    .into_iter()
                    .map(|u| u * key_unit)
                    .collect()
            };
            let ffn = cuts(c.n_ff as u32 / FFN_UNIT, &cumulative, &mut carry_f);
            for (i, device) in devices.iter_mut().enumerate() {
                device.blocks.push(Some(BlockSlice {
                    heads: heads[i]..heads[i + 1],
                    ffn: ffn[i] * FFN_UNIT..ffn[i + 1] * FFN_UNIT,
                }));
            }
        }
        let head = cuts(c.n_vocab as u32, &cumulative, &mut vec![0.0; n]);
        for (i, device) in devices.iter_mut().enumerate() {
            device.head = head[i]..head[i + 1];
        }
        Ok(Self { split: Split::Tensor, devices })
    }

    /// A layer split: device `i` holds a range of whole blocks with about
    /// `shares[i]` of the weights' bytes, the embedding table counted for
    /// the first device and the head for the last.
    pub fn layer(c: &Config, w: &Weights<'_>, shares: &[f64]) -> Result<Self, PlacementError> {
        let cumulative = check_shares(shares)?;
        let n = shares.len();
        if n == 1 {
            return Ok(Self::single(c));
        }
        let embedding = matrix_bytes(&w.token_embd);
        // A tied head is uploaded again on the last device.
        let head = matrix_bytes(w.head());
        let blocks: Vec<u64> = (0..c.n_layer).map(|l| block_bytes(w, l)).collect();
        let total = (embedding + head + blocks.iter().sum::<u64>()) as f64;
        let single = Self::single(c);
        let full = &single.devices[0].blocks;
        let mut devices: Vec<DevicePlan> = (0..n)
            .map(|i| DevicePlan {
                blocks: vec![None; c.n_layer],
                embedding: i == 0,
                head: if i + 1 == n { 0..c.n_vocab as u32 } else { 0..0 },
            })
            .collect();
        let mut device = 0;
        let mut used = embedding as f64;
        for (layer, bytes) in blocks.iter().enumerate() {
            // A block goes to the next device when more than half of it lies
            // past this device's share.
            while device + 1 < n && used + *bytes as f64 / 2.0 > cumulative[device] * total {
                device += 1;
            }
            devices[device].blocks[layer] = full[layer].clone();
            used += *bytes as f64;
        }
        if devices.iter().any(|d| d.blocks.iter().all(Option::is_none)) {
            return Err(PlacementError(format!(
                "the shares {shares:?} leave a device without a block"
            )));
        }
        Ok(Self { split: Split::Layer, devices })
    }

    /// The number of devices.
    pub fn n_devices(&self) -> usize {
        self.devices.len()
    }

    /// The device that holds the embedding table.
    pub fn embedding_device(&self) -> usize {
        self.devices.iter().position(|d| d.embedding).unwrap_or(0)
    }

    /// The range of blocks device `index` holds under a layer split, or all
    /// blocks otherwise.
    pub fn layers_of(&self, index: usize) -> Range<usize> {
        let blocks = &self.devices[index].blocks;
        let first = blocks.iter().position(Option::is_some).unwrap_or(0);
        let last = blocks.iter().rposition(Option::is_some).map_or(0, |l| l + 1);
        first..last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_follow_the_share_and_carry_the_rounding() {
        // Four KV heads at 0.625: 2.5 heads, so the blocks alternate 3 and 2.
        let mut carry = vec![0.0; 2];
        let got: Vec<u32> = (0..8).map(|_| cuts(4, &[0.625, 1.0], &mut carry)[1]).collect();
        assert_eq!(got.iter().sum::<u32>(), 20);
        assert!(got.iter().all(|&h| h == 2 || h == 3), "{got:?}");
        // An exact share has no error to carry.
        let mut carry = vec![0.0; 2];
        assert_eq!(cuts(8, &[0.625, 1.0], &mut carry), vec![0, 5, 8]);
        assert_eq!(carry[0], 0.0);
    }

    #[test]
    fn three_devices_get_ordered_cuts_that_cover_everything() {
        let mut carry = vec![0.0; 3];
        for _ in 0..16 {
            let c = cuts(68, &[0.5, 0.8, 1.0], &mut carry);
            assert_eq!((c[0], c[3]), (0, 68));
            assert!(c.windows(2).all(|w| w[0] <= w[1]), "{c:?}");
        }
    }

    #[test]
    fn shares_are_normalised_and_bad_ones_refused() {
        assert_eq!(check_shares(&[5.0, 3.0]).unwrap(), vec![0.625, 1.0]);
        assert!(check_shares(&[]).is_err());
        assert!(check_shares(&[0.0, 0.0]).is_err());
        assert!(check_shares(&[1.0, f64::NAN]).is_err());
    }
}
