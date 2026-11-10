//! Which device holds which layers, and what is left for expert slots.
//!
//! The model is split by layers: a device holds a contiguous range of them
//! with everything they need (dense weights, recurrent state, K/V and key
//! pools, scratch), and the experts of a layer live on the layer's device.
//! What a device has left after that is its budget for expert slots.
//!
//! The ranges are chosen so that the smallest share of its experts that any
//! device can hold in slots is as large as possible: a layer whose experts
//! are mostly outside VRAM makes every token wait for reads, wherever the
//! other layers are. The rule reads only what the devices report (free
//! memory, multiprocessors) and the model's sizes.

use std::ops::Range;

use ares_runtime::ModelError;

use super::load::{head_bytes, layer_bytes};
use super::part::{Frame, Options};
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

/// What a device offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Bytes the model may take: what the device reports free, less what is
    /// to stay free.
    pub budget: u64,
    /// The device's multiprocessors: the attention plans, and with them the
    /// scratch, are cut by it.
    pub multiprocessors: u32,
}

/// One device's share of the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePlan {
    /// The layers the device holds.
    pub layers: Range<usize>,
    /// Whether it holds the final mixer and the output head.
    pub head: bool,
    /// Bytes of dense weights.
    pub weights: u64,
    /// Bytes of recurrent state, for all sequence slots.
    pub state: u64,
    /// Bytes of the K/V pools and key pools of its attention layers.
    pub pools: u64,
    /// Bytes of scratch, kept window inputs, route records, step inputs and
    /// tables.
    pub step: u64,
    /// Bytes left for expert slots.
    pub slots: u64,
    /// Bytes of all experts of the device's layers.
    pub experts: u64,
}

impl DevicePlan {
    /// The share of its layers' experts the device can hold in slots.
    pub fn slot_share(&self) -> f64 {
        if self.experts == 0 { 1.0 } else { (self.slots as f64 / self.experts as f64).min(1.0) }
    }
}

/// The split of a model over devices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Per device, in the order of the layers.
    pub devices: Vec<DevicePlan>,
}

/// Device memory that is set aside next to what the model counts: the
/// driver rounds every allocation up, and the expert store has tables and
/// staging of its own.
const MARGIN: u64 = 32 << 20;

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

/// Bytes of one routed expert of `layer` in the file.
fn expert_bytes(w: &Weights<'_>, layer: usize) -> u64 {
    let moe = &w.layers[layer].moe;
    (moe.gate.expert_bytes() + moe.up.expert_bytes() + moe.down.expert_bytes()) as u64
}

impl Placement {
    /// The split of the model over `devices` that leaves the most room for
    /// expert slots where it is scarcest.
    ///
    /// # Errors
    ///
    /// When the dense parts do not fit: the model needs more devices or a
    /// smaller context.
    pub fn plan(
        c: &Config,
        w: &Weights<'_>,
        opts: &Options,
        devices: &[DeviceMemory],
    ) -> Result<Self, ModelError> {
        let (n_layer, n_dev) = (c.n_layer, devices.len());
        if n_dev == 0 || n_dev > n_layer {
            return Err(invalid(format!("{n_dev} devices for {n_layer} layers")));
        }
        let weights: Vec<u64> =
            (0..n_layer).map(|l| layer_bytes(c, w, l)).collect::<Result<_, _>>()?;
        let head = head_bytes(w)?;
        // best[d][b]: the best worst share when devices 0..=d hold the layers
        // 0..b, and the first layer of device d in that split.
        let mut best = vec![vec![(f64::NEG_INFINITY, 0usize); n_layer + 1]; n_dev];
        let mut plans = vec![vec![None; n_layer + 1]; n_dev];
        for d in 0..n_dev {
            // The devices after this one need a layer each.
            for b in d + 1..=n_layer - (n_dev - 1 - d) {
                let last = d == n_dev - 1;
                if last && b != n_layer {
                    continue;
                }
                let starts = if d == 0 { 0..1 } else { d..b };
                for a in starts {
                    let before = if d == 0 { f64::INFINITY } else { best[d - 1][a].0 };
                    if before == f64::NEG_INFINITY {
                        continue;
                    }
                    let Some(plan) = Self::device(c, w, opts, &devices[d], &weights, head, a..b)?
                    else {
                        continue;
                    };
                    let worst = before.min(plan.slot_share());
                    if worst > best[d][b].0 {
                        best[d][b] = (worst, a);
                        plans[d][b] = Some(plan);
                    }
                }
            }
        }
        if best[n_dev - 1][n_layer].0 == f64::NEG_INFINITY {
            return Err(invalid(format!(
                "the dense parts of the model ({} MiB of weights, and state, K/V for {} cells and \
                 scratch) do not fit on {n_dev} device(s) with {} MiB to give",
                (weights.iter().sum::<u64>() + head) >> 20,
                opts.cells,
                devices.iter().map(|d| d.budget).sum::<u64>() >> 20,
            )));
        }
        let mut chosen = Vec::with_capacity(n_dev);
        let mut end = n_layer;
        for d in (0..n_dev).rev() {
            let plan = plans[d][end].take().ok_or_else(|| invalid("no placement was kept"))?;
            end = plan.layers.start;
            chosen.push(plan);
        }
        chosen.reverse();
        Ok(Self { devices: chosen })
    }

    /// A split at given layers: device `d` holds `bounds[d]..bounds[d + 1]`
    /// (`bounds` starts at 0 and ends at the number of layers).
    ///
    /// # Errors
    ///
    /// Bounds that are not such a list, or a device whose share does not fit.
    pub fn at(
        c: &Config,
        w: &Weights<'_>,
        opts: &Options,
        devices: &[DeviceMemory],
        bounds: &[usize],
    ) -> Result<Self, ModelError> {
        let ordered = bounds.windows(2).all(|p| p[0] < p[1]);
        if bounds.len() != devices.len() + 1
            || bounds.first() != Some(&0)
            || bounds.last() != Some(&c.n_layer)
            || !ordered
        {
            return Err(invalid(format!(
                "layer bounds {bounds:?} for {} devices and {} layers",
                devices.len(),
                c.n_layer
            )));
        }
        let weights: Vec<u64> =
            (0..c.n_layer).map(|l| layer_bytes(c, w, l)).collect::<Result<_, _>>()?;
        let head = head_bytes(w)?;
        let devices = devices
            .iter()
            .zip(bounds.windows(2))
            .map(|(memory, pair)| {
                let layers = pair[0]..pair[1];
                Self::device(c, w, opts, memory, &weights, head, layers.clone())?.ok_or_else(|| {
                    invalid(format!(
                        "the layers {layers:?} do not fit in {} MiB",
                        memory.budget >> 20
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { devices })
    }

    /// The share of one device, or `None` when it does not fit there.
    fn device(
        c: &Config,
        w: &Weights<'_>,
        opts: &Options,
        memory: &DeviceMemory,
        weights: &[u64],
        head_bytes: u64,
        layers: Range<usize>,
    ) -> Result<Option<DevicePlan>, ModelError> {
        let head = layers.end == c.n_layer;
        let frame = Frame::new(c, opts, layers.clone(), head, memory.multiprocessors)?;
        let dense = weights[layers.clone()].iter().sum::<u64>() + if head { head_bytes } else { 0 };
        let state = frame.state_bytes_per_slot() * u64::from(opts.max_slots);
        let step = frame.step_bytes(opts);
        let fixed = dense + state + frame.pool_bytes() + step + MARGIN;
        let experts: u64 =
            layers.clone().map(|l| expert_bytes(w, l) * c.experts.count as u64).sum();
        // The fewest slots a step can need of every layer must fit as well;
        // the store checks that when it is planned.
        Ok(memory.budget.checked_sub(fixed).map(|left| DevicePlan {
            layers,
            head,
            weights: dense,
            state,
            pools: frame.pool_bytes(),
            step,
            slots: left.min(experts + (experts >> 6)),
            experts,
        }))
    }

    /// The number of devices.
    pub fn n_devices(&self) -> usize {
        self.devices.len()
    }

    /// The device that holds `layer`.
    pub fn device_of(&self, layer: usize) -> Option<usize> {
        self.devices.iter().position(|d| d.layers.contains(&layer))
    }
}

/// The bytes of page-locked RAM for the experts' tier in host memory: `want`
/// when given, otherwise a third of what the system reports available, and
/// never more than `limit`.
pub fn ram_tier_bytes(want: Option<u64>, limit: u64) -> u64 {
    let available = || {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    };
    want.unwrap_or_else(|| available().map_or(1 << 30, |bytes| bytes / 3)).min(limit)
}
