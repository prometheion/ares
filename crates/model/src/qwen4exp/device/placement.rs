//! Which device holds what of which layer, and what is left for expert
//! slots.
//!
//! **By layers** ([`Placement::plan`], [`Placement::at`]): a device holds a
//! contiguous range of layers with everything they need (dense weights,
//! recurrent state, K/V and key pools, scratch), and the experts of a layer
//! live on the layer's device. What a device has left after that is its
//! budget for expert slots. One sequence then uses one device at a time.
//!
//! **Both devices on every layer** ([`Placement::tensor`]): one device, the
//! mixer, holds the residual streams, every stream mixer, the router and the
//! attention layers; the other, the worker, gets the mixed input of every
//! half block and returns its share of the output. They share the key heads
//! of the recurrent layers by a share that is a parameter, one of them
//! computes the shared expert, and every routed expert has an owner
//! ([`Placement::owner`]): the device that computes it and holds it in a slot
//! when it is in VRAM. No weight is on both devices.
//!
//! Who owns an expert follows the memory first: each device's slots are to
//! be full of experts it owns, because an expert outside its slot costs more
//! than an unbalanced half block. Without a recorded routing a fixed rule
//! gives each device the share of every layer's experts that its slot bytes
//! are of all slot bytes. With one ([`Placement::assign`]) the hot experts
//! are dealt out by the devices' speed, and what a step can miss goes to the
//! device that an upload reaches faster.
//!
//! The ranges of a split by layers are chosen so that the smallest share of
//! its experts that any device can hold in slots is as large as possible: a
//! layer whose experts are mostly outside VRAM makes every token wait for
//! reads, wherever the other layers are. The rule reads only what the devices
//! report (free memory, multiprocessors) and the model's sizes.

use std::ops::Range;

use ares_runtime::ModelError;

use super::load::{LayerShare, Role, head_bytes, key_head_unit, layer_bytes};
use super::part::{Frame, Options, PartShape};
use super::roles::RoleTrial;
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::{Mixer, Weights};

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
    /// The layers the device holds, or holds a share of.
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
    /// Bytes of all experts the device owns: those of its layers, or its
    /// share of every layer's.
    pub experts: u64,
}

impl DevicePlan {
    /// The share of its layers' experts the device can hold in slots.
    pub fn slot_share(&self) -> f64 {
        if self.experts == 0 { 1.0 } else { (self.slots as f64 / self.experts as f64).min(1.0) }
    }
}

/// How two devices share every layer.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorPlan {
    /// The device that holds the residual streams and runs the mixers.
    pub mixer: usize,
    /// Per device and layer, what the device holds of the layer.
    pub shares: Vec<Vec<LayerShare>>,
    /// Per device, the share of every layer's experts it owns.
    pub expert_share: Vec<f64>,
    /// Per layer and expert the device that owns it, when the owners were
    /// assigned from a recorded routing ([`Placement::assign`]); otherwise
    /// an expert's owner follows from `expert_share` alone.
    pub owners: Option<Vec<Vec<u8>>>,
    /// The choices the placement was made from (measured at load unless they
    /// were given): another placement of the same choices on the same devices
    /// lays the state out the same way.
    pub choice: TensorChoice,
    /// What the trial at load measured of the worker's role, when it chose
    /// `choice` (`roles.rs`); `None` when the choice was given.
    pub trial: Option<RoleTrial>,
}

/// The choices of a placement in which two devices work on every layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorChoice {
    /// The device that holds the residual streams and runs every mixer: the
    /// one whose reads are faster.
    pub mixer: usize,
    /// The mixer device's share of the key heads of the recurrent layers;
    /// the worker computes the rest.
    pub core_share: f64,
    /// Whether the mixer device computes the shared expert; otherwise the
    /// worker does, while the mixer routes.
    pub shared_on_mixer: bool,
}

/// What one device's part is built as.
#[derive(Clone, Debug)]
pub struct PartPlan {
    /// What the part is to the residual streams.
    pub role: Role,
    /// The first layer it holds or shares.
    pub first: usize,
    /// What it holds of each layer from `first` on.
    pub shares: Vec<LayerShare>,
    /// Whether it holds the final mixer and the output head.
    pub head: bool,
}

/// The split of a model over devices.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// Per device, in the order of the layers.
    pub devices: Vec<DevicePlan>,
    /// How the devices share the layers when both work on every one.
    pub tensor: Option<TensorPlan>,
}

/// Device memory that is set aside next to what the model counts: the
/// driver rounds every allocation up, and the expert store has tables and
/// staging of its own.
pub(super) const MARGIN: u64 = 32 << 20;

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

/// Bytes of the dense weights of every layer held whole.
fn whole_layers(c: &Config, w: &Weights<'_>) -> Result<Vec<u64>, ModelError> {
    (0..c.n_layer).map(|l| layer_bytes(c, w, l, Role::Whole, &LayerShare::whole(c, l))).collect()
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
        let weights = whole_layers(c, w)?;
        let head = head_bytes(w);
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
        Ok(Self { devices: chosen, tensor: None })
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
        let weights = whole_layers(c, w)?;
        let head = head_bytes(w);
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
        Ok(Self { devices, tensor: None })
    }

    /// Both devices on every layer, as `choice` says.
    ///
    /// # Errors
    ///
    /// Other than two devices, a share outside 0 to 1, or a device whose
    /// share does not fit.
    pub fn tensor(
        c: &Config,
        w: &Weights<'_>,
        opts: &Options,
        devices: &[DeviceMemory],
        choice: &TensorChoice,
    ) -> Result<Self, ModelError> {
        if devices.len() != 2 || choice.mixer > 1 || !(0.0..=1.0).contains(&choice.core_share) {
            return Err(invalid(format!(
                "{} devices and {choice:?}: two devices share the layers, one of them the mixer",
                devices.len()
            )));
        }
        // The mixer takes the first key heads of a recurrent layer, in units
        // that keep the columns of the output projection whole blocks; what
        // the rounding of one layer leaves is carried into the next. An
        // attention layer is the mixer's.
        let mut shares = vec![Vec::with_capacity(c.n_layer), Vec::with_capacity(c.n_layer)];
        let mut carry = 0.0;
        for (layer, weights) in w.layers.iter().enumerate() {
            let (all, cut) = match &weights.mixer {
                Mixer::Attention(_) => (c.attention.n_head_kv as u32, c.attention.n_head_kv as u32),
                Mixer::Recurrent(r) => {
                    let unit = key_head_unit(c, r) as u32;
                    let units = c.recurrent.n_k_heads as u32 / unit;
                    let ideal = choice.core_share * f64::from(units);
                    let cut = ((ideal + carry).round().max(0.0) as u32).min(units);
                    carry += ideal - f64::from(cut);
                    (c.recurrent.n_k_heads as u32, cut * unit)
                }
            };
            shares[choice.mixer].push(LayerShare { heads: 0..cut, shared: choice.shared_on_mixer });
            shares[1 - choice.mixer]
                .push(LayerShare { heads: cut..all, shared: !choice.shared_on_mixer });
            debug_assert_eq!(shares[0].len(), layer + 1);
        }
        let head = head_bytes(w);
        let all_experts: u64 =
            (0..c.n_layer).map(|l| expert_bytes(w, l) * c.experts.count as u64).sum();
        let mut plans = Vec::with_capacity(2);
        for (device, memory) in devices.iter().enumerate() {
            let role = if device == choice.mixer { Role::Mixer } else { Role::Worker };
            let shape = PartShape {
                role,
                first: 0,
                shares: shares[device].clone(),
                head: role == Role::Mixer,
            };
            let frame = Frame::new(c, opts, &shape, memory.multiprocessors)?;
            let mut dense = if shape.head { head } else { 0 };
            for (layer, share) in shape.shares.iter().enumerate() {
                dense += layer_bytes(c, w, layer, role, share)?;
            }
            let state = frame.state_bytes_per_slot() * u64::from(opts.max_slots);
            let step = frame.step_bytes(opts);
            let fixed = dense + state + frame.pool_bytes() + step + MARGIN;
            let left = memory.budget.checked_sub(fixed).ok_or_else(|| {
                invalid(format!(
                    "the {role:?}'s share ({} MiB of weights, and state, K/V for {} cells and \
                     scratch) does not fit in {} MiB",
                    dense >> 20,
                    opts.cells,
                    memory.budget >> 20
                ))
            })?;
            plans.push(DevicePlan {
                layers: 0..c.n_layer,
                head: shape.head,
                weights: dense,
                state,
                pools: frame.pool_bytes(),
                step,
                slots: left,
                experts: 0,
            });
        }
        // A device owns the share of the experts that its slots are of all
        // slots; with more room than experts, every expert has a slot.
        let room: u64 = plans.iter().map(|p| p.slots).sum();
        let mut expert_share = Vec::with_capacity(2);
        for plan in &mut plans {
            let share = if room == 0 { 0.5 } else { plan.slots as f64 / room as f64 };
            plan.experts = (share * all_experts as f64) as u64;
            // A layer's owners are not cut exactly by the share, so a device
            // with room may hold more of a layer than its share of it.
            plan.slots = plan.slots.min(all_experts + (all_experts >> 6));
            expert_share.push(share);
        }
        Ok(Self {
            devices: plans,
            tensor: Some(TensorPlan {
                mixer: choice.mixer,
                shares,
                expert_share,
                owners: None,
                choice: *choice,
                trial: None,
            }),
        })
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
        let shape = PartShape::whole(c, layers.clone(), head);
        let frame = Frame::new(c, opts, &shape, memory.multiprocessors)?;
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

    /// Gives every expert its owner from a recorded routing, for two devices
    /// that share the layers: `counts[layer][expert]` says how often the
    /// expert was picked, `slots[device][layer]` how many experts of the
    /// layer the device can hold, `picks[device]` is the share of the picks
    /// the device's experts are to get (its speed's share of both), and
    /// `tail` is the device on which an expert outside its slot costs less.
    ///
    /// The experts that fit the two devices' slots, the most used, are dealt
    /// out hottest first, each to the device that is further behind its
    /// share of the picks, while that device has slots. So the faster device
    /// gets more of the hot experts and both fill their slots. Everything
    /// else, which is what a step can miss, is the `tail` device's: the other
    /// device owns no more than it holds and never waits for an expert.
    ///
    /// The owners are fixed from then on. A routing that drifts away from
    /// the recorded one leaves the device without the tail with experts that
    /// are no longer the hot ones; owners that follow the heat are not built.
    pub fn assign(
        &mut self,
        counts: &[Vec<u32>],
        slots: &[Vec<usize>],
        picks: [f64; 2],
        tail: usize,
    ) {
        let Some(tensor) = &mut self.tensor else { return };
        let (tail, held) = (tail.min(1), 1 - tail.min(1));
        let mut owners = Vec::with_capacity(counts.len());
        for (layer, counts) in counts.iter().enumerate() {
            let mut order: Vec<usize> = (0..counts.len()).collect();
            order.sort_by_key(|e| (std::cmp::Reverse(counts[*e]), *e));
            let room = |device: usize| slots.get(device).and_then(|s| s.get(layer)).copied();
            let (room_held, room_tail) = (room(held).unwrap_or(0), room(tail).unwrap_or(0));
            let mut owner = vec![tail as u8; counts.len()];
            let (mut heat_held, mut heat_tail) = (0.0f64, 0.0f64);
            let (mut n_held, mut n_tail) = (0usize, 0usize);
            for expert in order.into_iter().take(room_held + room_tail) {
                let heat = f64::from(counts[expert]);
                let behind = heat_held * picks[tail] <= heat_tail * picks[held];
                if n_held < room_held && (n_tail >= room_tail || behind) {
                    owner[expert] = held as u8;
                    heat_held += heat;
                    n_held += 1;
                } else {
                    heat_tail += heat;
                    n_tail += 1;
                }
            }
            owners.push(owner);
        }
        tensor.owners = Some(owners);
    }

    /// What device `index` is built as: its role, the layers it holds or
    /// shares and what it holds of each.
    pub fn part(&self, c: &Config, index: usize) -> PartPlan {
        let plan = &self.devices[index];
        match &self.tensor {
            Some(tensor) => PartPlan {
                role: if index == tensor.mixer { Role::Mixer } else { Role::Worker },
                first: plan.layers.start,
                shares: tensor.shares[index].clone(),
                head: plan.head,
            },
            None => PartPlan {
                role: Role::Whole,
                first: plan.layers.start,
                shares: plan.layers.clone().map(|layer| LayerShare::whole(c, layer)).collect(),
                head: plan.head,
            },
        }
    }

    /// The number of devices.
    pub fn n_devices(&self) -> usize {
        self.devices.len()
    }

    /// The device that holds `layer` when the model is split by layers.
    pub fn device_of(&self, layer: usize) -> Option<usize> {
        self.devices.iter().position(|d| d.layers.contains(&layer))
    }

    /// The device that owns expert `expert` of `layer`: it computes the
    /// expert, and holds it in a slot when the expert is in VRAM.
    pub fn owner(&self, layer: usize, expert: u32) -> usize {
        let Some(tensor) = &self.tensor else {
            return self.device_of(layer).unwrap_or(0);
        };
        if let Some(owner) =
            tensor.owners.as_ref().and_then(|o| o.get(layer)?.get(expert as usize).copied())
        {
            return usize::from(owner);
        }
        // A point per expert: the multiples of the golden ratio spread
        // evenly over [0, 1) whatever the numbers of the experts a text
        // uses, shifted from layer to layer.
        let x = (u64::from(expert) + 1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((layer as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let point = (x >> 11) as f64 / (1u64 << 53) as f64;
        let mut edge = 0.0;
        for (device, share) in tensor.expert_share.iter().enumerate() {
            edge += share;
            if point < edge {
                return device;
            }
        }
        tensor.expert_share.len() - 1
    }
}

/// The bytes of page-locked RAM for the experts' tier in host memory: `want`
/// when given, otherwise half of what the system reports available (a tool's
/// guess: the engine follows the machine instead), and never more than
/// `wanted`, what the experts that VRAM has no slot for take.
pub fn ram_tier_bytes(want: Option<u64>, wanted: u64) -> u64 {
    use ares_runtime::MemorySource;
    let available = || ares_runtime::Meminfo.available();
    want.unwrap_or_else(|| available().map_or(1 << 30, |bytes| bytes / 2)).min(wanted)
}
