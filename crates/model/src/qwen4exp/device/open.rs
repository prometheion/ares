//! The stages every way of opening the model's parts goes through: the
//! runtime's device threads, each opening its own part and meeting the others
//! between the stages (`runtime/parts.rs`), the hand-driven [`Session`], which
//! opens every part on one thread, and the trial of the second device's role
//! (`roles.rs`), which opens the parts without a store.
//!
//! In order: a part with its token rows ([`open_part`]); what its device has
//! left for expert slots ([`slot_room`]); the plan of the store
//! ([`store_config`]); the slots and the store (`feed.rs`); the feed with its
//! pruning mask and owners ([`open_feed`]); where the experts are; and, for two
//! devices on every layer, each part's peer ([`connect`]).
//!
//! [`Session`]: super::Session

use std::sync::Arc;
use std::time::Duration;

use ares_core::Ledger;
use ares_cuda::{Device, DevicePtr, ENTRY_CELLS, MAX_BATCH};
use ares_io::Reader;
use ares_runtime::ModelError;
use ares_runtime::expert_store::{CudaSlots, ExpertStore, Serve, StoreConfig};

use super::board::Board;
use super::feed::{Feed, Layout, plan_transit};
use super::invalid;
use super::load::Role;
use super::options::ModelOptions;
use super::part::{Options, Part, PartSetup, Ports};
use super::placement::{MARGIN, PartPlan, Placement};
use super::prune::{self, PruneMask};
use crate::blocks::rows::HostTable;
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

/// Opens the part of device `index` (`device`, booked in `ledger`) as
/// `placement` places it, with the token rows it stages a step's rows from.
#[allow(clippy::too_many_arguments)]
pub(super) fn open_part(
    c: &Config,
    w: &Weights<'_>,
    opts: &Options,
    placement: &Placement,
    index: usize,
    device: &Device,
    ledger: &mut Ledger,
    rows: &Arc<HostTable>,
) -> Result<Part, ModelError> {
    let PartPlan { role, first, shares, head } = placement.part(c, index);
    let setup = PartSetup { index, role, first, shares, head, device, ledger };
    let mut part = Part::open(c, w, opts, setup)?;
    part.set_token_rows(rows.clone());
    Ok(part)
}

/// What a device whose ledger has `free` bytes left has for expert slots: that,
/// less what the store keeps besides its slots (its tables and the host
/// shares the CPU computes through) and the margin the placement counts, and
/// at most `opts.slot_bytes`.
pub(super) fn slot_room(free: u64, layout: &Layout, c: &Config, opts: &ModelOptions) -> u64 {
    let layers = layout.specs.len() as u64;
    let tables = 4 * layers * c.experts.count as u64;
    let shares = match opts.serve {
        Serve::Stream => 0,
        _ => {
            let rows = opts.part.max_rows.min(MAX_BATCH);
            layers * layout.shapes[0].host_share_bytes(rows).next_multiple_of(16)
        }
    };
    free.saturating_sub(MARGIN + tables + shares).min(opts.slot_bytes.unwrap_or(u64::MAX))
}

/// The plan of the store for `vram` bytes of slots on each device and `ram`
/// bytes of the tier in host memory: the rows of a decode step, the experts a
/// row picks, the free slots kept per layer, whether the CPU computes what no
/// slot holds, and, for parts that run wide steps, the transit slots and the
/// staging that `room` (each device's [`slot_room`]) has space for.
pub(super) fn store_config(
    c: &Config,
    opts: &ModelOptions,
    layout: &Layout,
    reader: &Reader,
    vram: Vec<u64>,
    ram: u64,
    room: &[u64],
) -> StoreConfig {
    let rows = opts.part.max_rows.min(MAX_BATCH);
    let mut config = StoreConfig::new(rows, c.experts.used as u32, vram, ram);
    if let Some(reserve) = opts.reserve {
        config.vram_reserve = reserve;
    }
    config.cpu_serves = opts.serve != Serve::Stream;
    if opts.part.max_rows > MAX_BATCH {
        plan_transit(&mut config, layout, reader, opts.transit, room);
    }
    config
}

/// The feed of `store` for the parts whose route records are on `boards`:
/// reading ahead as `opts` asks, with the routers' pruning mask (made on
/// `device`, which the parts read it from) when the parts prune, and, for two
/// devices on every layer, which of them owns each expert.
#[allow(clippy::too_many_arguments)]
pub(super) fn open_feed(
    store: ExpertStore<CudaSlots>,
    boards: Vec<Arc<Board>>,
    placement: &Placement,
    layout: &Layout,
    c: &Config,
    opts: &ModelOptions,
    device: &Device,
) -> Result<(Feed, Option<Arc<PruneMask>>), ModelError> {
    let firsts = (0..placement.n_devices()).map(|d| placement.part(c, d).first).collect();
    let mut feed = Feed::new(store, boards, firsts, layout.clone(), c.experts.used as u32);
    feed.set_read_ahead(opts.read_ahead);
    let mask = match opts.part.prune {
        Some(_) => {
            let mask = Arc::new(PruneMask::new(device, c.n_layer, c.experts.count)?);
            feed.set_prune(mask.clone(), prune::owners(placement, c), c.experts.count);
            Some(mask)
        }
        None => None,
    };
    if placement.tensor.is_some() {
        feed.set_owners(placement.owners(c));
    }
    Ok((feed, mask))
}

/// What a part of two that share every layer tells the other: where its peer
/// writes to it, which of its layers it computes the recurrent or attention
/// half of, and where it takes the input of a half block in a step on VRAM
/// the expert slots lend.
#[derive(Clone, Debug, Default)]
pub(super) struct Peer {
    pub(super) ports: Option<Ports>,
    pub(super) cores: Vec<bool>,
    pub(super) lent: Option<DevicePtr>,
}

impl Peer {
    /// What `part` tells its peer, before any loan.
    pub(super) fn of(part: &Part) -> Self {
        Self {
            ports: part.ports(),
            cores: (0..part.n_layers()).map(|i| part.has_core(i)).collect(),
            lent: None,
        }
    }
}

/// Part `index` of `placement` learns its peer, when two devices share every
/// layer: where it writes and what the peer computes, and the mixer which
/// experts are the worker's ([`Placement::others`]).
pub(super) fn connect(
    part: &mut Part,
    placement: &Placement,
    c: &Config,
    index: usize,
    peer: Peer,
) -> Result<(), ModelError> {
    let Some(tensor) = &placement.tensor else { return Ok(()) };
    let ports = peer.ports.ok_or_else(|| invalid("a part that shares layers has no ports"))?;
    part.set_peer(ports, peer.cores, peer.lent)?;
    if index == tensor.mixer {
        part.set_owner(&placement.others(c, tensor.mixer))?;
    }
    Ok(())
}

/// [`connect`] for the parts of every device, opened on one thread.
pub(super) fn connect_all(
    parts: &mut [Part],
    placement: &Placement,
    c: &Config,
) -> Result<(), ModelError> {
    if placement.tensor.is_none() {
        return Ok(());
    }
    let peers: Vec<Peer> = parts.iter().map(Peer::of).collect();
    for (index, part) in parts.iter_mut().enumerate() {
        connect(part, placement, c, index, peers[1 - index].clone())?;
    }
    Ok(())
}

/// The memory of one part, for reports.
#[derive(Clone, Debug)]
pub struct PartReport {
    /// The part's index in the placement.
    pub index: usize,
    /// The device's name.
    pub device: String,
    /// What the part is to the residual streams.
    pub role: Role,
    /// The layers on it.
    pub layers: std::ops::Range<usize>,
    /// Bytes of dense weights.
    pub weights: u64,
    /// Bytes of expert slots, and how many slots that is.
    pub slots: u64,
    pub n_slots: usize,
    /// Bytes of recurrent state per sequence.
    pub state_per_slot: u64,
    /// Bytes of K, V and pooled keys per 1000 cells, and of the keys alone.
    pub kv_per_1000: u64,
    pub keys_per_1000: u64,
    /// Bytes of scratch, route records and step inputs.
    pub scratch: u64,
    /// Everything reserved for the model on the device.
    pub reserved: u64,
    /// How long the upload of the dense weights took.
    pub upload: Duration,
}

impl PartReport {
    /// The report of `part`, the `index`-th, with `n_slots` expert slots of
    /// `slots` bytes and `reserved` bytes booked on its device in all.
    pub(super) fn of(part: &Part, index: usize, slots: u64, n_slots: usize, reserved: u64) -> Self {
        let per_1000 = |bytes: u64| bytes * 1000 / u64::from(ENTRY_CELLS);
        Self {
            index,
            device: part.device().info().name.clone(),
            role: part.role(),
            layers: part.layer_range(),
            weights: part.weight_bytes(),
            slots,
            n_slots,
            state_per_slot: part.state_bytes_per_slot(),
            kv_per_1000: per_1000(part.page_bytes()),
            keys_per_1000: per_1000(part.key_page_bytes()),
            scratch: part.scratch_bytes(),
            reserved,
            upload: part.upload_time(),
        }
    }
}
