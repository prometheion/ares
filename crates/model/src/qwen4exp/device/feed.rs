//! The expert store as the parts of a model use it: how it is laid out for a
//! placement, how the slots of a device are made, and how the route records
//! the parts' routers leave are handed to it.
//!
//! The store is owned by one thread at a time. [`Session`](super::Session)
//! drives it from its own thread; the runtime's device threads of
//! [`Qwen4Exp`](super::Qwen4Exp) share it behind a lock that none of them
//! waits for in a step.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_core::Ledger;
use ares_cuda::{Device, ExpertShape, RouteGroup, RouteRecord};
use ares_gguf::Gguf;
use ares_io::{ExpertMap, Reader};
use ares_runtime::expert_store::cpu::{CpuPool, Placement as CpuPlacement, Topology};
use ares_runtime::expert_store::{CudaSlots, ExpertStore, HostRegion, LayerSpec, Serve, StorePlan};

use super::Error;
use super::board::Board;
use super::part::ExpertAt;
use super::placement::Placement;
use super::session::Fill;
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Where the experts' tensors are in the file, for a store with `per_layer`
/// entries for each layer of the model.
pub(super) fn expert_map(gguf: &Gguf, c: &Config, per_layer: usize) -> Result<ExpertMap, Error> {
    let patterns = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    Ok(if per_layer > 1 {
        let names: Vec<Vec<String>> = (0..c.n_layer * per_layer)
            .map(|at| {
                patterns.iter().map(|p| format!("blk.{}.{p}.weight", at / per_layer)).collect()
            })
            .collect();
        ExpertMap::new(gguf, &names)?
    } else {
        ExpertMap::discover(gguf, &patterns)?
    })
}

/// How the experts of a model are laid out for the store: where each layer's
/// matrices are in the file and in a slot, and which device runs which layer.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    /// One entry per layer of the store: a layer once per device that owns
    /// experts of it.
    pub specs: Vec<LayerSpec>,
    pub shapes: Vec<ExpertShape>,
    /// Entries of the store per layer of the model: 1, or the number of
    /// devices when they share every layer.
    pub per_layer: usize,
    /// The bytes of the experts a device would hold if it held its share of
    /// them all.
    pub expert_bytes: u64,
}

impl Layout {
    /// The layout of the experts of `placement`, and where their tensors are
    /// in the file.
    pub(super) fn new(
        gguf: &Gguf,
        c: &Config,
        w: &Weights<'_>,
        placement: &Placement,
    ) -> Result<(Self, ExpertMap), Error> {
        let per_layer = if placement.tensor.is_some() { placement.devices.len() } else { 1 };
        let store_layers = c.n_layer * per_layer;
        let map = expert_map(gguf, c, per_layer)?;
        let mut specs = Vec::with_capacity(store_layers);
        let mut shapes = Vec::with_capacity(store_layers);
        let mut expert_bytes = 0u64;
        for at in 0..store_layers {
            let layer = at / per_layer;
            let moe = &w.layers[layer].moe;
            let shape = ExpertShape {
                n_embd: c.n_embd as u32,
                n_ff: c.experts.ff as u32,
                n_experts: c.experts.count as u32,
                top_k: c.experts.used as u32,
                gate_ty: moe.gate.ty(),
                down_ty: moe.down.ty(),
            };
            let geometry = || invalid(format!("layer {layer}: the experts do not fit a slot"));
            let slot_bytes = shape.slot_bytes().ok_or_else(geometry)?;
            specs.push(LayerSpec {
                device: if per_layer > 1 {
                    at % per_layer
                } else {
                    placement
                        .device_of(layer)
                        .ok_or_else(|| invalid(format!("layer {layer} is on no device")))?
                },
                offsets: [
                    0,
                    shape.up_offset().ok_or_else(geometry)?,
                    shape.down_offset().ok_or_else(geometry)?,
                ],
                slot_bytes,
            });
            expert_bytes += slot_bytes * c.experts.count as u64 / per_layer as u64;
            shapes.push(shape);
        }
        Ok((Self { specs, shapes, per_layer, expert_bytes }, map))
    }

    /// The most bytes a tier in host memory can use: one slot per expert that
    /// has none in VRAM is all it ever holds, and a slot is the expert and a
    /// few blocks of the file.
    pub(super) fn ram_wanted(&self, vram_bytes: u64) -> u64 {
        (self.expert_bytes + (self.expert_bytes >> 4)).saturating_sub(vram_bytes).max(64 << 20)
    }

    /// The store layer of layer `layer` on device `part`.
    pub(super) fn store_layer(&self, part: usize, layer: usize) -> usize {
        if self.per_layer > 1 { layer * self.per_layer + part } else { layer }
    }
}

/// The slots of one device: its pools and tables, and the host shares that
/// let the CPU compute what a layer misses, reserved in the device's ledger
/// and in `pinned`.
pub(super) fn device_slots(
    device: &Device,
    ledger: &mut Ledger,
    pinned: &mut Ledger,
    pools: &[(u64, u32)],
    layout: &Layout,
    serve: Serve,
    rows: u32,
) -> Result<CudaSlots, Error> {
    let mut slots = CudaSlots::new(
        device,
        ledger,
        pinned,
        pools,
        layout.shapes.len(),
        layout.shapes[0].n_experts,
    )?;
    if serve != Serve::Stream {
        slots.add_host_shares(ledger, pinned, layout.shapes[0].host_share_bytes(rows))?;
    }
    Ok(slots)
}

/// Where the experts of the layers of `part` are on its device.
pub(super) fn places(
    store: &ExpertStore<CudaSlots>,
    layers: std::ops::Range<usize>,
    layout: &Layout,
    index: usize,
) -> Vec<ExpertAt> {
    layers
        .map(|layer| {
            let at = layout.store_layer(index, layer);
            let (device, pool) = store.place(at);
            debug_assert_eq!(device, index);
            let slots = store.device(device);
            ExpertAt {
                table: slots.table(at),
                pool: slots.pool(pool).0,
                host: slots.host_share(at),
            }
        })
        .collect()
}

/// The CPU workers of the store: all the cores the system reports but
/// `reserved`, which belong to the threads that drive the step.
pub(super) fn cpu_pool(workers: Option<usize>, reserved: usize) -> Result<CpuPool, Error> {
    let placement = match workers {
        Some(workers) => CpuPlacement::unpinned(workers.max(1)),
        None => CpuPlacement::for_machine(&Topology::read(), reserved),
    };
    CpuPool::new(&placement, Duration::from_millis(20))
        .map_err(|e| invalid(format!("the CPU workers of the expert store: {e}")))
}

/// The expert store and the route records that feed it.
pub(super) struct Feed {
    pub store: ExpertStore<CudaSlots>,
    boards: Vec<Arc<Board>>,
    /// The first layer of every part.
    firsts: Vec<usize>,
    layout: Layout,
    used: u32,
    rows: u32,
    /// Per part: the layers of the running step whose route records were
    /// handed to the store.
    followed: Vec<usize>,
    groups: Vec<RouteGroup>,
}

impl Feed {
    pub(super) fn new(
        store: ExpertStore<CudaSlots>,
        boards: Vec<Arc<Board>>,
        firsts: Vec<usize>,
        layout: Layout,
        used: u32,
    ) -> Self {
        Self {
            followed: vec![0; boards.len()],
            store,
            boards,
            firsts,
            layout,
            used,
            rows: 0,
            groups: Vec::new(),
        }
    }

    pub(super) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The layers of every part whose route records the store has.
    pub(super) fn followed(&self) -> &[usize] {
        &self.followed
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers, most wanted first, as far as the slots go: from the file into
    /// VRAM, and into RAM what VRAM has no room for. `owner` says which device
    /// owns an expert when the devices share the layers. Returns when nothing
    /// moves any more; no step may run meanwhile.
    pub(super) fn fill(
        &mut self,
        n_layer: usize,
        n_experts: usize,
        owner: &dyn Fn(usize, u32) -> usize,
        experts_of: &mut dyn FnMut(usize) -> Vec<u32>,
    ) -> Result<Fill, Error> {
        let started = Instant::now();
        self.store.take_stats();
        let before = self.store.reader().stats();
        let shared = self.layout.per_layer > 1;
        for layer in 0..n_layer {
            let wanted = experts_of(layer);
            for part in 0..self.layout.per_layer {
                let at = self.layout.store_layer(part, layer);
                let (vram, ram) = self.store.slots(at);
                let mut order: Vec<u32> = wanted
                    .iter()
                    .copied()
                    .filter(|expert| !shared || owner(layer, *expert) == part)
                    .collect();
                order.truncate((vram + ram).min(n_experts));
                self.store.fill(at, &order)?;
            }
        }
        let progress =
            |store: &ExpertStore<CudaSlots>| (store.stats().uploads, store.stats().demand_reads);
        let mut last = (progress(&self.store), Instant::now());
        while !self.store.is_settled() && last.1.elapsed() < Duration::from_secs(2) {
            self.store.poll()?;
            let now = progress(&self.store);
            if now != last.0 {
                last = (now, Instant::now());
            }
        }
        let after = self.store.reader().stats();
        let stats = self.store.take_stats();
        Ok(Fill {
            uploads: stats.uploads,
            read_bytes: after.prefetch.device_bytes + after.demand.device_bytes
                - before.prefetch.device_bytes
                - before.demand.device_bytes,
            time: started.elapsed(),
        })
    }

    /// A step of `rows` rows begins: its routers have not run.
    pub(super) fn begin_step(&mut self, rows: u32) {
        self.rows = rows;
        self.followed.fill(0);
        self.store.begin_step();
    }

    /// Hands the route records that have arrived to the store, per part in
    /// the order of its layers, of the first `enqueued[part]` layers, and lets
    /// the store work. Never waits.
    pub(super) fn pump(&mut self, enqueued: &[usize]) -> Result<(), Error> {
        for (index, board) in self.boards.iter().enumerate() {
            while self.followed[index] < enqueued[index] {
                let i = self.followed[index];
                if !board.arrived(i)? {
                    break;
                }
                let record = RouteRecord::parse(board.record(i), self.used, self.rows)
                    .ok_or_else(|| invalid("a route record that is none"))?;
                let at = self.layout.store_layer(index, self.firsts[index] + i);
                self.groups.clear();
                self.groups.extend(record.groups());
                // The store uploads what the layer misses or computes it from
                // the layer's rows, by what its measured costs say.
                self.store.routed_rows(
                    at,
                    record.token(),
                    self.rows as usize,
                    &self.groups,
                    board.rows(i, self.rows as usize),
                )?;
                self.followed[index] += 1;
            }
        }
        self.store.poll()?;
        Ok(())
    }

    /// Whether every route record of the `enqueued` layers is with the store.
    pub(super) fn is_followed(&self, enqueued: &[usize]) -> bool {
        self.followed == enqueued
    }

    /// The step has ended on the devices.
    pub(super) fn end_step(&mut self) -> Result<(), Error> {
        Ok(self.store.end_step()?)
    }

    /// What a step that stands still looks like, for the error that ends it.
    pub(super) fn standing_still(&self, enqueued: &[usize]) -> String {
        let stats = self.store.stats();
        format!(
            "per part {enqueued:?} layers are enqueued and {:?} have their route record handled; \
             the store waits for {} experts and is {}settled; in this step and since the \
             counters were read: {} picks, {} in a slot, {} computed on the CPU, {} uploads, {} \
             reads from the file",
            self.followed,
            self.store.waiting(),
            if self.store.is_settled() { "" } else { "not " },
            stats.picks,
            stats.hits,
            stats.cpu_experts,
            stats.uploads,
            stats.demand_reads,
        )
    }
}

/// Builds the store of a placement from the slots of its devices.
pub(super) fn build_store(
    plan: StorePlan,
    layout: &Layout,
    map: ExpertMap,
    reader: Reader,
    slots: Vec<CudaSlots>,
    region: HostRegion,
    cpu: Option<(CpuPool, Serve)>,
) -> Result<ExpertStore<CudaSlots>, Error> {
    let mut store = ExpertStore::new(plan, map, reader, slots, region, None)?;
    if let Some((pool, serve)) = cpu {
        store.enable_cpu(pool, layout.shapes.clone(), serve)?;
    }
    Ok(store)
}
