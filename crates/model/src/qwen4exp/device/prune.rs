//! Dynamic expert pruning, a fourth level beside VRAM, RAM and the file: a
//! pick of the router whose expert only the file holds may be left out or
//! stood in for by an expert a tier holds, so that the step does not wait for
//! the file (`ares_cuda::Prune` has the rules, `kernels/expert_router.cuh`
//! the kernels).
//!
//! The router does it on the device, right after its top-k: the record keeps
//! the router's picks and weights, and its groups, which the expert kernels
//! and the store work from, are what is computed. So the store never sees a
//! pick that was pruned, and a step whose rows touch nothing computes what it
//! computes without pruning, bit for bit.
//!
//! What the router needs to know is which experts a tier holds. Its slot table
//! does not say (it knows its own device's VRAM, not the other device's or
//! RAM), so the host writes a byte per expert of every layer of the model at
//! the start of every step from the store (`PruneMask::build`): 1 for an
//! expert in a VRAM slot of any device, in the RAM tier, or on its way to one.
//! Each part that routes copies its layers' bytes onto its device on its
//! stream before its first layer (`PartPrune::begin`). An expert that
//! arrives during the step counts from the next one.
//!
//! Only steps of up to eight rows (generation, windows of drafts) are pruned;
//! the slices of a prompt are read whole.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use ares_core::Ledger;
use ares_cuda::{Device, DeviceBuffer, NearEntry, PinnedBuffer, Prune, PruneLayer, Stream};
use ares_runtime::ModelError;
use ares_runtime::expert_store::{CudaSlots, ExpertNeighbours, ExpertStore};

use super::feed::Layout;
use super::placement::Placement;
use crate::qwen4exp::Config;

/// The pruning a model is opened with: the dial, and for
/// [`ares_cuda::PruneRule::Similar`] the table of similar experts.
#[derive(Clone, Debug)]
pub struct PruneSetup {
    pub prune: Prune,
    pub near: Option<Arc<ExpertNeighbours>>,
}

/// Which experts the tiers hold, a byte per expert of every layer of the
/// model, as the host writes it at the start of a step. Two buffers take
/// turns: the copies a step enqueued from one are done before the step after
/// the next writes it again.
pub struct PruneMask {
    n_layers: usize,
    n_experts: usize,
    inner: Mutex<MaskInner>,
}

struct MaskInner {
    buffers: [PinnedBuffer; 2],
    current: usize,
}

impl std::fmt::Debug for PruneMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PruneMask").field("layers", &self.n_layers).finish_non_exhaustive()
    }
}

impl PruneMask {
    /// A mask of `n_layers` layers of `n_experts` experts, every expert held
    /// until the store first writes it at the start of a step: nothing is
    /// pruned before.
    pub fn new(device: &Device, n_layers: usize, n_experts: usize) -> Result<Self, ModelError> {
        let len = (n_layers * n_experts).max(1);
        let mut buffers = [PinnedBuffer::new(device, len)?, PinnedBuffer::new(device, len)?];
        for buffer in &mut buffers {
            // SAFETY: the buffer is `len` bytes and nothing reads it yet.
            unsafe { buffer.as_mut_ptr().write_bytes(1, len) };
        }
        Ok(Self { n_layers, n_experts, inner: Mutex::new(MaskInner { buffers, current: 0 }) })
    }

    /// Page-locked bytes of the mask.
    pub fn bytes(n_layers: usize, n_experts: usize) -> u64 {
        2 * (n_layers * n_experts).max(1) as u64
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MaskInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes what the tiers of `store` hold into the buffer that is not
    /// current and makes it current. Call it when a step begins, before any
    /// part enqueues the step.
    pub(super) fn build(&self, store: &ExpertStore<CudaSlots>, layout: &Layout) {
        let mut inner = self.lock();
        let next = 1 - inner.current;
        let n = self.n_experts;
        let len = self.n_layers * n;
        // SAFETY: the buffer is `len` bytes; the copies of the step before the
        // last, which read it, are done (see the type).
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(inner.buffers[next].as_mut_ptr(), len) };
        bytes.fill(0);
        for (layer, out) in bytes.chunks_exact_mut(n).enumerate() {
            for part in 0..layout.per_layer {
                store.mark_present(layout.store_layer(part, layer), out);
            }
        }
        inner.current = next;
    }

    /// Enqueues on `stream` the copy of the current buffer's `layers` to
    /// `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must address `layers.len() * n_experts` bytes on the stream's
    /// device, alive until the copy has run.
    unsafe fn enqueue(
        &self,
        stream: &Stream,
        dst: ares_cuda::DevicePtr,
        layers: Range<usize>,
    ) -> Result<(), ModelError> {
        let inner = self.lock();
        let buffer = &inner.buffers[inner.current];
        let (at, len) = (layers.start * self.n_experts, layers.len() * self.n_experts);
        // SAFETY: the range is inside the buffer, which lives as long as the
        // mask, and the buffer is written again only two steps later.
        unsafe { stream.copy_from_host(dst, buffer.as_ptr().add(at), len)? };
        Ok(())
    }
}

/// The pruning of one part that routes: the dial, its layers' bytes of the
/// mask on its device, and its layers' table of similar experts.
#[derive(Debug)]
pub(super) struct PartPrune {
    prune: Prune,
    layers: Range<usize>,
    n_experts: u32,
    cached: DeviceBuffer,
    near: Option<(DeviceBuffer, u32)>,
    mask: Option<Arc<PruneMask>>,
}

impl PartPrune {
    /// The tables of the part that routes `layers` of a model with
    /// `n_experts` experts per layer, reserved in `ledger`.
    pub(super) fn new(
        device: &Device,
        ledger: &mut Ledger,
        setup: &PruneSetup,
        layers: Range<usize>,
        n_experts: usize,
    ) -> Result<Self, ModelError> {
        let cached_bytes = (layers.len() * n_experts).max(1) as u64;
        ledger.reserve("qwen4exp pruning: experts in a tier", cached_bytes)?;
        let cached = DeviceBuffer::new(device, cached_bytes)?;
        // Every expert held until a mask is attached: nothing is pruned.
        cached.write(0, &vec![1u8; cached_bytes as usize])?;
        let near = match &setup.near {
            Some(table) => {
                let key = table.key();
                if key.experts as usize != n_experts || (key.layers as usize) < layers.end {
                    return Err(ModelError::Invalid(format!(
                        "a table of similar experts for {} layers of {} experts, the model has {}",
                        key.layers, key.experts, n_experts
                    )));
                }
                let count = table.count();
                let mut bytes = Vec::with_capacity(layers.len() * n_experts * count * 8);
                for layer in layers.clone() {
                    for expert in 0..n_experts {
                        for near in table.of(layer, expert) {
                            let entry = NearEntry {
                                expert: near.expert,
                                closeness: near.closeness,
                                scale: near.scale,
                            };
                            bytes.extend_from_slice(&entry.to_bytes());
                        }
                    }
                }
                let len = (bytes.len() as u64).max(8);
                ledger.reserve("qwen4exp pruning: similar experts", len)?;
                let buffer = DeviceBuffer::new(device, len)?;
                buffer.write(0, &bytes)?;
                Some((buffer, count as u32))
            }
            None => None,
        };
        Ok(Self {
            prune: setup.prune,
            layers,
            n_experts: n_experts as u32,
            cached,
            near,
            mask: None,
        })
    }

    pub(super) fn set_mask(&mut self, mask: Arc<PruneMask>) {
        self.mask = Some(mask);
    }

    pub(super) fn prune(&self) -> &Prune {
        &self.prune
    }

    /// A step begins: the mask of its start comes onto the device.
    pub(super) fn begin(&self, stream: &Stream) -> Result<(), ModelError> {
        let Some(mask) = &self.mask else { return Ok(()) };
        // SAFETY: the buffer holds the part's layers.
        unsafe { mask.enqueue(stream, self.cached.ptr(), self.layers.clone()) }
    }

    /// Where the pruning of the part's layer `i` reads.
    pub(super) fn layer(&self, i: usize) -> PruneLayer {
        let (near, near_count) =
            self.near.as_ref().map_or((ares_cuda::DevicePtr::NULL, 0), |(b, n)| (b.ptr(), *n));
        let per_layer = u64::from(self.n_experts);
        PruneLayer {
            cached: self.cached.ptr().offset(i as u64 * per_layer),
            near: near.offset(if near_count == 0 {
                0
            } else {
                i as u64 * per_layer * 8 * u64::from(near_count)
            }),
            near_count,
        }
    }
}

/// What pruning did since the counters were read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PruneStats {
    /// Picks of the routers in steps that prune.
    pub picks: u64,
    /// Picks that were left out, and picks that an expert stood in for.
    pub skipped: u64,
    pub stood_in: u64,
}

/// With two devices on every layer: the mixer's index and, per layer and
/// expert, 1 where the other device owns the expert; `None` for whole
/// layers per device.
pub(super) fn owners(placement: &Placement, c: &Config) -> Option<(usize, Vec<u8>)> {
    let mixer = placement.tensor.as_ref()?.mixer;
    let owner = (0..c.n_layer)
        .flat_map(|layer| {
            (0..c.experts.count as u32)
                .map(move |expert| u8::from(placement.owner(layer, expert) != mixer))
        })
        .collect();
    Some((mixer, owner))
}
