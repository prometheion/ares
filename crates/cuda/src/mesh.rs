//! Several devices working on one step.
//!
//! In a tensor split every device holds a slice of each layer and computes a
//! partial result; the layer's output is the sum of the partials, and every
//! device needs that sum to go on. [`Mesh::sum_f32`] is that exchange.
//!
//! It is built so that the host never waits inside a step. Everything is
//! enqueued on the devices' streams, and the order between devices is kept on
//! the devices themselves. A model's whole forward pass, with two sums per
//! layer, can be enqueued in one go.
//!
//! There are three ways for a partial to cross, and the mesh picks the
//! fastest that applies:
//!
//! - **Push** (two devices with peer access, sums up to [`PUSH_MAX_VALUES`]):
//!   one kernel per device writes its partial straight into the other
//!   device's memory, waits for the other's, and adds. No copy, no event.
//! - **Peer copy** (peer access, any size or number of devices): each device
//!   pulls the others' partials over PCIe with the copy engine.
//! - **Host**: through a page-locked host buffer.
//!
//! Peer access between consumer cards needs patched kernel modules and a
//! matching driver library (see [`LIBRARY_ENV`](crate::LIBRARY_ENV)). With
//! those, a kernel can write into the other device's memory but reads from it
//! return garbage, which is why the direct path pushes.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::device::{Device, peer_access};
use crate::error::{Error, Result};
use crate::kernels::{Kernels, SUM_PEER_BLOCKS};
use crate::memory::{DeviceBuffer, DevicePtr, PinnedBuffer};
use crate::stream::{Event, Stream};

/// The largest sum the push kernel takes, in f32 values (512 KiB). Larger
/// partials are bound by the link's bandwidth, not by latency, and the copy
/// engine moves them without occupying the GPU's processors.
pub const PUSH_MAX_VALUES: u32 = 131_072;

/// How partial results travel between devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Directly from device to device.
    Peer,
    /// Through page-locked host memory.
    Host,
}

/// A page-locked buffer that only the driver reads and writes.
struct Staging {
    _buffer: PinnedBuffer,
    ptr: *mut u8,
}

// SAFETY: the pointer addresses memory owned by `_buffer` and is only ever
// handed to the driver, never dereferenced by this crate.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

impl Staging {
    fn new(device: &Device, len: usize) -> Result<Self> {
        let mut buffer = PinnedBuffer::new(device, len)?;
        let ptr = buffer.as_mut_ptr();
        Ok(Self { _buffer: buffer, ptr })
    }
}

/// What the push kernel needs on a device: where the peer writes.
struct PushTarget {
    /// Two slots of `push_values` f32 values; consecutive sums alternate, so
    /// the peer can write the next sum while this device still reads the last.
    recv: DeviceBuffer,
    /// One token per kernel block.
    flags: DeviceBuffer,
}

struct Member {
    device: Device,
    kernels: Kernels,
    /// Where this device receives the others' partials on the copy routes:
    /// one region of `max_values` f32 per other device, in device order.
    inbox: DeviceBuffer,
    /// Host route: where this device's partial waits for the others.
    staging: Option<Staging>,
    push: Option<PushTarget>,
    /// This device's partial is final (peer copy) or staged (host).
    ready: Event,
    /// This device has fetched every other partial.
    pulled: Event,
}

/// A group of devices that exchange partial results.
pub struct Mesh {
    members: Vec<Member>,
    route: Route,
    max_values: u32,
    push_values: u32,
    /// The number of the last push sum.
    token: AtomicU32,
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("devices", &self.members.iter().map(|m| m.device.ordinal()).collect::<Vec<_>>())
            .field("route", &self.route)
            .field("max_values", &self.max_values)
            .finish()
    }
}

impl Mesh {
    /// A mesh over `devices` for sums of up to `max_values` f32 values, on the
    /// fastest route all of them support.
    pub fn new(devices: &[Device], max_values: u32) -> Result<Self> {
        let mut all_peers = devices.len() > 1;
        for a in devices {
            for b in devices {
                if a.ordinal() != b.ordinal() && !peer_access(a.ordinal(), b.ordinal())?.0 {
                    all_peers = false;
                }
            }
        }
        Self::with_route(devices, max_values, if all_peers { Route::Peer } else { Route::Host })
    }

    /// A mesh on a given route. [`Route::Peer`] fails when a pair of devices
    /// has no peer access.
    pub fn with_route(devices: &[Device], max_values: u32, route: Route) -> Result<Self> {
        if devices.is_empty() || max_values == 0 {
            return Err(Error::Invalid {
                what: "mesh",
                detail: "a mesh needs at least one device and a size".into(),
            });
        }
        let max_bytes = u64::from(max_values) * 4;
        if route == Route::Peer {
            for a in devices {
                for b in devices {
                    if a.ordinal() != b.ordinal() {
                        a.enable_peer_access(b)?;
                    }
                }
            }
        }
        let push_values = if route == Route::Peer && devices.len() == 2 {
            max_values.min(PUSH_MAX_VALUES)
        } else {
            0
        };
        let others = devices.len().saturating_sub(1).max(1) as u64;
        let members = devices
            .iter()
            .map(|device| {
                let push = if push_values > 0 {
                    let flags = DeviceBuffer::new(device, u64::from(SUM_PEER_BLOCKS) * 4)?;
                    flags.write(0, &[0u8; SUM_PEER_BLOCKS as usize * 4])?;
                    let recv = DeviceBuffer::new(device, 2 * u64::from(push_values) * 4)?;
                    Some(PushTarget { recv, flags })
                } else {
                    None
                };
                Ok(Member {
                    device: device.clone(),
                    kernels: Kernels::load(device)?,
                    inbox: DeviceBuffer::new(device, max_bytes * others)?,
                    staging: match route {
                        Route::Host => Some(Staging::new(device, max_bytes as usize)?),
                        Route::Peer => None,
                    },
                    push,
                    ready: Event::new(device)?,
                    pulled: Event::new(device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { members, route, max_values, push_values, token: AtomicU32::new(0) })
    }

    pub fn route(&self) -> Route {
        self.route
    }

    pub fn devices(&self) -> impl Iterator<Item = &Device> {
        self.members.iter().map(|m| &m.device)
    }

    /// Enqueues the sum of the devices' partial results: afterwards each
    /// `partials[i]`, `n` f32 values on device `i`, holds the sum of all of
    /// them. `streams[i]` is device `i`'s stream; work enqueued on it later
    /// sees the sum.
    ///
    /// Nothing here waits for a device. Every device gets the bitwise same
    /// sum on every route.
    ///
    /// # Safety
    ///
    /// `streams[i]` must be a stream of device `i` of this mesh and
    /// `partials[i]` must address `n` f32 values on that device, alive until
    /// the enqueued work has run. Every sum of this mesh must be enqueued
    /// from one thread at a time and on the same streams, so that consecutive
    /// sums are ordered.
    pub unsafe fn sum_f32(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantees, with each partial as its own output.
        unsafe { self.sum(streams, partials, partials, n) }
    }

    /// Like [`Mesh::sum_f32`], but the sum is written to `outs[i]` on each
    /// device and the partials are left as they are. Use it where the next
    /// kernel can read its input from another buffer than the partial.
    ///
    /// # Safety
    ///
    /// As for [`Mesh::sum_f32`]; in addition `outs[i]` must address `n` f32
    /// values on device `i` that do not overlap any partial.
    pub unsafe fn sum_f32_to(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantees.
        unsafe { self.sum(streams, partials, outs, n) }
    }

    /// The sum of `partials` into `outs`, where `outs[i]` is `partials[i]` or
    /// does not overlap any partial.
    unsafe fn sum(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        let count = self.members.len();
        if streams.len() != count || partials.len() != count || outs.len() != count {
            return Err(Error::Invalid {
                what: "mesh sum",
                detail: format!(
                    "{count} devices, {} streams, {} partials, {} outputs",
                    streams.len(),
                    partials.len(),
                    outs.len()
                ),
            });
        }
        if n > self.max_values {
            return Err(Error::Invalid {
                what: "mesh sum",
                detail: format!("{n} values; the mesh was sized for {}", self.max_values),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let bytes = n as usize * 4;
        let slot = u64::from(self.max_values) * 4;

        if count == 1 {
            if outs[0] != partials[0] {
                // SAFETY: both ranges hold `n` values on the device and do
                // not overlap (the caller's guarantee).
                unsafe { streams[0].copy(outs[0], partials[0], bytes)? };
            }
            return Ok(());
        }

        if self.route == Route::Peer && n <= self.push_values {
            // One kernel per device. Each writes into the other's receive
            // slot and flags, then waits on its own.
            let token = self.token.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
            let slot_offset = u64::from(token % 2) * u64::from(self.push_values) * 4;
            for d in 0..2 {
                let own = self.members[d].push.as_ref().expect("push route has targets");
                let peer = self.members[1 - d].push.as_ref().expect("push route has targets");
                // SAFETY: the receive buffers hold two slots of
                // `push_values >= n` values and the flags one value per
                // block, on this device and on its peer, to which it has
                // peer access; the peer's half with the same token is
                // enqueued in this same loop. Data ranges are the caller's
                // guarantee.
                unsafe {
                    self.members[d].kernels.sum_peer_f32(
                        streams[d],
                        partials[d],
                        outs[d],
                        peer.recv.ptr().offset(slot_offset),
                        peer.flags.ptr(),
                        own.recv.ptr().offset(slot_offset),
                        own.flags.ptr(),
                        n,
                        token,
                    )?;
                }
            }
            return Ok(());
        }

        match self.route {
            Route::Peer => {
                // Every partial is final at this point of its stream.
                for (member, stream) in self.members.iter().zip(streams) {
                    member.ready.record(stream)?;
                }
                // Each device pulls the others' partials into its inbox...
                for (d, (member, stream)) in self.members.iter().zip(streams).enumerate() {
                    for (k, s) in (0..count).filter(|s| *s != d).enumerate() {
                        let source = &self.members[s];
                        stream.wait_event(&source.ready)?;
                        // SAFETY: the inbox has a region of `slot` bytes per
                        // other device and `bytes <= slot`; the source range
                        // is the caller's guarantee.
                        unsafe {
                            stream.copy_peer(
                                &member.device,
                                member.inbox.ptr().offset(k as u64 * slot),
                                &source.device,
                                partials[s],
                                bytes,
                            )?;
                        }
                    }
                    member.pulled.record(stream)?;
                }
                // ...and adds them only after everyone has pulled, because an
                // add in place overwrites the partial the others pull from,
                // and so may whatever the stream does next.
                for (d, (member, stream)) in self.members.iter().zip(streams).enumerate() {
                    for s in (0..count).filter(|s| *s != d) {
                        stream.wait_event(&self.members[s].pulled)?;
                    }
                    let mut own = partials[d];
                    for k in 0..count - 1 {
                        let received = member.inbox.ptr().offset(k as u64 * slot);
                        // SAFETY: all ranges hold `n` values on this device.
                        unsafe { member.kernels.add_to_f32(stream, outs[d], own, received, n)? };
                        own = outs[d];
                    }
                }
            }
            Route::Host => {
                // Each device copies its partial to its staging buffer...
                for ((member, stream), partial) in self.members.iter().zip(streams).zip(partials) {
                    let staging = member.staging.as_ref().expect("host route has staging");
                    // SAFETY: the staging buffer holds `slot >= bytes` bytes
                    // and lives as long as the mesh; only the driver touches
                    // it. The source range is the caller's guarantee.
                    unsafe { stream.copy_to_host(staging.ptr, *partial, bytes)? };
                    member.ready.record(stream)?;
                }
                // ...and the others fetch it from there and add it.
                for (d, (member, stream)) in self.members.iter().zip(streams).enumerate() {
                    let mut own = partials[d];
                    for (k, s) in (0..count).filter(|s| *s != d).enumerate() {
                        let source = &self.members[s];
                        let staging = source.staging.as_ref().expect("host route has staging");
                        let received = member.inbox.ptr().offset(k as u64 * slot);
                        stream.wait_event(&source.ready)?;
                        // SAFETY: as above; the staged data is complete once
                        // `ready` has happened. All device ranges hold `n`
                        // values on this device.
                        unsafe {
                            stream.copy_from_host(received, staging.ptr, bytes)?;
                            member.kernels.add_to_f32(stream, outs[d], own, received, n)?;
                        }
                        own = outs[d];
                    }
                    member.pulled.record(stream)?;
                }
                // A device's stream goes on only when the others have fetched
                // its staged partial. Otherwise its next sum could overwrite
                // the staging buffer under a slower device that is still
                // reading the previous one.
                for (d, stream) in streams.iter().enumerate() {
                    for s in (0..count).filter(|s| *s != d) {
                        stream.wait_event(&self.members[s].pulled)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// A one-way path for activations from one device to another: what a layer
/// split needs at each boundary between devices.
///
/// The sender does not wait for the receiver. It writes each parcel into the
/// next slot of a small ring on the receiver's side and goes on with its
/// next step, so both devices stay busy as long as there is independent work
/// (the chunks of a prompt, the rows of several sequences): this is what makes
/// a layer split a pipeline and not a relay. The sender is held back, on the
/// device, only when it is a whole ring ahead of the receiver.
///
/// The ring is the only memory a lane costs: `slots * max_values` f32 on the
/// receiver, plus the same in page-locked host memory on the host route.
pub struct Lane {
    from: Device,
    to: Device,
    route: Route,
    max_values: u32,
    /// On the receiver: `slots` regions of `max_values` f32.
    inbox: DeviceBuffer,
    /// Host route: one staging buffer per slot.
    staging: Vec<Staging>,
    /// Per slot, on the sender: the parcel has left.
    sent: Vec<Event>,
    /// Per slot, on the receiver: the parcel has been read.
    consumed: Vec<Event>,
    sent_count: AtomicU32,
    released_count: AtomicU32,
}

impl std::fmt::Debug for Lane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lane")
            .field("from", &self.from.ordinal())
            .field("to", &self.to.ordinal())
            .field("route", &self.route)
            .field("slots", &self.sent.len())
            .finish()
    }
}

/// A parcel on its way through a [`Lane`]. It must be received and then
/// released, in the order the parcels were sent.
#[derive(Debug)]
#[must_use = "a parcel that is not released blocks its slot of the lane"]
pub struct Parcel {
    sequence: u32,
    slot: u32,
    n: u32,
}

impl Parcel {
    /// The number of f32 values in the parcel.
    pub fn len(&self) -> u32 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl Lane {
    /// A lane from device `from` to device `to` of `mesh` (indices in the
    /// order the mesh was built with), with a ring of `slots` parcels of up
    /// to `max_values` f32 values. It uses the mesh's route.
    pub fn new(mesh: &Mesh, from: usize, to: usize, slots: u32, max_values: u32) -> Result<Self> {
        let count = mesh.members.len();
        if from >= count || to >= count || from == to || slots == 0 || max_values == 0 {
            return Err(Error::Invalid {
                what: "lane",
                detail: format!(
                    "from {from} to {to} of {count} devices, {slots} slots of {max_values} values"
                ),
            });
        }
        let (from, to) = (mesh.members[from].device.clone(), mesh.members[to].device.clone());
        let slot_bytes = u64::from(max_values) * 4;
        let staging = match mesh.route {
            Route::Host => (0..slots)
                .map(|_| Staging::new(&from, slot_bytes as usize))
                .collect::<Result<Vec<_>>>()?,
            Route::Peer => Vec::new(),
        };
        Ok(Self {
            inbox: DeviceBuffer::new(&to, slot_bytes * u64::from(slots))?,
            staging,
            sent: (0..slots).map(|_| Event::new(&from)).collect::<Result<Vec<_>>>()?,
            consumed: (0..slots).map(|_| Event::new(&to)).collect::<Result<Vec<_>>>()?,
            from,
            to,
            route: mesh.route,
            max_values,
            sent_count: AtomicU32::new(0),
            released_count: AtomicU32::new(0),
        })
    }

    /// The number of parcels the ring holds.
    pub fn slots(&self) -> u32 {
        self.sent.len() as u32
    }

    /// Enqueues, on the sender's stream, sending `n` f32 values at `src`.
    /// The sender's stream goes on without waiting for the receiver, unless
    /// the slot's previous parcel has not been read yet.
    ///
    /// Fails, with nothing enqueued, when `slots` parcels are sent and not
    /// yet released: the caller enqueues the receiving side of earlier
    /// parcels first.
    ///
    /// # Safety
    ///
    /// `sender` must be a stream of the lane's sending device and `src` must
    /// address `n` f32 values on it, alive until the enqueued work has run.
    /// All calls on one lane must come from one thread at a time.
    pub unsafe fn send_f32(&self, sender: &Stream, src: DevicePtr, n: u32) -> Result<Parcel> {
        if n > self.max_values {
            return Err(Error::Invalid {
                what: "lane",
                detail: format!("{n} values; the lane was sized for {}", self.max_values),
            });
        }
        let sequence = self.sent_count.load(Ordering::Relaxed);
        let in_flight = sequence.wrapping_sub(self.released_count.load(Ordering::Relaxed));
        if in_flight >= self.slots() {
            return Err(Error::Invalid {
                what: "lane",
                detail: format!(
                    "all {} slots hold parcels that are not released yet",
                    self.slots()
                ),
            });
        }
        let slot = sequence % self.slots();
        let bytes = n as usize * 4;
        // The slot's previous parcel must have been read. Its release was
        // enqueued before this call (checked above), so this waits for
        // exactly that; on the first round there is nothing to wait for.
        sender.wait_event(&self.consumed[slot as usize])?;
        // SAFETY: the slot is a region of `max_values >= n` values of the
        // inbox or of a staging buffer that lives as long as the lane; `src`
        // is the caller's guarantee.
        unsafe {
            match self.route {
                Route::Peer => {
                    sender.copy_peer(&self.to, self.slot_ptr(slot), &self.from, src, bytes)?
                }
                Route::Host => sender.copy_to_host(self.staging[slot as usize].ptr, src, bytes)?,
            }
        }
        self.sent[slot as usize].record(sender)?;
        self.sent_count.store(sequence.wrapping_add(1), Ordering::Relaxed);
        Ok(Parcel { sequence, slot, n })
    }

    /// Enqueues, on the receiver's stream, waiting for `parcel`, and returns
    /// where its values are on the receiving device. Work enqueued on that
    /// stream afterwards can read them there until the parcel is released.
    ///
    /// # Safety
    ///
    /// `receiver` must be a stream of the lane's receiving device, and
    /// `parcel` must come from this lane.
    pub unsafe fn receive(&self, receiver: &Stream, parcel: &Parcel) -> Result<DevicePtr> {
        let slot = parcel.slot;
        receiver.wait_event(&self.sent[slot as usize])?;
        if self.route == Route::Host {
            // SAFETY: the staging buffer and the inbox slot both hold
            // `max_values >= parcel.n` values and live as long as the lane.
            unsafe {
                receiver.copy_from_host(
                    self.slot_ptr(slot),
                    self.staging[slot as usize].ptr,
                    parcel.n as usize * 4,
                )?;
            }
        }
        Ok(self.slot_ptr(slot))
    }

    /// Enqueues, on the receiver's stream, the end of the receiver's use of
    /// `parcel`: work enqueued before this call has read it, and the slot may
    /// be written again once that work has run.
    ///
    /// Parcels are released in the order they were sent.
    pub fn release(&self, receiver: &Stream, parcel: Parcel) -> Result<()> {
        let expected = self.released_count.load(Ordering::Relaxed);
        if parcel.sequence != expected {
            return Err(Error::Invalid {
                what: "lane",
                detail: format!(
                    "parcel {} released while parcel {expected} is the oldest in flight",
                    parcel.sequence
                ),
            });
        }
        self.consumed[parcel.slot as usize].record(receiver)?;
        self.released_count.store(expected.wrapping_add(1), Ordering::Relaxed);
        Ok(())
    }

    fn slot_ptr(&self, slot: u32) -> DevicePtr {
        self.inbox.ptr().offset(u64::from(slot) * u64::from(self.max_values) * 4)
    }
}
