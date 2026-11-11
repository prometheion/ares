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
//!
//! [`Mesh::sum_f32`] makes every device wait until the sum is there. A model
//! whose step has independent rows can do better with the two halves of a sum
//! apart, which is what the **slots** of a mesh are for ([`Mesh::with_slots`]):
//! a device puts its partial into its outbox and *posts* it, goes on with
//! other rows while the copy engine carries it to the others, and later lets
//! its stream wait for the others' partials of that slot to *arrive*. What is
//! in a parcel and how the partials are added is the model's business (the
//! merge kernels of [`LayerKernels`](crate::LayerKernels) send f16); the mesh
//! moves bytes and keeps the order. A model with its own merge kernel also
//! gets the push path as [`Mesh::push`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::device::{Device, peer_access};
use crate::error::{Error, Result};
use crate::kernels::{Kernels, SUM_PEER_BLOCKS};
use crate::memory::{DeviceBuffer, DevicePtr, PinnedBuffer};
use crate::stream::{Event, Stream};

/// The largest sum the push kernel takes, in f32 values (512 KiB). Larger
/// partials are bound by the link's bandwidth, not by latency, and the copy
/// engine moves them without occupying the GPU's processors.
pub const PUSH_MAX_VALUES: u32 = 131_072;

/// The most rows of one sum pushed with [`Mesh::push`]: its kernel has a
/// block and a flag per row.
pub const PUSH_MAX_ROWS: u32 = 64;

/// The bytes the host route moves at a time: a device fetches a chunk of a
/// parcel as soon as it is in host memory, so the way in follows the way out
/// a chunk behind instead of a parcel behind. The size is the one of the
/// llama.cpp fork's copy-engine sum; other sizes were not measured.
const STAGE_CHUNK: usize = 512 << 10;

/// Where one device's kernel of a pushed sum writes and reads: see
/// [`Mesh::push`] and `kernels/layer_merge.cuh`.
#[derive(Clone, Copy, Debug)]
pub struct PushSlot {
    /// The peer's receive area for this sum (peer memory: write only).
    pub peer_recv: DevicePtr,
    /// The peer's flags, [`PUSH_MAX_ROWS`] of them (peer memory: write only).
    pub peer_flags: DevicePtr,
    /// This device's receive area for this sum.
    pub own_recv: DevicePtr,
    /// This device's flags.
    pub own_flags: DevicePtr,
    /// The bytes of a receive area.
    pub bytes: u64,
    /// The number of this sum.
    pub token: u32,
}

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
    /// One token per kernel block of [`Mesh::sum_f32`] or per row of
    /// [`Mesh::push`].
    flags: DeviceBuffer,
}

/// What the slots of a mesh need on one device.
struct SlotSide {
    /// Carries this device's parcels away: to the other devices (peer route)
    /// or to its staging (host route).
    out: Stream,
    /// Host route: fetches the others' parcels from their staging.
    fetch: Option<Stream>,
    /// One region of `slot_bytes` per slot: this device's parcels.
    outbox: DeviceBuffer,
    /// Per slot one region of `slot_bytes` per other device, in device order.
    inbox: DeviceBuffer,
    /// Host route: this device's parcels on their way, one buffer per slot.
    staging: Vec<Staging>,
    /// Per slot, on the device's own stream: the parcel is in the outbox.
    ready: Vec<Event>,
    /// Per slot, on `out`: the parcel has left the outbox.
    sent: Vec<Event>,
    /// Host route, per slot and chunk, on `out`: the chunk is in staging.
    staged: Vec<Vec<Event>>,
    /// Host route, per slot, on `fetch`: the others' parcels are in the inbox.
    fetched: Vec<Event>,
    /// Per slot, on the device's own stream: the inbox has been read.
    consumed: Vec<Event>,
    /// Per slot: the bytes of the parcel posted last.
    bytes: Vec<AtomicU64>,
}

/// What [`Mesh::release_waiters`] writes the flags with: a stream of the
/// device that nothing else uses, and the values to write, in page-locked
/// memory, so that the copy needs nothing the device's own streams hold.
struct Escape {
    stream: Stream,
    values: PinnedBuffer,
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
    /// Where the push flags are written from when a kernel waits for a peer
    /// that will not come; present with the push flags.
    escape: Option<Mutex<Escape>>,
    slots: Option<SlotSide>,
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
    /// The bytes of a slot; the slots themselves are in the members.
    slot_bytes: u64,
    /// Per device: the number of its last push sum. The devices count alike
    /// because they enqueue the same sums.
    tokens: Vec<AtomicU32>,
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
                    let n_flags = SUM_PEER_BLOCKS.max(PUSH_MAX_ROWS) as usize;
                    let flags = DeviceBuffer::new(device, n_flags as u64 * 4)?;
                    flags.write(0, &vec![0u8; n_flags * 4])?;
                    let recv = DeviceBuffer::new(device, 2 * u64::from(push_values) * 4)?;
                    Some(PushTarget { recv, flags })
                } else {
                    None
                };
                let escape = if push.is_some() {
                    let n_flags = SUM_PEER_BLOCKS.max(PUSH_MAX_ROWS) as usize;
                    Some(Mutex::new(Escape {
                        stream: Stream::new(device, 0)?,
                        values: PinnedBuffer::new(device, n_flags * 4)?,
                    }))
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
                    escape,
                    slots: None,
                    ready: Event::new(device)?,
                    pulled: Event::new(device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tokens = devices.iter().map(|_| AtomicU32::new(0)).collect();
        Ok(Self { members, route, max_values, push_values, slot_bytes: 0, tokens })
    }

    /// Gives the mesh `slots` slots for parcels of up to `slot_bytes` bytes:
    /// on every device an outbox and, per other device, an inbox of that many
    /// parcels, a stream for the copies, and on the host route the same bytes
    /// once more in page-locked host memory.
    pub fn with_slots(mut self, slots: u32, slot_bytes: u64) -> Result<Self> {
        if slots == 0 || slot_bytes == 0 {
            return Err(Error::Invalid {
                what: "mesh slots",
                detail: format!("{slots} slots of {slot_bytes} bytes"),
            });
        }
        let slot_bytes = slot_bytes.next_multiple_of(64);
        let others = self.members.len().saturating_sub(1).max(1) as u64;
        let chunks = (slot_bytes as usize).div_ceil(STAGE_CHUNK);
        let route = self.route;
        for member in &mut self.members {
            let device = &member.device.clone();
            let events = || (0..slots).map(|_| Event::new(device)).collect::<Result<Vec<_>>>();
            let host = route == Route::Host;
            member.slots = Some(SlotSide {
                out: Stream::new(device, 0)?,
                fetch: host.then(|| Stream::new(device, 0)).transpose()?,
                outbox: DeviceBuffer::new(device, slot_bytes * u64::from(slots))?,
                inbox: DeviceBuffer::new(device, slot_bytes * u64::from(slots) * others)?,
                staging: (0..if host { slots } else { 0 })
                    .map(|_| Staging::new(device, slot_bytes as usize))
                    .collect::<Result<Vec<_>>>()?,
                ready: events()?,
                sent: events()?,
                staged: (0..if host { slots } else { 0 })
                    .map(|_| (0..chunks).map(|_| Event::new(device)).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?,
                fetched: if host { events()? } else { Vec::new() },
                consumed: events()?,
                bytes: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            });
        }
        self.slot_bytes = slot_bytes;
        Ok(self)
    }

    pub fn route(&self) -> Route {
        self.route
    }

    pub fn devices(&self) -> impl Iterator<Item = &Device> {
        self.members.iter().map(|m| &m.device)
    }

    /// Lets every kernel of the push path that waits for another device's
    /// flag go on, now and for the sums enqueued after it, with whatever the
    /// other device has (or has not) written.
    ///
    /// The kernels of a push (`Mesh::sum_f32` of a few values, [`Mesh::push`])
    /// are enqueued by one host thread per device and wait on the device for
    /// their peer's. When a thread fails before it has enqueued its half, its
    /// peer's kernel waits for a flag that nobody sets, and everything behind
    /// it in the stream, the end of the step and the destruction of the
    /// device's buffers included, waits with it: the kernel gives up only
    /// after about 4 billion polls. This writes every flag ahead of every
    /// token that is likely to be asked for (a billion sums ahead), from the
    /// host, over a stream and a copy engine that the waiting kernel does not
    /// hold. The values the released kernels then add are meaningless, so it
    /// is for a step that has failed and whose results nobody reads; after it
    /// the mesh does no sum that can be trusted.
    ///
    /// It waits only for its own copies, never for a stream of the devices'
    /// work, so it returns even while a kernel spins. Work that the failed
    /// device had already queued still runs after it, and each of its pushes
    /// writes a flag of the other device with the token it had, which undoes
    /// the release for the kernels that wait for a later token:
    /// [`Mesh::release_waiters_until_idle`] is the call for the thread that
    /// failed.
    pub fn release_waiters(&self) -> Result<()> {
        for (rank, member) in self.members.iter().enumerate() {
            let (Some(push), Some(escape)) = (&member.push, &member.escape) else { continue };
            // The device waits for tokens counted by its own rank; a thread
            // that has failed counts no more, one that goes on counts a few
            // hundred more before it gives up.
            let ahead = self.tokens[rank].load(Ordering::Relaxed).wrapping_add(1 << 30);
            let mut escape = escape.lock().unwrap_or_else(|e| e.into_inner());
            let Escape { stream, values } = &mut *escape;
            for word in values.as_chunks_mut::<4>().0 {
                *word = ahead.to_le_bytes();
            }
            // SAFETY: the pinned buffer holds as many bytes as the flags
            // (one u32 per kernel block or row) and stays unchanged until
            // the stream has synchronised, which is below.
            unsafe { stream.copy_from_host(push.flags.ptr(), values.as_ptr(), values.len())? };
            stream.synchronize()?;
        }
        Ok(())
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
            let token = self.tokens[0].fetch_add(1, Ordering::Relaxed).wrapping_add(1);
            self.tokens[1].fetch_add(1, Ordering::Relaxed);
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

impl Mesh {
    /// [`Mesh::release_waiters`], again and again until everything `stream`
    /// had queued has run (or `patience` is out): the pushes it had queued
    /// before the failure write the other device's flags with the tokens they
    /// carry, one by one, and each of those writes is answered by the next
    /// release. After the last of them the flags stay ahead for good. Returns
    /// whether the stream went idle.
    pub fn release_waiters_until_idle(
        &self,
        stream: &Stream,
        patience: std::time::Duration,
    ) -> Result<bool> {
        let until = std::time::Instant::now() + patience;
        loop {
            self.release_waiters()?;
            if stream.is_idle()? {
                // Whatever it wrote last is answered now.
                self.release_waiters()?;
                return Ok(true);
            }
            if std::time::Instant::now() >= until {
                return Ok(false);
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }
}

/// The slots and the push of a mesh, for a model with its own merge kernels.
///
/// Unlike the sums above, these calls are made per device, each device's by
/// the thread that enqueues its stream, and the threads must keep an order
/// among them: counted per slot, a device's `k`-th [`Mesh::post`] comes after
/// every other device's `k-1`-th [`Mesh::consumed`] has returned, and its
/// `k`-th [`Mesh::arrive`] after every other device's `k`-th [`Mesh::post`]
/// has returned. (An event that a stream is told to wait for must have been
/// recorded before, and not again since.) `Links` of `ares-runtime` keeps
/// that order with two counters per slot.
impl Mesh {
    /// The number of slots.
    pub fn slots(&self) -> u32 {
        self.members[0].slots.as_ref().map_or(0, |s| s.ready.len() as u32)
    }

    /// The bytes of a slot: the stride of the regions [`Mesh::arrive`]
    /// returns.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// The bytes one pushed sum may write per device, or 0 when the mesh has
    /// no push path (no peer access, or not two devices).
    pub fn push_bytes(&self) -> u64 {
        u64::from(self.push_values) * 4
    }

    /// The next pushed sum of device `rank`: where its merge kernel
    /// (`LayerKernels::merge_push`) writes and reads. Every device must take
    /// the pushes of a step in the same order and enqueue a kernel for each;
    /// nothing else is needed between the devices, the kernels wait for each
    /// other on the devices. `None` when the mesh has no push path.
    pub fn push(&self, rank: usize) -> Option<PushSlot> {
        if self.push_values == 0 || rank >= 2 {
            return None;
        }
        let token = self.tokens[rank].fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        let bytes = self.push_bytes();
        let offset = u64::from(token % 2) * bytes;
        let own = self.members[rank].push.as_ref()?;
        let peer = self.members[1 - rank].push.as_ref()?;
        Some(PushSlot {
            peer_recv: peer.recv.ptr().offset(offset),
            peer_flags: peer.flags.ptr(),
            own_recv: own.recv.ptr().offset(offset),
            own_flags: own.flags.ptr(),
            bytes,
            token,
        })
    }

    fn side(&self, rank: usize, slot: u32) -> Result<&SlotSide> {
        self.members
            .get(rank)
            .and_then(|m| m.slots.as_ref())
            .filter(|s| (slot as usize) < s.ready.len())
            .ok_or_else(|| Error::Invalid {
                what: "mesh slot",
                detail: format!(
                    "slot {slot} of device {rank}; the mesh has {} slots",
                    self.slots()
                ),
            })
    }

    /// Where device `to` finds the parcel of device `from` in `slot`.
    fn inbox_ptr(&self, to: usize, slot: u32, from: usize) -> Result<DevicePtr> {
        let others = self.members.len() as u64 - 1;
        let place = if from < to { from } else { from - 1 };
        let region = u64::from(slot) * others + place as u64;
        Ok(self.side(to, slot)?.inbox.ptr().offset(region * self.slot_bytes))
    }

    /// Where device `rank` writes its next parcel of `slot`. Work enqueued on
    /// `stream` (the device's own) after this call waits until the slot's
    /// last parcel has left.
    pub fn outbox(&self, rank: usize, stream: &Stream, slot: u32) -> Result<DevicePtr> {
        let side = self.side(rank, slot)?;
        stream.wait_event(&side.sent[slot as usize])?;
        Ok(side.outbox.ptr().offset(u64::from(slot) * self.slot_bytes))
    }

    /// Sends the first `bytes` bytes of device `rank`'s outbox of `slot` to
    /// every other device, once the work enqueued on `stream` so far has run.
    /// `stream` itself does not wait for the copy.
    ///
    /// # Safety
    ///
    /// `stream` must be the stream of device `rank` on which its outbox was
    /// written, and the calls must keep the order described at this block.
    pub unsafe fn post(&self, rank: usize, stream: &Stream, slot: u32, bytes: u64) -> Result<()> {
        let side = self.side(rank, slot)?;
        if bytes > self.slot_bytes {
            return Err(Error::Invalid {
                what: "mesh slot",
                detail: format!("a parcel of {bytes} bytes; a slot holds {}", self.slot_bytes),
            });
        }
        let s = slot as usize;
        let source = side.outbox.ptr().offset(u64::from(slot) * self.slot_bytes);
        side.bytes[s].store(bytes, Ordering::Release);
        side.ready[s].record(stream)?;
        side.out.wait_event(&side.ready[s])?;
        let peers = (0..self.members.len()).filter(|p| *p != rank);
        match self.route {
            Route::Peer => {
                for p in peers {
                    // The peer's merge of the slot's last parcel has read it.
                    side.out.wait_event(&self.side(p, slot)?.consumed[s])?;
                    // SAFETY: both regions hold `slot_bytes >= bytes` bytes
                    // and live as long as the mesh.
                    unsafe {
                        side.out.copy_peer(
                            &self.members[p].device,
                            self.inbox_ptr(p, slot, rank)?,
                            &self.members[rank].device,
                            source,
                            bytes as usize,
                        )?;
                    }
                }
            }
            Route::Host => {
                // Every peer has fetched the slot's last parcel from staging.
                for p in peers {
                    side.out.wait_event(&self.side(p, slot)?.fetched[s])?;
                }
                let staging = side.staging[s].ptr;
                for (c, at) in (0..bytes as usize).step_by(STAGE_CHUNK).enumerate() {
                    let len = STAGE_CHUNK.min(bytes as usize - at);
                    // SAFETY: the staging buffer and the outbox region hold
                    // `slot_bytes >= bytes` bytes; only the driver touches
                    // the staging buffer.
                    unsafe {
                        side.out.copy_to_host(staging.add(at), source.offset(at as u64), len)?
                    };
                    side.staged[s][c].record(&side.out)?;
                }
            }
        }
        side.sent[s].record(&side.out)
    }

    /// Makes work enqueued on `stream` (device `rank`'s own) after this call
    /// wait until the other devices' parcels of `slot` are on this device,
    /// and returns where: one region per other device, in device order,
    /// [`Mesh::slot_bytes`] apart. They stay until [`Mesh::consumed`].
    ///
    /// # Safety
    ///
    /// `stream` must be the stream of device `rank` that reads the parcels,
    /// and the calls must keep the order described at this block.
    pub unsafe fn arrive(&self, rank: usize, stream: &Stream, slot: u32) -> Result<DevicePtr> {
        let side = self.side(rank, slot)?;
        let s = slot as usize;
        let peers = (0..self.members.len()).filter(|p| *p != rank);
        match (&side.fetch, self.route) {
            (Some(fetch), Route::Host) => {
                // This device's merge of the slot's last parcels has read them.
                fetch.wait_event(&side.consumed[s])?;
                for p in peers {
                    let theirs = self.side(p, slot)?;
                    let bytes = theirs.bytes[s].load(Ordering::Acquire) as usize;
                    let to = self.inbox_ptr(rank, slot, p)?;
                    for (c, at) in (0..bytes).step_by(STAGE_CHUNK).enumerate() {
                        let len = STAGE_CHUNK.min(bytes - at);
                        fetch.wait_event(&theirs.staged[s][c])?;
                        // SAFETY: the inbox region and the peer's staging
                        // buffer hold `slot_bytes >= bytes` bytes; the chunk
                        // is complete once its event has happened.
                        unsafe {
                            fetch.copy_from_host(
                                to.offset(at as u64),
                                theirs.staging[s].ptr.add(at).cast_const(),
                                len,
                            )?;
                        }
                    }
                }
                side.fetched[s].record(fetch)?;
                stream.wait_event(&side.fetched[s])?;
            }
            _ => {
                for p in peers {
                    stream.wait_event(&self.side(p, slot)?.sent[s])?;
                }
            }
        }
        Ok(side
            .inbox
            .ptr()
            .offset(u64::from(slot) * (self.members.len() as u64 - 1) * self.slot_bytes))
    }

    /// Ends device `rank`'s use of the parcels of `slot` that arrived: the
    /// work enqueued on `stream` so far has read them, and the slot may be
    /// written again once that work has run.
    pub fn consumed(&self, rank: usize, stream: &Stream, slot: u32) -> Result<()> {
        self.side(rank, slot)?.consumed[slot as usize].record(stream)
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
