//! The paths between device threads.
//!
//! `ares-cuda` orders work between devices on the devices themselves, so
//! the host never waits inside a step. Its sums and hand-overs, though, are
//! *one* call that touches the streams of several devices, and they must not
//! be called from two threads at once. With a thread per device that call has
//! to be made by one of them while the others wait their turn to go on
//! enqueuing. What waits there is host threads that are ahead of their
//! devices; the devices keep running whatever was enqueued before.
//!
//! - [`Links::sum`]: a rendezvous. Every device thread arrives with its
//!   partial result; the last one to arrive enqueues the sum on all streams
//!   and lets the others go. This is two atomic operations per thread per sum
//!   when the threads are in step, and spinning when one is behind.
//! - [`Links::post`], [`Links::arrive`], [`Links::consumed`]: the two halves
//!   of a sum apart, in the slots of the mesh, for a model that merges the
//!   partials with its own kernel. Each thread enqueues its own device's
//!   part; a thread waits only where its stream must wait for an event that
//!   another thread records (two counters per slot say which have been).
//!   [`Links::push`] is the same for a few rows and needs no order between
//!   the threads at all.
//! - [`Links::send`], [`Links::receive`], [`Links::release`]: the parcels of a
//!   layer split. The sender enqueues a parcel and puts it in a queue; the
//!   receiver takes it out when it is there. The receiver's stream may only
//!   wait for a parcel's event after the sender recorded it, which is why the
//!   receiver waits for the *enqueue* of the parcel on the host (not for the
//!   transfer on the device).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use ares_cuda::{Device, DevicePtr, Lane, Mesh, Parcel, PushSlot, Stream};

use crate::model::{LinkSpec, ModelError, Split};

/// The most devices in one step. A fixed bound keeps the rendezvous free of
/// allocation.
pub const MAX_DEVICES: usize = 8;

/// A rendezvous of the device threads in front of a sum.
struct Collective {
    mesh: Mesh,
    streams: Vec<Arc<Stream>>,
    pointers: Vec<AtomicU64>,
    arrived: AtomicUsize,
    generation: AtomicU64,
    /// Per device and slot (`device * slots + slot`): the parcels it posted
    /// and the arrivals whose use it ended.
    posted: Vec<AtomicU64>,
    consumed: Vec<AtomicU64>,
}

impl Collective {
    fn slot(&self, rank: usize, slot: u32) -> usize {
        rank * self.mesh.slots() as usize + slot as usize
    }
}

struct LaneLink {
    lane: Lane,
    state: Mutex<LaneState>,
}

struct LaneState {
    /// Parcels the sender enqueued and the receiver has not taken.
    sent: VecDeque<Parcel>,
    /// Parcels the receiver has taken and not released.
    received: VecDeque<Parcel>,
}

/// The paths a model uses between its device parts, built by the runtime from
/// the model's [`LinkSpec`].
pub struct Links {
    collective: Option<Collective>,
    lanes: Vec<LaneLink>,
    /// Set when a device thread failed, so threads that wait for it give up.
    abort: AtomicBool,
}

impl std::fmt::Debug for Links {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Links")
            .field("sums", &self.collective.is_some())
            .field("lanes", &self.lanes.len())
            .finish()
    }
}

impl Links {
    /// Builds the paths `split` needs over `devices`, whose streams are
    /// `streams` (in the same order).
    pub(crate) fn new(
        split: Split,
        devices: &[Device],
        streams: &[Arc<Stream>],
        spec: LinkSpec,
    ) -> Result<Self, ModelError> {
        if devices.len() > MAX_DEVICES {
            return Err(ModelError::Invalid(format!(
                "{} devices; a step can span {MAX_DEVICES}",
                devices.len()
            )));
        }
        let (mut collective, mut lanes) = (None, Vec::new());
        match split {
            Split::Single => {}
            Split::Tensor => {
                // The slots hold the parcels of the model's own merges: as
                // many as a step has groups of rows, each of the size the
                // model names for a parcel.
                let slots = spec.lane_slots.max(1);
                let counters = || (0..devices.len() * slots as usize).map(|_| AtomicU64::new(0));
                collective = Some(Collective {
                    mesh: Mesh::new(devices, spec.sum_values.max(1))?
                        .with_slots(slots, u64::from(spec.lane_values.max(1)) * 4)?,
                    streams: streams.to_vec(),
                    pointers: devices.iter().map(|_| AtomicU64::new(0)).collect(),
                    arrived: AtomicUsize::new(0),
                    generation: AtomicU64::new(0),
                    posted: counters().collect(),
                    consumed: counters().collect(),
                });
            }
            Split::Layer => {
                let mesh = Mesh::new(devices, 1)?;
                for hop in 0..devices.len().saturating_sub(1) {
                    lanes.push(LaneLink {
                        lane: Lane::new(&mesh, hop, hop + 1, spec.lane_slots, spec.lane_values)?,
                        state: Mutex::new(LaneState {
                            sent: VecDeque::with_capacity(spec.lane_slots as usize),
                            received: VecDeque::with_capacity(spec.lane_slots as usize),
                        }),
                    });
                }
            }
        }
        Ok(Self { collective, lanes, abort: AtomicBool::new(false) })
    }

    /// Makes every thread that waits in a rendezvous or for a parcel give up
    /// with [`ModelError::Aborted`], and every later wait too.
    pub(crate) fn abort(&self) {
        self.abort.store(true, Ordering::SeqCst);
    }

    /// Waits, spinning, until `done` holds. Yields the core after a while so
    /// that a partner that was descheduled can run.
    fn wait_until(&self, mut done: impl FnMut() -> bool) -> Result<(), ModelError> {
        let mut spins = 0u32;
        loop {
            if done() {
                return Ok(());
            }
            if self.abort.load(Ordering::Relaxed) {
                return Err(ModelError::Aborted);
            }
            spins += 1;
            if spins < 2000 {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
    }

    /// Sums `n` f32 values across all devices: afterwards the `n` values at
    /// `partial` on every device hold the sum of all devices' partials. Every
    /// device thread must call it, in the same order, with the same `n`. It
    /// returns when the sum is enqueued on every stream, not when it has run.
    pub fn sum(&self, rank: usize, partial: DevicePtr, n: u32) -> Result<(), ModelError> {
        let Some(c) = &self.collective else {
            return Err(ModelError::Invalid("this model is not split by tensor".into()));
        };
        let devices = c.streams.len();
        let generation = c.generation.load(Ordering::Acquire);
        c.pointers[rank].store(partial.as_raw(), Ordering::Release);
        if c.arrived.fetch_add(1, Ordering::AcqRel) + 1 < devices {
            self.wait_until(|| c.generation.load(Ordering::Acquire) != generation)?;
            // The last thread bumps the generation also when its sum failed.
            return if self.abort.load(Ordering::Acquire) {
                Err(ModelError::Aborted)
            } else {
                Ok(())
            };
        }
        // The last to arrive: every partial is published, and the others
        // are spinning on the generation.
        let stream_refs: [&Stream; MAX_DEVICES] = std::array::from_fn(|i| &*c.streams[i % devices]);
        // `DevicePtr` has no constructor from a number; an offset from null is one.
        let pointers: [DevicePtr; MAX_DEVICES] = std::array::from_fn(|i| {
            DevicePtr::NULL.offset(c.pointers[i % devices].load(Ordering::Acquire))
        });
        // SAFETY: each stream belongs to its device of the mesh and each
        // pointer addresses `n` f32 values of that device, published by the
        // thread that owns them and alive until the step has run. Only the
        // thread that completed the rendezvous gets here, and the next sum
        // starts only after the generation changes below, so the sums of the
        // mesh are enqueued one at a time.
        let result = unsafe { c.mesh.sum_f32(&stream_refs[..devices], &pointers[..devices], n) };
        c.arrived.store(0, Ordering::Release);
        if result.is_err() {
            self.abort();
        }
        c.generation.fetch_add(1, Ordering::Release);
        result.map_err(ModelError::from)
    }

    fn collective(&self) -> Result<&Collective, ModelError> {
        self.collective
            .as_ref()
            .ok_or_else(|| ModelError::Invalid("this model is not split by tensor".into()))
    }

    /// The bytes one pushed sum may write per device, or 0 when the devices
    /// have no push path.
    pub fn push_bytes(&self) -> u64 {
        self.collective.as_ref().map_or(0, |c| c.mesh.push_bytes())
    }

    /// The next pushed sum of device `rank`, for its merge kernel. Every
    /// device thread takes the pushes of a step in the same order; the
    /// threads need not wait for each other.
    pub fn push(&self, rank: usize) -> Result<PushSlot, ModelError> {
        self.collective()?
            .mesh
            .push(rank)
            .ok_or_else(|| ModelError::Invalid("the devices have no push path".into()))
    }

    /// The bytes of a slot: the stride of the regions [`Links::arrive`]
    /// returns.
    pub fn slot_bytes(&self) -> u64 {
        self.collective.as_ref().map_or(0, |c| c.mesh.slot_bytes())
    }

    /// Where device `rank` writes its next parcel of `slot`; what its stream
    /// enqueues next waits until the slot's last parcel has left.
    pub fn outbox(&self, rank: usize, slot: u32) -> Result<DevicePtr, ModelError> {
        let c = self.collective()?;
        Ok(c.mesh.outbox(rank, &c.streams[rank], slot)?)
    }

    /// Sends `bytes` bytes of device `rank`'s outbox of `slot` to the other
    /// devices, after what its stream has enqueued so far. The stream does
    /// not wait for the copy. Waits on the host until every other device
    /// thread has ended its use of the slot's last arrival.
    pub fn post(&self, rank: usize, slot: u32, bytes: u64) -> Result<(), ModelError> {
        let c = self.collective()?;
        let devices = c.streams.len();
        let sent = c.posted[c.slot(rank, slot)].load(Ordering::Acquire);
        self.wait_until(|| {
            (0..devices)
                .filter(|p| *p != rank)
                .all(|p| c.consumed[c.slot(p, slot)].load(Ordering::Acquire) >= sent)
        })?;
        // SAFETY: the stream is device `rank`'s own, on which the caller
        // wrote the outbox; every other device's `consumed` of the slot's
        // last parcel has returned (the wait above).
        let result = unsafe { c.mesh.post(rank, &c.streams[rank], slot, bytes) };
        if result.is_err() {
            self.abort();
        }
        c.posted[c.slot(rank, slot)].store(sent + 1, Ordering::Release);
        Ok(result?)
    }

    /// Makes device `rank`'s stream wait for the other devices' parcels of
    /// `slot` and returns where they are, one region per other device in
    /// device order, until [`Links::consumed`]. Waits on the host until every
    /// other device thread has posted its parcel.
    pub fn arrive(&self, rank: usize, slot: u32) -> Result<DevicePtr, ModelError> {
        let c = self.collective()?;
        let devices = c.streams.len();
        let used = c.consumed[c.slot(rank, slot)].load(Ordering::Acquire);
        self.wait_until(|| {
            (0..devices)
                .filter(|p| *p != rank)
                .all(|p| c.posted[c.slot(p, slot)].load(Ordering::Acquire) > used)
        })?;
        // SAFETY: the stream is device `rank`'s own; every other device's
        // post of this parcel has returned (the wait above), and none posts
        // the slot again before this device's `consumed`.
        Ok(unsafe { c.mesh.arrive(rank, &c.streams[rank], slot)? })
    }

    /// Ends device `rank`'s use of the parcels of `slot` that arrived: what
    /// its stream has enqueued so far has read them.
    pub fn consumed(&self, rank: usize, slot: u32) -> Result<(), ModelError> {
        let c = self.collective()?;
        let result = c.mesh.consumed(rank, &c.streams[rank], slot);
        if result.is_err() {
            self.abort();
        }
        c.consumed[c.slot(rank, slot)].fetch_add(1, Ordering::AcqRel);
        Ok(result?)
    }

    /// Enqueues, on `stream` of the sending device of hop `hop`, the hand-over
    /// of `n` f32 values at `src` to the next device, and queues the parcel
    /// for [`Links::receive`].
    pub fn send(
        &self,
        hop: usize,
        stream: &Stream,
        src: DevicePtr,
        n: u32,
    ) -> Result<(), ModelError> {
        let link = &self.lanes[hop];
        let mut state = link.state.lock().expect("lane state");
        // SAFETY: `stream` is the sending device's stream and `src` addresses
        // `n` values on it, alive until the step has run (the caller's
        // buffers). Lane calls are serialised by the lock.
        let parcel = unsafe { link.lane.send_f32(stream, src, n)? };
        state.sent.push_back(parcel);
        Ok(())
    }

    /// Enqueues, on `stream` of the receiving device, the wait for the oldest
    /// parcel of hop `hop` and returns where its values are on that device
    /// until [`Links::release`]. Waits on the host until the sender has
    /// enqueued that parcel.
    pub fn receive(&self, hop: usize, stream: &Stream) -> Result<DevicePtr, ModelError> {
        let link = &self.lanes[hop];
        let mut received = None;
        self.wait_until(|| {
            let mut state = link.state.lock().expect("lane state");
            let Some(parcel) = state.sent.pop_front() else { return false };
            // SAFETY: `stream` is the receiving device's stream and the
            // parcel came from this lane. Lane calls are serialised by the
            // lock.
            let ptr = unsafe { link.lane.receive(stream, &parcel) };
            received = Some(ptr.inspect(|_| state.received.push_back(parcel)));
            true
        })?;
        Ok(received.expect("set when the wait ended")?)
    }

    /// Enqueues, on `stream`, the end of the use of the oldest received
    /// parcel of hop `hop`: the slot may be sent into again after the work
    /// enqueued so far has read it.
    pub fn release(&self, hop: usize, stream: &Stream) -> Result<(), ModelError> {
        let link = &self.lanes[hop];
        let mut state = link.state.lock().expect("lane state");
        let parcel = state.received.pop_front().ok_or_else(|| {
            ModelError::Invalid("a parcel is released that was not received".into())
        })?;
        Ok(link.lane.release(stream, parcel)?)
    }
}
