//! Streams and events: how work is ordered on a device and across devices.
//!
//! A stream is a queue of work that the device runs in order. Enqueuing
//! returns at once, so one host thread can keep several devices busy. An
//! event marks a point in a stream; another stream, on the same device or a
//! different one, can wait for it. That is the mechanism behind a multi-GPU
//! step that never returns to the host between layers: each device's stream
//! is filled for the whole step, with waits on the other device's events
//! where partial results cross over.

use crate::device::Device;
use crate::error::Result;
use crate::memory::DevicePtr;
use crate::sys::*;

/// A queue of work on one device, destroyed when dropped.
#[derive(Debug)]
pub struct Stream {
    device: Device,
    raw: CUstream,
}

// SAFETY: a stream handle may be used from any thread; the driver serialises
// access to it.
unsafe impl Send for Stream {}
// SAFETY: as above.
unsafe impl Sync for Stream {}

impl Stream {
    /// A new stream. Among streams with pending work the device prefers lower
    /// `priority` values; 0 is the default. A decode stream that should not
    /// wait behind a prefill stream gets a lower number than it.
    pub fn new(device: &Device, priority: i32) -> Result<Self> {
        let driver = device.bind()?;
        let mut raw: CUstream = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the write of one handle.
        driver.check("cuStreamCreateWithPriority", unsafe {
            (driver.api.stream_create)(&mut raw, CU_STREAM_NON_BLOCKING, priority)
        })?;
        Ok(Self { device: device.clone(), raw })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn raw(&self) -> CUstream {
        self.raw
    }

    /// Waits until all work enqueued so far has finished.
    pub fn synchronize(&self) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: the stream is alive while `self` is.
        driver.check("cuStreamSynchronize", unsafe { (driver.api.stream_synchronize)(self.raw) })
    }

    /// Whether all work enqueued so far has finished, without waiting.
    pub fn is_idle(&self) -> Result<bool> {
        let driver = self.device.bind()?;
        // SAFETY: the stream is alive while `self` is.
        let code = unsafe { (driver.api.stream_query)(self.raw) };
        if code == CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        driver.check("cuStreamQuery", code).map(|()| true)
    }

    /// Makes work enqueued after this call wait until `event` has happened.
    /// The event may belong to another device.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: both handles are alive; flags must be 0.
        driver.check("cuStreamWaitEvent", unsafe {
            (driver.api.stream_wait_event)(self.raw, event.raw, 0)
        })
    }

    /// Enqueues a copy from host memory to the device.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes and stay valid and
    /// unchanged until the copy has run (the stream has passed this point).
    /// It should be page-locked memory; ordinary memory makes the call block.
    /// `dst` must address `len` bytes of live memory on this stream's device.
    pub unsafe fn copy_from_host(&self, dst: DevicePtr, src: *const u8, len: usize) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: guaranteed by the caller.
        driver.check("cuMemcpyHtoDAsync", unsafe {
            (driver.api.memcpy_htod_async)(dst.0, src.cast(), len, self.raw)
        })
    }

    /// Enqueues a copy from the device to host memory.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes, stay valid until the
    /// copy has run, and not be read before then. It should be page-locked
    /// memory. `src` must address `len` bytes of live memory on this stream's
    /// device.
    pub unsafe fn copy_to_host(&self, dst: *mut u8, src: DevicePtr, len: usize) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: guaranteed by the caller.
        driver.check("cuMemcpyDtoHAsync", unsafe {
            (driver.api.memcpy_dtoh_async)(dst.cast(), src.0, len, self.raw)
        })
    }

    /// Enqueues a copy within this stream's device.
    ///
    /// # Safety
    ///
    /// Both ranges must be live memory of this stream's device and must not
    /// overlap.
    pub unsafe fn copy(&self, dst: DevicePtr, src: DevicePtr, len: usize) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: guaranteed by the caller.
        driver.check("cuMemcpyDtoDAsync", unsafe {
            (driver.api.memcpy_dtod_async)(dst.0, src.0, len, self.raw)
        })
    }

    /// Enqueues a copy between any two addresses of the unified address space:
    /// device memory of this stream's device, or page-locked host memory that
    /// is mapped for it ([`crate::HostRegistration::mapped`]). The driver
    /// finds out from the addresses which way the bytes go.
    ///
    /// # Safety
    ///
    /// Both ranges must be live memory that this stream's device may access
    /// and must not overlap; host memory must stay valid and unchanged (or
    /// unread, for the destination) until the copy has run.
    pub unsafe fn copy_any(&self, dst: DevicePtr, src: DevicePtr, len: usize) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: guaranteed by the caller.
        driver.check("cuMemcpyAsync", unsafe {
            (driver.api.memcpy_async)(dst.0, src.0, len, self.raw)
        })
    }

    /// Enqueues a copy from `src` on `src_device` to `dst` on `dst_device`,
    /// ordered by this stream. With peer access enabled between the two
    /// devices the data moves directly between them; without it the driver
    /// routes it through host memory.
    ///
    /// # Safety
    ///
    /// Both ranges must be live memory of the device named with them.
    pub unsafe fn copy_peer(
        &self,
        dst_device: &Device,
        dst: DevicePtr,
        src_device: &Device,
        src: DevicePtr,
        len: usize,
    ) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: guaranteed by the caller; both contexts are alive.
        driver.check("cuMemcpyPeerAsync", unsafe {
            (driver.api.memcpy_peer_async)(
                dst.0,
                dst_device.context(),
                src.0,
                src_device.context(),
                len,
                self.raw,
            )
        })
    }

    /// Enqueues filling `len` bytes at `dst` with `value`.
    ///
    /// # Safety
    ///
    /// `dst` must address `len` bytes of live memory on this stream's device.
    pub unsafe fn fill(&self, dst: DevicePtr, value: u8, len: usize) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: guaranteed by the caller.
        driver.check("cuMemsetD8Async", unsafe {
            (driver.api.memset_d8_async)(dst.0, value, len, self.raw)
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the handle came from cuStreamCreate and is destroyed
            // once; the driver lets pending work finish first.
            unsafe { (driver.api.stream_destroy)(self.raw) };
        }
    }
}

/// A marker that records when a stream reaches a point, destroyed when
/// dropped.
#[derive(Debug)]
pub struct Event {
    device: Device,
    raw: CUevent,
}

// SAFETY: an event handle may be used from any thread.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl Event {
    /// An event for ordering. It keeps no timestamps, which makes recording
    /// it cheaper; use [`Event::timed`] to measure.
    pub fn new(device: &Device) -> Result<Self> {
        Self::create(device, CU_EVENT_DISABLE_TIMING)
    }

    /// An event that also records a device timestamp.
    pub fn timed(device: &Device) -> Result<Self> {
        Self::create(device, 0)
    }

    fn create(device: &Device, flags: u32) -> Result<Self> {
        let driver = device.bind()?;
        let mut raw: CUevent = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the write of one handle.
        driver.check("cuEventCreate", unsafe { (driver.api.event_create)(&mut raw, flags) })?;
        Ok(Self { device: device.clone(), raw })
    }

    /// Marks the current end of `stream`: the event has happened once the
    /// work enqueued before this call has finished. The stream must be on
    /// this event's device.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: both handles are alive.
        driver.check("cuEventRecord", unsafe { (driver.api.event_record)(self.raw, stream.raw) })
    }

    /// Whether the event has happened, without waiting.
    pub fn is_done(&self) -> Result<bool> {
        let driver = self.device.bind()?;
        // SAFETY: the handle is alive.
        let code = unsafe { (driver.api.event_query)(self.raw) };
        if code == CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        driver.check("cuEventQuery", code).map(|()| true)
    }

    /// Waits until the event has happened.
    pub fn synchronize(&self) -> Result<()> {
        let driver = self.device.bind()?;
        // SAFETY: the handle is alive.
        driver.check("cuEventSynchronize", unsafe { (driver.api.event_synchronize)(self.raw) })
    }

    /// Milliseconds of device time between two timed events that have both
    /// happened on this event's device.
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        let driver = self.device.bind()?;
        let mut ms = 0f32;
        // SAFETY: the pointer is valid for one float; both handles are alive.
        driver.check("cuEventElapsedTime", unsafe {
            (driver.api.event_elapsed_time)(&mut ms, start.raw, self.raw)
        })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the handle came from cuEventCreate and is destroyed once.
            unsafe { (driver.api.event_destroy)(self.raw) };
        }
    }
}
