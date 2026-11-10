//! Streams and events: how work is ordered on a device.
//!
//! A stream is a queue of work that the device runs in order, as in the CUDA
//! layer: enqueuing returns at once, and `synchronize` waits for everything
//! enqueued so far.
//!
//! # How commands are recorded and submitted
//!
//! Vulkan makes the host build command buffers and hand them to a queue; a
//! submission costs tens of microseconds, a recorded dispatch about one. A
//! step of the engine is thousands of small kernels, so submitting each would
//! cost more than running it. A stream therefore **records** every operation
//! into an open command buffer and **submits** the buffer in batches:
//!
//! - after [`DEFAULT_BATCH`] commands (or the count given to
//!   [`Stream::with_batch`]), so the device starts working on the first part of
//!   a long step while the host is still recording the rest;
//! - when [`flush`](Stream::flush), [`synchronize`](Stream::synchronize),
//!   [`is_idle`](Stream::is_idle), [`Event::record`] or
//!   [`wait_event`](Stream::wait_event) is called, or the stream is dropped.
//!
//! So a launch has no effect on the device until the batch it is in is
//! submitted. Code that enqueues work and then waits on something other than
//! the stream (a flag in mapped memory, another thread) must call `flush`
//! first.
//!
//! Each batch signals the stream's own timeline semaphore with the number of
//! batches submitted so far. That one counter answers everything: whether the
//! stream is idle, whether an event has happened, which command buffers are
//! finished and can be recorded into again (a stream keeps recycling its
//! command buffers and holds at most [`MAX_IN_FLIGHT`] unfinished ones, waiting
//! for the oldest beyond that, so a host that runs far ahead of the device is
//! slowed down instead of allocating without bound).
//!
//! **Order inside a stream.** Vulkan runs commands in no particular order
//! unless told otherwise, and addresses given to kernels are plain numbers, so
//! the stream cannot know which commands depend on which. It puts a global
//! memory barrier before every command: each one sees all the effects of every
//! earlier command of the stream and of everything submitted before it. That
//! is exactly what a CUDA stream guarantees for kernels. The barrier is one
//! recorded command and costs the dispatch no extra submission.
//!
//! **Streams share one queue.** The device is opened with one queue (see
//! [`Device`]), so streams are separate recording and waiting contexts, not
//! concurrent execution: batches run in the order they are submitted, and the
//! barriers also order a stream's commands after earlier batches of other
//! streams. Running a prefill stream next to a decode stream, as the CUDA
//! layer's stream priorities allow, needs a second queue; that is not built
//! yet.
//!
//! # Events
//!
//! An [`Event`] marks a point in a stream; another stream **of the same
//! device** can wait for it. Vulkan has no event that spans two logical devices
//! (an exported timeline semaphore could carry one, which is future work), so
//! unlike the CUDA layer's, a Vulkan event cannot order work between GPUs:
//! the host orders those, with `synchronize` and a staging buffer. Recording an
//! event submits the stream's batch, so that whatever waits for the event can
//! be sure the work it names is on its way.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::device::{Device, lock};
use crate::error::{Check, Error, Result};
use crate::memory::{DevicePtr, StagingBuffer};
use crate::module::{Args, Function};
use crate::transfer::{WORK_STAGES, full_barrier};

/// Commands a stream records before it submits the batch on its own.
pub const DEFAULT_BATCH: u32 = 256;

/// Submitted batches of a stream that may be unfinished at once.
pub const MAX_IN_FLIGHT: usize = 32;

struct State {
    /// The command buffer being recorded, if any.
    open: Option<vk::CommandBuffer>,
    /// Commands in the open buffer.
    commands: u32,
    /// The pipeline bound in the open buffer, to skip rebinding it.
    bound: vk::Pipeline,
    /// Command buffers whose batches have finished.
    free: Vec<vk::CommandBuffer>,
    /// Submitted command buffers and the timeline value that marks each done.
    in_flight: VecDeque<(vk::CommandBuffer, u64)>,
    /// The value of the last submission; the timeline reaches it when
    /// everything submitted has finished.
    submitted: u64,
    /// Events the next submission waits for. The streams are held so that the
    /// semaphores stay alive until the wait has been submitted.
    waits: Vec<(Arc<StreamInner>, u64)>,
}

pub(crate) struct StreamInner {
    device: Device,
    pool: vk::CommandPool,
    timeline: vk::Semaphore,
    batch: u32,
    state: Mutex<State>,
}

impl std::fmt::Debug for StreamInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stream(on {:?})", self.device)
    }
}

impl StreamInner {
    fn create(device: &Device, batch: u32) -> Result<Self> {
        let raw = device.raw();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.physical().queue_family);
        // SAFETY: the device is alive and the family was opened with it.
        let pool =
            unsafe { raw.create_command_pool(&pool_info, None) }.check("vkCreateCommandPool")?;
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the timelineSemaphore feature is enabled on the device.
        let timeline = match unsafe { raw.create_semaphore(&info, None) } {
            Ok(timeline) => timeline,
            Err(code) => {
                // SAFETY: the pool was created above and is unused.
                unsafe { raw.destroy_command_pool(pool, None) };
                return Err(Error::vk("vkCreateSemaphore", code));
            }
        };
        Ok(Self {
            device: device.clone(),
            pool,
            timeline,
            batch: batch.max(1),
            state: Mutex::new(State {
                open: None,
                commands: 0,
                bound: vk::Pipeline::null(),
                free: Vec::new(),
                in_flight: VecDeque::new(),
                submitted: 0,
                waits: Vec::new(),
            }),
        })
    }

    /// Opens a command buffer to record into: a finished one when there is
    /// one, else a new one.
    fn begin(&self, state: &mut State) -> Result<vk::CommandBuffer> {
        let raw = self.device.raw();
        // Hold back a host that is far ahead of the device.
        if state.free.is_empty()
            && state.in_flight.len() >= MAX_IN_FLIGHT
            && let Some((_, value)) = state.in_flight.front()
        {
            self.device.wait_timeline(self.timeline, *value)?;
        }
        let done = self.device.timeline_value(self.timeline)?;
        while let Some((cb, value)) = state.in_flight.front().copied() {
            if value > done {
                break;
            }
            state.in_flight.pop_front();
            state.free.push(cb);
        }
        let cb = match state.free.pop() {
            Some(cb) => cb,
            None => {
                let allocate = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                // SAFETY: the pool is alive and used under the state lock.
                unsafe { raw.allocate_command_buffers(&allocate) }
                    .check("vkAllocateCommandBuffers")?[0]
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the buffer is not recording or pending: it is new, or its
        // batch has finished (the timeline passed its value). The pool is used
        // under the state lock only.
        let begun = unsafe {
            raw.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .and_then(|()| raw.begin_command_buffer(cb, &begin))
        };
        if let Err(code) = begun {
            state.free.push(cb);
            return Err(Error::vk("vkBeginCommandBuffer", code));
        }
        state.open = Some(cb);
        state.commands = 0;
        state.bound = vk::Pipeline::null();
        Ok(cb)
    }

    /// Records one command, preceded by the barrier that orders it after
    /// everything before it. `record` may refuse (a bad address); nothing is
    /// recorded for it then. The batch is submitted when it reaches its size.
    fn record(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer, &mut vk::Pipeline) -> Result<()>,
    ) -> Result<()> {
        let mut state = lock(&self.state);
        self.record_in(&mut state, record)
    }

    fn record_in(
        &self,
        state: &mut State,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer, &mut vk::Pipeline) -> Result<()>,
    ) -> Result<()> {
        let raw = self.device.raw();
        let cb = match state.open {
            Some(cb) => cb,
            None => self.begin(state)?,
        };
        // SAFETY: the command buffer is recording and only this thread, under
        // the state lock, records into it.
        unsafe {
            raw.cmd_pipeline_barrier(
                cb,
                WORK_STAGES,
                WORK_STAGES,
                vk::DependencyFlags::empty(),
                &[full_barrier()],
                &[],
                &[],
            );
        }
        record(raw, cb, &mut state.bound)?;
        state.commands += 1;
        if state.commands >= self.batch {
            self.flush_in(state, false)?;
        }
        Ok(())
    }

    /// Ends the open command buffer and submits it, with the waits that were
    /// asked for since the last submission. With `force` a submission is made
    /// for pending waits even when no command was recorded after them.
    fn flush_in(&self, state: &mut State, force: bool) -> Result<()> {
        let cb = state.open.take();
        if cb.is_none() && !(force && !state.waits.is_empty()) {
            return Ok(());
        }
        state.commands = 0;
        state.bound = vk::Pipeline::null();
        let raw = self.device.raw();
        if let Some(cb) = cb {
            // The device's writes reach the host once the batch is done, for
            // staging buffers the host reads right after `synchronize`.
            let to_host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: the buffer is recording and owned by this stream.
            let ended = unsafe {
                raw.cmd_pipeline_barrier(
                    cb,
                    WORK_STAGES,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[to_host],
                    &[],
                    &[],
                );
                raw.end_command_buffer(cb)
            };
            if let Err(code) = ended {
                state.free.push(cb);
                return Err(Error::vk("vkEndCommandBuffer", code));
            }
        }
        let value = state.submitted + 1;
        let waits: Vec<(vk::Semaphore, u64)> =
            state.waits.iter().map(|(stream, value)| (stream.timeline, *value)).collect();
        let buffers = cb.as_slice();
        if let Err(error) =
            self.device.submit(buffers, &waits, Some((self.timeline, value)), vk::Fence::null())
        {
            state.free.extend(cb);
            return Err(error);
        }
        state.submitted = value;
        state.waits.clear();
        if let Some(cb) = cb {
            state.in_flight.push_back((cb, value));
        }
        Ok(())
    }

    pub(crate) fn flush(&self, force: bool) -> Result<()> {
        let mut state = lock(&self.state);
        self.flush_in(&mut state, force)
    }

    /// Submits everything recorded and waits for it, without holding the
    /// state lock while waiting, so other threads can keep recording.
    pub(crate) fn drain(&self) -> Result<()> {
        let value = {
            let mut state = lock(&self.state);
            self.flush_in(&mut state, true)?;
            state.submitted
        };
        self.device.wait_timeline(self.timeline, value)
    }

    fn is_idle(&self) -> Result<bool> {
        let value = {
            let mut state = lock(&self.state);
            self.flush_in(&mut state, true)?;
            state.submitted
        };
        Ok(self.device.timeline_value(self.timeline)? >= value)
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        let _ = self.drain();
        let raw = self.device.raw();
        // SAFETY: everything submitted has finished (or the device is lost);
        // the objects are destroyed once, and destroying the pool frees the
        // command buffers allocated from it.
        unsafe {
            raw.destroy_semaphore(self.timeline, None);
            raw.destroy_command_pool(self.pool, None);
        }
    }
}

/// A queue of work on one device, destroyed when dropped. Work still recorded
/// when the stream is dropped is submitted first, as CUDA's enqueued work runs
/// after its stream handle is gone.
#[derive(Debug)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

impl Stream {
    /// A new stream that submits every [`DEFAULT_BATCH`] commands.
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_batch(device, DEFAULT_BATCH)
    }

    /// A new stream that submits after every `commands` commands (at least 1).
    /// A small number starts the device sooner and costs a submission per
    /// batch; with 1 every command is submitted at once.
    pub fn with_batch(device: &Device, commands: u32) -> Result<Self> {
        let inner = Arc::new(StreamInner::create(device, commands)?);
        device.register_stream(&inner);
        Ok(Self { inner })
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Submits everything recorded so far to the device, without waiting for
    /// it. Cheap when there is nothing to submit.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush(false)
    }

    /// Waits until all work enqueued so far has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.drain()
    }

    /// Whether all work enqueued so far has finished, without waiting. This
    /// submits what is still recorded first (a stream nobody flushes would
    /// never become idle for a caller that polls).
    pub fn is_idle(&self) -> Result<bool> {
        self.inner.is_idle()
    }

    /// Commands recorded since the last submission.
    pub fn pending_commands(&self) -> u32 {
        lock(&self.inner.state).commands
    }

    /// Batches submitted so far.
    pub fn submissions(&self) -> u64 {
        lock(&self.inner.state).submitted
    }

    /// Makes work enqueued after this call wait until `event` has happened.
    /// The event must be of this stream's device; one that was never recorded
    /// is already done. Work enqueued before this call does not wait.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        if !event.device.same_device(&self.inner.device) {
            return Err(Error::invalid(
                "event",
                "an event can only be waited for on the device it belongs to; \
                 Vulkan has no event between two logical devices",
            ));
        }
        let Some((source, value)) = event.recorded() else { return Ok(()) };
        // Already ordered, or already happened.
        if Arc::ptr_eq(&source, &self.inner)
            || self.inner.device.timeline_value(source.timeline)? >= value
        {
            return Ok(());
        }
        let mut state = lock(&self.inner.state);
        // A wait holds back the whole batch it is submitted with, so the
        // commands recorded before it go out first.
        self.inner.flush_in(&mut state, false)?;
        state.waits.push((source, value));
        Ok(())
    }

    /// Enqueues a copy from a staging buffer to the device: the counterpart of
    /// the CUDA layer's `copy_from_host`, which takes a raw host pointer.
    /// Vulkan's copy commands need the buffer, not an address, on the host
    /// side.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `src_offset` must not change until the copy has run
    /// (the stream has passed this point). `dst` must address `len` bytes
    /// inside one live buffer of this stream's device, which is checked when
    /// the copy is recorded.
    pub unsafe fn copy_from_staging(
        &self,
        dst: DevicePtr,
        src: &StagingBuffer,
        src_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.check_staging(src, src_offset, len)?;
        if len == 0 {
            return Ok(());
        }
        let device = &self.inner.device;
        self.inner.record(|raw, cb, _| {
            let (buffer, offset) = device.resolve(dst, len as u64)?;
            let region = vk::BufferCopy::default()
                .src_offset(src_offset as u64)
                .dst_offset(offset)
                .size(len as u64);
            // SAFETY: both buffers are alive (the destination was just looked
            // up in the table of live buffers, and dropping a buffer drains
            // the stream first) and the ranges were checked.
            unsafe { raw.cmd_copy_buffer(cb, src.buffer(), buffer, &[region]) };
            Ok(())
        })
    }

    /// Enqueues a copy from the device to a staging buffer: the counterpart of
    /// the CUDA layer's `copy_to_host`.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `dst_offset` must not be read or written by the host
    /// until the copy has run. `src` must address `len` bytes inside one live
    /// buffer of this stream's device, which is checked when the copy is
    /// recorded.
    pub unsafe fn copy_to_staging(
        &self,
        dst: &StagingBuffer,
        dst_offset: usize,
        src: DevicePtr,
        len: usize,
    ) -> Result<()> {
        self.check_staging(dst, dst_offset, len)?;
        if len == 0 {
            return Ok(());
        }
        let device = &self.inner.device;
        self.inner.record(|raw, cb, _| {
            let (buffer, offset) = device.resolve(src, len as u64)?;
            let region = vk::BufferCopy::default()
                .src_offset(offset)
                .dst_offset(dst_offset as u64)
                .size(len as u64);
            // SAFETY: as in `copy_from_staging`.
            unsafe { raw.cmd_copy_buffer(cb, buffer, dst.buffer(), &[region]) };
            Ok(())
        })
    }

    /// Enqueues a copy within this stream's device.
    ///
    /// # Safety
    ///
    /// Both ranges must be inside one live buffer each (checked when the copy
    /// is recorded) and must not overlap.
    pub unsafe fn copy(&self, dst: DevicePtr, src: DevicePtr, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let device = &self.inner.device;
        self.inner.record(|raw, cb, _| {
            let (dst_buffer, dst_offset) = device.resolve(dst, len as u64)?;
            let (src_buffer, src_offset) = device.resolve(src, len as u64)?;
            let overlap = dst_buffer == src_buffer
                && dst_offset < src_offset + len as u64
                && src_offset < dst_offset + len as u64;
            if overlap {
                return Err(Error::invalid("device copy", "the ranges overlap"));
            }
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(len as u64);
            // SAFETY: as in `copy_from_staging`.
            unsafe { raw.cmd_copy_buffer(cb, src_buffer, dst_buffer, &[region]) };
            Ok(())
        })
    }

    /// Enqueues filling `len` bytes at `dst` with `value`.
    ///
    /// Vulkan fills whole 32-bit words, so unlike CUDA's `cuMemsetD8Async` this
    /// needs `dst` to be 4-byte aligned within its buffer and `len` a multiple
    /// of 4, and refuses anything else.
    ///
    /// # Safety
    ///
    /// `dst` must address `len` bytes inside one live buffer of this stream's
    /// device, which is checked when the fill is recorded.
    pub unsafe fn fill(&self, dst: DevicePtr, value: u8, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let device = &self.inner.device;
        self.inner.record(|raw, cb, _| {
            let (buffer, offset) = device.resolve(dst, len as u64)?;
            if !offset.is_multiple_of(4) || !len.is_multiple_of(4) {
                return Err(Error::invalid(
                    "device fill",
                    format!("{len} bytes at offset {offset} are not 4-byte aligned"),
                ));
            }
            let word = u32::from_ne_bytes([value; 4]);
            // SAFETY: as in `copy_from_staging`; the alignment was checked.
            unsafe { raw.cmd_fill_buffer(cb, buffer, offset, len as u64, word) };
            Ok(())
        })
    }

    /// Records one dispatch of `function` (see [`Function::launch`]).
    pub(crate) fn dispatch(&self, function: &Function, grid: [u32; 3], args: &Args) -> Result<()> {
        let device = &self.inner.device;
        if !function.device().same_device(device) {
            return Err(Error::invalid(
                "launch",
                "the kernel and the stream are on different devices",
            ));
        }
        let limits = device.physical().max_work_group_count;
        if grid.iter().zip(limits).any(|(groups, limit)| *groups > limit) {
            return Err(Error::invalid(
                "launch",
                format!("a grid of {grid:?} workgroups exceeds the device's limit of {limits:?}"),
            ));
        }
        if grid.contains(&0) {
            return Ok(());
        }
        let pipeline = function.raw();
        let layout = device.0.layout;
        self.inner.record(|raw, cb, bound| {
            // SAFETY: the buffer is recording; the pipeline is alive (the
            // caller holds the function) and uses the device's layout, whose
            // push constant range covers `args`.
            unsafe {
                if *bound != pipeline {
                    raw.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
                    *bound = pipeline;
                }
                raw.cmd_push_constants(
                    cb,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    args.as_bytes(),
                );
                raw.cmd_dispatch(cb, grid[0], grid[1], grid[2]);
            }
            Ok(())
        })
    }

    fn check_staging(&self, staging: &StagingBuffer, offset: usize, len: usize) -> Result<()> {
        if !staging.device().same_device(&self.inner.device) {
            return Err(Error::invalid("staging buffer", "it belongs to another device"));
        }
        match offset.checked_add(len) {
            Some(end) if end <= staging.len() => Ok(()),
            _ => Err(Error::invalid(
                "staging buffer range",
                format!("{len} bytes at {offset} do not fit in {} bytes", staging.len()),
            )),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // An event may keep the stream's internals alive; the work must go out
        // regardless.
        let _ = self.inner.flush(true);
    }
}

struct Recorded {
    stream: Arc<StreamInner>,
    value: u64,
}

/// A marker that records when a stream reaches a point, destroyed when
/// dropped. It works within one device (see the module docs).
#[derive(Debug)]
pub struct Event {
    device: Device,
    recorded: Mutex<Option<Recorded>>,
    /// One timestamp query, for events made with [`Event::timed`].
    query: Option<vk::QueryPool>,
}

impl std::fmt::Debug for Recorded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Recorded(at {})", self.value)
    }
}

impl Event {
    /// An event for ordering. It keeps no timestamp, which makes recording it
    /// cheaper; use [`Event::timed`] to measure.
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self { device: device.clone(), recorded: Mutex::new(None), query: None })
    }

    /// An event that also records a device timestamp.
    pub fn timed(device: &Device) -> Result<Self> {
        if device.physical().timestamp_valid_bits == 0 {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: "its queue does not support timestamps".into(),
            });
        }
        let info =
            vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(1);
        // SAFETY: the device is alive and its queue supports timestamps.
        let pool =
            unsafe { device.raw().create_query_pool(&info, None) }.check("vkCreateQueryPool")?;
        Ok(Self { device: device.clone(), recorded: Mutex::new(None), query: Some(pool) })
    }

    fn recorded(&self) -> Option<(Arc<StreamInner>, u64)> {
        lock(&self.recorded).as_ref().map(|r| (r.stream.clone(), r.value))
    }

    /// Marks the current end of `stream`: the event has happened once the
    /// work enqueued before this call has finished. The stream must be on this
    /// event's device. The stream's batch is submitted.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        let inner = &stream.inner;
        if !self.device.same_device(&inner.device) {
            return Err(Error::invalid("event", "the stream is on another device"));
        }
        let mut state = lock(&inner.state);
        if let Some(pool) = self.query {
            inner.record_in(&mut state, |raw, cb, _| {
                // SAFETY: the buffer is recording; the pool has one query,
                // which is reset before it is written. A recording from before
                // is ordered ahead of the reset by the barrier every command
                // starts with.
                unsafe {
                    raw.cmd_reset_query_pool(cb, pool, 0, 1);
                    raw.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, 0);
                }
                Ok(())
            })?;
        }
        inner.flush_in(&mut state, true)?;
        let value = state.submitted;
        drop(state);
        *lock(&self.recorded) = Some(Recorded { stream: inner.clone(), value });
        Ok(())
    }

    /// Whether the event has happened, without waiting. An event that was
    /// never recorded counts as having happened.
    pub fn is_done(&self) -> Result<bool> {
        match self.recorded() {
            None => Ok(true),
            Some((stream, value)) => Ok(self.device.timeline_value(stream.timeline)? >= value),
        }
    }

    /// Waits until the event has happened.
    pub fn synchronize(&self) -> Result<()> {
        match self.recorded() {
            None => Ok(()),
            Some((stream, value)) => self.device.wait_timeline(stream.timeline, value),
        }
    }

    /// Milliseconds of device time between two timed events that have both
    /// happened on this event's device. Waits for both.
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        let (Some(end_pool), Some(start_pool)) = (self.query, start.query) else {
            return Err(Error::invalid("event", "both events must be made with Event::timed"));
        };
        if !self.device.same_device(&start.device) {
            return Err(Error::invalid("event", "the events are on different devices"));
        }
        if self.recorded().is_none() || start.recorded().is_none() {
            return Err(Error::invalid("event", "both events must have been recorded"));
        }
        self.synchronize()?;
        start.synchronize()?;
        let read = |pool: vk::QueryPool| -> Result<u64> {
            let mut ticks = [0u64];
            // SAFETY: the pool is alive and its query was written by a command
            // that has finished.
            unsafe {
                self.device.raw().get_query_pool_results(
                    pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .check("vkGetQueryPoolResults")?;
            Ok(ticks[0])
        };
        let (end, begin) = (read(end_pool)?, read(start_pool)?);
        let physical = self.device.physical();
        // Only the low `timestamp_valid_bits` of a timestamp mean anything.
        let mask = match physical.timestamp_valid_bits {
            bits if bits >= 64 => u64::MAX,
            bits => (1u64 << bits) - 1,
        };
        let ticks = end.wrapping_sub(begin) & mask;
        Ok((ticks as f64 * f64::from(physical.timestamp_period) / 1e6) as f32)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(pool) = self.query {
            // The query may still be written by a batch in flight.
            let _ = self.synchronize();
            // SAFETY: the pool was created by `timed` and is destroyed once;
            // the batch that wrote it has finished.
            unsafe { self.device.raw().destroy_query_pool(pool, None) };
        }
    }
}
