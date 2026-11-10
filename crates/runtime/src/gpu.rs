//! The executor with a thread per device.
//!
//! Each device has a thread that owns its stream and its part of the model.
//! The conductor writes the step record once, in page-locked memory, and hands
//! every device thread a small message; each thread then
//!
//! 1. enqueues the upload of the record (one copy) and the changes to the
//!    sequences' page tables,
//! 2. calls the model, which enqueues all of its kernels for the whole step,
//! 3. enqueues the read-back of the logits that are wanted,
//! 4. waits once, for the end of the step on its stream, and
//! 5. takes the arg-max of each row it brought back.
//!
//! Between 1 and 4 the thread never waits for the device. Between devices the
//! order is kept on the devices (events, peer kernels); the only host-side
//! meeting is [`Links`], which exists because `ares-cuda`'s sum and
//! hand-over are single calls over several streams.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ares_core::units::MIB;
use ares_core::{Broker, Class, Growth, Ledger};
use ares_cuda::{Device, DeviceBuffer, Event, Kernels, PinnedBuffer, Stream};

use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::exec::{Clock, ExecError, Executor, Pick, StepMeta, StepOutput};
use crate::links::Links;
use crate::model::{DeviceContext, DeviceModel, Model, ModelError, StepContext};
use crate::record::{self, RecordView};
use crate::sync::{Bell, WakeCounts, Woke};
use crate::timing::DeviceTimes;

/// How the device threads are set up.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// The most sequences at once (page tables are kept per slot).
    pub max_slots: u32,
    /// Page-table entries per slot: the longest sequence in pages.
    pub table_stride: u32,
    /// How long a device thread spins for the next step before it parks.
    pub spin: Duration,
    /// VRAM left unreserved on each device.
    pub vram_headroom: u64,
    /// The most page-locked RAM the executor takes.
    pub pinned_limit: u64,
    /// The pool's pages, and the bytes of state they take on each device
    /// (the pages times the model's bytes per page there): reserved in the
    /// device's ledger and allocated before the model is built.
    pub pages: u32,
    pub page_bytes: Vec<u64>,
}

impl GpuConfig {
    /// The defaults for a runtime of `config` over `model`: page tables for
    /// every slot, long enough for the model's whole context.
    pub fn new(config: &RuntimeConfig, model: &dyn Model) -> Self {
        Self {
            max_slots: config.max_slots,
            table_stride: model.card().context_length.div_ceil(config.cells_per_page),
            spin: Duration::from_millis(2),
            vram_headroom: 256 * MIB,
            pinned_limit: 256 * MIB,
            pages: config.pages,
            page_bytes: {
                let layout = model.page_layout(config.cells_per_page);
                (0..model.devices())
                    .map(|d| layout.bytes_per_page(d) * u64::from(config.pages))
                    .collect()
            },
        }
    }
}

/// The step record in page-locked memory, written by the conductor and read
/// by the device threads.
struct PinnedRecord {
    _owner: PinnedBuffer,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is owned by this value and lives as long as it. Access
// follows the executor's protocol: the conductor writes only while no step
// is in flight, the device threads read only while one is.
unsafe impl Send for PinnedRecord {}
// SAFETY: as above.
unsafe impl Sync for PinnedRecord {}

impl PinnedRecord {
    fn new(device: &Device, len: usize) -> Result<Self, ares_cuda::Error> {
        let mut owner = PinnedBuffer::new(device, len)?;
        let ptr = owner.as_mut_ptr();
        Ok(Self { _owner: owner, ptr, len })
    }
}

enum Job {
    Step { meta: StepMeta, dispatched_ns: u64 },
    Quit,
}

/// What a device thread leaves for the conductor.
#[derive(Default)]
struct DeviceResult {
    error: Option<String>,
    times: DeviceTimes,
    /// One pick per wanted row, when this device has an output head.
    picks: Vec<Pick>,
    has_picks: bool,
}

/// The hand-over between the conductor and one device thread.
struct Mailbox {
    bell: Bell,
    job: Mutex<Option<Job>>,
    result: Mutex<DeviceResult>,
    spun: AtomicU64,
    parked: AtomicU64,
}

struct Shared {
    /// Rung by the last device thread to finish a step.
    done: Bell,
    remaining: AtomicUsize,
    clock: Clock,
}

struct Handle {
    mailbox: Arc<Mailbox>,
    thread: Option<JoinHandle<()>>,
}

/// An executor with a thread per device.
pub struct GpuExecutor {
    handles: Vec<Handle>,
    shared: Arc<Shared>,
    record: Arc<PinnedRecord>,
    links: Arc<Links>,
    seen: u64,
    in_flight: bool,
    /// Held so the page-locked memory stays accounted for as long as it exists.
    _broker: Broker,
}

impl std::fmt::Debug for GpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuExecutor").field("devices", &self.handles.len()).finish_non_exhaustive()
    }
}

/// What a device thread is built from.
struct Init {
    index: usize,
    model: Arc<dyn Model>,
    device: Device,
    stream: Arc<Stream>,
    kernels: Kernels,
    links: Arc<Links>,
    record: Arc<PinnedRecord>,
    mailbox: Arc<Mailbox>,
    shared: Arc<Shared>,
    config: GpuConfig,
    max_rows: u32,
}

impl GpuExecutor {
    /// Opens the model on `devices` (one part per device) and starts a thread
    /// for each. Returns when every part is built, or with the first error.
    pub fn start(
        model: Arc<dyn Model>,
        devices: Vec<Device>,
        config: GpuConfig,
    ) -> Result<Self, RuntimeError> {
        if devices.len() != model.devices() {
            return Err(RuntimeError::Config(format!(
                "the model spans {} devices and {} were given",
                model.devices(),
                devices.len()
            )));
        }
        let max_rows = model.max_rows();
        let record_bytes = record::max_bytes(max_rows, config.max_slots);

        // Page-locked memory is the host's scarcest kind; it is reserved
        // before it is taken.
        let mut broker = Broker::new("pinned RAM", config.pinned_limit);
        let consumer = broker.register("step record and read-back", Class::Required)?;
        let readback = devices.len() as u64 * u64::from(max_rows) * u64::from(model.vocab()) * 4;
        match broker.grow(&consumer, record_bytes as u64 + readback)? {
            Growth::Granted => {}
            Growth::Reclaim(_) => unreachable!("nobody else holds page-locked memory"),
        }

        let streams = devices
            .iter()
            .map(|d| Stream::new(d, 0).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let links = Arc::new(Links::new(model.split(), &devices, &streams, model.links())?);
        let record = Arc::new(PinnedRecord::new(&devices[0], record_bytes)?);
        let shared = Arc::new(Shared {
            done: Bell::new(),
            remaining: AtomicUsize::new(0),
            clock: Clock::new(),
        });

        let (ready_tx, ready_rx) = mpsc::channel();
        let mut handles = Vec::new();
        for (index, (device, stream)) in devices.into_iter().zip(streams).enumerate() {
            let mailbox = Arc::new(Mailbox {
                bell: Bell::new(),
                job: Mutex::new(None),
                result: Mutex::new(DeviceResult {
                    picks: Vec::with_capacity(max_rows as usize),
                    ..DeviceResult::default()
                }),
                spun: AtomicU64::new(0),
                parked: AtomicU64::new(0),
            });
            let kernels = Kernels::load(&device)?;
            let init = Init {
                index,
                model: model.clone(),
                device,
                stream,
                kernels,
                links: links.clone(),
                record: record.clone(),
                mailbox: mailbox.clone(),
                shared: shared.clone(),
                config: config.clone(),
                max_rows,
            };
            let ready = ready_tx.clone();
            let thread = thread::Builder::new()
                .name(format!("ares-device-{index}"))
                .spawn(move || worker_main(init, ready))
                .map_err(|e| RuntimeError::Thread(format!("cannot start a device thread: {e}")))?;
            handles.push(Handle { mailbox, thread: Some(thread) });
        }
        drop(ready_tx);

        let mut executor =
            Self { handles, shared, record, links, seen: 0, in_flight: false, _broker: broker };
        let mut failure = None;
        for _ in 0..executor.handles.len() {
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => failure = failure.or(Some(error)),
                Err(_) => {
                    failure = failure.or(Some(RuntimeError::Thread(
                        "a device thread ended while starting".into(),
                    )));
                }
            }
        }
        if let Some(error) = failure {
            executor.shutdown();
            return Err(error);
        }
        Ok(executor)
    }
}

impl Executor for GpuExecutor {
    fn devices(&self) -> usize {
        self.handles.len()
    }

    fn clock(&self) -> Clock {
        self.shared.clock
    }

    fn record(&mut self) -> &mut [u8] {
        assert!(!self.in_flight, "the record is written between collect and dispatch");
        // SAFETY: no step is in flight, so no device thread reads the record,
        // and `&mut self` makes this the only access from the conductor.
        unsafe { std::slice::from_raw_parts_mut(self.record.ptr, self.record.len) }
    }

    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError> {
        assert!(!self.in_flight, "a step is already running");
        self.shared.remaining.store(self.handles.len(), Ordering::SeqCst);
        let dispatched_ns = self.shared.clock.now_ns();
        for handle in &self.handles {
            *handle.mailbox.job.lock().expect("job slot") =
                Some(Job::Step { meta: *meta, dispatched_ns });
            handle.mailbox.bell.ring();
        }
        self.in_flight = true;
        Ok(())
    }

    fn collect(&mut self, spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError> {
        assert!(self.in_flight, "no step is running");
        // The waiter is the thread that collects, not the one that built the
        // executor; the first call from it is what binds the bell, later
        // calls find it bound.
        self.shared.done.bind();
        let (count, woke) = self.shared.done.wait_past(self.seen, spin);
        self.seen = count;
        self.in_flight = false;
        out.picks.clear();
        out.devices.clear();
        let mut have_picks = false;
        for (device, handle) in self.handles.iter().enumerate() {
            let result = handle.mailbox.result.lock().expect("result slot");
            if let Some(message) = &result.error {
                return Err(ExecError::Device { device, message: message.clone() });
            }
            out.devices.push(result.times);
            if result.has_picks {
                if have_picks {
                    merge_picks(&mut out.picks, &result.picks);
                } else {
                    out.picks.extend_from_slice(&result.picks);
                    have_picks = true;
                }
            }
        }
        Ok(woke)
    }

    fn device_wakes(&self) -> WakeCounts {
        self.handles.iter().fold(WakeCounts::default(), |sum, h| WakeCounts {
            spun: sum.spun + h.mailbox.spun.load(Ordering::Relaxed),
            parked: sum.parked + h.mailbox.parked.load(Ordering::Relaxed),
        })
    }

    fn shutdown(&mut self) {
        // Let any thread that waits on another give up first, so that a step
        // that failed on one device does not keep the others from quitting.
        self.links.abort();
        for handle in &self.handles {
            *handle.mailbox.job.lock().expect("job slot") = Some(Job::Quit);
            handle.mailbox.bell.ring();
        }
        for handle in &mut self.handles {
            if let Some(thread) = handle.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

impl Drop for GpuExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The best pick of two devices that each hold a slice of the vocabulary:
/// the higher logit, the lower token on a tie.
fn merge_picks(into: &mut [Pick], other: &[Pick]) {
    assert_eq!(into.len(), other.len(), "devices bring back the same rows");
    for (mine, theirs) in into.iter_mut().zip(other) {
        if theirs.logit > mine.logit || (theirs.logit == mine.logit && theirs.token < mine.token) {
            *mine = *theirs;
        }
    }
}

/// The arg-max of one row of logits. A value that is not finite means a sum
/// or a kernel went wrong, and is an error rather than a pick.
fn arg_max(row: &[u8], first_token: u32) -> Result<Pick, String> {
    let mut best = Pick { token: first_token, logit: f32::NEG_INFINITY };
    for (i, bytes) in row.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*bytes);
        if !value.is_finite() {
            return Err(format!("logit {} is {value}", first_token as usize + i));
        }
        if value > best.logit {
            best = Pick { token: first_token + i as u32, logit: value };
        }
    }
    Ok(best)
}

// --- the device thread ---------------------------------------------------

struct Worker {
    index: usize,
    devices: usize,
    shared: Arc<Shared>,
    device: Device,
    stream: Arc<Stream>,
    kernels: Kernels,
    links: Arc<Links>,
    model: Box<dyn DeviceModel>,
    record: Arc<PinnedRecord>,
    record_device: DeviceBuffer,
    tables: DeviceBuffer,
    table_stride: u32,
    max_slots: u32,
    /// The state pages, if the model keeps any.
    pages: Option<DeviceBuffer>,
    bytes_per_page: u64,
    logits_host: Option<PinnedBuffer>,
    start: Event,
    ends: [Event; 2],
    /// Which of `ends` the next step records.
    next_end: usize,
    have_previous: bool,
    clock: Clock,
    picks: Vec<Pick>,
}

fn worker_main(init: Init, ready: mpsc::Sender<Result<(), RuntimeError>>) {
    init.mailbox.bell.bind();
    let mailbox = init.mailbox.clone();
    let spin = init.config.spin;
    let mut worker = match Worker::new(init) {
        Ok(worker) => {
            let _ = ready.send(Ok(()));
            worker
        }
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    drop(ready);

    let mut seen = 0;
    loop {
        let (count, woke) = mailbox.bell.wait_past(seen, spin);
        seen = count;
        match woke {
            Woke::Spinning => mailbox.spun.fetch_add(1, Ordering::Relaxed),
            Woke::Parked => mailbox.parked.fetch_add(1, Ordering::Relaxed),
        };
        let job = mailbox.job.lock().expect("job slot").take();
        match job {
            Some(Job::Step { meta, dispatched_ns }) => {
                let outcome = worker.run_step(&meta, dispatched_ns);
                {
                    let mut result = mailbox.result.lock().expect("result slot");
                    match outcome {
                        Ok((times, produced)) => {
                            result.error = None;
                            result.times = times;
                            result.has_picks = produced;
                            std::mem::swap(&mut result.picks, &mut worker.picks);
                        }
                        Err(error) => {
                            result.error = Some(error.to_string());
                            worker.links.abort();
                        }
                    }
                }
                worker.finish();
            }
            Some(Job::Quit) => break,
            // A ring whose job was already taken (a quit racing a step).
            None => {}
        }
    }
    // Everything this thread enqueued must have run before its buffers go.
    let _ = worker.stream.synchronize();
}

impl Worker {
    fn new(init: Init) -> Result<Self, RuntimeError> {
        let Init {
            index,
            model,
            device,
            stream,
            kernels,
            links,
            record,
            shared,
            config,
            max_rows,
            ..
        } = init;
        let devices = model.devices();
        let free = device.mem_info()?.free;
        let mut ledger = Ledger::new(
            format!("CUDA{}", device.ordinal()),
            free.saturating_sub(config.vram_headroom),
        );
        let record_len = record::max_bytes(max_rows, config.max_slots) as u64;
        let tables_len = u64::from(config.max_slots) * u64::from(config.table_stride) * 4;
        ledger.reserve("step record (device copy)", record_len)?;
        ledger.reserve("page tables", tables_len)?;
        let page_bytes = config.page_bytes.get(index).copied().unwrap_or(0);
        if page_bytes > 0 {
            ledger.reserve("state pages", page_bytes)?;
        }
        let part = model
            .open(DeviceContext {
                index,
                devices,
                device: &device,
                kernels: &kernels,
                stream: &stream,
                ledger: &mut ledger,
                max_rows,
            })
            .map_err(|e| match e {
                ModelError::Memory(e) => RuntimeError::Memory(e),
                other => RuntimeError::Model(other),
            })?;
        let per_row = part.logits_per_row();
        let logits_host = (per_row > 0)
            .then(|| PinnedBuffer::new(&device, max_rows as usize * per_row as usize * 4))
            .transpose()?;
        let worker = Self {
            index,
            devices,
            clock: shared.clock,
            shared,
            record_device: DeviceBuffer::new(&device, record_len)?,
            tables: DeviceBuffer::new(&device, tables_len.max(16))?,
            table_stride: config.table_stride,
            max_slots: config.max_slots,
            pages: (page_bytes > 0).then(|| DeviceBuffer::new(&device, page_bytes)).transpose()?,
            bytes_per_page: page_bytes / u64::from(config.pages.max(1)),
            logits_host,
            start: Event::timed(&device)?,
            ends: [Event::timed(&device)?, Event::timed(&device)?],
            next_end: 0,
            have_previous: false,
            picks: Vec::with_capacity(max_rows as usize),
            device,
            stream,
            kernels,
            links,
            model: part,
            record,
        };
        // Page tables are never cleared in normal operation (a table decides
        // what is valid), but a defined start makes a bug show as a wrong
        // page and not as garbage.
        // SAFETY: the buffer is `tables_len` bytes on the stream's device.
        unsafe { worker.stream.fill(worker.tables.ptr(), 0, tables_len as usize)? };
        worker.stream.synchronize()?;
        Ok(worker)
    }

    /// Tells the conductor this device has finished the step; the last one to
    /// finish wakes it.
    fn finish(&self) {
        if self.shared.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.shared.done.ring();
        }
    }

    /// Runs one step on this device. Returns what was measured and whether
    /// this device brought picks back.
    fn run_step(
        &mut self,
        meta: &StepMeta,
        dispatched_ns: u64,
    ) -> Result<(DeviceTimes, bool), ModelError> {
        let mut times = DeviceTimes::default();
        let begun = Instant::now();
        times.wake_ns = self.clock.now_ns().saturating_sub(dispatched_ns);

        // SAFETY: the conductor wrote `meta.record_bytes` bytes of the record
        // before it handed over this step and does not touch the buffer until
        // every device thread has finished it.
        let bytes = unsafe { std::slice::from_raw_parts(self.record.ptr, meta.record_bytes) };
        let view = RecordView::new(bytes)
            .map_err(|e| ModelError::Invalid(format!("the step record is damaged: {e}")))?;

        // The upload: the record, then the changes to the page tables.
        let stream = &*self.stream;
        self.start.record(stream)?;
        // SAFETY: `bytes` is page-locked memory that stays unchanged until the
        // step has finished; the destination holds `record_bytes` bytes.
        unsafe { stream.copy_from_host(self.record_device.ptr(), bytes.as_ptr(), bytes.len())? };
        let entries_at = view.entries_offset_words() as u64;
        let mut taken = 0u64;
        for i in 0..view.updates() {
            let update = view.update(i);
            if update.slot >= self.max_slots || update.first + update.count > self.table_stride {
                return Err(ModelError::Invalid(format!(
                    "page-table update {update:?} is outside {} slots of {} entries",
                    self.max_slots, self.table_stride
                )));
            }
            let to = self.tables.ptr().offset(
                (u64::from(update.slot) * u64::from(self.table_stride) + u64::from(update.first))
                    * 4,
            );
            let from = self.record_device.ptr().offset((entries_at + taken) * 4);
            // SAFETY: both ranges are inside their buffers (checked above and
            // by the record's layout) and do not overlap.
            unsafe { stream.copy(to, from, update.count as usize * 4)? };
            taken += u64::from(update.count);
        }
        times.upload_ns = begun.elapsed().as_nanos() as u64;

        // The model's kernels for the whole step.
        let enqueue_begun = Instant::now();
        let logits = self.model.enqueue(&StepContext {
            index: self.index,
            devices: self.devices,
            device: &self.device,
            kernels: &self.kernels,
            stream,
            record: view,
            record_on_device: self.record_device.ptr(),
            tables: self.tables.ptr(),
            table_stride: self.table_stride,
            pages: self.pages.as_ref().map_or(ares_cuda::DevicePtr::NULL, DeviceBuffer::ptr),
            bytes_per_page: self.bytes_per_page,
            links: &self.links,
        })?;

        // The read-back of the rows whose logits are wanted, run by run.
        if let (Some(logits), Some(host)) = (logits, self.logits_host.as_mut()) {
            let row_bytes = logits.stride as usize * 4;
            let mut compact = 0usize;
            for i in 0..view.runs() {
                let (first, count) = view.run(i);
                // SAFETY: the source rows are inside the model's logits
                // buffer (the record's runs are inside its rows) and the
                // destination inside the read-back buffer, which holds a
                // row per step row; the buffer is not read before the step
                // has finished.
                unsafe {
                    stream.copy_to_host(
                        host.as_mut_ptr().add(compact * row_bytes),
                        logits.ptr.offset(first as u64 * row_bytes as u64),
                        count as usize * row_bytes,
                    )?;
                }
                compact += count as usize;
            }
        }
        let end = &self.ends[self.next_end];
        end.record(stream)?;
        times.enqueue_ns = enqueue_begun.elapsed().as_nanos() as u64;

        // The one wait of the step.
        let wait_begun = Instant::now();
        end.synchronize()?;
        times.readback_ns = wait_begun.elapsed().as_nanos() as u64;

        times.device_ns = (f64::from(end.elapsed_ms_since(&self.start)?) * 1e6) as u64;
        if self.have_previous && !meta.after_idle {
            let previous = &self.ends[1 - self.next_end];
            times.gap_ns =
                self.start.elapsed_ms_since(previous).ok().map(|ms| (f64::from(ms) * 1e6) as u64);
        }
        self.have_previous = true;
        self.next_end = 1 - self.next_end;

        // The arg-max of each row that came back.
        let pick_begun = Instant::now();
        self.picks.clear();
        let produced = logits.is_some() && self.logits_host.is_some();
        if let (Some(logits), Some(host)) = (logits, self.logits_host.as_ref()) {
            let row_bytes = logits.stride as usize * 4;
            for row in 0..view.picked_rows() as usize {
                let pick =
                    arg_max(&host[row * row_bytes..(row + 1) * row_bytes], logits.first_token)
                        .map_err(|e| ModelError::Invalid(format!("row {row}: {e}")))?;
                self.picks.push(pick);
            }
        }
        times.pick_ns = pick_begun.elapsed().as_nanos() as u64;
        times.finished_at_ns = self.clock.now_ns();
        Ok((times, produced))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn the_pick_is_the_largest_logit_and_the_lowest_token_on_a_tie() {
        let pick = arg_max(&bytes(&[0.5, -1.0, 3.0, 3.0, 2.0]), 100).unwrap();
        assert_eq!(pick, Pick { token: 102, logit: 3.0 });
        assert_eq!(arg_max(&bytes(&[-5.0, -2.0]), 0).unwrap().token, 1, "all negative still picks");
    }

    #[test]
    fn a_logit_that_is_not_a_number_is_an_error_and_not_a_pick() {
        let error = arg_max(&bytes(&[1.0, f32::NAN, 2.0]), 10).unwrap_err();
        assert!(error.contains("11") && error.contains("NaN"), "{error}");
        assert!(arg_max(&bytes(&[f32::INFINITY]), 0).is_err());
    }

    #[test]
    fn devices_with_slices_of_the_vocabulary_agree_on_the_best_of_all() {
        // Two devices, three rows. Device 1's slice starts at token 100.
        let mut best = vec![
            Pick { token: 5, logit: 1.0 },
            Pick { token: 7, logit: 2.0 },
            Pick { token: 9, logit: 3.0 },
        ];
        let other = [
            Pick { token: 105, logit: 4.0 },
            Pick { token: 107, logit: 2.0 },
            Pick { token: 109, logit: 0.0 },
        ];
        merge_picks(&mut best, &other);
        assert_eq!(best.iter().map(|p| p.token).collect::<Vec<_>>(), [105, 7, 9]);
        // On an equal logit the lower token wins, whichever device has it.
        let mut best = vec![Pick { token: 105, logit: 2.0 }];
        merge_picks(&mut best, &[Pick { token: 5, logit: 2.0 }]);
        assert_eq!(best[0].token, 5);
    }
}
