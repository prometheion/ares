//! Executors without a GPU, for testing the conductor and for measuring it
//! alone.
//!
//! [`HostExecutor`] runs no model. It reads the rows of the step record and
//! answers each with [`SyntheticModel::host_pick`], a fixed function of the
//! row's token and context, so the conductor's planning, page handling,
//! speculation, stopping and event delivery run exactly as with devices.
//! It is not an inference path: its picks mean nothing, and it exists so that
//! every decision the conductor makes can be checked by the default
//! `cargo test`, and so that the benchmark can show what the conductor costs
//! with devices that take no time.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::exec::{Clock, ExecError, Executor, Pick, StepMeta, StepOutput};
use crate::record::RecordView;
use crate::sync::{WakeCounts, Woke};
use crate::synthetic::SyntheticModel;
use crate::timing::DeviceTimes;

/// A latch a test closes to hold steps back and opens to let them finish.
#[derive(Clone, Debug)]
pub struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    /// A gate that starts open.
    pub fn new() -> Self {
        Self(Arc::new((Mutex::new(true), Condvar::new())))
    }

    pub fn close(&self) {
        *self.0.0.lock().expect("gate") = false;
    }

    pub fn open(&self) {
        *self.0.0.lock().expect("gate") = true;
        self.0.1.notify_all();
    }

    fn wait(&self, limit: Duration) -> bool {
        let (lock, condvar) = &*self.0;
        let (_open, timeout) = condvar
            .wait_timeout_while(lock.lock().expect("gate"), limit, |open| !*open)
            .expect("gate");
        !timeout.timed_out()
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

/// What the page tables of a step look like on the device, as the test
/// executor mirrors them: `[slot][entry]`.
pub type TableMirror = Arc<Mutex<Vec<Vec<u32>>>>;

type Observer = Box<dyn FnMut(&RecordView<'_>) + Send>;

/// An executor that computes picks on the host.
pub struct HostExecutor {
    model: Arc<SyntheticModel>,
    devices: usize,
    record: Vec<u8>,
    picks: Vec<Pick>,
    delay: Duration,
    gate: Option<Gate>,
    observer: Option<Observer>,
    dispatched_at: Instant,
    clock: Clock,
    tables: TableMirror,
    fail_at_step: Option<u64>,
    failing: bool,
}

impl std::fmt::Debug for HostExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostExecutor").field("delay", &self.delay).finish_non_exhaustive()
    }
}

impl HostExecutor {
    /// An executor for `model`, which sizes the record for `max_rows` rows and
    /// `max_slots` sequences.
    pub fn new(model: Arc<SyntheticModel>, max_rows: u32, max_slots: u32) -> Self {
        Self {
            model,
            devices: 1,
            record: vec![0; crate::record::max_bytes(max_rows, max_slots)],
            picks: Vec::new(),
            delay: Duration::ZERO,
            gate: None,
            observer: None,
            dispatched_at: Instant::now(),
            clock: Clock::new(),
            tables: Arc::new(Mutex::new(vec![Vec::new(); max_slots as usize])),
            fail_at_step: None,
            failing: false,
        }
    }

    /// Pretends to be `n` devices, for the shape of the timing tables.
    pub fn with_devices(mut self, n: usize) -> Self {
        self.devices = n;
        self
    }

    /// Makes every step take at least `delay`, as a device would.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Holds every step back while `gate` is closed.
    pub fn with_gate(mut self, gate: Gate) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Calls `observer` with the record of every step, before it runs.
    pub fn with_observer(mut self, observer: impl FnMut(&RecordView<'_>) + Send + 'static) -> Self {
        self.observer = Some(Box::new(observer));
        self
    }

    /// Makes the step with this number fail, as a device error would.
    pub fn failing_at(mut self, step: u64) -> Self {
        self.fail_at_step = Some(step);
        self
    }

    /// The page tables as the "device" holds them.
    pub fn tables(&self) -> TableMirror {
        self.tables.clone()
    }
}

impl Executor for HostExecutor {
    fn devices(&self) -> usize {
        self.devices
    }

    fn clock(&self) -> Clock {
        self.clock
    }

    fn record(&mut self) -> &mut [u8] {
        &mut self.record
    }

    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError> {
        let view = RecordView::new(&self.record[..meta.record_bytes])
            .map_err(|e| ExecError::Device { device: 0, message: e.to_string() })?;
        if let Some(observer) = &mut self.observer {
            observer(&view);
        }
        self.failing = self.fail_at_step == Some(meta.step);
        self.picks.clear();
        for i in 0..view.runs() {
            let (first, count) = view.run(i);
            for row in first..first + count {
                let row = view.row(row);
                let (token, logit) = self.model.host_pick(row.token, row.context);
                self.picks.push(Pick { token, logit });
            }
        }
        // The page-table updates, as the device copies them.
        let mut tables = self.tables.lock().expect("tables");
        let mut taken = 0;
        for i in 0..view.updates() {
            let update = view.update(i);
            let table = &mut tables[update.slot as usize];
            table.resize((update.first + update.count) as usize, 0);
            for k in 0..update.count {
                table[(update.first + k) as usize] = view.entry(taken + k);
            }
            taken += update.count;
        }
        self.dispatched_at = Instant::now();
        Ok(())
    }

    fn collect(&mut self, _spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError> {
        if let Some(gate) = &self.gate
            && !gate.wait(Duration::from_secs(20))
        {
            return Err(ExecError::Device {
                device: 0,
                message: "the test never opened the gate".into(),
            });
        }
        if let Some(left) = self.delay.checked_sub(self.dispatched_at.elapsed()) {
            std::thread::sleep(left);
        }
        if self.failing {
            return Err(ExecError::Device { device: 0, message: "injected failure".into() });
        }
        out.picks.clear();
        out.picks.extend_from_slice(&self.picks);
        out.devices.clear();
        let finished_at_ns = self.clock.now_ns();
        out.devices.resize(self.devices, DeviceTimes { finished_at_ns, ..DeviceTimes::default() });
        Ok(Woke::Spinning)
    }

    fn device_wakes(&self) -> WakeCounts {
        WakeCounts::default()
    }

    fn shutdown(&mut self) {}
}
