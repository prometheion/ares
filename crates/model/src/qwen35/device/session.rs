//! A model on the runtime's executor, driven one step at a time without the
//! conductor: the device threads, the links and the step record are the real
//! ones, the bookkeeping is the least that a few sequences need (every
//! sequence owns its pages from the start). Tests and measurements use it to
//! see the picks and the timing of every step, on one device or several.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_cuda::{Device, ENTRY_CELLS};
use ares_runtime::{
    Commit, Executor, GpuConfig, GpuExecutor, Model, Pick, RecordWriter, Row, RuntimeError,
    StepMeta, StepOutput, TableUpdate,
};

use super::runtime::Qwen35;

/// The rows of one sequence in a step.
#[derive(Clone, Copy, Debug)]
pub struct Rows<'a> {
    /// The sequence's slot.
    pub slot: u32,
    /// The position of the first row.
    pub position: u32,
    pub tokens: &'a [u32],
    /// Whether every row is certain to be accepted; otherwise the rows are a
    /// draft window, and [`Session::accept`] says how many count.
    pub direct: bool,
    /// Whose logits are wanted.
    pub wanted: Wanted,
}

/// Which rows of a sequence's rows get their logits picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wanted {
    None,
    Last,
    All,
}

/// What a step took.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTime {
    /// From handing the step to the devices to having their results.
    pub wall: Duration,
    /// Device time of the slowest device, from the upload of the record to the
    /// end of its last kernel.
    pub device: Duration,
}

/// A model on its devices, stepped by hand.
pub struct Session {
    model: Arc<Qwen35>,
    exec: GpuExecutor,
    out: StepOutput,
    step: u64,
    table_stride: u32,
    /// Per slot: the table entries the devices have.
    entries_sent: Vec<u32>,
    /// The outcome of the last step's draft windows, for the next record.
    commits: Vec<Commit>,
    /// The draft windows of the last step: first row, rows, slot.
    windows: Vec<(u32, u32, u32)>,
    last: StepTime,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("model", &self.model).finish_non_exhaustive()
    }
}

impl Session {
    /// Opens `model` on `devices` for `slots` sequences of up to `cells`
    /// positions each, leaving `headroom` bytes of each device unreserved.
    pub fn start(
        model: Arc<Qwen35>,
        devices: Vec<Device>,
        slots: u32,
        cells: u32,
        headroom: u64,
    ) -> Result<Self, RuntimeError> {
        let table_stride = cells.div_ceil(ENTRY_CELLS).max(1);
        let pages = slots * table_stride;
        let layout = model.page_layout(ENTRY_CELLS);
        let config = GpuConfig {
            max_slots: slots,
            table_stride,
            spin: Duration::from_millis(2),
            vram_headroom: headroom,
            pinned_limit: 1 << 30,
            pages,
            page_bytes: (0..model.devices())
                .map(|d| layout.bytes_per_page(d) * u64::from(pages))
                .collect(),
            cells_per_page: ENTRY_CELLS,
        };
        let exec = GpuExecutor::start(model.clone(), devices, config)?;
        Ok(Self {
            model,
            exec,
            out: StepOutput::default(),
            step: 0,
            table_stride,
            entries_sent: vec![0; slots as usize],
            commits: Vec::new(),
            windows: Vec::new(),
            last: StepTime::default(),
        })
    }

    /// The model.
    pub fn model(&self) -> &Arc<Qwen35> {
        &self.model
    }

    /// Runs one step with the rows of `parts` and returns the picks of the
    /// wanted rows, in order. The draft windows of the step before must have
    /// been settled with [`Session::accept`].
    pub fn step(&mut self, parts: &[Rows<'_>]) -> Result<&[Pick], RuntimeError> {
        let config = |m: String| RuntimeError::Config(m);
        if !self.windows.is_empty() {
            return Err(config("the draft windows of the last step were not settled".into()));
        }
        let n_rows: usize = parts.iter().map(|p| p.tokens.len()).sum();
        let full = |e: ares_runtime::RecordFull| RuntimeError::Config(e.to_string());
        let used = {
            let mut writer = RecordWriter::new(
                self.exec.record(),
                self.step + 1,
                parts.len() as u32,
                n_rows as u32,
            )
            .map_err(full)?;
            let mut runs = Vec::new();
            let mut updates = Vec::new();
            let mut row0 = 0u32;
            for part in parts {
                let n = part.tokens.len() as u32;
                let slot = part.slot as usize;
                if slot >= self.entries_sent.len() {
                    return Err(config(format!(
                        "slot {} of {}",
                        part.slot,
                        self.entries_sent.len()
                    )));
                }
                writer.segment(part.slot, part.position, n, part.direct).map_err(full)?;
                for (i, token) in part.tokens.iter().enumerate() {
                    let position = part.position + i as u32;
                    writer
                        .row(Row { slot: part.slot, position, token: *token, context: 0 })
                        .map_err(full)?;
                }
                match part.wanted {
                    Wanted::None => {}
                    Wanted::Last => runs.push((row0 + n - 1, 1)),
                    Wanted::All => runs.push((row0, n)),
                }
                // The pages the rows reach that the devices do not know yet.
                let need = (part.position + n).div_ceil(ENTRY_CELLS);
                if need > self.table_stride {
                    return Err(config(format!(
                        "position {} is past the {} cells of a sequence",
                        part.position + n,
                        self.table_stride * ENTRY_CELLS
                    )));
                }
                let sent = self.entries_sent[slot];
                if need > sent {
                    updates.push(TableUpdate { slot: part.slot, first: sent, count: need - sent });
                    self.entries_sent[slot] = need;
                }
                if !part.direct {
                    self.windows.push((row0, n, part.slot));
                }
                row0 += n;
            }
            writer.runs(&runs).map_err(full)?;
            writer.commits(&self.commits).map_err(full)?;
            let (model, stride) = (&self.model, self.table_stride);
            writer
                .updates(
                    updates.iter().copied(),
                    updates.iter().flat_map(|u| {
                        (u.first..u.first + u.count)
                            .map(move |entry| model.page_entry(u.slot * stride + entry))
                    }),
                )
                .map_err(full)?;
            writer.finish().map_err(full)?
        };
        self.commits.clear();
        self.step += 1;
        let begun = Instant::now();
        let device = |e: ares_runtime::ExecError| RuntimeError::Config(e.to_string());
        self.exec
            .dispatch(&StepMeta { step: self.step, record_bytes: used, after_idle: false })
            .map_err(device)?;
        self.exec.collect(Duration::from_millis(100), &mut self.out).map_err(device)?;
        self.last = StepTime {
            wall: begun.elapsed(),
            device: Duration::from_nanos(
                self.out.devices.iter().map(|d| d.device_ns).max().unwrap_or(0),
            ),
        };
        Ok(&self.out.picks)
    }

    /// Settles the draft windows of the last step: `accepted[i]` rows of the
    /// `i`-th of them count. The devices commit them with the next step.
    pub fn accept(&mut self, accepted: &[u32]) -> Result<(), RuntimeError> {
        if accepted.len() != self.windows.len() {
            return Err(RuntimeError::Config(format!(
                "{} outcomes for {} draft windows",
                accepted.len(),
                self.windows.len()
            )));
        }
        for ((row0, rows, slot), accepted) in self.windows.drain(..).zip(accepted) {
            self.commits.push(Commit { row0, rows, slot, accepted: *accepted });
        }
        Ok(())
    }

    /// What the last step took.
    pub fn last_time(&self) -> StepTime {
        self.last
    }

    /// What each device thread measured in the last step.
    pub fn device_times(&self) -> &[ares_runtime::DeviceTimes] {
        &self.out.devices
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.exec.shutdown();
    }
}
