//! The smallest driver of a device part: one device, one thread, every step
//! waited for. Tests and measurements use it to run steps, single blocks and
//! commits without the runtime, and to put state on the device and read it
//! back. The runtime drives the same [`Part`] through `Qwen35`.

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, ENTRY_CELLS, Event, Kernels, Segment, StateFormat, StepDims, StepRecord,
    Stream,
};
use ares_runtime::{ModelError, PinnedRam};

use super::part::{Options, PAGE_ENTRY, Part, PartSetup, StepEnv};
use super::placement::Placement;
use crate::qwen35::Config;
use crate::qwen35::weights::Weights;

/// A model on one device, driven step by step.
#[derive(Debug)]
pub struct Harness {
    device: Device,
    stream: Stream,
    part: Part,
    record: DeviceBuffer,
    tables: DeviceBuffer,
    pages: DeviceBuffer,
    ledger: Ledger,
    d: usize,
    state: StateFormat,
    /// The record of the last step, for its commit.
    last: Option<StepRecord>,
    start: Event,
    end: Event,
    last_ms: f32,
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Harness {
    /// Opens the whole model on `device` for `slots` sequences of up to
    /// `cells` positions each. Every sequence owns its pages from the start.
    pub fn open(
        device: &Device,
        c: &Config,
        w: &Weights<'_>,
        opts: &Options,
        slots: u32,
        cells: u32,
    ) -> Result<Self, ModelError> {
        Self::open_placed(device, c, w, &Placement::single(c), opts, slots, cells)
    }

    /// [`Harness::open`] for a placement on one device: [`Placement::single`],
    /// possibly with the MTP block ([`Placement::with_mtp`]).
    pub fn open_placed(
        device: &Device,
        c: &Config,
        w: &Weights<'_>,
        placement: &Placement,
        opts: &Options,
        slots: u32,
        cells: u32,
    ) -> Result<Self, ModelError> {
        if placement.n_devices() != 1 {
            return Err(ModelError::Invalid("a harness drives one device".into()));
        }
        let kernels = Kernels::load(device)?;
        let stream = Stream::new(device, 0)?;
        let table_stride = cells.div_ceil(ENTRY_CELLS).max(1);
        let pages = slots * table_stride;
        let mut ledger = Ledger::new(format!("CUDA{}", device.ordinal()), device.mem_info()?.free);
        let page_bytes = super::part::page_bytes(c, placement, 0, opts);
        let pool_bytes = (u64::from(pages) * page_bytes).max(64);
        let record_bytes =
            (128 + 32 * u64::from(slots) + 24 * u64::from(opts.max_rows)).next_multiple_of(64);
        ledger.reserve("state pages", pool_bytes)?;
        ledger.reserve("page tables", u64::from(pages) * 4)?;
        ledger.reserve("step record", record_bytes)?;
        // A harness has no conductor and keeps no checkpoints.
        let ram = PinnedRam::new(0);
        let part = Part::open(
            c,
            w,
            placement,
            opts,
            PartSetup {
                index: 0,
                device,
                kernels: &kernels,
                ledger: &mut ledger,
                max_slots: slots,
                pages,
                table_stride,
                checkpoints: 0,
                ram: &ram,
                table: None,
            },
        )?;
        let mut harness = Self {
            device: device.clone(),
            part,
            record: DeviceBuffer::new(device, record_bytes)?,
            tables: DeviceBuffer::new(device, (u64::from(pages) * 4).max(16))?,
            pages: DeviceBuffer::new(device, pool_bytes)?,
            ledger,
            d: c.n_embd,
            state: opts.state,
            last: None,
            start: Event::timed(device)?,
            end: Event::timed(device)?,
            last_ms: 0.0,
            stream,
        };
        // Slot `s` owns the pages `s * table_stride ..`, in order.
        let entries: Vec<u32> = (0..pages).map(|p| p | PAGE_ENTRY).collect();
        let table_bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        harness.tables.write(0, &table_bytes)?;
        for slot in 0..slots {
            let own =
                &entries[(slot * table_stride) as usize..((slot + 1) * table_stride) as usize];
            harness.part.update_table(slot, 0, own.iter().copied())?;
        }
        Ok(harness)
    }

    /// The part.
    pub fn part(&self) -> &Part {
        &self.part
    }

    /// The part, to change it.
    pub fn part_mut(&mut self) -> &mut Part {
        &mut self.part
    }

    /// The device's memory accounts: who reserved what.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Device milliseconds of the last step, from its first kernel to its last.
    pub fn last_step_ms(&self) -> f32 {
        self.last_ms
    }

    fn upload(&mut self, record: &StepRecord) -> Result<StepDims, ModelError> {
        let bytes = record.as_bytes();
        if bytes.len() as u64 > self.record.len() {
            return Err(ModelError::Invalid(format!(
                "a step record of {} bytes; the harness holds {}",
                bytes.len(),
                self.record.len()
            )));
        }
        self.record.write(0, bytes)?;
        Ok(record.dims())
    }

    /// Runs one step and waits for it. Returns the logits of `out_rows`, a
    /// row of the vocabulary each.
    pub fn step(
        &mut self,
        segments: &[Segment],
        tokens: &[u32],
        out_rows: &[u32],
    ) -> Result<Vec<f32>, ModelError> {
        let record = StepRecord::new(segments, tokens, None, out_rows)?;
        let dims = self.upload(&record)?;
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims,
            tokens,
            segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &[],
        };
        self.start.record(&self.stream)?;
        let logits = self.part.enqueue(&env)?;
        self.end.record(&self.stream)?;
        self.stream.synchronize()?;
        self.last_ms = self.end.elapsed_ms_since(&self.start)?;
        self.last = Some(record);
        let Some(logits) = logits else { return Ok(Vec::new()) };
        let mut bytes = vec![0u8; out_rows.len() * logits.stride as usize * 4];
        self.part.scratch().read(self.part.logits_offset(), &mut bytes)?;
        Ok(floats(&bytes))
    }

    /// Commits the draft windows of the last step: `accepted[i]` rows of its
    /// segment `i` (segments that were direct are skipped by the kernels).
    pub fn commit(&mut self, accepted: &[u32]) -> Result<(), ModelError> {
        let mut record =
            self.last.take().ok_or_else(|| ModelError::Invalid("no step to commit".into()))?;
        for (segment, rows) in accepted.iter().enumerate() {
            record.set_accepted(segment, *rows)?;
        }
        let dims = self.upload(&record)?;
        self.part.enqueue_commit(&self.stream, self.record.ptr(), dims)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Runs block `i` of the part alone on the rows `x` (`d` values each) of
    /// the given segments and returns the block's output rows. The block
    /// reads and advances the state on the device as in a step.
    pub fn block(
        &mut self,
        i: usize,
        segments: &[Segment],
        x: &[f32],
    ) -> Result<Vec<f32>, ModelError> {
        let n = x.len() / self.d;
        let tokens = vec![0; n];
        let record = StepRecord::new(segments, &tokens, None, &[])?;
        let dims = self.upload(&record)?;
        self.part.scratch().write(self.part.x_offset(), &bytes_of(x))?;
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims,
            tokens: &tokens,
            segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &[],
        };
        self.part.enqueue_begin(&env)?;
        // The rotary table of the step, which `enqueue` builds with the input.
        self.part.enqueue_rope(&env)?;
        self.part.enqueue_block(&env, i)?;
        self.part.enqueue_finish(&env)?;
        self.stream.synchronize()?;
        self.last = Some(record);
        let mut bytes = vec![0u8; x.len() * 4];
        self.part.scratch().read(self.part.x_offset(), &mut bytes)?;
        Ok(floats(&bytes))
    }

    /// Runs every block of the part in turn on the rows of `segments`, as one
    /// step, and returns the residual stream after each block. `input(i)`
    /// gives the rows block `i` reads (`d` values each), or `None` for what
    /// the block before it left; block 0 must be given its rows. The blocks
    /// read and advance the state on the device as in a step, and a draft
    /// window is committed with [`Harness::commit`] as after a step.
    pub fn blocks(
        &mut self,
        segments: &[Segment],
        mut input: impl FnMut(usize) -> Option<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        let n: usize = segments.iter().map(|s| s.n_rows as usize).sum();
        let tokens = vec![0; n];
        let record = StepRecord::new(segments, &tokens, None, &[])?;
        let dims = self.upload(&record)?;
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims,
            tokens: &tokens,
            segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &[],
        };
        self.part.enqueue_begin(&env)?;
        self.part.enqueue_rope(&env)?;
        let mut after = Vec::with_capacity(self.part.n_blocks());
        for i in 0..self.part.n_blocks() {
            match input(i) {
                Some(x) if x.len() == n * self.d => {
                    // The write does not go through the stream: the block
                    // before must have ended.
                    self.stream.synchronize()?;
                    self.part.scratch().write(self.part.x_offset(), &bytes_of(&x))?;
                }
                Some(x) => {
                    return Err(ModelError::Invalid(format!(
                        "{} values for {n} rows of {}",
                        x.len(),
                        self.d
                    )));
                }
                None if i == 0 => {
                    return Err(ModelError::Invalid("the first block needs its rows".into()));
                }
                None => {}
            }
            self.part.enqueue_block(&env, i)?;
            self.stream.synchronize()?;
            let mut bytes = vec![0u8; n * self.d * 4];
            self.part.scratch().read(self.part.x_offset(), &mut bytes)?;
            after.push(floats(&bytes));
        }
        self.part.enqueue_finish(&env)?;
        self.stream.synchronize()?;
        self.last = Some(record);
        Ok(after)
    }

    /// The stored keys and values of the first `cells` cells of the sequence
    /// in `slot`, in attention block `i`, as the bytes on the device: for
    /// every tile a byte that says whether it is a converted entry, then per
    /// KV head the tile means of a converted entry and the rows of K and of V
    /// that hold cells.
    pub fn read_kv(&self, i: usize, slot: u32, cells: u32) -> Result<Vec<u8>, ModelError> {
        self.stream.synchronize()?;
        let mut out = Vec::new();
        for tile in 0..cells.div_ceil(ENTRY_CELLS) {
            let place = self
                .part
                .kv_place(i, slot, tile)
                .ok_or_else(|| ModelError::Invalid(format!("block {i} is not attention")))?;
            let rows = (cells - tile * ENTRY_CELLS).min(ENTRY_CELLS) as u64;
            let geo = place.geo;
            let d = u64::from(geo.head_dim);
            out.push(u8::from(place.converted));
            let mut take = |buffer: &DeviceBuffer, at: u64, len: u64| {
                let mut bytes = vec![0u8; len as usize];
                buffer.read(at, &mut bytes).map(|()| out.extend(bytes))
            };
            for head in 0..u64::from(geo.kv_heads) {
                if place.converted {
                    let base = place.offset + head * geo.converted_head_bytes();
                    let (k_mean, v_mean) = (
                        u64::from(geo.k.mean_bytes(geo.head_dim)),
                        u64::from(geo.v.mean_bytes(geo.head_dim)),
                    );
                    let (k_row, v_row) = (
                        u64::from(geo.k.row_bytes(geo.head_dim)),
                        u64::from(geo.v.row_bytes(geo.head_dim)),
                    );
                    let v_at = base + k_mean + u64::from(ENTRY_CELLS) * k_row;
                    take(&self.pages, base, k_mean + rows * k_row)?;
                    take(&self.pages, v_at, v_mean + rows * v_row)?;
                } else {
                    let exact = self.part.exact_kv().ok_or_else(|| {
                        ModelError::Invalid("an exact entry without an exact pool".into())
                    })?;
                    let base = place.offset + head * u64::from(ENTRY_CELLS) * 4 * d;
                    take(exact, base, rows * 2 * d)?;
                    take(exact, base + u64::from(ENTRY_CELLS) * 2 * d, rows * 2 * d)?;
                }
            }
        }
        Ok(out)
    }

    /// Puts the recurrent state of block `i` and `slot` on the device: the
    /// convolution windows and the delta-rule matrices in the reference's
    /// layout.
    pub fn write_recurrent_state(
        &mut self,
        i: usize,
        slot: u32,
        conv: &[f32],
        delta: &[f32],
    ) -> Result<(), ModelError> {
        let [(conv_at, conv_len), (delta_at, delta_len)] = self
            .part
            .recurrent_state_at(i, slot)
            .ok_or_else(|| ModelError::Invalid(format!("block {i} is not recurrent")))?;
        let delta_bytes = match self.state {
            StateFormat::F32 => bytes_of(delta),
            StateFormat::F16 => {
                delta.iter().flat_map(|v| ares_quant::f32_to_f16(*v).to_le_bytes()).collect()
            }
        };
        if conv.len() as u64 * 4 != conv_len || delta_bytes.len() as u64 != delta_len {
            return Err(ModelError::Invalid(format!(
                "state of {} and {} values for a block that keeps {conv_len} and {delta_len} bytes",
                conv.len(),
                delta.len()
            )));
        }
        self.part.state().write(conv_at, &bytes_of(conv))?;
        self.part.state().write(delta_at, &delta_bytes)?;
        Ok(())
    }

    /// Reads the recurrent state of block `i` and `slot` back: convolution
    /// windows and delta-rule matrices as f32.
    pub fn read_recurrent_state(
        &self,
        i: usize,
        slot: u32,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        let [(conv_at, conv_len), (delta_at, delta_len)] = self
            .part
            .recurrent_state_at(i, slot)
            .ok_or_else(|| ModelError::Invalid(format!("block {i} is not recurrent")))?;
        self.stream.synchronize()?;
        let mut conv = vec![0u8; conv_len as usize];
        let mut delta = vec![0u8; delta_len as usize];
        self.part.state().read(conv_at, &mut conv)?;
        self.part.state().read(delta_at, &mut delta)?;
        let delta = match self.state {
            StateFormat::F32 => floats(&delta),
            StateFormat::F16 => delta
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| ares_quant::f16_to_f32(u16::from_le_bytes(*b)))
                .collect(),
        };
        Ok((floats(&conv), delta))
    }

    /// Writes the keys and values of the first cells of the sequence in
    /// `slot` into the pool of attention block `i`, in the model's K/V
    /// formats: `k` and `v` are `[cell][kv head][head_dim]`.
    pub fn write_kv(
        &mut self,
        i: usize,
        slot: u32,
        cells: u32,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), ModelError> {
        if cells == 0 {
            return Ok(());
        }
        let k_dev = DeviceBuffer::new(&self.device, (k.len() * 4) as u64)?;
        let v_dev = DeviceBuffer::new(&self.device, (v.len() * 4) as u64)?;
        k_dev.write(0, &bytes_of(k))?;
        v_dev.write(0, &bytes_of(v))?;
        self.part.write_kv(
            &self.stream,
            self.tables.ptr(),
            self.pages.ptr(),
            i,
            slot,
            0,
            cells,
            k_dev.ptr(),
            v_dev.ptr(),
        )
    }

    /// Adds the device time of the last step to the profile, if the part
    /// keeps one.
    pub fn flush_profile(&mut self) -> Result<(), ModelError> {
        self.stream.synchronize()?;
        self.part.flush_profile()
    }
}
