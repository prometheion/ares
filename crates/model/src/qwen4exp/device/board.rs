//! What a part leaves for the host about its routers: the route record of
//! every layer and the layer's input rows as the CPU reads them, copied on a
//! side stream, and the event that says the copy has run; and, in a step of
//! the decode kernels, the experts the next layers' routers are predicted to
//! pick (`part/predict.rs`), with an event of their own; in a wide step the
//! picks the next layer's router would make for every row.
//!
//! The device side belongs to the part, which enqueues the copies. The host
//! side belongs to whoever feeds the expert store ([`super::feed::Feed`]),
//! possibly another thread, so the board is shared and holds nothing that
//! changes except through the device's copies and two flags per layer of a
//! wide step: whether the feed wants a prediction there, and the rows of the
//! one the part has enqueued.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use ares_cuda::{Device, Event, PinnedBuffer, Stream};
use ares_runtime::ModelError;
use ares_runtime::expert_store::cpu::act_bytes;

/// The route records of a part's layers, as the host has them.
pub struct Board {
    records: PinnedBuffer,
    acts: PinnedBuffer,
    /// Per layer, recorded behind the copies of its record and its rows.
    copied: Vec<Event>,
    record_stride: usize,
    act_stride: usize,
    n_embd: usize,
    /// The predicted picks of the layers after each layer, `ahead` lists of
    /// `list_bytes` per layer, and per layer the event behind their copy.
    predicted: PinnedBuffer,
    predicted_copied: Vec<Event>,
    ahead: usize,
    list_bytes: usize,
    /// In wide steps: per layer the picks of every row by the next layer's router
    /// (`wide_bytes` per layer), the event behind their copy, the rows of the
    /// prediction enqueued and not yet taken (0 for none), and whether the feed
    /// wants one.
    wide: PinnedBuffer,
    wide_copied: Vec<Event>,
    wide_bytes: usize,
    wide_rows: Vec<AtomicU32>,
    wide_wanted: Vec<AtomicBool>,
}

impl std::fmt::Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Board").field("layers", &self.copied.len()).finish_non_exhaustive()
    }
}

impl Board {
    /// A board for `layers` layers whose records are `record_stride` bytes
    /// apart and whose rows are `act_stride` bytes apart, with `ahead` lists of
    /// predicted picks of `list_bytes` each per layer (`ahead` is 0 for a part
    /// that predicts nothing) and `wide_bytes` per layer for the predicted picks
    /// of a wide step (0 for none).
    pub(super) fn new(
        device: &Device,
        layers: usize,
        record_stride: usize,
        act_stride: usize,
        n_embd: usize,
        (ahead, list_bytes, wide_bytes): (usize, usize, usize),
    ) -> Result<Self, ModelError> {
        let predicted_layers = if ahead > 0 { layers } else { 0 };
        let wide_layers = if wide_bytes > 0 { layers } else { 0 };
        Ok(Self {
            records: PinnedBuffer::new(device, (record_stride * layers).max(64))?,
            acts: PinnedBuffer::new(device, (act_stride * layers).max(64))?,
            copied: (0..layers).map(|_| Event::new(device)).collect::<Result<_, _>>()?,
            record_stride,
            act_stride,
            n_embd,
            predicted: PinnedBuffer::new(device, (ahead * list_bytes * layers).max(64))?,
            predicted_copied: (0..predicted_layers)
                .map(|_| Event::new(device))
                .collect::<Result<_, _>>()?,
            ahead,
            list_bytes,
            wide: PinnedBuffer::new(device, (wide_bytes * layers).max(64))?,
            wide_copied: (0..wide_layers).map(|_| Event::new(device)).collect::<Result<_, _>>()?,
            wide_bytes,
            wide_rows: (0..wide_layers).map(|_| AtomicU32::new(0)).collect(),
            wide_wanted: (0..wide_layers).map(|_| AtomicBool::new(true)).collect(),
        })
    }

    /// Whether the part makes predictions in wide steps.
    pub fn predicts_wide(&self) -> bool {
        !self.wide_copied.is_empty()
    }

    /// Whether the feed wants the next layer's picks predicted at layer `i` of
    /// a wide step.
    pub fn wide_wanted(&self, i: usize) -> bool {
        self.wide_wanted.get(i).is_some_and(|w| w.load(Ordering::Acquire))
    }

    /// The feed's word on whether to predict at layer `i` of wide steps.
    pub fn set_wide_wanted(&self, i: usize, wanted: bool) {
        if let Some(w) = self.wide_wanted.get(i) {
            w.store(wanted, Ordering::Release);
        }
    }

    /// Where the device copies the predicted picks made at layer `i` of a wide
    /// step.
    pub(super) fn wide_dst(&self, i: usize) -> *mut u8 {
        // The only writer is the device's copy, ordered by `wide_copied`.
        self.wide.as_ptr().cast_mut().wrapping_add(i * self.wide_bytes)
    }

    /// Records, on the stream that did the copy, that the predicted picks of
    /// `rows` rows made at layer `i` of a wide step are copied.
    pub(super) fn wide_enqueued(
        &self,
        i: usize,
        rows: u32,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        self.wide_copied[i].record(stream)?;
        self.wide_rows[i].store(rows, Ordering::Release);
        Ok(())
    }

    /// The predicted picks made at layer `i` of a wide step, `top_k` ids per
    /// row, once they have reached the host; each is handed out once. Never
    /// waits.
    pub fn take_wide(&self, i: usize, top_k: usize) -> Result<Option<&[u8]>, ModelError> {
        let Some(rows) = self.wide_rows.get(i) else { return Ok(None) };
        let n = rows.load(Ordering::Acquire) as usize;
        if n == 0 || !self.wide_copied[i].is_done()? {
            return Ok(None);
        }
        rows.store(0, Ordering::Release);
        let len = (4 * n * top_k).min(self.wide_bytes);
        // SAFETY: the buffer holds `wide_bytes` per layer, and the copy that
        // wrote them has finished.
        Ok(Some(unsafe {
            std::slice::from_raw_parts(self.wide.as_ptr().add(i * self.wide_bytes), len)
        }))
    }

    /// The layers of the part.
    pub fn layers(&self) -> usize {
        self.copied.len()
    }

    /// How many layers after a layer its predictions name: 0 for a part that
    /// predicts nothing.
    pub fn ahead(&self) -> usize {
        self.ahead
    }

    /// Where the device copies the predicted picks of the layers after layer `i`.
    pub(super) fn ahead_dst(&self, i: usize) -> *mut u8 {
        // The only writer is the device's copy, ordered by `predicted_copied`.
        self.predicted.as_ptr().cast_mut().wrapping_add(i * self.ahead * self.list_bytes)
    }

    /// Records, on the stream that did the copy, that the predictions of layer
    /// `i` are copied.
    pub(super) fn ahead_enqueued(&self, i: usize, stream: &Stream) -> Result<(), ModelError> {
        Ok(self.predicted_copied[i].record(stream)?)
    }

    /// Whether the predictions made at layer `i` have reached the host: true
    /// also for a step that made none, whose lists say so by their token.
    /// Never waits.
    pub fn ahead_arrived(&self, i: usize) -> Result<bool, ModelError> {
        match self.predicted_copied.get(i) {
            Some(event) => Ok(event.is_done()?),
            None => Ok(true),
        }
    }

    /// The predicted picks of layer `i + 1 + d` made at layer `i`, as
    /// `ares_cuda::PredictedList` reads them: valid once
    /// [`Board::ahead_arrived`] is true for the step.
    pub fn ahead_list(&self, i: usize, d: usize) -> &[u8] {
        if d >= self.ahead || i >= self.predicted_copied.len() {
            return &[];
        }
        // SAFETY: the buffer holds `ahead` lists of `list_bytes` per layer, and
        // the caller reads them after the copy has finished.
        unsafe {
            std::slice::from_raw_parts(
                self.predicted.as_ptr().add((i * self.ahead + d) * self.list_bytes),
                self.list_bytes,
            )
        }
    }

    /// Where the device copies the record of layer `i`.
    pub(super) fn record_dst(&self, i: usize) -> *mut u8 {
        // The only writer is the device's copy, ordered by `copied`; the
        // pointer is derived from the buffer's address, not from a reference.
        self.records.as_ptr().cast_mut().wrapping_add(i * self.record_stride)
    }

    /// Where the device copies the input rows of layer `i`.
    pub(super) fn rows_dst(&self, i: usize) -> *mut u8 {
        self.acts.as_ptr().cast_mut().wrapping_add(i * self.act_stride)
    }

    /// Records, on the stream that did the copies, that those of layer `i`
    /// are done.
    pub(super) fn copies_enqueued(&self, i: usize, stream: &Stream) -> Result<(), ModelError> {
        Ok(self.copied[i].record(stream)?)
    }

    /// Whether the record of layer `i` of the step being run, with the
    /// layer's input rows, has reached the host. Never waits.
    pub fn arrived(&self, i: usize) -> Result<bool, ModelError> {
        Ok(self.copied[i].is_done()?)
    }

    /// The route record of layer `i`: valid once [`Board::arrived`] is true
    /// for the step.
    pub fn record(&self, i: usize) -> &[u8] {
        // SAFETY: the buffer holds a record per layer, `record_stride` apart,
        // and the caller reads it after the copy has finished.
        unsafe {
            std::slice::from_raw_parts(
                self.records.as_ptr().add(i * self.record_stride),
                self.record_stride,
            )
        }
    }

    /// The input rows of layer `i` as the CPU reads them for `rows` rows (the
    /// codes and scales of the quantized activations): valid with the
    /// layer's record.
    pub fn rows(&self, i: usize, rows: usize) -> &[u8] {
        let len = act_bytes(self.n_embd, rows).min(self.act_stride);
        // SAFETY: the buffer holds `act_stride` bytes per layer, and the
        // caller reads them after the copy has finished.
        unsafe { std::slice::from_raw_parts(self.acts.as_ptr().add(i * self.act_stride), len) }
    }
}
