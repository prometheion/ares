//! What a part leaves for the host about its routers: the route record of
//! every layer and the layer's input rows as the CPU reads them, copied on a
//! side stream, and the event that says the copy has run; and, in a step of
//! the decode kernels, the experts the next layers' routers are predicted to
//! pick (`part/predict.rs`), with an event of their own.
//!
//! The device side belongs to the part, which enqueues the copies. The host
//! side belongs to whoever feeds the expert store ([`super::feed::Feed`]),
//! possibly another thread, so the board is shared and holds nothing that
//! changes except through the device's copies.

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
    /// that predicts nothing).
    pub(super) fn new(
        device: &Device,
        layers: usize,
        record_stride: usize,
        act_stride: usize,
        n_embd: usize,
        (ahead, list_bytes): (usize, usize),
    ) -> Result<Self, ModelError> {
        let predicted_layers = if ahead > 0 { layers } else { 0 };
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
        })
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
