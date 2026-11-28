//! The prediction of what the part's next layers will pick, in a step of the decode kernels
//! (Pre-gated MoE, FATE, Run My AI): right after the router of a layer, the routers of the
//! next one or two layers of the part are applied to the rows that router read, and the
//! experts each would pick are copied to the host ([`Board::ahead_list`]). The expert store
//! starts to read those that no tier holds while the step is still on its way to their
//! layers.
//!
//! The prediction runs on a stream of its own that waits for the router's event; the step's
//! stream never waits for it within the step, and it writes only its own buffers (its logits
//! and lists). What the model computes is the same bit for bit with and without it. Before
//! the rows it reads are written again, in the next step, the step's stream waits for the
//! last prediction of the step before ([`Part::predictions_done`]), which has ended long
//! before.
//!
//! A step predicts only as many layers ahead as the board says ([`Board::ahead_depth`]): the
//! store's read-ahead takes the predictions only at the places of a list where they have been
//! picked at least half the time, and a step of a kind with no such place skips the routers
//! but now and then (`ExpertStore::ahead_depth`), which saves the device their 1.7 ms a step
//! on the RTX 4060 Ti where the tiers hold what the text uses.
//!
//! A wide step (a slice of a prompt) predicts the next layer only, where the board says the
//! feed wants it (where the step before read that layer's experts from the file): the next
//! layer's router as a float product on the f16 form of the rows, and its picks for every
//! row, which the feed counts per expert and hands to the store as the guesses of that layer
//! (`ExpertStore::transit_guess`). The step's stream waits only for the f16 copy of its rows,
//! which the next layer overwrites.
//!
//! [`Board::ahead_list`]: super::super::board::Board::ahead_list
//! [`Board::ahead_depth`]: super::super::board::Board::ahead_depth

use ares_core::GgmlType;
use ares_cuda::{DeviceBuffer, DevicePtr, Event, PrefillPath, Stream};
use ares_quant::act;
use ares_runtime::ModelError;

use super::{Frame, Part, invalid};

/// The buffers and the stream of the predictions of a part.
pub(super) struct State {
    stream: Stream,
    /// The logits of the routers (the layers ahead times the rows of a decode step times the
    /// experts, f32), then the lists of every layer of the part, `list_bytes` apart per layer
    /// ahead.
    buffer: DeviceBuffer,
    lists: u64,
    /// For wide steps, offsets in the buffer: the f16 form of the rows, their logits by the
    /// next layer's router, and their picks.
    wide_input: u64,
    wide_logits: u64,
    wide_record: u64,
    /// Recorded on the stream behind the f16 copy of a wide step's rows.
    input_copied: Event,
    /// Recorded on the stream behind the last prediction of a step.
    done: Event,
    /// Whether a prediction was enqueued since the step's stream last waited for `done`.
    pending: bool,
}

impl State {
    /// The buffers `f` sizes ([`Frame::predict_bytes`] in all).
    pub(super) fn open(device: &ares_cuda::Device, f: &Frame) -> Result<Self, ModelError> {
        let lists = f.ahead_logits_bytes.next_multiple_of(256);
        let wide_input = lists + f.predict_lists_bytes().max(64).next_multiple_of(256);
        let wide_logits = wide_input + f.wide_input_bytes.next_multiple_of(256);
        let wide_record = wide_logits + f.wide_logits_bytes.next_multiple_of(256);
        let len = wide_record + f.wide_record_bytes.next_multiple_of(256);
        debug_assert_eq!(len, f.predict_bytes());
        Ok(Self {
            stream: Stream::new(device, 0)?,
            buffer: DeviceBuffer::new(device, len)?,
            lists,
            wide_input,
            wide_logits,
            wide_record,
            input_copied: Event::new(device)?,
            done: Event::new(device)?,
            pending: false,
        })
    }
}

impl Part {
    /// Enqueues the prediction of the picks of the layers after layer `i` of the part for the
    /// `rows` rows of a decode step: `act` holds the rows the router of layer `i` read, and
    /// the router's event has been recorded on the step's stream. Nothing for a part without
    /// predictions, a layer without a layer behind it, a step of the many-row kernels, or a
    /// step the board says to predict nothing in.
    pub(super) fn enqueue_prediction(
        &mut self,
        i: usize,
        act: DevicePtr,
        rows: u32,
        token: u32,
    ) -> Result<(), ModelError> {
        let ahead = self.f.ahead_layers as usize;
        let depth = self.board.ahead_depth().min(ahead);
        let Some(state) = self.predict.as_mut() else { return Ok(()) };
        let layers = self.w.layers.len();
        let routers: Vec<DevicePtr> = (i + 1..layers.min(i + 1 + depth))
            .map(|at| {
                let router = self.w.layers[at].moe.router;
                router.map(|r| self.w.at(r)).ok_or_else(|| invalid("a layer without its router"))
            })
            .collect::<Result<_, _>>()?;
        if routers.is_empty() || Self::is_wide(rows) {
            return Ok(());
        }
        let shape = self.w.layers[i].moe.shape;
        let list_bytes = self.f.ahead_list_bytes;
        let lists = state.buffer.ptr().offset(state.lists + (i * ahead) as u64 * list_bytes);
        state.stream.wait_event(&self.routed[i])?;
        // SAFETY: the routers are F32 matrices of the part's weights of this shape, `act`
        // holds the activations of `rows` rows until the next step writes them (which waits
        // for `done`), and the logits and the lists are this state's own buffer, sized for
        // the layers ahead and the rows of a decode step. The board's lists of layer `i` are
        // pinned memory that the host reads only after the event behind this copy.
        unsafe {
            self.experts.predict(
                &state.stream,
                &shape,
                &routers,
                act,
                state.buffer.ptr(),
                lists,
                list_bytes,
                rows,
                token,
            )?;
            state.stream.copy_to_host(
                self.board.ahead_dst(i),
                lists,
                (routers.len() as u64 * list_bytes) as usize,
            )?;
        }
        self.board.ahead_enqueued(i, &state.stream)?;
        state.done.record(&state.stream)?;
        state.pending = true;
        Ok(())
    }

    /// Enqueues the prediction of the next layer's picks at layer `i` of a wide step of `rows`
    /// rows, on the step's `stream`: `act` holds the rows the router of layer `i` read (in
    /// the activations' layout) and the router's event has been recorded. Nothing when the
    /// board does not want it there, at the part's last layer, or for a router whose weights
    /// the f16 tile kernels do not take.
    pub(super) fn enqueue_wide_prediction(
        &mut self,
        stream: &Stream,
        i: usize,
        act: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        let Some(state) = self.predict.as_mut() else { return Ok(()) };
        if self.f.wide_input_bytes == 0 || !self.board.wide_wanted(i) {
            return Ok(());
        }
        let Some(next) = self.w.layers.get(i + 1) else { return Ok(()) };
        let moe = next.moe;
        let Some(router) = moe.router else { return Ok(()) };
        let (n_in, n_out) = (moe.shape.n_embd, moe.shape.n_experts);
        let tile = n_in.is_multiple_of(32).then_some(PrefillPath::F16);
        let Some(path) = moe.router_range.path(tile) else { return Ok(()) };
        let base = state.buffer.ptr();
        let (input, logits, record) = (
            base.offset(state.wide_input),
            base.offset(state.wide_logits),
            base.offset(state.wide_record),
        );
        let f32_at = act::Layout::of(u64::from(n_in) * u64::from(rows)).values_at;
        let ids = 4 * u64::from(moe.shape.top_k) * u64::from(rows);
        state.stream.wait_event(&self.routed[i])?;
        // SAFETY: `act` holds the activations of `rows` rows, which the step's stream does
        // not write again before it has waited for their f16 copy; the router is an F32
        // matrix of the part's weights; the input, the logits and the record are this
        // state's own buffer, sized for the most rows of a step. The board's ids of layer `i`
        // are pinned memory that the host reads only after the event behind this copy.
        unsafe {
            self.kernels.prepare_prefill_activations(
                &state.stream,
                PrefillPath::F16,
                act.offset(f32_at),
                input,
                n_in,
                rows,
            )?;
            state.input_copied.record(&state.stream)?;
            stream.wait_event(&state.input_copied)?;
            self.kernels.matmul(
                &state.stream,
                GgmlType::F32,
                path,
                self.w.at(router),
                input,
                logits,
                n_in,
                n_out,
                rows,
            )?;
            self.experts.pick_rows(&state.stream, &moe.shape, logits, record, rows)?;
            state.stream.copy_to_host(self.board.wide_dst(i), record.offset(16), ids as usize)?;
        }
        self.board.wide_enqueued(i, rows, &state.stream)?;
        state.done.record(&state.stream)?;
        state.pending = true;
        Ok(())
    }

    /// Makes the step's stream wait for the predictions enqueued so far: before a step writes
    /// the rows that the predictions of the step before read.
    pub(super) fn predictions_done(&mut self, stream: &Stream) -> Result<(), ModelError> {
        if let Some(state) = self.predict.as_mut().filter(|state| state.pending) {
            stream.wait_event(&state.done)?;
            state.pending = false;
        }
        Ok(())
    }
}
