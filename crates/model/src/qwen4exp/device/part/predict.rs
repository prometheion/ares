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
//! [`Board::ahead_list`]: super::super::board::Board::ahead_list

use ares_cuda::{DeviceBuffer, DevicePtr, Event, Stream};
use ares_runtime::ModelError;

use super::{Part, invalid};

/// The buffers and the stream of the predictions of a part.
pub(super) struct State {
    stream: Stream,
    /// The logits of the routers (the layers ahead times the rows of a decode step times the
    /// experts, f32), then the lists of every layer of the part, `list_bytes` apart per layer
    /// ahead.
    buffer: DeviceBuffer,
    lists: u64,
    /// Recorded on the stream behind the last prediction of a step.
    done: Event,
    /// Whether a prediction was enqueued since the step's stream last waited for `done`.
    pending: bool,
}

impl State {
    pub(super) fn open(
        device: &ares_cuda::Device,
        logits_bytes: u64,
        lists_bytes: u64,
    ) -> Result<Self, ModelError> {
        let lists = logits_bytes.next_multiple_of(256);
        Ok(Self {
            stream: Stream::new(device, 0)?,
            buffer: DeviceBuffer::new(device, lists + lists_bytes.max(64))?,
            lists,
            done: Event::new(device)?,
            pending: false,
        })
    }
}

impl Part {
    /// Enqueues the prediction of the picks of the layers after layer `i` of the part for the
    /// `rows` rows of a decode step: `act` holds the rows the router of layer `i` read, and
    /// the router's event has been recorded on the step's stream. Nothing for a part without
    /// predictions, a layer without a layer behind it, or a step of the many-row kernels.
    pub(super) fn enqueue_prediction(
        &mut self,
        i: usize,
        act: DevicePtr,
        rows: u32,
        token: u32,
    ) -> Result<(), ModelError> {
        let ahead = self.f.ahead_layers as usize;
        let Some(state) = self.predict.as_mut() else { return Ok(()) };
        let layers = self.w.layers.len();
        let routers: Vec<DevicePtr> = (i + 1..layers.min(i + 1 + ahead))
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
