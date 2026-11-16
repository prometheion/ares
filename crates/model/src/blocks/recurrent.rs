//! The **recurrent block**: the layers of the gated delta rule on a device,
//! with the state of every sequence slot (a convolution window and the
//! delta-rule matrices of each layer) and the inputs a draft window keeps.
//!
//! What a step's rows do to the state:
//!
//! - **Direct rows** (a slice of a prompt, a generated token without drafts)
//!   advance the state in their own launches.
//! - **Window rows** (a sequence's newest token and its drafts) leave the state
//!   as it is: the window reads it and keeps its inputs, the products the model
//!   wrote for its rows, until the commit ([`Recurrent::inputs`]).
//! - **The commit** of a window replays its accepted rows from the kept
//!   inputs into the state ([`Recurrent::commit`]), before the next step.
//!
//! The block declares its extents in a slot ([`Layout`]); the slot arena that
//! holds them, beside the state of other blocks, is [`super::slots::Arena`]. The products
//! that write the inputs, the norm and gate of the output and the product
//! after it are the model's launch plan.

use ares_core::Ledger;
use ares_cuda::{
    CommitSet, Device, DeviceBuffer, DevicePtr, LayerKernels, RecurrentShape, StateFormat,
    StepDims, Stream,
};
use ares_runtime::ModelError;

use super::invalid;
use super::slots::SlotLayout;

/// Where a recurrent layer keeps the state of a sequence in a slot of the
/// state arena, and where a draft window keeps the layer's inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layer {
    pub shape: RecurrentShape,
    /// Offsets in a slot: the convolution window and the delta-rule matrices.
    pub conv: u64,
    pub delta: u64,
    /// Offsets in the kept inputs: qkv, kv, alpha, beta.
    pub kept: [u64; 4],
}

/// Places the state of a layer of `shape` in `slot`: the convolution
/// windows, then the delta-rule matrices. Returns where each starts.
pub(crate) fn place(
    shape: &RecurrentShape,
    format: StateFormat,
    slot: &mut SlotLayout,
) -> (u64, u64) {
    let conv = slot.take(shape.conv_state_len() * 4);
    let delta = slot.take(shape.delta_state_len() * format.bytes());
    (conv, delta)
}

/// Lays out the recurrent layers of a device: their extents in a slot, which
/// they share with other state of the model, and their kept inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    format: StateFormat,
    /// The most rows of a draft window.
    window: u64,
    kept: u64,
}

impl Layout {
    /// No layers yet, for windows of up to `window_rows` rows (0: no windows).
    pub(crate) fn new(format: StateFormat, window_rows: u32) -> Self {
        Self { format, window: u64::from(window_rows), kept: 0 }
    }

    /// Places a layer of `shape` in `slot`, behind the extents declared there
    /// before, and its kept inputs behind those of the layers before.
    pub(crate) fn layer(&mut self, shape: RecurrentShape, slot: &mut SlotLayout) -> Layer {
        let (conv, delta) = place(&shape, self.format, slot);
        let mut kept = [0u64; 4];
        let sizes = [
            u64::from(shape.conv_dim()),
            u64::from(shape.key_dim() + shape.value_dim()),
            u64::from(shape.n_v_heads),
            u64::from(shape.n_v_heads),
        ];
        for (at, size) in kept.iter_mut().zip(sizes) {
            *at = self.kept;
            self.kept = (self.kept + self.window * size * 4).next_multiple_of(64);
        }
        Layer { shape, conv, delta, kept }
    }

    /// Bytes of the kept inputs of the layers placed so far.
    pub(crate) fn kept_bytes(&self) -> u64 {
        self.kept
    }
}

/// Where the rows of a step put a layer's inputs: the convolution's input
/// (`qkv`), its output of keys and values (`kv`), and the gates of the delta
/// rule (`alpha`, `beta`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inputs {
    pub qkv: DevicePtr,
    pub kv: DevicePtr,
    pub alpha: DevicePtr,
    pub beta: DevicePtr,
}

/// The recurrent layers of a device: how their delta-rule matrices are stored,
/// and the inputs a draft window keeps until its commit.
#[derive(Debug)]
pub(crate) struct Recurrent {
    format: StateFormat,
    kept: Option<DeviceBuffer>,
}

impl Recurrent {
    /// The block of layers laid out by `layout`; the kept inputs are reserved
    /// in `ledger` under a name that starts with `label`. Allocates nothing on
    /// a device whose layers keep nothing.
    pub(crate) fn new(
        device: &Device,
        ledger: &mut Ledger,
        label: &str,
        layout: &Layout,
    ) -> Result<Self, ModelError> {
        let bytes = layout.kept_bytes();
        if bytes > 0 {
            ledger.reserve(format!("{label} draft window inputs"), bytes)?;
        }
        let kept = (bytes > 0).then(|| DeviceBuffer::new(device, bytes)).transpose()?;
        Ok(Self { format: layout.format, kept })
    }

    /// Bytes a draft window's kept inputs take.
    pub(crate) fn kept_bytes(&self) -> u64 {
        self.kept.as_ref().map_or(0, DeviceBuffer::len)
    }

    /// Where the rows of a step put the inputs of `layer`: a step that verifies
    /// drafts (`deferred`) keeps them until its commit, any other uses the
    /// `scratch` every layer shares.
    pub(crate) fn inputs(
        &self,
        layer: &Layer,
        deferred: bool,
        scratch: Inputs,
    ) -> Result<Inputs, ModelError> {
        match (&self.kept, deferred) {
            (Some(kept), true) => {
                let k = |i: usize| kept.ptr().offset(layer.kept[i]);
                Ok(Inputs { qkv: k(0), kv: k(1), alpha: k(2), beta: k(3) })
            }
            (None, true) => Err(invalid("no draft windows were reserved")),
            _ => Ok(scratch),
        }
    }

    /// The window of `layer` over the rows of a step whose inputs are in
    /// `io` (the products wrote `qkv`, `alpha` and `beta`): the convolution
    /// (into `q` and `io.kv`, with its taps), the delta rule (into `out`, with
    /// the layer's `dt_bias` and `ssm_a`), and the commit of the convolution
    /// windows of the direct rows. Window rows leave the state as it is.
    /// `arena` is the state arena, `slot_bytes` the bytes of one of its slots.
    ///
    /// # Safety
    ///
    /// The inputs, `q` and `out` hold the step's rows of the layer's widths,
    /// the weights are the layer's on this device, and the arena holds a slot
    /// for every sequence the step record names.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn window(
        &self,
        kernels: &LayerKernels,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        layer: &Layer,
        eps: f32,
        io: &Inputs,
        q: DevicePtr,
        out: DevicePtr,
        taps: DevicePtr,
        dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        arena: DevicePtr,
        slot_bytes: u64,
    ) -> Result<(), ModelError> {
        let shape = layer.shape;
        let (conv, delta) = (arena.offset(layer.conv), arena.offset(layer.delta));
        let (conv_stride, delta_stride) = (slot_bytes / 4, slot_bytes / self.format.bytes());
        // SAFETY: the caller's contract.
        unsafe {
            kernels.conv_window(
                stream,
                step,
                dims,
                shape,
                Some(eps),
                io.qkv,
                taps,
                conv,
                conv_stride,
                q,
                io.kv,
            )?;
            kernels.delta_window(
                stream,
                step,
                dims,
                shape,
                self.format,
                q,
                io.kv,
                io.alpha,
                io.beta,
                dt_bias,
                ssm_a,
                delta,
                delta_stride,
                out,
            )?;
            kernels.conv_commit(
                stream,
                step,
                dims,
                shape,
                CommitSet::Direct,
                io.qkv,
                conv,
                conv_stride,
            )?;
        }
        Ok(())
    }

    /// Commits the draft windows of the last step to the state of `layer`:
    /// `step` is a record with that step's deferred segments and their
    /// accepted rows, whose inputs are kept.
    ///
    /// # Safety
    ///
    /// The kept inputs hold the rows of the last step, which nothing has
    /// overwritten since, and the arena a slot for every sequence the record
    /// names.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn commit(
        &self,
        kernels: &LayerKernels,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        layer: &Layer,
        dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        arena: DevicePtr,
        slot_bytes: u64,
    ) -> Result<(), ModelError> {
        let kept = self.kept.as_ref().ok_or_else(|| invalid("no draft windows were reserved"))?;
        let k = |i: usize| kept.ptr().offset(layer.kept[i]);
        // SAFETY: the caller's contract.
        unsafe {
            kernels.commit(
                stream,
                step,
                dims,
                layer.shape,
                self.format,
                k(0),
                k(1),
                k(2),
                k(3),
                dt_bias,
                ssm_a,
                arena.offset(layer.conv),
                slot_bytes / 4,
                arena.offset(layer.delta),
                slot_bytes / self.format.bytes(),
            )?;
        }
        Ok(())
    }
}
