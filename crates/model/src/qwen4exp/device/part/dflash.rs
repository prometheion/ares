//! The DFlash drafter in a part of `qwen4exp`: what Flash-Next gives the
//! shared drafter (`crate::dflash`).
//!
//! - **Its taps** are the mean of the four hyper-connection streams before
//!   each target layer, which is what its drafter was trained on: the residual
//!   after the read of the layer's first stream mixer (the read applies the
//!   write of the layer before), rows of `hc` streams of `n` values.
//! - **Its embedding of a token** is the row decoded on the host from the
//!   mapped file, as the model's own rows are.
//! - **Its draft head** is the output head after the final norm, in the file's
//!   layout, multiplied by the drafter's own products.
//!
//! The drafter is on the part that holds the streams of every layer and the
//! head: one device alone, or the mixer of two devices on every layer. A split
//! by layers would have to send the taps of the first device's layers to the
//! second; it is not built.

use ares_core::Ledger;
use ares_cuda::{Device, DevicePtr, Stream};
use ares_runtime::ModelError;

use super::{Part, Role, StepEnv, invalid};
use crate::dflash::{Drafter, DrafterSetup, File, PassSeq};

impl Part {
    /// Opens the drafter of `file` on this part, which must hold the streams
    /// of every layer the drafter reads and the head. `mask_row` is the
    /// model's embedding of the mask token.
    #[allow(clippy::too_many_arguments)]
    pub fn open_dflash(
        &mut self,
        file: &File,
        device: &Device,
        ledger: &mut Ledger,
        max_slots: u32,
        n_layer: usize,
        n_vocab: usize,
        head_rows: u32,
        mask_row: Vec<f32>,
    ) -> Result<(), ModelError> {
        if self.role == Role::Worker || self.w.head.is_none() {
            return Err(invalid(
                "the DFlash drafter needs the device that holds the streams and the head",
            ));
        }
        let held: Vec<usize> = self.w.layers.iter().map(|l| l.layer).collect();
        if let Some(layer) =
            file.config.target_layers.iter().find(|l| !held.contains(&(**l as usize)))
        {
            return Err(invalid(format!(
                "the DFlash drafter reads layer {layer}, which another device holds: the drafter \
                 is not built for a split by layers"
            )));
        }
        let setup = DrafterSetup {
            max_slots,
            max_rows: self.opts.max_rows,
            window_rows: self.opts.window_rows,
            head_rows,
        };
        let n = self.n as usize;
        self.dflash =
            Some(file.open_drafter(n, n_layer, n_vocab, device, ledger, setup, mask_row)?);
        Ok(())
    }

    /// The drafter of this part, for its report and its tests.
    pub fn dflash(&self) -> Option<&Drafter> {
        self.dflash.as_ref()
    }

    pub(crate) fn dflash_mut(&mut self) -> Option<&mut Drafter> {
        self.dflash.as_mut()
    }

    /// Tells the drafter that the step enqueued next has no pass.
    pub fn dflash_no_pass(&mut self) {
        if let Some(d) = &mut self.dflash {
            d.no_pass();
        }
    }

    /// The start of a step: the drafter is told its rows.
    pub fn dflash_begin(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let Some(d) = &mut self.dflash else { return Ok(()) };
        d.begin_segments(env.stream, env.segments)
    }

    /// After the read of the first mixer of model layer `layer`: its tap, when
    /// the drafter reads it.
    pub(super) fn dflash_tap(&mut self, env: &StepEnv<'_>, layer: usize) -> Result<(), ModelError> {
        let r = self.r();
        let Some(d) = &mut self.dflash else { return Ok(()) };
        let Some(tap) = d.config().target_layers.iter().position(|l| *l as usize == layer) else {
            return Ok(());
        };
        let rows = env.record.dims().n_rows;
        d.tap(&self.kernels, env.stream, tap, r, self.hc, u64::from(self.n), self.wide, 0, rows)
    }

    /// After the model's layers: the step's rows into the drafter's ring.
    pub fn dflash_inject(&mut self, stream: &Stream) -> Result<(), ModelError> {
        let Some(d) = &mut self.dflash else { return Ok(()) };
        d.inject(&self.kernels, stream)
    }

    /// The drafter's pass for the windows `seqs` of a step that only drafts:
    /// the layers, the draft head (the head view's first rows of the output
    /// head) and the selector. `embed` decodes a row of the embedding table.
    /// The drafts go into the step record's token array and into `list` (the
    /// MTP block's when it drafts on this device too), or the drafter's own.
    pub fn dflash_pass(
        &mut self,
        stream: &Stream,
        record_tokens: DevicePtr,
        seqs: &[PassSeq],
        embed: &dyn Fn(u32) -> Vec<f32>,
        list: Option<DevicePtr>,
    ) -> Result<(), ModelError> {
        let head = self.w.head.ok_or_else(|| invalid("no head on this part"))?;
        let output = self.w.mat(&head.output);
        let tokens = self.head_tokens();
        let Some(d) = &mut self.dflash else { return Err(invalid("no DFlash drafter")) };
        let (h, rows) = d.pass_layers(&self.kernels, stream, seqs, embed)?;
        let (ty, range) = (head.output.ty, head.output.range);
        d.head_product(&self.kernels, stream, ty, range, output, h, rows)?;
        let list = list.unwrap_or_else(|| d.own_list());
        d.pass_select(&self.kernels, stream, tokens, record_tokens, list)
    }
}
