//! The DFlash drafter in a part of `qwen35`: what the 27B gives the shared
//! drafter (`crate::dflash`).
//!
//! - **Its taps** are the residual stream `x` before each target block, one
//!   stream of rows of `d` values. Under a tensor split every device holds it
//!   bit for bit after the merge, so the drafter's device reads its own copy.
//! - **Its embedding of a token** is the row of the table in page-locked host
//!   memory, decoded on the host: the newest token of each sequence that
//!   drafts, and the mask token once at open.
//! - **Its draft head** is the draft plan's: the first rows of the output head
//!   on the device that embeds (the draft vocabulary when there is one), in
//!   whatever layout they have, multiplied by the part's own products.
//!
//! The drafter is on the device of the draft plan. A layer split has none.

use ares_core::Ledger;
use ares_cuda::{Device, DevicePtr, Stream};
use ares_runtime::ModelError;

use super::{Part, Prepared, StepEnv, invalid};
use crate::dflash::{Drafter, DrafterSetup, PassSeq};
use crate::qwen35::Config;
use crate::qwen35::device::Options;
use crate::qwen35::device::placement::Placement;
use crate::qwen35::weights::Weights;

/// Opens the drafter of `opts` on device `index` when that is the device of
/// the draft plan; `None` on any other device or without a drafter.
#[allow(clippy::too_many_arguments)]
pub(super) fn open(
    c: &Config,
    weights: &Weights<'_>,
    placement: &Placement,
    opts: &Options,
    index: usize,
    device: &Device,
    ledger: &mut Ledger,
    max_slots: u32,
) -> Result<Option<Drafter>, ModelError> {
    let Some(file) = &opts.dflash else { return Ok(None) };
    let Some(plan) = placement.draft.as_ref().filter(|d| d.device == index) else {
        return Ok(None);
    };
    file.config.fits(c.n_embd, c.n_layer, c.n_vocab).map_err(|e| invalid(e.to_string()))?;
    let w = file.weights().map_err(|e| invalid(e.to_string()))?;
    let mask_row = embedding_row(weights, file.config.mask_token)?;
    let setup = DrafterSetup {
        max_slots,
        max_rows: opts.max_rows,
        window_rows: opts.window_rows,
        head_rows: plan.head_rows,
    };
    Ok(Some(Drafter::open(&file.config, &w, device, ledger, setup, mask_row)?))
}

/// The row of `token` of the model's embedding table as f32, from the file.
fn embedding_row(weights: &Weights<'_>, token: u32) -> Result<Vec<f32>, ModelError> {
    let m = &weights.token_embd;
    if token as usize >= m.rows() {
        return Err(invalid(format!("token {token} past the embedding table")));
    }
    let mut out = vec![0f32; m.cols()];
    m.decode_row(token as usize, &mut out);
    Ok(out)
}

impl Part {
    /// Whether this part holds a DFlash drafter.
    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// The drafter of this part, for its report and its tests.
    pub fn dflash(&self) -> Option<&Drafter> {
        self.dflash.as_ref()
    }

    /// Tells the drafter that the step enqueued next has no pass.
    pub fn dflash_no_pass(&mut self) {
        if let Some(d) = &mut self.dflash {
            d.no_pass();
        }
    }

    /// The start of a step: the drafter is told its rows, whose taps the
    /// blocks give it and which it injects at the end.
    pub(super) fn dflash_begin(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let Some(d) = &mut self.dflash else { return Ok(()) };
        let rows: Vec<(u32, u32)> = env
            .segments
            .iter()
            .flat_map(|s| (0..s.n_rows).map(move |k| (s.slot, s.pos0 as u32 + k)))
            .collect();
        d.begin_step(env.stream, &rows)
    }

    /// Before block `i` (of the model, this part holds every block on the
    /// drafter's device): the residual stream of the step's rows (of the
    /// group being enqueued) when block `i` is a target layer.
    pub(super) fn dflash_tap(&mut self, env: &StepEnv<'_>, i: usize) -> Result<(), ModelError> {
        let Some(d) = &mut self.dflash else { return Ok(()) };
        let Some(tap) = d.config().target_layers.iter().position(|l| *l as usize == i) else {
            return Ok(());
        };
        let x =
            self.scratch.ptr().offset(self.s.x + u64::from(self.row_base) * u64::from(self.d) * 4);
        let rows = env.dims.n_rows;
        d.tap(&self.kernels, env.stream, tap, x, 1, 0, self.d, self.row_base, rows)
    }

    /// After the model's blocks: the step's rows into the drafter's ring.
    pub(super) fn dflash_inject(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let Some(d) = &mut self.dflash else { return Ok(()) };
        d.inject(&self.kernels, env.stream)
    }

    /// The drafter's pass for the windows `seqs` of a step that only drafts:
    /// the layers, the draft head and the selector. The drafts go into the
    /// step record's token array and into `list` at each window's place.
    pub fn dflash_pass(
        &mut self,
        stream: &Stream,
        record_tokens: DevicePtr,
        seqs: &[PassSeq],
        list: DevicePtr,
    ) -> Result<(), ModelError> {
        let Some(mut d) = self.dflash.take() else { return Err(invalid("no DFlash drafter")) };
        let result = (|| {
            let embed = self.embed.as_ref().ok_or_else(|| {
                invalid("the DFlash drafter needs the embedding table in host memory")
            })?;
            let shape = embed.shape();
            let decode = |token: u32| -> Vec<f32> {
                let mut out = vec![0f32; shape.n_embd as usize];
                if let Some(row) = embed.row(token) {
                    // A row of whole blocks decodes; a token past the table is
                    // zeros, as the gather gives it.
                    let _ = ares_quant::dequantize(shape.ty, row, &mut out);
                }
                out
            };
            let (h, rows) = d.pass_layers(&self.kernels, stream, seqs, &decode)?;
            let head = self.w.draft_head.ok_or_else(|| invalid("no draft head on this device"))?;
            // SAFETY: `h` holds `rows` rows of `d` values and the drafter's
            // logits `rows` rows of the draft head; the part's activation
            // scratch holds a step of `max_rows` rows, more than `rows`.
            unsafe {
                self.prepare(stream, h, self.d, rows, &[&head], Prepared::default())?;
                self.product(stream, &head, rows, d.logits())?;
            }
            d.pass_select(&self.kernels, stream, self.head_tokens(), record_tokens, list)
        })();
        self.dflash = Some(d);
        result
    }
}
