//! What an architecture implements to run in the step loop.
//!
//! The runtime knows steps, rows, devices and streams; it does not know what
//! a layer is. A model tells the runtime a few facts about itself (its
//! vocabulary, how it is spread over devices, what the host must supply per
//! row) and then, on each device's thread, enqueues the kernels of one step
//! for a batch of rows. [`SyntheticModel`](crate::SyntheticModel) is the one
//! implementation today, built from kernels that exist; the first real
//! architecture replaces it without touching the step loop.

use ares_api::{ChatFormat, EngineError, ModelCard, Prompt};
use ares_core::Ledger;
use ares_cuda::{Device, DevicePtr, Event, Kernels, Stream};
use ares_sampler::SampleParams;
use ares_state::PageLayout;

use crate::exec::Pick;
use crate::links::Links;
use crate::ram::PinnedRam;
use crate::record::RecordView;

/// How a model is spread over the devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// One device holds everything.
    Single,
    /// Every layer is on every device, each holding a slice; the devices sum
    /// their partial results twice per layer. All devices work on the same
    /// rows at the same time.
    Tensor,
    /// Each device holds a range of layers and hands the activations to the
    /// next at the boundary. The rows of a step flow through as a pipeline.
    Layer,
    /// Every layer is on every device, which exchange what they need among
    /// themselves, in memory they write to each other directly, ordered by
    /// the model's own kernels. The runtime builds no path between them, and
    /// the picks are made on the one device that holds the output head.
    Shared,
}

/// The sizes of the paths between devices that a model needs, so the runtime
/// can build them: the biggest sum (tensor split) and the ring of a hand-over
/// (layer split).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkSpec {
    /// The most f32 values one sum across devices takes.
    pub sum_values: u32,
    /// Parcels in flight on each hand-over between neighbouring devices. At
    /// least the number of groups of rows a step can be cut into.
    pub lane_slots: u32,
    /// The most f32 values one parcel takes.
    pub lane_values: u32,
}

/// What the host condenses of a sequence's tokens into the one number per row
/// that the step record carries (its *context*).
///
/// This exists because the synthetic model has no attention kernel to read
/// the sequence's earlier tokens: instead, the row's context selects a row of
/// a table that is added to the token's embedding. A model with attention
/// leaves it unused.
///
/// Two settings decide how repetitive the model's text is, which in turn
/// decides how well n-gram drafts do:
///
/// - `window`: the context is a function of the last `window` tokens (0: of
///   all tokens). A short window makes the text fall into a loop quickly,
///   because the next token is then a function of a small state.
/// - `epoch`: the function changes every `epoch` positions. In a loop, a draft
///   copied from earlier text is right until the epoch changes, and wrong
///   after it, so drafts are accepted and rejected both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextSpec {
    /// How many tokens the context reads, or 0 for all of them.
    pub window: u32,
    /// Positions per epoch, or 0 for one epoch.
    pub epoch: u32,
    /// The number of distinct contexts.
    pub buckets: u32,
}

/// The state before the first token.
const CONTEXT_SEED: u32 = 0x9e37_79b9;

fn mix(a: u32, b: u32) -> u32 {
    let mut h = u64::from(a) << 32 | u64::from(b);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h as u32
}

impl ContextSpec {
    /// A model that needs no context from the host.
    pub const NONE: ContextSpec = ContextSpec { window: 1, epoch: 0, buckets: 1 };

    /// The state of the token that ends `tail`, from the state of the token
    /// before it. `tail` holds the tokens up to and including this one; at
    /// least the last `window` of them (all of them for `window == 0` is not
    /// needed: the previous state stands for them).
    pub fn advance(&self, previous: u32, tail: &[u32]) -> u32 {
        let last = *tail.last().expect("a row has a token");
        if self.window == 0 {
            return mix(previous, last);
        }
        let from = tail.len().saturating_sub(self.window as usize);
        tail[from..].iter().fold(CONTEXT_SEED, |state, token| mix(state, *token))
    }

    /// The state before any token, for [`ContextSpec::advance`].
    pub fn start(&self) -> u32 {
        CONTEXT_SEED
    }

    /// The number the record carries for a row at `position` whose state is
    /// `state`.
    pub fn reduce(&self, state: u32, position: u32) -> u32 {
        let state = match position.checked_div(self.epoch) {
            Some(epoch) => mix(state, epoch),
            None => state,
        };
        state % self.buckets.max(1)
    }
}

/// Why a model's part of a step could not be built or enqueued.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Cuda(#[from] ares_cuda::Error),
    #[error(transparent)]
    Memory(#[from] ares_core::LedgerError),
    /// The configuration cannot be built on these devices.
    #[error("{0}")]
    Invalid(String),
    /// A device thread of the same step failed, so the exchange this one
    /// waited for will not happen.
    #[error("another device failed in the same step")]
    Aborted,
}

/// What a model offers the prefix cache: its device side carries out the
/// copies and the replay segments of the step record (see [`RecordView`](crate::RecordView)),
/// so that a prompt can start from pages and checkpoints that earlier requests
/// left.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixCache {
    /// The bytes one checkpoint of the model's recurrent state takes on each
    /// device, in the order of the device parts. All zero for a model without
    /// recurrent state, which needs no checkpoints and replays nothing.
    pub checkpoint_bytes: Vec<u64>,
    /// The most partly used pages (tails) the cache keeps for each slot of the
    /// runtime, or `None` for no limit. A model that keeps something on its
    /// devices for each of them, which cannot be dropped without the page, sizes
    /// it for this.
    pub tails_per_slot: Option<u32>,
}

impl PrefixCache {
    /// The bytes of one checkpoint on all devices together.
    pub fn bytes(&self) -> u64 {
        self.checkpoint_bytes.iter().sum()
    }
}

/// What a model offers the level of the prefix cache that is on SSD: its
/// device side moves pages and checkpoints between the devices and host
/// memory ([`DeviceModel::transfer`]), in a form that another load of the same
/// model can take back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Durable {
    /// Everything besides the tokens that changes the bytes of a page or of a
    /// checkpoint in transit: the weights, the formats and the precision of
    /// the state, and how the bytes are laid out over the devices. State
    /// written under another fingerprint is never read.
    pub fingerprint: Vec<u8>,
    /// Per device, the most bytes its share of a page takes in transit.
    pub page_bytes: Vec<u64>,
    /// Per device, the bytes of its share of a checkpoint.
    pub state_bytes: Vec<u64>,
    /// The cells before the start of a step that the device keeps in another
    /// form than the one a page ends in (a device brings pages into that form
    /// at the start of a step, by where the sequence stands then): a page is
    /// written when its sequence's last step began this far past it, or when
    /// the sequence ends. 0 for a model that stores a page in one form.
    pub settle_cells: u32,
}

/// What a device part is given to carry out a transfer.
#[derive(Debug)]
pub struct TransferContext<'a> {
    pub stream: &'a Stream,
    /// The memory of the state pages on this device, as in [`StepContext`].
    pub pages: DevicePtr,
}

/// One device's share of a [`Transfer`](crate::Transfer).
#[derive(Debug)]
pub struct DeviceTransfer {
    pub kind: crate::TransferKind,
    /// The pool page, or the checkpoint place.
    pub object: u32,
    pub keep_form: bool,
    /// Host memory with `room` bytes, aligned to 4096.
    pub ptr: *mut u8,
    pub room: usize,
    /// The bytes in use and their tag: read on the way in, set on the way out.
    pub len: usize,
    pub tag: u32,
}

// SAFETY: the pointer names memory of a transfer the conductor handed over
// and does not touch until every device has finished with it.
unsafe impl Send for DeviceTransfer {}

/// What a device part is built from.
#[derive(Debug)]
pub struct DeviceContext<'a> {
    /// Which device part this is, in `0..devices`.
    pub index: usize,
    pub devices: usize,
    pub device: &'a Device,
    pub kernels: &'a Kernels,
    pub stream: &'a Stream,
    /// The device's memory accounts: a part reserves every allocation here
    /// first, so a configuration that does not fit is refused at load, naming
    /// who holds what.
    pub ledger: &'a mut Ledger,
    /// The most rows a step has.
    pub max_rows: u32,
    /// The most sequences at once: slots of state that is kept per sequence.
    pub max_slots: u32,
    /// The pages of the pool and the cells of a page. The runtime has
    /// reserved and allocated `pages` times the model's bytes per page.
    pub pages: u32,
    pub cells_per_page: u32,
    /// Page-table entries per slot.
    pub table_stride: u32,
    /// The places for checkpoints of recurrent state that the conductor uses
    /// (the number [`GpuConfig::checkpoints`](crate::GpuConfig) names): a
    /// place holds [`PrefixCache::checkpoint_bytes`] of this device. 0 for a
    /// model without a prefix cache or a budget of none.
    pub checkpoints: u32,
    /// The engine's page-locked RAM. The places are locked here, in chunks, as
    /// they come into use ([`PinnedRam::lock_checkpoints`]).
    pub ram: &'a PinnedRam,
}

/// What the runtime gives a device part to enqueue one step.
#[derive(Debug)]
pub struct StepContext<'a> {
    pub index: usize,
    pub devices: usize,
    pub device: &'a Device,
    pub kernels: &'a Kernels,
    pub stream: &'a Stream,
    /// The step's rows, as the conductor wrote them.
    pub record: RecordView<'a>,
    /// The device copy of the record, uploaded on `stream` before the model
    /// is called.
    pub record_on_device: DevicePtr,
    /// The sequences' page tables on this device, `table_stride` entries per
    /// slot (a page number as [`Model::page_entry`] writes it), updated from
    /// the record before the model is called.
    pub tables: DevicePtr,
    pub table_stride: u32,
    /// The memory of the state pages on this device: the pool's pages times
    /// `bytes_per_page` bytes, which the model lays out (an array of pages per
    /// part of its state). Null when the model keeps no state in pages.
    pub pages: DevicePtr,
    pub bytes_per_page: u64,
    /// The paths to the other devices.
    pub links: &'a Links,
}

/// Where a device part left the logits of a step.
#[derive(Clone, Copy, Debug)]
pub struct Logits {
    /// The `i`-th row whose logits are wanted (in the order of the record's
    /// runs) has them at `ptr + i * stride * 4`, as f32. Rows nobody asked
    /// for have none: a row of a real vocabulary is a megabyte.
    pub ptr: DevicePtr,
    /// Values per row.
    pub stride: u32,
    /// The token of the first value of a row: a device that holds a slice of
    /// the vocabulary reports where its slice starts.
    pub first_token: u32,
}

/// Where a device part left the tokens it drafted itself for the windows of a
/// step: `tokens` u32 on the device, the drafts of each window with the
/// record's own-drafts flag in the order of the record's segments, window
/// after window. `u32::MAX` is a draft the part could not make.
#[derive(Clone, Copy, Debug)]
pub struct OwnDrafts {
    pub ptr: DevicePtr,
    pub tokens: u32,
}

/// One device's part of a model.
pub trait DeviceModel: Send {
    /// The number of logits per row this device part produces (0 for a part
    /// without an output head), so the runtime can size its read-back buffer
    /// before the first step.
    fn logits_per_row(&self) -> u32;

    /// Enqueues the kernels of one step for all rows of `step` on its stream,
    /// after the commit of the draft windows of the step before, whose
    /// outcome the record carries.
    ///
    /// Must not wait for the device: not for a stream, an event or a copy.
    /// The order between devices is kept on the devices, through the links.
    /// Returns where this device left the logits, or `None` for a device
    /// that holds no output head.
    fn enqueue(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError>;

    /// Waits until the step has ended on the device: `end` is recorded behind
    /// everything the step enqueued, the sampler's kernels and the read-back
    /// included. The default blocks in the driver. A model whose step needs
    /// the host to go on while the device runs (kernels that wait for an
    /// expert the host brings) serves it here and returns when `end` has
    /// happened and the host has nothing left to do for the step.
    fn wait(&mut self, end: &Event) -> Result<(), ModelError> {
        Ok(end.synchronize()?)
    }

    /// The tokens picked for the wanted rows of the step that has just ended,
    /// in the order of the record's runs, before the conductor has seen them.
    /// Only the device that holds the output head is told.
    fn picked(&mut self, _picks: &[Pick]) {}

    /// The tokens of the columns of this part's logits, a u32 per column on
    /// the device, when column `i` is not token `first_token + i`; null
    /// otherwise.
    fn logit_tokens(&self) -> DevicePtr {
        DevicePtr::NULL
    }

    /// Where the step enqueued last left the tokens this part drafted itself
    /// (a model with a draft block of its own); `None` when it drafted none.
    fn own_drafts(&self) -> Option<OwnDrafts> {
        None
    }

    /// Carries out this device's share of a transfer of the prefix cache's
    /// level on SSD, between steps. The part may enqueue copies on the stream;
    /// the runtime waits for the stream afterwards. Only a part of a model
    /// that says it is [`Model::durable`] is asked.
    fn transfer(
        &mut self,
        _cx: &TransferContext<'_>,
        _item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        Err(ModelError::Invalid("this model moves no state to the host".into()))
    }

    /// Device milliseconds the part's own drafter took in the step that ran
    /// last. Asked after that step has ended.
    fn own_draft_ms(&mut self) -> f64 {
        0.0
    }
}

/// An architecture.
pub trait Model: Send + Sync + 'static {
    /// The model as clients see it.
    fn card(&self) -> ModelCard;

    /// The number of tokens; picks are below it.
    fn vocab(&self) -> u32;

    /// Whether `token` ends a generation.
    fn ends(&self, token: u32) -> bool;

    /// The tokens of a prompt. Tokenizing is the model's business; refusing
    /// what it cannot tokenize is too.
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError>;

    /// What the model's chat template declares: how the model marks its
    /// thinking, which the front end needs to tell the reasoning in a reply
    /// from the answer. The default is `None`, a model whose template is not
    /// read: its replies are one text. Cheap, since it is asked for every
    /// chat request.
    fn chat_format(&self) -> Option<ChatFormat> {
        None
    }

    /// Appends the bytes of a generated token's text to `text`: nothing for
    /// a token that prints nothing, and possibly a part of a character, which
    /// the next tokens complete (the conductor holds bytes back until they
    /// are whole characters).
    fn piece(&self, token: u32, text: &mut Vec<u8>);

    /// What the host supplies per row besides sequence, position and token.
    fn context(&self) -> ContextSpec;

    fn split(&self) -> Split;

    /// The number of device parts, which is the number of devices needed.
    fn devices(&self) -> usize;

    /// The paths between devices the model needs.
    fn links(&self) -> LinkSpec;

    /// How the experts the routers picked were found since the engine
    /// started, for a model that streams them; `None` for any other.
    fn expert_stats(&self) -> Option<ares_api::ExpertStats> {
        None
    }

    /// The page-locked RAM the model's device parts hold besides what the
    /// runtime books for the step record and the read-back (the buffers a
    /// step needs, whatever the cache of checkpoints or a tier of experts
    /// holds), so that the engine's RAM limit counts it.
    fn pinned_ram(&self) -> u64 {
        0
    }

    /// The most rows of one step the model's buffers hold.
    fn max_rows(&self) -> u32;

    /// The rows of a piece of prompt: where the model's prompt kernels reach
    /// their rate. The conductor reads a prompt next to generating sequences
    /// in steps of this many rows (a smaller step pays the pass over the
    /// weights again and reads the prompt slower, a larger one stops the
    /// generating sequences for longer), and a prompt with no more rows left
    /// is read whole. At most [`Model::max_rows`]; that is the default.
    fn prompt_rows(&self) -> u32 {
        self.max_rows()
    }

    /// The most rows of one step whose logits can be wanted: every row of a
    /// generating sequence (its newest token and its drafts) and the last
    /// row of a prompt.
    fn logit_rows(&self) -> u32;

    /// The most rows of a step in which a sequence verifies drafts, or 0 for
    /// a model that takes none. A model with recurrent state keeps the
    /// inputs of such a step's rows until the next step commits the accepted
    /// ones, so it bounds them.
    fn draft_rows(&self) -> u32;

    /// How a request's tokens are picked where the request does not say: the
    /// sampling settings that suit a model are the model's.
    fn sampling(&self) -> SampleParams {
        SampleParams::GREEDY
    }

    /// The most cells of its sequence a row reads through attention:
    /// everything before it for dense attention (the default), the cells a
    /// sparse attention selects at most, 0 for a model without attention.
    /// The step time model counts the cells a step reads with it, so that a
    /// row deep in a long context is known to cost more than one at its start.
    fn attention_span(&self) -> u32 {
        u32::MAX
    }

    /// The most tokens the model drafts itself per sequence and step, with a
    /// draft block of its own that runs on the devices; 0 for a model
    /// without one. A window that takes them has the record's own-drafts
    /// flag and no tokens for its draft rows.
    fn own_drafts(&self) -> u32 {
        0
    }

    /// `Some` when the model's device side carries out the state copies, the
    /// page copies and the replay segments of the step record and keeps the
    /// places for checkpoints, which makes the runtime cache the pages and
    /// the recurrent state of prompts and start new prompts from them. The
    /// default is `None`: a model whose device side does not do this has no
    /// prefix cache, and every prompt is computed from its first token. A model
    /// with recurrent state must not answer `Some` before its device side
    /// restores, saves and replays: it would resume from state that is not there.
    fn prefix_cache(&self) -> Option<PrefixCache> {
        None
    }

    /// `Some` when the model's device side also moves pages and checkpoints
    /// to host memory and back ([`DeviceModel::transfer`]), which lets the
    /// prefix cache keep them on SSD. Only asked of a model with a prefix
    /// cache, for pages of `cells_per_page` cells.
    fn durable(&self, cells_per_page: u32) -> Option<Durable> {
        let _ = cells_per_page;
        None
    }

    /// The page-table entry of pool page `page`, as the model's kernels read
    /// it (a model may mark which of its pools a page is in).
    fn page_entry(&self, page: u32) -> u32;

    /// The state the model keeps per position of a sequence, on each device,
    /// for pages of `cells_per_page` cells. The runtime reserves the pool's
    /// pages on each device from it, in the device's ledger, before the model
    /// is built.
    fn page_layout(&self, cells_per_page: u32) -> PageLayout;

    /// Device part `part` could not be built (or its thread ended before it
    /// was): a model whose parts wait for each other while they open, to tell
    /// each other what they measured, lets the others go. Called on the
    /// thread of the part that failed, once, and not for a part that is open.
    fn abandon(&self, part: usize) {
        let _ = part;
    }

    /// Builds device part `cx.index`. Called on that device's thread.
    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(window: u32, epoch: u32) -> ContextSpec {
        ContextSpec { window, epoch, buckets: 1024 }
    }

    /// The reduced contexts of every position of `tokens`, as the conductor
    /// builds them: one `advance` per token from the state of the one before.
    fn contexts(spec: &ContextSpec, tokens: &[u32]) -> Vec<u32> {
        let mut state = spec.start();
        (0..tokens.len())
            .map(|i| {
                state = spec.advance(state, &tokens[..=i]);
                spec.reduce(state, i as u32)
            })
            .collect()
    }

    #[test]
    fn a_window_context_depends_only_on_the_last_tokens() {
        let spec = spec(2, 0);
        let a = contexts(&spec, &[9, 9, 5, 6, 7]);
        let b = contexts(&spec, &[1, 2, 3, 6, 7]);
        assert_eq!(a[4], b[4], "the same last two tokens give the same context");
        assert_ne!(a[3], a[4]);
        assert_ne!(a[4], contexts(&spec, &[1, 2, 3, 5, 7])[4]);
    }

    #[test]
    fn a_full_context_depends_on_every_token() {
        let spec = spec(0, 0);
        let a = contexts(&spec, &[1, 2, 3, 4]);
        assert_ne!(a[3], contexts(&spec, &[9, 2, 3, 4])[3]);
        assert_ne!(a[3], contexts(&spec, &[1, 2, 4, 4])[3]);
        // A prefix gives the same contexts as the longer sequence at those positions.
        assert_eq!(a[..2], contexts(&spec, &[1, 2])[..]);
    }

    #[test]
    fn an_epoch_changes_the_context_of_the_same_tokens() {
        let spec = spec(1, 4);
        let tokens = [7; 12];
        let c = contexts(&spec, &tokens);
        assert_eq!(c[0], c[3], "within an epoch the same token has the same context");
        assert_ne!(c[3], c[4], "the next epoch differs");
        assert_eq!(c[4], c[7]);
        assert!(c.iter().all(|c| *c < 1024));
    }
}
