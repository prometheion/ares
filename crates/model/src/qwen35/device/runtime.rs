//! `qwen35` as the runtime drives it: the [`Model`] that tells the runtime
//! what it is and builds a [`Part`] on each device's thread, and the
//! [`DeviceModel`] that turns the runtime's step into the part's.

use std::ops::Range;
use std::sync::Mutex;
use std::time::Duration;

use ares_api::{ChatFormat, EngineError, ModelCard, Prompt};
use ares_cuda::{DevicePtr, ENTRY_CELLS, PushSlot, Segment, Stream};
use ares_gguf::Gguf;
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, DeviceTransfer, Drafter, DrafterKind, Durable,
    LinkSpec, Links, Logits, Model, ModelDrafts, ModelError, PrefixCache, Split, StepContext,
    TransferContext,
};
use ares_state::PageLayout;
use ares_tokenizer::{ChatOptions, Tokenizer};

use super::choose::{Balance, LayoutReport};
use super::load::table_shape;
use super::part::{
    Exchange, MTP_KV, MtpBytes, MtpChain, MtpStep, Options, PAGE_ENTRY, Part, PartSetup, StepEnv,
    TAILS_PER_SLOT, declared, page_bytes,
};
use super::placement::{HalfShares, Placement, PlacementError};
use crate::blocks::rows::{EmbeddingReport, Stage};
use crate::blocks::slots::{self, Declared};
use crate::chat;
use crate::dflash::PassSeq;
use crate::qwen35::{Config, Weights};

/// How to spread the model over the devices.
#[derive(Clone, Debug, PartialEq)]
pub enum Spread {
    /// Everything on one device.
    Single,
    /// A tensor split with these shares per device.
    Tensor(Vec<f64>),
    /// A tensor split with shares per half block, and these for the head.
    TensorByHalf(Vec<f64>, HalfShares),
    /// A layer split with these shares of the weights' bytes per device.
    Layer(Vec<f64>),
}

/// What one device's part took, for the report at start.
#[derive(Clone, Debug, PartialEq)]
pub struct PartReport {
    /// The part's index in the placement.
    pub index: usize,
    /// The device's name.
    pub device: String,
    /// The number of blocks (or slices of blocks) on the device.
    pub blocks: usize,
    pub weight_bytes: u64,
    /// How long the upload of the weights took.
    pub upload: Duration,
    /// Recurrent state per sequence.
    pub state_bytes_per_slot: u64,
    /// One checkpoint of this device's share: the recurrent state of a
    /// sequence and, where the MTP block is, its carry row.
    pub checkpoint_bytes: u64,
    /// The scratch of a step of the most rows.
    pub scratch_bytes: u64,
    /// The kept inputs of draft windows.
    pub kept_bytes: u64,
    /// K and V per page of 128 cells.
    pub page_bytes: u64,
    /// The exact K/V entries of the attention blocks, when their formats have a
    /// tile mean: the second, small pool.
    pub exact_bytes: u64,
    /// What the model's MTP block takes, on the device that holds it.
    pub mtp: Option<MtpBytes>,
    /// Where the token embedding lies and what it takes.
    pub embedding: EmbeddingReport,
    /// Which weight matrices got which layout, and what the choice took.
    pub layouts: LayoutReport,
    /// Every reservation in the device's ledger once the part was built: the
    /// runtime's (step record, page tables, pool of pages) and the part's own.
    pub ledger: Vec<(String, u64)>,
}

/// Why a model could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error(transparent)]
    Model(#[from] crate::qwen35::Error),
    #[error(transparent)]
    Placement(#[from] PlacementError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

/// A `qwen35` model file, placed.
pub struct Qwen35 {
    gguf: Gguf,
    config: Config,
    placement: Placement,
    options: Options,
    tokenizer: Tokenizer,
    name: String,
    context: u32,
    /// When the model was set up, in seconds since the Unix epoch.
    created: u64,
    /// How chat prompts are rendered.
    chat: ChatOptions,
    /// What the chat template declares, read with `chat`.
    format: ChatFormat,
    reports: Mutex<Vec<PartReport>>,
    /// Where the parts of a tensor split, which the runtime opens at the
    /// same time, tell each other what the weight products cost them.
    balance: Option<Balance>,
    /// The page-locked host memory the device that embeds holds for the token
    /// embedding: none when it keeps the whole table.
    pinned: u64,
}

impl std::fmt::Debug for Qwen35 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35")
            .field("name", &self.name)
            .field("split", &self.placement.split)
            .field("devices", &self.placement.n_devices())
            .finish_non_exhaustive()
    }
}

impl Qwen35 {
    /// Reads the configuration and the tokenizer of `gguf` and decides the
    /// placement. `context` is the longest sequence the model is served with
    /// (at most the model's own context). Nothing is uploaded until the
    /// runtime opens the parts.
    pub fn new(
        gguf: Gguf,
        name: impl Into<String>,
        spread: &Spread,
        options: Options,
        context: u32,
    ) -> Result<Self, OpenError> {
        let config = Config::from_gguf(&gguf)?;
        let (placement, table) = {
            let weights = Weights::load(&gguf, &config)?;
            let table = table_shape(&weights.token_embd);
            let placement = match spread {
                Spread::Single => Placement::single(&config),
                Spread::Tensor(shares) => Placement::tensor(&config, shares)?,
                Spread::TensorByHalf(shares, halves) => {
                    Placement::tensor_by_half(&config, shares, halves)?
                }
                Spread::Layer(shares) => Placement::layer(&config, &weights, shares)?,
            };
            (placement, table)
        };
        // A model with a next-token-prediction block drafts with it wherever
        // it can (and is asked to): when windows may carry drafts, and the
        // device that embeds has the first rows of the head (not a layer
        // split). A DFlash drafter needs the same device and head.
        let drafts = options.window_rows > 1 && placement.split != Split::Layer;
        let mtp = config.n_nextn == 1 && options.mtp;
        let vocabulary = options.draft_vocabulary.as_deref();
        let placement = if drafts && mtp {
            placement.with_mtp(&config, vocabulary)?
        } else if drafts && options.dflash.is_some() {
            placement.with_head_view(&config, vocabulary, false)?
        } else {
            placement
        };
        if let Some(file) = &options.dflash {
            file.config
                .fits(config.n_embd, config.n_layer, config.n_vocab)
                .map_err(|e| PlacementError(e.to_string()))?;
        }
        let tokenizer =
            Tokenizer::from_gguf(&gguf).map_err(|e| OpenError::Tokenizer(e.to_string()))?;
        let context = context.clamp(1, config.n_ctx as u32);
        let balance = (placement.split == Split::Tensor && placement.n_devices() > 1)
            .then(|| Balance::new(placement.n_devices()));
        let pinned = if placement.stages_rows(&config) {
            table.bytes() + Stage::locked_bytes(&table, options.max_rows)
        } else {
            0
        };
        let format = chat::chat_format(&tokenizer, &ChatOptions::default());
        Ok(Self {
            gguf,
            config,
            placement,
            options,
            tokenizer,
            name: name.into(),
            context,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_secs()),
            format,
            chat: ChatOptions::default(),
            reports: Mutex::new(Vec::new()),
            balance,
            pinned,
        })
    }

    /// Renders chat prompts with `options`: whether the model thinks, and the
    /// variables a server passes to the chat template (`reasoning_effort`
    /// and the like).
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.format = chat::chat_format(&self.tokenizer, &options);
        self.chat = options;
        self
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Where the model's parts are.
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The tokenizer of the model file.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// What the parts took, by part index, once the runtime has opened them.
    pub fn reports(&self) -> Vec<PartReport> {
        let mut reports = self.reports.lock().unwrap_or_else(|e| e.into_inner()).clone();
        reports.sort_by_key(|r| r.index);
        reports
    }

    /// The drafts of one pass of the DFlash drafter, when it drafts: a block's
    /// less the newest token, at most a window's.
    fn dflash_drafts(&self) -> Option<u32> {
        let file = self.options.dflash.as_ref()?;
        self.placement.draft.as_ref()?;
        Some((file.config.drafts() as u32).min(self.options.window_rows.saturating_sub(1)))
    }

    /// The most rows of a sum across the devices that is pushed.
    fn rows_per_sum(&self) -> u32 {
        (ares_cuda::PUSH_MAX_VALUES / self.config.n_embd as u32).clamp(1, self.options.max_rows)
    }
}

impl Qwen35 {
    /// What each device part keeps of a sequence and of a page
    /// (`part/resume.rs`), which the prefix cache is told.
    fn declared(&self) -> Vec<Declared> {
        (0..self.placement.n_devices())
            .map(|d| declared(&self.config, &self.placement, d, &self.options))
            .collect()
    }

    /// Everything besides the tokens that changes the bytes of a page or of a
    /// checkpoint as the device parts hand them out
    /// ([`ares_runtime::Durable::fingerprint`]):
    ///
    /// - the model's files (size, header and samples of the weights);
    /// - the formats of K and of V of the attention blocks, and the draft
    ///   block's when the model drafts (its keys, values and carry are part
    ///   of the state);
    /// - the format of the recurrent state, and how the products round;
    /// - which heads of every block each device holds, since the bytes are
    ///   laid out per device.
    fn state_fingerprint(&self) -> Vec<u8> {
        let mut out = b"qwen35 state 1\0".to_vec();
        for part in 0..self.gguf.n_parts() {
            crate::files::identity(self.gguf.path(part as u16), &mut out);
        }
        let draft = self.placement.mtp().map(|draft| (draft.device, MTP_KV));
        let settings = format!(
            "kv {:?} draft {:?} state {:?} precision {:?} split {:?}",
            self.options.kv,
            draft,
            self.options.state,
            self.options.precision,
            self.placement.split
        );
        out.extend_from_slice(settings.as_bytes());
        for device in &self.placement.devices {
            out.push(0xff);
            for block in &device.blocks {
                let heads = block.as_ref().map_or(0..0, |slice| slice.heads.clone());
                out.extend_from_slice(&heads.start.to_le_bytes());
                out.extend_from_slice(&heads.end.to_le_bytes());
            }
        }
        out
    }
}

impl Model for Qwen35 {
    fn card(&self) -> ModelCard {
        ModelCard { id: self.name.clone(), context_length: self.context, created: self.created }
    }

    fn vocab(&self) -> u32 {
        self.config.n_vocab as u32
    }

    fn ends(&self, token: u32) -> bool {
        self.tokenizer.is_end_of_generation(token)
    }

    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        chat::encode(&self.tokenizer, &self.chat, prompt)
    }

    fn chat_format(&self) -> Option<ChatFormat> {
        Some(self.format.clone())
    }

    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        chat::piece(&self.tokenizer, token, text);
    }

    fn context(&self) -> ContextSpec {
        ContextSpec::NONE
    }

    fn split(&self) -> Split {
        self.placement.split
    }

    fn devices(&self) -> usize {
        self.placement.n_devices()
    }

    fn links(&self) -> LinkSpec {
        let d = self.config.n_embd as u32;
        let o = &self.options;
        // A layer split hands a long step over in groups of `pipeline_rows`
        // rows (`Part::plan_groups`) and any other step whole: a parcel is a
        // group, a step shorter than two groups, or a step with drafts.
        let (parcels, rows) = match o.pipeline_rows {
            0 => (1, o.max_rows),
            per => (o.max_rows.div_ceil(per), o.max_rows.min((2 * per - 1).max(o.window_rows))),
        };
        if self.placement.split == Split::Tensor {
            // A tensor split sums a long step in groups of at most
            // `pipeline_rows` rows, each in a slot of its own, and any other
            // step whole. A parcel is the wire form of a group's partial
            // result, counted here in words of four bytes.
            let rows = match o.pipeline_rows {
                0 => o.max_rows,
                per => o.max_rows.min(per.max(o.window_rows)),
            };
            return LinkSpec {
                sum_values: self.rows_per_sum() * d,
                lane_slots: parcels,
                lane_values: ares_cuda::partial_wire_bytes(o.wire(), rows, d).div_ceil(4) as u32,
            };
        }
        LinkSpec {
            sum_values: self.rows_per_sum() * d,
            // A slot more than a step sends, so the sender is never held back
            // by a receiver that still reads.
            lane_slots: parcels + 1,
            lane_values: rows * d,
        }
    }

    fn pinned_ram(&self) -> u64 {
        self.pinned
    }

    fn max_rows(&self) -> u32 {
        self.options.max_rows
    }

    fn prompt_rows(&self) -> u32 {
        // The group in which a long step is run: where the prompt kernels
        // reach their rate (see `Options::pipeline_rows`).
        match self.options.pipeline_rows {
            0 => self.options.max_rows,
            rows => rows.min(self.options.max_rows),
        }
    }

    fn logit_rows(&self) -> u32 {
        self.options.logit_rows
    }

    fn draft_rows(&self) -> u32 {
        self.options.window_rows
    }

    fn drafters(&self) -> Vec<Drafter> {
        let mut drafters = Vec::new();
        if self.placement.mtp().is_some() {
            drafters.push(Drafter {
                kind: DrafterKind::Mtp,
                drafts: self.options.window_rows.saturating_sub(1),
            });
        }
        if let Some(drafts) = self.dflash_drafts() {
            drafters.push(Drafter { kind: DrafterKind::Dflash, drafts });
        }
        drafters
    }

    fn prefix_cache(&self) -> Option<PrefixCache> {
        Some(slots::prefix_cache(&self.declared(), Some(TAILS_PER_SLOT)))
    }

    fn durable(&self, cells_per_page: u32) -> Option<Durable> {
        if cells_per_page != ENTRY_CELLS {
            return None;
        }
        Some(slots::durable(&self.declared(), self.state_fingerprint()))
    }

    fn page_entry(&self, page: u32) -> u32 {
        page | PAGE_ENTRY
    }

    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        // The attention kernels store 128 cells per entry; `open` refuses any
        // other page size, so the bytes are those of 128 cells.
        (0..self.placement.n_devices()).fold(
            PageLayout::new(self.placement.n_devices(), cells_per_page),
            |layout, device| {
                let bytes = page_bytes(&self.config, &self.placement, device, &self.options);
                layout.per_page(device, "K and V", bytes)
            },
        )
    }

    fn abandon(&self, part: usize) {
        // The parts of a tensor split open at the same time and tell each
        // other what they measured: one that fails first must not keep the
        // others waiting for it.
        if let Some(balance) = &self.balance {
            balance.give_up(part);
        }
    }

    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        if cx.cells_per_page != ENTRY_CELLS {
            return Err(ModelError::Invalid(format!(
                "pages of {} cells; the K/V entries of this model hold {ENTRY_CELLS}",
                cx.cells_per_page
            )));
        }
        let weights = Weights::load(&self.gguf, &self.config)
            .map_err(|e| ModelError::Invalid(e.to_string()))?;
        let index = cx.index;
        let part = Part::open_balanced(
            &self.config,
            &weights,
            &self.placement,
            &self.options,
            PartSetup {
                index,
                device: cx.device,
                kernels: cx.kernels,
                ledger: cx.ledger,
                max_slots: cx.max_slots,
                pages: cx.pages,
                table_stride: cx.table_stride,
                checkpoints: cx.checkpoints,
                ram: cx.ram,
            },
            self.balance.as_ref(),
        )?;
        self.reports.lock().unwrap_or_else(|e| e.into_inner()).push(PartReport {
            index,
            device: cx.device.info().name.clone(),
            blocks: part.n_blocks(),
            weight_bytes: part.weight_bytes(),
            upload: part.upload_time(),
            state_bytes_per_slot: part.state_bytes_per_slot(),
            checkpoint_bytes: part.checkpoint_bytes(),
            scratch_bytes: part.scratch_bytes(),
            kept_bytes: part.kept_bytes(),
            page_bytes: part.page_bytes(),
            exact_bytes: part.exact_bytes(),
            mtp: part.mtp_bytes(),
            embedding: part.embedding_report(),
            layouts: part.layouts().clone(),
            ledger: cx.ledger.entries().map(|(name, bytes)| (name.to_owned(), bytes)).collect(),
        });
        // The ids of the drafters in the record, in the order of `drafters`.
        let block_id = self.placement.mtp().map(|_| 1);
        let dflash_id = self.dflash_drafts().map(|_| 1 + u32::from(block_id.is_some()));
        Ok(Box::new(Qwen35Device {
            part,
            tokens: Vec::new(),
            segments: Vec::new(),
            replay: Vec::new(),
            chains: Vec::new(),
            passes: Vec::new(),
            block_id,
            dflash_id,
        }))
    }
}

/// A part behind the runtime's interface.
#[derive(Debug)]
struct Qwen35Device {
    part: Part,
    /// The tokens of the step's rows, read from the record.
    tokens: Vec<u32>,
    /// The segments of the step, read from the record once.
    segments: Vec<Segment>,
    /// The rows of the step that are replays.
    replay: Vec<Range<u32>>,
    /// The windows of the step whose drafts the model's MTP block makes,
    /// and those the DFlash drafter makes.
    chains: Vec<MtpChain>,
    passes: Vec<PassSeq>,
    /// The ids the record names the two drafters by, when the model has them.
    block_id: Option<u32>,
    dflash_id: Option<u32>,
}

/// The runtime's links as a part uses them.
struct LinkExchange<'a> {
    links: &'a Links,
    index: usize,
    stream: &'a Stream,
}

impl Exchange for LinkExchange<'_> {
    fn push_bytes(&self) -> u64 {
        self.links.push_bytes()
    }

    fn push(&self) -> Result<PushSlot, ModelError> {
        self.links.push(self.index)
    }

    fn slot_bytes(&self) -> u64 {
        self.links.slot_bytes()
    }

    fn outbox(&self, slot: u32) -> Result<DevicePtr, ModelError> {
        self.links.outbox(self.index, slot)
    }

    fn post(&self, slot: u32, bytes: u64) -> Result<(), ModelError> {
        self.links.post(self.index, slot, bytes)
    }

    fn arrive(&self, slot: u32) -> Result<DevicePtr, ModelError> {
        self.links.arrive(self.index, slot)
    }

    fn consumed(&self, slot: u32) -> Result<(), ModelError> {
        self.links.consumed(self.index, slot)
    }

    fn send(&self, src: DevicePtr, n: u32) -> Result<(), ModelError> {
        self.links.send(self.index, self.stream, src, n)
    }

    fn receive(&self) -> Result<DevicePtr, ModelError> {
        self.links.receive(self.index - 1, self.stream)
    }

    fn release(&self) -> Result<(), ModelError> {
        self.links.release(self.index - 1, self.stream)
    }
}

impl DeviceModel for Qwen35Device {
    fn logits_per_row(&self) -> u32 {
        self.part.logits_per_row()
    }

    fn enqueue(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        let record = &step.record;
        // The draft windows of the step before settle the cells they accepted
        // (in the tables of that step, which this record's updates may replace),
        // and the pages the pool freed give back what the part keeps for them.
        self.part.commit_windows((0..record.commits()).map(|i| {
            let commit = record.commit(i);
            (commit.slot, commit.accepted)
        }));
        self.part.release_pages((0..record.freed()).map(|i| record.freed_page(i)));
        if record.rows() == 0 {
            return Ok(None);
        }
        // The page tables as the device copy will be when the step runs.
        let mut taken = 0;
        for i in 0..record.updates() {
            let update = record.update(i);
            let entries = (taken..taken + update.count).map(|k| record.entry(k));
            self.part.update_table(update.slot, update.first, entries)?;
            taken += update.count;
        }
        // Those windows are committed to recurrent state, and the copies of the
        // prefix cache follow (saves, page copies, restores: the record's
        // documentation has the order and the reasons); then this step.
        if let Some((offset, dims)) = record.commit_record() {
            self.part.enqueue_commit(step.stream, step.record_on_device.offset(offset), dims)?;
        }
        // The MTP block follows the rows of the last step that count, before
        // the copies take its cells.
        self.part.mtp_follow(step.stream, step.tables, step.pages)?;
        self.part.enqueue_copies(step.stream, step.pages, record)?;
        self.tokens.clear();
        self.tokens.extend((0..record.rows()).map(|i| record.row(i).token));
        self.segments.clear();
        self.segments.extend((0..record.segments()).map(|i| record.segment(i)));
        self.replay.clear();
        for (i, segment) in self.segments.iter().enumerate() {
            if record.replay(i as u32) {
                self.replay.push(segment.row0..segment.row0 + segment.n_rows);
            }
        }
        // The windows whose drafts a drafter of the part makes, each at its
        // place in the step's list of drafts: the MTP block's chains, with
        // their newest token and the settings of the rows that verify the
        // drafts (all rows of a window are wanted, so the sampler rows of a
        // window are consecutive; `wanted` counts the wanted rows before it),
        // and the DFlash drafter's passes.
        self.chains.clear();
        self.passes.clear();
        let (mut bar, mut place) = (0, 0);
        for (i, segment) in self.segments.iter().enumerate() {
            let Some(drafter) = record.drafter(i as u32) else { continue };
            if segment.n_rows < 2 {
                continue;
            }
            let links = segment.n_rows - 1;
            // The bars of the window's links, when the record has them.
            let needs = if record.draft_bars() >= bar + links {
                (bar..bar + links).map(|k| record.draft_need(k)).collect()
            } else {
                Vec::new()
            };
            bar += links;
            if Some(drafter) == self.block_id {
                let wanted: u32 = (0..record.runs())
                    .map(|r| record.run(r))
                    .map(|(first, count)| segment.row0.clamp(first, first + count) - first)
                    .sum();
                self.chains.push(MtpChain {
                    segment: i as u32,
                    token: record.row(segment.row0).token,
                    rows: (0..links).map(|j| record.sampler(wanted + j).for_draft()).collect(),
                    needs,
                    place,
                });
            } else if Some(drafter) == self.dflash_id {
                self.passes.push(PassSeq {
                    slot: segment.slot,
                    position: segment.pos0 as u32,
                    newest: record.row(segment.row0).token,
                    record_row: segment.row0 + 1,
                    wanted: links,
                    place,
                });
            } else {
                return Err(ModelError::Invalid(format!("the part has no drafter {drafter}")));
            }
            place += links;
        }
        if self.part.has_mtp() {
            self.part.mtp_step(MtpStep {
                tokens: step.record_on_device.offset(record.tokens_offset()),
                chains: std::mem::take(&mut self.chains),
                chances: record.draft_chances(),
            })?;
        }
        if self.part.has_dflash() && self.passes.is_empty() {
            self.part.dflash_no_pass();
        }
        let exchange = LinkExchange { links: step.links, index: step.index, stream: step.stream };
        if record.drafts_only() {
            // The drafts and nothing of the model: the round that verifies
            // them is the next record. The MTP block's chains first, then
            // the DFlash drafter's passes into the same list.
            if self.part.has_mtp() {
                self.part.enqueue_chains(&StepEnv {
                    stream: step.stream,
                    step: step.record_on_device,
                    dims: record.dims(),
                    tokens: &self.tokens,
                    segments: &self.segments,
                    tables: step.tables,
                    pages: step.pages,
                    exchange: None,
                    replay: &self.replay,
                })?;
            }
            // The drafter is on one device; the others have nothing to do.
            if !self.passes.is_empty() && self.part.has_dflash() {
                let list = self.part.draft_list();
                let tokens = step.record_on_device.offset(record.tokens_offset());
                self.part.dflash_pass(step.stream, tokens, &self.passes, list)?;
            }
            return Ok(None);
        }
        self.part.enqueue(&StepEnv {
            stream: step.stream,
            step: step.record_on_device,
            dims: record.dims(),
            tokens: &self.tokens,
            segments: &self.segments,
            tables: step.tables,
            pages: step.pages,
            exchange: (step.devices > 1).then_some(&exchange as &dyn Exchange),
            replay: &self.replay,
        })
    }

    fn transfer(
        &mut self,
        cx: &TransferContext<'_>,
        item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        self.part.transfer(cx.stream, cx.pages, item)
    }

    fn logit_tokens(&self) -> DevicePtr {
        self.part.head_tokens()
    }

    fn model_drafts(&self) -> Option<ModelDrafts> {
        self.part.drafted().map(|d| ModelDrafts { ptr: d.ptr, tokens: d.tokens })
    }

    fn model_draft_probs(&mut self, out: &mut Vec<f32>) {
        self.part.draft_probs(out);
    }

    fn model_draft_ms(&mut self) -> f64 {
        // A failed read of the events is not worth failing the step for:
        // the cost then stays what it was measured as before.
        self.part.draft_ms().unwrap_or(0.0)
    }
}
