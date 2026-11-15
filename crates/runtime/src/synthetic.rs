//! A model made of the kernels that exist, for building and profiling the
//! step loop before an architecture does.
//!
//! Per layer it runs two blocks, each shaped like the blocks of a transformer
//! as far as the host sees them: an RMS norm, the activation quantizer, a
//! decode weight kernel on a random quantized matrix that widens the row
//! (`width -> inner`), a norm and a quantizer again, a second weight kernel
//! that narrows it (`inner -> width`), and an add into the residual stream.
//! The first block's inner width stands for attention, the second's for the
//! feed-forward network. Weight bytes, kernel launches and, under a tensor
//! split, sums across devices are what a real layer has; the arithmetic in
//! between is not a transformer's.
//!
//! The stream ends in a norm and a projection to the vocabulary, and the
//! pick is the arg-max of its logits, taken on the host from a small
//! read-back. A GPU sampler does not exist yet; this read-back is the one
//! synchronisation point of a step and its cost shows in the timing
//! ("wait for device" and "arg-max").
//!
//! # What the next token depends on
//!
//! A row's input is the sum of a table row chosen by its token and a table
//! row chosen by its *context*, a number the host derives from the sequence's
//! tokens ([`ContextSpec`]). Nothing in a row's computation touches another
//! row, and the kernels compute each row independently of the rows beside it
//! in a launch, so the next token of a sequence is a pure function of the
//! sequence's own tokens, bit for bit, whatever else is in the batch. The
//! context selects how repetitive the text is: see [`ContextSpec`].
//!
//! Tokens are bytes: a prompt is its UTF-8 bytes (characters of the model's
//! own alphabet map to their byte), and a picked byte prints as a printable
//! character ([`printable`](crate::printable)).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ares_api::{EngineError, ModelCard, Prompt};
use ares_core::GgmlType;
use ares_cuda::{DeviceBuffer, DevicePtr, Kernels, MAX_BATCH, PUSH_MAX_VALUES, PinnedBuffer};
use ares_state::PageLayout;

use crate::exec::Pick;
use crate::model::{
    ContextSpec, DeviceContext, DeviceModel, LinkSpec, Logits, Model, ModelError, OwnDrafts,
    PrefixCache, Split, StepContext,
};
use crate::text::{printable, unprintable};

/// The epsilon of every norm.
const EPS: f32 = 1e-5;

/// How a synthetic model is built.
#[derive(Clone, Debug)]
pub struct SyntheticConfig {
    /// The model's name for `GenerateRequest::model`.
    pub name: String,
    /// What the model declares about its chat template (it has none: a
    /// conversation is rendered as lines of `role: text`). A test sets it to
    /// have the front end read the replies as those of a model that thinks or
    /// calls tools.
    pub chat: Option<ares_api::ChatFormat>,
    pub split: Split,
    /// How many devices the model spans; 1 unless split.
    pub devices: usize,
    /// The residual stream's width.
    pub width: u32,
    pub layers: u32,
    /// The inner width of the attention-like block.
    pub attn_inner: u32,
    /// The inner width of the feed-forward-like block.
    pub ffn_inner: u32,
    /// The quantization of the block matrices.
    pub weight_type: GgmlType,
    /// How many distinct layers of weights are kept; layer `l` uses set
    /// `l % distinct_layers`. Fewer sets than layers keep VRAM moderate while
    /// every step still reads as many weight bytes as a model of `layers`
    /// layers would (the sets together are larger than any cache).
    pub distinct_layers: u32,
    /// The token that ends a generation, if any. Token 256 is an extra token
    /// that exists for this; any byte can be made the end token instead, to
    /// end text at a place the model is known to reach.
    pub eos: Option<u32>,
    /// The output head's quantization.
    pub head_type: GgmlType,
    /// What the host derives from the tokens per row; see [`ContextSpec`].
    pub context: ContextSpec,
    /// The longest sequence, prompt plus completion.
    pub context_length: u32,
    /// The most rows of a step.
    pub max_rows: u32,
    /// For a layer split: how many layers each device holds. Empty: even.
    pub layers_per_device: Vec<u32>,
    /// For a tensor split: each device's share of the rows of every block, as
    /// relative weights (the memory bandwidths of the devices, say). Empty:
    /// equal shares. Unequal cards given equal shares wait for the slower one
    /// at every sum.
    pub tensor_shares: Vec<f64>,
    /// Bytes of state the model declares per cell on each device. The
    /// synthetic model reads and writes none (its history reaches the kernels
    /// through the context of each row), so this only makes the runtime
    /// reserve and allocate the page arrays a real model would use.
    pub state_bytes_per_cell: u64,
    /// Whether the model takes part in the prefix cache, which makes the
    /// runtime cache the pages and checkpoints of prompts and start new
    /// prompts from them. `None`: it does not (the default, as for a model whose
    /// device side does not implement the step record's copies). `Some(bytes)`:
    /// it does, and it pretends to a recurrent state of `bytes` per checkpoint
    /// on every device; `Some(0)` pretends to none. Only the host executor
    /// ([`crate::testing::HostExecutor`]) keeps such a state and carries out the
    /// copies, to check what the conductor does with them; the model's device
    /// part has none and refuses to open with this set.
    pub prefix_cache: Option<u64>,
    /// Whether the model says its device side moves pages and checkpoints to
    /// the host and back, which lets the prefix cache keep them on SSD, and
    /// the number that stands for its weights in the store's fingerprint.
    /// `None`: it does not (the default). Only the host executor carries the
    /// transfers out. Needs `prefix_cache`.
    pub durable: Option<u64>,
    /// The cells before the start of a step whose pages the model says have
    /// not reached the form they keep ([`crate::Durable::settle_cells`]).
    pub settle_cells: u32,
    /// The links of a chain of drafts the model makes itself, like the draft
    /// block of a real one: 0 for none. The context must read one token
    /// (`context.window == 1`) so that a draft's context follows from the
    /// draft alone. The host executor drafts by [`SyntheticConfig::own_accuracy`]; the
    /// device part drafts what it has seen the model pick after the same token
    /// in the same context ([`SyntheticModel`]), which is right wherever the
    /// text repeats and the sampler picks as it did.
    pub own_drafts: u32,
    /// For the host executor: the chance that link `i` of a chain makes the
    /// draft the model itself would pick, given that the links before it did;
    /// the last number serves the links after it. Empty: every link is right.
    pub own_accuracy: Vec<f64>,
    /// For the host executor: the probability its drafter reports for a draft
    /// that is right and for one that is wrong. `None`: it reports none, and
    /// makes every link it is asked for.
    pub own_probability: Option<(f32, f32)>,
    pub seed: u64,
}

impl SyntheticConfig {
    /// The shape of the real model: width 5120, 64 layers, a feed-forward
    /// block of 17408 (the matrix the weight kernels were measured on), 4-bit
    /// weights, text that does not repeat.
    pub fn realistic(split: Split, devices: usize) -> Self {
        Self {
            name: "synthetic".to_owned(),
            chat: None,
            split,
            devices: if split == Split::Single { 1 } else { devices },
            width: 5120,
            layers: 64,
            attn_inner: 5120,
            ffn_inner: 17408,
            weight_type: GgmlType::Q4_K,
            distinct_layers: 4,
            eos: None,
            head_type: GgmlType::Q8_0,
            context: ContextSpec { window: 0, epoch: 0, buckets: 4096 },
            context_length: 4096,
            max_rows: 64,
            layers_per_device: Vec::new(),
            tensor_shares: Vec::new(),
            state_bytes_per_cell: 0,
            prefix_cache: None,
            durable: None,
            settle_cells: 0,
            own_drafts: 0,
            own_accuracy: Vec::new(),
            own_probability: None,
            seed: 0x5eed,
        }
    }

    /// A model of a few megabytes, for tests: it exercises every kernel and
    /// every path between devices and runs a step in well under a millisecond.
    pub fn small(split: Split, devices: usize) -> Self {
        Self {
            width: 512,
            layers: 4,
            attn_inner: 512,
            ffn_inner: 1024,
            distinct_layers: 2,
            context: ContextSpec { window: 0, epoch: 0, buckets: 512 },
            context_length: 512,
            max_rows: 64,
            ..Self::realistic(split, devices)
        }
    }

    /// Makes the text repetitive: the context reads the last `window` tokens
    /// and the function changes every `epoch` positions (0: never).
    pub fn repetitive(mut self, window: u32, epoch: u32) -> Self {
        self.context.window = window;
        self.context.epoch = epoch;
        self
    }

    /// Gives the model a draft block of its own with `links` links, of which
    /// link `i` is right with the chance `accuracy[i]` (see
    /// [`SyntheticConfig::own_accuracy`]). Needs a context of one token.
    pub fn with_own_drafts(mut self, links: u32, accuracy: &[f64]) -> Self {
        self.own_drafts = links;
        self.own_accuracy = accuracy.to_vec();
        self.context.window = 1;
        self
    }

    /// The context of the row of `token` at `position`, for a model whose
    /// context reads one token.
    pub fn context_of(&self, token: u32, position: u32) -> u32 {
        let spec = self.context;
        spec.reduce(spec.advance(spec.start(), &[token]), position)
    }

    /// The number of tokens: the bytes, and token 256 if it is the end token.
    pub fn vocab(&self) -> u32 {
        256 + u32::from(self.eos == Some(256))
    }

    /// The layers device `index` runs: all for a single device or a tensor
    /// split, a range for a layer split.
    pub fn layer_range(&self, index: usize) -> std::ops::Range<u32> {
        if self.split != Split::Layer {
            return 0..self.layers;
        }
        let counts: Vec<u32> = if self.layers_per_device.len() == self.devices {
            self.layers_per_device.clone()
        } else {
            (0..self.devices as u32)
                .map(|d| {
                    self.layers / self.devices as u32
                        + u32::from(d < self.layers % self.devices as u32)
                })
                .collect()
        };
        let start: u32 = counts[..index].iter().sum();
        start..start + counts[index]
    }

    /// The inner width of a block of `inner` that device `index` computes: all
    /// of it, or its share of a tensor split in whole blocks of the weight
    /// type (a slice of the rows of the first matrix and of the columns of
    /// the second must be whole blocks).
    pub fn local_inner(&self, inner: u32, index: usize) -> u32 {
        if self.split != Split::Tensor {
            return inner;
        }
        let unit = self.weight_type.block_values().max(32);
        let blocks = f64::from(inner / unit);
        let equal;
        let shares: &[f64] = if self.tensor_shares.len() == self.devices {
            &self.tensor_shares
        } else {
            equal = vec![1.0; self.devices];
            &equal
        };
        let total: f64 = shares.iter().sum();
        // Cumulative rounding, so the shares always add up to the whole.
        let edge = |k: usize| ((shares[..k].iter().sum::<f64>() / total) * blocks).round() as u32;
        (edge(index + 1) - edge(index)) * unit
    }

    /// The bytes of weights, embeddings and head that device `index` holds.
    pub fn weight_bytes(&self, index: usize) -> u64 {
        let range = self.layer_range(index);
        let sets = self.distinct_layers.min(range.end - range.start).max(1);
        let block = |inner: u32| {
            2 * self
                .weight_type
                .bytes_for(u64::from(self.width) * u64::from(self.local_inner(inner, index)))
                .unwrap_or(0)
        };
        let layers = u64::from(sets) * (block(self.attn_inner) + block(self.ffn_inner));
        let embeddings = 4 * u64::from(self.width) * u64::from(self.vocab() + self.context.buckets);
        let head = self
            .head_type
            .bytes_for(u64::from(self.width) * u64::from(self.head_rows(index).len() as u32))
            .unwrap_or(0);
        layers + embeddings + head
    }

    /// The tokens whose logits device `index` computes.
    fn head_rows(&self, index: usize) -> std::ops::Range<u32> {
        let vocab = self.vocab();
        match self.split {
            Split::Single => 0..vocab,
            Split::Tensor => {
                let per = vocab.div_ceil(self.devices as u32);
                (index as u32 * per).min(vocab)..((index as u32 + 1) * per).min(vocab)
            }
            Split::Layer => {
                if index + 1 == self.devices {
                    0..vocab
                } else {
                    0..0
                }
            }
            Split::Shared => 0..0,
        }
    }

    fn validate(&self) -> Result<(), ModelError> {
        let bad = |text: String| Err(ModelError::Invalid(text));
        if self.split == Split::Shared {
            return bad("the synthetic model has no shared split".into());
        }
        if self.devices == 0 || (self.split == Split::Single && self.devices != 1) {
            return bad(format!("{:?} over {} devices", self.split, self.devices));
        }
        if self.split != Split::Single && self.devices < 2 {
            return bad(format!("a {:?} split needs at least 2 devices", self.split));
        }
        if !Kernels::has_fast_matvec(self.weight_type) || !Kernels::has_fast_matvec(self.head_type)
        {
            return bad(format!(
                "{} or {} has no decode weight kernel",
                self.weight_type, self.head_type
            ));
        }
        let block = self.weight_type.block_values();
        let unit = block.max(32);
        for (what, n) in [
            ("width", self.width),
            ("attention inner width", self.attn_inner),
            ("feed-forward inner width", self.ffn_inner),
        ] {
            if n % unit != 0 {
                return bad(format!(
                    "{what} {n} is not a whole number of {} blocks of {block} values",
                    self.weight_type
                ));
            }
        }
        if self.split == Split::Tensor {
            if !self.tensor_shares.is_empty()
                && (self.tensor_shares.len() != self.devices
                    || self.tensor_shares.iter().any(|s| !s.is_finite() || *s <= 0.0))
            {
                return bad(format!(
                    "tensor shares {:?} are not {} positive numbers",
                    self.tensor_shares, self.devices
                ));
            }
            for (what, n) in [("attention", self.attn_inner), ("feed-forward", self.ffn_inner)] {
                for d in 0..self.devices {
                    if self.local_inner(n, d) == 0 {
                        return bad(format!(
                            "the {what} inner width {n} is too small to give device {d} a block"
                        ));
                    }
                }
            }
        }
        if !self.width.is_multiple_of(self.head_type.block_values()) {
            return bad(format!(
                "width {} is not a whole number of {} blocks",
                self.width, self.head_type
            ));
        }
        if self.layers == 0 || self.distinct_layers == 0 || self.max_rows == 0 {
            return bad("layers, distinct layers and rows must be at least 1".into());
        }
        if self.split == Split::Layer
            && !self.layers_per_device.is_empty()
            && (self.layers_per_device.len() != self.devices
                || self.layers_per_device.iter().sum::<u32>() != self.layers)
        {
            return bad(format!(
                "layers per device {:?} do not add up to {} layers over {} devices",
                self.layers_per_device, self.layers, self.devices
            ));
        }
        if self.own_drafts > 0 && self.context.window != 1 {
            return bad(
                "a draft's context follows from the draft alone only if it reads one token".into(),
            );
        }
        if self.own_accuracy.iter().any(|p| !(0.0..=1.0).contains(p)) {
            return bad(format!("accuracies {:?} are not chances", self.own_accuracy));
        }
        if self.eos.is_some_and(|t| t >= self.vocab()) {
            return bad(format!(
                "end token {:?} is not a token of {} tokens",
                self.eos,
                self.vocab()
            ));
        }
        if self.context.buckets == 0 || self.context_length < 2 {
            return bad("context buckets and length must be positive".into());
        }
        Ok(())
    }
}

/// A model whose layers are random.
#[derive(Debug)]
pub struct SyntheticModel {
    config: Arc<SyntheticConfig>,
    /// What the device parts have seen the model pick after a token in a
    /// context: their draft block drafts from it. The parts share it, so that
    /// all of them draft the same tokens.
    seen: Arc<Mutex<HashMap<(u32, u32), u32>>>,
}

impl SyntheticModel {
    /// Checks the configuration; nothing touches a device until the runtime
    /// opens the device parts.
    pub fn new(config: SyntheticConfig) -> Result<Self, ModelError> {
        config.validate()?;
        Ok(Self { config: Arc::new(config), seen: Arc::default() })
    }

    pub fn config(&self) -> &SyntheticConfig {
        &self.config
    }

    /// What the kernels would pick for a row whose token is `token` and whose
    /// context is `context`, computed on the host as a stand-in: any fixed
    /// function of the two. For executors that have no device
    /// ([`crate::testing`]); the GPU model computes its own.
    pub fn host_pick(&self, token: u32, context: u32) -> (u32, f32) {
        let mut h = u64::from(token) << 32 | u64::from(context);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        ((h % u64::from(self.config.vocab())) as u32, (h >> 40) as f32)
    }

    /// The context the row of `token` at `position` has: what the conductor
    /// derives from the tokens, for a model whose context reads one token.
    pub fn context_of(&self, token: u32, position: u32) -> u32 {
        self.config.context_of(token, position)
    }

    /// What the model's draft block would draft for the window whose newest
    /// token is `token` at `position` with `context`: `links` tokens. Link `j`
    /// is the token the model itself picks after the ones before it when a
    /// hash of where it is falls under the link's accuracy, and another token
    /// when it does not. For the executors that stand in for a device.
    pub fn host_drafts(&self, token: u32, context: u32, position: u32, links: u32) -> Vec<u32> {
        self.host_chain(token, context, position, links)
            .into_iter()
            .map(|(draft, _)| draft)
            .collect()
    }

    /// [`SyntheticModel::host_drafts`] with the probability the drafter gives
    /// each draft ([`SyntheticConfig::own_probability`]; NaN when it reports
    /// none).
    pub fn host_chain(
        &self,
        token: u32,
        context: u32,
        position: u32,
        links: u32,
    ) -> Vec<(u32, f32)> {
        let c = &self.config;
        let accuracy = |j: u32| {
            c.own_accuracy.get(j as usize).or(c.own_accuracy.last()).copied().unwrap_or(1.0)
        };
        let (mut token, mut context) = (token, context);
        let mut drafts = Vec::with_capacity(links as usize);
        for j in 0..links {
            let truth = self.host_pick(token, context).0;
            let at = position + 1 + j;
            let mut h = u64::from(truth) << 32 | u64::from(at);
            h = (h ^ (h >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd) ^ c.seed;
            h = (h ^ (h >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
            let right = unit < accuracy(j);
            let p = c.own_probability.map_or(
                f32::NAN,
                |(sure, unsure)| {
                    if right { sure } else { unsure }
                },
            );
            drafts.push((if right { truth } else { (truth + 1) % c.vocab() }, p));
            (token, context) = (truth, self.context_of(truth, at));
        }
        drafts
    }

    fn render(prompt: &Prompt) -> String {
        match prompt {
            Prompt::Text(text) => text.clone(),
            Prompt::Chat { messages, continue_final, .. } => {
                // A message to continue is the start of the reply.
                let (before, last) = match messages.split_last() {
                    Some((last, before)) if *continue_final => (before, Some(last)),
                    _ => (&messages[..], None),
                };
                let mut text = String::new();
                for message in before {
                    text.push_str(message.role.as_str());
                    text.push_str(": ");
                    text.push_str(&message.content);
                    text.push('\n');
                }
                text.push_str("assistant: ");
                if let Some(last) = last {
                    text.push_str(&last.content);
                }
                text
            }
        }
    }
}

impl Model for SyntheticModel {
    fn card(&self) -> ModelCard {
        ModelCard {
            id: self.config.name.clone(),
            context_length: self.config.context_length,
            created: 0,
        }
    }

    fn vocab(&self) -> u32 {
        self.config.vocab()
    }

    fn ends(&self, token: u32) -> bool {
        self.config.eos == Some(token)
    }

    fn chat_format(&self) -> Option<ares_api::ChatFormat> {
        self.config.chat.clone()
    }

    /// One token per byte. A character of the model's own alphabet (what
    /// [`printable`] produces) is its byte, so generated text fed back as a
    /// prompt gives the tokens it came from; any other character is its UTF-8
    /// bytes.
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        let text = Self::render(prompt);
        let mut tokens = Vec::with_capacity(text.len());
        let mut utf8 = [0u8; 4];
        for ch in text.chars() {
            match unprintable(ch) {
                Some(byte) => tokens.push(u32::from(byte)),
                None => tokens.extend(ch.encode_utf8(&mut utf8).bytes().map(u32::from)),
            }
        }
        Ok(tokens)
    }

    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        if let Ok(byte) = u8::try_from(token) {
            let mut utf8 = [0u8; 4];
            text.extend_from_slice(printable(byte).encode_utf8(&mut utf8).as_bytes());
        }
    }

    fn attention_span(&self) -> u32 {
        // No kernel of this model reads a sequence's earlier cells.
        0
    }

    fn context(&self) -> ContextSpec {
        self.config.context
    }

    fn split(&self) -> Split {
        self.config.split
    }

    fn devices(&self) -> usize {
        self.config.devices
    }

    fn links(&self) -> LinkSpec {
        let c = &self.config;
        // A sum covers as many rows as the push kernel takes at once.
        let sum_rows = (PUSH_MAX_VALUES / c.width).clamp(1, c.max_rows);
        LinkSpec {
            sum_values: sum_rows * c.width,
            lane_slots: c.max_rows.div_ceil(MAX_BATCH),
            lane_values: MAX_BATCH * c.width,
        }
    }

    fn max_rows(&self) -> u32 {
        self.config.max_rows
    }

    fn logit_rows(&self) -> u32 {
        self.config.max_rows
    }

    /// The model has no state that a rejected row could spoil, so any step
    /// may verify drafts.
    fn draft_rows(&self) -> u32 {
        self.config.max_rows
    }

    fn own_drafts(&self) -> u32 {
        self.config.own_drafts
    }

    fn prefix_cache(&self) -> Option<PrefixCache> {
        self.config.prefix_cache.map(|bytes| PrefixCache {
            checkpoint_bytes: vec![bytes; self.config.devices],
            tails_per_slot: None,
        })
    }

    fn durable(&self, cells_per_page: u32) -> Option<crate::Durable> {
        let weights = self.config.durable?;
        let devices = self.config.devices;
        // What the host executor writes for a page and for a checkpoint, all
        // of it as the first device's share.
        let first = |bytes: u64| (0..devices).map(|d| if d == 0 { bytes } else { 8 }).collect();
        let window = u64::from(self.config.context.window.max(1));
        Some(crate::Durable {
            fingerprint: [&b"synthetic"[..], &weights.to_le_bytes()].concat(),
            page_bytes: first(u64::from(cells_per_page) * crate::testing::CELL_BYTES as u64),
            state_bytes: first(crate::testing::STATE_BYTES as u64 + 4 * window),
            settle_cells: self.config.settle_cells,
        })
    }

    fn page_entry(&self, page: u32) -> u32 {
        page
    }

    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        let c = &self.config;
        (0..c.devices).fold(PageLayout::new(c.devices, cells_per_page), |layout, d| {
            layout.per_cell(d, "synthetic state", c.state_bytes_per_cell)
        })
    }

    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        Ok(Box::new(SyntheticDevice::build(&self.config, &self.seen, cx)?))
    }
}

// --- the device part -----------------------------------------------------

/// A quantized matrix on a device.
#[derive(Debug)]
struct Matrix {
    buffer: DeviceBuffer,
    ty: GgmlType,
    n_in: u32,
    n_out: u32,
}

/// A block: widen, narrow.
#[derive(Debug)]
struct Block {
    up: Matrix,
    down: Matrix,
    /// The inner width this device computes.
    inner: u32,
}

#[derive(Debug)]
struct SyntheticDevice {
    config: Arc<SyntheticConfig>,
    index: usize,
    layers: std::ops::Range<u32>,
    /// `[set][0 = attention, 1 = feed-forward]`.
    blocks: Vec<[Block; 2]>,
    token_table: DeviceBuffer,
    context_table: DeviceBuffer,
    head: Option<Matrix>,
    head_first_token: u32,
    /// The residual stream, `max_rows * width`.
    x: DeviceBuffer,
    /// The normalised input of a block.
    h: DeviceBuffer,
    /// The wide intermediate.
    u: DeviceBuffer,
    /// The narrowed output of a block (a partial result under tensor split).
    p: DeviceBuffer,
    /// The quantized activations, one group of rows at a time.
    q: DeviceBuffer,
    logits: Option<DeviceBuffer>,
    /// The token and the context every row of the step reads: the record's,
    /// except for the draft rows of a window whose drafts the model makes.
    inputs: Vec<(u32, u32)>,
    /// The inputs of the rows whose logits are wanted, in the order of the
    /// record's runs, to learn the picks by.
    wanted: Vec<(u32, u32)>,
    drafter: Option<Drafter>,
}

/// The draft block of a device part: it drafts what the model picked the last
/// time it read the same token in the same context.
#[derive(Debug)]
struct Drafter {
    seen: Arc<Mutex<HashMap<(u32, u32), u32>>>,
    /// The drafts of the step on the device, for the runtime to read back.
    on_device: DeviceBuffer,
    /// Where they come from: page-locked, so that the copy does not block.
    staged: PinnedBuffer,
    /// How many drafts the step that was enqueued has.
    count: u32,
}

impl SyntheticDevice {
    fn build(
        config: &Arc<SyntheticConfig>,
        seen: &Arc<Mutex<HashMap<(u32, u32), u32>>>,
        cx: DeviceContext<'_>,
    ) -> Result<Self, ModelError> {
        let c = config;
        if c.prefix_cache.is_some() {
            return Err(ModelError::Invalid(
                "this model pretends to a prefix cache, which only the host executor can carry \
                 out: its device part keeps no state"
                    .into(),
            ));
        }
        let index = cx.index;
        let layers = c.layer_range(index);
        let sets = c.distinct_layers.min(layers.end - layers.start).max(1);
        let head_tokens = c.head_rows(index);
        let rows = u64::from(cx.max_rows);
        let width = u64::from(c.width);
        let widest_inner =
            u64::from(c.local_inner(c.attn_inner, index).max(c.local_inner(c.ffn_inner, index)));

        let q_bytes = Kernels::quantized_activation_bytes(c.width, MAX_BATCH)
            .max(Kernels::quantized_activation_bytes((widest_inner as u32).max(32), MAX_BATCH));
        let activations = 4 * rows * (2 * width + widest_inner + width)
            + q_bytes
            + 4 * rows * u64::from(head_tokens.len() as u32);
        cx.ledger.reserve("synthetic weights", c.weight_bytes(index))?;
        cx.ledger.reserve("synthetic activations", activations)?;

        let mut rng = Rng::new(c.seed ^ (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut matrix = |ty: GgmlType, n_in: u32, n_out: u32| -> Result<Matrix, ModelError> {
            let bytes = ty.bytes_for(u64::from(n_in) * u64::from(n_out)).expect("whole blocks");
            let buffer = DeviceBuffer::new(cx.device, bytes)?;
            fill_random_blocks(&buffer, ty, &mut rng)?;
            Ok(Matrix { buffer, ty, n_in, n_out })
        };
        let mut blocks = Vec::new();
        for _ in 0..sets {
            let mut pair = Vec::new();
            for inner in [c.attn_inner, c.ffn_inner] {
                let local = c.local_inner(inner, index);
                pair.push(Block {
                    up: matrix(c.weight_type, c.width, local)?,
                    down: matrix(c.weight_type, local, c.width)?,
                    inner: local,
                });
            }
            let [a, b]: [Block; 2] = pair.try_into().map_err(|_| ()).expect("two blocks");
            blocks.push([a, b]);
        }
        let head = (!head_tokens.is_empty())
            .then(|| matrix(c.head_type, c.width, head_tokens.len() as u32))
            .transpose()?;

        let table = |n_rows: u32, rng: &mut Rng| -> Result<DeviceBuffer, ModelError> {
            let values: Vec<u8> =
                (0..u64::from(n_rows) * width).flat_map(|_| rng.unit().to_le_bytes()).collect();
            let buffer = DeviceBuffer::new(cx.device, values.len() as u64)?;
            buffer.write(0, &values)?;
            Ok(buffer)
        };
        let token_table = table(c.vocab(), &mut rng)?;
        let context_table = table(c.context.buckets, &mut rng)?;

        let scratch = |bytes: u64| DeviceBuffer::new(cx.device, bytes.max(16));
        Ok(Self {
            config: config.clone(),
            index,
            layers,
            blocks,
            token_table,
            context_table,
            head_first_token: head_tokens.start,
            logits: head.as_ref().map(|h| scratch(4 * rows * u64::from(h.n_out))).transpose()?,
            head,
            x: scratch(4 * rows * width)?,
            h: scratch(4 * rows * width)?,
            u: scratch(4 * rows * widest_inner)?,
            p: scratch(4 * rows * width)?,
            q: scratch(q_bytes)?,
            inputs: Vec::new(),
            wanted: Vec::new(),
            drafter: (c.own_drafts > 0)
                .then(|| -> Result<Drafter, ModelError> {
                    let bytes = 4 * rows;
                    Ok(Drafter {
                        seen: seen.clone(),
                        on_device: DeviceBuffer::new(cx.device, bytes)?,
                        staged: PinnedBuffer::new(cx.device, bytes as usize)?,
                        count: 0,
                    })
                })
                .transpose()?,
        })
    }

    /// Reads the inputs of the rows of the step from the record, and makes
    /// the drafts of the windows that take the model's own: link `j` is what
    /// the model was seen to pick after the draft before it (the newest token
    /// for the first), or `u32::MAX` where it was not seen, and the rows after
    /// such a draft read token 0.
    fn read_inputs(&mut self, cx: &StepContext<'_>) -> Result<(), ModelError> {
        let record = &cx.record;
        self.inputs.clear();
        self.inputs
            .extend((0..record.rows()).map(|r| (record.row(r).token, record.row(r).context)));
        self.wanted.clear();
        for i in 0..record.runs() {
            let (first, count) = record.run(i);
            self.wanted.extend_from_slice(&self.inputs[first as usize..(first + count) as usize]);
        }
        let Some(drafter) = &mut self.drafter else { return Ok(()) };
        let mut drafts = Vec::new();
        {
            let seen = drafter.seen.lock().expect("what the model picked");
            for i in (0..record.segments()).filter(|i| record.own_drafts(*i)) {
                let segment = record.segment(i);
                let newest = record.row(segment.row0);
                let mut input = Some((newest.token, newest.context));
                for link in 1..segment.n_rows {
                    let draft = input.and_then(|at| seen.get(&at).copied());
                    drafts.push(draft.unwrap_or(u32::MAX));
                    input = draft.map(|d| (d, self.config.context_of(d, newest.position + link)));
                    self.inputs[(segment.row0 + link) as usize] = input.unwrap_or((0, 0));
                }
            }
        }
        // Only the first device hands the drafts to the runtime: the others
        // have made the same ones for their own rows.
        drafter.count = if self.index == 0 { drafts.len() as u32 } else { 0 };
        if drafter.count > 0 {
            for (bytes, draft) in drafter.staged.as_chunks_mut::<4>().0.iter_mut().zip(&drafts) {
                *bytes = draft.to_le_bytes();
            }
            // SAFETY: `staged` holds the drafts (at most a row each) and is
            // not written again before the step has ended; `on_device` holds
            // a u32 for every row of a step.
            unsafe {
                cx.stream.copy_from_host(
                    drafter.on_device.ptr(),
                    drafter.staged.as_ptr(),
                    drafts.len() * 4,
                )?;
            }
        }
        Ok(())
    }

    fn at(buffer: &DeviceBuffer, row: u32, stride: u32) -> DevicePtr {
        buffer.ptr().offset(u64::from(row) * u64::from(stride) * 4)
    }

    /// Enqueues the input of rows `rows`: token embedding plus context
    /// embedding, one launch per row (there is no gather kernel).
    fn embed(&self, cx: &StepContext<'_>, rows: std::ops::Range<u32>) -> Result<(), ModelError> {
        let width = self.config.width;
        for r in rows {
            let (token, context) = self.inputs[r as usize];
            let token = Self::at(&self.token_table, token, width);
            let context = Self::at(&self.context_table, context, width);
            // SAFETY: the tables hold a row per token and per context (the
            // conductor only writes tokens below the vocabulary and contexts
            // below the bucket count) and `x` a row per step row, all alive
            // as long as `self`, which outlives the step.
            unsafe {
                cx.kernels.add_to_f32(
                    cx.stream,
                    Self::at(&self.x, r, width),
                    token,
                    context,
                    width,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueues one block for rows `rows`, which sum across devices when
    /// `sum` is set.
    fn block(
        &self,
        cx: &StepContext<'_>,
        block: &Block,
        rows: std::ops::Range<u32>,
        sum: bool,
    ) -> Result<(), ModelError> {
        let width = self.config.width;
        let n = rows.end - rows.start;
        let (kernels, stream) = (cx.kernels, cx.stream);
        // SAFETY: every buffer (all launches below) is a scratch or weight
        // buffer of `self`, sized for `max_rows` rows of the width used here,
        // and `rows` is below `max_rows` (the conductor never plans more).
        // Quantized activations are consumed by the matrix kernel that
        // follows before the next group overwrites them, by stream order.
        unsafe {
            kernels.rms_norm_f32(
                stream,
                Self::at(&self.x, rows.start, width),
                DevicePtr::NULL,
                Self::at(&self.h, rows.start, width),
                width,
                n,
                EPS,
            )?;
            for g in groups(&rows) {
                kernels.quantize_activations(
                    stream,
                    Self::at(&self.h, g.start, width),
                    self.q.ptr(),
                    width,
                    g.end - g.start,
                )?;
                kernels.matvec(
                    stream,
                    block.up.ty,
                    block.up.buffer.ptr(),
                    self.q.ptr(),
                    Self::at(&self.u, g.start, block.inner),
                    block.up.n_in,
                    block.up.n_out,
                    g.end - g.start,
                )?;
            }
            kernels.rms_norm_f32(
                stream,
                Self::at(&self.u, rows.start, block.inner),
                DevicePtr::NULL,
                Self::at(&self.u, rows.start, block.inner),
                block.inner,
                n,
                EPS,
            )?;
            for g in groups(&rows) {
                kernels.quantize_activations(
                    stream,
                    Self::at(&self.u, g.start, block.inner),
                    self.q.ptr(),
                    block.inner,
                    g.end - g.start,
                )?;
                kernels.matvec(
                    stream,
                    block.down.ty,
                    block.down.buffer.ptr(),
                    self.q.ptr(),
                    Self::at(&self.p, g.start, width),
                    block.down.n_in,
                    block.down.n_out,
                    g.end - g.start,
                )?;
            }
        }
        if sum {
            // As many rows per sum as the mesh was sized for.
            let per_sum = (PUSH_MAX_VALUES / width).clamp(1, self.config.max_rows);
            let mut start = rows.start;
            while start < rows.end {
                let end = (start + per_sum).min(rows.end);
                cx.links.sum(self.index, Self::at(&self.p, start, width), (end - start) * width)?;
                start = end;
            }
        }
        // SAFETY: as above; `x` and `p` do not overlap.
        unsafe {
            kernels.add_f32(
                stream,
                Self::at(&self.x, rows.start, width),
                Self::at(&self.p, rows.start, width),
                n * width,
            )?;
        }
        Ok(())
    }

    /// Enqueues this device's layers for rows `rows`.
    fn layers(&self, cx: &StepContext<'_>, rows: std::ops::Range<u32>) -> Result<(), ModelError> {
        let sum = self.config.split == Split::Tensor;
        for layer in self.layers.clone() {
            let set =
                &self.blocks[(layer % self.config.distinct_layers) as usize % self.blocks.len()];
            for block in set {
                self.block(cx, block, rows.clone(), sum)?;
            }
        }
        Ok(())
    }

    /// Enqueues the final norm and the projection to this device's logits
    /// for the rows among `rows` whose logits are wanted. The logits of the
    /// `i`-th wanted row of the step go to row `i` of the logits buffer.
    fn head(&self, cx: &StepContext<'_>, rows: std::ops::Range<u32>) -> Result<(), ModelError> {
        let (Some(head), Some(logits)) = (&self.head, &self.logits) else { return Ok(()) };
        let width = self.config.width;
        let mut before = 0;
        for i in 0..cx.record.runs() {
            let (first, count) = cx.record.run(i);
            let wanted = first.max(rows.start)..(first + count).min(rows.end);
            if wanted.start < wanted.end {
                let compact = before + (wanted.start - first);
                // SAFETY: as in `block`; `logits` holds a row of `head.n_out`
                // values per step row, and a step has no more wanted rows
                // than rows.
                unsafe {
                    cx.kernels.rms_norm_f32(
                        cx.stream,
                        Self::at(&self.x, wanted.start, width),
                        DevicePtr::NULL,
                        Self::at(&self.h, wanted.start, width),
                        width,
                        wanted.end - wanted.start,
                        EPS,
                    )?;
                    for g in groups(&wanted) {
                        cx.kernels.quantize_activations(
                            cx.stream,
                            Self::at(&self.h, g.start, width),
                            self.q.ptr(),
                            width,
                            g.end - g.start,
                        )?;
                        cx.kernels.matvec(
                            cx.stream,
                            head.ty,
                            head.buffer.ptr(),
                            self.q.ptr(),
                            Self::at(logits, compact + (g.start - wanted.start), head.n_out),
                            head.n_in,
                            head.n_out,
                            g.end - g.start,
                        )?;
                    }
                }
            }
            before += count;
        }
        Ok(())
    }

    fn logits(&self) -> Option<Logits> {
        let (head, logits) = (self.head.as_ref()?, self.logits.as_ref()?);
        Some(Logits { ptr: logits.ptr(), stride: head.n_out, first_token: self.head_first_token })
    }
}

/// Rows `rows` in groups of at most [`MAX_BATCH`], the most one launch of a
/// weight kernel takes.
fn groups(rows: &std::ops::Range<u32>) -> impl Iterator<Item = std::ops::Range<u32>> {
    let end = rows.end;
    rows.clone().step_by(MAX_BATCH as usize).map(move |start| start..(start + MAX_BATCH).min(end))
}

impl DeviceModel for SyntheticDevice {
    fn logits_per_row(&self) -> u32 {
        self.head.as_ref().map_or(0, |h| h.n_out)
    }

    fn picked(&mut self, picks: &[Pick]) {
        if let Some(drafter) = &self.drafter {
            let mut seen = drafter.seen.lock().expect("what the model picked");
            seen.extend(self.wanted.iter().copied().zip(picks.iter().map(|pick| pick.token)));
        }
    }

    fn own_drafts(&self) -> Option<OwnDrafts> {
        let drafter = self.drafter.as_ref()?;
        Some(OwnDrafts { ptr: drafter.on_device.ptr(), tokens: drafter.count })
    }

    fn enqueue(&mut self, cx: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        let n = cx.record.rows();
        if n == 0 {
            return Ok(None);
        }
        self.read_inputs(cx)?;
        match self.config.split {
            Split::Shared => {
                return Err(ModelError::Invalid("the synthetic model has no shared split".into()));
            }
            Split::Single | Split::Tensor => {
                // Every device holds the residual stream of all rows; under a
                // tensor split they compute the same rows at the same time.
                self.embed(cx, 0..n)?;
                self.layers(cx, 0..n)?;
                self.head(cx, 0..n)?;
            }
            Split::Layer => {
                // The rows go through in groups, so that the next device can
                // start on a group while this one works on the next: the
                // devices form a pipeline inside one step.
                let first = self.index == 0;
                let last = self.index + 1 == self.config.devices;
                let width = self.config.width;
                for g in groups(&(0..n)) {
                    if first {
                        self.embed(cx, g.clone())?;
                    } else {
                        let received = cx.links.receive(self.index - 1, cx.stream)?;
                        // SAFETY: the parcel holds the group's rows and `x`
                        // has room for them; the copy is ordered after the
                        // parcel's arrival by the stream.
                        unsafe {
                            cx.stream.copy(
                                Self::at(&self.x, g.start, width),
                                received,
                                (g.end - g.start) as usize * width as usize * 4,
                            )?;
                        }
                        cx.links.release(self.index - 1, cx.stream)?;
                    }
                    self.layers(cx, g.clone())?;
                    if last {
                        self.head(cx, g)?;
                    } else {
                        cx.links.send(
                            self.index,
                            cx.stream,
                            Self::at(&self.x, g.start, width),
                            (g.end - g.start) * width,
                        )?;
                    }
                }
            }
        }
        Ok(self.logits())
    }
}

// --- random weights --------------------------------------------------------

/// A small deterministic generator, so a model is the same on every run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// The byte offsets, within a block of the type, of its f16 fields that must
/// be finite for random bytes to be a usable matrix; every other byte may be
/// anything in these types.
fn scale_offsets(ty: GgmlType) -> &'static [usize] {
    match ty {
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        GgmlType::Q3_K => &[108],
        _ => &[0],
    }
}

/// The nearest f16 to `value` as bits, for the normal range random scales
/// use (|value| between about 6e-5 and 65000).
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent), "{value} is outside the normal f16 range");
    let mantissa = (bits >> 13) & 0x3ff;
    // Round to nearest: the dropped 13 bits decide, a carry into the exponent is fine.
    let rounded = ((exponent as u32) << 10 | mantissa) + ((bits >> 12) & 1);
    sign | rounded as u16
}

/// Fills `buffer` with random blocks of `ty` whose f16 scales are finite and
/// of both signs.
fn fill_random_blocks(
    buffer: &DeviceBuffer,
    ty: GgmlType,
    rng: &mut Rng,
) -> Result<(), ModelError> {
    let block = ty.block_bytes() as usize;
    // About 8 MiB per upload, in whole blocks.
    let chunk_blocks = ((8 << 20) / block).max(1);
    let total_blocks = buffer.len() as usize / block;
    let mut bytes = vec![0u8; chunk_blocks * block];
    let mut done = 0;
    while done < total_blocks {
        let n = chunk_blocks.min(total_blocks - done);
        let chunk = &mut bytes[..n * block];
        for word in chunk.chunks_mut(8) {
            let r = rng.next().to_le_bytes();
            word.copy_from_slice(&r[..word.len()]);
        }
        for b in 0..n {
            for &offset in scale_offsets(ty) {
                let magnitude = 0.002 + 0.02 * (rng.unit() * 0.5 + 0.5);
                let scale = if rng.next().is_multiple_of(2) { -magnitude } else { magnitude };
                let at = b * block + offset;
                chunk[at..at + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            }
        }
        buffer.write((done * block) as u64, chunk)?;
        done += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ares_api::{Message, Role};

    use super::*;

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        // Values of random scales, checked against a reference conversion.
        assert_eq!(f32_to_f16_bits(0.002), 0x1819);
        assert_eq!(f32_to_f16_bits(0.022), 0x25a2);
        assert_eq!(f32_to_f16_bits(-0.0137), 0xa304);
        assert_eq!(f32_to_f16_bits(0.01), 0x211f);
    }

    #[test]
    fn a_small_model_is_valid_for_every_split() {
        for (split, devices) in [(Split::Single, 1), (Split::Tensor, 2), (Split::Layer, 2)] {
            SyntheticModel::new(SyntheticConfig::small(split, devices)).unwrap();
            SyntheticModel::new(SyntheticConfig::realistic(split, devices)).unwrap();
        }
    }

    #[test]
    fn shapes_that_the_kernels_cannot_take_are_refused_with_the_reason() {
        let mut config = SyntheticConfig::small(Split::Tensor, 2);
        config.ffn_inner = 1024 + 100;
        let error = SyntheticModel::new(config).unwrap_err().to_string();
        assert!(error.contains("feed-forward") && error.contains("1124"), "{error}");

        let mut config = SyntheticConfig::small(Split::Single, 1);
        // An activation format, which no weight kernel reads as weights.
        config.weight_type = GgmlType::Q8_K;
        assert!(
            SyntheticModel::new(config)
                .unwrap_err()
                .to_string()
                .contains("no decode weight kernel")
        );

        assert!(SyntheticModel::new(SyntheticConfig::small(Split::Layer, 1)).is_err());
        let mut config = SyntheticConfig::small(Split::Layer, 2);
        config.layers_per_device = vec![1, 1];
        assert!(SyntheticModel::new(config).is_err(), "2 layers do not make 4");
    }

    #[test]
    fn a_tensor_split_gives_each_device_a_share_in_whole_blocks_that_adds_up() {
        let mut config = SyntheticConfig::realistic(Split::Tensor, 2);
        // Equal by default: 17408 is 68 blocks of 256.
        assert_eq!((config.local_inner(17408, 0), config.local_inner(17408, 1)), (8704, 8704));
        // 58 % and 42 % (the bandwidths of the two cards), in whole blocks.
        config.tensor_shares = vec![0.583, 0.417];
        let (a, b) = (config.local_inner(17408, 0), config.local_inner(17408, 1));
        assert_eq!((a % 256, b % 256, a + b), (0, 0, 17408));
        assert!((9_900..10_300).contains(&a), "{a}");
        let (a, b) = (config.local_inner(5120, 0), config.local_inner(5120, 1));
        assert_eq!(a + b, 5120);
        assert!(config.weight_bytes(0) > config.weight_bytes(1));
        // Three devices.
        let mut config = SyntheticConfig::realistic(Split::Tensor, 3);
        config.tensor_shares = vec![5.0, 3.0, 2.0];
        let parts: Vec<u32> = (0..3).map(|d| config.local_inner(17408, d)).collect();
        assert_eq!(parts.iter().sum::<u32>(), 17408, "{parts:?}");
        assert!(parts[0] > parts[1] && parts[1] > parts[2]);
        // Other splits are not sliced.
        assert_eq!(SyntheticConfig::realistic(Split::Layer, 2).local_inner(17408, 1), 17408);
        // A bad share is refused.
        let mut bad = SyntheticConfig::small(Split::Tensor, 2);
        bad.tensor_shares = vec![1.0, 0.0];
        assert!(SyntheticModel::new(bad).is_err());
    }

    #[test]
    fn layers_are_dealt_to_devices_in_ranges() {
        let mut config = SyntheticConfig::small(Split::Layer, 2);
        config.layers = 5;
        assert_eq!((config.layer_range(0), config.layer_range(1)), (0..3, 3..5));
        config.layers_per_device = vec![1, 4];
        assert_eq!((config.layer_range(0), config.layer_range(1)), (0..1, 1..5));
        let tensor = SyntheticConfig::small(Split::Tensor, 2);
        assert_eq!(tensor.layer_range(1), 0..4, "a tensor split runs every layer on every device");
    }

    #[test]
    fn the_head_is_on_the_last_device_of_a_layer_split_and_sliced_in_a_tensor_split() {
        let layer = SyntheticConfig::small(Split::Layer, 2);
        assert_eq!((layer.head_rows(0), layer.head_rows(1)), (0..0, 0..256));
        let mut tensor = SyntheticConfig::small(Split::Tensor, 2);
        tensor.eos = Some(256);
        assert_eq!((tensor.head_rows(0), tensor.head_rows(1)), (0..129, 129..257));
        assert_eq!(SyntheticConfig::small(Split::Single, 1).head_rows(0), 0..256);
    }

    #[test]
    fn weights_of_a_realistic_model_are_moderate_and_split_in_half_under_tensor_split() {
        let single = SyntheticConfig::realistic(Split::Single, 1).weight_bytes(0);
        let tensor = SyntheticConfig::realistic(Split::Tensor, 2).weight_bytes(0);
        assert!(single > 400 << 20 && single < 1 << 30, "{single}");
        assert!(tensor < single * 6 / 10, "{tensor} vs {single}");
    }

    #[test]
    fn prompts_are_bytes_and_generated_text_feeds_back_to_the_same_tokens() {
        let model = SyntheticModel::new(SyntheticConfig::small(Split::Single, 1)).unwrap();
        assert_eq!(model.encode(&Prompt::Text("Hi!".into())).unwrap(), [72, 105, 33]);
        // A character outside the alphabet is its UTF-8 bytes.
        assert_eq!(model.encode(&Prompt::Text("\u{e9}".into())).unwrap(), [0xc3, 0xa9]);
        // A generated byte, printed and fed back, is the same token.
        let mut bytes = Vec::new();
        (0..=255u32).for_each(|token| model.piece(token, &mut bytes));
        let text = String::from_utf8(bytes).unwrap();
        let tokens = model.encode(&Prompt::Text(text)).unwrap();
        assert_eq!(tokens, (0..=255u32).collect::<Vec<_>>());
        let mut nothing = Vec::new();
        model.piece(256, &mut nothing);
        assert!(nothing.is_empty());
    }

    #[test]
    fn a_chat_is_rendered_with_its_roles() {
        let model = SyntheticModel::new(SyntheticConfig::small(Split::Single, 1)).unwrap();
        let prompt = Prompt::Chat {
            messages: vec![Message::new(Role::User, "hi")],
            tools: None,
            variables: Vec::new(),
            continue_final: false,
        };
        let tokens = model.encode(&prompt).unwrap();
        let text: String = tokens.iter().map(|t| char::from(*t as u8)).collect();
        assert_eq!(text, "user: hi\nassistant: ");
    }

    #[test]
    fn the_host_stand_in_is_a_pure_function_below_the_vocabulary() {
        let mut config = SyntheticConfig::small(Split::Single, 1);
        config.eos = Some(256);
        let model = SyntheticModel::new(config).unwrap();
        for token in 0..300 {
            for context in [0, 1, 77, 511] {
                let (pick, _) = model.host_pick(token, context);
                assert!(pick < 257);
                assert_eq!(
                    model.host_pick(token, context),
                    (pick, model.host_pick(token, context).1)
                );
            }
        }
        assert_ne!(model.host_pick(5, 1).0, model.host_pick(5, 2).0, "the context matters");
    }
}
