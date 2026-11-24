//! A model in front of the engine that is another model with some of its
//! answers changed. Every method of `ares_runtime::Model` goes to the inner
//! model unless the test replaced it: a method a wrapper did not forward would
//! take the trait's default without a word and make the engine behave
//! otherwise than in production (for example price the steps of Flash-Next
//! with another estimate).

use std::path::PathBuf;
use std::sync::Arc;

use ares_api::{ChatFormat, EngineError, ExpertStats, ModelCard, Prompt};
use ares_gguf::{Gguf, GgufBuilder};
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, Drafter, Durable, EncodedPrompt, LinkSpec, Model,
    ModelError, PrefixCache, RecordView, Split, Vision,
};
use ares_sampler::SampleParams;
use ares_state::PageLayout;
use ares_tokenizer::Tokenizer;

/// How a test encodes a prompt in place of the model.
type Encode = Box<dyn Fn(&Prompt) -> Result<EncodedPrompt, EngineError> + Send + Sync>;

/// How a test opens a device part in place of the model.
type Open =
    Box<dyn Fn(DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> + Send + Sync>;

/// `inner` with what a test replaced.
pub struct Wrapped {
    inner: Arc<dyn Model>,
    /// The made-up words in place of the model's text ([`words`]).
    words: bool,
    /// Whether the model keeps its prefix cache, and what of it lasts.
    cache: bool,
    encode: Option<Encode>,
    vision: Option<Vision>,
    open: Option<Open>,
}

impl Wrapped {
    /// `inner` as it is.
    pub fn new(inner: Arc<dyn Model>) -> Self {
        Self { inner, words: false, cache: true, encode: None, vision: None, open: None }
    }

    /// With the made-up words in place of the model's text: a prompt is
    /// `t5 t9 ...`, a token prints as `t5 `, no token ends a reply and there is
    /// no chat template.
    pub fn words(self) -> Self {
        Self { words: true, ..self }
    }

    /// With its prefix cache (`true`, as the model has it) or without (the
    /// engine computes every prompt from its first token).
    pub fn cache(self, cache: bool) -> Self {
        Self { cache, ..self }
    }

    /// With prompts encoded by `encode`.
    pub fn encode(
        self,
        encode: impl Fn(&Prompt) -> Result<EncodedPrompt, EngineError> + Send + Sync + 'static,
    ) -> Self {
        Self { encode: Some(Box::new(encode)), ..self }
    }

    /// With images read through `vision`.
    pub fn vision(self, vision: Vision) -> Self {
        Self { vision: Some(vision), ..self }
    }

    /// With device parts opened by `open`.
    pub fn open(
        self,
        open: impl Fn(DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self { open: Some(Box::new(open)), ..self }
    }
}

/// The made-up words of `prompt`: `t5 t9` is the tokens 5 and 9.
pub fn words(prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
    let Prompt::Text(text) = prompt else {
        return Err(EngineError::Unsupported("the made-up model has no chat template".into()));
    };
    text.split_whitespace()
        .map(|word| {
            word.strip_prefix('t')
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| EngineError::InvalidRequest(format!("{word:?} is no token")))
        })
        .collect()
}

impl Model for Wrapped {
    fn card(&self) -> ModelCard {
        self.inner.card()
    }
    fn vocab(&self) -> u32 {
        self.inner.vocab()
    }
    fn ends(&self, token: u32) -> bool {
        !self.words && self.inner.ends(token)
    }
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        if self.encode.is_some() || self.words {
            Ok(self.encode_with_boundaries(prompt)?.tokens)
        } else {
            self.inner.encode(prompt)
        }
    }
    fn encode_with_boundaries(&self, prompt: &Prompt) -> Result<EncodedPrompt, EngineError> {
        match &self.encode {
            Some(encode) => encode(prompt),
            None if self.words => {
                Ok(EncodedPrompt { tokens: words(prompt)?, ..EncodedPrompt::default() })
            }
            None => self.inner.encode_with_boundaries(prompt),
        }
    }
    fn vision(&self) -> Option<Vision> {
        self.vision.or_else(|| self.inner.vision())
    }
    fn chat_format(&self) -> Option<ChatFormat> {
        if self.words { None } else { self.inner.chat_format() }
    }
    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        if self.words {
            text.extend_from_slice(format!("t{token} ").as_bytes());
        } else {
            self.inner.piece(token, text);
        }
    }
    fn context(&self) -> ContextSpec {
        self.inner.context()
    }
    fn split(&self) -> Split {
        self.inner.split()
    }
    fn devices(&self) -> usize {
        self.inner.devices()
    }
    fn links(&self) -> LinkSpec {
        self.inner.links()
    }
    fn expert_stats(&self) -> Option<ExpertStats> {
        self.inner.expert_stats()
    }
    fn saves_inside_steps(&self) -> u32 {
        self.inner.saves_inside_steps()
    }
    fn steps_wait_for_fetches(&self) -> bool {
        self.inner.steps_wait_for_fetches()
    }
    fn pinned_ram(&self) -> u64 {
        self.inner.pinned_ram()
    }
    fn max_rows(&self) -> u32 {
        self.inner.max_rows()
    }
    fn own_rows(&self) -> u32 {
        self.inner.own_rows()
    }
    fn prompt_rows(&self) -> u32 {
        self.inner.prompt_rows()
    }
    fn logit_rows(&self) -> u32 {
        self.inner.logit_rows()
    }
    fn draft_rows(&self) -> u32 {
        self.inner.draft_rows()
    }
    fn sampling(&self) -> SampleParams {
        self.inner.sampling()
    }
    fn attention_span(&self) -> u32 {
        self.inner.attention_span()
    }
    fn drafters(&self) -> Vec<Drafter> {
        self.inner.drafters()
    }
    fn prefix_cache(&self) -> Option<PrefixCache> {
        self.inner.prefix_cache().filter(|_| self.cache)
    }
    fn durable(&self, cells_per_page: u32) -> Option<Durable> {
        self.inner.durable(cells_per_page).filter(|_| self.cache)
    }
    fn page_entry(&self, page: u32) -> u32 {
        self.inner.page_entry(page)
    }
    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        self.inner.page_layout(cells_per_page)
    }
    fn abandon(&self, part: usize) {
        self.inner.abandon(part);
    }
    fn before_step(&self, record: &RecordView<'_>) -> Result<(), ModelError> {
        self.inner.before_step(record)
    }
    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        match &self.open {
            Some(open) => open(cx),
            None => self.inner.open(cx),
        }
    }
}

/// A tokenizer of the 256 bytes alone, which a made-up model needs to exist
/// and its tests do not use: their prompts are token ids. It has the chat
/// template `template`, if any.
pub fn byte_tokenizer(template: Option<&str>) -> Tokenizer {
    // The characters of GPT-2's byte-level alphabet.
    let tokens: Vec<String> = (0..=255u8)
        .map(|byte| {
            let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
            let moved =
                (0..byte).filter(|b| !matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)).count();
            let at = if printable { u32::from(byte) } else { 0x100 + moved as u32 };
            char::from_u32(at).unwrap().to_string()
        })
        .collect();
    let mut file = GgufBuilder::new()
        .str("general.architecture", "qwen4exp")
        .str("tokenizer.ggml.model", "gpt2")
        .str("tokenizer.ggml.pre", "qwen35")
        .str_array("tokenizer.ggml.tokens", &tokens)
        .i32_array("tokenizer.ggml.token_type", &vec![1; 256])
        .str_array::<&str>("tokenizer.ggml.merges", &[]);
    if let Some(template) = template {
        file = file.str("tokenizer.chat_template", template);
    }
    // A file per test binary and template, so that binaries running at once
    // do not write each other's (in the system's temporary directory when this
    // module is built into an example, which has no target directory of its own).
    let name = match template {
        Some(_) => format!("{}-tokenizer-chat.gguf", env!("CARGO_CRATE_NAME")),
        None => format!("{}-tokenizer.gguf", env!("CARGO_CRATE_NAME")),
    };
    let dir = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let path = dir.join(name);
    std::fs::write(&path, file.build()).unwrap();
    Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).expect("a tokenizer of bytes")
}
