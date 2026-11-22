//! The model file the suites of the routes (`real_tools`, `real_structured`,
//! `real_messages`, `real_responses`) run on in place of the small
//! Qwen3.5-0.8B, when `ARES_TEST_API_MODEL` names one: Qwen3.8-Flash-Next (the
//! first part of its split file) or the 27B.
//!
//! The file is loaded the way `ares --model <file>` loads it
//! (`ares_server::load`), on the devices the server would take, under the id
//! `q` that the requests of the suites name, with the flags of
//! `ARES_TEST_API_FLAGS` as `ares` reads them (default: `--ram 8G`, and
//! `--precision exact` for a model that has it). The usage profile of the
//! experts stays with the tests' build (`--cache` under its temporary
//! directory) unless the flags name another directory.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_API_MODEL=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
//!     ARES_TEST_API_FLAGS="--ram 8G --drafters mtp-q2_0.gguf" \
//!     cargo test --release -p ares-server --test real_tools -- --test-threads=1 --nocapture
//! ```
//!
//! Flash-Next's replies are not bit for bit the same from run to run (the CPU
//! computes some of the experts a step misses, and which ones depends on what
//! the tiers hold; the owner accepts that, `docs/OPEN.md`). Such a model is not
//! `repeatable`: where a suite compares two replies to the same request, it
//! compares them when they are the same tokens and checks each on its own
//! otherwise.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ares_api::Engine;
use ares_gguf::Gguf;
use ares_server::load::{self, Loaded};
use ares_server::options::{self, Kind, ModelOptions};
use ares_tokenizer::Json;

/// The model file the suites run on in place of the 0.8B, if one is named
/// (and the tests that open GPUs are enabled).
pub fn path() -> Option<PathBuf> {
    std::env::var_os("ARES_TEST_CUDA")?;
    std::env::var_os("ARES_TEST_API_MODEL").map(PathBuf::from)
}

/// A model loaded as the server loads it.
pub struct ApiModel {
    pub loaded: Loaded,
    pub path: PathBuf,
    /// Whether a request gives the same tokens every time it is asked.
    pub repeatable: bool,
}

impl ApiModel {
    /// Loads the file. `thinking` sets whether the model thinks when a
    /// request does not say (`--chat-var enable_thinking=...`); `None` leaves
    /// it to the model.
    pub fn load(path: &Path, thinking: Option<bool>) -> Self {
        let architecture =
            Gguf::open(path).expect("the model file opens").architecture().map(str::to_owned);
        // Flash-Next has one kind of arithmetic and refuses `--precision exact`.
        let exact = architecture.as_deref() != Some("qwen4exp");
        let default = if exact { "--ram 8G --precision exact" } else { "--ram 8G" };
        let flags = std::env::var("ARES_TEST_API_FLAGS").unwrap_or_else(|_| default.to_owned());
        let mut model = ModelOptions::default();
        model.load.model = path.to_owned();
        model.load.name = Some("q".to_owned());
        let mut words = flags.split_whitespace();
        while let Some(word) = words.next() {
            let (flag, kind) = options::lookup(word).unwrap_or_else(|| panic!("a flag: {word}"));
            let value = match kind {
                Kind::Value => words.next().unwrap_or_else(|| panic!("{flag} takes a value")),
                Kind::Switch(true) => "true",
                Kind::Switch(false) => "false",
            };
            options::set(&mut model, flag, value).unwrap_or_else(|e| panic!("{e}"));
        }
        if let Some(on) = thinking {
            model.load.chat_variables.retain(|(name, _)| name != "enable_thinking");
            model.load.chat_variables.push(("enable_thinking".to_owned(), Json::Bool(on)));
        }
        if model.load.cache.is_none() {
            let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
            let name = format!("api-model-{}", stem.unwrap_or_default());
            let profile = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
            model.load.cache = Some(profile);
        }
        let repeatable = model.load.precision == ares_cuda::Precision::Exact;
        let loaded = load::load(&model.load).expect("the model loads");
        eprintln!(
            "RES loaded {} ({}) with {flags:?}: {}repeatable",
            path.display(),
            architecture.as_deref().unwrap_or("?"),
            if repeatable { "" } else { "not " }
        );
        Self { loaded, path: path.to_owned(), repeatable }
    }

    pub fn engine(&self) -> Arc<dyn Engine> {
        self.loaded.engine()
    }
}

/// Whether two replies to the same request must be the same: always for a
/// repeatable model, and for one that is not when they are the same tokens
/// (the token hashes the chat route reports). Says so when they are not.
pub fn must_agree(repeatable: bool, hash: &str, other: &str, what: &str) -> bool {
    if repeatable || hash == other {
        return true;
    }
    println!("RES {what}: other tokens ({hash} and {other}), each reply checked on its own");
    false
}
