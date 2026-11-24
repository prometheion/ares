//! What the suites that run the engine in this process on GPUs share: the
//! models they run on, built here once (the 0.8B on the test device, the 27B
//! over both cards), the routes of the server over an engine, and the model
//! the suites of the routes (`real_tools`, `real_structured`,
//! `real_messages`, `real_responses`) run on in place of the 0.8B when
//! `ARES_TEST_API_MODEL` names one: Qwen3.8-Flash-Next (the first part of its
//! split file) or the 27B.
//!
//! That model is loaded the way `ares --model <file>` loads it
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
//!
//! A suite that includes this module includes `common` as well.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ares_api::Engine;
use ares_cuda::{Device, Precision, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig};
use ares_server::Config;
use ares_server::load::{self, Loaded};
use ares_server::options::{self, Kind, ModelOptions};
use ares_tokenizer::Tokenizer;
use ares_tokenizer::{ChatOptions, Json};
use axum::Router;

use crate::common::{body_text, model_file, post, router_of, test_device};

/// The 0.8B (`ARES_TEST_QWEN35_0_8B`).
pub fn small() -> Option<PathBuf> {
    model_file("ARES_TEST_QWEN35_0_8B", "the Qwen3.5-0.8B model file")
}

/// The 27B in Q4_K_S with its MTP block (`ARES_TEST_QWEN38_27B`), which needs
/// both cards.
pub fn large() -> Option<PathBuf> {
    model_file("ARES_TEST_QWEN38_27B", "the 27B Q4_K_S file (two GPUs)")
}

/// The 0.8B on the test device as the suites run it: exact products, so that
/// a text does not depend on what else is in a step, 4096 tokens for 4
/// sequences, under the id `q`. `thinking` sets the template's
/// `enable_thinking`; `None` leaves it to the template.
pub fn small_model(path: &Path, thinking: Option<bool>) -> (Arc<Runtime>, Arc<Qwen35>) {
    let options = Options {
        precision: Precision::Exact,
        max_rows: 128,
        logit_rows: 16,
        window_rows: 8,
        pipeline_rows: 32,
        ..Options::default()
    };
    let gguf = Gguf::open(path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Qwen35::new(gguf, tokenizer, "q", &Spread::Single, options, 4096).unwrap();
    let config =
        RuntimeConfig { pages: 96, cells_per_page: 128, max_slots: 4, ..RuntimeConfig::default() };
    let devices = vec![Device::open(test_device(), Wait::Block).unwrap()];
    start(model, thinking, config, devices)
}

/// The 27B over both cards with its MTP block and the n-gram drafter, with
/// exact products so that a row computes the same in a window as alone, and a
/// context of 1024 tokens, under the id `q`. `thinking` as for
/// [`small_model`].
pub fn large_model(path: &Path, thinking: Option<bool>) -> (Arc<Runtime>, Arc<Qwen35>) {
    let options = Options {
        precision: Precision::Exact,
        max_rows: 128,
        logit_rows: 16,
        window_rows: 4,
        ..Options::default()
    };
    let spread = Spread::Tensor(vec![0.625, 0.375]);
    let gguf = Gguf::open(path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let model = Qwen35::new(gguf, tokenizer, "q", &spread, options, 1024).unwrap();
    let config = RuntimeConfig {
        pages: 24,
        cells_per_page: 128,
        max_slots: 4,
        window_rows: 4,
        ..RuntimeConfig::default()
    };
    let devices =
        vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
    start(model, thinking, config, devices)
}

fn start(
    model: Qwen35,
    thinking: Option<bool>,
    config: RuntimeConfig,
    devices: Vec<Device>,
) -> (Arc<Runtime>, Arc<Qwen35>) {
    let model = Arc::new(match thinking {
        Some(on) => model.with_chat_options(ChatOptions {
            variables: vec![("enable_thinking".to_owned(), Json::Bool(on))],
            ..ChatOptions::default()
        }),
        None => model,
    });
    let as_model: Arc<dyn Model> = model.clone();
    let gpu = GpuConfig::new(&config, &*as_model);
    let runtime = Arc::new(Runtime::start_gpu(as_model, devices, config, gpu).unwrap());
    (runtime, model)
}

/// The routes of the server over `engine`, as `ares --model` serves it.
pub async fn router(engine: Arc<dyn Engine>) -> Router {
    router_of(engine, ModelOptions::default(), Config::default()).await
}

/// A request with a JSON body to `path` of `app`: its status and its body.
pub async fn ask(app: &Router, path: &str, body: &serde_json::Value) -> (u16, String) {
    let response = post(app, path, &body.to_string()).await;
    let status = response.status().as_u16();
    (status, body_text(response).await)
}

/// The server a suite of the routes runs on, and what it serves.
pub struct Served {
    pub app: Router,
    pub engine: Arc<dyn Engine>,
    /// The runtime, when it was built here (not for a model loaded as the
    /// server loads it).
    pub runtime: Option<Arc<Runtime>>,
    pub path: PathBuf,
    /// Whether a request gives the same tokens every time.
    pub repeatable: bool,
    /// The model loaded as the server loads it, which this keeps loaded.
    _loaded: Option<ApiModel>,
}

impl Served {
    /// The model of `ARES_TEST_API_MODEL` if one is named, else the 0.8B;
    /// `None` (and a note) when neither is. `thinking` as for
    /// [`small_model`].
    pub async fn any(thinking: Option<bool>) -> Option<Self> {
        if let Some(path) = api_path() {
            let model = ApiModel::load(&path, thinking);
            let engine = model.engine();
            return Some(Self {
                app: router(engine.clone()).await,
                engine,
                runtime: None,
                path,
                repeatable: model.repeatable,
                _loaded: Some(model),
            });
        }
        let path = small()?;
        let (runtime, _) = small_model(&path, thinking);
        Some(Self::over(runtime, path).await)
    }

    /// The server over `runtime`, which was built here from `path`.
    pub async fn over(runtime: Arc<Runtime>, path: PathBuf) -> Self {
        let engine: Arc<dyn Engine> = runtime.clone();
        let app = router(engine.clone()).await;
        Self { app, engine, runtime: Some(runtime), path, repeatable: true, _loaded: None }
    }

    /// A request with a JSON body: its status and its body.
    pub async fn ask(&self, path: &str, body: &serde_json::Value) -> (u16, String) {
        ask(&self.app, path, body).await
    }
}

/// The model file the suites run on in place of the 0.8B, if one is named
/// (and the tests that open GPUs are enabled).
pub fn api_path() -> Option<PathBuf> {
    std::env::var_os("ARES_TEST_CUDA")?;
    std::env::var_os("ARES_TEST_API_MODEL").map(PathBuf::from)
}

/// A model loaded as the server loads it.
pub struct ApiModel {
    pub loaded: Loaded,
    pub path: PathBuf,
    /// Whether a request gives the same tokens every time.
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
