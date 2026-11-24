//! The `ares` server: a model on the devices behind an HTTP API.
//!
//! [`load::load`] takes a model file and what the operator asked for to a
//! running engine on the devices, and says in the log what it decided and
//! what it took.
//!
//! [`router`] turns any [`Engine`] into an OpenAI-compatible server:
//! `/v1/chat/completions`, `/v1/completions` (plain and streamed), `/v1/models`,
//! `/health` and `/metrics`, llama-server's `/props`, `/tokenize` and
//! `/detokenize`, and the routes of router mode (`/models/load`,
//! `/models/unload`, `/models/sse`), into a server of Anthropic's Messages API on
//! `POST /v1/messages` and `/v1/messages/count_tokens` (the `anthropic`
//! module: a Messages request runs as the chat request that asks the same),
//! into a server of OpenAI's Responses API without state on
//! `POST /v1/responses` (the `responses` module, on the same footing), with
//! the routes that count the tokens of a request and generate nothing
//! (`/v1/responses/input_tokens`, `/v1/chat/completions/input_tokens`),
//! and into a System One server on
//! `POST /v1/systemone`. The HTTP side holds no model knowledge. It parses and
//! checks the request, hands a [`ares_api::GenerateRequest`] to the engine and
//! relays the events the engine pushes back, either as one JSON body or as
//! Server-Sent Events; or it hands a [`ares_api::DecisionRequest`] to the
//! engine and relays its one outcome as a JSON body.
//!
//! A request is served in full or refused: a field that asks for something
//! the server or the engine does not do is an error that names the field,
//! never dropped (`openai::request`). A sampling field it leaves out takes
//! the default its model was given, and its reply is held to the length the
//! model allows (the [`defaults`] module).
//!
//! A model that thinks has its reasoning returned apart from its answer, in
//! `reasoning_content`, and an assistant message of a request may carry it
//! back. What marks the reasoning is what the model's chat template declares
//! ([`ares_api::Engine::chat_format`]); the `reasoning` module sorts the text
//! by it and has the controls of llama-server for thinking.
//!
//! `/v1/systemone` has the request and response of llama.cpp's endpoint of
//! that name (the TypeSafe System One API): typed questions about a `state`,
//! answered by choosing among options. It is meant for every model ("Jev
//! mode"); the runtime does not decide yet, so it answers 501 today. The
//! `systemone` module says how the options that only a native decision model
//! has are handled. Decisions and generation requests run at the same time
//! and neither waits for the other.
//!
//! The one non-obvious piece is the hand-over between the engine's threads
//! and the async handlers, documented in the `bridge` module: what happens
//! when a client reads slower than the engine writes, how a client that
//! goes away cancels its request, and where a request's timings are taken.

mod access;
mod answer;
mod anthropic;
mod base64;
mod bridge;
mod chat_body;
pub mod defaults;
mod error;
mod generate;
mod images;
pub mod ini;
mod listen;
pub mod load;
mod metrics;
pub mod models;
mod openai;
pub mod options;
pub mod plan;
pub mod preset;
mod props;
mod qwen4exp;
mod reasoning;
mod responses;
mod routes;
mod sorter;
mod structured;
mod systemone;
mod tools;

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use ares_api::Engine;
use axum::Router;
use axum::routing::{any, get, post};
use tokio::net::TcpListener;
use tokio::sync::{Notify, watch};

pub use crate::access::Access;
use crate::bridge::Stop;
use crate::defaults::Defaults;
use crate::models::Models;
pub use crate::reasoning::ReasoningFormat;

/// Limits of the server. [`Config::default`] suits production; tests shrink
/// them to hit the limits cheaply.
#[derive(Debug, Clone)]
pub struct Config {
    /// How many events of one request may wait for its HTTP handler before
    /// the client counts as too slow and the request is cancelled. See the
    /// `bridge` module for why the engine is never made to wait instead.
    pub event_buffer: usize,
    /// The largest accepted request body. Long contexts are megabytes of JSON.
    pub max_body_bytes: usize,
    /// After a shutdown signal the server stops accepting connections and lets
    /// running requests finish for this long, then ends them.
    pub shutdown_grace: Duration,
    /// How the reasoning of a reply is returned when a request does not say
    /// (`--reasoning-format`).
    pub reasoning_format: ReasoningFormat,
    /// What a request gets for the fields it leaves out, and the most tokens
    /// of a reply (`--temperature` and the others of the `defaults` module).
    pub defaults: Defaults,
    /// Who may use the server: keys, names, origins (the `access` module).
    pub access: Access,
    /// How long a client may take to send a request (its head and its
    /// body), and how long one write of an answer may wait for it to take
    /// what was sent (`--timeout`; the `listen` module).
    pub timeout: Duration,
    /// Whether the log has the prompt and the reply of every request
    /// (`--log-text`).
    pub log_text: bool,
    /// How long a streamed reply waits for the engine's first step before
    /// its response starts without it (see `generate::Generation::first`).
    pub stream_wait: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            event_buffer: 1024,
            max_body_bytes: 64 << 20,
            shutdown_grace: Duration::from_secs(30),
            reasoning_format: ReasoningFormat::default(),
            defaults: Defaults::default(),
            access: Access::default(),
            timeout: Duration::from_secs(600),
            log_text: false,
            stream_wait: Duration::from_secs(10),
        }
    }
}

/// How long requests that were told to end get to write their last words to
/// their clients before the server returns without them.
const LAST_WORDS: Duration = Duration::from_secs(2);

/// The server's routes over `engine`, with the default [`Config`].
pub fn router(engine: Arc<dyn Engine>) -> Router {
    router_with(engine, Config::default())
}

/// The server's routes over `engine`.
///
/// Reads the engine's statistics once ([`Engine::stats`]), which may wait
/// for the engine's step.
pub fn router_with(engine: Arc<dyn Engine>, config: Config) -> Router {
    let models = Models::fixed(engine, config.reasoning_format, config.defaults.clone());
    router_over(Arc::new(models), config)
}

/// The server's routes over `models`: a request's `model` names one of them,
/// which is loaded if it is not. `config.reasoning_format` and
/// `config.defaults` are not read: each model has its own.
pub fn router_over(models: Arc<Models>, config: Config) -> Router {
    // Nobody raises this signal: a router that is served by someone else is
    // stopped by them.
    let (_, stop) = Stop::new();
    routes_over(models, config, stop)
}

fn routes_over(models: Arc<Models>, config: Config, stop: Stop) -> Router {
    let access = Arc::new(config.access.clone());
    Router::new()
        .route("/health", get(routes::health))
        .route("/metrics", get(metrics::metrics))
        .route("/v1/models", get(routes::models))
        .route("/props", get(props::props))
        .route("/tokenize", post(props::tokenize))
        .route("/detokenize", post(props::detokenize))
        .route("/models/load", post(routes::load_model))
        .route("/models/unload", post(routes::unload_model))
        .route("/models/sse", get(routes::model_events))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/systemone", post(systemone::decide))
        .route("/v1/messages", post(anthropic::messages).fallback(anthropic::method_not_allowed))
        .route(
            "/v1/messages/count_tokens",
            post(anthropic::count_tokens).fallback(anthropic::method_not_allowed),
        )
        .route("/v1/chat/completions/input_tokens", post(openai::chat_input_tokens))
        .route("/v1/responses", post(responses::create))
        .route("/v1/responses/input_tokens", post(responses::input_tokens))
        .route("/v1/responses/{id}", any(responses::stored))
        .route("/v1/responses/{id}/{what}", any(responses::stored))
        .fallback(routes::not_found)
        .method_not_allowed_fallback(routes::method_not_allowed)
        .with_state(routes::App { models, config, stop })
        .layer(axum::middleware::from_fn_with_state(access, access::guard))
}

/// Serves `engine` on `listener` until `shutdown` completes.
///
/// Shutdown is graceful: the listener closes at once and running requests get
/// [`Config::shutdown_grace`] to finish. Those that still run then are
/// cancelled in the engine and their clients told, and the function returns
/// when they are gone.
pub async fn serve(
    listener: TcpListener,
    engine: Arc<dyn Engine>,
    config: Config,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    serve_until(listener, engine, config, shutdown, std::future::pending()).await
}

/// Like [`serve`], with a second signal: when `now` completes during the
/// grace period, the wait for running requests ends at once.
pub async fn serve_until(
    listener: TcpListener,
    engine: Arc<dyn Engine>,
    config: Config,
    shutdown: impl Future<Output = ()> + Send + 'static,
    now: impl Future<Output = ()> + Send,
) -> io::Result<()> {
    let models = Models::fixed(engine, config.reasoning_format, config.defaults.clone());
    serve_models(listener, Arc::new(models), config, shutdown, now).await
}

/// Serves `models` on `listener` until `shutdown` completes, as [`serve`]
/// does for one engine; when `now` completes during the grace period, the
/// wait for running requests ends at once.
///
/// However it stops, no request is left in an engine when this returns,
/// unless a client did not take its last words within two seconds. The caller
/// can then shut the models down ([`Models::shutdown`]) and find them idle.
pub async fn serve_models(
    listener: TcpListener,
    models: Arc<Models>,
    config: Config,
    shutdown: impl Future<Output = ()> + Send + 'static,
    now: impl Future<Output = ()> + Send,
) -> io::Result<()> {
    let grace = config.shutdown_grace;
    let timeout = config.timeout;
    let draining = Arc::new(Notify::new());
    let begin_drain = draining.clone();
    let (stop_requests, stop) = Stop::new();
    let serving =
        listen::listen(listener, routes_over(models, config, stop), timeout, async move {
            shutdown.await;
            tracing::info!("shutting down: no new connections, waiting for running requests");
            begin_drain.notify_one();
        });
    tokio::pin!(serving);
    let patience = async move {
        draining.notified().await;
        tokio::select! {
            () = tokio::time::sleep(grace) => {
                tracing::warn!(?grace, "running requests did not finish in time: ending them");
            }
            () = now => tracing::warn!("not waiting for running requests: ending them"),
        }
    };
    tokio::select! {
        result = &mut serving => return result,
        () = patience => {}
    }
    end_requests(&stop_requests);
    match tokio::time::timeout(LAST_WORDS, serving).await {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!("a client did not take the end of its response: leaving it behind");
            Ok(())
        }
    }
}

/// Raises the signal on which every running request is cancelled in the
/// engine and ended with an error for its client.
fn end_requests(stop: &watch::Sender<bool>) {
    // Nobody listens when no request runs, and that is fine.
    let _ = stop.send(true);
}
