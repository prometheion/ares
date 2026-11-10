//! HTTP front end of the engine.
//!
//! [`router`] turns any [`Engine`] into an OpenAI-compatible server:
//! `/v1/chat/completions`, `/v1/completions` (plain and streamed), `/v1/models`
//! and `/health`, and into a System One server on `POST /v1/systemone`. The
//! server holds no model knowledge. It parses and checks the request, hands a
//! [`ares_api::GenerateRequest`] to the engine and relays the events the
//! engine pushes back, either as one JSON body or as Server-Sent Events; or it
//! hands a [`ares_api::DecisionRequest`] to the engine and relays its one
//! outcome as a JSON body.
//!
//! `/v1/systemone` has the request and response of llama.cpp's endpoint of
//! that name (the TypeSafe System One API): typed questions about a `state`,
//! answered by choosing among options. It is served for every model ("Jev
//! mode"), and the `systemone` module says how the options that only a native
//! decision model has are handled. Decisions and generation requests run at
//! the same time and neither waits for the other.
//!
//! The one non-obvious piece is the hand-over between the engine's threads
//! and the async handlers, documented in the `bridge` module: what happens
//! when a client reads slower than the engine writes, and how a client that
//! goes away cancels its request.

mod bridge;
mod error;
mod no_model;
mod routes;
mod systemone;
mod wire;

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use ares_api::Engine;
use axum::Router;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tokio::sync::Notify;

pub use no_model::NoModelEngine;

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
    /// running requests finish for this long, then gives up on them.
    pub shutdown_grace: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            event_buffer: 1024,
            max_body_bytes: 64 << 20,
            shutdown_grace: Duration::from_secs(30),
        }
    }
}

/// The server's routes over `engine`, with the default [`Config`].
pub fn router(engine: Arc<dyn Engine>) -> Router {
    router_with(engine, Config::default())
}

/// The server's routes over `engine`.
pub fn router_with(engine: Arc<dyn Engine>, config: Config) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/systemone", post(routes::systemone))
        .fallback(routes::not_found)
        .method_not_allowed_fallback(routes::method_not_allowed)
        .with_state(routes::App { engine, config })
}

/// Serves `engine` on `listener` until `shutdown` completes.
///
/// Shutdown is graceful: the listener closes at once, running requests get
/// [`Config::shutdown_grace`] to finish, and then the function returns
/// regardless, so that one stuck stream cannot keep the process alive for
/// ever. Requests still running at that point are cancelled in the engine when
/// their tasks are dropped with the runtime.
pub async fn serve(
    listener: TcpListener,
    engine: Arc<dyn Engine>,
    config: Config,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    let grace = config.shutdown_grace;
    let draining = Arc::new(Notify::new());
    let begin_drain = draining.clone();
    let serving = async move {
        axum::serve(listener, router_with(engine, config))
            .with_graceful_shutdown(async move {
                shutdown.await;
                tracing::info!("shutting down: no new connections, waiting for running requests");
                begin_drain.notify_one();
            })
            .await
    };
    let deadline = async move {
        draining.notified().await;
        tokio::time::sleep(grace).await;
    };
    tokio::select! {
        result = serving => result,
        () = deadline => {
            tracing::warn!(?grace, "running requests did not finish in time, giving up on them");
            Ok(())
        }
    }
}
