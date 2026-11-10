//! The `ares` server binary.
//!
//! `ares --model <file.gguf>` binds the port, loads the model onto the
//! devices (see `ares_server::load`), serves it, and on SIGINT or SIGTERM
//! stops in order: no new connections, running requests finish or are ended,
//! the engine's threads are joined, the devices released, and only then does
//! the process exit.

mod cli;

use std::io::IsTerminal;
use std::process::ExitCode;

use ares_server::load::{self, Loaded};
use ares_server::{Config, serve_until};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    let options = match cli::parse(std::env::args_os().skip(1)) {
        Ok(cli::Command::Run(options)) => *options,
        Ok(cli::Command::Help) => {
            print!("{}", cli::USAGE);
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("error: {error}\nrun 'ares --help' for the options");
            return ExitCode::from(2);
        }
    };

    // Logs go to stderr: stdout is for output a user asks for, like --help.
    // Colors only on a terminal, so that a log file or journal stays plain text.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .init();

    match run(options).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            tracing::error!("{message}");
            ExitCode::FAILURE
        }
    }
}

async fn run(options: cli::Options) -> Result<(), String> {
    let mut signals = Signals::install();

    // The port first: a port that is taken is known at once, before seconds
    // are spent on the weights. Connections wait in the backlog until the
    // model is loaded.
    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .await
        .map_err(|e| format!("cannot listen on {}:{}: {e}", options.host, options.port))?;
    let address =
        listener.local_addr().map_err(|e| format!("cannot read the listen address: {e}"))?;

    let load_options = options.load;
    let mut loading = tokio::task::spawn_blocking(move || load::load(&load_options));
    let mut stop_after_load = false;
    let loaded = loop {
        tokio::select! {
            result = &mut loading => {
                break result
                    .map_err(|e| format!("loading the model failed: {e}"))?
                    .map_err(|e| e.to_string())?;
            }
            () = signals.next() => {
                if stop_after_load {
                    tracing::warn!("second signal: exiting without cleaning up");
                    std::process::exit(1);
                }
                tracing::warn!(
                    "stopping as soon as the model is loaded: the upload of the weights cannot \
                     be interrupted. A second signal exits at once"
                );
                stop_after_load = true;
            }
        }
    };
    if stop_after_load {
        return unload(loaded).await;
    }

    tracing::info!(%address, model = loaded.name(), "listening");
    let (drain, drained) = oneshot::channel::<()>();
    let (hurry, hurried) = oneshot::channel::<()>();
    tokio::spawn(async move {
        signals.next().await;
        let _ = drain.send(());
        signals.next().await;
        let _ = hurry.send(());
        signals.next().await;
        tracing::warn!("third signal: exiting without cleaning up");
        std::process::exit(1);
    });
    let served = serve_until(
        listener,
        loaded.engine(),
        Config::default(),
        async {
            let _ = drained.await;
        },
        async {
            // A sender that is gone means no second signal can come.
            if hurried.await.is_err() {
                std::future::pending::<()>().await;
            }
        },
    )
    .await;
    // The engine is stopped whatever became of the HTTP side.
    unload(loaded).await?;
    served.map_err(|e| format!("the server failed: {e}"))?;
    tracing::info!("stopped");
    Ok(())
}

/// Stops the engine and releases the devices, on a thread that may wait for
/// the engine's threads to end.
async fn unload(loaded: Loaded) -> Result<(), String> {
    tokio::task::spawn_blocking(move || loaded.unload())
        .await
        .map_err(|e| format!("stopping the engine failed: {e}"))
}

/// SIGINT and SIGTERM, one after the other: the first asks for an orderly
/// stop, the second for a stop that does not wait for running requests, and
/// the third ends the process.
struct Signals {
    streams: Option<(tokio::signal::unix::Signal, tokio::signal::unix::Signal)>,
}

impl Signals {
    fn install() -> Self {
        use tokio::signal::unix::{SignalKind, signal};
        // Failing to install a handler leaves the process killable only by
        // the default action; serve rather than refuse to start.
        let streams = match (signal(SignalKind::terminate()), signal(SignalKind::interrupt())) {
            (Ok(terminate), Ok(interrupt)) => Some((terminate, interrupt)),
            _ => {
                tracing::warn!("cannot install signal handlers: an orderly stop is unavailable");
                None
            }
        };
        Self { streams }
    }

    /// Completes when the next signal arrives.
    async fn next(&mut self) {
        let Some((terminate, interrupt)) = &mut self.streams else {
            return std::future::pending().await;
        };
        tokio::select! {
            _ = terminate.recv() => tracing::info!("SIGTERM received"),
            _ = interrupt.recv() => tracing::info!("SIGINT received"),
        }
    }
}
