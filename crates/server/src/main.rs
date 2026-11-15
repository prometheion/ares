//! The `ares` server binary.
//!
//! `ares --model <file.gguf>` binds the port, loads the model onto the
//! devices (see `ares_server::load`), serves it, and on SIGINT or SIGTERM
//! stops in order: no new connections, running requests finish or are ended,
//! the engine's threads are joined, the devices released, and only then does
//! the process exit. `ares --models-preset <file.ini>` serves the models of a
//! preset file the same way, each loaded when a request names it (see
//! `ares_server::models`).

mod cli;

use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;

use ares_server::load::DeviceLoader;
use ares_server::models::{AcquireError, Models, SystemClock};
use ares_server::options::ModelOptions;
use ares_server::preset::Presets;
use ares_server::{Config, plan, serve_models};
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
    // One model on the command line is the one preset of a server without a
    // preset file. A file that cannot be served is refused here, with its
    // line, before anything is started.
    let cli::Options { host, port, presets, load, reasoning_format, defaults, given } = options;
    let (presets, single) = match &presets {
        Some(file) => match Presets::read(file, &given) {
            Ok(presets) => (presets, None),
            Err(error) => {
                eprintln!("error: {error}\nrun 'ares --help' for the options");
                return ExitCode::from(2);
            }
        },
        None => {
            let presets = Presets::single(ModelOptions { load, reasoning_format, defaults });
            let id = presets.models[0].id.clone();
            (presets, Some(id))
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

    match run(&host, port, presets, single).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            tracing::error!("{message}");
            ExitCode::FAILURE
        }
    }
}

async fn run(
    host: &str,
    port: u16,
    presets: Presets,
    single: Option<String>,
) -> Result<(), String> {
    let mut signals = Signals::install();

    // The port first: a port that is taken is known at once, before seconds
    // are spent on the weights. Connections wait in the backlog until the
    // model is loaded.
    let listener = TcpListener::bind((host, port))
        .await
        .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?;
    let address =
        listener.local_addr().map_err(|e| format!("cannot read the listen address: {e}"))?;

    // The most RAM of all loaded models together, when nobody names it: all
    // the machine has. The engines hold what is available below it.
    let total = ares_runtime::Meminfo::total();
    let offered = presets.models.len();
    // What a request to each model gets for the fields it leaves out, once.
    for preset in &presets.models {
        tracing::info!(model = preset.id.as_str(), "{}", preset.options.defaults);
    }
    let models = Arc::new(Models::start(
        presets,
        Arc::new(DeviceLoader),
        Arc::new(SystemClock::default()),
        plan::default_ram(total),
    ));

    match &single {
        // One model on the command line is loaded before the server listens,
        // and a server that cannot load it does not start.
        Some(id) => {
            let mut stop_after_load = false;
            let loading = models.wait_loaded(id);
            tokio::pin!(loading);
            loop {
                tokio::select! {
                    result = &mut loading => {
                        match result {
                            Ok(()) => break,
                            Err(AcquireError::Failed { reason, .. }) => return Err(reason),
                            Err(other) => return Err(format!("loading the model failed: {other:?}")),
                        }
                    }
                    () = signals.next() => {
                        if stop_after_load {
                            tracing::warn!("second signal: exiting without cleaning up");
                            std::process::exit(1);
                        }
                        tracing::warn!(
                            "stopping as soon as the model is loaded: the upload of the weights \
                             cannot be interrupted. A second signal exits at once"
                        );
                        stop_after_load = true;
                    }
                }
            }
            if stop_after_load {
                models.shutdown().await;
                return Ok(());
            }
            tracing::info!(%address, model = id.as_str(), "listening");
        }
        // The models of a preset file are loaded when a request names them,
        // or from now on in the background if their sections say so.
        None => tracing::info!(%address, models = offered, "listening"),
    }

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
    let served = serve_models(
        listener,
        models.clone(),
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
    // Every model is unloaded and the devices are released, whatever became
    // of the HTTP side.
    models.shutdown().await;
    served.map_err(|e| format!("the server failed: {e}"))?;
    tracing::info!("stopped");
    Ok(())
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
