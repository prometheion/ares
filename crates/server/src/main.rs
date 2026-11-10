//! The `ares` server binary.
//!
//! Until a runtime exists it serves [`NoModelEngine`]: the HTTP side is
//! complete and every completion and decision request is answered with a
//! `501` that says no model architecture is implemented. A `--model` file is opened and described
//! in the log, which proves the file is readable, and then closed again;
//! loading it is the runtime's job.

mod cli;

use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;

use ares_core::units::Bytes;
use ares_gguf::Gguf;
use ares_server::{Config, NoModelEngine, serve};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    let options = match cli::parse(std::env::args_os().skip(1)) {
        Ok(cli::Command::Run(options)) => options,
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
    let mut architecture = None;
    if let Some(path) = &options.model {
        let gguf = Gguf::open(path).map_err(|e| format!("cannot open the model: {e}"))?;
        architecture = gguf.architecture().map(str::to_owned);
        tracing::info!(
            path = %path.display(),
            architecture = architecture.as_deref().unwrap_or("(none declared)"),
            size = %Bytes(gguf.file_size()),
            parts = gguf.n_parts(),
            tensors = gguf.tensors().len(),
            "opened the model file"
        );
    }

    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .await
        .map_err(|e| format!("cannot listen on {}:{}: {e}", options.host, options.port))?;
    let address =
        listener.local_addr().map_err(|e| format!("cannot read the listen address: {e}"))?;
    tracing::info!(%address, "listening");
    tracing::warn!(
        "no model architecture is implemented yet: /v1/models is empty, completions and decisions \
         answer 501"
    );

    let engine = Arc::new(NoModelEngine::new().with_architecture(architecture));
    serve(listener, engine, Config::default(), shutdown_signal())
        .await
        .map_err(|e| format!("the server failed: {e}"))?;
    tracing::info!("stopped");
    Ok(())
}

/// Completes on the first SIGINT or SIGTERM. A second one ends the process at
/// once, for an operator who does not want to wait for running requests.
async fn shutdown_signal() {
    wait_for_signal().await;
    tokio::spawn(async {
        wait_for_signal().await;
        tracing::warn!("second signal: exiting without waiting");
        std::process::exit(1);
    });
}

#[cfg(unix)]
async fn wait_for_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    // Failing to install a handler leaves the process killable only by the
    // default action; stay up and serve rather than refusing to start.
    let (Ok(mut term), Ok(mut int)) =
        (signal(SignalKind::terminate()), signal(SignalKind::interrupt()))
    else {
        tracing::warn!("cannot install signal handlers: graceful shutdown is unavailable");
        std::future::pending::<()>().await;
        return;
    };
    tokio::select! {
        _ = term.recv() => tracing::info!("SIGTERM received"),
        _ = int.recv() => tracing::info!("SIGINT received"),
    }
}

#[cfg(not(unix))]
async fn wait_for_signal() {
    if tokio::signal::ctrl_c().await.is_ok() {
        tracing::info!("Ctrl-C received");
    } else {
        std::future::pending::<()>().await;
    }
}
