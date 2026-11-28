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
mod restart;

use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;

use ares_server::load::DeviceLoader;
use ares_server::models::{AcquireError, Models, SystemClock};
use ares_server::preset::Presets;
use ares_server::{Config, serve_models};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    // Only the variables of Ares' own options are read (`ARES_CONTEXT`), never
    // a name without the prefix.
    let environment = |name: &str| std::env::var_os(name);
    let options = match cli::parse_with(std::env::args_os().skip(1), &environment) {
        Ok(cli::Command::Run(options)) => *options,
        Ok(cli::Command::Help) => {
            print!("{}", cli::usage());
            return ExitCode::SUCCESS;
        }
        Ok(cli::Command::Version) => {
            println!("{}", ares_server::BUILD_INFO);
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
    let cli::Options { host, port, timeout, log, access, presets, vram, model, given } = options;
    let (presets, single) = match &presets {
        Some(file) => match Presets::read(file, &given) {
            Ok(presets) => (presets, None),
            Err(error) => {
                eprintln!("error: {error}\nrun 'ares --help' for the options");
                return ExitCode::from(2);
            }
        },
        None => {
            let presets = Presets::single(model);
            let id = presets.models[0].id.clone();
            (presets, Some(id))
        }
    };

    // The VRAM cap is the server's, for the devices of every model: one that
    // cannot be is refused before anything is started.
    let used: Option<Vec<u32>> = presets.models.iter().try_fold(Vec::new(), |mut all, preset| {
        all.extend(preset.options.load.devices.as_ref()?);
        Some(all)
    });
    if let Err(error) = ares_server::load::set_vram_cap(vram, used.as_deref()) {
        eprintln!("error: {error}\nrun 'ares --help' for the options");
        return ExitCode::from(2);
    }
    if let Err(error) = start_log(&log) {
        eprintln!("error: {error}");
        return ExitCode::from(2);
    }
    // The first line of the log names the binary.
    tracing::info!("{}", ares_server::BUILD_INFO);
    log_panics();

    let code = match run(&host, port, timeout, log.text, &access, presets, single).await {
        Ok(()) => 0,
        Err(message) => {
            tracing::error!("{message}");
            1
        }
    };
    // The engine has stopped in order, or never started, and the log is
    // written (its writers do not buffer). What may still run are threads
    // that outlive an engine: the watch of the machine's memory, which can
    // hold a model's tier of experts for a moment and free it there, and a
    // device thread that did not stop in time. The handlers for the end of
    // the process would tear the CUDA driver down under them (`exit_now`),
    // and the async runtime would wait for its blocking threads.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    exit_now(code)
}

/// Writes a panic into the log, with its thread, its place and the stack, before
/// the default hook prints it and the release build aborts: a log file
/// (`--log-file`) or a journal then says why a core dump was made.
fn log_panics() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let place = info.location().map(|at| format!("{}:{}", at.file(), at.line()));
        tracing::error!(
            thread = thread.name().unwrap_or("unnamed"),
            place = place.as_deref().unwrap_or("unknown"),
            stack = %std::backtrace::Backtrace::force_capture(),
            "panicked: {}",
            info.payload_as_str().unwrap_or("without a message"),
        );
        default(info);
    }));
}

/// Where the log goes: standard error (stdout is for output a user asks for,
/// like --help), or the file of `--log-file`; as text, with colors only on a
/// terminal so that a log file or journal stays plain, or as JSON.
fn start_log(log: &cli::LogOptions) -> Result<(), String> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let file = match &log.file {
        Some(path) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("cannot open the log file '{}': {e}", path.display()))?,
        ),
        None => None,
    };
    match (file, log.json) {
        (Some(file), true) => builder.json().with_writer(std::sync::Mutex::new(file)).init(),
        (Some(file), false) => {
            builder.with_ansi(false).with_writer(std::sync::Mutex::new(file)).init();
        }
        (None, true) => builder.json().with_writer(std::io::stderr).init(),
        (None, false) => {
            builder.with_writer(std::io::stderr).with_ansi(std::io::stderr().is_terminal()).init()
        }
    }
    Ok(())
}

async fn run(
    host: &str,
    port: u16,
    timeout: std::time::Duration,
    log_text: bool,
    access: &cli::AccessOptions,
    presets: Presets,
    single: Option<String>,
) -> Result<(), String> {
    let mut signals = Signals::install();

    // The port first: a port that is taken is known at once, before seconds
    // are spent on the weights. Connections wait in the backlog until the
    // model is loaded. An image that restarted in place takes the socket of
    // the image before over.
    let listener = match restart::inherited() {
        Some(listener) => TcpListener::from_std(listener)
            .map_err(|e| format!("cannot serve the inherited listening socket: {e}"))?,
        None => TcpListener::bind((host, port))
            .await
            .map_err(|e| format!("cannot listen on {host}:{port}: {e}"))?,
    };
    // A second descriptor of it, which a restart in place hands on.
    let kept = restart::keep(&listener)?;
    let address =
        listener.local_addr().map_err(|e| format!("cannot read the listen address: {e}"))?;
    let access = access.access(address.ip().is_loopback())?;
    if access.keys.is_empty() && !address.ip().is_loopback() {
        tracing::warn!(
            %address,
            "the server listens beyond this machine and wants no key: anyone who reaches the \
             port can use it (--api-key, --api-key-file)"
        );
    }
    let config = Config { access, timeout, log_text, ..Config::default() };

    let offered = presets.models.len();
    // What a request to each model gets for the fields it leaves out, once.
    for preset in &presets.models {
        tracing::info!(model = preset.id.as_str(), "{}", preset.options.defaults);
    }
    let models =
        Arc::new(Models::start(presets, Arc::new(DeviceLoader), Arc::new(SystemClock::default())));

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
                            // A first step that did not come back: a new image
                            // would load again what may stick again, so the
                            // supervisor decides.
                            Err(AcquireError::Restarting { reason }) => {
                                tracing::error!(
                                    "{reason}: before the server served anything; exiting with \
                                     {} for the supervisor to start it again",
                                    restart::EXIT_RESTART
                                );
                                exit_now(restart::EXIT_RESTART);
                            }
                            Err(other) => return Err(format!("loading the model failed: {other:?}")),
                        }
                    }
                    () = signals.next() => {
                        if stop_after_load {
                            tracing::warn!("second signal: exiting without cleaning up");
                            exit_now(1);
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

    // How far the stop has come: 1, no new connections and running requests
    // finish; 2, they are ended. The signals step it up one at a time, and a
    // restart in place takes it to 2 at once: every request is answered that
    // the server restarts.
    let (stage, staged) = watch::channel(0u8);
    let raise = move |to: u8| stage.send_modify(|now| *now = (*now).max(to));
    let raised = raise.clone();
    tokio::spawn(async move {
        signals.next().await;
        raised(1);
        signals.next().await;
        raised(2);
        signals.next().await;
        tracing::warn!("third signal: exiting without cleaning up");
        exit_now(1);
    });
    let asked = models.clone();
    tokio::spawn(async move {
        asked.restart_asked().await;
        raise(2);
    });
    let at = |stage: u8| {
        let mut staged = staged.clone();
        async move {
            if staged.wait_for(|now| *now >= stage).await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    };
    let served = serve_models(listener, models.clone(), config, at(1), at(2)).await;
    if let Some(reason) = models.restarting() {
        // The other models are unloaded in order (their prompt caches are
        // written) while the driver answers, for a while at most; the stuck
        // one is left to the new image.
        if tokio::time::timeout(RESTART_UNLOADS, models.shutdown()).await.is_err() {
            tracing::warn!("the other models did not unload in time: restarting without them");
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        restart::restart_in_place(kept, &reason);
    }
    // Every model is unloaded and the devices are released, whatever became
    // of the HTTP side.
    models.shutdown().await;
    served.map_err(|e| format!("the server failed: {e}"))?;
    tracing::info!("stopped");
    Ok(())
}

/// How long a restart in place waits for the models that are not stuck to
/// unload in order.
const RESTART_UNLOADS: std::time::Duration = std::time::Duration::from_secs(60);

/// Ends the process at once: as a signal asked, while the engine's threads may
/// still be inside the CUDA driver (a load uploads its weights, a step runs),
/// and after an orderly stop, beside threads that outlived the engine.
/// `std::process::exit` would run the driver's handlers for the end of the
/// process (`atexit`), which tear its state down under those threads, and the
/// driver may abort then: a SIGABRT and a core dump of all the RAM the process
/// holds.
/// `_exit` hands the process to the kernel, which frees everything.
fn exit_now(code: i32) -> ! {
    // SAFETY: `_exit` ends the process without running anything of it; no
    // state of this program is touched.
    unsafe { libc::_exit(code) }
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
