//! The front of the runtime: an [`Engine`] over a conductor thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::Duration;

use ares_api::{
    DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, EventSink, GenerateRequest,
    ModelCard, RequestId,
};
use ares_cuda::Device;

use crate::conductor::{Command, Conductor, SequenceInfo, StatsCopy, Submission};
use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::exec::Executor;
use crate::gpu::{GpuConfig, GpuExecutor};
use crate::model::Model;
use crate::timing::Snapshot;

/// How long a question to the conductor (statistics, inspection) may take.
const ASK_TIMEOUT: Duration = Duration::from_secs(30);

/// The inference runtime: a conductor thread and the executor it drives,
/// behind the [`Engine`] trait.
///
/// [`Engine::submit`] and [`Engine::cancel`] never block. They check the
/// request on the calling thread and put it in the conductor's channel. The
/// conductor puts it in line and starts it when the pool has room for it:
/// nothing is reserved for the longest a request may get, and nothing is
/// refused because the engine is busy. Dropping the runtime shuts it down.
pub struct Runtime {
    model: Arc<dyn Model>,
    card: ModelCard,
    /// The longest prompt a request may have, in tokens, plus one: what the
    /// model's context and the whole page pool hold.
    limit: u32,
    /// `None` after shutdown. Sending happens under the lock, so a request
    /// is either in the channel before the shutdown command or refused.
    commands: Mutex<Option<Sender<Command>>>,
    conductor: Mutex<Option<JoinHandle<()>>>,
    conductor_id: ThreadId,
    /// Cleared when the conductor thread has ended, and with it the device
    /// threads the conductor's executor owns.
    running: Arc<AtomicBool>,
}

/// Clears the flag when the conductor thread is done, however it ends.
struct Running(Arc<AtomicBool>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").field("model", &self.card.id).finish_non_exhaustive()
    }
}

impl Runtime {
    /// Starts a runtime over `executor`, which must have been built for
    /// `model`. Returns when the conductor is ready for requests.
    pub fn start(
        model: Arc<dyn Model>,
        executor: Box<dyn Executor>,
        config: RuntimeConfig,
    ) -> Result<Self, RuntimeError> {
        if config.pages == 0 || config.cells_per_page == 0 || config.max_slots == 0 {
            return Err(RuntimeError::Config(
                "pages, cells per page and slots must be at least 1".into(),
            ));
        }
        if let Some(cache) = model.prefix_cache()
            && cache.checkpoint_bytes.len() != model.devices()
        {
            return Err(RuntimeError::Config(format!(
                "the model names checkpoint bytes for {} devices and has {}",
                cache.checkpoint_bytes.len(),
                model.devices()
            )));
        }
        let card = model.card();
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let limit = card.context_length.min(config.pages.saturating_mul(config.cells_per_page));
        let warm_up = config.warm_up;
        let mut conductor = Conductor::new(model.clone(), config, executor, rx);
        let running = Arc::new(AtomicBool::new(true));
        let alive = running.clone();
        let thread = thread::Builder::new()
            .name("ares-conductor".to_owned())
            .spawn(move || {
                // Declared first, so it is dropped last: after the conductor
                // and the executor it owns, whose device threads it joins.
                let _running = Running(alive);
                let ready = if warm_up { conductor.warm_up() } else { Ok(()) };
                let failed = ready.is_err();
                let _ = ready_tx.send(ready);
                if !failed {
                    conductor.run();
                }
            })
            .map_err(|e| RuntimeError::Thread(format!("cannot start the conductor: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let _ = thread.join();
                return Err(RuntimeError::Device { device: 0, message: error.to_string() });
            }
            Err(_) => {
                let _ = thread.join();
                return Err(RuntimeError::Thread("the conductor ended while starting".into()));
            }
        }
        Ok(Self {
            model,
            card,
            limit,
            commands: Mutex::new(Some(tx)),
            conductor_id: thread.thread().id(),
            conductor: Mutex::new(Some(thread)),
            running,
        })
    }

    /// Starts a runtime with a thread per device over `devices`.
    pub fn start_gpu(
        model: Arc<dyn Model>,
        devices: Vec<Device>,
        config: RuntimeConfig,
        gpu: GpuConfig,
    ) -> Result<Self, RuntimeError> {
        let executor = GpuExecutor::start(model.clone(), devices, gpu)?;
        Self::start(model, Box::new(executor), config)
    }

    /// Whether the conductor and the device threads are still alive. False
    /// once [`Runtime::shutdown`] has returned.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// A copy of the timing measurements so far, or `None` when the conductor
    /// is gone. The conductor only copies its samples (a fraction of a
    /// millisecond); the percentiles are computed on the caller's thread.
    pub fn snapshot(&self) -> Option<Snapshot> {
        self.ask(|reply| Command::Stats { reset: false, reply }).map(StatsCopy::into_snapshot)
    }

    /// Like [`Runtime::snapshot`], and starts the measurements afresh.
    pub fn take_snapshot(&self) -> Option<Snapshot> {
        self.ask(|reply| Command::Stats { reset: true, reply }).map(StatsCopy::into_snapshot)
    }

    /// The running sequences and their page tables, for tests and logs.
    pub fn sequences(&self) -> Option<Vec<SequenceInfo>> {
        self.ask(|reply| Command::Inspect { reply })
    }

    fn ask<T>(&self, command: impl FnOnce(mpsc::SyncSender<T>) -> Command) -> Option<T> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.commands.lock().expect("commands").as_ref()?.send(command(reply)).ok()?;
        answer.recv_timeout(ASK_TIMEOUT).ok()
    }

    /// Ends every request with a terminal event, stops the threads and joins
    /// them. Idempotent; also done when the runtime is dropped.
    pub fn shutdown(&self) {
        // Taking the sender under the lock closes the door: a `submit` that
        // got in before has its command ahead of the shutdown in the channel.
        let sender = self.commands.lock().expect("commands").take();
        if let Some(sender) = sender {
            let _ = sender.send(Command::Shutdown);
        }
        let handle = self.conductor.lock().expect("conductor").take();
        // The last handle can be dropped by a sink running on the conductor
        // itself; a thread cannot join itself.
        if let Some(handle) = handle
            && thread::current().id() != self.conductor_id
        {
            let _ = handle.join();
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Engine for Runtime {
    fn models(&self) -> Vec<ModelCard> {
        vec![self.card.clone()]
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        if request.model != self.card.id {
            return Err(EngineError::ModelNotFound(request.model));
        }
        request.sampling.validate()?;
        // The sampler on the devices has no penalties yet. A value that
        // switches a penalty off asks for nothing; any other is refused, so
        // that no request gets text that ignores what it asked for.
        let sampling = &request.sampling;
        for (name, value, off) in [
            ("presence_penalty", sampling.presence_penalty, 0.0),
            ("frequency_penalty", sampling.frequency_penalty, 0.0),
            ("repeat_penalty", sampling.repeat_penalty, 1.0),
        ] {
            if value.is_some_and(|v| v != off) {
                return Err(EngineError::InvalidRequest(format!(
                    "{name} is not implemented: the sampler has temperature, top_k, top_p, \
                     min_p and seed. Leave {name} out or set it to {off}"
                )));
            }
        }
        if request.max_tokens == Some(0) {
            return Err(EngineError::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let prompt = self.model.encode(&request.prompt)?;
        if prompt.is_empty() {
            return Err(EngineError::InvalidRequest(
                "the prompt is empty: there is nothing to continue".into(),
            ));
        }
        // At least one token must fit after the prompt, in the model's
        // context and in the page pool. This is the one refusal that depends
        // on a size, and it does not depend on what else is running.
        if prompt.len() >= self.limit as usize {
            return Err(EngineError::ContextOverflow {
                prompt_tokens: u32::try_from(prompt.len()).unwrap_or(u32::MAX),
                limit: self.limit,
            });
        }
        let submission = Box::new(Submission {
            id: request.id,
            prompt,
            max_tokens: request.max_tokens,
            stop: request.stop,
            sampling: request.sampling,
            sink: events,
        });
        let sent = match self.commands.lock().expect("commands").as_ref() {
            Some(sender) => sender.send(Command::Submit(submission)).is_ok(),
            None => false,
        };
        if !sent {
            return Err(EngineError::Internal("the engine has shut down".into()));
        }
        Ok(())
    }

    fn decide(&self, _: DecisionRequest, _: DecisionSink) -> Result<(), EngineError> {
        Err(EngineError::Unsupported(
            "decisions are not implemented by the runtime skeleton: scoring option rows needs \
             the model's logits for forked sequences"
                .into(),
        ))
    }

    fn cancel(&self, id: RequestId) {
        if let Some(sender) = self.commands.lock().expect("commands").as_ref() {
            let _ = sender.send(Command::Cancel(id));
        }
    }

    fn stats(&self) -> Option<EngineStats> {
        let snapshot = self.snapshot()?;
        let c = &snapshot.counters;
        Some(EngineStats {
            steps: c.steps,
            prompt_tokens: c.prefill_rows,
            generated_tokens: c.tokens,
            drafts_proposed: c.draft_rows,
            drafts_accepted: c.drafts_accepted,
            requests_finished: c.requests_finished,
            requests_cancelled: c.requests_cancelled,
            // Nothing is refused for lack of room: a request waits in line.
            requests_refused: 0,
            sequences: snapshot.active,
            pages_free: snapshot.pages_free,
            pages: snapshot.pages_capacity,
        })
    }
}
