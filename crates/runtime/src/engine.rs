//! The front of the runtime: an [`Engine`] over a conductor thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::Duration;

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, EventSink,
    GenerateRequest, ModelCard, Prompt, RequestId,
};
use ares_cuda::Device;

use crate::conductor::{Command, Conductor, SequenceInfo, StatsCopy, Submission};
use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::exec::{ExecError, Executor};
use crate::gpu::{GpuConfig, GpuExecutor};
use crate::model::Model;
use crate::sampling::TokenFilter;
use crate::structured::Grammars;
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
    /// The grammar engine of structured output.
    grammars: Grammars,
    /// `None` after shutdown. Sending happens under the lock, so a request
    /// is either in the channel before the shutdown command or refused.
    commands: Mutex<Option<Sender<Command>>>,
    conductor: Mutex<Option<JoinHandle<()>>>,
    conductor_id: ThreadId,
    /// Cleared when the conductor thread has ended, and with it the device
    /// threads the conductor's executor owns.
    running: Arc<AtomicBool>,
    /// Why the engine accepts no more requests, once its devices have failed.
    failure: Arc<OnceLock<String>>,
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
        if config.window_rows == 0 {
            return Err(RuntimeError::Config(
                "a window has at least one row, its newest token (1: no drafts)".into(),
            ));
        }
        if !(config.prefill_slowdown > 1.0 && config.prefill_slowdown.is_finite()) {
            return Err(RuntimeError::Config(format!(
                "the slowdown of generation while a prompt is read must be above 1, got {}",
                config.prefill_slowdown
            )));
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
        let grammars = Grammars::start(&*model);
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let limit = card.context_length.min(config.pages.saturating_mul(config.cells_per_page));
        let warm_up = config.warm_up;
        let mut conductor = Conductor::new(model.clone(), config, executor, rx);
        let failure = conductor.failure();
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
                return Err(match error {
                    ExecError::Device { device, message } => {
                        RuntimeError::Device { device, message }
                    }
                    other => RuntimeError::Thread(other.to_string()),
                });
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
            grammars,
            commands: Mutex::new(Some(tx)),
            conductor_id: thread.thread().id(),
            conductor: Mutex::new(Some(thread)),
            running,
            failure,
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

    /// Starts a request whose picks are restricted by `filter` (see
    /// [`TokenFilter`]): what [`Engine::submit`] does for a request that has
    /// none.
    pub fn submit_filtered(
        &self,
        request: GenerateRequest,
        events: EventSink,
        filter: Box<dyn TokenFilter>,
    ) -> Result<(), EngineError> {
        self.start_request(request, events, Some(filter))
    }

    fn start_request(
        &self,
        request: GenerateRequest,
        events: EventSink,
        filter: Option<Box<dyn TokenFilter>>,
    ) -> Result<(), EngineError> {
        if request.model != self.card.id {
            return Err(EngineError::ModelNotFound(request.model));
        }
        request.sampling.validate()?;
        if request.max_tokens == Some(0) {
            return Err(EngineError::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let filter = match (filter, &request.structure) {
            (Some(filter), _) => Some(filter),
            (None, Some(structure)) => Some(self.grammars.filter(structure)?),
            (None, None) => None,
        };
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
            draft: request.draft,
            filter,
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

    fn chat_format(&self, model: &str) -> Option<ChatFormat> {
        (model == self.card.id).then(|| self.model.chat_format()).flatten()
    }

    fn count_tokens(&self, model: &str, prompt: &Prompt) -> Result<u32, EngineError> {
        if model != self.card.id {
            return Err(EngineError::ModelNotFound(model.to_owned()));
        }
        Ok(u32::try_from(self.model.encode(prompt)?.len()).unwrap_or(u32::MAX))
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        self.start_request(request, events, None)
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

    fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
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
            prefix_hits: c.prefix_hits,
            cached_cells: c.cached_cells,
            replayed_rows: c.replay_rows,
            checkpoints_saved: c.checkpoints_saved,
            checkpoints_restored: c.checkpoints_restored,
            checkpoints_dropped: c.checkpoints_dropped,
            page_copies: c.page_copies,
            sequences: snapshot.active,
            pages_free: snapshot.pages_free,
            pages: snapshot.pages_capacity,
            cached_pages: snapshot.cached_pages,
            checkpoint_ram: snapshot.checkpoint_ram,
            experts: self.model.expert_stats().unwrap_or_default(),
            ssd: ares_api::SsdStats {
                pages_written: c.ssd.pages_written,
                checkpoints_written: c.ssd.states_written,
                bytes_written: c.ssd.bytes_written,
                write_ns: c.ssd.write_ns,
                pages_read: c.ssd.pages_read,
                checkpoints_read: c.ssd.states_read,
                bytes_read: c.ssd.bytes_read,
                read_ns: c.ssd.read_ns,
                transfer_ns: c.ssd.transfer_ns,
                store_bytes: c.ssd.store_bytes,
                pending: c.ssd.pending,
            },
        })
    }
}
