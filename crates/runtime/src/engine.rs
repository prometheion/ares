//! The front of the runtime: an [`Engine`] over a conductor thread.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::Duration;

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, EventSink,
    GenerateRequest, Json, Labels, ModelCard, Prompt, RequestId, SINGLE_LABELS, ThinkingFormat,
};
use ares_cuda::Device;

use crate::conductor::{
    Command, Conductor, DecisionSubmission, Role, SequenceInfo, StatsCopy, Submission,
};
use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::exec::{ExecError, Executor};
use crate::gpu::{GpuConfig, GpuExecutor};
use crate::guard::Guard;
use crate::model::{EncodedPrompt, Model};
use crate::sampling::TokenFilter;
use crate::structured::Grammars;
use crate::timing::Snapshot;
use crate::watch::{Thinking, Watch};

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
    /// What the engine was asked so far, for models that pause long steps
    /// ([`Executor::asks`]).
    asks: Option<Arc<AtomicU64>>,
    conductor: Mutex<Option<JoinHandle<()>>>,
    conductor_id: ThreadId,
    /// Cleared when the conductor thread has ended, and with it the device
    /// threads the conductor's executor owns.
    running: Arc<AtomicBool>,
    /// Why the engine accepts no more requests, once its devices have failed.
    failure: Arc<OnceLock<String>>,
    /// Set when the watchdog found a step that does not end: the threads
    /// that hang are not waited for when the engine goes.
    stuck: Arc<AtomicBool>,
    /// The requests the engine holds, running and in line: one for every
    /// sink it has not let go of.
    held: Arc<AtomicU32>,
    /// The most requests that may wait in line while the slots are taken;
    /// `None`: the line has no limit.
    queue: Option<u32>,
    slots: u32,
    /// The text of the chat template conversations are rendered with.
    template: Option<String>,
    /// The labels a decision reads its options by, found in the model's
    /// tokenizer at the first decision.
    labels: OnceLock<Result<Labels, String>>,
}

/// One request the engine holds; gives its place up when the sink that
/// carries it is dropped, however the request ends.
struct Held(Arc<AtomicU32>);

impl Drop for Held {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
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
        if !(config.generation_floor >= 0.0 && config.generation_floor.is_finite()) {
            return Err(RuntimeError::Config(format!(
                "the rate generation keeps while a prompt is read must be 0 or more tokens a \
                 second, got {}",
                config.generation_floor
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
        let (queue, slots, watchdog) = (config.queue, config.max_slots, config.watchdog);
        // When the prompt cache must give page-locked memory back, the
        // conductor is woken to give the coldest checkpoints and pages up,
        // also while nothing runs.
        if let Some(ram) = executor.ram() {
            let wake = Mutex::new(tx.clone());
            ram.on_cache(move |_| {
                let _ = wake.lock().unwrap_or_else(|e| e.into_inner()).send(Command::Trim);
            });
        }
        let asks = executor.asks();
        let mut conductor = Conductor::new(model.clone(), config, executor, rx);
        let failure = conductor.failure();
        let heartbeat = conductor.heartbeat();
        let running = Arc::new(AtomicBool::new(true));
        let stuck = Arc::new(AtomicBool::new(false));
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
        if let Some(limit) = watchdog {
            watch(heartbeat, limit, running.clone(), failure.clone(), stuck.clone())?;
        }
        Ok(Self {
            model,
            card,
            limit,
            grammars,
            stuck,
            commands: Mutex::new(Some(tx)),
            asks,
            conductor_id: thread.thread().id(),
            conductor: Mutex::new(Some(thread)),
            running,
            failure,
            held: Arc::new(AtomicU32::new(0)),
            queue,
            slots,
            template: None,
            labels: OnceLock::new(),
        })
    }

    /// Counts a request or cancellation sent to the conductor, for a model that
    /// pauses a long step so that the conductor can answer it.
    fn asked(&self) {
        if let Some(asks) = &self.asks {
            asks.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// The runtime, which says that conversations are rendered with the
    /// chat template `source` ([`Engine::chat_template`]). The template
    /// itself is the model's business: this is what the engine tells.
    pub fn with_chat_template(mut self, source: Option<String>) -> Self {
        self.template = source;
        self
    }

    /// Starts a runtime with a thread per device over `devices`. The engine
    /// holds whatever the RAM limit of `gpu` allows.
    pub fn start_gpu(
        model: Arc<dyn Model>,
        devices: Vec<Device>,
        config: RuntimeConfig,
        gpu: GpuConfig,
    ) -> Result<Self, RuntimeError> {
        Self::start_gpu_following(model, devices, config, gpu, None)
    }

    /// [`Runtime::start_gpu`] for an engine that follows the memory of
    /// `machine` below the RAM limit of `gpu` (see [`crate::Machine`]).
    pub fn start_gpu_following(
        model: Arc<dyn Model>,
        devices: Vec<Device>,
        config: RuntimeConfig,
        gpu: GpuConfig,
        machine: Option<Arc<crate::Machine>>,
    ) -> Result<Self, RuntimeError> {
        let executor = GpuExecutor::start_following(model.clone(), devices, gpu, machine)?;
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
        let EncodedPrompt { tokens: prompt, boundaries, images } =
            self.model.encode_with_boundaries(&request.prompt)?;
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
        // A full line refuses: the slots are taken and as many wait as may.
        // The conductor, which knows the line, decides: a request of a better
        // priority than the worst in line that has not started takes that
        // one's place, which is refused; else this one is refused, as an
        // event (a route gives it its status all the same).
        let held = self.held.fetch_add(1, Ordering::AcqRel) + 1;
        let place = Held(self.held.clone());
        let displace = self.queue.is_some_and(|queue| held > self.slots.saturating_add(queue));
        let events = EventSink::new(move |event| {
            let _place = &place;
            events.send(event)
        });
        let format = self.model.chat_format().and_then(|format| format.thinking);
        let thinking = format
            .as_ref()
            .map(|format| thinking(&*self.model, &prompt, format, request.structure.as_ref()));
        // Every reply is guarded against a loop; a budget ends the thinking.
        let forced = match &format {
            Some(format) => {
                let message = request.sampling.reasoning_budget_message.clone().unwrap_or_default();
                let mut forced = self.model.encode(&Prompt::Text(message + &format.end))?;
                // A token the model adds before a text (one that writes
                // nothing) is not part of the end of the thinking.
                let mut piece = Vec::new();
                while let Some(&first) = forced.first() {
                    piece.clear();
                    self.model.piece(first, &mut piece);
                    if !piece.is_empty() {
                        break;
                    }
                    forced.remove(0);
                }
                forced
            }
            // A model that does not think has no thinking to end: a budget
            // (a default of the model, say) has nothing to bound.
            None => Vec::new(),
        };
        let model = self.model.clone();
        let filter: Option<Box<dyn TokenFilter>> = Some(Box::new(Guard::new(
            Arc::new(move |token, text: &mut Vec<u8>| model.piece(token, text)),
            filter,
            thinking.clone(),
            request.sampling.reasoning_budget_tokens,
            forced,
        )));
        let submission = Box::new(Submission {
            id: request.id,
            prompt,
            boundaries,
            images,
            max_tokens: request.max_tokens,
            stop: Watch::new(&request.stop, thinking),
            sampling: request.sampling,
            draft: request.draft,
            filter,
            priority: request.priority,
            displace,
            sink: events,
            role: Role::Request,
        });
        let sent = match self.commands.lock().expect("commands").as_ref() {
            Some(sender) => sender.send(Command::Submit(submission)).is_ok(),
            None => false,
        };
        self.asked();
        if !sent {
            return Err(EngineError::Internal("the engine has shut down".into()));
        }
        Ok(())
    }

    /// The labels a decision reads the options of the model by, found in its
    /// tokenizer (see [`ares_api::Labels`]): what [`Engine::decide`] uses, for
    /// whoever needs to ask a question as the engine does.
    pub fn labels(&self) -> Result<Labels, EngineError> {
        self.labels
            .get_or_init(|| labels_of(&*self.model))
            .clone()
            .map_err(EngineError::Unsupported)
    }

    /// [`Engine::decide`] with `labels` for the labels of the options in place
    /// of the ones found in the model's tokenizer: for a model whose tokenizer
    /// has none (the synthetic model of the tests reads a byte per token).
    pub fn decide_with(
        &self,
        request: DecisionRequest,
        outcome: DecisionSink,
        labels: &Labels,
    ) -> Result<(), EngineError> {
        if request.model != self.card.id {
            return Err(EngineError::ModelNotFound(request.model));
        }
        request.validate()?;
        // Each question as a prompt of its own, thinking off, as Flide's
        // next_token format asks a chat model.
        let mut prompts = Vec::with_capacity(request.questions.len());
        let mut tokens = Vec::with_capacity(request.questions.len());
        for question in &request.questions {
            let rendered = question.render(labels)?;
            prompts.push(
                self.model.encode(&ares_api::question_prompt(&request.state, &rendered.text))?,
            );
            tokens.push(rendered.tokens);
        }
        // The start every prompt shares: each keeps at least its last token,
        // whose logits are read.
        let shortest = prompts.iter().map(Vec::len).min().unwrap_or(0);
        let mut shared = 0;
        while shared + 1 < shortest && prompts.iter().all(|p| p[shared] == prompts[0][shared]) {
            shared += 1;
        }
        let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let total = shared + prompts.iter().map(|p| p.len() - shared).sum::<usize>();
        // Flide's limits, then what the model's context and the pool hold.
        if longest > ares_api::MAX_BRANCH_TOKENS {
            return Err(EngineError::InvalidRequest(format!(
                "state plus longest question is {longest} tokens, limit is {}",
                ares_api::MAX_BRANCH_TOKENS
            )));
        }
        if total > ares_api::MAX_REQUEST_TOKENS {
            return Err(EngineError::InvalidRequest(format!(
                "request is {total} tokens, limit is {}",
                ares_api::MAX_REQUEST_TOKENS
            )));
        }
        if longest >= self.limit as usize {
            return Err(EngineError::ContextOverflow {
                prompt_tokens: u32::try_from(longest).unwrap_or(u32::MAX),
                limit: self.limit,
            });
        }
        let submission = Box::new(DecisionSubmission {
            id: request.id,
            questions: request.questions,
            prompts,
            labels: tokens,
            shared,
            priority: request.priority,
            sink: outcome,
        });
        let sent = match self.commands.lock().expect("commands").as_ref() {
            Some(sender) => sender.send(Command::Decide(submission)).is_ok(),
            None => false,
        };
        self.asked();
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
        if self.stuck.load(Ordering::Acquire) {
            // A step that never ended: the conductor waits for it and would
            // never be joined. It and its devices are left behind, with the
            // memory they hold, until the process ends (the failure says why).
            return;
        }
        // The last handle can be dropped by a sink running on the conductor
        // itself; a thread cannot join itself.
        if let Some(handle) = handle
            && thread::current().id() != self.conductor_id
        {
            let _ = handle.join();
        }
    }
}

/// Starts the watchdog of an engine: a thread that looks every second at how
/// long the step in flight has run, and fails the engine when it has run
/// longer than `limit` (`RuntimeConfig::watchdog`). It ends with the
/// conductor.
fn watch(
    heartbeat: Arc<crate::conductor::Heartbeat>,
    limit: Duration,
    running: Arc<AtomicBool>,
    failure: Arc<OnceLock<String>>,
    stuck: Arc<AtomicBool>,
) -> Result<(), RuntimeError> {
    thread::Builder::new()
        .name("ares-watchdog".to_owned())
        .spawn(move || {
            while running.load(Ordering::Acquire) && failure.get().is_none() {
                thread::sleep(Duration::from_secs(1).min(limit));
                let Some(busy) = heartbeat.busy() else { continue };
                if busy > limit {
                    let why = format!(
                        "the engine has been in one step for {} s, longer than its watchdog \
                         allows ({} s): a device does not finish its work",
                        busy.as_secs(),
                        limit.as_secs()
                    );
                    stuck.store(true, Ordering::Release);
                    let _ = failure.set(why);
                    return;
                }
            }
        })
        .map(drop)
        .map_err(|e| RuntimeError::Thread(format!("cannot start the watchdog: {e}")))
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
        Ok(u32::try_from(self.tokenize(model, prompt)?.len()).unwrap_or(u32::MAX))
    }

    fn tokenize(&self, model: &str, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        if model != self.card.id {
            return Err(EngineError::ModelNotFound(model.to_owned()));
        }
        self.model.encode(prompt)
    }

    fn pieces(&self, model: &str, tokens: &[u32]) -> Result<Vec<Vec<u8>>, EngineError> {
        if model != self.card.id {
            return Err(EngineError::ModelNotFound(model.to_owned()));
        }
        let vocab = self.model.vocab();
        tokens
            .iter()
            .map(|&token| {
                if token >= vocab {
                    return Err(EngineError::InvalidRequest(format!(
                        "token {token} is not one of the {vocab} tokens of the model"
                    )));
                }
                let mut bytes = Vec::new();
                self.model.piece(token, &mut bytes);
                Ok(bytes)
            })
            .collect()
    }

    fn reads_images(&self, model: &str) -> bool {
        model == self.card.id && self.model.vision().is_some()
    }

    fn chat_template(&self, model: &str) -> Option<String> {
        (model == self.card.id).then(|| self.template.clone()).flatten()
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        self.start_request(request, events, None)
    }

    fn decide(&self, request: DecisionRequest, outcome: DecisionSink) -> Result<(), EngineError> {
        if request.model != self.card.id {
            return Err(EngineError::ModelNotFound(request.model));
        }
        let labels = self
            .labels
            .get_or_init(|| labels_of(&*self.model))
            .as_ref()
            .map_err(|why| EngineError::Unsupported(why.clone()))?;
        self.decide_with(request, outcome, labels)
    }

    fn cancel(&self, id: RequestId) {
        if let Some(sender) = self.commands.lock().expect("commands").as_ref() {
            let _ = sender.send(Command::Cancel(id));
        }
        self.asked();
    }

    fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
    }

    fn end_thinking(&self, id: RequestId) -> Result<bool, EngineError> {
        self.ask(|reply| Command::EndThinking { id, reply })
            .ok_or_else(|| EngineError::Internal("the engine is not running".into()))
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
            requests_outranked: c.requests_outranked,
            requests_displaced: c.requests_displaced,
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
            ram: ares_api::RamCacheStats {
                pages: c.ram.pages,
                bytes: c.ram.bytes,
                pages_in: c.ram.pages_in,
                pages_read: c.ram.pages_read,
                pages_dropped: c.ram.pages_dropped,
                transfer_ns: c.ram.transfer_ns,
            },
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
                checkpoints_skipped: c.ssd.states_skipped,
            },
        })
    }
}

/// The labels a decision reads the options of `model` by: each label's token
/// where the reply starts, after the chat template's generation prompt.
///
/// Ported from Flide's `NextToken._reply_token` and `NextToken._pairs`
/// (`flide/formats/next_token.py`, <https://github.com/OmniJev/Flide>,
/// commit caffe660df88db747d6da9f95f7d6759fea38698): the prompt of a question
/// is cut after its last control token (the start of the reply, which the
/// template writes after it, is what is left), and a label is the one token
/// that text and the label add to the text alone. Every one of
/// [`SINGLE_LABELS`] must be such a token, and no two the same; the two-letter
/// labels are those that are and differ from every label before them, up to
/// [`Labels::MAX_PAIRS`].
fn labels_of(model: &dyn crate::model::Model) -> Result<Labels, String> {
    let unusable = |why: String| format!("this model cannot answer decisions: {why}");
    let probe = model
        .encode(&ares_api::question_prompt(&Json::Null, "x"))
        .map_err(|e| unusable(e.to_string()))?;
    let mut piece = Vec::new();
    let cut = probe
        .iter()
        .rposition(|&token| {
            piece.clear();
            model.piece(token, &mut piece);
            piece.is_empty()
        })
        .map_or(0, |at| at + 1);
    let mut reply = Vec::new();
    for &token in &probe[cut..] {
        model.piece(token, &mut reply);
    }
    let reply = String::from_utf8_lossy(&reply).into_owned();
    let encode = |text: String| model.encode(&Prompt::Text(text)).unwrap_or_default();
    let base = encode(reply.clone());
    let token_of = |label: &str| -> Option<u32> {
        let tokens = encode(format!("{reply}{label}"));
        (tokens.len() == base.len() + 1 && tokens[..base.len()] == base[..])
            .then(|| tokens[base.len()])
    };
    let mut singles = Vec::with_capacity(SINGLE_LABELS.len());
    for label in SINGLE_LABELS {
        match token_of(label) {
            Some(token) if !singles.contains(&token) => singles.push(token),
            _ => {
                return Err(unusable(format!(
                    "label {label:?} is not one distinct token after this model's chat prompt"
                )));
            }
        }
    }
    let mut pairs = Vec::new();
    for label in Labels::pair_candidates() {
        if pairs.len() == Labels::MAX_PAIRS {
            break;
        }
        if let Some(token) = token_of(&label)
            && !singles.contains(&token)
            && !pairs.iter().any(|(_, t)| *t == token)
        {
            pairs.push((label, token));
        }
    }
    Ok(Labels { singles, pairs })
}

/// Where the thinking of a reply to `prompt` is: inside, when the last tag of
/// the thinking the prompt holds is the opening one (the template opened the
/// thinking for the model); else the reply may open it with its first text.
/// The prompt is read backwards, its last tokens first, until a tag is found.
/// A call of a tool may end the thinking when the request's rule says so.
fn thinking(
    model: &dyn crate::model::Model,
    prompt: &[u32],
    format: &ThinkingFormat,
    structure: Option<&ares_api::Structure>,
) -> Thinking {
    let (open, close) = (format.start_tag().to_owned(), format.end_tag().to_owned());
    let reversed = |tag: &str| tag.bytes().rev().collect::<Vec<u8>>();
    let (open_back, close_back) = (reversed(&open), reversed(&close));
    // The bytes of the prompt from its end, reversed.
    let mut back: Vec<u8> = Vec::new();
    let mut inside = false;
    let mut piece = Vec::new();
    for &token in prompt.iter().rev() {
        piece.clear();
        model.piece(token, &mut piece);
        let searched = back.len().saturating_sub(open_back.len().max(close_back.len()));
        back.extend(piece.iter().rev());
        let found = |tag: &[u8]| {
            (!tag.is_empty())
                .then(|| back[searched..].windows(tag.len()).position(|w| w == tag))
                .flatten()
                .map(|at| at + searched)
        };
        match (found(&open_back), found(&close_back)) {
            (Some(o), Some(c)) => {
                inside = o < c;
                break;
            }
            (Some(_), None) => {
                inside = true;
                break;
            }
            (None, Some(_)) => break,
            (None, None) => {}
        }
    }
    let interrupt = structure.and_then(|s| s.interrupt.as_ref()).map(|i| i.text.clone());
    Thinking { inside, open, close, interrupt }
}
