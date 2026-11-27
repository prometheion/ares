//! The models a server offers, of which those are loaded that requests ask
//! for: the presets of a preset file (router mode), or the one preset of
//! `ares --model`, which is loaded before the server listens. Every request
//! takes its model here; there is no other way to serve one.
//!
//! A [`Models`] holds the presets ([`crate::preset`]) and, for each, whether
//! its model is loaded. A request names a model by its id or an alias and
//! takes a [`Lease`] on it ([`Models::acquire`]): if the model is not loaded
//! the request waits until it is, and while the lease lives the model is not
//! unloaded. One task, the manager, does every load and every unload, one at
//! a time, so that what the devices report as free is what the next load
//! finds.
//!
//! # Making room
//!
//! Nothing counts models. A load is planned by the loader against what the
//! devices have free and what the RAM limit has left ([`Site`]), and when the
//! plan says that it does not fit ([`LoadFailure::NoRoom`]) the manager
//! unloads the model that has been idle the longest and plans again, until it
//! fits or nothing is left to unload. A model that is serving requests is
//! never unloaded under them: when only such models are in the way, the one
//! used least recently gives way, which means that it gets no new requests
//! (they wait in line behind the load, and load it again afterwards) and is
//! unloaded when its running requests have ended. The wait of the new request
//! is therefore bounded by the requests that run when it arrives, and it
//! waits; no request is answered 503 for a model that is busy.
//!
//! # Idle unload
//!
//! A model whose preset has an `idle-unload` is unloaded when no request has
//! used it for that long. The time is the [`Clock`]'s, which tests set.
//!
//! # Failures
//!
//! A load that fails leaves its model unloaded with the reason, which the
//! requests that waited for it get and `/v1/models` shows; the next request
//! tries again, and the other models are not touched. An engine that says it
//! has failed ([`Engine::failure`]: one of its devices) gets no more requests,
//! is unloaded when the requests it failed have ended, and is listed as
//! failed with the engine's reason until a request loads it again.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ares_api::{Engine, ModelCard};
use tokio::sync::{Notify, broadcast, watch};

use crate::ReasoningFormat;
use crate::bridge::Flight;
use crate::defaults::Defaults;
use crate::options::ModelOptions;
use crate::preset::{Preset, Presets};

/// What a load finds next to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    /// Whether other models of this server are loaded, which could be
    /// unloaded for this one. A loader that finds too little room then says
    /// [`LoadFailure::NoRoom`], and makes do with what there is otherwise.
    pub beside: bool,
    /// The most RAM the model's engine may use. `None`: the loader's own
    /// default (the model is alone and the server's limit is not named).
    pub ram: Option<u64>,
    /// Whether `ram` is what the server's limit has left next to the loaded
    /// models, and not a number the preset names.
    pub ram_is_rest: bool,
}

/// Why a model was not loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadFailure {
    /// The devices or the RAM cannot hold it next to what is loaded.
    NoRoom(String),
    /// Anything that unloading other models does not change.
    Failed(String),
}

/// Loads models. The server's loader puts them on the devices
/// ([`crate::load::DeviceLoader`]); tests script one.
pub trait Loader: Send + Sync + 'static {
    /// Plans and loads the model of `preset`. Takes as long as the weights
    /// take; called from a thread that may block, one call at a time.
    fn load(&self, preset: &Preset, site: &Site) -> Result<Box<dyn Unit>, LoadFailure>;

    /// The server stops: a load that runs now or later ends as soon as it
    /// can, with what it has (the weights reach the devices; what fills a
    /// model's caches afterwards is cut short). Called from any thread.
    fn stop(&self) {}
}

/// A loaded model.
pub trait Unit: Send + 'static {
    /// The engine that serves it.
    fn engine(&self) -> Arc<dyn Engine>;
    /// The most RAM its engine may use: what it takes of the server's limit.
    fn ram(&self) -> u64;
    /// Ends what still runs and gives back everything the model holds.
    /// Returns when that is done; called from a thread that may block.
    fn unload(self: Box<Self>);
}

/// The loader of an engine that is already there, for a program that builds
/// its engine itself and serves it (a test, an example): every preset it
/// loads is that engine, and an unload leaves the engine to its owner.
#[derive(Clone)]
pub struct Given(pub Arc<dyn Engine>);

impl std::fmt::Debug for Given {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Given").finish_non_exhaustive()
    }
}

impl Loader for Given {
    fn load(&self, _: &Preset, _: &Site) -> Result<Box<dyn Unit>, LoadFailure> {
        Ok(Box::new(self.clone()))
    }
}

impl Unit for Given {
    fn engine(&self) -> Arc<dyn Engine> {
        self.0.clone()
    }

    fn ram(&self) -> u64 {
        0
    }

    fn unload(self: Box<Self>) {}
}

/// The time the idle unload goes by.
pub trait Clock: Send + Sync + 'static {
    /// Time since some fixed moment.
    fn now(&self) -> Duration;
    /// Completes when [`Clock::now`] has reached `at`.
    fn sleep_until(&self, at: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

/// The machine's clock.
#[derive(Debug)]
pub struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }

    fn sleep_until(&self, at: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(tokio::time::sleep_until((self.0 + at).into()))
    }
}

/// A change of a model's state, as `GET /models/sse` sends it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEvent {
    pub model: String,
    /// `loading`, `loaded`, `unloading`, `unloaded` or `failed`.
    pub status: &'static str,
    /// Why: of a failure the reason, of an unload what asked for it.
    pub reason: Option<String>,
}

/// A model as `GET /v1/models` lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct Listed {
    pub id: String,
    pub aliases: Vec<String>,
    /// The card of its engine: of the loaded model, or as it was last loaded.
    pub card: Option<ModelCard>,
    pub status: ModelEvent,
    /// What a request to the model gets for the fields it leaves out.
    pub defaults: Arc<Defaults>,
}

/// Why a request does not get its model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquireError {
    /// No preset answers to the name; `available` are the ids there are.
    NotFound { model: String, available: Vec<String> },
    /// The model could not be loaded.
    Failed { model: String, reason: String },
    /// The model is not loaded, and the caller did not ask for a load.
    NotLoaded { model: String },
    /// The server is stopping.
    Stopping,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
}

/// A model that is loaded: its unit, the engine it serves, the record of
/// its requests in flight, and the RAM it holds.
struct Running {
    unit: Box<dyn Unit>,
    engine: Arc<dyn Engine>,
    flight: Arc<Flight>,
    ram: u64,
}

struct Entry {
    preset: Arc<Preset>,
    /// The preset's, for the requests to take with their hold.
    defaults: Arc<Defaults>,
    phase: Phase,
    loaded: Option<Running>,
    card: Option<ModelCard>,
    /// Requests that hold the model.
    leases: usize,
    /// Requests that wait for it.
    waiters: usize,
    /// Of those, the ones that waited for the load that brought it: each
    /// takes its hold, or goes away, before the model may be unloaded again.
    owed: usize,
    /// `attempts` after the load that brought it.
    loaded_at: u64,
    /// When a request last took or gave back the model.
    last_used: Duration,
    /// Its place in the line of loads, while someone wants it loaded.
    wanted: Option<u64>,
    /// A load nobody waits for: at start, or asked through the endpoint.
    asked: bool,
    /// Its load found no room and waits for a busy model to end its requests.
    blocked: Option<String>,
    /// The model whose load this one is unloaded for, once it is idle.
    gives_way_to: Option<usize>,
    /// An unload asked through the endpoint, for when it is idle.
    unload_asked: bool,
    /// What its engine said when it failed.
    broken: Option<String>,
    /// Loads that have ended, well or not.
    attempts: u64,
    /// Why the last load failed, or why the model was taken down.
    failure: Option<String>,
    /// The status last sent as an event.
    said: &'static str,
}

impl Entry {
    /// Whether requests hold the model or are about to.
    fn busy(&self) -> bool {
        self.leases > 0 || self.owed > 0
    }

    fn status(&self) -> (&'static str, Option<String>) {
        match self.phase {
            Phase::Loading => ("loading", None),
            Phase::Loaded => ("loaded", None),
            Phase::Unloading => ("unloading", None),
            Phase::Unloaded if self.wanted.is_some() => {
                ("loading", self.blocked.as_ref().map(|why| format!("waiting for room: {why}")))
            }
            Phase::Unloaded => match &self.failure {
                Some(why) => ("failed", Some(why.clone())),
                None => ("unloaded", None),
            },
        }
    }
}

struct State {
    entries: Vec<Entry>,
    tickets: u64,
    stopping: bool,
    /// The server takes no more requests and lets those that run finish
    /// ([`Models::drain`]): nothing more is loaded, and a request that waits
    /// for a model that is not loaded is told that the server stops.
    draining: bool,
}

struct Shared {
    state: Mutex<State>,
    loader: Arc<dyn Loader>,
    clock: Arc<dyn Clock>,
    /// Wakes the manager.
    wake: Notify,
    /// Counts the changes of state, for the requests that wait for one.
    version: watch::Sender<u64>,
    events: broadcast::Sender<ModelEvent>,
    /// The most RAM of all loaded models together, when the operator named
    /// it.
    ram: Option<u64>,
}

enum Job {
    Load(usize),
    Unload(usize, String),
    Wait(Option<Duration>),
    Exit,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Tells the requests that wait, and the manager, that the state changed.
    fn changed(&self) {
        self.version.send_modify(|version| *version += 1);
        self.wake.notify_one();
    }

    /// Sends the status of `entry` as an event if it is not the one sent last.
    fn say(&self, entry: &mut Entry, reason: Option<String>) {
        let (status, own) = entry.status();
        if status == entry.said {
            return;
        }
        entry.said = status;
        let reason = reason.or(own);
        match (&reason, status) {
            (Some(why), "failed") => tracing::error!(model = %entry.preset.id, "{status}: {why}"),
            // An engine that failed is unloaded for it: the failure is an error.
            (Some(why), "unloading") if entry.broken.is_some() => {
                tracing::error!(model = %entry.preset.id, "{status} ({why})")
            }
            (Some(why), _) => tracing::info!(model = %entry.preset.id, "{status} ({why})"),
            (None, _) => tracing::info!(model = %entry.preset.id, "{status}"),
        }
        let _ = self.events.send(ModelEvent { model: entry.preset.id.clone(), status, reason });
    }

    fn next_job(&self) -> Job {
        let now = self.clock.now();
        let mut state = self.lock();
        let stopping = state.stopping;
        let ending = stopping || state.draining;
        let mut changed = false;
        for entry in &mut state.entries {
            // A load that nobody waits for any more is not done.
            if entry.wanted.is_some() && (ending || (entry.waiters == 0 && !entry.asked)) {
                (entry.wanted, entry.asked, entry.blocked) = (None, false, None);
                self.say(entry, None);
                changed = true;
            }
            if entry.phase == Phase::Loaded
                && entry.broken.is_none()
                && let Some(why) = entry.loaded.as_ref().and_then(|loaded| loaded.engine.failure())
            {
                entry.broken = Some(why);
            }
        }
        // Nobody gives way to a load that is not wanted any more, and the
        // loads that waited behind that one plan again: another model may
        // have to give way for them.
        for index in 0..state.entries.len() {
            if let Some(taker) = state.entries[index].gives_way_to
                && state.entries[taker].wanted.is_none()
            {
                state.entries[index].gives_way_to = None;
                for entry in &mut state.entries {
                    entry.blocked = None;
                }
                changed = true;
            }
        }
        if changed {
            self.version.send_modify(|version| *version += 1);
        }

        let mut deadline: Option<Duration> = None;
        for (index, entry) in state.entries.iter().enumerate() {
            if entry.phase != Phase::Loaded {
                continue;
            }
            // At the end nothing waits for a request: the engine ends them.
            let why = if stopping {
                Some("the server is stopping".to_owned())
            } else if entry.leases > 0 {
                None
            } else if let Some(why) = &entry.broken {
                Some(format!("its engine failed: {why}"))
            } else if entry.owed > 0 {
                // The requests it was loaded for come first.
                None
            } else if entry.unload_asked {
                Some("asked to".to_owned())
            } else if let Some(taker) = entry.gives_way_to {
                Some(format!("to make room for {}", state.entries[taker].preset.id))
            } else if let Some(idle) = entry.preset.idle_unload {
                let due = entry.last_used + idle;
                if now >= due {
                    Some(format!("idle for {} s", (now - entry.last_used).as_secs()))
                } else {
                    deadline = Some(deadline.map_or(due, |known| known.min(due)));
                    None
                }
            } else {
                None
            };
            if let Some(why) = why {
                return Job::Unload(index, why);
            }
        }
        if stopping {
            return Job::Exit;
        }
        let next = state
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.phase == Phase::Unloaded && e.blocked.is_none())
            .filter_map(|(index, e)| e.wanted.map(|ticket| (ticket, index)))
            .min();
        match next {
            Some((_, index)) => Job::Load(index),
            None => Job::Wait(deadline),
        }
    }

    async fn unload(&self, index: usize, why: String) {
        let unit = {
            let mut state = self.lock();
            let entry = &mut state.entries[index];
            entry.phase = Phase::Unloading;
            self.say(entry, Some(why));
            entry.loaded.take().map(|loaded| loaded.unit)
        };
        self.changed();
        if let Some(unit) = unit {
            // The engine's handles above are gone; the unit's is the last.
            let _ = tokio::task::spawn_blocking(move || unit.unload()).await;
        }
        let mut state = self.lock();
        for entry in &mut state.entries {
            // There is room that was not there: whoever waited for it plans again.
            entry.blocked = None;
        }
        let entry = &mut state.entries[index];
        entry.phase = Phase::Unloaded;
        (entry.unload_asked, entry.gives_way_to, entry.owed) = (false, None, 0);
        if let Some(why) = entry.broken.take() {
            entry.failure =
                Some(format!("the model was taken down because its engine failed: {why}"));
            entry.attempts += 1;
        }
        self.say(entry, None);
        drop(state);
        self.changed();
    }

    /// What the load of entry `index` finds, or why the RAM limit has no room
    /// for it.
    fn site(&self, state: &State, index: usize) -> Result<Site, String> {
        let others = || state.entries.iter().filter_map(|e| e.loaded.as_ref());
        let booked: u64 = others().map(|loaded| loaded.ram).sum();
        let beside = others().next().is_some();
        let own = state.entries[index].preset.options.load.ram;
        // Without a limit of the server there is nothing to fit into: every
        // engine follows the memory the machine has, and they share it
        // through the one watch over it.
        let Some(limit) = self.ram else {
            return Ok(Site { beside, ram: own, ram_is_rest: false });
        };
        let left = limit.saturating_sub(booked);
        match own {
            Some(own) if own > left => Err(format!(
                "its {own} bytes of RAM do not fit in the {left} bytes that the server's limit \
                 of {limit} bytes has left"
            )),
            Some(own) => Ok(Site { beside, ram: Some(own), ram_is_rest: false }),
            // Alone, the whole of a limit the operator named is as asked.
            None => Ok(Site { beside, ram: Some(left), ram_is_rest: beside }),
        }
    }

    async fn load(&self, index: usize) {
        let (preset, site) = {
            let mut state = self.lock();
            let site = self.site(&state, index);
            let entry = &mut state.entries[index];
            entry.phase = Phase::Loading;
            self.say(entry, None);
            (entry.preset.clone(), site)
        };
        self.changed();
        let result = match site {
            Err(why) => Err(LoadFailure::NoRoom(why)),
            Ok(site) => {
                let loader = self.loader.clone();
                let load = move || {
                    let unit = loader.load(&preset, &site)?;
                    let engine = unit.engine();
                    // The engine's totals are read where waiting is allowed.
                    let flight = Arc::new(Flight::new(&*engine));
                    Ok(Running { ram: unit.ram(), unit, engine, flight })
                };
                tokio::task::spawn_blocking(load).await.unwrap_or_else(|lost| {
                    Err(LoadFailure::Failed(format!("the load ended without a result: {lost}")))
                })
            }
        };
        let mut state = self.lock();
        let now = self.clock.now();
        let failed = match result {
            Ok(loaded) => {
                let entry = &mut state.entries[index];
                // The card of the model the preset names, which the engine
                // knows by the preset's id.
                let cards = loaded.engine.models();
                let own = cards.iter().position(|card| card.id == entry.preset.id);
                entry.card = cards.into_iter().nth(own.unwrap_or(0)).or(entry.card.take());
                entry.loaded = Some(loaded);
                entry.phase = Phase::Loaded;
                (entry.wanted, entry.asked, entry.failure) = (None, false, None);
                entry.attempts += 1;
                // Whoever waits now waited for this load.
                (entry.owed, entry.loaded_at) = (entry.waiters, entry.attempts);
                entry.last_used = now;
                self.say(entry, None);
                None
            }
            Err(LoadFailure::Failed(why)) => Some(why),
            Err(LoadFailure::NoRoom(why)) => {
                state.entries[index].phase = Phase::Unloaded;
                let others = |busy: bool| {
                    let entries = state.entries.iter().enumerate();
                    entries
                        .filter(move |(other, e)| {
                            *other != index && e.phase == Phase::Loaded && e.busy() == busy
                        })
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(other, _)| other)
                };
                if let Some(idle) = others(false) {
                    // Unloaded at the manager's next turn; then this load again.
                    state.entries[idle].gives_way_to = Some(index);
                    None
                } else if let Some(busy) = others(true) {
                    // One busy model gives way at a time; when it is gone,
                    // whoever still finds no room plans again.
                    if !state.entries.iter().any(|e| e.gives_way_to.is_some()) {
                        state.entries[busy].gives_way_to = Some(index);
                    }
                    let entry = &mut state.entries[index];
                    entry.blocked = Some(why);
                    self.say(entry, None);
                    None
                } else {
                    Some(why)
                }
            }
        };
        if let Some(why) = failed {
            let entry = &mut state.entries[index];
            entry.phase = Phase::Unloaded;
            (entry.wanted, entry.asked, entry.blocked) = (None, false, None);
            entry.failure = Some(why);
            entry.attempts += 1;
            self.say(entry, None);
        }
        drop(state);
        self.changed();
    }

    async fn manage(self: Arc<Self>) {
        loop {
            match self.next_job() {
                Job::Exit => break,
                Job::Load(index) => self.load(index).await,
                Job::Unload(index, why) => self.unload(index, why).await,
                Job::Wait(None) => self.wake.notified().await,
                Job::Wait(Some(until)) => {
                    tokio::select! {
                        () = self.wake.notified() => {}
                        () = self.clock.sleep_until(until) => {}
                    }
                }
            }
        }
    }

    fn resolve(&self, state: &State, name: &str) -> Result<usize, AcquireError> {
        let answers = |e: &Entry| e.preset.id == name || e.preset.aliases.iter().any(|a| a == name);
        state.entries.iter().position(answers).ok_or_else(|| AcquireError::NotFound {
            model: name.to_owned(),
            available: state.entries.iter().map(|e| e.preset.id.clone()).collect(),
        })
    }

    /// Puts entry `index` in the line of loads, if it is not there.
    fn want(&self, state: &mut State, index: usize) {
        if state.entries[index].wanted.is_none() {
            state.tickets += 1;
            state.entries[index].wanted = Some(state.tickets);
            let entry = &mut state.entries[index];
            self.say(entry, None);
        }
    }
}

/// One request's wait for a model; gives its place up when dropped, with
/// its hold on the model or without.
struct Waiter<'a> {
    shared: &'a Shared,
    index: usize,
    /// The loads of the model that had ended when the wait began, or when it
    /// began again.
    since: u64,
}

impl Waiter<'_> {
    /// Whether the model was loaded since this wait began: for this request,
    /// among others.
    fn owed(&self, entry: &Entry) -> bool {
        entry.phase == Phase::Loaded && self.since < entry.loaded_at
    }
}

impl Drop for Waiter<'_> {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        let entry = &mut state.entries[self.index];
        entry.waiters -= 1;
        if self.owed(entry) {
            entry.owed = entry.owed.saturating_sub(1);
        }
        drop(state);
        // A load nobody waits for is taken out of the line, and a model
        // that owes nobody may make room.
        self.shared.wake.notify_one();
    }
}

/// What the clones of one [`Lease`] share, which gives the model back when
/// the last is dropped.
struct Hold {
    engine: Option<Arc<dyn Engine>>,
    flight: Arc<Flight>,
    /// The id of the model in its engine, which a request may have named by
    /// an alias.
    id: String,
    reasoning_format: ReasoningFormat,
    defaults: Arc<Defaults>,
    /// The models and the model's place among them.
    home: (Arc<Shared>, usize),
}

impl Drop for Hold {
    fn drop(&mut self) {
        let engine = self.engine.take();
        let (shared, index) = &self.home;
        let broken = engine.as_ref().and_then(|engine| engine.failure());
        // The engine's handle goes before the model counts as free.
        drop(engine);
        let mut state = shared.lock();
        let entry = &mut state.entries[*index];
        entry.leases -= 1;
        entry.last_used = shared.clock.now();
        if entry.broken.is_none() {
            entry.broken = broken;
        }
        drop(state);
        shared.wake.notify_one();
    }
}

/// A request's hold on a loaded model: while a lease or a clone of it lives,
/// the model is not unloaded.
#[derive(Clone)]
pub struct Lease(Arc<Hold>);

impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lease").field("model", &self.0.id).finish_non_exhaustive()
    }
}

impl Lease {
    /// The engine that serves the model.
    pub fn engine(&self) -> &Arc<dyn Engine> {
        self.0.engine.as_ref().expect("the engine is there until the lease is dropped")
    }

    pub(crate) fn flight(&self) -> &Arc<Flight> {
        &self.0.flight
    }

    /// The id the engine knows the model by: a request that named an alias
    /// is handed to the engine under this one.
    pub fn id(&self) -> &str {
        &self.0.id
    }

    /// How the reasoning of a reply is returned when a request does not say.
    pub fn reasoning_format(&self) -> ReasoningFormat {
        self.0.reasoning_format
    }

    /// What a request gets for the fields it leaves out, and the most tokens
    /// of its reply.
    pub fn defaults(&self) -> &Defaults {
        &self.0.defaults
    }
}

/// The models of a server. See the [module documentation](self).
pub struct Models {
    shared: Arc<Shared>,
    manager: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl std::fmt::Debug for Models {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Models").finish_non_exhaustive()
    }
}

impl Models {
    /// The models of `presets`, none loaded yet; those that are to be loaded
    /// at start are put in line. Without a RAM limit of the presets the
    /// engines share what the machine has available instead of parts of a
    /// limit. Must be called inside a Tokio runtime: the manager's task
    /// starts here.
    pub fn start(presets: Presets, loader: Arc<dyn Loader>, clock: Arc<dyn Clock>) -> Self {
        let entries: Vec<Entry> = presets
            .models
            .into_iter()
            .zip(1u64..)
            .map(|(preset, ticket)| Entry {
                wanted: preset.load_on_startup.then_some(ticket),
                asked: preset.load_on_startup,
                defaults: Arc::new(preset.options.defaults.clone()),
                preset: Arc::new(preset),
                phase: Phase::Unloaded,
                loaded: None,
                card: None,
                leases: 0,
                waiters: 0,
                owed: 0,
                loaded_at: 0,
                last_used: Duration::ZERO,
                blocked: None,
                gives_way_to: None,
                unload_asked: false,
                broken: None,
                attempts: 0,
                failure: None,
                said: "unloaded",
            })
            .collect();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                tickets: entries.len() as u64,
                entries,
                stopping: false,
                draining: false,
            }),
            loader,
            clock,
            wake: Notify::new(),
            version: watch::channel(0).0,
            events: broadcast::channel(256).0,
            ram: presets.ram,
        });
        let manager = tokio::spawn(shared.clone().manage());
        Self { shared, manager: Mutex::new(Some(manager)) }
    }

    /// The models of a server over `engine`, which is already there ([`Given`]),
    /// as `ares --model` serves one: a preset with `options` for each model the
    /// engine serves, loaded before this returns. Must be called inside a
    /// Tokio runtime.
    pub async fn of_engine(engine: Arc<dyn Engine>, options: ModelOptions) -> Self {
        let presets = engine.models().into_iter().map(|card| Preset {
            id: card.id,
            aliases: Vec::new(),
            options: options.clone(),
            load_on_startup: true,
            idle_unload: None,
        });
        let presets = Presets { ram: None, models: presets.collect() };
        let clock = Arc::new(SystemClock::default());
        let models = Self::start(presets, Arc::new(Given(engine)), clock);
        for id in models.offered() {
            // A load of a given engine does not fail.
            let _ = models.wait_loaded(&id).await;
        }
        models
    }

    /// The model `name` for one request: loaded if it is not, which the
    /// request waits for. Dropping the future (the client went away) gives
    /// the wait up; a load that has begun is finished all the same.
    pub async fn acquire(&self, name: &str) -> Result<Lease, AcquireError> {
        self.take(name, true).await
    }

    /// The model `name` if it is loaded now; nothing is loaded for this.
    pub async fn loaded(&self, name: &str) -> Result<Lease, AcquireError> {
        self.take(name, false).await
    }

    async fn take(&self, name: &str, load: bool) -> Result<Lease, AcquireError> {
        let shared = &self.shared;
        let mut version = shared.version.subscribe();
        let mut waiting: Option<Waiter<'_>> = None;
        loop {
            {
                let mut state = shared.lock();
                if state.stopping {
                    return Err(AcquireError::Stopping);
                }
                let draining = state.draining;
                let index = shared.resolve(&state, name)?;
                let now = shared.clock.now();
                let entry = &mut state.entries[index];
                if entry.phase == Phase::Loaded
                    && entry.broken.is_none()
                    && let Some(why) = entry.loaded.as_ref().and_then(|l| l.engine.failure())
                {
                    entry.broken = Some(why);
                    shared.wake.notify_one();
                }
                // A model that is to be unloaded takes no request but those
                // that waited for its load.
                let owed = waiting.as_ref().is_some_and(|waiter| waiter.owed(entry));
                let free = entry.broken.is_none()
                    && (owed || (entry.gives_way_to.is_none() && !entry.unload_asked));
                if let (Phase::Loaded, true, Some(loaded)) = (entry.phase, free, &entry.loaded) {
                    entry.leases += 1;
                    entry.last_used = now;
                    return Ok(Lease(Arc::new(Hold {
                        engine: Some(loaded.engine.clone()),
                        flight: loaded.flight.clone(),
                        id: entry.preset.id.clone(),
                        reasoning_format: entry.preset.options.reasoning_format,
                        defaults: entry.defaults.clone(),
                        home: (shared.clone(), index),
                    })));
                }
                if !load {
                    return Err(AcquireError::NotLoaded { model: entry.preset.id.clone() });
                }
                // The server stops: what is not loaded is loaded for nobody.
                if draining {
                    return Err(AcquireError::Stopping);
                }
                let attempts = entry.attempts;
                match &mut waiting {
                    None => {
                        entry.waiters += 1;
                        waiting = Some(Waiter { shared, index, since: attempts });
                    }
                    Some(waiter) if attempts > waiter.since => {
                        if let (Phase::Unloaded, Some(reason)) = (entry.phase, &entry.failure) {
                            let (model, reason) = (entry.preset.id.clone(), reason.clone());
                            return Err(AcquireError::Failed { model, reason });
                        }
                        // Loaded and unloaded again before this request had
                        // its turn: it waits for the next load. A model that
                        // is loaded now (its engine failed) owes it nothing.
                        if entry.phase != Phase::Loaded {
                            waiter.since = attempts;
                        }
                    }
                    Some(_) => {}
                }
                if entry.phase == Phase::Unloaded {
                    shared.want(&mut state, index);
                    shared.wake.notify_one();
                }
            }
            if version.changed().await.is_err() {
                return Err(AcquireError::Stopping);
            }
        }
    }

    /// Asks for the model `name` to be loaded, as `POST /models/load` does.
    /// Returns at once; the load is seen in the list and in the events.
    pub fn ask_load(&self, name: &str) -> Result<(), AcquireError> {
        let shared = &self.shared;
        let mut state = shared.lock();
        if state.stopping {
            return Err(AcquireError::Stopping);
        }
        let index = shared.resolve(&state, name)?;
        let entry = &mut state.entries[index];
        entry.unload_asked = false;
        if entry.phase == Phase::Unloaded {
            entry.asked = true;
            shared.want(&mut state, index);
        }
        drop(state);
        shared.changed();
        Ok(())
    }

    /// Asks for the model `name` to be unloaded, as `POST /models/unload`
    /// does: it gets no new requests and is unloaded when those that run have
    /// ended. Requests that come for it meanwhile load it again afterwards.
    /// Returns at once.
    pub fn ask_unload(&self, name: &str) -> Result<(), AcquireError> {
        let shared = &self.shared;
        let mut state = shared.lock();
        let index = shared.resolve(&state, name)?;
        let entry = &mut state.entries[index];
        // A load that nobody waits for is taken back as well.
        entry.asked = false;
        if matches!(entry.phase, Phase::Loaded | Phase::Loading) {
            entry.unload_asked = true;
        }
        drop(state);
        shared.changed();
        Ok(())
    }

    /// The ids of the presets.
    pub fn offered(&self) -> Vec<String> {
        self.shared.lock().entries.iter().map(|e| e.preset.id.clone()).collect()
    }

    /// The preset that answers to `name` (an id or an alias), loaded or not;
    /// `None` for a name no preset has.
    pub fn preset(&self, name: &str) -> Option<crate::preset::Preset> {
        let state = self.shared.lock();
        let index = self.shared.resolve(&state, name).ok()?;
        Some((*state.entries[index].preset).clone())
    }

    /// Every model with its state, in the order of the preset file.
    pub fn list(&self) -> Vec<Listed> {
        let state = self.shared.lock();
        let listed = state.entries.iter().map(|entry| {
            let (status, reason) = entry.status();
            let model = entry.preset.id.clone();
            Listed {
                id: model.clone(),
                aliases: entry.preset.aliases.clone(),
                card: entry.card.clone(),
                status: ModelEvent { model, status, reason },
                defaults: entry.defaults.clone(),
            }
        });
        listed.collect()
    }

    /// The changes of state from now on, with the list as it is now.
    pub fn watch(&self) -> (Vec<Listed>, broadcast::Receiver<ModelEvent>) {
        // Subscribed first: a change between the two is sent twice, not lost.
        let events = self.shared.events.subscribe();
        (self.list(), events)
    }

    /// Completes when the load of `name` that is in line or runs has ended:
    /// with the reason, if the model is not loaded then.
    pub async fn wait_loaded(&self, name: &str) -> Result<(), AcquireError> {
        self.acquire(name).await.map(drop)
    }

    /// The server is going to stop: a load that runs ends as soon as it can
    /// ([`Loader::stop`]), so that the stop does not wait for what fills its
    /// caches. Called at the first signal, before the running requests are
    /// waited for.
    pub fn stop_loads(&self) {
        self.shared.loader.stop();
    }

    /// The server takes no more requests and lets those that run finish (the
    /// first signal): a load that runs ends as soon as it can
    /// ([`Models::stop_loads`]), nothing more is loaded, and a request that
    /// waits for a model that is not loaded is told now that the server
    /// stops, instead of waiting for the load and running into the end of
    /// the drain. A loaded model serves on until [`Models::shutdown`].
    pub fn drain(&self) {
        self.stop_loads();
        self.shared.lock().draining = true;
        self.shared.changed();
    }

    /// Unloads every model and ends the manager. Requests that wait for a
    /// model are told that the server is stopping; a load that runs is
    /// finished first, since the weights' way to the devices is not
    /// interrupted, but what fills its caches afterwards is cut short
    /// ([`Models::stop_loads`]). Returns when everything is released.
    pub async fn shutdown(&self) {
        self.stop_loads();
        self.shared.lock().stopping = true;
        self.shared.changed();
        let manager = self.manager.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(manager) = manager {
            let _ = manager.await;
        }
    }
}
