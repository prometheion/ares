//! Router mode without a GPU: the state machine of `ares_server::models` on a
//! scripted machine, and its routes.
//!
//! The machine is a loader with made-up devices: a model takes as many units
//! of a device as its preset names tokens of context, or all a device has
//! free when it names none, and what it takes comes back when it is unloaded.
//! That is the rule of the real loader in small, and the router decides by
//! it alone. Time is a clock the tests set; nothing here sleeps for a
//! duration (a test waits for a condition, however long the machine takes).

mod common;

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{Engine, EngineError, FinishReason, ModelCard};
use ares_server::models::{AcquireError, Clock, Lease, LoadFailure, Loader, Models, Site, Unit};
use ares_server::preset::{Preset, Presets};
use ares_server::{Config, router_over};
use common::{DecisionPlan, Gate, MockEngine, Plan, Step, done, eventually, get, json, post, text};
use futures_util::StreamExt;
use tokio::sync::watch;

// --- the clock --------------------------------------------------------------

struct ManualClock(watch::Sender<Duration>);

impl ManualClock {
    fn new() -> Arc<Self> {
        Arc::new(Self(watch::channel(Duration::ZERO).0))
    }

    fn advance(&self, seconds: u64) {
        self.0.send_modify(|now| *now += Duration::from_secs(seconds));
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        *self.0.borrow()
    }

    fn sleep_until(&self, at: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        let mut now = self.0.subscribe();
        Box::pin(async move {
            let _ = now.wait_for(|now| *now >= at).await;
        })
    }
}

// --- the machine ------------------------------------------------------------

/// The least a model without a named context is loaded with next to others.
const LEAST: u64 = 2;

#[derive(Default)]
struct Machine {
    /// Free units of every device.
    free: Mutex<Vec<u64>>,
    /// What the loader was asked to do, in order.
    log: Mutex<Vec<String>>,
    /// Loads that wait for the test.
    latches: Mutex<HashMap<String, Gate>>,
    /// Models whose file is broken.
    broken_files: Mutex<HashSet<String>>,
    /// What every load found.
    sites: Mutex<Vec<(String, Site)>>,
    /// The engine of every model, as last loaded.
    engines: Mutex<HashMap<String, Arc<MockEngine>>>,
    /// What a model's engine does with a request, where it is not the usual.
    scripts: Mutex<HashMap<String, Vec<Step>>>,
}

impl Machine {
    fn new(devices: &[u64]) -> Arc<Self> {
        Arc::new(Self { free: Mutex::new(devices.to_vec()), ..Self::default() })
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    fn free(&self) -> Vec<u64> {
        self.free.lock().unwrap().clone()
    }

    fn latch(&self, model: &str) -> Gate {
        let latch = Gate::default();
        self.latches.lock().unwrap().insert(model.to_owned(), latch.clone());
        latch
    }

    fn engine(&self, model: &str) -> Arc<MockEngine> {
        self.engines.lock().unwrap()[model].clone()
    }

    fn site(&self, model: &str) -> Site {
        let sites = self.sites.lock().unwrap();
        sites.iter().rev().find(|(id, _)| id == model).expect("the model was loaded").1
    }
}

struct Loads(Arc<Machine>);

struct Running {
    machine: Arc<Machine>,
    id: String,
    engine: Arc<MockEngine>,
    taken: Vec<(usize, u64)>,
    ram: u64,
}

impl Unit for Running {
    fn engine(&self) -> Arc<dyn Engine> {
        self.engine.clone()
    }

    fn ram(&self) -> u64 {
        self.ram
    }

    fn unload(self: Box<Self>) {
        let mut free = self.machine.free.lock().unwrap();
        for (device, units) in &self.taken {
            free[*device] += units;
        }
        self.machine.log.lock().unwrap().push(format!("unload {}", self.id));
    }
}

impl Loader for Loads {
    fn load(&self, preset: &Preset, site: &Site) -> Result<Box<dyn Unit>, LoadFailure> {
        let machine = &self.0;
        let id = preset.id.clone();
        machine.sites.lock().unwrap().push((id.clone(), *site));
        if machine.broken_files.lock().unwrap().contains(&id) {
            return Err(LoadFailure::Failed(format!("the file of {id} is not a model")));
        }
        // The plan: what the model takes of which device.
        let devices = preset.options.load.devices.clone().unwrap_or_else(|| vec![0]);
        let mut free = machine.free.lock().unwrap();
        let mut taken = Vec::new();
        for device in devices {
            let device = device as usize;
            let units = match preset.options.load.context {
                Some(context) => u64::from(context),
                None => free[device],
            };
            let least = if preset.options.load.context.is_some() { units } else { LEAST };
            if free[device] < least || (units == 0 && site.beside) {
                return Err(LoadFailure::NoRoom(format!(
                    "device {device} has {} free and {id} takes {least}",
                    free[device]
                )));
            }
            if units == 0 {
                return Err(LoadFailure::Failed(format!("device {device} has no room for {id}")));
            }
            taken.push((device, units));
        }
        for (device, units) in &taken {
            free[*device] -= units;
        }
        drop(free);
        machine.log.lock().unwrap().push(format!("load {id}"));
        // The weights take their time.
        let latch = machine.latches.lock().unwrap().remove(&id);
        if let Some(latch) = latch {
            latch.wait();
        }
        let script = machine.scripts.lock().unwrap().get(&id).cloned().unwrap_or_else(|| {
            vec![text(&format!("hello from {id}")), done(FinishReason::Stop, 3, 2)]
        });
        let card = ModelCard { id: id.clone(), context_length: 4096, created: 1_700_000_000 };
        let refusal = EngineError::Unsupported("no decisions here".into());
        let engine = MockEngine::with_models(
            vec![card],
            Plan::Script(script),
            DecisionPlan::Reject(refusal),
        );
        machine.engines.lock().unwrap().insert(id.clone(), engine.clone());
        let ram = site.ram.unwrap_or(0);
        Ok(Box::new(Running { machine: machine.clone(), id, engine, taken, ram }))
    }
}

/// A server over the presets of `ini` on a machine with `devices`.
struct World {
    machine: Arc<Machine>,
    clock: Arc<ManualClock>,
    models: Arc<Models>,
}

fn presets(ini: &str) -> Presets {
    Presets::parse(ini, Path::new("models.ini"), &[], &|_| true).expect("the presets are valid")
}

fn world(ini: &str, devices: &[u64]) -> World {
    let machine = Machine::new(devices);
    let clock = ManualClock::new();
    let loader = Arc::new(Loads(machine.clone()));
    let models = Arc::new(Models::start(presets(ini), loader, clock.clone()));
    World { machine, clock, models }
}

impl World {
    fn status(&self, model: &str) -> (String, Option<String>) {
        let listed = self.models.list();
        let status = listed.iter().find(|m| m.id == model).expect("listed").status.clone().unwrap();
        (status.status.to_owned(), status.reason)
    }

    fn is(&self, model: &str, status: &str) -> bool {
        self.status(model).0 == status
    }

    /// A request for `model` that runs next to the test.
    fn request(&self, model: &str) -> tokio::task::JoinHandle<Result<Lease, AcquireError>> {
        let (models, model) = (self.models.clone(), model.to_owned());
        tokio::spawn(async move { models.acquire(&model).await })
    }

    async fn until(&self, model: &str, status: &str) {
        eventually(&format!("{model} to be {status}"), || self.is(model, status)).await;
    }

    async fn until_logged(&self, line: &str) {
        eventually(line, || self.machine.log().iter().any(|l| l == line)).await;
    }
}

const THREE: &str = "\
[a]
model = /a.gguf
aliases = first, gpt-4o
context = 4
[b]
model = /b.gguf
context = 4
[c]
model = /c.gguf
context = 6
[d]
model = /d.gguf
devices = 1
";

// --- load on request ----------------------------------------------------------

#[tokio::test]
async fn a_request_loads_the_model_it_names_by_id_or_alias() {
    let w = world(THREE, &[10, 10]);
    assert!(w.is("a", "unloaded") && w.is("b", "unloaded"));
    assert!(w.machine.log().is_empty(), "nothing is loaded before a request");

    let lease = w.models.acquire("gpt-4o").await.unwrap();
    assert_eq!(lease.id(), Some("a"), "the engine is asked under the model's id");
    assert_eq!(lease.engine().models()[0].id, "a");
    assert!(w.is("a", "loaded") && w.is("b", "unloaded"));
    // A second request finds it loaded.
    let again = w.models.acquire("a").await.unwrap();
    assert_eq!(w.machine.log(), ["load a"]);
    drop((lease, again));

    let unknown = w.models.acquire("nope").await.unwrap_err();
    assert_eq!(
        unknown,
        AcquireError::NotFound {
            model: "nope".into(),
            available: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        }
    );
    // Asking whether a model is loaded loads nothing.
    let not_loaded = w.models.loaded("b").await.unwrap_err();
    assert_eq!(not_loaded, AcquireError::NotLoaded { model: "b".into() });
    assert_eq!(w.machine.log(), ["load a"]);
}

#[tokio::test]
async fn requests_for_a_model_that_is_loading_wait_for_the_one_load() {
    let w = world(THREE, &[10, 10]);
    let weights = w.machine.latch("a");
    let requests: Vec<_> = ["a", "first", "a"].iter().map(|name| w.request(name)).collect();
    w.until_logged("load a").await;
    assert_eq!(w.status("a"), ("loading".to_owned(), None));
    assert!(requests.iter().all(|request| !request.is_finished()));

    weights.open();
    for request in requests {
        assert_eq!(request.await.unwrap().unwrap().id(), Some("a"));
    }
    assert_eq!(w.machine.log(), ["load a"], "one load serves all three");
}

#[tokio::test]
async fn a_request_that_goes_away_while_it_waits_leaves_no_load_behind_it() {
    let w = world(THREE, &[10, 10]);
    // A load that has begun is finished, and the model is then idle.
    let weights = w.machine.latch("a");
    let gone = w.request("a");
    w.until_logged("load a").await;
    // b waits in line behind the load of a, and its only request goes away.
    let queued = w.request("b");
    eventually("b to be in line", || w.is("b", "loading")).await;
    queued.abort();
    gone.abort();
    let _ = (queued.await, gone.await);
    weights.open();
    w.until("a", "loaded").await;

    // The line is served in order: had b still been in it, it came before d.
    drop(w.models.acquire("d").await.unwrap());
    assert_eq!(w.machine.log(), ["load a", "load d"]);
    assert!(w.is("b", "unloaded"));
    // Nothing holds a: it gives way like any idle model.
    w.models.ask_unload("a").unwrap();
    w.until("a", "unloaded").await;
}

// --- making room ----------------------------------------------------------------

#[tokio::test]
async fn the_model_idle_the_longest_makes_room_and_no_other() {
    let w = world(THREE, &[10, 10]);
    drop(w.models.acquire("a").await.unwrap());
    w.clock.advance(1);
    drop(w.models.acquire("b").await.unwrap());
    w.clock.advance(1);
    // a is used again: b is now the one idle the longest.
    drop(w.models.acquire("a").await.unwrap());
    w.clock.advance(1);
    assert_eq!(w.machine.free(), [2, 10]);

    // c takes 6 of the device and 2 are free: one model has to go, and one is enough.
    drop(w.models.acquire("c").await.unwrap());
    assert_eq!(w.machine.log(), ["load a", "load b", "unload b", "load c"]);
    assert!(w.is("a", "loaded") && w.is("b", "unloaded") && w.is("c", "loaded"));
    assert_eq!(w.machine.free(), [0, 10]);

    // A model for the other device needs nobody's room: three are loaded,
    // because the devices hold three.
    drop(w.models.acquire("d").await.unwrap());
    assert_eq!(w.machine.log().len(), 5);
    assert!(w.is("a", "loaded") && w.is("c", "loaded") && w.is("d", "loaded"));
    assert_eq!(w.machine.free(), [0, 0], "without a context it took all the device had");
    // It was planned as a model that others are loaded next to.
    assert!(w.machine.site("d").beside && !w.machine.site("a").beside);
}

#[tokio::test]
async fn a_model_that_serves_requests_is_not_unloaded_under_them() {
    let ini = "[a]\nmodel = /a.gguf\ncontext = 8\n[b]\nmodel = /b.gguf\ncontext = 8\n";
    let w = world(ini, &[10]);
    let running = w.models.acquire("a").await.unwrap();

    // b needs the room of a, which is busy: b's request waits.
    let for_b = w.request("b");
    eventually("b to wait for room", || w.status("b").1.is_some()).await;
    let (status, reason) = w.status("b");
    assert_eq!(status, "loading");
    assert!(reason.unwrap().starts_with("waiting for room: device 0 has 2 free and b takes 8"));
    assert!(w.is("a", "loaded") && !for_b.is_finished());
    assert_eq!(w.machine.log(), ["load a"], "a is not unloaded under its request");
    // a gives way: a request that comes for it now waits in line behind b.
    let for_a = w.request("a");
    eventually("a's new request to be in line", || {
        w.models.list().iter().any(|m| m.id == "a") && !for_a.is_finished()
    })
    .await;
    assert_eq!(
        w.models.loaded("a").await.unwrap_err(),
        AcquireError::NotLoaded { model: "a".into() },
        "a model that gives way takes no new request"
    );

    // The running request ends: a is unloaded, b loaded, and b's request served.
    drop(running);
    let on_b = for_b.await.unwrap().unwrap();
    assert_eq!(on_b.id(), Some("b"));
    assert_eq!(w.machine.log(), ["load a", "unload a", "load b"]);
    assert!(!for_a.is_finished(), "b is busy now, and a waits for it in turn");
    drop(on_b);
    assert_eq!(for_a.await.unwrap().unwrap().id(), Some("a"));
    assert_eq!(w.machine.log(), ["load a", "unload a", "load b", "unload b", "load a"]);
}

#[tokio::test]
async fn a_busy_model_keeps_serving_when_the_request_that_wanted_its_room_goes_away() {
    let ini = "[a]\nmodel = /a.gguf\ncontext = 8\n[b]\nmodel = /b.gguf\ncontext = 8\n";
    let w = world(ini, &[10]);
    let running = w.models.acquire("a").await.unwrap();
    let for_b = w.request("b");
    eventually("b to wait for room", || w.status("b").1.is_some()).await;
    for_b.abort();
    let _ = for_b.await;
    // Nobody wants b any more: a takes requests again and is not unloaded.
    let again = w.models.acquire("a").await.unwrap();
    drop((running, again));
    assert!(w.is("a", "loaded") && w.is("b", "unloaded"));
    assert_eq!(w.machine.log(), ["load a"]);
}

#[tokio::test]
async fn two_loads_that_wait_for_one_busy_model_are_both_served_when_one_of_them_goes_away() {
    let ini = "[a]\nmodel = /a.gguf\ncontext = 8\n[b]\nmodel = /b.gguf\ncontext = 8\n\
               [c]\nmodel = /c.gguf\ncontext = 8\n";
    let w = world(ini, &[10]);
    let running = w.models.acquire("a").await.unwrap();
    let for_b = w.request("b");
    eventually("b to wait for room", || w.status("b").1.is_some()).await;
    let for_c = w.request("c");
    eventually("c to wait for room", || w.status("c").1.is_some()).await;
    // The request a gave way to goes away: the other still gets its room.
    for_b.abort();
    let _ = for_b.await;
    drop(running);
    assert_eq!(for_c.await.unwrap().unwrap().id(), Some("c"));
    assert_eq!(w.machine.log(), ["load a", "unload a", "load c"]);
}

#[tokio::test]
async fn the_ram_limit_makes_room_like_a_device_does() {
    let ini = "\
[*]
ram = 10
[a]
model = /a.gguf
context = 1
ram = 6
[b]
model = /b.gguf
context = 1
ram = 6
[c]
model = /c.gguf
context = 1
";
    let w = world(ini, &[10]);
    drop(w.models.acquire("a").await.unwrap());
    assert_eq!(w.machine.site("a"), Site { beside: false, ram: Some(6), ram_is_rest: false });
    // The device has room for both; the server's RAM has not.
    drop(w.models.acquire("b").await.unwrap());
    assert_eq!(w.machine.log(), ["load a", "unload a", "load b"]);
    // A model without a part of its own takes what the limit has left.
    drop(w.models.acquire("c").await.unwrap());
    assert_eq!(w.machine.site("c"), Site { beside: true, ram: Some(4), ram_is_rest: true });
    assert!(w.is("b", "loaded") && w.is("c", "loaded"));
}

/// Without a limit of the server there are no parts of a limit to fit into:
/// every model is loaded with its own limit or none, whatever is loaded
/// beside it, and none is unloaded for another's RAM. What they hold they
/// share by following the machine's memory (`ares_runtime::Machine`).
#[tokio::test]
async fn without_a_limit_of_the_server_no_model_makes_room_for_anothers_ram() {
    let ini = "\
[a]
model = /a.gguf
context = 1
[b]
model = /b.gguf
context = 1
ram = 6
[c]
model = /c.gguf
context = 1
";
    let w = world(ini, &[10]);
    for model in ["a", "b", "c"] {
        drop(w.models.acquire(model).await.unwrap());
    }
    assert_eq!(w.machine.site("a"), Site { beside: false, ram: None, ram_is_rest: false });
    assert_eq!(w.machine.site("b"), Site { beside: true, ram: Some(6), ram_is_rest: false });
    assert_eq!(w.machine.site("c"), Site { beside: true, ram: None, ram_is_rest: false });
    assert_eq!(w.machine.log(), ["load a", "load b", "load c"]);
}

// --- idle unload ----------------------------------------------------------------

#[tokio::test]
async fn a_model_is_unloaded_when_it_was_idle_for_its_time_and_loaded_again_on_request() {
    let ini = "\
[a]
model = /a.gguf
context = 4
idle-unload = 60
[b]
model = /b.gguf
context = 4
";
    let w = world(ini, &[10]);
    let lease = w.models.acquire("a").await.unwrap();
    drop(w.models.acquire("b").await.unwrap());
    // A request that runs longer than the idle time keeps its model.
    w.clock.advance(1000);
    assert!(w.is("a", "loaded"));
    drop(lease);
    // The time counts from the end of the last request.
    w.clock.advance(59);
    drop(w.models.acquire("b").await.unwrap());
    assert!(w.is("a", "loaded"), "one second before its time");
    assert_eq!(w.machine.free(), [2]);
    w.clock.advance(1);
    w.until("a", "unloaded").await;
    assert_eq!(w.machine.log(), ["load a", "load b", "unload a"]);
    assert_eq!(w.machine.free(), [6], "what it held is back");
    // b has no idle time and stays, however long.
    w.clock.advance(100_000);
    drop(w.models.acquire("a").await.unwrap());
    assert!(w.is("a", "loaded") && w.is("b", "loaded"));
    assert_eq!(w.machine.log(), ["load a", "load b", "unload a", "load a"]);
}

// --- failures -------------------------------------------------------------------

#[tokio::test]
async fn a_load_that_fails_is_told_to_everyone_who_waited_and_leaves_the_others_serving() {
    let w = world(THREE, &[10, 10]);
    drop(w.models.acquire("a").await.unwrap());
    w.machine.broken_files.lock().unwrap().insert("b".into());
    let first = w.request("b");
    let second = w.request("b");
    for request in [first, second] {
        assert_eq!(
            request.await.unwrap().unwrap_err(),
            AcquireError::Failed {
                model: "b".into(),
                reason: "the file of b is not a model".into()
            }
        );
    }
    assert_eq!(w.status("b"), ("failed".into(), Some("the file of b is not a model".into())));
    // The others are not touched, and still serve.
    assert!(w.is("a", "loaded"));
    drop(w.models.acquire("a").await.unwrap());
    assert_eq!(w.machine.log(), ["load a"]);

    // The next request tries again.
    w.machine.broken_files.lock().unwrap().clear();
    drop(w.models.acquire("b").await.unwrap());
    assert_eq!(w.status("b"), ("loaded".into(), None));
}

#[tokio::test]
async fn a_model_that_fits_nowhere_fails_with_the_plan_when_nothing_is_left_to_unload() {
    let ini = "[a]\nmodel = /a.gguf\ncontext = 4\n[huge]\nmodel = /h.gguf\ncontext = 99\n";
    let w = world(ini, &[10]);
    drop(w.models.acquire("a").await.unwrap());
    let error = w.models.acquire("huge").await.unwrap_err();
    let AcquireError::Failed { reason, .. } = error else { panic!("{error:?}") };
    assert_eq!(reason, "device 0 has 10 free and huge takes 99");
    assert_eq!(w.machine.log(), ["load a", "unload a"], "the idle model made way first");
}

#[tokio::test]
async fn an_engine_that_fails_takes_its_model_down_and_no_other() {
    let w = world(THREE, &[10, 10]);
    let on_a = w.models.acquire("a").await.unwrap();
    drop(w.models.acquire("d").await.unwrap());
    w.machine.engine("a").set_failure("device 0 failed: Xid 79");

    // It gets no new request, and is unloaded when the ones it failed have ended.
    let next = w.request("a");
    eventually("a's next request to wait", || !next.is_finished()).await;
    assert!(w.is("a", "loaded"));
    drop(on_a);
    let error = next.await.unwrap().unwrap_err();
    let AcquireError::Failed { model, reason } = error else { panic!("{error:?}") };
    assert_eq!(model, "a");
    assert_eq!(
        reason,
        "the model was taken down because its engine failed: device 0 failed: Xid 79"
    );
    assert_eq!(w.status("a"), ("failed".into(), Some(reason)));
    assert_eq!(w.machine.log(), ["load a", "load d", "unload a"]);
    // The model on the other device serves on.
    drop(w.models.acquire("d").await.unwrap());
    assert!(w.is("d", "loaded"));
    // A request loads the failed one again.
    drop(w.models.acquire("a").await.unwrap());
    assert_eq!(w.status("a"), ("loaded".into(), None));
}

// --- start and shutdown -----------------------------------------------------------

#[tokio::test]
async fn models_marked_for_it_are_loaded_at_start_and_one_model_alone_is_never_unloaded() {
    let ini = "\
[a]
model = /a.gguf
context = 4
[b]
model = /b.gguf
context = 4
load-on-startup = true
";
    let w = world(ini, &[10]);
    w.until("b", "loaded").await;
    assert_eq!(w.machine.log(), ["load b"]);

    // `ares --model`: one preset, loaded at start, its RAM the loader's default.
    let machine = Machine::new(&[10]);
    let clock = ManualClock::new();
    let mut options = ares_server::options::ModelOptions::default();
    ares_server::options::set(&mut options, "--model", "/models/only.gguf").unwrap();
    let models =
        Models::start(Presets::single(options), Arc::new(Loads(machine.clone())), clock.clone());
    models.wait_loaded("only").await.unwrap();
    assert_eq!(machine.site("only"), Site { beside: false, ram: None, ram_is_rest: false });
    clock.advance(1_000_000);
    drop(models.acquire("only").await.unwrap());
    assert_eq!(machine.log(), ["load only"]);
    // A load at start that fails is told to whoever waits for it.
    let machine = Machine::new(&[10]);
    machine.broken_files.lock().unwrap().insert("only".into());
    let mut options = ares_server::options::ModelOptions::default();
    ares_server::options::set(&mut options, "--model", "/models/only.gguf").unwrap();
    let models = Models::start(Presets::single(options), Arc::new(Loads(machine)), clock);
    assert!(matches!(models.wait_loaded("only").await, Err(AcquireError::Failed { .. })));
}

#[tokio::test]
async fn a_shutdown_unloads_everything_and_tells_the_requests_that_wait() {
    let w = world(THREE, &[10, 10]);
    drop(w.models.acquire("a").await.unwrap());
    drop(w.models.acquire("d").await.unwrap());
    // A load is under way, and a request waits for it.
    let weights = w.machine.latch("b");
    let waiting = w.request("b");
    w.until_logged("load b").await;

    let (models, stopped) = (w.models.clone(), Arc::new(Mutex::new(false)));
    let flag = stopped.clone();
    let shutdown = tokio::spawn(async move {
        models.shutdown().await;
        *flag.lock().unwrap() = true;
    });
    assert_eq!(waiting.await.unwrap().unwrap_err(), AcquireError::Stopping);
    assert!(!*stopped.lock().unwrap(), "the load that runs is finished first");
    weights.open();
    shutdown.await.unwrap();

    let mut log = w.machine.log();
    assert_eq!(log[..3], ["load a", "load d", "load b"]);
    log.sort();
    assert_eq!(log[3..], ["unload a", "unload b", "unload d"]);
    assert_eq!(w.machine.free(), [10, 10], "everything is given back");
    assert_eq!(w.models.acquire("a").await.unwrap_err(), AcquireError::Stopping);
}

// --- the routes ---------------------------------------------------------------------

#[tokio::test]
async fn the_routes_list_load_and_unload_models_and_a_request_loads_its_own() {
    let w = world(THREE, &[10, 10]);
    let app = router_over(w.models.clone(), Config::default());

    let (status, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(status, 200);
    let ids: Vec<_> = list["data"].as_array().unwrap().iter().map(|m| m["id"].clone()).collect();
    assert_eq!(ids, ["a", "b", "c", "d"]);
    assert_eq!(list["data"][0]["status"], serde_json::json!({ "value": "unloaded" }));
    assert_eq!(list["data"][0]["aliases"], serde_json::json!(["first", "gpt-4o"]));
    assert!(list["data"][0].get("context_length").is_none(), "not known before a load");

    // A chat request under an alias loads the model and is answered by it.
    let body = r#"{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}"#;
    let (status, answer) = json(post(&app, "/v1/chat/completions", body).await).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(answer["model"], "a");
    assert_eq!(answer["choices"][0]["message"]["content"], "hello from a");
    assert_eq!(w.machine.engine("a").submitted()[0].model, "a");
    let (_, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(list["data"][0]["status"]["value"], "loaded");
    assert_eq!(list["data"][0]["context_length"], 4096);
    assert_eq!(list["data"][1]["status"]["value"], "unloaded");

    // The endpoints answer at once; the state follows.
    let (status, answer) = json(post(&app, "/models/load", r#"{"model": "b"}"#).await).await;
    assert_eq!((status.as_u16(), &answer), (200, &serde_json::json!({ "success": true })));
    w.until("b", "loaded").await;
    let (status, _) = json(post(&app, "/models/unload", r#"{"model": "first"}"#).await).await;
    assert_eq!(status, 200);
    w.until("a", "unloaded").await;
    assert_eq!(w.machine.log(), ["load a", "load b", "unload a"]);

    // What cannot be done says why.
    let (status, error) = json(post(&app, "/models/load", r#"{"model": "x"}"#).await).await;
    assert_eq!(status, 404);
    assert_eq!(error["error"]["code"], "model_not_found");
    assert!(error["error"]["message"].as_str().unwrap().contains("available models: a, b, c, d"));
    let (status, error) = json(post(&app, "/models/unload", r#"{"name": "a"}"#).await).await;
    assert_eq!((status.as_u16(), error["error"]["param"].as_str()), (400, Some("name")));
    let body = r#"{"model": "x", "prompt": "hi"}"#;
    let (status, error) = json(post(&app, "/v1/completions", body).await).await;
    assert_eq!((status.as_u16(), error["error"]["code"].as_str()), (404, Some("model_not_found")));

    // A model whose load fails is a 503 for its requests and a state in the list.
    w.machine.broken_files.lock().unwrap().insert("c".into());
    let body = r#"{"model": "c", "prompt": "hi"}"#;
    let response = post(&app, "/v1/completions", body).await;
    assert!(response.headers().get("retry-after").is_none());
    let (status, error) = json(response).await;
    assert_eq!(status, 503);
    assert_eq!(error["error"]["code"], "model_load_failed");
    assert_eq!(
        error["error"]["message"],
        "the model 'c' could not be loaded: the file of c is not a model"
    );
    let (_, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(
        list["data"][2]["status"],
        serde_json::json!({ "value": "failed", "reason": "the file of c is not a model" })
    );

    // The statistics are a loaded model's.
    let (status, error) = json(get(&app, "/metrics").await).await;
    assert_eq!((status.as_u16(), error["error"]["param"].as_str()), (400, Some("model")));
    let (status, error) = json(get(&app, "/metrics?model=a").await).await;
    assert_eq!((status.as_u16(), error["error"]["code"].as_str()), (503, Some("model_not_loaded")));
}

#[tokio::test]
async fn a_request_waits_for_its_load_and_a_client_that_leaves_gives_the_wait_up() {
    let w = world(THREE, &[10, 10]);
    let app = router_over(w.models.clone(), Config::default());
    let weights = w.machine.latch("a");
    let body = r#"{"model": "a", "prompt": "hi", "stream": true}"#;
    let waiting = tokio::spawn({
        let app = app.clone();
        async move { post(&app, "/v1/completions", body).await }
    });
    w.until_logged("load a").await;
    assert!(!waiting.is_finished(), "no answer before the model is loaded");
    // A second client leaves while it waits.
    let leaving = tokio::spawn({
        let app = app.clone();
        async move { post(&app, "/v1/completions", body).await }
    });
    eventually("the second request to wait", || !leaving.is_finished()).await;
    leaving.abort();
    let _ = leaving.await;

    weights.open();
    let response = waiting.await.unwrap();
    assert_eq!(response.status(), 200);
    let stream = common::parse_sse(&common::body_text(response).await);
    assert!(stream.done);
    assert_eq!(stream.chunks[0]["choices"][0]["text"], "hello from a");
    assert_eq!(w.machine.engine("a").submitted().len(), 1, "the one who left asked nothing");
}

#[tokio::test]
async fn the_event_stream_says_every_state_and_every_change() {
    let w =
        world("[a]\nmodel = /a.gguf\ncontext = 4\nidle-unload = 5\n[b]\nmodel = /b.gguf\n", &[10]);
    let app = router_over(w.models.clone(), Config::default());
    w.machine.broken_files.lock().unwrap().insert("b".into());
    let response = get(&app, "/models/sse").await;
    assert_eq!(response.status(), 200);
    assert!(response.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let mut body = response.into_body().into_data_stream();

    drop(w.models.acquire("a").await.unwrap());
    assert!(w.models.acquire("b").await.is_err());
    w.clock.advance(5);
    w.until("a", "unloaded").await;

    let expected = [
        ("a", "unloaded", None),
        ("b", "unloaded", None),
        ("a", "loading", None),
        ("a", "loaded", None),
        ("b", "loading", None),
        ("b", "failed", Some("the file of b is not a model")),
        ("a", "unloading", Some("idle for 5 s")),
        ("a", "unloaded", None),
    ];
    let mut seen = String::new();
    let limit = tokio::time::Instant::now() + Duration::from_secs(20);
    while common::sse_payloads(&seen).len() < expected.len() {
        let chunk = tokio::time::timeout_at(limit, body.next()).await;
        let chunk = chunk.expect("the events came").expect("the stream is open").unwrap();
        seen.push_str(std::str::from_utf8(&chunk).unwrap());
    }
    let events: Vec<serde_json::Value> =
        common::sse_payloads(&seen).iter().map(|p| serde_json::from_str(p).unwrap()).collect();
    for (event, (model, status, reason)) in events.iter().zip(expected) {
        let mut data = serde_json::json!({ "status": status });
        if let Some(reason) = reason {
            data["reason"] = reason.into();
        }
        let wanted = serde_json::json!({ "model": model, "event": "model_status", "data": data });
        assert_eq!(*event, wanted, "{events:?}");
    }
}

#[tokio::test]
async fn a_server_over_one_engine_is_as_it_was() {
    // No presets: the engine decides which names it answers to, the list is
    // its own, and nothing can be loaded or unloaded.
    let engine = MockEngine::script(vec![text("hi"), done(FinishReason::Stop, 1, 1)]);
    let app = common::app(&engine);
    let (_, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(list["data"][0]["id"], "mock");
    assert!(list["data"][0].get("status").is_none());
    let (status, _) = json(post(&app, "/models/unload", r#"{"model": "mock"}"#).await).await;
    assert_eq!(status, 503);
    let (status, _) = json(get(&app, "/models/sse").await).await;
    assert_eq!(status, 501);
}

/// A small model file with a tokenizer and a chat template, and no tensor:
/// enough for what `/props` reads of a model that is not loaded.
fn tiny_model_file(dir: &Path) -> std::path::PathBuf {
    let char_of = |byte: u8| {
        if matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF) {
            return char::from(byte);
        }
        let moved =
            (0..byte).filter(|b| !matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)).count();
        char::from_u32(0x100 + moved as u32).expect("a character")
    };
    let mut tokens: Vec<String> = (0..=255u8).map(|b| char_of(b).to_string()).collect();
    tokens.extend(["<|im_start|>".to_owned(), "<|im_end|>".to_owned()]);
    let mut types = vec![1i32; 256];
    types.extend([3, 3]);
    let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
                    {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    let file = ares_gguf::GgufBuilder::new()
        .str("general.architecture", "tiny")
        .u32("tiny.context_length", 8192)
        .str("tokenizer.ggml.model", "gpt2")
        .str("tokenizer.ggml.pre", "qwen35")
        .str_array("tokenizer.ggml.tokens", &tokens)
        .i32_array("tokenizer.ggml.token_type", &types)
        .str_array::<&str>("tokenizer.ggml.merges", &[])
        .u32("tokenizer.ggml.eos_token_id", 257)
        .str("tokenizer.chat_template", template)
        .build();
    let path = dir.join("tiny.gguf");
    std::fs::write(&path, file).unwrap();
    path
}

#[tokio::test]
async fn props_of_a_model_that_is_not_loaded_are_read_from_its_preset_and_file() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("props-unloaded");
    std::fs::create_dir_all(&dir).unwrap();
    let path = tiny_model_file(&dir);
    let ini = format!(
        "[t]\nmodel = {}\ntemperature = 0.5\n[u]\nmodel = {}\ncontext = 2048\n",
        path.display(),
        path.display()
    );
    let w = world(&ini, &[10, 10]);
    let app = router_over(w.models.clone(), Config::default());
    let (status, props) = json(get(&app, "/props?model=t").await).await;
    assert_eq!(status, 200, "{props}");
    assert_eq!(props["model"], "t");
    assert_eq!(props["loaded"], false);
    assert_eq!(props["context_length"], 8192, "the file's own");
    assert_eq!(props["default_generation_settings"]["params"]["temperature"], 0.5);
    assert!(props["chat_template"].as_str().unwrap().contains("<|im_start|>"), "{props}");
    assert_eq!(props["chat_template_caps"]["supports_tools"], false);
    // A context the preset names is what it says.
    let (_, props) = json(get(&app, "/props?model=u").await).await;
    assert_eq!(props["context_length"], 2048);
    // Nothing was loaded for it.
    assert!(w.is("t", "unloaded") && w.is("u", "unloaded"));
    // A name no preset has is not found.
    let (status, _) = json(get(&app, "/props?model=nope").await).await;
    assert_eq!(status, 404);
}
