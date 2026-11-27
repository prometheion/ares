//! The usage profile of the experts of a model the runtime drives, and the
//! fill of the tiers from it.
//!
//! The store's heat (a count of every expert's picks in which recent ones
//! weigh more) is what the tiers are decided from while the model runs, and
//! it is lost when the model goes. Here it outlives the model: it is written
//! to a profile ([`ExpertProfile`]) while the model serves, once a minute
//! when steps ran, and when the model is dropped (at an unload and a clean
//! stop); a model that is opened again starts from it.
//!
//! - **When the engine begins to serve** ([`Usage::serving`], where the
//!   server reads its warm-up text when there is no profile, after the
//!   runtime measured what a step costs): the heat becomes the profile's (the
//!   measurement's steps route a made-up text; their heat was forgotten when
//!   they ended, [`Model::measured`](ares_runtime::Model::measured)) and the tiers
//!   are filled from it in one sweep through the file, VRAM with the hottest
//!   experts of every layer and the RAM tier with the next, into what is
//!   empty or worth nothing. A thread keeps the profile from then on.
//! - **While nothing runs** (no step for [`IDLE_AFTER`]): that thread fills
//!   what is empty or worth nothing, in VRAM and in the RAM tier, from the
//!   heat, the hottest first, a few experts at a time as prefetches. A step that begins
//!   stops it at once: the store does nothing of a fill while a step runs,
//!   and what is on its way then (two experts from the file, an upload per
//!   device) takes its slot on its own.
//!
//! The profile is per layer of the model and expert, whatever the placement:
//! when both devices share every layer, an expert's heat is the larger of
//! the two halves of the store's layer, and it is seeded where the expert is
//! owned. The MTP block's experts are not in it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ares_gguf::Gguf;
use ares_runtime::expert_store::{
    CudaSlots, ExpertProfile, ExpertStore, FillOrder, FillPace, FillStats, ProfileKey, StoreError,
    TierFill,
};

use super::feed::Layout;
use super::flow::Flow;
use super::placement::Placement;
use crate::qwen4exp::Config;

/// How long no step must have run before the empty slots are filled.
pub const IDLE_AFTER: Duration = Duration::from_secs(1);

/// How often the keeper looks while it fills, and while it does not.
const FILL_TICK: Duration = Duration::from_millis(5);
const IDLE_TICK: Duration = Duration::from_millis(200);

/// How often the profile is written while steps run.
const SAVE_EVERY: Duration = Duration::from_secs(60);

/// How long an idle fill may wait for room before it is given up until the
/// tiers change.
const STUCK: Duration = Duration::from_secs(2);

/// How long the fill at the start may stand still before it gives up the
/// rest: a drive that does not answer must not keep the model from serving.
pub(super) const START_STALL: Duration = Duration::from_secs(60);

/// What the start found of the profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileFound {
    /// There was none.
    None,
    /// There was one and it was not taken, for this reason.
    Ignored(String),
    /// It was taken; this many experts had heat in it.
    Found { known: usize },
}

/// What the start made of the usage profile.
#[derive(Clone, Debug)]
pub struct ProfileReport {
    /// Where the profile is read from and written to.
    pub path: PathBuf,
    pub found: ProfileFound,
    /// What the fill from the profile brought, and how long it took.
    pub filled: Option<(FillStats, Duration)>,
}

/// What the idle fill did since the engine started, and what the tiers of
/// the model's layers hold.
#[derive(Clone, Copy, Debug, Default)]
pub struct UsageStats {
    /// What the fills while nothing ran brought, summed.
    pub idle: FillStats,
    /// Fills that ran to their end, and fills that a step stopped.
    pub idle_done: u64,
    pub idle_stopped: u64,
    /// Experts in VRAM slots and the slots, and the same for the RAM tier.
    pub vram: (usize, usize),
    pub ram: (usize, usize),
    /// Profiles written.
    pub saved: u64,
}

/// Which layers of the store hold the experts of each layer of the model,
/// and which experts each of them owns.
#[derive(Clone, Debug)]
struct LayerMap {
    key: ProfileKey,
    /// Per layer of the model: its layers of the store, each with the experts
    /// it owns (`None`: all of them).
    layers: Vec<Vec<(usize, Option<Vec<bool>>)>>,
}

impl LayerMap {
    fn new(key: ProfileKey, layout: &Layout, placement: &Placement) -> Self {
        let experts = key.experts as usize;
        let layers = (0..key.layers as usize)
            .map(|layer| {
                (0..layout.per_layer)
                    .map(|part| {
                        let owned = (layout.per_layer > 1).then(|| {
                            (0..experts as u32).map(|e| placement.owner(layer, e) == part).collect()
                        });
                        (layout.store_layer(part, layer), owned)
                    })
                    .collect()
            })
            .collect();
        Self { key, layers }
    }

    /// Every layer of the store that holds experts of the model's layers.
    fn store_layers(&self) -> Vec<usize> {
        self.layers.iter().flatten().map(|(at, _)| *at).collect()
    }

    /// Sets the store's heat of the model's layers to `profile`'s.
    fn seed(&self, store: &mut ExpertStore<CudaSlots>, profile: &ExpertProfile) {
        for (layer, parts) in self.layers.iter().enumerate() {
            let heat = profile.layer(layer);
            for (at, owned) in parts {
                let values: Vec<f32> = match owned {
                    None => heat.to_vec(),
                    Some(owned) => {
                        heat.iter().zip(owned).map(|(h, o)| if *o { *h } else { 0.0 }).collect()
                    }
                };
                store.set_heat(*at, &values);
            }
        }
    }

    /// The store's heat of the model's layers as a profile.
    fn profile_of(&self, store: &ExpertStore<CudaSlots>) -> ExpertProfile {
        let mut profile = ExpertProfile::new(self.key);
        for (layer, parts) in self.layers.iter().enumerate() {
            let out = profile.layer_mut(layer);
            for (at, _) in parts {
                for (out, heat) in out.iter_mut().zip(store.heat().layer(*at)) {
                    *out = out.max(*heat);
                }
            }
        }
        profile
    }

    /// Experts in the slots of the model's layers, and the slots, in VRAM and
    /// in the RAM tier.
    fn occupancy(&self, store: &ExpertStore<CudaSlots>) -> ((usize, usize), (usize, usize)) {
        self.store_layers().into_iter().fold(((0, 0), (0, 0)), |(vram, ram), at| {
            let (held_vram, held_ram) = store.held(at);
            let (slots_vram, slots_ram) = store.slots(at);
            ((vram.0 + held_vram, vram.1 + slots_vram), (ram.0 + held_ram, ram.1 + slots_ram))
        })
    }
}

/// Writes log lines for the model's owner (the server logs them).
pub type Log = Arc<dyn Fn(&str) + Send + Sync>;

/// The usage profile of a model: nothing for a model that keeps none.
#[derive(Default)]
pub(super) struct Usage(Option<Arc<Inner>>);

impl std::fmt::Debug for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Usage").field("dir", &self.0.as_ref().map(|i| &i.dir)).finish()
    }
}

struct Inner {
    dir: PathBuf,
    key: ProfileKey,
    log: Log,
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    found: Option<ProfileFound>,
    /// The profile, until the engine serves and the store is seeded from it.
    profile: Option<ExpertProfile>,
    filled: Option<(FillStats, Duration)>,
    map: Option<Arc<LayerMap>>,
    flow: Option<Arc<Flow>>,
    serving: bool,
    /// The store's clock when the profile was last written (or when the
    /// engine began to serve): a profile is written only when steps ran.
    saved: Option<u64>,
    /// The last error a write gave, which is logged once.
    error: Option<String>,
    stats: UsageStats,
    stop: bool,
    keeper: Option<JoinHandle<()>>,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes the store's heat as the profile, when steps ran since it was
    /// last written. Returns the path, or `None` when there was nothing new.
    fn save(&self, state: &mut State) -> Option<Result<PathBuf, String>> {
        let (flow, map) = (state.flow.as_ref()?, state.map.as_ref()?);
        if !state.serving {
            return None;
        }
        let (profile, clock) = {
            let feed = flow.lock_feed();
            if state.saved == Some(feed.store.clock()) {
                return None;
            }
            (map.profile_of(&feed.store), feed.store.clock())
        };
        let written = profile.write(&self.dir).map_err(|e| e.to_string());
        match &written {
            Ok(_) => {
                state.saved = Some(clock);
                state.stats.saved += 1;
                state.error = None;
            }
            Err(error) if state.error.as_ref() != Some(error) => {
                (self.log)(&format!(
                    "expert profile {}: cannot be written: {error}",
                    self.key.path(&self.dir).display()
                ));
                state.error = Some(error.clone());
            }
            Err(_) => {}
        }
        Some(written)
    }
}

impl Usage {
    /// The profile of the model in `gguf` kept in `dir`: read now, so that
    /// the start can fill the tiers from it. `log` gets the lines the owner
    /// should know of (a profile that cannot be written).
    pub(super) fn open(dir: &Path, gguf: &Gguf, c: &Config, log: Log) -> Self {
        let key = profile_key(gguf, c);
        let (found, profile) = match ExpertProfile::read(dir, &key) {
            Ok(Some(profile)) => (ProfileFound::Found { known: profile.known() }, Some(profile)),
            Ok(None) => (ProfileFound::None, None),
            Err(why) => (ProfileFound::Ignored(why.to_string()), None),
        };
        let state = State { found: Some(found), profile, ..State::default() };
        Self(Some(Arc::new(Inner {
            dir: dir.to_owned(),
            key,
            log,
            state: Mutex::new(state),
            wake: Condvar::new(),
        })))
    }

    /// The runtime's flow, which holds the store, once it is made, and where
    /// the experts of every layer of the model are in it.
    pub(super) fn attach(&self, flow: &Arc<Flow>, layout: &Layout, placement: &Placement) {
        if let Some(inner) = &self.0 {
            let mut state = inner.lock();
            state.map = Some(Arc::new(LayerMap::new(inner.key, layout, placement)));
            state.flow = Some(flow.clone());
        }
    }

    /// The engine serves from now on. The heat becomes the profile's (or none:
    /// the steps that measured the price of a step route a made-up text), the
    /// tiers are filled from it, and the profile is kept from now on. Returns
    /// what the start made of the profile; `None` for a model that keeps none
    /// or before the parts are open. Call it before the first request.
    pub(super) fn serving(&self) -> Option<ProfileReport> {
        let inner = self.0.as_ref()?;
        let mut state = inner.lock();
        let (flow, map) = (state.flow.clone()?, state.map.clone()?);
        if !state.serving {
            state.serving = true;
            let profile = state.profile.take().unwrap_or_else(|| ExpertProfile::new(inner.key));
            map.seed(&mut flow.lock_feed().store, &profile);
            if profile.known() > 0 {
                state.filled = Some(fill_from(&flow, &map, &inner.log));
            }
            state.saved = Some(flow.lock_feed().store.clock());
            let keeper = inner.clone();
            let spawned = std::thread::Builder::new()
                .name("ares-expert-usage".into())
                .spawn(move || keep(&keeper));
            match spawned {
                Ok(handle) => state.keeper = Some(handle),
                Err(error) => (inner.log)(&format!(
                    "expert profile: no thread to keep it ({error}); it is written at the end only"
                )),
            }
        }
        Some(ProfileReport {
            path: inner.key.path(&inner.dir),
            found: state.found.clone().unwrap_or(ProfileFound::None),
            filled: state.filled,
        })
    }

    /// Fills the tiers from the store's heat as the start does from a
    /// profile ([`fill_from`]); `None` before the engine serves.
    pub(super) fn fill_from_heat(&self) -> Option<(FillStats, Duration)> {
        let inner = self.0.as_ref()?;
        let (flow, map) = {
            let state = inner.lock();
            if !state.serving {
                return None;
            }
            (state.flow.clone()?, state.map.clone()?)
        };
        Some(fill_from(&flow, &map, &inner.log))
    }

    /// Writes the profile now, if steps ran since it was last written.
    pub(super) fn save(&self) -> Option<Result<PathBuf, String>> {
        let inner = self.0.as_ref()?;
        let mut state = inner.lock();
        inner.save(&mut state)
    }

    /// What the idle fill did and what the tiers hold.
    pub(super) fn stats(&self) -> Option<UsageStats> {
        let inner = self.0.as_ref()?;
        let state = inner.lock();
        let mut stats = state.stats;
        if let (Some(flow), Some(map)) = (&state.flow, &state.map) {
            (stats.vram, stats.ram) = map.occupancy(&flow.lock_feed().store);
        }
        Some(stats)
    }
}

/// The key of the usage profile of the model in `gguf`: the fingerprint of
/// its files and the shape of its experts. The table of similar experts
/// (`ExpertNeighbours`) is kept under the same key.
pub fn profile_key(gguf: &Gguf, c: &Config) -> ProfileKey {
    let mut identity = b"qwen4exp\0".to_vec();
    for part in 0..gguf.n_parts() {
        crate::files::identity(gguf.path(part as u16), &mut identity);
    }
    ProfileKey::new(&identity, c.n_layer as u32, c.experts.count as u32)
}

/// Seeds the heat of `store` with `profile` and fills the tiers from it in one
/// sweep through the file, as a start of the runtime does
/// ([`Usage::serving`]), for a store that one thread drives (`Session`).
pub(super) fn seed_and_fill(
    store: &mut ExpertStore<CudaSlots>,
    layout: &Layout,
    placement: &Placement,
    profile: &ExpertProfile,
) -> Result<(FillStats, Duration), StoreError> {
    let map = LayerMap::new(*profile.key(), layout, placement);
    map.seed(store, profile);
    let begun = Instant::now();
    let mut fill = store.plan_fill(&map.store_layers(), FillOrder::File);
    let mut seen = (usize::MAX, Instant::now());
    loop {
        store.advance_fill(&mut fill, FillPace::START)?;
        if fill.is_done() && store.is_quiet() {
            break;
        }
        let left = fill.left() + usize::from(!store.is_quiet());
        if left != seen.0 {
            seen = (left, Instant::now());
        } else if seen.1.elapsed() > START_STALL {
            break;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    store.take_stats();
    Ok((fill.stats(), begun.elapsed()))
}

/// Fills the tiers of the model's layers from the heat, in one sweep through
/// the file at the pace of a start, VRAM's free slots first and the RAM tier
/// behind them, while no request runs. The store's lock is let go between
/// the turns, so that the watch of the machine's memory can make the tier
/// give memory back meanwhile. Returns what was brought and how long it took.
fn fill_from(flow: &Flow, map: &LayerMap, log: &Log) -> (FillStats, Duration) {
    let begun = Instant::now();
    let mut fill = flow.lock_feed().store.plan_fill(&map.store_layers(), FillOrder::File);
    let mut seen = (usize::MAX, Instant::now());
    loop {
        let left = {
            let mut feed = flow.lock_feed();
            if let Err(error) = feed.store.advance_fill(&mut fill, FillPace::START) {
                log(&format!("expert profile: the fill of the tiers stopped: {error}"));
                break;
            }
            if fill.is_done() && feed.store.is_quiet() {
                break;
            }
            fill.left() + usize::from(!feed.store.is_quiet())
        };
        if left != seen.0 {
            seen = (left, Instant::now());
        } else if seen.1.elapsed() > START_STALL {
            log(&format!("expert profile: the fill of the tiers stood still; {left} experts left"));
            break;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    // What the fill read and uploaded is not what steps missed.
    flow.lock_feed().store.take_stats();
    (fill.stats(), begun.elapsed())
}

impl Drop for Usage {
    /// The keeper stops, and the profile is written a last time.
    fn drop(&mut self) {
        let Some(inner) = &self.0 else { return };
        let keeper = {
            let mut state = inner.lock();
            state.stop = true;
            inner.wake.notify_all();
            state.keeper.take()
        };
        if let Some(keeper) = keeper {
            let _ = keeper.join();
        }
        let mut state = inner.lock();
        inner.save(&mut state);
        state.flow = None;
    }
}

/// The keeper's thread: fills the empty slots while nothing runs and writes
/// the profile while steps run, until the model goes.
fn keep(inner: &Inner) {
    let mut fill: Option<TierFill> = None;
    // The store's clock when it was last seen to move, and since when it
    // stands.
    let mut seen = (u64::MAX, Instant::now());
    // What the tiers were when a fill was last planned: the clock and the
    // bytes the RAM tier held. Nothing is planned again before one changes.
    let mut planned: Option<(u64, u64)> = None;
    // What the fill had left when it last moved, and when: a fill that waits
    // for room that does not come (a tier that gives memory back) is given up
    // until the tiers change.
    let mut moved = (usize::MAX, Instant::now());
    let mut last_save = Instant::now();
    loop {
        let mut state = inner.lock();
        let tick = if fill.is_some() { FILL_TICK } else { IDLE_TICK };
        if !state.stop {
            state = inner.wake.wait_timeout(state, tick).unwrap_or_else(|e| e.into_inner()).0;
        }
        if state.stop {
            return;
        }
        let (Some(flow), Some(map)) = (state.flow.clone(), state.map.clone()) else { return };
        let mut feed = flow.lock_feed();
        let store = &mut feed.store;
        let clock = store.clock();
        if clock != seen.0 || store.in_step() {
            if fill.take().is_some_and(|f| !f.is_done()) {
                state.stats.idle_stopped += 1;
            }
            seen = (clock, Instant::now());
        }
        if seen.1.elapsed() >= IDLE_AFTER {
            let tiers = (clock, store.ram_bytes().0);
            if fill.is_none() && planned != Some(tiers) {
                planned = Some(tiers);
                let plan = store.plan_fill(&map.store_layers(), FillOrder::Heat);
                fill = (!plan.is_done()).then_some(plan);
                moved = (usize::MAX, Instant::now());
            }
            if let Some(f) = &mut fill {
                let before = f.stats();
                let advanced = store.advance_fill(f, FillPace::IDLE);
                let after = f.stats();
                let idle = &mut state.stats.idle;
                idle.vram += after.vram - before.vram;
                idle.ram += after.ram - before.ram;
                idle.reads += after.reads - before.reads;
                idle.read_bytes += after.read_bytes - before.read_bytes;
                idle.dropped += after.dropped - before.dropped;
                let left = f.left() + usize::from(!store.is_quiet());
                if left != moved.0 {
                    moved = (left, Instant::now());
                }
                match advanced {
                    Err(error) => {
                        (inner.log)(&format!("expert profile: the idle fill stopped: {error}"));
                        fill = None;
                    }
                    Ok(()) if f.is_done() => {
                        state.stats.idle_done += 1;
                        fill = None;
                        planned = Some((clock, store.ram_bytes().0));
                    }
                    Ok(()) if moved.1.elapsed() > STUCK => {
                        state.stats.idle_stopped += 1;
                        fill = None;
                        planned = Some((clock, store.ram_bytes().0));
                    }
                    Ok(()) => {}
                }
            }
        }
        drop(feed);
        if last_save.elapsed() >= SAVE_EVERY {
            last_save = Instant::now();
            inner.save(&mut state);
        }
    }
}
