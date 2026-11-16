//! Tests of filling the tiers from the heat (`expert_store/fill.rs`) on a
//! small model file and slots in ordinary memory (`HostSlots`), whose copies
//! run only when the test says so, as in `tests/expert_store.rs`.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Config, ExpertMap, Reader};
use ares_runtime::expert_store::{
    ExpertStore, FillOrder, FillPace, FillTier, HostRegion, HostSlots, LayerSpec, StoreConfig,
    StorePlan, TierFill,
};

const LAYERS: usize = 3;
const EXPERTS: u32 = 16;
const LENS: [usize; 3] = [128, 128, 96];
const OFFSETS: [u64; 3] = [0, 128, 256];
const SLOT_BYTES: u64 = 352;

/// A scratch directory that goes away even when the test fails.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ares-expert-fill-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn byte(layer: usize, part: usize, expert: u32, i: usize) -> u8 {
    (layer * 131 + part * 71 + expert as usize * 29 + i * 7 + 3) as u8
}

fn expert_bytes(layer: usize, expert: u32) -> Vec<u8> {
    let mut slot = vec![0u8; SLOT_BYTES as usize];
    for part in 0..3 {
        for i in 0..LENS[part] {
            slot[OFFSETS[part] as usize + i] = byte(layer, part, expert, i);
        }
    }
    slot
}

/// A model of `LAYERS` layers of `EXPERTS` experts with F32 matrices, the
/// layers one after the other in the file.
fn model(dir: &Scratch) -> Gguf {
    let mut builder = GgufBuilder::new().str("general.architecture", "test");
    for layer in 0..LAYERS {
        for (part, (name, dims)) in [
            ("ffn_gate_exps", [8u64, 4, u64::from(EXPERTS)]),
            ("ffn_up_exps", [8, 4, u64::from(EXPERTS)]),
            ("ffn_down_exps", [4, 6, u64::from(EXPERTS)]),
        ]
        .into_iter()
        .enumerate()
        {
            let data: Vec<u8> = (0..EXPERTS)
                .flat_map(|e| (0..LENS[part]).map(move |i| byte(layer, part, e, i)))
                .collect();
            builder =
                builder.tensor(&format!("blk.{layer}.{name}.weight"), &dims, GgmlType::F32, data);
        }
    }
    let path = dir.0.join("model.gguf");
    builder.write(&path).unwrap();
    Gguf::open(&path).unwrap()
}

struct Rig {
    store: ExpertStore<HostSlots>,
    _dir: Scratch,
}

/// A store with `vram_slots` slots per layer in VRAM (one of them kept free
/// between steps) and `ram_slots` in RAM, the RAM tier in pieces of
/// `pieces.0` slots of which `pieces.1` are locked at the start.
fn rig(vram_slots: u64, ram_slots: u64, pieces: Option<(usize, usize)>) -> Rig {
    let dir = Scratch::new();
    let gguf = model(&dir);
    let map =
        ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
    let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
    let layers = [LayerSpec { device: 0, offsets: OFFSETS, slot_bytes: SLOT_BYTES }; LAYERS];
    let mut config =
        StoreConfig::new(1, 2, vec![vram_slots * SLOT_BYTES * LAYERS as u64], u64::MAX / 4);
    config.vram_reserve = 1;
    config.rebalance_every = 0;
    let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
    let ram_slot = probe.ram_slot_bytes(0) as usize;
    config.ram_bytes = ram_slots * LAYERS as u64 * ram_slot as u64;
    if let Some((per_piece, _)) = pieces {
        config.ram_piece = (per_piece * ram_slot) as u64;
    }
    let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
    assert_eq!(plan.slots(0), (vram_slots as usize, ram_slots as usize));
    let device = HostSlots::new(&plan.device_pools(0), LAYERS, EXPERTS);
    let mut region = HostRegion::unpinned(plan.ram_bytes(), plan.block());
    if let Some((per_piece, start)) = pieces {
        region = region.starting_with(start * (per_piece * ram_slot).next_multiple_of(4096));
    }
    let store = ExpertStore::new(plan, map, reader, vec![device], region, None).unwrap();
    Rig { store, _dir: dir }
}

impl Rig {
    /// Gives `layer` the heat `heat[e]` for the experts named, none for the
    /// others.
    fn heat(&mut self, layer: usize, heat: &[(u32, f32)]) {
        let mut values = vec![0.0; EXPERTS as usize];
        for (expert, h) in heat {
            values[*expert as usize] = *h;
        }
        self.store.set_heat(layer, &values);
    }

    /// Lets reads finish and the device run until nothing is on its way.
    fn settle(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            self.store.poll().unwrap();
            self.store.device_mut(0).run(usize::MAX);
            self.store.poll().unwrap();
            if self.store.is_quiet() {
                return;
            }
            assert!(Instant::now() < deadline, "the store does not settle: {:?}", self.store);
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Runs a fill to its end.
    fn fill(&mut self, fill: &mut TierFill, pace: FillPace) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !(fill.is_done() && self.store.is_quiet()) {
            self.store.advance_fill(fill, pace).unwrap();
            self.store.device_mut(0).run(usize::MAX);
            assert!(Instant::now() < deadline, "the fill does not end: {fill:?}");
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// The device has the expert's table entry and its bytes in that slot.
    fn assert_usable(&self, layer: usize, expert: u32) {
        let entry = self.store.device(0).entry(layer, expert);
        assert!(entry >= 0, "layer {layer} expert {expert} has no entry");
        let (_, pool) = self.store.place(layer);
        assert_eq!(
            self.store.device(0).slot(pool, entry as u32),
            expert_bytes(layer, expert),
            "layer {layer} expert {expert} in slot {entry}"
        );
    }

    /// What VRAM and RAM hold of `layer`.
    fn held(&self, layer: usize) -> (BTreeSet<u32>, BTreeSet<u32>) {
        let vram = (0..EXPERTS).filter(|e| self.store.in_vram(layer, *e)).collect();
        let ram = (0..EXPERTS).filter(|e| self.store.in_ram(layer, *e)).collect();
        (vram, ram)
    }

    /// Every (layer, expert) that a tier holds.
    fn everything(&self) -> BTreeSet<(usize, u32)> {
        (0..LAYERS)
            .flat_map(|l| (0..EXPERTS).map(move |e| (l, e)))
            .filter(|(l, e)| self.store.in_vram(*l, *e) || self.store.in_ram(*l, *e))
            .collect()
    }
}

fn set(experts: &[u32]) -> BTreeSet<u32> {
    experts.iter().copied().collect()
}

const ALL: [usize; LAYERS] = [0, 1, 2];

#[test]
fn a_fill_brings_the_hottest_into_vram_and_the_next_into_ram() {
    // Four VRAM slots a layer, one of them kept free; four RAM slots.
    let mut rig = rig(4, 4, None);
    rig.heat(
        0,
        &[(5, 9.0), (7, 8.0), (2, 7.0), (11, 6.0), (0, 5.0), (14, 4.0), (3, 3.0), (9, 2.5)],
    );
    rig.heat(1, &[(1, 1.0), (2, 0.5)]);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::File);
    rig.fill(&mut fill, FillPace::START);

    let (vram, ram) = rig.held(0);
    assert_eq!(vram, set(&[2, 5, 7]), "the three hottest take the free slots but one");
    // RAM keeps the next four: the copies of those VRAM took on their way up
    // are worth nothing there and gave way.
    let outside: BTreeSet<u32> = ram.difference(&vram).copied().collect();
    assert_eq!(outside, set(&[0, 3, 11, 14]));
    assert!(!rig.store.in_ram(0, 9) && !rig.store.in_vram(0, 9), "no room was left for it");
    for expert in [2, 5, 7] {
        rig.assert_usable(0, expert);
    }
    // A layer with two experts of any heat brings two, one without brings
    // nothing.
    assert_eq!(rig.held(1).0, set(&[1, 2]));
    assert!(rig.held(2).0.is_empty() && rig.held(2).1.is_empty());
    let stats = fill.stats();
    assert_eq!((stats.vram, stats.ram, stats.reads, stats.dropped), (5, 4, 9, 0));
    assert_eq!(stats.read_bytes, rig.store.stats().demand_read_bytes);
}

#[test]
fn the_start_reads_in_the_order_of_the_file_and_the_idle_fill_the_hottest_first() {
    let mut rig = rig(4, 4, None);
    rig.heat(0, &[(9, 1.0), (3, 2.0), (12, 3.0), (6, 9.0), (1, 0.5)]);
    rig.heat(2, &[(4, 4.0), (8, 0.7)]);
    rig.heat(1, &[(15, 6.0)]);

    // At the start: layer by layer, each in the order of its tensors.
    let file = rig.store.plan_fill(&ALL, FillOrder::File).todo();
    let order: Vec<(usize, u32)> = file.iter().map(|(l, e, _)| (*l, *e)).collect();
    assert_eq!(order, [(0, 1), (0, 3), (0, 6), (0, 9), (0, 12), (1, 15), (2, 4), (2, 8)]);
    let to_vram: BTreeSet<(usize, u32)> =
        file.iter().filter(|i| i.2 == FillTier::Vram).map(|(l, e, _)| (*l, *e)).collect();
    assert_eq!(to_vram, [(0, 6), (0, 12), (0, 3), (1, 15), (2, 4), (2, 8)].into());

    // Idle: those meant for VRAM, the hottest first over all layers, then
    // those meant for RAM.
    let idle = rig.store.plan_fill(&ALL, FillOrder::Heat).todo();
    let order: Vec<(usize, u32)> = idle.iter().map(|(l, e, _)| (*l, *e)).collect();
    assert_eq!(order, [(0, 6), (1, 15), (2, 4), (0, 12), (0, 3), (2, 8), (0, 9), (0, 1)]);

    // The experts come in that order: one read at a time, each arrives
    // before the next is started.
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::File);
    let one = FillPace { reads: 1, uploads: 1, demand: true };
    let mut arrived = Vec::new();
    while !fill.is_done() {
        let before = rig.everything();
        rig.store.advance_fill(&mut fill, one).unwrap();
        rig.settle();
        arrived.extend(rig.everything().difference(&before).copied());
    }
    assert_eq!(arrived, [(0, 1), (0, 3), (0, 6), (0, 9), (0, 12), (1, 15), (2, 4), (2, 8)]);
}

#[test]
fn a_step_stops_the_idle_fill_at_once() {
    let mut rig = rig(4, 4, None);
    rig.heat(0, &[(1, 5.0), (2, 4.0), (3, 3.0), (4, 2.0), (5, 1.0)]);
    rig.heat(1, &[(6, 5.0), (7, 4.0), (8, 3.0)]);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    let planned = fill.left();
    rig.store.advance_fill(&mut fill, FillPace::IDLE).unwrap();
    // Two experts are on their way from the file, as prefetches.
    assert_eq!(fill.stats().reads, FillPace::IDLE.reads as u64);
    assert_eq!(rig.store.stats().guess_reads, 2);
    // The hottest of each of the first two layers: the order is the heat's.
    assert_eq!(fill.todo()[0], (0, 2, FillTier::Vram));

    // A request comes: its step begins and the fill does nothing more.
    rig.store.begin_step();
    for _ in 0..5 {
        rig.store.advance_fill(&mut fill, FillPace::IDLE).unwrap();
        rig.store.device_mut(0).run(usize::MAX);
    }
    assert_eq!(fill.stats().reads, 2, "no read is started while a step runs");
    assert_eq!(fill.left(), planned, "and nothing else of the fill moves");
    // The step's own miss is read and comes into its slot.
    assert_eq!(rig.store.routed(2, &[13]).unwrap(), 1);
    rig.settle();
    rig.assert_usable(2, 13);
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.demand_reads, stats.guess_reads), (1, 2));
    // What was on its way arrived and took its slot in RAM; nothing went up.
    assert!(rig.store.in_ram(0, 1) && rig.store.in_ram(1, 6));
    assert!(!rig.store.in_vram(0, 1) && !rig.store.in_vram(1, 6));
    assert_eq!(rig.everything().len(), 3);

    // Idle again: a new plan goes on from what is there.
    drop(fill);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    rig.fill(&mut fill, FillPace::IDLE);
    assert_eq!(rig.held(0).0, set(&[1, 2, 3]));
    assert_eq!(rig.held(1).0, set(&[6, 7, 8]));
    // The first two came up from RAM without a read.
    assert_eq!(fill.stats().reads, 6);
}

#[test]
fn a_fill_evicts_nothing_that_is_worth_something() {
    // Three VRAM slots a layer, one kept free: two experts go up. Two RAM
    // slots.
    let mut rig = rig(3, 2, None);
    rig.heat(0, &[(1, 9.0), (2, 8.0), (3, 7.0), (4, 6.0)]);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::File);
    rig.fill(&mut fill, FillPace::START);
    let (vram, ram) = rig.held(0);
    assert_eq!(vram, set(&[1, 2]));
    assert_eq!(ram, set(&[3, 4]), "RAM holds the next two");

    // Hotter experts that no tier holds: there is no room for them, and the
    // members are worth something.
    rig.heat(0, &[(1, 9.0), (2, 8.0), (3, 7.0), (4, 6.0), (10, 50.0), (11, 40.0)]);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    assert!(fill.is_done(), "nothing is planned: {:?}", fill.todo());
    rig.fill(&mut fill, FillPace::IDLE);
    assert_eq!(rig.held(0), (set(&[1, 2]), set(&[3, 4])));

    // A member without heat is room.
    rig.heat(0, &[(1, 9.0), (3, 7.0), (10, 50.0), (11, 40.0)]);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    assert_eq!(fill.todo(), [(0, 10, FillTier::Ram)], "expert 4 is worth nothing now");
    rig.fill(&mut fill, FillPace::IDLE);
    assert_eq!(rig.held(0), (set(&[1, 2]), set(&[3, 10])));
}

#[test]
fn the_fill_follows_the_ram_tier_and_never_fights_a_shrink() {
    // Four RAM slots a layer in pieces of two slots, all locked at the start;
    // three VRAM slots, one kept free.
    let mut rig = rig(3, 4, Some((2, 6)));
    for layer in ALL {
        let heat: Vec<(u32, f32)> = (0..EXPERTS).map(|e| (e, 1.0 + e as f32)).collect();
        rig.heat(layer, &heat);
    }
    let (locked, _) = rig.store.ram_bytes();
    let piece = locked / 6;

    // The tier is to give two pieces back: while it holds more than it may,
    // nothing is read into it, and so nothing goes up through it either.
    rig.store.set_ram_target(locked - 2 * piece);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    for _ in 0..5 {
        rig.store.advance_fill(&mut fill, FillPace::IDLE).unwrap();
        rig.store.device_mut(0).run(usize::MAX);
    }
    assert_eq!(fill.stats().reads, 0, "no read while the tier shrinks");
    assert_eq!(rig.store.shrink_ram(Duration::from_secs(1)).unwrap(), 2);
    drop(rig.store.take_released());
    assert_eq!(rig.store.ram_slots(), 8);

    // What is left is filled, no more.
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    rig.fill(&mut fill, FillPace::IDLE);
    let held_ram: usize = ALL.iter().map(|l| rig.store.held(*l).1).sum();
    assert_eq!(held_ram, 8, "every slot the tier holds now, and not one more");
    for layer in ALL {
        let (vram, _) = rig.held(layer);
        assert_eq!(vram, set(&[14, 15]), "layer {layer}: the hottest in its free VRAM slots");
    }

    // The tier may hold all again: the pieces come back empty and the next
    // fill takes them.
    rig.store.set_ram_target(locked);
    assert_eq!(rig.store.grow_ram(10).unwrap(), 2);
    assert_eq!(rig.store.ram_slots(), 12);
    let mut fill = rig.store.plan_fill(&ALL, FillOrder::Heat);
    assert_eq!(fill.todo().len(), 4);
    rig.fill(&mut fill, FillPace::IDLE);
    let held_ram: usize = ALL.iter().map(|l| rig.store.held(*l).1).sum();
    assert_eq!(held_ram, 12);
    // What RAM held stays: the hottest outside VRAM.
    for layer in ALL {
        let (_, ram) = rig.held(layer);
        assert!(ram.contains(&13), "layer {layer}: {ram:?}");
    }
}
