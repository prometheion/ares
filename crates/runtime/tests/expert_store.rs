//! Tests of the expert store on a small model file and slots in ordinary
//! memory (`HostSlots`), whose copies and table writes run only when the test
//! says so: what the store does can then be checked at every point a device
//! could be at.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Config, ExpertMap, Reader};
use ares_runtime::expert_store::{
    ExpertStore, HostRegion, HostSlots, LayerSpec, StoreConfig, StorePlan,
};

const LAYERS: usize = 3;
const EXPERTS: u32 = 16;
/// Bytes of gate, up and down of one expert, and where they lie in a slot.
const LENS: [usize; 3] = [128, 128, 96];
const OFFSETS: [u64; 3] = [0, 128, 256];
const SLOT_BYTES: u64 = 352;

/// A scratch directory that goes away even when the test fails.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ares-expert-store-{}-{}",
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

/// Byte `i` of matrix `part` of an expert: different everywhere.
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

/// A model of `LAYERS` layers of `EXPERTS` experts with F32 matrices.
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

/// Heat that halves with every step, so that what was used longest ago is
/// the coldest.
fn recent_first(config: &mut StoreConfig) {
    config.decay_every = 1;
    config.heat_decay = 0.5;
}

struct Rig {
    store: ExpertStore<HostSlots>,
    _dir: Scratch,
}

/// A store with `vram_slots` slots per layer in VRAM and `ram_slots` in RAM.
fn rig(vram_slots: u64, ram_slots: u64, tune: impl FnOnce(&mut StoreConfig)) -> Rig {
    let dir = Scratch::new();
    let gguf = model(&dir);
    let map =
        ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
    let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
    let layers = [LayerSpec { device: 0, offsets: OFFSETS, slot_bytes: SLOT_BYTES }; LAYERS];
    // What a RAM slot takes is the plan's business; ask it with a large
    // budget first.
    let mut config =
        StoreConfig::new(1, 2, vec![vram_slots * SLOT_BYTES * LAYERS as u64], u64::MAX / 4);
    config.vram_reserve = 1;
    config.rebalance_every = 0;
    tune(&mut config);
    let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
    let ram_slot = probe.ram_bytes() / (probe.slots(0).1 * LAYERS);
    config.ram_bytes = ram_slots * (ram_slot * LAYERS) as u64;
    let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
    assert_eq!(plan.slots(0), (vram_slots as usize, ram_slots as usize));
    let device = HostSlots::new(&plan.device_pools(0), LAYERS, EXPERTS);
    let region = HostRegion::unpinned(plan.ram_bytes(), plan.block());
    Rig {
        store: ExpertStore::new(plan, map, reader, vec![device], region, None).unwrap(),
        _dir: dir,
    }
}

impl Rig {
    /// Lets reads finish and the device run until nothing is wanted.
    fn settle(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            self.store.poll().unwrap();
            self.store.device_mut(0).run(usize::MAX);
            self.store.poll().unwrap();
            if self.store.is_settled() {
                return;
            }
            assert!(Instant::now() < deadline, "the store does not settle: {:?}", self.store);
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// One step that routes `picks[l]` in layer `l`.
    fn step(&mut self, picks: &[&[u32]]) {
        self.store.begin_step();
        for (layer, picks) in picks.iter().enumerate() {
            self.store.routed(layer, picks).unwrap();
            self.settle();
            for expert in *picks {
                self.assert_usable(layer, *expert);
            }
        }
        self.store.end_step().unwrap();
        self.settle();
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
}

#[test]
fn a_missing_pick_comes_from_the_file_into_a_slot() {
    let mut rig = rig(4, 4, |_| {});
    rig.store.begin_step();
    assert_eq!(rig.store.routed(0, &[3, 5]).unwrap(), 2);
    assert_eq!(rig.store.waiting(), 2);
    assert!(!rig.store.in_vram(0, 3));
    rig.settle();
    assert!(rig.store.in_vram(0, 3) && rig.store.in_vram(0, 5));
    rig.assert_usable(0, 3);
    rig.assert_usable(0, 5);
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.picks, stats.hits, stats.miss_from_disk), (2, 0, 2));
    assert_eq!((stats.demand_reads, stats.uploads, stats.early_uploads), (2, 2, 0));
    assert_eq!(stats.miss_ms.len(), 2);
    assert_eq!(stats.upload_bytes, 2 * 352);

    // The next step finds them.
    rig.store.begin_step();
    assert_eq!(rig.store.routed(0, &[5, 3]).unwrap(), 0);
    rig.store.end_step().unwrap();
    let stats = rig.store.take_stats();
    assert_eq!((stats.picks, stats.hits, stats.demand_reads), (2, 2, 0));
}

#[test]
fn an_entry_appears_only_after_the_bytes() {
    let mut rig = rig(4, 4, |_| {});
    rig.store.begin_step();
    rig.store.routed(1, &[9, 2]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while rig.store.waiting() > 0 {
        rig.store.poll().unwrap();
        // The device runs one operation at a time; whatever entry it has
        // must name a slot that holds the expert.
        rig.store.device_mut(0).run(1);
        for expert in [9, 2] {
            if rig.store.device(0).entry(1, expert) >= 0 {
                rig.assert_usable(1, expert);
            } else {
                assert!(!rig.store.in_vram(1, expert), "the store is ahead of the device");
            }
        }
        assert!(Instant::now() < deadline);
    }
    rig.assert_usable(1, 9);
    rig.assert_usable(1, 2);
}

#[test]
fn an_evicted_expert_goes_down_to_ram_and_comes_back_without_a_read() {
    // 3 slots per layer in VRAM (2 for a step, 1 free), 6 in RAM.
    let mut rig = rig(3, 6, recent_first);
    rig.step(&[&[0, 1][..]]);
    rig.step(&[&[2, 3][..]]);
    rig.step(&[&[4, 5][..]]);
    let stats = rig.store.take_stats();
    assert_eq!(stats.demand_reads, 6);
    assert!(stats.evictions >= 3, "{stats:?}");
    // Experts 0 and 1 left VRAM. They were read into RAM on their way up and
    // are still there, or came down again when they were evicted.
    assert!(!rig.store.in_vram(0, 0) && !rig.store.in_vram(0, 1));
    assert!(rig.store.in_ram(0, 0) && rig.store.in_ram(0, 1));
    assert_eq!(rig.store.device(0).entry(0, 0), -1);

    rig.step(&[&[0, 1][..]]);
    let stats = rig.store.take_stats();
    assert_eq!((stats.miss_from_ram, stats.miss_from_disk, stats.demand_reads), (2, 0, 0));
    assert_eq!(stats.uploads, 2);
}

#[test]
fn what_vram_evicts_and_ram_has_lost_is_copied_down() {
    // RAM holds 2 experts per layer: the copies of the first experts are
    // long gone when VRAM evicts them, so they must come down from the slot.
    let mut rig = rig(4, 2, recent_first);
    rig.step(&[&[0, 1][..]]);
    rig.step(&[&[2, 3][..]]);
    assert!(!rig.store.in_ram(0, 0) || !rig.store.in_ram(0, 1), "RAM has two slots");
    rig.store.take_stats();
    // Experts 0 to 3 fill VRAM's four slots; the fifth and sixth push two out.
    rig.step(&[&[4, 5][..]]);
    let stats = rig.store.take_stats();
    assert!(stats.demotions >= 1, "{stats:?}");
    let down: Vec<u32> =
        (0..4).filter(|e| !rig.store.in_vram(0, *e) && rig.store.in_ram(0, *e)).collect();
    assert!(!down.is_empty(), "an evicted expert is in RAM");
    // It comes back from RAM with the right bytes: the copy down put them
    // where a read would have.
    let back = down[0];
    let other = (0..4).find(|e| rig.store.in_vram(0, *e)).unwrap();
    rig.step(&[&[back, other][..]]);
    let stats = rig.store.take_stats();
    assert_eq!((stats.miss_from_ram, stats.demand_reads), (1, 0));
}

#[test]
fn a_guess_arrives_before_its_layer_and_takes_only_free_slots() {
    let mut rig = rig(4, 8, |config| config.vram_reserve = 2);
    // Fill layer 1: two slots hold experts, two are kept free.
    rig.step(&[&[][..], &[0, 1][..]]);
    rig.store.take_stats();

    rig.store.begin_step();
    rig.store.routed(0, &[6, 7]).unwrap();
    // While layer 0 runs, layer 1's picks are predicted: three experts, of
    // which two can have the free slots.
    rig.store.predicted(1, &[10, 11, 12], usize::MAX).unwrap();
    rig.settle_guesses();
    let early: Vec<u32> = [10, 11, 12].into_iter().filter(|e| rig.store.in_vram(1, *e)).collect();
    assert_eq!(early.len(), 2, "two free slots");
    for expert in &early {
        rig.assert_usable(1, *expert);
    }
    // Nothing of layer 1 was removed: the step may still pick it.
    rig.assert_usable(1, 0);
    rig.assert_usable(1, 1);

    // The router of layer 1 picks a predicted expert and one that was there.
    assert_eq!(rig.store.routed(1, &[early[0], 0]).unwrap(), 0);
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!(stats.early_uploads, 2);
    assert_eq!(stats.guess_reads, 3);
    assert_eq!((stats.picks, stats.hits), (4, 2));
}

#[test]
fn an_unsure_guess_is_uploaded_from_ram_but_not_read_from_the_file() {
    let mut rig = rig(4, 8, |config| {
        recent_first(config);
        config.vram_reserve = 2;
    });
    // Expert 4 gets into RAM and out of VRAM again; expert 13 is only in the file.
    rig.step(&[&[][..], &[4, 5][..]]);
    rig.step(&[&[][..], &[6, 7][..]]);
    rig.step(&[&[][..], &[8, 9][..]]);
    assert!(!rig.store.in_vram(1, 4) && rig.store.in_ram(1, 4));
    rig.store.take_stats();

    rig.store.begin_step();
    rig.store.routed(0, &[1, 2]).unwrap();
    // Nothing of what is named may be read: both are unsure.
    rig.store.predicted(1, &[4, 13], 0).unwrap();
    rig.settle_guesses();
    assert!(rig.store.in_vram(1, 4), "what RAM has is uploaded early");
    assert!(!rig.store.in_ram(1, 13) && !rig.store.in_vram(1, 13));
    rig.store.routed(1, &[4, 13]).unwrap();
    rig.settle();
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.guess_reads, stats.early_uploads), (0, 1));
    assert_eq!(
        stats.demand_reads, 3,
        "experts 1 and 2 of layer 0, and expert 13 when it was picked"
    );
}

#[test]
fn slots_follow_the_heat_between_layers() {
    let mut rig = rig(5, 8, |config| {
        config.rebalance_every = 4;
        config.decay_every = 0;
    });
    // Layer 0 uses many experts, layer 2 always the same two.
    for step in 0..8u32 {
        let a = [step % 8, (step + 3) % 8 + 8];
        rig.step(&[&a[..], &[1, 2][..], &[4, 5][..]]);
    }
    let stats = rig.store.take_stats();
    assert!(stats.moved_slots > 0, "{stats:?}");
    let (wide, _) = rig.store.slots(0);
    let (narrow, _) = rig.store.slots(2);
    assert!(wide > 5 && narrow < 5, "layer 0 has {wide} slots, layer 2 {narrow}");
    assert_eq!((0..LAYERS).map(|l| rig.store.slots(l).0).sum::<usize>(), 15);
    // The layers still work with their new slots.
    rig.step(&[&[0, 9][..], &[1, 2][..], &[4, 5][..]]);
    rig.step(&[&[14, 15][..], &[3, 2][..], &[5, 6][..]]);
}

#[test]
fn a_prompt_takes_a_layers_experts_in_waves_and_fetches_each_once() {
    // 4 slots per layer in VRAM, 8 in RAM (a wave, the next one, and what
    // comes down from VRAM); the prompt routes to all 16.
    let mut rig = rig(4, 8, recent_first);
    rig.step(&[&[][..], &[3, 9][..]]);
    rig.store.take_stats();
    let counts: Vec<u32> = (0..EXPERTS).map(|e| 1 + e % 5).collect();
    let plan = rig.store.plan_waves(1, &counts, 3).expect("room for a wave");
    assert_eq!(plan.in_place, [3, 9]);
    assert_eq!(plan.waves.len(), 5, "14 experts, 3 at a time");
    for (i, wave) in plan.waves.iter().enumerate() {
        let next = plan.waves.get(i + 1).map_or(&[][..], |w| &w.experts[..]);
        rig.store.wave(1, &wave.experts, next).unwrap();
        rig.settle();
        // Now the wave's kernels would run: every expert of it is usable.
        for expert in &wave.experts {
            rig.assert_usable(1, *expert);
        }
    }
    rig.store.record_counts(1, &counts, 1.0);
    let stats = rig.store.take_stats();
    assert_eq!(stats.uploads, 14, "each expert once");
    assert_eq!(stats.demand_reads + stats.guess_reads, 14, "each read once: {stats:?}");
    assert!(stats.guess_reads >= 8, "the next wave is read while one is computed: {stats:?}");
    // A decode step afterwards works as before.
    rig.step(&[&[1, 2][..], &[3, 9][..]]);
}

impl Rig {
    /// Lets guesses be read and uploaded as far as they can be.
    fn settle_guesses(&mut self) {
        let deadline = Instant::now() + Duration::from_millis(300);
        while Instant::now() < deadline {
            self.store.poll().unwrap();
            self.store.device_mut(0).run(usize::MAX);
            std::thread::sleep(Duration::from_micros(200));
        }
    }
}
