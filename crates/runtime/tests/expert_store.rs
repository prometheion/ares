//! Tests of the expert store on a small model file and slots in ordinary
//! memory (`HostSlots`), whose copies and table writes run only when the test
//! says so: what the store does can then be checked at every point a device
//! could be at.

use std::any::Any;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Config, ExpertMap, Reader};
use ares_runtime::expert_store::{
    ExpertStore, FillOrder, FillPace, HostRegion, LayerSpec, Locker, ScratchLoan, StoreConfig,
    StoreError, StorePlan,
};
use ares_runtime::testing::HostSlots;

mod expert_rig;

use expert_rig::*;

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
    assert_eq!((stats.demand_reads, stats.uploads), (2, 2));
    assert_eq!(stats.miss_ms.len(), 2);
    assert_eq!(stats.upload_bytes, 2 * 352);
    // The layer waited once, and last for an expert from the file.
    assert_eq!(stats.waits_last, [0, 0, 1, 0]);
    assert_eq!(stats.wait_last_ms[2], stats.wait_ms);

    // The next step finds them.
    rig.store.begin_step();
    assert_eq!(rig.store.routed(0, &[5, 3]).unwrap(), 0);
    rig.store.end_step().unwrap();
    let stats = rig.store.take_stats();
    assert_eq!((stats.picks, stats.hits, stats.demand_reads), (2, 2, 0));
}

#[test]
fn a_resident_layer_holds_all_its_experts_outside_the_tiers_and_never_misses() {
    let dir = Scratch::new();
    let gguf = model(&dir);
    let map =
        ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
    let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
    // Layer 0 is a drafter's: all 16 experts stay in VRAM. The other two
    // share what is left, 4 slots each.
    let mut layers =
        [LayerSpec { device: 0, offsets: OFFSETS, slot_bytes: SLOT_BYTES, resident: false };
            LAYERS];
    layers[0].resident = true;
    let vram = (16 + 2 * 4) * SLOT_BYTES;
    let mut config = StoreConfig::new(1, 2, vec![vram], u64::MAX / 4);
    config.vram_reserve = 1;
    config.rebalance_every = 1;
    recent_first(&mut config);
    let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
    config.ram_bytes = 64 * probe.ram_slot_bytes(1);
    let plan = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
    assert_eq!(plan.slots(0).0, 16, "every expert of the resident layer");
    assert_eq!((plan.slots(1).0, plan.slots(2).0), (4, 4));
    assert_eq!(plan.vram_bytes(0), vram);
    // A budget that does not hold the resident experts and the others' floor
    // is refused, and says why.
    let small = StoreConfig { vram_bytes: vec![15 * SLOT_BYTES], ..config.clone() };
    let refused = StorePlan::new(small, &map, &reader, &layers, None).unwrap_err().to_string();
    assert!(refused.contains("resident"), "{refused}");
    let short = StoreConfig { vram_bytes: vec![17 * SLOT_BYTES], ..config.clone() };
    assert!(StorePlan::new(short, &map, &reader, &layers, None).is_err());

    let device = HostSlots::new(&plan.device_pools(0), LAYERS, EXPERTS);
    let region = HostRegion::unpinned(plan.ram_bytes(), plan.block());
    let mut rig =
        Rig { store: ExpertStore::new(plan, map, reader, vec![device], region).unwrap(), dir };
    assert!(rig.store.is_resident(0) && !rig.store.is_resident(1));
    // Filled once, through the few RAM slots the layer passes its experts
    // through.
    rig.fill(0, &(0..EXPERTS).collect::<Vec<_>>());
    for expert in 0..EXPERTS {
        rig.assert_usable(0, expert);
    }
    rig.store.take_stats();
    // Steps that use every expert of the other layers in turn, with heat that
    // forgets fast and slots that move between layers at every step: the
    // resident layer gives up nothing and finds every pick.
    for round in 0..8u32 {
        let others = [round * 2 % EXPERTS, (round * 2 + 1) % EXPERTS];
        rig.step(&[&[round % EXPERTS, (round + 5) % EXPERTS][..], &others[..], &others[..]]);
    }
    let stats = rig.store.take_stats();
    assert!(stats.evictions > 0, "the other layers churn: {stats:?}");
    for expert in 0..EXPERTS {
        rig.assert_usable(0, expert);
    }
    assert_eq!(rig.store.slots(0).0, 16);
    rig.store.begin_step();
    assert_eq!(rig.store.routed(0, &[1, 7, 15]).unwrap(), 0);
    rig.store.end_step().unwrap();
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
    assert_eq!(stats.waits_last, [0, 1, 0, 0], "the layer waited for an upload from RAM");
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

/// `n` experts of `layer` that no tier holds, the lowest numbers first.
fn cold(rig: &Rig, layer: usize, n: usize) -> Vec<u32> {
    (0..EXPERTS)
        .filter(|e| !rig.store.in_vram(layer, *e) && !rig.store.in_ram(layer, *e))
        .take(n)
        .collect()
}

impl Rig {
    /// `steps` steps of one row in which a prediction one layer ahead names `places` experts
    /// of `layer` (1 or 2) that no tier holds, the first `picked` of which its router then
    /// picks. The counts of the read-ahead are the store's, whatever the layer: layer 2
    /// teaches them and leaves layer 1 as it is.
    fn predict_steps(&mut self, layer: usize, steps: usize, places: usize, picked: usize) {
        self.predict_steps_of(1, layer, steps, places, picked);
    }

    /// [`Rig::predict_steps`] with steps of `rows` rows.
    fn predict_steps_of(
        &mut self,
        rows: usize,
        layer: usize,
        steps: usize,
        places: usize,
        picked: usize,
    ) {
        for _ in 0..steps {
            self.store.begin_step();
            for before in 0..layer {
                self.store.routed(before, &[1]).unwrap();
            }
            self.settle();
            let named = cold(self, layer, places);
            assert_eq!(named.len(), places, "the tiers leave enough experts in the file");
            self.store.ahead(layer, &named, 1, rows);
            self.store.poll().unwrap();
            self.store.routed(layer, &named[..picked]).unwrap();
            self.settle();
            self.store.end_step().unwrap();
            self.settle();
        }
    }
}

#[test]
fn reads_ahead_start_where_predictions_have_been_picked_more_often_than_not() {
    let mut rig = rig(4, 8, recent_first);
    // Between steps a prediction does nothing.
    rig.store.ahead(1, &[10], 1, 1);
    rig.store.poll().unwrap();
    assert_eq!(rig.store.stats().predicted, 0);

    // The first place of a list is picked every time, the second never: nothing is read
    // until the counts have samples, and then only the first place.
    rig.predict_steps(2, 16, 2, 1);
    let stats = rig.store.take_stats();
    assert_eq!((stats.predicted, stats.predicted_picked), (32, 16));
    assert_eq!((stats.predicted_in_file, stats.predicted_in_file_picked), (32, 16));
    assert_eq!((stats.ahead_reads, stats.ahead_left), (0, 32), "no samples yet");
    rig.predict_steps(1, 4, 2, 1);
    let stats = rig.store.take_stats();
    assert_eq!((stats.ahead_reads, stats.ahead_used, stats.ahead_wasted), (4, 4, 0));
    assert_eq!(stats.ahead_left, 4, "the second place is not read");
    assert_eq!(stats.ahead_read_bytes, stats.guess_read_bytes);

    // A read ahead goes into RAM and its record finds it there.
    rig.store.begin_step();
    rig.store.routed(0, &[1]).unwrap();
    let named = cold(&rig, 1, 1);
    // What a tier holds is named and not read; a layer whose record has arrived is not
    // predicted any more.
    let held = (0..EXPERTS).find(|e| rig.store.in_vram(1, *e)).unwrap();
    rig.store.ahead(1, &[named[0], held], 1, 1);
    rig.store.ahead(0, &[12], 1, 1);
    rig.settle_guesses(|store| store.in_ram(1, named[0]));
    assert!(!rig.store.in_vram(1, named[0]), "a read ahead goes to RAM, not to VRAM");
    assert_eq!(rig.store.routed(1, &[named[0]]).unwrap(), 1);
    rig.settle();
    rig.assert_usable(1, named[0]);
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.predicted, stats.predicted_picked), (2, 1), "one was in VRAM");
    assert_eq!((stats.ahead_reads, stats.ahead_used, stats.miss_from_ram), (1, 1, 1));
}

#[test]
fn a_step_predicts_only_as_far_as_a_place_is_worth_a_read_and_now_and_then() {
    let mut rig = rig(4, 8, recent_first);
    // Before any counts no place qualifies: one step in eight predicts, as far as it can.
    let depths: Vec<usize> = (0..16).map(|_| rig.store.ahead_depth(1, 2)).collect();
    assert_eq!(depths.iter().filter(|d| **d == 2).count(), 2);
    assert!(depths.iter().all(|d| *d == 0 || *d == 2));
    // Steps of several rows are counted apart.
    assert_eq!(rig.store.ahead_depth(4, 2), 0);

    // Predictions one layer ahead that are picked: every step of one row predicts one
    // layer ahead, and every eighth two.
    rig.predict_steps(2, 16, 1, 1);
    let depths: Vec<usize> = (0..16).map(|_| rig.store.ahead_depth(1, 2)).collect();
    assert!(depths.iter().all(|d| *d >= 1), "{depths:?}");
    assert_eq!(depths.iter().filter(|d| **d == 2).count(), 2, "{depths:?}");
    // A part that predicts one layer ahead does so every step.
    assert!((0..16).all(|_| rig.store.ahead_depth(1, 1) == 1));
    // The several rows have no counts yet.
    assert_eq!(rig.store.ahead_depth(4, 2), 0);

    // Once the first place is no longer picked, no place qualifies again.
    rig.predict_steps(2, 48, 1, 0);
    let depths: Vec<usize> = (0..16).map(|_| rig.store.ahead_depth(1, 1)).collect();
    assert_eq!(depths.iter().filter(|d| **d == 1).count(), 2, "{depths:?}");
}

#[test]
fn a_read_ahead_takes_only_room_that_nobody_uses() {
    let mut rig = rig(4, 8, recent_first);
    rig.predict_steps(2, 17, 1, 1);
    // Layer 1's RAM tier fills with experts that steps used, most recently first.
    for e in 0..12u32 {
        rig.step(&[&[][..], &[e][..]]);
    }
    rig.store.take_stats();
    let ram_before: Vec<u32> = (0..EXPERTS).filter(|e| rig.store.in_ram(1, *e)).collect();
    rig.predict_steps(1, 1, 1, 0);
    let stats = rig.store.take_stats();
    assert_eq!((stats.ahead_reads, stats.ahead_left), (0, 1), "no room nobody uses");
    let ram_after: Vec<u32> = (0..EXPERTS).filter(|e| rig.store.in_ram(1, *e)).collect();
    assert_eq!(ram_before, ram_after, "the RAM tier lost nothing to the read-ahead");
}

#[test]
fn what_steps_of_several_rows_pick_says_nothing_of_steps_of_one() {
    let mut rig = rig(4, 8, recent_first);
    // In steps of several rows the first place is picked every time; in steps of one row it
    // is never picked.
    rig.predict_steps_of(8, 2, 9, 1, 1);
    rig.predict_steps(2, 17, 1, 0);
    rig.predict_steps_of(8, 2, 9, 1, 1);
    rig.store.take_stats();
    rig.predict_steps(1, 1, 1, 0);
    rig.predict_steps_of(8, 1, 1, 1, 1);
    let stats = rig.store.take_stats();
    assert_eq!((stats.ahead_reads, stats.ahead_used, stats.ahead_left), (1, 1, 1));
}

#[test]
fn a_read_ahead_that_its_record_does_not_need_is_dropped() {
    let mut rig = rig(4, 8, recent_first);
    rig.predict_steps(2, 17, 1, 1);
    rig.store.take_stats();
    rig.store.begin_step();
    rig.store.routed(0, &[1]).unwrap();
    rig.settle();
    let named = cold(&rig, 1, 1);
    rig.store.ahead(1, &named, 1, 1);
    rig.store.poll().unwrap();
    // The record arrives before the read has come back, or after: either way the read is
    // done with and nothing waits for it.
    let other = cold(&rig, 1, 2)[1];
    assert_eq!(rig.store.routed(1, &[other]).unwrap(), 1);
    rig.settle();
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.ahead_reads, stats.ahead_wasted, stats.ahead_used), (1, 1, 0));
    assert_eq!(stats.failed_reads, 0, "a cancelled read ahead is no failed read");
    assert!(rig.store.is_settled());
    assert!(!rig.store.in_vram(1, named[0]));
}

/// A fill between steps that names an expert whose read ahead its record cancelled, while
/// the cancelled read is still on its way, takes the read as coming and starts none of its
/// own: the read then fails, and the store reads the expert again for the fill. Whether
/// the read has come back before the fill depends on the reader's thread, so the scene is
/// played often enough that some runs meet it.
#[test]
fn a_fill_gets_the_expert_whose_read_ahead_was_cancelled_on_its_way() {
    let (mut lost, runs) = (Vec::new(), 100);
    for run in 0..runs {
        // One chunk of 64 bytes at the drive at a time: an expert's read is several.
        let reader = Config { prefetch_depth: 1, prefetch_chunk: 64, ..Config::default() };
        let mut rig = rig_reading(4, 8, None, reader, recent_first);
        rig.predict_steps(2, 17, 1, 1);
        rig.store.begin_step();
        rig.store.routed(0, &[1]).unwrap();
        rig.settle();
        let named = cold(&rig, 1, 2);
        rig.store.ahead(1, &named[..1], 1, 1);
        rig.store.poll().unwrap();
        // The record picks another expert: the read ahead is cancelled on its way.
        rig.store.routed(1, &named[1..]).unwrap();
        rig.store.end_step().unwrap();
        rig.fill(1, &named[..1]);
        if !rig.store.in_vram(1, named[0]) && !rig.store.in_ram(1, named[0]) {
            lost.push(run);
        }
    }
    assert!(lost.is_empty(), "the fills of runs {lost:?} of {runs} left the expert in no tier");
}

#[test]
fn reads_ahead_keep_to_what_the_drive_takes_of_prefetches() {
    // The reader takes one chunk of 256 bytes of prefetches at once: less than an expert,
    // so two reads ahead are on their way at a time, the one at the drive and one queued
    // behind it. It takes one chunk of any kind at a time, so the drive is never measured
    // under load, and the read-ahead has no bound of bytes per step, only its bound in
    // flight (`ahead.rs`). With such a bound the third would follow only on a drive that
    // is idle for much of a step: on a slow one (a CI runner's) the steps before spent
    // their time on their own reads, and the bound let two reads start at most.
    let reader =
        Config { queue_depth: 1, prefetch_depth: 1, prefetch_chunk: 256, ..Config::default() };
    let mut rig = rig_reading(4, 8, None, reader, recent_first);
    rig.predict_steps(2, 17, 3, 3);
    rig.store.take_stats();
    rig.store.begin_step();
    rig.store.routed(0, &[1]).unwrap();
    rig.settle();
    let named = cold(&rig, 1, 3);
    rig.store.ahead(1, &named, 1, 1);
    rig.store.poll().unwrap();
    assert_eq!(rig.store.stats().ahead_reads, 2, "two in flight");
    rig.settle_guesses(|store| named.iter().all(|e| store.in_ram(1, *e)));
    assert_eq!(rig.store.stats().ahead_reads, 3, "the third follows");
    rig.store.routed(1, &named[1..2]).unwrap();
    rig.settle();
    rig.store.end_step().unwrap();
    let stats = rig.store.take_stats();
    assert_eq!((stats.ahead_used, stats.ahead_wasted), (1, 2));
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

impl Rig {
    /// Brings `experts` of `layer` into the tiers as a model's fill from a
    /// list does ([`ExpertStore::plan_listed_fill`]), and lets the store
    /// settle.
    fn fill(&mut self, layer: usize, experts: &[u32]) {
        let mut fill = self.store.plan_listed_fill(&[(layer, experts.to_vec())], FillOrder::File);
        let deadline = Instant::now() + Duration::from_secs(20);
        while !fill.is_done() {
            self.store.advance_fill(&mut fill, FillPace::START).unwrap();
            self.store.device_mut(0).run(usize::MAX);
            assert!(Instant::now() < deadline, "the fill does not end: {:?}", self.store);
            std::thread::sleep(Duration::from_micros(200));
        }
        self.settle();
    }

    /// Lets guesses be read and uploaded until `there` says that what the
    /// test expects has arrived, however long the reader's thread takes for
    /// it, and for a few rounds more, in which what should not follow would
    /// show. (A step whose guesses are not all placed does not settle, so the
    /// test names what it waits for.)
    fn settle_guesses(&mut self, there: impl Fn(&ExpertStore<HostSlots>) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut more = 50;
        while more > 0 {
            self.store.poll().unwrap();
            self.store.device_mut(0).run(usize::MAX);
            if there(&self.store) {
                more -= 1;
            } else {
                assert!(Instant::now() < deadline, "the guesses never arrived: {:?}", self.store);
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }
}

/// The CPU half of the store, on a model whose experts have the types the
/// CPU kernels compute (IQ2_S gate and up, Q2_0 down) and a mock device.
mod hybrid {
    use std::sync::Arc;

    use ares_cuda::{ExpertShape, RouteGroup};
    use ares_runtime::expert_store::Serve;
    use ares_runtime::expert_store::cpu::{
        CpuExpert, CpuPool, Features, LayerJob, Placement, quantize_activations,
    };

    use super::*;

    const LAYERS: usize = 2;
    const EXPERTS: u32 = 16;
    const SHAPE: ExpertShape = ExpertShape {
        n_embd: 256,
        n_ff: 64,
        n_experts: EXPERTS,
        top_k: 2,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };

    /// Matrix `part` of an expert: bytes that differ everywhere and keep
    /// every scale small (all below 0x40).
    fn matrix(layer: usize, part: usize, expert: u32) -> Vec<u8> {
        let len = if part < 2 { SHAPE.gate_bytes() } else { SHAPE.down_bytes() }.unwrap() as usize;
        let seed = layer * 131 + part * 71 + expert as usize * 29;
        (0..len).map(|i| ((i * 31 + seed * 17) ^ (i >> 5)) as u8 & 0x3f).collect()
    }

    struct Rig {
        store: ExpertStore<HostSlots>,
        act: Vec<u8>,
        _dir: Scratch,
    }

    /// The model file of the rigs, its map and reader, and the layers' slots.
    fn model() -> (Scratch, ExpertMap, Reader, [LayerSpec; LAYERS]) {
        let dir = Scratch::new();
        let mut builder = GgufBuilder::new().str("general.architecture", "test");
        for layer in 0..LAYERS {
            for (part, (name, dims)) in [
                ("ffn_gate_exps", [256u64, 64, u64::from(EXPERTS)]),
                ("ffn_up_exps", [256, 64, u64::from(EXPERTS)]),
                ("ffn_down_exps", [64, 256, u64::from(EXPERTS)]),
            ]
            .into_iter()
            .enumerate()
            {
                let data: Vec<u8> = (0..EXPERTS).flat_map(|e| matrix(layer, part, e)).collect();
                let ty = if part < 2 { SHAPE.gate_ty } else { SHAPE.down_ty };
                builder = builder.tensor(&format!("blk.{layer}.{name}.weight"), &dims, ty, data);
            }
        }
        let path = dir.0.join("model.gguf");
        builder.write(&path).unwrap();
        let gguf = Gguf::open(&path).unwrap();
        let map =
            ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
        let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
        let slot_bytes = SHAPE.slot_bytes().unwrap();
        let spec = LayerSpec {
            device: 0,
            offsets: [0, SHAPE.up_offset().unwrap(), SHAPE.down_offset().unwrap()],
            slot_bytes,
            resident: false,
        };
        (dir, map, reader, [spec; LAYERS])
    }

    fn rig(vram_slots: u64, ram_slots: u64, serve: Serve) -> Rig {
        let (dir, map, reader, layers) = model();
        let slot_bytes = layers[0].slot_bytes;
        let mut config =
            StoreConfig::new(1, 2, vec![vram_slots * slot_bytes * LAYERS as u64], u64::MAX / 4);
        config.vram_reserve = 0;
        config.rebalance_every = 0;
        let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
        let ram_slot = probe.ram_slot_bytes(0) as usize;
        config.ram_bytes = ram_slots * (ram_slot * LAYERS) as u64;
        let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
        assert_eq!(plan.slots(0), (vram_slots as usize, ram_slots as usize));
        let device = HostSlots::new(&plan.device_pools(0), LAYERS, EXPERTS);
        let region = HostRegion::unpinned(plan.ram_bytes(), plan.block());
        let mut store = ExpertStore::new(plan, map, reader, vec![device], region).unwrap();
        let pool = CpuPool::new(&Placement::unpinned(2), Duration::from_millis(1)).unwrap();
        store.enable_cpu(pool, vec![SHAPE; LAYERS], serve).unwrap();
        let x: Vec<f32> = (0..256).map(|i| ((i * 37 % 255) as f32 - 127.0) / 64.0).collect();
        Rig { store, act: quantize_activations(&x, 256), _dir: dir }
    }

    /// Gives the cost model an upload of a microsecond, many times: the link then has room
    /// for every promotion a test starts between two steps, whatever the machine's timing
    /// (the room is what the link does in the time of a step).
    fn fast_link(rig: &mut Rig) {
        let model = rig.store.cost_model_mut().expect("a CPU half");
        for _ in 0..64 {
            model.record_upload(0, Duration::from_micros(1));
        }
    }

    fn group(expert: u32, weight: f32) -> RouteGroup {
        let mut weights = [0.0; 8];
        weights[0] = weight;
        RouteGroup { expert, slot: None, rows: 1, weight: weights }
    }

    /// What the kernels give for `groups` of `layer` and one row, computed
    /// here from the file's bytes on this thread.
    fn reference(layer: usize, act: &[u8], groups: &[RouteGroup]) -> Vec<f32> {
        reference_rows(layer, act, groups, 1)
    }

    /// [`reference`] for a step of `n_rows` rows.
    fn reference_rows(layer: usize, act: &[u8], groups: &[RouteGroup], n_rows: usize) -> Vec<f32> {
        let matrices: Vec<[Vec<u8>; 3]> = groups
            .iter()
            .map(|g| {
                [matrix(layer, 0, g.expert), matrix(layer, 1, g.expert), matrix(layer, 2, g.expert)]
            })
            .collect();
        let experts: Vec<CpuExpert> = groups
            .iter()
            .zip(&matrices)
            .map(|(g, m)| CpuExpert {
                gate: m[0].as_ptr(),
                up: m[1].as_ptr(),
                down: m[2].as_ptr(),
                rows: g.rows,
                weight: g.weight,
            })
            .collect();
        let features = Features::detect();
        // SAFETY: the matrices are whole and outlive the job.
        let job =
            Arc::new(unsafe { LayerJob::new(features, &SHAPE, n_rows, act, &experts) }.unwrap());
        job.work();
        job.sums().unwrap().to_vec()
    }

    impl Rig {
        /// Polls, and lets the device run, until `done` says so.
        fn until(&mut self, what: &str, done: impl Fn(&ExpertStore<HostSlots>) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                self.store.poll().unwrap();
                self.store.device_mut(0).run(usize::MAX);
                self.store.poll().unwrap();
                if done(&self.store) {
                    return;
                }
                assert!(Instant::now() < deadline, "never {what}: {:?}", self.store);
                std::thread::sleep(Duration::from_micros(200));
            }
        }

        fn delivered(&mut self, layer: usize, token: u32) -> Vec<f32> {
            self.until("delivered", |store| {
                store.device(0).share(layer).delivered.as_ref().is_some_and(|(t, _)| *t == token)
            });
            self.store.device(0).share(layer).delivered.clone().unwrap().1
        }
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        let err: f64 = got.iter().zip(want).map(|(g, w)| f64::from(g - w).powi(2)).sum();
        let norm: f64 = want.iter().map(|w| f64::from(*w).powi(2)).sum();
        assert!(
            norm > 0.0 && (err / norm).sqrt() < 1e-5,
            "sums differ by {:e}",
            (err / norm).sqrt()
        );
    }

    /// The slots a loan of VRAM gives back come back empty, many at once: they fill with hot
    /// experts of RAM as they come back, all of them, not a step's room on the link at a time
    /// (which is one here: the link is slow), as long as RAM holds hot experts. Swaps still
    /// wait for room.
    #[test]
    fn the_slots_a_loan_gives_back_fill_at_once() {
        let mut rig = rig(8, 16, Serve::Measured);
        {
            let model = rig.store.cost_model_mut().expect("a CPU half");
            for _ in 0..64 {
                model.record_upload(0, Duration::from_secs(1));
            }
        }
        let act = rig.act.clone();
        // Steps that use twelve experts of every layer, two at a time: the tiers learn them.
        for step in 0..30u32 {
            rig.store.begin_step();
            for layer in 0..LAYERS {
                let a = (step * 2) % 12;
                let groups = [group(a, 0.5), group(a + 1, 0.5)];
                rig.store.routed_rows(layer, step, 1, &groups, &act).unwrap();
                rig.until("served", |store| store.is_settled() && store.device(0).waiting() == 0);
            }
            rig.store.end_step().unwrap();
            rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        }
        let lendable = rig.store.lendable_pieces(0)[0];
        let loan = lend_one(&mut rig.store, lendable / 2).unwrap().expect("a loan");
        rig.until("lent", |store| store.is_settled());
        assert!(rig.store.scratch_ready(0).unwrap());
        rig.store.take_stats();
        rig.store.return_scratch(0).unwrap();
        let promotions = rig.store.stats().promotions;
        assert!(
            promotions >= u64::from(loan.slots).min(4),
            "{promotions} promotions into {} slots given back",
            loan.slots
        );
    }

    #[test]
    fn the_cpu_computes_what_a_step_misses() {
        let mut rig = rig(4, 4, Serve::Cpu);
        fast_link(&mut rig);
        let act = rig.act.clone();
        let groups = [group(3, 0.25), group(5, 0.75)];
        rig.store.begin_step();
        // Both experts are only in the file: claimed at once, computed
        // when their bytes are in RAM.
        assert_eq!(rig.store.routed_rows(0, 7, 1, &groups, &act).unwrap(), 2);
        let sums = rig.delivered(0, 7);
        assert_close(&sums, &reference(0, &act, &groups));
        let device = rig.store.device(0);
        assert_eq!(device.share(0).claimed, Some((7, vec![0, 1])));
        assert_eq!(
            (device.entry(0, 3), device.entry(0, 5)),
            (-1, -1),
            "a claimed expert gets no slot"
        );
        assert!(rig.store.in_ram(0, 3) && rig.store.in_ram(0, 5));
        assert_eq!(rig.store.stats().cpu_experts, 2);
        // The CPU's sums waited for the reads of their experts.
        assert_eq!(rig.store.stats().waits_last, [0, 0, 1, 0]);
        rig.store.end_step().unwrap();

        // Between the steps the heat promotes them: the bytes, then the entry.
        rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        let device = rig.store.device(0);
        for expert in [3, 5] {
            let entry = device.entry(0, expert);
            assert!(entry >= 0, "expert {expert} was not promoted");
            let slot = device.slot(rig.store.place(0).1, entry as u32);
            assert_eq!(slot[..64], matrix(0, 0, expert)[..64]);
        }
        // The next step finds them in their slots.
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 8, 1, &groups, &act).unwrap(), 0);
        rig.store.end_step().unwrap();
    }

    #[test]
    fn a_promotion_shows_only_when_no_step_is_at_its_layer() {
        let mut rig = rig(4, 4, Serve::Cpu);
        let act = rig.act.clone();
        let groups = [group(9, 1.0)];
        rig.store.begin_step();
        rig.store.routed_rows(1, 1, 1, &groups, &act).unwrap();
        let first = rig.delivered(1, 1);
        // The step ends and the promotion starts, but the device has not
        // run the copy when the next step is already at the layer.
        rig.store.end_step().unwrap();
        assert_eq!(rig.store.stats().promotions, 1);
        rig.store.begin_step();
        assert_eq!(
            rig.store.routed_rows(1, 2, 1, &groups, &act).unwrap(),
            1,
            "not on the device yet"
        );
        // The CPU serves it again, and although the bytes land meanwhile,
        // the entry does not appear while the layer is open.
        let second = rig.delivered(1, 2);
        assert_eq!(first, second);
        assert_eq!(rig.store.device(0).share(1).claimed, Some((2, vec![0])));
        rig.until("idle device", |store| store.device(0).waiting() == 0);
        assert_eq!(
            rig.store.device(0).entry(1, 9),
            -1,
            "an entry for a claimed expert of an open layer"
        );
        rig.store.end_step().unwrap();
        rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        assert!(rig.store.device(0).entry(1, 9) >= 0);
    }

    /// One step that claims `expert` of `layer` and ends: its promotion is
    /// then enqueued on the device, which has not run it.
    fn step_that_promotes(rig: &mut Rig, layer: usize, token: u32, expert: u32) {
        let act = rig.act.clone();
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(layer, token, 1, &[group(expert, 1.0)], &act).unwrap(), 1);
        rig.delivered(layer, token);
        let before = rig.store.stats().promotions;
        rig.store.end_step().unwrap();
        assert_eq!(rig.store.stats().promotions, before + 1);
        assert_eq!(rig.store.device(0).entry(layer, expert), -1, "the device has not run");
    }

    /// An expert of the running step gets a table entry or a claim, never
    /// both. Here the promotion lands, and the store hears of it, before the
    /// record of its layer arrives: the entry is written, and the record
    /// then finds a hit, so nothing is claimed and no sums are sent.
    #[test]
    fn a_promotion_that_lands_before_its_layers_record_is_a_hit() {
        let mut rig = rig(4, 4, Serve::Cpu);
        let act = rig.act.clone();
        step_that_promotes(&mut rig, 1, 1, 9);
        rig.store.begin_step();
        rig.until("the entry", |store| store.device(0).entry(1, 9) >= 0);
        let hits = rig.store.stats().hits;
        assert_eq!(rig.store.routed_rows(1, 2, 1, &[group(9, 1.0)], &act).unwrap(), 0);
        assert_eq!(rig.store.stats().hits, hits + 1);
        rig.until("idle device", |store| store.device(0).waiting() == 0);
        let share = rig.store.device(0).share(1);
        assert_eq!(share.claimed, Some((1, vec![0])), "a claim for an expert with an entry");
        assert_eq!(share.delivered.as_ref().map(|d| d.0), Some(1), "sums nobody waits for");
        rig.store.end_step().unwrap();
    }

    /// The other order: the record arrives first, so the expert is claimed,
    /// and its entry is held back while the kernels of the layer may run. It
    /// is written when the record of the next layer says that they have,
    /// which is before the step ends.
    #[test]
    fn a_claimed_experts_entry_follows_when_the_step_is_past_the_layer() {
        let mut rig = rig(4, 4, Serve::Cpu);
        let act = rig.act.clone();
        step_that_promotes(&mut rig, 0, 1, 9);
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 2, 1, &[group(9, 1.0)], &act).unwrap(), 1);
        rig.delivered(0, 2);
        rig.until("idle device", |store| store.device(0).waiting() == 0);
        assert_eq!(rig.store.device(0).entry(0, 9), -1, "an entry while the layer is open");
        // The record of layer 1: the step is past layer 0.
        assert_eq!(rig.store.routed_rows(1, 2, 1, &[group(4, 1.0)], &act).unwrap(), 1);
        rig.until("the entry", |store| store.device(0).entry(0, 9) >= 0);
        rig.delivered(1, 2);
        rig.store.end_step().unwrap();
    }

    /// A claim names groups by their place in the record, also when groups
    /// before them are in slots, and the sums are those of the claimed
    /// experts alone.
    #[test]
    fn a_claim_names_the_groups_of_the_record_that_have_no_slot() {
        let mut rig = rig(4, 4, Serve::Cpu);
        let act = rig.act.clone();
        step_that_promotes(&mut rig, 0, 1, 3);
        rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        assert!(rig.store.device(0).entry(0, 3) >= 0);
        let groups = [group(3, 0.25), group(5, 0.75)];
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 2, 1, &groups, &act).unwrap(), 1);
        let sums = rig.delivered(0, 2);
        assert_eq!(rig.store.device(0).share(0).claimed, Some((2, vec![1])));
        assert_close(&sums, &reference(0, &act, &groups[1..]));
        rig.store.end_step().unwrap();
    }

    /// A claimed expert keeps its RAM slot until the layer's sums are
    /// delivered, and the sums wait for every claimed expert: were the CPU
    /// to claim more experts that must be read than the layer has RAM slots,
    /// the last of them would wait for a slot that only its own delivery
    /// frees, and the layer would never end. The store claims what RAM can
    /// keep and uploads the rest.
    #[test]
    fn the_cpu_claims_no_more_than_ram_can_keep() {
        // Two RAM slots per layer; a step of two rows that needs three
        // experts, all of them only in the file.
        let mut rig = rig(4, 2, Serve::Cpu);
        let x: Vec<f32> = (0..512).map(|i| ((i * 37 % 255) as f32 - 127.0) / 64.0).collect();
        let act = quantize_activations(&x, 256);
        let pick = |expert: u32, rows: u32| {
            let mut weight = [0.0; 8];
            for (b, w) in weight.iter_mut().enumerate().take(2) {
                if rows >> b & 1 != 0 {
                    *w = 0.5;
                }
            }
            RouteGroup { expert, slot: None, rows, weight }
        };
        let groups = [pick(3, 0b01), pick(5, 0b10), pick(7, 0b11)];
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 1, 2, &groups, &act).unwrap(), 3);
        let sums = rig.delivered(0, 1);
        assert_eq!(rig.store.device(0).share(0).claimed, Some((1, vec![0, 1])));
        assert_close(&sums, &reference_rows(0, &act, &groups[..2], 2));
        // The third comes in a slot, as a miss does without a CPU half.
        rig.until("the third expert's entry", |store| store.device(0).entry(0, 7) >= 0);
        let stats = rig.store.stats();
        assert_eq!((stats.cpu_experts, stats.picks), (2, 3));
        assert!(stats.uploads >= 1);
        rig.store.end_step().unwrap();
    }

    /// A layer whose VRAM slots are fewer than its step picks: the store uploads into the
    /// slots whose experts the step did not pick, and the CPU takes the misses beyond them,
    /// although the cost model gives them to the link. An upload never evicts a pick: the
    /// kernels of the layer are about to read it.
    #[test]
    fn misses_beyond_the_slots_a_step_left_go_to_the_cpu() {
        let mut rig = rig(2, 8, Serve::Measured);
        let act = rig.act.clone();
        // A first step times an upload: the model then gives the link all misses but one.
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 1, 1, &[group(3, 1.0)], &act).unwrap(), 1);
        rig.until("the upload", |store| store.device(0).entry(0, 3) >= 0);
        rig.store.end_step().unwrap();
        rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        let slot3 = rig.store.device(0).entry(0, 3);
        assert!(slot3 >= 0);
        // Expert 3 holds one of the two slots and is picked again: one slot is left for three
        // misses, so one is uploaded and two are computed by the CPU.
        let groups = [group(3, 0.25), group(5, 0.25), group(6, 0.25), group(7, 0.25)];
        let cpu_before = rig.store.stats().cpu_experts;
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 2, 1, &groups, &act).unwrap(), 3);
        let sums = rig.delivered(0, 2);
        let (_, claimed) = rig.store.device(0).share(0).claimed.clone().expect("a claim");
        assert_eq!(claimed.len(), 2, "claimed {claimed:?}");
        assert_eq!(rig.store.stats().cpu_experts, cpu_before + 2);
        let claimed_groups: Vec<RouteGroup> = claimed.iter().map(|g| groups[*g as usize]).collect();
        assert_close(&sums, &reference(0, &act, &claimed_groups));
        rig.until("idle device", |store| store.device(0).waiting() == 0);
        let device = rig.store.device(0);
        assert_eq!(device.entry(0, 3), slot3, "the pick lost its slot");
        let slot = device.slot(rig.store.place(0).1, slot3 as u32);
        assert_eq!(slot[..64], matrix(0, 0, 3)[..64], "the pick's slot was written over");
        rig.store.end_step().unwrap();
    }

    /// Where the CPU computes what VRAM cannot take, a layer may have fewer VRAM slots than
    /// a step can pick, and RAM keeps the rest; without the CPU the plan is refused, and the
    /// refusal names the device and what each layer needs.
    #[test]
    fn a_plan_with_fewer_vram_slots_than_a_step_keeps_the_rest_in_ram() {
        let (_dir, map, reader, layers) = model();
        let slot_bytes = layers[0].slot_bytes;
        // Steps of 4 rows of 2 picks; room for 2 slots per layer in VRAM.
        let config = |cpu_serves: bool, ram: u64| {
            let mut config =
                StoreConfig::new(4, 2, vec![2 * slot_bytes * LAYERS as u64], u64::MAX / 4);
            config.vram_reserve = 0;
            config.cpu_serves = cpu_serves;
            if ram > 0 {
                config.ram_bytes = ram;
            }
            config
        };
        let error = StorePlan::new(config(false, 0), &map, &reader, &layers, None).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("device 0") && text.contains("needs 8 slots"), "{text}");
        let probe = StorePlan::new(config(true, 0), &map, &reader, &layers, None).unwrap();
        let ram_slot = probe.ram_slot_bytes(0);
        let plan = StorePlan::new(
            config(true, 8 * ram_slot * LAYERS as u64),
            &map,
            &reader,
            &layers,
            None,
        )
        .unwrap();
        for layer in 0..LAYERS {
            let (vram, ram) = plan.slots(layer);
            assert_eq!(vram, 2);
            assert!(ram >= 6, "layer {layer} has {ram} slots in RAM");
        }
        let error = StorePlan::new(
            config(true, 4 * ram_slot * LAYERS as u64),
            &map,
            &reader,
            &layers,
            None,
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("the RAM budget") && text.contains("needs 6 slots in RAM"), "{text}");
    }

    /// Two promotions chosen at the end of one step, in a layer whose RAM
    /// tier is as small as it may be: the first evicts a VRAM member that
    /// RAM no longer has, and the copy down takes the RAM slot of the second
    /// expert. The second promotion then finds nothing to upload and does
    /// not happen (the store used to stop here: it held a RAM copy that was
    /// gone); the expert stays with the CPU until it is read again.
    #[test]
    fn a_promotion_whose_ram_copy_gave_way_does_not_start() {
        let mut rig = rig(4, 2, Serve::Cpu);
        fast_link(&mut rig);
        let act = rig.act.clone();
        // One step that claims two experts and ends; `settled` says whether
        // to let the device and the reads run empty afterwards.
        let step = |rig: &mut Rig, token: u32, experts: [u32; 2]| {
            let groups = [group(experts[0], 0.5), group(experts[1], 0.5)];
            rig.store.begin_step();
            rig.store.routed_rows(0, token, 1, &groups, &act).unwrap();
            rig.delivered(0, token);
            rig.store.end_step().unwrap();
            rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        };
        // Four experts fill the VRAM slots of the layer, two per step.
        step(&mut rig, 1, [1, 2]);
        step(&mut rig, 2, [3, 4]);
        for expert in 1..=4 {
            assert!(rig.store.device(0).entry(0, expert) >= 0, "expert {expert} was not promoted");
        }
        // Two more, used three times: hot enough to take slots from two of them (a promotion
        // needs more than 1.25 times a member's heat and one more use).
        step(&mut rig, 3, [5, 6]);
        assert_eq!(rig.store.stats().evictions, 0, "as hot as the members: no slot yet");
        step(&mut rig, 4, [5, 6]);
        assert_eq!(rig.store.stats().evictions, 0, "not hotter by the rule: no slot yet");
        step(&mut rig, 5, [5, 6]);
        let stats = rig.store.stats();
        assert_eq!((stats.evictions, stats.demotions), (1, 1), "{stats:?}");
        let device = rig.store.device(0);
        let promoted: Vec<u32> = [5, 6].into_iter().filter(|e| device.entry(0, *e) >= 0).collect();
        assert_eq!(promoted.len(), 1, "one promotion, and one that had lost its bytes");
        // The other is still served: read again and computed on the CPU.
        let left = if promoted[0] == 5 { 6 } else { 5 };
        let groups = [group(left, 1.0)];
        rig.store.begin_step();
        assert_eq!(rig.store.routed_rows(0, 6, 1, &groups, &act).unwrap(), 1);
        assert_close(&rig.delivered(0, 6), &reference(0, &act, &groups));
        rig.store.end_step().unwrap();
    }

    /// The link is made far sooner than the CPU here, so the split gives it every miss; but an
    /// upload becomes a member only by a free slot or by beating the coldest member as a
    /// promotion would (`SwapRule`). The others go up on loan: into the slot of a member on
    /// loan, or of the coldest member, which is how the slots on loan come to be. The hot
    /// members keep their slots, and a miss that comes back often enough becomes a member.
    #[test]
    fn misses_go_up_on_loan_and_only_heat_makes_a_member() {
        let mut rig = rig(3, 8, Serve::Measured);
        let act = rig.act.clone();
        let types = (SHAPE.gate_ty, SHAPE.down_ty);
        let model = rig.store.cost_model_mut().expect("a CPU half");
        for _ in 0..64 {
            model.record_upload(0, Duration::from_micros(1));
            for pairs in [1, 2, 4, 8] {
                model.record_job(types, pairs, Duration::from_millis(10 * pairs as u64));
            }
        }
        let step = |rig: &mut Rig, token: u32, experts: &[u32]| {
            let groups: Vec<RouteGroup> = experts.iter().map(|e| group(*e, 0.5)).collect();
            rig.store.begin_step();
            rig.store.routed_rows(0, token, 1, &groups, &act).unwrap();
            rig.until("served", |store| {
                let device = store.device(0);
                experts.iter().all(|e| device.entry(0, *e) >= 0)
                    || device.share(0).delivered.as_ref().is_some_and(|(t, _)| *t == token)
            });
            rig.store.end_step().unwrap();
            rig.until("settled", |store| store.is_settled() && store.device(0).waiting() == 0);
        };
        let entry = |rig: &Rig, expert: u32| rig.store.device(0).entry(0, expert);
        // Experts 1, 2 and 3 take the free slots as members; 1 and 2 are used three times.
        step(&mut rig, 1, &[1, 2, 3]);
        step(&mut rig, 2, &[1, 2]);
        step(&mut rig, 3, &[1, 2]);
        let (one, two) = (entry(&rig, 1), entry(&rig, 2));
        assert!(one >= 0 && two >= 0 && entry(&rig, 3) >= 0);
        assert_eq!(rig.store.stats().loans, 0);
        // Three misses, each once: up on loan, the first into the slot of the coldest member,
        // the others into the slot on loan.
        for (token, expert) in [(10, 9), (11, 10), (12, 11)] {
            step(&mut rig, token, &[expert]);
            assert!(entry(&rig, expert) >= 0 && rig.store.on_loan(0, expert), "expert {expert}");
        }
        assert_eq!(rig.store.stats().loans, 3);
        assert_eq!((entry(&rig, 1), entry(&rig, 2)), (one, two), "a hot member lost its slot");
        assert_eq!((entry(&rig, 3), entry(&rig, 9), entry(&rig, 10)), (-1, -1, -1));
        assert_eq!(rig.store.loans(), 1, "one slot on loan, passed from miss to miss");
        assert_eq!(rig.store.stats().promotions, 0);
        // Expert 9 again and again: on loan while it does not beat the members (heat 3 each:
        // it needs more than 1.25 * 3 + 1); then a member.
        for token in 13..16 {
            step(&mut rig, token, &[9]);
            assert!(rig.store.on_loan(0, 9), "use {}", token - 11);
        }
        step(&mut rig, 16, &[9]);
        assert!(rig.store.in_vram(0, 9) && !rig.store.on_loan(0, 9), "use 5 made no member");
        assert!(entry(&rig, 1) >= 0 && entry(&rig, 2) >= 0);
    }

    #[test]
    fn the_cost_model_serves_a_miss_one_way_or_the_other() {
        let mut rig = rig(4, 4, Serve::Measured);
        let act = rig.act.clone();
        for (step, expert) in [2u32, 6, 11, 2, 13].into_iter().enumerate() {
            let token = 10 + step as u32;
            let groups = [group(expert, 1.0)];
            rig.store.begin_step();
            let missing = rig.store.routed_rows(0, token, 1, &groups, &act).unwrap();
            // The layer goes on when the expert has a slot or the sums came.
            rig.until("served", |store| {
                let device = store.device(0);
                device.entry(0, expert) >= 0
                    || device.share(0).delivered.as_ref().is_some_and(|(t, _)| *t == token)
            });
            if rig.store.device(0).entry(0, expert) < 0 {
                assert_eq!(missing, 1);
                assert_close(&rig.delivered(0, token), &reference(0, &act, &groups));
            }
            rig.store.end_step().unwrap();
        }
        let stats = rig.store.stats();
        assert_eq!(stats.picks, 5);
        assert!(stats.cpu_experts + stats.uploads >= 4, "every miss was served: {stats:?}");
    }
}

/// The CPU half next to the real kernels: layers of experts on a GPU, the
/// store with slots for half of them, and every layer of every step compared
/// with the same kernels on a pool that holds every expert.
///
/// This is the place where "an expert computed on the CPU changes nothing"
/// can be asked sharply. The rows a layer reads here do not depend on what
/// another layer gave, so the two results differ by the rounding of the
/// CPU's arithmetic against the device's and by nothing else: an expert that
/// is added twice or not at all, sums of another record, bytes of another
/// expert in a slot, all show at the layer and step where they happen, at
/// the size of an expert's output. A whole model cannot be asked that way:
/// its discrete choices (a router's pick, the blocks an attention layer
/// reads) turn a difference of rounding into another answer now and then,
/// with every expert on the device as well.
///
/// The tests open a GPU and are skipped unless `ARES_TEST_CUDA=1` is set;
/// `ARES_TEST_DEVICES` names the device (the first of a list; default 0).
mod device {
    use ares_core::Ledger;
    use ares_cuda::{
        Device, DeviceBuffer, Event, ExpertKernels, ExpertShape, Kernels, PinnedBuffer, RouteGroup,
        RouteRecord, Stream, Wait,
    };
    use ares_runtime::expert_store::cpu::{CpuPool, Placement, act_bytes};
    use ares_runtime::expert_store::{CudaSlots, Serve};

    use super::*;

    const LAYERS: usize = 4;
    const EXPERTS: u32 = 48;
    const MAX_ROWS: u32 = 8;
    const N: usize = 256;
    const SHAPE: ExpertShape = ExpertShape {
        n_embd: N as u32,
        n_ff: 64,
        n_experts: EXPERTS,
        top_k: 3,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };

    /// A small generator with a fixed sequence (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        /// Roughly normal, with a rare value ten times as large.
        fn activation(&mut self) -> f32 {
            let v: f32 = (0..6).map(|_| self.unit() - 0.5).sum();
            if self.next().is_multiple_of(61) { 10.0 * v } else { v }
        }
    }

    /// Matrix `part` of an expert: random blocks with scales between 1/64
    /// and 1/8 (both types keep an f16 scale at the start of a block).
    fn matrix(layer: usize, part: usize, expert: u32) -> Vec<u8> {
        let (len, block) = if part < 2 {
            (SHAPE.gate_bytes().unwrap() as usize, 82)
        } else {
            (SHAPE.down_bytes().unwrap() as usize, 18)
        };
        let mut rng =
            Rng(0x9E37_79B9 ^ ((layer * 131 + part * 71 + expert as usize * 29) as u64) << 20);
        let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        for blk in bytes.chunks_exact_mut(block) {
            let d = 0x2400u16 + (rng.next() % 0xC00) as u16;
            blk[..2].copy_from_slice(&d.to_le_bytes());
        }
        bytes
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
        let err: f64 =
            got.iter().zip(want).map(|(g, w)| (f64::from(*g) - f64::from(*w)).powi(2)).sum();
        let norm: f64 = want.iter().map(|w| f64::from(*w).powi(2)).sum();
        (err / norm.max(1e-30)).sqrt()
    }

    /// A step that stands still: kernels wait on the device for the host,
    /// and unwinding would free the buffers under them, which waits for the
    /// very kernels that wait. So the process says what it knows and ends.
    fn stuck(what: String) -> ! {
        eprintln!("{what}");
        std::process::abort()
    }

    /// What a run found.
    #[derive(Debug, Default)]
    struct Found {
        steps: usize,
        /// Rows of layers with a claimed expert, and without.
        rows_with_claims: usize,
        rows_without: usize,
        /// The largest relative difference of a row of a layer with claims
        /// from the same row with every expert on the device, and where.
        worst: f64,
        worst_at: (usize, usize, usize),
        /// Rows of layers with claims that came out bit for bit the same,
        /// and rows that differ by more than the order of float additions
        /// explains (a hidden value on the boundary between two codes).
        equal_with_claims: usize,
        beyond_float_order: usize,
        cpu_experts: u64,
        promotions: u64,
        uploads: u64,
        demotions: u64,
        demand_reads: u64,
    }

    /// Runs `steps` steps of `LAYERS` layers with `ram_slots` RAM slots per
    /// layer and misses served as `serve` says. The rows of a layer change
    /// every second step, so that what one step missed is on its way into a
    /// slot, or has just arrived, when the next step picks it again.
    fn run(device: &Device, serve: Serve, ram_slots: u64, steps: usize) -> Found {
        let dir = Scratch::new();
        let mut builder = GgufBuilder::new().str("general.architecture", "test");
        for layer in 0..LAYERS {
            for (part, (name, dims)) in [
                ("ffn_gate_exps", [N as u64, 64, u64::from(EXPERTS)]),
                ("ffn_up_exps", [N as u64, 64, u64::from(EXPERTS)]),
                ("ffn_down_exps", [64, N as u64, u64::from(EXPERTS)]),
            ]
            .into_iter()
            .enumerate()
            {
                let data: Vec<u8> = (0..EXPERTS).flat_map(|e| matrix(layer, part, e)).collect();
                let ty = if part < 2 { SHAPE.gate_ty } else { SHAPE.down_ty };
                builder = builder.tensor(&format!("blk.{layer}.{name}.weight"), &dims, ty, data);
            }
        }
        let path = dir.0.join("model.gguf");
        builder.write(&path).unwrap();
        let gguf = Gguf::open(&path).unwrap();
        let map =
            ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
        let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
        let slot_bytes = SHAPE.slot_bytes().unwrap();
        let (up_offset, down_offset) = (SHAPE.up_offset().unwrap(), SHAPE.down_offset().unwrap());
        let spec = LayerSpec {
            device: 0,
            offsets: [0, up_offset, down_offset],
            slot_bytes,
            resident: false,
        };
        let layers = [spec; LAYERS];
        // Slots for what one step of eight rows can pick, which is half of
        // a layer's experts.
        let vram_slots = u64::from(MAX_ROWS * SHAPE.top_k);
        let mut config = StoreConfig::new(
            MAX_ROWS,
            SHAPE.top_k,
            vec![vram_slots * slot_bytes * LAYERS as u64],
            u64::MAX / 4,
        );
        config.vram_reserve = 0;
        config.rebalance_every = 0;
        let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
        let ram_slot = probe.ram_slot_bytes(0) as usize;
        config.ram_bytes = ram_slots * (ram_slot * LAYERS) as u64;
        let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
        assert_eq!(plan.slots(0), (vram_slots as usize, ram_slots as usize));

        let mut vram = Ledger::new("CUDA", 1 << 30);
        let mut pinned = Ledger::new("pinned RAM", 1 << 30);
        let mut slots = CudaSlots::new(
            device,
            &[],
            &mut vram,
            &mut pinned,
            &plan.device_pools(0),
            LAYERS,
            EXPERTS,
        )
        .unwrap();
        slots.add_host_shares(&mut vram, &mut pinned, SHAPE.host_share_bytes(MAX_ROWS)).unwrap();
        let region = HostRegion::pinned(device, &mut pinned, plan.ram_bytes()).unwrap();
        let mut store = ExpertStore::new(plan, map, reader, vec![slots], region).unwrap();
        let pool = CpuPool::new(&Placement::unpinned(2), Duration::from_millis(1)).unwrap();
        store.enable_cpu(pool, vec![SHAPE; LAYERS], serve).unwrap();

        // The step's side of the device: one router for all layers (their
        // rows differ), per layer the rows, the record and the output, and a
        // second pool that holds every expert of every layer.
        let kernels = Kernels::load(device).unwrap();
        let experts = ExpertKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let side = Stream::new(device, 0).unwrap();
        let mut rng = Rng(0xC0FF_EE11);
        let router = DeviceBuffer::new(device, SHAPE.router_bytes()).unwrap();
        let weights: Vec<f32> = (0..EXPERTS as usize * N).map(|_| rng.unit() - 0.5).collect();
        router.write(0, &f32_bytes(&weights)).unwrap();
        let row_bytes = (MAX_ROWS as usize * N * 4) as u64;
        let buffer = |bytes: u64| DeviceBuffer::new(device, bytes).unwrap();
        let x: Vec<DeviceBuffer> = (0..LAYERS).map(|_| buffer(row_bytes)).collect();
        let act: Vec<DeviceBuffer> = (0..LAYERS)
            .map(|_| buffer(Kernels::quantized_activation_bytes(N as u32, MAX_ROWS)))
            .collect();
        let record: Vec<DeviceBuffer> =
            (0..LAYERS).map(|_| buffer(SHAPE.record_bytes(MAX_ROWS))).collect();
        let out: Vec<DeviceBuffer> = (0..LAYERS).map(|_| buffer(row_bytes)).collect();
        let out_all = buffer(row_bytes);
        let logits = buffer(SHAPE.logits_bytes(MAX_ROWS));
        let hidden = buffer(SHAPE.hidden_bytes(MAX_ROWS));
        let all_pool = buffer(LAYERS as u64 * u64::from(EXPERTS) * slot_bytes);
        let all_table = buffer(4 * LAYERS as u64 * u64::from(EXPERTS));
        for layer in 0..LAYERS {
            let mut entries = Vec::new();
            for expert in 0..EXPERTS {
                let slot = layer as u64 * u64::from(EXPERTS) + u64::from(expert);
                for (part, offset) in [0, up_offset, down_offset].into_iter().enumerate() {
                    all_pool
                        .write(slot * slot_bytes + offset, &matrix(layer, part, expert))
                        .unwrap();
                }
                entries.extend((slot as i32).to_le_bytes());
            }
            all_table.write(4 * layer as u64 * u64::from(EXPERTS), &entries).unwrap();
        }
        let record_stride = SHAPE.record_bytes(MAX_ROWS) as usize;
        let act_stride = act_bytes(N, MAX_ROWS as usize).next_multiple_of(64);
        let records_host = PinnedBuffer::new(device, LAYERS * record_stride).unwrap();
        let acts_host = PinnedBuffer::new(device, LAYERS * act_stride).unwrap();
        let event = || Event::new(device).unwrap();
        let routed: Vec<Event> = (0..LAYERS).map(|_| event()).collect();
        let copied: Vec<Event> = (0..LAYERS).map(|_| event()).collect();
        let done = event();

        // Every expert is asked for, as a session's fill does: the tiers
        // take what fits (the first reads to arrive get the VRAM slots), and
        // the fill is over when nothing has moved for a while.
        let all: Vec<u32> = (0..EXPERTS).collect();
        let lists: Vec<(usize, Vec<u32>)> = (0..LAYERS).map(|layer| (layer, all.clone())).collect();
        let mut fill = store.plan_listed_fill(&lists, FillOrder::File);
        let patience = Duration::from_secs(30);
        let mut last = (fill.left(), Instant::now());
        while !(fill.is_done() && store.is_settled())
            && last.1.elapsed() < Duration::from_millis(300)
        {
            store.advance_fill(&mut fill, FillPace::START).unwrap();
            if fill.left() != last.0 {
                last = (fill.left(), Instant::now());
            }
        }
        assert_eq!(store.stats().uploads, vram_slots * LAYERS as u64, "every slot was filled");
        store.take_stats();

        let mut found = Found::default();
        for step in 0..steps {
            // The rows of the layers: new every second step, one row or eight.
            let rows: u32 = if (step / 2).is_multiple_of(3) { MAX_ROWS } else { 1 };
            for (layer, x) in x.iter().enumerate() {
                let mut rng = Rng(0xA5A5_0000 + ((step / 2) * LAYERS + layer) as u64);
                let values: Vec<f32> = (0..rows as usize * N).map(|_| rng.activation()).collect();
                x.write(0, &f32_bytes(&values)).unwrap();
            }
            let token = step as u32 + 1;
            let n_record = SHAPE.record_bytes(rows) as usize;
            let n_act = act_bytes(N, rows as usize);
            store.begin_step();
            store.device_mut(0).fence(&stream).unwrap();
            for layer in 0..LAYERS {
                let slots = store.device(0);
                let (pool, _) = slots.pool(store.place(layer).1);
                // SAFETY: every buffer was sized for eight rows of this
                // shape; the table, the pool and the host share are the
                // store's, which keeps their rules; the pinned copies are
                // read after the event behind them.
                unsafe {
                    stream.fill(out[layer].ptr(), 0, row_bytes as usize).unwrap();
                    kernels
                        .quantize_activations(
                            &stream,
                            x[layer].ptr(),
                            act[layer].ptr(),
                            N as u32,
                            rows,
                        )
                        .unwrap();
                    experts
                        .route(
                            &stream,
                            &SHAPE,
                            router.ptr(),
                            act[layer].ptr(),
                            logits.ptr(),
                            slots.table(layer),
                            record[layer].ptr(),
                            rows,
                            token,
                        )
                        .unwrap();
                    routed[layer].record(&stream).unwrap();
                    side.wait_event(&routed[layer]).unwrap();
                    side.copy_to_host(
                        records_host.as_ptr().add(layer * record_stride).cast_mut(),
                        record[layer].ptr(),
                        n_record,
                    )
                    .unwrap();
                    side.copy_to_host(
                        acts_host.as_ptr().add(layer * act_stride).cast_mut(),
                        act[layer].ptr(),
                        n_act,
                    )
                    .unwrap();
                    copied[layer].record(&side).unwrap();
                    experts
                        .experts(
                            &stream,
                            &SHAPE,
                            pool,
                            slots.table(layer),
                            record[layer].ptr(),
                            act[layer].ptr(),
                            hidden.ptr(),
                            out[layer].ptr(),
                            slots.host_share(layer),
                            rows,
                            None,
                        )
                        .unwrap();
                }
            }
            done.record(&stream).unwrap();

            // The host follows the records, as a session does, while the
            // device is up to four layers ahead of it.
            let started = Instant::now();
            let mut claimed = [false; LAYERS];
            for layer in 0..LAYERS {
                while !copied[layer].is_done().unwrap() {
                    store.poll().unwrap();
                    if started.elapsed() > patience {
                        stuck(format!("step {step}: no record of layer {layer}: {store:?}"));
                    }
                }
                // SAFETY: the copies of this layer's record and rows have run.
                let (bytes, rows_host) = unsafe {
                    (
                        std::slice::from_raw_parts(
                            records_host.as_ptr().add(layer * record_stride),
                            n_record,
                        ),
                        std::slice::from_raw_parts(
                            acts_host.as_ptr().add(layer * act_stride),
                            n_act,
                        ),
                    )
                };
                let parsed = RouteRecord::parse(bytes, SHAPE.top_k, rows).expect("a record");
                assert_eq!(parsed.token(), token);
                let groups: Vec<RouteGroup> = parsed.groups().collect();
                let before = store.stats().cpu_experts;
                store.routed_rows(layer, token, rows as usize, &groups, rows_host).unwrap();
                claimed[layer] = store.stats().cpu_experts > before;
            }
            while !done.is_done().unwrap() {
                store.poll().unwrap();
                if started.elapsed() > patience {
                    stuck(format!(
                        "step {step} of {rows} rows does not end: {store:?}, {:?}",
                        store.stats()
                    ));
                }
            }
            store.end_step().unwrap();

            // The same records and rows on the pool that holds everything.
            let mut got = vec![0u8; rows as usize * N * 4];
            let mut want = vec![0u8; rows as usize * N * 4];
            for layer in 0..LAYERS {
                // SAFETY: as above, with the table and pool of all experts
                // and no host share.
                unsafe {
                    stream.fill(out_all.ptr(), 0, row_bytes as usize).unwrap();
                    experts
                        .experts(
                            &stream,
                            &SHAPE,
                            all_pool.ptr(),
                            all_table.ptr().offset(4 * layer as u64 * u64::from(EXPERTS)),
                            record[layer].ptr(),
                            act[layer].ptr(),
                            hidden.ptr(),
                            out_all.ptr(),
                            None,
                            rows,
                            None,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
                out[layer].read(0, &mut got).unwrap();
                out_all.read(0, &mut want).unwrap();
                let (got, want) = (bytes_f32(&got), bytes_f32(&want));
                for row in 0..rows as usize {
                    let (g, w) = (&got[row * N..(row + 1) * N], &want[row * N..(row + 1) * N]);
                    assert!(w.iter().any(|v| *v != 0.0), "a row without experts");
                    if !claimed[layer] {
                        // Uploaded or resident: the device computed it all.
                        assert!(
                            g == w,
                            "step {step} layer {layer} row {row}: no claim, another result"
                        );
                        found.rows_without += 1;
                        continue;
                    }
                    found.rows_with_claims += 1;
                    let error = rel_rms(g, w);
                    found.equal_with_claims += usize::from(g == w);
                    found.beyond_float_order += usize::from(error > 1e-5);
                    if error > found.worst {
                        (found.worst, found.worst_at) = (error, (step, layer, row));
                    }
                }
            }
            found.steps += 1;
        }
        let stats = store.take_stats();
        found.cpu_experts = stats.cpu_experts;
        found.promotions = stats.promotions;
        found.uploads = stats.uploads;
        found.demotions = stats.demotions;
        found.demand_reads = stats.demand_reads;
        found
    }

    fn gpu() -> Option<Device> {
        if std::env::var_os("ARES_TEST_CUDA").is_none() {
            eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
            return None;
        }
        let wanted = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0".into());
        let ordinal: u32 =
            wanted.split(',').next().and_then(|d| d.trim().parse().ok()).unwrap_or(0);
        Some(Device::open(ordinal, Wait::Spin).expect("open device"))
    }

    /// The pieces of two pools given up at the end of a step, with the RAM
    /// tier locked for a device by the driver (`cuMemHostRegister`), which
    /// refuses a second lock of a range.
    #[test]
    fn a_piece_that_is_given_up_is_not_registered_again_before_its_release_is_dropped() {
        let Some(device) = gpu() else { return };
        let region = |plan: &StorePlan| {
            let mut pinned = Ledger::new("pinned RAM", 1 << 30);
            HostRegion::pinned(&device, &mut pinned, plan.ram_bytes()).unwrap()
        };
        give_up_pieces_of_two_pools(region, || None);
    }

    fn check(serve: Serve, ram_slots: u64, steps: usize) -> Option<Found> {
        let device = gpu()?;
        let found = run(&device, serve, ram_slots, steps);
        eprintln!("  {serve:?}, {ram_slots} RAM slots per layer: {found:?}");
        assert!(found.cpu_experts > 100, "the run was to compute experts on the CPU: {found:?}");
        assert!(found.promotions > 20, "and to promote some between the steps: {found:?}");
        // An expert left out, added twice or computed from other bytes is
        // some hundredths of a row at the least. The CPU's rounding against
        // the device's is the order of float additions, seven digits down,
        // and where a hidden value sits on the boundary between two 8-bit
        // codes, one step of that value, three digits down.
        assert!(
            found.worst < 1e-2,
            "step {} layer {} row {}: {:.2e} from the result with every expert on the device",
            found.worst_at.0,
            found.worst_at.1,
            found.worst_at.2,
            found.worst
        );
        Some(found)
    }

    /// Every miss on the CPU, the bytes of every expert in RAM: claims, and
    /// promotions that are on their way or land while the next step runs.
    #[test]
    fn a_layer_with_experts_from_the_cpu_is_the_devices_layer() {
        check(Serve::Cpu, u64::from(EXPERTS), 120);
    }

    /// The store's own choice between the CPU and the link.
    #[test]
    fn a_layer_served_both_ways_is_the_devices_layer() {
        check(Serve::Measured, u64::from(EXPERTS), 120);
    }

    /// A RAM tier of six slots per layer, fewer than a step of eight rows
    /// misses in a layer: experts are read from the file for a claim, copied
    /// down when VRAM evicts them, give way in RAM while other claims keep
    /// theirs, and what the CPU cannot keep until its sums are delivered is
    /// uploaded instead.
    #[test]
    fn a_small_ram_tier_changes_nothing_either() {
        for serve in [Serve::Cpu, Serve::Measured] {
            let Some(found) = check(serve, 6, 120) else { return };
            assert!(found.demand_reads > 100 && found.demotions > 20, "{found:?}");
            if serve == Serve::Cpu {
                assert!(found.uploads > found.promotions, "nothing went to the link: {found:?}");
            }
        }
    }
}

/// A store with `transit` transit slots and `staging` staging slots besides its tiers.
fn transit_rig(transit: u32, staging: u32) -> Rig {
    rig(4, 4, |config| {
        config.transit_slots = vec![transit];
        config.staging_slots = staging;
    })
}

impl Rig {
    /// What has arrived in transit for `layer`, each checked against the file.
    fn take_transit(&mut self, layer: usize) -> Vec<u32> {
        let mut arrived = Vec::new();
        self.store.transit_take(layer, &mut arrived);
        let (pool, stride, _) = self.store.transit_pool(layer).unwrap();
        assert_eq!(stride, SLOT_BYTES);
        for (expert, slot) in &arrived {
            assert_eq!(
                self.store.device(0).slot(pool, *slot),
                expert_bytes(layer, *expert),
                "layer {layer} expert {expert} in transit slot {slot}"
            );
        }
        arrived.into_iter().map(|(expert, _)| expert).collect()
    }
}

#[test]
fn a_wanted_expert_passes_through_transit_and_the_tiers_stay() {
    let mut rig = transit_rig(2, 2);
    rig.step(&[&[1, 2]]);
    rig.store.take_stats();

    rig.store.begin_step();
    // Three experts for two transit slots: the third waits for a slot.
    rig.store.transit_want(0, &[7, 9, 11]).unwrap();
    assert_eq!(rig.store.transit_pending(0), 3);
    rig.settle();
    assert_eq!(rig.take_transit(0), [7, 9]);
    assert_eq!(rig.store.transit_pending(0), 1);
    // Asking again changes nothing.
    rig.store.transit_want(0, &[9, 11]).unwrap();
    rig.settle();
    assert!(rig.take_transit(0).is_empty());
    rig.store.transit_release(0, &[7, 9]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [11]);
    assert_eq!(rig.store.transit_pending(0), 0);
    rig.store.transit_release(0, &[11]).unwrap();
    rig.store.end_step().unwrap();
    rig.settle();

    // No tier took them, and the tiers hold what they held.
    for expert in [7, 9, 11] {
        assert!(!rig.store.in_vram(0, expert) && !rig.store.in_ram(0, expert));
    }
    rig.assert_usable(0, 1);
    rig.assert_usable(0, 2);
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_uploads, stats.transit_from_ram, stats.transit_reads), (3, 0, 3));
    assert_eq!(stats.transit_upload_bytes, 3 * 352);
    assert_eq!((stats.demand_reads, stats.uploads, stats.evictions, stats.demotions), (0, 0, 0, 0));
    assert_eq!((stats.transit_ahead_reads, stats.transit_ahead_used), (0, 0));
}

#[test]
fn an_expert_in_ram_goes_into_transit_from_its_ram_slot_and_stays_there() {
    // The members of the VRAM tier are far hotter than expert 7, which a
    // step brings from the file and leaves in the RAM tier.
    let mut rig = full_rig(100);
    let mut counts = [0u32; EXPERTS as usize];
    counts[7] = 5;
    for from_ram in [false, true] {
        rig.store.begin_step();
        rig.store.transit_want(0, &[7]).unwrap();
        rig.settle();
        assert_eq!(rig.take_transit(0), [7]);
        rig.store.record_counts(0, &counts, 1.0);
        rig.store.transit_release(0, &[7]).unwrap();
        rig.store.end_step().unwrap();
        rig.settle();
        assert!(rig.store.in_ram(0, 7) && !rig.store.in_vram(0, 7));
        let stats = rig.store.take_stats();
        assert_eq!(
            (stats.transit_uploads, stats.transit_from_ram, stats.transit_reads),
            (1, u64::from(from_ram), u64::from(!from_ram))
        );
        assert_eq!((stats.transit_adopted, stats.evictions, stats.demotions), (0, 0, 0));
    }
}

#[test]
fn a_guess_is_read_ahead_and_the_want_that_names_it_takes_it() {
    let mut rig = transit_rig(2, 4);
    rig.store.begin_step();
    // Guesses take at most half of staging: two of the three are read.
    rig.store.transit_guess(1, &[3, 4, 5], None).unwrap();
    rig.settle();
    assert_eq!(rig.store.stats().transit_ahead_reads, 2);
    assert!(rig.take_transit(1).is_empty(), "a guess is not uploaded");

    // The record names one of them and an expert nobody guessed.
    let named = rig.store.transit_want(1, &[4, 8]).unwrap();
    assert_eq!((named.guessed, named.unguessed), (1, 1));
    rig.settle();
    assert_eq!(rig.take_transit(1), [4, 8]);
    rig.store.transit_release(1, &[4, 8]).unwrap();
    // The layer is over: the guesses nobody named are dropped.
    rig.store.transit_forget(1).unwrap();
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_reads, stats.transit_ahead_used), (1, 1));
    assert_eq!(stats.transit_ahead_unused, stats.transit_ahead_reads - 1);
    assert_eq!(stats.transit_uploads, 2);
    for expert in [3, 4, 5, 8] {
        assert!(!rig.store.in_vram(1, expert) && !rig.store.in_ram(1, expert));
    }

    // Everything is free again: a full round of wants goes through.
    rig.store.begin_step();
    rig.store.transit_want(2, &[1, 2]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(2), [1, 2]);
    rig.store.transit_release(2, &[1, 2]).unwrap();
    rig.store.end_step().unwrap();
}

/// A store with transit whose four VRAM slots of layer 0 hold experts 1 to 4,
/// with the heat of 100 rows each but expert 4, which has that of `cold` rows.
fn full_rig(cold: u32) -> Rig {
    let mut rig = rig(4, 4, |config| {
        config.transit_slots = vec![2];
        config.staging_slots = 2;
        config.vram_reserve = 0;
    });
    rig.step(&[&[1, 2]]);
    rig.step(&[&[3, 4]]);
    let mut counts = [0u32; EXPERTS as usize];
    counts[1..4].fill(100);
    counts[4] = cold;
    rig.store.record_counts(0, &counts, 1.0);
    for expert in 1..=4 {
        rig.assert_usable(0, expert);
    }
    rig.store.take_stats();
    rig
}

#[test]
fn a_guess_the_ram_tier_holds_goes_up_into_a_free_transit_slot_before_its_record() {
    // Expert 7 comes from the file in one step and stays in the RAM tier.
    let mut rig = full_rig(100);
    let mut counts = [0u32; EXPERTS as usize];
    counts[7] = 5;
    rig.store.begin_step();
    rig.store.transit_want(0, &[7]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7]);
    rig.store.record_counts(0, &counts, 1.0);
    rig.store.transit_release(0, &[7]).unwrap();
    rig.store.end_step().unwrap();
    rig.settle();
    assert!(rig.store.in_ram(0, 7) && !rig.store.in_vram(0, 7));
    rig.store.take_stats();

    // Guessed for the layer, it goes up into a transit slot while nothing named waits for
    // one, and nothing hands it over before a record names it; then it is there at once.
    rig.store.begin_step();
    rig.store.transit_guess(0, &[7], None).unwrap();
    rig.settle();
    assert_eq!(rig.store.stats().transit_guess_uploads, 1);
    assert!(rig.take_transit(0).is_empty(), "a guess is not handed over");
    rig.store.transit_want(0, &[7]).unwrap();
    assert_eq!(rig.take_transit(0), [7]);
    rig.store.transit_release(0, &[7]).unwrap();
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_uploads, stats.transit_from_ram), (1, 1), "uploaded once");
    assert_eq!((stats.transit_guess_uploads_used, stats.transit_guess_uploads_unused), (1, 0));
    assert!(rig.store.in_ram(0, 7), "the RAM copy stays");

    // A guess no record names gives its slot back when its layer is done with.
    rig.store.begin_step();
    rig.store.transit_guess(0, &[7], None).unwrap();
    rig.settle();
    rig.store.transit_forget(0).unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_guess_uploads, stats.transit_guess_uploads_unused), (1, 1));
    // Both transit slots are free: a record that names two gets both.
    rig.store.transit_want(1, &[8, 9]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(1), [8, 9]);
    rig.store.transit_release(1, &[8, 9]).unwrap();
    rig.store.end_step().unwrap();
}

#[test]
fn an_expert_the_heat_ranks_high_stays_in_ram_after_its_transit() {
    // Every member of the VRAM tier is far hotter than what comes through.
    let mut rig = full_rig(100);
    rig.store.begin_step();
    rig.store.transit_want(0, &[7, 9]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7, 9]);
    // The step used expert 7 for five rows; of expert 9 the heat knows nothing.
    let mut counts = [0u32; EXPERTS as usize];
    counts[7] = 5;
    rig.store.record_counts(0, &counts, 1.0);
    rig.store.transit_release(0, &[7, 9]).unwrap();
    rig.store.end_step().unwrap();
    rig.settle();
    assert!(rig.store.in_ram(0, 7) && !rig.store.in_ram(0, 9));
    assert!(!rig.store.in_vram(0, 7));
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_admitted, stats.transit_adopted, stats.evictions), (1, 0, 0));

    // From RAM it goes up without a read, and it is the file's bytes.
    rig.store.begin_step();
    rig.store.transit_want(0, &[7]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7]);
    rig.store.transit_release(0, &[7]).unwrap();
    rig.store.end_step().unwrap();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_from_ram, stats.transit_reads), (1, 0));
}

#[test]
fn a_store_without_transit_says_so() {
    let mut rig = rig(4, 4, |_| {});
    assert!(rig.store.transit_pool(0).is_none());
    assert_eq!(rig.store.staging_slots(), 0);
    assert!(rig.store.transit_want(0, &[1]).is_err());
    // A guess is only a hint.
    rig.store.transit_guess(0, &[1], None).unwrap();
}

#[test]
fn an_expert_let_go_on_its_way_is_brought_when_it_is_named_again() {
    let mut rig = transit_rig(2, 2);
    rig.store.begin_step();
    rig.store.transit_want(0, &[7]).unwrap();
    // The step ends early (it failed) while the read is on its way, and the
    // next step names the expert again.
    rig.store.begin_step();
    assert_eq!(rig.store.transit_pending(0), 1, "let go, and still there until its read is back");
    rig.store.transit_want(0, &[7]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7]);
    rig.store.transit_release(0, &[7]).unwrap();
    rig.store.end_step().unwrap();
    rig.settle();

    // Nothing was lost on the way: both slots of each kind are free.
    rig.store.begin_step();
    rig.store.transit_want(1, &[3, 4]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(1), [3, 4]);
    rig.store.transit_release(1, &[3, 4]).unwrap();
    rig.store.end_step().unwrap();
}

#[test]
fn what_waits_for_a_slot_is_not_on_its_way() {
    let mut rig = transit_rig(2, 2);
    rig.store.begin_step();
    rig.store.transit_want(0, &[7, 9, 11]).unwrap();
    // Two reads are on their way; the third has no staging slot to be read to.
    assert_eq!((rig.store.transit_pending(0), rig.store.transit_moving(0)), (3, 2));
    rig.settle();
    // Two have arrived, and the third waits for what they hold: nothing moves
    // until a round has read them and they are released.
    assert_eq!((rig.store.transit_pending(0), rig.store.transit_moving(0)), (1, 0));
    assert_eq!(rig.take_transit(0), [7, 9]);
    rig.store.transit_release(0, &[7, 9]).unwrap();
    assert_eq!(rig.store.transit_moving(0), 1);
    rig.settle();
    assert_eq!(rig.take_transit(0), [11]);
    rig.store.transit_release(0, &[11]).unwrap();
    rig.store.end_step().unwrap();
}

#[test]
fn an_expert_hotter_than_the_coldest_member_takes_its_slot_from_transit() {
    // Expert 4 holds a slot with the heat of two rows (and one pick).
    let mut rig = full_rig(2);
    rig.store.begin_step();
    rig.store.transit_want(0, &[7, 9]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7, 9]);
    // Fifty rows of the step used expert 7 and three used expert 9.
    let mut counts = [0u32; EXPERTS as usize];
    counts[7] = 50;
    counts[9] = 3;
    rig.store.record_counts(0, &counts, 1.0);
    // The round that read them has not ended: nothing moves yet.
    assert!(!rig.store.in_vram(0, 7) && rig.store.in_vram(0, 4));
    rig.store.transit_release(0, &[7, 9]).unwrap();
    // The coldest member is gone at once, and the newcomer shows when the
    // device has copied it.
    assert!(!rig.store.in_vram(0, 4) && !rig.store.in_vram(0, 7));
    rig.settle();
    rig.assert_usable(0, 7);
    assert!(rig.store.in_vram(0, 7));
    assert_eq!(rig.store.device(0).entry(0, 4), -1);
    // Expert 9 is not hotter than a member by the margin: it does not stay in
    // VRAM. What left VRAM is in RAM, an upload away.
    assert!(!rig.store.in_vram(0, 9));
    assert!(rig.store.in_ram(0, 4));
    for expert in 1..=3 {
        rig.assert_usable(0, expert);
    }
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_adopted, stats.evictions, stats.uploads), (1, 1, 0));
    assert_eq!((stats.transit_uploads, stats.transit_reads), (2, 2));

    // The next step finds it in its slot, and transit is free again.
    rig.store.begin_step();
    assert_eq!(rig.store.routed(0, &[7]).unwrap(), 0);
    rig.store.transit_want(1, &[5, 6]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(1), [5, 6]);
    rig.store.transit_release(1, &[5, 6]).unwrap();
    rig.store.end_step().unwrap();
}

#[test]
fn an_expert_with_heat_takes_a_free_slot_from_transit() {
    // Two of the four slots of every layer are free.
    let mut rig = transit_rig(2, 2);
    rig.step(&[&[1, 2]]);
    rig.store.take_stats();
    rig.store.begin_step();
    rig.store.transit_want(0, &[7, 9]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7, 9]);
    let mut counts = [0u32; EXPERTS as usize];
    counts[7] = 1;
    rig.store.record_counts(0, &counts, 1.0);
    rig.store.transit_release(0, &[7, 9]).unwrap();
    rig.settle();
    rig.assert_usable(0, 7);
    rig.assert_usable(0, 1);
    rig.assert_usable(0, 2);
    // Of expert 9 the heat knows nothing: it is forgotten.
    assert!(!rig.store.in_vram(0, 9) && !rig.store.in_ram(0, 9));
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_adopted, stats.transit_admitted), (1, 0));
}

#[test]
fn a_swap_that_copies_nothing_down_needs_no_margin() {
    // Expert 4 holds a VRAM slot with the heat of one pick, and RAM has its
    // bytes: to let it go costs nothing.
    let mut rig = full_rig(0);
    assert!(rig.store.in_ram(0, 4));
    rig.store.begin_step();
    rig.store.transit_want(0, &[7, 9]).unwrap();
    rig.settle();
    assert_eq!(rig.take_transit(0), [7, 9]);
    // Two rows used expert 7 and one used expert 9: 7 is hotter than 4, but
    // not by the margin of a swap; 9 is as hot as 4.
    let mut counts = [0u32; EXPERTS as usize];
    counts[7] = 2;
    counts[9] = 1;
    rig.store.record_counts(0, &counts, 1.0);
    rig.store.transit_release(0, &[7, 9]).unwrap();
    rig.settle();
    rig.assert_usable(0, 7);
    assert!(!rig.store.in_vram(0, 4) && rig.store.in_ram(0, 4));
    // Expert 9 came from the file and is hotter than the copies RAM keeps of
    // what VRAM holds: it stays in RAM, though one row used it.
    assert!(!rig.store.in_vram(0, 9) && rig.store.in_ram(0, 9));
    rig.store.end_step().unwrap();
    rig.settle();
    let stats = rig.store.take_stats();
    assert_eq!((stats.transit_adopted, stats.transit_admitted), (1, 1));
    assert_eq!((stats.evictions, stats.demotions), (1, 0));
}

// --- a RAM tier that follows the memory it is allowed ------------------------

/// The slots every layer has in the RAM tier.
fn ram_slots(rig: &Rig) -> Vec<usize> {
    (0..LAYERS).map(|layer| rig.store.slots(layer).1).collect()
}

/// The experts of `layer` that are in the RAM tier.
fn in_ram(rig: &Rig, layer: usize) -> Vec<u32> {
    (0..EXPERTS).filter(|expert| rig.store.in_ram(layer, *expert)).collect()
}

#[test]
fn a_tier_that_starts_small_has_slots_for_what_is_locked_and_every_layer_its_share() {
    // 8 slots per layer are planned in pieces of 3, and half of the 8 pieces is locked.
    let rig = rig_in_pieces(3, 8, Some((3, 4)), recent_first);
    let (locked, whole) = rig.store.ram_bytes();
    assert!(locked > 0 && locked <= whole / 2 + whole / 8, "{locked} of {whole}");
    let slots = ram_slots(&rig);
    let total: usize = slots.iter().sum();
    assert_eq!(total, rig.store.ram_slots());
    assert!((6..=14).contains(&total), "{slots:?}");
    let (least, most) = (slots.iter().min().unwrap(), slots.iter().max().unwrap());
    assert!(most - least <= 1 && *least >= 2, "the layers share what is there: {slots:?}");
    assert_eq!(rig.store.ram_target(), locked);
}

#[test]
fn the_tier_gives_pieces_back_keeps_what_is_hot_and_takes_the_pieces_again() {
    let mut rig = rig_in_pieces(3, 8, Some((3, 8)), recent_first);
    assert_eq!(ram_slots(&rig), [8; LAYERS]);
    let whole = rig.store.ram_bytes().0;
    // Eight experts per layer pass through RAM; 6 and 7 are used most, and 0
    // and 1 are in VRAM at the end, so their copies in RAM are worth least.
    for pair in [[0, 1], [2, 3], [4, 5], [6, 7], [6, 7], [6, 7], [0, 1]] {
        rig.step(&[&pair[..], &pair[..], &pair[..]]);
    }
    for layer in 0..LAYERS {
        assert_eq!(in_ram(&rig, layer), [0, 1, 2, 3, 4, 5, 6, 7]);
    }
    rig.store.take_stats();

    // Half of it must go, and the owner hears of it.
    let heard = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let told = heard.clone();
    rig.store.on_shrink(Box::new(move |shrink| told.lock().unwrap().push(shrink)));
    rig.store.set_ram_target(whole / 2);
    let given = rig.store.shrink_ram(Duration::ZERO).unwrap();
    assert!(given > 0);
    let (locked, _) = rig.store.ram_bytes();
    assert!(locked <= whole / 2, "{locked} of {whole}");
    let want = ares_runtime::expert_store::RamShrink {
        pieces: given,
        bytes: whole - locked,
        held: locked,
        target: whole / 2,
        at_floor: false,
    };
    assert_eq!(*heard.lock().unwrap(), [want]);
    let released = rig.store.take_released();
    assert_eq!(released.len(), given);
    assert_eq!(released.iter().map(|piece| piece.len() as u64).sum::<u64>(), whole - locked);
    drop(released);
    let slots = ram_slots(&rig);
    assert!(slots.iter().all(|n| (3..=4).contains(n)), "every layer gave its share: {slots:?}");
    for (layer, n) in slots.iter().enumerate() {
        let kept = in_ram(&rig, layer);
        assert_eq!(kept.len(), *n, "what stays fills the slots: {kept:?}");
        assert!(kept.contains(&6) && kept.contains(&7), "the hottest stay: {kept:?}");
        assert!(!kept.contains(&0) && !kept.contains(&1), "copies of what VRAM holds go first");
    }
    let stats = rig.store.take_stats();
    assert_eq!(stats.pieces_released, given as u64);
    assert!(stats.piece_evictions >= 12, "{stats:?}");

    // What stayed is where the store looks for it: it comes up from RAM
    // with its own bytes (the step checks them), wherever it was copied to.
    rig.step(&[&[6, 7][..], &[6, 7][..], &[6, 7][..]]);
    rig.step(&[&[2, 3][..], &[2, 3][..], &[2, 3][..]]);
    let stats = rig.store.take_stats();
    assert!(stats.miss_from_ram >= 6, "{stats:?}");

    // The memory is free again: the pieces come back with empty slots, and
    // what is read next lands in them.
    rig.store.set_ram_target(u64::MAX);
    assert_eq!(rig.store.grow_ram(1).unwrap(), 1, "a piece at a time when it is asked so");
    assert_eq!(rig.store.grow_ram(usize::MAX).unwrap(), given - 1);
    assert_eq!(rig.store.ram_bytes().0, whole);
    assert_eq!(ram_slots(&rig), [8; LAYERS]);
    for pair in [[8, 9], [10, 11], [12, 13], [2, 3], [6, 7], [8, 9], [12, 13]] {
        rig.step(&[&pair[..], &pair[..], &pair[..]]);
    }
    assert_eq!(rig.store.take_stats().pieces_locked, given as u64);
}

#[test]
fn nothing_is_given_up_while_a_step_runs_and_the_end_of_the_step_gives_it() {
    let mut rig = rig_in_pieces(3, 8, Some((3, 8)), recent_first);
    for pair in [[0, 1], [2, 3], [4, 5]] {
        rig.step(&[&pair[..], &pair[..], &pair[..]]);
    }
    let whole = rig.store.ram_bytes().0;
    rig.store.begin_step();
    rig.store.routed(0, &[4, 5]).unwrap();
    rig.settle();
    rig.store.set_ram_target(whole / 2);
    assert_eq!(rig.store.shrink_ram(Duration::ZERO).unwrap(), 0);
    assert_eq!(rig.store.ram_bytes().0, whole);
    for layer in 1..LAYERS {
        rig.store.routed(layer, &[4, 5]).unwrap();
        rig.settle();
    }
    rig.store.end_step().unwrap();
    assert!(rig.store.ram_bytes().0 <= whole / 2);
    assert!(!rig.store.take_released().is_empty());
    rig.settle();
    rig.step(&[&[0, 1][..], &[2, 3][..], &[4, 5][..]]);
}

#[test]
fn a_tier_keeps_two_slots_for_every_layer_whatever_it_is_told() {
    let mut rig = rig_in_pieces(3, 8, Some((3, 8)), recent_first);
    rig.step(&[&[0, 1][..], &[0, 1][..], &[0, 1][..]]);
    rig.store.set_ram_target(0);
    assert!(rig.store.shrink_ram(Duration::ZERO).unwrap() > 0);
    assert!(rig.store.ram_at_floor());
    let slots = ram_slots(&rig);
    assert!(slots.iter().all(|n| *n >= 2), "{slots:?}");
    assert!(slots.iter().sum::<usize>() <= 2 * LAYERS + 4, "{slots:?}");
    assert!(rig.store.ram_bytes().0 > 0);
    // It asks no more until it is told something new, and it still serves.
    assert_eq!(rig.store.shrink_ram(Duration::ZERO).unwrap(), 0);
    for pair in [[2, 3], [4, 5], [0, 1]] {
        rig.step(&[&pair[..], &pair[..], &pair[..]]);
    }
}

/// Refuses every lock while it is full, as the books of the RAM limit refuse a piece of the
/// tier that the limit has no room for.
#[derive(Clone, Default)]
struct Books(Arc<std::sync::atomic::AtomicBool>);

impl Books {
    fn full(&self, full: bool) {
        self.0.store(full, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Locker for Books {
    unsafe fn lock(&mut self, _: *mut u8, len: usize) -> Result<Box<dyn Any + Send>, StoreError> {
        if self.0.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(StoreError::NoRoom(format!("no room for {len} bytes more")));
        }
        Ok(Box::new(()))
    }
}

/// A piece of the tier that the RAM limit has no room for is no error: the tier stays as it
/// is, serves on, and takes the piece once there is room (2026-11-25, `fix/model-sweep`:
/// `Flow::before_step` grows the tier by a piece, and the books' refusal became the step's
/// error, which took Flash-Next down).
#[test]
fn a_piece_the_ram_limit_has_no_room_for_leaves_the_tier_as_it_is() {
    let books = Books::default();
    let mut rig = rig_locked(3, 8, (3, 8), Box::new(books.clone()), recent_first);
    rig.step(&[&[0, 1][..], &[0, 1][..], &[0, 1][..]]);
    let whole = rig.store.ram_bytes().0;
    rig.store.set_ram_target(whole / 2);
    let given = rig.store.shrink_ram(Duration::ZERO).unwrap();
    assert!(given > 0);
    drop(rig.store.take_released());
    let held = rig.store.ram_bytes().0;
    // The tier may hold all again, but the limit has no room now.
    rig.store.set_ram_target(u64::MAX);
    books.full(true);
    assert_eq!(rig.store.grow_ram(1).unwrap(), 0, "the refusal is no error");
    assert_eq!(rig.store.ram_bytes().0, held, "the tier stays as it is");
    for pair in [[2, 3], [4, 5]] {
        rig.step(&[&pair[..], &pair[..], &pair[..]]);
    }
    books.full(false);
    assert_eq!(rig.store.grow_ram(usize::MAX).unwrap(), given, "and takes them with room");
    assert_eq!(rig.store.ram_bytes().0, whole);
    rig.step(&[&[6, 7][..], &[6, 7][..], &[6, 7][..]]);
}

// --- pools whose pieces differ in size, locked as the driver locks -----------

/// Ranges locked the way `cuMemHostRegister` locks them: a range that
/// overlaps one that is locked is refused (the driver's
/// `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`), and a lock ends when what
/// it returned is dropped.
#[derive(Clone, Default)]
struct DriverLocks(Arc<Mutex<Vec<(usize, usize)>>>);

impl DriverLocks {
    fn bytes(&self) -> usize {
        self.0.lock().unwrap().iter().map(|(start, end)| end - start).sum()
    }
}

struct DriverLock {
    locks: DriverLocks,
    start: usize,
}

impl Drop for DriverLock {
    fn drop(&mut self) {
        self.locks.0.lock().unwrap().retain(|(start, _)| *start != self.start);
    }
}

impl Locker for DriverLocks {
    unsafe fn lock(&mut self, ptr: *mut u8, len: usize) -> Result<Box<dyn Any + Send>, StoreError> {
        let (start, end) = (ptr as usize, ptr as usize + len);
        let mut locked = self.0.lock().unwrap();
        if locked.iter().any(|(s, e)| *s < end && start < *e) {
            return Err(StoreError::Plan(format!(
                "{len} bytes at {ptr:p} overlap a locked range: already registered"
            )));
        }
        locked.push((start, end));
        Ok(Box::new(DriverLock { locks: self.clone(), start }))
    }
}

/// Bytes of each matrix of an expert in the first four layers of
/// [`mixed_model`], and in the fifth.
const SMALL: usize = 4096;
const LARGE: usize = 10240;

/// Four layers of experts with matrices of `SMALL` bytes and a fifth with
/// matrices of `LARGE` bytes: the RAM tier has a pool for each size.
fn mixed_model(dir: &Scratch) -> Gguf {
    let mut builder = GgufBuilder::new().str("general.architecture", "test");
    for layer in 0..5 {
        let dims = if layer < 4 { [32u64, 32] } else { [64, 40] };
        let len = if layer < 4 { SMALL } else { LARGE };
        for (part, name) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
            let data: Vec<u8> =
                (0..EXPERTS).flat_map(|e| (0..len).map(move |i| byte(layer, part, e, i))).collect();
            let shape = [dims[0], dims[1], u64::from(EXPERTS)];
            builder =
                builder.tensor(&format!("blk.{layer}.{name}.weight"), &shape, GgmlType::F32, data);
        }
    }
    let path = dir.0.join("mixed.gguf");
    builder.write(&path).unwrap();
    Gguf::open(&path).unwrap()
}

/// One step of the store over [`mixed_model`] that routes `picks` in every
/// layer and checks that the device has their bytes.
fn mixed_step(store: &mut ExpertStore<HostSlots>, picks: &[u32]) -> Result<(), StoreError> {
    let settle = |store: &mut ExpertStore<HostSlots>| {
        let deadline = Instant::now() + Duration::from_secs(20);
        while {
            store.poll().unwrap();
            store.device_mut(0).run(usize::MAX);
            store.poll().unwrap();
            !store.is_settled()
        } {
            assert!(Instant::now() < deadline, "the store does not settle: {store:?}");
            std::thread::sleep(Duration::from_micros(200));
        }
    };
    store.begin_step();
    for layer in 0..5 {
        store.routed(layer, picks).unwrap();
        settle(store);
        let len = if layer < 4 { SMALL } else { LARGE };
        for expert in picks {
            let entry = store.device(0).entry(layer, *expert);
            assert!(entry >= 0, "layer {layer} expert {expert} has no entry");
            let (_, pool) = store.place(layer);
            let slot = store.device(0).slot(pool, entry as u32);
            for part in 0..3 {
                let want: Vec<u8> = (0..len).map(|i| byte(layer, part, *expert, i)).collect();
                assert_eq!(&slot[part * len..(part + 1) * len], &want[..], "layer {layer}");
            }
        }
    }
    store.end_step()?;
    settle(store);
    Ok(())
}

/// The end of a step gives a piece up and may then lock one: never a piece
/// whose release has not been dropped, because its lock still covers it and
/// the driver refuses to lock it again. This was how Flash-Next's requests
/// failed with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED` while the machine's
/// memory moved: the tier gave up a small piece and then a large one, and the
/// small one's pool, which then had the smallest share of its slots, wanted
/// its piece back, which fit in what the large one left below the target.
#[test]
fn a_piece_that_is_given_up_is_not_locked_again_before_its_release_is_dropped() {
    let locks = DriverLocks::default();
    let region = |plan: &StorePlan| {
        let locker = Box::new(locks.clone());
        HostRegion::elastic(plan.ram_bytes(), plan.ram_bytes(), locker).unwrap()
    };
    give_up_pieces_of_two_pools(region, || Some(locks.bytes() as u64));
}

/// The test above on [`mixed_model`], with the RAM tier in the region that
/// `region` makes for the plan; `locked` tells the bytes its locker holds,
/// where that is known.
fn give_up_pieces_of_two_pools(
    region: impl FnOnce(&StorePlan) -> HostRegion,
    locked: impl Fn() -> Option<u64>,
) {
    let dir = Scratch::new();
    let gguf = mixed_model(&dir);
    let map =
        ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
    let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
    let spec = |len: u64| LayerSpec {
        device: 0,
        offsets: [0, len, 2 * len],
        slot_bytes: 3 * len,
        resident: false,
    };
    let layers: Vec<LayerSpec> =
        (0..5).map(|layer| spec(if layer < 4 { SMALL } else { LARGE } as u64)).collect();
    let vram_round = 3 * (4 * SMALL + LARGE) as u64;
    let mut config = StoreConfig::new(1, 2, vec![4 * vram_round], u64::MAX / 4);
    config.vram_reserve = 1;
    config.rebalance_every = 0;
    recent_first(&mut config);
    let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
    let (small, large) = (probe.ram_slot_bytes(0) as usize, probe.ram_slot_bytes(4) as usize);
    // Six slots per layer: 24 small ones in pieces of three, 6 large ones in
    // pieces of one, which are smaller than a piece of three small ones.
    config.ram_bytes = 6 * (4 * small + large) as u64;
    config.ram_piece = 3 * small as u64;
    let (small_piece, large_piece) =
        ((3 * small).next_multiple_of(4096), large.next_multiple_of(4096));
    assert!(large * 2 > 3 * small, "one large slot per piece: {small} {large}");
    assert!(large_piece + 4096 <= small_piece, "{small_piece} {large_piece}");
    let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
    assert_eq!((plan.slots(0).1, plan.slots(4).1), (6, 6));
    let device = HostSlots::new(&plan.device_pools(0), 5, EXPERTS);
    let region = region(&plan);
    let mut store = ExpertStore::new(plan, map, reader, vec![device], region).unwrap();
    let whole = store.ram_bytes().0;
    let agree = |store: &ExpertStore<HostSlots>, more: usize, what: &str| {
        if let Some(bytes) = locked() {
            assert_eq!(bytes, store.ram_bytes().0 + more as u64, "{what}");
        }
    };
    agree(&store, 0, "every piece is locked at the start");
    for picks in [[0, 1], [2, 3], [4, 5], [0, 1]] {
        mixed_step(&mut store, &picks).unwrap();
    }

    // While a step runs the machine takes a little more than a large piece.
    // The step's end gives up the last large piece (both pools have all
    // their slots: the higher piece goes) and then the last small piece;
    // the large pool has the smaller share then, and its piece fits in what
    // is left below the target.
    store.set_ram_target(whole - (large_piece + 4096) as u64);
    mixed_step(&mut store, &[6, 7]).expect("the end of the step locks nothing twice");
    let released = store.take_released();
    let lens: Vec<usize> = released.iter().map(|piece| piece.len()).collect();
    assert_eq!(lens, [large_piece, small_piece], "the pieces given up, in their order");
    let (after, _) = store.ram_bytes();
    assert_eq!(after, whole - (large_piece + small_piece) as u64, "nothing was locked again");
    agree(&store, large_piece + small_piece, "the releases hold their locks");
    // The owner locks what the tier may hold between steps: not the large
    // piece while its release holds its lock.
    assert_eq!(store.grow_ram(1).unwrap(), 0, "a piece on its way back is not locked");
    drop(released);
    agree(&store, 0, "until they are dropped");

    // Once the releases are dropped, the lock between steps takes the large
    // piece again, and the tier serves from it.
    assert_eq!(store.grow_ram(1).unwrap(), 1);
    mixed_step(&mut store, &[8, 9]).unwrap();
    assert_eq!(store.ram_bytes().0, after + large_piece as u64);
    agree(&store, 0, "the books and the locks agree");
    for picks in [[10, 11], [0, 1], [12, 13], [6, 7]] {
        mixed_step(&mut store, &picks).unwrap();
    }
}

/// VRAM lent by the slots: the last slots of the pool leave every layer, what
/// stays in them moves into slots the layers gave (their coldest), a step
/// while the loan runs never uses them, and given back they are the layers'
/// again, empty.
#[test]
fn a_loan_of_vram_takes_the_last_slots_of_the_pool_and_gives_them_back_empty() {
    // Eight slots per layer in VRAM, all of RAM; a loan leaves a layer two:
    // its top-k.
    let mut rig = rig(8, 16, |_| {});
    for step in 0..2 {
        let picks: Vec<Vec<u32>> = (0..LAYERS)
            .map(|l| (0..8).map(|e| (e + step * 3 + l as u32) % EXPERTS).collect())
            .collect();
        let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
        rig.step(&picks);
    }
    let (_, pool) = rig.store.place(0);
    let slots_before: Vec<usize> = (0..LAYERS).map(|l| rig.store.slots(l).0).collect();
    assert_eq!(slots_before, [8, 8, 8]);
    assert_eq!(rig.store.lendable_pieces(0), [(3 * (8 - 2)) as u64 * SLOT_BYTES]);
    assert_eq!(
        lend_one(&mut rig.store, 19 * SLOT_BYTES).unwrap(),
        None,
        "more than the layers have"
    );
    // Ten slots: the last ten of the pool of 24.
    let loan = lend_one(&mut rig.store, 10 * SLOT_BYTES - 5).unwrap().expect("a loan");
    assert_eq!((loan.pool, loan.first, loan.slots, loan.stride), (pool, 14, 10, SLOT_BYTES));
    rig.settle();
    assert!(rig.store.scratch_ready(0).unwrap());
    assert_eq!(rig.store.scratch_loans(0), Some(vec![loan]));
    // No table entry names a lent slot, and every expert VRAM holds is usable
    // where its entry says.
    let mut held = 0;
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let entry = rig.store.device(0).entry(layer, expert);
            if entry >= 0 {
                assert!((entry as u32) < loan.first, "layer {layer} expert {expert} in {entry}");
                rig.assert_usable(layer, expert);
                held += 1;
            }
        }
    }
    assert_eq!(held, 24 - 10, "what the layers kept");
    // A step while the loan runs: what it misses goes into slots outside it.
    let picks: Vec<Vec<u32>> = (0..LAYERS).map(|l| vec![15 - l as u32, 14 - l as u32]).collect();
    let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
    rig.step(&picks);
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let entry = rig.store.device(0).entry(layer, expert);
            assert!(entry < loan.first as i32, "layer {layer} expert {expert} in {entry}");
        }
    }
    // Given back: the slots are the layers' again and fill.
    rig.store.return_scratch(0).unwrap();
    assert_eq!(rig.store.scratch_loans(0), None);
    let slots_after: Vec<usize> = (0..LAYERS).map(|l| rig.store.slots(l).0).collect();
    assert_eq!(slots_after, slots_before);
    let picks: Vec<Vec<u32>> = (0..LAYERS).map(|_| (8..16).collect()).collect();
    let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
    rig.step(&picks);
    let stats = rig.store.take_stats();
    assert_eq!(stats.scratch_loans, 1);
    assert!(stats.scratch_moves > 0, "what stayed in the lent slots moved");
}

/// Every expert whose table entry names a slot is usable there, and no entry names a slot
/// below `dead`; returns how many experts VRAM holds.
fn entries_above(rig: &Rig, dead: u32) -> usize {
    let mut held = 0;
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let entry = rig.store.device(0).entry(layer, expert);
            if entry >= 0 {
                assert!(entry as u32 >= dead, "layer {layer} expert {expert} in {entry}");
                rig.assert_usable(layer, expert);
                held += 1;
            }
        }
    }
    held
}

/// The VRAM tier gives the memory of pieces back from the low end of its pool when it may hold
/// less, moves what stays into the slots the layers give, keeps the slots a loan takes, and
/// takes the pieces again, empty, when it may hold more.
#[test]
fn the_vram_tier_gives_pieces_back_and_takes_them_again() {
    // Eight slots per layer, one pool of 24; pieces of four slots.
    let piece = 4 * SLOT_BYTES;
    let mut rig = rig_vram_in_pieces(8, 16, piece, |_| {});
    let whole = 24 * SLOT_BYTES;
    assert_eq!(rig.store.vram_elastic(0), (whole, whole));
    for step in 0..3 {
        let picks: Vec<Vec<u32>> = (0..LAYERS)
            .map(|l| (0..8).map(|e| (e + step * 3 + l as u32) % EXPERTS).collect())
            .collect();
        let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
        rig.step(&picks);
    }
    // Every layer keeps a slot free for an expert that arrives before its layer runs.
    assert_eq!(entries_above(&rig, 0), 21);
    // Two pieces less: slots 0 to 7 go, the layers give eight slots, the coldest, and what
    // lay in the first eight moves into the slots given.
    rig.store.set_vram_target(0, whole - 2 * piece);
    assert_eq!(rig.store.shrink_vram(0, Duration::from_secs(1)).unwrap(), 2);
    assert_eq!(rig.store.vram_elastic(0), (whole - 2 * piece, whole));
    assert_eq!((0..LAYERS).map(|l| rig.store.slots(l).0).sum::<usize>(), 16);
    // The memory goes back once the device has done the moves.
    assert!(rig.store.device(0).is_mapped(0, 0), "the moves have not run");
    rig.settle();
    assert!(!rig.store.device(0).is_mapped(0, 0) && !rig.store.device(0).is_mapped(0, 1));
    assert!(rig.store.device(0).is_mapped(0, 2));
    assert_eq!(entries_above(&rig, 8), 16);
    // Steps go on in what is left (the device faults on a slot without memory).
    for step in 0..3 {
        let picks: Vec<Vec<u32>> =
            (0..LAYERS).map(|l| vec![(step + l as u32) % EXPERTS, 15 - step]).collect();
        let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
        rig.step(&picks);
    }
    entries_above(&rig, 8);
    // A loan takes the last slots, as far as they have memory: 16 live, ten above the floor of
    // a loan (two a layer).
    assert_eq!(rig.store.lendable_pieces(0), [10 * SLOT_BYTES]);
    // What loans take stays: with six slots kept for them, the tier gives until the layers'
    // floor would not leave them.
    rig.store.keep_for_loans(0, &[6 * SLOT_BYTES]);
    rig.store.set_vram_target(0, 0);
    let given = rig.store.shrink_vram(0, Duration::from_secs(1)).unwrap();
    rig.settle();
    assert!(given >= 1 && rig.store.vram_at_floor(0), "{given}");
    let (held, _) = rig.store.vram_elastic(0);
    assert!(held >= 6 * SLOT_BYTES + 6 * SLOT_BYTES, "the floors and the loan: {held}");
    let dead = ((whole - held) / SLOT_BYTES) as u32;
    entries_above(&rig, dead);
    let loan = lend_one(&mut rig.store, 6 * SLOT_BYTES).unwrap().expect("the loan still fits");
    assert!(loan.first >= dead, "{loan:?} below {dead}");
    rig.settle();
    rig.store.return_scratch(0).unwrap();
    // Room again: the pieces come back, the highest first, as far as the device has room.
    rig.store.set_vram_target(0, u64::MAX);
    rig.store.device_mut(0).set_room(Some(1));
    assert_eq!(rig.store.grow_vram(0, 10).unwrap(), 1);
    rig.store.device_mut(0).set_room(None);
    let left = (whole - rig.store.vram_elastic(0).0) / piece;
    assert_eq!(rig.store.grow_vram(0, 10).unwrap() as u64, left);
    assert_eq!(rig.store.vram_elastic(0), (whole, whole));
    assert_eq!((0..LAYERS).map(|l| rig.store.slots(l).0).sum::<usize>(), 24);
    let picks: Vec<Vec<u32>> = (0..LAYERS).map(|_| (8..16).collect()).collect();
    let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
    rig.step(&picks);
    entries_above(&rig, 0);
    let stats = rig.store.take_stats();
    assert_eq!(stats.vram_pieces_taken, stats.vram_pieces_released);
}

/// A slot across the edge of two pieces needs both: it goes with the first of them.
#[test]
fn a_slot_across_two_pieces_goes_with_either() {
    // Pieces of 1000 bytes over slots of 352: slot 2 lies in pieces 0 and 1.
    let mut rig = rig_vram_in_pieces(8, 16, 1000, |_| {});
    let picks: Vec<Vec<u32>> =
        (0..LAYERS).map(|l| (0..8).map(|e| e + l as u32).collect()).collect();
    let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
    rig.step(&picks);
    rig.store.set_vram_target(0, 24 * SLOT_BYTES - 1000);
    assert_eq!(rig.store.shrink_vram(0, Duration::from_secs(1)).unwrap(), 1);
    rig.settle();
    // Slots 0, 1 and 2 have no memory now.
    assert_eq!((0..LAYERS).map(|l| rig.store.slots(l).0).sum::<usize>(), 21);
    entries_above(&rig, 3);
    let few: Vec<Vec<u32>> = (0..LAYERS).map(|l| vec![9 + l as u32, 15]).collect();
    let few: Vec<&[u32]> = few.iter().map(Vec::as_slice).collect();
    rig.step(&few);
    entries_above(&rig, 3);
    // Taken again, all three come back.
    rig.store.set_vram_target(0, u64::MAX);
    assert_eq!(rig.store.grow_vram(0, 1).unwrap(), 1);
    assert_eq!((0..LAYERS).map(|l| rig.store.slots(l).0).sum::<usize>(), 24);
}

/// A pool whose memory stays as it is gives nothing back.
#[test]
fn a_pool_without_pieces_gives_nothing_back() {
    let mut rig = rig(8, 16, |_| {});
    assert_eq!(rig.store.vram_elastic(0), (0, 0));
    rig.store.set_vram_target(0, 0);
    assert_eq!(rig.store.shrink_vram(0, Duration::ZERO).unwrap(), 0);
    assert_eq!((0..LAYERS).map(|l| rig.store.slots(l).0).sum::<usize>(), 24);
}

/// Lends `bytes` of device 0 in one piece, of its largest pool.
fn lend_one(
    store: &mut ExpertStore<HostSlots>,
    bytes: u64,
) -> Result<Option<ScratchLoan>, StoreError> {
    Ok(store.lend_pieces(0, &[bytes])?.and_then(|loans| loans.first().copied()))
}

/// Loans one after the other, with steps between them that move experts around: every loan
/// finds a home for what stays in its slots, and no table entry names a lent slot.
#[test]
fn loans_one_after_the_other_find_a_home_for_what_stays_in_their_slots() {
    // Slots move between layers by their use (as in the engine), between loans and while one
    // runs.
    let mut rig = rig(8, 16, |config| config.rebalance_every = 1);
    let picks_of = |step: u32| -> Vec<Vec<u32>> {
        (0..LAYERS)
            .map(|l| (0..8).map(|e| (e * 3 + step * 5 + l as u32) % EXPERTS).collect())
            .collect()
    };
    for round in 0..12u32 {
        for step in 0..3 {
            let picks = picks_of(round * 3 + step);
            let picks: Vec<&[u32]> = picks.iter().map(Vec::as_slice).collect();
            rig.step(&picks);
        }
        let slots = 4 + (round % 3) as u64 * 3;
        let loan = lend_one(&mut rig.store, slots * SLOT_BYTES).unwrap().expect("a loan");
        rig.settle();
        assert!(rig.store.scratch_ready(0).unwrap());
        // A step on the loan: its misses go into slots outside it.
        let picks = picks_of(round * 7 + 1);
        let picks: Vec<&[u32]> = picks.iter().map(|p| &p[..2]).collect();
        rig.step(&picks);
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                let entry = rig.store.device(0).entry(layer, expert);
                assert!(entry < loan.first as i32, "round {round}: layer {layer} expert {expert}");
            }
        }
        rig.store.return_scratch(0).unwrap();
    }
}

/// A loan of VRAM in pieces: the last slots of each pool it asks of (the
/// pools of slots of two sizes lie apart), every layer keeping its floor, no
/// table entry naming a lent slot of either pool while the loan runs; given
/// back, every pool's slots are the layers' again.
#[test]
fn a_loan_takes_the_last_slots_of_every_pool_it_asks_and_gives_them_back() {
    let dir = Scratch::new();
    let gguf = mixed_model(&dir);
    let map =
        ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
    let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
    let spec = |len: u64| LayerSpec {
        device: 0,
        offsets: [0, len, 2 * len],
        slot_bytes: 3 * len,
        resident: false,
    };
    let layers: Vec<LayerSpec> =
        (0..5).map(|layer| spec(if layer < 4 { SMALL } else { LARGE } as u64)).collect();
    // Eight slots per layer in VRAM, sixteen in RAM.
    let vram_round = 3 * (4 * SMALL + LARGE) as u64;
    let mut config = StoreConfig::new(1, 2, vec![8 * vram_round], u64::MAX / 4);
    config.vram_reserve = 1;
    config.rebalance_every = 0;
    let probe = StorePlan::new(config.clone(), &map, &reader, &layers, None).unwrap();
    let (small, large) = (probe.ram_slot_bytes(0) as usize, probe.ram_slot_bytes(4) as usize);
    config.ram_bytes = 16 * (4 * small + large) as u64;
    let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
    let device = HostSlots::new(&plan.device_pools(0), 5, EXPERTS);
    let region = HostRegion::unpinned(plan.ram_bytes(), plan.block());
    let mut store = ExpertStore::new(plan, map, reader, vec![device], region).unwrap();
    for picks in [[0, 1], [2, 3], [4, 5], [6, 7]] {
        mixed_step(&mut store, &picks).unwrap();
    }
    // The small pool (four layers) holds more bytes than the large one: it is
    // the first piece. Each layer keeps two slots: a row's two picks.
    let (small_slot, large_slot) = (3 * SMALL as u64, 3 * LARGE as u64);
    assert_eq!(store.lendable_pieces(0), [4 * 6 * small_slot, 6 * large_slot]);
    let pieces = [7 * small_slot, 3 * large_slot - 10];
    let loans = store.lend_pieces(0, &pieces).unwrap().expect("a loan");
    assert_eq!(loans.len(), 2);
    let (small_pool, large_pool) = (store.place(0).1, store.place(4).1);
    assert_eq!(
        loans.iter().map(|l| (l.pool, l.first, l.slots)).collect::<Vec<_>>(),
        [(small_pool, 32 - 7, 7), (large_pool, 8 - 3, 3)]
    );
    while !store.scratch_ready(0).unwrap() {
        store.poll().unwrap();
        store.device_mut(0).run(usize::MAX);
    }
    assert_eq!(store.scratch_loans(0), Some(loans.clone()));
    let pools: Vec<usize> = (0..5).map(|layer| store.place(layer).1).collect();
    let lent = |layer: usize, entry: i32| {
        loans.iter().any(|l| l.pool == pools[layer] && entry >= 0 && entry as u32 >= l.first)
    };
    for layer in 0..5 {
        for expert in 0..EXPERTS {
            let entry = store.device(0).entry(layer, expert);
            assert!(!lent(layer, entry), "layer {layer} expert {expert} in lent slot {entry}");
        }
    }
    // A step while the loan runs uses the slots outside it.
    store.end_step().ok();
    mixed_step(&mut store, &[8, 9]).unwrap();
    for layer in 0..5 {
        for expert in 0..EXPERTS {
            let entry = store.device(0).entry(layer, expert);
            assert!(!lent(layer, entry), "layer {layer} expert {expert} in lent slot {entry}");
        }
    }
    store.return_scratch(0).unwrap();
    assert_eq!(store.scratch_loans(0), None);
    let slots: Vec<usize> = (0..5).map(|l| store.slots(l).0).collect();
    assert_eq!(slots, [8; 5]);
    for picks in [[10, 11], [12, 13]] {
        mixed_step(&mut store, &picks).unwrap();
    }
}
