//! The small model file and the store on slots in ordinary memory
//! (`HostSlots`) that the expert store's suites share: copies and table
//! writes run only when a test says so, so what the store does can be
//! checked at every point a device could be at.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Config, ExpertMap, Reader};
use ares_runtime::expert_store::{
    ExpertStore, HostRegion, LayerSpec, Locker, StoreConfig, StorePlan,
};
use ares_runtime::testing::HostSlots;

pub const LAYERS: usize = 3;
pub const EXPERTS: u32 = 16;
/// Bytes of gate, up and down of one expert, and where they lie in a slot.
pub const LENS: [usize; 3] = [128, 128, 96];
pub const OFFSETS: [u64; 3] = [0, 128, 256];
pub const SLOT_BYTES: u64 = 352;

/// A scratch directory that goes away even when the test fails.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ares-expert-rig-{}-{}",
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
pub fn byte(layer: usize, part: usize, expert: u32, i: usize) -> u8 {
    (layer * 131 + part * 71 + expert as usize * 29 + i * 7 + 3) as u8
}

pub fn expert_bytes(layer: usize, expert: u32) -> Vec<u8> {
    let mut slot = vec![0u8; SLOT_BYTES as usize];
    for part in 0..3 {
        for i in 0..LENS[part] {
            slot[OFFSETS[part] as usize + i] = byte(layer, part, expert, i);
        }
    }
    slot
}

/// A model of `LAYERS` layers of `EXPERTS` experts with F32 matrices.
pub fn model(dir: &Scratch) -> Gguf {
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
pub fn recent_first(config: &mut StoreConfig) {
    config.decay_every = 1;
    config.heat_decay = 0.5;
}

pub struct Rig {
    pub store: ExpertStore<HostSlots>,
    /// The directory of the model file, which lives as long as the store.
    pub dir: Scratch,
}

/// A store with `vram_slots` slots per layer in VRAM and `ram_slots` in RAM.
pub fn rig(vram_slots: u64, ram_slots: u64, tune: impl FnOnce(&mut StoreConfig)) -> Rig {
    rig_in_pieces(vram_slots, ram_slots, None, tune)
}

/// [`rig`] with a RAM tier that lies in pieces of `pieces.0` slots, of which
/// `pieces.1` pieces are locked at the start.
pub fn rig_in_pieces(
    vram_slots: u64,
    ram_slots: u64,
    pieces: Option<(usize, usize)>,
    tune: impl FnOnce(&mut StoreConfig),
) -> Rig {
    rig_reading(vram_slots, ram_slots, pieces, Config::default(), tune)
}

/// [`rig_in_pieces`] with a reader of the file made with `reader`.
pub fn rig_reading(
    vram_slots: u64,
    ram_slots: u64,
    pieces: Option<(usize, usize)>,
    reader: Config,
    tune: impl FnOnce(&mut StoreConfig),
) -> Rig {
    build(vram_slots, ram_slots, pieces, reader, None, None, tune)
}

/// [`rig`] whose device has the memory of its pools in pieces of `piece`
/// bytes, which the VRAM tier gives back and takes again.
pub fn rig_vram_in_pieces(
    vram_slots: u64,
    ram_slots: u64,
    piece: u64,
    tune: impl FnOnce(&mut StoreConfig),
) -> Rig {
    build(vram_slots, ram_slots, None, Config::default(), None, Some(piece), tune)
}

/// [`rig_in_pieces`] whose RAM tier `locker` locks, as the books of the RAM
/// limit and a device do.
pub fn rig_locked(
    vram_slots: u64,
    ram_slots: u64,
    pieces: (usize, usize),
    locker: Box<dyn Locker>,
    tune: impl FnOnce(&mut StoreConfig),
) -> Rig {
    build(vram_slots, ram_slots, Some(pieces), Config::default(), Some(locker), None, tune)
}

fn build(
    vram_slots: u64,
    ram_slots: u64,
    pieces: Option<(usize, usize)>,
    reader: Config,
    locker: Option<Box<dyn Locker>>,
    vram_piece: Option<u64>,
    tune: impl FnOnce(&mut StoreConfig),
) -> Rig {
    let dir = Scratch::new();
    let gguf = model(&dir);
    let map =
        ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]).unwrap();
    let reader = Reader::open_gguf(&gguf, reader).unwrap();
    let layers =
        [LayerSpec { device: 0, offsets: OFFSETS, slot_bytes: SLOT_BYTES, resident: false };
            LAYERS];
    // What a RAM slot takes is the plan's business; ask it with a large
    // budget first.
    let mut config =
        StoreConfig::new(1, 2, vec![vram_slots * SLOT_BYTES * LAYERS as u64], u64::MAX / 4);
    config.vram_reserve = 1;
    config.rebalance_every = 0;
    tune(&mut config);
    // Transit and staging are room of their own, next to the tiers'.
    let (transit, staging) =
        (config.transit_slots.first().copied().unwrap_or(0), config.staging_slots);
    config.vram_bytes[0] += u64::from(transit) * SLOT_BYTES;
    let probe = StorePlan::new(
        StoreConfig { staging_slots: 0, ..config.clone() },
        &map,
        &reader,
        &layers,
        None,
    )
    .unwrap();
    let ram_slot = probe.ram_slot_bytes(0) as usize;
    config.ram_bytes = (ram_slots * LAYERS as u64 + u64::from(staging)) * ram_slot as u64;
    if let Some((per_piece, _)) = pieces {
        config.ram_piece = (per_piece * ram_slot) as u64;
    }
    let plan = StorePlan::new(config, &map, &reader, &layers, None).unwrap();
    assert_eq!(plan.slots(0), (vram_slots as usize, ram_slots as usize));
    assert_eq!(plan.transit_slots(0), (transit, staging));
    let device = HostSlots::new(&plan.device_pools(0), LAYERS, EXPERTS);
    let device = match vram_piece {
        Some(piece) => device.with_pieces(piece),
        None => device,
    };
    let mut region = match locker {
        Some(locker) => HostRegion::elastic(plan.ram_bytes(), plan.ram_bytes(), locker).unwrap(),
        None => HostRegion::unpinned(plan.ram_bytes(), plan.block()),
    };
    if let Some((per_piece, start)) = pieces {
        // A piece is whole pages.
        region = region.starting_with(start * (per_piece * ram_slot).next_multiple_of(4096));
    }
    Rig { store: ExpertStore::new(plan, map, reader, vec![device], region).unwrap(), dir }
}

impl Rig {
    /// Lets reads finish and the device run until nothing is wanted.
    pub fn settle(&mut self) {
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

    /// Lets reads finish and the device run until nothing is on its way
    /// (wants and guesses may be left: what a fill waits for).
    pub fn quiet(&mut self) {
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

    /// One step that routes `picks[l]` in layer `l`.
    pub fn step(&mut self, picks: &[&[u32]]) {
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
    pub fn assert_usable(&self, layer: usize, expert: u32) {
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
