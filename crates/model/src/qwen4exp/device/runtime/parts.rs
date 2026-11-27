//! How the runtime's device threads open their parts: each thread opens its
//! own part through the stages of `open.rs` and meets the others twice
//! ([`Meeting`]): the plan of the expert store needs every device's room, and
//! the store needs every device's slots. The last to arrive at the second
//! meeting builds the store, the loans of wide steps and passes, and the
//! [`Flow`] the threads share; each part then learns where its experts are and
//! where its peer receives.

use std::sync::{Arc, Mutex};

use ares_core::Ledger;
use ares_cuda::{ENTRY_CELLS, MAX_BATCH};
use ares_runtime::expert_store::{
    CudaSlots, ExpertStore, HostRegion, PIECE, RamShrink, Serve, StoreError, StorePlan,
};
use ares_runtime::{DeviceContext, DeviceModel, Model, ModelError, PinnedRam};

use super::super::board::Board;
use super::super::feed::{MtpExperts, build_store, cpu_pool, device_slots, expert_map, places};
use super::super::flow::{Flow, Routing};
use super::super::invalid;
use super::super::misses::SpanPicks;
use super::super::mtp::MtpBlock;
use super::super::ngram::{self, NgramFeed};
use super::super::open::{self, PartReport, Peer};
use super::super::part::{ExpertAt, Frame, Options, Part, PartShape, PassSizes, Pieces};
use super::super::placement::PartPlan;
use super::super::prune::PruneMask;
use super::super::resume::Resume;
use super::device::Qwen4ExpDevice;
use super::meeting::Meeting;
use super::{Qwen4Exp, pass, widest_rows};
use crate::blocks::far::FarPool;
use crate::qwen3vl::TowerAt;
use crate::qwen4exp::weights::{MtpWeights, Weights};

/// What the parts hand each other while they open.
pub(super) struct Build {
    /// Per part: the VRAM it has left for slots after its dense parts and the
    /// store's tables, what it tells its peer, and its route records.
    room: Vec<u64>,
    peers: Vec<Peer>,
    boards: Vec<Option<Arc<Board>>>,
    plan: Option<PlanBox>,
    slots: Vec<Option<CudaSlots>>,
    places: Vec<Vec<ExpertAt>>,
    /// Per part: the expert slots it has in VRAM, and their bytes.
    n_slots: Vec<usize>,
    slot_bytes: Vec<u64>,
    flow: Option<Arc<Flow>>,
    /// Whether a part holds the MTP block, and where its experts are.
    mtp: bool,
    mtp_place: Option<ExpertAt>,
    /// What the tiers hold, for the routers' pruning.
    prune_mask: Option<Arc<PruneMask>>,
    /// Per part: the bytes of VRAM a step wider than its own scratch runs on,
    /// where the expert slots of its device lend them, and where its input of
    /// a half block lies in them; the multiprocessors of its device; the rows
    /// of the widest step, as far as the slots can lend for it.
    lent: Vec<u64>,
    lent_at: Vec<Option<Pieces>>,
    lent_raw: Vec<u64>,
    /// Per part: the bytes each piece of its loan holds, and what each pool of its device can
    /// lend (what the scratch was laid out in).
    lent_pieces: Vec<Vec<u64>>,
    lendable: Vec<Vec<u64>>,
    multiprocessors: Vec<u32>,
    lone_rows: u32,
    /// The most rows of a prompt read layer by layer (0: none).
    pass_rows: u32,
    pass_sizes: Vec<PassSizes>,
    /// Per part: the K and V of its attention layers beyond VRAM, if any.
    far: Vec<Option<Arc<Mutex<FarPool>>>>,
}

impl Build {
    /// What the parts of `n` devices have handed each other before any has
    /// arrived.
    pub(super) fn new(n: usize) -> Self {
        Self {
            room: vec![0; n],
            peers: vec![Peer::default(); n],
            boards: vec![None; n],
            plan: None,
            slots: (0..n).map(|_| None).collect(),
            places: vec![Vec::new(); n],
            n_slots: vec![0; n],
            slot_bytes: vec![0; n],
            flow: None,
            mtp: false,
            mtp_place: None,
            prune_mask: None,
            lent: vec![0; n],
            lent_at: vec![None; n],
            lent_raw: vec![0; n],
            lent_pieces: vec![Vec::new(); n],
            lendable: vec![Vec::new(); n],
            multiprocessors: vec![0; n],
            lone_rows: 0,
            pass_rows: 0,
            pass_sizes: Vec::new(),
            far: vec![None; n],
        }
    }
}

/// The plan of the expert store and what it was made from.
struct PlanBox {
    plan: StorePlan,
    map: ares_io::ExpertMap,
    reader: ares_io::Reader,
    /// The bytes of the RAM tier that are locked at the start.
    ram_start: u64,
}

impl Qwen4Exp {
    /// Opens the part of the device thread `cx` is of, the parts of the other
    /// devices at the same time on their threads.
    pub(super) fn open_part(
        &self,
        mut cx: DeviceContext<'_>,
    ) -> Result<Box<dyn DeviceModel>, ModelError> {
        self.check_setup(&cx)?;
        let (c, o) = (&self.config, &self.opts.part);
        let (index, n) = (cx.index, self.placement.n_devices());
        let tower = self.open_tower(&mut cx)?;
        let weights = Weights::load(&self.gguf, c).map_err(|e| invalid(e.to_string()))?;
        let mut part = open::open_part(
            c,
            &weights,
            o,
            &self.placement,
            index,
            cx.device,
            cx.ledger,
            &self.rows,
        )?;
        // The K and V of the pages beyond VRAM: host memory the store's flow
        // locks as the pages come into use.
        let far = part.open_far(cx.ram)?.map(|pool| Arc::new(Mutex::new(pool)));
        // The drafters, next to the part that holds the head: before the
        // room for slots is counted, their weights and rings are taken from
        // the experts'.
        let mut block = self.open_mtp(&part, index, &mut cx)?;
        let dflash_embedding = self.open_dflash(&mut part, index, &mut cx)?;

        // The room each device has left decides the slots of the store.
        let room = open::slot_room(cx.ledger.free(), &self.layout, c, &self.opts);
        let (peer, board, holds_mtp) = (Peer::of(&part), part.board(), block.is_some());
        let (lent, multiprocessors) = (part.lent_bytes(), cx.device.info().multiprocessors);
        self.meeting.meet(
            n,
            |build| {
                build.lent[index] = lent;
                build.far[index] = far.clone();
                build.multiprocessors[index] = multiprocessors;
                build.room[index] = room;
                build.peers[index] = peer;
                build.boards[index] = Some(board);
                build.mtp |= holds_mtp;
                Ok(())
            },
            |build| self.plan_store(build, cx.ram),
        )?;

        // The slots of this device; the last to arrive builds the store.
        let pools = {
            let build = self.meeting.lock();
            let planned = build.value.plan.as_ref().ok_or_else(|| invalid("no plan was made"))?;
            planned.plan.device_pools(index)
        };
        let mut pinned = Ledger::new("pinned RAM (booked by the engine)", u64::MAX / 2);
        let rows = o.max_rows.min(MAX_BATCH);
        let layout = &self.layout;
        let slots =
            device_slots(cx.device, cx.ledger, &mut pinned, &pools, layout, self.opts.serve, rows)?;
        self.meeting.meet(
            n,
            |build| {
                build.slots[index] = Some(slots);
                Ok(())
            },
            |build| self.build_flow(build, cx.ram, cx.device),
        )?;

        // This part learns where its experts are, and its peer's ports.
        let (flow, places, lendable, mask, peer, slots, n_slots) = {
            let build = self.meeting.lock();
            let build = &build.value;
            if let Some(block) = &mut block {
                let at = build.mtp_place.ok_or_else(|| invalid("the MTP block has no slots"))?;
                block.set_experts(at)?;
            }
            let _ = self.lone_rows.set(build.lone_rows);
            let _ = self.pass_span.set(build.pass_rows);
            // Two devices on every layer: what the other tells this one.
            let peer = self.placement.tensor.is_some().then(|| {
                let other = 1 - index;
                let lent =
                    build.lent_at[other].as_ref().map(|at| at.address(build.lent_raw[other]));
                Peer { lent, ..build.peers[other].clone() }
            });
            (
                build.flow.clone().ok_or_else(|| invalid("no flow was made"))?,
                build.places[index].clone(),
                build.lendable[index].clone(),
                build.prune_mask.clone(),
                peer,
                build.slot_bytes[index],
                build.n_slots[index],
            )
        };
        part.set_experts(places)?;
        if flow.pass_rows > 0 {
            if !part.takes_passes() {
                return Err(invalid("a part cannot read prompts layer by layer"));
            }
            part.prepare_passes()?;
        }
        let PartPlan { role, first, shares, head } = self.placement.part(c, index);
        part.lend_rows(c, &PartShape { role, first, shares, head }, self.max_rows(), &lendable)?;
        if let Some(mask) = mask {
            part.set_prune_mask(mask);
        }
        if let Some(peer) = peer {
            open::connect(&mut part, &self.placement, c, index, peer)?;
        }
        let report = PartReport::of(&part, index, slots, n_slots, cx.ledger.reserved());
        self.reports.lock().unwrap_or_else(|e| e.into_inner()).push(report);
        // The checkpoints of this device: what the model said it keeps of a
        // sequence here is what it copies.
        let feeds = part.ngram_layer().map_or(0, |_| ngram::saved_bytes(c.ngram.as_ref()));
        let resume = Resume::open(cx.device, cx.ram, cx.checkpoints, &part, block.as_ref(), feeds)?;
        if resume.bytes() != self.declared[index].checkpoint {
            return Err(invalid(format!(
                "a checkpoint of device part {index} takes {} bytes; the model declared {}",
                resume.bytes(),
                self.declared[index].checkpoint
            )));
        }
        let routing = Routing::of(&self.placement, c, index);
        // The host's copy of a pass's picks, for the part that routes: taken here, for the
        // widest pass, since a pass runs inside a step.
        let picks = (flow.pass_rows > 0 && routing.mixer.is_none())
            .then(|| SpanPicks::new(cx.device, flow.pass_rows, flow.top_k).map(Arc::new))
            .transpose()?;
        let mut device = Qwen4ExpDevice::new(
            self,
            index,
            part,
            flow,
            routing,
            resume,
            tower,
            block,
            dflash_embedding,
        );
        device.picks = picks;
        Ok(Box::new(device))
    }

    /// An error when the runtime was set up for other slots, pages or rows
    /// than the model.
    fn check_setup(&self, cx: &DeviceContext<'_>) -> Result<(), ModelError> {
        let o = &self.opts.part;
        let rows = o.max_rows.max(self.opts.pass_rows);
        if cx.max_slots != o.max_slots
            || cx.pages != o.pages()
            || cx.table_stride != o.table_stride()
            || cx.cells_per_page != ENTRY_CELLS
            || cx.max_rows != rows
        {
            return Err(invalid(format!(
                "the runtime was set up for {} slots, {} pages of {} cells, tables of {} \
                 entries and steps of {} rows; the model for {}, {}, {ENTRY_CELLS}, {} and {rows}",
                cx.max_slots,
                cx.pages,
                cx.cells_per_page,
                cx.table_stride,
                cx.max_rows,
                o.max_slots,
                o.pages(),
                o.table_stride(),
            )));
        }
        Ok(())
    }

    /// The image tower, on the part that runs it, before the part and its
    /// expert slots take what is left of the device.
    fn open_tower(&self, cx: &mut DeviceContext<'_>) -> Result<Option<TowerAt>, ModelError> {
        match &self.text.projector {
            Some((projector, device)) if *device == cx.index => {
                cx.ledger.reserve("image tower", projector.vram_bytes())?;
                let tower = projector
                    .load(cx.device, cx.kernels)
                    .map_err(|e| invalid(format!("the image tower: {e}")))?;
                Ok(Some(tower))
            }
            _ => Ok(None),
        }
    }

    /// The MTP block, on the part that holds the head.
    fn open_mtp(
        &self,
        part: &Part,
        index: usize,
        cx: &mut DeviceContext<'_>,
    ) -> Result<Option<MtpBlock>, ModelError> {
        let (Some(draft), Some(output)) = (&self.mtp, part.output()) else { return Ok(None) };
        if index != self.head_device() {
            return Ok(None);
        }
        let c = &self.config;
        let head = MtpWeights::load(draft, c).map_err(|e| invalid(e.to_string()))?;
        let block = MtpBlock::open(
            c,
            &head,
            &self.opts.part,
            index,
            output,
            part.head_tokens(),
            &self.view,
            self.rows.clone(),
            self.mtp_watch.clone(),
            cx.device,
            cx.ledger,
        )?;
        *self.mtp_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(block.report());
        Ok(Some(block))
    }

    /// The DFlash drafter, on the part that holds the head; the embedding rows
    /// of its newest tokens.
    fn open_dflash(
        &self,
        part: &mut Part,
        index: usize,
        cx: &mut DeviceContext<'_>,
    ) -> Result<Option<Arc<crate::blocks::rows::HostTable>>, ModelError> {
        let Some(file) = &self.dflash else { return Ok(None) };
        if index != self.head_device() {
            return Ok(None);
        }
        let c = &self.config;
        let mut mask = vec![0f32; c.n_embd];
        self.rows.decode(file.config.mask_token, &mut mask)?;
        part.open_dflash(
            file,
            cx.device,
            cx.ledger,
            cx.max_slots,
            c.n_layer,
            c.n_vocab,
            self.view.rows,
            mask,
        )?;
        Ok(Some(self.rows.clone()))
    }

    /// The plan of the expert store, by the last part to arrive at the first
    /// meeting: the slots each device's room holds (its experts and a 64th
    /// more, or the MTP block's too), the tier in RAM the engine's limit
    /// leaves, and the transit of wide steps.
    fn plan_store(&self, build: &mut Build, ram: &PinnedRam) -> Result<(), ModelError> {
        let (c, layout, n) = (&self.config, &self.layout, self.placement.n_devices());
        let vram: Vec<u64> = (0..n)
            .map(|d| {
                let own = layout.mtp_bytes(d)
                    + if self.placement.tensor.is_some() {
                        layout.expert_bytes
                    } else {
                        self.placement.devices[d].experts
                    };
                build.room[d].min(own + (own >> 6))
            })
            .collect();
        let reader = super::super::feed::reader(&self.gguf, self.mtp.as_deref())?;
        let head = self
            .mtp
            .as_ref()
            .map(|draft| MtpWeights::load(draft, c))
            .transpose()
            .map_err(|e| invalid(e.to_string()))?;
        let mtp_experts = self.mtp.as_ref().zip(head.as_ref()).map(|(gguf, weights)| MtpExperts {
            gguf,
            weights,
            device: self.head_device(),
        });
        let map = expert_map(&self.gguf, c, layout.per_layer, mtp_experts.as_ref())?;
        // The tier is planned for all it can ever hold, what the limit
        // leaves and the experts without a slot in VRAM can use, and
        // starts with what it may hold now. Staging is sized for the
        // tier as it starts.
        let reach = ram.obtainable_experts().min(layout.ram_wanted(vram.iter().sum()));
        ram.want_experts(reach, 0);
        let ram_start = ram.experts_allowance().min(reach);
        let mut config =
            open::store_config(c, &self.opts, layout, &reader, vram, ram_start, &build.room);
        config.ram_bytes = reach;
        config.ram_piece = self.opts.ram_piece.unwrap_or(PIECE) as u64;
        let plan = StorePlan::new(config, &map, &reader, &layout.specs, None).map_err(|e| {
            // With both devices on every layer each holds a share of the
            // experts of every layer, and every one of those needs its slots.
            let shared = self.placement.tensor.is_some()
                && matches!(e, StoreError::Plan(ref text) if text.starts_with("device"));
            invalid(if shared {
                format!(
                    "{e}. Both devices hold a share of the experts of each of the {} layers; \
                     with the layers split between the devices (--split layer) each holds the \
                     slots of its own layers only",
                    c.n_layer
                )
            } else {
                e.to_string()
            })
        })?;
        build.plan = Some(PlanBox { plan, map, reader, ram_start });
        Ok(())
    }

    /// The store and the flow, by the last part to arrive at the second
    /// meeting, with what the parts learn of them: where their experts are,
    /// the loans of wide steps and passes, the pruning mask.
    fn build_flow(
        &self,
        build: &mut Build,
        ram: &PinnedRam,
        device: &ares_cuda::Device,
    ) -> Result<(), ModelError> {
        let (c, layout, n) = (&self.config, &self.layout, self.placement.n_devices());
        let serve = self.opts.serve;
        let PlanBox { plan, map, reader, ram_start } =
            build.plan.take().ok_or_else(|| invalid("no plan was made"))?;
        let slots: Vec<CudaSlots> = build.slots.iter_mut().filter_map(Option::take).collect();
        for d in 0..n {
            build.slot_bytes[d] = plan.vram_bytes(d);
        }
        let reach = plan.ram_bytes() as u64;
        let tier =
            HostRegion::elastic(plan.ram_bytes(), ram_start as usize, ram.expert_pieces(device))
                .map_err(|e| invalid(e.to_string()))?;
        // Cores for the threads that drive the step: the conductor and
        // one for each device.
        let cpu = (serve != Serve::Stream)
            .then(|| cpu_pool(self.opts.cpu_workers, 1 + n).map(|pool| (pool, serve)))
            .transpose()?;
        let store = build_store(plan, layout, map, reader, slots, tier, cpu)?;
        let ram_fixed = store.ram_fixed_bytes();
        self.lend(build, &store)?;
        for d in 0..n {
            let PartPlan { first, shares, .. } = self.placement.part(c, d);
            let layers = first..first + shares.len();
            build.places[d] = places(&store, layers.clone(), layout, d);
            build.n_slots[d] = layers.map(|l| store.slots(layout.store_layer(d, l)).0).sum();
        }
        let boards = build
            .boards
            .iter()
            .map(|b| b.clone().ok_or_else(|| invalid("a part has no route records")))
            .collect::<Result<Vec<_>, _>>()?;
        if self.mtp.is_some() {
            let (device, pool) = store.place(0);
            let slots = store.device(device);
            build.mtp_place = Some(ExpertAt {
                table: slots.table(0),
                pool: slots.pool(pool).0,
                host: slots.host_share(0),
            });
        }
        let (mut feed, mask) =
            open::open_feed(store, boards, &self.placement, layout, c, &self.opts, device)?;
        build.prune_mask = mask;
        if let Some(log) = self.log.clone() {
            feed.store.on_shrink(Box::new(move |shrink| log(&shrink_line(&shrink))));
        }
        if build.mtp {
            feed.set_mtp();
            feed.fill_mtp(c.experts.count)?;
        }
        let o = &self.opts.part;
        let ngram = NgramFeed::open(&self.gguf, c, self.opts.row_cache, o.max_slots)?;
        let used = c.experts.used as u32;
        let flow = Arc::new(Flow {
            lent: build.lent_pieces.clone(),
            lent_at: build.lent_at.clone(),
            pass_rows: build.pass_rows,
            pass_sizes: std::mem::take(&mut build.pass_sizes),
            far: build.far.iter().flatten().cloned().collect(),
            ..Flow::new(feed, ngram, n, self.opts.stall, o.own_rows(), used, ram.clone())
        });
        let _ = self.flow.set(flow.clone());
        self.usage.attach(&flow, layout, &self.placement);
        // From here on the tier follows what it is allowed: its
        // staging slots are the part it never gives up.
        ram.want_experts(reach, ram_fixed);
        let follower = Arc::downgrade(&flow);
        ram.on_experts(move |ram, allowance| {
            if let Some(flow) = follower.upgrade() {
                flow.follow_ram(ram, allowance);
            }
        });
        build.flow = Some(flow);
        Ok(())
    }

    /// What the expert slots of `store` lend: the rows of the widest step a
    /// prompt read alone takes (the rows the parts were opened for, or the
    /// most below them, in multiples of `ROW_GRAIN`, whose scratch every
    /// device's slots can lend, laid out in a piece of each of its pools; the
    /// plan's estimate may hold more) and where each device's slots lend it
    /// (always the same slots); and the widest prompt read layer by layer, its
    /// kept rows in what every device's slots can lend.
    fn lend(&self, build: &mut Build, store: &ExpertStore<CudaSlots>) -> Result<(), ModelError> {
        let (c, o, n) = (&self.config, &self.opts.part, self.placement.n_devices());
        let own = o.own_rows();
        for d in 0..n {
            build.lendable[d] = store.lendable_pieces(d);
        }
        let lent_of = |build: &Build, d: usize, rows: u32| {
            let PartPlan { role, first, shares, head } = self.placement.part(c, d);
            let shape = PartShape { role, first, shares, head };
            let opts = Options { max_rows: rows, own_rows: own, ..o.clone() };
            let mp = build.multiprocessors[d];
            let frame = Frame::new_in(c, &opts, &shape, mp, &build.lendable[d])?;
            Ok::<_, ModelError>(frame.lent_pieces().map(|pieces| (pieces, frame.lent_raw())))
        };
        let asked = if build.lent.iter().any(|b| *b > 0) { o.max_rows } else { own };
        let rows = widest_rows(own, asked, |rows| {
            let mut fits = true;
            for d in 0..n {
                fits &= lent_of(build, d, rows)?.is_some();
            }
            Ok::<bool, ModelError>(fits)
        })?;
        build.lone_rows = rows;
        // Where K and V lie partly in host memory, beside a copy of every far
        // slot of a layer: what the loan of a pass at the longest context
        // holds of them.
        let far_entry = |d: usize| {
            build.far[d]
                .as_ref()
                .map_or(0, |far| far.lock().unwrap_or_else(|e| e.into_inner()).entry_bytes())
        };
        let far_slots = build
            .far
            .iter()
            .flatten()
            .map(|far| far.lock().unwrap_or_else(|e| e.into_inner()).slots())
            .max()
            .unwrap_or(0);
        build.pass_sizes = (0..n)
            .map(|d| PassSizes::of(c, self.placement.part(c, d).role).with_far(far_entry(d)))
            .collect();
        build.pass_rows = if self.reads_layer_by_layer() {
            pass::pass_span(own, self.opts.pass_rows, |rows| {
                pass::pass_fits(&build.pass_sizes, &build.lendable, own, rows, far_slots)
            })
        } else {
            0
        };
        if build.pass_rows <= own {
            build.pass_rows = 0;
        }
        for d in 0..n {
            let (pieces, raw) = match rows > own {
                true => lent_of(build, d, rows)?
                    .ok_or_else(|| invalid("the slots lend less than they said"))?,
                false => (Vec::new(), 0),
            };
            build.lent[d] = pieces.iter().sum();
            build.lent_raw[d] = raw;
            build.lent_at[d] = None;
            build.lent_pieces[d] = pieces.clone();
            if pieces.is_empty() {
                continue;
            }
            let loans = store
                .scratch_pieces_at(d, &pieces)
                .ok_or_else(|| invalid("the slots lend less than they said"))?;
            // Each piece starts in the layout where the pieces before it
            // end as the scratch was laid out: at what each pool can lend.
            let mut start = 0;
            let mut at = Vec::with_capacity(loans.len());
            let mut loans = loans.iter();
            for (piece, lendable) in pieces.iter().zip(&build.lendable[d]) {
                if *piece > 0 {
                    let loan = loans.next().ok_or_else(|| invalid("a piece without its loan"))?;
                    let (base, _) = store.device(d).pool(loan.pool);
                    at.push((start, base.offset(u64::from(loan.first) * loan.stride)));
                }
                start += lendable / 256 * 256;
            }
            build.lent_at[d] = Some(Pieces(at));
        }
        Ok(())
    }
}

/// The log line of a shrink of the tier of experts in RAM.
fn shrink_line(shrink: &RamShrink) -> String {
    use ares_core::units::Bytes;
    let RamShrink { pieces, bytes, held, target, at_floor } = *shrink;
    let floor = if at_floor { "; it is at its floor and keeps the rest" } else { "" };
    format!(
        "the tier of experts in RAM gave {pieces} piece(s), {}, back to the machine: it holds {} \
         and may hold {}{floor}",
        Bytes(bytes),
        Bytes(held),
        Bytes(target),
    )
}

/// The meeting of the parts of `n` devices while they open.
pub(super) fn meeting(n: usize) -> Meeting<Build> {
    Meeting::new(Build::new(n))
}
