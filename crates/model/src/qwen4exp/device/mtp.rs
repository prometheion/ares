//! Flash-Next's MTP block (section 4 of the specification) as the layer of
//! the shared MTP driver (`blocks/mtp.rs`), which runs its chains, its picks
//! on the device, its carry and its following of the model.
//!
//! The layer is a part of one layer on the device that holds the output head
//! ([`Part::open_mtp`]): attention with its indexer and a mixture of experts
//! between stream mixers, which runs steps of its own in front of the
//! model's. The model gives the driver its hidden state where it is tapped:
//! the residual streams after the last layer (`hc` streams of `n` values a
//! row), which the model part keeps until its next step begins. The block's
//! cell `i` pairs the residual at cell `i` with the token at cell `i + 1`, so a
//! pair of the token at position `p` is cell `p - 1` (the driver's lag of 1).
//!
//! - **Rows that only write.** A cell the model has passed gets its keys and
//!   values and the indexer's key, and reads nothing
//!   ([`Part::enqueue_keys`]), in steps of up to [`MAX_BATCH`] rows.
//! - **Links.** The whole layer, the final mixer and the draft head: the first
//!   rows of the model's output head as the head view orders them. The
//!   block's experts are resident in VRAM (layer 0 of the expert store), so a
//!   link never waits for the store, and the chain stays on the device: the
//!   host neither waits for a link nor reads its picks before the step ends.
//! - **State that is put back.** The indexer's carried keys of a sequence are
//!   saved behind link 0 and put back behind the last link; keys and values of
//!   the links' cells are written again when the model has passed them.
//!
//! The block's pages are its own memory, addressed through the page tables
//! the conductor gives the model: page `p` of a sequence holds the same cells
//! of the model's attention layers and of the block.
//!
//! **The prefix cache** (`resume.rs`). A checkpoint keeps the block's slot
//! (its indexer's carried keys) and the sequence's carry; a page copy and a
//! page in transit take the block's cells and keys with the model's. The rows
//! of a replay have their cells in the cache, the block's too (whoever
//! computed them had the same tokens), so the block follows a replay with the
//! carry alone and writes nothing. Since cell `i` waits for the token of
//! cell `i + 1`, the last cell of a page is written one step after the page
//! is full, when the sequence goes on: a page whose sequence ended at its end
//! is settled without it (its tile can then be converted when the pool needs
//! room), and a row that would write into a tile that is converted already
//! (a cached page that another sequence went past) reads the cell and writes
//! nothing, as a replay. Either way only the drafts can differ.

use std::ops::Range;
use std::sync::Arc;

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, KeptRows, LayerKernels, MAX_BATCH, Segment, Stream,
};
use ares_runtime::ModelError;

use super::part::{ExpertAt, Options, Part, StepEnv};
use crate::blocks::head::HeadView;
use crate::blocks::invalid;
use crate::blocks::mtp::{Driver, DriverSetup, LinkLogits, MtpLayer, Rows};
pub use crate::blocks::mtp::{MtpStep, MtpTimes};
use crate::blocks::rows::HostTable;
use crate::blocks::slots::{Extent, PageArray, PageGroup};
use crate::qwen4exp::Config;
use crate::qwen4exp::device::load::Mat;
use crate::qwen4exp::reference::mtp_config;
use crate::qwen4exp::weights::MtpWeights;

/// What the block took, for reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MtpReport {
    /// The device part that holds the block.
    pub index: usize,
    /// Bytes of its dense weights on the device.
    pub weights: u64,
    /// Bytes of its keys, values and indexer keys for all pages.
    pub pools: u64,
    /// Bytes of its scratch, carries, tables and step records.
    pub buffers: u64,
    /// Rows of the embedding table kept on the device for the tokens a link
    /// can pick (the head view's), and their bytes.
    pub kept_rows: u32,
    pub kept_bytes: u64,
}

/// A chain the block made: for tests and measurements that compare its
/// drafts with the reference's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpDrafted {
    /// The sequence slot.
    pub slot: u32,
    /// The cell of the model the chain's first draft is a guess for: one
    /// behind the newest token's.
    pub position: u32,
    /// The newest token, which link 0 is paired with.
    pub token: u32,
    /// The drafts, link after link.
    pub drafts: Vec<u32>,
}

/// What an MTP block tells whoever watches it.
#[derive(Debug, Default)]
pub(super) struct Watch {
    pub times: MtpTimes,
    /// The newest chains, oldest first, at most [`Watch::KEPT`].
    pub recent: std::collections::VecDeque<MtpDrafted>,
}

impl Watch {
    const KEPT: usize = 1024;
}

/// A chain of the step that ran last, until its drafts are read.
#[derive(Clone, Copy, Debug)]
struct Asked {
    slot: u32,
    position: u32,
    token: u32,
    place: u32,
    links: u32,
}

pub(super) struct MtpBlock {
    part: Part,
    driver: Driver,
    layers: LayerKernels,
    /// Keys, values and indexer keys of the block's cells, for every page.
    pools: DeviceBuffer,
    /// The rows of the embedding table of the tokens a link can pick, and
    /// their tokens when they are not the first ones: the driver reads them.
    _kept: DeviceBuffer,
    _kept_tokens: Option<DeviceBuffer>,
    serial: u32,
    wide: u64,
    head_rows: u32,
    watch: Arc<std::sync::Mutex<Watch>>,
    asked: Vec<Asked>,
    report: MtpReport,
}

impl std::fmt::Debug for MtpBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpBlock").field("report", &self.report).finish_non_exhaustive()
    }
}

/// The block's part as the driver drives it.
struct BlockLayer<'a> {
    part: &'a mut Part,
    serial: &'a mut u32,
    pools: DevicePtr,
    wide: u64,
    head_rows: u32,
}

impl BlockLayer<'_> {
    /// The step of the block for `rows`, whose input starts from the rows'
    /// hidden states and embeddings.
    fn env<'b>(&mut self, rows: &'b Rows<'b>, input: DevicePtr) -> StepEnv<'b> {
        // The token marks the block's route records: never 0, and never the
        // same for two steps that follow each other.
        *self.serial = self.serial.wrapping_add(1).max(1);
        StepEnv {
            stream: rows.stream,
            record: rows.record,
            segments: rows.segments,
            token: *self.serial,
            tokens: &[],
            ngram: None,
            inbound: Some(input),
            xbase: 0,
            pools: self.pools,
            replay: rows.replay,
        }
    }
}

impl MtpLayer for BlockLayer<'_> {
    fn lag(&self) -> u32 {
        1
    }

    fn keys_inputs(&self) -> Option<(DevicePtr, DevicePtr)> {
        None
    }

    fn bytes(&mut self, _: &[Segment], _: bool, _: &mut [u8]) -> Result<usize, ModelError> {
        // The part sends its records and plans itself, from its ring.
        Ok(0)
    }

    fn keys(&mut self, rows: &Rows<'_>) -> Result<(), ModelError> {
        let n = rows.record.dims().n_rows;
        // SAFETY: the driver's rows hold `n` hidden states and embeddings.
        let input = unsafe { self.part.mtp_input(rows.stream, rows.h, rows.e, n)? };
        let env = self.env(rows, input);
        self.part.begin(&env)?;
        self.part.enqueue_keys(&env, 0)?;
        self.part.end_keys(&env)
    }

    fn link(
        &mut self,
        rows: &Rows<'_>,
        out: DevicePtr,
        _logits: DevicePtr,
    ) -> Result<LinkLogits, ModelError> {
        let n = rows.record.dims().n_rows;
        // SAFETY: as in `keys`.
        let input = unsafe { self.part.mtp_input(rows.stream, rows.h, rows.e, n)? };
        let env = self.env(rows, input);
        self.part.begin(&env)?;
        self.part.enqueue_layer(&env, 0)?;
        self.part.end(&env)?;
        // The residual of the link's rows is the next link's hidden state.
        // SAFETY: the part's residual holds the link's rows, `out` the
        // driver's room for them.
        unsafe {
            let bytes = (u64::from(n) * self.wide * 4) as usize;
            rows.stream.copy(out, self.part.residual(), bytes)?;
        }
        Ok(LinkLogits { ptr: self.part.head_logits(), stride: self.head_rows })
    }

    fn forget_links(&mut self) {
        // Nothing is kept: the part plans a link when it runs it.
    }

    fn save(&mut self, stream: &Stream, slots: &[u32]) -> Result<(), ModelError> {
        slots.iter().try_for_each(|&slot| self.part.save_slot(stream, slot))
    }

    fn restore(&mut self, stream: &Stream, slots: &[u32]) -> Result<(), ModelError> {
        slots.iter().try_for_each(|&slot| self.part.restore_slot(stream, slot))
    }

    fn writes_converted(&self, slot: u32, cell: u32) -> bool {
        self.part.writes_converted(slot, cell)
    }

    fn settle(&mut self, slot: u32, cells: u32) {
        self.part.settle(slot, cells);
    }
}

impl MtpBlock {
    /// Builds the block on `device`, next to the model part `index` that
    /// holds the output head `output` (in the order of `view`, whose tokens
    /// `head_tokens` names on the device). `opts` are the model part's: the
    /// block has the same sequences, pages and formats, and steps of up to
    /// [`MAX_BATCH`] rows. `table` is the embedding table on the host.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open(
        c: &Config,
        w: &MtpWeights<'_>,
        opts: &Options,
        index: usize,
        output: (Mat, DevicePtr),
        head_tokens: DevicePtr,
        view: &HeadView,
        table: Arc<HostTable>,
        watch: Arc<std::sync::Mutex<Watch>>,
        device: &Device,
        ledger: &mut Ledger,
    ) -> Result<Self, ModelError> {
        let block = mtp_config(c);
        // The model's steps, which the block follows, have up to this many
        // rows; its own have up to `MAX_BATCH`.
        let model_rows = opts.max_rows;
        let opts = Options {
            max_rows: MAX_BATCH,
            logit_rows: MAX_BATCH,
            window_rows: 0,
            host_logits: false,
            profile: None,
            ..opts.clone()
        };
        let before = ledger.reserved();
        // The draft head is the first rows of the head in the view's order.
        let (mat, at) = output;
        if view.rows == 0 || view.rows > mat.n_out {
            return Err(invalid(format!(
                "a draft head of {} rows of a head of {}",
                view.rows, mat.n_out
            )));
        }
        let head = (Mat { n_out: view.rows, ..mat }, at);
        let part = Part::open_mtp(&block, w, &opts, index, head, device, ledger)?;
        let weights = part.weight_bytes();
        let pool_bytes = part.pool_bytes();
        ledger.reserve("qwen4exp MTP block K, V and keys", pool_bytes)?;
        let pools = DeviceBuffer::new(device, pool_bytes.max(64))?;
        // The rows of the embedding table of the tokens a link can pick.
        let shape = table.shape();
        let tokens = view.draft_tokens();
        let kept_bytes = u64::from(view.rows) * shape.row_bytes();
        ledger.reserve("qwen4exp MTP block: kept embedding rows", kept_bytes)?;
        let kept = DeviceBuffer::new(device, kept_bytes.max(64))?;
        let mut bytes = vec![0u8; kept_bytes as usize];
        table.stage(&tokens, &mut bytes)?;
        kept.write(0, &bytes)?;
        let kept_tokens = if view.order.is_none() {
            None
        } else {
            let list: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let buffer = DeviceBuffer::new(device, list.len() as u64)?;
            buffer.write(0, &list)?;
            Some(buffer)
        };
        let wide = c.wide() as u64;
        let kept_rows = KeptRows {
            rows: kept.ptr(),
            tokens: kept_tokens.as_ref().map_or(DevicePtr::NULL, DeviceBuffer::ptr),
            count: view.rows,
        };
        let driver = Driver::open(
            device,
            ledger,
            DriverSetup {
                label: "qwen4exp MTP block",
                width: wide as u32,
                max_slots: opts.max_slots,
                max_rows: model_rows,
                window_rows: MAX_BATCH,
                head_rows: view.rows,
                head_tokens,
                table: shape,
                kept: kept_rows,
                host: Some(table),
            },
            MAX_BATCH,
            false,
            |_, _| 0,
        )?;
        let report = MtpReport {
            index,
            weights,
            pools: pool_bytes,
            buffers: ledger.reserved() - before - weights - pool_bytes - kept_bytes,
            kept_rows: view.rows,
            kept_bytes,
        };
        Ok(Self {
            layers: part.layers(),
            part,
            driver,
            pools,
            _kept: kept,
            _kept_tokens: kept_tokens,
            serial: 0,
            wide,
            head_rows: view.rows,
            watch,
            asked: Vec::new(),
            report,
        })
    }

    /// Tells the block where its experts are: layer 0 of the store.
    pub(super) fn set_experts(&mut self, at: ExpertAt) -> Result<(), ModelError> {
        self.part.set_experts(vec![at])
    }

    pub(super) fn report(&self) -> MtpReport {
        self.report
    }

    /// Entries `first..` of the page table of `slot` are now `entries`: the
    /// block reads the tables the model reads.
    pub(super) fn update_table(
        &mut self,
        stream: &Stream,
        slot: u32,
        first: u32,
        entries: impl IntoIterator<Item = u32>,
    ) -> Result<(), ModelError> {
        self.part.update_table(stream, slot, first, entries)
    }

    /// What a checkpoint keeps of the block: its slot (the carried keys of
    /// its indexer) and the sequence's carry.
    pub(super) fn extents(&self) -> [Extent; 2] {
        let row = self.wide * 4;
        [
            self.part.arena_extent(),
            Extent { base: self.driver.carry().ptr(), stride: row, bytes: row },
        ]
    }

    /// What a page holds of the block: its K/V group and its indexer's pooled
    /// keys, in the block's own pools. A tail whose one cell is its sequence's
    /// last has none of the block's cells yet.
    pub(super) fn page_parts(
        &mut self,
    ) -> Result<(Option<PageGroup<'_>>, Vec<PageArray>), ModelError> {
        self.part.page_parts(self.pools.ptr(), true)
    }

    /// The pool freed these pages: the block gives back what it kept for them.
    pub(super) fn release_pages(&mut self, pages: &[u32]) {
        self.part.release_pages(pages.iter().copied());
    }

    /// Runs `f` with the driver and the block's part as its layer.
    fn with_layer<T>(
        &mut self,
        f: impl FnOnce(&mut Driver, &mut BlockLayer<'_>, &LayerKernels) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let mut layer = BlockLayer {
            part: &mut self.part,
            serial: &mut self.serial,
            pools: self.pools.ptr(),
            wide: self.wide,
            head_rows: self.head_rows,
        };
        f(&mut self.driver, &mut layer, &self.layers)
    }

    /// Step 1, before the tables of the step are written: the cells the
    /// model's last step has passed. `accepted` says how many rows of the
    /// window of a slot that started at a row count (`None` when the record
    /// names no outcome for it), and `residual` is where the model's part
    /// left the residual of that step's rows (`D` f32 a row).
    pub(super) fn follow(
        &mut self,
        stream: &Stream,
        accepted: &dyn Fn(u32, u32) -> Option<u32>,
        residual: DevicePtr,
    ) -> Result<(), ModelError> {
        self.with_layer(|driver, layer, layers| {
            let pools = layer.pools;
            driver.follow(layer, layers, stream, accepted, residual, DevicePtr::NULL, pools)
        })?;
        self.watch.lock().unwrap_or_else(|e| e.into_inner()).times = self.driver.times();
        Ok(())
    }

    /// The rotary positions of the rows of the step enqueued next, where they
    /// are not their cells (see `blocks::mtp::Driver::set_rotary`).
    pub(super) fn set_rotary(&mut self, rotary: Vec<[i32; 4]>) {
        self.driver.set_rotary(rotary);
    }

    /// Steps 2 and 3, after the tables of the step are written: the newest
    /// cell of every sequence of the step and the chains of `step`. `tokens`
    /// are the tokens of the step's rows, `replay` its rows that are a
    /// replay. `only` is a step that only drafts. Unless `only`, the step's
    /// rows are kept for the next step to follow.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn step(
        &mut self,
        stream: &Stream,
        segments: &[Segment],
        tokens: &[u32],
        replay: &[Range<u32>],
        step: Option<MtpStep>,
        only: bool,
    ) -> Result<(), ModelError> {
        self.asked.clear();
        if let Some(step) = step.filter(|s| !s.chains.is_empty()) {
            for chain in &step.chains {
                let s = segments
                    .get(chain.segment as usize)
                    .ok_or_else(|| invalid(format!("the step has no segment {}", chain.segment)))?;
                self.asked.push(Asked {
                    slot: s.slot,
                    position: s.pos0 as u32 + 1,
                    token: tokens[s.row0 as usize],
                    place: chain.place,
                    links: chain.rows.len() as u32,
                });
            }
            self.driver.draft_step(step)?;
        }
        self.with_layer(|driver, layer, layers| {
            let pools = layer.pools;
            driver.step(
                layer,
                layers,
                stream,
                segments,
                tokens,
                replay,
                DevicePtr::NULL,
                pools,
                only,
            )
        })
    }

    /// The step's list of drafts, which every drafter of the device writes at
    /// the places of its windows.
    pub(super) fn list(&self) -> DevicePtr {
        self.driver.list()
    }

    /// The shared driver, whose drafts of a step the part hands out
    /// (`blocks/step.rs`).
    pub(super) fn driver(&self) -> &Driver {
        &self.driver
    }

    pub(super) fn driver_mut(&mut self) -> &mut Driver {
        &mut self.driver
    }

    /// The chains of the step that ran last, with their drafts as the host
    /// reads them now that the step has ended: the slot, and the newest token
    /// followed by the drafts the chain made. They go to the watchers too.
    pub(super) fn chains(&mut self) -> Vec<(u32, Vec<u32>)> {
        if self.asked.is_empty() {
            return Vec::new();
        }
        let end = self.asked.iter().map(|a| a.place + a.links).max().unwrap_or(0);
        let mut drafts = vec![u32::MAX; end as usize];
        self.driver.picked(&mut drafts);
        let mut out = Vec::with_capacity(self.asked.len());
        let mut watch = self.watch.lock().unwrap_or_else(|e| e.into_inner());
        for a in std::mem::take(&mut self.asked) {
            let made: Vec<u32> = drafts[a.place as usize..(a.place + a.links) as usize]
                .iter()
                .copied()
                .take_while(|t| *t != u32::MAX)
                .collect();
            if watch.recent.len() == Watch::KEPT {
                watch.recent.pop_front();
            }
            watch.recent.push_back(MtpDrafted {
                slot: a.slot,
                position: a.position,
                token: a.token,
                drafts: made.clone(),
            });
            let mut tokens = vec![a.token];
            tokens.extend(made);
            out.push((a.slot, tokens));
        }
        watch.times = self.driver.times();
        out
    }
}
