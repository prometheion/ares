//! The model's own draft block on its device: the next-token-prediction block
//! of `qwen35` (multi-token prediction), run as a drafter.
//!
//! What the block computes is in `reference/mtp.rs`. Here it runs in two ways,
//! both inside the step and without a trip to the host:
//!
//! - **The chain**, before the model's own blocks. For every sequence that
//!   drafts with the block, link 0 reads the sequence's newest token and the
//!   final hidden state that token was picked from (kept on the device from
//!   the step before, see `kernels/sampler_draft.cuh`) and picks the token
//!   after it; link `j` reads the pick and the output of link `j - 1`. Each
//!   pick is made on the device by the sampler, from the draft head, with the
//!   settings and the noise of the row that will verify it, and is written
//!   straight into the token array of the step record, where the embedding of
//!   the verify window reads it, and into a list the host reads at the end of
//!   the step together with the picks.
//! - **Following the model**, after its blocks. The block's keys and values
//!   at a position depend on that position's token and the model's hidden
//!   state of the position before only, so for every row of the step they are
//!   computed from what the model just computed and written into the block's
//!   own K/V pool (the same pages and tables as the model's attention blocks,
//!   one more array of entries). Rows of a window whose drafts are rejected
//!   are written too: their cells are past the sequence's end and the next
//!   step overwrites them. This is also what replaces the provisional keys
//!   and values a chain wrote from its own outputs.
//!
//! The whole block is on one device, the one that embeds, and its head is the
//! first rows of the output head there (the draft vocabulary when the
//! placement has one), so it takes no weights twice. That device keeps the
//! rows of the embedding table for the tokens the chain can pick (`embed.rs`):
//! a link embeds its pick from them, and the embedding of the verify window
//! reads them for the same tokens. The newest token of a chain is known to
//! the host, which stages its row with the step's records. The chain works
//! in a scratch of its own, sized for the sequences that can draft at once;
//! following the model reuses the step's scratch, which is free by then.

use ares_core::Ledger;
use ares_cuda::{
    AttendBuffers, AttentionShape, AttnRow, DeviceBuffer, DevicePtr, ENTRY_CELLS, Event, Kernels,
    PinnedBuffer, Planner, PrefillPath, SampleBatch, SamplePlan, SamplerKernels, Segment, StepDims,
    StepRecord,
};
use ares_runtime::ModelError;
use ares_sampler::{SamplePick, SampleRow};

use super::super::load::{DeviceWeights, DraftW, MixerW};
use super::kv;
use super::{
    AttentionAt, DRAFT_KV, Options, Part, Phase, Prepared, StepEnv, TAILS_PER_SLOT, geometry,
    invalid,
};
use crate::qwen35::Config;

/// Bytes reserved per link and sequence for the attention plan of a link.
const PLAN_BYTES: u64 = 32 * 1024;

/// A sequence that drafts with the block in the coming step.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftChain {
    /// The segment of the step that verifies the drafts: its first row is
    /// the sequence's newest token, its other rows take the drafts.
    pub segment: u32,
    /// The newest token.
    pub token: u32,
    /// How the rows that verify the drafts are sampled, one per draft: link
    /// `j` of the chain and row `j` of the window pick the same position.
    pub rows: Vec<SampleRow>,
}

/// What the block drafts in the coming step.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftStep {
    /// The token array of the step record on this device: the chain writes
    /// the drafts into it.
    pub tokens: DevicePtr,
    pub chains: Vec<DraftChain>,
}

/// Where a step left the tokens it drafted: `tokens` u32 on the device, the
/// drafts of each chain of the [`DraftStep`] in order, chain after chain.
/// `u32::MAX` is a draft that could not be picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drafted {
    pub ptr: DevicePtr,
    pub tokens: u32,
}

/// What the draft block takes on its device, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftBytes {
    /// The block's weights (the draft head is rows of the output head and
    /// counts nothing).
    pub weights: u64,
    /// Its K and V per page of 128 cells.
    pub page: u64,
    /// Its exact K/V entries and the page table they need, when its formats
    /// have a tile mean.
    pub exact: u64,
    /// The hidden states kept from step to step, for all sequence slots.
    pub carry: u64,
    /// The chain's scratch, the sampler's included.
    pub scratch: u64,
    /// The records of a step: links, plans, rows.
    pub records: u64,
    /// What the step's own scratch grew by for the block.
    pub step_scratch: u64,
}

/// Device time of the draft block, added up over steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DraftTimes {
    /// Steps counted.
    pub steps: u64,
    /// Links of all chains of those steps (a link of several sequences
    /// counts once).
    pub links: u64,
    /// Milliseconds of the chains.
    pub chain_ms: f64,
    /// Milliseconds of following the model.
    pub follow_ms: f64,
}

/// What the block needs to know of the part it is in.
#[derive(Clone, Copy, Debug)]
pub(super) struct Setup {
    pub max_slots: u32,
    pub pages: u32,
    pub table_stride: u32,
    /// The attention blocks keep exact K/V entries, so the step's page table
    /// names exact slots that the block's layer must not read.
    pub main_exact: bool,
}

/// What the block adds to the sizes a part plans its memory from.
#[derive(Clone, Copy, Debug)]
pub(super) struct DraftSizes {
    pub at: AttentionAt,
    /// Bytes of the block's pool.
    pub pool: u64,
    /// Values of all KV heads of a row: the step's K and V buffers hold them.
    pub kv_values: u64,
    /// Bytes of partial attention states a link's plan may need.
    pub attn_state: u64,
    /// Bytes of prepared activations the chain needs, quantized and f16.
    pub act_q: u64,
    pub act_h: u64,
}

/// The sizes for the draft block of `w`, whose pool starts at byte `pool` of
/// the pages; `None` on a device without it.
pub(super) fn sizes(
    c: &Config,
    w: &DeviceWeights,
    max_slots: u32,
    pages: u32,
    target_blocks: u32,
    pool: u64,
) -> Option<DraftSizes> {
    let draft = w.draft.as_ref()?;
    let MixerW::Attention(a) = &draft.block.mixer else { return None };
    let geo = geometry(c, DRAFT_KV, a.n_kv);
    let rows = max_slots;
    let widest = draft.block.ffn.as_ref().map_or(0, |f| f.gate.n_out).max(a.out.n_in).max(a.q.n_in);
    let slots = 20 * u64::from(target_blocks) / u64::from(a.n_kv) + 64;
    Some(DraftSizes {
        at: AttentionAt { geo, pool, exact: 0 },
        pool: u64::from(pages) * geo.converted_entry_bytes(),
        kv_values: u64::from(a.n_kv) * u64::from(geo.head_dim),
        attn_state: slots * u64::from(a.n_head) * (u64::from(geo.head_dim) + 1) * 4,
        act_q: Kernels::quantized_activation_bytes(widest, rows),
        act_h: Kernels::prefill_activation_bytes(PrefillPath::F16, widest, rows),
    })
}

/// Offsets of the chain's scratch.
#[derive(Clone, Copy, Debug, Default)]
struct Scratch {
    e: u64,
    h: u64,
    x: u64,
    xn: u64,
    p: u64,
    qg: u64,
    k_raw: u64,
    v: u64,
    q: u64,
    gate: u64,
    k: u64,
    o: u64,
    g1: u64,
    g2: u64,
    rope: u64,
    /// Per link the block's output of every chain: the next link's hidden state.
    out: u64,
    logits: u64,
    /// Per link the picks of every chain.
    picks: u64,
    drafts: u64,
    sampler: u64,
    exchange: u64,
    len: u64,
}

/// The events around one step's draft work.
#[derive(Debug)]
struct Clock {
    /// Start and end of the chains: always recorded, because what a link
    /// costs decides how many are worth drafting.
    chain: [Event; 2],
    /// Start and end of following the model, when the part keeps a profile.
    follow: Option<[Event; 2]>,
    /// Whether the last step recorded them and they are not read yet.
    chain_pending: bool,
    follow_pending: bool,
    links: u64,
    /// Milliseconds of the chains of the last step.
    last_chain_ms: f64,
}

/// The draft block of a part.
#[derive(Debug)]
pub(super) struct Mtp {
    sampler: SamplerKernels,
    pub(super) at: AttentionAt,
    /// The K/V group of the block's layer: its exact pool and table.
    pub(super) kv: kv::Group,
    w: DraftW,
    /// Per slot `carry_rows` rows of `d` values (see `sampler_draft.cuh`).
    carry: DeviceBuffer,
    carry_rows: u32,
    scratch: DeviceBuffer,
    s: Scratch,
    /// What a step's kernels read besides the step record: the records,
    /// settings, indices and plans of the links and the rows of the step,
    /// written on the host and uploaded with one copy.
    host: PinnedBuffer,
    records: DeviceBuffer,
    max_chains: u32,
    max_links: u32,
    /// The tokens of the head's rows on the device, or null when row `r` is
    /// token `r`.
    head_tokens: DevicePtr,
    pending: Option<DraftStep>,
    /// Where the rows of the step are in `records`, for following.
    follow_rows: u64,
    /// Where the rows of the chains' newest tokens that the host staged are
    /// in `records` (link 0 embeds them); null when it staged none.
    newest: DevicePtr,
    drafted: u32,
    clock: Clock,
    times: DraftTimes,
    step_scratch: u64,
}

impl Mtp {
    /// Reserves and allocates what the block needs besides its weights and
    /// its pool.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open(
        c: &Config,
        device: &ares_cuda::Device,
        ledger: &mut Ledger,
        w: &DeviceWeights,
        opts: &Options,
        setup: Setup,
        sizes: Option<DraftSizes>,
        step_scratch: u64,
    ) -> Result<Option<Self>, ModelError> {
        let Setup { max_slots, pages, table_stride, main_exact } = setup;
        let (Some(sizes), Some(draft)) = (sizes, w.draft.as_ref()) else { return Ok(None) };
        let MixerW::Attention(a) = &draft.block.mixer else {
            return Err(invalid("the draft block is an attention block"));
        };
        let ffn = draft.block.ffn.as_ref().ok_or_else(|| invalid("the draft block has no ffn"))?;
        let Some(embedding) = &w.embedding else {
            return Err(invalid("the draft block's device does not embed"));
        };
        if embedding.kept.is_none() {
            return Err(invalid("the draft block's device keeps no rows of the embedding table"));
        }
        // The newest tokens' rows go with the records when the host stages rows.
        let staged_rows =
            if embedding.stages() { u64::from(max_slots) * embedding.table.row_bytes() } else { 0 };
        let d = c.n_embd as u64;
        let rows = u64::from(max_slots);
        let links = u64::from(opts.window_rows.saturating_sub(1).max(1));
        let (hq, hkv) = (u64::from(a.n_head) * u64::from(sizes.at.geo.head_dim), sizes.kv_values);
        let inner = u64::from(ffn.gate.n_out);
        let head_rows = u64::from(draft.head.n_out);

        let mut s = Scratch::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        s.e = take(rows * d * 4);
        s.h = take(rows * d * 4);
        s.x = take(rows * d * 4);
        s.xn = take(rows * d * 4);
        s.p = take(rows * d * 4);
        s.qg = take(rows * 2 * hq * 4);
        s.k_raw = take(rows * hkv * 4);
        s.v = take(rows * hkv * 4);
        s.q = take(rows * hq * 4);
        s.gate = take(rows * hq * 4);
        s.k = take(rows * hkv * 4);
        s.o = take(rows * hq * 4);
        s.g1 = take(rows * inner * 4);
        s.g2 = take(rows * inner * 4);
        s.rope = take(rows * (c.attention.rope.dim as u64 / 2) * 8);
        s.out = take(links * rows * d * 4);
        s.logits = take(rows * head_rows * 4);
        s.picks = take(links * rows * SamplePick::BYTES as u64);
        s.drafts = take(links * rows * 4);
        s.sampler = take(SamplerKernels::scratch_bytes(max_slots));
        s.exchange = take(u64::from(SamplerKernels::exchange_values(max_slots, 1)) * 4);
        s.len = len;

        let carry_rows = opts.window_rows.max(1);
        let carry_bytes = rows * u64::from(carry_rows) * d * 4;
        // Per link: a record, the settings, two index arrays and a plan per
        // sequence; then the carry index of link 0 and the rows of the step.
        let per_link =
            64 + rows * (32 + 4 + 16 + 16) + rows * (32 + 8 + 8) + 64 + rows * PLAN_BYTES;
        let record_bytes =
            (links * per_link + rows * 4 + u64::from(opts.max_rows) * 16 + 256 + staged_rows)
                .next_multiple_of(64);
        ledger.reserve("qwen35 draft block: hidden states kept", carry_bytes)?;
        ledger.reserve("qwen35 draft block: chain scratch", s.len)?;
        ledger.reserve("qwen35 draft block: step records", record_bytes)?;
        let carry = DeviceBuffer::new(device, carry_bytes)?;
        let scratch = DeviceBuffer::new(device, s.len)?;
        carry.write(0, &vec![0u8; carry_bytes as usize])?;
        scratch.write(s.sampler, &vec![0u8; SamplerKernels::scratch_bytes(max_slots) as usize])?;
        let sizing =
            kv::Sizing { slots: max_slots, rows: opts.max_rows, tails: max_slots * TAILS_PER_SLOT };
        let exact_slots = kv::slots_for(DRAFT_KV.0, DRAFT_KV.1, sizing);
        let pages_kv = kv::Group::new(
            device,
            ledger,
            "draft block",
            vec![sizes.at],
            DRAFT_KV,
            exact_slots,
            pages,
            max_slots,
            table_stride,
            exact_slots > 0 || main_exact,
        )?;
        let clock = Clock {
            chain: [Event::timed(device)?, Event::timed(device)?],
            follow: match opts.profile {
                Some(_) => Some([Event::timed(device)?, Event::timed(device)?]),
                None => None,
            },
            chain_pending: false,
            follow_pending: false,
            links: 0,
            last_chain_ms: 0.0,
        };
        Ok(Some(Self {
            sampler: SamplerKernels::load(device)?,
            at: sizes.at,
            kv: pages_kv,
            w: draft.clone(),
            carry,
            carry_rows,
            scratch,
            s,
            host: PinnedBuffer::new(device, record_bytes as usize)?,
            records: DeviceBuffer::new(device, record_bytes)?,
            max_chains: max_slots,
            max_links: links as u32,
            head_tokens: w.head_tokens.map_or(DevicePtr::NULL, |at| w.at(at)),
            pending: None,
            follow_rows: 0,
            newest: DevicePtr::NULL,
            drafted: 0,
            clock,
            times: DraftTimes::default(),
            step_scratch,
        }))
    }

    fn at(&self, offset: u64) -> DevicePtr {
        self.scratch.ptr().offset(offset)
    }

    /// Adds the events of the last step to the times. The step has run.
    fn flush_clock(&mut self) -> Result<(), ModelError> {
        let clock = &mut self.clock;
        if clock.chain_pending || clock.follow_pending {
            self.times.steps += 1;
        }
        if std::mem::take(&mut clock.chain_pending) {
            clock.last_chain_ms = f64::from(clock.chain[1].elapsed_ms_since(&clock.chain[0])?);
            self.times.chain_ms += clock.last_chain_ms;
            self.times.links += std::mem::take(&mut clock.links);
        }
        if let (true, Some(follow)) = (std::mem::take(&mut clock.follow_pending), &clock.follow) {
            self.times.follow_ms += f64::from(follow[1].elapsed_ms_since(&follow[0])?);
        }
        Ok(())
    }
}

/// One link of a step's chains, as its kernels need it.
struct Link {
    /// Rows: the chains that are at least this long.
    n: u32,
    /// The rows of the next link.
    n_next: u32,
    dims: StepDims,
    /// Offsets in the records buffer.
    record: u64,
    tokens: u64,
    params: u64,
    record_rows: u64,
    draft_at: u64,
    plan_at: u64,
    plan: ares_cuda::Plan,
    sample: SamplePlan,
}

/// Where the block's exact entries are: null when its formats have none.
fn exact_pool(m: &Mtp) -> DevicePtr {
    if m.kv.has_exact() { m.kv.exact().offset(m.at.exact) } else { DevicePtr::NULL }
}

fn words(buf: &mut [u8], at: u64, values: impl IntoIterator<Item = u32>) -> u64 {
    let mut end = at as usize;
    for v in values {
        buf[end..end + 4].copy_from_slice(&v.to_le_bytes());
        end += 4;
    }
    (end as u64).next_multiple_of(16)
}

impl Part {
    /// Whether this part holds the model's draft block.
    pub fn has_draft_block(&self) -> bool {
        self.mtp.is_some()
    }

    /// The most drafts the block makes per sequence in a step.
    pub fn draft_links(&self) -> u32 {
        self.mtp.as_ref().map_or(0, |m| m.max_links)
    }

    /// What the block takes on this device.
    pub fn draft_bytes(&self) -> Option<DraftBytes> {
        let m = self.mtp.as_ref()?;
        let mat = |mat: &super::super::load::Mat| mat.bytes();
        let MixerW::Attention(a) = &m.w.block.mixer else { return None };
        let ffn = m.w.block.ffn.as_ref()?;
        let weights =
            [&m.w.proj_e, &m.w.proj_h, &a.q, &a.k, &a.v, &a.out, &ffn.gate, &ffn.up, &ffn.down]
                .iter()
                .map(|x| mat(x))
                .sum::<u64>()
                + 5 * u64::from(self.d) * 4;
        Some(DraftBytes {
            weights,
            page: m.at.geo.converted_entry_bytes(),
            exact: m.kv.bytes(),
            carry: m.carry.len(),
            scratch: m.scratch.len(),
            records: m.records.len(),
            step_scratch: m.step_scratch,
        })
    }

    /// Device time of the draft block so far (following the model is counted
    /// only when the part keeps a profile). Includes the last step only after
    /// the next one was enqueued or [`Part::flush_profile`] was called.
    pub fn draft_times(&self) -> DraftTimes {
        self.mtp.as_ref().map_or_else(DraftTimes::default, |m| m.times)
    }

    /// Device milliseconds of the chains of the step that ran last (0 when it
    /// drafted nothing). The step must have ended.
    pub fn draft_chain_ms(&mut self) -> Result<f64, ModelError> {
        let Some(m) = &mut self.mtp else { return Ok(0.0) };
        m.flush_clock()?;
        Ok(m.clock.last_chain_ms)
    }

    /// Tells the part what its draft block drafts in the step that is
    /// enqueued next. Without this call a step drafts nothing.
    pub fn draft_step(&mut self, step: DraftStep) -> Result<(), ModelError> {
        let m = self.mtp.as_mut().ok_or_else(|| invalid("this part holds no draft block"))?;
        if step.chains.len() > m.max_chains as usize
            || step.chains.iter().any(|c| c.rows.is_empty() || c.rows.len() > m.max_links as usize)
        {
            return Err(invalid(format!(
                "{} sequences draft, the longest {} tokens; the block takes {} sequences of 1 to {}",
                step.chains.len(),
                step.chains.iter().map(|c| c.rows.len()).max().unwrap_or(0),
                m.max_chains,
                m.max_links
            )));
        }
        m.pending = Some(step);
        Ok(())
    }

    /// Where the step enqueued last left its drafts; `None` when it drafted
    /// nothing.
    pub fn drafted(&self) -> Option<Drafted> {
        let m = self.mtp.as_ref()?;
        (m.drafted > 0).then(|| Drafted { ptr: m.at(m.s.drafts), tokens: m.drafted })
    }

    /// The block's exact K/V entries (its cells are in them while their tile is
    /// exact: see [`DRAFT_KV`]), for tests that read the cells it wrote.
    pub fn draft_exact_entries(&self) -> Option<&DeviceBuffer> {
        self.mtp.as_ref().and_then(|m| m.kv.exact_buffer())
    }

    /// The row of the carry that `slot`'s next chain starts from (the final
    /// hidden state of its last row) and its bytes: what a checkpoint keeps
    /// of the block. `None` on a device without the block.
    pub(super) fn carry_row(&self, slot: u32) -> Option<(DevicePtr, u64)> {
        let m = self.mtp.as_ref()?;
        let row = u64::from(self.d) * 4;
        Some((m.carry.ptr().offset(u64::from(slot) * u64::from(m.carry_rows) * row), row))
    }

    /// The hidden states kept per sequence and the rows per sequence, for
    /// tests that put a hidden state on the device.
    pub fn draft_carry(&self) -> Option<(&DeviceBuffer, u32)> {
        self.mtp.as_ref().map(|m| (&m.carry, m.carry_rows))
    }

    /// The chain's scratch and, in it, the offsets of the block's output of
    /// link `link` (a row of `d` values per chain), of the logits of the last
    /// link that ran (a row of the draft head per chain) and of the picks of
    /// link `link` ([`SamplePick`] records): for tests.
    pub fn draft_scratch(&self, link: u32) -> Option<(&DeviceBuffer, [u64; 3])> {
        let m = self.mtp.as_ref()?;
        let rows = u64::from(m.max_chains);
        Some((
            &m.scratch,
            [
                m.s.out + u64::from(link) * rows * u64::from(self.d) * 4,
                m.s.logits,
                m.s.picks + u64::from(link) * rows * SamplePick::BYTES as u64,
            ],
        ))
    }

    /// The tokens of the columns of this part's logits (a u32 each) when a
    /// draft vocabulary reorders the output head; null when column `i` is
    /// the token of the part's first row plus `i`.
    pub fn head_tokens(&self) -> DevicePtr {
        self.w.head_tokens.map_or(DevicePtr::NULL, |at| self.w.at(at))
    }

    /// The rows of the draft head.
    pub fn draft_head_rows(&self) -> u32 {
        self.mtp.as_ref().map_or(0, |m| m.w.head.n_out)
    }

    /// The commit of the last step's windows, for the draft block: the hidden
    /// state of each window's last accepted row becomes the one its sequence
    /// goes on from.
    pub(super) fn draft_commit(
        &mut self,
        stream: &ares_cuda::Stream,
        step: DevicePtr,
        dims: StepDims,
    ) -> Result<(), ModelError> {
        let Some(m) = &self.mtp else { return Ok(()) };
        // SAFETY: `step` is the commit record the caller uploaded; the carry
        // holds `carry_rows` rows for every slot.
        unsafe {
            m.sampler.commit_hidden(
                stream,
                step,
                dims.n_segs,
                m.carry.ptr(),
                m.carry_rows,
                self.d,
            )?;
        }
        Ok(())
    }

    /// For tests of the block alone: the draft block's part of
    /// [`Part::enqueue_commit`].
    pub fn commit_drafts(
        &mut self,
        stream: &ares_cuda::Stream,
        step: DevicePtr,
        dims: StepDims,
    ) -> Result<(), ModelError> {
        self.draft_commit(stream, step, dims)
    }

    /// Before the model's blocks: uploads what the draft block's kernels of
    /// this step read and enqueues the chains.
    pub(super) fn draft_before(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let Some(mut m) = self.mtp.take() else { return Ok(()) };
        let result = self.draft_chains(env, &mut m);
        self.mtp = Some(m);
        result
    }

    /// After the model's blocks and its head: the block's keys and values of
    /// the step's rows, and the hidden states the next step starts from.
    pub(super) fn draft_after(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let Some(mut m) = self.mtp.take() else { return Ok(()) };
        let (x, xn) = (self.at(self.s.x), self.at(self.s.xn));
        let result = (|| {
            if let Some(follow) = &m.clock.follow {
                follow[0].record(env.stream)?;
            }
            // SAFETY: `x` holds the residual stream of the step's rows after
            // the last block, `xn` has room for as many rows.
            unsafe {
                self.kernels.rms_norm_f32(
                    env.stream,
                    x,
                    self.w.at(self.w.output_norm),
                    xn,
                    self.d,
                    env.dims.n_rows,
                    self.eps,
                )?;
            }
            self.draft_follow(env, &m)?;
            if let Some(follow) = &m.clock.follow {
                follow[1].record(env.stream)?;
                m.clock.follow_pending = true;
            }
            self.mark(env.stream, Phase::Other)
        })();
        self.mtp = Some(m);
        result
    }

    /// For tests of the block alone: follows the rows of `env` with `hidden`
    /// as the model's final hidden states of those rows (`d` values each),
    /// after [`Part::enqueue_drafts`] for the same step.
    pub fn enqueue_follow(&mut self, env: &StepEnv<'_>, hidden: &[f32]) -> Result<(), ModelError> {
        let Some(m) = self.mtp.take() else { return Err(invalid("no draft block")) };
        let result = (|| {
            if hidden.len() != env.dims.n_rows as usize * self.d as usize {
                return Err(invalid("one hidden state per row of the step"));
            }
            let bytes: Vec<u8> = hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
            env.stream.synchronize()?;
            self.scratch.write(self.s.xn, &bytes)?;
            self.stage_embedding(env)?;
            self.draft_follow(env, &m)
        })();
        self.mtp = Some(m);
        result
    }

    /// For tests of the block alone: what [`Part::enqueue`] does for the
    /// draft block before the model's blocks, with the preparation of the
    /// step's K/V entries that comes first there.
    pub fn enqueue_drafts(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.mtp.is_none() {
            return Err(invalid("no draft block"));
        }
        // The exact entries and the table of the block's layer are made ready
        // for the step, as `Part::enqueue` does in its first stage.
        self.prepare_kv(env)?;
        self.draft_before(env)
    }

    /// Writes the records of the step and enqueues their upload and the
    /// chains.
    fn draft_chains(&mut self, env: &StepEnv<'_>, m: &mut Mtp) -> Result<(), ModelError> {
        m.flush_clock()?;
        m.clock.last_chain_ms = 0.0;
        let step =
            m.pending.take().unwrap_or(DraftStep { tokens: DevicePtr::NULL, chains: Vec::new() });
        let stream = env.stream;
        let geo = m.at.geo;

        // The chains, longest first, so that the chains of a link are the
        // first rows of the link before; `first[c]` is where chain `c` of the
        // caller's order has its drafts in the list for the host.
        let mut first = Vec::with_capacity(step.chains.len());
        let mut total = 0u32;
        for chain in &step.chains {
            let segment = env
                .segments
                .get(chain.segment as usize)
                .ok_or_else(|| invalid(format!("the step has no segment {}", chain.segment)))?;
            if segment.direct || segment.n_rows as usize != chain.rows.len() + 1 {
                return Err(invalid(format!(
                    "segment {segment:?} does not verify {} drafts",
                    chain.rows.len()
                )));
            }
            first.push(total);
            total += chain.rows.len() as u32;
        }
        let mut order: Vec<usize> = (0..step.chains.len()).collect();
        order.sort_by_key(|&c| std::cmp::Reverse(step.chains[c].rows.len()));
        let n_links = order.first().map_or(0, |&c| step.chains[c].rows.len());

        let capacity = m.host.len() as u64;
        let full = || invalid("the draft block's step records do not fit their buffer");
        let buf: &mut [u8] = &mut m.host;
        let mut end = 0u64;
        let mut links = Vec::with_capacity(n_links);
        for j in 0..n_links {
            let chains: Vec<&DraftChain> = order
                .iter()
                .map(|&c| &step.chains[c])
                .take_while(|chain| chain.rows.len() > j)
                .collect();
            let n = chains.len() as u32;
            let n_next =
                order.iter().filter(|&&c| step.chains[c].rows.len() > j + 1).count() as u32;
            let segments: Vec<Segment> = chains
                .iter()
                .enumerate()
                .map(|(row, chain)| {
                    let s = env.segments[chain.segment as usize];
                    Segment {
                        row0: row as u32,
                        n_rows: 1,
                        slot: s.slot,
                        pos0: s.pos0 + j as i32,
                        direct: true,
                    }
                })
                .collect();
            // Link 0 reads the newest tokens; the others get theirs on the
            // device from the picks of the link before.
            let tokens: Vec<u32> =
                chains.iter().map(|c| if j == 0 { c.token } else { 0 }).collect();
            let record = StepRecord::new(&segments, &tokens, None, &[])?;
            let bytes = record.as_bytes();
            let mut planner = Planner::new(geo, self.plan_config, n);
            for (row, segment) in segments.iter().enumerate() {
                let table = segment.slot * self.table_stride;
                let need = (segment.pos0 as u32 + 1).div_ceil(ENTRY_CELLS) as usize;
                let entries = &m.kv.entries(&self.page_tab)[table as usize..table as usize + need];
                planner.dense(table, entries, row as u32, 1, segment.pos0 as u32);
            }
            let plan = planner.finish();
            let plan_bytes = plan.layout().bytes;
            if plan.state_bytes(&geo) > self.s.attn_state_bytes {
                return Err(invalid("a draft link's attention plan needs more partial states"));
            }
            let record_at = end;
            let after = record_at + (bytes.len() as u64).next_multiple_of(16);
            let need = after + u64::from(n) * (32 + 8 + 8) + 48 + plan_bytes;
            if need > capacity {
                return Err(full());
            }
            buf[record_at as usize..record_at as usize + bytes.len()].copy_from_slice(bytes);
            let off_tokens =
                u32::from_le_bytes(bytes[20..24].try_into().expect("the record's header"));
            let params = after;
            let mut at = params;
            for chain in &chains {
                buf[at as usize..at as usize + SampleRow::BYTES]
                    .copy_from_slice(&chain.rows[j].to_bytes());
                at += SampleRow::BYTES as u64;
            }
            let record_rows = at;
            let rows_of =
                chains.iter().map(|c| env.segments[c.segment as usize].row0 + 1 + j as u32);
            let draft_at = words(buf, record_rows, rows_of);
            let slots = order.iter().take(n as usize).map(|&c| first[c] + j as u32);
            let plan_at = words(buf, draft_at, slots);
            plan.write(&mut buf[plan_at as usize..(plan_at + plan_bytes) as usize]);
            end = (plan_at + plan_bytes).next_multiple_of(16);
            links.push(Link {
                n,
                n_next,
                dims: record.dims(),
                record: record_at,
                tokens: record_at + u64::from(off_tokens),
                params,
                record_rows,
                draft_at,
                plan_at,
                sample: SamplePlan::of(chains.iter().map(|c| &c.rows[j])),
                plan,
            });
        }
        // The rows of the carry link 0 starts from: row 0 of each sequence.
        let carry_index = end;
        let starts = order
            .iter()
            .map(|&c| env.segments[step.chains[c].segment as usize].slot * m.carry_rows);
        if carry_index + order.len() as u64 * 4 + 16 + u64::from(env.dims.n_rows) * 16 > capacity {
            return Err(full());
        }
        let follow_rows = words(buf, carry_index, starts);
        let mut rows = Vec::with_capacity(env.dims.n_rows as usize);
        rows.resize(env.dims.n_rows as usize, AttnRow::default());
        for s in env.segments {
            for i in 0..s.n_rows {
                rows[(s.row0 + i) as usize] = AttnRow {
                    table: s.slot * self.table_stride,
                    pos: s.pos0 as u32 + i,
                    slot_base: 0,
                    n_slots: 0,
                };
            }
        }
        let row_bytes = AttnRow::to_bytes(&rows);
        buf[follow_rows as usize..follow_rows as usize + row_bytes.len()]
            .copy_from_slice(&row_bytes);
        end = follow_rows + row_bytes.len() as u64;
        // The rows of the newest tokens, for link 0, in the order of its rows;
        // the host stages them when the device does not keep every row.
        let newest_at = end.next_multiple_of(16);
        m.newest = match &self.embed {
            Some(host) => {
                let size = host.row_bytes();
                let rows = order.iter().map(|&c| step.chains[c].token);
                if newest_at + order.len() as u64 * size as u64 > capacity {
                    return Err(full());
                }
                for (i, token) in rows.enumerate() {
                    if let Some(row) = host.row(token) {
                        let at = newest_at as usize + i * size;
                        buf[at..at + size].copy_from_slice(row);
                    }
                }
                end = newest_at + order.len() as u64 * size as u64;
                m.records.ptr().offset(newest_at)
            }
            None => DevicePtr::NULL,
        };
        m.follow_rows = follow_rows;
        m.drafted = total;
        // SAFETY: the pinned buffer is not written again before the next
        // step's records, which are written after this step has run; the
        // device buffer has its size.
        unsafe { stream.copy_from_host(m.records.ptr(), m.host.as_ptr(), end as usize)? };
        if links.is_empty() {
            return Ok(());
        }

        m.clock.chain[0].record(stream)?;
        m.clock.chain_pending = true;
        m.clock.links = links.len() as u64;
        let rec = |at: u64| m.records.ptr().offset(at);
        // SAFETY: the carry holds a row 0 for every slot and the chain's
        // scratch a row of `d` values per chain.
        unsafe {
            m.sampler.gather_hidden(
                stream,
                m.at(m.s.h),
                m.carry.ptr(),
                rec(carry_index),
                links[0].n,
                self.d,
            )?;
        }
        for (j, link) in links.iter().enumerate() {
            let next_tokens = links.get(j + 1).map_or(DevicePtr::NULL, |next| rec(next.tokens));
            self.draft_link(env, m, j as u32, link, step.tokens, next_tokens)?;
        }
        m.clock.chain[1].record(stream)?;
        self.mark(stream, Phase::Other)
    }

    /// Enqueues link `j`: the block on the rows of the link, the draft head,
    /// the picks, and the picks to where they are read.
    fn draft_link(
        &mut self,
        env: &StepEnv<'_>,
        m: &Mtp,
        j: u32,
        link: &Link,
        record_tokens: DevicePtr,
        next_tokens: DevicePtr,
    ) -> Result<(), ModelError> {
        let (stream, n, d, eps) = (env.stream, link.n, self.d, self.eps);
        let w = &m.w;
        let MixerW::Attention(a) = &w.block.mixer else { return Err(invalid("draft attention")) };
        let ffn = w.block.ffn.as_ref().ok_or_else(|| invalid("draft ffn"))?;
        // Link 0 embeds the newest tokens, whose rows the host staged; the
        // others embed the picks of the link before, which the device keeps.
        let source = self
            .w
            .embed_rows(if j == 0 { m.newest } else { DevicePtr::NULL })
            .ok_or_else(|| invalid("the draft block's device does not embed"))?;
        let attention = self.attention.clone().ok_or_else(|| invalid("attention kernels"))?;
        let geo = m.at.geo;
        let s = m.s;
        let rows = u64::from(m.max_chains);
        let rec = m.records.ptr();
        let step = rec.offset(link.record);
        let (e, x, xn, p) = (m.at(s.e), m.at(s.x), m.at(s.xn), m.at(s.p));
        // Link 0 reads the hidden states gathered from the carries, link j the
        // output of the link before.
        let out_of = |link: u32| m.at(s.out + u64::from(link) * rows * u64::from(d) * 4);
        let h = if j == 0 { m.at(s.h) } else { out_of(j - 1) };
        let out = out_of(j);
        let picks = m.at(s.picks + u64::from(j) * rows * SamplePick::BYTES as u64);
        let pool = env.pages.offset(m.at.pool);
        let (tables, exact) = (m.kv.table(env.tables), exact_pool(m));
        let shape = AttentionShape { n_head: a.n_head, n_kv_head: a.n_kv, head_dim: geo.head_dim };
        let inner = ffn.gate.n_out;
        // SAFETY: the chain's scratch holds `max_chains` rows of every buffer
        // and `n` is at most that; the records hold what `draft_chains` wrote
        // for this link; weights, tables and pool as in `Part::attend`.
        unsafe {
            let plan = link.plan.on_device(rec.offset(link.plan_at));
            self.layers.embed_gather(stream, step, link.dims, &source, e)?;
            // x = proj_e e' + proj_h h'
            self.kernels.rms_norm_f32(stream, e, self.w.at(w.enorm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&w.proj_e], Prepared::default())?;
            self.product(stream, &w.proj_e, n, x)?;
            self.kernels.rms_norm_f32(stream, h, self.w.at(w.hnorm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&w.proj_h], Prepared::default())?;
            self.product(stream, &w.proj_h, n, p)?;
            self.kernels.add_f32(stream, x, p, n * d)?;
            // Attention over the block's own keys and values.
            self.kernels.rms_norm_f32(stream, x, self.w.at(w.block.attn_norm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&a.q, &a.k, &a.v], Prepared::default())?;
            self.product(stream, &a.q, n, m.at(s.qg))?;
            self.product(stream, &a.k, n, m.at(s.k_raw))?;
            self.product(stream, &a.v, n, m.at(s.v))?;
            self.layers.rope_table(stream, step, link.dims, m.at(s.rope), self.rope)?;
            self.layers.attn_prep(
                stream,
                step,
                link.dims,
                shape,
                self.rope.dim,
                eps,
                m.at(s.qg),
                m.at(s.k_raw),
                self.w.at(a.q_norm),
                self.w.at(a.k_norm),
                m.at(s.rope),
                m.at(s.q),
                Some(m.at(s.gate)),
                m.at(s.k),
            )?;
            attention.append(
                stream,
                &geo,
                plan.rows(),
                n,
                m.at(s.k),
                m.at(s.v),
                tables,
                exact,
                pool,
                self.at(self.s.status),
            )?;
            attention.attend(
                stream,
                &geo,
                &plan,
                &AttendBuffers {
                    tables,
                    exact_pool: exact,
                    converted_pool: pool,
                    q: m.at(s.q),
                    gate: Some(m.at(s.gate)),
                    out: m.at(s.o),
                    scratch: self.at(self.s.attn_state),
                },
                1.0 / (self.attn_head_dim as f32).sqrt(),
            )?;
            self.prepare(stream, m.at(s.o), a.out.n_in, n, &[&a.out], Prepared::default())?;
            self.product(stream, &a.out, n, p)?;
            self.kernels.add_f32(stream, x, p, n * d)?;
            // The feed-forward part.
            self.kernels.rms_norm_f32(stream, x, self.w.at(w.block.post_norm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&ffn.gate, &ffn.up], Prepared::default())?;
            self.product(stream, &ffn.gate, n, m.at(s.g1))?;
            self.product(stream, &ffn.up, n, m.at(s.g2))?;
            self.swiglu(stream, m.at(s.g1), m.at(s.g2), inner, n, &ffn.down)?;
            self.product(stream, &ffn.down, n, p)?;
            self.kernels.add_f32(stream, x, p, n * d)?;
            // The output, the draft head and the picks.
            self.kernels.rms_norm_f32(stream, x, self.w.at(w.head_norm), out, d, n, eps)?;
            self.prepare(stream, out, d, n, &[&w.head], Prepared::default())?;
            self.product(stream, &w.head, n, m.at(s.logits))?;
            let batch = SampleBatch {
                logits: m.at(s.logits),
                stride: w.head.n_out,
                n_rows: n,
                capacity: m.max_chains,
                n_local: w.head.n_out,
                tokens: m.head_tokens,
                first: 0,
                // A draft is picked from the logits as they are: no corrections,
                // masks or probabilities (see `SampleRow::for_draft`).
                inverse: DevicePtr::NULL,
                vocab: 0,
                params: rec.offset(link.params),
                entries: DevicePtr::NULL,
                masks: DevicePtr::NULL,
                mask_words: 0,
                out: picks,
                logprobs: DevicePtr::NULL,
                scratch: m.at(s.sampler),
                exchange: m.at(s.exchange),
                devices: 1,
                rank: 0,
            };
            m.sampler.sample(stream, &batch, link.sample, |_, _| Ok::<(), ares_cuda::Error>(()))?;
            m.sampler.draft_put(
                stream,
                picks,
                n,
                next_tokens,
                link.n_next,
                record_tokens,
                rec.offset(link.record_rows),
                m.at(s.drafts),
                rec.offset(link.draft_at),
            )?;
        }
        Ok(())
    }

    /// The block's keys and values of the step's rows from the final hidden
    /// states in the step's `xn`, and those hidden states into the carries.
    fn draft_follow(&mut self, env: &StepEnv<'_>, m: &Mtp) -> Result<(), ModelError> {
        let (stream, dims, n, d, eps) = (env.stream, env.dims, env.dims.n_rows, self.d, self.eps);
        let w = &m.w;
        let MixerW::Attention(a) = &w.block.mixer else { return Err(invalid("draft attention")) };
        let source = self.embed_rows()?;
        let attention = self.attention.clone().ok_or_else(|| invalid("attention kernels"))?;
        let geo = m.at.geo;
        let s = self.s;
        let (x, xn, p) = (self.at(s.x), self.at(s.xn), self.at(s.p));
        let (k_raw, k, v) = (self.at(s.w2), self.at(s.k), self.at(s.v));
        // Only the keys are wanted of what prepares attention's inputs: the
        // queries it computes next to them come from, and go to, memory
        // nothing reads.
        let shape = AttentionShape { n_head: a.n_kv, n_kv_head: a.n_kv, head_dim: geo.head_dim };
        // SAFETY: after the model's blocks and its head the step's scratch is
        // free; its buffers hold the step's rows of `d` values and of all KV
        // heads of the block (`Part::open` sized them for it); `xn` holds the
        // final hidden states. Weights, tables and pool as in `Part::attend`.
        unsafe {
            // The hidden state of the position before each row, then this
            // step's into the carries.
            m.sampler.shift_hidden(
                stream,
                env.step,
                dims.n_segs,
                dims.max_seg_rows,
                xn,
                m.carry.ptr(),
                m.carry_rows,
                x,
                d,
            )?;
            m.sampler.keep_hidden(
                stream,
                env.step,
                dims.n_segs,
                dims.max_seg_rows,
                xn,
                m.carry.ptr(),
                m.carry_rows,
                d,
            )?;
            self.layers.embed_gather(stream, env.step, dims, &source, p)?;
            self.kernels.rms_norm_f32(stream, p, self.w.at(w.enorm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&w.proj_e], Prepared::default())?;
            self.product(stream, &w.proj_e, n, p)?;
            self.kernels.rms_norm_f32(stream, x, self.w.at(w.hnorm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&w.proj_h], Prepared::default())?;
            self.product(stream, &w.proj_h, n, x)?;
            self.kernels.add_f32(stream, x, p, n * d)?;
            self.kernels.rms_norm_f32(stream, x, self.w.at(w.block.attn_norm), xn, d, n, eps)?;
            self.prepare(stream, xn, d, n, &[&a.k, &a.v], Prepared::default())?;
            self.product(stream, &a.k, n, k_raw)?;
            self.product(stream, &a.v, n, v)?;
            self.layers.rope_table(stream, env.step, dims, self.at(s.rope), self.rope)?;
            self.layers.attn_prep(
                stream,
                env.step,
                dims,
                shape,
                self.rope.dim,
                eps,
                self.at(s.w1),
                k_raw,
                self.w.at(a.q_norm),
                self.w.at(a.k_norm),
                self.at(s.rope),
                self.at(s.q),
                None,
                k,
            )?;
            // The rows of a replay have their cells in the pool already.
            m.kv.append(
                &attention,
                stream,
                &m.at,
                m.records.ptr().offset(m.follow_rows),
                k,
                v,
                env.tables,
                env.pages,
                self.at(s.status),
                env.replay,
                0,
                n,
            )?;
        }
        Ok(())
    }

    /// Adds the draft block's events of the last step to its times.
    pub(super) fn draft_flush(&mut self) -> Result<(), ModelError> {
        match &mut self.mtp {
            Some(m) => m.flush_clock(),
            None => Ok(()),
        }
    }
}
