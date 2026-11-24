//! What the server asks of the model before it opens it: where a spread puts
//! the weights (the placement [`Qwen35::new`] makes) and what each device
//! would take, estimated without a device.
//!
//! The runtime reserves the pages of sequence state before the parts are
//! built, and a part's sizes cannot be asked without a device. So the context
//! that fits comes from an estimate ([`Footprint`]): weights from the
//! placement and the tensor table, recurrent state and the scratch of a step
//! from the model's dimensions. The server logs what each part really took
//! next to it once the model is loaded.
//!
//! Also here: the arithmetic of a tensor split's shares ([`fit_shares`]) and
//! where the KV heads of the attention blocks go for a context
//! ([`Estimate::attention_for_context`]).
//!
//! [`Qwen35::new`]: super::Qwen35::new

use ares_core::units::MIB;
use ares_gguf::Gguf;
use ares_runtime::Split;

use super::part::{Options, attention_state_bytes, exact_bytes, page_bytes};
use super::placement::{Placement, PlacementError, block_parts, matrix_bytes};
use super::runtime::Spread;
use crate::qwen35::{Config, Result, Weights};

/// What the parts of a model take on one device, by estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Footprint {
    pub weights: u64,
    /// The scratch of a step and the small buffers.
    pub step: u64,
    /// The recurrent state of one sequence.
    pub per_sequence: u64,
    /// One page of K and V.
    pub per_page: u64,
    /// The exact K/V entries of the formats with a tile mean, for the
    /// sequences that run at once: a pool of its own, whatever the context.
    pub exact: u64,
}

impl Footprint {
    /// What the device takes without K and V, with `parallel` sequences.
    pub fn fixed(&self, parallel: u32) -> u64 {
        self.weights + self.step + self.exact + self.per_sequence * u64::from(parallel)
    }

    /// What the device takes with `parallel` sequences and `pages` pages.
    pub fn total(&self, parallel: u32, pages: u32) -> u64 {
        self.fixed(parallel) + self.per_page * u64::from(pages)
    }
}

/// What the whole model takes when it is shared out: the part that is
/// divided among the devices by their shares, and what every device takes
/// besides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Whole {
    pub shared: u64,
    pub per_device: u64,
}

/// A model file read for the plan: its configuration and its table of
/// tensors (no weight is read).
pub struct Estimate<'a> {
    config: Config,
    weights: Weights<'a>,
}

impl std::fmt::Debug for Estimate<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Estimate").field("config", &self.config).finish_non_exhaustive()
    }
}

impl<'a> Estimate<'a> {
    /// Reads the configuration and the tensor table of `gguf`.
    pub fn new(gguf: &'a Gguf) -> Result<Self> {
        let config = Config::from_gguf(gguf)?;
        let weights = Weights::load(gguf, &config)?;
        Ok(Self { config, weights })
    }

    /// The same with the MTP block of `file` (architecture
    /// [`crate::qwen35::MTP_ARCHITECTURE`]) in place of the model file's: a
    /// file that is no MTP block of this model is refused by what it is.
    pub fn with_mtp_file(self, file: &'a Gguf) -> Result<Self> {
        let weights = self.weights.with_mtp_file(file, &self.config)?;
        Ok(Self { weights, ..self })
    }

    /// [`Estimate::new`] with the MTP block's file that `options` names.
    pub fn for_options(gguf: &'a Gguf, options: &'a Options) -> Result<Self> {
        let estimate = Self::new(gguf)?;
        match &options.mtp_file {
            Some(file) => estimate.with_mtp_file(file),
            None => Ok(estimate),
        }
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The weights' tensor table.
    pub(super) fn weights(&self) -> &Weights<'a> {
        &self.weights
    }

    /// Where `spread` puts the weights, with the drafter where the model
    /// opened with `options` puts it: a model with an MTP block (in its file
    /// or a file of its own) drafts with it when the options ask for it and
    /// windows may carry drafts, on the device with the first rows of the head
    /// (the one that embeds, or the last one of a layer split). A DFlash
    /// drafter reads taps of the model's layers on the device that embeds,
    /// which a layer split does not give it.
    pub fn place(
        &self,
        spread: &Spread,
        options: &Options,
    ) -> std::result::Result<Placement, PlacementError> {
        let c = &self.config;
        let placement = spread.place(c, &self.weights)?;
        let drafts = options.window_rows > 1;
        let mtp = (c.n_nextn == 1 || options.mtp_file.is_some()) && options.mtp;
        let vocabulary = options.draft_vocabulary.as_deref();
        if drafts && mtp {
            placement.with_mtp(c, vocabulary)
        } else if drafts && options.dflash.is_some() && placement.split != Split::Layer {
            placement.with_head_view(c, vocabulary, false)
        } else {
            Ok(placement)
        }
    }

    /// What `placement` puts on each device with `options`, for `parallel`
    /// sequences on devices of `multiprocessors` multiprocessors (the last
    /// one's for a device the list does not reach).
    ///
    /// The weights follow the loader (`load.rs`): a device's slice of every
    /// block, the embedding rows the draft chain picks, its rows of the head,
    /// the MTP block. The scratch of a step follows `Part::open` with the
    /// widths of the whole model, which a slice does not exceed, and the
    /// partial states of the attention plans that the device's heads and its
    /// multiprocessors take ([`attention_state_bytes`]). Both are estimates:
    /// the loader aligns and rearranges, and the part has more small buffers
    /// than are counted here, for which a margin is added.
    pub fn footprint(
        &self,
        placement: &Placement,
        options: &Options,
        parallel: u32,
        multiprocessors: &[u32],
    ) -> Vec<Footprint> {
        let (c, w) = (&self.config, &self.weights);
        let r = &c.recurrent;
        let a = &c.attention;
        let (d, f) = (c.n_embd as u64, c.n_ff as u64);
        let (kd, vd, cd) = (r.key_dim() as u64, r.value_dim() as u64, r.conv_dim() as u64);
        let (hq, hkv) = ((a.n_head * a.head_dim) as u64, (a.n_head_kv * a.head_dim) as u64);
        let rows = u64::from(options.max_rows);
        let widest = d.max(vd).max(hq).max(f);
        let per_row = 4
            * (3 * d
                + cd.max(2 * hq).max(f)
                + vd.max(hkv).max(f)
                + 2 * r.n_v_heads as u64
                + kd.max(hq)
                + (kd + vd).max(hq)
                + vd.max(hq)
                + 2 * hkv)
            + 4 * a.rope.dim as u64
            + 3 * widest;
        let values_per_key = (r.n_v_heads / r.n_k_heads.max(1)) as u64;

        placement
            .devices
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                let mut weights = 0u64;
                let mut per_sequence = 0u64;
                for (layer, slice) in plan.blocks.iter().enumerate() {
                    let Some(slice) = slice else { continue };
                    let (mixer, ffn) = block_parts(w, layer);
                    let heads = slice.heads.len();
                    weights += part_of(ffn, slice.ffn.len(), c.n_ff);
                    if c.is_attention(layer) {
                        weights += part_of(mixer, heads, a.n_head_kv);
                    } else {
                        weights += part_of(mixer, heads, r.n_k_heads);
                        // The convolution's window and the delta-rule
                        // matrices of the heads this device computes.
                        let keys = heads as u64 * r.head_k_dim as u64;
                        let value_heads = heads as u64 * values_per_key;
                        let values = value_heads * r.head_v_dim as u64;
                        let conv = (2 * keys + values) * (r.conv_kernel as u64).saturating_sub(1);
                        let delta = value_heads * (r.head_k_dim * r.head_v_dim) as u64;
                        per_sequence += (conv + delta) * options.state.bytes();
                    }
                }
                // The embedding table lies in host memory. The device that
                // embeds keeps the rows of the tokens the draft chain picks,
                // as a matrix of their own unless the head is the table,
                // whose rows they are then.
                if let Some(draft) =
                    placement.mtp().filter(|draft| draft.device == index && w.output.is_some())
                {
                    weights +=
                        part_of(matrix_bytes(&w.token_embd), draft.head_rows as usize, c.n_vocab);
                }
                weights += part_of(matrix_bytes(w.head()), plan.head.len(), c.n_vocab);
                if let (Some(_), Some(nextn)) =
                    (placement.mtp().filter(|draft| draft.device == index), &w.nextn)
                {
                    let at = &nextn.attention;
                    weights += matrix_bytes(&nextn.block.ffn_gate)
                        + matrix_bytes(&nextn.block.ffn_up)
                        + matrix_bytes(&nextn.block.ffn_down)
                        + matrix_bytes(&at.q)
                        + matrix_bytes(&at.k)
                        + matrix_bytes(&at.v)
                        + matrix_bytes(&at.out)
                        + matrix_bytes(&nextn.eh_proj);
                }
                // A DFlash drafter on the device of the draft plan: its
                // weights, its ring per sequence, its scratch.
                let mut step_extra = 0u64;
                if let (Some(file), Some(draft)) =
                    (&options.dflash, placement.draft.as_ref().filter(|d| d.device == index))
                    && let Ok(fw) = file.weights()
                {
                    let setup = |slots: u32| crate::dflash::DrafterSetup {
                        max_slots: slots,
                        max_rows: options.max_rows,
                        window_rows: options.window_rows,
                        head_rows: draft.head_rows,
                    };
                    let one = crate::dflash::bytes_of(&file.config, &fw, &setup(1));
                    let two = crate::dflash::bytes_of(&file.config, &fw, &setup(2));
                    weights += one.weights;
                    per_sequence += two.total() - one.total();
                    step_extra += one.scratch + one.records;
                }
                let logits = u64::from(options.logit_rows) * 4 * (d + plan.head.len() as u64);
                let sms =
                    multiprocessors.get(index).or(multiprocessors.last()).copied().unwrap_or(1);
                step_extra += attention_state_bytes(c, placement, index, options, parallel, sms);
                Footprint {
                    weights,
                    // The margin: alignment of the arena, the sampler,
                    // records, attention plans, links, the MTP block's
                    // buffers.
                    step: rows * per_row + logits + weights / 64 + 64 * MIB + step_extra,
                    per_sequence,
                    per_page: page_bytes(c, placement, index, options),
                    exact: exact_bytes(c, placement, index, options, parallel),
                }
            })
            .collect()
    }

    /// The footprint of `spread` placed ([`Estimate::place`]).
    pub fn footprint_of(
        &self,
        spread: &Spread,
        options: &Options,
        parallel: u32,
        multiprocessors: &[u32],
    ) -> std::result::Result<Vec<Footprint>, PlacementError> {
        Ok(self.footprint(&self.place(spread, options)?, options, parallel, multiprocessors))
    }

    /// What the whole model takes with `options` and `pages` pages of context
    /// for `parallel` sequences when it is shared out over devices of
    /// `multiprocessors` multiprocessors (the most any of them has).
    pub fn whole(
        &self,
        options: &Options,
        parallel: u32,
        pages: u32,
        multiprocessors: u32,
    ) -> Whole {
        // One device with everything gives the totals. A split divides
        // weights, recurrent state and K/V by the shares, near enough, and
        // repeats the scratch of a step on every device.
        let single = Placement::single(&self.config);
        let all = self.footprint(&single, options, parallel, &[multiprocessors])[0];
        let margin = all.weights / 64;
        Whole {
            shared: all.weights
                + all.exact
                + margin
                + all.per_sequence * u64::from(parallel)
                + all.per_page * u64::from(pages),
            per_device: all.step - margin,
        }
    }

    /// The shares of the attention heads of a tensor split by `shares` for a
    /// context of `pages` pages ([`heads_for_context`]), or, when the devices
    /// cannot hold that many wherever the heads are, the shares with which
    /// they hold the most ([`attention_by_room`]); by estimate. `free` is what
    /// each device has free, `headroom` what is left unreserved of each.
    /// `None` when no device has room for the heads next to its share of the
    /// weights.
    ///
    /// What a device takes besides the heads is its footprint with every head
    /// on another device: its share of the weights, the scratch, the recurrent
    /// state, and the K and V of the MTP block, which has heads of its own on
    /// the device that holds it (296 MiB for 262 144 tokens of Qwen3.8-27B with
    /// K and V in `q5`). What the heads take is the difference with all of
    /// them on the device, the scratch that grows with the weights included. A
    /// device holds the heads of each block whole, so it can hold up to a head
    /// more than its share (about 100 MiB of the 27B's at that context): the
    /// shares are checked against the footprint of the placement they make,
    /// and a device that is over gives heads to the others until none is (when
    /// a round leaves a device over, the context that fits is smaller, which
    /// the server's plan says or refuses).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_for_context(
        &self,
        options: &Options,
        parallel: u32,
        free: &[u64],
        multiprocessors: &[u32],
        shares: &[f64],
        headroom: u64,
        pages: u32,
    ) -> std::result::Result<Option<Vec<f64>>, PlacementError> {
        /// Rounds of checking the shares against the footprint they make: one
        /// moves the heads a device is over by, at least one, and the scratch
        /// of the partial states grows with the most heads a block of the
        /// device has.
        const ROUNDS: usize = 8;
        let c = &self.config;
        let n = free.len();
        let footprint_of = |attention: Vec<f64>| {
            let spread = Spread::TensorWithAttention(shares.to_vec(), attention);
            self.footprint_of(&spread, options, parallel, multiprocessors)
        };
        let alone_on =
            |on: usize| -> Vec<f64> { (0..n).map(|i| f64::from(u8::from(i == on))).collect() };
        let usable: Vec<u64> = free.iter().map(|f| f.saturating_sub(headroom)).collect();
        let (mut room, mut room_for_pages) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut heads, mut page) = (0, 0);
        for (i, usable) in usable.iter().enumerate() {
            let all = footprint_of(alone_on(i))?[i];
            let none = footprint_of(alone_on((i + 1) % n))?[i];
            room.push(usable.saturating_sub(none.fixed(parallel)));
            room_for_pages.push(usable.saturating_sub(none.total(parallel, pages)));
            heads = heads.max(all.fixed(parallel).saturating_sub(none.fixed(parallel)));
            page = page.max(all.per_page.saturating_sub(none.per_page));
        }
        let Some(mut attention) = heads_for_context(shares, &room_for_pages, heads, page, pages)
        else {
            return Ok(attention_by_room(&room));
        };
        let per_share = (heads as f64 + page as f64 * f64::from(pages)).max(1.0);
        // A device that is over gives up what it is over by, and at least a
        // head: the heads of a block are whole, so less may move none.
        let all_heads =
            (0..c.n_layer).filter(|layer| c.is_attention(*layer)).count() * c.attention.n_head_kv;
        let head = 1.0 / all_heads.max(1) as f64;
        let mut most: Vec<f64> =
            room_for_pages.iter().map(|r| (*r as f64 / per_share).min(1.0)).collect();
        for _ in 0..ROUNDS {
            let planned = footprint_of(attention.clone())?;
            let over: Vec<u64> = planned
                .iter()
                .zip(&usable)
                .map(|(f, usable)| f.total(parallel, pages).saturating_sub(*usable))
                .collect();
            if over.iter().all(|o| *o == 0) {
                break;
            }
            for ((most, over), share) in most.iter_mut().zip(&over).zip(&attention) {
                if *over > 0 {
                    *most = most.min(share - (*over as f64 / per_share).max(head));
                }
            }
            match fit_shares(shares, &most) {
                Some(fitted) => attention = fitted,
                None => return Ok(attention_by_room(&room)),
            }
        }
        Ok(Some(attention))
    }
}

fn part_of(bytes: u64, have: usize, of: usize) -> u64 {
    (bytes * have as u64).div_ceil(of.max(1) as u64)
}

/// Lowers the shares of devices that cannot hold theirs and gives the rest
/// to the others in proportion to what they asked for. `most[i]` is the
/// largest share device `i` has memory for. `None` when the devices together
/// cannot hold the model.
pub fn fit_shares(wanted: &[f64], most: &[f64]) -> Option<Vec<f64>> {
    let mut shares = wanted.to_vec();
    let mut full = vec![false; shares.len()];
    // Each round fills at least one device, so the loop ends.
    for _ in 0..=shares.len() {
        let mut over = 0.0;
        for i in 0..shares.len() {
            if !full[i] && shares[i] > most[i] {
                over += shares[i] - most[i];
                shares[i] = most[i].max(0.0);
                full[i] = true;
            }
        }
        if over <= 1e-9 {
            return Some(shares);
        }
        let open: f64 = (0..shares.len()).filter(|i| !full[*i]).map(|i| shares[i]).sum();
        if open <= 0.0 {
            return None;
        }
        for i in 0..shares.len() {
            if !full[i] {
                shares[i] += over * shares[i] / open;
            }
        }
    }
    None
}

/// Each device's share of the attention heads that lets the devices hold the
/// most context, from `room[i]`: the bytes device `i` has left once it holds
/// everything of the model but the attention heads, their weights and their
/// K and V.
///
/// A head's weights and its K and V take the same bytes on every device, so
/// with `a[i]` of the heads, `w` bytes of all heads' weights and `p` of all
/// heads' K and V per page, `n` pages fit when `a[i] * (w + p * n) <= room[i]`
/// on every device. The most pages fit when the heads follow the rooms. `None`
/// when no device has room.
pub fn attention_by_room(room: &[u64]) -> Option<Vec<f64>> {
    let total: u64 = room.iter().sum();
    (total > 0).then(|| room.iter().map(|r| *r as f64 / total as f64).collect())
}

/// Each device's share of the attention heads for `pages` pages of context:
/// `wanted` (the shares of the weights) lowered where a device's `room` (as
/// for [`attention_by_room`]) cannot hold its heads' weights, `heads` bytes for
/// all heads, and their K and V, `page` bytes a page for all heads, and the
/// rest given to the others ([`fit_shares`]). The heads stay with the weights
/// whenever the context fits that way: attention work on a device that is not
/// the fastest makes a long sequence slower. `None` when the devices cannot
/// hold that many pages.
pub fn heads_for_context(
    wanted: &[f64],
    room: &[u64],
    heads: u64,
    page: u64,
    pages: u32,
) -> Option<Vec<f64>> {
    let all = heads as f64 + page as f64 * f64::from(pages);
    let most: Vec<f64> = room.iter().map(|r| (*r as f64 / all.max(1.0)).min(1.0)).collect();
    fit_shares(wanted, &most)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ares_cuda::CacheFormat;
    use std::path::Path;

    /// Memory of every device the server leaves unreserved (its
    /// `VRAM_HEADROOM`).
    const HEADROOM: u64 = 256 * MIB;

    /// The owner's production settings on the two development cards, as the
    /// owner's fork serves them (2026-11-22): Qwen3.8-27B UD-Q4_K_S
    /// (`ARES_TEST_QWEN38_27B`), a tensor split of the weights 0.625 / 0.375,
    /// 262 144 tokens of context for two sequences, K and V in `q5`, the MTP
    /// block drafting with a draft vocabulary (`ARES_TEST_DRAFT_VOCAB`, else the
    /// head's rows), and the free memory the cards had: 11.52 and 11.57 GiB.
    /// The heads by the context once left the RTX 4070 planned above its free
    /// memory (the MTP block's K and V were not counted), and the load died; the
    /// plan now holds the context on both devices within what they have. A
    /// context they cannot hold is refused by estimate, before anything is
    /// loaded (here with K and V in `f16`).
    #[test]
    fn the_27b_at_the_owners_settings_plans_within_free_memory() {
        let Some(path) = std::env::var_os("ARES_TEST_QWEN38_27B") else {
            eprintln!("skipped: set ARES_TEST_QWEN38_27B");
            return;
        };
        let gguf = Gguf::open(&path).unwrap();
        let vocabulary = std::env::var_os("ARES_TEST_DRAFT_VOCAB")
            .map(|p| crate::files::read_draft_vocabulary(Path::new(&p)).unwrap().into());
        let parallel = 2;
        // The server's default window: the rows of one launch of the few-row
        // kernels.
        let window_rows = parallel * ares_cuda::MAX_BATCH;
        let options = |kv: CacheFormat| Options {
            kv: (kv, kv),
            logit_rows: window_rows.max(parallel),
            window_rows,
            draft_vocabulary: vocabulary.clone(),
            mtp: true,
            ..Options::default()
        };
        let free = [11_796 * MIB, 11_847 * MIB];
        // The RTX 4070 and the RTX 3060.
        let multiprocessors = [46, 28];
        let shares = [0.625, 0.375];
        let context: u32 = 262_144;
        let pages = context.div_ceil(ares_cuda::ENTRY_CELLS) + parallel;
        for (kv, fits) in [(CacheFormat::Q5, true), (CacheFormat::F16, false)] {
            let options = options(kv);
            let estimate = Estimate::new(&gguf).unwrap();
            let attention = estimate
                .attention_for_context(
                    &options,
                    parallel,
                    &free,
                    &multiprocessors,
                    &shares,
                    HEADROOM,
                    pages,
                )
                .unwrap()
                .unwrap();
            let spread = Spread::TensorWithAttention(shares.to_vec(), attention.clone());
            let placement = estimate.place(&spread, &options).unwrap();
            let footprint = estimate.footprint(&placement, &options, parallel, &multiprocessors);
            let mut all_fit = true;
            for (device, f) in footprint.iter().enumerate() {
                let planned = f.total(parallel, pages);
                eprintln!(
                    "{kv:?}: device {device}: heads {:.3}, {} MiB planned for {pages} pages of \
                     {} MiB free",
                    attention[device],
                    planned / MIB,
                    free[device] / MIB
                );
                if fits {
                    assert!(planned + HEADROOM <= free[device], "{kv:?}: device {device}");
                }
                all_fit &= planned + HEADROOM <= free[device];
            }
            assert_eq!(all_fit, fits, "{kv:?}: {pages} pages asked");
            if fits {
                plans_fit_their_states(
                    estimate.config(),
                    &placement,
                    &options,
                    parallel,
                    &multiprocessors,
                );
            }
        }
    }

    /// The attention plans of prompts on each device of `placement` take no more partial
    /// states than the device reserves for them (`attention_state_bytes`, which the part
    /// reserves as well): with every number of KV heads its attention blocks hold, a step
    /// of 256 rows at cell 512 that reads the exact first tile, converted tiles and exact
    /// newest ones (the step of the owner's settings that needed 768 slots where 680 were
    /// reserved, 2026-11-23), a full step past the first tiles, and two sequences that
    /// read a common prefix.
    fn plans_fit_their_states(
        c: &Config,
        placement: &Placement,
        options: &Options,
        parallel: u32,
        multiprocessors: &[u32],
    ) {
        use super::super::PAGE_ENTRY;
        let a = &c.attention;
        let entries = |cells: u32, base: u32, shared: u32, exact_from: u32| -> Vec<u32> {
            (0..cells.div_ceil(ares_cuda::ENTRY_CELLS))
                .map(|e| {
                    let id = if e < shared { 900_000 + e } else { base + e };
                    let exact = e == 0 || e * ares_cuda::ENTRY_CELLS >= exact_from;
                    if exact { id } else { id | PAGE_ENTRY }
                })
                .collect()
        };
        for (index, plan) in placement.devices.iter().enumerate() {
            let reserved = attention_state_bytes(
                c,
                placement,
                index,
                options,
                parallel,
                multiprocessors[index],
            );
            let mut heads: Vec<u32> = (0..c.n_layer)
                .filter(|layer| c.is_attention(*layer))
                .filter_map(|layer| plan.blocks[layer].as_ref())
                .map(|slice| slice.heads.len() as u32)
                .filter(|h| *h > 0)
                .collect();
            heads.sort_unstable();
            heads.dedup();
            for kv_heads in heads {
                let geo = ares_cuda::CacheGeometry {
                    k: options.kv.0,
                    v: options.kv.1,
                    head_dim: a.head_dim as u32,
                    kv_heads,
                    kv_total: a.n_head_kv as u32,
                    first_kv_head: 0,
                    q_per_kv: (a.n_head / a.n_head_kv) as u32,
                };
                let config = ares_cuda::PlanConfig::for_multiprocessors(multiprocessors[index]);
                let steps: [&[(u32, u32, u32, u32)]; 3] = [
                    &[(256, 512, 0, 384)],
                    &[(512, 3584, 0, 3456)],
                    &[(300, 9000, 40, 8832), (212, 7000, 40, 6912)],
                ];
                for step in steps {
                    let rows: u32 = step.iter().map(|s| s.0).sum();
                    let mut planner = ares_cuda::Planner::new(geo, config, rows);
                    let mut row = 0;
                    for (i, &(n, pos0, shared, exact_from)) in step.iter().enumerate() {
                        let table = 10_000 * i as u32;
                        let cells = entries(pos0 + n, 100_000 * (i as u32 + 1), shared, exact_from);
                        planner.dense(table, &cells, row, n, pos0);
                        row += n;
                    }
                    let needed = planner.finish().state_bytes(&geo);
                    assert!(
                        needed <= reserved,
                        "device {index}, {kv_heads} KV heads, {step:?}: {needed} bytes of partial \
                         states, {reserved} reserved"
                    );
                }
            }
        }
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn attention_heads_follow_the_room_and_that_holds_the_most_pages() {
        // Two devices with 2 and 6 GiB left; the heads' weights take 1 GiB,
        // their K and V 1 MiB a page.
        let (w, p) = (1u64 << 30, 1u64 << 20);
        let room = [2u64 << 30, 6 << 30];
        let shares = attention_by_room(&room).unwrap();
        assert_eq!(shares, vec![0.25, 0.75]);
        let pages = |a: &[f64]| {
            (0..2)
                .map(|i| ((room[i] as f64 / a[i] - w as f64) / p as f64).floor() as u64)
                .min()
                .unwrap()
        };
        // (8 GiB - 1 GiB) / 1 MiB: every byte of both devices holds context.
        assert_eq!(pages(&shares), 7 << 10);
        for other in [[0.5, 0.5], [0.2, 0.8], [0.3, 0.7]] {
            assert!(pages(&other) < pages(&shares), "{other:?}");
        }
        assert_eq!(attention_by_room(&[0, 0]), None);
        assert_eq!(attention_by_room(&[0, 5]), Some(vec![0.0, 1.0]));
    }

    #[test]
    fn attention_heads_stay_with_the_weights_until_the_context_needs_room() {
        // Heads of 1 GiB of weights and 1 MiB of K and V a page; 2 and 6 GiB
        // of room; the weights 0.625 and 0.375.
        let (heads, page) = (1u64 << 30, 1u64 << 20);
        let (room, wanted) = ([2u64 << 30, 6 << 30], [0.625, 0.375]);
        let fit = |pages| heads_for_context(&wanted, &room, heads, page, pages);
        // 2 GiB hold 0.625 of 1 GiB + 2048 pages: the heads stay.
        assert_eq!(fit(2048), Some(wanted.to_vec()));
        // 4096 pages: the first device holds 2/5 of the heads, the rest moves.
        let moved = fit(4096).unwrap();
        assert!((moved[0] - 0.4).abs() < 1e-9 && (moved[1] - 0.6).abs() < 1e-9, "{moved:?}");
        // The most the devices hold, (8 - 1) GiB / 1 MiB, still fits; one
        // page more does not.
        let most = fit(7 << 10).unwrap();
        assert!((most[0] - 0.25).abs() < 1e-9, "{most:?}");
        assert_eq!(fit((7 << 10) + 1), None);
    }

    #[test]
    fn a_device_that_is_too_small_gives_its_excess_to_the_others() {
        // Everything fits: nothing changes.
        assert!(close(&fit_shares(&[0.6, 0.4], &[0.7, 0.5]).unwrap(), &[0.6, 0.4]));
        // The first can hold half: the second takes the rest.
        assert!(close(&fit_shares(&[0.6, 0.4], &[0.5, 0.9]).unwrap(), &[0.5, 0.5]));
        // The excess is split in proportion, and may fill a second device.
        let shares = fit_shares(&[0.5, 0.3, 0.2], &[0.2, 0.4, 1.0]).unwrap();
        assert!(close(&shares, &[0.2, 0.4, 0.4]), "{shares:?}");
        assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // Together they are too small.
        assert_eq!(fit_shares(&[0.5, 0.5], &[0.3, 0.3]), None);
        assert_eq!(fit_shares(&[1.0], &[0.9]), None);
    }
}
