//! What the server asks of the model before it opens it: the rows of a step,
//! where the layers go and what a page takes on each device, and the widest
//! step the expert slots can lend for.

use ares_cuda::{Device, MAX_BATCH};
use ares_gguf::Gguf;

use super::super::feed::Layout;
use super::super::options::ModelOptions;
use super::super::part::{Frame, Options, PartShape};
use super::super::placement::{Chosen, PartPlan, Placement, choose};
use super::OpenError;
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

/// The rows of a step to open the model with on `devices`: as many as the scratch of a step
/// may take of the smaller device's memory (a thirty-second; the rest is slots for experts and
/// the pools of K and V), as a power of two between eight (the decode kernels' limit) and
/// 4096.
///
/// A step of more rows reads the experts of a layer once for more of them, and every byte of
/// the scratch is a byte less for expert slots, which the generating steps miss: measured on
/// Flash-Next, 2048 rows (1.4 GiB of scratch on a 12 GiB card) cost a third to a half of the
/// generation speed through the slots they took. The scratch grows with the rows, about 0.7 MiB
/// a row for Flash-Next; `opts` supplies the other choices of the part, whose `max_rows` is
/// ignored.
pub fn step_rows(gguf: &Gguf, devices: &[Device], opts: &Options) -> Result<u32, OpenError> {
    let config = Config::from_gguf(gguf)?;
    let room = devices.iter().map(|d| d.info().total_memory / 32).min().unwrap_or(0);
    let mut rows = WIDE_ROWS;
    while rows > MAX_BATCH {
        let opts = Options { max_rows: rows, ..opts.clone() };
        // The scratch of the part with everything: the widest a part is.
        let shape = PartShape::whole(&config, 0..config.n_layer, true);
        let frame = Frame::new(&config, &opts, &shape, 1)?;
        if frame.step_bytes(&opts) <= room {
            break;
        }
        rows /= 2;
    }
    Ok(rows)
}

/// Whether a model opened with `opts` on `devices` devices, with a DFlash drafter or
/// without (`dflash`), reads a prompt wider than the parts' own scratch layer by layer
/// ([`ModelOptions::pass_rows`]): its micro-batches take the many-row kernels, K and V
/// have no exact pool (whose tiles convert per step for every layer at once), no DFlash
/// drafter reads the residual of a step's rows at some layers, and the placement is one the
/// pass is built for.
pub fn reads_layer_by_layer(opts: &Options, dflash: bool) -> bool {
    opts.own_rows() >= 2 * MAX_BATCH
        && crate::blocks::kv::Group::window_limit(opts.kv.0, opts.kv.1).is_none()
        && !dflash
}

/// The most rows of a step the model is opened with.
const WIDE_ROWS: u32 = 4096;

/// The rows of a lent step are a multiple of this ([`widest_rows`]).
pub const ROW_GRAIN: u32 = 256;

/// The rows of the widest step from `asked` down to `own` that `fits` holds: `asked`, or the
/// largest multiple of [`ROW_GRAIN`] (of `own` where that is less) below it that fits, or
/// `own` when no more rows do. `fits` must hold for fewer rows wherever it holds for more (the
/// scratch grows with the rows), so the rows are found by halving the range between them.
pub fn widest_rows<E>(
    own: u32,
    asked: u32,
    mut fits: impl FnMut(u32) -> Result<bool, E>,
) -> Result<u32, E> {
    if asked <= own {
        return Ok(own);
    }
    if fits(asked)? {
        return Ok(asked);
    }
    let grain = ROW_GRAIN.min(own).max(1);
    // `lo` grains fit (or are no more than `own`), `hi` grains do not.
    let (mut lo, mut hi) = (own / grain, asked.div_ceil(grain));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if mid * grain <= own || fits(mid * grain)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo * grain).max(own))
}

/// The rows of the widest step a prompt read alone takes on `devices` placed as
/// `placement`: `asked`, or the largest multiple of 256 rows (`ROW_GRAIN`) below it whose
/// scratch the expert slots of every device can lend (`expert_store/scratch.rs`),
/// above `opts.part.own_rows()`; that if none is. `pages` is what the pools of K and
/// V hold, `page_bytes` a page on each device, as [`place`] gave them.
///
/// A device lends what the slots of its pools (each the layers whose experts take
/// slots of one size) hold above a row's picks of each of those layers (the store's
/// floor for a loan), a piece of each pool, in which the scratch is laid out without a
/// buffer cut by the end of a piece; a step of more rows also
/// keeps the rest of what it needs (its inputs, its exchange, its predictions,
/// about 50 KiB a row on Flash-Next against 0.5 to 0.7 MiB of scratch) for
/// good, which the slots lose, as do the parcels a split by layers hands the
/// residual over in (80 KiB a row on the receiving device); what is estimated
/// here leaves a tenth of the lendable room spare for what the plan of the
/// store takes besides (transit). `drafter` is what the drafters take of the bytes for
/// slots on the device with the head (the MTP block's experts stay resident there), which
/// the slots of that device do not have.
#[allow(clippy::too_many_arguments)]
pub fn batch_rows(
    gguf: &Gguf,
    devices: &[Device],
    opts: &ModelOptions,
    placement: &Placement,
    pages: u32,
    page_bytes: &[u64],
    asked: u32,
    drafter: u64,
) -> Result<u32, OpenError> {
    let config = Config::from_gguf(gguf)?;
    let weights = Weights::load(gguf, &config)?;
    let (layout, _) = Layout::new(gguf, &config, &weights, placement, None)?;
    let own = opts.part.own_rows();
    // A layer keeps a row's picks while a loan runs (`ExpertStore::scratch_floor`).
    let floor = config.experts.used as u64;
    // Per device: the bytes of a slot of each of its layers, and of the
    // layers of each of its pools (slots of one size), the largest first: a
    // loan takes a piece of each.
    let rounds: Vec<(u64, Vec<u64>)> = (0..devices.len())
        .map(|d| {
            let mut sizes: Vec<(u64, u64)> = Vec::new();
            for spec in layout.specs.iter().filter(|s| s.device == d && !s.resident) {
                match sizes.iter_mut().find(|(size, _)| *size == spec.slot_bytes) {
                    Some((_, sum)) => *sum += spec.slot_bytes,
                    None => sizes.push((spec.slot_bytes, spec.slot_bytes)),
                }
            }
            let all = sizes.iter().map(|(_, sum)| sum).sum();
            let mut pools: Vec<u64> = sizes.iter().map(|(_, sum)| *sum).collect();
            pools.sort_unstable_by(|a, b| b.cmp(a));
            (all, pools)
        })
        .collect();
    widest_rows(own, asked.max(own), |rows| {
        let wide = Options { max_rows: rows, own_rows: own, ..opts.part.clone() };
        let narrow = Options { max_rows: own, own_rows: 0, ..opts.part.clone() };
        let fits = devices.iter().enumerate().all(|(d, device)| {
            let PartPlan { role, first, shares, head } = placement.part(&config, d);
            let shape = PartShape { role, first, shares, head };
            let mp = device.info().multiprocessors;
            let (Ok(f), Ok(g)) =
                (Frame::new(&config, &wide, &shape, mp), Frame::new(&config, &narrow, &shape, mp))
            else {
                return false;
            };
            let plan = &placement.devices[d];
            // A split by layers hands the residual of a step's rows to the next
            // device in a ring of two parcels, which the receiver holds.
            let lanes = if placement.tensor.is_none() && d > 0 {
                2 * u64::from(rows - own) * config.wide() as u64 * 4
            } else {
                0
            };
            let extra = f.step_bytes(&wide).saturating_sub(g.step_bytes(&narrow)) + lanes;
            let pool = u64::from(pages) * page_bytes.get(d).copied().unwrap_or(0);
            let drafters = if plan.head { drafter } else { 0 };
            let room = plan.slots.saturating_sub(pool + extra + drafters);
            let (all, pools) = &rounds[d];
            let per_layer = (room / (*all).max(1)).saturating_sub(floor);
            let pieces: Vec<u64> = pools.iter().map(|sum| per_layer * sum / 10 * 9).collect();
            Frame::new_in(&config, &wide, &shape, mp, &pieces)
                .is_ok_and(|f| f.lent_pieces().is_some())
        });
        Ok(fits)
    })
}

/// Where the layers of the model in `gguf` would go on `devices` with `opts`,
/// and the bytes one page of the pools would take on each device. Nothing is
/// kept on the devices; the placement of two devices that share every layer
/// measures a read of a stream mixer on each and, unless `opts.spread` gives
/// the worker's role, times both roles with parts opened for the trial and
/// closed again (`roles.rs`): the placement's `TensorPlan::trial` says what
/// that found. A caller that places the model again pins the choice.
pub fn place(
    gguf: &Gguf,
    devices: &[Device],
    opts: &ModelOptions,
) -> Result<(Placement, Vec<u64>), OpenError> {
    let config = Config::from_gguf(gguf)?;
    let weights = Weights::load(gguf, &config)?;
    place_in(gguf, &config, &weights, devices, opts)
}

/// What a page takes on each device of `placement`: the converted entries of
/// K and V of its attention layers, and the indexer's keys. The first lies in
/// host memory for a page beyond VRAM ([`Options::near_pages`]), the second in
/// VRAM for every page.
pub fn page_parts(
    gguf: &Gguf,
    devices: &[Device],
    opts: &ModelOptions,
    placement: &Placement,
) -> Result<Vec<(u64, u64)>, OpenError> {
    let config = Config::from_gguf(gguf)?;
    let mut parts = Vec::with_capacity(devices.len());
    for (index, device) in devices.iter().enumerate() {
        let PartPlan { role, first, shares, head } = placement.part(&config, index);
        let shape = PartShape { role, first, shares, head };
        let frame = Frame::new(&config, &opts.part, &shape, device.info().multiprocessors)?;
        parts.push(frame.page_parts());
    }
    Ok(parts)
}

pub(super) fn place_in(
    gguf: &Gguf,
    config: &Config,
    weights: &Weights<'_>,
    devices: &[Device],
    opts: &ModelOptions,
) -> Result<(Placement, Vec<u64>), OpenError> {
    let Chosen { placement, .. } = choose(gguf, config, weights, devices, opts)?;
    let pages = u64::from(opts.part.pages());
    let mut page_bytes = Vec::with_capacity(devices.len());
    for (index, device) in devices.iter().enumerate() {
        let PartPlan { role, first, shares, head } = placement.part(config, index);
        let shape = PartShape { role, first, shares, head };
        let frame = Frame::new(config, &opts.part, &shape, device.info().multiprocessors)?;
        page_bytes.push(frame.pool_bytes().div_ceil(pages));
    }
    Ok((placement, page_bytes))
}

#[cfg(test)]
mod widest_tests {
    use super::*;

    #[test]
    fn the_widest_step_is_a_multiple_of_the_grain_that_fits() {
        let fits = |most: u32| move |rows: u32| Ok::<bool, ()>(rows <= most);
        assert_eq!(widest_rows(512, 8192, fits(9000)), Ok(8192));
        assert_eq!(widest_rows(512, 8192, fits(3700)), Ok(3584));
        assert_eq!(widest_rows(512, 8192, fits(767)), Ok(512));
        assert_eq!(widest_rows(512, 8192, fits(768)), Ok(768));
        assert_eq!(widest_rows(512, 512, fits(0)), Ok(512));
        assert_eq!(widest_rows(512, 1000, fits(999)), Ok(768));
        // A part that keeps scratch for fewer rows than the grain takes its own as the grain.
        assert_eq!(widest_rows(8, 32, fits(20)), Ok(16));
    }
}
