//! What the second of two devices that share every layer does, measured when
//! the model is placed (the owner's rule of 2026-11-22: no option).
//!
//! The device that holds the streams (the mixer) runs every stream mixer, the
//! routers, the attention layers and the head. The other one, the worker,
//! either computes a share of the recurrent heads and the shared expert
//! besides the experts it owns ([`TensorChoice::core_share`] below 1), or
//! holds experts only and the mixer computes every dense part. A share takes
//! work off the mixer in a decode step; it costs an exchange per recurrent
//! layer, and in a wide step the worker's time for its heads, which a slower
//! worker spends while the mixer waits. Which way is faster depends on the two
//! devices, so both are timed and the faster kept.
//!
//! A trial opens the parts of each choice in turn on the empty devices, as
//! the choice places them, with every expert in a slot of zeros: a decode
//! step and a wide step then run as they run when no expert is missed, the
//! dense parts with their real weights and the experts' kernels reading as
//! many bytes as real ones (the slots are several times the L2 cache, so the
//! reads come from memory). What a trial cannot see is the experts a step
//! misses, which the two choices hold about equally: experts only gives the
//! worker the bytes of its dense share for slots, and the mixer as many
//! fewer.
//!
//! The choice whose decode step and wide step take the least time together,
//! each counted by how much faster it is (the product of the two times), is
//! kept: a decode step is some fifty times a prompt row, so milliseconds alone
//! would let the decode step decide what the prompts pay for.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_core::Ledger;
use ares_cuda::{Device, DeviceBuffer, MAX_BATCH, Segment, StepRecord, Stream};
use ares_gguf::Gguf;

use super::Error;
use super::feed::Layout;
use super::open;
use super::part::{ExpertAt, Options, Part, StepEnv, ngram_shape};
use super::placement::{DeviceMemory, Placement, TensorChoice};
use crate::blocks::rows::{HostTable, TOKEN_EMBD};
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

/// Decode steps of a trial that are not timed (the first ones after the parts
/// open), and those that are: their median counts.
const DECODE_WARM: u32 = 3;
const DECODE_TIMED: u32 = 9;

/// Wide steps of a trial: one of a few rows that is not timed (the first launches of
/// the many-row kernels), then the timed ones, of which the fastest counts.
const WIDE_WARM_ROWS: u32 = 16;
const WIDE_TIMED: u32 = 1;

/// The most rows of the trial's wide step: a step of the parts' own scratch
/// up to this.
const WIDE_ROWS: u32 = 512;

/// The slots of zeros every expert of a layer is read from are at least this
/// many times the device's L2 cache, so that the experts' kernels read from
/// memory as they do in a step.
const SLOTS_OVER_L2: u64 = 4;

/// What one choice measured.
#[derive(Clone, Debug, PartialEq)]
pub struct RoleTime {
    pub choice: TensorChoice,
    /// The median of the timed decode steps of one row.
    pub decode: Duration,
    /// The fastest timed wide step; `None` when the model runs no wide step.
    pub wide: Option<Duration>,
    /// How long opening the choice's parts took, and the whole of its trial.
    pub opened: Duration,
    pub took: Duration,
}

impl RoleTime {
    /// What the choices are compared by: the product of the decode step and
    /// the wide step, in milliseconds squared, or the decode step alone.
    pub fn cost(&self) -> f64 {
        let decode = self.decode.as_secs_f64() * 1e3;
        decode * self.wide.map_or(1.0, |w| w.as_secs_f64() * 1e3)
    }
}

/// What a trial of the worker's role measured and kept.
#[derive(Clone, Debug, PartialEq)]
pub struct RoleTrial {
    /// The choices that were timed.
    pub times: Vec<RoleTime>,
    /// The choices that do not fit the devices, and why.
    pub refused: Vec<(TensorChoice, String)>,
    /// The rows of the wide step (0 for none).
    pub wide_rows: u32,
    /// The choice kept.
    pub kept: TensorChoice,
    /// How long the trial took, parts opened and closed.
    pub took: Duration,
}

/// Times every choice of `choices` on `devices` (two, the first choice's
/// mixer the same in all) and keeps the fastest. A choice that does not fit
/// with the model's options `part` is refused; when none fits, the first is
/// kept, and placing it says why it does not fit.
pub(super) fn trial(
    gguf: &Gguf,
    c: &Config,
    w: &Weights<'_>,
    devices: &[Device],
    part: &Options,
    memory: &[DeviceMemory],
    choices: &[TensorChoice],
) -> Result<RoleTrial, Error> {
    let started = Instant::now();
    let wide_rows = if part.own_rows() > MAX_BATCH { part.own_rows().min(WIDE_ROWS) } else { 0 };
    // A sequence of a short wide step and the decode steps, then the timed
    // wide steps after them.
    let cells = (WIDE_WARM_ROWS + wide_rows * WIDE_TIMED + DECODE_WARM + DECODE_TIMED + 1).max(256);
    let opts = Options {
        max_rows: wide_rows.max(1),
        own_rows: 0,
        logit_rows: 1,
        window_rows: 0,
        max_slots: 1,
        cells,
        pool_pages: 0,
        // The trial's few cells are all in VRAM: its parts open no far pool.
        near_pages: None,
        host_logits: false,
        profile: None,
        prune: None,
        ..part.clone()
    };
    let mapped = Gguf::open(gguf.path(0)).map_err(|e| Error::Invalid(e.to_string()))?;
    let table = Arc::new(HostTable::mapped(Arc::new(mapped), TOKEN_EMBD)?);
    let mut times = Vec::with_capacity(choices.len());
    let mut refused = Vec::new();
    for choice in choices {
        // The placement as the load will make it, with the model's options:
        // a choice that does not fit is no choice.
        let placement = match Placement::tensor(c, w, part, memory, choice) {
            Ok(placement) => placement,
            Err(e) => {
                refused.push((*choice, e.to_string()));
                continue;
            }
        };
        match time_choice(gguf, c, w, devices, &opts, memory, &placement, &table, wide_rows) {
            Ok(time) => times.push(RoleTime { choice: *choice, ..time }),
            Err(e) if is_out_of_memory(&e) => refused.push((*choice, e.to_string())),
            Err(e) => return Err(e),
        }
    }
    let kept = faster(&times).unwrap_or(choices[0]);
    Ok(RoleTrial { times, refused, wide_rows, kept, took: started.elapsed() })
}

/// The choice of `times` that costs the least ([`RoleTime::cost`]); the first
/// of equal ones.
fn faster(times: &[RoleTime]) -> Option<TensorChoice> {
    times.iter().min_by(|a, b| a.cost().total_cmp(&b.cost())).map(|t| t.choice)
}

fn is_out_of_memory(e: &Error) -> bool {
    e.is_no_room()
}

/// The median decode step and the fastest wide step of `placement`'s parts,
/// opened with `opts` and every expert in a slot of zeros.
#[allow(clippy::too_many_arguments)]
fn time_choice(
    gguf: &Gguf,
    c: &Config,
    w: &Weights<'_>,
    devices: &[Device],
    opts: &Options,
    memory: &[DeviceMemory],
    placement: &Placement,
    table: &Arc<HostTable>,
    wide_rows: u32,
) -> Result<RoleTime, Error> {
    let started = Instant::now();
    let tensor =
        placement.tensor.as_ref().ok_or_else(|| Error::invalid("a trial of no tensor split"))?;
    let mixer = tensor.mixer;
    let mut ledgers: Vec<Ledger> = devices
        .iter()
        .zip(memory)
        .map(|(d, m)| Ledger::new(format!("CUDA{} trial", d.ordinal()), m.budget))
        .collect();
    let mut parts = Vec::with_capacity(2);
    for (index, device) in devices.iter().enumerate() {
        let ledger = &mut ledgers[index];
        parts.push(open::open_part(c, w, opts, placement, index, device, ledger, table)?);
    }
    open::connect_all(&mut parts, placement, c)?;

    // The pools of K/V, and every expert of a layer in one of a few slots of
    // zeros: expert `e` in slot `e` modulo their number.
    let (layout, _) = Layout::new(gguf, c, w, placement, None)?;
    let n_experts = c.experts.count;
    let mut pools = Vec::with_capacity(2);
    let mut slots = Vec::with_capacity(2);
    let mut tables = Vec::with_capacity(2);
    let mut addresses: Vec<Vec<Vec<u64>>> = vec![Vec::new(); 2];
    for (index, device) in devices.iter().enumerate() {
        let part = &mut parts[index];
        pools.push(DeviceBuffer::new(device, part.pool_bytes().max(64))?);
        part.own_pages()?;
        let stride = (0..c.n_layer)
            .map(|layer| layout.specs[layout.store_layer(index, layer)].slot_bytes)
            .max()
            .unwrap_or(1)
            .max(1);
        let l2 = u64::from(device.info().l2_cache_bytes);
        let count = (SLOTS_OVER_L2 * l2)
            .div_ceil(stride)
            .max(2 * c.experts.used as u64)
            .min(n_experts as u64)
            .max(1);
        let pool = DeviceBuffer::new(device, count * stride)?;
        let stream = Stream::new(device, 0)?;
        // SAFETY: the whole buffer, of this device.
        unsafe { stream.fill(pool.ptr(), 0, (count * stride) as usize)? };
        stream.synchronize()?;
        let entries: Vec<u8> = (0..c.n_layer)
            .flat_map(|_| (0..n_experts).flat_map(|e| ((e as u64 % count) as i32).to_le_bytes()))
            .collect();
        let table = DeviceBuffer::new(device, entries.len() as u64)?;
        table.write(0, &entries)?;
        let row = 4 * n_experts as u64;
        part.set_experts(
            (0..c.n_layer)
                .map(|layer| ExpertAt {
                    table: table.ptr().offset(layer as u64 * row),
                    pool: pool.ptr(),
                    host: None,
                })
                .collect(),
        )?;
        addresses[index] = (0..c.n_layer)
            .map(|layer| {
                (0..n_experts as u32)
                    .map(|e| {
                        if placement.owner(layer, e) == index {
                            part.slot_address(layer, e % count as u32)
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();
        slots.push(pool);
        tables.push(table);
    }
    let streams: Vec<Stream> =
        devices.iter().map(|d| Stream::new(d, 0)).collect::<Result<_, _>>()?;
    let mut run = Runner {
        c,
        parts: &mut parts,
        streams: &streams,
        pools: &pools,
        addresses: &addresses,
        mixer,
        position: 0,
        step: 0,
        ngram: ngram_shape(c).map_or(0, |g| g.fetched_row_bytes()),
    };
    let opened = started.elapsed();
    let mut decode = Vec::with_capacity(DECODE_TIMED as usize);
    let mut wide = None;
    if wide_rows > 0 {
        run.step(WIDE_WARM_ROWS.min(wide_rows))?;
    }
    for _ in 0..DECODE_WARM {
        run.step(1)?;
    }
    for _ in 0..DECODE_TIMED {
        decode.push(run.step(1)?);
    }
    if wide_rows > 0 {
        for _ in 0..WIDE_TIMED {
            let took = run.step(wide_rows)?;
            wide = Some(wide.map_or(took, |w: Duration| w.min(took)));
        }
    }
    decode.sort();
    Ok(RoleTime {
        choice: tensor.choice,
        decode: decode[decode.len() / 2],
        wide,
        opened,
        took: started.elapsed(),
    })
}

/// The steps of a trial on the parts of one choice.
struct Runner<'a> {
    c: &'a Config,
    parts: &'a mut [Part],
    streams: &'a [Stream],
    pools: &'a [DeviceBuffer],
    /// Per device and layer, where each expert the device owns is (0 for the
    /// others): the tables of a wide step's round.
    addresses: &'a [Vec<Vec<u64>>],
    mixer: usize,
    /// The next cell of the one sequence, and the steps so far.
    position: u32,
    step: u32,
    /// Bytes of n-gram table rows a row needs (0 without the block).
    ngram: u64,
}

impl Runner<'_> {
    /// One step of `rows` rows of the sequence, timed from its first launch to
    /// the end of both devices.
    fn step(&mut self, rows: u32) -> Result<Duration, Error> {
        let c = self.c;
        self.step += 1;
        let (mixer, worker) = (self.mixer, 1 - self.mixer);
        // Tokens spread over the vocabulary; what they are changes no time.
        let tokens: Vec<u32> =
            (0..rows).map(|r| ((self.position + r) * 7919 + 13) % c.n_vocab as u32).collect();
        let segments =
            [Segment { row0: 0, n_rows: rows, slot: 0, pos0: self.position as i32, direct: true }];
        let record = StepRecord::new(&segments, &tokens, None, &[rows - 1])?;
        self.position += rows;
        // The rows of the n-gram table: zeros, whose decoded values are zero.
        let rows_of_table = vec![0u8; (u64::from(rows) * self.ngram) as usize];
        let per_step = 8 * (c.n_layer as u32 + 1);
        let xbase = self.step.wrapping_mul(per_step).wrapping_add(1);
        let ngram_layer = self.parts[mixer].ngram_layer();
        let env = |device: usize, layer: usize| StepEnv {
            stream: &self.streams[device],
            record: &record,
            segments: &segments,
            token: self.step,
            tokens: &tokens,
            ngram: ngram_layer.filter(|at| layer >= *at).map(|_| rows_of_table.as_slice()),
            inbound: None,
            xbase,
            pools: self.pools[device].ptr(),
            replay: &[],
        };
        let started = Instant::now();
        for device in [mixer, worker] {
            self.parts[device].begin(&env(device, 0))?;
        }
        for i in 0..c.n_layer {
            for device in [mixer, worker] {
                self.parts[device].enqueue_layer(&env(device, i), i)?;
            }
            if rows > MAX_BATCH {
                // The mixer computes its experts at once; the worker once the
                // picks have reached the host, as in a step of the session.
                self.parts[mixer].experts_round(
                    &env(mixer, i),
                    i,
                    &self.addresses[mixer][i],
                    None,
                )?;
                let board = self.parts[mixer].board();
                let waiting = Instant::now();
                while !board.arrived(i)? {
                    if waiting.elapsed() > Duration::from_secs(30) {
                        return Err(Error::invalid(format!(
                            "the picks of layer {i} did not arrive"
                        )));
                    }
                    std::hint::spin_loop();
                }
                let picks = board.record(i).to_vec();
                self.parts[worker].experts_round(
                    &env(worker, i),
                    i,
                    &self.addresses[worker][i],
                    Some(&picks),
                )?;
            }
        }
        let last = c.n_layer - 1;
        self.parts[worker].end(&env(worker, last))?;
        self.parts[mixer].end(&env(mixer, last))?;
        for stream in self.streams {
            stream.synchronize()?;
        }
        Ok(started.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(core_share: f64, decode_ms: f64, wide_ms: Option<f64>) -> RoleTime {
        RoleTime {
            choice: TensorChoice { mixer: 0, core_share, shared_on_mixer: core_share >= 1.0 },
            decode: Duration::from_secs_f64(decode_ms / 1e3),
            wide: wide_ms.map(|ms| Duration::from_secs_f64(ms / 1e3)),
            opened: Duration::ZERO,
            took: Duration::ZERO,
        }
    }

    /// Each kind of step counts by how much faster it is: a decode step 3 %
    /// slower is worth a wide step 27 % faster, and not one 2 % faster; a
    /// model without wide steps is decided by the decode step.
    #[test]
    fn the_choice_faster_in_both_steps_together_is_kept() {
        let share = |decode, wide| time(0.75, decode, wide);
        let alone = |decode, wide| time(1.0, decode, wide);
        let kept = |a: RoleTime, b: RoleTime| faster(&[a, b]).unwrap().core_share;
        assert_eq!(kept(share(17.4, Some(1020.0)), alone(18.0, Some(800.0))), 1.0);
        assert_eq!(kept(share(17.4, Some(1020.0)), alone(18.0, Some(1000.0))), 0.75);
        assert_eq!(kept(share(17.4, None), alone(18.0, None)), 0.75);
        assert_eq!(kept(share(18.4, None), alone(18.0, None)), 1.0);
        assert_eq!(faster(&[]), None);
    }
}
