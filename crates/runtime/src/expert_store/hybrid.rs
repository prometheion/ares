//! Serving a miss on the CPU or over the link, and promoting experts into
//! VRAM without a step waiting for it.
//!
//! An expert that a step needs and that is in no slot can be brought to the
//! device (the kernels wait for its slot) or computed on the CPU from its
//! bytes in the RAM tier (the kernels wait for the host's sums). Which is
//! sooner depends on what else the layer misses, on how many rows use the
//! expert, on the cores that are free and on what the link is busy with, so
//! it is decided per layer and step from two measured numbers
//! ([`CostModel`]): the time of a CPU job by its size, and the time of one
//! upload on each device. The CPU's part of a layer is one job for all its
//! experts; the uploads follow each other on the link; the layer goes on
//! when the later of the two is done, so the split is the one whose later
//! half ends first.
//!
//! Serving a miss and deciding what VRAM holds are separate. An expert the
//! CPU served stays where it was. What moves into VRAM is decided between
//! steps, from the heat ([`ExpertStore::promote_hot`]): the bytes are
//! uploaded into the victim's slot while the next steps run, and the table
//! entry is written only when the bytes are there and no step is at the
//! layer. Until then the expert is simply not on the device, and a step that
//! needs it gets it from the CPU: a promotion never makes a step wait.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::{ExpertShape, RouteGroup};
use ares_weights::{Start, Swap, Want};

use super::cpu::{CpuExpert, CpuPool, LayerJob, MAX_ROWS, act_bytes};
use super::{DeviceOp, ExpertStore, Phase, SlotDevice, Source, StoreError, Wanted};

/// How the store serves an expert a step needs that is in no slot.
///
/// The engine runs [`Serve::Measured`]. The other two exist to measure it
/// against: each is what the measured choice degenerates to when one side
/// is never the faster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Serve {
    /// Every miss is uploaded and the device waits for it.
    Stream,
    /// Every miss is computed on the CPU, as far as the layer's RAM slots
    /// can keep the experts of one step; what is beyond that is uploaded.
    Cpu,
    /// Per layer and step, whichever the cost model says is sooner.
    Measured,
}

/// A running mean that follows what it measures.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    value: f64,
    samples: u32,
}

impl Mean {
    /// The first samples count fully, later ones a tenth.
    fn add(&mut self, sample: f64) {
        self.samples = self.samples.saturating_add(1);
        let weight = (1.0 / f64::from(self.samples)).max(0.1);
        self.value += weight * (sample - self.value);
    }

    fn get(&self) -> Option<f64> {
        (self.samples > 0).then_some(self.value)
    }
}

/// Jobs are timed by size class: 1, 2, 3 to 4, 5 to 8, 9 to 16, more pairs.
/// A job has a fixed part (handing it to the workers, the last task's tail)
/// that a class per size holds without a model of it.
const SIZES: usize = 6;

fn size_class(pairs: usize) -> usize {
    (pairs.max(1).next_power_of_two().trailing_zeros() as usize).min(SIZES - 1)
}

/// The types of a layer's experts, of gate and up and of down: what a CPU job of the layer
/// is timed by. Each pair of types a model has is timed by itself, because the time of a job
/// depends on both and differs widely between them: the types with a kernel written for them
/// (IQ2_S, IQ2_XXS and IQ1_M for gate and up, Q2_0 for down) multiply the file's bytes, every
/// other type is decoded row by row first (`cpu/kernels.rs`).
pub type ExpertTypes = (GgmlType, GgmlType);

/// One expert a layer misses, as the split sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Miss {
    /// Rows of the step that use the expert.
    pub pairs: u32,
    /// The expert's heat: of two that could be uploaded, the hotter one is
    /// the better guest of a slot.
    pub heat: f32,
    /// The expert cannot be uploaded now: its bytes are already on their way
    /// into a slot whose entry will not be written during this step.
    pub cpu_only: bool,
}

/// What the split between CPU and link is decided from: measured times only.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Seconds per (expert, row) pair of a CPU job, by the types of the
    /// layer's experts and by size class, from the creation of the job to its
    /// last task. An entry per pair of types that was timed.
    cpu: Vec<(ExpertTypes, [Mean; SIZES])>,
    /// Seconds one expert takes on a device's upload stream.
    upload: Vec<Mean>,
}

impl CostModel {
    pub(super) fn new(devices: usize) -> Self {
        Self { cpu: Vec::new(), upload: vec![Mean::default(); devices] }
    }

    /// A CPU job of `pairs` pairs of a layer whose experts have the types
    /// `types` took `time`.
    pub fn record_job(&mut self, types: ExpertTypes, pairs: usize, time: Duration) {
        if pairs == 0 {
            return;
        }
        let at = self.cpu.iter().position(|(t, _)| *t == types).unwrap_or_else(|| {
            self.cpu.push((types, [Mean::default(); SIZES]));
            self.cpu.len() - 1
        });
        self.cpu[at].1[size_class(pairs)].add(time.as_secs_f64() / pairs as f64);
    }

    /// One expert took `time` on the upload stream of `device`.
    pub fn record_upload(&mut self, device: usize, time: Duration) {
        self.upload[device].add(time.as_secs_f64());
    }

    /// The expected time of a CPU job of `pairs` pairs; `None` before any
    /// job of a layer of those types was timed. A size that was never seen
    /// takes the time per pair of the nearest size that was.
    pub fn job_time(&self, types: ExpertTypes, pairs: usize) -> Option<f64> {
        if pairs == 0 {
            return Some(0.0);
        }
        let (_, classes) = self.cpu.iter().find(|(t, _)| *t == types)?;
        let class = size_class(pairs);
        let per_pair = (0..SIZES)
            .filter(|c| classes[*c].samples > 0)
            .min_by_key(|c| c.abs_diff(class))
            .and_then(|c| classes[c].get())?;
        Some(per_pair * pairs as f64)
    }

    /// The expected time of one upload on `device`; `None` before any was timed.
    pub fn upload_time(&self, device: usize) -> Option<f64> {
        self.upload[device].get()
    }

    /// Divides the experts a layer misses between the CPU (`true`) and the
    /// link (`false`): the split for which the later of the CPU's job and
    /// the last upload ends soonest. `queued` uploads are ahead on the link.
    ///
    /// Uploads go to the experts with the most rows first (an upload costs
    /// the same however many rows use the expert, a CPU job costs per row),
    /// then to the hottest. With equal ends the CPU takes more: its work
    /// leaves the tiers and the link as they are. A side that was never
    /// timed is tried, one expert at a time, so that it gets timed.
    pub fn split(
        &self,
        types: ExpertTypes,
        device: usize,
        misses: &[Miss],
        queued: usize,
    ) -> Vec<bool> {
        let mut on_cpu = vec![true; misses.len()];
        let mut order: Vec<usize> = (0..misses.len()).filter(|i| !misses[*i].cpu_only).collect();
        order.sort_by(|a, b| {
            let (a, b) = (&misses[*a], &misses[*b]);
            b.pairs.cmp(&a.pairs).then(b.heat.total_cmp(&a.heat))
        });
        let pairs = |on_cpu: &[bool]| -> usize {
            misses.iter().zip(on_cpu).filter(|(_, c)| **c).map(|(m, _)| m.pairs as usize).sum()
        };
        let Some(upload) = self.upload_time(device) else {
            // Never timed: one upload tells.
            if let Some(first) = order.first() {
                on_cpu[*first] = false;
            }
            return on_cpu;
        };
        if self.job_time(types, 1).is_none() {
            // The CPU was never timed for these types: it gets one expert, the
            // one that costs it least, and the link the rest.
            let cheapest = order.last().copied();
            for i in &order {
                on_cpu[*i] = Some(*i) == cheapest;
            }
            return on_cpu;
        }
        let end = |on_cpu: &[bool], uploads: usize| -> f64 {
            let cpu = self.job_time(types, pairs(on_cpu)).unwrap_or(0.0);
            let link = if uploads == 0 { 0.0 } else { (queued + uploads) as f64 * upload };
            cpu.max(link)
        };
        let mut best = (end(&on_cpu, 0), 0);
        let mut trial = on_cpu.clone();
        for (n, i) in order.iter().enumerate() {
            trial[*i] = false;
            let t = end(&trial, n + 1);
            if t < best.0 {
                best = (t, n + 1);
            }
        }
        for i in &order[..best.1] {
            on_cpu[*i] = false;
        }
        on_cpu
    }
}

/// When a hot expert takes a VRAM slot from a cold one, and how much of that
/// may be on the link at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromoteRule {
    /// A promotion needs `heat_in > margin * heat_out + bonus`, so that two
    /// experts of similar heat do not trade places. Without a bonus an
    /// expert that was just used takes the slot of one that was not for
    /// some hundred steps: on recorded routing a third of a token's picks
    /// are picks of the token before it, and with a bonus of 1 (an expert
    /// must be used twice) VRAM served 93.9 % of the picks where it serves
    /// 95.7 % when every miss moves in.
    pub margin: f32,
    pub bonus: f32,
    /// The most promotions on their way per device: what bounds the link
    /// time promotions take from the uploads a step waits for.
    pub in_flight: usize,
}

impl Default for PromoteRule {
    fn default() -> Self {
        Self { margin: 1.25, bonus: 0.0, in_flight: 8 }
    }
}

/// An expert the CPU will compute for an open layer once its bytes are in RAM.
#[derive(Clone, Copy, Debug)]
struct Late {
    expert: u32,
    rows: u32,
    weight: [f32; MAX_ROWS],
}

/// The CPU's part of a layer of the running step.
#[derive(Debug)]
struct Claims {
    layer: usize,
    token: u32,
    n_rows: usize,
    act: Vec<u8>,
    /// Claimed experts whose bytes are still being read.
    late: Vec<Late>,
    jobs: Vec<Arc<LayerJob>>,
    /// Experts held in RAM for the jobs.
    held: Vec<u32>,
    since: Instant,
}

/// The CPU half of a store.
#[derive(Debug)]
pub(super) struct Hybrid {
    pool: CpuPool,
    shapes: Vec<ExpertShape>,
    serve: Serve,
    rule: PromoteRule,
    pub(super) model: CostModel,
    /// The layers of the running step that wait for the host's sums.
    open: Vec<Claims>,
    /// Promotions whose bytes are on their way or there, and whose table
    /// entry is not written yet.
    promoting: HashSet<(usize, u32)>,
    /// Promotions whose bytes have arrived: (layer, swap).
    landed: Vec<(usize, Swap)>,
    /// The experts the running step has claimed, by layer.
    claimed: HashSet<(usize, u32)>,
    /// Per device: promotions on the upload stream.
    in_flight: Vec<usize>,
    /// Per device: when its upload stream last finished an expert, and how
    /// many it has not finished.
    last_upload: Vec<Instant>,
    pub(super) queued_uploads: Vec<usize>,
    /// Sums of the open layer being delivered, kept to reuse the memory.
    sums: Vec<f32>,
}

impl<D: SlotDevice> ExpertStore<D> {
    /// Gives the store its CPU half: `pool` computes, `shapes[layer]` is the
    /// shape of each layer's experts, and misses are served as `serve` says.
    /// The devices must have host shares for the layers they run
    /// ([`super::CudaSlots::add_host_shares`]).
    ///
    /// The cost model starts from a measurement, not from a guess: a few
    /// jobs of each gate type and size are run here, on bytes of the RAM
    /// tier's own memory, before any step.
    pub fn enable_cpu(
        &mut self,
        pool: CpuPool,
        shapes: Vec<ExpertShape>,
        serve: Serve,
    ) -> Result<(), StoreError> {
        if shapes.len() != self.layers.len() {
            return Err(StoreError::Plan(format!(
                "{} expert shapes for {} layers",
                shapes.len(),
                self.layers.len()
            )));
        }
        let mut model = CostModel::new(self.devices.len());
        calibrate(&pool, &shapes, &mut model)?;
        self.hybrid = Some(Hybrid {
            pool,
            shapes,
            serve,
            rule: PromoteRule::default(),
            model,
            open: Vec::new(),
            promoting: HashSet::new(),
            landed: Vec::new(),
            claimed: HashSet::new(),
            in_flight: vec![0; self.devices.len()],
            last_upload: vec![Instant::now(); self.devices.len()],
            queued_uploads: vec![0; self.devices.len()],
            sums: Vec::new(),
        });
        Ok(())
    }

    /// How misses are served: [`Serve::Stream`] without a CPU half.
    pub fn serve(&self) -> Serve {
        self.hybrid.as_ref().map_or(Serve::Stream, |h| h.serve)
    }

    /// Changes how misses are served, between steps: for measurements that
    /// compare the ways on one store.
    pub fn set_serve(&mut self, serve: Serve) {
        debug_assert!(!self.in_step, "between steps only");
        if let Some(hybrid) = self.hybrid.as_mut() {
            hybrid.serve = serve;
        }
    }

    /// The measured costs the split is decided from.
    pub fn cost_model(&self) -> Option<&CostModel> {
        self.hybrid.as_ref().map(|h| &h.model)
    }

    /// The route record of `layer` of the running step has arrived, with
    /// the rows' activations: `groups` are the record's groups in its order,
    /// `token` its token, and `act` the first
    /// [`act_bytes`] bytes of what
    /// `ares_quantize_act` wrote for the layer's `n_rows` input rows.
    ///
    /// Returns how many of the experts are in no slot. Each of those is now
    /// on its way to the device, where the kernels wait for its slot, or
    /// claimed for the CPU, where they wait for the sums that
    /// [`ExpertStore::poll`] delivers.
    pub fn routed_rows(
        &mut self,
        layer: usize,
        token: u32,
        n_rows: usize,
        groups: &[RouteGroup],
        act: &[u8],
    ) -> Result<usize, StoreError> {
        let picks: Vec<u32> = groups.iter().map(|g| g.expert).collect();
        if self.serve() == Serve::Stream {
            return self.routed(layer, &picks);
        }
        let Some(hybrid) = self.hybrid.as_ref() else { return self.routed(layer, &picks) };
        let shape = hybrid.shapes[layer];
        if n_rows == 0 || n_rows > MAX_ROWS || act.len() < act_bytes(shape.n_embd as usize, n_rows)
        {
            return Err(StoreError::Plan(format!(
                "layer {layer}: {n_rows} rows with {} bytes of activations",
                act.len()
            )));
        }

        for phase in &mut self.phase[..layer] {
            *phase = Phase::Passed;
        }
        self.phase[layer] = Phase::Open;
        self.heat.record(layer, &picks);
        self.guesses.forget_layer(layer);
        self.wants.retain(|w| w.layer != layer || w.demand);
        self.worth_stale[layer] = true;
        self.vram.sets[layer].routed(&picks, self.step);
        self.ram.sets[layer].routed(&picks, self.step);

        // The misses, and where each one's bytes are.
        let now = Instant::now();
        let mut missing = 0;
        let mut candidates: Vec<(usize, Miss, Source)> = Vec::new();
        // The misses left without a read or a job: claimed, they add nothing.
        let mut left: Vec<usize> = Vec::new();
        // Whether the layer waits for the file anyway: a read that a row which
        // must run needs. A read for draft rows next to it costs nothing more.
        let mut layer_waits = groups.iter().any(|group| {
            let (vram, ram) = (&self.vram.sets[layer], &self.ram.sets[layer]);
            !vram.contains(group.expert)
                && !vram.is_loading(group.expert)
                && !ram.contains(group.expert)
                && !self.windows.may_leave(group.rows)
        });
        for (g, group) in groups.iter().enumerate() {
            let expert = group.expert;
            self.stats.picks += 1;
            if self.vram.sets[layer].contains(expert) {
                self.stats.hits += 1;
                continue;
            }
            missing += 1;
            let promoting = hybrid.promoting.contains(&(layer, expert));
            if self.vram.sets[layer].is_loading(expert) && !promoting {
                // An upload with its entry behind it: the kernels get a slot.
                self.stats.miss_on_its_way += 1;
                if !self.wants.iter().any(|w| w.layer == layer && w.expert == expert) {
                    self.wants.push(Wanted {
                        layer,
                        expert,
                        demand: true,
                        since: now,
                        source: Source::OnItsWay,
                        cpu: false,
                    });
                }
                continue;
            }
            let source = if self.ram.sets[layer].contains(expert) {
                self.stats.miss_from_ram += 1;
                Source::Ram
            } else if self.ram.sets[layer].is_loading(expert) {
                self.stats.miss_on_its_way += 1;
                Source::OnItsWay
            } else {
                self.stats.miss_from_disk += 1;
                Source::Disk
            };
            // A row that verifies a draft waits for the file only while it is
            // worth the wait (`windows.rs`), and a row left already needs
            // nothing.
            let from_file = source == Source::Disk || source == Source::OnItsWay;
            let mut leave = self.windows.all_left(group.rows);
            if !leave && from_file && self.windows.may_leave(group.rows) {
                let cost = if layer_waits { 0.0 } else { self.read_ms as f32 };
                if self.windows.stake(group.rows) > cost {
                    self.windows.charge(group.rows, cost);
                    layer_waits = true;
                } else {
                    leave = true;
                }
            }
            if leave {
                if from_file {
                    self.stats.left_reads += 1;
                    // The token of a left row is likely the next step's
                    // first, which will route here again.
                    if self.config.read_left {
                        self.guesses.guess(&mut self.ram.sets, layer, &[expert], self.step + 1);
                    }
                } else {
                    self.stats.left_experts += 1;
                }
                self.windows.leave(group.rows);
                left.push(g);
                continue;
            }
            let miss = Miss {
                pairs: group.rows.count_ones(),
                heat: self.heat.layer(layer)[expert as usize],
                cpu_only: promoting,
            };
            candidates.push((g, miss, source));
        }
        if candidates.is_empty() && left.is_empty() {
            if missing > 0 {
                self.poll()?;
            }
            return Ok(missing);
        }

        let device = self.layers[layer].device;
        let misses: Vec<Miss> = candidates.iter().map(|c| c.1).collect();
        let on_cpu = match hybrid.serve {
            Serve::Measured => {
                let types = (shape.gate_ty, shape.down_ty);
                hybrid.model.split(types, device, &misses, hybrid.queued_uploads[device])
            }
            _ => vec![true; misses.len()],
        };

        // What the CPU claims stays in its RAM slot until the layer's sums
        // are delivered, and the sums wait for every claimed expert. So the
        // CPU may claim only as many experts that are yet to be read as the
        // layer has RAM slots beside what is kept there anyway: the claimed
        // experts that are in RAM and the promotions on their way. An expert
        // beyond that is uploaded, like a miss the cost model gives to the
        // link; claimed, it would wait for a slot that only its own
        // delivery frees, and the layer would never end.
        let in_ram = |c: &(usize, Miss, Source)| c.2 == Source::Ram;
        let kept = candidates.iter().zip(&on_cpu).filter(|(c, cpu)| **cpu && in_ram(c)).count()
            + hybrid.promoting.iter().filter(|(l, _)| *l == layer).count();
        let mut room = self.ram.sets[layer].n_slots().saturating_sub(kept);

        let mut claimed: Vec<u32> = Vec::new();
        let mut claimed_experts: Vec<u32> = Vec::new();
        let mut ready: Vec<CpuExpert> = Vec::new();
        let mut held: Vec<u32> = Vec::new();
        let mut late: Vec<Late> = Vec::new();
        for ((g, _, source), cpu) in candidates.iter().zip(&on_cpu) {
            let group = &groups[*g];
            self.wants.retain(|w| w.layer != layer || w.expert != group.expert);
            let cpu = *cpu && (*source == Source::Ram || room > 0);
            if cpu && *source != Source::Ram {
                room -= 1;
            }
            if !cpu {
                self.wants.push(Wanted {
                    layer,
                    expert: group.expert,
                    demand: true,
                    since: now,
                    source: *source,
                    cpu: false,
                });
                continue;
            }
            claimed.push(*g as u32);
            claimed_experts.push(group.expert);
            self.stats.cpu_experts += 1;
            self.stats.cpu_pairs += u64::from(group.rows.count_ones());
            if *source == Source::Ram {
                ready.push(self.cpu_expert(layer, group.expert, group.rows, group.weight)?);
                self.ram.sets[layer].hold(group.expert);
                held.push(group.expert);
            } else {
                late.push(Late { expert: group.expert, rows: group.rows, weight: group.weight });
                self.wants.push(Wanted {
                    layer,
                    expert: group.expert,
                    demand: true,
                    since: now,
                    source: *source,
                    cpu: true,
                });
            }
        }
        for g in left {
            claimed.push(g as u32);
            claimed_experts.push(groups[g].expert);
        }
        self.waits.push((layer, now));
        if !claimed.is_empty() {
            self.devices[device].claim(layer, token, &claimed)?;
            let mut claims = Claims {
                layer,
                token,
                n_rows,
                act: act[..act_bytes(shape.n_embd as usize, n_rows)].to_vec(),
                late,
                jobs: Vec::new(),
                held,
                since: now,
            };
            if let Some(hybrid) = self.hybrid.as_mut() {
                hybrid.claimed.extend(claimed_experts.into_iter().map(|e| (layer, e)));
                if !ready.is_empty() {
                    claims.jobs.push(start_job(hybrid, &claims, &ready)?);
                }
                hybrid.open.push(claims);
            }
        }
        self.poll()?;
        Ok(missing)
    }

    /// Where the matrices of `expert` of `layer` are in the RAM tier.
    fn cpu_expert(
        &self,
        layer: usize,
        expert: u32,
        rows: u32,
        weight: [f32; MAX_ROWS],
    ) -> Result<CpuExpert, StoreError> {
        let Some(slot) = self.ram.sets[layer].slot(expert) else {
            return Err(StoreError::Plan(format!(
                "expert {expert} of layer {layer} is not in RAM"
            )));
        };
        let skips = self.ram_skip[self.ram.pool_of[layer]][self.ram.slot(layer, slot) as usize];
        // SAFETY: each matrix starts `skip` bytes into its part of the
        // slot, inside the region.
        let part = |part: usize| unsafe {
            self.ram_part(layer, slot, part).add(skips[part] as usize).cast_const()
        };
        Ok(CpuExpert { gate: part(0), up: part(1), down: part(2), rows, weight })
    }

    /// A claimed expert whose read has landed: its job starts. Returns
    /// whether the expert was one the CPU waits for.
    pub(super) fn late_job(&mut self, layer: usize, expert: u32) -> Result<bool, StoreError> {
        let Some(at) = self.hybrid.as_ref().and_then(|h| {
            h.open
                .iter()
                .position(|c| c.layer == layer && c.late.iter().any(|l| l.expert == expert))
        }) else {
            return Ok(false);
        };
        let Some(late) = self.hybrid.as_mut().and_then(|h| {
            let claims = &mut h.open[at];
            let which = claims.late.iter().position(|l| l.expert == expert)?;
            Some(claims.late.swap_remove(which))
        }) else {
            return Ok(false);
        };
        let ready = self.cpu_expert(layer, expert, late.rows, late.weight)?;
        self.ram.sets[layer].hold(expert);
        if let Some(hybrid) = self.hybrid.as_mut() {
            let job = start_job(hybrid, &hybrid.open[at], &[ready])?;
            let claims = &mut hybrid.open[at];
            claims.held.push(expert);
            claims.jobs.push(job);
        }
        Ok(true)
    }

    /// Delivers the sums of every open layer whose jobs have all run, and
    /// writes the entries of promotions that have arrived.
    pub(super) fn poll_cpu(&mut self) -> Result<(), StoreError> {
        let Some(hybrid) = self.hybrid.as_mut() else { return Ok(()) };
        let mut at = 0;
        while at < hybrid.open.len() {
            let claims = &hybrid.open[at];
            if !claims.late.is_empty() || !claims.jobs.iter().all(|job| job.is_done()) {
                at += 1;
                continue;
            }
            let claims = hybrid.open.swap_remove(at);
            let shape = hybrid.shapes[claims.layer];
            let mut sums = std::mem::take(&mut hybrid.sums);
            sums.clear();
            sums.resize(claims.n_rows * shape.n_embd as usize, 0.0);
            for job in &claims.jobs {
                if let Some(part) = job.sums() {
                    sums.iter_mut().zip(part).for_each(|(s, p)| *s += p);
                }
                if let Some(time) = job.elapsed() {
                    hybrid.model.record_job((shape.gate_ty, shape.down_ty), job.pairs(), time);
                }
                self.stats.cpu_jobs += 1;
                self.stats.cpu_busy_ms += job.busy().as_secs_f64() * 1e3;
            }
            let device = self.layers[claims.layer].device;
            self.devices[device].deliver(claims.layer, claims.token, &sums)?;
            hybrid.sums = sums;
            for expert in &claims.held {
                self.ram.sets[claims.layer].release(*expert);
            }
            self.stats.cpu_ms.push(claims.since.elapsed().as_secs_f32() * 1e3);
        }

        // Promotions whose bytes are on the device: the entry is written
        // at once, unless the running step has claimed the expert and is
        // still at the layer. A claimed expert must not appear in a slot
        // before the layer's kernels have all run, also after its sums were
        // delivered: the down kernel looks at the table once more.
        let mut at = 0;
        while at < hybrid.landed.len() {
            let (layer, swap) = hybrid.landed[at];
            if self.in_step
                && self.phase[layer] == Phase::Open
                && hybrid.claimed.contains(&(layer, swap.load))
            {
                at += 1;
                continue;
            }
            hybrid.landed.swap_remove(at);
            hybrid.promoting.remove(&(layer, swap.load));
            let g = self.layers[layer];
            self.vram.sets[layer].commit(&swap);
            self.devices[g.device].set_entry(
                layer,
                swap.load,
                self.vram.slot(layer, swap.slot) as i32,
            )?;
            self.ram.sets[layer].release(swap.load);
            self.ram.sets[layer].unprotect(swap.load);
            self.worth_stale[layer] = true;
        }
        Ok(())
    }

    /// An upload has left the upload stream of `device`: its time goes into
    /// the cost model. `promotion` says what to do with a promotion's swap.
    pub(super) fn upload_done(
        &mut self,
        device: usize,
        enqueued: Instant,
        promotion: Option<(usize, Swap)>,
    ) {
        let Some(hybrid) = self.hybrid.as_mut() else { return };
        let now = Instant::now();
        // The stream was busy with this expert since the later of its
        // enqueue and the end of the one before it.
        let start = enqueued.max(hybrid.last_upload[device]);
        hybrid.model.record_upload(device, now.duration_since(start));
        hybrid.last_upload[device] = now;
        hybrid.queued_uploads[device] = hybrid.queued_uploads[device].saturating_sub(1);
        if let Some(landed) = promotion {
            hybrid.in_flight[device] = hybrid.in_flight[device].saturating_sub(1);
            hybrid.landed.push(landed);
        }
    }

    /// An upload was enqueued on `device`.
    pub(super) fn upload_queued(&mut self, device: usize) {
        if let Some(hybrid) = self.hybrid.as_mut() {
            hybrid.queued_uploads[device] += 1;
        }
    }

    /// Between steps: lets the hottest experts that RAM holds take VRAM
    /// slots from members that are colder by the rule's margin. The bytes go
    /// up while the next steps run; the entry follows when they are there
    /// ([`ExpertStore::poll`]). Until then a step that needs the expert gets
    /// it from the CPU.
    pub(super) fn promote_hot(&mut self) -> Result<(), StoreError> {
        let Some(hybrid) = self.hybrid.as_ref() else { return Ok(()) };
        let rule = hybrid.rule;
        // The best promotion of every layer, the largest gain first.
        let mut wanted: Vec<(f32, usize, u32)> = Vec::new();
        for layer in 0..self.layers.len() {
            let device = self.layers[layer].device;
            if hybrid.in_flight[device] >= rule.in_flight {
                continue;
            }
            let (vram, ram) = (&self.vram.sets[layer], &self.ram.sets[layer]);
            let heat = self.heat.layer(layer);
            let mut members: Vec<f32> = Vec::new();
            let mut outside: Vec<(f32, u32)> = Vec::new();
            for (expert, h) in heat.iter().enumerate() {
                let expert = expert as u32;
                if vram.contains(expert) {
                    members.push(*h);
                } else if *h > 0.0 && !vram.is_loading(expert) && ram.contains(expert) {
                    outside.push((*h, expert));
                }
            }
            if outside.is_empty() {
                continue;
            }
            outside.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            members.sort_by(f32::total_cmp);
            // Free slots go to the hottest outsiders as they are; behind
            // them the next hottest take the slots of the coldest members,
            // each pair as long as the rule holds for it.
            let free = vram.n_free();
            for (rank, (h, expert)) in outside.into_iter().enumerate() {
                if rank < free {
                    wanted.push((h, layer, expert));
                    continue;
                }
                match members.get(rank - free) {
                    Some(cold) if h > rule.margin * cold + rule.bonus => {
                        wanted.push((h - cold, layer, expert));
                    }
                    _ => break,
                }
            }
        }
        wanted.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, layer, expert) in wanted {
            let device = self.layers[layer].device;
            let in_flight = self.hybrid.as_ref().map_or(usize::MAX, |h| h.in_flight[device]);
            if in_flight >= rule.in_flight {
                continue;
            }
            if self.upload(layer, expert, Want::Demand, false)? {
                self.stats.promotions += 1;
                if let Some(hybrid) = self.hybrid.as_mut() {
                    hybrid.in_flight[device] += 1;
                    hybrid.promoting.insert((layer, expert));
                }
            }
        }
        Ok(())
    }

    /// The workers, for work the store does not schedule itself: the CPU's
    /// part of a prompt pass.
    pub fn cpu_pool(&self) -> Option<&CpuPool> {
        self.hybrid.as_ref().map(|h| &h.pool)
    }

    /// Asks for `experts` of `layer` in the RAM tier, read from the file if
    /// they are in no tier: for experts the CPU will compute outside a step
    /// (a prompt pass). Follow them with [`ExpertStore::in_ram`].
    pub fn read_for_cpu(&mut self, layer: usize, experts: &[u32]) {
        let now = Instant::now();
        for expert in experts {
            if self.ram.sets[layer].contains(*expert)
                || self.wants.iter().any(|w| w.layer == layer && w.expert == *expert && w.cpu)
            {
                continue;
            }
            self.wants.push(Wanted {
                layer,
                expert: *expert,
                demand: true,
                since: now,
                source: Source::Disk,
                cpu: true,
            });
        }
    }

    /// Keeps `expert` of `layer` in its RAM slot and says where its
    /// matrices are, for a job the caller runs; `None` when RAM does not
    /// hold it. Every hold needs its [`ExpertStore::release_from_cpu`], after
    /// the job has ended.
    pub fn hold_for_cpu(
        &mut self,
        layer: usize,
        expert: u32,
        rows: u32,
        weight: [f32; MAX_ROWS],
    ) -> Result<Option<CpuExpert>, StoreError> {
        if !self.ram.sets[layer].contains(expert) {
            return Ok(None);
        }
        let found = self.cpu_expert(layer, expert, rows, weight)?;
        self.ram.sets[layer].hold(expert);
        Ok(Some(found))
    }

    /// Ends a [`ExpertStore::hold_for_cpu`]: the expert may give way again.
    pub fn release_from_cpu(&mut self, layer: usize, expert: u32) {
        self.ram.sets[layer].release(expert);
        self.ram.sets[layer].unprotect(expert);
        self.worth_stale[layer] = true;
    }

    /// A step begins: nothing is claimed.
    pub(super) fn cpu_begin_step(&mut self) {
        if let Some(hybrid) = self.hybrid.as_mut() {
            hybrid.claimed.clear();
        }
    }

    /// Whether `layer` of the running step waits for the host's sums.
    pub(super) fn cpu_is_open(&self, layer: usize) -> bool {
        self.hybrid.as_ref().is_some_and(|h| h.open.iter().any(|c| c.layer == layer))
    }

    /// Experts of the running step whose sums the CPU is computing.
    pub(super) fn cpu_waiting(&self) -> usize {
        self.hybrid.as_ref().map_or(0, |h| h.open.iter().map(|c| c.held.len()).sum())
    }

    /// Whether the CPU half has nothing open.
    pub(super) fn cpu_is_settled(&self) -> bool {
        self.hybrid.as_ref().is_none_or(|h| h.open.is_empty() && h.promoting.is_empty())
    }

    /// Starts the upload of an expert into a VRAM slot, if RAM holds it and
    /// a slot can be had now; returns whether it did. With `entry` the table
    /// entry follows the bytes at once (an upload a step waits for);
    /// without, it is written later, by [`ExpertStore::poll`] (a promotion).
    pub(super) fn upload(
        &mut self,
        layer: usize,
        expert: u32,
        want: Want,
        entry: bool,
    ) -> Result<bool, StoreError> {
        // Asked here and not by the caller alone: of several promotions that
        // were chosen together, the first may have copied its victim down
        // into the RAM slot of the second.
        if !self.ram.sets[layer].contains(expert) {
            return Ok(false);
        }
        if self.vram.sets[layer].n_free() == 0 && !self.can_evict(layer) {
            return Ok(false);
        }
        let Start::Slot(swap) =
            self.vram.sets[layer].start_read(expert, want, self.step, self.heat.layer(layer))
        else {
            return Ok(false);
        };
        let g = self.layers[layer];
        let slot = self.vram.slot(layer, swap.slot);
        // The RAM copy stays where it is until the device has read it.
        self.ram.sets[layer].hold(expert);
        if let Some(evicted) = swap.evict {
            self.stats.evictions += 1;
            self.devices[g.device].set_entry(layer, evicted, -1)?;
            self.demote(layer, evicted, slot)?;
        }
        let Some(ram_slot) = self.ram.sets[layer].slot(expert) else {
            return Err(StoreError::Plan("a held expert left the RAM tier".into()));
        };
        let skips = self.ram_skip[self.ram.pool_of[layer]][self.ram.slot(layer, ram_slot) as usize];
        for (part, skip) in skips.iter().enumerate() {
            // SAFETY: the source is the expert's bytes in its RAM slot,
            // which the hold keeps in place until the mark below is passed.
            unsafe {
                let src = self.ram_part(layer, ram_slot, part).add(*skip as usize);
                self.devices[g.device].upload(
                    g.device_pool,
                    slot,
                    g.offsets[part],
                    src,
                    g.lens[part],
                )?;
            }
            self.stats.upload_bytes += g.lens[part] as u64;
        }
        if entry {
            self.devices[g.device].set_entry(layer, expert, slot as i32)?;
        }
        let mark = self.devices[g.device].mark(false)?;
        let op = if entry {
            DeviceOp::Upload { layer, swap, since: Instant::now() }
        } else {
            DeviceOp::Promotion { layer, swap, since: Instant::now() }
        };
        self.device_ops[g.device][0].push_back((mark, op));
        self.upload_queued(g.device);
        self.stats.uploads += 1;
        Ok(true)
    }
}

/// Makes the job of `experts` for an open layer and hands it to the pool.
fn start_job(
    hybrid: &Hybrid,
    claims: &Claims,
    experts: &[CpuExpert],
) -> Result<Arc<LayerJob>, StoreError> {
    let shape = &hybrid.shapes[claims.layer];
    // SAFETY: the matrices are whole experts of the layer's shape in the RAM
    // tier (the store read or copied them there), and the store holds each
    // of them in its slot until the layer's sums have been delivered, which
    // is after the job has run.
    let job = unsafe {
        LayerJob::new(hybrid.pool.features(), shape, claims.n_rows, &claims.act, experts)
    }
    .map_err(|e| StoreError::Plan(e.to_string()))?;
    let job = Arc::new(job);
    hybrid.pool.submit(Arc::clone(&job));
    Ok(job)
}

/// Bytes of a matrix of `ty` for a job that is only timed: they vary, so that
/// the codebook lookups do, and every byte is below 0x40, which keeps every
/// scale (an f16 wherever it lies) small and finite. The bytes that hold the
/// exponent of a float, or of a block of MXFP4, are set so that no weight is
/// subnormal or near it: the products with such weights are subnormal, which
/// a CPU computes many times slower, and no real weight is.
pub(super) fn timing_bytes(ty: GgmlType, len: usize, seed: usize) -> Vec<u8> {
    let mut bytes: Vec<u8> =
        (0..len).map(|i| ((i * 31 + seed * 17) ^ (i >> 7)) as u8 & 0x3f).collect();
    // The width of a value or block, and where in it the exponent is: 0x38 to
    // 0x3f there is 2^-15 to 2^0 for the float types, 0x78 to 0x7b a block
    // scale of 2^-7 to 2^-4 for MXFP4.
    let (width, at, exponent, mask) = match ty {
        GgmlType::F32 => (4, 3, 0x38, 0x07),
        GgmlType::F16 | GgmlType::BF16 => (2, 1, 0x38, 0x07),
        GgmlType::MXFP4 => (ty.block_bytes() as usize, 0, 0x78, 0x03),
        _ => return bytes,
    };
    for value in bytes.chunks_exact_mut(width) {
        value[at] = exponent | (value[at] & mask);
    }
    bytes
}

/// Times jobs of every pair of types the model's experts have, of 1, 2, 4
/// and 8 pairs, on the pool: the cost model's first numbers. The bytes are
/// arbitrary (the time of a kernel does not depend on them) with finite
/// scales.
fn calibrate(
    pool: &CpuPool,
    shapes: &[ExpertShape],
    model: &mut CostModel,
) -> Result<(), StoreError> {
    let mut seen: Vec<ExpertTypes> = Vec::new();
    for shape in shapes {
        let types = (shape.gate_ty, shape.down_ty);
        if seen.contains(&types) {
            continue;
        }
        seen.push(types);
        let plan = |what: &str| StoreError::Plan(format!("calibration: {what}"));
        let gate_bytes = shape.gate_bytes().ok_or_else(|| plan("gate size"))? as usize;
        let down_bytes = shape.down_bytes().ok_or_else(|| plan("down size"))? as usize;
        // Several experts, so that a job reads memory the last one did not
        // leave in the cache: 8 experts are 9 to 12 MB in the types of the
        // model the kernels were written for.
        let experts: Vec<[Vec<u8>; 3]> = (0..8)
            .map(|e| {
                [
                    timing_bytes(shape.gate_ty, gate_bytes, e),
                    timing_bytes(shape.gate_ty, gate_bytes, e + 8),
                    timing_bytes(shape.down_ty, down_bytes, e + 16),
                ]
            })
            .collect();
        let n_in = shape.n_embd as usize;
        for rows in [1usize, 2, 4, 8] {
            let x: Vec<f32> =
                (0..rows * n_in).map(|i| ((i * 37 % 255) as f32 - 127.0) / 64.0).collect();
            let act = super::cpu::quantize_activations(&x, n_in);
            for round in 0..6 {
                let m = &experts[(round + rows) % experts.len()];
                let expert = CpuExpert {
                    gate: m[0].as_ptr(),
                    up: m[1].as_ptr(),
                    down: m[2].as_ptr(),
                    rows: (1u32 << rows) - 1,
                    weight: [0.1; MAX_ROWS],
                };
                // SAFETY: the matrices have the sizes of the shape and
                // outlive the job, which is waited for below.
                let job = unsafe { LayerJob::new(pool.features(), shape, rows, &act, &[expert]) }
                    .map_err(|e| StoreError::Plan(e.to_string()))?;
                let job = Arc::new(job);
                pool.submit(Arc::clone(&job));
                let start = Instant::now();
                while !job.is_done() {
                    if start.elapsed() > Duration::from_secs(30) {
                        return Err(plan("the workers do not finish a job"));
                    }
                    std::hint::spin_loop();
                }
                // The first rounds wake the workers and warm the tables.
                if let (true, Some(time)) = (round >= 2, job.elapsed()) {
                    model.record_job(types, rows, time);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The types of a layer whose jobs the test models are timed for.
    const FAST: ExpertTypes = (GgmlType::IQ2_S, GgmlType::Q2_0);

    fn model(cpu_per_pair: f64, upload: f64) -> CostModel {
        let mut model = CostModel::new(1);
        for pairs in [1, 2, 4, 8, 16, 32] {
            model.record_job(FAST, pairs, Duration::from_secs_f64(cpu_per_pair * pairs as f64));
        }
        model.record_upload(0, Duration::from_secs_f64(upload));
        model
    }

    fn miss(pairs: u32, heat: f32) -> Miss {
        Miss { pairs, heat, cpu_only: false }
    }

    #[test]
    fn the_split_ends_soonest() {
        // 100 us per pair on the CPU, 150 us per upload.
        let m = model(100e-6, 150e-6);
        // One expert, one row: the CPU is sooner.
        assert_eq!(m.split(FAST, 0, &[miss(1, 1.0)], 0), vec![true]);
        // One expert that eight rows use: one upload beats 800 us of CPU.
        assert_eq!(m.split(FAST, 0, &[miss(8, 1.0)], 0), vec![false]);
        // Six experts of one row: all on the CPU end at 600 us; with the
        // two hottest uploaded the link ends at 300 and the CPU at 400; a
        // third upload would end at 450.
        let six: Vec<Miss> = (0..6).map(|i| miss(1, i as f32)).collect();
        assert_eq!(m.split(FAST, 0, &six, 0), vec![true, true, true, true, false, false]);
        // With four uploads queued on the link, the CPU takes them all.
        assert_eq!(m.split(FAST, 0, &six, 4), vec![true; 6]);
        // The expert with the most rows goes up first.
        let mixed = [miss(1, 9.0), miss(4, 1.0), miss(1, 1.0)];
        assert_eq!(m.split(FAST, 0, &mixed, 0), vec![true, false, true]);
    }

    #[test]
    fn an_expert_on_its_way_stays_with_the_cpu() {
        let m = model(100e-6, 10e-6);
        let mut misses = vec![miss(4, 1.0), miss(4, 1.0)];
        misses[0].cpu_only = true;
        assert_eq!(m.split(FAST, 0, &misses, 0), vec![true, false]);
    }

    #[test]
    fn a_side_that_was_never_timed_is_tried() {
        let mut m = CostModel::new(1);
        m.record_job(FAST, 1, Duration::from_micros(100));
        // No upload was timed: one expert goes up.
        assert_eq!(m.split(FAST, 0, &[miss(1, 1.0), miss(1, 2.0)], 0), vec![true, false]);
        // No job of this type was timed: the CPU gets one expert.
        m.record_upload(0, Duration::from_micros(200));
        assert_eq!(
            m.split((GgmlType::IQ1_M, GgmlType::Q2_0), 0, &[miss(2, 1.0), miss(1, 1.0)], 0),
            vec![false, true]
        );
    }

    #[test]
    fn every_pair_of_types_is_timed_by_itself() {
        // A layer whose jobs take 100 us a pair, and layers that differ from it in the
        // type of gate and up or in the type of down, whose jobs take 50 times that.
        let mut m = model(100e-6, 150e-6);
        let slow_gate = (GgmlType::Q4_K, GgmlType::Q2_0);
        let slow_down = (GgmlType::IQ2_S, GgmlType::Q6_K);
        for types in [slow_gate, slow_down] {
            // Never timed: the CPU gets one expert, which times it.
            assert_eq!(m.job_time(types, 1), None);
            assert_eq!(m.split(types, 0, &[miss(1, 1.0), miss(1, 2.0)], 0), vec![true, false]);
            for pairs in [1, 2, 4, 8] {
                m.record_job(types, pairs, Duration::from_secs_f64(5e-3 * pairs as f64));
            }
        }
        // What the first layer's jobs take is what it was.
        assert_eq!(m.split(FAST, 0, &[miss(1, 1.0)], 0), vec![true]);
        let near = |t: Option<f64>, want: f64| t.is_some_and(|t| (t - want).abs() < 1e-9);
        assert!(near(m.job_time(FAST, 1), 100e-6));
        for types in [slow_gate, slow_down] {
            assert!(near(m.job_time(types, 2), 10e-3));
            // An upload of 150 us is sooner than 5 ms of CPU, for one expert and for
            // six, also behind 20 uploads that are queued.
            assert_eq!(m.split(types, 0, &[miss(1, 1.0)], 0), vec![false]);
            assert_eq!(m.split(types, 0, &[miss(1, 1.0); 6], 20), vec![false; 6]);
        }
        // An expert that cannot be uploaded now is the CPU's whatever it costs.
        let held = Miss { pairs: 1, heat: 1.0, cpu_only: true };
        assert_eq!(m.split(slow_gate, 0, &[held, miss(1, 1.0)], 0), vec![true, false]);
    }

    #[test]
    fn job_times_come_from_the_nearest_size() {
        let mut m = CostModel::new(1);
        assert_eq!(m.job_time(FAST, 3), None);
        m.record_job(FAST, 1, Duration::from_micros(200));
        m.record_job(FAST, 8, Duration::from_micros(800));
        let near = |a: Option<f64>, b: f64| a.is_some_and(|a| (a - b).abs() < 1e-9);
        assert!(near(m.job_time(FAST, 1), 200e-6));
        assert!(near(m.job_time(FAST, 2), 400e-6));
        assert!(near(m.job_time(FAST, 6), 600e-6));
        assert!(near(m.job_time(FAST, 40), 4000e-6));
        assert_eq!(m.job_time(FAST, 0), Some(0.0));
    }
}
