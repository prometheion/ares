//! The CPU's part of a wide step: misses of a layer that few rows picked.
//!
//! A wide step (a slice of a prompt) brings what a layer misses through transit
//! (`transit.rs`): one upload per expert, whatever number of rows picked it. In a step of a
//! few rows (a turn whose prompt the cache holds but for its last rows, the end of a
//! prompt) a layer misses hundreds of experts picked by one or two rows each, and their
//! uploads take seconds on a link of a few GB/s, where the CPU computes a row of an expert
//! in tens of microseconds. So the CPU takes the misses of a wide step that are in the RAM
//! tier and that the cost model gives it, as in a decode step: the split whose later half
//! ends first, uploads for the experts with the most rows ([`CostModel::split_by`], with the
//! time per pair of this work, which is measured apart from the decode steps' jobs: its jobs
//! are packed differently). Transit brings the others; the layer ends when both have.
//!
//! A job ([`LayerJob`]) takes at most [`MAX_ROWS`] rows, so the rows are packed: each
//! expert's rows in pieces of at most that many, and the pieces, in the order of their
//! first row, into jobs whose rows together are at most that many. An expert whose rows
//! fall into several jobs is read once for each; in a step of a few rows that is rare,
//! and the cost model learns what it costs.
//!
//! The arithmetic is the decode steps' CPU half: the int8 codes of the rows' mixed input
//! and the device's sums per block of 32 (`cpu/mod.rs`).
//!
//! [`CostModel::split_by`]: super::CostModel::split_by

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use super::super::cpu::{CpuExpert, LayerJob, MAX_ROWS, act_bytes};
use super::super::{ExpertStore, SlotDevice, StoreError};
use super::{Miss, Serve};

/// An expert of a layer of a wide step that the CPU computes: the rows of the step that
/// picked it, ascending, with their weights.
#[derive(Clone, Debug, PartialEq)]
pub struct WidePick {
    pub expert: u32,
    pub rows: Vec<(u32, f32)>,
}

/// The CPU's jobs for a layer of a wide step ([`ExpertStore::wide_cpu`]). The experts stay
/// in their RAM slots until [`ExpertStore::wide_cpu_end`]; dropped without that, they stay
/// there for good.
#[derive(Debug)]
pub struct WideCpu {
    layer: usize,
    /// Each job with the rows of the step that its rows are.
    jobs: Vec<(Arc<LayerJob>, Vec<u32>)>,
    held: Vec<u32>,
    pairs: usize,
    rows: Range<u32>,
    since: Instant,
}

impl WideCpu {
    /// Whether every job has run.
    pub fn is_done(&self) -> bool {
        self.jobs.iter().all(|(job, _)| job.is_done())
    }

    /// The rows of the step whose sums the jobs compute: from the first that picked one of
    /// the experts to the last.
    pub fn rows(&self) -> Range<u32> {
        self.rows.clone()
    }

    /// The experts and the (expert, row) pairs.
    pub fn experts(&self) -> usize {
        self.held.len()
    }

    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// The jobs the experts were packed into.
    pub fn jobs(&self) -> usize {
        self.jobs.len()
    }
}

/// A job of [`pack`]: its rows, ascending, and its pieces (the index of a pick and the range
/// of its rows).
type Packed = (Vec<u32>, Vec<(usize, Range<usize>)>);

/// Packs the rows of `picks` into jobs of at most [`MAX_ROWS`] rows: the pieces of each
/// expert's rows, in the order of their first row, into the job before as long as the rows
/// of the job stay that few. Returns per job its rows, ascending, and its pieces (the index
/// of the pick and the range of its rows).
fn pack(picks: &[WidePick]) -> Vec<Packed> {
    let mut pieces: Vec<(usize, Range<usize>)> = Vec::new();
    for (p, pick) in picks.iter().enumerate() {
        let mut at = 0;
        while at < pick.rows.len() {
            let end = (at + MAX_ROWS).min(pick.rows.len());
            pieces.push((p, at..end));
            at = end;
        }
    }
    let first_row = |(p, range): &(usize, Range<usize>)| picks[*p].rows[range.start].0;
    pieces.sort_by_key(|piece| (first_row(piece), piece.0));
    let mut jobs: Vec<Packed> = Vec::new();
    for (p, range) in pieces {
        let rows = picks[p].rows[range.clone()].iter().map(|(row, _)| *row);
        let fits = jobs.last().is_some_and(|(job, _)| {
            let new = rows.clone().filter(|row| !job.contains(row)).count();
            job.len() + new <= MAX_ROWS
        });
        if !fits {
            jobs.push((Vec::new(), Vec::new()));
        }
        let (job, members) = jobs.last_mut().expect("a job");
        for row in rows {
            if !job.contains(&row) {
                job.push(row);
            }
        }
        members.push((p, range));
    }
    for (rows, _) in &mut jobs {
        rows.sort_unstable();
    }
    jobs
}

/// The activations of `rows` of a step of `n_rows` rows of `n` values (`act`, as
/// [`act_bytes`] lays them out) as the activations of a step of just those rows.
fn gather(act: &[u8], n: usize, n_rows: usize, rows: &[u32]) -> Vec<u8> {
    let scales = n / 32 * 8;
    let from = (n_rows * n).next_multiple_of(16);
    let to = (rows.len() * n).next_multiple_of(16);
    let mut out = vec![0u8; act_bytes(n, rows.len())];
    for (j, row) in rows.iter().enumerate() {
        let r = *row as usize;
        out[j * n..(j + 1) * n].copy_from_slice(&act[r * n..(r + 1) * n]);
        out[to + j * scales..to + (j + 1) * scales]
            .copy_from_slice(&act[from + r * scales..from + (r + 1) * scales]);
    }
    out
}

impl<D: SlotDevice> ExpertStore<D> {
    /// Which of the misses of `layer` in a wide step the CPU computes: `misses` are the
    /// experts with no slot and the number of rows that picked each. Only experts in the RAM
    /// tier can go to the CPU; the cost model splits those between the CPU and the link,
    /// behind the uploads of the others (which transit brings from the file). None without
    /// a CPU half or when every miss is streamed ([`Serve::Stream`]); every one it can take
    /// with [`Serve::Cpu`].
    pub fn wide_split(&self, layer: usize, misses: &[(u32, u32)]) -> Vec<bool> {
        let mut on_cpu = vec![false; misses.len()];
        let Some(hybrid) = self.hybrid.as_ref() else { return on_cpu };
        if hybrid.serve == Serve::Stream || misses.is_empty() {
            return on_cpu;
        }
        let ram = &self.ram.sets[layer];
        let able: Vec<usize> = (0..misses.len()).filter(|i| ram.contains(misses[*i].0)).collect();
        if hybrid.serve == Serve::Cpu {
            able.iter().for_each(|i| on_cpu[*i] = true);
            return on_cpu;
        }
        let heat = self.heat.layer(layer);
        let candidates: Vec<Miss> = able
            .iter()
            .map(|i| Miss {
                pairs: misses[*i].1,
                heat: heat[misses[*i].0 as usize],
                cpu_only: false,
            })
            .collect();
        let shape = hybrid.shapes[layer];
        let types = (shape.gate_ty, shape.down_ty);
        let device = self.layers[layer].device;
        let others = misses.len() - able.len();
        let model = &hybrid.model;
        let split =
            model.split_by(|pairs| model.wide_time(types, pairs), device, &candidates, others);
        for (i, cpu) in able.iter().zip(split) {
            on_cpu[*i] = cpu;
        }
        on_cpu
    }

    /// Starts the CPU's jobs for `picks` of `layer` of a wide step of `n_rows` rows whose
    /// mixed input is `act` (the first [`act_bytes`] bytes of the step's int8 activations).
    /// Every pick must be in the RAM tier ([`ExpertStore::wide_split`] says which are); it is
    /// held in its slot until [`ExpertStore::wide_cpu_end`].
    pub fn wide_cpu(
        &mut self,
        layer: usize,
        n_rows: usize,
        act: &[u8],
        picks: &[WidePick],
    ) -> Result<WideCpu, StoreError> {
        let since = Instant::now();
        let hybrid = self.hybrid.as_ref().ok_or_else(|| StoreError::Plan("no CPU half".into()))?;
        let shape = hybrid.shapes[layer];
        let n = shape.n_embd as usize;
        if act.len() < act_bytes(n, n_rows) {
            return Err(StoreError::Plan(format!(
                "layer {layer}: {} bytes of activations for {n_rows} rows",
                act.len()
            )));
        }
        let bad = |pick: &WidePick| {
            pick.rows.is_empty()
                || pick.rows.iter().any(|(row, _)| *row as usize >= n_rows)
                || pick.rows.windows(2).any(|w| w[0].0 >= w[1].0)
                || !self.ram.sets[layer].contains(pick.expert)
        };
        if let Some(pick) = picks.iter().find(|pick| bad(pick)) {
            return Err(StoreError::Plan(format!(
                "layer {layer}: expert {} with rows {:?} is not one the CPU can take",
                pick.expert, pick.rows
            )));
        }
        let mut jobs = Vec::new();
        for (rows, members) in pack(picks) {
            let job_act = gather(act, n, n_rows, &rows);
            let mut experts = Vec::with_capacity(members.len());
            for (p, range) in members {
                let pick = &picks[p];
                let (mut mask, mut weight) = (0u32, [0.0f32; MAX_ROWS]);
                for (row, w) in &pick.rows[range] {
                    let b = rows.binary_search(row).expect("the job has the row");
                    mask |= 1 << b;
                    weight[b] = *w;
                }
                experts.push(self.cpu_expert(layer, pick.expert, mask, weight)?);
            }
            jobs.push((experts, rows, job_act));
        }
        for pick in picks {
            self.ram.sets[layer].hold(pick.expert);
        }
        let held: Vec<u32> = picks.iter().map(|pick| pick.expert).collect();
        let hybrid = self.hybrid.as_ref().ok_or_else(|| StoreError::Plan("no CPU half".into()))?;
        let mut started = Vec::with_capacity(jobs.len());
        for (experts, rows, job_act) in jobs {
            let experts: &[CpuExpert] = &experts;
            // SAFETY: the matrices are whole experts of the layer's shape in their RAM slots,
            // which are held until `wide_cpu_end`, after every job has run.
            let job = unsafe {
                LayerJob::new(hybrid.pool.features(), &shape, rows.len(), &job_act, experts)
            }
            .map_err(|e| StoreError::Plan(e.to_string()))?;
            let job = Arc::new(job);
            hybrid.pool.submit(Arc::clone(&job));
            started.push((job, rows));
        }
        let pairs = picks.iter().map(|pick| pick.rows.len()).sum();
        let rows = match (
            picks.iter().filter_map(|pick| pick.rows.first()).map(|(row, _)| *row).min(),
            picks.iter().filter_map(|pick| pick.rows.last()).map(|(row, _)| *row).max(),
        ) {
            (Some(first), Some(last)) => first..last + 1,
            _ => 0..0,
        };
        self.stats.cpu_experts += picks.len() as u64;
        self.stats.cpu_pairs += pairs as u64;
        Ok(WideCpu { layer, jobs: started, held, pairs, rows, since })
    }

    /// Ends the CPU's part `cpu` of a layer of a wide step, whose jobs have all run: adds
    /// their sums into `sums` (the rows of [`WideCpu::rows`], `n_embd` values each), lets
    /// the experts go from their RAM slots and times the work for the cost model.
    pub fn wide_cpu_end(&mut self, cpu: WideCpu, sums: &mut [f32]) -> Result<(), StoreError> {
        let Some(hybrid) = self.hybrid.as_mut() else {
            return Err(StoreError::Plan("no CPU half".into()));
        };
        if !cpu.is_done() {
            return Err(StoreError::Plan("the CPU's jobs of a wide step have not run".into()));
        }
        let shape = hybrid.shapes[cpu.layer];
        let n = shape.n_embd as usize;
        let first = cpu.rows.start as usize;
        if sums.len() != cpu.rows.len() * n {
            return Err(StoreError::Plan(format!(
                "{} sums for {} rows of {n}",
                sums.len(),
                cpu.rows.len()
            )));
        }
        for (job, rows) in &cpu.jobs {
            let part = job.sums().expect("the job has run");
            for (j, row) in rows.iter().enumerate() {
                let to = (*row as usize - first) * n;
                let into = &mut sums[to..to + n];
                into.iter_mut().zip(&part[j * n..(j + 1) * n]).for_each(|(s, p)| *s += p);
            }
            self.stats.cpu_jobs += 1;
            self.stats.cpu_busy_ms += job.busy().as_secs_f64() * 1e3;
        }
        let begun = cpu.jobs.iter().map(|(job, _)| job.created()).min();
        let ended = cpu.jobs.iter().filter_map(|(job, _)| job.ended()).max();
        if let (Some(begun), Some(ended)) = (begun, ended) {
            let types = (shape.gate_ty, shape.down_ty);
            hybrid.model.record_wide(types, cpu.pairs, ended.saturating_duration_since(begun));
            let ms = |at: Instant| at.saturating_duration_since(cpu.since).as_secs_f32() * 1e3;
            self.stats.cpu_start_ms.push(ms(begun));
            self.stats.cpu_done_ms.push(ms(ended));
        }
        for expert in &cpu.held {
            self.ram.sets[cpu.layer].release(*expert);
        }
        self.stats.cpu_ms.push(cpu.since.elapsed().as_secs_f32() * 1e3);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(expert: u32, rows: &[u32]) -> WidePick {
        WidePick { expert, rows: rows.iter().map(|r| (*r, 0.5)).collect() }
    }

    #[test]
    fn rows_are_packed_into_jobs_of_at_most_eight() {
        // Experts of one row each over twelve rows, one of two rows, and one of eleven: the
        // pieces go into jobs by their first row while the job's rows stay eight.
        let mut picks: Vec<WidePick> = (0..12).map(|r| pick(r, &[r])).collect();
        picks.push(pick(20, &[3, 9]));
        picks.push(pick(21, &(0..11).collect::<Vec<u32>>()));
        let jobs = pack(&picks);
        for (rows, members) in &jobs {
            assert!(rows.len() <= MAX_ROWS, "{rows:?}");
            for (p, range) in members {
                for (row, _) in &picks[*p].rows[range.clone()] {
                    assert!(rows.contains(row), "row {row} of pick {p} is in its job");
                }
            }
        }
        // Every pair is in one job exactly.
        let pairs: usize = jobs.iter().flat_map(|(_, m)| m).map(|(_, range)| range.len()).sum();
        assert_eq!(pairs, 12 + 2 + 11);
        // The expert of eleven rows is read twice, its pieces of eight and three rows.
        let pieces = jobs.iter().flat_map(|(_, m)| m).filter(|(p, _)| *p == 13).count();
        assert_eq!(pieces, 2);
        assert!(jobs.len() <= 5, "{} jobs", jobs.len());
    }

    #[test]
    fn the_rows_of_a_job_are_gathered_with_their_scales() {
        let n = 64;
        let n_rows = 3;
        let x: Vec<f32> = (0..n * n_rows).map(|i| (i as f32 * 0.37).sin()).collect();
        let act = crate::expert_store::cpu::quantize_activations(&x, n);
        let rows = [2u32, 0];
        let mut picked = Vec::new();
        for row in rows {
            picked.extend_from_slice(&x[row as usize * n..(row as usize + 1) * n]);
        }
        assert_eq!(
            gather(&act, n, n_rows, &rows),
            crate::expert_store::cpu::quantize_activations(&picked, n)
        );
    }
}
