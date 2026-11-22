//! The threads that compute experts, and where they sit.
//!
//! Workers only compute: they take [`LayerJob`]s from a queue and run their
//! tasks. Tasks are claimed one at a time, so cores of different speed share
//! a job by how fast each one is, without the pool knowing their speeds.
//!
//! A worker that finds nothing spins for a while before it sleeps: the jobs
//! of a step come a few milliseconds apart and each takes a fraction of one,
//! so a worker that had to be woken would arrive when the job is over.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{CpuError, Features, LayerJob, MEASURED, Team, ThreadSpeed, thread_speed};

/// The logical CPUs this process may run on, by kind.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Topology {
    /// The first thread of every performance core (of every core, on a CPU
    /// with one kind of core).
    pub performance: Vec<usize>,
    /// The other threads of those cores, in the order of their cores.
    pub siblings: Vec<usize>,
    /// The efficiency cores.
    pub efficiency: Vec<usize>,
}

/// Reads a list like `0-3,8,10-11` as the kernel prints CPU sets.
fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|p| !p.is_empty()) {
        let (first, last) = part.split_once('-').unwrap_or((part, part));
        if let (Ok(first), Ok(last)) = (first.trim().parse::<usize>(), last.trim().parse::<usize>())
        {
            cpus.extend(first..=last);
        }
    }
    cpus
}

fn read_cpu_list(path: &str) -> Vec<usize> {
    std::fs::read_to_string(path).map(|text| parse_cpu_list(&text)).unwrap_or_default()
}

/// The CPUs the scheduler lets this process use.
fn allowed_cpus() -> Vec<usize> {
    // SAFETY: an all-zero set is a valid `cpu_set_t`, and the call writes at
    // most its size into it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize).filter(|cpu| libc::CPU_ISSET(*cpu, &set)).collect()
    }
}

impl Topology {
    /// The topology of this machine, from sysfs: the kinds of core as the
    /// kernel's PMUs name them (`cpu_core`, `cpu_atom`) and each core's
    /// threads. A machine without that split has performance cores only.
    pub fn read() -> Self {
        let allowed = allowed_cpus();
        let efficiency = read_cpu_list("/sys/devices/cpu_atom/cpus");
        Self::from_lists(&allowed, &efficiency, |cpu| {
            read_cpu_list(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
        })
    }

    /// Sorts `allowed` into kinds: `efficiency` names the efficiency cores,
    /// `threads_of(cpu)` the logical CPUs of `cpu`'s core.
    fn from_lists(
        allowed: &[usize],
        efficiency: &[usize],
        threads_of: impl Fn(usize) -> Vec<usize>,
    ) -> Self {
        let mut topology = Self::default();
        for cpu in allowed {
            if efficiency.contains(cpu) {
                topology.efficiency.push(*cpu);
            } else if threads_of(*cpu)
                .iter()
                .filter(|t| allowed.contains(t))
                .min()
                .is_none_or(|t| t == cpu)
            {
                topology.performance.push(*cpu);
            } else {
                topology.siblings.push(*cpu);
            }
        }
        topology
    }
}

/// Which logical CPU each worker is bound to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// One entry per worker; `None` leaves the worker to the scheduler.
    pub cpus: Vec<Option<usize>>,
}

impl Placement {
    /// A worker on each of `cpus`.
    pub fn pinned(cpus: &[usize]) -> Self {
        Self { cpus: cpus.iter().map(|cpu| Some(*cpu)).collect() }
    }

    /// `workers` workers that the scheduler places.
    pub fn unpinned(workers: usize) -> Self {
        Self { cpus: vec![None; workers] }
    }

    /// The placement the engine uses on a machine with `topology`: one
    /// worker on the first thread of every performance core but `reserved`
    /// of them, and one on every efficiency core.
    ///
    /// The reserved cores are for the threads that must not wait behind a
    /// worker: the conductor, which spins through a step. The second threads
    /// of the performance cores get no worker: a second worker on a core
    /// shares its vector units with the first, and those threads are where
    /// the device threads and the I/O completions run. Nothing here is a
    /// count of this or that CPU: what the system reports decides, and how
    /// fast each worker's core is, the worker measures itself.
    pub fn for_machine(topology: &Topology, reserved: usize) -> Self {
        let mut cpus: Vec<usize> = topology.performance.iter().skip(reserved).copied().collect();
        cpus.extend(&topology.efficiency);
        if cpus.is_empty() {
            // A machine with nothing to spare still computes its misses.
            return Self::unpinned(1);
        }
        Self::pinned(&cpus)
    }
}

struct Shared {
    queue: Mutex<VecDeque<Arc<LayerJob>>>,
    /// Jobs in the queue: lets an idle worker look without the lock.
    queued: AtomicUsize,
    stop: AtomicBool,
    sleepers: AtomicUsize,
    sleep: Mutex<()>,
    wake: Condvar,
    spin: Duration,
    /// What each worker measured of its core when it started.
    speeds: Mutex<Vec<ThreadSpeed>>,
    team: std::sync::OnceLock<Team>,
}

impl Shared {
    /// The oldest job that still has tasks to claim.
    fn take(&self) -> Option<Arc<LayerJob>> {
        if self.queued.load(Ordering::SeqCst) == 0 {
            return None;
        }
        let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
        while queue.front().is_some_and(|job| job.is_claimed()) {
            queue.pop_front();
            self.queued.fetch_sub(1, Ordering::SeqCst);
        }
        queue.front().cloned()
    }

    fn worker(&self, cpu: Option<usize>, vector: bool) {
        if let Some(cpu) = cpu {
            // SAFETY: an all-zero set is valid, `cpu` is below the set's
            // size (checked when the pool was built), and the call reads
            // the set only. A refusal leaves the thread unbound, which is
            // slower and not wrong.
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(cpu, &mut set);
                libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
            }
        }
        // On its core now: how fast it is there.
        let speed = thread_speed(vector);
        self.speeds.lock().unwrap_or_else(PoisonError::into_inner).push(speed);
        let mut idle_since = Instant::now();
        let mut spins = 0u32;
        while !self.stop.load(Ordering::Relaxed) {
            if let Some(job) = self.take() {
                job.work_in(self.team.get());
                idle_since = Instant::now();
                spins = 0;
                continue;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if !spins.is_multiple_of(256) || idle_since.elapsed() < self.spin {
                continue;
            }
            // Asleep until a job is posted. The count of sleepers goes up
            // before the queue is looked at once more, and a poster looks at
            // the count after it has queued, so one of the two sees the other.
            let guard = self.sleep.lock().unwrap_or_else(PoisonError::into_inner);
            self.sleepers.fetch_add(1, Ordering::SeqCst);
            if self.queued.load(Ordering::SeqCst) == 0 && !self.stop.load(Ordering::SeqCst) {
                drop(self.wake.wait(guard).unwrap_or_else(PoisonError::into_inner));
            }
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
            idle_since = Instant::now();
        }
    }
}

/// The worker threads of the CPU's share of the experts.
pub struct CpuPool {
    features: Features,
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for CpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuPool").field("workers", &self.threads.len()).finish_non_exhaustive()
    }
}

impl CpuPool {
    /// Starts a worker per entry of `placement`. An idle worker spins for
    /// `spin` before it sleeps. What the CPU has is asked here
    /// ([`Features::detect`]), once for the pool's jobs.
    pub fn new(placement: &Placement, spin: Duration) -> Result<Self, CpuError> {
        let features = Features::detect();
        if let Some(cpu) =
            placement.cpus.iter().flatten().find(|c| **c >= libc::CPU_SETSIZE as usize)
        {
            return Err(CpuError::Job(format!("no logical CPU {cpu}")));
        }
        // The tables are built here, not by the first job.
        super::tables::tables();
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            queued: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
            spin,
            speeds: Mutex::new(Vec::new()),
            team: std::sync::OnceLock::new(),
        });
        let mut threads = Vec::with_capacity(placement.cpus.len());
        for (n, cpu) in placement.cpus.iter().copied().enumerate() {
            let shared = Arc::clone(&shared);
            let vector = features.vector;
            let thread = std::thread::Builder::new()
                .name(format!("ares-expert-{n}"))
                .spawn(move || shared.worker(cpu, vector))
                .map_err(|e| CpuError::Job(format!("cannot start a worker: {e}")))?;
            threads.push(thread);
        }
        // Every worker measures its core first (a few milliseconds); the
        // team is what they found together.
        let started = Instant::now();
        loop {
            let speeds = shared.speeds.lock().unwrap_or_else(PoisonError::into_inner);
            if speeds.len() == threads.len() {
                let _ = shared.team.set(Team::of(&speeds));
                break;
            }
            drop(speeds);
            if started.elapsed() > Duration::from_secs(10) {
                return Err(CpuError::Job("the workers did not start".into()));
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        Ok(Self { features, shared, threads })
    }

    /// For each worker, in no particular order: per gate type with a kernel
    /// of its own (IQ2_S, IQ2_XXS, IQ1_M, IQ3_XXS, IQ4_XS, Q4_K, Q5_K) whether it fetches
    /// codebook entries by a gather, and the nanoseconds 32 hidden values of
    /// one row take it.
    pub fn speeds(&self) -> Vec<([bool; MEASURED], [u32; MEASURED])> {
        let speeds = self.shared.speeds.lock().unwrap_or_else(PoisonError::into_inner);
        speeds.iter().map(|s| (s.gather, s.tile_ns)).collect()
    }

    /// What the CPU has beyond the floor: what a job for this pool is made with.
    pub fn features(&self) -> Features {
        self.features
    }

    /// The number of workers.
    pub fn workers(&self) -> usize {
        self.threads.len()
    }

    /// Hands `job` to the workers. The caller follows it with
    /// [`LayerJob::is_done`]; it may also call [`LayerJob::work`] itself.
    pub fn submit(&self, job: Arc<LayerJob>) {
        if job.is_claimed() {
            return;
        }
        {
            let mut queue = self.shared.queue.lock().unwrap_or_else(PoisonError::into_inner);
            queue.push_back(job);
            self.shared.queued.fetch_add(1, Ordering::SeqCst);
        }
        if self.shared.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = self.shared.sleep.lock().unwrap_or_else(PoisonError::into_inner);
            self.shared.wake.notify_all();
        }
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        {
            let _guard = self.shared.sleep.lock().unwrap_or_else(PoisonError::into_inner);
            self.shared.wake.notify_all();
        }
        for thread in self.threads.drain(..) {
            // A worker that panicked has nothing left to clean up.
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_cpu_lists() {
        assert_eq!(parse_cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert!(parse_cpu_list("").is_empty());
    }

    #[test]
    fn sorts_cores_by_kind() {
        // Two performance cores of two threads each and two efficiency
        // cores, of which the process may not use CPU 5.
        let threads = |cpu: usize| match cpu {
            0 | 1 => vec![0, 1],
            2 | 3 => vec![2, 3],
            other => vec![other],
        };
        let topology = Topology::from_lists(&[0, 1, 2, 3, 4], &[4, 5], threads);
        assert_eq!(topology.performance, vec![0, 2]);
        assert_eq!(topology.siblings, vec![1, 3]);
        assert_eq!(topology.efficiency, vec![4]);
        // The first thread of a core is not allowed: the second stands for it.
        let topology = Topology::from_lists(&[1, 2, 3], &[], threads);
        assert_eq!(topology.performance, vec![1, 2]);
        assert_eq!(topology.siblings, vec![3]);
        // One core is the conductor's; a machine with one core still gets a worker.
        assert_eq!(Placement::for_machine(&topology, 1), Placement::pinned(&[2]));
        assert_eq!(Placement::for_machine(&topology, 2), Placement::unpinned(1));
        let hybrid = Topology::from_lists(&[0, 1, 2, 3, 4], &[4, 5], threads);
        assert_eq!(Placement::for_machine(&hybrid, 1), Placement::pinned(&[2, 4]));
    }
}
