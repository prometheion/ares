//! When page-locked host memory may change: never while a device runs a step.
//!
//! Allocating, freeing, locking and unlocking host memory for the devices
//! (`cuMemHostAlloc`, `cuMemFreeHost`, `cuMemHostRegister`,
//! `cuMemHostUnregister`) take a lock of the driver's that every call of the
//! process needs, and a free or an unlock waits under it for the work the
//! devices have queued (PyTorch's host allocator says the same of
//! `cudaFreeHost`; the CUDA C Programming Guide lists a page-locked
//! allocation among the operations that synchronize streams, "Implicit
//! Synchronization"). A step's kernels can wait for the host: for an expert
//! that is on its way, for the host's share of a layer, for the peer's half
//! of a layer. The host thread that ends such a wait enqueues a copy for it,
//! and the copy needs the driver's lock. A free from another thread while a
//! step runs then waits for the kernel, the kernel for the thread, and the
//! thread for the free: the engine stands still for good.
//!
//! So the rule, for every thread of the process: **page-locked memory changes
//! only while no device runs a job.** The thread that hands jobs to the
//! devices holds [`Steps`] from the hand-over to the end of the job (an
//! executor's dispatch to its collect), and each device thread marks itself
//! [`InStep`] while it runs one. Then:
//!
//! - [`change`] runs a lock or an allocation once no job runs, and keeps new
//!   jobs from beginning until it returns; inside a job (a thread marked
//!   [`InStep`] or holding [`Steps`]) it refuses instead of waiting for
//!   itself. [`try_change`] runs it only if no job runs now: for a thread
//!   that holds something a job may wait for.
//! - [`retire`] gives memory back: at once when no job runs, otherwise when
//!   the last job that runs ends, on the thread that ends it. A free never
//!   waits for a job, and a job never waits for a free in its middle.
//!
//! Device memory goes the same way: `cuMemFree` "may perform implicit
//! synchronization" too (the driver's API reference), so a
//! [`crate::DeviceBuffer`] dropped during a job (a plan's buffer that grows
//! inside a step) or beside another engine's step is freed where no job runs.
//! Before a caller looks at what a device has free ([`crate::Device::mem_info`])
//! or when an allocation finds no room, [`settle`] frees what waits.
//!
//! [`crate::PinnedBuffer`] and [`crate::HostRegistration`] go through these
//! themselves, so no owner of page-locked memory can break the rule; an owner
//! that books the bytes elsewhere retires the memory together with its booking,
//! so that the books give the bytes back when the memory is really free.
//!
//! The state is the process's, not an engine's: the driver's lock is.

use std::cell::Cell;
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::error::{Error, Result};

/// What the gate knows of the process.
struct State {
    /// Jobs that run on the devices now: one per [`Steps`] held.
    steps: usize,
    /// A thread changes page-locked memory now.
    changing: bool,
    /// Threads that wait in [`change`]: no job begins before they have run.
    waiting: usize,
    /// What was given back while jobs ran, dropped when the last one ends.
    later: Vec<Box<dyn Send>>,
}

/// The gate: the process has one ([`GATE`]); the tests make their own.
struct Gate {
    state: Mutex<State>,
    turn: Condvar,
}

/// The process's gate.
static GATE: Gate = Gate::new();

thread_local! {
    /// How many jobs this thread is inside: device threads that run one, and
    /// the thread that holds [`Steps`].
    static IN_STEP: Cell<u32> = const { Cell::new(0) };
    /// This thread changes page-locked memory now: what it frees and locks
    /// meanwhile is done at once.
    static CHANGING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is inside a job of the devices.
pub fn in_step() -> bool {
    IN_STEP.get() > 0
}

impl Gate {
    const fn new() -> Self {
        Self {
            state: Mutex::new(State { steps: 0, changing: false, waiting: 0, later: Vec::new() }),
            turn: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        self.turn.wait(guard).unwrap_or_else(|e| e.into_inner())
    }

    fn begin_steps(&self) {
        let mut state = self.state();
        while state.changing || state.waiting > 0 {
            state = self.wait(state);
        }
        state.steps += 1;
        drop(state);
        IN_STEP.set(IN_STEP.get() + 1);
    }

    fn end_steps(&self) {
        IN_STEP.set(IN_STEP.get().saturating_sub(1));
        let mut state = self.state();
        state.steps -= 1;
        if state.steps > 0 {
            return;
        }
        // The last job has ended: what was given back meanwhile goes now,
        // before the next job begins.
        if state.later.is_empty() || state.changing {
            drop(state);
            self.turn.notify_all();
            return;
        }
        state.changing = true;
        drop(state);
        self.run_changing(|| ());
    }

    /// Runs `f` as the one thread that changes page-locked memory (the caller
    /// set `changing`): what waits to be given back goes first, so that the
    /// memory a change takes finds room, and what `f` gives back goes at once.
    fn run_changing<T>(&self, f: impl FnOnce() -> T) -> T {
        let finish = Finish { gate: self, outer: CHANGING.replace(true) };
        let later = std::mem::take(&mut self.state().later);
        drop(later);
        let out = f();
        drop(finish);
        out
    }

    fn change<T>(&self, what: &'static str, f: impl FnOnce() -> T) -> Result<T> {
        if CHANGING.get() {
            return Ok(f());
        }
        if in_step() {
            return Err(Error::Invalid {
                what,
                detail: "page-locked memory cannot change inside a step of the devices: a \
                         kernel of the step may wait for this thread (see \"Host memory\" in \
                         docs/dev/architecture/host-memory.md)"
                    .into(),
            });
        }
        let mut state = self.state();
        state.waiting += 1;
        while state.steps > 0 || state.changing {
            state = self.wait(state);
        }
        state.waiting -= 1;
        state.changing = true;
        drop(state);
        Ok(self.run_changing(f))
    }

    fn try_change<T>(&self, f: impl FnOnce() -> T) -> Option<T> {
        if CHANGING.get() {
            return Some(f());
        }
        if in_step() {
            return None;
        }
        let mut state = self.state();
        if state.steps > 0 || state.changing || state.waiting > 0 {
            return None;
        }
        state.changing = true;
        drop(state);
        Some(self.run_changing(f))
    }

    fn settle(&self) -> bool {
        if CHANGING.get() || in_step() || self.state().later.is_empty() {
            return false;
        }
        self.change("memory given back during a job", || ()).is_ok()
    }

    fn retire(&self, item: impl Send + 'static) {
        if CHANGING.get() {
            drop(item);
            return;
        }
        let mut state = self.state();
        if in_step() || state.steps > 0 || state.changing {
            state.later.push(Box::new(item));
            return;
        }
        state.changing = true;
        drop(state);
        self.run_changing(|| drop(item));
    }
}

/// Ends a change: what was given back meanwhile goes too, then jobs may begin.
struct Finish<'a> {
    gate: &'a Gate,
    outer: bool,
}

impl Drop for Finish<'_> {
    fn drop(&mut self) {
        loop {
            let mut state = self.gate.state();
            if state.later.is_empty() {
                state.changing = false;
                drop(state);
                CHANGING.set(self.outer);
                self.gate.turn.notify_all();
                return;
            }
            let later = std::mem::take(&mut state.later);
            drop(state);
            drop(later);
        }
    }
}

/// The devices run a job while this is held: a step, or a transfer between
/// the devices and the host. Held by the thread that hands the job over, from
/// before the hand-over until the job has ended on every device.
#[must_use = "the job runs while the value is held"]
pub struct Steps(&'static Gate);

impl std::fmt::Debug for Steps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Steps")
    }
}

impl Steps {
    /// Begins a job: waits while page-locked memory changes or a thread waits
    /// to change it.
    pub fn begin() -> Self {
        Self::begin_on(&GATE)
    }

    fn begin_on(gate: &'static Gate) -> Self {
        gate.begin_steps();
        Self(gate)
    }
}

impl Drop for Steps {
    fn drop(&mut self) {
        self.0.end_steps();
    }
}

/// A device thread is inside a job while this is held: what it gives back
/// waits for the end of the job, and it cannot lock or allocate page-locked
/// memory.
#[must_use = "the thread is inside a job while the value is held"]
#[derive(Debug)]
pub struct InStep(());

impl InStep {
    pub fn mark() -> Self {
        IN_STEP.set(IN_STEP.get() + 1);
        Self(())
    }
}

impl Drop for InStep {
    fn drop(&mut self) {
        IN_STEP.set(IN_STEP.get().saturating_sub(1));
    }
}

/// Runs `f`, which locks, unlocks, allocates or frees page-locked memory (`what`
/// names it), once no job runs on the devices; no job begins until it returns.
/// Inside a job it refuses: the job would wait for its own end.
pub fn change<T>(what: &'static str, f: impl FnOnce() -> T) -> Result<T> {
    GATE.change(what, f)
}

/// Runs `f` as [`change`] does if no job runs now and no other thread changes
/// page-locked memory or waits to; `None` otherwise, without waiting. For a
/// thread that holds what a job may wait for (the expert store, say).
pub fn try_change<T>(f: impl FnOnce() -> T) -> Option<T> {
    GATE.try_change(f)
}

/// Gives `item` back, which frees or unlocks page-locked memory when it is
/// dropped: at once when no job runs, otherwise when the last job ends (or a
/// thread that changes page-locked memory now finishes). Never waits.
pub fn retire(item: impl Send + 'static) {
    GATE.retire(item)
}

/// Frees what was given back during a job and waits to go ([`retire`]): waits
/// until no job runs, as [`change`] does. Does nothing, and returns false,
/// inside a job or when nothing waits. For a thread that is about to
/// look at what is free, or found no room, and holds nothing a job waits for.
pub fn settle() -> bool {
    GATE.settle()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// Counts its drops.
    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn what_is_given_back_in_a_step_goes_when_the_last_step_ends() {
        static GATE: Gate = Gate::new();
        let dropped = Arc::new(AtomicUsize::new(0));
        let first = Steps::begin_on(&GATE);
        let second = Steps::begin_on(&GATE);
        GATE.retire(Counted(dropped.clone()));
        // A device thread inside the step, which holds no `Steps` itself.
        let device = std::thread::spawn({
            let dropped = dropped.clone();
            move || {
                let _inside = InStep::mark();
                GATE.retire(Counted(dropped));
                assert!(GATE.change("a test", || ()).is_err(), "no change inside a step");
                assert!(GATE.try_change(|| ()).is_none());
            }
        });
        device.join().unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(first);
        assert_eq!(dropped.load(Ordering::SeqCst), 0, "a step still runs");
        drop(second);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
        // Without a step it goes at once.
        GATE.retire(Counted(dropped.clone()));
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn settling_frees_what_a_step_gave_back_once_the_step_ends() {
        static GATE: Gate = Gate::new();
        let dropped = Arc::new(AtomicUsize::new(0));
        assert!(!GATE.settle(), "nothing waits");
        let step = Steps::begin_on(&GATE);
        GATE.retire(Counted(dropped.clone()));
        assert!(!GATE.settle(), "no settling inside a step");
        let settler = std::thread::spawn(|| GATE.settle());
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(dropped.load(Ordering::SeqCst), 0, "the step still runs");
        // The step's end frees it, or the settler right after.
        drop(step);
        settler.join().unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert!(!GATE.settle(), "nothing waits any more");
    }

    #[test]
    fn a_change_waits_for_the_step_and_the_next_step_waits_for_the_change() {
        static GATE: Gate = Gate::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        let step = Steps::begin_on(&GATE);
        assert!(GATE.try_change(|| ()).is_none(), "a step runs");
        let changer = std::thread::spawn({
            let order = order.clone();
            move || {
                GATE.change("a test", || {
                    order.lock().unwrap().push("change");
                    std::thread::sleep(Duration::from_millis(50));
                })
                .unwrap();
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        order.lock().unwrap().push("step ends");
        drop(step);
        // The changer waits or runs now; a new step begins after it.
        while order.lock().unwrap().len() < 2 {
            std::thread::yield_now();
        }
        let next = Steps::begin_on(&GATE);
        order.lock().unwrap().push("next step");
        drop(next);
        changer.join().unwrap();
        assert_eq!(*order.lock().unwrap(), ["step ends", "change", "next step"]);
        assert_eq!(GATE.try_change(|| 7), Some(7), "nothing runs");
    }

    #[test]
    fn what_a_change_gives_back_and_what_waited_go_within_it() {
        static GATE: Gate = Gate::new();
        let dropped = Arc::new(AtomicUsize::new(0));
        let step = Steps::begin_on(&GATE);
        GATE.retire(Counted(dropped.clone()));
        // The step ends with nothing left: the drop of the step frees it.
        drop(step);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        let inner = dropped.clone();
        GATE.change("a test", move || {
            // Nested calls run at once inside a change.
            GATE.retire(Counted(inner.clone()));
            assert_eq!(inner.load(Ordering::SeqCst), 2);
            assert_eq!(GATE.change("nested", || 1).unwrap(), 1);
        })
        .unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}
