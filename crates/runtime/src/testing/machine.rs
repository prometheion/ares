//! A machine whose memory and clock a test sets, for books that follow it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::ram::{Machine, Rule};

/// The machine the books of a test follow ([`PinnedRam::following`]): what it
/// has available and its clock are the test's, and it is looked at when the
/// test says so, so "the machine has no room for checkpoints" is a line of a
/// test.
///
/// [`PinnedRam::following`]: crate::PinnedRam::following
#[derive(Debug)]
pub struct TestMachine {
    available: Arc<AtomicU64>,
    ms: Arc<AtomicU64>,
    machine: Arc<Machine>,
}

impl TestMachine {
    /// A machine with `available` bytes, followed by the default rule.
    pub fn new(available: u64) -> Self {
        Self::with_rule(available, Rule::default())
    }

    pub fn with_rule(available: u64, rule: Rule) -> Self {
        let (bytes, ms) = (Arc::new(AtomicU64::new(available)), Arc::new(AtomicU64::new(0)));
        let (source, clock) = (bytes.clone(), ms.clone());
        let machine = Machine::new(
            rule,
            move || Some(source.load(Ordering::SeqCst)),
            move || Duration::from_millis(clock.load(Ordering::SeqCst)),
        );
        Self { available: bytes, ms, machine }
    }

    /// The watch over it, for the books.
    pub fn machine(&self) -> &Arc<Machine> {
        &self.machine
    }

    /// The machine has `available` bytes from now on, 50 ms have passed, and
    /// it has been looked at: the books' allowances are what that leaves them.
    pub fn has(&self, available: u64) {
        self.available.store(available, Ordering::SeqCst);
        self.ms.fetch_add(50, Ordering::SeqCst);
        self.machine.look().expect("the machine says what it has");
    }
}
