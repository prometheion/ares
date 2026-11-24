//! Threads that open their parts at the same time and wait for each other at
//! a few points (`parts.rs`).

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ares_runtime::ModelError;

use super::super::invalid;

/// How long a part waits for the others to arrive while they open: the
/// longest thing between two meetings is the upload of a device's weights
/// from a cold drive.
const MEET_LIMIT: Duration = Duration::from_secs(1800);

/// Threads that build their parts at the same time and wait for each other at
/// a few points: the plan of the expert store needs every device's room, the
/// store needs every device's slots, and a part needs to know where its peer
/// receives.
pub(super) struct Meeting<T> {
    state: Mutex<MeetState<T>>,
    changed: Condvar,
}

pub(super) struct MeetState<T> {
    pub(super) value: T,
    arrived: usize,
    generation: u64,
    failed: Option<String>,
}

impl<T> Meeting<T> {
    pub(super) fn new(value: T) -> Self {
        Self {
            state: Mutex::new(MeetState { value, arrived: 0, generation: 0, failed: None }),
            changed: Condvar::new(),
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, MeetState<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Lets every thread that waits give up: one of the parts failed.
    pub(super) fn fail(&self, message: String) {
        let mut state = self.lock();
        state.failed.get_or_insert(message);
        self.changed.notify_all();
    }

    /// `add` runs on the shared value for each of `n` threads as it arrives;
    /// the last of them also runs `last`. Returns when all have arrived and
    /// `last` has run, or with an error when any of them failed.
    pub(super) fn meet(
        &self,
        n: usize,
        add: impl FnOnce(&mut T) -> Result<(), ModelError>,
        last: impl FnOnce(&mut T) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let gone = |state: &MeetState<T>| {
            state.failed.as_ref().map(|why| invalid(format!("another part failed: {why}")))
        };
        let mut state = self.lock();
        if let Some(error) = gone(&state) {
            return Err(error);
        }
        let mut outcome = add(&mut state.value);
        if outcome.is_ok() {
            state.arrived += 1;
            if state.arrived == n {
                outcome = last(&mut state.value);
                state.arrived = 0;
                state.generation += 1;
            }
        }
        if let Err(error) = outcome {
            state.failed.get_or_insert(error.to_string());
            self.changed.notify_all();
            return Err(error);
        }
        if state.arrived == 0 {
            self.changed.notify_all();
            return Ok(());
        }
        let generation = state.generation;
        let started = Instant::now();
        while state.generation == generation {
            if let Some(error) = gone(&state) {
                return Err(error);
            }
            let left = MEET_LIMIT.saturating_sub(started.elapsed());
            if left.is_zero() {
                return Err(invalid("the other parts of the model did not arrive"));
            }
            state = self.changed.wait_timeout(state, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        // The last to arrive may have failed in what it made of the others'.
        gone(&state).map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn parts_that_meet_see_what_the_last_to_arrive_made_of_what_they_added() {
        let meeting = Meeting::new(Vec::<usize>::new());
        let made = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for part in 0..3 {
                let (meeting, made) = (&meeting, &made);
                scope.spawn(move || {
                    meeting
                        .meet(
                            3,
                            |added| {
                                added.push(part);
                                Ok(())
                            },
                            |added| {
                                // Once, and with every part's contribution.
                                made.fetch_add(1, Ordering::SeqCst);
                                added.sort_unstable();
                                assert_eq!(added, &[0, 1, 2]);
                                Ok(())
                            },
                        )
                        .unwrap();
                    // Returned: the last has run.
                    assert_eq!(made.load(Ordering::SeqCst), 1);
                });
            }
        });
        // A second meeting of the same parts works as well.
        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| meeting.meet(3, |_| Ok(()), |_| Ok(())).unwrap());
            }
        });
    }

    #[test]
    fn a_part_that_fails_lets_the_others_give_up_instead_of_waiting() {
        let meeting = Meeting::new(());
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| meeting.meet(2, |_| Ok(()), |_| Ok(())));
            // The other part fails before it arrives.
            meeting.fail("no memory for the weights".into());
            let error = waiting.join().unwrap().unwrap_err().to_string();
            assert!(error.contains("no memory for the weights"), "{error}");
        });
        // And a part that arrives later does not wait either.
        assert!(meeting.meet(2, |_| Ok(()), |_| Ok(())).is_err());
    }

    #[test]
    fn a_failure_in_what_the_last_makes_reaches_everyone() {
        let meeting = Meeting::new(());
        let errors: Vec<bool> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        meeting
                            .meet(2, |_| Ok(()), |_| Err(invalid("the store does not fit")))
                            .is_err()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(errors, [true, true]);
    }
}
