//! The devices' memory and how the engine follows it, as it follows the
//! machine's RAM (`ram/machine.rs`).
//!
//! A load plans what the engine takes of a device against what is free then,
//! less a cap or a reserve (`ares_cuda::vram`). Another program on the card can
//! grow afterwards: a desktop compositor that drives a display, a browser, a
//! second engine. Device memory is not taken back from anyone by the driver,
//! so a program that finds the card full fails to draw or to compute. One
//! watch per process ([`VramWatch::system`]) looks at every device a loaded
//! model uses, through NVML, which counts what this process holds and what is
//! free without a context: every 50 ms, every 5 ms while memory is being given
//! back.
//!
//! - *The headroom* is what the engine leaves free: the reserve the device's
//!   state asks for now (a display, other processes that hold memory:
//!   `ares_cuda::vram::reserve`; none under a cap, where the operator decided),
//!   and what other programs can take while the engine gives way: their fastest
//!   growth seen times the time the engine takes to react. That is the RAM
//!   watch's rule ([`crate::Rule`]), with nothing assumed of the growth before
//!   one is seen and at most a sixteenth of the device above the reserve.
//! - *Giving back.* When less than the headroom is free, the elastic users of
//!   the device's memory together may hold that much less, at once: the slots
//!   of the experts of a mixture of experts ([`VramUser`]), between steps.
//!   Nothing else gives way: the weights, the K and V of sequences, the step
//!   buffers and the drafters are what the model runs on. When nothing more
//!   can go, the watch says so in the log, once for each time it happens, and
//!   the engine runs on.
//! - *Taking.* When more than the headroom and a band has been free for a
//!   while (3 s, doubled for every give that came soon after a take, up to 5
//!   minutes), the users may hold the surplus, never more than a cap leaves.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use ares_core::units::Bytes;
use ares_cuda::vram::{Display, RESERVE_SHARE, VramLimit, displays, reserve, vram_limit};
use ares_cuda::{DeviceInfo, PciAddress};

use crate::ram::{Follow, Rule, Verdict, share};

/// What an elastic user holds of one device's memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramHeld {
    /// The device, by the driver's ordinal.
    pub ordinal: u32,
    /// The bytes it holds there that can go, and the most it can hold.
    pub held: u64,
    pub most: u64,
    /// It holds more than it may and found nothing more to give.
    pub at_floor: bool,
}

/// A user of device memory that gives way when other programs need it: the
/// tier of experts in VRAM.
pub trait VramUser: Send + Sync {
    /// What it holds of each device it uses.
    fn vram(&self) -> Vec<VramHeld>;

    /// It may hold `bytes` of device `ordinal` from now on: it gives back
    /// what is above that between steps, and takes what it may when there is
    /// room.
    fn allow_vram(&self, ordinal: u32, bytes: u64);
}

/// The rule of the watch over a device of `total` bytes: the RAM watch's,
/// with no growth assumed until one is seen and the pieces of the tier.
fn rule(total: u64) -> Rule {
    Rule {
        reserve: 0,
        speed: 0,
        piece: crate::expert_store::VRAM_PIECE,
        most: total / RESERVE_SHARE,
        ..Rule::default()
    }
}

/// How long the displays DRM lists are taken as they were read.
const DISPLAYS_FOR: Duration = Duration::from_secs(10);

/// A device the watch looks at.
struct Watched {
    ordinal: u32,
    name: String,
    pci: PciAddress,
    total: u64,
    /// The loaded models that use it.
    users: usize,
    follow: Follow,
    /// What the elastic users may hold together.
    target: u64,
    displays: Option<(Instant, Vec<Display>)>,
    /// The log said that nothing more gives way, and nothing was taken since.
    short: bool,
}

/// What one look at a device found and decided, for logs and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramLook {
    pub ordinal: u32,
    pub free: u64,
    pub own: u64,
    pub headroom: u64,
    pub verdict: Verdict,
    /// What the elastic users hold, and may hold.
    pub held: u64,
    pub target: u64,
}

struct State {
    devices: Vec<Watched>,
    members: Vec<Weak<dyn VramUser>>,
    log: Option<Arc<dyn Fn(String) + Send + Sync>>,
    /// The log said that the devices are not watched.
    blind: bool,
}

/// The watch over the memory of the devices, for all engines of a process.
pub struct VramWatch {
    started: Instant,
    state: Mutex<State>,
    watching: AtomicBool,
}

impl std::fmt::Debug for VramWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state();
        f.debug_struct("VramWatch")
            .field("devices", &state.devices.iter().map(|d| d.ordinal).collect::<Vec<_>>())
            .field("members", &state.members.len())
            .finish()
    }
}

impl VramWatch {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            state: Mutex::new(State {
                devices: Vec::new(),
                members: Vec::new(),
                log: None,
                blind: false,
            }),
            watching: AtomicBool::new(false),
        }
    }

    /// The watch of this process, with the thread that looks.
    pub fn system() -> Arc<Self> {
        static SYSTEM: OnceLock<Arc<VramWatch>> = OnceLock::new();
        SYSTEM
            .get_or_init(|| {
                let watch = Arc::new(VramWatch::new());
                watch.watch();
                watch
            })
            .clone()
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts the thread that looks, once.
    fn watch(self: &Arc<Self>) {
        if self.watching.swap(true, Ordering::AcqRel) {
            return;
        }
        let weak = Arc::downgrade(self);
        let spawned = std::thread::Builder::new().name("ares-vram".into()).spawn(move || {
            let period = Rule::default().period;
            loop {
                let Some(watch) = weak.upgrade() else { return };
                let owed = watch.look().iter().any(|l| l.held > l.target);
                drop(watch);
                std::thread::sleep(if owed { period / 10 } else { period });
            }
        });
        if spawned.is_err() {
            self.watching.store(false, Ordering::Release);
        }
    }

    /// Where the watch writes what it saw: the server's log.
    pub fn on_log(&self, log: Arc<dyn Fn(String) + Send + Sync>) {
        self.state().log = Some(log);
    }

    fn say(state: &State, line: String) {
        if let Some(log) = &state.log {
            log(line);
        }
    }

    /// A loaded model uses the device of `info`: the watch looks at it until
    /// [`VramWatch::release`] says no model uses it any more. Without NVML
    /// the devices are not watched, and the log says so once.
    pub fn hold(&self, info: &DeviceInfo) {
        let mut state = self.state();
        if let Err(error) = ares_cuda::nvml_available() {
            if !std::mem::replace(&mut state.blind, true) {
                Self::say(
                    &state,
                    format!(
                        "the devices' memory is not watched while models run ({error}): when \
                         another program's VRAM grows, the expert slots do not give way"
                    ),
                );
            }
            return;
        }
        if let Some(known) = state.devices.iter_mut().find(|d| d.ordinal == info.ordinal) {
            known.users += 1;
            return;
        }
        state.devices.push(Watched {
            ordinal: info.ordinal,
            name: info.name.clone(),
            pci: info.pci,
            total: info.total_memory,
            users: 1,
            follow: Follow::new(rule(info.total_memory)),
            target: u64::MAX,
            displays: None,
            short: false,
        });
    }

    /// A model that used device `ordinal` is unloaded.
    pub fn release(&self, ordinal: u32) {
        let mut state = self.state();
        if let Some(known) = state.devices.iter_mut().find(|d| d.ordinal == ordinal) {
            known.users = known.users.saturating_sub(1);
        }
        state.devices.retain(|d| d.users > 0);
    }

    /// `member` gives way when the devices it uses need memory.
    pub fn admit(&self, member: Weak<dyn VramUser>) {
        self.state().members.push(member);
    }

    /// One look at every device: decides, divides among the users of each,
    /// and tells them what they may hold.
    pub fn look(&self) -> Vec<VramLook> {
        let now = self.started.elapsed();
        let mut allowed: Vec<(Arc<dyn VramUser>, u32, u64)> = Vec::new();
        let mut looks = Vec::new();
        {
            let mut state = self.state();
            let members: Vec<Arc<dyn VramUser>> =
                state.members.iter().filter_map(Weak::upgrade).collect();
            state.members.retain(|m| m.strong_count() > 0);
            let held: Vec<Vec<VramHeld>> = members.iter().map(|m| m.vram()).collect();
            let mut lines = Vec::new();
            for device in &mut state.devices {
                let Ok(seen) = ares_cuda::device_use(device.pci) else { continue };
                let limit = vram_limit(device.ordinal);
                // The reserve the device's state asks for now; none under a cap.
                let reserve_now = match limit {
                    Some(VramLimit::Cap(_)) => 0,
                    _ => {
                        let fresh = device
                            .displays
                            .as_ref()
                            .is_some_and(|(at, _)| at.elapsed() < DISPLAYS_FOR);
                        if !fresh {
                            device.displays = Some((Instant::now(), displays(device.pci)));
                        }
                        let shown = device.displays.as_ref().map_or(&[][..], |(_, d)| d.as_slice());
                        reserve(device.total, seen.display, shown, seen.others_bytes()).bytes
                    }
                };
                device.follow.set_reserve(reserve_now, reserve_now + device.total / RESERVE_SHARE);
                let verdict = device.follow.look(now, seen.free, seen.own);
                let headroom = device.follow.headroom();
                let band = device.follow.band();
                let on_device: Vec<(usize, VramHeld)> = held
                    .iter()
                    .enumerate()
                    .flat_map(|(m, list)| {
                        list.iter().filter(|h| h.ordinal == device.ordinal).map(move |h| (m, *h))
                    })
                    .collect();
                let held_here: u64 = on_device.iter().map(|(_, h)| h.held).sum();
                let spare = seen.free.saturating_sub(headroom);
                let mut target = match verdict {
                    Verdict::Give(missing) => held_here.saturating_sub(missing),
                    Verdict::Take(more) => held_here + more,
                    Verdict::Stay => device.target.min(held_here + spare),
                };
                // Under a cap the users never grow the process beyond it.
                if let Some(VramLimit::Cap(cap)) = limit {
                    target = target.min(held_here + cap.saturating_sub(seen.own));
                }
                device.target = target;
                if let Verdict::Give(missing) = verdict {
                    let nothing = on_device.iter().all(|(_, h)| h.held == 0 || h.at_floor);
                    if nothing && !std::mem::replace(&mut device.short, true) {
                        lines.push(format!(
                            "device {} ({}): other programs need {} more VRAM than is free ({} \
                             free, {} kept for them), and nothing more of the engine's gives way: \
                             the weights, the K and V of sequences, the step buffers and the \
                             drafters are what the model runs on{}",
                            device.ordinal,
                            device.name,
                            Bytes(missing),
                            Bytes(seen.free),
                            Bytes(headroom),
                            if on_device.is_empty() {
                                String::new()
                            } else {
                                format!(
                                    "; the expert slots hold {} at their floor",
                                    Bytes(held_here)
                                )
                            }
                        ));
                    }
                } else if seen.free >= headroom + band {
                    device.short = false;
                }
                let wants: Vec<u64> = on_device.iter().map(|(_, h)| h.most).collect();
                for ((m, _), bytes) in on_device.iter().zip(share(target, &wants)) {
                    allowed.push((members[*m].clone(), device.ordinal, bytes));
                }
                looks.push(VramLook {
                    ordinal: device.ordinal,
                    free: seen.free,
                    own: seen.own,
                    headroom,
                    verdict,
                    held: held_here,
                    target,
                });
            }
            for line in lines {
                Self::say(&state, line);
            }
        }
        // The users act outside the watch's lock: a give takes a step's time.
        for (member, ordinal, bytes) in allowed {
            member.allow_vram(ordinal, bytes);
        }
        looks
    }
}

#[cfg(test)]
mod tests {
    use ares_core::units::GIB;

    use super::*;

    #[test]
    fn the_rule_of_a_device_assumes_no_growth_until_one_is_seen() {
        let rule = rule(12 * GIB);
        assert_eq!((rule.reserve, rule.speed, rule.most), (0, 0, 768 << 20));
        let mut follow = Follow::new(rule);
        assert_eq!(follow.headroom(), 0);
        // A reserve of 768 MiB for a display; a program that takes a GiB in a
        // second adds what it takes while the engine reacts (50 + 100 ms).
        follow.set_reserve(768 << 20, (768 << 20) + (768 << 20));
        assert_eq!(follow.headroom(), 768 << 20);
        let ms = Duration::from_millis(1);
        follow.look(Duration::ZERO, 8 * GIB, 2 * GIB);
        follow.look(1000 * ms, 7 * GIB, 2 * GIB);
        let expected = (768 << 20) + (GIB as f64 * 0.15) as u64;
        assert!(follow.headroom().abs_diff(expected) < (1 << 20), "{}", follow.headroom());
        // What the engine itself takes is not growth of the others.
        let mut follow = Follow::new(rule);
        follow.look(Duration::ZERO, 8 * GIB, 2 * GIB);
        follow.look(1000 * ms, 7 * GIB, 3 * GIB);
        assert_eq!(follow.headroom(), 0);
    }
}
