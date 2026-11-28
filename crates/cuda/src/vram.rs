//! How much of a device's memory the engine may take: all it finds free, the
//! operator's cap, or what a reserve for other programs leaves.
//!
//! The server sets one [`VramLimit`] for every device it uses
//! ([`set_vram_limit`]) before a model is planned there, and every plan asks
//! [`crate::Device::room`] instead of the driver's free memory: the server's
//! choice of devices and shares, the placement of a mixture of experts, the
//! ledgers of the runtime, the probes of the weight layouts. A device without
//! a limit (tests and tools) gives all it has free.
//!
//! The reserve is decided from the device ([`reserve`]): a device that drives
//! a display, or on which other processes hold memory, keeps room for them to
//! grow into. See "VRAM" in `docs/dev/architecture/devices.md` for the rule
//! and why.

use std::path::Path;
use std::sync::Mutex;

use ares_core::units::Bytes;

use crate::device::PciAddress;
use crate::nvml::DeviceUse;

/// What the engine leaves on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VramLimit {
    /// The most the process may hold on the device, everything counted that
    /// the driver counts for it (its context and the kernels' code too):
    /// `--vram`. The operator decided what is left: no reserve is kept.
    Cap(u64),
    /// What the engine leaves free for other programs ([`reserve`]).
    Reserve(u64),
}

/// The limits of the devices, by the driver's ordinal.
static LIMITS: Mutex<Vec<(u32, VramLimit)>> = Mutex::new(Vec::new());

/// Sets what the engine leaves on device `ordinal` (`None`: nothing).
pub fn set_vram_limit(ordinal: u32, limit: Option<VramLimit>) {
    let mut limits = LIMITS.lock().unwrap_or_else(|e| e.into_inner());
    limits.retain(|(known, _)| *known != ordinal);
    if let Some(limit) = limit {
        limits.push((ordinal, limit));
    }
}

/// What the engine leaves on device `ordinal`.
pub fn vram_limit(ordinal: u32) -> Option<VramLimit> {
    let limits = LIMITS.lock().unwrap_or_else(|e| e.into_inner());
    limits.iter().find(|(known, _)| *known == ordinal).map(|(_, limit)| *limit)
}

/// What a plan may still take of a device that has `free` bytes free, under
/// `limit`, when the process holds `own` there (needed for a cap only).
pub fn room_under(free: u64, limit: Option<VramLimit>, own: Option<u64>) -> Option<u64> {
    match limit {
        None => Some(free),
        Some(VramLimit::Reserve(reserve)) => Some(free.saturating_sub(reserve)),
        Some(VramLimit::Cap(cap)) => Some(free.min(cap.saturating_sub(own?))),
    }
}

/// Screens' worth of memory a display is given: a compositor keeps three
/// frames of every display it drives (one shown, one drawn, one queued) and
/// every window draws into two or three buffers of its own, so 32 holds the
/// frames and windows that cover the screen ten times over: a browser, an
/// editor and a terminal, with a margin. Only one of the two terms of the
/// display's reserve ([`reserve`]).
pub const SCREENS_PER_DISPLAY: u64 = 32;

/// The share of a device's memory a display keeps at least, and the most
/// that other processes keep: a sixteenth, the share the watch over the
/// machine's RAM leaves at most (see "Host memory").
pub const RESERVE_SHARE: u64 = 16;

/// A display a device drives, as Linux's DRM reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Display {
    /// The connector (`DP-1`, `HDMI-A-2`).
    pub connector: String,
    /// The width and height of its preferred mode, when DRM lists one.
    pub size: Option<(u32, u32)>,
}

impl Display {
    /// The bytes of one frame of the display, four bytes a pixel.
    pub fn frame_bytes(&self) -> u64 {
        self.size.map_or(0, |(w, h)| u64::from(w) * u64::from(h) * 4)
    }
}

/// The displays connected to the device at `pci`, read from
/// `/sys/class/drm`. Empty where Linux has no DRM or the driver's module does
/// not register the device there.
pub fn displays(pci: PciAddress) -> Vec<Display> {
    displays_in(Path::new("/sys/class/drm"), pci)
}

/// [`displays`] in the tree at `root`.
pub fn displays_in(root: &Path, pci: PciAddress) -> Vec<Display> {
    let Ok(entries) = std::fs::read_dir(root) else { return Vec::new() };
    let address = pci.to_string();
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // A connector is `card<N>-<connector>`; its card is `card<N>`.
        let Some((card, connector)) = name.split_once('-') else { continue };
        if !card.strip_prefix("card").is_some_and(|n| n.chars().all(|c| c.is_ascii_digit())) {
            continue;
        }
        let device = std::fs::read_link(root.join(card).join("device")).ok();
        let on_device = device
            .as_ref()
            .and_then(|link| link.file_name())
            .is_some_and(|file| file.to_string_lossy().eq_ignore_ascii_case(&address));
        if !on_device {
            continue;
        }
        let read = |file: &str| std::fs::read_to_string(entry.path().join(file)).ok();
        if read("status").is_none_or(|status| status.trim() != "connected") {
            continue;
        }
        let size = read("modes").and_then(|modes| {
            let first = modes.lines().next()?;
            let (w, h) = first.split_once('x')?;
            let digits = |s: &str| s.chars().take_while(char::is_ascii_digit).collect::<String>();
            Some((digits(w).parse().ok()?, digits(h).parse().ok()?))
        });
        found.push(Display { connector: connector.to_owned(), size });
    }
    found.sort_by(|a, b| a.connector.cmp(&b.connector));
    found
}

/// What the reserve of a device is made of ([`reserve`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reserve {
    /// What a display keeps.
    pub display: u64,
    /// What the other processes that hold memory there keep.
    pub others: u64,
    /// The larger of the two: what the engine leaves free.
    pub bytes: u64,
    /// Why, for the log.
    pub why: String,
}

/// The reserve of a device of `total` bytes on which other processes hold
/// `others` bytes, which drives a display by NVML (`display`) or has
/// `displays` connected by DRM. The two needs are not added: a desktop's
/// programs are the other processes that hold memory.
///
/// - **A display**: a sixteenth of the device ([`RESERVE_SHARE`]), or
///   [`SCREENS_PER_DISPLAY`] frames of every display DRM lists a size for,
///   whichever is more. The sixteenth is what a device keeps whose displays
///   say nothing more; the frames are for large displays on small cards.
/// - **Other processes**: room to grow by what they hold, at most a
///   sixteenth of the device. A program that allocates on the device grows
///   by what it has, or stays.
pub fn reserve(total: u64, display: bool, displays: &[Display], others: u64) -> Reserve {
    let share = total / RESERVE_SHARE;
    let frames: u64 = displays.iter().map(Display::frame_bytes).sum::<u64>() * SCREENS_PER_DISPLAY;
    let drives = display || !displays.is_empty();
    let for_display = if drives { share.max(frames) } else { 0 };
    let for_others = others.min(share);
    let bytes = for_display.max(for_others);
    let mut reasons = Vec::new();
    if drives {
        let named: Vec<String> = displays
            .iter()
            .map(|d| match d.size {
                Some((w, h)) => format!("{} {w}x{h}", d.connector),
                None => d.connector.clone(),
            })
            .collect();
        reasons.push(format!(
            "it drives a display{}: {} ({})",
            if named.is_empty() { String::new() } else { format!(" ({})", named.join(", ")) },
            Bytes(for_display),
            if frames > share {
                format!("{SCREENS_PER_DISPLAY} frames of its displays")
            } else {
                format!("a {RESERVE_SHARE}th of its memory")
            }
        ));
    }
    if others > 0 {
        reasons.push(format!(
            "other processes hold {} there: {} for them to grow into ({})",
            Bytes(others),
            Bytes(for_others),
            if others > share {
                format!("a {RESERVE_SHARE}th of its memory")
            } else {
                "as much as they hold".to_owned()
            }
        ));
    }
    let why = if reasons.is_empty() {
        "no display and no other process on it".to_owned()
    } else {
        reasons.join("; ")
    };
    Reserve { display: for_display, others: for_others, bytes, why }
}

/// The reserve of the device at `pci` of `total` bytes as it is now, with
/// what NVML says of it (`None` when NVML cannot be used: then DRM alone tells
/// of a display, and other processes are not seen).
pub fn reserve_of(pci: PciAddress, total: u64) -> (Reserve, Option<DeviceUse>) {
    let seen = crate::nvml::device_use(pci).ok();
    let (display, others) = seen.as_ref().map_or((false, 0), |u| (u.display, u.others_bytes()));
    (reserve(total, display, &displays(pci), others), seen)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    fn display(connector: &str, size: Option<(u32, u32)>) -> Display {
        Display { connector: connector.into(), size }
    }

    #[test]
    fn a_display_keeps_a_sixteenth_or_its_frames_and_others_room_to_double() {
        // No display and nobody else: nothing is kept.
        let none = reserve(12 * GIB, false, &[], 0);
        assert_eq!((none.bytes, none.why.as_str()), (0, "no display and no other process on it"));
        // A display NVML sees, of a size DRM does not say: a sixteenth.
        assert_eq!(reserve(12 * GIB, true, &[], 0).bytes, 768 * MIB);
        assert_eq!(reserve(24 * GIB, true, &[], 0).bytes, 1536 * MIB);
        // 1080p on a 12 GiB card: 32 frames are 253 MiB, the sixteenth more.
        let full_hd = [display("DP-1", Some((1920, 1080)))];
        let r = reserve(12 * GIB, false, &full_hd, 0);
        assert_eq!(r.bytes, 768 * MIB);
        assert!(r.why.starts_with("it drives a display (DP-1 1920x1080): 768.0 MiB"), "{}", r.why);
        // 4K on an 8 GiB card: the frames are more.
        let uhd = [display("HDMI-A-1", Some((3840, 2160)))];
        let r = reserve(8 * GIB, true, &uhd, 0);
        assert_eq!(r.bytes, 3840 * 2160 * 4 * 32);
        assert!(r.why.contains("32 frames of its displays"), "{}", r.why);
        // Two displays add their frames.
        let two = [display("DP-1", Some((3840, 2160))), display("DP-2", Some((2560, 1440)))];
        assert_eq!(reserve(8 * GIB, false, &two, 0).display, (3840 * 2160 + 2560 * 1440) * 128);
        // Other processes: room to double, at most a sixteenth.
        assert_eq!(reserve(12 * GIB, false, &[], 300 * MIB).bytes, 300 * MIB);
        let r = reserve(12 * GIB, false, &[], 5 * GIB);
        assert_eq!(r.bytes, 768 * MIB);
        assert!(r.why.contains("other processes hold 5.00 GiB there"), "{}", r.why);
        // A desktop's programs are the others: the larger need, not the sum.
        let r = reserve(12 * GIB, true, &full_hd, 400 * MIB);
        assert_eq!((r.display, r.others, r.bytes), (768 * MIB, 400 * MIB, 768 * MIB));
    }

    #[test]
    fn the_room_is_what_is_free_less_the_reserve_or_what_the_cap_leaves() {
        assert_eq!(room_under(10 * GIB, None, None), Some(10 * GIB));
        assert_eq!(room_under(10 * GIB, Some(VramLimit::Reserve(GIB)), None), Some(9 * GIB));
        assert_eq!(room_under(MIB, Some(VramLimit::Reserve(GIB)), None), Some(0));
        // Under a cap of 8 GiB, a process that holds 3 GiB may take 5 more,
        // and no more than is free; without its own count the cap cannot hold.
        let cap = Some(VramLimit::Cap(8 * GIB));
        assert_eq!(room_under(10 * GIB, cap, Some(3 * GIB)), Some(5 * GIB));
        assert_eq!(room_under(4 * GIB, cap, Some(3 * GIB)), Some(4 * GIB));
        assert_eq!(room_under(10 * GIB, cap, Some(9 * GIB)), Some(0));
        assert_eq!(room_under(10 * GIB, cap, None), None);
    }

    #[test]
    fn limits_are_kept_per_device() {
        // Ordinals no other test uses.
        set_vram_limit(900, Some(VramLimit::Cap(GIB)));
        set_vram_limit(901, Some(VramLimit::Reserve(MIB)));
        assert_eq!(vram_limit(900), Some(VramLimit::Cap(GIB)));
        set_vram_limit(900, Some(VramLimit::Reserve(2 * MIB)));
        assert_eq!(vram_limit(900), Some(VramLimit::Reserve(2 * MIB)));
        assert_eq!(vram_limit(901), Some(VramLimit::Reserve(MIB)));
        set_vram_limit(900, None);
        assert_eq!(vram_limit(900), None);
        assert_eq!(vram_limit(902), None);
    }

    #[test]
    fn the_displays_of_a_device_are_its_connected_connectors() {
        let root = std::env::temp_dir().join(format!("ares-drm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let pci = |bus| PciAddress { domain: 0, bus, device: 0 };
        let devices = root.join("devices");
        for (card, bus) in [("card0", 1), ("card1", 6)] {
            let device = devices.join(pci(bus).to_string());
            std::fs::create_dir_all(&device).unwrap();
            std::fs::create_dir_all(root.join(card)).unwrap();
            std::os::unix::fs::symlink(&device, root.join(card).join("device")).unwrap();
        }
        let connector = |name: &str, status: &str, modes: &str| {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("status"), format!("{status}\n")).unwrap();
            std::fs::write(dir.join("modes"), modes).unwrap();
        };
        connector("card0-DP-1", "connected", "2560x1440\n1920x1080\n");
        connector("card0-DP-2", "disconnected", "");
        connector("card0-HDMI-A-1", "connected", "");
        connector("card1-DP-4", "connected", "1920x1080i\n");
        std::fs::create_dir_all(root.join("renderD128")).unwrap();
        std::fs::write(root.join("version"), "drm 1.1.0").unwrap();
        assert_eq!(
            displays_in(&root, pci(1)),
            [display("DP-1", Some((2560, 1440))), display("HDMI-A-1", None)]
        );
        assert_eq!(displays_in(&root, pci(6)), [display("DP-4", Some((1920, 1080)))]);
        assert_eq!(displays_in(&root, pci(9)), []);
        assert_eq!(displays_in(&root.join("nowhere"), pci(1)), []);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
