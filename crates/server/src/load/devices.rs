//! The devices of a load: which the operator named or the driver reports,
//! what each is and has free, how they reach each other, and which one runs
//! the image tower. Facts the devices and Linux report; nothing is measured.

use ares_core::units::Bytes;
use ares_cuda::{Device, DeviceInfo, Wait};
use ares_model::qwen3vl;

use super::{LoadError, join, refuse};
use crate::plan::LoadOptions;

/// A device that was opened, what it had free then, and what a plan may take
/// of it ([`Device::room`]: the free memory less a reserve, or under a cap).
pub(crate) struct Held {
    pub(crate) device: Device,
    pub(crate) free: u64,
    pub(crate) room: u64,
    /// What the engine leaves on the device, for the log.
    pub(crate) limit: String,
}

/// Opens the devices the operator named, or all the driver reports, and says
/// in the log what each is. A device of a class this build does not run
/// ([`runnable`]) is skipped.
pub(crate) fn hold_devices(options: &LoadOptions) -> Result<(Vec<u32>, Vec<Held>), LoadError> {
    let known = ares_cuda::devices()?;
    if known.is_empty() {
        return Err(refuse("the driver reports no CUDA device"));
    }
    if let Ok((major, minor)) = ares_cuda::driver_version() {
        tracing::info!(
            "CUDA driver {major}.{minor}, {} device(s){}",
            known.len(),
            match std::env::var_os(ares_cuda::LIBRARY_ENV) {
                Some(path) =>
                    format!(", library {} ({})", path.to_string_lossy(), ares_cuda::LIBRARY_ENV),
                None => String::new(),
            }
        );
    }
    let ordinals: Vec<u32> = match &options.devices {
        Some(list) => {
            for (i, ordinal) in list.iter().enumerate() {
                if !known.iter().any(|info| info.ordinal == *ordinal) {
                    return Err(refuse(format!(
                        "--devices: there is no device {ordinal}; the driver reports {}",
                        join(known.iter().map(|info| info.ordinal), ", ")
                    )));
                }
                if list[..i].contains(ordinal) {
                    return Err(refuse(format!("--devices names device {ordinal} twice")));
                }
            }
            list.clone()
        }
        None => known.iter().map(|info| info.ordinal).collect(),
    };
    if ordinals.is_empty() {
        return Err(refuse("--devices names no device"));
    }
    let ordinals = runnable(&known, &ordinals)?;
    let mut held = Vec::new();
    for ordinal in &ordinals {
        let device = Device::open(*ordinal, Wait::Block)?;
        let limit = super::vram::limit(&device)?;
        let free = device.mem_info()?.free;
        let room = device.room()?;
        held.push(Held { device, free, room, limit });
        describe(&held[held.len() - 1]);
    }
    Ok((ordinals, held))
}

/// The devices of `ordinals` (all in `known`) this build runs
/// ([`DeviceInfo::check_class`]): CUDA runs only on the GPU classes it was tested
/// on. A device of another class is skipped with a log line while others remain;
/// when none remains, the load is refused with the reason of each.
fn runnable(known: &[DeviceInfo], ordinals: &[u32]) -> Result<Vec<u32>, LoadError> {
    let mut runs = Vec::new();
    let mut refused = Vec::new();
    for ordinal in ordinals {
        let info = known.iter().find(|info| info.ordinal == *ordinal).expect("a known device");
        match info.check_class() {
            Ok(()) => runs.push(*ordinal),
            Err(error) => refused.push(error.to_string()),
        }
    }
    if runs.is_empty() {
        return Err(refuse(refused.join("\n")));
    }
    for reason in &refused {
        tracing::warn!("skipped: {reason}");
    }
    Ok(runs)
}

/// The device the image tower runs on: in the RAM home, where every new image
/// uploads its weights, the one with the fastest link to the host, as Linux
/// reports it, and then the most multiprocessors; in the VRAM home the most
/// multiprocessors, then the link. Facts the devices report, no measurement:
/// which device computes a block faster is not measured at load.
pub(crate) fn tower_device(infos: &[&DeviceInfo], home: qwen3vl::Home) -> usize {
    let link = |info: &DeviceInfo| link_rate(info).unwrap_or(0.0);
    let key = |info: &DeviceInfo| match home {
        qwen3vl::Home::Ram => (link(info), f64::from(info.multiprocessors)),
        qwen3vl::Home::Vram => (f64::from(info.multiprocessors), link(info)),
    };
    (0..infos.len())
        .max_by(|&a, &b| {
            key(infos[a])
                .partial_cmp(&key(infos[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.cmp(&a))
        })
        .unwrap_or(0)
}

/// [`tower_device`] among `held`, for a tower whose weights stay in `home`.
pub(crate) fn tower_of(held: &[Held], home: qwen3vl::Home) -> usize {
    tower_device(&held.iter().map(|h| h.device.info()).collect::<Vec<_>>(), home)
}

/// The rate of a device's link to its slot, as Linux reports it: transfers per
/// second times lanes (only its ratio between devices is used).
fn link_rate(info: &DeviceInfo) -> Option<f64> {
    let base = format!("/sys/bus/pci/devices/{}", info.pci);
    let read = |name: &str| std::fs::read_to_string(format!("{base}/{name}")).ok();
    let speed: f64 = read("current_link_speed")?.split_whitespace().next()?.parse().ok()?;
    let width: f64 = read("current_link_width")?.trim().parse().ok()?;
    Some(speed * width)
}

/// The link of a device to its slot, as Linux reports it.
fn pci_link(info: &DeviceInfo) -> Option<String> {
    let base = format!("/sys/bus/pci/devices/{}", info.pci);
    let read = |name: &str| {
        std::fs::read_to_string(format!("{base}/{name}")).ok().map(|s| s.trim().to_owned())
    };
    Some(format!("{} x{}", read("current_link_speed")?, read("current_link_width")?))
}

/// The bound on a device's memory bandwidth that it reports, in bits per
/// second: only its ratio between devices is used.
pub(crate) fn bandwidth(info: &DeviceInfo) -> f64 {
    f64::from(info.memory_bus_width_bits) * f64::from(info.memory_clock_khz) * 1e3
}

pub(crate) fn describe(held: &Held) {
    let info = held.device.info();
    tracing::info!(
        "device {}: {}, compute capability {}.{}, {} of {} free, memory interface {} bit at {} \
         MHz, PCI {}{}; {}: a plan may take {}",
        info.ordinal,
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        Bytes(held.free),
        Bytes(info.total_memory),
        info.memory_bus_width_bits,
        info.memory_clock_khz / 1000,
        info.pci,
        pci_link(info).map(|link| format!(" ({link})")).unwrap_or_default(),
        held.limit,
        Bytes(held.room),
    );
}

/// Says how the devices of a split reach each other. This is the rule of
/// `ares_cuda::Mesh::new`: the direct route when every pair has peer access.
pub(crate) fn describe_links(held: &[Held]) -> Result<(), LoadError> {
    if held.len() < 2 {
        return Ok(());
    }
    let mut all = true;
    for a in held {
        for b in held {
            let (from, to) = (a.device.ordinal(), b.device.ordinal());
            if from != to {
                let (access, rank) = ares_cuda::peer_access(from, to)?;
                all &= access;
                tracing::info!(
                    "link {from} -> {to}: {}",
                    if access {
                        format!("peer access (the driver's rank of the link: {rank})")
                    } else {
                        "no peer access".to_owned()
                    }
                );
            }
        }
    }
    if all {
        tracing::info!("the devices exchange directly, from device to device");
    } else {
        tracing::warn!(
            "the devices exchange through page-locked host memory, which is slower for a tensor \
             split: {} names a driver library with peer access, if the system has one",
            ares_cuda::LIBRARY_ENV
        );
    }
    Ok(())
}

/// The ordinals of `held`, for the log.
pub(crate) fn ordinals(held: &[Held]) -> String {
    join(held.iter().map(|h| h.device.ordinal()), ", ")
}

/// What a plan may take of each of `held`, in their order.
pub(crate) fn room_of(held: &[Held]) -> Vec<u64> {
    held.iter().map(|h| h.room).collect()
}

/// The multiprocessors of each of `held`, in their order.
pub(crate) fn multiprocessors_of(held: &[Held]) -> Vec<u32> {
    held.iter().map(|h| h.device.info().multiprocessors).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device of a class this build does not run is skipped while another
    /// remains and refused when none does, for every device and for `--devices`.
    #[test]
    fn devices_of_a_class_this_build_does_not_run_are_skipped() {
        let card = |ordinal, compute_capability| DeviceInfo {
            ordinal,
            name: format!("Card {ordinal}"),
            uuid: [0; 16],
            compute_capability,
            total_memory: 12 << 30,
            pci: ares_cuda::PciAddress { domain: 0, bus: ordinal + 1, device: 0 },
            multiprocessors: 28,
            memory_bus_width_bits: 192,
            memory_clock_khz: 7_501_000,
            max_shared_per_block_optin: 101_376,
            l2_cache_bytes: 3 << 20,
        };
        // No build has code for 7.5 or 6.1: `build.rs` builds nothing below 8.0.
        let runs = ares_cuda::BUILT_CLASSES[0];
        let known = [card(0, (7, 5)), card(1, runs), card(2, (6, 1)), card(3, runs)];
        assert_eq!(runnable(&known, &[0, 1, 2, 3]).unwrap(), [1, 3]);
        assert_eq!(runnable(&known, &[2, 3]).unwrap(), [3]);
        assert_eq!(runnable(&known, &[1]).unwrap(), [1]);
        let refused = runnable(&known, &[0]).unwrap_err().to_string();
        assert!(refused.starts_with("device 0 (Card 0) has compute capability 7.5"), "{refused}");
        assert!(refused.contains("through the Vulkan backend"), "{refused}");
        let refused = runnable(&known, &[2, 0]).unwrap_err().to_string();
        let lines: Vec<&str> = refused.lines().collect();
        assert_eq!(lines.len(), 2, "{refused}");
        assert!(lines[0].starts_with("device 2 (Card 2) has compute capability 6.1"), "{refused}");
        assert!(lines[1].starts_with("device 0 (Card 0) has compute capability 7.5"), "{refused}");
    }
}
