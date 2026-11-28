//! The VRAM the engine leaves on each device: the operator's cap
//! (`--vram`), or a reserve for other programs where the device shows the
//! need, and the line of the log that says what a plan may take there.
//!
//! The cap is the server's, for all loaded models together, and it counts
//! everything the driver counts for the process on the device (NVML). It is
//! checked once when the server starts and set on every device a load opens
//! (`ares_cuda::vram`), before anything is planned there. A device without a
//! cap gets the reserve its state asks for at that moment
//! (`ares_cuda::vram::reserve`).

use std::sync::Mutex;

use ares_core::units::Bytes;
use ares_cuda::Device;
use ares_cuda::vram::{VramLimit, set_vram_limit};

use crate::plan::VramCap;

/// The cap the server was started with.
static CAP: Mutex<Option<VramCap>> = Mutex::new(None);

/// Sets the server's cap (`None`: none), after checking it against the
/// devices the driver reports and `used`, the devices the models may use
/// (`None`: all). A cap needs NVML, which counts what the process holds on a
/// device. `Err` says why the cap cannot be.
pub fn set_vram_cap(cap: Option<VramCap>, used: Option<&[u32]>) -> Result<(), String> {
    if let Some(cap) = &cap {
        ares_cuda::nvml_available().map_err(|e| {
            format!(
                "--vram needs NVML (libnvidia-ml.so.1, which comes with the NVIDIA driver) to \
                 count what the engine holds on a device: {e}"
            )
        })?;
        let devices = ares_cuda::devices().map_err(|e| format!("--vram: {e}"))?;
        // What a device has, the driver's own reserve included: the size the
        // card is sold with, which is what an operator names.
        let sizes: Vec<(u32, u64)> = devices
            .iter()
            .map(|info| {
                let total = ares_cuda::device_use(info.pci).map_or(info.total_memory, |u| u.total);
                (info.ordinal, total)
            })
            .collect();
        cap.check(&sizes, used).map_err(|why| format!("--vram: {why}"))?;
    }
    *CAP.lock().unwrap_or_else(|e| e.into_inner()) = cap;
    Ok(())
}

/// The cap of device `ordinal`.
fn cap_of(ordinal: u32) -> Option<u64> {
    CAP.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|cap| cap.of(ordinal))
}

/// Sets what the engine leaves on `device`, its cap or the reserve its state
/// asks for now, and returns the words of the log for it.
pub(crate) fn limit(device: &Device) -> Result<String, ares_cuda::Error> {
    let info = device.info();
    let ordinal = info.ordinal;
    match cap_of(ordinal) {
        Some(cap) => {
            set_vram_limit(ordinal, Some(VramLimit::Cap(cap)));
            let own = ares_cuda::device_use(info.pci)?.own;
            Ok(format!(
                "a cap of {} (--vram), of which the engine holds {} there already",
                Bytes(cap),
                Bytes(own),
            ))
        }
        None => {
            let (reserve, _) = ares_cuda::vram::reserve_of(info.pci, info.total_memory);
            set_vram_limit(ordinal, Some(VramLimit::Reserve(reserve.bytes)));
            Ok(match reserve.bytes {
                0 => format!("no reserve: {}", reserve.why),
                bytes => {
                    format!("a reserve of {} for other programs: {}", Bytes(bytes), reserve.why)
                }
            })
        }
    }
}
