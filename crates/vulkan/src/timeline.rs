//! A timeline semaphore that several devices share: the one thing Vulkan has
//! that orders work between logical devices on the devices themselves.
//!
//! The semaphore is made on the first device, exported as a file descriptor
//! (`VK_KHR_external_semaphore_fd`, `OPAQUE_FD`) and imported by the others;
//! afterwards every device has its own handle of one counter. A batch of one
//! device sets the counter to a value when it has finished, and a batch of
//! another waits until the counter has reached it. The host is not involved
//! once the batches are submitted.

use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

use ash::vk;

use crate::device::Device;
use crate::error::{Check, Error, Result};
use crate::stream::Stream;

struct Handle {
    device: Device,
    semaphore: vk::Semaphore,
}

/// A timeline counter that streams of several devices set and wait for.
///
/// Values only grow. A value is set by [`SharedTimeline::signal`] after the
/// work enqueued before it, and [`SharedTimeline::wait`] makes the work
/// enqueued after it wait for the value; which values are used for what is
/// up to the caller (the [`Mesh`](crate::Mesh) numbers its exchanges).
pub struct SharedTimeline {
    handles: Vec<Handle>,
}

impl std::fmt::Debug for SharedTimeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedTimeline")
            .field("devices", &self.handles.iter().map(|h| h.device.ordinal()).collect::<Vec<_>>())
            .finish()
    }
}

impl SharedTimeline {
    /// A counter at 0 shared by `devices`. Fails when a device cannot export
    /// or import timeline semaphores, or when the devices are not of one
    /// driver.
    pub fn new(devices: &[Device]) -> Result<Self> {
        let Some(first) = devices.first() else {
            return Err(Error::invalid("shared timeline", "no devices"));
        };
        for device in devices {
            let sharing = &device.physical().sharing;
            if !sharing.timeline_fd || device.0.semaphore_fd.is_none() {
                return Err(Error::Unsupported {
                    device: device.info().name.clone(),
                    reason: "it cannot share timeline semaphores (VK_KHR_external_semaphore_fd)"
                        .into(),
                });
            }
            if sharing.driver_uuid != first.physical().sharing.driver_uuid {
                return Err(Error::Unsupported {
                    device: device.info().name.clone(),
                    reason: "it is not of the same driver as the first device".into(),
                });
            }
        }
        let mut timeline = Self { handles: Vec::new() };
        // The exporting handle.
        let semaphore = create(first, true)?;
        timeline.handles.push(Handle { device: first.clone(), semaphore });
        for device in &devices[1..] {
            let fd = export(first, semaphore)?;
            let imported = create(device, false)?;
            // Pushed before the import, so that the handle is destroyed if it fails.
            timeline.handles.push(Handle { device: device.clone(), semaphore: imported });
            import(device, imported, fd)?;
        }
        Ok(timeline)
    }

    fn handle(&self, device: &Device) -> Result<&Handle> {
        self.handles.iter().find(|h| h.device.same_device(device)).ok_or_else(|| {
            Error::invalid(
                "shared timeline",
                "the stream is on a device the timeline is not shared with",
            )
        })
    }

    /// Enqueues on `stream`: the counter is set to `value` once everything
    /// enqueued on the stream before this call has finished. The stream's
    /// batch is submitted at once, so that a device that waits for the value
    /// can rely on the work being on its way.
    ///
    /// `value` must be greater than every value set before.
    pub fn signal(&self, stream: &Stream, value: u64) -> Result<()> {
        stream.signal_shared(self.handle(stream.device())?.semaphore, value)
    }

    /// Enqueues on `stream`: work enqueued after this call starts only when
    /// the counter has reached `value`.
    pub fn wait(&self, stream: &Stream, value: u64) -> Result<()> {
        stream.wait_shared(self.handle(stream.device())?.semaphore, value)
    }

    /// The counter as `device` sees it right now.
    pub fn value(&self, device: &Device) -> Result<u64> {
        device.timeline_value(self.handle(device)?.semaphore)
    }
}

impl Drop for SharedTimeline {
    fn drop(&mut self) {
        // Streams may hold waits and signals on the semaphores, recorded or
        // submitted. Every device submits what it has before any of them
        // waits (a device may wait for a value another one has yet to set).
        for handle in &self.handles {
            handle.device.flush_streams();
        }
        for handle in &self.handles {
            let _ = handle.device.synchronize();
        }
        for handle in &self.handles {
            // SAFETY: nothing on the device uses the semaphore any more; each
            // handle is destroyed once.
            unsafe { handle.device.raw().destroy_semaphore(handle.semaphore, None) };
        }
    }
}

fn create(device: &Device, exportable: bool) -> Result<vk::Semaphore> {
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let mut export_info = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    let mut info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    if exportable {
        info = info.push_next(&mut export_info);
    }
    // SAFETY: the timelineSemaphore feature is enabled on the device and
    // `info` describes a timeline semaphore.
    unsafe { device.raw().create_semaphore(&info, None) }.check("vkCreateSemaphore")
}

fn export(device: &Device, semaphore: vk::Semaphore) -> Result<OwnedFd> {
    let loader = device.0.semaphore_fd.as_ref().expect("checked by new");
    let info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    // SAFETY: the semaphore was created exportable as this handle type.
    let fd = unsafe { loader.get_semaphore_fd(&info) }.check("vkGetSemaphoreFdKHR")?;
    // SAFETY: a new descriptor that nothing else owns.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn import(device: &Device, semaphore: vk::Semaphore, fd: OwnedFd) -> Result<()> {
    let loader = device.0.semaphore_fd.as_ref().expect("checked by new");
    let info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
        .fd(fd.as_raw_fd());
    // SAFETY: the semaphore is a timeline semaphore of this device that no
    // batch uses yet; the descriptor is an exported semaphore.
    unsafe { loader.import_semaphore_fd(&info) }.check("vkImportSemaphoreFdKHR")?;
    // The driver owns the descriptor now.
    let _ = fd.into_raw_fd();
    Ok(())
}
