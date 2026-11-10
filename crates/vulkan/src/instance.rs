//! The Vulkan instance, and what the loader reports about the physical
//! devices.
//!
//! The instance is created once per process and never destroyed, like the
//! driver of the CUDA layer: it holds no device memory, and every logical
//! device opened later borrows it for the rest of the process.

use std::sync::OnceLock;

use ash::{Entry, vk};

use crate::device::{DeviceInfo, DeviceType, PciAddress};
use crate::error::{Check, Error, Result};

/// The Vulkan version this crate is written against. Buffer device addresses,
/// timeline semaphores and subgroup operations are core in it, so a device
/// that reports at least this version needs none of their extensions.
pub(crate) const REQUIRED_API: u32 = vk::API_VERSION_1_2;

/// Push constants are how a kernel receives its addresses and sizes. 128 bytes
/// is the most every Vulkan device guarantees.
pub(crate) const PUSH_CONSTANT_BYTES: u32 = 128;

/// The least subgroup width the kernels allow: a workgroup of 256 invocations
/// then has at most 32 subgroups, which is what their reductions are sized for.
pub(crate) const MIN_SUBGROUP_SIZE: u32 = 8;

pub(crate) struct Instance {
    /// Keeps the loader library mapped for as long as the instance lives.
    _entry: Entry,
    pub(crate) raw: ash::Instance,
    /// Every non-software device, in the loader's order.
    pub(crate) physical: Vec<Physical>,
}

/// A physical device and the facts about it that opening and using it need.
pub(crate) struct Physical {
    pub(crate) raw: vk::PhysicalDevice,
    pub(crate) info: DeviceInfo,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    /// The heap that `info.total_memory` is the size of, and that `mem_info`
    /// reports on.
    pub(crate) device_heap: usize,
    /// The queue family the engine submits to; `u32::MAX` when there is none.
    pub(crate) queue_family: u32,
    pub(crate) timestamp_valid_bits: u32,
    /// Nanoseconds per timestamp tick.
    pub(crate) timestamp_period: f32,
    pub(crate) max_work_group_count: [u32; 3],
    /// Whether `VK_EXT_memory_budget` is available (it is then enabled when
    /// the device is opened).
    pub(crate) memory_budget: bool,
    /// Why the engine cannot use this device, if it cannot.
    pub(crate) unusable: Option<String>,
}

static INSTANCE: OnceLock<std::result::Result<Instance, Error>> = OnceLock::new();

/// The process-wide instance, created on first use. Every later call returns
/// the same outcome, including the same error.
pub(crate) fn instance() -> Result<&'static Instance> {
    INSTANCE.get_or_init(Instance::create).as_ref().map_err(Clone::clone)
}

impl Instance {
    fn create() -> Result<Self> {
        // SAFETY: loading the loader runs its initialisers, which have no
        // preconditions on the calling process.
        let entry = unsafe { Entry::load() }.map_err(|e| Error::LibraryNotFound(e.to_string()))?;
        // SAFETY: the entry points were loaded from the loader just above.
        let version = unsafe { entry.try_enumerate_instance_version() }
            .check("vkEnumerateInstanceVersion")?
            .unwrap_or(vk::API_VERSION_1_0);
        if version < REQUIRED_API {
            return Err(Error::Unsupported {
                device: "the Vulkan loader".into(),
                reason: format!("it implements Vulkan {}, 1.2 is needed", version_string(version)),
            });
        }
        let app = vk::ApplicationInfo::default()
            .application_name(c"ares")
            .engine_name(c"ares")
            .api_version(REQUIRED_API);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `info` and the application info it points to are alive for
        // the call; no layers or extensions are requested.
        let raw = unsafe { entry.create_instance(&info, None) }.check("vkCreateInstance")?;
        // SAFETY: the instance was created just above.
        let handles =
            unsafe { raw.enumerate_physical_devices() }.check("vkEnumeratePhysicalDevices")?;
        let mut physical: Vec<Physical> =
            handles.into_iter().filter_map(|handle| describe(&raw, handle)).collect();
        for (ordinal, device) in physical.iter_mut().enumerate() {
            device.info.ordinal = ordinal as u32;
        }
        Ok(Self { _entry: entry, raw, physical })
    }
}

/// `(major, minor, patch)` of a packed Vulkan version.
pub(crate) fn unpack_version(version: u32) -> (u32, u32, u32) {
    (vk::api_version_major(version), vk::api_version_minor(version), vk::api_version_patch(version))
}

fn version_string(version: u32) -> String {
    let (major, minor, patch) = unpack_version(version);
    format!("{major}.{minor}.{patch}")
}

/// Reads everything about a physical device that does not need a logical one.
/// Returns `None` for processor-based implementations (llvmpipe and the like),
/// which the engine never uses.
fn describe(instance: &ash::Instance, raw: vk::PhysicalDevice) -> Option<Physical> {
    // SAFETY: `raw` was enumerated from this live instance; the queries only
    // write the structures passed to them.
    let props = unsafe { instance.get_physical_device_properties(raw) };
    let device_type = match props.device_type {
        vk::PhysicalDeviceType::CPU => return None,
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::Virtual,
        _ => DeviceType::Other,
    };
    let name = props
        .device_name_as_c_str()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();

    // SAFETY: as above.
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(raw) }.unwrap_or_default();
    let has_extension = |wanted: &std::ffi::CStr| {
        extensions.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|name| name == wanted))
    };
    let has_pci = has_extension(ash::ext::pci_bus_info::NAME);
    let memory_budget = has_extension(ash::ext::memory_budget::NAME);

    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    {
        let mut props2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut id).push_next(&mut subgroup);
        if has_pci {
            props2 = props2.push_next(&mut pci);
        }
        // SAFETY: the chain holds only structures of properties this device
        // reports (the PCI one only when its extension is listed).
        unsafe { instance.get_physical_device_properties2(raw, &mut props2) };
    }

    // SAFETY: as above.
    let memory = unsafe { instance.get_physical_device_memory_properties(raw) };
    // The largest device-local heap is the video memory. A second, small one
    // (a window of the first that the CPU can map) is not additional memory.
    let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
    let device_heap = heaps
        .iter()
        .enumerate()
        .filter(|(_, heap)| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .max_by_key(|(_, heap)| heap.size)
        .map(|(index, _)| index);

    // SAFETY: as above.
    let families = unsafe { instance.get_physical_device_queue_family_properties(raw) };
    // A family that can compute but not draw is kept clear of the desktop's
    // graphics work, which is where the engine's long kernels belong.
    let compute = |f: &vk::QueueFamilyProperties| f.queue_flags.contains(vk::QueueFlags::COMPUTE);
    let family = families
        .iter()
        .position(|f| compute(f) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .or_else(|| families.iter().position(compute));

    let mut unusable = None;
    let mut fail = |reason: String| {
        unusable.get_or_insert(reason);
    };
    if props.api_version < REQUIRED_API {
        fail(format!("it implements Vulkan {}, 1.2 is needed", version_string(props.api_version)));
    } else {
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
        // SAFETY: the device reports Vulkan 1.2, which defines the structure.
        unsafe { instance.get_physical_device_features2(raw, &mut features) };
        let int64 = features.features.shader_int64 == vk::TRUE;
        if v12.buffer_device_address != vk::TRUE {
            fail("it lacks bufferDeviceAddress".into());
        }
        if v12.timeline_semaphore != vk::TRUE {
            fail("it lacks timelineSemaphore".into());
        }
        if !int64 {
            fail("it lacks shaderInt64".into());
        }
    }
    if family.is_none() {
        fail("it has no queue family that can compute".into());
    }
    if device_heap.is_none() {
        fail("it reports no device-local memory".into());
    }
    if props.limits.max_push_constants_size < PUSH_CONSTANT_BYTES {
        fail(format!(
            "it allows {} bytes of push constants, {PUSH_CONSTANT_BYTES} are needed",
            props.limits.max_push_constants_size
        ));
    }
    if !subgroup.supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC)
        || !subgroup.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
    {
        fail("compute shaders cannot use subgroup arithmetic".into());
    }
    if subgroup.subgroup_size < MIN_SUBGROUP_SIZE {
        fail(format!(
            "its subgroups have {} invocations, at least {MIN_SUBGROUP_SIZE} are needed",
            subgroup.subgroup_size
        ));
    }

    let device_heap = device_heap.unwrap_or(0);
    let family_index = family.unwrap_or(0);
    let info = DeviceInfo {
        ordinal: 0,
        name,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        device_type,
        api_version: unpack_version(props.api_version),
        driver_version: props.driver_version,
        total_memory: heaps.get(device_heap).map_or(0, |heap| heap.size),
        pci: has_pci.then_some(PciAddress {
            domain: pci.pci_domain,
            bus: pci.pci_bus,
            device: pci.pci_device,
            function: pci.pci_function,
        }),
        uuid: id.device_uuid,
        subgroup_size: subgroup.subgroup_size,
    };
    Some(Physical {
        raw,
        info,
        memory,
        device_heap,
        queue_family: family.map_or(u32::MAX, |f| f as u32),
        timestamp_valid_bits: families.get(family_index).map_or(0, |f| f.timestamp_valid_bits),
        timestamp_period: props.limits.timestamp_period,
        max_work_group_count: props.limits.max_compute_work_group_count,
        memory_budget,
        unusable,
    })
}
