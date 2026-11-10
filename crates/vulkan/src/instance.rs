//! The Vulkan instance, and what the loader reports about the physical
//! devices.
//!
//! The instance is created once per process and never destroyed, like the
//! driver of the CUDA layer: it holds no device memory, and every logical
//! device opened later borrows it for the rest of the process.

use std::sync::OnceLock;

use ash::{Entry, vk};

use crate::device::{ComponentType, CooperativeMatrix, DeviceInfo, DeviceType, PciAddress};
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
    /// Why the quantized weight kernels (`mmvq_*`, `quantize_act`) cannot run
    /// here, if they cannot. They need integer dot products, 8- and 16-bit
    /// buffer access and clustered subgroup operations; every desktop GPU of the
    /// last few years has them, so this is `None` there.
    pub(crate) quantized_unsupported: Option<String>,
    /// Every device extension the driver lists.
    pub(crate) extensions: Vec<String>,
    /// What the device can share with other devices and with host memory.
    pub(crate) sharing: Sharing,
}

/// What a device can export and import, as the loader reports it (reading it
/// opens nothing). That a handle type is listed does not mean data crosses:
/// the mesh checks that with a real exchange when it is made.
#[derive(Clone, Debug, Default)]
pub(crate) struct Sharing {
    /// Buffers of device memory can be exported to and imported from a file
    /// descriptor of the driver's own kind (`OPAQUE_FD`).
    pub(crate) opaque_fd: bool,
    /// The same for `DMA_BUF`.
    pub(crate) dma_buf: bool,
    /// Page-aligned host memory can be imported as a buffer
    /// (`VK_EXT_external_memory_host`).
    pub(crate) host_pointer: bool,
    /// The alignment host memory must have to be imported.
    pub(crate) host_alignment: u64,
    /// A timeline semaphore can be exported to and imported from a file
    /// descriptor (`OPAQUE_FD`).
    pub(crate) timeline_fd: bool,
    /// `VkPhysicalDeviceIDProperties::driverUUID`: memory and semaphores can only
    /// be shared between devices of one driver.
    pub(crate) driver_uuid: [u8; 16],
}

/// The usage every buffer of this crate is made with.
pub(crate) const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

/// Device extensions that only share memory and semaphores. They are enabled
/// whenever the driver lists them: they change nothing for a device that
/// shares nothing.
pub(crate) const SHARING_EXTENSIONS: &[&std::ffi::CStr] = &[
    ash::khr::external_memory::NAME,
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    ash::ext::external_memory_host::NAME,
    ash::khr::external_semaphore::NAME,
    ash::khr::external_semaphore_fd::NAME,
];

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
            handles.into_iter().filter_map(|handle| describe(&entry, &raw, handle)).collect();
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
fn describe(entry: &Entry, instance: &ash::Instance, raw: vk::PhysicalDevice) -> Option<Physical> {
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

    let has_dot_product = has_extension(ash::khr::shader_integer_dot_product::NAME);
    let has_coop = has_extension(ash::khr::cooperative_matrix::NAME);
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut dot = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
    let has_host_memory = has_extension(ash::ext::external_memory_host::NAME);
    let mut host_properties = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    {
        let mut props2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut id).push_next(&mut subgroup);
        if has_host_memory {
            props2 = props2.push_next(&mut host_properties);
        }
        if has_pci {
            props2 = props2.push_next(&mut pci);
        }
        if has_dot_product {
            props2 = props2.push_next(&mut dot);
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
    let mut coop_supported = false;
    let mut quantized_unsupported = None;
    let mut no_quantized = |reason: String| {
        quantized_unsupported.get_or_insert(reason);
    };
    if props.api_version < REQUIRED_API {
        fail(format!("it implements Vulkan {}, 1.2 is needed", version_string(props.api_version)));
        no_quantized("the device is not usable at all".into());
    } else {
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut dot_features = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut coop_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut v11).push_next(&mut v12);
        if has_dot_product {
            features = features.push_next(&mut dot_features);
        }
        if has_coop {
            features = features.push_next(&mut coop_features);
        }
        // SAFETY: the device reports Vulkan 1.2, which defines the structures
        // (the dot product one only when its extension is listed).
        unsafe { instance.get_physical_device_features2(raw, &mut features) };
        let int64 = features.features.shader_int64 == vk::TRUE;
        let int16 = features.features.shader_int16 == vk::TRUE;
        coop_supported = has_coop && coop_features.cooperative_matrix == vk::TRUE;
        if !has_dot_product || dot_features.shader_integer_dot_product != vk::TRUE {
            no_quantized(
                "it lacks shaderIntegerDotProduct (VK_KHR_shader_integer_dot_product)".into(),
            );
        }
        if v12.vulkan_memory_model != vk::TRUE {
            no_quantized("it lacks vulkanMemoryModel".into());
        }
        if v12.shader_int8 != vk::TRUE || v12.storage_buffer8_bit_access != vk::TRUE {
            no_quantized("it lacks shaderInt8 or storageBuffer8BitAccess".into());
        }
        if !int16 || v11.storage_buffer16_bit_access != vk::TRUE {
            no_quantized("it lacks shaderInt16 or storageBuffer16BitAccess".into());
        }
        if !subgroup.supported_operations.contains(vk::SubgroupFeatureFlags::CLUSTERED)
            || subgroup.subgroup_size < 32
            || subgroup.subgroup_size % 32 != 0
        {
            no_quantized(format!(
                "the weight kernels need clustered subgroup operations and subgroups of 32 or 64 \
                 invocations (this device: {})",
                subgroup.subgroup_size
            ));
        }
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

    // The shapes of matrix the tensor cores multiply for a whole subgroup, if the device
    // exposes them through VK_KHR_cooperative_matrix.
    let cooperative_matrix = if coop_supported {
        let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        // SAFETY: the extension is supported and `raw` belongs to the instance.
        unsafe { loader.get_physical_device_cooperative_matrix_properties(raw) }
            .unwrap_or_default()
            .iter()
            .filter(|p| p.scope == vk::ScopeKHR::SUBGROUP)
            .map(|p| CooperativeMatrix {
                m: p.m_size,
                n: p.n_size,
                k: p.k_size,
                a: ComponentType::from_raw(p.a_type),
                b: ComponentType::from_raw(p.b_type),
                c: ComponentType::from_raw(p.c_type),
                result: ComponentType::from_raw(p.result_type),
                saturating: p.saturating_accumulation == vk::TRUE,
            })
            .collect()
    } else {
        Vec::new()
    };
    let sharing = describe_sharing(
        instance,
        raw,
        &has_extension,
        id.driver_uuid,
        has_host_memory.then_some(host_properties.min_imported_host_pointer_alignment),
    );
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
        cooperative_matrix,
        integer_dot_product: has_dot_product
            && dot.integer_dot_product4x8_bit_packed_mixed_signedness_accelerated == vk::TRUE
            && dot.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE,
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
        quantized_unsupported,
        extensions: extensions
            .iter()
            .filter_map(|e| e.extension_name_as_c_str().ok())
            .map(|name| name.to_string_lossy().into_owned())
            .collect(),
        sharing,
    })
}

/// What the device can export and import, read from the loader's capability
/// queries.
fn describe_sharing(
    instance: &ash::Instance,
    raw: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
    driver_uuid: [u8; 16],
    host_alignment: Option<u64>,
) -> Sharing {
    let buffer = |handle: vk::ExternalMemoryHandleTypeFlags,
                  wanted: vk::ExternalMemoryFeatureFlags| {
        let info =
            vk::PhysicalDeviceExternalBufferInfo::default().usage(BUFFER_USAGE).handle_type(handle);
        let mut props = vk::ExternalBufferProperties::default();
        // SAFETY: `raw` belongs to the instance; the query only writes `props`.
        unsafe { instance.get_physical_device_external_buffer_properties(raw, &info, &mut props) };
        props.external_memory_properties.external_memory_features.contains(wanted)
    };
    let both =
        vk::ExternalMemoryFeatureFlags::EXPORTABLE | vk::ExternalMemoryFeatureFlags::IMPORTABLE;
    let fd = has_extension(ash::khr::external_memory_fd::NAME);
    let mut timeline =
        vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let semaphore_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
        .push_next(&mut timeline);
    let mut semaphore = vk::ExternalSemaphoreProperties::default();
    // SAFETY: as above.
    unsafe {
        instance.get_physical_device_external_semaphore_properties(
            raw,
            &semaphore_info,
            &mut semaphore,
        )
    };
    Sharing {
        opaque_fd: fd && buffer(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD, both),
        dma_buf: fd
            && has_extension(ash::ext::external_memory_dma_buf::NAME)
            && buffer(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT, both),
        host_pointer: host_alignment.is_some()
            && buffer(
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                vk::ExternalMemoryFeatureFlags::IMPORTABLE,
            ),
        host_alignment: host_alignment.unwrap_or(0),
        timeline_fd: has_extension(ash::khr::external_semaphore_fd::NAME)
            && semaphore.external_semaphore_features.contains(
                vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
                    | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE,
            ),
        driver_uuid,
    }
}
