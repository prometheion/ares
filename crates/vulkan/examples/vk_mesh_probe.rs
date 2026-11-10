//! What Vulkan lets two devices share on this machine: the device groups the loader
//! reports, the external memory and semaphore handle types each device supports, and
//! what really happens when memory of one device is imported by the other, or both
//! import the same host pages. It prints every call that fails with its result code.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_mesh_probe
//! ```

use std::ffi::{CStr, c_void};

use ash::vk;

struct Gpu {
    name: String,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    extensions: Vec<String>,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue_family: u32,
}

fn main() {
    // SAFETY: loading the Vulkan loader.
    let entry = unsafe { ash::Entry::load() }.expect("the Vulkan loader");
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    // SAFETY: an instance without layers.
    let instance = unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
    }
    .expect("instance");

    // --- device groups ---------------------------------------------------------------
    // SAFETY: the instance is alive.
    let groups = unsafe { instance.enumerate_physical_device_groups_len() }.unwrap_or(0);
    let mut group_list = vec![vk::PhysicalDeviceGroupProperties::default(); groups];
    // SAFETY: as above; the list has `groups` entries.
    let _ = unsafe { instance.enumerate_physical_device_groups(&mut group_list) };
    println!("== physical device groups: {groups}");
    for (i, g) in group_list.iter().enumerate() {
        println!(
            "  group {i}: {} device(s), subset allocation {}",
            g.physical_device_count,
            g.subset_allocation == vk::TRUE
        );
    }

    // --- the devices --------------------------------------------------------------------
    // SAFETY: as above.
    let physicals = unsafe { instance.enumerate_physical_devices() }.unwrap();
    let mut gpus = Vec::new();
    for physical in physicals {
        // SAFETY: as above.
        let props = unsafe { instance.get_physical_device_properties(physical) };
        if props.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        let name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
        // SAFETY: as above.
        let exts = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
        let extensions: Vec<String> = exts
            .iter()
            .filter_map(|e| e.extension_name_as_c_str().ok())
            .map(|n| n.to_string_lossy().into_owned())
            .collect();
        let has = |n: &str| extensions.iter().any(|e| e == n);
        println!("\n{name}: external-sharing extensions");
        for n in [
            "VK_KHR_external_memory",
            "VK_KHR_external_memory_fd",
            "VK_EXT_external_memory_dma_buf",
            "VK_EXT_external_memory_host",
            "VK_KHR_external_semaphore",
            "VK_KHR_external_semaphore_fd",
            "VK_KHR_external_fence_fd",
            "VK_EXT_image_drm_format_modifier",
            "VK_KHR_device_group",
            "VK_EXT_pci_bus_info",
            "VK_KHR_timeline_semaphore",
            "VK_EXT_pageable_device_local_memory",
            "VK_NV_external_memory_rdma",
        ] {
            println!("  {n}: {}", if has(n) { "yes" } else { "no" });
        }
        // SAFETY: as above.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family =
            families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE)).unwrap()
                as u32;
        let want = [
            "VK_KHR_external_memory",
            "VK_KHR_external_memory_fd",
            "VK_EXT_external_memory_dma_buf",
            "VK_EXT_external_memory_host",
            "VK_KHR_external_semaphore",
            "VK_KHR_external_semaphore_fd",
        ];
        let names: Vec<std::ffi::CString> =
            want.iter().filter(|n| has(n)).map(|n| std::ffi::CString::new(*n).unwrap()).collect();
        let name_ptrs: Vec<*const i8> = names.iter().map(|n| n.as_ptr()).collect();
        let prio = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prio)];
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true);
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&name_ptrs)
            .push_next(&mut f2);
        // SAFETY: the extensions were listed by the device.
        let device = unsafe { instance.create_device(physical, &info, None) }.expect("device");
        // SAFETY: as above.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        gpus.push(Gpu { name, physical, device, extensions, memory, queue_family });
    }

    // --- capabilities ------------------------------------------------------------------
    println!("\n== external buffer capabilities (storage + transfer usage)");
    let handle_types = [
        ("OPAQUE_FD", vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
        ("DMA_BUF", vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
        ("HOST_ALLOCATION", vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT),
        ("HOST_MAPPED_FOREIGN", vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT),
    ];
    for gpu in &gpus {
        println!("{}", gpu.name);
        for (label, ty) in handle_types {
            let info = vk::PhysicalDeviceExternalBufferInfo::default()
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .handle_type(ty);
            let mut props = vk::ExternalBufferProperties::default();
            // SAFETY: the instance and device are alive.
            unsafe {
                instance.get_physical_device_external_buffer_properties(
                    gpu.physical,
                    &info,
                    &mut props,
                )
            };
            let p = props.external_memory_properties;
            println!(
                "  {label}: features {:?}, exportable from {:?}, compatible with {:?}",
                p.external_memory_features,
                p.export_from_imported_handle_types,
                p.compatible_handle_types
            );
        }
        for (label, ty) in [
            ("OPAQUE_FD", vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
            ("SYNC_FD", vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
        ] {
            for timeline in [true, false] {
                let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(if timeline {
                    vk::SemaphoreType::TIMELINE
                } else {
                    vk::SemaphoreType::BINARY
                });
                let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
                    .handle_type(ty)
                    .push_next(&mut kind);
                let mut props = vk::ExternalSemaphoreProperties::default();
                // SAFETY: as above.
                unsafe {
                    instance.get_physical_device_external_semaphore_properties(
                        gpu.physical,
                        &info,
                        &mut props,
                    )
                };
                println!(
                    "  semaphore {label} ({}): features {:?}, compatible with {:?}",
                    if timeline { "timeline" } else { "binary" },
                    props.external_semaphore_features,
                    props.compatible_handle_types
                );
            }
        }
    }

    if gpus.len() < 2 {
        println!("\none device: no cross-device attempts");
        return;
    }

    // --- cross-device attempts --------------------------------------------------------
    let find_type = |gpu: &Gpu,
                     bits: u32,
                     want: vk::MemoryPropertyFlags,
                     avoid: vk::MemoryPropertyFlags|
     -> Option<u32> {
        (0..gpu.memory.memory_type_count).find(|&i| {
            let flags = gpu.memory.memory_types[i as usize].property_flags;
            bits & (1 << i) != 0 && flags.contains(want) && !flags.intersects(avoid)
        })
    };
    println!("\n== exporting memory of one device and importing it into the other");
    for (src, dst) in [(0usize, 1usize), (1, 0)] {
        let (a, b) = (&gpus[src], &gpus[dst]);
        for (label, ty) in handle_types.iter().take(2) {
            let ty = *ty;
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ty);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(1 << 20)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .push_next(&mut external);
            // SAFETY: the device is alive.
            let buf = match unsafe { a.device.create_buffer(&buffer_info, None) } {
                Ok(b) => b,
                Err(e) => {
                    println!(
                        "  {} -> {} {label}: vkCreateBuffer (export) failed: {e:?}",
                        a.name, b.name
                    );
                    continue;
                }
            };
            // SAFETY: as above.
            let req = unsafe { a.device.get_buffer_memory_requirements(buf) };
            let Some(ty_index) = find_type(
                a,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ) else {
                println!("  no memory type");
                continue;
            };
            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(ty);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty_index)
                .push_next(&mut export)
                .push_next(&mut dedicated);
            // SAFETY: valid allocation info.
            let memory = match unsafe { a.device.allocate_memory(&alloc, None) } {
                Ok(m) => m,
                Err(e) => {
                    println!(
                        "  {} -> {} {label}: vkAllocateMemory (export) failed: {e:?}",
                        a.name, b.name
                    );
                    continue;
                }
            };
            let fd = match ty {
                vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
                | vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT => {
                    let loader = ash::khr::external_memory_fd::Device::new(&instance, &a.device);
                    let get = vk::MemoryGetFdInfoKHR::default().memory(memory).handle_type(ty);
                    // SAFETY: the memory was exported with this handle type.
                    unsafe { loader.get_memory_fd(&get) }
                }
                _ => unreachable!(),
            };
            let fd = match fd {
                Ok(fd) => fd,
                Err(e) => {
                    println!("  {} -> {} {label}: vkGetMemoryFdKHR failed: {e:?}", a.name, b.name);
                    continue;
                }
            };
            println!("  {} -> {} {label}: exported (fd {fd})", a.name, b.name);
            // Import into b.
            let loader = ash::khr::external_memory_fd::Device::new(&instance, &b.device);
            let mut props = vk::MemoryFdPropertiesKHR::default();
            // SAFETY: a valid fd of the handle type.
            let property_result = unsafe {
                (loader.fp().get_memory_fd_properties_khr)(b.device.handle(), ty, fd, &mut props)
            };
            println!(
                "    vkGetMemoryFdPropertiesKHR on the importer: {property_result:?} (memory types {:#x})",
                props.memory_type_bits
            );
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ty);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(1 << 20)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .push_next(&mut external);
            // SAFETY: as above.
            let imported = unsafe { b.device.create_buffer(&buffer_info, None) }.unwrap();
            // SAFETY: as above.
            let req_b = unsafe { b.device.get_buffer_memory_requirements(imported) };
            let bits = if property_result == vk::Result::SUCCESS {
                props.memory_type_bits & req_b.memory_type_bits
            } else {
                req_b.memory_type_bits
            };
            let Some(ty_b) = find_type(
                b,
                bits,
                vk::MemoryPropertyFlags::empty(),
                vk::MemoryPropertyFlags::empty(),
            ) else {
                println!("    importer: no compatible memory type (bits {bits:#x})");
                continue;
            };
            let mut import = vk::ImportMemoryFdInfoKHR::default().handle_type(ty).fd(fd);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(imported);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty_b)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            // SAFETY: as above; on failure the fd stays ours.
            match unsafe { b.device.allocate_memory(&alloc, None) } {
                Ok(_) => println!("    vkAllocateMemory (import) on {}: SUCCESS", b.name),
                Err(e) => println!("    vkAllocateMemory (import) on {}: FAILED {e:?}", b.name),
            }
        }
    }

    println!("\n== both devices importing the same host pages (VK_EXT_external_memory_host)");
    let host_size = 1usize << 20;
    // SAFETY: an anonymous mapping of whole pages.
    let host = unsafe { libc_mmap(host_size) };
    for gpu in &gpus {
        if !gpu.extensions.iter().any(|e| e == "VK_EXT_external_memory_host") {
            println!("  {}: extension missing", gpu.name);
            continue;
        }
        let loader = ash::ext::external_memory_host::Device::new(&instance, &gpu.device);
        let mut props = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: a page-aligned host pointer.
        let r = unsafe {
            (loader.fp().get_memory_host_pointer_properties_ext)(
                gpu.device.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                host,
                &mut props,
            )
        };
        println!(
            "  {}: vkGetMemoryHostPointerPropertiesEXT {r:?}, memory types {:#x}",
            gpu.name, props.memory_type_bits
        );
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(host_size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .push_next(&mut external);
        // SAFETY: as above.
        let buffer = unsafe { gpu.device.create_buffer(&buffer_info, None) }.unwrap();
        // SAFETY: as above.
        let req = unsafe { gpu.device.get_buffer_memory_requirements(buffer) };
        let bits = props.memory_type_bits & req.memory_type_bits;
        let Some(ty) = find_type(
            gpu,
            bits,
            vk::MemoryPropertyFlags::empty(),
            vk::MemoryPropertyFlags::empty(),
        ) else {
            println!("    no memory type");
            continue;
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(host);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(host_size as u64)
            .memory_type_index(ty)
            .push_next(&mut import)
            .push_next(&mut flags);
        // SAFETY: as above.
        match unsafe { gpu.device.allocate_memory(&alloc, None) } {
            Ok(memory) => {
                // SAFETY: bound once.
                let bound = unsafe { gpu.device.bind_buffer_memory(buffer, memory, 0) };
                println!(
                    "    import: SUCCESS, bind {bound:?}, memory type {ty} flags {:?}",
                    gpu.memory.memory_types[ty as usize].property_flags
                );
            }
            Err(e) => println!("    import: FAILED {e:?}"),
        }
        let _ = gpu.queue_family;
    }
    let _ = CStr::from_bytes_with_nul(b"\0");
}

/// A zeroed anonymous mapping of `len` bytes (page aligned).
///
/// # Safety
///
/// Calls `mmap`.
unsafe fn libc_mmap(len: usize) -> *mut c_void {
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
    }
    // PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS
    // SAFETY: the arguments describe a fresh anonymous mapping.
    unsafe { mmap(std::ptr::null_mut(), len, 3, 0x22, -1, 0) }
}
