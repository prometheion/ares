//! What Vulkan lets two devices share on this machine: the device groups the loader
//! reports, the external memory and semaphore handle types each device supports, and
//! what really happens when memory of one device is imported by the other, or both
//! import the same host pages. It prints every call that fails with its result code.
//!
//! The first half uses the raw Vulkan calls and prints every failing call with its result
//! code. The second half ([`data`]) moves real data through each way of sharing that the
//! first half found to be accepted, with the engine's own device layer, because a call that
//! succeeds does not prove that the other device sees what was written (or reads what was
//! there).
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_mesh_probe
//! ```

use std::ffi::c_void;

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
    drop(gpus);
    data::run();
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

/// What really crosses between the devices: data written and read through kernels and through
/// the copy engine, and ordering through a shared timeline semaphore.
mod data {
    use std::sync::Arc;
    use std::time::Instant;

    use ares_vulkan::{
        Args, Device, DeviceBuffer, DevicePtr, Function, HostPages, Module, ShareKind,
        SharedTimeline, Stream, Wait, device_count,
    };

    const N: usize = 1 << 16;

    /// `out[i] = in[i]` as a kernel (the sum kernel with one operand).
    struct Copier(Function);

    impl Copier {
        fn new(device: &Device) -> Self {
            let image = include_bytes!(concat!(env!("OUT_DIR"), "/mesh_sum.spv"));
            Self(Module::load(device, image).unwrap().function("main").unwrap())
        }

        fn copy(&self, stream: &Stream, dst: DevicePtr, src: DevicePtr, n: usize) {
            let mut args = Args::new().ptr(dst).ptr(src);
            for _ in 1..8 {
                args = args.ptr(DevicePtr::NULL);
            }
            let args = args.u32(1).u32(n as u32);
            // SAFETY: the sum kernel with one operand; both ranges hold `n` values.
            unsafe { self.0.launch(stream, [(n as u32).div_ceil(256), 1, 1], args).unwrap() };
        }
    }

    fn pattern(seed: f32) -> Vec<u8> {
        (0..N).flat_map(|i| (seed + i as f32).to_le_bytes()).collect()
    }

    fn check(label: &str, got: &[u8], seed: f32) {
        let want = pattern(seed);
        let (got_words, want_words) = (got.as_chunks::<4>().0, want.as_chunks::<4>().0);
        let wrong = got_words.iter().zip(want_words).filter(|(a, b)| a != b).count();
        let first = got_words.iter().zip(want_words).position(|(a, b)| a != b);
        if wrong == 0 {
            println!("    {label}: correct");
        } else {
            println!(
                "    {label}: WRONG, {wrong} of {N} values differ (first at {first:?}, got {:?})",
                &got[..16.min(got.len())]
            );
        }
    }

    fn zeros() -> Vec<u8> {
        vec![0u8; N * 4]
    }

    pub fn run() {
        if device_count().unwrap_or(0) < 2 {
            return;
        }
        let devices: Vec<Device> =
            (0..2).map(|i| Device::open_with(i, Wait::Spin).unwrap()).collect();
        let streams: Vec<Stream> = devices.iter().map(|d| Stream::new(d).unwrap()).collect();
        let copiers: Vec<Copier> = devices.iter().map(Copier::new).collect();
        let bytes = N * 4;

        for kind in [ShareKind::OpaqueFd, ShareKind::DmaBuf] {
            println!("\n== video memory of one device imported by the other ({kind:?}), data");
            for (owner, user) in [(0usize, 1usize), (1, 0)] {
                println!(
                    "  memory of {} used from {}",
                    devices[owner].info().name,
                    devices[user].info().name
                );
                let exported =
                    match DeviceBuffer::new_exportable(&devices[owner], bytes as u64, kind) {
                        Ok(b) => b,
                        Err(e) => {
                            println!("    export failed: {e}");
                            continue;
                        }
                    };
                let shared = match exported.export() {
                    Ok(s) => s,
                    Err(e) => {
                        println!("    export fd failed: {e}");
                        continue;
                    }
                };
                let view = match DeviceBuffer::import(&devices[user], shared) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("    import failed: {e}");
                        continue;
                    }
                };
                println!(
                    "    import: ok (view at {:#x}, owner at {:#x})",
                    view.ptr().as_raw(),
                    exported.ptr().as_raw()
                );
                let own_src = DeviceBuffer::new(&devices[owner], bytes as u64).unwrap();
                let user_src = DeviceBuffer::new(&devices[user], bytes as u64).unwrap();
                let user_dst = DeviceBuffer::new(&devices[user], bytes as u64).unwrap();
                // The user writes into the owner's memory: with a kernel, and with a copy.
                for (label, kernel) in
                    [("user kernel writes", true), ("user copy engine writes", false)]
                {
                    exported.write(0, &zeros()).unwrap();
                    user_src.write(0, &pattern(1.0)).unwrap();
                    if kernel {
                        copiers[user].copy(&streams[user], view.ptr(), user_src.ptr(), N);
                    } else {
                        // SAFETY: whole buffers of the stream's device.
                        unsafe { streams[user].copy(view.ptr(), user_src.ptr(), bytes).unwrap() };
                    }
                    streams[user].synchronize().unwrap();
                    let mut got = vec![0u8; bytes];
                    exported.read(0, &mut got).unwrap();
                    check(label, &got, 1.0);
                }
                // The user reads the owner's memory.
                for (label, kernel) in
                    [("user kernel reads", true), ("user copy engine reads", false)]
                {
                    exported.write(0, &pattern(7.0)).unwrap();
                    user_dst.write(0, &zeros()).unwrap();
                    if kernel {
                        copiers[user].copy(&streams[user], user_dst.ptr(), view.ptr(), N);
                    } else {
                        // SAFETY: whole buffers of the stream's device.
                        unsafe { streams[user].copy(user_dst.ptr(), view.ptr(), bytes).unwrap() };
                    }
                    streams[user].synchronize().unwrap();
                    let mut got = vec![0u8; bytes];
                    user_dst.read(0, &mut got).unwrap();
                    check(label, &got, 7.0);
                }
                let _ = &own_src;
            }
        }

        println!("\n== host pages imported by both devices, data");
        let refs: Vec<&Device> = devices.iter().collect();
        let pages = Arc::new(HostPages::new(&refs, bytes).unwrap());
        let views: Vec<DeviceBuffer> =
            devices.iter().map(|d| DeviceBuffer::import_host(d, &pages).unwrap()).collect();
        println!(
            "  import: ok (views at {:#x} and {:#x}, {} bytes)",
            views[0].ptr().as_raw(),
            views[1].ptr().as_raw(),
            pages.len()
        );
        let local: Vec<DeviceBuffer> =
            devices.iter().map(|d| DeviceBuffer::new(d, bytes as u64).unwrap()).collect();
        let host_read = || -> Vec<u8> {
            let mut out = vec![0u8; bytes];
            for (i, b) in out.iter_mut().enumerate() {
                // SAFETY: inside the pages; the devices are idle.
                *b = unsafe { pages.as_ptr().add(i).read_volatile() };
            }
            out
        };
        for d in 0..2 {
            let name = &devices[d].info().name;
            for (label, kernel) in [("kernel", true), ("copy engine", false)] {
                // device writes, host reads
                // SAFETY: inside the pages; the devices are idle.
                unsafe { pages.as_ptr().write_bytes(0, bytes) };
                local[d].write(0, &pattern(3.0)).unwrap();
                if kernel {
                    copiers[d].copy(&streams[d], views[d].ptr(), local[d].ptr(), N);
                } else {
                    // SAFETY: whole buffers of the stream's device.
                    unsafe { streams[d].copy(views[d].ptr(), local[d].ptr(), bytes).unwrap() };
                }
                streams[d].synchronize().unwrap();
                check(&format!("{name}: {label} writes, host reads"), &host_read(), 3.0);
                // host writes, device reads
                let want = pattern(5.0);
                for (i, b) in want.iter().enumerate() {
                    // SAFETY: inside the pages; the devices are idle.
                    unsafe { pages.as_ptr().add(i).write_volatile(*b) };
                }
                local[d].write(0, &zeros()).unwrap();
                if kernel {
                    copiers[d].copy(&streams[d], local[d].ptr(), views[d].ptr(), N);
                } else {
                    // SAFETY: as above.
                    unsafe { streams[d].copy(local[d].ptr(), views[d].ptr(), bytes).unwrap() };
                }
                streams[d].synchronize().unwrap();
                let mut got = vec![0u8; bytes];
                local[d].read(0, &mut got).unwrap();
                check(&format!("{name}: host writes, {label} reads"), &got, 5.0);
            }
        }
        // One device writes, the other reads, with the host out of the middle.
        for (w, r) in [(0usize, 1usize), (1, 0)] {
            // SAFETY: the devices are idle.
            unsafe { pages.as_ptr().write_bytes(0, bytes) };
            local[w].write(0, &pattern(9.0)).unwrap();
            local[r].write(0, &zeros()).unwrap();
            copiers[w].copy(&streams[w], views[w].ptr(), local[w].ptr(), N);
            streams[w].synchronize().unwrap();
            copiers[r].copy(&streams[r], local[r].ptr(), views[r].ptr(), N);
            streams[r].synchronize().unwrap();
            let mut got = vec![0u8; bytes];
            local[r].read(0, &mut got).unwrap();
            check(
                &format!(
                    "{} kernel writes, {} kernel reads",
                    devices[w].info().name,
                    devices[r].info().name
                ),
                &got,
                9.0,
            );
        }

        println!("\n== a kernel polling host memory that the host writes while it polls");
        // The ways a kernel can read a word again and again: a plain load of a `volatile
        // coherent` buffer, the same on the upper half of a 64-bit word, and two atomics.
        // The host writes the awaited value 50 ms after the kernel starts. 16 million polls
        // take 0.2 s for a load that is answered by a cache, 2 s or more for one that is not.
        let module_poll = |device: &Device| {
            Module::load(device, include_bytes!(concat!(env!("OUT_DIR"), "/mesh_poll_test.spv")))
                .unwrap()
                .function("main")
                .unwrap()
        };
        for (mode, label) in [
            (0u32, "volatile coherent 64-bit load"),
            (1, "volatile coherent 32-bit load"),
            (2, "atomicAdd(word, 0)"),
            (3, "atomicLoad, device scope"),
        ] {
            for d in 0..2 {
                let function = module_poll(&devices[d]);
                let out = DeviceBuffer::new(&devices[d], 4096).unwrap();
                out.write(0, &[0u8; 64]).unwrap();
                let p = views[d].ptr();
                let args =
                    Args::new().ptr(p).ptr(p).ptr(p).ptr(out.ptr()).u32(mode).u32(1 << 24).u32(7);
                // SAFETY: the host pages hold two words; the output buffer four.
                unsafe { function.launch(&streams[d], [1, 1, 1], args).unwrap() };
                // SAFETY: inside the pages; the kernel only reads.
                unsafe { (pages.as_ptr().add(4) as *mut u32).write_volatile(0) };
                let start = Instant::now();
                streams[d].flush().unwrap();
                std::thread::sleep(std::time::Duration::from_millis(50));
                // SAFETY: as above.
                unsafe { (pages.as_ptr().add(4) as *mut u32).write_volatile(7) };
                streams[d].synchronize().unwrap();
                let mut got = [0u8; 8];
                out.read(0, &mut got).unwrap();
                println!(
                    "    {label} on {}: {} after {} polls, {:?}",
                    devices[d].info().name,
                    if got[0] == 1 { "saw the write" } else { "never saw the write" },
                    u32::from_le_bytes(got[4..8].try_into().unwrap()),
                    start.elapsed()
                );
            }
        }
        // SAFETY: the devices are idle.
        unsafe { (pages.as_ptr().add(4) as *mut u32).write_volatile(0) };

        println!("\n== a timeline semaphore shared by both devices");
        match SharedTimeline::new(&devices) {
            Err(e) => println!("  share failed: {e}"),
            Ok(timeline) => {
                println!("  export / import: ok");
                for s in &streams {
                    s.link(&streams[0]);
                }
                // Device 0 writes into the host pages and signals 1; device 1 waits for 1 and
                // copies the pages. The order is only right if the wait works.
                for (w, r) in [(0usize, 1usize), (1, 0)] {
                    // SAFETY: the devices are idle.
                    unsafe { pages.as_ptr().write_bytes(0, bytes) };
                    local[w].write(0, &pattern(11.0)).unwrap();
                    local[r].write(0, &zeros()).unwrap();
                    let base = timeline.value(&devices[0]).unwrap();
                    // A long kernel chain before the signal makes a missing wait show up.
                    for _ in 0..200 {
                        copiers[w].copy(&streams[w], views[w].ptr(), local[w].ptr(), N);
                    }
                    timeline.signal(&streams[w], base + 1).unwrap();
                    timeline.wait(&streams[r], base + 1).unwrap();
                    copiers[r].copy(&streams[r], local[r].ptr(), views[r].ptr(), N);
                    streams[r].synchronize().unwrap();
                    streams[w].synchronize().unwrap();
                    let mut got = vec![0u8; bytes];
                    local[r].read(0, &mut got).unwrap();
                    check(&format!("signal on {w}, wait on {r}"), &got, 11.0);
                    println!(
                        "    counter seen from device 0: {}, from device 1: {}",
                        timeline.value(&devices[0]).unwrap(),
                        timeline.value(&devices[1]).unwrap()
                    );
                }
                // The cost of a hand-over between the two devices, back and forth.
                let base = timeline.value(&devices[0]).unwrap();
                let hops = 400u64;
                let start = Instant::now();
                for k in 0..hops / 2 {
                    let t = base + 2 * k;
                    timeline.wait(&streams[0], t).unwrap();
                    timeline.signal(&streams[0], t + 1).unwrap();
                    timeline.wait(&streams[1], t + 1).unwrap();
                    timeline.signal(&streams[1], t + 2).unwrap();
                }
                streams[0].synchronize().unwrap();
                streams[1].synchronize().unwrap();
                println!(
                    "    ping-pong: {:.1} us per hand-over (host submits included)",
                    start.elapsed().as_secs_f64() * 1e6 / hops as f64
                );
            }
        }
    }
}
