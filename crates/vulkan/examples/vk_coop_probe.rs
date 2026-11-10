//! What the tensor cores offer through Vulkan on each device: the shapes and element
//! types of `VK_KHR_cooperative_matrix` (what `DeviceInfo::cooperative_matrix` holds),
//! its NVIDIA predecessor `VK_NV_cooperative_matrix`, and the extensions that go beyond
//! them (`VK_NV_cooperative_matrix2`, `VK_NV_cooperative_vector`), asked of the driver
//! directly. Nothing is allocated on a device.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_coop_probe
//! ```

use std::ffi::{CStr, c_void};

use ares_vulkan::{ComponentType, devices};
use ash::vk;

/// `VkCooperativeMatrixFlexibleDimensionsPropertiesNV`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Flexible {
    s_type: i32,
    p_next: *mut c_void,
    m_granularity: u32,
    n_granularity: u32,
    k_granularity: u32,
    a: i32,
    b: i32,
    c: i32,
    result: i32,
    saturating: u32,
    scope: i32,
    workgroup_invocations: u32,
}

type FlexibleFn =
    unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut Flexible) -> vk::Result;

fn component(raw: i32) -> String {
    format!("{:?}", ComponentType::from_raw_for_probe(raw))
}

trait Probe {
    fn from_raw_for_probe(raw: i32) -> ComponentType;
}

impl Probe for ComponentType {
    fn from_raw_for_probe(raw: i32) -> ComponentType {
        match raw {
            0 => ComponentType::F16,
            1 => ComponentType::F32,
            2 => ComponentType::F64,
            3 => ComponentType::I8,
            4 => ComponentType::I16,
            5 => ComponentType::I32,
            6 => ComponentType::I64,
            7 => ComponentType::U8,
            8 => ComponentType::U16,
            9 => ComponentType::U32,
            10 => ComponentType::U64,
            1_000_141_000 => ComponentType::Bf16,
            1_000_491_002 => ComponentType::E4M3,
            1_000_491_003 => ComponentType::E5M2,
            other => ComponentType::Other(other),
        }
    }
}

fn main() {
    println!("== through ares-vulkan (VK_KHR_cooperative_matrix, subgroup scope)");
    for info in devices().expect("devices") {
        println!("{} (subgroup {}):", info.name, info.subgroup_size);
        if info.cooperative_matrix.is_empty() {
            println!("  none");
        }
        for c in &info.cooperative_matrix {
            println!(
                "  {:>2}x{:>2}x{:>2}  A {:?}  B {:?}  C {:?}  result {:?}{}",
                c.m,
                c.n,
                c.k,
                c.a,
                c.b,
                c.c,
                c.result,
                if c.saturating { "  saturating" } else { "" }
            );
        }
    }

    // SAFETY: loading the Vulkan loader, as the crate does.
    let entry = unsafe { ash::Entry::load() }.expect("the Vulkan loader");
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    // SAFETY: a plain instance without layers or extensions.
    let instance = unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
    }
    .expect("instance");
    // SAFETY: the instance is alive.
    let physical = unsafe { instance.enumerate_physical_devices() }.expect("devices");
    println!("\n== asked of the driver directly");
    for pd in physical {
        // SAFETY: as above.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if props.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        let name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
        // SAFETY: as above.
        let extensions = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap();
        let has = |wanted: &str| {
            extensions
                .iter()
                .filter_map(|e| e.extension_name_as_c_str().ok())
                .any(|n| n.to_str() == Ok(wanted))
        };
        println!("{name}");
        for ext in [
            "VK_KHR_cooperative_matrix",
            "VK_NV_cooperative_matrix",
            "VK_NV_cooperative_matrix2",
            "VK_NV_cooperative_vector",
            "VK_KHR_shader_integer_dot_product",
            "VK_NV_shader_sm_builtins",
        ] {
            println!("  {ext}: {}", if has(ext) { "yes" } else { "no" });
        }

        if has("VK_NV_cooperative_matrix") {
            let loader = ash::nv::cooperative_matrix::Instance::new(&entry, &instance);
            let query = loader.fp().get_physical_device_cooperative_matrix_properties_nv;
            let mut count = 0u32;
            // SAFETY: the extension is supported; a count query, then the list.
            unsafe { query(pd, &mut count, std::ptr::null_mut()) };
            let mut list = vec![vk::CooperativeMatrixPropertiesNV::default(); count as usize];
            // SAFETY: `list` has `count` entries with their structure types set.
            unsafe { query(pd, &mut count, list.as_mut_ptr()) };
            println!("  VK_NV_cooperative_matrix: {} shapes (subgroup scope)", list.len());
            for c in list.iter().filter(|c| c.scope == vk::ScopeNV::SUBGROUP) {
                println!(
                    "    {:>2}x{:>2}x{:>2}  A {}  B {}  C {}  D {}",
                    c.m_size,
                    c.n_size,
                    c.k_size,
                    component(c.a_type.as_raw()),
                    component(c.b_type.as_raw()),
                    component(c.c_type.as_raw()),
                    component(c.d_type.as_raw())
                );
            }
        }

        if has("VK_NV_cooperative_matrix2") {
            // SAFETY: the name is a NUL-terminated string; the loader returns null or the
            // function of the instance.
            let f = unsafe {
                (entry.static_fn().get_instance_proc_addr)(
                    instance.handle(),
                    c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr(),
                )
            };
            if let Some(f) = f {
                // SAFETY: the signature is the one in the specification.
                let f: FlexibleFn = unsafe { std::mem::transmute(f) };
                let mut count = 0u32;
                // SAFETY: a count query.
                unsafe { f(pd, &mut count, std::ptr::null_mut()) };
                let mut list = vec![
                    Flexible {
                        s_type: 1_000_593_001,
                        p_next: std::ptr::null_mut(),
                        m_granularity: 0,
                        n_granularity: 0,
                        k_granularity: 0,
                        a: 0,
                        b: 0,
                        c: 0,
                        result: 0,
                        saturating: 0,
                        scope: 0,
                        workgroup_invocations: 0,
                    };
                    count as usize
                ];
                // SAFETY: `list` has `count` entries with the right structure type.
                unsafe { f(pd, &mut count, list.as_mut_ptr()) };
                println!("  VK_NV_cooperative_matrix2 flexible dimensions: {count} entries");
                for c in list.iter().take(count as usize) {
                    println!(
                        "    granularity {}x{}x{}  A {}  B {}  C {}  result {}  scope {}  {} invocations{}",
                        c.m_granularity,
                        c.n_granularity,
                        c.k_granularity,
                        component(c.a),
                        component(c.b),
                        component(c.c),
                        component(c.result),
                        c.scope,
                        c.workgroup_invocations,
                        if c.saturating != 0 { "  saturating" } else { "" }
                    );
                }
            }
        }
        let _: Option<&CStr> = None;
    }
    // SAFETY: nothing was created from the instance.
    unsafe { instance.destroy_instance(None) };
}
