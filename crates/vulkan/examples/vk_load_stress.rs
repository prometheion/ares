//! Loads the kernels of every device at the same time, each on its own thread, again and
//! again: a check of whether the driver survives pipelines being made for several devices
//! at once (`vkCreateComputePipelines` from one thread per device).
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_load_stress -- [rounds] [serial]
//! ```
//!
//! `serial` loads the devices one after the other instead, for comparison.

use ares_core::GgmlType;
use ares_vulkan::{Device, Kernels, devices};

fn main() {
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let serial = std::env::args().nth(2).is_some_and(|a| a == "serial");
    let ordinals: Vec<u32> = devices().expect("devices").iter().map(|d| d.ordinal).collect();
    let types = [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::IQ4_XS, GgmlType::Q8_0];
    let load = |ordinal: u32| {
        let device = Device::open(ordinal).expect("open");
        let kernels = Kernels::load(&device).expect("load");
        kernels.prepare(&types).expect("prepare");
    };
    for round in 0..rounds {
        if serial {
            for &ordinal in &ordinals {
                load(ordinal);
            }
        } else {
            std::thread::scope(|scope| {
                for &ordinal in &ordinals {
                    scope.spawn(move || load(ordinal));
                }
            });
        }
        println!("round {round} done");
    }
}
