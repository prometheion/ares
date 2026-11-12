//! Times the sampler on each device, next to CUDA's where the same GPU has both: the pick of
//! 1, 4 and 16 rows of the vocabulary of Qwen (248320 logits) for greedy rows, rows with top-k
//! and top-p, and rows with top-p alone. The time is that of the whole pick on the device (all
//! its launches), per batch and per row.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_sampler_bench -- [--device N] [--no-cuda]
//! ```
//!
//! CUDA runs when the driver is there (`ARES_LIBCUDA` names it). Run it under `flock` when other
//! work shares the GPUs.

mod vulkan {
    pub use ares_vulkan::{
        Device, DeviceBuffer, DevicePtr, Error, Event, Result, SampleBatch, SampleParams,
        SamplePick, SamplePlan, SampleRow, SamplerKernels, Stream,
    };

    fn open(ordinal: u32) -> Result<Device> {
        Device::open(ordinal)
    }

    fn new_stream(device: &Device) -> Result<Stream> {
        Stream::new(device)
    }

    include!("core.rs");
}

mod cuda {
    pub use ares_cuda::{
        Device, DeviceBuffer, DevicePtr, Error, Event, Result, SampleBatch, SampleParams,
        SamplePick, SamplePlan, SampleRow, SamplerKernels, Stream, Wait,
    };

    fn open(ordinal: u32) -> Result<Device> {
        Device::open(ordinal, Wait::Block)
    }

    fn new_stream(device: &Device) -> Result<Stream> {
        Stream::new(device, 0)
    }

    include!("core.rs");
}

/// The vocabulary of Qwen3.8.
pub const VOCAB: usize = 248_320;

/// What the rows of a batch ask for.
#[derive(Debug)]
pub struct Setting {
    pub name: &'static str,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let only = arg("--device").map(|d| d.parse::<u32>().unwrap());
    let with_cuda = !args.iter().any(|a| a == "--no-cuda");
    let settings = [
        Setting { name: "greedy", temperature: 0.0, top_k: 0, top_p: 1.0 },
        Setting { name: "top-k 20, top-p 0.95", temperature: 1.0, top_k: 20, top_p: 0.95 },
        Setting { name: "top-p 0.95", temperature: 1.0, top_k: 0, top_p: 0.95 },
    ];
    let cuda_devices = if with_cuda { ares_cuda::devices().ok() } else { None };
    for info in ares_vulkan::devices().expect("the Vulkan loader") {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let twin = cuda_devices.as_ref().and_then(|d| d.iter().find(|c| c.uuid == info.uuid));
        println!("\n## {}\n", info.name);
        println!(
            "| setting | rows | launches | Vulkan us | Vulkan us/row | CUDA us | CUDA us/row | Vulkan / CUDA |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for setting in &settings {
            for rows in [1usize, 4, 16] {
                let (v, launches) = vulkan::time(info.ordinal, setting, rows).expect("Vulkan");
                let c = twin.map(|c| cuda::time(c.ordinal, setting, rows).expect("CUDA").0);
                let (cu, cur, ratio) = match c {
                    Some(c) => (
                        format!("{c:.0}"),
                        format!("{:.0}", c / rows as f64),
                        format!("{:.2}", v / c),
                    ),
                    None => ("-".into(), "-".into(), "-".into()),
                };
                println!(
                    "| {} | {rows} | {launches} | {v:.0} | {:.0} | {cu} | {cur} | {ratio} |",
                    setting.name,
                    v / rows as f64
                );
            }
        }
    }
}
