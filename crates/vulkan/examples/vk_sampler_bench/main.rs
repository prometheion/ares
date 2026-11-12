//! Times the sampler on each device, next to CUDA's where the same GPU has both: the pick of 1, 4
//! and 16 rows of the vocabulary of Qwen (248320 logits) for the settings of
//! `the_time_of_a_pick` in `crates/cuda/tests/sampler.rs`, with the same rows and the same
//! method on both backends: for every pick the logits (and the corrections) are written
//! fresh, the stream is drained, and the time is that of the launches of the pick to the end of
//! the stream. The median of 40 picks is printed.
//!
//! ```sh
//! flock <lock> cargo run --release -p ares-vulkan --example vk_sampler_bench -- [--device N] [--no-cuda]
//! ```
//!
//! CUDA runs when the driver is there (`ARES_LIBCUDA` names it). Run it under `flock` when other
//! work shares the GPUs; the figures are noisy while anything else compiles or computes.

mod vulkan {
    pub use ares_vulkan::{
        Device, DeviceBuffer, DevicePtr, Error, Result, SampleBatch, SamplePlan, SamplerKernels,
        Stream,
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
        Device, DeviceBuffer, DevicePtr, Error, Result, SampleBatch, SamplePlan, SamplerKernels,
        Stream, Wait,
    };

    fn open(ordinal: u32) -> Result<Device> {
        Device::open(ordinal, Wait::Block)
    }

    fn new_stream(device: &Device) -> Result<Stream> {
        Stream::new(device, 0)
    }

    include!("core.rs");
}

use ares_sampler::{Logprobs, SampleParams};

/// The vocabulary of Qwen3.8.
pub const VOCAB: usize = 248_320;

/// What the rows of a batch ask for.
#[derive(Debug)]
pub struct Setting {
    pub name: &'static str,
    pub params: SampleParams,
    /// Whether the rows have 64 corrections, as a penalty over the last 64 tokens gives.
    pub entries: bool,
    /// Whether the rows have a mask that allows a third of the tokens.
    pub mask: bool,
}

fn settings() -> Vec<Setting> {
    let sampled =
        SampleParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 9, ..SampleParams::GREEDY };
    let lp = |after_chain| Logprobs { top: 5, chosen: true, after_chain };
    let case = |name, params, entries, mask| Setting { name, params, entries, mask };
    vec![
        case("greedy", SampleParams::GREEDY, false, false),
        case("top-k 20, top-p 0.95", sampled, false, false),
        case("top-p 0.95", SampleParams { top_k: 0, ..sampled }, false, false),
        case("greedy with penalties", SampleParams::GREEDY, true, false),
        case("greedy with a mask", SampleParams::GREEDY, false, true),
        case("top-k, top-p, penalties and a mask", sampled, true, true),
        case(
            "greedy with logprobs",
            SampleParams { logprobs: lp(false), ..SampleParams::GREEDY },
            false,
            false,
        ),
        case(
            "top-k, top-p with logprobs after the chain",
            SampleParams { logprobs: lp(true), ..sampled },
            false,
            false,
        ),
        case(
            "top-n-sigma 2, typical 0.9, top-k, top-p",
            SampleParams { typical_p: 0.9, top_n_sigma: 2.0, ..sampled },
            false,
            false,
        ),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let only = arg("--device").map(|d| d.parse::<u32>().unwrap());
    let with_cuda = !args.iter().any(|a| a == "--no-cuda");
    let cuda_devices = if with_cuda { ares_cuda::devices().ok() } else { None };
    for info in ares_vulkan::devices().expect("the Vulkan loader") {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let twin = cuda_devices.as_ref().and_then(|d| d.iter().find(|c| c.uuid == info.uuid));
        println!("\n## {}\n", info.name);
        println!("| setting | rows | launches | Vulkan us | CUDA us | Vulkan / CUDA |");
        println!("| --- | ---: | ---: | ---: | ---: | ---: |");
        for setting in &settings() {
            for rows in [1usize, 4, 16] {
                let (v, launches) = vulkan::time(info.ordinal, setting, rows).expect("Vulkan");
                let c = twin.map(|c| cuda::time(c.ordinal, setting, rows).expect("CUDA").0);
                let (cu, ratio) = match c {
                    Some(c) => (format!("{c:.0}"), format!("{:.2}", v / c)),
                    None => ("-".into(), "-".into()),
                };
                println!("| {} | {rows} | {launches} | {v:.0} | {cu} | {ratio} |", setting.name);
            }
        }
    }
}
