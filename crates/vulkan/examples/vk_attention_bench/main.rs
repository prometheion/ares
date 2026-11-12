//! Times the attention kernels on each device, next to CUDA's where the same GPU has both, with
//! the shape of Qwen3.8-27B: head size 256, 4 KV heads, 6 query heads per KV head.
//!
//! The cases: windows of 1, 8 and 64 rows (a decode step, a step with drafts, a verified
//! window) over histories of 1k, 32k and 262k cells in `f16`, `q8` and `q4` (K and V in the
//! same format), and one row that sees a list of 2051 cells (blocks of 4 cells and the newest
//! 3) of a history of 32k. The time is that of one attention layer (a step of the model has
//! 16), K and V cycle over several layers' pools so that the reads come from memory and not
//! from the L2 cache. "GB/s" is the bytes of K and V of the visible cells, read once, per
//! second: a kernel that reads cells more than once for lack of a wide item shows it as a lower
//! number.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_attention_bench -- \
//!     [--pairs q8/q8,f16/f16,q4/q4] [--hist 1024,32768,262144] [--rows 1,8,64] \
//!     [--list 2051] [--device N] [--iters N] [--blocks N] [--no-cuda]
//! ```
//!
//! CUDA runs when the driver is there (`ARES_LIBCUDA` names it); `--blocks` overrides the
//! number of workgroups Vulkan's planner aims for. Run it under `flock` when other work shares
//! the GPUs.

use ares_vulkan::CacheFormat;

mod vulkan {
    pub use ares_vulkan::{
        AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
        Event, Planner, Result, Stream,
    };
    pub const BACKEND: &str = "vulkan";

    fn open(ordinal: u32) -> Result<Device> {
        Device::open(ordinal)
    }

    fn new_stream(device: &Device) -> Result<Stream> {
        Stream::new(device)
    }

    fn plan_config(_: &Device, kernels: &AttentionKernels) -> ares_vulkan::PlanConfig {
        kernels.plan_config()
    }

    include!("core.rs");
}

mod cuda {
    pub use ares_cuda::{
        AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
        Event, Planner, Result, Stream, Wait,
    };
    pub const BACKEND: &str = "cuda";

    fn open(ordinal: u32) -> Result<Device> {
        Device::open(ordinal, Wait::Block)
    }

    fn new_stream(device: &Device) -> Result<Stream> {
        Stream::new(device, 0)
    }

    fn plan_config(device: &Device, _: &AttentionKernels) -> ares_cuda::PlanConfig {
        ares_cuda::PlanConfig::for_multiprocessors(device.info().multiprocessors)
    }

    include!("core.rs");
}

/// What a case looks at.
#[derive(Debug)]
pub enum Case {
    /// `n` consecutive rows, the newest cells of the sequence, attending causally.
    Window(usize),
    /// One row that sees a list of this many cells.
    List(usize),
}

#[derive(Debug)]
pub struct BenchCase {
    pub k_name: &'static str,
    pub v_name: &'static str,
    pub k: CacheFormat,
    pub v: CacheFormat,
    pub hist: usize,
    pub case: Case,
}

impl BenchCase {
    fn label(&self) -> String {
        let what = match self.case {
            Case::Window(rows) => format!("{rows} rows"),
            Case::List(cells) => format!("list of {cells}"),
        };
        format!("{}/{} {what} over {}", self.k_name, self.v_name, self.hist)
    }

    /// Bytes of K and V of the cells the case sees, read once.
    fn useful_bytes(&self) -> f64 {
        let cells = match self.case {
            Case::Window(_) => self.hist + 8,
            Case::List(cells) => cells,
        };
        let d = 256u32;
        cells as f64 * 4.0 * f64::from(self.k.row_bytes(d) + self.v.row_bytes(d))
    }
}

#[derive(Debug)]
pub struct Opts {
    pub iters: usize,
    pub blocks: Option<u32>,
    pub l2_bytes: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let list = |flag: &str, default: &str| -> Vec<String> {
        arg(flag).unwrap_or_else(|| default.to_owned()).split(',').map(str::to_owned).collect()
    };
    let name = |s: &str| -> &'static str {
        ["f16", "q8", "q7", "q6", "q5", "q4"].into_iter().find(|n| *n == s).expect("a format")
    };
    let format = |s: &str| -> CacheFormat {
        CacheFormat::ALL
            [["f16", "q8", "q7", "q6", "q5", "q4"].iter().position(|n| *n == s).expect("a format")]
    };
    let pairs: Vec<(&'static str, &'static str)> = list("--pairs", "f16/f16,q8/q8,q4/q4")
        .iter()
        .map(|p| {
            let (k, v) = p.split_once('/').expect("pairs are k/v");
            (name(k), name(v))
        })
        .collect();
    let hist: Vec<usize> =
        list("--hist", "1024,32768,262144").iter().map(|x| x.parse().unwrap()).collect();
    let rows: Vec<usize> = list("--rows", "1,8,64").iter().map(|x| x.parse().unwrap()).collect();
    let list_cells: Option<usize> = match arg("--list").as_deref() {
        Some("0") => None,
        other => Some(other.map_or(2051, |x| x.parse().unwrap())),
    };
    let only = arg("--device").map(|d| d.parse::<u32>().unwrap());
    let iters: usize = arg("--iters").map_or(30, |x| x.parse().unwrap());
    let blocks = arg("--blocks").map(|x| x.parse::<u32>().unwrap());
    let with_cuda = !args.iter().any(|a| a == "--no-cuda");

    let mut cases = Vec::new();
    for (k, v) in &pairs {
        for &h in &hist {
            for &r in &rows {
                cases.push(BenchCase {
                    k_name: k,
                    v_name: v,
                    k: format(k),
                    v: format(v),
                    hist: h,
                    case: Case::Window(r),
                });
            }
        }
        if let Some(cells) = list_cells {
            cases.push(BenchCase {
                k_name: k,
                v_name: v,
                k: format(k),
                v: format(v),
                hist: 32768,
                case: Case::List(cells),
            });
        }
    }

    let cuda_devices = if with_cuda { ares_cuda::devices().ok() } else { None };
    for info in ares_vulkan::devices().expect("the Vulkan loader") {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        // The same GPU in CUDA's list: the driver has no order in common with the loader.
        let twin = cuda_devices.as_ref().and_then(|d| d.iter().find(|c| c.uuid == info.uuid));
        let limit = twin.map(|c| {
            f64::from(c.memory_bus_width_bits) / 8.0 * f64::from(c.memory_clock_khz) * 2.0 / 1e6
        });
        let opts = Opts {
            iters,
            blocks,
            l2_bytes: twin.map_or(32.0 * 1048576.0, |c| f64::from(c.l2_cache_bytes)),
        };
        println!(
            "\n## {} (memory limit {}, L2 {} MiB)\n",
            info.name,
            limit.map_or("unknown".to_owned(), |l| format!("about {l:.0} GB/s")),
            opts.l2_bytes / 1048576.0
        );
        let vk = vulkan::run(info.ordinal, &opts, &cases).expect("Vulkan");
        let cu = twin.map(|c| cuda::run(c.ordinal, &opts, &cases).expect("CUDA"));
        println!(
            "| case | Vulkan items | Vulkan us | Vulkan GB/s | CUDA items | CUDA us | CUDA GB/s | Vulkan / CUDA |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for (i, c) in cases.iter().enumerate() {
            let gbps = |us: f64| c.useful_bytes() / us / 1e3;
            let cell = |r: Option<(f64, usize)>| match r {
                Some((us, items)) => {
                    (items.to_string(), format!("{us:.1}"), format!("{:.0}", gbps(us)))
                }
                None => ("-".into(), "skipped".into(), "-".into()),
            };
            let v = cell(vk[i]);
            let (cuda_cells, ratio) = match cu.as_ref().map(|r| r[i]) {
                Some(r) => {
                    let ratio = match (vk[i], r) {
                        (Some(a), Some(b)) => format!("{:.2}", a.0 / b.0),
                        _ => "-".into(),
                    };
                    (cell(r), ratio)
                }
                None => (("-".into(), "-".into(), "-".into()), "-".into()),
            };
            println!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                c.label(),
                v.0,
                v.1,
                v.2,
                cuda_cells.0,
                cuda_cells.1,
                cuda_cells.2,
                ratio
            );
        }
    }
}
