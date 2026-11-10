//! The rate of the tensor cores through `VK_KHR_cooperative_matrix`, per shape and
//! element type: a subgroup multiplies matrices in a loop on registers only
//! (`shaders/coop_rate.glsl`). To be set next to the `mma` rates the CUDA layer's probe
//! measures (RTX 4070: 234 TOPS int8, 117 f16 with f16 accumulation, 58.6 f16 with f32
//! accumulation; RTX 3060: 102 / 51.2 / 25.6), which are the cap of GeForce cards.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_coop_rate [device]
//! ```

use ares_vulkan::{
    Args, ComponentType, Device, DeviceBuffer, Event, Function, Module, Result, Stream, devices,
};

struct Variant {
    name: &'static str,
    image: &'static [u8],
    m: u32,
    n: u32,
    k: u32,
    a: ComponentType,
    c: ComponentType,
}

macro_rules! variant {
    ($name:literal, $file:literal, $m:expr, $n:expr, $k:expr, $a:expr, $c:expr) => {
        Variant {
            name: $name,
            image: include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".spv")),
            m: $m,
            n: $n,
            k: $k,
            a: $a,
            c: $c,
        }
    };
}

fn main() -> Result<()> {
    use ComponentType::{F16, F32, I8, I32};
    let variants = [
        variant!("int8 -> int32   16x16x32", "coop_rate_i8_16x16x32", 16, 16, 32, I8, I32),
        variant!("int8 -> int32   16x 8x32", "coop_rate_i8_16x8x32", 16, 8, 32, I8, I32),
        variant!("f16 -> f16      16x16x16", "coop_rate_f16f16_16x16x16", 16, 16, 16, F16, F16),
        variant!("f16 -> f16      16x 8x16", "coop_rate_f16f16_16x8x16", 16, 8, 16, F16, F16),
        variant!("f16 -> f32      16x16x16", "coop_rate_f16f32_16x16x16", 16, 16, 16, F16, F32),
        variant!("f16 -> f32      16x 8x16", "coop_rate_f16f32_16x8x16", 16, 8, 16, F16, F32),
    ];
    let only: Option<u32> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        println!("Vulkan{} {}", info.ordinal, info.name);
        let device = Device::open(info.ordinal)?;
        let stream = Stream::new(&device)?;
        let sink = DeviceBuffer::new(&device, 1 << 20)?;
        let (start, end) = (Event::timed(&device)?, Event::timed(&device)?);
        for v in &variants {
            let offered = info
                .cooperative_matrix
                .iter()
                .any(|c| (c.m, c.n, c.k, c.a, c.c) == (v.m, v.n, v.k, v.a, v.c));
            if !offered {
                println!("  {}: not offered", v.name);
                continue;
            }
            let function: Function = Module::load(&device, v.image)?.function("main")?;
            let (groups, iterations) = (4096u32, 2048u32);
            let launch = || -> Result<()> {
                let args = Args::new().ptr(sink.ptr()).u32(iterations);
                // SAFETY: the sink is 1 MiB and is written, if at all, below 4096 * 128 floats... only for an impossible value.
                unsafe { function.launch(&stream, [groups, 1, 1], args) }
            };
            launch()?;
            stream.synchronize()?;
            let mut times = Vec::new();
            for _ in 0..5 {
                start.record(&stream)?;
                launch()?;
                end.record(&stream)?;
                end.synchronize()?;
                times.push(f64::from(end.elapsed_ms_since(&start)?));
            }
            times.sort_by(f64::total_cmp);
            let ms = times[2];
            // 4 subgroups per workgroup, 8 products per iteration, 2 m n k operations each.
            let ops = 2.0
                * f64::from(v.m)
                * f64::from(v.n)
                * f64::from(v.k)
                * 8.0
                * f64::from(iterations)
                * 4.0
                * f64::from(groups);
            println!(
                "  {}: {:7.2} ms  {:7.1} T{}ops",
                v.name,
                ms,
                ops / (ms * 1e-3) / 1e12,
                if matches!(v.a, I8) { "" } else { "FL" }
            );
        }
    }
    Ok(())
}
