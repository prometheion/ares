//! Times the Q4_K matrix-vector kernel on every Vulkan device.
//!
//! Decoding a token reads every weight once, so the speed of a weight kernel is
//! how many bytes of weights it reads per second, compared with the memory
//! bandwidth of the card. This prints, per device, the time per launch and the
//! rate in GB/s of weight bytes for a 17408 x 5120 matrix (the shape of a
//! feed-forward projection of a 27B model, in both orientations) with 1 and 8
//! rows of activations, so the numbers can be set next to the CUDA kernel's.
//! Each shape is timed twice: with the kernel the crate ships, and with a port
//! of the CUDA kernel's thread layout (see `CudaLayout`), which is what a CUDA
//! run of the reference kernel should be compared with.
//!
//! The weights are spread over several copies that the launches rotate through,
//! so every launch reads data that is not in the GPU's L2 cache (36 MiB on an
//! RTX 4070): a repeated launch on one matrix would measure the cache. Each run
//! is a batch of launches followed by one wait, the way a step enqueues work;
//! the figure is the median over the runs.
//!
//! It opens the devices and takes about 350 MiB of video memory on each, so run
//! it only when that is free:
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_matvec_bench [device]
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_vulkan::{
    Args, Device, DeviceBuffer, DevicePtr, Function, Kernels, Module, Result, Stream, devices,
};

/// The port of the CUDA kernel's thread layout, for comparison with the tuned
/// kernel (`shaders/matvec_q4_K_cuda_layout.comp`): one invocation per unit of
/// 64 values, one workgroup per output row, the workgroup as wide as the row
/// has units up to 256. It needs the same `sum32_f32` pass the tuned kernel
/// does.
struct CudaLayout {
    sum32: Function,
    /// Index `i` has `32 * (i + 1)` invocations per workgroup.
    matvec: Vec<Function>,
}

impl CudaLayout {
    fn load(device: &Device) -> Result<Self> {
        let sum32 =
            Module::load(device, include_bytes!(concat!(env!("OUT_DIR"), "/sum32_f32.spv")))?;
        let matvec = Module::load(
            device,
            include_bytes!(concat!(env!("OUT_DIR"), "/matvec_q4_K_cuda_layout.spv")),
        )?;
        Ok(Self {
            sum32: sum32.function("main")?,
            matvec: (1..=8u32)
                .map(|i| matvec.function_with("main", &[(0, 32 * i)]))
                .collect::<Result<_>>()?,
        })
    }

    /// # Safety
    ///
    /// As `Kernels::matvec_q4_k`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch(
        &self,
        stream: &Stream,
        w: DevicePtr,
        x: DevicePtr,
        scratch: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_batch: u32,
    ) -> Result<()> {
        let n_groups = n_batch * (n_in / 32);
        let args = Args::new().ptr(x).ptr(scratch).u32(n_groups);
        // SAFETY: the argument list matches sum32_f32.comp; the buffers are the caller's.
        unsafe { self.sum32.launch(stream, [n_groups.div_ceil(256), 1, 1], args)? };
        let threads = (n_in / 64).next_multiple_of(32).min(256);
        let args = Args::new().ptr(w).ptr(x).ptr(scratch).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
        // SAFETY: the argument list matches the shader's push constants.
        unsafe { self.matvec[(threads / 32 - 1) as usize].launch(stream, [n_out, 1, 1], args) }
    }
}

/// Launches per timed run, and runs per figure.
const LAUNCHES: usize = 50;
const RUNS: usize = 21;
/// Launches before the first run, so pipelines, caches and clocks are warm.
const WARM_UP: usize = 30;
/// The weights are spread over copies until they exceed this many bytes.
const MIN_FOOTPRINT: u64 = 256 << 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random Q4_K blocks with finite scales.
fn random_q4k(rng: &mut Rng, blocks: usize) -> Vec<u8> {
    const SCALES: [u16; 6] = [0x2e66, 0x3266, 0xae66, 0x1c00, 0x3800, 0xb000];
    let mut bytes = Vec::with_capacity(blocks * 144);
    for _ in 0..blocks {
        bytes.extend_from_slice(&SCALES[(rng.next() % 6) as usize].to_le_bytes());
        bytes.extend_from_slice(&SCALES[(rng.next() % 6) as usize].to_le_bytes());
        bytes.extend((0..140).map(|_| (rng.next() >> 32) as u8));
    }
    bytes
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn bench_shape(
    device: &Device,
    kernels: &Kernels,
    reference: &CudaLayout,
    n_in: u32,
    n_out: u32,
) -> Result<()> {
    let blocks = (n_in / 256) as usize * n_out as usize;
    let matrix_bytes = GgmlType::Q4_K.bytes_for(u64::from(n_in) * u64::from(n_out)).unwrap();
    let copies = MIN_FOOTPRINT.div_ceil(matrix_bytes).max(2) as usize;
    println!(
        "  {n_out} rows x {n_in} columns: {:.1} MB of Q4_K weights, {copies} copies",
        matrix_bytes as f64 / 1e6
    );

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let data = random_q4k(&mut rng, blocks);
    let weights: Vec<DeviceBuffer> = (0..copies)
        .map(|_| {
            let buffer = DeviceBuffer::new(device, matrix_bytes)?;
            buffer.write(0, &data)?;
            Ok(buffer)
        })
        .collect::<Result<_>>()?;
    let stream = Stream::new(device)?;

    for (label, tuned) in [("tuned kernel", true), ("CUDA thread layout", false)] {
        for n_batch in [1u32, 8] {
            let x_values: Vec<u8> = (0..n_in * n_batch)
                .flat_map(|i| {
                    (((i * 2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5).to_le_bytes()
                })
                .collect();
            let x = DeviceBuffer::new(device, x_values.len() as u64)?;
            x.write(0, &x_values)?;
            let scratch = DeviceBuffer::new(device, u64::from(n_batch * n_in / 32) * 4)?;
            let y = DeviceBuffer::new(device, u64::from(n_batch * n_out) * 4)?;

            let launch = |i: usize| -> Result<()> {
                let w = weights[i % copies].ptr();
                // SAFETY: the buffers were sized for this shape just above and
                // outlive the synchronisation that follows every batch.
                unsafe {
                    if tuned {
                        kernels.matvec_q4_k(
                            &stream,
                            w,
                            x.ptr(),
                            scratch.ptr(),
                            y.ptr(),
                            n_in,
                            n_out,
                            n_batch,
                        )
                    } else {
                        reference.launch(
                            &stream,
                            w,
                            x.ptr(),
                            scratch.ptr(),
                            y.ptr(),
                            n_in,
                            n_out,
                            n_batch,
                        )
                    }
                }
            };
            for i in 0..WARM_UP {
                launch(i)?;
            }
            stream.synchronize()?;

            let mut seconds = Vec::with_capacity(RUNS);
            for run in 0..RUNS {
                let started = Instant::now();
                for i in 0..LAUNCHES {
                    launch(run * LAUNCHES + i)?;
                }
                stream.synchronize()?;
                seconds.push(started.elapsed().as_secs_f64() / LAUNCHES as f64);
            }
            let (best, worst) = (
                seconds.iter().copied().fold(f64::INFINITY, f64::min),
                seconds.iter().copied().fold(0.0, f64::max),
            );
            let per_launch = median(seconds);
            println!(
                "    {label:<18} {n_batch} row{}: {:8.4} ms/launch  {:7.1} GB/s   (min {:.4}, max {:.4} ms; \
             median of {RUNS} runs of {LAUNCHES} launches)",
                if n_batch == 1 { " " } else { "s" },
                per_launch * 1e3,
                matrix_bytes as f64 / per_launch / 1e9,
                best * 1e3,
                worst * 1e3,
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let only: Option<u32> = std::env::args().nth(1).and_then(|arg| arg.parse().ok());
    for info in devices()? {
        if only.is_some_and(|ordinal| ordinal != info.ordinal) {
            continue;
        }
        let pci = info.pci.map(|p| p.to_string()).unwrap_or_else(|| "no PCI info".into());
        let u = info.uuid;
        let uuid = format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            u[0],
            u[1],
            u[2],
            u[3],
            u[4],
            u[5],
            u[6],
            u[7],
            u[8],
            u[9],
            u[10],
            u[11],
            u[12],
            u[13],
            u[14],
            u[15]
        );
        println!(
            "Vulkan{}: {} ({} MiB, PCI {pci}, UUID {uuid}, subgroup {})",
            info.ordinal,
            info.name,
            info.total_memory >> 20,
            info.subgroup_size
        );
        let device = Device::open(info.ordinal)?;
        let started = Instant::now();
        let kernels = Kernels::load(&device)?;
        println!(
            "  Kernels::load: {:.1} ms (shaders compiled for this GPU)",
            started.elapsed().as_secs_f64() * 1e3
        );
        let reference = CudaLayout::load(&device)?;
        bench_shape(&device, &kernels, &reference, 5120, 17408)?;
        bench_shape(&device, &kernels, &reference, 17408, 5120)?;
    }
    Ok(())
}
