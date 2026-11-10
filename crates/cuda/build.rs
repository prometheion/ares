//! Compiles the CUDA kernels into two fat binaries that the crate embeds.
//!
//! The engine talks to the driver API only (`libcuda.so`, loaded at run
//! time), so the CUDA toolkit is needed to build but not to run. `nvcc`
//! produces machine code for the GPU generations listed in `ARCHS` (each only
//! if the toolkit supports it) and PTX for the lowest compute capability the
//! toolkit can target, which the driver compiles on first load for any GPU
//! that has no exact match.
//!
//! How far down the engine reaches is decided by the toolkit that builds it:
//! CUDA 13.4 lists `compute_75` (Turing) as its lowest target, so a build here
//! gets PTX from 75 on, while a toolkit that still lists Pascal (`compute_61`)
//! produces PTX that a GTX 10 card can use. Compute capability 6.1 is the
//! floor, because the weight kernels use `__dp4a`; targets below it are not
//! built, and there is no fallback code for GPUs the kernels cannot run on.
//!
//! The kernels are in two modules. `kernels/module.cu` has everything that runs
//! on every GPU from compute capability 6.1: the small kernels and the decode
//! weight kernels. `kernels/prefill.cu` has the tensor-core weight kernels for
//! prompt processing, which use the `mma` shapes and `cp.async` of Ampere and
//! are therefore built for compute capability 8.0 and newer only (PTX for the
//! lowest such target the toolkit lists, 8.0 with CUDA 13.4). A GPU below that
//! loads the first module and not the second; there are no prefill kernels
//! for it and nothing stands in for them.
//!
//! Environment:
//! - `NVCC`: path of the compiler (default: `$CUDA_PATH/bin/nvcc`,
//!   `$CUDA_HOME/bin/nvcc`, `nvcc` on `PATH`, `/opt/cuda/bin/nvcc`,
//!   `/usr/local/cuda/bin/nvcc`).
//! - `ARES_CUDA_ARCHS`: comma-separated compute capabilities to build
//!   machine code for, for example `86,89` (default: every entry of `ARCHS`
//!   the compiler supports). Fewer entries build faster.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Compute capabilities that get their own machine code, each only if the
/// toolkit supports it. 75 is Turing (GTX 16, RTX 20), the oldest generation
/// CUDA 13.4 can target. 86 covers RTX 30 (and runs on 87 and 89), 89 is RTX
/// 40, 120 is RTX 50.
const ARCHS: &[u32] = &[75, 86, 89, 120];

/// The lowest compute capability the kernels can be built for: `__dp4a`, the
/// integer dot product of the weight kernels, starts at 6.1.
const FLOOR: u32 = 61;

/// The lowest compute capability of the prefill module: the integer
/// `mma.m16n8k32`, the f16 `mma.m16n8k16` and `cp.async` start with Ampere.
const TENSOR_FLOOR: u32 = 80;

/// The lowest compute capability the toolkit lists that the kernels compile
/// for as they are (`floor` and up): the PTX baseline. It is 75 for the first
/// module and 80 for the second with CUDA 13.4, the toolkit of the development
/// machine.
fn baseline(supported: &[u32], floor: u32) -> Option<u32> {
    supported.iter().copied().filter(|arch| *arch >= floor).min()
}

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=ARES_CUDA_ARCHS");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "nvcc was not found. The CUDA toolkit (13.0 or newer) is needed to build the \
             kernels; set NVCC to the compiler or CUDA_PATH to the toolkit."
        )
    });

    let supported = supported_archs(&nvcc);
    let wanted: Vec<u32> = match env::var("ARES_CUDA_ARCHS") {
        Ok(list) => list
            .split(',')
            .map(|a| {
                a.trim().parse().unwrap_or_else(|_| panic!("ARES_CUDA_ARCHS: bad entry '{a}'"))
            })
            .collect(),
        Err(_) => ARCHS.to_vec(),
    };
    // The two modules build side by side: each nvcc run is single-threaded and
    // the prefill module is the slower one.
    let modules = [
        ("module.cu", "kernels.fatbin", FLOOR),
        ("prefill.cu", "prefill.fatbin", TENSOR_FLOOR),
        ("layers.cu", "layers.fatbin", FLOOR),
        ("attention.cu", "attention.fatbin", TENSOR_FLOOR),
    ];
    let children: Vec<_> = modules
        .iter()
        .map(|(source, fatbin, floor)| {
            let Some(baseline) = baseline(&supported, *floor) else {
                panic!(
                    "{} lists no compute capability of {}.{} or newer for {source} (`--list-gpu-arch` \
                     printed nothing usable); CUDA 13.0 or newer is needed",
                    nvcc.display(),
                    floor / 10,
                    floor % 10
                );
            };
            let mut cmd = Command::new(&nvcc);
            cmd.args(["-fatbin", "-O3", "-std=c++17", "--no-exceptions", "-Wno-deprecated-gpu-targets"]);
            cmd.arg("-I").arg("kernels");
            // PTX for the baseline: any GPU without exact machine code compiles it.
            cmd.arg("-gencode").arg(format!("arch=compute_{baseline},code=compute_{baseline}"));
            for arch in wanted.iter().filter(|a| supported.contains(a) && **a >= *floor) {
                cmd.arg("-gencode").arg(format!("arch=compute_{arch},code=sm_{arch}"));
            }
            cmd.arg("-o").arg(out_dir.join(fatbin)).arg(Path::new("kernels").join(source));
            let child = cmd
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap_or_else(|e| panic!("cannot run {}: {e}", nvcc.display()));
            (*source, child)
        })
        .collect();
    for (source, child) in children {
        let output = child.wait_with_output().expect("nvcc");
        if !output.status.success() {
            panic!(
                "nvcc failed on {source} ({}):\n{}{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("NVCC") {
        return Some(PathBuf::from(path));
    }
    let mut candidates = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = env::var_os(var) {
            candidates.push(Path::new(&root).join("bin/nvcc"));
        }
    }
    if let Some(paths) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&paths).map(|dir| dir.join("nvcc")));
    }
    candidates.push(PathBuf::from("/opt/cuda/bin/nvcc"));
    candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    candidates.into_iter().find(|path| path.is_file())
}

/// The compute capabilities `nvcc --list-gpu-arch` reports (`compute_86` -> 86).
fn supported_archs(nvcc: &Path) -> Vec<u32> {
    let output = Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", nvcc.display()));
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .filter_map(|line| line.strip_prefix("compute_")?.parse().ok())
        .collect()
}
