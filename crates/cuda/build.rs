//! Compiles the CUDA kernels into fat binaries that the crate embeds.
//!
//! The engine talks to the driver API only (`libcuda.so`, loaded at run
//! time), so the CUDA toolkit is needed to build but not to run. `nvcc`
//! produces machine code for the GPU classes the engine was tested on and for
//! nothing else: no PTX, so no other GPU compiles the kernels on its first
//! load and runs code nobody ran. The crate refuses a device of any other
//! class when it opens it (`ares_cuda::BUILT_CLASSES`); such GPUs are to run
//! through the Vulkan backend.
//!
//! The kernels are in several modules, each a fat binary of its own so that
//! they compile side by side: `kernels/module.cu` (the small kernels and the
//! decode weight kernels), `kernels/prefill.cu` and `prefill_more.cu` (the
//! tensor-core weight kernels), the many-row expert kernels
//! (`kernels/expertrows.cu` and `expertrows_{k,i,l,f}.cu`), the image tower's
//! module (`kernels/tower.cu`) and the others below. Every module is built for
//! the same classes.
//!
//! Environment:
//! - `NVCC`: path of the compiler (default: `$CUDA_PATH/bin/nvcc`,
//!   `$CUDA_HOME/bin/nvcc`, `nvcc` on `PATH`, `/opt/cuda/bin/nvcc`,
//!   `/usr/local/cuda/bin/nvcc`).
//! - `ARES_CUDA_ARCHS`: comma-separated compute capabilities to build
//!   machine code for, for example `86,89,120` (default: `TESTED`). A
//!   developer names a class here to build it on purpose: every named class is
//!   built, the build says which of them are untested, and a class the toolkit
//!   does not list, or one below Ampere (`AMPERE`), is an error.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The compute capabilities the engine was run and tuned on, and the default
/// build: 86 is Ampere (RTX 30), 89 is Ada (RTX 40). Machine code runs only on
/// the class it was built for here: the crate refuses a device whose compute
/// capability is not one of the built classes exactly, so 8.7 (Jetson Orin),
/// which could run 8.6 code, is refused too.
const TESTED: &[u32] = &[86, 89];

/// The lowest compute capability the kernels compile for: the tensor-core
/// kernels use the integer and f16 `mma` shapes, `cp.async` and `ldmatrix` of
/// Ampere.
const AMPERE: u32 = 80;

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

    let archs = classes(&nvcc);
    // The classes the crate runs, for `ares_cuda::BUILT_CLASSES`.
    let list: Vec<String> =
        archs.iter().map(|arch| format!("({}, {})", arch / 10, arch % 10)).collect();
    fs::write(out_dir.join("classes.rs"), format!("&[{}]\n", list.join(", ")))
        .expect("write classes.rs");

    // The modules build side by side: each nvcc run is single-threaded.
    let modules = [
        ("module.cu", "kernels.fatbin"),
        ("prefill.cu", "prefill.fatbin"),
        ("prefill_more.cu", "prefill_more.fatbin"),
        ("layers.cu", "layers.fatbin"),
        ("attention.cu", "attention.fatbin"),
        ("experts.cu", "experts.fatbin"),
        ("expertrows.cu", "expertrows.fatbin"),
        ("expertrows_k.cu", "expertrows_k.fatbin"),
        ("expertrows_i.cu", "expertrows_i.fatbin"),
        ("expertrows_l.cu", "expertrows_l.fatbin"),
        ("expertrows_f.cu", "expertrows_f.fatbin"),
        ("q4.cu", "q4.fatbin"),
        ("q4_reads.cu", "q4_reads.fatbin"),
        ("mmv_exact.cu", "mmv_exact.fatbin"),
        ("tower.cu", "tower.fatbin"),
        ("sampler.cu", "sampler.fatbin"),
        ("dflash.cu", "dflash.fatbin"),
    ];
    let children: Vec<_> = modules
        .iter()
        .map(|(source, fatbin)| {
            let mut cmd = Command::new(&nvcc);
            cmd.args(["-fatbin", "-O3", "-std=c++17", "--no-exceptions"]);
            cmd.arg("-I").arg("kernels");
            for arch in &archs {
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

/// The compute capabilities to build: `TESTED`, or what `ARES_CUDA_ARCHS`
/// names. Each must be one the toolkit lists and at least `AMPERE`; each
/// named class that is not tested gets a build warning.
fn classes(nvcc: &Path) -> Vec<u32> {
    let named = env::var("ARES_CUDA_ARCHS").ok();
    let mut archs: Vec<u32> = match &named {
        Some(list) => list
            .split(',')
            .map(|a| {
                a.trim().parse().unwrap_or_else(|_| panic!("ARES_CUDA_ARCHS: bad entry '{a}'"))
            })
            .collect(),
        None => TESTED.to_vec(),
    };
    archs.sort_unstable();
    archs.dedup();
    if archs.is_empty() {
        panic!("ARES_CUDA_ARCHS names no compute capability");
    }
    let supported = supported_archs(nvcc);
    let class_of = |arch: u32| format!("{}.{}", arch / 10, arch % 10);
    let tested: Vec<String> = TESTED.iter().map(|arch| class_of(*arch)).collect();
    for arch in &archs {
        let class = class_of(*arch);
        if *arch < AMPERE {
            panic!(
                "ARES_CUDA_ARCHS names {arch}: the kernels need Ampere (8.0) or newer; compute \
                 capability {class} is to run through the Vulkan backend"
            );
        }
        if !supported.contains(arch) {
            panic!(
                "{} does not list compute capability {class} (`--list-gpu-arch`); CUDA 13.0 or \
                 newer is needed",
                nvcc.display()
            );
        }
        if !TESTED.contains(arch) {
            println!(
                "cargo:warning=building compute capability {class} as ARES_CUDA_ARCHS names it: \
                 it is untested (the engine was tested on {})",
                tested.join(", ")
            );
        }
    }
    archs
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
