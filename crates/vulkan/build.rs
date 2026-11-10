//! Compiles the GLSL compute shaders in `shaders/` to SPIR-V that the crate
//! embeds.
//!
//! The engine talks to Vulkan through the loader, which is loaded at run time,
//! so the shader compiler is needed to build and not to run. Every
//! `shaders/*.comp` becomes `$OUT_DIR/<name>.spv`; `shaders/*.glsl` are
//! includes and are not compiled on their own.
//!
//! Environment:
//! - `GLSLC`: path of the compiler (default: `glslc` on `PATH`, then
//!   `$VULKAN_SDK/bin/glslc`, `/usr/bin/glslc`, `/usr/local/bin/glslc`).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Output rows of the matrix kernel that one workgroup computes. The shader
/// takes it as `ARES_MATVEC_ROWS`; the host divides the launch by it. Defined
/// here so that the two cannot disagree.
const MATVEC_ROWS: u32 = 4;

/// The launch shape of the quantized weight kernels (`ARES_MMVQ_WARPS` and
/// `ARES_MMVQ_ROWS` in `shaders/mmvq.glsl`): a workgroup of this many warps of 32
/// lanes, each warp computing this many output rows. The same two numbers as the
/// CUDA kernels' `ARES_MMVQ_WARPS` / `ARES_MMVQ_ROWS`; defined here so that the
/// shaders and the launch cannot disagree.
const MMVQ_WARPS: u32 = 4;
const MMVQ_ROWS: u32 = 2;

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let glslc = find_glslc().unwrap_or_else(|| {
        panic!(
            "glslc was not found. The shader compiler from the Vulkan SDK (or the shaderc \
             package) is needed to build the kernels; set GLSLC to the compiler."
        )
    });

    fs::write(
        out.join("shader_constants.rs"),
        format!(
            "/// Output rows per workgroup of `matvec_q4_K`.\n\
             #[allow(dead_code)]\n\
             pub(crate) const MATVEC_ROWS: u32 = {MATVEC_ROWS};\n\
             /// Threads and output rows per workgroup of the quantized weight kernels.\n\
             #[allow(dead_code)]\n\
             pub(crate) const MMVQ_THREADS: u32 = {};\n\
             #[allow(dead_code)]\n\
             pub(crate) const MMVQ_ROWS_PER_BLOCK: u32 = {};\n",
            MMVQ_WARPS * 32,
            MMVQ_WARPS * MMVQ_ROWS
        ),
    )
    .expect("write shader_constants.rs");

    let mut sources: Vec<PathBuf> = fs::read_dir("shaders")
        .expect("the shaders directory")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "comp"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "shaders/ holds no .comp file");

    for source in sources {
        let name = source.file_stem().expect("a file name").to_string_lossy().into_owned();
        let target = out.join(format!("{name}.spv"));
        let mut cmd = Command::new(&glslc);
        // Vulkan 1.2 is the baseline of the crate: buffer device addresses,
        // timeline semaphores and subgroup operations are all core there.
        // -Werror makes a shader warning fail the build instead of scrolling by.
        cmd.args(["--target-env=vulkan1.2", "-fshader-stage=compute", "-O", "-Werror"]);
        cmd.arg("-I").arg("shaders");
        cmd.arg(format!("-DARES_MATVEC_ROWS={MATVEC_ROWS}"));
        cmd.arg(format!("-DARES_MMVQ_WARPS={MMVQ_WARPS}"));
        cmd.arg(format!("-DARES_MMVQ_ROWS={MMVQ_ROWS}"));
        cmd.arg("-o").arg(&target).arg(&source);
        let output = cmd.output().unwrap_or_else(|e| panic!("cannot run {}: {e}", glslc.display()));
        if !output.status.success() {
            panic!(
                "glslc failed on {} ({}):\n{}{}",
                source.display(),
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

fn find_glslc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("GLSLC") {
        return Some(PathBuf::from(path));
    }
    let mut candidates = Vec::new();
    if let Some(paths) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&paths).map(|dir| dir.join("glslc")));
    }
    if let Some(root) = env::var_os("VULKAN_SDK") {
        candidates.push(Path::new(&root).join("bin/glslc"));
    }
    candidates.push(PathBuf::from("/usr/bin/glslc"));
    candidates.push(PathBuf::from("/usr/local/bin/glslc"));
    candidates.into_iter().find(|path| path.is_file())
}
