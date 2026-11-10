//! Compiles the GLSL compute shaders in `shaders/` to SPIR-V that the crate
//! embeds.
//!
//! The engine talks to Vulkan through the loader, which is loaded at run time,
//! so the shader compiler is needed to build and not to run. Every
//! `shaders/*.comp` becomes `$OUT_DIR/<name>.spv`; `shaders/*.glsl` are
//! includes and are not compiled on their own. The compilations run on all cores.
//!
//! The weight kernels (`mmvq_*.comp`, `mmvf_*.comp`) are built in a shape per kernel (see
//! `Shape`): output rows per warp, one shader for each number of activation rows (1 to 8,
//! `-DARES_NB`) instead of one that cuts a loop of 8 by a branch, and blocks per trip of the
//! loop (`-DARES_UNROLL`). `$OUT_DIR/shader_images.rs` lists every shader with its variants
//! for the host.
//!
//! Environment:
//! - `GLSLC`: path of the compiler (default: `glslc` on `PATH`, then
//!   `$VULKAN_SDK/bin/glslc`, `/usr/bin/glslc`, `/usr/local/bin/glslc`).
//! - `ARES_SWEEP=rows,per_batch,unroll` (for example `4,1,2`): builds every weight kernel in
//!   that shape, for measuring the choices with `vk_mmvq_bench`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Output rows of the matrix kernel that one workgroup computes. The shader
/// takes it as `ARES_MATVEC_ROWS`; the host divides the launch by it. Defined
/// here so that the two cannot disagree.
const MATVEC_ROWS: u32 = 4;

/// The launch shape of the quantized weight kernels (`ARES_MMVQ_WARPS` and
/// `ARES_MMVQ_ROWS` in `shaders/mmvq.glsl`): a workgroup of this many warps of 32
/// lanes, each warp computing `rows` output rows. The same numbers as the CUDA kernels'
/// `ARES_MMVQ_WARPS` / `ARES_MMVQ_ROWS`; defined here so that the shaders and the
/// launch cannot disagree.
const MMVQ_WARPS: u32 = 4;

/// How one weight kernel is built. Measured per type on the two development cards
/// (`vk_mmvq_bench`), see `docs/ARCHITECTURE.md`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Output rows per warp.
    rows: u32,
    /// One shader per number of activation rows (1 to 8), each with its loop over the
    /// rows unrolled to exactly that many, instead of one shader that cuts a loop of 8
    /// by a branch.
    per_batch: bool,
    /// Blocks of a row that a lane loads before it uses any (1, 2 or 4).
    unroll: u32,
}

/// What fits most types on both development cards: 4 rows per warp, a shader per number of
/// activation rows, no unrolling over blocks.
const DEFAULT_SHAPE: Shape = Shape { rows: 4, per_batch: true, unroll: 1 };

/// The kernels for which another shape was faster, as (name, rows, per_batch, unroll). Chosen
/// by `vk_mmvq_bench` over rows in {2, 4, 8} x unroll in {1, 2} with a shader per batch (and
/// the original 2 rows with the loop cut by a branch), on an RTX 4070 and an RTX 3060: the
/// shape with the least mean ratio to the best of the shapes over 1, 2, 4 and 8 rows and both
/// cards; for these the default was 1.5 to 7 % worse on that measure.
const SHAPES: [(&str, u32, bool, u32); 14] = [
    ("mmvq_q5_K", 8, true, 1),
    ("mmvq_q6_K", 4, true, 2),
    ("mmvq_q3_K", 4, true, 2),
    ("mmvq_q8_0", 2, true, 2),
    ("mmvq_q4_0", 8, true, 1),
    ("mmvq_q4_1", 4, true, 2),
    ("mmvq_q5_0", 8, true, 1),
    ("mmvq_q5_1", 4, true, 2),
    ("mmvq_q2_K", 8, true, 1),
    ("mmvq_tq2_0", 8, true, 2),
    ("mmvq_mxfp4", 4, true, 2),
    ("mmvf_f32", 2, false, 1),
    ("mmvf_f16", 4, true, 2),
    ("mmvf_bf16", 4, true, 2),
];

/// The shape of the kernel `name` (`mmvq_q4_K`, ...). `ARES_SWEEP=rows,per_batch,unroll`
/// (for example `4,1,0`) gives it to every kernel, for measuring the choices.
fn shape_of(name: &str) -> Shape {
    if let Ok(sweep) = env::var("ARES_SWEEP") {
        let v: Vec<u32> = sweep.split(',').map(|x| x.trim().parse().expect("ARES_SWEEP")).collect();
        return Shape { rows: v[0], per_batch: v[1] != 0, unroll: v[2] };
    }
    SHAPES
        .iter()
        .find(|(n, ..)| *n == name)
        .map_or(DEFAULT_SHAPE, |&(_, rows, per_batch, unroll)| Shape { rows, per_batch, unroll })
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=ARES_SWEEP");
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
             /// Threads per workgroup of the quantized weight kernels.\n\
             #[allow(dead_code)]\n\
             pub(crate) const MMVQ_THREADS: u32 = {};\n\
             /// Warps per workgroup of the quantized weight kernels.\n\
             #[allow(dead_code)]\n\
             pub(crate) const MMVQ_WARPS: u32 = {MMVQ_WARPS};\n",
            MMVQ_WARPS * 32,
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

    // The weight kernels come in variants (see `Shape`); everything else is one shader.
    let mut images = String::from(
        "pub(crate) struct Image {\n    pub name: &'static str,\n    pub rows: u32,\n    \
         pub variants: &'static [&'static [u8]],\n}\n\n\
         pub(crate) static IMAGES: &[Image] = &[\n",
    );
    // The compilations are independent: collect them, run them on all cores.
    struct Job {
        source: PathBuf,
        target: PathBuf,
        rows: u32,
        batch: u32,
        unroll: u32,
    }
    let mut jobs = Vec::new();
    let mut listing = Vec::new();
    for source in sources {
        let name = source.file_stem().expect("a file name").to_string_lossy().into_owned();
        let weight = name.starts_with("mmvq_") || name.starts_with("mmvf_");
        let shape = if weight { shape_of(&name) } else { DEFAULT_SHAPE };
        let batches: Vec<u32> = if shape.per_batch { (1..=8).collect() } else { vec![0] };
        let mut variants = Vec::new();
        for n in batches {
            let file = if weight { format!("{name}_b{n}") } else { name.clone() };
            jobs.push(Job {
                source: source.clone(),
                target: out.join(format!("{file}.spv")),
                rows: shape.rows,
                batch: n,
                unroll: shape.unroll,
            });
            variants.push(file);
        }
        listing.push((name, shape.rows, variants));
    }
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let threads = thread::available_parallelism().map_or(1, |n| n.get()).min(jobs.len());
    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    if failure.lock().expect("lock").is_some() {
                        break;
                    }
                    let mut cmd = Command::new(&glslc);
                    // Vulkan 1.2 is the baseline of the crate: buffer device addresses,
                    // timeline semaphores and subgroup operations are all core there.
                    // -Werror makes a shader warning fail the build instead of scrolling by.
                    cmd.args(["--target-env=vulkan1.2", "-fshader-stage=compute", "-O", "-Werror"]);
                    cmd.arg("-I").arg("shaders");
                    cmd.arg(format!("-DARES_MATVEC_ROWS={MATVEC_ROWS}"));
                    cmd.arg(format!("-DARES_MMVQ_WARPS={MMVQ_WARPS}"));
                    cmd.arg(format!("-DARES_MMVQ_ROWS={}", job.rows));
                    cmd.arg(format!("-DARES_NB={}", job.batch));
                    cmd.arg(format!("-DARES_UNROLL={}", job.unroll));
                    cmd.arg("-o").arg(&job.target).arg(&job.source);
                    let output = cmd
                        .output()
                        .unwrap_or_else(|e| panic!("cannot run {}: {e}", glslc.display()));
                    if !output.status.success() {
                        *failure.lock().expect("lock") = Some(format!(
                            "glslc failed on {} ({}):\n{}{}",
                            job.source.display(),
                            output.status,
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        ));
                    }
                }
            });
        }
    });
    if let Some(message) = failure.into_inner().expect("lock") {
        panic!("{message}");
    }
    for (name, rows, variants) in listing {
        images.push_str(&format!(
            "    Image {{ name: \"{name}\", rows: {rows}, variants: &[{}] }},\n",
            variants
                .iter()
                .map(|v| format!(
                    "include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{v}.spv\")) as &[u8]"
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    images.push_str("];\n");
    fs::write(out.join("shader_images.rs"), images).expect("write shader_images.rs");
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
