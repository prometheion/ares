//! The two real projectors against what llama.cpp computed for them. These
//! tests need files that are not in the repository and say so when they skip:
//!
//! - `ARES_TEST_QWEN3VL_27B`: `mmproj-BF16.gguf` of Qwen3.8-27B;
//! - `ARES_TEST_QWEN3VL_FLASH`: `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf`;
//! - `ARES_TEST_QWEN3VL_REF`: a directory of what llama.cpp's `llama-mtmd-debug`
//!   wrote on the CPU: `<27b|flash>/<pattern>-<side>-fa<off|on>.bin`
//!   (`[i32 rows][i32 width][f32 ...]`, `MTMD_DEBUG_EMBEDDINGS`), beside each
//!   `.sums` (the header line and the sum of every node of its graph) and
//!   `pixels/<pattern>-<side>.f32` (the square image it was given: three
//!   `f32` a pixel, already normalised);
//! - `ARES_TEST_QWEN3VL_SIDE`: the largest side to run, 512 unless set (the
//!   pass is slow on the CPU);
//! - `ARES_TEST_QWEN3VL_PILLOW`: a directory of resizes made by Pillow, see
//!   `the_resize_matches_pillow`.
//!
//! Run with `--nocapture` to see the differences and the times; with
//! `ARES_TEST_QWEN3VL_PEAKS` set also the largest values of every block, with
//! `ARES_TEST_QWEN3VL_STAGES` every stage's sum beside llama.cpp's.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ares_gguf::Gguf;
use ares_model::qwen3vl::{
    Arithmetic, Config, Limits, Reference, Stage, Weights, resize_bicubic, target_size,
};

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set {name} to run this test");
            None
        }
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// A dump of `MTMD_DEBUG_EMBEDDINGS`: (rows, width, values).
fn read_dump(path: &Path) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let int = |at: usize| i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let values = bytes[8..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    (int(0), int(4), values)
}

/// The sum llama.cpp printed for every named node of its graph.
fn read_sums(path: &Path) -> HashMap<String, f64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut sums = HashMap::new();
    let mut name = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("common_debug_cb_eval:") {
            name = rest.split(" = (").next().map(|n| n.trim().to_owned());
        } else if let (Some(value), Some(node)) = (line.trim().strip_prefix("sum = "), name.take())
        {
            sums.entry(node).or_insert_with(|| value.parse().unwrap());
        }
    }
    sums
}

/// The root of the summed squared difference over that of `b`'s squares, and
/// the largest difference.
fn distance(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut d, mut n, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let diff = f64::from(*x) - f64::from(*y);
        d += diff * diff;
        n += f64::from(*y).powi(2);
        worst = worst.max(diff.abs());
    }
    ((d / n).sqrt(), worst)
}

fn all_finite(what: &str, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()), "{what} has a value that is not finite");
}

/// The cases of one projector up to the largest side: (pattern, side).
fn cases(dir: &Path) -> Vec<(String, usize)> {
    let largest = std::env::var("ARES_TEST_QWEN3VL_SIDE").map_or(512, |s| s.parse().unwrap());
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let name = entry.unwrap().file_name().into_string().unwrap();
        let Some(stem) = name.strip_suffix("-faoff.bin") else { continue };
        let (pattern, side) = stem.rsplit_once('-').unwrap();
        let side: usize = side.parse().unwrap();
        if side <= largest {
            found.push((pattern.to_owned(), side));
        }
    }
    found.sort_by_key(|(pattern, side)| (*side, pattern.clone()));
    assert!(!found.is_empty(), "no dumps in {}", dir.display());
    found
}

/// How far the reference with llama.cpp's roundings may be from llama.cpp's
/// rows without flash attention, as the relative root of the squared
/// difference; see `docs/ARCHITECTURE.md` for where the number comes from.
const FORK_ARITHMETIC: f64 = 2e-2;
/// The same for the exact reference: llama.cpp's CPU path rounds every input
/// of a BF16 matrix to bf16, which the exact reference does not.
const EXACT: f64 = 3e-2;
/// How far the sum of a stage may be from the sum llama.cpp printed, as a
/// part of the sum of the magnitudes.
const STAGE_SUM: f64 = 1e-2;

fn against_llama_cpp(model: &str, tag: &str, n_out: usize) {
    let (Some(path), Some(reference_dir)) = (env_path(model), env_path("ARES_TEST_QWEN3VL_REF"))
    else {
        return;
    };
    let gguf = Gguf::open(path).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    assert_eq!(
        (config.patch, config.merge, config.n_embd, config.n_ff, config.n_layer, config.n_head),
        (16, 2, 1152, 4304, 27, 16)
    );
    assert_eq!((config.n_out, config.table_side, config.head_dim()), (n_out, 48, 72));
    assert_eq!((config.mean, config.std, config.eps), ([0.5; 3], [0.5; 3], 1e-6));
    let weights = Weights::load(&gguf, &config).unwrap();
    let dir = reference_dir.join(tag);

    for (pattern, side) in cases(&dir) {
        let case = format!("{tag} {pattern}-{side}");
        let pixels = read_f32(&reference_dir.join(format!("pixels/{pattern}-{side}.f32")));
        let (rows, width, theirs) = read_dump(&dir.join(format!("{pattern}-{side}-faoff.bin")));
        assert_eq!((rows, width), ((side / 32) * (side / 32), n_out), "{case}");
        all_finite(&format!("{case}: llama.cpp's dump"), &theirs);
        let sums = read_sums(&dir.join(format!("{pattern}-{side}-faoff.sums")));

        // With llama.cpp's roundings: every stage's sum, then the rows.
        let reference = Reference::new(&config, &weights).with_arithmetic(Arithmetic::ForkCpu);
        let mut worst_sum = (0.0f64, String::new());
        let mut compared = 0;
        let started = Instant::now();
        let ours = reference
            .encode_traced(&pixels, side, side, &mut |stage, values| {
                let Some(name) = stage.fork_name(config.n_layer) else { return };
                all_finite(&format!("{case}: {name}"), values);
                let theirs = *sums.get(&name).unwrap_or_else(|| panic!("{case}: no node {name}"));
                // Summed as llama.cpp sums it: one `f32` running sum in order.
                let sum = values.iter().fold(0.0f32, |s, v| s + v);
                let size: f64 = values.iter().map(|v| f64::from(v.abs())).sum();
                let off = (f64::from(sum) - theirs).abs() / size.max(1e-30);
                if std::env::var_os("ARES_TEST_QWEN3VL_STAGES").is_some() {
                    eprintln!("  {name}: sum {sum} llama.cpp {theirs} off {off:.1e}");
                }
                if off > worst_sum.0 {
                    worst_sum = (off, name);
                }
                compared += 1;
            })
            .unwrap();
        let fork_seconds = started.elapsed().as_secs_f64();
        assert_eq!((ours.nx, ours.ny, ours.width), (side / 32, side / 32, n_out), "{case}");
        all_finite(&format!("{case}: the reference with llama.cpp's roundings"), &ours.data);
        assert_eq!(compared, 2 + 11 * config.n_layer + 3, "{case}: stages compared");
        let (fork_rel, fork_max) = distance(&ours.data, &theirs);

        // The pass as defined, with the largest value of every block.
        let reference = Reference::new(&config, &weights);
        let mut peaks = vec![[0.0f32; 6]; config.n_layer];
        let started = Instant::now();
        let exact = reference
            .encode_traced(&pixels, side, side, &mut |stage, values| {
                let (layer, slot) = match stage {
                    Stage::Qkv(l) => (l, 0),
                    Stage::Attention(l) => (l, 1),
                    Stage::AttentionOut(l) => (l, 2),
                    Stage::Hidden(l) => (l, 3),
                    Stage::Mlp(l) => (l, 4),
                    Stage::Block(l) => (l, 5),
                    _ => return,
                };
                let peak = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                peaks[layer][slot] = peaks[layer][slot].max(peak);
            })
            .unwrap();
        let exact_seconds = started.elapsed().as_secs_f64();
        all_finite(&format!("{case}: the exact reference"), &exact.data);
        let (exact_rel, exact_max) = distance(&exact.data, &theirs);
        let largest = theirs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (between, _) = distance(&ours.data, &exact.data);

        eprintln!(
            "{case}: {rows} rows, largest |value| {largest:.2}; with llama.cpp's roundings \
             rel {fork_rel:.2e} max {fork_max:.2e} (worst stage sum {:.1e} at {}), {fork_seconds:.1} s; \
             exact rel {exact_rel:.2e} max {exact_max:.2e}, {exact_seconds:.1} s; the two \
             against each other rel {between:.2e}",
            worst_sum.0, worst_sum.1
        );
        // llama.cpp against itself: the same image with flash attention.
        let flash = dir.join(format!("{pattern}-{side}-faon.bin"));
        if flash.exists() {
            let (_, _, with_flash) = read_dump(&flash);
            let (rel, max) = distance(&with_flash, &theirs);
            let (ours_rel, _) = distance(&exact.data, &with_flash);
            eprintln!(
                "{case}: llama.cpp with flash attention against without: rel {rel:.2e} max \
                 {max:.2e}; exact reference against it: rel {ours_rel:.2e}"
            );
        }
        if std::env::var_os("ARES_TEST_QWEN3VL_PEAKS").is_some() {
            eprintln!(
                "{case}: largest |value| per block: qkv, attention, attention out, hidden, mlp out, residual"
            );
            for (l, p) in peaks.iter().enumerate() {
                eprintln!(
                    "  block {l:2}: {:8.1} {:8.1} {:8.1} {:8.1} {:8.1} {:8.1}",
                    p[0], p[1], p[2], p[3], p[4], p[5]
                );
            }
        }
        let column = |slot: usize| peaks.iter().map(|p| p[slot]).fold(0.0f32, f32::max);
        eprintln!(
            "{case}: largest over the blocks: qkv {:.1}, attention {:.1}, attention out {:.1}, \
             hidden {:.1}, mlp out {:.1}, residual {:.1}",
            column(0),
            column(1),
            column(2),
            column(3),
            column(4),
            column(5)
        );
        assert!(
            worst_sum.0 < STAGE_SUM,
            "{case}: sum of {} is off by {:.2e}",
            worst_sum.1,
            worst_sum.0
        );
        assert!(fork_rel < FORK_ARITHMETIC, "{case}: with llama.cpp's roundings {fork_rel:.2e}");
        assert!(exact_rel < EXACT, "{case}: exact {exact_rel:.2e}");
    }
}

#[test]
fn the_27b_projector_gives_the_rows_of_llama_cpp() {
    against_llama_cpp("ARES_TEST_QWEN3VL_27B", "27b", 5120);
}

#[test]
fn the_flash_next_projector_gives_the_rows_of_llama_cpp() {
    against_llama_cpp("ARES_TEST_QWEN3VL_FLASH", "flash", 2560);
}

/// `ARES_TEST_QWEN3VL_PILLOW` holds `manifest.txt` with one line per case,
/// `name width height out_width out_height differing`, and for each
/// `name.src.rgb` (the bytes fed) and `name.out.rgb`: what Pillow's bicubic
/// resize returned for the columns and then, in a second call, for the rows,
/// which is the order llama.cpp filters in. (One call of Pillow 13 filters
/// the rows first when that leaves the smaller picture between the passes;
/// `differing` counts the bytes in which that result is another.) The output
/// size is the one a second transcription of the size rule gave, with the
/// limits of 8 and 4096 rows.
#[test]
fn the_resize_matches_pillow() {
    let Some(dir) = env_path("ARES_TEST_QWEN3VL_PILLOW") else { return };
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).unwrap();
    let shape = Config {
        patch: 16,
        merge: 2,
        n_embd: 1152,
        n_ff: 4304,
        n_layer: 27,
        n_head: 16,
        eps: 1e-6,
        mean: [0.5; 3],
        std: [0.5; 3],
        n_out: 5120,
        table_side: 48,
    };
    let mut cases = 0;
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let n: Vec<usize> = fields[1..5].iter().map(|f| f.parse().unwrap()).collect();
        let source = std::fs::read(dir.join(format!("{}.src.rgb", fields[0]))).unwrap();
        let expected = std::fs::read(dir.join(format!("{}.out.rgb", fields[0]))).unwrap();
        let size = target_size(&shape, Limits::default(), n[0], n[1]).unwrap();
        assert_eq!(size, (n[2], n[3]), "{line}: the size rule");
        let ours = resize_bicubic(&source, n[0], n[1], n[2], n[3]);
        let differing = ours.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(ours.len(), expected.len(), "{line}");
        assert_eq!(differing, 0, "{line}: {differing} of {} bytes differ", ours.len());
        cases += 1;
    }
    assert!(cases > 0, "no cases in the manifest");
    eprintln!("{cases} resizes equal to Pillow's passes in llama.cpp's order, byte for byte");
}
