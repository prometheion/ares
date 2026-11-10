//! Compares the reference with what llama.cpp computed (captured by
//! `tools/capture`), or two captures with each other.
//!
//! ```sh
//! cargo run --release -p ares-model --example compare -- MODEL.gguf CAPTURE_DIR [--all-logits]
//! cargo run --release -p ares-model --example compare -- --captures CAPTURE_A CAPTURE_B
//! ```
//!
//! The first form runs the reference on the captured prompt, then feeds it the
//! tokens llama.cpp generated, one at a time, and prints for every block the
//! relative difference of the hidden state, for the logits the same plus the
//! top-two margin. It then lets the reference generate greedily on its own and
//! says where its tokens leave llama.cpp's. The second form prints the same
//! tables between two captures of llama.cpp (for example with another KV cache
//! type or another thread count), which measures how far llama.cpp is from
//! itself.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ares_gguf::Gguf;
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

/// `|a - b| / |b|` over the whole slice.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n).sqrt()
}

fn norm(x: &[f32]) -> f64 {
    x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt()
}

fn top2(x: &[f32]) -> (usize, f32, f32) {
    let (mut best, mut second) = (0usize, f32::NEG_INFINITY);
    for i in 1..x.len() {
        if x[i] > x[best] {
            second = x[best];
            best = i;
        } else if x[i] > second {
            second = x[i];
        }
    }
    (best, x[best], second)
}

struct Capture {
    dir: PathBuf,
}

impl Capture {
    fn f32(&self, name: &str) -> Vec<f32> {
        read_f32(&self.dir.join(name))
    }
}

/// Prints the per-block table of the hidden states.
fn block_table(
    title: &str,
    n_layer: usize,
    n_embd: usize,
    is_attention: impl Fn(usize) -> bool,
    prefill: impl Fn(usize) -> (Vec<f32>, Vec<f32>),
    steps: impl Fn(usize) -> (Vec<f32>, Vec<f32>),
) {
    println!("\n{title}");
    println!(
        "| block | kind | prefill, all rows | prefill, median row | prefill, worst row | last prompt row | generation steps, mean | generation steps, worst |"
    );
    println!("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for l in 0..n_layer {
        let (a, b) = prefill(l);
        let last = a.len() - n_embd;
        let (sa, sb) = steps(l);
        let per_step: Vec<f64> =
            sa.chunks_exact(n_embd).zip(sb.chunks_exact(n_embd)).map(|(x, y)| rel(x, y)).collect();
        let mut rows: Vec<f64> =
            a.chunks_exact(n_embd).zip(b.chunks_exact(n_embd)).map(|(x, y)| rel(x, y)).collect();
        rows.sort_by(f64::total_cmp);
        println!(
            "| {l} | {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
            if is_attention(l) { "attn" } else { "delta" },
            rel(&a, &b),
            rows[rows.len() / 2],
            rows[rows.len() - 1],
            rel(&a[last..], &b[last..]),
            per_step.iter().sum::<f64>() / per_step.len() as f64,
            per_step.iter().copied().fold(0.0, f64::max),
        );
    }
}

fn captures_against(a: &Path, b: &Path) {
    let (a, b) = (Capture { dir: a.into() }, Capture { dir: b.into() });
    let n_layer = 24;
    let n_embd = 1024;
    block_table(
        &format!("{} against {}", a.dir.display(), b.dir.display()),
        n_layer,
        n_embd,
        |l| (l + 1) % 4 == 0,
        |l| (a.f32(&format!("prefill.l_out-{l}.f32")), b.f32(&format!("prefill.l_out-{l}.f32"))),
        |l| (a.f32(&format!("steps.l_out-{l}.f32")), b.f32(&format!("steps.l_out-{l}.f32"))),
    );
    logits_table(&a.f32("steps.logits.f32"), &b.f32("steps.logits.f32"), 248320, "second");
    let (ga, gb) =
        (read_ids(&a.dir.join("generated.tokens")), read_ids(&b.dir.join("generated.tokens")));
    match ga.iter().zip(&gb).position(|(x, y)| x != y) {
        None => println!("\ngenerated tokens: identical ({})", ga.len()),
        Some(i) => println!("\ngenerated tokens: first difference at step {i}"),
    }
}

/// Logits of the steps, row by row: relative difference, and where the top
/// token and the margin to the second one stand.
fn logits_table(a: &[f32], b: &[f32], n_vocab: usize, what: &str) {
    println!(
        "\nlogits per step (row 0 is the last prompt token), this is the {what} against the other:"
    );
    println!(
        "| step | relative difference | max abs difference | top token (other / this) | margin of the other | margin of this |"
    );
    println!("| ---: | ---: | ---: | --- | ---: | ---: |");
    let mut worst: f64 = 0.0;
    let mut flips = Vec::new();
    for (s, (x, y)) in a.chunks_exact(n_vocab).zip(b.chunks_exact(n_vocab)).enumerate() {
        let (tx, vx, sx) = top2(x);
        let (ty, vy, sy) = top2(y);
        let max_abs = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        worst = worst.max(rel(x, y));
        if tx != ty {
            flips.push(s);
        }
        if s < 3 || s % 8 == 0 || tx != ty {
            println!(
                "| {s} | {:.2e} | {max_abs:.2e} | {ty} / {tx} | {:.3} | {:.3} |",
                rel(x, y),
                vy - sy,
                vx - sx
            );
        }
    }
    println!(
        "\nworst relative difference over all steps: {worst:.2e}; steps where the top token differs: {flips:?}"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--captures") {
        captures_against(Path::new(&args[1]), Path::new(&args[2]));
        return;
    }
    let (model_path, cap_dir) = (&args[0], Path::new(&args[1]));
    let all_logits = args.iter().any(|a| a == "--all-logits");

    let gguf = Gguf::open(model_path).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    let reference = Reference::new(&config, &weights);
    let cap = Capture { dir: cap_dir.into() };

    let prompt = read_ids(&cap.dir.join("prompt.tokens"));
    let generated = read_ids(&cap.dir.join("generated.tokens"));
    let (n_embd, n_vocab) = (config.n_embd, config.n_vocab);
    println!("prompt of {} tokens, {} generated by llama.cpp", prompt.len(), generated.len());

    // Prefill.
    let mut state = SequenceState::new(&config);
    let started = Instant::now();
    let rows = if all_logits { LogitRows::All } else { LogitRows::Last };
    let pre = reference.forward(&mut state, &prompt, rows).unwrap();
    let prefill_time = started.elapsed();
    println!(
        "prefill: {:.2} s, {:.1} tokens/s",
        prefill_time.as_secs_f64(),
        prompt.len() as f64 / prefill_time.as_secs_f64()
    );

    // Teacher forcing: the reference is fed the tokens llama.cpp picked.
    let mut step_hidden: Vec<Vec<f32>> = vec![Vec::new(); config.n_layer];
    let mut step_logits: Vec<f32> = pre.logits_of(prompt.len() - 1).unwrap().to_vec();
    let free_state = state.clone();
    let started = Instant::now();
    for &token in &generated {
        let w = reference.forward(&mut state, &[token], LogitRows::Last).unwrap();
        for (l, h) in step_hidden.iter_mut().enumerate() {
            h.extend_from_slice(w.hidden_row(l, 0));
        }
        step_logits.extend_from_slice(w.logits_of(0).unwrap());
    }
    let step_time = started.elapsed();
    println!(
        "generation: {:.1} ms per token",
        step_time.as_secs_f64() * 1e3 / generated.len() as f64
    );

    block_table(
        "hidden state after every block, reference against llama.cpp (relative L2 difference)",
        config.n_layer,
        n_embd,
        |l| config.is_attention(l),
        |l| (pre.hidden[l].clone(), cap.f32(&format!("prefill.l_out-{l}.f32"))),
        |l| {
            // llama.cpp's steps include one row more than we fed (the last token is
            // generated, not fed); `steps.l_out` has one row per fed token.
            (step_hidden[l].clone(), cap.f32(&format!("steps.l_out-{l}.f32")))
        },
    );
    // Block by block from llama.cpp's own hidden states: the reference runs each
    // block on what llama.cpp fed it, so the differences do not accumulate.
    println!(
        "\nevery block alone, fed with the hidden state llama.cpp gave it (prompt rows, no earlier state)"
    );
    println!(
        "| block | kind | all rows | median row | worst row | last prompt row | relative to what the block adds |"
    );
    println!("| ---: | --- | ---: | ---: | ---: | ---: | ---: |");
    let empty = SequenceState::new(&config);
    for l in 0..config.n_layer {
        let input: Vec<f32> = if l == 0 {
            let mut x = vec![0.0f32; prompt.len() * n_embd];
            for (row, &t) in prompt.iter().enumerate() {
                weights.token_embd.decode_row(t as usize, &mut x[row * n_embd..(row + 1) * n_embd]);
            }
            x
        } else {
            cap.f32(&format!("prefill.l_out-{}.f32", l - 1))
        };
        let out = reference.block(l, &empty, &input);
        let theirs = cap.f32(&format!("prefill.l_out-{l}.f32"));
        let mut rows: Vec<f64> = out
            .chunks_exact(n_embd)
            .zip(theirs.chunks_exact(n_embd))
            .map(|(x, y)| rel(x, y))
            .collect();
        let last = rows[rows.len() - 1];
        rows.sort_by(f64::total_cmp);
        // The same difference measured against the block's update, theirs - input.
        let update: Vec<f32> = theirs.iter().zip(&input).map(|(t, i)| t - i).collect();
        let diff: Vec<f32> = out.iter().zip(&theirs).map(|(o, t)| o - t).collect();
        println!(
            "| {l} | {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
            if config.is_attention(l) { "attn" } else { "delta" },
            rel(&out, &theirs),
            rows[rows.len() / 2],
            rows[rows.len() - 1],
            last,
            norm(&diff) / norm(&update),
        );
    }

    let (final_a, final_b) = (&pre.final_norm, cap.f32("prefill.result_norm.f32"));
    println!(
        "\nfinal norm, prefill: {:.2e} over all rows, {:.2e} at the last row",
        rel(final_a, &final_b),
        rel(&final_a[final_a.len() - n_embd..], &final_b[final_b.len() - n_embd..])
    );

    let theirs = cap.f32("steps.logits.f32");
    logits_table(&step_logits, &theirs, n_vocab, "reference");

    if all_logits {
        let picks: Vec<i32> =
            cap.f32("prefill.argmax.i32").iter().map(|v| v.to_bits() as i32).collect();
        let mut differ = 0;
        for (row, &pick) in picks.iter().enumerate() {
            let (t, _, _) = top2(pre.logits_of(row).unwrap());
            differ += usize::from(t as i32 != pick);
        }
        println!("\nprompt rows whose greedy pick differs: {differ} of {}", prompt.len());
    }

    // Free run: the reference picks its own tokens.
    let mut state = free_state;
    let mut logits = pre.logits_of(prompt.len() - 1).unwrap().to_vec();
    let mut own = Vec::new();
    for _ in 0..generated.len() {
        let (t, _, _) = top2(&logits);
        own.push(t as u32);
        let w = reference.forward(&mut state, &[t as u32], LogitRows::Last).unwrap();
        logits = w.logits_of(0).unwrap().to_vec();
    }
    match own.iter().zip(&generated).position(|(a, b)| a != b) {
        None => {
            println!("\ngreedy tokens of the reference: identical to llama.cpp's ({})", own.len())
        }
        Some(i) => {
            let (_, v1, v2) = top2(&theirs[i * n_vocab..(i + 1) * n_vocab]);
            println!(
                "\ngreedy tokens of the reference: first difference at step {i} (reference {}, llama.cpp {}); \
                 llama.cpp's margin there is {:.3}",
                own[i],
                generated[i],
                v1 - v2
            );
        }
    }
}
