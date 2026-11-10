//! Runs the `qwen4exp` reference on a real model file and writes a routing
//! trace: for every token and layer the experts the router picked, in rank
//! order, with their weights. The trace is what expert tiers and reads from
//! the file are replayed against.
//!
//! ```sh
//! q4trace run MODEL.gguf OUT_DIR NAME TEXT_FILE [--chat] [--gen N] [--window W] [--threads T]
//!             [--probe-rule]
//! q4trace stats FILE.q4route...
//! q4trace tap MODEL.gguf TAP_FILE TOKENS_FILE
//! ```
//!
//! `run` reads `TEXT_FILE` (UTF-8), with `--chat` wrapped as one user turn of
//! the model's chat format, runs it through the reference in windows of `W`
//! rows (default 32) and then generates `N` tokens greedily, one per window.
//! It writes `OUT_DIR/NAME.q4route` (rewritten after every window, so a run
//! that is stopped leaves a valid shorter trace), `NAME.tokens.txt` (the ids)
//! and `NAME.txt` (the text of all tokens), and with `--gen` the logits of the
//! last prompt row as `NAME.prompt-logits.f32`. `--probe-rule` runs the first
//! window past the dense limit of the indexer under both selection rules
//! (difference 1 of the specification) and prints how far their logits are
//! apart; the trace itself always follows the specification's rule.
//!
//! `tap` compares the reference with a file Strata wrote with
//! `STRATA_TAP_FILE`: a header (`"STAP"`, `u32` version 1, `u32` N, `u32`
//! number of tapped layers, then the layers as `i32`), then per row of every
//! verify window `i32` position, `i32` token and for each tapped layer `N`
//! floats, the mean of the residual streams after the layer. `TOKENS_FILE`
//! holds the whole sequence, prompt and continuation. Rows of drafts that
//! were rejected are
//! skipped: a row counts when its token and those of the rows before it in
//! its window are the sequence's.
//!
//! The format of a trace, all little endian:
//!
//! ```text
//! offset 0   8 bytes  "Q4ROUTE1"
//!        8   u32      version, 1
//!        12  u32      L, layers
//!        16  u32      E, experts per layer
//!        20  u32      K, experts per token and layer
//!        24  u32      T, tokens in the trace
//!        28  u32      P, how many of them are the given text (the rest were generated)
//!        32  u32      record size in bytes: K * 2 + K * 4
//!        36  28 bytes zero
//!        64  T * u32  the token ids
//!        64 + 4 T     T * L records, token-major: record (t, l) at 64 + 4 T + (t * L + l) * size
//! record:    K * u16  expert ids, the largest weight first (equal weights: lower id first)
//!            K * f32  their weights, which sum to 1
//! ```

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ares_gguf::Gguf;
use ares_model::qwen4exp::{
    Config, LogitRows, Observer, Reference, SelectionRule, SequenceState, Weights, Window,
};
use ares_tokenizer::{EncodeOptions, Tokenizer};

type AnyError = Box<dyn std::error::Error>;

const MAGIC: &[u8; 8] = b"Q4ROUTE1";

/// A routing trace in memory.
struct Trace {
    n_layer: usize,
    n_expert: usize,
    k: usize,
    n_prompt: usize,
    tokens: Vec<u32>,
    /// `[token][layer][K]`.
    ids: Vec<u16>,
    weights: Vec<f32>,
}

impl Trace {
    fn push(&mut self, window: &Window) {
        let rows = window.rows();
        self.tokens.extend_from_slice(&window.tokens);
        for row in 0..rows {
            for routing in &window.routing {
                self.ids.extend(
                    routing.ids[row * self.k..(row + 1) * self.k].iter().map(|&i| i as u16),
                );
                self.weights.extend_from_slice(&routing.weights[row * self.k..(row + 1) * self.k]);
            }
        }
    }

    fn write(&self, path: &Path) -> std::io::Result<()> {
        let mut out = Vec::with_capacity(64 + self.tokens.len() * (4 + self.n_layer * self.k * 6));
        out.extend_from_slice(MAGIC);
        let header = [
            1,
            self.n_layer as u32,
            self.n_expert as u32,
            self.k as u32,
            self.tokens.len() as u32,
            self.n_prompt as u32,
            (self.k * 6) as u32,
        ];
        for value in header {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.resize(64, 0);
        for token in &self.tokens {
            out.extend_from_slice(&token.to_le_bytes());
        }
        for (ids, weights) in self.ids.chunks_exact(self.k).zip(self.weights.chunks_exact(self.k)) {
            for id in ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
            for weight in weights {
                out.extend_from_slice(&weight.to_le_bytes());
            }
        }
        std::fs::write(path, out)
    }

    fn read(path: &Path) -> Result<Self, AnyError> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 64 || &bytes[..8] != MAGIC {
            return Err(format!("{} is not a routing trace", path.display()).into());
        }
        let u32_at = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
        };
        let (n_layer, n_expert, k, t, n_prompt) =
            (u32_at(12), u32_at(16), u32_at(20), u32_at(24), u32_at(28));
        let tokens = (0..t).map(|i| u32_at(64 + 4 * i) as u32).collect();
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        let mut at = 64 + 4 * t;
        for _ in 0..t * n_layer {
            ids.extend(
                (0..k).map(|j| u16::from_le_bytes([bytes[at + 2 * j], bytes[at + 2 * j + 1]])),
            );
            at += 2 * k;
            weights.extend((0..k).map(|j| f32::from_bits(u32_at(at + 4 * j) as u32)));
            at += 4 * k;
        }
        Ok(Self { n_layer, n_expert, k, n_prompt, tokens, ids, weights })
    }

    fn picks(&self, token: usize, layer: usize) -> &[u16] {
        let at = (token * self.n_layer + layer) * self.k;
        &self.ids[at..at + self.k]
    }

    /// Prints how the picks of consecutive tokens overlap and how many
    /// experts the trace touches.
    fn stats(&self, name: &str) {
        let (t, l, k) = (self.tokens.len(), self.n_layer, self.k);
        if t < 2 {
            return;
        }
        println!(
            "{name}: {t} tokens ({} given, {} generated), {l} layers, {k} of {} experts",
            self.n_prompt,
            t - self.n_prompt,
            self.n_expert
        );
        // Per layer: the mean share of a token's picks that the token before it also
        // picked, and the distinct experts over the whole trace.
        let (mut overlap, mut distinct) = (vec![0.0f64; l], vec![0usize; l]);
        for layer in 0..l {
            let mut seen = BTreeSet::new();
            let mut shared = 0usize;
            for token in 0..t {
                let picks = self.picks(token, layer);
                seen.extend(picks.iter().copied());
                if token > 0 {
                    let before = self.picks(token - 1, layer);
                    shared += picks.iter().filter(|p| before.contains(p)).count();
                }
            }
            overlap[layer] = shared as f64 / ((t - 1) * k) as f64;
            distinct[layer] = seen.len();
        }
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" ");
        println!("  overlap with the token before, per layer: {}", fmt(&overlap));
        println!(
            "  overlap mean over layers {:.3}, min {:.3}, max {:.3}",
            overlap.iter().sum::<f64>() / l as f64,
            overlap.iter().copied().fold(f64::INFINITY, f64::min),
            overlap.iter().copied().fold(0.0, f64::max)
        );
        println!(
            "  distinct experts over the trace, per layer: {}",
            distinct.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")
        );
        println!(
            "  distinct (layer, expert) pairs over the trace: {} of {}",
            distinct.iter().sum::<usize>(),
            l * self.n_expert
        );
        // Per token: the (layer, expert) pairs it touches that none of the `back`
        // tokens before it touched.
        for back in [1usize, 4, 16, 64, usize::MAX] {
            let mut new_pairs = 0usize;
            let mut counted = 0usize;
            for token in 1..t {
                let first = token.saturating_sub(back);
                if back != usize::MAX && token < back {
                    continue;
                }
                counted += 1;
                for layer in 0..l {
                    for pick in self.picks(token, layer) {
                        if !(first..token).any(|b| self.picks(b, layer).contains(pick)) {
                            new_pairs += 1;
                        }
                    }
                }
            }
            if counted > 0 {
                let label = if back == usize::MAX {
                    "all tokens".to_owned()
                } else {
                    format!("{back} token(s)")
                };
                println!(
                    "  pairs of a token not touched by the {label} before it: mean {:.1} of {}",
                    new_pairs as f64 / counted as f64,
                    l * k
                );
            }
        }
        // The weight the first picks carry.
        let mut by_rank = vec![0.0f64; k];
        for record in self.weights.chunks_exact(k) {
            for (sum, &w) in by_rank.iter_mut().zip(record) {
                *sum += f64::from(w);
            }
        }
        let records = (t * l) as f64;
        println!(
            "  mean weight by rank: {}",
            fmt(&by_rank.iter().map(|s| s / records).collect::<Vec<_>>())
        );
    }
}

/// Reads the experts a row picked into the page cache before the reference
/// decodes them from the mapping. A page fault on a cold mapping reads a few
/// pages at a time; one read of an expert's whole matrix is several times
/// faster, and several run at once. An expert is read once per window: the
/// page cache of a machine that is short of memory does not keep it until the
/// next one. The result of the reference does not depend on any of this.
struct Prefetch {
    files: Vec<std::fs::File>,
    /// Per layer the three stacked expert tensors: part, offset, bytes of one expert.
    stacks: Vec<[(usize, u64, usize); 3]>,
    done: BTreeSet<(usize, u32)>,
    bytes: u64,
    seconds: f64,
}

impl Prefetch {
    fn new(gguf: &Gguf, config: &Config) -> Result<Self, AnyError> {
        let files = (0..gguf.n_parts())
            .map(|part| std::fs::File::open(gguf.path(part as u16)))
            .collect::<Result<_, _>>()?;
        let mut stacks = Vec::new();
        for layer in 0..config.n_layer {
            let mut stack = [(0, 0, 0); 3];
            for (slot, what) in ["gate", "up", "down"].into_iter().enumerate() {
                let name = format!("blk.{layer}.ffn_{what}_exps.weight");
                let info = gguf.tensor(&name).ok_or(format!("{name} is missing"))?;
                stack[slot] = (
                    usize::from(info.part),
                    info.offset,
                    info.size as usize / config.experts.count,
                );
            }
            stacks.push(stack);
        }
        Ok(Self { files, stacks, done: BTreeSet::new(), bytes: 0, seconds: 0.0 })
    }
}

impl Observer for Prefetch {
    fn routing(&mut self, layer: usize, _row: usize, ids: &[u32], _weights: &[f32]) {
        use std::os::unix::fs::FileExt;
        let mut ranges = Vec::new();
        for &id in ids {
            if self.done.insert((layer, id)) {
                for &(part, offset, size) in &self.stacks[layer] {
                    ranges.push((part, offset + u64::from(id) * size as u64, size));
                }
            }
        }
        if ranges.is_empty() {
            return;
        }
        let started = Instant::now();
        let files = &self.files;
        std::thread::scope(|scope| {
            for chunk in ranges.chunks(ranges.len().div_ceil(8)) {
                scope.spawn(move || {
                    let mut buffer = Vec::new();
                    for &(part, offset, size) in chunk {
                        buffer.resize(size, 0u8);
                        // A failed read only loses the speed-up.
                        let _ = files[part].read_exact_at(&mut buffer, offset);
                    }
                });
            }
        });
        self.bytes += ranges.iter().map(|r| r.2 as u64).sum::<u64>();
        self.seconds += started.elapsed().as_secs_f64();
    }
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

fn run(args: &[String]) -> Result<(), AnyError> {
    let [model, out, name, text_file, rest @ ..] = args else {
        return Err("usage: q4trace run MODEL.gguf OUT_DIR NAME TEXT_FILE [--chat] [--gen N] [--window W] [--threads T]".into());
    };
    let (mut chat, mut n_gen, mut window_rows, mut threads) = (false, 0usize, 32usize, 12usize);
    let mut probe = false;
    let mut rest = rest.iter();
    while let Some(flag) = rest.next() {
        let mut number = || -> Result<usize, AnyError> {
            Ok(rest.next().ok_or("a number is missing")?.parse()?)
        };
        match flag.as_str() {
            "--chat" => chat = true,
            "--probe-rule" => probe = true,
            "--gen" => n_gen = number()?,
            "--window" => window_rows = number()?.max(1),
            "--threads" => threads = number()?,
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;

    let started = Instant::now();
    let gguf = Gguf::open(model)?;
    let config = Config::from_gguf(&gguf)?;
    let weights = Weights::load(&gguf, &config)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    println!(
        "opened in {:.1} s: {} layers, {} experts, vocabulary {}",
        started.elapsed().as_secs_f64(),
        config.n_layer,
        config.experts.count,
        config.n_vocab
    );

    let text = std::fs::read_to_string(text_file)?;
    let text = text.trim_end();
    let prompt = if chat {
        format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        text.to_owned()
    };
    let prompt_tokens =
        tokenizer.encode(&prompt, EncodeOptions { add_special: false, parse_special: chat });
    println!("{} prompt tokens", prompt_tokens.len());

    let reference = Reference::new(&config, &weights).threads(threads).keep_residual(false);
    let mut prefetch = Prefetch::new(&gguf, &config)?;
    let mut state = SequenceState::new(&config);
    let mut trace = Trace {
        n_layer: config.n_layer,
        n_expert: config.experts.count,
        k: config.experts.used,
        n_prompt: prompt_tokens.len(),
        tokens: Vec::new(),
        ids: Vec::new(),
        weights: Vec::new(),
    };
    let trace_path = out.join(format!("{name}.q4route"));
    let save = |trace: &Trace| -> Result<(), AnyError> {
        trace.write(&trace_path)?;
        let ids: Vec<String> = trace.tokens.iter().map(u32::to_string).collect();
        std::fs::write(out.join(format!("{name}.tokens.txt")), ids.join(" ") + "\n")?;
        let text = tokenizer.decode(&trace.tokens, true)?;
        std::fs::write(out.join(format!("{name}.txt")), text)?;
        Ok(())
    };

    // The given text, in windows.
    let mut next = None;
    let mut at = 0;
    while at < prompt_tokens.len() {
        let end = (at + window_rows).min(prompt_tokens.len());
        let last = end == prompt_tokens.len() && n_gen > 0;
        if probe && at >= config.indexer.dense_limit() {
            probe = false;
            probe_rule(&config, &weights, threads, &state, &prompt_tokens[at..end])?;
        }
        let t0 = Instant::now();
        let logits = if last { LogitRows::Last } else { LogitRows::None };
        prefetch.done.clear();
        let window = reference.forward_observed(
            &mut state,
            &prompt_tokens[at..end],
            logits,
            &mut prefetch,
        )?;
        trace.push(&window);
        save(&trace)?;
        let secs = t0.elapsed().as_secs_f64();
        println!("cells {at}..{end}: {secs:.1} s, {:.2} s per token", secs / (end - at) as f64);
        if last {
            let logits = window.logits_of(window.rows() - 1).ok_or("the last row has logits")?;
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(out.join(format!("{name}.prompt-logits.f32")), bytes)?;
            next = Some(argmax(logits));
        }
        at = end;
    }

    // Greedy generation, one token per window.
    let stop: BTreeSet<u32> =
        [gguf.u64("tokenizer.ggml.eos_token_id"), gguf.u64("tokenizer.ggml.bos_token_id")]
            .into_iter()
            .flatten()
            .map(|id| id as u32)
            .collect();
    let gen_started = Instant::now();
    let mut generated = 0;
    while let Some(token) = next.take() {
        if generated == n_gen || stop.contains(&token) {
            println!("stopped at token {token} after {generated} generated tokens");
            break;
        }
        prefetch.done.clear();
        let window =
            reference.forward_observed(&mut state, &[token], LogitRows::Last, &mut prefetch)?;
        trace.push(&window);
        generated += 1;
        next = Some(argmax(window.logits_of(0).ok_or("the row has logits")?));
        if generated % 8 == 0 {
            save(&trace)?;
            println!(
                "generated {generated}: {:.2} s per token",
                gen_started.elapsed().as_secs_f64() / generated as f64
            );
        }
    }
    save(&trace)?;
    if generated > 0 {
        println!(
            "generated {generated} tokens, {:.2} s per token",
            gen_started.elapsed().as_secs_f64() / generated as f64
        );
    }
    println!(
        "total {:.0} s; experts read ahead: {:.2} GiB in {:.1} s",
        started.elapsed().as_secs_f64(),
        prefetch.bytes as f64 / (1u64 << 30) as f64,
        prefetch.seconds
    );
    trace.stats(name);
    Ok(())
}

/// Runs one window from `state` under both selection rules and prints how
/// far the two are apart, by the size of the row's tail.
fn probe_rule(
    config: &Config,
    weights: &Weights<'_>,
    threads: usize,
    state: &SequenceState,
    tokens: &[u32],
) -> Result<(), AnyError> {
    let run = |rule: SelectionRule| {
        Reference::new(config, weights)
            .threads(threads)
            .keep_residual(false)
            .selection_rule(rule)
            .window(state, tokens, LogitRows::All)
    };
    let (budget, whole) = (run(SelectionRule::CellBudget)?, run(SelectionRule::WholeBlocks)?);
    let (v, k, block) = (config.n_vocab, config.experts.used, config.indexer.block);
    println!("selection rules at cells {}..{}:", state.position(), state.position() + tokens.len());
    for tail in 0..block {
        let (mut rows, mut worst, mut sum, mut same_pick, mut other_experts) =
            (0, 0.0f64, 0.0f64, 0, 0);
        for row in (0..tokens.len()).filter(|r| (state.position() + r + 1) % block == tail) {
            let (a, b) =
                (&budget.logits[row * v..(row + 1) * v], &whole.logits[row * v..(row + 1) * v]);
            let (mut diff, mut norm) = (0.0f64, 0.0f64);
            for (&x, &y) in a.iter().zip(b) {
                diff += (f64::from(x) - f64::from(y)).powi(2);
                norm += f64::from(x).powi(2);
            }
            let rel = (diff / norm).sqrt();
            rows += 1;
            worst = worst.max(rel);
            sum += rel;
            same_pick += usize::from(argmax(a) == argmax(b));
            for (ra, rb) in budget.routing.iter().zip(&whole.routing) {
                let (ia, ib) = (&ra.ids[row * k..(row + 1) * k], &rb.ids[row * k..(row + 1) * k]);
                other_experts += ia.iter().filter(|id| !ib.contains(id)).count();
            }
        }
        if rows > 0 {
            println!(
                "  tail of {tail} cells ({} cells differ): {rows} rows, logits rel rms mean {:.2e} worst {worst:.2e}, same greedy pick {same_pick}/{rows}, routed experts that differ {other_experts} of {}",
                block - 1 - tail,
                sum / rows as f64,
                rows * config.n_layer * k
            );
        }
    }
    Ok(())
}

/// Keeps, for every row and layer, the mean of the residual streams after
/// the layer.
struct StreamMeans {
    n_embd: usize,
    /// `[layer][cell]` to `N` values.
    means: Vec<Vec<Vec<f32>>>,
}

impl Observer for StreamMeans {
    fn tensor(&mut self, layer: Option<usize>, name: &'static str, rows: usize, data: &[f32]) {
        let (Some(layer), "layer_out") = (layer, name) else { return };
        let wide = data.len() / rows;
        let streams = wide / self.n_embd;
        for row in data.chunks_exact(wide) {
            let mean = (0..self.n_embd)
                .map(|i| {
                    let sum: f64 = (0..streams).map(|c| f64::from(row[c * self.n_embd + i])).sum();
                    (sum / streams as f64) as f32
                })
                .collect();
            self.means[layer].push(mean);
        }
    }
}

fn tap(args: &[String]) -> Result<(), AnyError> {
    let [model, tap_file, tokens_file] = args else {
        return Err("usage: q4trace tap MODEL.gguf TAP_FILE TOKENS_FILE".into());
    };
    let gguf = Gguf::open(model)?;
    let config = Config::from_gguf(&gguf)?;
    let weights = Weights::load(&gguf, &config)?;
    let n = config.n_embd;
    let bytes = std::fs::read(tap_file)?;
    let word = |at: usize| -> Option<usize> {
        let b = bytes.get(at * 4..at * 4 + 4)?;
        usize::try_from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok()
    };
    if bytes.get(..4) != Some(b"STAP") || word(1) != Some(1) || word(2) != Some(n) {
        return Err(format!("{tap_file} is not a version 1 tap file for a width of {n}").into());
    }
    let n_taps = word(3).ok_or("a short header")?;
    let layers: Vec<usize> =
        (0..n_taps).map(|i| word(4 + i).ok_or("a short header")).collect::<Result<_, _>>()?;
    let bytes = &bytes[(4 + n_taps) * 4..];
    let tokens: Vec<u32> = std::fs::read_to_string(tokens_file)?
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|w| !w.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;

    // Strata's rows: the first valid one per cell.
    let record = 8 + layers.len() * n * 4;
    let mut theirs: Vec<Option<&[u8]>> = vec![None; tokens.len()];
    let (mut before, mut valid, mut first_cell) = (i64::MIN, true, None);
    for rec in bytes.chunks_exact(record) {
        let pos = i64::from(i32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]));
        let token = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
        if pos != before + 1 {
            valid = true; // a new window
        }
        before = pos;
        let cell = usize::try_from(pos)?;
        valid = valid && tokens.get(cell) == Some(&token);
        if valid && theirs[cell].is_none() {
            theirs[cell] = Some(&rec[8..]);
            first_cell.get_or_insert(cell);
        }
    }
    let cells: Vec<usize> = (0..tokens.len()).filter(|&c| theirs[c].is_some()).collect();
    let first_cell = first_cell.ok_or("the tap file has no row of this sequence")?;
    println!(
        "{} records, {} cells of the sequence tapped, the first is cell {first_cell}",
        bytes.len() / record,
        cells.len()
    );

    let reference = Reference::new(&config, &weights).keep_residual(false);
    let mut state = SequenceState::new(&config);
    let mut means = StreamMeans { n_embd: n, means: vec![Vec::new(); config.n_layer] };
    let end = cells.last().map_or(0, |c| c + 1);
    for chunk in tokens[..end].chunks(16) {
        reference.forward_observed(&mut state, chunk, LogitRows::None, &mut means)?;
    }

    println!(
        "layer: rel rms difference of the stream mean at the first tapped cell | median, worst over the other cells"
    );
    for (slot, &layer) in layers.iter().enumerate() {
        let rel = |cell: usize| -> f64 {
            let ours = &means.means[layer][cell];
            let row = &theirs[cell].unwrap_or(&[])[slot * n * 4..(slot + 1) * n * 4];
            let (mut diff, mut norm) = (0.0f64, 0.0f64);
            for (x, b) in ours.iter().zip(row.as_chunks::<4>().0) {
                let y = f32::from_le_bytes(*b);
                diff += (f64::from(*x) - f64::from(y)).powi(2);
                norm += f64::from(*x).powi(2);
            }
            (diff / norm).sqrt()
        };
        let mut others: Vec<f64> =
            cells.iter().filter(|&&c| c != first_cell).map(|&c| rel(c)).collect();
        others.sort_by(f64::total_cmp);
        let (median, worst) = match others.as_slice() {
            [] => (f64::NAN, f64::NAN),
            v => (v[v.len() / 2], v[v.len() - 1]),
        };
        println!(
            "{layer:2} {} {:.2e} | {median:.2e} {worst:.2e}",
            if config.is_attention(layer) { "att" } else { "rec" },
            rel(first_cell)
        );
    }
    Ok(())
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("tap") => tap(&args[1..]),
        Some("stats") => {
            for file in &args[1..] {
                Trace::read(Path::new(file))?.stats(file);
            }
            Ok(())
        }
        _ => Err("usage: q4trace run ... | q4trace stats FILE... | q4trace tap ...".into()),
    }
}
