//! Measures what a read-ahead of experts could know before a router has run:
//! a tool for development, not a part of the server.
//!
//! ```text
//! qwen4exp_predict record --model FIRST_PART.gguf --out DIR [--devices 0]
//!     [--text-file german.txt,code.txt] [--prompt 256] [--generate 128] [--ram-gib 9]
//! qwen4exp_predict analyze --model FIRST_PART.gguf --in DIR/german.q4hid,DIR/code.q4hid
//!     [--threads 12] [--rows 400]
//! qwen4exp_predict drive --model FIRST_PART.gguf [--experts 160] [--class demand|prefetch]
//! ```
//!
//! - `record` runs the model on the GPUs: every text file is put into a chat
//!   prompt (the first `--prompt` tokens of it), read in probed steps, and
//!   answered greedily for `--generate` tokens. For every row and layer it
//!   writes the residual behind the layer, the output of the layer's first
//!   half, the router's picks, and which tier held every expert of the layer
//!   when the row's step began.
//! - `drive` reads experts of the file with 1 to 32 of them in flight and
//!   says how long one takes and what the drive delivers: what a read-ahead
//!   can hide a read behind, and how many reads it may keep in flight.
//! - `analyze` needs no GPU. For every row and every layer `T` it asks each
//!   candidate what it would have named for `T` before the router of `T` ran,
//!   and compares that with the picks: the router of `T` on the residual of
//!   an earlier point (through the mixer of `T`), the router of `T` on the
//!   input an earlier layer's own router read, the picks of the row before,
//!   the picks the same token had the last time, and the most used experts.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read as _, Write as _};
use std::sync::Mutex;
use std::time::Instant;

use ares_cuda::{Device, Wait};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    LayerOut, Options, Probe, Rows, Session, SessionOptions, Wanted,
};
use ares_model::qwen4exp::{Config, Reference, Weights};
use ares_tokenizer::{ChatOptions, EncodeOptions, Message, Tokenizer};

type AnyError = Box<dyn std::error::Error>;

const MAGIC: &[u8; 4] = b"Q4HD";

/// The dimensions a recording was made with.
#[derive(Clone, Copy, Debug)]
struct Dims {
    n_layer: usize,
    wide: usize,
    n_embd: usize,
    used: usize,
    n_experts: usize,
}

/// One row of a recording.
struct Row {
    /// Whether the row was generated (otherwise it is a row of the prompt).
    generated: bool,
    token: u32,
    /// Per layer and expert: 0 in no tier, 1 in RAM, 2 in VRAM, when the step began.
    tier: Vec<u8>,
    /// Per layer the picks, in rank order.
    picks: Vec<u32>,
    /// Per layer the residual behind it.
    residual: Vec<f32>,
    /// Per layer the output of its first half.
    core: Vec<f32>,
}

fn write_f32(out: &mut impl std::io::Write, values: &[f32]) -> std::io::Result<()> {
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32(input: &mut impl std::io::Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    input.read_exact(&mut bytes)?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn read_u32(input: &mut impl std::io::Read, n: usize) -> std::io::Result<Vec<u32>> {
    let mut bytes = vec![0u8; n * 4];
    input.read_exact(&mut bytes)?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect())
}

/// Keeps what every layer of a probed step left.
struct Keep {
    dims: Dims,
    rows: usize,
    residual: Vec<Vec<f32>>,
    core: Vec<Vec<f32>>,
    picks: Vec<Vec<Vec<u32>>>,
}

impl Probe for Keep {
    fn before(&mut self, _layer: usize) -> Option<Vec<f32>> {
        None
    }

    fn after(&mut self, layer: usize, out: LayerOut<'_>) {
        debug_assert_eq!(out.residual.len(), self.rows * self.dims.wide);
        self.residual[layer] = out.residual.to_vec();
        self.core[layer] = out.core.to_vec();
        self.picks[layer] = out.picks;
    }
}

fn arg_map() -> Result<(String, HashMap<String, String>), AnyError> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("record or analyze")?;
    let mut map = HashMap::new();
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        map.insert(flag, value);
    }
    Ok((mode, map))
}

fn record(args: &HashMap<String, String>) -> Result<(), AnyError> {
    let model = args.get("--model").ok_or("--model FILE is needed")?;
    let out_dir = args.get("--out").ok_or("--out DIR is needed")?;
    let n_prompt: usize = args.get("--prompt").map_or(Ok(256), |v| v.parse())?;
    let n_generate: usize = args.get("--generate").map_or(Ok(128), |v| v.parse())?;
    let devices: Vec<u32> = args
        .get("--devices")
        .map_or("0", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if devices.len() != 1 {
        return Err("the recording runs on one device: the store's layers are the model's".into());
    }
    let gguf = Gguf::open(model)?;
    let c = Config::from_gguf(&gguf)?;
    let w = Weights::load(&gguf, &c)?;
    let opts = SessionOptions {
        part: Options { cells: 2048, ..Options::default() },
        ram_tier: args
            .get("--ram-gib")
            .map(|g| g.parse::<f64>().map(|g| (g * (1u64 << 30) as f64) as u64))
            .transpose()?,
        ..SessionOptions::default()
    };
    let devices: Vec<Device> =
        devices.iter().map(|d| Device::open(*d, Wait::Spin)).collect::<Result<_, _>>()?;
    let mut session = Session::open(&gguf, &c, &w, devices, opts)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let dims = Dims {
        n_layer: c.n_layer,
        wide: c.wide(),
        n_embd: c.n_embd,
        used: c.experts.used,
        n_experts: c.experts.count,
    };
    let texts = args.get("--text-file").ok_or("--text-file A.txt,B.txt is needed")?;
    for path in texts.split(',') {
        let name =
            std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("text");
        let text = std::fs::read_to_string(path)?;
        // The text is cut to the prompt length by tokens, then asked about.
        let plain = EncodeOptions { add_special: false, parse_special: false };
        let ids = tokenizer.encode(&text, plain);
        let cut = tokenizer.decode(&ids[..ids.len().min(n_prompt)], false)?;
        let cut = String::from_utf8_lossy(&cut);
        let question = "Continue this in your own words, in the language and the form of the text.";
        let chat = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
        let prompt = tokenizer.render_chat(
            &[Message::new("user", format!("{cut}\n\n{question}"))],
            &[],
            &chat,
        )?;
        let tokens =
            tokenizer.encode(&prompt, EncodeOptions { add_special: true, parse_special: true });
        let file = std::fs::File::create(format!("{out_dir}/{name}.q4hid"))?;
        let mut out = BufWriter::with_capacity(8 << 20, file);
        out.write_all(MAGIC)?;
        for v in [dims.n_layer, dims.wide, dims.n_embd, dims.used, dims.n_experts] {
            out.write_all(&(v as u32).to_le_bytes())?;
        }
        session.restart(0)?;
        let started = Instant::now();
        let mut rows_written = 0usize;
        let mut step =
            |session: &mut Session<'_>, tokens: &[u32], generated: bool| -> Result<u32, AnyError> {
                // Where every expert is before the step.
                let store = session.store();
                let mut tier = vec![0u8; dims.n_layer * dims.n_experts];
                for layer in 0..dims.n_layer {
                    for expert in 0..dims.n_experts {
                        tier[layer * dims.n_experts + expert] =
                            if store.in_vram(layer, expert as u32) {
                                2
                            } else {
                                u8::from(store.in_ram(layer, expert as u32))
                            };
                    }
                }
                let mut keep = Keep {
                    dims,
                    rows: tokens.len(),
                    residual: vec![Vec::new(); dims.n_layer],
                    core: vec![Vec::new(); dims.n_layer],
                    picks: vec![Vec::new(); dims.n_layer],
                };
                let picks = session.step_probed(
                    &[Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last }],
                    &mut keep,
                )?;
                let pick = picks[0];
                for (row, token) in tokens.iter().enumerate() {
                    out.write_all(&[u8::from(generated)])?;
                    out.write_all(&token.to_le_bytes())?;
                    out.write_all(&tier)?;
                    for layer in 0..dims.n_layer {
                        for id in &keep.picks[layer][row] {
                            out.write_all(&id.to_le_bytes())?;
                        }
                    }
                    for layer in 0..dims.n_layer {
                        write_f32(&mut out, &keep.residual[layer][row * dims.wide..][..dims.wide])?;
                    }
                    for layer in 0..dims.n_layer {
                        write_f32(&mut out, &keep.core[layer][row * dims.n_embd..][..dims.n_embd])?;
                    }
                    rows_written += 1;
                }
                Ok(pick)
            };
        let mut pick = 0;
        for chunk in tokens.chunks(8) {
            pick = step(&mut session, chunk, false)?;
        }
        let mut reply = Vec::new();
        for _ in 0..n_generate {
            if tokenizer.is_end_of_generation(pick) {
                break;
            }
            reply.push(pick);
            pick = step(&mut session, &[pick], true)?;
        }
        out.flush()?;
        println!(
            "{name}: {} prompt rows, {} generated, {rows_written} rows written in {:.1} s\n  -> {:?}",
            tokens.len(),
            reply.len(),
            started.elapsed().as_secs_f64(),
            String::from_utf8_lossy(&tokenizer.decode(&reply, false)?)
        );
    }
    Ok(())
}

fn load(path: &str, limit: usize) -> Result<(Dims, Vec<Row>), AnyError> {
    let mut input = BufReader::with_capacity(8 << 20, std::fs::File::open(path)?);
    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(format!("{path} is not a recording").into());
    }
    let head = read_u32(&mut input, 5)?;
    let dims = Dims {
        n_layer: head[0] as usize,
        wide: head[1] as usize,
        n_embd: head[2] as usize,
        used: head[3] as usize,
        n_experts: head[4] as usize,
    };
    let mut rows = Vec::new();
    loop {
        let mut kind = [0u8; 1];
        if input.read_exact(&mut kind).is_err() {
            break;
        }
        let token = read_u32(&mut input, 1)?[0];
        let mut tier = vec![0u8; dims.n_layer * dims.n_experts];
        input.read_exact(&mut tier)?;
        let picks = read_u32(&mut input, dims.n_layer * dims.used)?;
        let residual = read_f32(&mut input, dims.n_layer * dims.wide)?;
        let core = read_f32(&mut input, dims.n_layer * dims.n_embd)?;
        rows.push(Row { generated: kind[0] != 0, token, tier, picks, residual, core });
        if rows.len() >= limit {
            break;
        }
    }
    Ok((dims, rows))
}

/// The experts of the `k` largest logits, largest first.
fn top(logits: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    order.truncate(k);
    order
}

/// The most experts a candidate names per row and layer.
const DEPTH: usize = 48;
/// The numbers of named experts the tables show.
const WIDTHS: [usize; 5] = [10, 16, 24, 32, 48];
/// The farthest distance, in layers.
const FAR: usize = 3;

/// What a candidate named for one row and layer, most likely first.
type Named = Vec<u32>;

/// The lists of the router candidates of one row: per target layer and distance.
#[derive(Default, Clone)]
struct Routed {
    /// Router of `T` through the mixer of `T` on the residual in the middle of layer
    /// `T - d` (where that layer's own router reads), `d` in `0..=FAR`; `d = 0` is the
    /// router itself and checks the recording.
    mid: Vec<Vec<Named>>,
    /// The same on the residual behind layer `T - d`, `d` in `1..=FAR`.
    end: Vec<Vec<Named>>,
    /// Router of `T` on the input the router of layer `T - d` read, `d` in `1..=FAR`.
    input: Vec<Vec<Named>>,
}

fn routed(reference: &Reference<'_, '_>, dims: Dims, row: &Row) -> Routed {
    let (n_layer, wide, n) = (dims.n_layer, dims.wide, dims.n_embd);
    let residual = |layer: usize| &row.residual[layer * wide..(layer + 1) * wide];
    // The residual in the middle of every layer but the first, and what its router read.
    let mut mid: Vec<Vec<f32>> = vec![Vec::new(); n_layer];
    let mut input: Vec<Vec<f32>> = vec![Vec::new(); n_layer];
    for layer in 1..n_layer {
        mid[layer] = reference.after_core(
            layer,
            residual(layer - 1),
            &row.core[layer * n..(layer + 1) * n],
            1,
        );
        input[layer] = reference.expert_input(layer, &mid[layer], 1);
    }
    let mut out = Routed {
        mid: vec![vec![Vec::new(); FAR + 1]; n_layer],
        end: vec![vec![Vec::new(); FAR + 1]; n_layer],
        input: vec![vec![Vec::new(); FAR + 1]; n_layer],
    };
    for target in 1..n_layer {
        for d in 0..=FAR {
            if target > d {
                let m = if d == 0 {
                    input[target].clone()
                } else {
                    reference.expert_input(target, &mid[target - d], 1)
                };
                out.mid[target][d] = top(&reference.router_logits(target, &m, 1), DEPTH);
            }
            if d >= 1 && target >= d {
                let m = reference.expert_input(target, residual(target - d), 1);
                out.end[target][d] = top(&reference.router_logits(target, &m, 1), DEPTH);
            }
            if d >= 1 && target > d {
                out.input[target][d] =
                    top(&reference.router_logits(target, &input[target - d], 1), DEPTH);
            }
        }
    }
    out
}

/// Hits and sizes of one candidate at one width.
#[derive(Clone, Copy, Default)]
struct Tally {
    /// Picks, and those of them the candidate named.
    picks: u64,
    found: u64,
    /// Experts the candidate named.
    named: u64,
    /// The same over what no tier held when the step began: picks only on the file, those
    /// named, and named experts that were only on the file (what a read-ahead would read)
    /// with those of them that were picked.
    cold_picks: u64,
    cold_found: u64,
    cold_named: u64,
    cold_used: u64,
}

impl Tally {
    fn add(&mut self, named: &[u32], picks: &[u32], tier: &[u8]) {
        self.picks += picks.len() as u64;
        self.named += named.len() as u64;
        for pick in picks {
            let hit = named.contains(pick);
            self.found += u64::from(hit);
            if tier[*pick as usize] == 0 {
                self.cold_picks += 1;
                self.cold_found += u64::from(hit);
            }
        }
        for expert in named {
            if tier[*expert as usize] == 0 {
                self.cold_named += 1;
                self.cold_used += u64::from(picks.contains(expert));
            }
        }
    }

    fn line(&self) -> String {
        let pct = |a: u64, b: u64| if b == 0 { f64::NAN } else { 100.0 * a as f64 / b as f64 };
        format!(
            "recall {:5.1} % precision {:5.1} % | on the file only: recall {:5.1} % of {:6} picks, \
             {:5.1} % of {:6} named were picked",
            pct(self.found, self.picks),
            pct(self.found, self.named),
            pct(self.cold_found, self.cold_picks),
            self.cold_picks,
            pct(self.cold_used, self.cold_named),
            self.cold_named,
        )
    }
}

/// The tallies of every candidate: by name, then by width.
#[derive(Default)]
struct Board {
    names: Vec<String>,
    tallies: Vec<[Tally; WIDTHS.len()]>,
    /// For the candidates that are shown per layer: per layer the tally at 16 named.
    by_layer: HashMap<String, Vec<Tally>>,
}

impl Board {
    fn add(
        &mut self,
        name: &str,
        layer: usize,
        n_layer: usize,
        list: &[u32],
        picks: &[u32],
        tier: &[u8],
    ) {
        let at = self.names.iter().position(|n| n == name).unwrap_or_else(|| {
            self.names.push(name.to_owned());
            self.tallies.push([Tally::default(); WIDTHS.len()]);
            self.names.len() - 1
        });
        for (w, width) in WIDTHS.iter().enumerate() {
            self.tallies[at][w].add(&list[..list.len().min(*width)], picks, tier);
        }
        let per =
            self.by_layer.entry(name.to_owned()).or_insert_with(|| vec![Tally::default(); n_layer]);
        per[layer].add(&list[..list.len().min(16)], picks, tier);
    }
}

fn analyze(args: &HashMap<String, String>) -> Result<(), AnyError> {
    let model = args.get("--model").ok_or("--model FILE is needed")?;
    let inputs = args.get("--in").ok_or("--in A.q4hid,B.q4hid is needed")?;
    let threads: usize = args.get("--threads").map_or(Ok(12), |v| v.parse())?;
    let limit: usize = args.get("--rows").map_or(Ok(usize::MAX), |v| v.parse())?;
    let only_generated = args.get("--only").is_some_and(|v| v == "generated");
    let gguf = Gguf::open(model)?;
    let c = Config::from_gguf(&gguf)?;
    let w = Weights::load(&gguf, &c)?;
    println!(
        "{} layers, {} experts of which {} are picked, mixer rank {}, {} streams of {}",
        c.n_layer, c.experts.count, c.experts.used, c.mixer_rank, c.n_stream, c.n_embd
    );
    for path in inputs.split(',') {
        let (dims, rows) = load(path, limit)?;
        let started = Instant::now();
        // The router candidates of every row, in parallel over the rows.
        let lists: Vec<Mutex<Option<Routed>>> = rows.iter().map(|_| Mutex::new(None)).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let reference = Reference::new(&c, &w).threads(1);
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if at >= rows.len() {
                            break;
                        }
                        if only_generated && !rows[at].generated {
                            continue;
                        }
                        let made = routed(&reference, dims, &rows[at]);
                        *lists[at].lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
                    }
                });
            }
        });
        let lists: Vec<Routed> = lists
            .into_iter()
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()).unwrap_or_default())
            .collect();
        println!(
            "\n== {path}: {} rows ({} generated), candidates computed in {:.0} s",
            rows.len(),
            rows.iter().filter(|r| r.generated).count(),
            started.elapsed().as_secs_f64()
        );

        for generated in [false, true] {
            if only_generated && !generated {
                continue;
            }
            let mut board = Board::default();
            // Per candidate and rank of its list: named experts that were only on the
            // file, and those of them that were picked.
            let mut by_rank: HashMap<String, Vec<(u64, u64)>> = HashMap::new();
            // What the same token picked the last time, and how often every expert was
            // picked so far, per layer.
            let mut by_token: HashMap<u32, Vec<u32>> = HashMap::new();
            let mut counts = vec![0u32; dims.n_layer * dims.n_experts];
            let mut before: Option<&Row> = None;
            let mut before_routed: Option<&Routed> = None;
            for (row, routed) in rows.iter().zip(&lists) {
                let counted = row.generated == generated;
                for layer in 0..dims.n_layer {
                    let picks = &row.picks[layer * dims.used..(layer + 1) * dims.used];
                    let tier = &row.tier[layer * dims.n_experts..(layer + 1) * dims.n_experts];
                    if counted {
                        let mut add = |name: &str, list: &[u32]| {
                            board.add(name, layer, dims.n_layer, list, picks, tier);
                        };
                        for (name, lists) in [("middle", &routed.mid), ("input", &routed.input)] {
                            for (d, list) in lists[layer].iter().enumerate().skip(1) {
                                let ranks = by_rank
                                    .entry(format!("router on the {name} of T-{d}"))
                                    .or_insert_with(|| vec![(0, 0); 16]);
                                for (rank, expert) in list.iter().take(16).enumerate() {
                                    if tier[*expert as usize] == 0 {
                                        ranks[rank].0 += 1;
                                        ranks[rank].1 += u64::from(picks.contains(expert));
                                    }
                                }
                            }
                        }
                        for d in 0..=FAR {
                            if !routed.mid[layer][d].is_empty() {
                                add(
                                    &format!("router on the middle of T-{d}"),
                                    &routed.mid[layer][d],
                                );
                            }
                            if d >= 1 && !routed.end[layer][d].is_empty() {
                                add(&format!("router on the end of T-{d}"), &routed.end[layer][d]);
                            }
                            if d >= 1 && !routed.input[layer][d].is_empty() {
                                add(
                                    &format!("router on the input of T-{d}"),
                                    &routed.input[layer][d],
                                );
                            }
                        }
                        let last: Vec<u32> = before.map_or_else(Vec::new, |b| {
                            b.picks[layer * dims.used..(layer + 1) * dims.used].to_vec()
                        });
                        if before.is_some() {
                            add("the picks of the row before", &last);
                        }
                        if let Some(ranked) = before_routed
                            && ranked.mid.get(layer).is_some_and(|lists| !lists[0].is_empty())
                        {
                            add("the router's ranking at the row before", &ranked.mid[layer][0]);
                        }
                        if let Some(seen) = by_token.get(&row.token) {
                            add(
                                "the picks of the same token the last time",
                                &seen[layer * dims.used..(layer + 1) * dims.used],
                            );
                        }
                        let heat = &counts[layer * dims.n_experts..(layer + 1) * dims.n_experts];
                        let mut hot: Vec<u32> = (0..dims.n_experts as u32).collect();
                        hot.sort_by_key(|e| std::cmp::Reverse(heat[*e as usize]));
                        hot.truncate(DEPTH);
                        add("the most used so far", &hot);
                        // Together: the early router's ten and the row before.
                        for d in 1..=2 {
                            if routed.mid[layer][d].is_empty() {
                                continue;
                            }
                            let mut both: Vec<u32> = routed.mid[layer][d][..16].to_vec();
                            for e in &last {
                                if !both.contains(e) {
                                    both.push(*e);
                                }
                            }
                            // One width only: the sixteen and what the row before adds.
                            let name =
                                format!("router on the middle of T-{d} (16) and the row before");
                            let at =
                                board.names.iter().position(|n| *n == name).unwrap_or_else(|| {
                                    board.names.push(name.clone());
                                    board.tallies.push([Tally::default(); WIDTHS.len()]);
                                    board.names.len() - 1
                                });
                            board.tallies[at][1].add(&both, picks, tier);
                        }
                    }
                    for pick in picks {
                        counts[layer * dims.n_experts + *pick as usize] += 1;
                    }
                }
                by_token.insert(row.token, row.picks.clone());
                before = Some(row);
                before_routed = Some(routed);
            }
            println!("\n-- {}", if generated { "generated rows" } else { "prompt rows" });
            for (name, tallies) in board.names.iter().zip(&board.tallies) {
                println!("{name}:");
                for (w, width) in WIDTHS.iter().enumerate() {
                    if tallies[w].named > 0 {
                        println!("  {width:2} named: {}", tallies[w].line());
                    }
                }
            }
            let mut names: Vec<&String> = by_rank.keys().collect();
            names.sort();
            for name in names {
                let line: Vec<String> =
                    by_rank[name].iter().map(|(named, used)| format!("{used}/{named}")).collect();
                println!(
                    "picked of the named on the file only, by rank, {name}:\n  {}",
                    line.join(" ")
                );
            }
            for name in [
                "router on the middle of T-1",
                "router on the middle of T-2",
                "router on the input of T-1",
                "the picks of the same token the last time",
                "the picks of the row before",
                "the router's ranking at the row before",
            ] {
                if let Some(per) = board.by_layer.get(name) {
                    let recall: Vec<String> = per
                        .iter()
                        .map(|t| {
                            if t.picks == 0 {
                                "--".into()
                            } else {
                                format!("{:.0}", 100.0 * t.found as f64 / t.picks as f64)
                            }
                        })
                        .collect();
                    println!("recall per layer, 16 named, {name}:\n  {}", recall.join(" "));
                }
            }
        }
    }
    Ok(())
}

/// Reads experts of the file with `k` of them in flight, for several `k`: how long one
/// takes from its three reads being queued to the last coming back, and what the drive
/// delivers. The experts are spread over the file by a fixed sequence, read directly.
fn drive(args: &HashMap<String, String>) -> Result<(), AnyError> {
    let model = args.get("--model").ok_or("--model FILE is needed")?;
    let per_depth: usize = args.get("--experts").map_or(Ok(160), |v| v.parse())?;
    let class = match args.get("--class").map(String::as_str) {
        Some("prefetch") => ares_io::Class::Prefetch,
        _ => ares_io::Class::Demand,
    };
    let gguf = Gguf::open(model)?;
    let map =
        ares_io::ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])?;
    let mut config = ares_io::Config::default();
    let kib = |name: &str, to: &mut usize| -> Result<(), AnyError> {
        if let Some(v) = args.get(name) {
            *to = v.parse::<usize>()? << 10;
        }
        Ok(())
    };
    kib("--demand-chunk-kib", &mut config.demand_chunk)?;
    kib("--prefetch-chunk-kib", &mut config.prefetch_chunk)?;
    if let Some(v) = args.get("--prefetch-depth") {
        config.prefetch_depth = v.parse()?;
    }
    if let Some(v) = args.get("--queue-depth") {
        config.queue_depth = v.parse()?;
    }
    println!(
        "queue depth {}, prefetch depth {}, chunks: demand {} KiB, prefetch {} KiB",
        config.queue_depth,
        config.prefetch_depth,
        config.demand_chunk >> 10,
        config.prefetch_chunk >> 10
    );
    let reader = ares_io::Reader::open_gguf(&gguf, config)?;
    let layers: Vec<usize> = map.moe_layers().collect();
    let n_experts = map.n_experts(layers[0]).ok_or("no experts")?;
    let sizes: Vec<usize> =
        map.ranges(layers[0], 0).ok_or("no ranges")?.map(|range| range.len >> 10).collect();
    println!("the matrices of an expert: {sizes:?} KiB");
    println!(
        "drives {:?}, every file read directly: {}, {} layers of {n_experts} experts",
        reader.drives().iter().map(|d| d.name.clone()).collect::<Vec<_>>(),
        reader.all_direct(),
        layers.len()
    );
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) as usize
    };
    let mut key = 0u64;
    for depth in [1usize, 2, 4, 8, 16, 32] {
        let started = Instant::now();
        let mut in_flight: HashMap<u64, (Instant, u8)> = HashMap::new();
        let mut ms: Vec<f64> = Vec::new();
        let mut bytes = 0u64;
        let mut queued = 0usize;
        let mut done = Vec::new();
        while ms.len() < per_depth {
            while in_flight.len() < depth && queued < per_depth {
                let (layer, expert) = (layers[next() % layers.len()], next() % n_experts);
                let id = key;
                key += 1;
                queued += 1;
                let mut parts = 0;
                for (part, range) in map.ranges(layer, expert).ok_or("no ranges")?.enumerate() {
                    let span = reader.span(range.file, range.offset, range.len)?;
                    let buffer = ares_io::Buffer::aligned(span.len, reader.buffer_alignment());
                    bytes += span.len as u64;
                    reader
                        .submit(ares_io::Read {
                            key: id * 4 + part as u64,
                            file: range.file,
                            offset: range.offset,
                            len: range.len,
                            class,
                            buffer,
                        })
                        .map_err(|rejected| rejected.error.to_string())?;
                    parts += 1;
                }
                in_flight.insert(id, (Instant::now(), parts));
            }
            if reader.poll(&mut done) == 0 {
                std::hint::spin_loop();
                continue;
            }
            for completion in done.drain(..) {
                completion.result.as_ref().map_err(|e| e.to_string())?;
                let id = completion.key / 4;
                if let Some((since, left)) = in_flight.get_mut(&id) {
                    *left -= 1;
                    if *left == 0 {
                        ms.push(since.elapsed().as_secs_f64() * 1e3);
                        in_flight.remove(&id);
                    }
                }
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        ms.sort_by(f64::total_cmp);
        println!(
            "{depth:2} experts in flight: {:.2} ms an expert at the median, {:.2} at the 90th \
             percentile, {:.2} the slowest; {:.2} GiB/s, {:.0} experts/s (load {})",
            ms[ms.len() / 2],
            ms[ms.len() * 9 / 10],
            ms[ms.len() - 1],
            bytes as f64 / (1u64 << 30) as f64 / seconds,
            ms.len() as f64 / seconds,
            std::fs::read_to_string("/proc/loadavg")
                .unwrap_or_default()
                .split(' ')
                .next()
                .unwrap_or("?"),
        );
    }
    Ok(())
}

fn main() {
    let result = arg_map().and_then(|(mode, args)| match mode.as_str() {
        "record" => record(&args),
        "analyze" => analyze(&args),
        "drive" => drive(&args),
        other => Err(format!("{other}: record or analyze").into()),
    });
    if let Err(error) = result {
        eprintln!("qwen4exp_predict: {error}");
        std::process::exit(1);
    }
}
