//! Runs a `qwen35` model file on one or two GPUs through the runtime's
//! executor and reports what it takes and how fast it is.
//!
//! ```text
//! qwen35_run <model.gguf> [--split single|tensor|layer] [--shares 0.625,0.375]
//!     [--devices 0,1] [--precision fast|exact] [--max-rows 512] [--context 4096]
//!     [--chat] [--generate 48] [--capture DIR]... [--decode-at 1000,32000]
//!     [--decode-steps 64] [--prefill 2048] [--text FILE] [--state f32|f16]
//!     [--pipeline-rows 256] [--no-profile] [--balance FILE] [--balance-out FILE]
//!     [--step-rows N]
//! ```
//!
//! - `--chat`: a German and an English question through the chat template,
//!   greedy, decoded with the tokenizer.
//! - `--capture DIR`: a directory of `tools/capture` (`prompt.tokens`,
//!   `prefill.argmax.i32`, `prefill.logits_last.f32`, `generated.tokens`):
//!   top-1 agreement of every row with llama.cpp and the greedy tokens.
//! - `--decode-at N`: milliseconds per decode step with `N` tokens of context,
//!   with the device time by kind of kernel.
//! - `--prefill N`: prompt tokens per second for `N` tokens, in steps of
//!   `--step-rows` rows (default: `--max-rows`).
//! - `--balance-out FILE`: with a tensor split over two devices and
//!   `--decode-at`, writes the share of device 0 in every half block at which
//!   both devices would take equally long, from the device times of the
//!   decode steps. `--balance FILE` places the model by such a file.
//!
//! - `--engine N`: the whole engine instead of the session (conductor, picks
//!   on the device, drafts): three chat requests (German prose, English prose,
//!   code) of `--generate` tokens, each most-probable-token and, with
//!   `--sample`, sampled, with at most `N` drafts per window (0: none). With
//!   a model that has a draft block the drafts are its own; `--draft-vocab
//!   FILE` restricts them to the tokens of a list (an id per line). `--out
//!   DIR` keeps the answers, to compare runs. `--chat-var name=value` sets a
//!   variable of the chat template (`reasoning_effort=medium`). Reports
//!   tokens per second, acceptance by draft position and the time of a step
//!   by kind.
//!
//! Lines that start with `RES` are results.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_cuda::{Device, Precision, StateFormat, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::Config;
use ares_model::qwen35::device::placement::{HalfShares, Placement};
use ares_model::qwen35::device::{Options, PHASES, Profile, Qwen35, Rows, Session, Spread, Wanted};
use ares_runtime::Pick;
use ares_tokenizer::{ChatOptions, EncodeOptions, Message};

const MIB: f64 = 1024.0 * 1024.0;

struct Args {
    model: PathBuf,
    split: String,
    shares: Vec<f64>,
    devices: Vec<u32>,
    precision: Precision,
    state: StateFormat,
    max_rows: u32,
    context: u32,
    chat: bool,
    generate: usize,
    captures: Vec<PathBuf>,
    decode_at: Vec<u32>,
    decode_steps: usize,
    prefill: Option<usize>,
    text: Option<PathBuf>,
    pipeline_rows: u32,
    profile: bool,
    balance: Option<PathBuf>,
    balance_out: Option<PathBuf>,
    step_rows: Option<usize>,
    engine: Option<u32>,
    sample: bool,
    draft_vocab: Option<PathBuf>,
    out: Option<PathBuf>,
    chat_vars: Vec<(String, String)>,
}

fn parse() -> Args {
    let mut args = Args {
        model: PathBuf::new(),
        split: "single".into(),
        shares: vec![0.625, 0.375],
        devices: vec![0],
        precision: Precision::Fast,
        state: StateFormat::F32,
        max_rows: 512,
        context: 4096,
        chat: false,
        generate: 48,
        captures: Vec::new(),
        decode_at: Vec::new(),
        decode_steps: 64,
        prefill: None,
        text: None,
        pipeline_rows: 256,
        profile: true,
        balance: None,
        balance_out: None,
        step_rows: None,
        engine: None,
        sample: false,
        draft_vocab: None,
        out: None,
        chat_vars: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    let list = |v: String| v.split(',').map(str::to_owned).collect::<Vec<_>>();
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--split" => args.split = value(),
            "--shares" => {
                args.shares = list(value()).iter().map(|s| s.parse().expect("a share")).collect()
            }
            "--devices" => {
                args.devices = list(value()).iter().map(|s| s.parse().expect("a device")).collect()
            }
            "--precision" => {
                args.precision = match value().as_str() {
                    "exact" => Precision::Exact,
                    _ => Precision::Fast,
                }
            }
            "--state" => {
                args.state = match value().as_str() {
                    "f16" => StateFormat::F16,
                    _ => StateFormat::F32,
                }
            }
            "--max-rows" => args.max_rows = value().parse().expect("rows"),
            "--context" => args.context = value().parse().expect("cells"),
            "--chat" => args.chat = true,
            "--generate" => args.generate = value().parse().expect("tokens"),
            "--capture" => args.captures.push(value().into()),
            "--decode-at" => {
                args.decode_at = list(value()).iter().map(|s| s.parse().expect("tokens")).collect()
            }
            "--decode-steps" => args.decode_steps = value().parse().expect("steps"),
            "--prefill" => args.prefill = Some(value().parse().expect("tokens")),
            "--text" => args.text = Some(value().into()),
            "--pipeline-rows" => args.pipeline_rows = value().parse().expect("rows"),
            "--no-profile" => args.profile = false,
            "--balance" => args.balance = Some(value().into()),
            "--balance-out" => args.balance_out = Some(value().into()),
            "--step-rows" => args.step_rows = Some(value().parse().expect("rows")),
            "--engine" => args.engine = Some(value().parse().expect("drafts")),
            "--sample" => args.sample = true,
            "--draft-vocab" => args.draft_vocab = Some(value().into()),
            "--out" => args.out = Some(value().into()),
            "--chat-var" => {
                let pair = value();
                let (name, text) = pair.split_once('=').expect("--chat-var name=value");
                args.chat_vars.push((name.to_owned(), text.to_owned()));
            }
            other if args.model.as_os_str().is_empty() => args.model = other.into(),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!args.model.as_os_str().is_empty(), "usage: qwen35_run <model.gguf> [options]");
    args
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn read_words<const N: usize>(path: &Path) -> Vec<[u8; N]> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<N>().0.to_vec()
}

/// Feeds `tokens` of the sequence in slot 0 from `position` in steps of at
/// most `chunk` rows. Returns the picks of the wanted rows and the time of
/// all steps.
fn feed(
    session: &mut Session,
    tokens: &[u32],
    position: usize,
    chunk: usize,
    wanted: Wanted,
) -> (Vec<Pick>, Duration) {
    let mut picks = Vec::new();
    let mut at = 0;
    let begun = Instant::now();
    while at < tokens.len() {
        let n = chunk.min(tokens.len() - at);
        let last = at + n == tokens.len();
        let wanted = match wanted {
            Wanted::Last if !last => Wanted::None,
            other => other,
        };
        let rows = Rows {
            slot: 0,
            position: (position + at) as u32,
            tokens: &tokens[at..at + n],
            direct: true,
            wanted,
        };
        picks.extend_from_slice(session.step(&[rows]).expect("a step"));
        at += n;
    }
    (picks, begun.elapsed())
}

/// Greedy generation of `n` tokens after `first`, the pick of the last
/// prompt row. Returns the tokens (with `first`) and the wall and device
/// milliseconds of every step.
fn generate(
    session: &mut Session,
    first: u32,
    position: usize,
    n: usize,
) -> (Vec<u32>, Vec<(f64, f64)>) {
    let mut tokens = vec![first];
    let mut times = Vec::new();
    while tokens.len() < n {
        let token = *tokens.last().expect("a token");
        let rows = Rows {
            slot: 0,
            position: (position + tokens.len() - 1) as u32,
            tokens: &[token],
            direct: true,
            wanted: Wanted::Last,
        };
        let pick = session.step(&[rows]).expect("a step")[0];
        let time = session.last_time();
        times.push((time.wall.as_secs_f64() * 1e3, time.device.as_secs_f64() * 1e3));
        tokens.push(pick.token);
    }
    (tokens, times)
}

/// Prints the device time per step and kind of kernel that `profile` holds.
fn print_profile(profile: &Profile, what: &str) {
    for (index, (steps, ms)) in profile.totals().iter().enumerate() {
        let per_step: Vec<String> = PHASES
            .iter()
            .map(|p| format!("{p:?} {:.2}", ms[*p as usize] / *steps as f64))
            .collect();
        let total: f64 = ms.iter().sum::<f64>() / *steps as f64;
        println!(
            "RES {what} device {index}: {} ms per step, sum {total:.2} ({steps} steps)",
            per_step.join(", ")
        );
    }
}

/// Reads the shares of device 0 per half block that [`balance`] wrote.
fn read_balance(path: &Path) -> HalfShares {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut halves = HalfShares { mixer: Vec::new(), ffn: Vec::new() };
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let v: Vec<f64> = line.split_whitespace().map(|t| t.parse().expect("a number")).collect();
        halves.mixer.push(vec![v[1], 1.0 - v[1]]);
        halves.ffn.push(vec![v[2], 1.0 - v[2]]);
    }
    halves
}

/// What the devices of a tensor split over two devices lose by waiting for
/// each other at the sums of a decode step, and the shares per half block at
/// which they would not: device 0's share of a half is moved to where both
/// devices' times for it, taken as proportional to their shares, are equal.
fn balance(profile: &Profile, placement: &Placement, config: &Config, out: Option<&Path>) {
    let (merges, totals) = (profile.merges(), profile.totals());
    let halves = 2 * config.n_layer;
    if merges.len() != 2 || merges.iter().any(|m| m.len() != halves + 1) {
        return;
    }
    let steps = totals[0].0.max(1) as f64;
    // The own work of each device in front of merge `j`, per step.
    let own = |device: usize, j: usize| merges[device][j][0] / steps;
    let (mut sum0, mut sum1, mut slower) = (0.0, 0.0, 0.0);
    let mut text =
        String::from("# block, share of device 0 in the mixer, in the feed-forward part\n");
    for layer in 0..config.n_layer {
        let slice = placement.devices[0].blocks[layer].as_ref().expect("a slice of every block");
        let heads = if config.is_attention(layer) {
            config.attention.n_head_kv
        } else {
            config.recurrent.n_k_heads
        };
        let now =
            [slice.heads.len() as f64 / heads as f64, slice.ffn.len() as f64 / config.n_ff as f64];
        let mut wanted = [0.0; 2];
        for half in 0..2 {
            let j = 2 * layer + half + 1;
            let (t0, t1) = (own(0, j), own(1, j));
            sum0 += t0;
            sum1 += t1;
            slower += f64::max(t0, t1);
            let f = now[half];
            wanted[half] = if t0 > 0.0 && t1 > 0.0 && f > 0.0 && f < 1.0 {
                f * t1 / (f * t1 + (1.0 - f) * t0)
            } else {
                f
            };
        }
        text.push_str(&format!("{layer} {:.4} {:.4}\n", wanted[0], wanted[1]));
    }
    println!(
        "RES balance: own work per decode step {sum0:.2} | {sum1:.2} ms, the slower device of every half block {slower:.2} ms: {:.2} ms per step are waiting at sums that are not balanced",
        slower - f64::max(sum0, sum1)
    );
    if let Some(path) = out {
        std::fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("RES balance: shares per half block written to {}", path.display());
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
}

fn main() {
    let args = parse();
    let spread = match (args.split.as_str(), &args.balance) {
        ("single", _) => Spread::Single,
        ("tensor", Some(file)) => Spread::TensorByHalf(args.shares.clone(), read_balance(file)),
        ("tensor", None) => Spread::Tensor(args.shares.clone()),
        ("layer", _) => Spread::Layer(args.shares.clone()),
        (other, _) => panic!("unknown split {other}"),
    };
    if let Some(drafts) = args.engine {
        return engine(&args, &spread, drafts);
    }
    let profile = Arc::new(Profile::new());
    let logit_rows = 16;
    let options = Options {
        precision: args.precision,
        state: args.state,
        max_rows: args.max_rows,
        logit_rows,
        window_rows: 0,
        pipeline_rows: args.pipeline_rows,
        profile: args.profile.then(|| profile.clone()),
        ..Options::default()
    };
    let devices: Vec<Device> =
        args.devices.iter().map(|d| Device::open(*d, Wait::Block).expect("open device")).collect();
    let free_before: Vec<u64> =
        devices.iter().map(|d| d.mem_info().expect("memory").free).collect();

    let begun = Instant::now();
    let gguf = Gguf::open(&args.model).expect("the model file");
    let name = args.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let model =
        Arc::new(Qwen35::new(gguf, name, &spread, options, args.context).expect("the model"));
    let mut session =
        Session::start(model.clone(), devices.clone(), 1, args.context, 64 << 20).expect("start");
    let load = begun.elapsed();
    println!(
        "RES load {:.2} s ({:?} split over {} devices, {:?}, max rows {}, context {})",
        load.as_secs_f64(),
        model.placement().split,
        devices.len(),
        args.precision,
        args.max_rows,
        args.context
    );
    for (report, (device, before)) in model.reports().iter().zip(devices.iter().zip(&free_before)) {
        let free = device.mem_info().expect("memory").free;
        println!(
            "RES device {} {}: {} blocks, weights {:.0} MiB (upload {:.2} s), recurrent state {:.1} MiB per sequence, K/V {:.2} MiB per 1000 tokens, step scratch {:.1} MiB, all VRAM taken {:.0} MiB, free {:.0} MiB",
            report.index,
            report.device,
            report.blocks,
            report.weight_bytes as f64 / MIB,
            report.upload.as_secs_f64(),
            report.state_bytes_per_slot as f64 / MIB,
            report.page_bytes as f64 / 128.0 * 1000.0 / MIB,
            report.scratch_bytes as f64 / MIB,
            (*before - free) as f64 / MIB,
            free as f64 / MIB,
        );
    }
    let tokenizer = model.tokenizer();
    let chunk = args.step_rows.unwrap_or(args.max_rows as usize).min(args.max_rows as usize);

    if args.chat {
        let questions = [
            "Erkl\u{e4}re in drei S\u{e4}tzen, warum der Himmel blau ist.",
            "Explain in three sentences why the sky is blue.",
            "Schreibe eine Python-Funktion, die pr\u{fc}ft, ob eine Zahl eine Primzahl ist.",
        ];
        for question in questions {
            let chat = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
            let prompt = tokenizer
                .render_chat(&[Message::new("user", question)], &[], &chat)
                .expect("the chat template");
            let ids =
                tokenizer.encode(&prompt, EncodeOptions { add_special: true, parse_special: true });
            let (picks, _) = feed(&mut session, &ids, 0, chunk, Wanted::Last);
            let (tokens, times) = generate(&mut session, picks[0].token, ids.len(), args.generate);
            let end = tokens.iter().position(|t| tokenizer.is_end_of_generation(*t));
            let text = tokenizer.decode(&tokens[..end.unwrap_or(tokens.len())], false);
            let text = String::from_utf8_lossy(&text.expect("decodable tokens")).into_owned();
            let mut wall: Vec<f64> = times.iter().map(|t| t.0).collect();
            println!(
                "RES chat ({} prompt tokens, {} generated{}, {:.1} ms per token): {question}\n--- answer ---\n{text}\n--------------",
                ids.len(),
                tokens.len(),
                if end.is_some() { ", ended" } else { "" },
                median(&mut wall)
            );
        }
    }

    for dir in &args.captures {
        let ids = read_ids(&dir.join("prompt.tokens"));
        let llama_picks: Vec<u32> = read_words::<4>(&dir.join("prefill.argmax.i32"))
            .iter()
            .map(|b| i32::from_le_bytes(*b) as u32)
            .collect();
        let llama_last: Vec<f32> = read_words::<4>(&dir.join("prefill.logits_last.f32"))
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let llama_generated = read_ids(&dir.join("generated.tokens"));
        let n = ids.len();
        let (picks, _) = feed(&mut session, &ids, 0, logit_rows as usize, Wanted::All);
        let agree = picks.iter().zip(&llama_picks).filter(|(p, l)| p.token == **l).count();
        let last = picks[n - 1];
        // llama.cpp's own lead of its pick over its runner-up, where we differ.
        let llama_best = llama_last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "RES capture {}: top-1 {agree}/{n} rows equal llama.cpp's; last row picks {} with logit {:.4}, llama.cpp picks {} with {:.4} and gives ours {:.4}",
            dir.display(),
            last.token,
            last.logit,
            llama_picks[n - 1],
            llama_best,
            llama_last[last.token as usize],
        );
        let (tokens, _) = generate(&mut session, last.token, n, llama_generated.len().min(40));
        let same = tokens.iter().zip(&llama_generated).take_while(|(a, b)| a == b).count();
        let ours = tokenizer.decode(&tokens, false).expect("decodable tokens");
        let theirs = tokenizer.decode(&llama_generated, false).expect("decodable tokens");
        println!(
            "RES capture {}: {same}/{} greedy tokens equal llama.cpp's\n--- ours ---\n{}\n--- llama.cpp ---\n{}\n-------------",
            dir.display(),
            tokens.len(),
            String::from_utf8_lossy(&ours),
            String::from_utf8_lossy(&theirs),
        );
    }

    // Long text for the speed measurements: a file, repeated as often as needed.
    let corpus: Vec<u32> = match &args.text {
        Some(path) => {
            let text = std::fs::read(path).expect("the text file");
            tokenizer.encode(&text, EncodeOptions { add_special: true, parse_special: false })
        }
        None => Vec::new(),
    };
    let long = |n: usize| -> Vec<u32> {
        assert!(!corpus.is_empty(), "--text FILE is needed for the speed measurements");
        corpus.iter().copied().cycle().take(n).collect()
    };

    if let Some(n) = args.prefill {
        let tokens = long(n);
        // Twice: the first pass pays for the first use of every kernel shape.
        for pass in 0..2 {
            profile.reset();
            let (_, time) = feed(&mut session, &tokens, 0, chunk, Wanted::Last);
            println!(
                "RES prefill pass {pass}: {n} tokens in steps of {chunk} rows: {:.0} ms, {:.0} tokens/s",
                time.as_secs_f64() * 1e3,
                n as f64 / time.as_secs_f64()
            );
            // The profile of a step is closed by the next one, so the last
            // step of the pass is not in these totals.
            print_profile(&profile, &format!("prefill pass {pass}"));
        }
    }

    for &context in &args.decode_at {
        let tokens = long(context as usize);
        let (picks, time) = feed(&mut session, &tokens, 0, chunk, Wanted::Last);
        println!(
            "RES context {context}: read in {:.1} s ({:.0} tokens/s)",
            time.as_secs_f64(),
            f64::from(context) / time.as_secs_f64()
        );
        // A few steps to warm up, then the measured ones; one more step
        // closes the profile of the last measured one.
        let (warm, _) = generate(&mut session, picks[0].token, tokens.len(), 4);
        profile.reset();
        let position = tokens.len() + warm.len() - 1;
        let (generated, times) =
            generate(&mut session, *warm.last().expect("a token"), position, args.decode_steps + 2);
        let mut wall: Vec<f64> = times[..args.decode_steps].iter().map(|t| t.0).collect();
        let mut device: Vec<f64> = times[..args.decode_steps].iter().map(|t| t.1).collect();
        let (wall, device) = (median(&mut wall), median(&mut device));
        // The fastest step is the one the host disturbed least: on a busy
        // machine the device threads lose their cores between kernels.
        let fastest = times[..args.decode_steps].iter().map(|t| t.0).fold(f64::INFINITY, f64::min);
        println!(
            "RES decode at context {context}: {wall:.2} ms per step wall (median of {}, fastest {fastest:.2}), {device:.2} ms on the slowest device, {:.1} tokens/s",
            args.decode_steps,
            1e3 / wall
        );
        print_profile(&profile, &format!("decode at context {context}"));
        balance(&profile, model.placement(), model.config(), args.balance_out.as_deref());
        let text = tokenizer.decode(&generated, false).expect("decodable tokens");
        println!("RES text after context {context}: {:?}", String::from_utf8_lossy(&text));
    }
}

/// One request through the engine. Returns the answer, the seconds from its
/// first piece of text to its end and the completion tokens.
fn ask(
    runtime: &ares_runtime::Runtime,
    id: u64,
    question: &str,
    sampling: ares_api::Sampling,
    max_tokens: u32,
) -> (String, f64, u32) {
    use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Role};
    use std::sync::{Mutex, mpsc};
    let (tx, rx) = mpsc::channel();
    let state = Arc::new(Mutex::new((String::new(), None::<Instant>)));
    let sink = {
        let state = state.clone();
        EventSink::new(move |event| {
            match event {
                Event::Text(piece) => {
                    let mut state = state.lock().unwrap();
                    state.0.push_str(&piece);
                    state.1.get_or_insert_with(Instant::now);
                }
                Event::Done { usage, .. } => {
                    let _ = tx.send(Ok(usage.completion_tokens));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            true
        })
    };
    runtime
        .submit(
            GenerateRequest {
                model: String::new(),
                prompt: Prompt::Chat {
                    messages: vec![ares_api::Message::new(Role::User, question)],
                    tools: None,
                },
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                id: RequestId(id),
            },
            sink,
        )
        .expect("submit");
    let tokens = rx.recv().expect("the request ends").expect("the request");
    let (text, first) = std::mem::take(&mut *state.lock().unwrap());
    (text, first.map_or(0.0, |t| t.elapsed().as_secs_f64()), tokens)
}

/// `--engine`: chat requests through the conductor, with the picks made on
/// the devices and, when the model has a draft block, its own drafts.
fn engine(args: &Args, spread: &Spread, drafts: u32) {
    use ares_runtime::{DevicePhase, GpuConfig, Phase, Runtime, RuntimeConfig};
    let vocabulary: Option<Arc<[u32]>> = args.draft_vocab.as_ref().map(|path| {
        let text = std::fs::read_to_string(path).expect("the draft vocabulary");
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.parse().expect("a token id"))
            .collect()
    });
    let profile = Arc::new(Profile::new());
    let options = Options {
        precision: args.precision,
        state: args.state,
        max_rows: args.max_rows,
        logit_rows: 16,
        window_rows: if drafts > 0 { drafts + 1 } else { 0 },
        draft_vocabulary: vocabulary,
        pipeline_rows: args.pipeline_rows,
        profile: args.profile.then(|| profile.clone()),
        ..Options::default()
    };
    let devices: Vec<Device> =
        args.devices.iter().map(|d| Device::open(*d, Wait::Block).expect("open device")).collect();
    let free_before: Vec<u64> =
        devices.iter().map(|d| d.mem_info().expect("memory").free).collect();
    let begun = Instant::now();
    let gguf = Gguf::open(&args.model).expect("the model file");
    let chat = ChatOptions {
        variables: args
            .chat_vars
            .iter()
            .map(|(name, text)| (name.clone(), ares_tokenizer::Json::String(text.clone())))
            .collect(),
        ..ChatOptions::default()
    };
    let model = Arc::new(
        Qwen35::new(gguf, String::new(), spread, options, args.context)
            .expect("model")
            .with_chat_options(chat),
    );
    let config = RuntimeConfig {
        pages: 2 * args.context.div_ceil(128) + 2,
        cells_per_page: 128,
        max_slots: 2,
        speculation: drafts > 0,
        max_draft: drafts.max(1),
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let runtime = Runtime::start_gpu(model.clone(), devices.clone(), config, gpu).expect("start");
    println!(
        "RES engine: loaded in {:.1} s, at most {drafts} drafts",
        begun.elapsed().as_secs_f64()
    );
    for (report, (device, before)) in model.reports().iter().zip(devices.iter().zip(&free_before)) {
        let taken = before - device.mem_info().expect("memory").free;
        println!(
            "RES device {} ({}): weights {:.0} MiB, all taken {:.0} MiB",
            report.index,
            report.device,
            report.weight_bytes as f64 / MIB,
            taken as f64 / MIB
        );
        if let Some(d) = &report.draft {
            println!(
                "RES device {} draft block: weights {:.1} MiB, K/V {:.3} MiB per 1000 tokens, kept \
                 hidden states {:.2} MiB, chain scratch {:.2} MiB, records {:.2} MiB, step scratch \
                 grown by at most {:.2} MiB",
                report.index,
                d.weights as f64 / MIB,
                d.page as f64 / 128.0 * 1000.0 / MIB,
                d.carry as f64 / MIB,
                d.scratch as f64 / MIB,
                d.records as f64 / MIB,
                d.step_scratch as f64 / MIB
            );
        }
    }

    let questions = [
        (
            "german",
            "Schreibe einen ausf\u{fc}hrlichen Aufsatz \u{fc}ber die Geschichte der Eisenbahn in \
             Deutschland, von den Anf\u{e4}ngen im 19. Jahrhundert bis heute.",
        ),
        (
            "english",
            "Write a detailed essay about the history of the printing press and how it changed \
             Europe.",
        ),
        (
            "code",
            "Write a Rust module that implements an LRU cache with get, put and a capacity limit, \
             with unit tests.",
        ),
    ];
    let greedy = ares_api::Sampling { temperature: Some(0.0), ..ares_api::Sampling::default() };
    let sampled = ares_api::Sampling {
        temperature: Some(1.0),
        top_k: Some(20),
        top_p: Some(0.95),
        min_p: Some(0.0),
        seed: Some(20_261_110),
        ..ares_api::Sampling::default()
    };
    let mut settings = vec![("greedy", greedy)];
    if args.sample {
        settings.push(("sampled", sampled));
    }
    // The first request also pays for whatever the first steps of a process
    // pay; it is not measured.
    let _ = ask(&runtime, 1, "Say hello.", settings[0].1.clone(), 8);
    let mut id = 1;
    for (how, sampling) in &settings {
        for (name, question) in questions {
            id += 1;
            let _ = runtime.take_snapshot();
            profile.reset();
            let (text, seconds, tokens) =
                ask(&runtime, id, question, sampling.clone(), args.generate as u32);
            let Some(snapshot) = runtime.snapshot() else { panic!("the engine is gone") };
            let c = snapshot.counters;
            if let Some(dir) = &args.out {
                std::fs::create_dir_all(dir).expect("the output directory");
                std::fs::write(dir.join(format!("{name}.{how}.txt")), &text).expect("write");
            }
            let step = snapshot.phase(Phase::Step);
            println!(
                "RES {name} {how}: {tokens} tokens in {seconds:.2} s after the first = {:.1} \
                 tokens/s; {} steps ({} with rows of a prompt), {:.2} tokens per step, step \
                 median {:.2} ms mean {:.2} ms",
                f64::from(tokens.saturating_sub(1)) / seconds.max(1e-9),
                c.steps,
                c.steps - c.window_rows.min(c.steps),
                c.tokens as f64 / c.window_rows.max(1) as f64,
                step.p50_us / 1e3,
                step.mean_us / 1e3,
            );
            for (source, label) in [(0, "lookup"), (1, "own")] {
                let tested = c.drafts_tested_at[source];
                if tested[0] == 0 {
                    continue;
                }
                let rates: Vec<String> = tested
                    .iter()
                    .zip(&c.drafts_accepted_at[source])
                    .take_while(|(t, _)| **t > 0)
                    .map(|(t, a)| format!("{:.2} ({a}/{t})", *a as f64 / *t as f64))
                    .collect();
                println!(
                    "RES {name} {how}: {label} drafts accepted by position: {}",
                    rates.join(", ")
                );
            }
            if c.own_draft_steps > 0 {
                println!(
                    "RES {name} {how}: draft block {:.2} ms per step that drafts ({:.2} links), \
                     {:.2} ms per link",
                    c.own_draft_us as f64 / 1e3 / c.own_draft_steps as f64,
                    c.own_draft_links as f64 / c.own_draft_steps as f64,
                    c.own_draft_us as f64 / 1e3 / c.own_draft_links.max(1) as f64,
                );
            }
            let devices: Vec<String> = (0..snapshot.devices.len())
                .map(|d| {
                    format!(
                        "device {d}: {:.2} ms on the device, sampler {:.3} ms, enqueue {:.2} ms",
                        snapshot.device_phase(d, DevicePhase::Device).p50_us / 1e3,
                        snapshot.device_phase(d, DevicePhase::Pick).p50_us / 1e3,
                        snapshot.device_phase(d, DevicePhase::Enqueue).p50_us / 1e3,
                    )
                })
                .collect();
            let slowest = (0..snapshot.devices.len())
                .map(|d| snapshot.device_phase(d, DevicePhase::Device).p50_us)
                .fold(0.0, f64::max);
            println!(
                "RES {name} {how}: per step (medians): {}; host besides the devices {:.3} ms \
                 (plan {:.3}, record {:.3}, commit {:.3})",
                devices.join("; "),
                (step.p50_us - slowest) / 1e3,
                snapshot.phase(Phase::Plan).p50_us / 1e3,
                snapshot.phase(Phase::Write).p50_us / 1e3,
                snapshot.phase(Phase::Commit).p50_us / 1e3,
            );
            if args.profile {
                print_profile(&profile, &format!("{name} {how}"));
            }
            let shown: String = text.chars().take(160).collect();
            println!("RES {name} {how} text: {}", shown.replace('\n', " "));
        }
    }
    runtime.shutdown();
}
