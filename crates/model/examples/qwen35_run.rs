//! Runs a `qwen35` model file on one or two GPUs through the runtime's
//! executor and reports what it takes and how fast it is.
//!
//! ```text
//! qwen35_run <model.gguf> [--split single|tensor|layer] [--shares 0.625,0.375]
//!     [--devices 0,1] [--precision fast|exact] [--max-rows 512] [--context 4096]
//!     [--chat] [--generate 48] [--capture DIR]... [--decode-at 1000,32000]
//!     [--decode-steps 64] [--prefill 2048] [--text FILE] [--state f32|f16]
//!     [--pipeline-rows 256] [--no-profile]
//! ```
//!
//! - `--chat`: a German and an English question through the chat template,
//!   greedy, decoded with the tokenizer.
//! - `--capture DIR`: a directory of `tools/capture` (`prompt.tokens`,
//!   `prefill.argmax.i32`, `prefill.logits_last.f32`, `generated.tokens`):
//!   top-1 agreement of every row with llama.cpp and the greedy tokens.
//! - `--decode-at N`: milliseconds per decode step with `N` tokens of context,
//!   with the device time by kind of kernel.
//! - `--prefill N`: prompt tokens per second for `N` tokens.
//!
//! Lines that start with `RES` are results.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_cuda::{Device, Precision, StateFormat, Wait};
use ares_gguf::Gguf;
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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
}

fn main() {
    let args = parse();
    let spread = match args.split.as_str() {
        "single" => Spread::Single,
        "tensor" => Spread::Tensor(args.shares.clone()),
        "layer" => Spread::Layer(args.shares.clone()),
        other => panic!("unknown split {other}"),
    };
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
    let chunk = args.max_rows as usize;

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
            let (_, time) = feed(&mut session, &tokens, 0, chunk, Wanted::Last);
            println!(
                "RES prefill pass {pass}: {n} tokens in steps of {chunk} rows: {:.0} ms, {:.0} tokens/s",
                time.as_secs_f64() * 1e3,
                n as f64 / time.as_secs_f64()
            );
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
        for (index, (steps, ms)) in profile.totals().iter().enumerate() {
            let per_step: Vec<String> = PHASES
                .iter()
                .map(|p| format!("{p:?} {:.2}", ms[*p as usize] / *steps as f64))
                .collect();
            let total: f64 = ms.iter().sum::<f64>() / *steps as f64;
            println!(
                "RES decode at context {context} device {index}: {} ms per step, sum {total:.2} ({steps} steps)",
                per_step.join(", ")
            );
        }
        let text = tokenizer.decode(&generated, false).expect("decodable tokens");
        println!("RES text after context {context}: {:?}", String::from_utf8_lossy(&text));
    }
}
