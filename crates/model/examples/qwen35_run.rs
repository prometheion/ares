//! Runs a `qwen35` model file on one or two GPUs through the runtime's
//! executor and reports what it takes and how fast it is.
//!
//! ```text
//! qwen35_run <model.gguf> [--split single|tensor|layer] [--shares 0.625,0.375]
//!     [--devices 0,1] [--precision fast|exact] [--max-rows N] [--context 4096]
//!     [--chat] [--generate 48] [--capture DIR]... [--decode-at 1000,32000]
//!     [--decode-steps 64] [--prefill 2048] [--text FILE] [--state f32|f16]
//!     [--pipeline-rows 256] [--no-profile] [--balance FILE] [--balance-out FILE]
//!     [--step-rows N] [--layouts measured|file|device] [--prompt-weight 1]
//!     [--show-layouts] [--kv q8,q8] [--fingerprint FILE]
//! ```
//!
//! - `--max-rows N`: the rows of a step (and so the scratch); by default the
//!   default of the model's options.
//! - `--kv K,V`: the formats of K and of V of the attention blocks (`f16`, `q8`
//!   to `q4`; the default is `q8,q8`); the MTP block's are always `q4`.
//!   With `q7` to `q4` a second, small pool of exact entries holds the first
//!   tile of a sequence and its newest cells.
//! - `--chat`: a German and an English question through the chat template,
//!   greedy, decoded with the tokenizer.
//! - `--capture DIR`: a directory of `tools/capture` (`prompt.tokens`,
//!   `prefill.argmax.i32`, `prefill.logits_last.f32`, `generated.tokens`):
//!   top-1 agreement of every row with llama.cpp and the greedy tokens.
//! - `--decode-at N`: milliseconds per decode step with `N` tokens of context,
//!   with the device time by kind of kernel.
//! - `--prefill N`: prompt tokens per second for `N` tokens, in steps of
//!   `--step-rows` rows (default: `--max-rows`).
//! - `--layouts`: how every weight matrix gets its layout on a device:
//!   `measured` (the default) times both layouts per matrix at load and takes
//!   the device layout where it does not make a decode step slower, and of
//!   those assignments the one that reads prompts fastest; with
//!   `--prompt-weight W` a percent of prompt speed buys `W` percent of decode
//!   speed. `file` and `device` take one layout for every matrix that can
//!   have it, for comparison. `--show-layouts` prints every probed shape
//!   with its times.
//! - `--balance-out FILE`: with a tensor split over two devices and
//!   `--decode-at`, writes the share of device 0 in every half block at which
//!   both devices would take equally long, from the device times of the
//!   decode steps. `--balance FILE` places the model by such a file.
//!
//! - `--engine W`: the whole engine instead of the session (conductor, picks
//!   on the device, drafts): seven chat requests (German prose, English prose,
//!   new code, a rewrite of a given file, then two more of each of the first
//!   three kinds) of `--generate` tokens, each
//!   most-probable-token and, with `--sample`, sampled, with windows of up to
//!   `W` rows (0: no drafts). With a model that has an MTP block the engine
//!   drafts with it and with the n-gram drafter, and chooses for every
//!   step whose drafts and how many from the price of a step (measured at
//!   load) and the acceptance (measured while running); `--draft-vocab FILE`
//!   restricts the block's drafts to the tokens of a list (an id per line).
//!   `--out DIR` keeps the answers, to compare runs. `--undrafted` asks every
//!   request again without drafts (`"draft": false`) and a third time with
//!   them, and says where the texts part; `--requests N` asks the first `N`
//!   of the seven. `--chat-var name=value`
//!   sets a variable of the chat template (`reasoning_effort=medium`). Reports
//!   tokens per second, acceptance by draft position, the lengths of the
//!   drafts and the time of a step by kind.
//!   `--dflash FILE` drafts with a DFlash2 drafter as well, `--no-mtp`
//!   without the model's MTP block, `--mtp-file FILE` with the MTP block of a
//!   file of its own (architecture `qwen35-mtp`).
//! - `--conversation TOKENS`: the prefix cache on the devices. A conversation
//!   whose history is about `TOKENS` tokens (a slice of `--text`) has a second
//!   turn, and a conversation of the same size that the engine has never seen
//!   has one too; the time to the first token of each, the rows replayed and
//!   computed, what a save and a restore of a checkpoint took on each device
//!   (copies of the device's recurrent state to and from page-locked RAM),
//!   and the RAM locked are reported. `--checkpoints N` is the number of
//!   checkpoints the engine keeps (16 for `--conversation`, none otherwise);
//!   `--window W` the rows of a sequence's window (8).
//!
//! - `--fingerprint FILE`: the picks of fixed steps over `--text` (a prompt,
//!   generated tokens, windows of drafts, every row of a short prompt), a line
//!   per pick with its token and its logit's bits, to compare two builds bit
//!   for bit (with `--layouts file` or `device`: the measured layouts change
//!   between loads).
//!
//! Lines that start with `RES` are results.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_cuda::{CacheFormat, Device, Precision, StateFormat, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::Config;
use ares_model::qwen35::device::placement::{HalfShares, Placement};
use ares_model::qwen35::device::{
    Layouts, Options, PHASES, PartReport, Profile, Qwen35, Rows, Session, Spread, Wanted,
};
use ares_runtime::Pick;
use ares_tokenizer::{ChatOptions, EncodeOptions, Message};

const MIB: f64 = 1024.0 * 1024.0;

/// The file of the rewrite task of `--engine`.
const REWRITE_SOURCE: &str = "use std::collections::HashMap;

/// A least-recently-used cache with a fixed capacity.
pub struct Lru<K, V> {
    capacity: usize,
    tick: u64,
    items: HashMap<K, (V, u64)>,
}

impl<K: std::hash::Hash + Eq + Clone, V> Lru<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, tick: 0, items: HashMap::with_capacity(capacity) }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.tick += 1;
        let tick = self.tick;
        self.items.get_mut(key).map(|entry| {
            entry.1 = tick;
            &entry.0
        })
    }

    pub fn put(&mut self, key: K, value: V) {
        self.tick += 1;
        if self.items.len() >= self.capacity && !self.items.contains_key(&key) {
            let oldest = self.items.iter().min_by_key(|(_, (_, used))| *used).map(|(k, _)| k.clone());
            if let Some(oldest) = oldest {
                self.items.remove(&oldest);
            }
        }
        self.items.insert(key, (value, self.tick));
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
";

struct Args {
    model: PathBuf,
    split: String,
    shares: Vec<f64>,
    devices: Vec<u32>,
    precision: Precision,
    state: StateFormat,
    kv: (CacheFormat, CacheFormat),
    max_rows: Option<u32>,
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
    conversation: Option<usize>,
    checkpoints: Option<u64>,
    window: u32,
    sample: bool,
    undrafted: bool,
    requests: usize,
    draft_vocab: Option<PathBuf>,
    /// A DFlash drafter file, and whether the model's MTP block drafts.
    dflash: Option<PathBuf>,
    mtp: bool,
    mtp_file: Option<PathBuf>,
    out: Option<PathBuf>,
    chat_vars: Vec<(String, String)>,
    layouts: Layouts,
    show_layouts: bool,
    fingerprint: Option<PathBuf>,
}

impl Args {
    /// The rows of a step: the one asked for, or the default of the options.
    fn rows(&self) -> u32 {
        self.max_rows.unwrap_or(Options::default().max_rows)
    }
}

fn parse() -> Args {
    let mut args = Args {
        model: PathBuf::new(),
        split: "single".into(),
        shares: vec![0.625, 0.375],
        devices: vec![0],
        precision: Precision::Fast,
        state: StateFormat::F32,
        kv: (CacheFormat::Q8, CacheFormat::Q8),
        max_rows: None,
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
        conversation: None,
        checkpoints: None,
        window: 8,
        sample: false,
        undrafted: false,
        requests: usize::MAX,
        draft_vocab: None,
        dflash: None,
        mtp: true,
        mtp_file: None,
        out: None,
        chat_vars: Vec::new(),
        layouts: Layouts::default(),
        show_layouts: false,
        fingerprint: None,
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
            "--kv" => args.kv = formats(&value()),
            "--max-rows" => args.max_rows = Some(value().parse().expect("rows")),
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
            "--engine" => args.engine = Some(value().parse().expect("rows of a window")),
            "--conversation" => args.conversation = Some(value().parse().expect("tokens")),
            "--checkpoints" => args.checkpoints = Some(value().parse().expect("checkpoints")),
            "--window" => args.window = value().parse().expect("rows of a window"),
            "--sample" => args.sample = true,
            "--undrafted" => args.undrafted = true,
            "--requests" => args.requests = value().parse().expect("requests"),
            "--draft-vocab" => args.draft_vocab = Some(value().into()),
            "--dflash" => args.dflash = Some(value().into()),
            "--no-mtp" => args.mtp = false,
            "--mtp-file" => args.mtp_file = Some(value().into()),
            "--out" => args.out = Some(value().into()),
            "--layouts" => {
                args.layouts = match value().as_str() {
                    "measured" => Layouts::default(),
                    "file" => Layouts::File,
                    "device" => Layouts::Device,
                    other => panic!("--layouts: measured, file or device, not {other}"),
                }
            }
            "--prompt-weight" => {
                let weight = value().parse().expect("the weight of prompt speed");
                args.layouts = Layouts::Measured { prompt_weight: weight };
            }
            "--show-layouts" => args.show_layouts = true,
            "--fingerprint" => args.fingerprint = Some(value().into()),
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

/// `K,V` as two K/V formats.
fn formats(text: &str) -> (CacheFormat, CacheFormat) {
    let name = |name: &str| {
        CacheFormat::ALL
            .into_iter()
            .find(|f| format!("{f:?}").eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("a K/V format (f16, q8 to q4), not {name}"))
    };
    match text.split_once(',') {
        Some((k, v)) => (name(k), name(v)),
        None => (name(text), name(text)),
    }
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

/// The picks of fixed steps, written to `out` as the token and the bits of
/// its logit per wanted row, for comparing two builds bit for bit: `prompt`
/// read in steps of `chunk` rows and 40 tokens generated from it; the same
/// again verified in windows of four rows whose drafts are the generated
/// tokens, every third window with a wrong second draft; then `rows` read in
/// steps of 16 with every row's pick.
fn fingerprint(session: &mut Session, prompt: &[u32], rows: &[u32], chunk: usize, out: &Path) {
    let mut lines = Vec::new();
    let mut note = |pick: &Pick| lines.push(format!("{} {:08x}", pick.token, pick.logit.to_bits()));
    let (picks, _) = feed(session, prompt, 0, chunk, Wanted::Last);
    note(&picks[0]);
    let mut generated = vec![picks[0].token];
    while generated.len() < 40 {
        let token = *generated.last().expect("a token");
        let position = (prompt.len() + generated.len() - 1) as u32;
        let rows = Rows { slot: 0, position, tokens: &[token], direct: true, wanted: Wanted::Last };
        let pick = session.step(&[rows]).expect("a step")[0];
        note(&pick);
        generated.push(pick.token);
    }
    // Windows: the drafts are the generated tokens, so most are accepted.
    let (picks, _) = feed(session, prompt, 0, chunk, Wanted::Last);
    note(&picks[0]);
    let (mut done, mut window) = (1usize, 0usize);
    let mut last = picks[0].token;
    while done + 4 < generated.len() {
        let mut tokens = vec![last];
        tokens.extend_from_slice(&generated[done..done + 3]);
        if window % 3 == 2 {
            tokens[2] = 13;
        }
        let position = (prompt.len() + done - 1) as u32;
        let rows = Rows { slot: 0, position, tokens: &tokens, direct: false, wanted: Wanted::All };
        let picks = session.step(&[rows]).expect("a window").to_vec();
        picks.iter().for_each(&mut note);
        let mut accepted = 1;
        while accepted < 4 && picks[accepted - 1].token == tokens[accepted] {
            accepted += 1;
        }
        session.accept(&[accepted as u32]).expect("the outcome of a window");
        last = picks[accepted - 1].token;
        done += accepted;
        window += 1;
    }
    // Every row's pick, in steps of 16 rows.
    let (picks, _) = feed(session, rows, 0, 16, Wanted::All);
    picks.iter().for_each(&mut note);
    let text = lines.join("\n") + "\n";
    std::fs::write(out, &text).expect("the fingerprint file");
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    println!("RES fingerprint: {} picks, hash {hash:016x}, in {}", lines.len(), out.display());
}

/// Prints where a device's token embedding lies and what it takes.
fn print_embedding(report: &PartReport) {
    let e = &report.embedding;
    if !e.embeds {
        return;
    }
    let kept = match (e.kept_rows, e.shared_with_head) {
        (0, _) => "no rows on the device".to_owned(),
        (rows, true) => format!("{rows} rows on the device, the first rows of the head"),
        (rows, false) => {
            format!(
                "{rows} rows ({:.1} MiB) on the device for the draft chain",
                e.kept_bytes as f64 / MIB
            )
        }
    };
    println!(
        "RES embedding device {}: table {:.1} MiB, {:.1} MiB page-locked RAM (the table and the rows of a step), {kept}, {:.1} MiB of the step scratch for staged rows",
        report.index,
        e.table_bytes as f64 / MIB,
        e.host_bytes as f64 / MIB,
        e.staged_bytes as f64 / MIB,
    );
}

/// Prints the page-locked RAM the model holds for the token embedding and
/// what the kernel counts for this process. Memory the driver page-locks is
/// shared memory of the process to the kernel (`RssShmem`), with every other
/// buffer the driver locked for it: the step record, the read-back, the
/// staging of the uploads. `VmLck` and `VmPin` do not count it.
fn print_locked(model: &Qwen35) {
    use ares_runtime::Model;
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let shared = status
        .lines()
        .find_map(|line| {
            line.strip_prefix("RssShmem:")?.split_whitespace().next()?.parse::<f64>().ok()
        })
        .unwrap_or(0.0);
    println!(
        "RES RAM locked: {:.1} MiB for the token embedding (booked as required); the kernel counts {:.1} MiB of shared memory in this process, which is all the driver locked for it",
        model.pinned_ram() as f64 / MIB,
        shared / 1024.0,
    );
}

/// Prints every reservation of a device's ledger, and what the ledger does not
/// know of the memory the device holds.
fn print_ledger(report: &PartReport, held: u64, taken: u64) {
    let total: u64 = report.ledger.iter().map(|(_, bytes)| bytes).sum();
    for (name, bytes) in &report.ledger {
        println!("RES ledger device {}: {:>9.2} MiB  {name}", report.index, *bytes as f64 / MIB);
    }
    println!(
        "RES ledger device {}: {:>9.2} MiB reserved in all; taken {:.2} MiB, so {:.2} MiB are \
         not in the ledger (the sampler, allocation granularity, kernels loaded); the device \
         holds {:.2} MiB in all",
        report.index,
        total as f64 / MIB,
        taken as f64 / MIB,
        (taken as f64 - total as f64) / MIB,
        held as f64 / MIB,
    );
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

/// Prints which layouts the weight matrices of a device got and what the
/// choice took; with `all`, every probed shape and its times.
fn print_layouts(report: &PartReport, all: bool) {
    let l = &report.layouts;
    println!(
        "RES layout device {}: {} matrices ({:.0} MiB) in the file's layout, {} ({:.0} MiB) in the \
         device layout, padding {:.2} MiB; {} shapes probed in {:.0} ms on {:.0} MiB, kernel \
         tables {:.0} ms",
        report.index,
        l.matrices[0],
        l.bytes[0] as f64 / MIB,
        l.matrices[1],
        l.bytes[1] as f64 / MIB,
        l.padding as f64 / MIB,
        l.probes.len(),
        l.probes_time.as_secs_f64() * 1e3,
        l.probe_bytes as f64 / MIB,
        l.tables.as_secs_f64() * 1e3,
    );
    if !l.paths.is_empty() {
        let paths: Vec<String> = l
            .paths
            .iter()
            .map(|c| {
                let ms = c
                    .probe_ms
                    .map(|ms| if ms.is_finite() { format!("{ms:.3}") } else { "-".into() });
                format!(
                    "{} {:?}{} (int8 {} f16 {} f16 flushed {} ms)",
                    c.ty.name(),
                    c.path,
                    if c.measured { "" } else { " kept" },
                    ms[0],
                    ms[1],
                    ms[2]
                )
            })
            .collect();
        println!("RES layout device {} tile paths: {}", report.index, paths.join(", "));
    }
    if !l.probes.is_empty() {
        println!(
            "RES layout device {}: by the probes the weight products of a decode step take \
             {:.2} ms in the file's layout and {:.2} ms as chosen, those of a group of prompt \
             rows {:.1} and {:.1} ms; chosen knowing the times of {} other device(s)",
            report.index,
            l.decode_ms[0],
            l.decode_ms[1],
            l.prompt_ms[0],
            l.prompt_ms[1],
            l.balanced_with,
        );
    }
    if !all {
        return;
    }
    for (n_in, rows, ms) in &l.preparations {
        println!(
            "RES layout device {} preparing {rows} row(s) of {n_in}: 8-bit {:.4} ms, f16 {:.4} ms",
            report.index, ms[0], ms[1]
        );
    }
    for p in &l.probes {
        println!(
            "RES layout device {} {:>7} {:>6} x {:<5}: {} row(s) file {:.4} device {:.4} ms; {} \
             rows file {:.3} device {:.3} ms -> {} matrices in the file's layout, {} in the \
             device layout",
            report.index,
            p.ty.name(),
            p.n_out,
            p.n_in,
            p.decode_rows,
            p.file_ms[0],
            p.device_ms[0],
            p.prompt_rows,
            p.file_ms[1],
            p.device_ms[1],
            p.matrices[0],
            p.matrices[1],
        );
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
    if let Some(window) = args.engine {
        return engine(&args, &spread, window);
    }
    if let Some(history) = args.conversation {
        return conversation(&args, &spread, history);
    }
    let profile = Arc::new(Profile::new());
    let logit_rows = 16;
    let options = Options {
        precision: args.precision,
        layouts: args.layouts,
        state: args.state,
        kv: args.kv,
        max_rows: args.rows(),
        logit_rows,
        // The fingerprint verifies windows of drafts.
        window_rows: if args.fingerprint.is_some() { 8 } else { 0 },
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
        args.rows(),
        args.context
    );
    for (report, (device, before)) in model.reports().iter().zip(devices.iter().zip(&free_before)) {
        let memory = device.mem_info().expect("memory");
        let free = memory.free;
        println!(
            "RES device {} {}: {} blocks, weights {:.0} MiB (upload {:.2} s), recurrent state {:.1} MiB per sequence, K/V {:.2} MiB per 1000 tokens, exact K/V entries {:.1} MiB, step scratch {:.1} MiB, all VRAM taken {:.0} MiB (the device holds {:.0} MiB in all), free {:.0} MiB",
            report.index,
            report.device,
            report.blocks,
            report.weight_bytes as f64 / MIB,
            report.upload.as_secs_f64(),
            report.state_bytes_per_slot as f64 / MIB,
            report.page_bytes as f64 / 128.0 * 1000.0 / MIB,
            (report.exact_bytes + report.mtp.as_ref().map_or(0, |d| d.exact)) as f64 / MIB,
            report.scratch_bytes as f64 / MIB,
            (*before - free) as f64 / MIB,
            (memory.total - free) as f64 / MIB,
            free as f64 / MIB,
        );
        print_embedding(report);
        print_ledger(report, memory.total - free, *before - free);
        print_layouts(report, args.show_layouts);
    }
    print_locked(&model);
    let tokenizer = model.tokenizer();
    let chunk = args.step_rows.unwrap_or(args.rows() as usize).min(args.rows() as usize);

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

    if let Some(out) = &args.fingerprint {
        fingerprint(&mut session, &long(1100), &corpus[..200.min(corpus.len())], chunk, out);
    }

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
        let fastest_device =
            times[..args.decode_steps].iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
        println!(
            "RES decode at context {context}: {wall:.2} ms per step wall (median of {}, fastest {fastest:.2}), {device:.2} ms on the slowest device (fastest {fastest_device:.3}), {:.1} tokens/s",
            args.decode_steps,
            1e3 / wall
        );
        print_profile(&profile, &format!("decode at context {context}"));
        balance(&profile, model.placement(), model.config(), args.balance_out.as_deref());
        let text = tokenizer.decode(&generated, false).expect("decodable tokens");
        println!("RES text after context {context}: {:?}", String::from_utf8_lossy(&text));
    }
}

/// One request through the engine, with drafts or without (`"draft": false`).
/// Returns the answer, the seconds from its first piece of text to its end and
/// the completion tokens.
fn ask(
    runtime: &ares_runtime::Runtime,
    id: u64,
    question: &str,
    sampling: ares_api::Sampling,
    max_tokens: u32,
    draft: bool,
) -> (String, f64, u32) {
    use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Role};
    use std::sync::{Mutex, mpsc};
    let (tx, rx) = mpsc::channel();
    let state = Arc::new(Mutex::new((String::new(), None::<Instant>)));
    let sink = {
        let state = state.clone();
        EventSink::new(move |event| {
            match event {
                Event::Logprobs(_) => {}
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
                    variables: Vec::new(),
                    continue_final: false,
                },
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft,
                structure: None,
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
/// the devices and, when the model has an MTP block, its own drafts.
fn engine(args: &Args, spread: &Spread, window: u32) {
    use ares_runtime::{DevicePhase, GpuConfig, Model, Phase, Runtime, RuntimeConfig};
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
        layouts: args.layouts,
        state: args.state,
        kv: args.kv,
        max_rows: args.rows(),
        logit_rows: (2 * window).max(16),
        window_rows: if window > 1 { 2 * window } else { 0 },
        draft_vocabulary: vocabulary,
        mtp: args.mtp,
        mtp_file: args
            .mtp_file
            .as_ref()
            .map(|path| Arc::new(Gguf::open(path).expect("the file of the MTP block"))),
        dflash: args.dflash.as_ref().map(|path| {
            Arc::new(ares_model::dflash::File::open(path).expect("the DFlash drafter"))
        }),
        pipeline_rows: args.pipeline_rows,
        profile: args.profile.then(|| profile.clone()),
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
        window_rows: window.max(1),
        cache_ram: args.checkpoints.unwrap_or(0) * model.prefix_cache().map_or(0, |c| c.bytes()),
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let runtime = Runtime::start_gpu(model.clone(), devices.clone(), config, gpu).expect("start");
    println!(
        "RES engine: loaded in {:.1} s, windows of at most {window} rows",
        begun.elapsed().as_secs_f64()
    );
    for (report, (device, before)) in model.reports().iter().zip(devices.iter().zip(&free_before)) {
        let memory = device.mem_info().expect("memory");
        let taken = before - memory.free;
        println!(
            "RES device {} ({}): weights {:.0} MiB, all taken {:.0} MiB (the device holds {:.0} MiB in all)",
            report.index,
            report.device,
            report.weight_bytes as f64 / MIB,
            taken as f64 / MIB,
            (memory.total - memory.free) as f64 / MIB,
        );
        print_embedding(report);
        print_ledger(report, memory.total - memory.free, taken);
        print_layouts(report, args.show_layouts);
        if let Some(d) = &report.mtp {
            println!(
                "RES device {} MTP block: weights {:.1} MiB, K/V {:.3} MiB per 1000 tokens, exact \
                 K/V entries {:.2} MiB, kept hidden states {:.2} MiB, chain scratch {:.2} MiB, \
                 records {:.2} MiB, step scratch grown by at most {:.2} MiB",
                report.index,
                d.weights as f64 / MIB,
                d.page as f64 / 128.0 * 1000.0 / MIB,
                d.exact as f64 / MIB,
                d.carry as f64 / MIB,
                d.scratch as f64 / MIB,
                d.records as f64 / MIB,
                d.step_scratch as f64 / MIB
            );
        }
    }
    print_locked(&model);

    // A file to write again with two names changed: text to copy.
    let rewrite = format!(
        "Rewrite the following Rust file so that `Lru` is called `Cache` and `tick` is called \
         `clock`, and change nothing else. Give the whole file.\n\n```rust\n{REWRITE_SOURCE}```"
    );
    let questions: [(&str, &str); 7] = [
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
        ("rewrite", &rewrite),
        (
            "german2",
            "Erkl\u{e4}re ausf\u{fc}hrlich, wie ein Compiler Quelltext in Maschinencode \u{fc}bersetzt, \
             Phase f\u{fc}r Phase.",
        ),
        ("english2", "Explain in detail how TCP congestion control works and why it was needed."),
        (
            "code2",
            "Write a Python program that reads a CSV file, groups the rows by a column and prints \
             the mean of another column per group, with argument parsing and error handling.",
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
    let _ = ask(&runtime, 1, "Say hello.", settings[0].1.clone(), 8, true);
    let mut id = 1;
    let (mut compared, mut parted) = (0u32, 0u32);
    for (how, sampling) in &settings {
        for (name, question) in questions.into_iter().take(args.requests) {
            id += 1;
            let _ = runtime.take_snapshot();
            profile.reset();
            let (text, seconds, tokens) =
                ask(&runtime, id, question, sampling.clone(), args.generate as u32, true);
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
            for (source, label) in [(0, "ngram"), (1, "mtp"), (2, "dflash")] {
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
            for (source, label) in [(0, "ngram"), (1, "mtp"), (2, "dflash")] {
                let lengths = c.draft_lengths[source];
                if lengths.iter().all(|n| *n == 0) {
                    continue;
                }
                let shown: Vec<String> = lengths
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| **n > 0)
                    .map(|(len, n)| format!("{len}: {n}"))
                    .collect();
                println!(
                    "RES {name} {how}: {label} windows by drafts verified: {}",
                    shown.join(", ")
                );
            }
            if c.model_draft_steps > 0 {
                println!(
                    "RES {name} {how}: MTP block {:.2} ms per step that drafts ({:.2} links), \
                     {:.2} ms per link",
                    c.model_draft_us as f64 / 1e3 / c.model_draft_steps as f64,
                    c.model_draft_links as f64 / c.model_draft_steps as f64,
                    c.model_draft_us as f64 / 1e3 / c.model_draft_links.max(1) as f64,
                );
            }
            if c.model_links_asked > 0 {
                let by_p: Vec<String> = c
                    .model_tested_by_p
                    .iter()
                    .zip(&c.model_accepted_by_p)
                    .enumerate()
                    .filter(|(_, (tested, _))| **tested > 0)
                    .map(|(bin, (t, a))| format!("{:.2}: {a}/{t}", bin as f64 / 16.0))
                    .collect();
                println!(
                    "RES {name} {how}: own links asked {}, not made {}, made and not verified {}; \
                     accepted of verified by the drafter's probability (ranges of 1/16, from): {}",
                    c.model_links_asked,
                    c.model_links_cut,
                    c.model_drafts_unverified,
                    by_p.join(", ")
                );
            }
            let prices = |table: &[(u32, f64)]| {
                let shown: Vec<String> =
                    table.iter().take(8).map(|(n, ms)| format!("{n}: {ms:.2}")).collect();
                shown.join(", ")
            };
            let chains: Vec<String> = snapshot.chain_ms.iter().map(|c| prices(c)).collect();
            println!(
                "RES {name} {how}: prices now, ms by rows: {}; drafters by drafts: {}",
                prices(&snapshot.step_ms),
                chains.join(" | ")
            );
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
            if args.undrafted {
                // The same request without drafts, and with them once more: the
                // second and third start from what the first left in the cache.
                let _ = runtime.take_snapshot();
                id += 1;
                let (plain, _, plain_tokens) =
                    ask(&runtime, id, question, sampling.clone(), args.generate as u32, false);
                let plain_steps = runtime.snapshot().map_or(0, |s| s.counters.steps);
                id += 1;
                let (again, ..) =
                    ask(&runtime, id, question, sampling.clone(), args.generate as u32, true);
                let shared =
                    |a: &str, b: &str| a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
                let against = |other: &str| {
                    if other == text {
                        "the same text".to_owned()
                    } else {
                        format!(
                            "parts after {} of {} characters",
                            shared(&text, other),
                            text.chars().count()
                        )
                    }
                };
                compared += 1;
                parted += u32::from(plain != text);
                println!(
                    "RES {name} {how}: without drafts ({plain_tokens} tokens in {plain_steps} \
                     steps): {}; with drafts again: {}",
                    against(&plain),
                    against(&again),
                );
                if let Some(dir) = &args.out {
                    std::fs::write(dir.join(format!("{name}.{how}.undrafted.txt")), &plain)
                        .expect("write");
                }
            }
        }
    }
    if args.undrafted {
        println!(
            "RES drafts and the text: {parted} of {compared} replies with drafts are not the reply \
             without"
        );
    }
    runtime.shutdown();
}

/// One request of a conversation: the messages, the seconds from the submit to
/// the first piece of text, the answer and the usage.
fn converse(
    runtime: &ares_runtime::Runtime,
    id: u64,
    messages: Vec<ares_api::Message>,
    max_tokens: u32,
) -> (f64, String, ares_api::Usage) {
    use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId};
    use std::sync::{Mutex, mpsc};
    let (tx, rx) = mpsc::channel();
    let state = Arc::new(Mutex::new((String::new(), None::<Instant>)));
    let sink = {
        let state = state.clone();
        EventSink::new(move |event| {
            match event {
                Event::Logprobs(_) => {}
                Event::Text(piece) => {
                    let mut state = state.lock().unwrap();
                    state.0.push_str(&piece);
                    state.1.get_or_insert_with(Instant::now);
                }
                Event::Done { usage, .. } => {
                    let _ = tx.send(Ok(usage));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            true
        })
    };
    let begun = Instant::now();
    runtime
        .submit(
            GenerateRequest {
                model: String::new(),
                prompt: Prompt::Chat {
                    messages,
                    tools: None,
                    variables: Vec::new(),
                    continue_final: false,
                },
                sampling: ares_api::Sampling { temperature: Some(0.0), ..Default::default() },
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft: true,
                structure: None,
                id: RequestId(id),
            },
            sink,
        )
        .expect("submit");
    let usage = rx.recv().expect("the request ends").expect("the request");
    let (text, first) = std::mem::take(&mut *state.lock().unwrap());
    (first.map_or(f64::NAN, |t| (t - begun).as_secs_f64()), text, usage)
}

/// `--conversation`: the second turn of a conversation from the cache, and
/// the second turn of one the engine has never seen.
fn conversation(args: &Args, spread: &Spread, history: usize) {
    use ares_api::{Message, Role};
    use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig};
    let text = std::fs::read_to_string(args.text.as_ref().expect("--text FILE is needed"))
        .expect("the text");
    let profile = Arc::new(Profile::new());
    let options = Options {
        precision: args.precision,
        layouts: args.layouts,
        state: args.state,
        kv: args.kv,
        max_rows: args.rows(),
        logit_rows: (2 * args.window).max(16),
        window_rows: if args.window > 1 { 2 * args.window } else { 0 },
        pipeline_rows: args.pipeline_rows,
        profile: args.profile.then(|| profile.clone()),
        ..Options::default()
    };
    let devices: Vec<Device> =
        args.devices.iter().map(|d| Device::open(*d, Wait::Block).expect("open device")).collect();
    let chat = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
    let model = Arc::new(
        Qwen35::new(
            Gguf::open(&args.model).expect("the model file"),
            String::new(),
            spread,
            options,
            args.context,
        )
        .expect("model")
        .with_chat_options(chat),
    );
    let tokenizer = model.tokenizer();
    let ids = tokenizer.encode(&text, EncodeOptions { add_special: false, parse_special: false });
    assert!(
        ids.len() >= 2 * history + 64,
        "--text has {} tokens, two conversations of {history} need more",
        ids.len()
    );
    let document = |from: usize| -> String {
        let bytes = tokenizer.decode(&ids[from..from + history], false).expect("decodable tokens");
        String::from_utf8_lossy(&bytes).into_owned()
    };
    let checkpoints = args.checkpoints.unwrap_or(16);
    let per_page = 128;
    let config = RuntimeConfig {
        // The two conversations, their answers and room to run.
        pages: 2 * (history + 1024).div_ceil(per_page) as u32 + 16,
        cells_per_page: per_page as u32,
        max_slots: 2,
        window_rows: args.window.max(1),
        cache_ram: checkpoints * model.prefix_cache().map_or(0, |c| c.bytes()),
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let begun = Instant::now();
    let runtime = Runtime::start_gpu(model.clone(), devices.clone(), config, gpu).expect("start");
    println!(
        "RES conversation: loaded in {:.1} s, {checkpoints} checkpoints of {:.1} MiB",
        begun.elapsed().as_secs_f64(),
        model.prefix_cache().map_or(0, |c| c.bytes()) as f64 / MIB
    );
    let reports = model.reports();
    for r in &reports {
        println!(
            "RES conversation device {} ({}): one checkpoint is {:.1} MiB ({:.1} MiB recurrent state per sequence)",
            r.index,
            r.device,
            r.checkpoint_bytes as f64 / MIB,
            r.state_bytes_per_slot as f64 / MIB
        );
    }
    // What it takes to lock the memory of one checkpoint on each device.
    for (device, r) in devices.iter().zip(&reports) {
        let begun = Instant::now();
        let locked =
            ares_cuda::PinnedBuffer::new(device, r.checkpoint_bytes as usize).expect("lock");
        println!(
            "RES conversation device {}: locking {:.1} MiB of RAM took {:.1} ms",
            r.index,
            r.checkpoint_bytes as f64 / MIB,
            begun.elapsed().as_secs_f64() * 1e3
        );
        drop(locked);
    }

    let first_question = |doc: &str| {
        format!(
            "Below is a document.\n\n{doc}\n\nSummarize the first section of the document in two sentences."
        )
    };
    let second_question = "List three key terms of the document.";
    let _ = converse(&runtime, 1, vec![Message::new(Role::User, "Say hello.")], 8);
    let mut id = 1;
    for round in 0..2 {
        // A conversation the engine has not seen: the second turn from nothing.
        let other = first_question(&document(history + 32 * round));
        id += 1;
        let cold_prompt = vec![
            Message::new(Role::User, other),
            Message::new(Role::Assistant, "This is the answer to the first question."),
            Message::new(Role::User, second_question),
        ];
        let _ = runtime.take_snapshot();
        profile.reset();
        let (cold_ttft, _, cold) = converse(&runtime, id, cold_prompt, 24);
        let snapshot = runtime.snapshot().expect("the engine runs");
        let report = *snapshot.request(ares_api::RequestId(id)).expect("reported");
        println!(
            "RES conversation {history} tokens, second turn cold: {} prompt tokens, {} cached, {} replayed, {} computed; first token after {:.0} ms ({:.0} prompt tokens/s)",
            cold.prompt_tokens,
            cold.cached_prompt_tokens,
            report.replayed_rows,
            report.computed_rows,
            cold_ttft * 1e3,
            f64::from(cold.prompt_tokens) / cold_ttft
        );

        // A conversation of the same size: the first turn, then the second.
        let doc = document(32 * round);
        let first = vec![Message::new(Role::User, first_question(&doc))];
        id += 1;
        let (first_ttft, answer, one) = converse(&runtime, id, first.clone(), 32);
        println!(
            "RES conversation {history} tokens, first turn: {} prompt tokens, first token after {:.0} ms",
            one.prompt_tokens,
            first_ttft * 1e3
        );
        let mut next = first;
        next.push(Message::new(Role::Assistant, answer));
        next.push(Message::new(Role::User, second_question));
        id += 1;
        let _ = runtime.take_snapshot();
        profile.reset();
        let (ttft, _, two) = converse(&runtime, id, next, 24);
        let snapshot = runtime.snapshot().expect("the engine runs");
        let report = *snapshot.request(ares_api::RequestId(id)).expect("reported");
        let c = snapshot.counters;
        println!(
            "RES conversation {history} tokens, second turn from the cache: {} prompt tokens, {} cached, {} replayed, {} computed; first token after {:.0} ms (cold: {:.0} ms); checkpoints restored {}, page copies {}, RAM locked {:.0} MiB",
            two.prompt_tokens,
            two.cached_prompt_tokens,
            report.replayed_rows,
            report.computed_rows,
            ttft * 1e3,
            cold_ttft * 1e3,
            c.checkpoints_restored,
            c.page_copies,
            snapshot.checkpoint_ram as f64 / MIB
        );
        // The copies of the two turns, per device: the saves of the first
        // turn belong to the step before and are counted with the next.
        let _ = converse(&runtime, 1_000_000 + id, vec![Message::new(Role::User, "Say hello.")], 4);
        for (device, times) in profile.copies().iter().enumerate() {
            println!(
                "RES conversation device {device}: {} saves of {:.2} ms each, {} restores of {:.2} ms each, {} page copies of {:.3} ms each; {} places locked on first use, {:.1} ms of host waiting in all",
                times.saves,
                times.save_ms / times.saves.max(1) as f64,
                times.restores,
                times.restore_ms / times.restores.max(1) as f64,
                times.page_copies,
                times.page_ms / times.page_copies.max(1) as f64,
                times.locks,
                times.lock_ms,
            );
        }
    }
    runtime.shutdown();
}
