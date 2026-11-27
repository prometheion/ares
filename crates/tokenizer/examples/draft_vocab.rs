//! Makes and checks draft vocabularies: the files of token ids that
//! `--draft-vocab` gives the model drafters (MTP, DFlash), which can then
//! propose only those tokens.
//!
//! ```text
//! cargo run --release -p ares-tokenizer --example draft_vocab -- info MODEL.gguf
//! cargo run --release -p ares-tokenizer --example draft_vocab -- make MODEL.gguf \
//!     --coverage 0.995 --title TEXT --out FILE \
//!     --corpus PATH [--weight W] [--name NAME] [--licence TEXT] [--source TEXT] ...
//! cargo run --release -p ares-tokenizer --example draft_vocab -- check MODEL.gguf FILE \
//!     --corpus PATH [--weight W] [--name NAME] ...
//! ```
//!
//! The text is tokenized as the server tokenizes it: by the tokenizer of the
//! model file ([`Tokenizer::from_gguf`]), as text a user wrote (control tokens
//! stay text). A `--corpus` is one part of the corpus: a file, or a directory
//! read recursively in the order of the names, without hidden entries, files
//! with a NUL byte and files that are not UTF-8. The options after a
//! `--corpus` describe that part: its weight (1 when not given), its name (the
//! last component of the path), its licence and its source, which the header
//! names.
//!
//! **Coverage** is the share of the corpus' tokens that are in the list, with
//! each part counted by its weight: a part's tokens count as their frequency in
//! that part times the part's share of the weights, so a large part does not
//! outweigh a small one. A weight says how much of what the model writes is of
//! that kind.
//!
//! - `make` takes the tokens by that frequency, the most frequent first (on a
//!   tie the lower id), until the coverage reaches `--coverage`, or the `--ids`
//!   most frequent, and always the tokens a drafter must be able to propose
//!   (those of `info`). It writes them in ascending order under a header of
//!   `#` lines that names the tokenizer, the corpus, the coverage and the date,
//!   and prints how many ids other coverages would take.
//! - `check` reads a list as the server reads it (an id per line, blank lines
//!   and lines that begin with `#` left out), compares the tokenizer its header
//!   names with the model's, and prints what the list covers of each part and
//!   of their mix, and the most frequent tokens it lacks. That is the check of a
//!   list on one's own text.
//! - `info` prints the model's tokenizer, its fingerprint and the tokens kept
//!   always: the end-of-generation tokens, the control and user-defined tokens
//!   the chat template writes, and the tokens of the thinking and tool-call
//!   tags the template uses.
//!
//! The **fingerprint** is the SHA-256 of the values of `tokenizer.ggml.model`,
//! `tokenizer.ggml.pre`, `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`
//! and `tokenizer.ggml.merges`, in that order and in the encoding of GGUF
//! without the type tags: a string as its length (u64, little-endian) and its
//! bytes, an array as its length (u64) and its elements, a token type as an
//! i32; a key that is missing as a length of 0. Two files with the same
//! fingerprint give every text the same tokens.

use std::collections::{BTreeSet, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime};

use ares_gguf::{Gguf, Value};
use ares_tokenizer::{
    ChatOptions, EncodeOptions, Json, Message, TokenKind, Tokenizer, Tool, ToolCall,
};
use sha2::{Digest, Sha256};

const USAGE: &str = "usage:
  draft_vocab info MODEL.gguf
  draft_vocab make MODEL.gguf (--coverage C | --ids N) --out FILE [--title TEXT] [--date YYYY-MM-DD]
      [--threads N] --corpus PATH [--weight W] [--name NAME] [--licence TEXT] [--source TEXT] ...
  draft_vocab check MODEL.gguf FILE [--threads N] [--missing N] --corpus PATH [--weight W] [--name NAME] ...";

/// The coverages for which `make` prints the number of ids they take.
const CURVE: [f64; 8] = [0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9999];

/// The size of the pieces a file is tokenized in.
const CHUNK: usize = 1 << 20;

/// The first line of every file `make` writes.
const FIRST_LINE: &str = "# Ares draft vocabulary (--draft-vocab): the token ids model drafters may propose, one per line.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("info") => info(&args[1..]),
        Some("make") => make(&args[1..]),
        Some("check") => check(&args[1..]),
        _ => Err(USAGE.to_owned()),
    };
    if let Err(e) = result {
        eprintln!("draft_vocab: {e}");
        std::process::exit(2);
    }
}

// ---------------------------------------------------------------------------
// The tokenizer

/// The tokenizer of a model file, with what the files of this tool say of it.
struct Model {
    tokenizer: Tokenizer,
    /// `gpt2/qwen35, 248320 tokens, sha256:...`: the header's `tokenizer:` line.
    description: String,
    /// The file name the tokenizer was read from.
    file: String,
}

impl Model {
    fn open(path: &str) -> Result<Self, String> {
        let gguf = Gguf::open(path).map_err(|e| format!("{path}: {e}"))?;
        let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|e| format!("{path}: {e}"))?;
        let kind = format!(
            "{}/{}",
            gguf.str("tokenizer.ggml.model").unwrap_or("?"),
            gguf.str("tokenizer.ggml.pre").unwrap_or("?")
        );
        let description =
            format!("{kind}, {} tokens, sha256:{}", tokenizer.n_tokens(), fingerprint(&gguf));
        let file = Path::new(path)
            .file_name()
            .map_or_else(|| path.to_owned(), |name| name.to_string_lossy().into_owned());
        Ok(Self { tokenizer, description, file })
    }
}

/// The fingerprint of a model file's tokenizer (see the head of this file).
fn fingerprint(gguf: &Gguf) -> String {
    let mut hash = Sha256::new();
    let mut put = |bytes: &[u8]| {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    };
    for key in ["tokenizer.ggml.model", "tokenizer.ggml.pre"] {
        put(gguf.str(key).unwrap_or("").as_bytes());
    }
    let array = |key: &str| gguf.get(key).and_then(|value| value.as_array());
    for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"] {
        let Some(values) = array(key) else {
            hash.update(0u64.to_le_bytes());
            continue;
        };
        hash.update(values.len().to_le_bytes());
        for value in values.iter() {
            match value {
                Value::String(bytes) => {
                    hash.update((bytes.len() as u64).to_le_bytes());
                    hash.update(bytes);
                }
                other => {
                    let number = other.as_i64().and_then(|n| i32::try_from(n).ok());
                    hash.update(number.unwrap_or(i32::MIN).to_le_bytes());
                }
            }
        }
    }
    hash.finalize().iter().fold(String::new(), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    })
}

/// A token a drafter must always be able to propose, and why.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Kept {
    id: u32,
    why: &'static str,
}

/// The tokens a drafter must always be able to propose: the end-of-generation
/// tokens, the control and user-defined tokens of a conversation the chat
/// template renders (with thinking, a call of a tool and its result), and the
/// tokens of the thinking and tool-call tags the template writes. Sorted by
/// id, each once, with the first reason found.
fn kept_tokens(tokenizer: &Tokenizer) -> Vec<Kept> {
    let mut kept: Vec<Kept> = Vec::new();
    let mut add = |id: u32, why: &'static str| {
        if !kept.iter().any(|k| k.id == id) {
            kept.push(Kept { id, why });
        }
    };
    for id in 0..tokenizer.n_tokens() as u32 {
        if tokenizer.is_end_of_generation(id) {
            add(id, "end of generation");
        }
    }
    let special = EncodeOptions { add_special: false, parse_special: true };
    if let Some(prompt) = probe_conversation(tokenizer) {
        for id in tokenizer.encode(&prompt, special) {
            if matches!(tokenizer.kind(id), Some(TokenKind::Control | TokenKind::UserDefined)) {
                add(id, "chat template");
            }
        }
    }
    if let Some(thinking) = tokenizer.thinking_format() {
        for text in [&thinking.start, &thinking.end, &thinking.prompt_on, &thinking.prompt_off] {
            tokenizer.encode(text, special).into_iter().for_each(|id| add(id, "thinking tags"));
        }
    }
    if let Some(tools) = tokenizer.tool_format() {
        for text in [
            &tools.call_open,
            &tools.name_close,
            &tools.param_open,
            &tools.key_close,
            &tools.param_close,
            &tools.call_close,
            &tools.before_calls,
            &tools.between_calls,
        ] {
            tokenizer.encode(text, special).into_iter().for_each(|id| add(id, "tool-call tags"));
        }
    }
    kept.sort();
    kept
}

/// A conversation through the model's chat template: a system prompt, tools, a
/// user message, an assistant that thinks and calls a tool, the tool's result,
/// and the start of the next answer. `None` without a template that renders.
fn probe_conversation(tokenizer: &Tokenizer) -> Option<String> {
    let text = |value: &str| Json::String(value.to_owned());
    let tool = Tool {
        name: "run".to_owned(),
        description: "Runs a command.".to_owned(),
        parameters: Json::Object(vec![
            ("type".to_owned(), text("object")),
            (
                "properties".to_owned(),
                Json::Object(vec![(
                    "command".to_owned(),
                    Json::Object(vec![("type".to_owned(), text("string"))]),
                )]),
            ),
        ]),
    };
    let mut assistant = Message::new("assistant", "Here.");
    assistant.reasoning_content = "Run it.".to_owned();
    assistant.tool_calls = vec![ToolCall {
        id: Some("call_0".to_owned()),
        name: "run".to_owned(),
        arguments: r#"{"command": "ls"}"#.to_owned(),
    }];
    let mut result = Message::new("tool", "a b");
    result.name = "run".to_owned();
    result.tool_call_id = "call_0".to_owned();
    let messages = [
        Message::new("system", "Be brief."),
        Message::new("user", "List the files."),
        assistant,
        result,
    ];
    let options = ChatOptions { add_generation_prompt: true, ..ChatOptions::default() };
    tokenizer.render_chat(&messages, &[tool], &options).ok()
}

/// A token's text for a person: its bytes, with escapes.
fn shown(tokenizer: &Tokenizer, id: u32) -> String {
    let piece = tokenizer.piece(id, true).unwrap_or(b"");
    format!("{:?}", String::from_utf8_lossy(piece))
}

// ---------------------------------------------------------------------------
// The corpus

/// One part of the corpus, as the command line gives it.
#[derive(Clone, Debug, Default)]
struct PartSpec {
    path: PathBuf,
    weight: f64,
    name: String,
    licence: Option<String>,
    source: Option<String>,
}

/// What the tokens of one part came to.
#[derive(Clone, Debug)]
struct Counted {
    spec: PartSpec,
    /// The tokens of each id.
    counts: Vec<u64>,
    tokens: u64,
    bytes: u64,
    files: usize,
    /// Files left out: with a NUL byte, or not UTF-8.
    skipped: usize,
}

/// The options common to `make` and `check`, and the parts.
#[derive(Debug, Default)]
struct Options {
    positional: Vec<String>,
    parts: Vec<PartSpec>,
    coverage: Option<f64>,
    ids: Option<usize>,
    out: Option<PathBuf>,
    title: Option<String>,
    date: Option<String>,
    threads: usize,
    missing: usize,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options { threads: 4, missing: 30, ..Options::default() };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if !arg.starts_with("--") {
            options.positional.push(arg.to_owned());
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))?.clone();
        i += 2;
        let number = |what: &str| -> Result<f64, String> {
            value.parse::<f64>().map_err(|_| format!("{arg} {value}: {what}"))
        };
        let part = options.parts.last_mut();
        match (arg, part) {
            ("--corpus", _) => {
                let path = PathBuf::from(&value);
                let name = path
                    .file_name()
                    .map_or_else(|| value.clone(), |name| name.to_string_lossy().into_owned());
                options.parts.push(PartSpec { path, weight: 1.0, name, ..PartSpec::default() });
            }
            ("--weight", Some(part)) => {
                part.weight = number("a weight is a number above 0")?;
                if !(part.weight > 0.0 && part.weight.is_finite()) {
                    return Err(format!("--weight {value}: a weight is a number above 0"));
                }
            }
            ("--name", Some(part)) => part.name = value,
            ("--licence", Some(part)) => part.licence = Some(value),
            ("--source", Some(part)) => part.source = Some(value),
            ("--weight" | "--name" | "--licence" | "--source", None) => {
                return Err(format!("{arg} describes the --corpus before it, and there is none"));
            }
            ("--coverage", _) => {
                let coverage = number("a coverage is above 0 and at most 1")?;
                if !(coverage > 0.0 && coverage <= 1.0) {
                    return Err(format!("--coverage {value}: a coverage is above 0 and at most 1"));
                }
                options.coverage = Some(coverage);
            }
            ("--ids", _) => {
                let ids = value.parse().map_err(|_| format!("--ids {value}: a number of ids"))?;
                options.ids = Some(ids);
            }
            ("--out", _) => options.out = Some(PathBuf::from(value)),
            ("--title", _) => options.title = Some(value),
            ("--date", _) => options.date = Some(value),
            ("--threads", _) => {
                options.threads = value
                    .parse()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| format!("--threads {value}: a number above 0"))?;
            }
            ("--missing", _) => {
                options.missing =
                    value.parse().map_err(|_| format!("--missing {value}: a number"))?;
            }
            _ => return Err(format!("unknown option {arg}\n{USAGE}")),
        }
    }
    let mut names = HashSet::new();
    for part in &options.parts {
        if !names.insert(part.name.as_str()) {
            return Err(format!("two parts are named '{}': give one --name", part.name));
        }
        for text in [&part.name].into_iter().chain(&part.licence).chain(&part.source) {
            if text.contains('\n') {
                return Err(format!("'{text}': the header takes no line breaks"));
            }
        }
    }
    Ok(options)
}

/// The files of a part: the path itself, or every file under it in the order
/// of the names, without hidden entries.
fn files_of(path: &Path) -> Result<Vec<PathBuf>, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_file() {
        return Ok(vec![path.to_owned()]);
    }
    let mut files = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')))
        .collect();
    entries.sort();
    for entry in entries {
        let meta = std::fs::metadata(&entry).map_err(|e| format!("{}: {e}", entry.display()))?;
        if meta.is_dir() {
            files.extend(files_of(&entry)?);
        } else if meta.is_file() {
            files.push(entry);
        }
    }
    Ok(files)
}

/// Where `text` may be cut so that the tokens of the pieces are the tokens of
/// the whole: after a line break that a printable ASCII character follows. No
/// word of the pre-tokenizer reaches across a line break into such a character
/// (a run of white space that ends in a line break is a word of its own, and
/// nothing else holds a line break and a character after it), and the words
/// after the cut do not depend on what came before it. Pieces of about
/// `size` bytes, longer where no such place comes sooner.
fn pieces(text: &[u8], size: usize) -> Vec<&[u8]> {
    let mut pieces = Vec::new();
    let mut start = 0;
    while text.len() - start > size {
        let from = start + size;
        let cut = text[from..]
            .windows(2)
            .position(|pair| pair[0] == b'\n' && (0x21..=0x7e).contains(&pair[1]))
            .map(|at| from + at + 1);
        let Some(cut) = cut else { break };
        pieces.push(&text[start..cut]);
        start = cut;
    }
    pieces.push(&text[start..]);
    pieces
}

/// Counts the tokens of every part, with `threads` threads over the files.
fn count(
    tokenizer: &Tokenizer,
    parts: &[PartSpec],
    threads: usize,
) -> Result<Vec<Counted>, String> {
    let started = Instant::now();
    let mut jobs: Vec<(usize, PathBuf)> = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let files = files_of(&part.path)?;
        if files.is_empty() {
            return Err(format!("{}: no files", part.path.display()));
        }
        jobs.extend(files.into_iter().map(|file| (index, file)));
    }
    let n_vocab = tokenizer.n_tokens();
    let empty = || -> Vec<Counted> {
        parts
            .iter()
            .map(|spec| Counted {
                spec: spec.clone(),
                counts: vec![0; n_vocab],
                tokens: 0,
                bytes: 0,
                files: 0,
                skipped: 0,
            })
            .collect()
    };
    let counted = Mutex::new(empty());
    let next = AtomicUsize::new(0);
    let failure = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(jobs.len()) {
            scope.spawn(|| {
                let mut mine = empty();
                let mut ids = Vec::new();
                loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    let Some((index, file)) = jobs.get(job) else { break };
                    let text = match std::fs::read(file) {
                        Ok(text) => text,
                        Err(e) => {
                            *failure.lock().unwrap() = Some(format!("{}: {e}", file.display()));
                            break;
                        }
                    };
                    let part = &mut mine[*index];
                    if text.contains(&0) || std::str::from_utf8(&text).is_err() {
                        part.skipped += 1;
                        continue;
                    }
                    part.files += 1;
                    part.bytes += text.len() as u64;
                    for piece in pieces(&text, CHUNK) {
                        ids.clear();
                        tokenizer.encode_into(piece, EncodeOptions::default(), &mut ids);
                        part.tokens += ids.len() as u64;
                        for &id in &ids {
                            part.counts[id as usize] += 1;
                        }
                    }
                }
                let mut all = counted.lock().unwrap();
                for (sum, part) in all.iter_mut().zip(mine) {
                    sum.counts.iter_mut().zip(&part.counts).for_each(|(s, c)| *s += c);
                    sum.tokens += part.tokens;
                    sum.bytes += part.bytes;
                    sum.files += part.files;
                    sum.skipped += part.skipped;
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    let counted = counted.into_inner().unwrap();
    for part in &counted {
        if part.tokens == 0 {
            return Err(format!("{}: no text", part.spec.path.display()));
        }
        eprintln!(
            "{}: {} in {} files ({} left out), {} tokens",
            part.spec.name,
            megabytes(part.bytes),
            part.files,
            part.skipped,
            thousands(part.tokens)
        );
    }
    eprintln!("counted in {:.1} s", started.elapsed().as_secs_f64());
    Ok(counted)
}

/// The frequency of every token in the mix of the parts: its frequency in each
/// part times the part's share of the weights. The frequencies add up to 1.
fn mixed(parts: &[Counted]) -> Vec<f64> {
    let total: f64 = parts.iter().map(|p| p.spec.weight).sum();
    let mut frequency = vec![0.0; parts.first().map_or(0, |p| p.counts.len())];
    for part in parts {
        let scale = part.spec.weight / total / part.tokens as f64;
        frequency.iter_mut().zip(&part.counts).for_each(|(f, &c)| *f += c as f64 * scale);
    }
    frequency
}

/// What a set of tokens covers of a frequency table.
fn covered(frequency: &[f64], set: &BTreeSet<u32>) -> f64 {
    set.iter().map(|&id| frequency.get(id as usize).copied().unwrap_or(0.0)).sum()
}

/// What `make` takes: the kept tokens, then the most frequent others (lower id
/// first on a tie) until `coverage` is reached or the list holds `ids` tokens.
fn select(
    frequency: &[f64],
    kept: &[u32],
    coverage: Option<f64>,
    ids: Option<usize>,
) -> BTreeSet<u32> {
    let mut set: BTreeSet<u32> = kept.iter().copied().collect();
    let mut sum = covered(frequency, &set);
    let mut order: Vec<u32> = (0..frequency.len() as u32)
        .filter(|id| frequency[*id as usize] > 0.0 && !set.contains(id))
        .collect();
    order.sort_by(|a, b| frequency[*b as usize].total_cmp(&frequency[*a as usize]).then(a.cmp(b)));
    for id in order {
        let full = match (coverage, ids) {
            (_, Some(n)) => set.len() >= n,
            // A margin for the rounding of the sum.
            (Some(target), None) => sum >= target - 1e-12,
            (None, None) => false,
        };
        if full {
            break;
        }
        set.insert(id);
        sum += frequency[id as usize];
    }
    set
}

/// How many ids each coverage of `CURVE` takes, the kept tokens included.
fn curve(frequency: &[f64], kept: &[u32]) -> Vec<(f64, usize)> {
    CURVE.iter().map(|&c| (c, select(frequency, kept, Some(c), None).len())).collect()
}

// ---------------------------------------------------------------------------
// The commands

fn info(args: &[String]) -> Result<(), String> {
    let [path] = args else { return Err(USAGE.to_owned()) };
    let model = Model::open(path)?;
    println!("{}: {}", model.file, model.description);
    println!("kept always:");
    for kept in kept_tokens(&model.tokenizer) {
        println!("  {:>7} {:<28} {}", kept.id, shown(&model.tokenizer, kept.id), kept.why);
    }
    Ok(())
}

fn make(args: &[String]) -> Result<(), String> {
    let options = parse(args)?;
    let [path] = options.positional.as_slice() else { return Err(USAGE.to_owned()) };
    let out = options.out.as_ref().ok_or("make needs --out FILE")?;
    if options.coverage.is_some() == options.ids.is_some() {
        return Err("make takes one of --coverage and --ids".to_owned());
    }
    if options.parts.is_empty() {
        return Err("make needs a --corpus".to_owned());
    }
    let model = Model::open(path)?;
    let kept = kept_tokens(&model.tokenizer);
    let kept_ids: Vec<u32> = kept.iter().map(|k| k.id).collect();
    let parts = count(&model.tokenizer, &options.parts, options.threads)?;
    let frequency = mixed(&parts);
    let set = select(&frequency, &kept_ids, options.coverage, options.ids);
    for (coverage, ids) in curve(&frequency, &kept_ids) {
        println!("coverage {coverage}: {ids} ids");
    }
    let date = options.date.clone().unwrap_or_else(today);
    let text = file_text(&model, &options, &parts, &kept, &set, &date);
    std::fs::write(out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("{}: {} ids, coverage {:.5}", out.display(), set.len(), covered(&frequency, &set));
    Ok(())
}

/// The file `make` writes: the header, then the ids in ascending order.
fn file_text(
    model: &Model,
    options: &Options,
    parts: &[Counted],
    kept: &[Kept],
    set: &BTreeSet<u32>,
    date: &str,
) -> String {
    let frequency = mixed(parts);
    let total: f64 = parts.iter().map(|p| p.spec.weight).sum();
    let mut text = String::new();
    let mut line = |line: String| {
        text.push_str(line.trim_end());
        text.push('\n');
    };
    line(FIRST_LINE.to_owned());
    if let Some(title) = &options.title {
        line(format!("# title: {title}"));
    }
    line(format!("# tokenizer: {}", model.description));
    line(format!("# read from: {}", model.file));
    line(format!("# made: {date}, by crates/tokenizer/examples/draft_vocab.rs"));
    let rule = match (options.coverage, options.ids) {
        (Some(coverage), _) => format!("coverage {coverage}"),
        (None, Some(ids)) => format!("{ids} ids"),
        (None, None) => unreachable!("make takes one of them"),
    };
    line(format!(
        "# rule: the most frequent tokens of the weighted corpus up to {rule}, and the tokens kept \
         always"
    ));
    for part in parts {
        let share = part.spec.weight / total;
        line(format!(
            "# part: {}, weight {} (share {share:.3}), {} in {} files, {} tokens, licence {}, {}",
            part.spec.name,
            part.spec.weight,
            megabytes(part.bytes),
            thousands(part.files as u64),
            thousands(part.tokens),
            part.spec.licence.as_deref().unwrap_or("not given"),
            part.spec.source.as_deref().unwrap_or("source not given")
        ));
    }
    let each: Vec<String> = parts
        .iter()
        .map(|part| format!("{} {:.5}", part.spec.name, part_coverage(part, set)))
        .collect();
    line(format!("# coverage: {:.5} ({})", covered(&frequency, set), each.join(", ")));
    let n_kept = kept.len();
    line(format!(
        "# ids: {}, of which {} kept always ({} by frequency alone)",
        set.len(),
        n_kept,
        set.len() - n_kept
    ));
    let shown_kept: Vec<String> =
        kept.iter().map(|k| format!("{} {}", k.id, shown(&model.tokenizer, k.id))).collect();
    line(format!("# kept always: {}", shown_kept.join(", ")));
    for id in set {
        line(id.to_string());
    }
    text
}

/// What a set of tokens covers of one part.
fn part_coverage(part: &Counted, set: &BTreeSet<u32>) -> f64 {
    let hits: u64 = set.iter().map(|&id| part.counts.get(id as usize).copied().unwrap_or(0)).sum();
    hits as f64 / part.tokens as f64
}

/// A draft vocabulary file as the server reads it, and the `tokenizer:` line
/// of its header if it has one.
struct ListFile {
    ids: Vec<u32>,
    tokenizer: Option<String>,
}

fn read_list(path: &str) -> Result<ListFile, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut ids = Vec::new();
    let mut tokenizer = None;
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("# tokenizer:") {
            tokenizer = Some(value.trim().to_owned());
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let id = line.parse::<u32>();
        ids.push(id.map_err(|_| format!("{path}: line {}: '{line}' is no token id", number + 1))?);
    }
    Ok(ListFile { ids, tokenizer })
}

fn check(args: &[String]) -> Result<(), String> {
    let options = parse(args)?;
    let [path, list_path] = options.positional.as_slice() else { return Err(USAGE.to_owned()) };
    if options.parts.is_empty() {
        return Err("check needs a --corpus".to_owned());
    }
    let model = Model::open(path)?;
    let list = read_list(list_path)?;
    let n_vocab = model.tokenizer.n_tokens() as u32;
    let mut set = BTreeSet::new();
    for &id in &list.ids {
        if id >= n_vocab || !set.insert(id) {
            return Err(format!(
                "{list_path}: token {id} is listed twice or is not one of the {n_vocab} tokens; \
                 the server refuses the file"
            ));
        }
    }
    if set.is_empty() {
        return Err(format!("{list_path}: the list is empty; the server refuses it"));
    }
    match &list.tokenizer {
        Some(named) if *named == model.description => {
            println!("tokenizer: the model's ({})", model.description);
        }
        Some(named) => println!(
            "tokenizer: NOT the model's: the file names {named}, the model has {}",
            model.description
        ),
        None => println!("tokenizer: the file names none"),
    }
    let missing_kept: Vec<String> = kept_tokens(&model.tokenizer)
        .into_iter()
        .filter(|k| !set.contains(&k.id))
        .map(|k| format!("{} {} ({})", k.id, shown(&model.tokenizer, k.id), k.why))
        .collect();
    if missing_kept.is_empty() {
        println!("kept always: all in the list");
    } else {
        println!("kept always: missing {}", missing_kept.join(", "));
    }
    let parts = count(&model.tokenizer, &options.parts, options.threads)?;
    let frequency = mixed(&parts);
    println!("{} ids", set.len());
    for part in &parts {
        println!(
            "  {:<16} coverage {:.5} (weight {}, {} tokens)",
            part.spec.name,
            part_coverage(part, &set),
            part.spec.weight,
            thousands(part.tokens)
        );
    }
    if parts.len() > 1 {
        println!("  {:<16} coverage {:.5}", "the mix", covered(&frequency, &set));
    }
    let mut missing: Vec<u32> =
        (0..n_vocab).filter(|id| frequency[*id as usize] > 0.0 && !set.contains(id)).collect();
    missing
        .sort_by(|a, b| frequency[*b as usize].total_cmp(&frequency[*a as usize]).then(a.cmp(b)));
    if !missing.is_empty() && options.missing > 0 {
        println!("the most frequent tokens the list lacks (share of the mix):");
        for &id in missing.iter().take(options.missing) {
            println!(
                "  {:>7} {:<28} {:.6}",
                id,
                shown(&model.tokenizer, id),
                frequency[id as usize]
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Numbers and dates

fn megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

/// `1234567` as `1,234,567`.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Today's date in UTC, `YYYY-MM-DD`.
fn today() -> String {
    let seconds =
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    civil_date((seconds / 86_400) as i64)
}

/// The date of a day counted from 1970-01-01 (Howard Hinnant's
/// `civil_from_days`).
fn civil_date(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{self, IM_END, TempDir};

    /// The ready files of the repository.
    fn ready_files() -> Vec<PathBuf> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/user/draft-vocab");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map(|entries| entries.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        files.retain(|p| p.extension().is_some_and(|e| e == "txt"));
        files.sort();
        files
    }

    /// The ready files: what the server needs of them (ids of the vocabulary,
    /// each once), ascending, with a header; with `ARES_TEST_QWEN35_0_8B`, the
    /// tokenizer they name is that model's and they hold the kept tokens.
    #[test]
    fn the_pieces_of_a_text_have_the_tokens_of_the_whole() {
        let tokenizer = common::tiny_tokenizer("draft-vocab-pieces");
        let text = "hello world\n\nhello\n world\nhello  \n\n  world\nü\nhello\n".repeat(40);
        let whole = tokenizer.encode(&text, EncodeOptions::default());
        for size in [1, 7, 64, 1000] {
            let cut = pieces(text.as_bytes(), size);
            assert!(size > 100 || cut.len() > 5, "{size}: few pieces");
            assert_eq!(cut.concat(), text.as_bytes());
            let ids: Vec<u32> = cut
                .iter()
                .flat_map(|piece| tokenizer.encode(piece, EncodeOptions::default()))
                .collect();
            assert_eq!(ids, whole, "pieces of about {size} bytes");
        }
    }

    /// The same with the tokenizer of a real model on the texts of the
    /// crate's corpus (`ARES_TEST_QWEN35_0_8B`).
    #[test]
    fn the_pieces_of_real_text_have_the_tokens_of_the_whole() {
        let Some(path) = std::env::var_os("ARES_TEST_QWEN35_0_8B") else {
            eprintln!("skipped: set ARES_TEST_QWEN35_0_8B=<model.gguf>");
            return;
        };
        let model = Model::open(&path.to_string_lossy()).unwrap();
        let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
        let mut text = Vec::new();
        for file in files_of(&corpus).unwrap() {
            let bytes = std::fs::read(&file).unwrap();
            if std::str::from_utf8(&bytes).is_ok() {
                text.extend_from_slice(&bytes);
            }
        }
        let whole = model.tokenizer.encode(&text, EncodeOptions::default());
        for size in [16, 300, 5000] {
            let ids: Vec<u32> = pieces(&text, size)
                .iter()
                .flat_map(|piece| model.tokenizer.encode(piece, EncodeOptions::default()))
                .collect();
            assert_eq!(ids, whole, "pieces of about {size} bytes");
        }
    }

    #[test]
    fn the_fingerprint_is_the_sha256_of_the_tokenizer_values() {
        let tiny = common::tiny();
        let (_dir, gguf) = tiny.open("draft-vocab-fingerprint");
        // The bytes of the definition, written out by hand.
        let mut bytes = Vec::new();
        let string = |bytes: &mut Vec<u8>, s: &[u8]| {
            bytes.extend((s.len() as u64).to_le_bytes());
            bytes.extend(s);
        };
        string(&mut bytes, b"gpt2");
        string(&mut bytes, b"qwen35");
        let tokens = common::token_texts();
        bytes.extend((tokens.len() as u64).to_le_bytes());
        tokens.iter().for_each(|t| string(&mut bytes, t.as_bytes()));
        let types = common::token_types();
        bytes.extend((types.len() as u64).to_le_bytes());
        types.iter().for_each(|t| bytes.extend(t.to_le_bytes()));
        let merges = common::merge_texts();
        bytes.extend((merges.len() as u64).to_le_bytes());
        merges.iter().for_each(|m| string(&mut bytes, m.as_bytes()));
        let expected: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(fingerprint(&gguf), expected);

        // Another merge order is another tokenizer.
        let mut merges = common::merge_texts();
        merges.swap(0, 1);
        let (_other_dir, other) = common::tiny().merges(merges).open("draft-vocab-fingerprint-2");
        assert_ne!(fingerprint(&other), expected);
    }

    #[test]
    fn a_part_counts_by_its_weight_not_its_size() {
        let part = |name: &str, weight: f64, counts: Vec<u64>| Counted {
            spec: PartSpec { name: name.to_owned(), weight, ..PartSpec::default() },
            tokens: counts.iter().sum(),
            counts,
            bytes: 0,
            files: 1,
            skipped: 0,
        };
        // Part a has ten times the tokens of part b and a quarter of the weight.
        let parts = [part("a", 1.0, vec![900, 100, 0, 0]), part("b", 3.0, vec![0, 0, 60, 40])];
        let frequency = mixed(&parts);
        let expected = [0.225, 0.025, 0.45, 0.3];
        for (f, e) in frequency.iter().zip(expected) {
            assert!((f - e).abs() < 1e-12, "{frequency:?}");
        }
        let set: BTreeSet<u32> = [0, 2].into();
        assert!((covered(&frequency, &set) - 0.675).abs() < 1e-12);
        assert!((part_coverage(&parts[0], &set) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn the_most_frequent_tokens_up_to_the_coverage_and_the_kept_ones_always() {
        // Token 4 is the kept one; 1 and 3 tie.
        let frequency = [0.5, 0.2, 0.0, 0.2, 0.0, 0.1];
        let pick = |c: Option<f64>, n: Option<usize>| -> Vec<u32> {
            select(&frequency, &[4], c, n).into_iter().collect()
        };
        assert_eq!(pick(Some(0.5), None), [0, 4]);
        assert_eq!(pick(Some(0.6), None), [0, 1, 4], "the lower id of a tie first");
        assert_eq!(pick(Some(0.9), None), [0, 1, 3, 4]);
        assert_eq!(pick(Some(1.0), None), [0, 1, 3, 4, 5], "never a token that does not occur");
        assert_eq!(pick(None, Some(3)), [0, 1, 4]);
        assert_eq!(pick(None, Some(1)), [4], "the kept tokens even past --ids");
        let curve = curve(&frequency, &[4]);
        assert_eq!(curve[0], (0.9, 4));
    }

    #[test]
    fn a_made_file_is_read_back_as_the_server_reads_it() {
        let dir = TempDir::new("draft-vocab-file");
        let model_path = dir.path().join("model.gguf");
        common::tiny().builder().write(&model_path).unwrap();
        let corpus = dir.path().join("corpus");
        std::fs::create_dir_all(corpus.join("sub")).unwrap();
        std::fs::write(corpus.join("a.txt"), "hello world\nhello\n").unwrap();
        std::fs::write(corpus.join("sub/b.txt"), "hello hello ü\n").unwrap();
        std::fs::write(corpus.join(".hidden"), "world world world\n").unwrap();
        std::fs::write(corpus.join("binary"), b"he\0llo").unwrap();
        let args: Vec<String> = [
            &model_path.to_string_lossy(),
            "--coverage",
            "0.6",
            "--title",
            "test",
            "--corpus",
            &corpus.to_string_lossy(),
            "--licence",
            "CC0-1.0",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let options = parse(&args).unwrap();
        let model = Model::open(&model_path.to_string_lossy()).unwrap();
        let parts = count(&model.tokenizer, &options.parts, 2).unwrap();
        assert_eq!((parts[0].files, parts[0].skipped), (2, 1), "the hidden file is not read");
        let kept = kept_tokens(&model.tokenizer);
        assert_eq!(kept, [Kept { id: IM_END, why: "end of generation" }]);
        let frequency = mixed(&parts);
        let kept_ids: Vec<u32> = kept.iter().map(|k| k.id).collect();
        let set = select(&frequency, &kept_ids, options.coverage, None);
        let text = file_text(&model, &options, &parts, &kept, &set, "2026-11-27");
        assert!(text.starts_with(FIRST_LINE), "{text}");
        assert!(text.contains("# part: corpus, weight 1 (share 1.000), "), "{text}");
        assert!(text.contains(", licence CC0-1.0, source not given\n"), "{text}");
        let path = dir.path().join("list.txt");
        std::fs::write(&path, &text).unwrap();
        let list = read_list(&path.to_string_lossy()).unwrap();
        assert_eq!(list.ids, set.iter().copied().collect::<Vec<_>>());
        assert_eq!(list.tokenizer.as_deref(), Some(model.description.as_str()));
        assert!(list.ids.contains(&IM_END));
    }

    #[test]
    fn options_describe_the_part_before_them() {
        let args: Vec<String> =
            ["m", "--corpus", "x/de", "--weight", "2", "--corpus", "y", "--name", "code"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let options = parse(&args).unwrap();
        let parts: Vec<(&str, f64)> =
            options.parts.iter().map(|p| (p.name.as_str(), p.weight)).collect();
        assert_eq!(parts, [("de", 2.0), ("code", 1.0)]);
        let refused =
            |args: &[&str]| parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).is_err();
        assert!(refused(&["--weight", "2"]), "a weight without a part");
        assert!(refused(&["--corpus", "a", "--weight", "0"]));
        assert!(refused(&["--corpus", "a", "--corpus", "b/a"]), "two parts named a");
        assert!(refused(&["--coverage", "1.5"]));
        assert!(refused(&["--colour", "red"]));
    }

    #[test]
    fn dates() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(11_016), "2000-02-29");
        assert_eq!(civil_date(20_784), "2026-11-27");
    }

    #[test]
    fn the_ready_files_fit_their_tokenizer() {
        let files = ready_files();
        assert!(!files.is_empty(), "no ready files");
        let model = std::env::var_os("ARES_TEST_QWEN35_0_8B")
            .map(|path| Model::open(&path.to_string_lossy()).unwrap());
        for file in files {
            let name = file.display().to_string();
            let list = read_list(&name).unwrap();
            assert!(list.ids.windows(2).all(|w| w[0] < w[1]), "{name}: not ascending");
            assert!(list.ids.last().is_some_and(|&id| id < 248_320), "{name}");
            let text = std::fs::read_to_string(&file).unwrap();
            assert!(text.starts_with(FIRST_LINE), "{name}");
            for key in ["# tokenizer:", "# made:", "# part:", "# coverage:", "# ids:"] {
                assert!(text.contains(key), "{name}: no '{key}' line");
            }
            let Some(model) = &model else { continue };
            assert_eq!(list.tokenizer.as_ref(), Some(&model.description), "{name}");
            for kept in kept_tokens(&model.tokenizer) {
                assert!(list.ids.binary_search(&kept.id).is_ok(), "{name}: lacks {kept:?}");
            }
        }
        if model.is_none() {
            eprintln!("the tokenizer not compared: set ARES_TEST_QWEN35_0_8B=<model.gguf>");
        }
    }
}
