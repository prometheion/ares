//! Measures the lookup drafter against the n-gram drafter on real text.
//!
//! ```sh
//! cargo run --release -p ares-sched --example lookup_bench -- <model.gguf> [repository]
//! ```
//!
//! The text is this repository's own: its documentation (English), its Rust
//! sources (code) and the German files of the tokenizer's corpus and the
//! capture prompts, tokenized with the vocabulary of the model file. Nothing
//! runs a model. A text is replayed as if a model generated it: at every
//! step the drafter proposes, the proposal is compared with what the text
//! really continues with, and the step advances by the accepted tokens and
//! one more, as verification would.
//!
//! Three ways to replay, per kind of text:
//!
//! - **plain**: every file is a request that generates the file from an
//!   empty prompt. A lookup drafter can only help where the text repeats
//!   itself, so this is its floor.
//! - **edit**: the file is the prompt, and the request writes it again with a
//!   few tokens changed every 150. This is what the drafter is for.
//! - **again**: the file was written by an earlier request; a new request
//!   with an empty prompt writes it again with the same edits. Only a corpus
//!   that is shared between requests can help here.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ares_gguf::Gguf;
use ares_sched::{
    DraftCost, LookupConfig, LookupCorpus, LookupDrafter, NgramDrafter, best_draft_len,
};
use ares_tokenizer::{EncodeOptions, Tokenizer};

const MAX_DRAFT: usize = 16;
/// The step cost the draft length is chosen with: Qwen3.8-27B on the
/// development machine, 33 ms and 1.1 ms per row.
const BASE_MS: f64 = 33.0;
const PER_ROW_MS: f64 = 1.1;

#[derive(Default, Clone, Copy)]
struct Tally {
    tokens: u64,
    steps: u64,
    proposals: u64,
    proposed: u64,
    accepted: u64,
    ms: f64,
}

impl Tally {
    fn record(&mut self, proposed: usize, accepted: usize) {
        self.steps += 1;
        self.tokens += accepted as u64 + 1;
        self.proposals += u64::from(proposed > 0);
        self.proposed += proposed as u64;
        self.accepted += accepted as u64;
        self.ms += BASE_MS + PER_ROW_MS * (1 + proposed) as f64;
    }

    fn row(&self, name: &str) -> String {
        let per = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        format!(
            "| {name} | {:.3} | {:.2} | {:.2} | {:.1} % | {:.3} | {:.2} | {:.1} |",
            per(self.proposals, self.tokens),
            per(self.proposed, self.proposals),
            per(self.accepted, self.proposals),
            100.0 * per(self.accepted, self.tokens),
            per(self.proposed - self.accepted, self.tokens),
            per(self.tokens, self.steps),
            1000.0 * self.tokens as f64 / self.ms,
        )
    }
}

const HEADER: &str = "| drafter | proposals per token | proposed length | accepted length | \
tokens from drafts | wasted rows per token | tokens per step | tokens/s at 33 + 1.1 ms |\n\
| --- | --- | --- | --- | --- | --- | --- | --- |";

fn accepted(draft: &[u32], rest: &[u32]) -> usize {
    draft.iter().zip(rest).take_while(|(a, b)| a == b).count()
}

/// The request writes `output` after `prompt`, drafted by the n-gram lookup.
fn replay_ngram(prompt: &[u32], output: &[u32], tally: &mut Tally) {
    let mut drafter = NgramDrafter::new(3);
    drafter.extend(prompt);
    let mut at = 0;
    while at < output.len() {
        // The newest token is known to the drafter only after the step, as
        // in the engine; the drafts are for the tokens after it.
        drafter.push(output[at]);
        let draft = drafter.draft(MAX_DRAFT.min(output.len() - at - 1));
        let (proposed, ok) = (draft.len(), accepted(draft, &output[at + 1..]));
        tally.record(proposed, ok);
        drafter.extend(&output[at + 1..at + 1 + ok]);
        at += ok + 1;
    }
}

/// Acceptance by table cell: tokens that were reached, and of them accepted.
type Cells = [[u64; 2]; 40];

/// The same with the lookup drafter. `by_confidence` cuts every draft to the
/// length that [`best_draft_len`] chooses from the drafter's confidences.
fn replay_lookup(
    corpus: &mut LookupCorpus,
    prompt: &[u32],
    output: &[u32],
    by_confidence: bool,
    tally: &mut Tally,
    cells: &mut Cells,
) {
    let mut drafter = LookupDrafter::new();
    drafter.extend(corpus, prompt);
    let (mut draft, mut confidence) = (Vec::new(), Vec::new());
    let mut at = 0;
    while at < output.len() {
        drafter.push(corpus, output[at]);
        drafter.draft(corpus, MAX_DRAFT.min(output.len() - at - 1), &mut draft, &mut confidence);
        if by_confidence {
            let free = DraftCost { fixed_ms: 0.0, per_token_ms: 0.0 };
            let len = best_draft_len(&confidence, BASE_MS, PER_ROW_MS, 1, 1.0, free).len;
            draft.truncate(len);
        }
        let ok = accepted(&draft, &output[at + 1..]);
        for (i, cell) in drafter.cells().iter().take(draft.len()).enumerate() {
            if i <= ok {
                cells[*cell as usize][0] += 1;
                cells[*cell as usize][1] += u64::from(i < ok);
            }
        }
        tally.record(draft.len(), ok);
        drafter.observe(corpus, draft.len(), ok);
        drafter.extend(corpus, &output[at + 1..at + 1 + ok]);
        at += ok + 1;
    }
}

/// The file with a few tokens changed every 150: what a model writes when it
/// is asked for a small change.
fn edited(tokens: &[u32]) -> Vec<u32> {
    let mut out = tokens.to_vec();
    let n = out.len();
    for at in (100..n.saturating_sub(4)).step_by(150) {
        for k in 0..3 {
            out[at + k] = tokens[(at * 7 + k * 13) % n];
        }
    }
    out
}

fn files(root: &Path, dirs: &[&str], extension: &str, skip: &[&str], limit: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack: Vec<PathBuf> = dirs.iter().map(|d| root.join(d)).collect();
    while let Some(path) = stack.pop() {
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target" || n == ".git") {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(&path) {
                stack.extend(entries.flatten().map(|e| e.path()));
            }
        } else if path.extension().is_some_and(|e| e == extension)
            && !skip.iter().any(|s| path.ends_with(s))
        {
            found.push(path);
        }
    }
    found.sort();
    // The same files every run, and not more text than is useful.
    let mut bytes = 0;
    found.retain(|path| {
        bytes += std::fs::metadata(path).map_or(0, |m| m.len() as usize);
        bytes <= limit
    });
    found
}

fn tokenize(tokenizer: &Tokenizer, paths: &[PathBuf]) -> Vec<Vec<u32>> {
    paths
        .iter()
        .filter_map(|path| std::fs::read(path).ok())
        .map(|text| tokenizer.encode(&text, EncodeOptions::default()))
        .filter(|tokens| tokens.len() >= 32)
        .collect()
}

/// The default settings, or `LOOKUP_MIN_MATCH` and `LOOKUP_MAX_HOPS` from the
/// environment, to see what they are worth.
fn config() -> LookupConfig {
    let from_env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
    let default = LookupConfig::default();
    LookupConfig {
        min_match: from_env("LOOKUP_MIN_MATCH").unwrap_or(default.min_match),
        max_hops: from_env("LOOKUP_MAX_HOPS").unwrap_or(default.max_hops),
        ..default
    }
}

fn measure(kind: &str, texts: &[Vec<u32>], cells: &mut Cells) {
    let total: usize = texts.iter().map(Vec::len).sum();
    println!("\n### {kind}: {} files, {total} tokens\n", texts.len());
    let config = config();
    for (replay, describe) in [
        ("plain", "every file written from an empty prompt"),
        ("edit", "the file is the prompt and is written again with edits"),
        ("again", "an earlier request wrote the file; a new one writes it with edits"),
    ] {
        let (mut ngram, mut own, mut shared, mut cut) =
            (Tally::default(), Tally::default(), Tally::default(), Tally::default());
        // One corpus for all requests of a run, as the engine has it.
        let mut corpus = LookupCorpus::new(config);
        let mut corpus_cut = LookupCorpus::new(config);
        for text in texts {
            let (prompt, output): (&[u32], Vec<u32>) = match replay {
                "plain" => (&[], text.clone()),
                "edit" => (text, edited(text)),
                _ => (&[], edited(text)),
            };
            if replay == "again" {
                // The earlier request.
                LookupDrafter::new().extend(&mut corpus, text);
                LookupDrafter::new().extend(&mut corpus_cut, text);
            }
            replay_ngram(prompt, &output, &mut ngram);
            // A corpus of its own per request: what the automaton gains over
            // the n-gram lookup without any sharing.
            let mut alone = LookupCorpus::new(config);
            replay_lookup(&mut alone, prompt, &output, false, &mut own, &mut [[0; 2]; 40]);
            replay_lookup(&mut corpus, prompt, &output, false, &mut shared, cells);
            replay_lookup(&mut corpus_cut, prompt, &output, true, &mut cut, &mut [[0; 2]; 40]);
        }
        println!("**{replay}** ({describe})\n\n{HEADER}");
        println!("{}", ngram.row("n-gram (3), per request"));
        println!("{}", own.row("lookup, corpus per request"));
        println!("{}", shared.row("lookup, shared corpus"));
        println!("{}", cut.row("lookup, shared, length by confidence"));
        let none = 1000.0 / (BASE_MS + PER_ROW_MS);
        println!("\nWithout drafts: {none:.1} tokens/s.\n");
    }
}

/// Nanoseconds per token of reading text into each drafter and of asking for
/// a draft at every token.
fn timing(texts: &[Vec<u32>]) {
    let total: usize = texts.iter().map(Vec::len).sum();
    let mut best = [f64::MAX; 4];
    for _ in 0..5 {
        let start = Instant::now();
        let mut drafters = Vec::new();
        for text in texts {
            let mut drafter = NgramDrafter::new(3);
            drafter.extend(text);
            drafters.push(drafter);
        }
        best[0] = best[0].min(start.elapsed().as_nanos() as f64 / total as f64);

        let start = Instant::now();
        let mut hits = 0usize;
        for text in texts {
            let mut drafter = NgramDrafter::new(3);
            for token in text {
                drafter.push(*token);
                hits += drafter.draft(MAX_DRAFT).len();
            }
        }
        best[1] = best[1].min(start.elapsed().as_nanos() as f64 / total as f64 - best[0]);

        let mut corpus = LookupCorpus::new(config());
        let start = Instant::now();
        for text in texts {
            LookupDrafter::new().extend(&mut corpus, text);
        }
        best[2] = best[2].min(start.elapsed().as_nanos() as f64 / total as f64);

        let mut corpus = LookupCorpus::new(config());
        let (mut draft, mut confidence) = (Vec::new(), Vec::new());
        let start = Instant::now();
        for text in texts {
            let mut drafter = LookupDrafter::new();
            for token in text {
                drafter.push(&mut corpus, *token);
                hits += drafter.draft(&corpus, MAX_DRAFT, &mut draft, &mut confidence);
            }
        }
        best[3] = best[3].min(start.elapsed().as_nanos() as f64 / total as f64 - best[2]);
        std::hint::black_box(hits);
    }
    println!("\n### Cost on the host, {total} tokens, best of 5\n");
    println!("| drafter | ns per token read | ns per draft asked |\n| --- | --- | --- |");
    println!("| n-gram (3) | {:.0} | {:.0} |", best[0], best[1]);
    println!("| lookup | {:.0} | {:.0} |", best[2], best[3]);
    let corpus = LookupCorpus::new(config());
    println!(
        "\nA corpus of {} MiB holds {} to {} tokens.",
        corpus.bytes() >> 20,
        corpus.tokens_per_generation(),
        2 * corpus.tokens_per_generation()
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or("usage: lookup_bench <model.gguf> [repository]")?;
    let root = args
        .next()
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."), PathBuf::from);
    let gguf = Gguf::open(&model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let english = files(&root, &["docs", "README.md"], "md", &[], 2 << 20);
    let code = files(
        &root,
        &["crates"],
        "rs",
        // Generated tables are not text anyone writes.
        &["tables.rs", "unicode_table.rs"],
        2 << 20,
    );
    let german = vec![
        root.join("crates/tokenizer/tests/corpus/german.txt"),
        root.join("crates/model/tools/capture/prompts/german.txt"),
    ];
    let mut cells: Cells = [[0; 2]; 40];
    let texts = [
        ("English (documentation)", tokenize(&tokenizer, &english)),
        ("Code (Rust)", tokenize(&tokenizer, &code)),
        ("German", tokenize(&tokenizer, &german)),
    ];
    println!("Vocabulary: {} tokens of {model}. {:?}", tokenizer.n_tokens(), config());
    for (kind, texts) in &texts {
        measure(kind, texts, &mut cells);
    }
    let all: Vec<Vec<u32>> = texts.iter().flat_map(|(_, t)| t.iter().cloned()).collect();
    timing(&all);

    println!("\n### Acceptance by context length and agreement (shared corpus, all replays)\n");
    println!("| context | once | always | mostly | seldom |\n| --- | --- | --- | --- | --- |");
    let names = ["1", "2", "3", "4", "5-6", "7-8", "9-12", "13-16", "17-32", "33+"];
    for (row, name) in names.iter().enumerate() {
        let cell = |column: usize| {
            let [tried, ok] = cells[row * 4 + column];
            if tried == 0 {
                "-".to_owned()
            } else {
                format!("{:.2} ({tried})", ok as f64 / tried as f64)
            }
        };
        println!("| {name} | {} | {} | {} | {} |", cell(0), cell(1), cell(2), cell(3));
    }
    Ok(())
}
