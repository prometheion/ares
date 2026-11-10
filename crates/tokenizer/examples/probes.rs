//! Compares the Jinja of this crate with llama.cpp's on many small templates.
//!
//! ```text
//! python3 -I tools/make_probes.py > probes.jsonl
//! cargo run --release -p ares-tokenizer --example probes -- REFTPL probes.jsonl [-v]
//! ```
//!
//! `REFTPL` is `tools/reference/reftpl` built against llama.cpp (see the source for how).
//! Every probe is a template and a variable `v`; llama.cpp renders it, and so does
//! [`ChatTemplate`]. The results are sorted:
//!
//! - same: both gave the same text;
//! - both error: both refused;
//! - we refuse a template that uses something outside the dialect (and llama.cpp renders it or
//!   not): a cost of supporting only what the models use;
//! - different: both rendered, with different text (printed, the first of each template);
//! - only we error / only llama.cpp errors: one rendered where the other failed. Of these,
//!   only the second can put a prompt in front of a model that llama.cpp would not have given.
//!
//! The probes that come out different or one-sided are what the crate documents as its
//! differences from llama.cpp, and the list of them should only shrink.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use ares_tokenizer::{ChatError, ChatOptions, ChatTemplate, Json, Message};

fn get<'a>(json: &'a Json, key: &str) -> Option<&'a Json> {
    match json {
        Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

struct Reference {
    path: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Reference {
    fn start(path: &str) -> Self {
        let mut child = Command::new(path)
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the reference");
        let stdin = child.stdin.take().expect("a pipe");
        let stdout = BufReader::new(child.stdout.take().expect("a pipe"));
        Self { path: path.to_owned(), child, stdin, stdout }
    }

    /// The prompt llama.cpp renders for a request line, or why it did not. A reference that
    /// crashes is started again and counts as an error.
    fn render(&mut self, line: &str) -> Result<String, String> {
        let mut response = String::new();
        let _ = writeln!(self.stdin, "{line}");
        self.stdout.read_line(&mut response).expect("read the reference");
        if response.is_empty() {
            let _ = self.child.wait();
            *self = Self::start(&self.path.clone());
            return Err("the reference crashed".into());
        }
        let json = Json::parse(&response).map_err(|e| format!("unparsable response: {e}"))?;
        match (get(&json, "ok"), get(&json, "error")) {
            (Some(Json::String(prompt)), _) => Ok(prompt.clone()),
            (_, Some(Json::String(error))) => Err(error.clone()),
            _ => Err(format!("unexpected response {response}")),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, reftpl, probes, rest @ ..] = &args[..] else {
        eprintln!("usage: probes REFTPL PROBES.jsonl [-v]");
        std::process::exit(2);
    };
    let verbose = rest.iter().any(|a| a == "-v");
    let mut reference = Reference::start(reftpl);
    let text = std::fs::read_to_string(probes).expect("read the probes");

    let (
        mut same,
        mut both_error,
        mut we_refuse,
        mut different,
        mut only_we_error,
        mut only_they_error,
    ) = (0, 0, 0, 0, 0, 0);
    let mut shown: BTreeMap<(&str, String), usize> = BTreeMap::new();
    for line in text.lines() {
        let probe = Json::parse(line).expect("a probe is JSON");
        let (Some(Json::String(label)), Some(Json::String(source))) =
            (get(&probe, "label"), get(&probe, "template"))
        else {
            panic!("a probe has a label and a template");
        };
        let mut options = ChatOptions { add_generation_prompt: false, ..ChatOptions::default() };
        if let Some(Json::Object(variables)) = get(&probe, "kwargs") {
            options.variables = variables.clone();
        }
        let theirs = reference.render(line);
        let ours = match ChatTemplate::new(source, "", "") {
            Err(error @ ChatError::Unsupported { .. }) => Err((true, error.to_string())),
            Err(error) => Err((false, error.to_string())),
            Ok(template) => template
                .render(&[Message::new("user", "x")], &[], &options)
                .map_err(|e| (false, e.to_string())),
        };
        let value = get(&probe, "kwargs").map(Json::dump).unwrap_or_default();
        let mut show = |kind: &'static str, theirs: String, ours: String| {
            let count = shown.entry((kind, label.clone())).or_default();
            *count += 1;
            if verbose || *count == 1 {
                println!("{kind} {label:?} {value}\n  llama.cpp: {theirs}\n  here:      {ours}");
            }
        };
        match (&theirs, &ours) {
            (Ok(a), Ok(b)) if a == b => same += 1,
            (Err(_), Err((false, _))) => both_error += 1,
            (_, Err((true, _))) => we_refuse += 1,
            (Ok(a), Ok(b)) => {
                different += 1;
                show("DIFFERENT", format!("{a:?}"), format!("{b:?}"));
            }
            (Ok(a), Err((false, e))) => {
                only_we_error += 1;
                show("ONLY WE ERROR", format!("{a:?}"), e.clone());
            }
            (Err(e), Ok(b)) => {
                only_they_error += 1;
                let first_line: String =
                    e.trim().lines().next().unwrap_or_default().chars().take(100).collect();
                show("ONLY LLAMA.CPP ERRORS", first_line, format!("{b:?}"));
            }
        }
    }
    println!(
        "same {same}, both error {both_error}, we refuse the template {we_refuse}, different {different}, \
         only we error {only_we_error}, only llama.cpp errors {only_they_error}"
    );
}
