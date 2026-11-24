//! What a step of a grammar costs on the vocabulary of a real model: the time
//! to compile, and the time of a mask along a random walk through the grammar.
//!
//! `ARES_TEST_QWEN35_0_8B=<model.gguf> cargo run --release -p ares-grammar \
//!  --example mask_cost -- [steps]`

use std::time::Instant;

use ares_api::ToolFormat;
use ares_gguf::Gguf;
use ares_grammar::calls::{CallRule, Callable};
use ares_grammar::{Compiler, Source, Vocabulary};
use ares_tokenizer::Tokenizer;

const SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "name": {"type": "string", "minLength": 1, "maxLength": 40},
    "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "email": {"type": "string", "pattern": "^[a-z0-9._]+@[a-z0-9.]+\\.[a-z]{2,}$"},
    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b", "c", "dd", "eee"]}, "maxItems": 8},
    "address": {
      "type": "object",
      "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"},
        "zip": {"type": "string", "pattern": "^[0-9]{5}$"},
        "country": {"enum": ["DE", "FR", "US", "UK"]}
      },
      "required": ["street", "city", "zip"],
      "additionalProperties": false
    },
    "scores": {"type": "array", "items": {"type": "number"}, "minItems": 1},
    "active": {"type": "boolean"},
    "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}
  },
  "required": ["name", "age", "email", "address", "active"],
  "additionalProperties": false
}"#;

const JSON_GBNF: &str = include_str!("../tests/grammars/json.gbnf");

/// The grammar of the calls of two tools in the format of Qwen3.8's template.
/// `text` is true for a tool whose arguments are strings (a random walk then
/// stays in the text of one), false for one whose arguments are JSON.
fn calls(text: bool) -> String {
    let format = ToolFormat {
        call_open: "<tool_call>\n<function=".into(),
        name_close: ">\n".into(),
        param_open: "<parameter=".into(),
        key_close: ">\n".into(),
        param_close: "\n</parameter>\n".into(),
        call_close: "</function>\n</tool_call>".into(),
        before_calls: "\n\n".into(),
        between_calls: "\n".into(),
    };
    let write: serde_json::Value = serde_json::from_str(
        r#"{"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}"#,
    )
    .unwrap();
    let set: serde_json::Value = serde_json::from_str(
        r#"{"type": "object", "properties": {"level": {"type": "integer"}, "mode": {"enum": ["quiet", "loud"]}, "on": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "number"}}}, "required": ["level", "mode", "on", "tags"]}"#,
    )
    .unwrap();
    let tools = [Callable {
        name: if text { "write_file" } else { "set_volume" },
        parameters: Some(if text { &write } else { &set }),
    }];
    CallRule { format: &format, tools: &tools, parallel: true, triggered: false }.lark().unwrap()
}

fn main() {
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let path = std::env::var("ARES_TEST_QWEN35_0_8B").expect("ARES_TEST_QWEN35_0_8B");
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    let n = tokenizer.n_tokens() as u32;
    let pieces: Vec<Vec<u8>> =
        (0..n).map(|t| tokenizer.piece(t, false).unwrap_or(b"").to_vec()).collect();
    let ends: Vec<u32> = (0..n).filter(|&t| tokenizer.is_end_of_generation(t)).collect();
    let t = Instant::now();
    let vocab = Vocabulary::new(&pieces, &ends).unwrap();
    let compiler = Compiler::new(vocab).unwrap();
    println!("vocabulary of {n} tokens, {} end tokens: {:.1} ms", ends.len(), ms(t));
    println!("{}", peak_memory());

    let (text_calls, json_calls) = (calls(true), calls(false));
    for (name, source) in [
        ("tool calls, string arguments", Source::Lark(&text_calls)),
        ("tool calls, JSON arguments", Source::Lark(&json_calls)),
        ("json schema (moderate)", Source::JsonSchema(SCHEMA)),
        ("json_object", Source::JsonSchema(r#"{"type":"object"}"#)),
        ("llama.cpp json.gbnf", Source::Gbnf(JSON_GBNF)),
        ("gbnf: yes or no", Source::Gbnf(r#"root ::= "yes" | "no""#)),
    ] {
        let t = Instant::now();
        let mut grammar = compiler.compile(source).unwrap();
        let compile = ms(t);
        let mut times = Vec::new();
        let mut taken = 0usize;
        let mut seed = 7u64;
        for round in 0..5 {
            let mut cursor = grammar.start();
            let mut tokens = Vec::new();
            for _ in 0..steps / 5 {
                let t = Instant::now();
                let mask = cursor.mask().unwrap();
                times.push(ms(t));
                let allowed: Vec<u32> = (0..n)
                    .filter(|&t| mask[(t / 32) as usize] >> (t % 32) & 1 != 0 && !ends.contains(&t))
                    .collect();
                if allowed.is_empty() {
                    break;
                }
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407 + round);
                tokens.push(allowed[(seed >> 33) as usize % allowed.len()]);
                assert_eq!(cursor.goto(&tokens), tokens.len());
                taken += 1;
            }
        }
        times.sort_by(f64::total_cmp);
        let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
        println!(
            "{name}: compile {compile:.2} ms, mask over {taken} steps: median {:.3} ms, p90 {:.3} ms, max {:.3} ms",
            at(0.5),
            at(0.9),
            at(1.0)
        );
    }
}

/// The most memory the process has held, from the kernel.
fn peak_memory() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).map(str::to_owned))
        .unwrap_or_default()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
