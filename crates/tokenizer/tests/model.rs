//! The tokenizer of a real model against llama.cpp's results.
//!
//! These tests need a model file, so they run only when `ARES_TEST_GGUF` names one
//! and say so otherwise:
//!
//! ```text
//! ARES_TEST_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf cargo test -p ares-tokenizer --test model
//! ```
//!
//! The expected results are in `tests/golden/`, written by llama.cpp's own tools (see
//! `generate.py` and `generate_chat.py` there). With `ARES_TEST_LLAMA_TOKENIZE` set to
//! a working `llama-tokenize` as well, one test asks the program itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ares_gguf::Gguf;
use ares_tokenizer::{
    ChatOptions, Content, EncodeOptions, Json, Message, TokenKind, Tokenizer, Tool, ToolCall,
};

fn model() -> Option<(Gguf, Tokenizer)> {
    let Some(path) = std::env::var_os("ARES_TEST_GGUF") else {
        eprintln!(
            "skipped: set ARES_TEST_GGUF=<model.gguf> to compare a real tokenizer with llama.cpp"
        );
        return None;
    };
    let gguf = Gguf::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    Some((gguf, tokenizer))
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn corpus() -> Vec<(String, Vec<u8>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            (path.file_name().unwrap().to_str().unwrap().to_owned(), std::fs::read(&path).unwrap())
        })
        .collect()
}

fn options(special: bool) -> EncodeOptions {
    EncodeOptions { add_special: false, parse_special: special }
}

#[test]
fn gives_the_token_ids_of_llama_cpp_for_the_corpus() {
    let Some((_, tokenizer)) = model() else { return };
    let golden = std::fs::read_to_string(golden_dir().join("qwen35_ids.txt")).unwrap();
    let mut expected: HashMap<(String, bool), Vec<u32>> = HashMap::new();
    for line in golden.lines() {
        let mut fields = line.splitn(3, '\t');
        let (name, mode, ids) =
            (fields.next().unwrap(), fields.next().unwrap(), fields.next().unwrap());
        let ids = ids.split_whitespace().map(|id| id.parse().unwrap()).collect();
        expected.insert((name.to_owned(), mode == "special"), ids);
    }
    let files = corpus();
    assert_eq!(expected.len(), files.len() * 2, "the golden file is for this corpus");
    let mut tokens = 0;
    for (name, text) in &files {
        for special in [true, false] {
            let want = &expected[&(name.clone(), special)];
            let got = tokenizer.encode(text, options(special));
            assert_eq!(&got, want, "{name} (special tokens parsed: {special})");
            tokens += got.len();
        }
    }
    eprintln!("{} files, {tokens} token ids equal to llama.cpp's", files.len());
}

#[test]
fn decodes_the_corpus_to_the_text_it_came_from() {
    let Some((_, tokenizer)) = model() else { return };
    for (name, text) in corpus() {
        for special in [true, false] {
            let ids = tokenizer.encode(&text, options(special));
            let back = tokenizer.decode(&ids, true).unwrap();
            if std::str::from_utf8(&text).is_ok() {
                assert_eq!(back, text, "{name}");
            } else {
                // The bytes that start no character came back as U+FFFD (the surrogates, which
                // strict UTF-8 refuses but llama.cpp accepts, came back as they were); what
                // is left encodes to the same tokens.
                assert_ne!(back, text, "{name}");
                assert_eq!(tokenizer.encode(&back, options(special)), ids, "{name}");
            }
        }
    }
}

#[test]
fn knows_the_vocabulary_of_the_qwen_models() {
    let Some((gguf, tokenizer)) = model() else { return };
    assert_eq!(gguf.str("tokenizer.ggml.pre"), Some("qwen35"));
    assert_eq!(tokenizer.n_tokens(), 248_320);
    assert!(!tokenizer.adds_bos() && !tokenizer.adds_eos());
    let id = |text| tokenizer.special_token_id(text).unwrap();
    let (start, end, endoftext) = (id("<|im_start|>"), id("<|im_end|>"), id("<|endoftext|>"));
    assert_eq!((start, end, endoftext), (248_045, 248_046, 248_044));
    let special = tokenizer.special_tokens();
    assert_eq!(special.eos, Some(end));
    assert_eq!(special.eot, Some(end));
    assert_eq!(special.fim_prefix, Some(id("<|fim_prefix|>")));
    assert!(tokenizer.is_end_of_generation(end) && tokenizer.is_end_of_generation(endoftext));
    assert!(!tokenizer.is_end_of_generation(start));
    assert_eq!(tokenizer.kind(start), Some(TokenKind::Control));
    assert_eq!(tokenizer.kind(id("<think>")), Some(TokenKind::UserDefined));
    assert_eq!(tokenizer.kind(248_077), Some(TokenKind::Unused));
    assert_eq!(tokenizer.kind(0), Some(TokenKind::Normal));
    // The user defined tags are found without asking, the control tokens only on request.
    let text = "<think>x</think><|im_end|>";
    assert_eq!(tokenizer.encode(text, options(true)), [id("<think>"), 87, id("</think>"), end]);
    assert!(!tokenizer.encode(text, options(false)).contains(&end));
}

fn get<'a>(json: &'a Json, key: &str) -> Option<&'a Json> {
    match json {
        Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn string(json: Option<&Json>) -> Option<String> {
    match json {
        Some(Json::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// The messages, tools and options of a request in the shape of the OpenAI API. `None` for a
/// request that has no counterpart here (images).
fn request(body: &Json) -> Option<(Vec<Message>, Vec<Tool>, ChatOptions)> {
    let Some(Json::Array(list)) = get(body, "messages") else { return None };
    let mut messages = Vec::new();
    for item in list {
        let mut message = Message { role: string(get(item, "role"))?, ..Message::default() };
        match get(item, "content") {
            Some(Json::String(text)) => message.content = Content::Text(text.clone()),
            Some(Json::Array(parts)) => {
                let mut texts = Vec::new();
                for part in parts {
                    if string(get(part, "type")).as_deref() != Some("text") {
                        return None;
                    }
                    texts.push(string(get(part, "text"))?);
                }
                message.content = Content::Parts(texts);
            }
            _ => {}
        }
        if let Some(Json::Array(calls)) = get(item, "tool_calls") {
            for call in calls {
                let function = get(call, "function")?;
                let arguments = match get(function, "arguments")? {
                    Json::String(text) => text.clone(),
                    other => other.dump(),
                };
                message.tool_calls.push(ToolCall {
                    id: string(get(call, "id")),
                    name: string(get(function, "name"))?,
                    arguments,
                });
            }
        }
        message.reasoning_content = string(get(item, "reasoning_content")).unwrap_or_default();
        message.name = string(get(item, "name")).unwrap_or_default();
        message.tool_call_id = string(get(item, "tool_call_id")).unwrap_or_default();
        messages.push(message);
    }
    let mut tools = Vec::new();
    if let Some(Json::Array(list)) = get(body, "tools") {
        for tool in list {
            let function = get(tool, "function")?;
            tools.push(Tool {
                name: string(get(function, "name"))?,
                description: string(get(function, "description")).unwrap_or_default(),
                parameters: get(function, "parameters").cloned().unwrap_or(Json::Object(vec![])),
            });
        }
    }
    let mut options = ChatOptions::default();
    if let Some(Json::Bool(false)) = get(body, "add_generation_prompt") {
        options.add_generation_prompt = false;
    }
    if let Some(Json::Object(kwargs)) = get(body, "chat_template_kwargs") {
        for (key, value) in kwargs {
            if let (true, Json::Bool(on)) = (key == "enable_thinking", value) {
                options.enable_thinking = *on;
            }
            options.variables.push((key.clone(), value.clone()));
        }
    }
    Some((messages, tools, options))
}

/// FNV-1a over the bytes, which names the fixture of a template.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01B3))
}

#[test]
fn renders_the_prompts_of_llama_cpp() {
    let Some((gguf, tokenizer)) = model() else { return };
    let template = gguf.str("tokenizer.chat_template").expect("the model has a chat template");
    let name = format!("chat_{:016x}.jsonl", fnv1a(template.as_bytes()));
    let path = golden_dir().join(&name);
    let Ok(golden) = std::fs::read_to_string(&path) else {
        eprintln!(
            "skipped: no fixture {name} for this model's chat template (see tests/golden/generate_chat.py)"
        );
        return;
    };
    let (mut rendered, mut refused) = (0, 0);
    for (n, line) in golden.lines().enumerate() {
        let case = Json::parse(line).unwrap();
        let body = get(&case, "request").unwrap();
        let ours = request(body).map(|(m, t, o)| tokenizer.render_chat(&m, &t, &o));
        match (get(&case, "prompt"), &ours) {
            (Some(Json::String(want)), Some(Ok(got))) => {
                assert_eq!(got, want, "request {n}: {}", body.dump());
                rendered += 1;
            }
            (None, Some(Err(_))) => refused += 1,
            // A request with an image: llama.cpp refuses it as well, without a multimodal projector.
            (None, None) => refused += 1,
            (want, got) => panic!("request {n}: llama.cpp {want:?}, here {got:?}: {}", body.dump()),
        }
    }
    assert!(rendered > 100 && refused > 0, "{rendered} prompts, {refused} refusals");
    eprintln!(
        "{name}: {rendered} prompts equal to llama.cpp's, {refused} conversations refused by both"
    );
}

/// Asks `llama-tokenize` itself, for a corpus of the files and of texts made up here.
#[test]
fn agrees_with_llama_tokenize() {
    let Some(binary) = std::env::var_os("ARES_TEST_LLAMA_TOKENIZE") else {
        eprintln!(
            "skipped: set ARES_TEST_LLAMA_TOKENIZE=<llama-tokenize> (and ARES_TEST_GGUF) to compare with the program"
        );
        return;
    };
    let Some((gguf, tokenizer)) = model() else { return };
    let model_path = gguf.path(0).to_owned();

    let mut texts = corpus();
    let mut rng = 0x1234_5678_9ABC_DEF1u64;
    let mut next = |n: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % n as u64) as usize
    };
    let pieces = [
        "the",
        " The",
        "Straße",
        " größer",
        "'s",
        "'LL",
        " don't",
        "123",
        " 4567",
        "\n",
        "\n\n ",
        "  ",
        "\t",
        "\r\n",
        "!!",
        " (",
        "));",
        "日本語",
        "😀",
        "👨‍👩‍👧",
        "e\u{301}",
        "\u{a0}",
        "\u{3000}",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</tool_call>",
        "\u{fffd}",
    ];
    for i in 0..60 {
        let mut text = Vec::new();
        for _ in 0..1 + next(40) {
            text.extend_from_slice(pieces[next(pieces.len())].as_bytes());
        }
        if i % 7 == 0 {
            text.push(0xFF); // invalid UTF-8 as well
            text.extend_from_slice(&[0xE2, 0x82]);
        }
        texts.push((format!("generated-{i}"), text));
    }

    let dir = std::env::temp_dir().join(format!("ares-tokenizer-model-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let run = |binary: &Path, name: &str, text: &[u8], special: bool| -> Vec<u32> {
        let file = dir.join(format!("{name}-{special}"));
        std::fs::write(&file, text).unwrap();
        let mut command = std::process::Command::new(binary);
        command
            .arg("-m")
            .arg(&model_path)
            .args(["--ids", "--no-escape", "--log-disable", "-f"])
            .arg(&file);
        if !special {
            command.arg("--no-parse-special");
        }
        // Only the vocabulary is read, and no GPU has any business here.
        let out = command.env("CUDA_VISIBLE_DEVICES", "").output().expect("run llama-tokenize");
        assert!(
            out.status.success(),
            "llama-tokenize failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        let inner = text.trim().trim_start_matches('[').trim_end_matches(']').trim();
        if inner.is_empty() {
            Vec::new()
        } else {
            inner.split(',').map(|id| id.trim().parse().unwrap()).collect()
        }
    };
    let binary = PathBuf::from(binary);
    let work = std::sync::Mutex::new(
        texts.iter().flat_map(|t| [(t, true), (t, false)]).collect::<Vec<_>>(),
    );
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                loop {
                    let Some(((name, text), special)) = work.lock().unwrap().pop() else { break };
                    assert_eq!(
                        tokenizer.encode(text, options(special)),
                        run(&binary, name, text, special),
                        "{name} (special: {special})"
                    );
                }
            });
        }
    });
    let _ = std::fs::remove_dir_all(dir);
}
