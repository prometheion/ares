//! Structured output through the server, with the real Qwen3.5-0.8B on a GPU:
//! a JSON schema is obeyed and the reply parses, `json_object` parses, a GBNF
//! grammar is obeyed, and with thinking on the thinking is free and the answer
//! obeys. The cost of the grammar in generation speed is measured at the end
//! (lines that start with `RES`).
//!
//! The server runs in this process over the real engine (requests go through
//! the router without a socket). Needs `ARES_TEST_CUDA=1` and
//! `ARES_TEST_QWEN35_0_8B` (the 0.8B); `ARES_TEST_DEVICE` names the device
//! (default 0). With `ARES_TEST_API_MODEL` the tests run on that file instead,
//! loaded as `ares --model` loads it (Flash-Next, the 27B: see
//! `gpu/mod.rs`).
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_0_8B=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test real_structured -- --test-threads=1
//! ```

use ares_gguf::Gguf;
use ares_tokenizer::{EncodeOptions, Tokenizer};
use serde_json::{Value, json};

mod common;
mod gpu;

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

/// The server over the engine.
struct Served(gpu::Served);

impl std::ops::Deref for Served {
    type Target = gpu::Served;

    fn deref(&self) -> &gpu::Served {
        &self.0
    }
}

impl Served {
    /// The 0.8B, or the model of `ARES_TEST_API_MODEL`. With `thinking` the
    /// model thinks when a request does not say otherwise; without, it does
    /// not.
    async fn start(thinking: bool) -> Option<Self> {
        gpu::Served::any(Some(thinking)).await.map(Self)
    }

    /// The token of the tag that closes the thinking, which the chat template
    /// of the model names.
    fn thinking_end(&self) -> u32 {
        let format = self.engine.chat_format("q").expect("a chat format");
        let close = format.thinking.expect("the model thinks").end_tag().to_owned();
        let tokenizer = Tokenizer::from_gguf(&Gguf::open(&self.path).unwrap()).unwrap();
        let tokens =
            tokenizer.encode(&close, EncodeOptions { add_special: false, parse_special: true });
        assert_eq!(tokens.len(), 1, "{close} is {tokens:?}");
        tokens[0]
    }

    async fn post(&self, path: &str, body: &Value) -> (u16, Value) {
        let (status, text) = self.ask(path, body).await;
        (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    /// The first choice of a request that is to succeed.
    async fn chat(&self, question: &str, extra: Value) -> Value {
        let mut body = json!({
            "model": "q",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 300,
            "temperature": 0.7,
        });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let (status, answer) = self.post(CHAT, &body).await;
        assert_eq!(status, 200, "{answer}");
        answer["choices"][0].clone()
    }
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 120},
            "hobbies": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            "mood": {"enum": ["happy", "sad", "angry"]},
        },
        "required": ["name", "age", "hobbies", "mood"],
        "additionalProperties": false,
    })
}

/// Whether `value` is a person as `person_schema` describes.
fn is_person(value: &Value) -> bool {
    let Some(object) = value.as_object() else { return false };
    let keys_ok = object.len() == 4
        && ["name", "age", "hobbies", "mood"].iter().all(|k| object.contains_key(*k));
    keys_ok
        && object["name"].as_str().is_some_and(|s| !s.is_empty())
        && object["age"].as_i64().is_some_and(|n| (0..=120).contains(&n))
        && object["hobbies"]
            .as_array()
            .is_some_and(|h| (1..=3).contains(&h.len()) && h.iter().all(Value::is_string))
        && ["happy", "sad", "angry"].contains(&object["mood"].as_str().unwrap_or(""))
}

fn content(choice: &Value) -> &str {
    choice["message"]["content"].as_str().or_else(|| choice["text"].as_str()).unwrap_or_default()
}

#[tokio::test]
async fn a_json_schema_is_obeyed_and_the_reply_parses() {
    let Some(served) = Served::start(false).await else { return };
    let wrapped = json!({
        "type": "json_schema",
        "json_schema": {"name": "person", "strict": true, "schema": person_schema()},
    });
    // The three ways to ask for the same schema, with several seeds each.
    for seed in 1..=4 {
        for extra in [
            json!({"response_format": wrapped, "seed": seed}),
            json!({"json_schema": person_schema(), "seed": seed}),
            json!({"response_format": {"type": "json_object", "schema": person_schema()}, "seed": seed}),
        ] {
            let choice = served.chat("Describe a made-up person.", extra.clone()).await;
            let text = content(&choice);
            let value: Value = serde_json::from_str(text)
                .unwrap_or_else(|e| panic!("{extra}: {text:?} does not parse: {e}"));
            assert!(is_person(&value), "{extra}: {text:?}");
            assert_eq!(choice["finish_reason"], "stop", "{text:?}");
        }
    }
    // The same on completions, which have no chat template.
    let body = json!({
        "model": "q",
        "prompt": "A made-up person, as JSON: ",
        "json_schema": person_schema(),
        "max_tokens": 300,
        "temperature": 0.7,
        "seed": 3,
    });
    let (status, answer) = served.post(COMPLETIONS, &body).await;
    assert_eq!(status, 200, "{answer}");
    let text = answer["choices"][0]["text"].as_str().unwrap();
    assert!(is_person(&serde_json::from_str(text).expect(text)), "{text:?}");
}

#[tokio::test]
async fn json_object_gives_an_object_that_parses() {
    let Some(served) = Served::start(false).await else { return };
    for seed in 1..=4 {
        let extra = json!({"response_format": {"type": "json_object"}, "seed": seed});
        let choice = served.chat("Tell me about the weather today.", extra).await;
        let text = content(&choice);
        let value: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        assert!(value.is_object(), "{text:?}");
        assert_eq!(choice["finish_reason"], "stop", "{text:?}");
    }
}

/// Whether `text` is three lines, each a dash, a space and lower case letters.
fn is_list(text: &str) -> bool {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    lines.len() == 3
        && lines.iter().all(|line| {
            line.strip_prefix("- ").and_then(|rest| rest.strip_suffix('\n')).is_some_and(|word| {
                !word.is_empty() && word.bytes().all(|b| b.is_ascii_lowercase())
            })
        })
}

#[tokio::test]
async fn a_gbnf_grammar_is_obeyed() {
    let Some(served) = Served::start(false).await else { return };
    let list = "root ::= (\"- \" [a-z]+ \"\\n\"){3}";
    for seed in 1..=4 {
        let choice = served
            .chat("Name three fruits, one per line.", json!({"grammar": list, "seed": seed}))
            .await;
        let text = content(&choice);
        assert!(is_list(text), "seed {seed}: {text:?}");
        assert_eq!(choice["finish_reason"], "stop");
    }
    let choice = served
        .chat("Is the sky blue? Answer yes or no.", json!({"grammar": "root ::= \"yes\" | \"no\""}))
        .await;
    assert!(["yes", "no"].contains(&content(&choice)), "{:?}", content(&choice));
    // A completion with a grammar over digits only.
    let body = json!({
        "model": "q", "prompt": "The year the wall fell: ", "grammar": "root ::= [0-9]{4}",
        "max_tokens": 20, "temperature": 0,
    });
    let (status, answer) = served.post(COMPLETIONS, &body).await;
    assert_eq!(status, 200, "{answer}");
    let text = answer["choices"][0]["text"].as_str().unwrap();
    assert!(text.len() == 4 && text.bytes().all(|b| b.is_ascii_digit()), "{text:?}");
}

#[tokio::test]
async fn a_grammar_that_is_wrong_is_a_400_that_names_it() {
    let Some(served) = Served::start(false).await else { return };
    let ask = |extra: Value| {
        let mut body = json!({"model": "q", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    let (status, answer) = served.post(CHAT, &ask(json!({"grammar": "root ::= nothing"}))).await;
    assert_eq!(status, 400, "{answer}");
    let message = answer["error"]["message"].as_str().unwrap();
    assert!(message.contains("grammar") && message.contains("'nothing'"), "{message}");
    let (status, answer) =
        served.post(CHAT, &ask(json!({"json_schema": {"type": "string", "pattern": "("}}))).await;
    assert_eq!(status, 400, "{answer}");
    assert!(answer["error"]["message"].as_str().unwrap().contains("json_schema"), "{answer}");
}

#[tokio::test]
async fn with_thinking_on_the_thinking_is_free_and_the_answer_obeys() {
    let Some(served) = Served::start(true).await else { return };
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "integer"}, "unit": {"enum": ["apples", "pears"]}},
        "required": ["answer", "unit"],
        "additionalProperties": false,
    });
    let format = json!({"type": "json_schema", "json_schema": {"name": "a", "schema": schema}});
    let question = "Anna has 3 apples and buys 4 more. How many apples does she have?";
    let reasoning_of = |choice: &Value| {
        choice["message"]["reasoning_content"].as_str().unwrap_or_default().to_owned()
    };

    // The thinking is free: under the rule it is the thinking of the same
    // request without a rule, character for character. The limit cuts both
    // while the model thinks (a model that is done sooner has the same
    // thinking all the same, and then an answer that differs).
    let cut = json!({"max_tokens": 200, "temperature": 0});
    let free = served.chat(question, cut.clone()).await;
    let mut ruled = cut;
    ruled["response_format"] = format.clone();
    let ruled = served.chat(question, ruled).await;
    let thinking = reasoning_of(&ruled);
    eprintln!(
        "RES thinking, cut at 200 tokens: {} bytes of reasoning with the rule, {} without",
        thinking.len(),
        reasoning_of(&free).len()
    );
    // A model that is not repeatable may think otherwise in another run, so
    // there the two are compared and not held to be the same.
    if served.repeatable {
        assert_eq!(thinking, reasoning_of(&free), "the rule changed the thinking");
    } else {
        let same = thinking == reasoning_of(&free);
        println!(
            "RES thinking with the rule and without: {}",
            if same { "the same" } else { "other" }
        );
    }
    // It is prose, which no JSON value is, and which the schema would have
    // refused from its first character.
    assert!(thinking.len() > 40, "the model did not think: {ruled}");
    assert!(serde_json::from_str::<Value>(thinking.trim()).is_err(), "{thinking:?}");

    // The answer obeys once the thinking is over. When that is, is the
    // model's business: the 0.8B, taking the most probable token, thinks about
    // this question in a loop of six lines to any limit, with the rule and
    // without. So the token of the closing tag gets a bias, the least of a
    // ladder with which the reply leaves its thinking and ends by itself;
    // none where the model leaves by itself. A bias brings the tag forward
    // and changes nothing else: the tag is no part of the answer's JSON.
    let close = served.thinking_end();
    let mut left = None;
    for bias in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0] {
        let choice = served
            .chat(
                question,
                json!({
                    "response_format": format,
                    "max_tokens": 600,
                    "temperature": 0,
                    "logit_bias": [[close, bias]],
                }),
            )
            .await;
        eprintln!(
            "RES thinking, a bias of {bias} on the closing tag: {} bytes of reasoning, answer \
             {:?}, finish {}",
            reasoning_of(&choice).len(),
            choice["message"]["content"].as_str().unwrap_or_default(),
            choice["finish_reason"]
        );
        if choice["finish_reason"] == "stop" {
            left = Some(choice);
            break;
        }
    }
    let choice = left.expect("no bias of the ladder made the reply leave its thinking");
    let message = &choice["message"];
    let reasoning = reasoning_of(&choice);
    let text = message["content"].as_str().unwrap_or_default();
    // The thinking came first and is the start of the free thinking above, or
    // all of it.
    assert!(reasoning.len() > 40, "the reply left its thinking at once: {message}");
    let shared = reasoning.len().min(thinking.len());
    let same = reasoning.is_char_boundary(shared)
        && thinking.is_char_boundary(shared)
        && reasoning[..shared].trim_end() == thinking[..shared].trim_end();
    if served.repeatable {
        assert!(same, "the same thinking up to the tag: {reasoning:?} and {thinking:?}");
    } else if !same {
        // Two runs of a model that is not repeatable, with other tokens.
        println!("RES the thinking up to the tag is not the free thinking (not repeatable)");
    }
    // The answer obeys.
    let answer: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    let object = answer.as_object().expect("an object");
    assert!(object["answer"].is_i64() && object.len() == 2, "{text:?}");
    assert!(["apples", "pears"].contains(&object["unit"].as_str().unwrap()), "{text:?}");
    // A request that turns thinking off gets the answer from its first token.
    let choice = served
        .chat(
            question,
            json!({
                "response_format": format,
                "chat_template_kwargs": {"enable_thinking": false},
                "temperature": 0,
            }),
        )
        .await;
    assert!(choice["message"]["reasoning_content"].is_null(), "{choice}");
    let text = content(&choice);
    assert!(serde_json::from_str::<Value>(text).is_ok_and(|v| v["answer"].is_i64()), "{text:?}");
}

/// Tokens a second of a request: the timings the server reports.
fn speed(answer: &Value) -> f64 {
    answer["timings"]["predicted_per_second"].as_f64().expect("timings")
}

#[tokio::test]
async fn the_grammar_costs_what_the_mask_costs_per_step() {
    let Some(served) = Served::start(false).await else { return };
    // About 400 tokens of numbers either way.
    let schema = json!({"type": "array", "items": {"type": "integer"}, "minItems": 150});
    let free = json!({
        "model": "q", "messages": [{"role": "user", "content": "Count from 1 to 150, separated by commas."}],
        "max_tokens": 400, "temperature": 0, "ignore_eos": true,
    });
    let mut with_schema = free.clone();
    with_schema["response_format"] =
        json!({"type": "json_schema", "json_schema": {"name": "n", "schema": schema}});
    with_schema["ignore_eos"] = json!(false);
    let object = json!({"type": "json_object"});
    let mut with_object = free.clone();
    with_object["response_format"] = object;
    with_object["ignore_eos"] = json!(false);
    // Warm up, then three of each, taking the best.
    served.post(CHAT, &free).await;
    let mut best = [0.0f64; 3];
    let mut tokens = [0u64; 3];
    for _ in 0..3 {
        for (i, body) in [&free, &with_schema, &with_object].into_iter().enumerate() {
            let (status, answer) = served.post(CHAT, body).await;
            assert_eq!(status, 200, "{answer}");
            best[i] = best[i].max(speed(&answer));
            tokens[i] = answer["usage"]["completion_tokens"].as_u64().unwrap();
        }
    }
    eprintln!("RES free text:           {:7.1} tokens/s over {} tokens", best[0], tokens[0]);
    eprintln!("RES json schema (array): {:7.1} tokens/s over {} tokens", best[1], tokens[1]);
    eprintln!("RES json_object:         {:7.1} tokens/s over {} tokens", best[2], tokens[2]);
    // A mask of a few microseconds on the host and a pass over the logits on
    // the device must not halve the speed.
    assert!(best[1] > best[0] * 0.5, "{best:?}");
}
