//! The `ares` binary serving a real model from a GPU, over real sockets: the
//! basics of a server (tokens, properties, streams, stop strings, a client
//! that leaves, requests at once), what is refused, the signals, the prompt
//! cache across turns and a server with no RAM to spare, and what a request
//! asks of the sampler. `real_drafts.rs` has the drafts, `real_flash_next.rs`
//! Qwen3.8-Flash-Next.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_0_8B` (a small `qwen35`
//! file, Qwen3.5-0.8B); `ARES_TEST_DEVICE` names the device (default 0) and
//! `ARES_TEST_PORT` the first of the ten ports on 127.0.0.1 the tests
//! listen on (default 18951). Every test starts its own server.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_0_8B=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test -p ares-server --test real -- --test-threads=1
//! ```
//!
//! The servers run as every server does, with the fast products: a row's
//! rounding depends on the size of the step that computes it, so a reply
//! computed next to another request, or with its prompt from the cache, is
//! compared with the one it should be up to a near tie
//! (`same_up_to_a_near_tie`), unless its tokens follow with a wide margin.

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::thread;
use std::time::Duration;

use common::model_file;
use common::process::{
    Client, ENGLISH, GERMAN, LIMIT, Server, ares, eventually, same_up_to_a_near_tie, serve,
};
use common::socket::{Reply, read_events, streamed_text};
use serde_json::{Value, json};

mod common;

/// The model to serve, or `None` (and a note) when the tests are not enabled.
fn model() -> Option<PathBuf> {
    model_file("ARES_TEST_QWEN35_0_8B", "a small qwen35 model file")
}

fn port(offset: u16) -> u16 {
    common::process::port(18951, offset)
}

/// Tokens and properties, the count of the thinking, the stop string that
/// ended a reply and a prefill, with the model's own template.
#[test]
fn the_basics_of_a_server_work_with_a_real_model() {
    let Some(model) = model() else { return };
    let server = serve(&model, port(5), &[]);
    let props = server.get("/props").json();
    assert!(props["chat_template"].as_str().unwrap().contains("<|im_start|>"), "{props}");
    assert_eq!(props["chat_template_caps"]["supports_reasoning"], true, "{props}");
    assert_eq!(props["context_length"], 8192);

    // The tokens of a chat request are the prompt the chat route counts, and
    // tokens go back to their text.
    let chat = server.chat(ENGLISH, json!({}));
    let tokens = server.post("/tokenize", &chat).json()["tokens"].as_array().unwrap().len();
    let counted = server.post("/v1/chat/completions/input_tokens", &chat).json();
    assert_eq!(counted["input_tokens"], tokens);
    let text = json!({"content": GERMAN, "with_pieces": true});
    let pieces = server.post("/tokenize", &text).json()["tokens"].clone();
    let ids: Vec<Value> = pieces.as_array().unwrap().iter().map(|p| p["id"].clone()).collect();
    let back = server.post("/detokenize", &json!({"tokens": ids})).json();
    assert_eq!(back["content"], GERMAN);

    // A reply that thinks: its thinking is counted to the closing tag.
    let reply =
        server.post("/v1/chat/completions", &server.chat(ENGLISH, json!({"max_tokens": 600})));
    let reply = reply.json();
    let usage = &reply["usage"];
    let reasoning = usage["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap();
    let completion = usage["completion_tokens"].as_u64().unwrap();
    let message = &reply["choices"][0]["message"];
    let thought = message["reasoning_content"].as_str().unwrap_or_default();
    let thought = json!({"content": thought});
    let thought = server.post("/tokenize", &thought).json()["tokens"].as_array().unwrap().len();
    if message["content"].as_str().is_some_and(|content| !content.is_empty()) {
        // The reasoning's tokens and the layout and tag around it: a token or
        // two the tokenizer and the model cut differently.
        assert!(reasoning < completion, "{reply}");
        assert!(reasoning.abs_diff(thought as u64 + 2) <= 4, "{reasoning} and {thought}: {reply}");
    } else {
        assert_eq!(reasoning, completion, "cut inside the thinking: {reply}");
    }

    // A stop string from the middle of a greedy reply ends it and is named.
    let plain = json!({"max_tokens": 40, "chat_template_kwargs": {"enable_thinking": false}});
    let free = server.post("/v1/chat/completions", &server.chat(ENGLISH, plain.clone())).json();
    let answer = free["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
    let stop: String = answer.chars().skip(20).take(4).collect();
    let mut stopped = plain.clone();
    stopped["stop"] = json!([stop]);
    let cut = server.post("/v1/chat/completions", &server.chat(ENGLISH, stopped)).json();
    let choice = &cut["choices"][0];
    assert_eq!(choice["stop_reason"], stop, "{cut}");
    assert_eq!(choice["message"]["content"], answer[..answer.find(&stop).unwrap()], "{cut}");
    let messages = json!({
        "model": server.model,
        "max_tokens": 40,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": ENGLISH}],
        "stop_sequences": [stop],
    });
    let message = server.post("/v1/messages", &messages).json();
    assert_eq!(message["stop_reason"], "stop_sequence", "{message}");
    assert_eq!(message["stop_sequence"], stop);

    // A prefill: the reply goes on from the given start of the answer.
    let start: String = answer.chars().take(20).collect();
    let mut prefill = messages.clone();
    prefill["messages"] = json!([
        {"role": "user", "content": ENGLISH},
        {"role": "assistant", "content": start},
    ]);
    prefill["stop_sequences"] = json!([]);
    let rest = server.post("/v1/messages", &prefill).json();
    assert!(["max_tokens", "end_turn"].contains(&rest["stop_reason"].as_str().unwrap()), "{rest}");
    let rest = rest["content"][0]["text"].as_str().unwrap_or_default().to_owned();
    // The template lays the turn out as its own, so the reply need not be the
    // free one's; it goes on, and does not begin the answer again.
    assert!(!rest.trim().is_empty() && !rest.starts_with(&start), "{start:?} then {rest:?}");
    server.stop();
}

#[test]
fn a_chat_gives_the_same_text_streamed_and_plain_for_the_same_seed() {
    let Some(model) = model() else { return };
    let server = serve(&model, port(0), &["--chat-var", "enable_thinking=false"]);

    assert_eq!(server.get("/health").body, r#"{"status":"ok"}"#);
    let models = server.get("/v1/models").json();
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["context_length"], 8192);
    assert!(models["data"][0]["created"].as_u64().unwrap() > 1_600_000_000, "{models}");

    for question in [GERMAN, ENGLISH] {
        let sampled = json!({
            "max_tokens": 48, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.02,
            "seed": 20_261_110,
        });
        let plain = server.post("/v1/chat/completions", &server.chat(question, sampled.clone()));
        assert_eq!(plain.status, 200, "{}", plain.body);
        let plain = plain.json();
        let text = plain["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
        assert!(text.chars().filter(|c| c.is_alphabetic()).count() > 20, "an answer: {text:?}");
        assert_eq!(plain["choices"][0]["finish_reason"], "length");
        let usage = &plain["usage"];
        assert_eq!(usage["completion_tokens"], 48);
        let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap();
        assert!((10..80).contains(&prompt_tokens), "the question through the template: {usage}");
        assert_eq!(usage["total_tokens"], prompt_tokens + 48);
        // What the engine computed before may start the prompt: the start of
        // the template, at least, is the same for every question.
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
        assert!(cached < prompt_tokens, "the last token is always computed: {usage}");
        let timings = &plain["timings"];
        assert_eq!(timings["prompt_n"], prompt_tokens - cached, "the cached part is not computed");
        assert_eq!(timings["predicted_n"], 48);
        assert!(timings["prompt_ms"].as_f64().unwrap() > 0.0, "{timings}");
        assert!(timings["predicted_per_second"].as_f64().unwrap() > 1.0, "{timings}");
        assert!(timings["draft_n"].as_u64().is_some(), "it ran alone: {timings}");
        assert!(timings["draft_n_accepted"].as_u64() <= timings["draft_n"].as_u64());

        let mut streaming = sampled.clone();
        streaming["stream"] = json!(true);
        streaming["stream_options"] = json!({"include_usage": true});
        let reply = server.post("/v1/chat/completions", &server.chat(question, streaming));
        assert_eq!(reply.status, 200, "{}", reply.body);
        let (chunks, done) = reply.events();
        assert!(done, "the stream ends with [DONE]");
        // The stream's prompt came from the cache but for its last token, which
        // a step of one row computed.
        same_up_to_a_near_tie("streamed and plain, same seed", &streamed_text(&chunks), &text, 24);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let finish = &chunks[chunks.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        assert_eq!(finish["timings"]["predicted_n"], 48);
        // The same prompt a second time comes from the cache, all but its
        // last token; the rest of the usage is the same.
        let mut streamed_usage = chunks[chunks.len() - 1]["usage"].clone();
        let streamed_cached = streamed_usage["prompt_tokens_details"]["cached_tokens"].as_u64();
        assert_eq!(streamed_cached, Some(prompt_tokens - 1), "{streamed_usage}");
        streamed_usage["prompt_tokens_details"] = usage["prompt_tokens_details"].clone();
        assert_eq!(streamed_usage, *usage, "the usage chunk");

        // Another seed is another text; the most probable token needs none.
        let mut other = sampled.clone();
        other["seed"] = json!(4711);
        let other = server.post("/v1/chat/completions", &server.chat(question, other)).json();
        assert_ne!(other["choices"][0]["message"]["content"], text, "the seed is used");
        let greedy = json!({"max_tokens": 24, "temperature": 0});
        let ask = || {
            let reply = server.post("/v1/chat/completions", &server.chat(question, greedy.clone()));
            reply.json()["choices"][0]["message"]["content"].as_str().unwrap().to_owned()
        };
        same_up_to_a_near_tie("the most probable token, twice", &ask(), &ask(), 24);
    }
    server.stop();
}

#[test]
fn completions_stop_strings_and_token_limits_end_a_text_where_they_should() {
    let Some(model) = model() else { return };
    let server = serve(&model, port(1), &[]);
    let complete = |extra: Value| {
        let mut body = json!({
            "model": server.model,
            "prompt": "The first ten prime numbers are 2, 3, 5,",
            "temperature": 0,
        });
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        let reply = server.post("/v1/completions", &body);
        assert_eq!(reply.status, 200, "{}", reply.body);
        reply.json()
    };

    let whole = complete(json!({"max_tokens": 32}));
    assert_eq!(whole["object"], "text_completion");
    let text = whole["choices"][0]["text"].as_str().unwrap();
    assert_eq!(whole["usage"]["completion_tokens"], 32);
    assert_eq!(whole["choices"][0]["finish_reason"], "length");
    assert!(text.contains('7'), "the model goes on counting: {text:?}");

    // Fewer tokens are the start of the same text.
    let short = complete(json!({"max_tokens": 5}));
    let start = short["choices"][0]["text"].as_str().unwrap();
    assert_eq!(short["usage"]["completion_tokens"], 5);
    assert!(text.starts_with(start) && start.len() < text.len(), "{start:?} of {text:?}");

    // A stop string from the middle of the text ends it right before.
    let middle = text.len() / 2;
    let from = (middle..text.len()).find(|i| text.is_char_boundary(*i)).unwrap();
    let stop: String = text[from..].chars().take(3).collect();
    let cut = text.find(&stop).unwrap();
    let stopped = complete(json!({"max_tokens": 32, "stop": [stop.as_str(), "\u{1}never"]}));
    assert_eq!(stopped["choices"][0]["text"], text[..cut], "stopped before {stop:?}");
    assert_eq!(stopped["choices"][0]["finish_reason"], "stop");
    assert!(stopped["usage"]["completion_tokens"].as_u64().unwrap() <= 32);

    // The same through a stream: the held-back text never shows.
    let reply = server.post(
        "/v1/completions",
        &json!({
            "model": server.model, "prompt": "The first ten prime numbers are 2, 3, 5,",
            "temperature": 0, "max_tokens": 32, "stop": stop, "stream": true,
        }),
    );
    let (chunks, done) = reply.events();
    assert!(done);
    assert_eq!(streamed_text(&chunks), text[..cut]);
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
    server.stop();
}

#[test]
fn a_client_that_leaves_frees_its_sequence() {
    let Some(model) = model() else { return };
    let server = serve(&model, port(2), &[]);
    let pages = server.metric("pages");
    // A page is free, or the prefix cache holds what was computed on it.
    let unused = || server.metric("pages_free") + server.metric("cached_pages");
    assert_eq!(unused(), pages);
    assert_eq!(server.metric("sequences"), 0);

    // No token limit: the request would run until the context is full.
    let body = server.chat("Write a very long story about a lighthouse.", json!({"stream": true}));
    let mut stream = server.send("POST", "/v1/chat/completions", &body.to_string());
    read_events(&mut stream, 20);
    assert_eq!(server.metric("sequences"), 1);
    assert!(server.metric("pages_free") < pages, "the sequence holds pages");
    drop(stream);

    eventually("the sequence to leave the engine", || server.metric("sequences") == 0);
    assert_eq!(unused(), pages, "all pages are back, free or cached");
    assert!(server.metric("cached_pages") >= 1, "what it computed stays in the cache");
    assert_eq!(server.metric("requests_cancelled_total"), 1);
    assert_eq!(server.metric("requests_finished_total"), 0);

    // The same for a client that leaves a request that is not streamed.
    let plain = server.chat("Write a very long story about a harbour.", json!({}));
    let waiting = server.send("POST", "/v1/chat/completions", &plain.to_string());
    eventually("the request to run", || server.metric("sequences") == 1);
    drop(waiting);
    eventually("the sequence to leave the engine", || server.metric("sequences") == 0);
    assert_eq!(unused(), pages);
    assert_eq!(server.metric("requests_cancelled_total"), 2);

    // The engine is as good as new for the next request.
    let reply = server.post(
        "/v1/chat/completions",
        &server.chat(ENGLISH, json!({"max_tokens": 16, "temperature": 0})),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.json()["usage"]["completion_tokens"], 16);
    server.stop();
}

#[test]
fn two_requests_run_at_once_and_each_gets_its_own_text() {
    let Some(model) = model() else { return };
    let process = serve(&model, port(3), &["--parallel", "2"]);
    let server: &Client = &process;
    let ask = |question: &str, seed: u64| {
        let extra = json!({"max_tokens": 160, "temperature": 0.7, "seed": seed, "stream": true});
        let reply = server.post("/v1/chat/completions", &server.chat(question, extra));
        assert_eq!(reply.status, 200, "{}", reply.body);
        let (chunks, done) = reply.events();
        assert!(done);
        let finish = chunks.last().unwrap()["choices"][0]["finish_reason"].clone();
        (streamed_text(&chunks), finish, chunks.last().unwrap()["timings"].clone())
    };

    // Alone, one after the other.
    let (german, _, alone) = ask(GERMAN, 1);
    let (english, _, _) = ask(ENGLISH, 2);
    assert!(alone["draft_n"].as_u64().is_some(), "{alone}");

    // Together.
    let most = std::sync::atomic::AtomicU64::new(0);
    let (german_together, english_together) = thread::scope(|scope| {
        let first = scope.spawn(|| ask(GERMAN, 1));
        let second = scope.spawn(|| ask(ENGLISH, 2));
        while !(first.is_finished() && second.is_finished()) {
            most.fetch_max(server.metric("sequences"), std::sync::atomic::Ordering::Relaxed);
        }
        (first.join().unwrap(), second.join().unwrap())
    });
    assert_eq!(most.into_inner(), 2, "both sequences were in the engine at the same time");
    // Alone, a request read all of its prompt but the template's start; next to
    // the other one it reads the last token, the rest from the cache.
    same_up_to_a_near_tie("German next to another request", &german_together.0, &german, 24);
    same_up_to_a_near_tie("English next to another request", &english_together.0, &english, 24);
    assert_ne!(german_together.0, english_together.0, "each its own text");
    assert_eq!(german_together.1, "length");
    for timings in [&german_together.2, &english_together.2] {
        assert_eq!(timings["predicted_n"], 160);
        assert!(timings.get("draft_n").is_none(), "drafts are counted for both: {timings}");
    }
    assert_eq!(server.metric("requests_finished_total"), 4);
    assert_eq!(server.metric("sequences"), 0);
    assert_eq!(server.metric("pages_free") + server.metric("cached_pages"), server.metric("pages"));
    process.stop();
}

#[test]
fn what_the_engine_cannot_do_is_refused_by_name() {
    let Some(model) = model() else { return };
    let server = serve(&model, port(4), &[]);
    let refused = |path: &str, body: Value, status: u16, words: &[&str]| {
        let reply = server.post(path, &body);
        let error = reply.json()["error"].clone();
        assert_eq!(reply.status, status, "{body}: {}", reply.body);
        let message = error["message"].as_str().unwrap().to_owned();
        for word in words {
            assert!(message.contains(word), "{body}: '{message}' should mention '{word}'");
        }
        error
    };
    let chat = |extra: Value| server.chat("Hi", extra);
    let chat_path = "/v1/chat/completions";

    // The sampler has its limits, and a request past them is told which.
    refused(chat_path, chat(json!({"repeat_last_n": 5000})), 400, &["repeat_last_n", "1024"]);
    refused(chat_path, chat(json!({"adaptive_target": 0.5})), 400, &["adaptive_target"]);
    refused(chat_path, chat(json!({"logit_bias": {"99999999": -100}})), 400, &["logit_bias"]);
    refused(chat_path, chat(json!({"samplers": ["top_k", "penalties"]})), 400, &["penalties"]);
    refused(chat_path, chat(json!({"top_logprobs": 2})), 400, &["top_logprobs"]);
    // Nor has the server what this asks for.
    refused(chat_path, chat(json!({"n": 17})), 400, &["n must be between 1 and 16"]);
    // A grammar is served, and one that is wrong is told by name.
    refused(
        chat_path,
        chat(json!({"grammar": "root ::= nothing"})),
        400,
        &["grammar", "'nothing'"],
    );
    refused(
        chat_path,
        chat(json!({"response_format": {"type": "xml"}})),
        400,
        &["response_format", "xml"],
    );
    // Tools are served, as functions the client runs. What a request asks
    // about them beyond that is told by name: a tool of another kind, two
    // tools of one name, a choice the tools do not cover, a second rule for
    // the reply next to them, the functions of the first chat API, and tools
    // where there is no chat template.
    let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
    let tools = |extra: Value| {
        let mut body = chat(extra);
        body["tools"] = json!([tool]);
        body
    };
    refused(
        chat_path,
        chat(json!({"tools": [{"type": "web_search"}]})),
        400,
        &["tools[0].type", "\"web_search\"", "other kinds of tools are not served"],
    );
    refused(chat_path, chat(json!({"tools": [tool, tool]})), 400, &["two tools named 'f'"]);
    refused(
        chat_path,
        chat(json!({"tool_choice": "required"})),
        400,
        &["tool_choice", "the request has no tools"],
    );
    let named = json!({"type": "function", "function": {"name": "g"}});
    refused(
        chat_path,
        tools(json!({"tool_choice": named})),
        400,
        &["tool_choice", "'g'", "not among the tools"],
    );
    for (rule, value) in [
        ("grammar", json!("root ::= \"a\"")),
        ("json_schema", json!({"type": "object"})),
        ("response_format", json!({"type": "json_object"})),
    ] {
        let mut body = tools(json!({}));
        body[rule] = value;
        let error = refused(chat_path, body, 400, &["tools cannot be combined with", rule]);
        assert_eq!(error["param"], "tools");
    }
    refused(
        chat_path,
        chat(json!({"functions": [tool["function"]]})),
        400,
        &["functions", "tools"],
    );
    refused(
        "/v1/completions",
        json!({"model": server.model, "prompt": "Hi", "tools": [tool]}),
        400,
        &["tools is not supported", "/v1/completions has no chat template"],
    );
    // A budget for the thinking is served (`thinking_budget_tokens`, see the
    // routes' tests), so it is no longer among what is refused.
    let image = json!({
        "model": server.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    });
    refused(chat_path, image, 400, &["image_url", "not of a format"]);
    // A real image, to a model loaded without a projector.
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    let image = json!({
        "model": server.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png}")}},
            {"type": "text", "text": "What is this?"},
        ]}],
    });
    refused(chat_path, image, 400, &["projector", "--mmproj"]);
    let error = refused(chat_path, json!({"model": "gpt-4", "messages": []}), 400, &["messages"]);
    assert_eq!(error["param"], "messages");
    let mut other = chat(json!({}));
    other["model"] = json!("gpt-4");
    let error = refused(chat_path, other, 404, &["'gpt-4'", &server.model]);
    assert_eq!(error["code"], "model_not_found");
    // A prompt that does not fit is told so with both numbers.
    let long = "word ".repeat(9000);
    let error =
        refused("/v1/completions", json!({"model": server.model, "prompt": long}), 400, &["8192"]);
    assert_eq!(error["code"], "context_length_exceeded");
    // Decisions need more of the runtime than it has.
    let decision = json!({
        "model": server.model, "state": "s",
        "questions": {"q": {"type": "noul", "instructions": "x"}},
    });
    refused("/v1/systemone", decision, 501, &["decisions are not implemented"]);
    refused("/v1/embeddings", json!({"model": server.model, "input": "x"}), 404, &["no endpoint"]);

    // None of that reached the devices, and a neutral value is no refusal.
    assert_eq!(server.metric("steps_total"), 0);
    let quiet = chat(json!({
        "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1.0, "logit_bias": {},
        "n": 1, "max_tokens": 4, "user": "someone", "logprobs": false,
    }));
    let reply = server.post(chat_path, &quiet);
    assert_eq!(reply.status, 200, "{}", reply.body);
    server.stop();
}

#[test]
fn a_signal_lets_a_running_request_finish_and_then_stops_in_order() {
    let Some(model) = model() else { return };
    let mut server = serve(&model, port(5), &[]);
    let body = server.chat(ENGLISH, json!({"stream": true, "max_tokens": 400, "temperature": 0}));
    let mut stream = server.send("POST", "/v1/chat/completions", &body.to_string());
    let mut raw = read_events(&mut stream, 5);

    server.signal("TERM");
    server.wait_for_log("no new connections");
    // The listener is closed; the request that runs is not disturbed.
    eventually("the listener to close", || server.connect().is_err());
    stream.read_to_end(&mut raw).unwrap();
    let (chunks, done) = Reply::parse(&raw).events();
    assert!(done, "the request ran to its end");
    let finish = chunks.last().unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(finish["timings"]["predicted_n"], 400);

    let status = server.wait_exit();
    let log = server.logged().join("\n");
    assert!(status.success(), "{status:?}\n{log}");
    assert!(log.contains("released"), "the device is given back: {log}");
    assert!(log.contains("threads joined, devices released"), "{log}");
    assert!(log.contains("running=false"), "the engine's threads have ended: {log}");
    assert!(log.trim_end().ends_with("stopped"), "{log}");
}

#[test]
fn a_second_signal_ends_a_running_request_and_still_stops_in_order() {
    let Some(model) = model() else { return };
    let mut server = serve(&model, port(6), &[]);
    // No token limit: this request does not end by itself in time.
    let body = server.chat("Write a very long story about a river.", json!({"stream": true}));
    let mut stream = server.send("POST", "/v1/chat/completions", &body.to_string());
    let mut raw = read_events(&mut stream, 5);

    server.signal("INT");
    server.wait_for_log("no new connections");
    thread::sleep(Duration::from_millis(300));
    assert!(server.exited().is_none(), "the first signal waits for the request");
    server.signal("INT");

    // The client is told, and the stream does not look complete.
    stream.read_to_end(&mut raw).unwrap();
    let (chunks, done) = Reply::parse(&raw).events();
    assert!(!done, "no [DONE] after a cut");
    assert_eq!(chunks.last().unwrap()["error"]["code"], "cancelled");
    assert!(streamed_text(&chunks).len() > 10, "the text up to the cut arrived");

    let status = server.wait_exit();
    let log = server.logged().join("\n");
    assert!(status.success(), "a stop that was asked for twice is still a clean one: {log}");
    assert!(log.contains("the server is stopping: ending the request"), "{log}");
    assert!(log.contains("threads joined, devices released"), "{log}");
    assert!(log.contains("running=false"), "{log}");
}

/// The sentences of a text that a model copies without a doubt: no two are
/// alike.
fn passage(sentences: usize) -> String {
    const THINGS: [&str; 8] = [
        "lanterns",
        "marshes",
        "apples",
        "weather vanes",
        "letters",
        "paper boats",
        "keys",
        "timetables",
    ];
    const PEOPLE: [&str; 8] = [
        "The harbour master",
        "A courier",
        "Our engineer",
        "The baker",
        "Her grandmother",
        "The clockmaker",
        "A botanist",
        "The watchman",
    ];
    (0..sentences)
        .map(|i| {
            format!(
                "{} counted {} {} on day {}.",
                PEOPLE[i * 5 % 8],
                11 + i * 7 % 90,
                THINGS[i * 3 % 8],
                i + 1
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Two turns of a conversation on `server`: the first message to copy a text,
/// then its answer and a second message. Returns the usage, the timings and
/// the answer of each turn.
fn two_turns(server: &Client) -> [(Value, Value, String); 2] {
    let first = format!("{}\n\nCopy the text above word for word.", passage(60));
    let ask = |messages: Value| {
        let body = json!({
            "model": server.model, "messages": messages, "max_tokens": 48, "temperature": 0,
        });
        let reply = server.post("/v1/chat/completions", &body);
        assert_eq!(reply.status, 200, "{}", reply.body);
        let reply = reply.json();
        let text = reply["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
        (reply["usage"].clone(), reply["timings"].clone(), text)
    };
    let one = ask(json!([{"role": "user", "content": first}]));
    let two = ask(json!([
        {"role": "user", "content": first},
        {"role": "assistant", "content": one.2},
        {"role": "user", "content": "Now copy the first sentence again."},
    ]));
    [one, two]
}

#[test]
fn a_second_turn_starts_from_what_the_first_turn_computed_and_says_so() {
    let Some(model) = model() else { return };
    let mut server = serve(&model, port(7), &["--chat-var", "enable_thinking=false"]);
    let [(usage, timings, answer), (usage2, timings2, answer2)] = two_turns(&server);

    // The first turn computes its whole prompt.
    let prompt = usage["prompt_tokens"].as_u64().unwrap();
    assert!(prompt > 5 * 128, "five pages and a bit: {usage}");
    let first_cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(first_cached < 16, "nothing but the start of the template: {usage}");
    assert_eq!(timings["prompt_n"], prompt - first_cached);

    // The second turn reads the first one's pages and goes on from its
    // checkpoint: the whole pages of the first prompt at least.
    let prompt2 = usage2["prompt_tokens"].as_u64().unwrap();
    let cached = usage2["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(prompt2 > prompt, "the second prompt holds the first, the answer and a question");
    assert!(cached >= prompt / 128 * 128, "{usage2} after a prompt of {prompt} tokens");
    assert_eq!(timings2["prompt_n"], prompt2 - cached, "the cached part is not computed");
    eprintln!(
        "RES second turn: {prompt2} prompt tokens, {cached} from the cache, {} computed",
        prompt2 - cached
    );
    assert!(answer.contains("harbour master"), "a copy of the text: {answer:?}");
    assert!(answer2.chars().count() > 10, "an answer: {answer2:?}");

    // The counters say the same.
    assert_eq!(server.metric("prefix_hits_total"), 1 + u64::from(first_cached > 0));
    assert!(server.metric("cached_cells_total") >= cached);
    assert!(server.metric("checkpoints_saved_total") >= 1);
    assert_eq!(server.metric("checkpoints_restored_total"), 1);
    assert!(server.metric("replayed_rows_total") < prompt2, "not everything is read again");
    let held = server.metric("checkpoint_ram_bytes");
    assert!(held > 0 && held < 1 << 30, "page-locked memory for the checkpoints: {held} bytes");
    assert!(server.logged().iter().any(|l| l.contains("plan: RAM")), "{:#?}", server.logged());
    server.stop();
}

/// The server started as [`serve`] starts it, with `ram` as its RAM limit,
/// which it refuses: its exit status and what it said. The port is named so
/// that a server that starts after all takes no other, and such a server is
/// killed.
fn refused_ram(model: &PathBuf, port: u16, flags: &[&str], ram: &str) -> (ExitStatus, String) {
    let device = common::test_device().to_string();
    let mut command = ares();
    command
        .arg("--model")
        .arg(model)
        .args(["--port", &port.to_string(), "--devices", &device])
        .args(["--context", "8192", "--ram", ram])
        .args(flags);
    let mut server = Server::spawn(command, LIMIT);
    let status = server.wait_exit();
    (status, server.logged().join("\n"))
}

/// A server whose `--ram` is exactly what the engine needs: the step buffers,
/// the token embedding table and the one checkpoint place the prompt cache
/// keeps whatever else the limit and the machine leave. A limit below that is
/// refused with the shares named; at that limit checkpoints are taken in that
/// place, and a second turn answers as a server with room does. (The place
/// holds the newest checkpoint, the end of the first turn; this model's
/// template writes the reply into the second prompt otherwise than it was
/// generated, so the second turn parts from the first before that end and
/// replays the cached rows.)
#[test]
fn a_server_with_no_ram_to_spare_keeps_one_checkpoint_and_answers_as_one_with() {
    let Some(model) = model() else { return };
    let flags = ["--chat-var", "enable_thinking=false"];
    let with = serve(&model, port(7), &flags);
    let [_, with_second] = two_turns(&with);
    with.stop();

    // To run at all the engine page-locks its step buffers and the token
    // embedding table, which lies in host memory, and the prompt cache keeps
    // one checkpoint place. A limit below that is refused, and the refusal
    // names the shares and their sum in bytes.
    let (status, said) = refused_ram(&model, port(8), &flags, "4M");
    let refusal = said.lines().find(|l| l.contains("--ram 4.0 MiB is less than")).unwrap_or(&said);
    assert!(
        !status.success()
            && refusal.contains("for its step buffers")
            && refusal.contains("for the token embedding table")
            && refusal.contains("one checkpoint of recurrent state"),
        "{said}"
    );
    let required: u64 = refusal
        .split_once(" bytes)")
        .and_then(|(before, _)| before.rsplit_once('('))
        .and_then(|(_, bytes)| bytes.parse().ok())
        .unwrap_or_else(|| panic!("the refusal names what is needed in bytes: {refusal}"));
    assert!(required > 4 << 20, "{refusal}");
    // One byte less than that is refused as well.
    let (status, said) = refused_ram(&model, port(8), &flags, &(required - 1).to_string());
    assert!(!status.success() && said.contains("is less than"), "{said}");

    // Exactly that much is a server with no RAM to spare: the limit leaves
    // the prompt cache its one checkpoint place (the 0.8B's is 20 MB) and
    // nothing more.
    let required = required.to_string();
    let mut few = Vec::from(flags);
    few.extend(["--ram", &required]);
    let mut without = serve(&model, port(8), &few);
    assert!(
        without.logged().iter().any(|l| l.contains("plan: RAM")
            && l.contains("as asked")
            && l.contains("for the token embedding table")
            && l.contains("room for 1")
            && l.contains("one checkpoint place")),
        "{:#?}",
        without.logged()
    );
    let [_, without_second] = two_turns(&without);
    let cached = without_second.0["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(cached >= 5 * 128, "the pages are cached all the same: {}", without_second.0);
    eprintln!(
        "RES no RAM to spare: {} checkpoints saved, {} restored, {} bytes locked for them; {} of \
         {cached} cached rows replayed",
        without.metric("checkpoints_saved_total"),
        without.metric("checkpoints_restored_total"),
        without.metric("checkpoint_ram_bytes"),
        without.metric("replayed_rows_total"),
    );
    assert!(without.metric("checkpoints_saved_total") >= 1, "checkpoints are taken in the place");
    assert!(without.metric("checkpoint_ram_bytes") > 0, "the place holds one");
    assert_eq!(without_second.2, with_second.2, "the second answer, with and without checkpoints");
    without.stop();
}

/// The server run with `flags` on the small model, which is refused before
/// any device is touched: its exit status and what it said.
fn refused(model: &PathBuf, flags: &[&str]) -> (ExitStatus, String) {
    let output = ares().arg("--model").arg(model).args(flags).output().unwrap();
    (output.status, String::from_utf8_lossy(&output.stderr).into_owned())
}

#[test]
fn the_choice_of_drafters_is_refused_by_name_before_a_device_is_touched() {
    let Some(model) = std::env::var_os("ARES_TEST_QWEN35_0_8B").map(PathBuf::from) else {
        eprintln!("skipped: set ARES_TEST_QWEN35_0_8B to a small qwen35 model file");
        return;
    };
    // A drafter from a file that is not there is refused by its name.
    let (status, said) = refused(&model, &["--drafters", "/models/dflash2.gguf"]);
    assert!(!status.success() && said.contains("/models/dflash2.gguf"), "{said}");
    // The small model has no MTP block of its own.
    let (status, said) = refused(&model, &["--drafters", "mtp"]);
    assert!(!status.success() && said.contains("no MTP block"), "{said}");
    // A window, or a vocabulary, for no drafter does nothing.
    let (status, said) = refused(&model, &["--drafters", "none", "--window", "8"]);
    assert!(!status.success() && said.contains("without a model drafter"), "{said}");
    let (status, said) = refused(&model, &["--drafters", "none", "--draft-vocab", "v.txt"]);
    assert!(!status.success() && said.contains("no drafter of that kind"), "{said}");
}

/// The text of the first choice of an answer, and the ids of its tokens with
/// the probabilities (`logprob`) the answer reports.
fn first_choice(reply: &Reply) -> (String, Vec<Value>) {
    assert_eq!(reply.status, 200, "{}", reply.body);
    let value = reply.json();
    let choice = &value["choices"][0];
    let tokens = choice["logprobs"]["content"].as_array().cloned().unwrap_or_default();
    (choice["message"]["content"].as_str().unwrap_or_default().to_owned(), tokens)
}

#[test]
fn the_sampler_serves_what_a_request_sets_and_reports_the_probabilities() {
    let Some(model) = model() else { return };
    // The 0.8B thinks first unless told not to: the answers are the reply.
    let server = serve(&model, port(9), &["--chat-var", "enable_thinking=false"]);
    let chat_path = "/v1/chat/completions";
    let question = "Name three colours.";
    let greedy = |extra: Value| {
        let mut body = server.chat(question, json!({"temperature": 0, "max_tokens": 16}));
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        body
    };

    // The probabilities of a greedy answer: the token with the best
    // probability is the one that came out, and the alternatives are in order.
    let (plain, tokens) = first_choice(
        &server.post(chat_path, &greedy(json!({"logprobs": true, "top_logprobs": 5}))),
    );
    assert!(tokens.len() >= 8, "{tokens:?}");
    for token in &tokens {
        let top = token["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 5);
        assert_eq!(top[0]["id"], token["id"], "the greedy token is the most probable");
        assert_eq!(top[0]["logprob"], token["logprob"]);
        let lp: Vec<f64> = top.iter().map(|t| t["logprob"].as_f64().unwrap()).collect();
        assert!(lp.windows(2).all(|w| w[0] >= w[1]) && lp[0] <= 0.0);
        assert!(lp.iter().map(|l| l.exp()).sum::<f64>() <= 1.0 + 1e-4);
    }
    // After the chain a greedy row has one token left, with probability 1.
    let (_, after) = first_choice(
        &server.post(chat_path, &greedy(json!({"n_probs": 3, "post_sampling_probs": true}))),
    );
    assert!(
        after.iter().all(|t| t["prob"] == 1.0 && t["top_probs"].as_array().unwrap().len() == 1)
    );

    // A banned token is not picked, and the best of the rest is: the first
    // token of the answer becomes the second best of the first answer.
    let first = &tokens[0];
    let second = first["top_logprobs"][1]["id"].clone();
    let (banned, banned_tokens) =
        first_choice(&server.post(
            chat_path,
            &greedy(json!({"logit_bias": [[first["id"], false]], "logprobs": true})),
        ));
    assert_eq!(banned_tokens[0]["id"], second, "the answer {plain:?} then {banned:?}");
    // The probabilities are the model's own, so they do not move with the ban
    // (the prompt came from the cache the second time, which rounds a little
    // differently).
    let (a, b) = (
        banned_tokens[0]["logprob"].as_f64().unwrap(),
        first["top_logprobs"][1]["logprob"].as_f64().unwrap(),
    );
    assert!((a - b).abs() < 0.05, "{a} against {b}");

    // A penalty and a seed: the same request says the same twice, and
    // a strong penalty says something else than none.
    let penalised = |seed: u64| {
        let body = greedy(json!({
            "temperature": 0.7, "seed": seed, "repeat_penalty": 1.5, "frequency_penalty": 0.8,
            "max_tokens": 40,
        }));
        first_choice(&server.post(chat_path, &body)).0
    };
    assert_eq!(penalised(5), penalised(5));
    let free = first_choice(
        &server.post(chat_path, &greedy(json!({"temperature": 0.7, "seed": 5, "max_tokens": 40}))),
    )
    .0;
    assert_ne!(penalised(5), free);

    // Every other sampler answers.
    for extra in [
        json!({"mirostat": 2, "mirostat_tau": 3.0}),
        json!({"mirostat": 1}),
        json!({"typical_p": 0.9, "top_n_sigma": 3.0}),
        json!({"xtc_probability": 0.5, "xtc_threshold": 0.2}),
        json!({"dynatemp_range": 0.5, "dynatemp_exponent": 1.5}),
        json!({"dry_multiplier": 0.8, "dry_penalty_last_n": 64}),
        json!({"samplers": ["temperature", "min_p", "top_p"], "min_p": 0.05, "top_p": 0.9}),
        json!({"samplers": "ktpmt", "top_k": 40}),
        json!({"ignore_eos": true, "logit_bias": {"Hello": 1.5}}),
    ] {
        let mut body = greedy(extra.clone());
        body["temperature"] = json!(0.8);
        body["seed"] = json!(3);
        let (text, _) = first_choice(&server.post(chat_path, &body));
        assert!(!text.is_empty(), "{extra}");
    }

    // The completions of one prompt are requests of their own with their own
    // seeds, and the engine reads the prompt once.
    let steps = server.metric("prefix_hits_total");
    let mut body = greedy(json!({"temperature": 0.9, "seed": 40, "n": 3, "max_tokens": 24}));
    body["messages"][0]["content"] = json!("Tell me a very short story about a lighthouse keeper.");
    let reply = server.post(chat_path, &body);
    assert_eq!(reply.status, 200, "{}", reply.body);
    let choices = reply.json()["choices"].as_array().unwrap().clone();
    assert_eq!(choices.len(), 3);
    let texts: Vec<&str> =
        choices.iter().map(|c| c["message"]["content"].as_str().unwrap()).collect();
    assert!(texts[0] != texts[1] || texts[1] != texts[2], "{texts:?}");
    for (i, text) in texts.iter().enumerate() {
        let mut one = body.clone();
        one["n"] = json!(1);
        one["seed"] = json!(40 + i as u64);
        // The text the seed gives, up to the model's rounding: the choices ran
        // side by side, so a sampled text can part from the one it has alone
        // where two tokens are that close.
        let alone = first_choice(&server.post(chat_path, &one)).0;
        let same = text.chars().zip(alone.chars()).take_while(|(a, b)| a == b).count();
        eprintln!("RES n choice {i}: {same} of {} characters equal the request alone", text.len());
        assert!(same >= 10, "choice {i}: {text:?} against {alone:?}");
    }
    assert!(
        server.metric("prefix_hits_total") >= steps + 2,
        "the other completions read the cache"
    );
    server.stop();
}
