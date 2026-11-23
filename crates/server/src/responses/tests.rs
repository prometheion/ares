use std::collections::VecDeque;

use ares_api::{Json, Usage};
use serde_json::{Value, json};

use super::reply::{Change, Ending, Head, Output, change_events};
use super::request::{Echo, renamed, to_chat};
use crate::answer::Written;
use crate::bridge::Timings;
use crate::chat_body::Purpose;
use crate::error::ApiError;
use crate::reasoning::Split;
use crate::reply::Piece;

fn read(body: Value) -> Json {
    serde_json::from_value(body).unwrap()
}

/// The chat body a Responses body comes to, and what the response says back.
fn converted(body: Value) -> (Value, Echo) {
    let (chat, echo) =
        to_chat(&read(body), Purpose::Reply).unwrap_or_else(|e| panic!("{}", e.message));
    (serde_json::to_value(chat).unwrap(), echo)
}

fn chat(body: Value) -> Value {
    converted(body).0
}

fn with(extra: Value) -> Value {
    let mut body = json!({"model": "m", "input": "hi"});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

/// The messages of the chat body the items of an input come to.
fn messages(input: Value) -> Value {
    chat(json!({"model": "m", "input": input}))["messages"].clone()
}

/// The message of the refusal of a body, and the field it names.
fn refused(body: Value) -> (String, String) {
    let error = to_chat(&read(body.clone()), Purpose::Reply)
        .err()
        .unwrap_or_else(|| panic!("not refused: {body}"));
    assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
    (error.param.unwrap_or_default(), error.message)
}

fn weather() -> Value {
    json!({
        "type": "function",
        "name": "get_weather",
        "description": "The weather of a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "strict": true,
    })
}

// --- the request ---------------------------------------------------------------

#[test]
fn the_plain_fields_go_over_under_the_names_of_the_chat_body() {
    let body = with(json!({
        "instructions": "Be brief.",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_output_tokens": 64,
        "stream": true,
        "store": false,
        "background": false,
        "truncation": "disabled",
        "include": ["reasoning.encrypted_content"],
        "stream_options": {"include_obfuscation": false},
        "metadata": {"run": "7"},
        "client_metadata": {"x-codex-turn": "1"},
        "user": "u1",
        "safety_identifier": "s",
        "service_tier": "auto",
        "prompt_cache_key": "k",
        "prompt_cache_retention": "24h",
        "prompt_cache_options": {"retention": "24h"},
        "previous_response_id": null,
    }));
    let (chat, echo) = converted(body);
    assert_eq!(
        chat,
        json!({
            "model": "m",
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 64,
            "stream": true,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
        })
    );
    assert_eq!(
        serde_json::to_value([
            &echo.instructions,
            &echo.max_output_tokens,
            &echo.metadata,
            &echo.effort,
            &echo.temperature,
            &echo.format,
            &echo.tool_choice,
            &echo.tools,
            &echo.top_p,
            &echo.user,
        ])
        .unwrap(),
        json!(["Be brief.", 64, {"run": "7"}, null, 0.5, {"type": "text"}, "auto", [], 0.9, "u1"])
    );
    assert!(echo.parallel_tool_calls);

    // What a request leaves out is said back as nothing.
    let (_, echo) = converted(with(json!({})));
    assert_eq!(
        (&echo.instructions, &echo.temperature, &echo.user),
        (&Json::Null, &Json::Null, &Json::Null)
    );
    assert_eq!(echo.metadata, Json::Object(Vec::new()));
}

#[test]
fn a_count_has_no_stream() {
    let (chat, _) = to_chat(&read(with(json!({"stream": true}))), Purpose::Count).unwrap();
    assert_eq!(chat.get("stream"), None);
    assert!(to_chat(&read(with(json!({"stream": "yes"}))), Purpose::Count).is_err());
}

#[test]
fn messages_are_the_messages_of_the_chat_body_and_their_parts_one_text() {
    assert_eq!(
        messages(json!([
            {"role": "user", "content": "One."},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Two, "},
                {"type": "input_text", "text": "three."},
            ]},
            {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed", "content": [
                {"type": "output_text", "text": "Four.", "annotations": [], "logprobs": []},
            ]},
            {"role": "assistant", "content": "Five."},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "No."}]},
            {"role": "user", "content": []},
        ])),
        json!([
            {"role": "user", "content": "One."},
            {"role": "user", "content": "Two, three."},
            {"role": "assistant", "content": "Four."},
            {"role": "assistant", "content": "Five."},
            {"role": "assistant", "content": "No."},
            {"role": "user", "content": ""},
        ])
    );
}

#[test]
fn an_image_of_the_user_makes_the_message_parts_of_the_chat_body() {
    let url = "data:image/png;base64,AA==";
    assert_eq!(
        messages(json!([
            {"role": "user", "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": url, "detail": "auto"},
            ]},
        ])),
        json!([
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": url}},
            ]},
        ])
    );
}

#[test]
fn the_instructions_before_a_conversation_are_one_system_message() {
    // Codex puts its instructions at the head of the input; the template of
    // Qwen3.5-0.8B takes a system message only as the first message.
    let body = json!({
        "model": "m",
        "instructions": "Be brief.",
        "input": [
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": "You are a coding agent."},
            ]},
            {"role": "system", "content": "Sandbox: read-only."},
            {"role": "developer", "content": ""},
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "The model changed."},
            {"role": "user", "content": "go on"},
        ],
    });
    assert_eq!(
        chat(body)["messages"],
        json!([
            {
                "role": "system",
                "content": "Be brief.\n\nYou are a coding agent.\n\nSandbox: read-only.",
            },
            {"role": "user", "content": "hi"},
            // Instructions inside a conversation stay where they stand.
            {"role": "developer", "content": "The model changed."},
            {"role": "user", "content": "go on"},
        ])
    );
    // Instructions alone are a conversation too.
    assert_eq!(
        chat(json!({"model": "m", "instructions": "Say hi.", "input": []}))["messages"],
        json!([{"role": "system", "content": "Say hi."}])
    );
}

#[test]
fn the_items_of_a_turn_are_one_message_of_the_assistant() {
    assert_eq!(
        messages(json!([
            {"role": "user", "content": "Weather in Köln and Bonn?"},
            {"type": "reasoning", "id": "rs_1", "summary": [], "status": "completed", "content": [
                {"type": "reasoning_text", "text": "Two cities, two calls."},
            ]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "I look it up."},
            ]},
            {"type": "function_call", "id": "fc_1", "status": "completed", "call_id": "call_a",
             "name": "get_weather", "arguments": "{\"city\":\"Köln\"}"},
            {"type": "function_call", "call_id": "call_b", "name": "get_weather",
             "arguments": "{\"city\":\"Bonn\"}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "12 °C"},
            {"type": "function_call_output", "id": "fco_1", "status": "completed",
             "call_id": "call_b", "name": "get_weather", "output": [
                {"type": "input_text", "text": "11 "},
                {"type": "input_text", "text": "°C"},
            ]},
            // The next turn: its reasoning begins another message.
            {"type": "reasoning", "summary": [], "content": [
                {"type": "reasoning_text", "text": "Both are mild."},
            ]},
            {"type": "function_call", "call_id": "call_c", "name": "now", "arguments": ""},
            {"type": "function_call_output", "call_id": "call_c", "output": ""},
        ])),
        json!([
            {"role": "user", "content": "Weather in Köln and Bonn?"},
            {
                "role": "assistant",
                "content": "I look it up.",
                "reasoning_content": "Two cities, two calls.",
                "tool_calls": [
                    {"id": "call_a", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Köln\"}"}},
                    {"id": "call_b", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Bonn\"}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "12 °C"},
            {"role": "tool", "tool_call_id": "call_b", "content": "11 °C", "name": "get_weather"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Both are mild.",
                "tool_calls": [
                    {"id": "call_c", "type": "function",
                     "function": {"name": "now", "arguments": ""}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_c", "content": ""},
        ])
    );
}

#[test]
fn an_item_that_does_not_continue_a_turn_begins_another_message() {
    assert_eq!(
        messages(json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"type": "function_call", "call_id": "c1", "name": "now", "arguments": "{}"},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "r"}]},
            {"type": "reasoning", "summary": [], "content": [{"type": "text", "text": "s"}]},
            {"role": "assistant", "content": "c"},
        ])),
        json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "now", "arguments": "{}"}},
            ]},
            // Two reasoning items of one turn are its reasoning, line by line.
            {"role": "assistant", "content": "c", "reasoning_content": "r\ns"},
        ])
    );
}

#[test]
fn the_reasoning_of_an_item_is_its_content_and_nothing_else_of_it() {
    // A summary is not the reasoning, and encrypted content is text only
    // the service that wrote it can read.
    assert_eq!(
        messages(json!([
            {"role": "user", "content": "q"},
            {"type": "reasoning", "id": "rs_x", "encrypted_content": "gAAAAAB...",
             "summary": [{"type": "summary_text", "text": "Thought about it."}]},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q"},
            {"type": "reasoning", "summary": [], "content": null, "encrypted_content": null},
            {"type": "reasoning", "summary": [], "encrypted_content": "",
             "content": [{"type": "reasoning_text", "text": " kept as it is \n"}]},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q"},
            // A reply the limit cut inside its thinking, sent back.
            {"type": "reasoning", "summary": [], "status": "incomplete",
             "content": [{"type": "reasoning_text", "text": "cut"}]},
            {"role": "user", "content": "go on"},
        ])),
        json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning_content": " kept as it is \n"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "reasoning_content": "cut"},
            {"role": "user", "content": "go on"},
        ])
    );
}

#[test]
fn tools_and_the_choice_among_them_are_those_of_the_chat_body() {
    let now = json!({"type": "function", "name": "now", "parameters": null, "strict": null});
    let body = with(json!({
        "tools": [weather(), now],
        "tool_choice": {"type": "function", "name": "now"},
        "parallel_tool_calls": false,
    }));
    let (chat, echo) = converted(body.clone());
    assert_eq!(
        chat["tools"],
        json!([
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "The weather of a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }},
            {"type": "function", "function": {"name": "now"}},
        ])
    );
    assert_eq!(chat["tool_choice"], json!({"type": "function", "function": {"name": "now"}}));
    assert_eq!(chat["parallel_tool_calls"], json!(false));
    // The response says them back as the request wrote them.
    assert_eq!(serde_json::to_value(&echo.tools).unwrap(), body["tools"]);
    assert_eq!(serde_json::to_value(&echo.tool_choice).unwrap(), body["tool_choice"]);
    assert!(!echo.parallel_tool_calls);

    for word in ["auto", "none", "required"] {
        let chat = self::chat(with(json!({"tools": [weather()], "tool_choice": word})));
        assert_eq!(chat["tool_choice"], json!(word));
    }
    // An empty list is no tools, and Codex's "auto" without tools is fine.
    let chat = self::chat(with(json!({"tools": [], "tool_choice": "auto"})));
    assert_eq!((chat.get("tools"), &chat["tool_choice"]), (None, &json!("auto")));
}

#[test]
fn reasoning_is_the_effort_of_the_chat_body_and_text_its_response_format() {
    let body = with(json!({
        "reasoning": {"effort": "low", "summary": "auto"},
        "text": {"format": {
            "type": "json_schema",
            "name": "answer",
            "description": "An answer.",
            "strict": true,
            "schema": {"type": "object", "properties": {"n": {"type": "integer"}}},
        }},
    }));
    let (chat, echo) = converted(body.clone());
    assert_eq!(chat["reasoning_effort"], json!("low"));
    assert_eq!(
        chat["response_format"],
        json!({"type": "json_schema", "json_schema": {
            "schema": {"type": "object", "properties": {"n": {"type": "integer"}}},
        }})
    );
    assert_eq!(serde_json::to_value(&echo.effort).unwrap(), json!("low"));
    assert_eq!(serde_json::to_value(&echo.format).unwrap(), body["text"]["format"]);

    let chat = self::chat(with(json!({"text": {"format": {"type": "json_object"}}})));
    assert_eq!(chat["response_format"], json!({"type": "json_object"}));
    for nothing in [
        json!({"text": {"format": {"type": "text"}}}),
        json!({"text": {}}),
        json!({"text": {"format": null, "verbosity": null}}),
        // What Codex sends when no effort is set.
        json!({"reasoning": null}),
        json!({"reasoning": {}}),
    ] {
        let chat = self::chat(with(nothing));
        assert_eq!((chat.get("response_format"), chat.get("reasoning_effort")), (None, None));
    }
    // A format next to tools that are only in the prompt is served.
    let chat = self::chat(with(json!({
        "tools": [weather()],
        "tool_choice": "none",
        "text": {"format": {"type": "json_object"}},
    })));
    assert_eq!(chat["response_format"], json!({"type": "json_object"}));
}

#[test]
fn what_is_not_served_is_refused_by_its_name() {
    let item = |item: Value| json!({"model": "m", "input": [item]});
    let part = |part: Value| item(json!({"role": "user", "content": [part]}));
    let cases: Vec<(Value, &str, &str)> = vec![
        // State.
        (with(json!({"store": true})), "store", "keeps no responses"),
        (with(json!({"previous_response_id": "resp_1"})), "previous_response_id", "keeps no"),
        (with(json!({"conversation": "conv_1"})), "conversation", "no conversations"),
        (with(json!({"conversation": {"id": "conv_1"}})), "conversation", "no conversations"),
        (with(json!({"background": true})), "background", "none is kept"),
        (item(json!({"type": "item_reference", "id": "msg_1"})), "input[0]", "item_reference"),
        (with(json!({"prompt": {"id": "pmpt_1"}})), "prompt", "templates"),
        (with(json!({"truncation": "middle"})), "truncation", "must be"),
        // What the response would have to hold.
        (
            with(json!({"include": ["message.output_text.logprobs"]})),
            "include",
            "message.output_text.logprobs",
        ),
        (
            with(json!({"include": ["reasoning.encrypted_content", "file_search_call.results"]})),
            "include",
            "runs no tools",
        ),
        (with(json!({"include": ["everything"]})), "include", "everything"),
        (with(json!({"include": "reasoning.encrypted_content"})), "include", "an array"),
        (with(json!({"top_logprobs": 3})), "top_logprobs", "probabilities"),
        (
            with(json!({"stream_options": {"include_obfuscation": true}})),
            "stream_options",
            "padding",
        ),
        (with(json!({"stream_options": {"include_usage": true}})), "stream_options", "usage"),
        (with(json!({"reasoning": {"summary": "detailed"}})), "reasoning", "no summaries"),
        (with(json!({"reasoning": {"summary": "short"}})), "reasoning", "must be"),
        (with(json!({"reasoning": {"generate_summary": "auto"}})), "reasoning", "generate_summary"),
        (with(json!({"reasoning": {"effort": 3}})), "reasoning", "a string"),
        (with(json!({"text": {"verbosity": "low"}})), "text", "verbosity"),
        (with(json!({"text": {"format": {"type": "json_schema"}}})), "text.format", "schema"),
        (with(json!({"text": {"format": {"type": "xml"}}})), "text.format", "xml"),
        (
            with(json!({"text": {"format": {"type": "json_object", "schema": {}}}})),
            "text.format",
            "json_schema",
        ),
        (
            with(json!({"tools": [weather()], "text": {"format": {"type": "json_object"}}})),
            "text.format",
            "one rule",
        ),
        (with(json!({"max_output_tokens": 0})), "max_output_tokens", "at least 1"),
        (with(json!({"max_tool_calls": 3})), "max_tool_calls", "no tools of its own"),
        (with(json!({"prompt_cache_options": "24h"})), "prompt_cache_options", "an object"),
        (with(json!({"metadata": "x"})), "metadata", "an object"),
        (with(json!({"user": 7})), "user", "a string"),
        // Members of other APIs, and the sampler's fields of the chat route.
        (with(json!({"messages": []})), "messages", "input"),
        (with(json!({"max_tokens": 5})), "max_tokens", "max_output_tokens"),
        (with(json!({"response_format": {}})), "response_format", "text.format"),
        (with(json!({"reasoning_effort": "low"})), "reasoning_effort", "reasoning.effort"),
        (with(json!({"top_k": 40})), "top_k", "not a parameter of this route"),
        (with(json!({"seed": 1})), "seed", "not a parameter"),
        (with(json!({"stop": ["x"]})), "stop", "not a parameter"),
        // Tools that are not functions.
        (with(json!({"tools": [{"type": "web_search"}]})), "tools", "web_search"),
        (with(json!({"tools": [{"type": "file_search", "vector_store_ids": []}]})), "tools", "run"),
        (with(json!({"tools": [{"type": "code_interpreter"}]})), "tools", "code_interpreter"),
        (with(json!({"tools": [{"type": "computer_use_preview"}]})), "tools", "computer_use"),
        (with(json!({"tools": [{"type": "mcp", "server_label": "x"}]})), "tools", "mcp"),
        (with(json!({"tools": [{"type": "local_shell"}]})), "tools", "local_shell"),
        (with(json!({"tools": [{"type": "custom", "name": "patch"}]})), "tools", "free text"),
        (with(json!({"tools": [{"name": "now"}]})), "tools", "type is required"),
        (with(json!({"tools": [{"type": "function"}]})), "tools", "name"),
        (
            with(json!({"tools": [{"type": "function", "name": "f", "defer_loading": true}]})),
            "tools",
            "defer_loading",
        ),
        (
            with(json!({"tools": [{"type": "function", "function": {"name": "f"}}]})),
            "tools",
            "tools[0].function",
        ),
        (
            with(json!({"tools": [weather()], "tool_choice": {"type": "web_search"}})),
            "tool_choice",
            "web_search",
        ),
        (
            with(json!({"tools": [weather()], "tool_choice": {
                "type": "allowed_tools", "mode": "auto", "tools": [],
            }})),
            "tool_choice",
            "subset",
        ),
        (
            with(json!({"tools": [weather()], "tool_choice": {
                "type": "function", "function": {"name": "get_weather"},
            }})),
            "tool_choice",
            "tool_choice.function",
        ),
        // Input that is neither text nor an image, an image that is not
        // sent, and items of tools the server does not run.
        (
            part(json!({"type": "input_image", "file_id": "f"})),
            "input[0].content[0]",
            "files are not kept",
        ),
        (
            part(json!({"type": "input_image", "image_url": "data:", "detail": "max"})),
            "input[0].content[0]",
            "detail",
        ),
        (part(json!({"type": "input_file", "file_id": "f"})), "input[0].content[0]", "files"),
        (part(json!({"type": "input_audio"})), "input[0].content[0]", "audio"),
        (part(json!({"type": "refusal", "refusal": "No."})), "input[0].content[0]", "refusal"),
        (part(json!({"type": "input_text"})), "input[0].content[0]", "text is required"),
        (item(json!({"type": "web_search_call", "id": "ws_1"})), "input[0]", "no tools of its own"),
        (item(json!({"type": "custom_tool_call", "call_id": "c"})), "input[0]", "custom tools"),
        (item(json!({"type": "compaction", "encrypted_content": "x"})), "input[0]", "OpenAI"),
        (item(json!({"type": "hologram"})), "input[0]", "hologram"),
        (item(json!({"content": "x"})), "input[0]", "type is required"),
        (item(json!("hi")), "input[0]", "an object"),
        (item(json!({"role": "tool", "content": "x"})), "input[0].role", "must be"),
        (item(json!({"role": "user"})), "input[0].content", "required"),
        (item(json!({"role": "user", "content": "x", "phase": "commentary"})), "input[0]", "phase"),
        (
            item(json!({"type": "function_call", "call_id": "c", "name": "f", "arguments": "[1]"})),
            "input[0]",
            "JSON object",
        ),
        (
            item(json!({"type": "function_call", "call_id": "c", "name": "f", "arguments": {}})),
            "input[0]",
            "a string",
        ),
        (
            item(json!({"type": "function_call", "name": "f", "arguments": "{}"})),
            "input[0]",
            "call_id is required",
        ),
        (
            item(json!({"type": "function_call_output", "call_id": "c"})),
            "input[0]",
            "output is required",
        ),
        (
            item(json!({"type": "function_call_output", "call_id": "c", "output": [
                {"type": "input_image", "image_url": "data:"},
            ]})),
            "input[0].output[0]",
            "an image stands in a message of the user",
        ),
        (
            item(json!({"type": "reasoning", "summary": [], "content": [{"type": "image"}]})),
            "input[0].content[0]",
            "image",
        ),
        (item(json!({"type": "reasoning", "summary": "none"})), "input[0]", "summary"),
        (json!({"model": "m"}), "input", "required"),
        (json!({"model": "m", "input": []}), "input", "empty"),
        (json!({"model": "m", "input": 7}), "input", "a string or an array"),
        (with(json!({"instructions": ["a"]})), "instructions", "a string"),
    ];
    for (body, param, word) in cases {
        let (named, message) = refused(body.clone());
        assert_eq!(named, param, "{body}: {message}");
        assert!(message.contains(word), "{body}: {message}");
    }
}

#[test]
fn an_error_of_the_chat_reader_has_the_name_of_this_api() {
    let error = renamed(ApiError::invalid_param(
        "reasoning_effort",
        "reasoning_effort is not supported: the chat template of this model does not use it",
    ));
    assert_eq!(error.param.as_deref(), Some("reasoning.effort"));
    assert!(error.message.starts_with("reasoning.effort is not supported"), "{}", error.message);
    let error = renamed(ApiError::invalid_param("model", "model is required"));
    assert_eq!(
        (error.param.as_deref(), error.message.as_str()),
        (Some("model"), "model is required")
    );
}

// --- the reply -----------------------------------------------------------------

fn text(reasoning: &str, content: &str) -> Piece {
    Piece::Text(Split { reasoning: reasoning.to_owned(), content: content.to_owned() })
}

fn call(index: usize, name: &str) -> Piece {
    Piece::Call { index, id: format!("call_r_{index}"), name: name.to_owned() }
}

fn arguments(index: usize, json: &str) -> Piece {
    Piece::Arguments { index, json: json.to_owned() }
}

fn head() -> Head {
    let (_, echo) = converted(with(json!({"tools": [weather()], "metadata": {"run": "7"}})));
    Head { id: "resp_r".to_owned(), model: "m".to_owned(), created_at: 1_700_000_000, echo }
}

fn ending(finish: &'static str) -> Ending {
    let usage = Usage {
        prompt_tokens: 30,
        completion_tokens: 12,
        cached_prompt_tokens: 20,
        ..Usage::default()
    };
    let timings = Timings {
        prompt_tokens: 10,
        prompt_ms: 4.0,
        generated_tokens: 12,
        generated_ms: 22.0,
        drafts: None,
        output_hash: 0xab,
    };
    Ending { finish, usage, timings }
}

/// A reply in `steps`, ended by `finish`: its output, and the events of its
/// stream with their data parsed.
fn replied(steps: Vec<Vec<Piece>>, finish: &'static str) -> (Output, Vec<(String, Value)>) {
    let head = head();
    let ending = ending(finish);
    let mut output = Output::new("r".to_owned());
    let mut sequence = 0;
    let mut events: VecDeque<Written> = VecDeque::from(head.begin(&mut sequence));
    let mut write = |output: &Output, changes: Vec<Change>| {
        for change in &changes {
            change_events(output, change, &mut sequence, &mut events);
        }
    };
    for pieces in steps {
        let changes = output.push(pieces);
        write(&output, changes);
    }
    let changes = output.finish(ending.cut());
    write(&output, changes);
    events.push_back(head.end(output.items(), &ending, &mut sequence));
    let events = events
        .into_iter()
        .map(|(name, data)| (name.to_owned(), serde_json::from_str(&data).unwrap()))
        .collect();
    (output, events)
}

fn names(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(name, _)| name.as_str()).collect()
}

#[test]
fn a_reply_is_its_items_in_the_order_the_model_wrote_them() {
    let steps = vec![
        vec![text("Two ", "")],
        vec![text("calls.", "I look "), text("", "it up.")],
        vec![call(0, "get_weather"), arguments(0, "{\"city\":")],
        vec![arguments(0, "\"Köln\"}"), call(1, "now"), arguments(1, "{}")],
    ];
    let (output, events) = replied(steps, "stop");
    assert_eq!(
        serde_json::to_value(output.items()).unwrap(),
        json!([
            {
                "type": "reasoning",
                "id": "rs_r_0",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Two calls."}],
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_r_1",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "I look it up.",
                    "annotations": [],
                    "logprobs": [],
                }],
            },
            {
                "type": "function_call",
                "id": "fc_r_2",
                "status": "completed",
                "call_id": "call_r_0",
                "name": "get_weather",
                "arguments": "{\"city\":\"Köln\"}",
            },
            {
                "type": "function_call",
                "id": "fc_r_3",
                "status": "completed",
                "call_id": "call_r_1",
                "name": "now",
                "arguments": "{}",
            },
        ])
    );

    assert_eq!(
        names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (number, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], name.as_str());
        assert_eq!(data["sequence_number"], number, "{name}");
    }
    let data = |at: usize| &events[at].1;
    assert_eq!(
        data(2),
        &json!({
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": {
                "type": "reasoning", "id": "rs_r_0", "summary": [], "content": [],
                "status": "in_progress",
            },
        })
    );
    assert_eq!(
        data(3),
        &json!({
            "type": "response.content_part.added",
            "sequence_number": 3,
            "item_id": "rs_r_0",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "reasoning_text", "text": ""},
        })
    );
    assert_eq!(
        data(5),
        &json!({
            "type": "response.reasoning_text.delta",
            "sequence_number": 5,
            "item_id": "rs_r_0",
            "output_index": 0,
            "content_index": 0,
            "delta": "calls.",
        })
    );
    assert_eq!(data(6)["text"], "Two calls.");
    assert_eq!(data(7)["part"], json!({"type": "reasoning_text", "text": "Two calls."}));
    assert_eq!(data(8)["item"], serde_json::to_value(&output.items()[0]).unwrap());
    assert_eq!(
        data(10)["part"],
        json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []})
    );
    assert_eq!(
        data(11),
        &json!({
            "type": "response.output_text.delta",
            "sequence_number": 11,
            "item_id": "msg_r_1",
            "output_index": 1,
            "content_index": 0,
            "delta": "I look ",
            "logprobs": [],
        })
    );
    assert_eq!((&data(13)["text"], &data(13)["logprobs"]), (&json!("I look it up."), &json!([])));
    assert_eq!(
        data(16)["item"],
        json!({
            "type": "function_call", "id": "fc_r_2", "status": "in_progress",
            "call_id": "call_r_0", "name": "get_weather", "arguments": "",
        })
    );
    assert_eq!(
        data(19),
        &json!({
            "type": "response.function_call_arguments.done",
            "sequence_number": 19,
            "item_id": "fc_r_2",
            "output_index": 2,
            "name": "get_weather",
            "arguments": "{\"city\":\"Köln\"}",
        })
    );
    // The stream ends with the response that is returned whole.
    let whole: Value =
        serde_json::from_str(&head().response(output.items(), &ending("stop"))).unwrap();
    assert_eq!(data(25)["response"], whole);
}

#[test]
fn the_response_has_the_items_the_status_the_usage_and_what_the_request_said() {
    let (output, events) = replied(vec![vec![text("", "Mild.")]], "stop");
    let whole: Value =
        serde_json::from_str(&head().response(output.items(), &ending("stop"))).unwrap();
    assert_eq!(
        whole,
        json!({
            "id": "resp_r",
            "object": "response",
            "created_at": 1_700_000_000,
            "status": "completed",
            "background": false,
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "m",
            "output": [{
                "type": "message",
                "id": "msg_r_0",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text", "text": "Mild.", "annotations": [], "logprobs": [],
                }],
            }],
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {"effort": null, "summary": null},
            "store": false,
            "temperature": null,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [weather()],
            "top_p": null,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 30,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 12,
                "total_tokens": 42,
            },
            "user": null,
            "metadata": {"run": "7"},
            "timings": {
                "prompt_n": 10,
                "prompt_ms": 4.0,
                "prompt_per_second": 2500.0,
                "predicted_n": 12,
                "predicted_ms": 22.0,
                "predicted_per_second": 500.0,
                "token_hash": "00000000000000ab",
            },
        })
    );
    // The first two events have the response before anything is known of
    // its reply.
    let mut begun = whole.clone();
    begun["status"] = json!("in_progress");
    begun["output"] = json!([]);
    begun["usage"] = Value::Null;
    begun.as_object_mut().unwrap().remove("timings");
    assert_eq!(events[0].1["response"], begun);
    assert_eq!(events[1].1["response"], begun);
}

#[test]
fn a_reply_the_limit_cut_is_incomplete_and_so_is_the_item_it_cut() {
    let steps = vec![vec![text("Hm.", "")], vec![call(0, "get_weather"), arguments(0, "{\"ci")]];
    let (output, events) = replied(steps, "length");
    let items = serde_json::to_value(output.items()).unwrap();
    assert_eq!(items[0]["status"], "completed");
    assert_eq!(
        (&items[1]["status"], &items[1]["arguments"]),
        (&json!("incomplete"), &json!("{\"ci"))
    );
    let (name, last) = events.last().unwrap();
    assert_eq!(name, "response.incomplete");
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(last["response"]["incomplete_details"], json!({"reason": "max_output_tokens"}));
    assert_eq!(last["response"]["output"], items);
    // A call that was cut still ends with its events.
    let tail = names(&events);
    assert_eq!(
        tail[tail.len() - 3..],
        [
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.incomplete"
        ]
    );

    // A reply the limit cut inside its thinking has the reasoning alone.
    let (output, _) = replied(vec![vec![text("So", "")]], "length");
    let items = serde_json::to_value(output.items()).unwrap();
    assert_eq!(items.as_array().unwrap().len(), 1);
    assert_eq!(
        (&items[0]["type"], &items[0]["status"]),
        (&json!("reasoning"), &json!("incomplete"))
    );

    // A reply without text has no items.
    let (output, events) = replied(vec![vec![], vec![text("", "")]], "stop");
    assert!(output.items().is_empty());
    assert_eq!(names(&events), ["response.created", "response.in_progress", "response.completed"]);
}

#[test]
fn a_reply_sent_back_is_the_turn_the_chat_route_would_get_back() {
    let steps = vec![
        vec![text("Two cities:\n- Köln\n- Bonn", "I look it up.")],
        vec![call(0, "get_weather"), arguments(0, "{\"city\":\"Köln\"}")],
        vec![call(1, "get_weather"), arguments(1, "{\"city\":\"Bonn\"}")],
    ];
    let (output, _) = replied(steps, "stop");
    let mut input = vec![json!({"role": "user", "content": "Weather in Köln and Bonn?"})];
    // What a client does with the output of a response: it appends the
    // items as they are, and the outputs of the calls.
    input.extend(serde_json::to_value(output.items()).unwrap().as_array().unwrap().clone());
    input.push(json!({"type": "function_call_output", "call_id": "call_r_0", "output": "12 °C"}));
    input.push(json!({"type": "function_call_output", "call_id": "call_r_1", "output": "11 °C"}));
    assert_eq!(
        messages(Value::Array(input)),
        json!([
            {"role": "user", "content": "Weather in Köln and Bonn?"},
            {
                "role": "assistant",
                "content": "I look it up.",
                "reasoning_content": "Two cities:\n- Köln\n- Bonn",
                "tool_calls": [
                    {"id": "call_r_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Köln\"}"}},
                    {"id": "call_r_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Bonn\"}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_r_0", "content": "12 °C"},
            {"role": "tool", "tool_call_id": "call_r_1", "content": "11 °C"},
        ])
    );
}
