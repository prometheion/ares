use ares_api::{Json, Usage};
use serde_json::{Value, json};

use super::edits::{Edit, Keep, ToolUses, Trigger, Turns};
use super::reply::{Block, BlockEvent, Blocks, Content, Head, SIGNATURE, block_event, stop_reason};
use super::request::{Asked, to_chat};
use crate::chat_body::Purpose;
use crate::reasoning::Split;
use crate::reply::Piece;

fn read(body: Value) -> Json {
    serde_json::from_value(body).unwrap()
}

/// The chat body a Messages body comes to.
fn chat(body: Value) -> Value {
    serde_json::to_value(asked(body).0).unwrap()
}

/// The chat body and what else a Messages body asks for.
fn asked(body: Value) -> (Json, Asked) {
    to_chat(&read(body), Purpose::Reply).unwrap_or_else(|e| panic!("{}", e.message))
}

fn with(extra: Value) -> Value {
    let mut body = json!({
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
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
        "name": "get_weather",
        "description": "The weather of a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    })
}

#[test]
fn the_plain_fields_go_over_under_the_names_of_the_chat_body() {
    let body = with(json!({
        "system": "Be brief.",
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "stream": true,
        "stop_sequences": ["END", "\n\n"],
        "metadata": {"user_id": "u1"},
        "service_tier": "auto",
    }));
    assert_eq!(
        chat(body),
        json!({
            "model": "m",
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 64,
            "stream": true,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            "stop": ["END", "\n\n"],
        })
    );
}

#[test]
fn system_blocks_are_one_system_message() {
    let marker = json!({"type": "ephemeral", "ttl": "1h"});
    let body = with(json!({"system": [
        {"type": "text", "text": "You are terse."},
        {"type": "text", "text": "Answer in German.", "cache_control": marker},
    ]}));
    assert_eq!(
        chat(body)["messages"][0],
        json!({"role": "system", "content": "You are terse.\nAnswer in German."})
    );
    // No text, no message.
    assert_eq!(chat(with(json!({"system": []})))["messages"].as_array().unwrap().len(), 1);
    assert_eq!(chat(with(json!({"system": ""})))["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn the_stamp_of_the_billing_header_is_the_same_in_every_request() {
    let header = |stamp: &str| {
        format!(
            "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch={stamp};"
        )
    };
    let system = |stamp: &str| {
        let body = with(json!({"system": [
            {"type": "text", "text": header(stamp)},
            {"type": "text", "text": "You are Claude Code. cch=12345;"},
        ]}));
        chat(body)["messages"][0]["content"].as_str().unwrap().to_owned()
    };
    assert_eq!(system("a5145"), system("0beef"));
    assert_eq!(system("a5145"), format!("{}\nYou are Claude Code. cch=12345;", header("fffff")));
    // A system prompt that is a string, with the text after the stamp on the
    // same line, as Claude Code sends it.
    let string = |stamp: &str| {
        let body = with(json!({"system": format!("{}You are Claude Code.", header(stamp))}));
        chat(body)["messages"][0]["content"].as_str().unwrap().to_owned()
    };
    assert_eq!(string("a5145"), string("e0f9c"));
    assert_eq!(string("a5145"), format!("{}You are Claude Code.", header("fffff")));
    // Any five bytes before the `;`, as upstream: also characters of more
    // than one byte.
    assert_eq!(string("a-b_c"), string("12345"));
    assert_eq!(string("\u{e4}\u{f6}a"), string("12345"));
    // Only that line, and only a stamp of that form; a prompt without the
    // header is untouched.
    for other in [
        "cch=a5145; is text",
        "x-anthropic-billing-header: cch=a51;",
        "x-anthropic-billing-header: cch=a5145",
        "x-anthropic-billing-header: cch=a51456;",
        "x-anthropic-billing-header: none",
        "You are Claude Code. x-anthropic-billing-header: cch=a5145;",
    ] {
        assert_eq!(chat(with(json!({"system": other})))["messages"][0]["content"], other);
        let blocks = with(json!({"system": [{"type": "text", "text": other}]}));
        assert_eq!(chat(blocks)["messages"][0]["content"], other);
    }
}

#[test]
fn a_turn_of_the_user_keeps_the_order_of_its_blocks() {
    let body = with(json!({"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Köln"}},
            {"type": "tool_use", "id": "toolu_2", "name": "now", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "12 °C"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "is_error": true, "content": [
                {"type": "text", "text": "no clock"},
                {"type": "text", "text": "try later"},
            ]},
            {"type": "text", "text": "and now?"},
        ]},
    ]}));
    assert_eq!(
        chat(body)["messages"],
        json!([
            {"role": "user", "content": "one\ntwo"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "toolu_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"city": "Köln"}}},
                {"id": "toolu_2", "type": "function", "function": {"name": "now", "arguments": {}}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "12 °C"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "Error: no clock\ntry later"},
            {"role": "user", "content": "and now?"},
        ])
    );
    // A failed call that says nothing still says that it failed.
    let failed = with(json!({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "is_error": true},
    ]}]}));
    assert_eq!(chat(failed)["messages"][0]["content"], "Error");
}

#[test]
fn the_thinking_of_a_turn_goes_back_as_its_reasoning_whatever_the_signature() {
    for signature in [json!(SIGNATURE), json!(""), json!("EqQBCkgIARABGAIiQL..."), Value::Null] {
        let body = with(json!({"messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "2 and 2\nmake 4", "signature": signature},
                {"type": "redacted_thinking", "data": "b3BhcXVl"},
                {"type": "text", "text": "4"},
            ]},
            {"role": "user", "content": "sure?"},
        ]}));
        assert_eq!(
            chat(body)["messages"][1],
            json!({"role": "assistant", "content": "4", "reasoning_content": "2 and 2\nmake 4"})
        );
    }
    // Without thinking the message has no reasoning, which a template can tell
    // from an empty one.
    let body = with(json!({"messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]}));
    assert_eq!(chat(body)["messages"][1], json!({"role": "assistant", "content": "b"}));
}

/// What Claude Code sends: its environment as a turn of the system after the
/// first question, and the tokens left as one at the end.
#[test]
fn a_turn_of_the_system_joins_the_system_prompt_or_is_read_as_a_turn_of_the_user() {
    let body = with(json!({
        "system": [{"type": "text", "text": "You are terse."}],
        "messages": [
            {"role": "system", "content": "Answer in German."},
            {"role": "user", "content": "What does calc.py do?"},
            {"role": "system", "content": [{"type": "text", "text": "# Environment"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "calc.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "def add(a, b): ..."},
            ]},
            {"role": "system", "content": [{"type": "text", "text": "<total_tokens>999</total_tokens>",
                "cache_control": {"type": "ephemeral"}}]},
        ],
    }));
    let got = chat(body);
    assert_eq!(
        got["messages"],
        json!([
            {"role": "system", "content": "You are terse.\nAnswer in German."},
            {"role": "user", "content": "What does calc.py do?"},
            {"role": "system", "content": "# Environment"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "toolu_1", "type": "function",
                 "function": {"name": "Read", "arguments": {"file_path": "calc.py"}}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "def add(a, b): ..."},
            {"role": "system", "content": "<total_tokens>999</total_tokens>"},
        ])
    );
    // The reader of the chat body, the rule of every route, makes the later
    // ones turns of the user.
    let json: Json = serde_json::from_value(got.clone()).unwrap();
    let parsed =
        crate::wire::from_json(crate::wire::Endpoint::Chat, &json, ares_api::RequestId(1)).unwrap();
    let ares_api::Prompt::Chat { messages, .. } = &parsed.request.prompt else { panic!() };
    let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "user", "assistant", "tool", "user"]);
    // The reply is a new turn, not a continuation of the last one.
    assert!(got.get("continue_final_message").is_none(), "{got}");
    // A turn of the system first is the system prompt when there is no other.
    let first = with(json!({"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]}));
    assert_eq!(
        chat(first)["messages"],
        json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
    );
}

/// The edits of `context_management` are read with Anthropic's defaults and
/// carried out once the model is known (the `edits` module); the chat body
/// keeps every thinking, and the turns are counted as Anthropic counts them.
#[test]
fn context_edits_are_read_with_anthropics_defaults_and_the_turns_counted() {
    let turn = |n: u32| {
        json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": format!("thought {n}"), "signature": SIGNATURE},
            {"type": "text", "text": format!("answer {n}")},
        ]})
    };
    let question = json!({"role": "user", "content": "go on"});
    let results = json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": "r"}]});
    let messages = json!([
        question, turn(1), results, turn(2), question, {"role": "assistant", "content": "plain"},
        question,
    ]);
    let edits = |management: Value| {
        let mut body = with(json!({"messages": messages}));
        if !management.is_null() {
            body["context_management"] = management;
        }
        let (chat, asked) = asked(body);
        let chat = serde_json::to_value(chat).unwrap();
        let thoughts = chat["messages"].as_array().unwrap().iter();
        assert_eq!(thoughts.filter(|m| m["reasoning_content"].is_string()).count(), 2);
        assert_eq!(asked.turns, Turns { of_assistant: vec![0, 0, 1], current: 2 });
        asked.edits
    };
    assert_eq!(edits(Value::Null), None);
    assert_eq!(edits(json!({"edits": []})), Some(vec![]));
    let thinking = |keep: Value| {
        let mut edit = json!({"type": "clear_thinking_20251015"});
        if !keep.is_null() {
            edit["keep"] = keep;
        }
        edits(json!({"edits": [edit]})).unwrap()
    };
    assert_eq!(thinking(json!("all")), [Edit::ClearThinking { keep: Keep::All }]);
    assert_eq!(thinking(Value::Null), [Edit::ClearThinking { keep: Keep::Turns(1) }]);
    assert_eq!(
        thinking(json!({"type": "thinking_turns", "value": 2})),
        [Edit::ClearThinking { keep: Keep::Turns(2) }]
    );
    assert_eq!(
        edits(json!({"edits": [{"type": "clear_tool_uses_20250919"}]})).unwrap(),
        [Edit::ClearToolUses(ToolUses::default())]
    );
    let full = json!({"edits": [
        {"type": "clear_thinking_20251015", "keep": "all"},
        {"type": "clear_tool_uses_20250919", "trigger": {"type": "tool_uses", "value": 5},
         "keep": {"type": "tool_uses", "value": 1}, "clear_at_least": {"type": "input_tokens",
         "value": 500}, "exclude_tools": ["web_search"], "clear_tool_inputs": true},
    ]});
    assert_eq!(
        edits(full).unwrap()[1],
        Edit::ClearToolUses(ToolUses {
            trigger: Trigger::ToolUses(5),
            keep: 1,
            clear_at_least: Some(500),
            exclude: vec!["web_search".into()],
            clear_inputs: true,
        })
    );
    // Anthropic asks for the thinking edit first; values are checked.
    let late = with(json!({"context_management": {"edits": [
        {"type": "clear_tool_uses_20250919"}, {"type": "clear_thinking_20251015"}]}}));
    assert!(refused(late).1.contains("first"));
    for bad in [
        json!({"type": "clear_tool_uses_20250919", "trigger": {"type": "turns", "value": 1}}),
        json!({"type": "clear_tool_uses_20250919", "keep": {"type": "input_tokens", "value": 1}}),
        json!({"type": "clear_tool_uses_20250919", "exclude_tools": "web_search"}),
        json!({"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 0}}),
        json!({"type": "compact_20260112"}),
    ] {
        refused(with(json!({"context_management": {"edits": [bad]}})));
    }
}

#[test]
fn tools_and_the_choice_among_them_are_those_of_the_chat_body() {
    let function = json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "The weather of a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }});
    let bare = json!({"name": "now", "input_schema": {"type": "object"}, "type": "custom",
        "strict": true, "eager_input_streaming": true, "defer_loading": false,
        "cache_control": {"type": "ephemeral"}});
    let got = chat(with(json!({"tools": [weather(), bare]})));
    assert_eq!(got["tools"][0], function);
    assert_eq!(
        got["tools"][1],
        json!({"type": "function", "function": {"name": "now", "parameters": {"type": "object"}}})
    );
    assert!(got.get("tool_choice").is_none() && got.get("parallel_tool_calls").is_none());

    let choice = |choice: Value| chat(with(json!({"tools": [weather()], "tool_choice": choice})));
    assert_eq!(choice(json!({"type": "auto"}))["tool_choice"], "auto");
    assert_eq!(choice(json!({"type": "any"}))["tool_choice"], "required");
    assert_eq!(choice(json!({"type": "none"}))["tool_choice"], "none");
    assert_eq!(
        choice(json!({"type": "tool", "name": "get_weather"}))["tool_choice"],
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
    let one = choice(json!({"type": "any", "disable_parallel_tool_use": true}));
    assert_eq!(one["parallel_tool_calls"], false);
    let many = choice(json!({"type": "auto", "disable_parallel_tool_use": false}));
    assert_eq!(many["parallel_tool_calls"], true);
    // No tools, no section for them in the prompt.
    assert!(chat(with(json!({"tools": []}))).get("tools").is_none());
}

#[test]
fn thinking_is_the_switch_of_the_chat_template_and_the_budget_the_chat_routes() {
    let variables = |thinking: Value| chat(with(json!({"thinking": thinking})));
    let on = json!({"enable_thinking": true});
    let enabled = variables(json!({"type": "enabled", "budget_tokens": 31999}));
    assert_eq!(enabled["chat_template_kwargs"], on);
    assert_eq!(enabled["reasoning_budget_tokens"], 31999, "{enabled}");
    assert!(variables(json!({"type": "adaptive"})).get("reasoning_budget_tokens").is_none());
    assert_eq!(variables(json!({"type": "enabled"}))["chat_template_kwargs"], on);
    assert_eq!(variables(json!({"type": "adaptive"}))["chat_template_kwargs"], on);
    // How the thinking is to be shown: it comes back whole in every case.
    for display in ["summarized", "omitted", "updates"] {
        let adaptive = variables(json!({"type": "adaptive", "display": display}));
        assert_eq!(adaptive["chat_template_kwargs"], on);
        let enabled = variables(json!({"type": "enabled", "budget_tokens": 9, "display": display}));
        assert_eq!(enabled["chat_template_kwargs"], on);
    }
    assert_eq!(
        variables(json!({"type": "disabled"}))["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    // A request that does not say leaves it to the server.
    assert!(chat(with(json!({}))).get("chat_template_kwargs").is_none());
}

#[test]
fn output_config_is_the_schema_and_the_effort_of_the_chat_body() {
    let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
    let got = chat(with(json!({"output_config": {
        "format": {"type": "json_schema", "schema": schema},
        "effort": "low",
    }})));
    assert_eq!(
        got["response_format"],
        json!({"type": "json_schema", "json_schema": {"schema": schema}})
    );
    assert_eq!(got["reasoning_effort"], "low");
    // Anthropic's default level is the template's default (Claude Code sends
    // it with every request; the template of Flash-Next does not know it).
    let high = chat(with(json!({"output_config": {"effort": "high"}})));
    assert!(high.get("reasoning_effort").is_none(), "{high}");
    assert_eq!(chat(with(json!({"output_config": {"effort": "max"}})))["reasoning_effort"], "max");
    // With tools that stay in the prompt the schema is the rule.
    let none = json!({"tools": [weather()], "tool_choice": {"type": "none"},
        "output_config": {"format": {"type": "json_schema", "schema": schema}}});
    assert!(chat(with(none)).get("response_format").is_some());
}

#[test]
fn a_count_needs_no_limit_and_has_no_stream() {
    let body =
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true});
    let counted = to_chat(&read(body.clone()), Purpose::Count).unwrap().0;
    assert_eq!(
        serde_json::to_value(counted).unwrap(),
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    );
    assert_eq!(refused(body).0, "max_tokens");
}

#[test]
fn images_become_image_url_parts_of_the_chat_body() {
    let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}});
    let part = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}});
    let text = |t: &str| json!({"type": "text", "text": t});
    let body = with(json!({"messages": [{"role": "user", "content": [
        text("a"), image.clone(), text("b"),
        {"type": "tool_result", "tool_use_id": "t", "content": [text("c"), image.clone()]},
        text("d"),
    ]}]}));
    let chat = chat(body);
    assert_eq!(
        chat["messages"],
        json!([
            {"role": "user", "content": [text("a"), part.clone(), text("b")]},
            {"role": "tool", "tool_call_id": "t", "content": [text("c"), part.clone()]},
            {"role": "user", "content": "d"},
        ])
    );
}

#[test]
fn what_is_not_served_is_refused_by_its_name() {
    let user = |block: Value| with(json!({"messages": [{"role": "user", "content": [block]}]}));
    let assistant = |block: Value| {
        with(json!({"messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [block]},
            {"role": "user", "content": "b"},
        ]}))
    };
    let tool = |extra: Value| {
        let mut tool = weather();
        tool.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        with(json!({"tools": [tool]}))
    };
    let cases: Vec<(Value, &str, &str)> = vec![
        // Input that is neither text nor an image, and an image at a URL.
        (user(json!({"type": "document", "source": {}})), "messages[0].content[0]", "'document'"),
        (
            user(json!({"type": "image", "source": {"type": "url", "url": "https://a/b.png"}})),
            "messages[0].content[0].source",
            "not fetched from URLs",
        ),
        (
            user(
                json!({"type": "image", "source": {"type": "base64", "media_type": "text/plain", "data": "AA=="}}),
            ),
            "messages[0].content[0].source",
            "is not an image",
        ),
        // Tools the server would have to run, and what they leave behind.
        (
            with(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})),
            "tools",
            "tools[0]: a tool of the type 'web_search_20250305' is not supported",
        ),
        (
            with(json!({"tools": [{"type": "bash_20250124", "name": "bash"}]})),
            "tools",
            "'bash_20250124'",
        ),
        (
            assistant(
                json!({"type": "server_tool_use", "id": "s", "name": "web_search", "input": {}}),
            ),
            "messages[1].content[0]",
            "'server_tool_use' is not supported: the server runs no tools",
        ),
        (
            with(json!({"mcp_servers": [{"type": "url", "url": "https://x"}]})),
            "mcp_servers",
            "runs no tools",
        ),
        (tool(json!({"defer_loading": true})), "tools", "tools[0].defer_loading is not supported"),
        (
            tool(json!({"input_examples": [{}]})),
            "tools",
            "tools[0].input_examples is not supported",
        ),
        // Members nobody knows, anywhere.
        (
            with(json!({"context_management": {"edits": [
                {"type": "clear_tool_results_20990101"},
            ]}})),
            "context_management.edits[0]",
            "'clear_tool_results_20990101' is not supported",
        ),
        (
            with(json!({"context_management": {"edits": [
                {"type": "clear_thinking_20251015", "keep": {"type": "tool_uses", "value": 2}},
            ]}})),
            "context_management.edits[0]",
            "thinking_turns",
        ),
        (with(json!({"top_a": 0.2})), "top_a", "top_a is not supported"),
        (with(json!({"stop": ["x"]})), "stop", "stop_sequences"),
        (
            user(json!({"type": "text", "text": "a", "citations": [{"type": "char_location"}]})),
            "messages[0].content[0]",
            "citations is not supported",
        ),
        (
            user(json!({"type": "text", "text": "a", "extra": 1})),
            "messages[0].content[0]",
            ".extra is not supported",
        ),
        (
            with(json!({"thinking": {"type": "adaptive", "display": "loud"}})),
            "thinking",
            "thinking.display must be",
        ),
        (
            with(json!({"thinking": {"type": "disabled", "display": "omitted"}})),
            "thinking",
            "thinking.display is not supported",
        ),
        (
            with(json!({"output_config": {"format": {"type": "json_schema", "schema": {}}},
                "tools": [weather()]})),
            "output_config.format",
            "cannot be combined with tools",
        ),
        // Shapes.
        (with(json!({"max_tokens": 0})), "max_tokens", "at least 1"),
        (with(json!({"messages": []})), "messages", "must not be empty"),
        (
            with(json!({"messages": [{"role": "developer", "content": "x"}]})),
            "messages[0].role",
            "\"user\", \"assistant\" or \"system\"",
        ),
        (
            with(json!({"messages": [
                {"role": "user", "content": "x"},
                {"role": "system", "content": [{"type": "image"}]},
            ]})),
            "messages[1].content[0]",
            "'image'",
        ),
        (with(json!({"messages": [{"role": "user"}]})), "messages[0].content", "is required"),
        (
            with(json!({"messages": [{"role": "user", "content": []}]})),
            "messages[0].content",
            "must not be empty",
        ),
        (with(json!({"system": [{"type": "image"}]})), "system[0]", "'image'"),
        (with(json!({"stop_sequences": [""]})), "stop_sequences", "empty string"),
        (with(json!({"stop_sequences": "END"})), "stop_sequences", "an array of strings"),
        (
            with(json!({"thinking": {"type": "on"}})),
            "thinking",
            "\"enabled\", \"adaptive\" or \"disabled\"",
        ),
        (
            with(json!({"thinking": {"type": "enabled", "budget_tokens": 0}})),
            "thinking",
            "at least 1",
        ),
        (with(json!({"tool_choice": {"type": "any"}})), "tool_choice", "has no tools"),
        (
            with(json!({"tools": [weather()], "tool_choice": {"type": "tool", "name": "nope"}})),
            "tool_choice",
            "'nope', which is not among the tools",
        ),
        (
            with(json!({"tools": [weather()], "tool_choice": "auto"})),
            "tool_choice",
            "must be an object",
        ),
        (
            with(json!({"tools": [{"name": "f"}]})),
            "tools",
            "tools[0].input_schema must be an object",
        ),
        (with(json!({"tools": [{"input_schema": {}}]})), "tools", "tools[0].name"),
        (
            assistant(json!({"type": "tool_use", "id": "t", "name": "f", "input": "{}"})),
            "messages[1].content[0]",
            "input must be an object",
        ),
        (
            user(json!({"type": "tool_result", "content": "x"})),
            "messages[0].content[0]",
            "tool_use_id is required",
        ),
        (with(json!({"cache_control": {"type": "forever"}})), "cache_control", "\"ephemeral\""),
        (with(json!({"metadata": "u1"})), "metadata", "an object"),
    ];
    for (body, param, fragment) in cases {
        let (named, message) = refused(body.clone());
        assert_eq!(named, param, "{message}: {body}");
        assert!(message.contains(fragment), "{message:?} lacks {fragment:?}");
    }
}

fn text(reasoning: &str, content: &str) -> Piece {
    Piece::Text(Split { reasoning: reasoning.into(), content: content.into() })
}

fn call(index: usize, name: &str) -> Piece {
    Piece::Call { index, id: format!("toolu_x_{index}"), name: name.into() }
}

fn arguments(index: usize, json: &str) -> Piece {
    Piece::Arguments { index, json: json.into() }
}

/// The events of a reply that came in `steps`, and its content.
fn blocks(steps: Vec<Vec<Piece>>) -> (Vec<BlockEvent>, Content) {
    let mut blocks = Blocks::default();
    let mut events = Vec::new();
    for pieces in steps {
        events.extend(blocks.push(pieces));
    }
    events.extend(blocks.finish());
    let mut content = Content::default();
    content.take(events.clone());
    (events, content)
}

#[test]
fn a_reply_is_its_blocks_in_the_order_the_model_wrote_them() {
    let (events, content) = blocks(vec![
        vec![text("The user ", "")],
        vec![text("wants weather.", "I will ")],
        vec![text("", "look."), call(0, "get_weather"), arguments(0, "{\"city\":")],
        vec![arguments(0, "\"Köln\"}"), call(1, "now"), arguments(1, "{}")],
    ]);
    let weather = Block::ToolUse {
        id: "toolu_x_0".into(),
        name: "get_weather".into(),
        input: "{\"city\":\"Köln\"}".into(),
    };
    assert_eq!(
        content.0,
        vec![
            Block::Thinking("The user wants weather.".into(), SIGNATURE.into()),
            Block::Text("I will look.".into()),
            weather,
            Block::ToolUse { id: "toolu_x_1".into(), name: "now".into(), input: "{}".into() },
        ]
    );
    // Every block opens, gets its deltas and closes before the next opens,
    // and a thinking block gets its signature before it closes.
    let names: Vec<String> = events
        .iter()
        .map(|event| {
            let (name, data) = block_event(event);
            let data: Value = serde_json::from_str(&data).unwrap();
            assert_eq!(data["type"], name);
            let detail = match name {
                "content_block_start" => data["content_block"]["type"].as_str().unwrap().to_owned(),
                "content_block_delta" => data["delta"]["type"].as_str().unwrap().to_owned(),
                _ => "stop".to_owned(),
            };
            format!("{}:{detail}", data["index"])
        })
        .collect();
    assert_eq!(
        names,
        [
            "0:thinking",
            "0:thinking_delta",
            "0:thinking_delta",
            "0:signature_delta",
            "0:stop",
            "1:text",
            "1:text_delta",
            "1:text_delta",
            "1:stop",
            "2:tool_use",
            "2:input_json_delta",
            "2:input_json_delta",
            "2:stop",
            "3:tool_use",
            "3:input_json_delta",
            "3:stop",
        ]
    );
}

#[test]
fn a_reply_without_text_has_no_blocks_and_thinking_may_come_again() {
    assert_eq!(blocks(vec![vec![], vec![text("", "")]]), (Vec::new(), Content::default()));
    // Reasoning after a call is a block of its own.
    let (_, content) = blocks(vec![vec![text("a", ""), call(0, "now"), text("b", "c")]]);
    assert_eq!(
        content.0,
        vec![
            Block::Thinking("a".into(), SIGNATURE.into()),
            Block::ToolUse { id: "toolu_x_0".into(), name: "now".into(), input: String::new() },
            Block::Thinking("b".into(), SIGNATURE.into()),
            Block::Text("c".into()),
        ]
    );
}

#[test]
fn the_message_has_the_blocks_the_reason_and_the_usage() {
    let (_, content) = blocks(vec![vec![
        text("hm", "ok"),
        call(0, "get_weather"),
        arguments(0, "{\"city\":\"Köln\",\"days\":3.0}"),
        call(1, "cut"),
        arguments(1, "{\"city\":\"Kö"),
    ]]);
    let head = Head { id: "msg_1".into(), model: "m".into(), edits: None };
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 7,
        cached_prompt_tokens: 60,
        output_hash: 0,
        ..Usage::default()
    };
    let text = head.message(&content, stop_reason("stop", 2, None), None, usage);
    // The input is the text the model wrote, not a reading of it.
    assert!(text.contains(r#""input":{"city":"Köln","days":3.0}}"#), "{text}");
    let message: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        message,
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [
                {"type": "thinking", "thinking": "hm", "signature": SIGNATURE},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "toolu_x_0", "name": "get_weather",
                 "input": {"city": "Köln", "days": 3.0}},
                // A call the limit cut has no whole input.
                {"type": "tool_use", "id": "toolu_x_1", "name": "cut", "input": {}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 40,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 60,
                "output_tokens": 7,
            },
        })
    );
    assert_eq!(stop_reason("stop", 0, None), "end_turn");
    assert_eq!(stop_reason("stop", 0, Some("\n\n")), "stop_sequence");
    assert_eq!(stop_reason("stop", 1, Some("\n\n")), "tool_use");
    assert_eq!(stop_reason("length", 0, None), "max_tokens");
    assert_eq!(stop_reason("length", 1, None), "max_tokens");
}

#[test]
fn a_reply_sent_back_is_the_turn_the_chat_route_would_get_back() {
    // What the reply's blocks say, a client sends back as they are.
    let (_, content) = blocks(vec![vec![
        text("2 and 2\nmake 4", "Let me check."),
        call(0, "add"),
        arguments(0, "{\"a\":2,\"b\":2}"),
    ]]);
    let head = Head { id: "msg_1".into(), model: "m".into(), edits: None };
    let message: Value =
        serde_json::from_str(&head.message(&content, "tool_use", None, Usage::default())).unwrap();
    let body = with(json!({"messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": message["content"]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_x_0", "content": "4"},
        ]},
    ]}));
    assert_eq!(
        chat(body)["messages"],
        json!([
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "Let me check.",
             "reasoning_content": "2 and 2\nmake 4",
             "tool_calls": [{"id": "toolu_x_0", "type": "function",
                 "function": {"name": "add", "arguments": {"a": 2, "b": 2}}}]},
            {"role": "tool", "tool_call_id": "toolu_x_0", "content": "4"},
        ])
    );
}

/// `thinking.display: "omitted"`: the block has an empty text and its
/// signature carries the thinking, which comes back into the prompt when the
/// block is sent back as it came.
#[test]
fn omitted_thinking_is_carried_by_the_signature_and_comes_back() {
    let reasoning = "Größe: 2 + 2 = 4 🙂\nso the answer is 4.";
    let mut blocks = Blocks::new(true);
    let mut events = blocks.push(vec![Piece::Text(Split {
        reasoning: reasoning[..10].to_owned(),
        content: String::new(),
    })]);
    events.extend(blocks.push(vec![Piece::Text(Split {
        reasoning: reasoning[10..].to_owned(),
        content: "4".to_owned(),
    })]));
    events.extend(blocks.finish());
    assert!(!events.iter().any(|e| matches!(e, BlockEvent::ThinkingDelta { .. })), "{events:?}");
    let mut content = Content::default();
    content.take(events.clone());
    let Block::Thinking(text, signature) = &content.0[0] else { panic!("{content:?}") };
    assert_eq!(text, "");
    assert!(signature.starts_with(super::reply::CARRIED), "{signature}");
    // The stream says the same: an empty start and the signature.
    let written: Vec<_> = events.iter().map(block_event).collect();
    assert!(written[0].1.contains(r#""thinking":"""#), "{written:?}");
    assert!(written.iter().any(|(_, data)| data.contains(signature.as_str())));

    // Sent back as it came, the block is the reasoning of its message.
    let back = with(json!({"messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": signature},
            {"type": "text", "text": "4"},
        ]},
        {"role": "user", "content": "and 3+3?"},
    ]}));
    assert_eq!(chat(back)["messages"][1]["reasoning_content"], reasoning);
    // A text that is there wins; a carried signature that is not base64 is refused.
    let edited = with(json!({"messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "mine", "signature": signature},
        ]},
        {"role": "user", "content": "go on"},
    ]}));
    assert_eq!(chat(edited)["messages"][1]["reasoning_content"], "mine");
    let broken = with(json!({"messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": "ares-reasoning:***"},
        ]},
        {"role": "user", "content": "go on"},
    ]}));
    assert!(refused(broken).0.ends_with(".signature"));
    // Shown thinking keeps the fixed signature.
    let mut shown = Blocks::default();
    let mut events =
        shown.push(vec![Piece::Text(Split { reasoning: "r".into(), content: "a".into() })]);
    events.extend(shown.finish());
    assert!(events.contains(&BlockEvent::Signature { index: 0, signature: SIGNATURE.into() }));
    assert!(
        asked(with(json!({"thinking": {"type": "adaptive", "display": "omitted"}})))
            .1
            .omit_thinking
    );
    assert!(
        !asked(with(json!({"thinking": {"type": "adaptive", "display": "summarized"}})))
            .1
            .omit_thinking
    );
}

/// The edits that were applied are in the message and in the last
/// `message_delta` of a stream, for a request that asked for edits.
#[test]
fn applied_edits_are_in_the_message_and_the_last_delta() {
    use super::edits::Applied;
    let applied = vec![Applied::Thinking { turns: 2, tokens: 40 }];
    let head = Head { id: "msg_1".into(), model: "m".into(), edits: Some(applied) };
    let message: Value = serde_json::from_str(&head.message(
        &Content::default(),
        "end_turn",
        None,
        Usage::default(),
    ))
    .unwrap();
    let expected = json!({"applied_edits": [{"type": "clear_thinking_20251015",
        "cleared_thinking_turns": 2, "cleared_input_tokens": 40}]});
    assert_eq!(message["context_management"], expected);
    let [delta, stop] = head.message_end("end_turn", None, Usage::default());
    let delta: Value = serde_json::from_str(&delta.1).unwrap();
    assert_eq!(delta["type"], "message_delta");
    assert_eq!(delta["context_management"], expected);
    assert_eq!(stop.0, "message_stop");
    // None asked: no member; asked and nothing applied: an empty list.
    let plain = Head { edits: None, ..head.clone() };
    let message: Value = serde_json::from_str(&plain.message(
        &Content::default(),
        "end_turn",
        None,
        Usage::default(),
    ))
    .unwrap();
    assert!(message.get("context_management").is_none());
    let [delta, _] =
        Head { edits: Some(vec![]), ..head }.message_end("end_turn", None, Usage::default());
    let delta: Value = serde_json::from_str(&delta.1).unwrap();
    assert_eq!(delta["context_management"], json!({"applied_edits": []}));
}
