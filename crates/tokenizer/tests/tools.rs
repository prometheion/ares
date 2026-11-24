//! How the real chat templates write the calls of tools.
//!
//! The templates are read from the model files named by `ARES_TEST_QWEN35_0_8B`
//! (Qwen3.5-0.8B), `ARES_TEST_QWEN38_27B` (Qwen3.8-27B) and
//! `ARES_TEST_FLASH_NEXT` (Qwen3.8-Flash-Next, first part); a test says so and
//! passes when a file is not named. Reading the metadata of a file needs no GPU.

use ares_gguf::Gguf;
use ares_tokenizer::{ChatOptions, EncodeOptions, Json, Message, Tokenizer, Tool, ToolCall};

const FILES: [(&str, &str); 3] = [
    ("ARES_TEST_QWEN35_0_8B", "Qwen3.5-0.8B"),
    ("ARES_TEST_QWEN38_27B", "Qwen3.8-27B"),
    ("ARES_TEST_FLASH_NEXT", "Qwen3.8-Flash-Next"),
];

fn tokenizer_of(variable: &str, what: &str) -> Option<Tokenizer> {
    let Some(path) = std::env::var_os(variable) else {
        eprintln!("skipped: set {variable} to the model file of {what}");
        return None;
    };
    Some(Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap())
}

/// The form all three templates write a call in.
fn tagged() -> ares_api::ToolFormat {
    ares_api::ToolFormat {
        call_open: "<tool_call>\n<function=".into(),
        name_close: ">\n".into(),
        param_open: "<parameter=".into(),
        key_close: ">\n".into(),
        param_close: "\n</parameter>\n".into(),
        call_close: "</function>\n</tool_call>".into(),
        before_calls: "\n\n".into(),
        between_calls: "\n".into(),
    }
}

fn weather() -> Tool {
    let schema = r#"{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}"#;
    Tool {
        name: "get_weather".into(),
        description: "The weather of a city.".into(),
        parameters: Json::parse(schema).unwrap(),
    }
}

#[test]
fn the_three_templates_write_calls_in_the_tagged_form() {
    for (variable, what) in FILES {
        let Some(tokenizer) = tokenizer_of(variable, what) else { continue };
        assert_eq!(tokenizer.tool_format(), Some(tagged()), "{what}");
    }
}

#[test]
fn the_tags_of_a_call_are_tokens_of_their_own_and_print_as_text() {
    for (variable, what) in FILES {
        let Some(tokenizer) = tokenizer_of(variable, what) else { continue };
        for tag in ["<tool_call>", "</tool_call>"] {
            let tokens =
                tokenizer.encode(tag, EncodeOptions { add_special: false, parse_special: true });
            assert_eq!(tokens.len(), 1, "{what}: {tag} is {tokens:?}");
            // A generated token reaches the reply as its piece without the
            // control tokens: the tag must be among the text.
            assert_eq!(tokenizer.piece(tokens[0], false).unwrap(), tag.as_bytes(), "{what}");
        }
    }
}

#[test]
fn tools_a_call_and_its_result_are_rendered_by_the_three_templates() {
    for (variable, what) in FILES {
        let Some(tokenizer) = tokenizer_of(variable, what) else { continue };
        let mut assistant = Message::new("assistant", "I will look it up.");
        assistant.tool_calls = vec![
            ToolCall {
                id: Some("call_1".into()),
                name: "get_weather".into(),
                arguments: r#"{"city":"Köln","days":3}"#.into(),
            },
            ToolCall {
                id: Some("call_2".into()),
                name: "get_weather".into(),
                arguments: r#"{"city":"Bonn\nBeuel"}"#.into(),
            },
        ];
        let mut first = Message::new("tool", "12 degrees");
        first.tool_call_id = "call_1".into();
        let mut second = Message::new("tool", "13 degrees");
        second.tool_call_id = "call_2".into();
        let messages = [Message::new("user", "How is the weather?"), assistant, first, second];
        let text = tokenizer
            .render_chat(&messages, &[weather()], &ChatOptions::default())
            .unwrap_or_else(|e| panic!("{what}: {e}"));

        // The tools are listed for the model, with the schema of each.
        assert!(
            text.contains(
                "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\""
            ),
            "{what}: {text}"
        );
        assert!(text.contains("</tools>"), "{what}");
        // The calls stand in the assistant's turn as the format says, a
        // string as it is and a number as JSON.
        let calls = "I will look it up.\n\n<tool_call>\n<function=get_weather>\n\
                     <parameter=city>\nKöln\n</parameter>\n<parameter=days>\n3\n</parameter>\n\
                     </function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n\
                     <parameter=city>\nBonn\nBeuel\n</parameter>\n</function>\n</tool_call><|im_end|>\n";
        assert!(text.contains(calls), "{what}: {text}");
        // The results come back in one user turn.
        let results = "<|im_start|>user\n<tool_response>\n12 degrees\n</tool_response>\n\
                       <tool_response>\n13 degrees\n</tool_response><|im_end|>\n";
        assert!(text.contains(results), "{what}: {text}");
        assert!(text.ends_with("<|im_start|>assistant\n<think>\n"), "{what}");
    }
}
