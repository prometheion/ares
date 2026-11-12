//! A call opened inside the thinking, through the real chat templates.
//!
//! A model that opens a call without closing its thinking generates the
//! reasoning, the marker and the call. The server hands the client the
//! reasoning and the call; the client sends both back with the result of the
//! call; the chat template writes the turn into the next prompt. The test
//! here plays that through with the template and the vocabulary of a model
//! file: what the template declares, what the server makes of the tokens,
//! what the next prompt looks like, and how much of the generated sequence it
//! repeats. The engine is not run. The model files are named by
//! `ARES_TEST_GGUF` (Qwen3.5-0.8B), `ARES_TEST_QWEN38_27B` (Qwen3.8-27B) and
//! `ARES_TEST_FLASH_NEXT` (Qwen3.8-Flash-Next, first part); the test says so
//! and passes when a file is not named.

use ares_api::ChatFormat;
use ares_gguf::Gguf;
use ares_tokenizer::{ChatOptions, EncodeOptions, Message, Tokenizer, ToolCall};

use super::*;
use crate::reasoning::Splitter;
use crate::reply::{Piece, Sorter};

const FILES: [(&str, &str); 3] = [
    ("ARES_TEST_GGUF", "Qwen3.5-0.8B"),
    ("ARES_TEST_QWEN38_27B", "Qwen3.8-27B"),
    ("ARES_TEST_FLASH_NEXT", "Qwen3.8-Flash-Next"),
];

const SCHEMA: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}"#;
const REASONING: &str = "The user asks for the weather in Köln.\nI look it up.";
const ARGUMENTS: &str = r#"{"city":"Köln","days":3}"#;

/// A reply as the server sorts it: the reasoning, the content, and each call
/// as its name and its arguments.
type Sorted = (String, String, Vec<(String, String)>);

struct Rig {
    name: &'static str,
    tokenizer: Tokenizer,
}

impl Rig {
    fn options(&self) -> ChatOptions {
        // The server's default: the thinking of earlier turns stays in the prompt.
        let preserve = ("preserve_reasoning".to_owned(), ares_tokenizer::Json::Bool(true));
        ChatOptions { variables: vec![preserve], ..ChatOptions::default() }
    }

    fn render(&self, messages: &[Message]) -> String {
        let weather = ares_tokenizer::Tool {
            name: "get_weather".into(),
            description: "The weather of a city.".into(),
            parameters: ares_tokenizer::Json::parse(SCHEMA).unwrap(),
        };
        self.tokenizer
            .render_chat(messages, &[weather], &self.options())
            .unwrap_or_else(|e| panic!("{}: {e}", self.name))
    }

    fn tokens(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, EncodeOptions { add_special: false, parse_special: true })
    }

    /// What the server makes of the tokens a model generated: the sorter of
    /// a request with tools that may be called, fed the text of the tokens
    /// one after the other.
    fn sorted(&self, format: &ChatFormat, generated: &[u32]) -> Sorted {
        let thinking = format.thinking.as_ref().expect("the template thinks");
        let tools = [Tool {
            name: "get_weather".into(),
            parameters: Some(serde_json::from_str(SCHEMA).unwrap()),
        }];
        let calls = splitter(format.tools.clone().unwrap(), &tools, ares_api::RequestId(1));
        let mut sorter =
            Sorter::new(Splitter::extracting(thinking, &thinking.prompt_on), Some(calls));
        let mut pieces = Vec::new();
        let mut detokenizer = self.tokenizer.detokenizer(false);
        for id in generated {
            let mut bytes = Vec::new();
            detokenizer.push(*id, &mut bytes).unwrap();
            pieces.extend(sorter.push(&String::from_utf8(bytes).unwrap()));
        }
        let mut bytes = Vec::new();
        detokenizer.finish(&mut bytes);
        pieces.extend(sorter.push(&String::from_utf8(bytes).unwrap()));
        pieces.extend(sorter.finish());
        let mut sorted = Sorted::default();
        for piece in pieces {
            match piece {
                Piece::Text(split) => {
                    sorted.0.push_str(&split.reasoning);
                    sorted.1.push_str(&split.content);
                }
                Piece::Call { name, .. } => sorted.2.push((name, String::new())),
                Piece::Arguments { index, json } => sorted.2[index].1.push_str(&json),
            }
        }
        sorted
    }
}

fn rigs() -> Vec<Rig> {
    let mut rigs = Vec::new();
    for (variable, name) in FILES {
        let Some(path) = std::env::var_os(variable) else {
            eprintln!("skipped: set {variable} to the model file of {name}");
            continue;
        };
        let tokenizer = Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap();
        rigs.push(Rig { name, tokenizer });
    }
    rigs
}

#[test]
fn a_call_opened_inside_the_thinking_goes_back_as_the_template_writes_a_call_after_it() {
    for rig in rigs() {
        let name = rig.name;
        // The marker and the tags are what the template declares.
        let format = ares_model::chat::chat_format(&rig.tokenizer, &rig.options());
        let thinking = format.thinking.clone().expect(name);
        let tools = format.tools.clone().expect(name);
        assert_eq!((tools.trigger(), thinking.end_tag()), ("<tool_call>", "</think>"), "{name}");
        assert_eq!(
            (thinking.start.as_str(), thinking.end.as_str()),
            ("<think>\n", "\n</think>\n\n")
        );
        let call = format!(
            "{}get_weather{}{}city{}Köln{}{}days{}3{}{}",
            tools.call_open,
            tools.name_close,
            tools.param_open,
            tools.key_close,
            tools.param_close,
            tools.param_open,
            tools.key_close,
            tools.param_close,
            tools.call_close
        );

        let question = Message::new("user", "How is the weather in Köln?");
        let prompt_text = rig.render(std::slice::from_ref(&question));
        assert!(prompt_text.ends_with(&thinking.prompt_on), "{name}: the prompt opens the block");
        let prompt = rig
            .tokenizer
            .encode(&prompt_text, EncodeOptions { add_special: true, parse_special: true });
        let end = rig.tokenizer.special_token_id("<|im_end|>").expect("a token");

        // The model opens the call inside its thinking, or closes the
        // thinking first as the template would have it.
        let inside = format!("{REASONING}\n\n{call}");
        let after = format!("{REASONING}{}{call}", thinking.end);
        let mut trips = Vec::new();
        for said in [&inside, &after] {
            let mut generated = rig.tokens(said);
            generated.push(end);
            let (reasoning, content, calls) = rig.sorted(&format, &generated);
            assert_eq!((reasoning.as_str(), content.as_str()), (REASONING, ""), "{name}");
            assert_eq!(calls, [("get_weather".to_owned(), ARGUMENTS.to_owned())], "{name}");

            // The client sends the reply back with the result of the call.
            let mut reply = Message::new("assistant", content);
            reply.reasoning_content = reasoning;
            reply.tool_calls = vec![ToolCall {
                id: Some("call_1".into()),
                name: calls[0].0.clone(),
                arguments: calls[0].1.clone(),
            }];
            let mut result = Message::new("tool", "12 degrees");
            result.tool_call_id = "call_1".into();
            let next_text = rig.render(&[question.clone(), reply, result]);
            let next = rig
                .tokenizer
                .encode(&next_text, EncodeOptions { add_special: true, parse_special: true });
            let sequence: Vec<u32> = prompt.iter().chain(&generated).copied().collect();
            let kept = sequence.iter().zip(&next).take_while(|(a, b)| a == b).count();
            trips.push((next_text, sequence.len(), kept));
        }

        // Either way the turn stands in the next prompt as the template
        // writes a call after thinking: the block closed, then the call.
        let turn = format!("{}{REASONING}{}{call}<|im_end|>\n", thinking.start, thinking.end);
        assert!(trips[0].0.contains(&turn), "{name}: {}", trips[0].0);
        assert_eq!(trips[0].0, trips[1].0, "{name}: one prompt for both replies");
        assert!(trips[0].0.ends_with(&thinking.prompt_on), "{name}");

        // A reply that closed its thinking is repeated by the next prompt,
        // all of it. One that did not is repeated up to the end of its
        // reasoning: from the white space before the marker on the template
        // writes the closing tag, and the call is computed again.
        let (_, closed_len, closed_kept) = trips[1];
        assert_eq!(closed_kept, closed_len, "{name}: the closed reply is repeated");
        let (_, open_len, open_kept) = trips[0];
        let reasoning_tokens = rig.tokens(REASONING).len();
        assert_eq!(open_kept, prompt.len() + reasoning_tokens, "{name}");
        eprintln!(
            "{name}: a call inside the thinking: {} of {} generated tokens are computed again",
            open_len - open_kept,
            open_len - prompt.len()
        );
    }
}
