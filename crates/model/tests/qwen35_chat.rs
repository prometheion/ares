//! The chat prompt of `qwen35` as the engine builds it for a request: the
//! reasoning of an assistant message, the variables of a request, and what the
//! model's template declares. Nothing here touches a device: the model is set
//! up (`Qwen35::new`) and asked for tokens.
//!
//! Needs `ARES_TEST_QWEN35_MODEL`, the Qwen3.5-0.8B file. Its template writes
//! the thinking of an assistant turn into the prompt only after the last user
//! message, so the tests put the reply last.

use std::path::PathBuf;

use ares_api::{ChatFormat, Json as Request, Message, Prompt, Role};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::Model;
use ares_tokenizer::{ChatOptions, EncodeOptions, Json, Tokenizer};

fn path() -> Option<PathBuf> {
    match std::env::var_os("ARES_TEST_QWEN35_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to run this test");
            None
        }
    }
}

fn model(path: &PathBuf, chat: ChatOptions) -> Qwen35 {
    Qwen35::new(Gguf::open(path).unwrap(), "q", &Spread::Single, Options::default(), 4096)
        .unwrap()
        .with_chat_options(chat)
}

/// A question and the reply to it, which has its reasoning if `reasoning`.
fn conversation(reasoning: Option<&str>) -> Vec<Message> {
    let mut reply = Message::new(Role::Assistant, "4");
    reply.reasoning_content = reasoning.map(str::to_owned);
    vec![Message::new(Role::User, "What is 2 + 2?"), reply]
}

fn prompt(messages: Vec<Message>, variables: Vec<(String, Request)>) -> Prompt {
    Prompt::Chat { messages, tools: None, variables }
}

/// The tokens of the same text with `text`'s own tokenizer.
fn tokens(path: &PathBuf, text: &str) -> Vec<u32> {
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap();
    tokenizer.encode(text, EncodeOptions { add_special: true, parse_special: true })
}

#[test]
fn the_reasoning_of_an_assistant_message_is_written_into_the_prompt() {
    let Some(path) = path() else { return };
    let qwen = model(&path, ChatOptions::default());
    let with = qwen.encode(&prompt(conversation(Some("Two and two.")), Vec::new())).unwrap();
    let without = qwen.encode(&prompt(conversation(None), Vec::new())).unwrap();
    assert_eq!(
        with,
        tokens(
            &path,
            "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n\
             <|im_start|>assistant\n<think>\nTwo and two.\n</think>\n\n4<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        )
    );
    assert_ne!(with, without, "the template wrote the reasoning, and an empty block without it");
    assert_eq!(
        without,
        tokens(
            &path,
            "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        )
    );
}

#[test]
fn the_variables_of_a_request_win_over_the_ones_the_server_started_with() {
    let Some(path) = path() else { return };
    let started = ChatOptions {
        variables: vec![("enable_thinking".to_owned(), Json::Bool(false))],
        ..ChatOptions::default()
    };
    let qwen = model(&path, started);
    let messages = || vec![Message::new(Role::User, "What is 2 + 2?")];
    let closed = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let open = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    // The server's variable, then the request's.
    assert_eq!(qwen.encode(&prompt(messages(), Vec::new())).unwrap(), tokens(&path, closed));
    let thinking = vec![("enable_thinking".to_owned(), Request::Bool(true))];
    assert_eq!(qwen.encode(&prompt(messages(), thinking)).unwrap(), tokens(&path, open));
    // A request that does not mention it leaves the server's value alone, and
    // the options of the model are the same for the next one.
    assert_eq!(qwen.encode(&prompt(messages(), Vec::new())).unwrap(), tokens(&path, closed));
}

#[test]
fn the_model_declares_what_its_template_says_about_thinking() {
    let Some(path) = path() else { return };
    let thinking = |chat: ChatOptions| {
        let ChatFormat { thinking, .. } =
            model(&path, chat).chat_format().expect("a model with a template");
        thinking.expect("the template thinks")
    };
    let format = thinking(ChatOptions::default());
    assert_eq!(format.start, "<think>\n");
    assert_eq!(format.end, "\n</think>\n\n");
    assert_eq!(format.prompt_on, "<think>\n");
    assert_eq!(format.prompt_off, "<think>\n\n</think>\n\n");
    assert!(format.on_by_default && format.switchable());
    // This template does not read an effort.
    assert!(!format.effort);

    // Whether the model thinks when a request does not say is what the server started it with.
    let off = ChatOptions {
        variables: vec![("enable_thinking".to_owned(), Json::Bool(false))],
        ..ChatOptions::default()
    };
    assert!(!thinking(off).on_by_default);
    let off = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
    assert!(!thinking(off).on_by_default);
}
