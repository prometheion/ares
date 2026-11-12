//! Text to token ids and back for the byte-level BPE vocabularies of GGUF
//! models, and the chat templates that turn a conversation into text.
//!
//! The crate reproduces llama.cpp. A [`Tokenizer`] built from a model file
//! gives the same token ids as `llama-tokenize` for the same text, and the chat
//! templates give the same prompt as `llama-server --jinja`; the tests compare
//! with both. What llama.cpp does with text it was not written for (malformed
//! UTF-8, for example) is reproduced too, because a different id in a prompt is
//! a different model input.
//!
//! ```no_run
//! use ares_gguf::Gguf;
//! use ares_tokenizer::{ChatOptions, EncodeOptions, Message, Tokenizer};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let gguf = Gguf::open("model.gguf")?;
//! let tokenizer = Tokenizer::from_gguf(&gguf)?;
//!
//! // Text a user wrote: the text of control tokens stays text.
//! let ids = tokenizer.encode("Hello, world!", EncodeOptions::default());
//! assert_eq!(tokenizer.decode(&ids, false)?, b"Hello, world!");
//!
//! // A conversation: the template writes the control tokens, which are turned
//! // into tokens when the prompt is encoded.
//! let messages = [Message::new("system", "Be brief."), Message::new("user", "Hi")];
//! let prompt = tokenizer.render_chat(&messages, &[], &ChatOptions::default())?;
//! let ids = tokenizer.encode(&prompt, EncodeOptions { add_special: true, parse_special: true });
//!
//! // Tokens of a reply, as they arrive.
//! let mut detokenizer = tokenizer.detokenizer(false);
//! let mut text = Vec::new();
//! for id in [9419, 1814] {
//!     detokenizer.push(id, &mut text)?;
//! }
//! detokenizer.finish(&mut text);
//! # Ok(())
//! # }
//! ```
//!
//! # What is supported
//!
//! Only what the engine's models use, and nothing else:
//!
//! - `tokenizer.ggml.model = gpt2`, byte-level BPE, with the pre-tokenizer
//!   `tokenizer.ggml.pre = qwen35` (Qwen3.5, Qwen3.8 and the models built on
//!   them). A model file that names another vocabulary type or pre-tokenizer
//!   gets an [`Error`] that names it; nothing falls back to another one.
//! - Chat templates in the dialect of Jinja that the templates of those models
//!   are written in. A template that uses a filter, test, function, method or
//!   statement outside it is refused by [`ChatTemplate::new`] with the name and
//!   the line, and not run.
//!
//! # Where it differs from llama.cpp
//!
//! - llama.cpp invents a beginning-of-text token (id 11) for a model that has
//!   none; here there is none.
//! - A conversation is rendered, not continued: llama.cpp's `continue_final_message`
//!   and the prefilling of an assistant message, images in messages, and the
//!   variables `datetime` and `date_string` are not supported.
//! - In templates, a few things follow MiniJinja where llama.cpp does something
//!   else: `~` and `+` write floats, lists and mappings their own way; an index
//!   into a string (`s[0]`, `s[-1]`) is by character where llama.cpp indexes
//!   bytes; operators on values of unlike types (`1 < 'a'`, `true + 1`) may
//!   succeed where llama.cpp raises an error; whole numbers above 2^53 are exact
//!   in arithmetic where llama.cpp computes with doubles. (`|length` of a string
//!   counts bytes, as llama.cpp does.) None of this occurs in the templates of
//!   the supported models.
//!
//! # Speed
//!
//! Encoding runs at tens of megabytes per second on one core, more than an order
//! of magnitude faster than llama.cpp's tokenizer, and
//! [`Tokenizer::encode_parallel`] scales that with the cores. What bounds it is the
//! lookup of merges: most of the time goes into probing a hash table of several
//! megabytes for pairs of tokens.

#![forbid(unsafe_code)]

mod bpe;
mod cache;
mod chat;
mod decode;
mod error;
mod jinja;
mod json;
mod pretok;
mod reasoning;
mod tokenizer;
mod tools;
mod unicode;
#[rustfmt::skip]
mod unicode_table;
mod vocab;

pub use chat::{ChatError, ChatOptions, ChatTemplate, Content, Message, Tool, ToolCall};
pub use decode::Detokenizer;
pub use error::{Error, Result};
pub use json::{Json, JsonError};
pub use reasoning::ReasoningFormat;
pub use tokenizer::{EncodeOptions, TokenKind, Tokenizer};
pub use tools::ToolFormat;
pub use vocab::SpecialTokens;
