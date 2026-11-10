//! A small byte-level BPE vocabulary, built in the tests, and helpers to use it.
//!
//! The ids are fixed so that tests can say what they expect: 0 to 255 are the
//! bytes (in the order of their values), then the merged words below, then the
//! special tokens.

#![allow(dead_code)] // every test file uses some of it

use std::path::PathBuf;

use ares_gguf::{Gguf, GgufBuilder};
use ares_tokenizer::Tokenizer;

pub const HE: u32 = 256;
pub const LL: u32 = 257;
pub const HELL: u32 = 258;
pub const HELLO: u32 = 259;
pub const SPACE_W: u32 = 260;
pub const OR: u32 = 261;
pub const SPACE_WOR: u32 = 262;
pub const LD: u32 = 263;
pub const SPACE_WORLD: u32 = 264;
pub const U_UMLAUT: u32 = 265;
pub const IM_START: u32 = 266;
pub const IM_END: u32 = 267;
pub const THINK: u32 = 268;
pub const END_THINK: u32 = 269;
pub const UNUSED: u32 = 270;
/// Two line breaks and two spaces, which make the way text is cut into words visible.
pub const NEWLINES: u32 = 271;
pub const SPACES: u32 = 272;
pub const N_TOKENS: usize = 273;

/// The character GPT-2's byte-level alphabet writes `byte` as.
pub fn alphabet_char(byte: u8) -> char {
    let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
    if printable {
        return char::from(byte);
    }
    // The other 68 bytes are numbered in order from U+0100.
    let moved = (0..byte).filter(|b| !matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)).count();
    char::from_u32(0x100 + moved as u32).expect("a character")
}

/// `text` written in the byte-level alphabet, as token texts are.
pub fn alphabet(text: &str) -> String {
    text.bytes().map(alphabet_char).collect()
}

pub const MERGES: &[(&str, &str)] = &[
    ("h", "e"),
    ("l", "l"),
    ("he", "ll"),
    ("hell", "o"),
    (" ", "w"),
    ("o", "r"),
    (" w", "or"),
    ("l", "d"),
    (" wor", "ld"),
];

/// The tokens of the vocabulary, as the file stores them.
pub fn token_texts() -> Vec<String> {
    let mut tokens: Vec<String> = (0..=255u8).map(|b| alphabet_char(b).to_string()).collect();
    for text in ["he", "ll", "hell", "hello", " w", "or", " wor", "ld", " world"] {
        tokens.push(alphabet(text));
    }
    tokens.push("\u{c3}\u{bc}".to_owned()); // the two bytes of ü, each as its character
    for text in ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "[PAD270]"] {
        tokens.push(text.to_owned());
    }
    tokens.push(alphabet("\n\n"));
    tokens.push(alphabet("  "));
    assert_eq!(tokens.len(), N_TOKENS);
    tokens
}

pub fn token_types() -> Vec<i32> {
    let mut types = vec![1; 266];
    types.extend([3, 3, 4, 4, 5, 1, 1]);
    types
}

pub fn merge_texts() -> Vec<String> {
    let mut merges: Vec<String> =
        MERGES.iter().map(|(a, b)| format!("{} {}", alphabet(a), alphabet(b))).collect();
    // The two bytes of ü (C3 BC), as their characters.
    merges.push("\u{c3} \u{bc}".to_owned());
    merges.push(format!("{} {}", alphabet("\n"), alphabet("\n")));
    merges.push(format!("{} {}", alphabet(" "), alphabet(" ")));
    merges
}

/// A complete model file with the vocabulary, in which tests change what they
/// are about. A key can be replaced or left out; extra keys are added at the end.
pub struct Tiny {
    model: Option<ares_gguf_value::Str>,
    pre: Option<String>,
    tokens: Option<Vec<String>>,
    types: Option<Vec<i32>>,
    merges: Option<Vec<String>>,
    extra: Vec<Extra>,
}

enum Extra {
    Bool(&'static str, bool),
    U32(&'static str, u32),
    Str(&'static str, String),
    I32s(&'static str, Vec<i32>),
}

mod ares_gguf_value {
    /// `tokenizer.ggml.model` is a string, or for one test a number.
    pub enum Str {
        Text(String),
        Number(u32),
    }
}

pub fn tiny() -> Tiny {
    Tiny {
        model: Some(ares_gguf_value::Str::Text("gpt2".into())),
        pre: Some("qwen35".into()),
        tokens: Some(token_texts()),
        types: Some(token_types()),
        merges: Some(merge_texts()),
        extra: Vec::new(),
    }
}

impl Tiny {
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(ares_gguf_value::Str::Text(model.into()));
        self
    }

    pub fn model_as_number(mut self, number: u32) -> Self {
        self.model = Some(ares_gguf_value::Str::Number(number));
        self
    }

    pub fn no_model(mut self) -> Self {
        self.model = None;
        self
    }

    pub fn pre(mut self, pre: &str) -> Self {
        self.pre = Some(pre.into());
        self
    }

    pub fn no_pre(mut self) -> Self {
        self.pre = None;
        self
    }

    pub fn tokens(mut self, tokens: Vec<String>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    pub fn no_tokens(mut self) -> Self {
        self.tokens = None;
        self
    }

    pub fn types(mut self, types: Vec<i32>) -> Self {
        self.types = Some(types);
        self
    }

    pub fn merges(mut self, merges: Vec<String>) -> Self {
        self.merges = Some(merges);
        self
    }

    pub fn no_merges(mut self) -> Self {
        self.merges = None;
        self
    }

    pub fn bool(mut self, key: &'static str, value: bool) -> Self {
        self.extra.push(Extra::Bool(key, value));
        self
    }

    pub fn u32(mut self, key: &'static str, value: u32) -> Self {
        self.extra.push(Extra::U32(key, value));
        self
    }

    pub fn str(mut self, key: &'static str, value: &str) -> Self {
        self.extra.push(Extra::Str(key, value.into()));
        self
    }

    pub fn i32s(mut self, key: &'static str, value: Vec<i32>) -> Self {
        self.extra.push(Extra::I32s(key, value));
        self
    }

    pub fn builder(&self) -> GgufBuilder {
        let mut b = GgufBuilder::new().str("general.architecture", "tiny");
        match &self.model {
            Some(ares_gguf_value::Str::Text(model)) => b = b.str("tokenizer.ggml.model", model),
            Some(ares_gguf_value::Str::Number(n)) => b = b.u32("tokenizer.ggml.model", *n),
            None => {}
        }
        if let Some(pre) = &self.pre {
            b = b.str("tokenizer.ggml.pre", pre);
        }
        if let Some(tokens) = &self.tokens {
            b = b.str_array("tokenizer.ggml.tokens", tokens);
        }
        if let Some(types) = &self.types {
            b = b.i32_array("tokenizer.ggml.token_type", types);
        }
        if let Some(merges) = &self.merges {
            b = b.str_array("tokenizer.ggml.merges", merges);
        }
        b = b
            .u32("tokenizer.ggml.eos_token_id", IM_END)
            .u32("tokenizer.ggml.padding_token_id", UNUSED);
        for extra in &self.extra {
            b = match extra {
                Extra::Bool(key, value) => b.bool(key, *value),
                Extra::U32(key, value) => b.u32(key, *value),
                Extra::Str(key, value) => b.str(key, value),
                Extra::I32s(key, value) => b.i32_array(key, value),
            };
        }
        b
    }

    /// Writes the file and opens it.
    pub fn open(&self, tag: &str) -> (TempDir, Gguf) {
        open(&self.builder(), tag)
    }

    /// The tokenizer of the file, or why there is none.
    pub fn load(&self, tag: &str) -> Result<Tokenizer, ares_tokenizer::Error> {
        let (_dir, gguf) = self.open(tag);
        Tokenizer::from_gguf(&gguf)
    }
}

/// A fresh directory under the system temp dir, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ares-tokenizer-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Opens the model a builder describes.
pub fn open(builder: &GgufBuilder, tag: &str) -> (TempDir, Gguf) {
    let dir = TempDir::new(tag);
    let path = dir.path().join("model.gguf");
    builder.write(&path).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    (dir, gguf)
}

/// The tokenizer of the tiny vocabulary.
pub fn tiny_tokenizer(tag: &str) -> Tokenizer {
    tiny().load(tag).unwrap()
}

/// A repeatable stream of numbers for tests that need variety.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
