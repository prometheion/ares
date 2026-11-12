//! A small vocabulary and a random walk through a grammar, for the tests.
#![allow(dead_code)]

use ares_grammar::{Compiler, Cursor, Grammar, Source, Vocabulary};

/// 256 tokens of one byte each, a few longer ones, one control token that
/// stands for no text, and the end token last.
pub struct Toy {
    pub vocab: Vocabulary,
    pub compiler: Compiler,
    pub eos: u32,
    pub control: u32,
}

const LONG: &[&str] = &[
    "{\"", "\":", "\", ", "true", "false", "null", "name", "age", "id", "[\"", "\"]", ", ", "  ",
    "\n\n", "\\n", "\\u", "10", "100", "abc", "xyz", "<think>", "</think>",
];

pub fn toy() -> Toy {
    toy_with(&[])
}

/// The toy vocabulary with tokens for the texts of `extra` as well.
pub fn toy_with(extra: &[&str]) -> Toy {
    let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    // Bytes of a character that continues another are tokens too; a model's
    // vocabulary has them, and the grammar must put them together.
    pieces.extend(LONG.iter().map(|s| s.as_bytes().to_vec()));
    for text in extra {
        if text.len() > 1 && !pieces.iter().any(|piece| piece == text.as_bytes()) {
            pieces.push(text.as_bytes().to_vec());
        }
    }
    let control = pieces.len() as u32;
    pieces.push(Vec::new());
    let eos = pieces.len() as u32;
    pieces.push(Vec::new());
    let vocab = Vocabulary::new(&pieces, &[eos]).expect("a vocabulary");
    let compiler = Compiler::new(vocab.clone()).expect("a compiler");
    Toy { vocab, compiler, eos, control }
}

impl Toy {
    pub fn grammar(&self, source: Source<'_>) -> Grammar {
        self.compiler.compile(source).unwrap_or_else(|e| panic!("{source:?}: {e}"))
    }

    pub fn text(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| self.vocab.piece(*t).to_vec()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The token whose text is `text`.
    pub fn token(&self, text: &str) -> u32 {
        self.vocab.find(text.as_bytes()).unwrap_or_else(|| panic!("no token for {text:?}"))
    }

    pub fn tokens(&self, text: &str) -> Vec<u32> {
        // The longest token that fits, as a plain tokenizer would take it.
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let best = (0..self.vocab.len())
                .filter(|&t| {
                    let p = self.vocab.piece(t);
                    !p.is_empty() && bytes[at..].starts_with(p)
                })
                .max_by_key(|&t| self.vocab.piece(t).len())
                .expect("a byte token");
            at += self.vocab.piece(best).len();
            out.push(best);
        }
        out
    }
}

pub fn allowed(words: &[u32], token: u32) -> bool {
    words.get((token / 32) as usize).is_some_and(|w| w >> (token % 32) & 1 != 0)
}

pub fn allowed_tokens(words: &[u32]) -> Vec<u32> {
    (0..words.len() as u32 * 32).filter(|&t| allowed(words, t)).collect()
}

/// A walk through the grammar that picks among the allowed tokens by a seed,
/// and ends at the end token when the grammar can end (and a coin says so).
pub fn walk(toy: &Toy, grammar: &mut Grammar, seed: u64, limit: usize) -> (Vec<u32>, bool) {
    let mut cursor = grammar.start();
    let mut rng = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut tokens = Vec::new();
    for _ in 0..limit {
        let mask = cursor.mask().expect("a mask");
        let options = allowed_tokens(&mask);
        assert!(!options.is_empty(), "a step with nothing allowed after {:?}", toy.text(&tokens));
        let can_end = allowed(&mask, toy.eos);
        if can_end && next().is_multiple_of(4) {
            return (tokens, true);
        }
        let options: Vec<u32> = options.into_iter().filter(|&t| t != toy.eos).collect();
        if options.is_empty() {
            return (tokens, true);
        }
        let token = options[(next() % options.len() as u64) as usize];
        tokens.push(token);
        assert_eq!(cursor.goto(&tokens), tokens.len(), "the cursor refused an allowed token");
    }
    (tokens, false)
}

pub fn cursor_after(toy: &Toy, grammar: &mut Grammar, text: &str) -> Cursor {
    let mut cursor = grammar.start();
    let tokens = toy.tokens(text);
    assert_eq!(cursor.goto(&tokens), tokens.len(), "{text:?} is not a prefix of the grammar");
    cursor
}

/// Whether `text` is a whole sentence of the grammar.
pub fn accepts(toy: &Toy, grammar: &mut Grammar, text: &str) -> bool {
    let mut cursor = grammar.start();
    let tokens = toy.tokens(text);
    cursor.goto(&tokens) == tokens.len() && cursor.accepting()
}

/// Whether `text` can still become a sentence of the grammar.
pub fn viable(toy: &Toy, grammar: &mut Grammar, text: &str) -> bool {
    let mut cursor = grammar.start();
    let tokens = toy.tokens(text);
    cursor.goto(&tokens) == tokens.len()
}
