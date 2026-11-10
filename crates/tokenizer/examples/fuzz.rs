//! Compares the tokenizer with llama.cpp's on many texts.
//!
//! ```text
//! cargo run --release -p ares-tokenizer --example fuzz -- MODEL.gguf REFTOK COUNT SEED [FILE...]
//! ```
//!
//! `REFTOK` is `tools/reference/reftok` built against llama.cpp (see the source for how).
//! The texts are every FILE whole, its lines, and pieces of random sizes, and COUNT texts
//! put together from a pool of awkward fragments (white space of every kind, letters of
//! many scripts, combining marks, emoji, numbers, special tokens and pieces of them,
//! malformed UTF-8) and from random bytes and code points. Each is encoded with and
//! without `parse_special` and compared token by token; random token sequences are decoded
//! with and without special tokens and compared byte by byte. The first differences are
//! printed with the bytes that caused them, and the exit code is 1 when there were any.

use std::io::{BufReader, Read, Write};
use std::process::{Command, Stdio};

use ares_gguf::Gguf;
use ares_tokenizer::{EncodeOptions, Tokenizer};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Fragments that text is put together from. They are chosen for the places where a
/// pre-tokenizer can go wrong.
#[rustfmt::skip]
const FRAGMENTS: &[&str] = &[
    // English, contractions in every case
    "the", "The", "quick", "Hello", "world", "don", "'t", "'S", "'re", "'VE", "'m", "'ll", "'D", "I'm", "it's", "'", "''",
    "x'y", "'\u{130}", "'\u{212A}", "'ſ", "'ﬁ", "'ｓ", "'Re", "'rE", "'lL", "'LE", "'v", "’s", "´s",
    // white space of every kind
    " ", "  ", "   ", "    ", "        ", "\t", "\t\t", "\n", "\n\n", "\n\n\n", "\r", "\r\n", "\r\n\r\n", " \n", "\n ", " \n ",
    "\t\n", "\u{b}", "\u{c}", "\u{a0}", "\u{2003}", "\u{2028}", "\u{2029}", "\u{3000}", "\u{85}", "\u{1680}", "\u{202f}",
    "\u{205f}", "\u{180e}", "\u{1c}", "\u{0}", "\u{1}", "\u{7f}", "\u{200b}", "\u{200c}", "\u{200d}", "\u{feff}", "\u{ad}",
    // punctuation and symbols
    ".", ",", "!", "?", ";", ":", "-", "--", "---", "_", "=", "==", "+", "*", "/", "\\", "(", ")", "[", "]", "{", "}", "<",
    ">", "</", "/>", "&", "|", "~", "`", "```", "#", "$", "%", "^", "@", "\"", "->", "=>", "::", "...", "€", "£", "¥", "©",
    "®", "™", "°", "±", "×", "÷", "→", "∑", "√", "∞", "«", "»", "„", "—", "…", "·",
    // numbers
    "0", "1", "12", "123", "1234", "2024", "3.14", "1e-5", "0xFF", "1,000,000", "٣", "١٢٣", "½", "²", "①", "Ⅷ", "५", "０１２",
    // scripts and marks
    "ä", "Ö", "ß", "ü", "Straße", "Größe", "Übergrößenträger", "é", "naïve", "ç", "ñ", "αβγ", "Привет", "мир", "שלום",
    "مرحبا", "हिन्दी", "नमस्ते", "สวัสดี", "你好", "世界", "日本語", "ひらがな", "カタカナ", "한국어", "ﬁ", "\u{301}",
    "\u{300}\u{301}", "e\u{301}", "a\u{308}", "\u{64b}", "\u{e31}", "\u{fe0f}", "Ａ", "ａｂｃ",
    // emoji and rare planes
    "😀", "😀😀", "👨‍👩‍👧‍👦", "👍🏽", "🇩🇪", "❤️", "🎉", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "𠮷", "\u{10FFFF}",
    // the tokens of the chat format and pieces of them
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_start|>user\n", "<|im_start|>assistant\n", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<|fim_prefix|>", "<|file_sep|>", "<|vision_start|>",
    "<|image_pad|>", "<tts_pad>", "<|im_", "<|im_start", "im_start|>", "<|", "|>", "<think", "<tool_call", "[PAD248077]",
    "<|im_start|><|im_end|>", "<|im_start|>|im_end|>",
    // code and data
    "def ", "fn main() {", "#include <stdio.h>", "SELECT * FROM t;", "<div class=\"x\">", "https://example.com/a?b=c&d=e#f",
    "user@example.com", "{\"a\": [1, 2.5, null, true]}", "    return x;", "// comment", "/* c */", "if (a && b) {",
];

/// Byte strings that are not UTF-8, in every way the decoder can fail.
#[rustfmt::skip]
const MALFORMED: &[&[u8]] = &[
    b"\x80", b"\xBF", b"\xC0\x80", b"\xC1\xBF", b"\xC3", b"\xC3\x28", b"\xE2\x82", b"\xE2\x82\xAC", b"\xE0\x80\x80",
    b"\xED\xA0\x80", b"\xED\xBF\xBF", b"\xF0\x9F\x98", b"\xF0\x9F\x98\x80", b"\xF4\x90\x80\x80", b"\xF5\x80\x80\x80",
    b"\xF8\x88\x80\x80\x80", b"\xFF", b"\xFE", b"\xEF\xBF\xBD",
];

fn random_text(rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::new();
    match rng.below(12) {
        0 => out.extend((0..rng.below(40)).map(|_| rng.next() as u8)),
        1 => {
            for _ in 0..rng.below(30) {
                let cpt = match rng.below(4) {
                    0 => rng.below(0x80),
                    1 => rng.below(0x800),
                    2 => rng.below(0x10000),
                    _ => rng.below(0x110000),
                } as u32;
                match char::from_u32(cpt) {
                    Some(c) => out.extend_from_slice(c.to_string().as_bytes()),
                    // A surrogate, as three bytes by hand.
                    None => out.extend_from_slice(&[
                        0xE0 | (cpt >> 12) as u8,
                        0x80 | ((cpt >> 6) & 0x3F) as u8,
                        0x80 | (cpt & 0x3F) as u8,
                    ]),
                }
            }
        }
        // Long runs and long words, for the paths of BPE that short words do not reach.
        2 => {
            for _ in 0..1 + rng.below(6) {
                let unit = FRAGMENTS[rng.below(FRAGMENTS.len())];
                let bound = if rng.below(4) == 0 { 400 } else { 40 };
                for _ in 0..1 + rng.below(bound) {
                    out.extend_from_slice(unit.as_bytes());
                }
            }
        }
        3 => {
            for _ in 0..1 + rng.below(4) {
                let alphabet = if rng.below(3) == 0 { 3 } else { 26 };
                out.extend((0..1 + rng.below(700)).map(|_| b'a' + rng.below(alphabet) as u8));
                out.push(b' ');
            }
        }
        _ => {
            for _ in 0..1 + rng.below(25) {
                if rng.below(12) == 0 {
                    out.extend_from_slice(MALFORMED[rng.below(MALFORMED.len())]);
                } else {
                    out.extend_from_slice(FRAGMENTS[rng.below(FRAGMENTS.len())].as_bytes());
                }
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, model, reftok, count, seed, files @ ..] = &args[..] else {
        eprintln!("usage: fuzz MODEL.gguf REFTOK COUNT SEED [FILE...]");
        std::process::exit(2);
    };
    let count: usize = count.parse().expect("COUNT is a number");
    let seed: u64 = seed.parse().expect("SEED is a number");
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(model).expect("open the model"))
        .expect("build the tokenizer");
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

    let mut texts: Vec<Vec<u8>> = vec![Vec::new()];
    for file in files {
        let data = std::fs::read(file).expect("read a file");
        texts.extend(data.split_inclusive(|b| *b == b'\n').map(<[u8]>::to_vec));
        let mut at = 0;
        while at < data.len() {
            let size = 1 + rng.below(300);
            texts.push(data[at..(at + size).min(data.len())].to_vec());
            at += size;
        }
        texts.push(data);
    }
    texts.extend((0..count).map(|_| random_text(&mut rng)));

    // Requests for the reference: u8 mode, u8 flags, u32 length, bytes.
    // Mode 0 encodes (flags: 1 add_special, 2 parse_special); 1 and 2 decode with and
    // without special tokens.
    let mut requests: Vec<(u8, u8, Vec<u8>)> = Vec::new();
    for text in &texts {
        for flags in [0, 2, 3] {
            requests.push((0, flags, text.clone()));
        }
    }
    let n_tokens = tokenizer.n_tokens();
    for _ in 0..count / 4 + 10 {
        let ids: Vec<u8> = (0..rng.below(20))
            .flat_map(|_| {
                let id = match rng.below(3) {
                    0 => 248_044 + rng.below(40),
                    1 => rng.below(300),
                    _ => rng.below(n_tokens),
                };
                (id as u32).to_le_bytes()
            })
            .collect();
        requests.push((1, 0, ids.clone()));
        requests.push((2, 0, ids));
    }

    let mut child = Command::new(reftok)
        .arg(model)
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start the reference");
    let mut stdin = child.stdin.take().expect("a pipe");
    let to_send = requests.clone();
    let writer = std::thread::spawn(move || {
        for (mode, flags, data) in to_send {
            stdin.write_all(&[mode, flags]).unwrap();
            stdin.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
            stdin.write_all(&data).unwrap();
        }
    });
    let mut stdout = BufReader::new(child.stdout.take().expect("a pipe"));
    let (mut mismatches, mut bytes, mut tokens) = (0, 0, 0);
    for (mode, flags, data) in &requests {
        let mut length = [0u8; 4];
        stdout.read_exact(&mut length).unwrap();
        let length = u32::from_le_bytes(length) as usize;
        if *mode == 0 {
            let mut raw = vec![0u8; length * 4];
            stdout.read_exact(&mut raw).unwrap();
            let reference: Vec<u32> =
                raw.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            let options =
                EncodeOptions { add_special: flags & 1 != 0, parse_special: flags & 2 != 0 };
            let ours = tokenizer.encode(data, options);
            bytes += data.len();
            tokens += reference.len();
            if ours != reference {
                mismatches += 1;
                if mismatches <= 15 {
                    println!(
                        "MISMATCH {options:?}\n  bytes {data:02x?}\n  llama.cpp {reference:?}\n  here      {ours:?}"
                    );
                }
            }
        } else {
            let mut reference = vec![0u8; length];
            stdout.read_exact(&mut reference).unwrap();
            let ids: Vec<u32> =
                data.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            let ours = tokenizer.decode(&ids, *mode == 1).unwrap();
            if ours != reference {
                mismatches += 1;
                if mismatches <= 15 {
                    println!(
                        "DECODE MISMATCH special={} ids={ids:?}\n  llama.cpp {reference:02x?}\n  here      {ours:02x?}",
                        *mode == 1
                    );
                }
            }
        }
    }
    writer.join().unwrap();
    let _ = child.wait();
    println!(
        "{} requests, {bytes} bytes, {tokens} tokens from llama.cpp, {mismatches} mismatches",
        requests.len()
    );
    if mismatches > 0 {
        std::process::exit(1);
    }
}
