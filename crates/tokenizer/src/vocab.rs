//! The vocabulary of a model file: the texts of the tokens, what kind each is,
//! and which ones have a meaning to the engine (end of text, fill-in-the-middle).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use ares_gguf::{Gguf, Value, ValueType};

use crate::bpe::{Merge, Merges};
use crate::error::{Error, Result};
use crate::unicode::decode;

/// What a token is, as llama.cpp keeps it: bit flags, because a token can be
/// more than one thing (a token the file calls normal but that is plainly an
/// end-of-text marker is made a control token as well).
pub(crate) mod attr {
    pub(crate) const UNKNOWN: u8 = 1 << 0;
    pub(crate) const UNUSED: u8 = 1 << 1;
    pub(crate) const NORMAL: u8 = 1 << 2;
    pub(crate) const CONTROL: u8 = 1 << 3;
    pub(crate) const USER_DEFINED: u8 = 1 << 4;
    /// The tokens that the text can name: control, unknown and user defined.
    pub(crate) const SPECIAL: u8 = UNKNOWN | CONTROL | USER_DEFINED;
}

/// The ids of the tokens the engine treats specially. A token is `None` when the
/// model has none of that kind.
///
/// The ids come from the file's `tokenizer.ggml.*_token_id` keys. Where a key
/// is missing, `eot` and the fill-in-the-middle tokens are found by their
/// names (`<|im_end|>`, `<|fim_prefix|>`, ...), as llama.cpp does. Unlike
/// llama.cpp, a missing `bos` or `eos` is not replaced by a made-up id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    /// Beginning of text.
    pub bos: Option<u32>,
    /// End of text, which the model writes at the end of a reply.
    pub eos: Option<u32>,
    /// End of a turn. Often the same token as `eos`.
    pub eot: Option<u32>,
    /// End of a message.
    pub eom: Option<u32>,
    /// A token for what the vocabulary cannot represent.
    pub unk: Option<u32>,
    /// Separator between texts.
    pub sep: Option<u32>,
    /// Padding.
    pub padding: Option<u32>,
    /// Masked position.
    pub mask: Option<u32>,
    /// Fill in the middle: the text before the hole.
    pub fim_prefix: Option<u32>,
    /// Fill in the middle: the text after the hole.
    pub fim_suffix: Option<u32>,
    /// Fill in the middle: where the generated text starts.
    pub fim_middle: Option<u32>,
    /// Fill in the middle: padding.
    pub fim_pad: Option<u32>,
    /// Fill in the middle: the name of the repository.
    pub fim_repo: Option<u32>,
    /// Fill in the middle: the separator between files.
    pub fim_file_sep: Option<u32>,
}

/// Byte strings back to back, addressed by index.
#[derive(Default)]
pub(crate) struct Pool {
    bytes: Vec<u8>,
    ends: Vec<u32>,
}

impl Pool {
    fn push(&mut self, item: &[u8]) {
        self.bytes.extend_from_slice(item);
        self.ends
            .push(u32::try_from(self.bytes.len()).expect("a vocabulary is smaller than 4 GiB"));
    }

    pub(crate) fn get(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }

    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }
}

impl std::fmt::Debug for Vocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocab")
            .field("tokens", &self.len())
            .field("special", &self.special)
            .finish()
    }
}

/// The tokens of a model.
pub(crate) struct Vocab {
    /// The text of each token as the file stores it.
    pub(crate) texts: Pool,
    /// What each token stands for in output text: its bytes, or empty for tokens
    /// that never appear in it.
    pub(crate) pieces: Pool,
    pub(crate) attrs: Vec<u8>,
    pub(crate) special: SpecialTokens,
    /// The ids that end a generation, sorted.
    pub(crate) end_of_generation: Vec<u32>,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    /// The token of each single byte, which every word starts from.
    pub(crate) byte_token: [u32; 256],
    pub(crate) merges: Merges,
}

/// The characters GPT-2's byte-level BPE writes bytes as: printable ASCII and
/// Latin-1 stand for themselves, the other 68 bytes are moved to U+0100 and up.
/// Returns the byte of a character, or `None` for a character that is no byte.
fn byte_of_char(c: u32) -> Option<u8> {
    const LAST: usize = 0x100 + 68;
    static TABLE: std::sync::LazyLock<[i16; LAST]> = std::sync::LazyLock::new(|| {
        let mut table = [-1i16; LAST];
        let mut moved = 0x100;
        for byte in 0..=255u8 {
            let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
            let c = if printable {
                usize::from(byte)
            } else {
                moved += 1;
                moved - 1
            };
            table[c] = i16::from(byte);
        }
        table
    });
    TABLE.get(c as usize).and_then(|b| u8::try_from(*b).ok())
}

/// The bytes a token text stands for, when it is written in the byte alphabet
/// only. `None` for a text with any other character, or malformed UTF-8.
fn decode_alphabet(text: &[u8], out: &mut Vec<u8>) -> bool {
    let start = out.len();
    let mut i = 0;
    while i < text.len() {
        let Some((c, len)) = decode(&text[i..]) else {
            out.truncate(start);
            return false;
        };
        let Some(byte) = byte_of_char(c) else {
            out.truncate(start);
            return false;
        };
        out.push(byte);
        i += len;
    }
    true
}

/// A hasher for byte strings that is fast and needs no protection against
/// hostile keys: the keys here are the tokens of a model file.
#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0x517C_C1B7_2722_0A95;
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(*word)).wrapping_mul(K);
        }
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        self.0 =
            (self.0.rotate_left(5) ^ u64::from_le_bytes(tail) ^ bytes.len() as u64).wrapping_mul(K);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type ByteMap<'a> = HashMap<&'a [u8], u32, BuildHasherDefault<FastHasher>>;

fn type_name(value: &Value) -> String {
    format!("{:?}", value.value_type()).to_lowercase()
}

fn string<'a>(gguf: &'a Gguf, key: &'static str) -> Result<Option<&'a str>> {
    match gguf.get(key) {
        None => Ok(None),
        Some(value) => value.as_str().map(Some).ok_or_else(|| Error::WrongType {
            key,
            expected: "a string",
            found: type_name(&value),
        }),
    }
}

fn flag(gguf: &Gguf, key: &'static str) -> Result<Option<bool>> {
    match gguf.get(key) {
        None => Ok(None),
        Some(value) => value.as_bool().map(Some).ok_or_else(|| Error::WrongType {
            key,
            expected: "a boolean",
            found: type_name(&value),
        }),
    }
}

fn token_id(gguf: &Gguf, key: &'static str, n_tokens: usize) -> Result<Option<u32>> {
    let Some(value) = gguf.get(key) else { return Ok(None) };
    let id = value.as_u64().ok_or_else(|| Error::WrongType {
        key,
        expected: "a token id",
        found: type_name(&value),
    })?;
    match u32::try_from(id) {
        Ok(id) if (id as usize) < n_tokens => Ok(Some(id)),
        _ => Err(Error::Invalid(format!(
            "'{key}' is {id}, but the vocabulary has {n_tokens} tokens"
        ))),
    }
}

/// The pre-tokenizer of a vocabulary, from `tokenizer.ggml.pre`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pre {
    /// The Qwen3.5 family and everything that builds on it.
    Qwen35,
}

impl Vocab {
    pub(crate) fn load(gguf: &Gguf) -> Result<(Self, Pre)> {
        let model = string(gguf, "tokenizer.ggml.model")?
            .ok_or(Error::MissingKey { key: "tokenizer.ggml.model" })?;
        if model != "gpt2" {
            return Err(Error::UnsupportedModel { model: model.to_owned() });
        }
        let pre_name = string(gguf, "tokenizer.ggml.pre")?
            .ok_or(Error::MissingKey { key: "tokenizer.ggml.pre" })?;
        let pre = match pre_name {
            "qwen35" => Pre::Qwen35,
            other => return Err(Error::UnsupportedPre { pre: other.to_owned() }),
        };
        // These change the text before the vocabulary sees it, and only for other
        // kinds of tokenizer; a file that asks for them is not one of ours.
        for key in [
            "tokenizer.ggml.add_space_prefix",
            "tokenizer.ggml.remove_extra_whitespaces",
            "tokenizer.ggml.clean_spaces",
        ] {
            if flag(gguf, key)?.unwrap_or(false) {
                return Err(Error::UnsupportedOption { key });
            }
        }

        let tokens = gguf
            .array("tokenizer.ggml.tokens")
            .filter(|a| a.elem_type() == ValueType::String)
            .ok_or(Error::MissingKey { key: "tokenizer.ggml.tokens" })?;
        let n_tokens = tokens.len() as usize;
        if n_tokens == 0 || u32::try_from(n_tokens).is_err() {
            return Err(Error::Invalid(format!("the vocabulary has {n_tokens} tokens")));
        }
        let types = match gguf.array("tokenizer.ggml.token_type") {
            None => None,
            Some(array) => {
                let types = array.to_i32_vec().ok_or(Error::WrongType {
                    key: "tokenizer.ggml.token_type",
                    expected: "an array of integers",
                    found: "something else".into(),
                })?;
                if types.len() < n_tokens {
                    return Err(Error::Invalid(format!(
                        "token_type has {} entries for {n_tokens} tokens",
                        types.len()
                    )));
                }
                Some(types)
            }
        };

        let mut texts = Pool::default();
        let mut attrs = Vec::with_capacity(n_tokens);
        // The bytes of every token whose text is written in the byte alphabet
        // only. These are the tokens that BPE can build.
        let mut alphabetic = Pool::default();
        let mut alphabetic_of = Vec::with_capacity(n_tokens);
        let mut scratch = Vec::new();
        for (i, raw) in tokens.strings().enumerate() {
            let renamed;
            let text = if raw.is_empty() {
                renamed = format!("[EMPTY_{i}]");
                renamed.as_bytes()
            } else {
                raw
            };
            let kind = match types.as_ref().map_or(1, |t| t[i]) {
                1 => attr::NORMAL,
                2 => attr::UNKNOWN,
                3 => attr::CONTROL,
                4 => attr::USER_DEFINED,
                5 => attr::UNUSED,
                6 => {
                    return Err(Error::Invalid(format!(
                        "token {i} is a byte token; byte-fallback vocabularies are not supported"
                    )));
                }
                _ => 0,
            };
            scratch.clear();
            let is_alphabetic = decode_alphabet(text, &mut scratch);
            if kind == attr::NORMAL && !is_alphabetic {
                return Err(Error::Invalid(format!(
                    "token {i} ({}) is a normal token with a character that is no byte of the \
                     byte-level alphabet",
                    String::from_utf8_lossy(text)
                )));
            }
            alphabetic_of.push(is_alphabetic);
            alphabetic.push(if is_alphabetic { &scratch } else { b"" });
            texts.push(text);
            attrs.push(kind);
        }
        if texts.len() != n_tokens {
            return Err(Error::Invalid("the token list is shorter than it says".into()));
        }

        // The ids of the tokens by their bytes. A text that is not alphabetic has
        // no entry, and two tokens with the same text are a damaged vocabulary.
        let mut by_bytes: ByteMap = HashMap::with_capacity_and_hasher(n_tokens, Default::default());
        let mut by_text: ByteMap = HashMap::with_capacity_and_hasher(n_tokens, Default::default());
        for (i, is_alphabetic) in alphabetic_of.iter().enumerate() {
            if let Some(other) = by_text.insert(texts.get(i), i as u32) {
                return Err(Error::Invalid(format!(
                    "tokens {other} and {i} have the same text ({})",
                    String::from_utf8_lossy(texts.get(i))
                )));
            }
            if *is_alphabetic {
                by_bytes.insert(alphabetic.get(i), i as u32);
            }
        }

        let mut byte_token = [0u32; 256];
        for (byte, slot) in byte_token.iter_mut().enumerate() {
            let bytes = [byte as u8];
            *slot = *by_bytes.get(&bytes[..]).ok_or_else(|| {
                Error::Invalid(format!("the vocabulary has no token for the byte {byte:#04x}"))
            })?;
        }

        let merges = load_merges(gguf, &by_bytes)?;

        let mut special = SpecialTokens {
            bos: token_id(gguf, "tokenizer.ggml.bos_token_id", n_tokens)?,
            eos: token_id(gguf, "tokenizer.ggml.eos_token_id", n_tokens)?,
            eot: token_id(gguf, "tokenizer.ggml.eot_token_id", n_tokens)?,
            eom: token_id(gguf, "tokenizer.ggml.eom_token_id", n_tokens)?,
            unk: token_id(gguf, "tokenizer.ggml.unknown_token_id", n_tokens)?,
            sep: token_id(gguf, "tokenizer.ggml.seperator_token_id", n_tokens)?,
            padding: token_id(gguf, "tokenizer.ggml.padding_token_id", n_tokens)?,
            mask: token_id(gguf, "tokenizer.ggml.mask_token_id", n_tokens)?,
            fim_prefix: token_id(gguf, "tokenizer.ggml.fim_pre_token_id", n_tokens)?,
            fim_suffix: token_id(gguf, "tokenizer.ggml.fim_suf_token_id", n_tokens)?,
            fim_middle: token_id(gguf, "tokenizer.ggml.fim_mid_token_id", n_tokens)?,
            fim_pad: token_id(gguf, "tokenizer.ggml.fim_pad_token_id", n_tokens)?,
            fim_repo: token_id(gguf, "tokenizer.ggml.fim_rep_token_id", n_tokens)?,
            fim_file_sep: token_id(gguf, "tokenizer.ggml.fim_sep_token_id", n_tokens)?,
        };

        // Tokens that the file does not name are found by their text. A token
        // found this way is a control token whatever the file says about it.
        let mut by_name = |slot: &mut Option<u32>, names: &[&str]| {
            if slot.is_some() {
                return;
            }
            if let Some(id) = names.iter().find_map(|n| by_text.get(n.as_bytes())) {
                *slot = Some(*id);
                attrs[*id as usize] |= attr::CONTROL;
            }
        };
        by_name(&mut special.eot, &["<|im_end|>", "<|endoftext|>"]);
        by_name(&mut special.fim_prefix, &["<|fim_prefix|>"]);
        by_name(&mut special.fim_suffix, &["<|fim_suffix|>"]);
        by_name(&mut special.fim_middle, &["<|fim_middle|>"]);
        by_name(&mut special.fim_pad, &["<|fim_pad|>"]);
        by_name(&mut special.fim_repo, &["<|fim_repo|>", "<|repo_name|>"]);
        by_name(&mut special.fim_file_sep, &["<|file_sep|>"]);

        // The ids that end a generation: the end markers by name, and the ones
        // the model names for its end and for the filler of fill-in-the-middle.
        let mut end_of_generation: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|name| by_text.get(name.as_bytes()).copied())
            .collect();
        for id in end_of_generation.iter() {
            attrs[*id as usize] |= attr::CONTROL;
        }
        end_of_generation.extend(
            [
                special.fim_pad,
                special.fim_repo,
                special.fim_file_sep,
                special.eos,
                special.eot,
                special.eom,
            ]
            .into_iter()
            .flatten(),
        );
        end_of_generation.sort_unstable();
        end_of_generation.dedup();

        let add_bos = flag(gguf, "tokenizer.ggml.add_bos_token")?.unwrap_or(false);
        let add_eos = flag(gguf, "tokenizer.ggml.add_eos_token")?.unwrap_or(false);
        if add_bos && special.bos.is_none() {
            return Err(Error::Invalid(
                "the model adds a beginning-of-text token but has no 'tokenizer.ggml.bos_token_id'"
                    .into(),
            ));
        }
        if add_eos && special.eos.is_none() {
            return Err(Error::Invalid(
                "the model adds an end-of-text token but has no 'tokenizer.ggml.eos_token_id'"
                    .into(),
            ));
        }

        drop((by_bytes, by_text));
        // What a token writes into text: its bytes if it is normal, its text if
        // it is one that the text can name, nothing for the rest.
        let mut pieces = Pool::default();
        for (i, kind) in attrs.iter().enumerate() {
            pieces.push(if kind & attr::SPECIAL != 0 {
                texts.get(i)
            } else if kind & attr::NORMAL != 0 {
                alphabetic.get(i)
            } else {
                b""
            });
        }

        let vocab = Self {
            texts,
            pieces,
            attrs,
            special,
            end_of_generation,
            add_bos,
            add_eos,
            byte_token,
            merges,
        };
        Ok((vocab, pre))
    }

    pub(crate) fn len(&self) -> usize {
        self.texts.len()
    }
}

/// Reads the merge list. Each entry is two token texts separated by a space;
/// the model gives them in the order they apply.
fn load_merges(gguf: &Gguf, by_bytes: &ByteMap) -> Result<Merges> {
    let list = gguf
        .array("tokenizer.ggml.merges")
        .filter(|a| a.elem_type() == ValueType::String)
        .ok_or(Error::MissingKey { key: "tokenizer.ggml.merges" })?;
    let mut merges = Merges::with_capacity(list.len() as usize);
    let (mut left, mut right, mut joined) = (Vec::new(), Vec::new(), Vec::new());
    for (rank, entry) in list.strings().enumerate() {
        // llama.cpp splits at the first space after the first character.
        let Some(space) = entry.iter().skip(1).position(|b| *b == b' ').map(|p| p + 1) else {
            continue;
        };
        left.clear();
        right.clear();
        if !decode_alphabet(&entry[..space], &mut left)
            || !decode_alphabet(&entry[space + 1..], &mut right)
        {
            continue;
        }
        // A merge of something that is no token can never apply: the words
        // that merges work on are made of tokens.
        let (Some(&l), Some(&r)) = (by_bytes.get(&left[..]), by_bytes.get(&right[..])) else {
            continue;
        };
        joined.clear();
        joined.extend_from_slice(&left);
        joined.extend_from_slice(&right);
        let Some(&token) = by_bytes.get(&joined[..]) else {
            return Err(Error::Invalid(format!(
                "merge {rank} ({}) joins two tokens into something that is no token; \
                 the encoder relies on every merge producing a token",
                String::from_utf8_lossy(entry)
            )));
        };
        merges.insert(l, r, Merge { rank: rank as u32, token });
    }
    Ok(merges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_byte_characters_to_bytes_like_gpt2() {
        assert_eq!(byte_of_char('!' as u32), Some(0x21));
        assert_eq!(byte_of_char('~' as u32), Some(0x7E));
        assert_eq!(byte_of_char('¡' as u32), Some(0xA1));
        assert_eq!(byte_of_char('ÿ' as u32), Some(0xFF));
        // Bytes that are not printable are written as U+0100 and up, in order:
        // NUL is Ā, space is Ġ, newline is Ċ, DEL is ġ, 0xAD is Ń.
        assert_eq!(byte_of_char(0x100), Some(0x00));
        assert_eq!(byte_of_char('Ġ' as u32), Some(0x20));
        assert_eq!(byte_of_char('Ċ' as u32), Some(0x0A));
        assert_eq!(byte_of_char('ġ' as u32), Some(0x7F));
        assert_eq!(byte_of_char('Ń' as u32), Some(0xAD));
        assert_eq!(byte_of_char(0x143), Some(0xAD));
        assert_eq!(byte_of_char(0x144), None);
        assert_eq!(byte_of_char(0x20), None);
        assert_eq!(byte_of_char('日' as u32), None);
    }

    #[test]
    fn decodes_texts_of_the_byte_alphabet() {
        let mut out = Vec::new();
        assert!(decode_alphabet("ĠhelloĊ".as_bytes(), &mut out));
        assert_eq!(out, b" hello\n");
        let mut out = vec![b'x'];
        assert!(!decode_alphabet("a日".as_bytes(), &mut out));
        assert_eq!(out, b"x", "a failed decode leaves the buffer as it was");
        assert!(!decode_alphabet(&[0xFF], &mut out));
    }

    #[test]
    fn hashes_byte_strings_of_every_length() {
        let keys: Vec<Vec<u8>> =
            (0..40usize).map(|len| (0..len).map(|i| (i * 7) as u8).collect()).collect();
        let mut map: ByteMap = HashMap::default();
        for (i, key) in keys.iter().enumerate() {
            map.insert(key, i as u32);
        }
        assert_eq!(map.len(), 40);
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(map[&key[..]], i as u32);
        }
    }
}
