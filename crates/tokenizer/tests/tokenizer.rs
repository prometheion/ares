//! Encoding, decoding and loading, on a vocabulary built in the test.

mod common;

use ares_tokenizer::{EncodeOptions, Error, TokenKind};
use common::*;

fn plain() -> EncodeOptions {
    EncodeOptions { add_special: false, parse_special: false }
}

fn parsing() -> EncodeOptions {
    EncodeOptions { add_special: false, parse_special: true }
}

fn byte(b: u8) -> u32 {
    u32::from(b)
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(byte).collect()
}

#[test]
fn can_be_shared_between_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<ares_tokenizer::Tokenizer>();
    shared::<ares_tokenizer::ChatTemplate>();
}

#[test]
fn applies_the_merges_in_the_order_of_the_file() {
    let t = tiny_tokenizer("order");
    assert_eq!(t.encode("hello world", plain()), [HELLO, SPACE_WORLD]);
    assert_eq!(t.encode("hello", plain()), [HELLO]);
    assert_eq!(t.encode("hell", plain()), [HELL]);
    // "he" (merge 0) first, then "ll" (merge 1) twice from the left, then "he" + "ll" (merge 2).
    assert_eq!(t.encode("hellll", plain()), [HELL, LL]);
    assert_eq!(t.encode("helo", plain()), [HE, byte(b'l'), byte(b'o')]);
    assert_eq!(t.encode("xyz", plain()), bytes("xyz"));
    assert_eq!(t.encode("", plain()), Vec::<u32>::new());
}

#[test]
fn cuts_the_text_into_words_before_merging() {
    let t = tiny_tokenizer("words");
    // "hello" and " world" never merge across the space, although "o" + " " ... exists as bytes.
    assert_eq!(t.encode("hello  world", plain()), [HELLO, byte(b' '), SPACE_WORLD]);
    assert_eq!(t.pre_tokenize("hello  world!\n"), [&b"hello"[..], b" ", b" world", b"!\n"]);
    // White space is a word of its own, as far as the last line break; the merges
    // for two line breaks and two spaces show where the words end.
    assert_eq!(t.encode("a\n\n\nb", plain()), [byte(b'a'), NEWLINES, byte(b'\n'), byte(b'b')]);
    assert_eq!(
        t.encode("a    b", plain()),
        [byte(b'a'), SPACES, byte(b' '), byte(b' '), byte(b'b')]
    );
    // The last space of a run joins the word that follows, so these two do not merge.
    assert_eq!(
        t.encode("a\n\n  b", plain()),
        [byte(b'a'), NEWLINES, byte(b' '), byte(b' '), byte(b'b')]
    );
    assert_eq!(t.encode("hello\nworld", plain()), [HELLO, byte(b'\n'), byte(b'w'), OR, LD]);
}

#[test]
fn encodes_multibyte_text_through_its_bytes() {
    let t = tiny_tokenizer("utf8");
    assert_eq!(t.encode("ü", plain()), [U_UMLAUT]);
    assert_eq!(t.encode("€", plain()), [0xE2, 0x82, 0xAC]);
    assert_eq!(t.encode("😀", plain()), [0xF0, 0x9F, 0x98, 0x80]);
}

#[test]
fn reads_each_byte_that_is_not_utf8_as_a_replacement_character() {
    let t = tiny_tokenizer("invalid");
    let replacement = t.encode("\u{FFFD}", plain());
    assert_eq!(replacement, [0xEF, 0xBF, 0xBD]);
    assert_eq!(t.encode(b"a\xFFb", plain()), [byte(b'a'), 0xEF, 0xBF, 0xBD, byte(b'b')]);
    // A truncated three-byte sequence is two replacements, not one.
    assert_eq!(t.encode(b"\xE2\x82", plain()), [replacement.clone(), replacement.clone()].concat());
    // A surrogate is accepted and written as the three bytes it came in.
    assert_eq!(t.encode([0xED, 0xA0, 0x80], plain()), [0xED, 0xA0, 0x80]);
    // And the bytes of the replacement decode to the replacement, not to the bytes given.
    assert_eq!(t.decode(&t.encode(b"\xFF", plain()), true).unwrap(), "\u{FFFD}".as_bytes());
}

#[test]
fn recognises_control_tokens_only_when_asked() {
    let t = tiny_tokenizer("parse");
    assert_eq!(t.encode("a<|im_start|>b", parsing()), [byte(b'a'), IM_START, byte(b'b')]);
    let without = t.encode("<|im_start|>", plain());
    assert!(!without.contains(&IM_START));
    assert_eq!(t.decode(&without, true).unwrap(), b"<|im_start|>", "it is text then");
}

#[test]
fn always_recognises_user_defined_tokens() {
    let t = tiny_tokenizer("user");
    for options in [plain(), parsing()] {
        assert_eq!(t.encode("<think>hello</think>", options), [THINK, HELLO, END_THINK]);
    }
    assert_eq!(t.kind(THINK), Some(TokenKind::UserDefined));
    assert_eq!(t.kind(IM_START), Some(TokenKind::Control));
    assert_eq!(t.kind(UNUSED), Some(TokenKind::Unused));
    assert_eq!(t.kind(0), Some(TokenKind::Normal));
    assert_eq!(t.kind(N_TOKENS as u32), None);
}

#[test]
fn text_that_looks_like_an_unused_token_is_ordinary_text() {
    let t = tiny_tokenizer("unused");
    let ids = t.encode("[PAD270]", parsing());
    assert!(!ids.contains(&UNUSED));
    assert_eq!(t.decode(&ids, true).unwrap(), b"[PAD270]");
}

#[test]
fn cuts_special_tokens_longest_first_and_never_inside_one_already_cut() {
    // "ab" and "bcd" overlap in "abcd": the longer one is cut first and the shorter
    // can no longer be found, as in llama.cpp; a scan from the left would find "ab".
    let mut texts = token_texts();
    texts.extend(["ab".to_owned(), "bcd".to_owned()]);
    let mut types = token_types();
    types.extend([3, 3]);
    let t = tiny().tokens(texts).types(types).load("overlap").unwrap();
    let (ab, bcd) = (N_TOKENS as u32, N_TOKENS as u32 + 1);
    assert_eq!(t.encode("abcd", parsing()), [byte(b'a'), bcd]);
    assert_eq!(t.encode("ab", parsing()), [ab]);
    assert_eq!(t.encode("abab bcd", parsing()), [ab, ab, byte(b' '), bcd]);
    assert_eq!(t.special_token_id("bcd"), Some(bcd));
    assert_eq!(t.special_token_id("hello"), None);
}

#[test]
fn adds_the_tokens_the_model_asks_for() {
    let t = tiny()
        .u32("tokenizer.ggml.bos_token_id", IM_START)
        .bool("tokenizer.ggml.add_bos_token", true)
        .bool("tokenizer.ggml.add_eos_token", true)
        .load("special")
        .unwrap();
    assert!(t.adds_bos() && t.adds_eos());
    let with = EncodeOptions { add_special: true, parse_special: false };
    assert_eq!(t.encode("hello", with), [IM_START, HELLO, IM_END]);
    assert_eq!(t.encode("", with), [IM_START, IM_END]);
    assert_eq!(t.encode("hello", plain()), [HELLO]);

    let t = tiny_tokenizer("nospecial");
    assert!(!t.adds_bos() && !t.adds_eos());
    assert_eq!(
        t.encode("hello", EncodeOptions { add_special: true, parse_special: false }),
        [HELLO]
    );
}

#[test]
fn knows_the_special_tokens_of_the_model() {
    let t = tiny_tokenizer("ids");
    let special = t.special_tokens();
    assert_eq!(special.eos, Some(IM_END));
    assert_eq!(special.padding, Some(UNUSED));
    // Not in the file, but found by name.
    assert_eq!(special.eot, Some(IM_END));
    assert_eq!(special.bos, None, "a missing bos is not made up");
    assert!(t.is_end_of_generation(IM_END));
    assert!(!t.is_end_of_generation(IM_START));
    assert!(!t.is_end_of_generation(HELLO));
}

#[test]
fn decodes_to_the_bytes_that_were_encoded() {
    let t = tiny_tokenizer("roundtrip");
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let words = [
        "hello",
        " world",
        "hell",
        "ld",
        "or",
        "ü",
        "€",
        "😀",
        " ",
        "  ",
        "\n",
        "\t",
        "x",
        "'s",
        "12",
        "!",
        "<think>",
        "</think>",
        "<|im_start|>",
        "[PAD270]",
    ];
    for _ in 0..500 {
        let n = 1 + rng.below(12);
        let text: String = (0..n).map(|_| words[rng.below(words.len())]).collect();
        for options in [plain(), parsing()] {
            let ids = t.encode(&text, options);
            assert_eq!(t.decode(&ids, true).unwrap(), text.as_bytes(), "{text:?} {options:?}");
        }
    }
}

#[test]
fn leaves_control_tokens_out_of_text_unless_asked() {
    let t = tiny_tokenizer("hide");
    let ids = [IM_START, byte(b'h'), THINK, UNUSED, IM_END];
    // Unused tokens are never text.
    assert_eq!(t.decode(&ids, true).unwrap(), b"<|im_start|>h<think><|im_end|>");
    assert_eq!(t.decode(&ids, false).unwrap(), b"h<think>");
    assert_eq!(t.piece(IM_START, false).unwrap(), b"");
    assert_eq!(t.piece(IM_START, true).unwrap(), b"<|im_start|>");
    assert_eq!(t.piece(U_UMLAUT, false).unwrap(), "ü".as_bytes());
    assert_eq!(t.token_text(U_UMLAUT).unwrap(), "Ã¼".as_bytes());
}

#[test]
fn refuses_to_decode_a_token_that_does_not_exist() {
    let t = tiny_tokenizer("range");
    let error = t.decode(&[HELLO, N_TOKENS as u32], true).unwrap_err();
    assert!(
        matches!(error, Error::TokenOutOfRange { id, n_tokens } if id == N_TOKENS as u32 && n_tokens == N_TOKENS),
        "{error}"
    );
    assert!(t.piece(u32::MAX, true).is_err());
}

#[test]
fn refuses_vocabularies_it_does_not_implement_and_names_them() {
    let load = |tiny: common::Tiny, tag: &str| tiny.load(tag).unwrap_err();
    let error = load(tiny().model("llama"), "model");
    assert!(matches!(&error, Error::UnsupportedModel { model } if model == "llama"), "{error}");
    assert!(error.to_string().contains("'llama'"));

    let error = load(tiny().model("bert"), "bert");
    assert!(matches!(error, Error::UnsupportedModel { .. }));

    let error = load(tiny().pre("llama3"), "pre");
    assert!(matches!(&error, Error::UnsupportedPre { pre } if pre == "llama3"), "{error}");
    assert!(error.to_string().contains("'llama3'"));
    // Also one that llama.cpp does not know.
    assert!(matches!(load(tiny().pre("nonsense"), "pre2"), Error::UnsupportedPre { .. }));

    for key in [
        "tokenizer.ggml.add_space_prefix",
        "tokenizer.ggml.remove_extra_whitespaces",
        "tokenizer.ggml.clean_spaces",
    ] {
        let error = load(tiny().bool(key, true), "option");
        assert!(matches!(error, Error::UnsupportedOption { key: k } if k == key), "{key}");
    }
}

#[test]
fn refuses_files_without_what_the_tokenizer_needs() {
    let error = tiny().no_model().load("nomodel").unwrap_err();
    assert!(matches!(error, Error::MissingKey { key: "tokenizer.ggml.model" }), "{error}");
    let error = tiny().no_pre().load("nopre").unwrap_err();
    assert!(matches!(error, Error::MissingKey { key: "tokenizer.ggml.pre" }), "{error}");
    let error = tiny().no_tokens().load("notokens").unwrap_err();
    assert!(matches!(error, Error::MissingKey { key: "tokenizer.ggml.tokens" }), "{error}");
    let error = tiny().no_merges().load("nomerges").unwrap_err();
    assert!(matches!(error, Error::MissingKey { key: "tokenizer.ggml.merges" }), "{error}");
    let error = tiny().model_as_number(3).load("wrongtype").unwrap_err();
    assert!(matches!(error, Error::WrongType { key: "tokenizer.ggml.model", .. }), "{error}");
}

#[test]
fn refuses_a_damaged_vocabulary() {
    let invalid = |tiny: common::Tiny, tag: &str| match tiny.load(tag).unwrap_err() {
        Error::Invalid(what) => what,
        other => panic!("{other}"),
    };
    // A byte without a token.
    let mut texts = token_texts();
    texts[usize::from(b'q')] = "qq".to_owned();
    let what = invalid(tiny().tokens(texts), "nobyte");
    assert!(what.contains("0x71"), "{what}");

    // Two tokens with one text.
    let mut texts = token_texts();
    texts[270] = "<think>".to_owned();
    let what = invalid(tiny().tokens(texts), "duplicate");
    assert!(what.contains("268") && what.contains("270"), "{what}");

    // A merge whose result is no token.
    let mut merges = merge_texts();
    merges.push(format!("{} {}", alphabet("x"), alphabet("y")));
    let what = invalid(tiny().merges(merges), "merge");
    assert!(what.contains("merge 12"), "{what}");

    // A normal token that is not written in the byte alphabet.
    let mut texts = token_texts();
    texts[256] = "日本".to_owned();
    let what = invalid(tiny().tokens(texts), "alphabet");
    assert!(what.contains("token 256"), "{what}");

    // A byte token, which this vocabulary type does not have.
    let mut types = token_types();
    types[5] = 6;
    let what = invalid(tiny().types(types), "bytetype");
    assert!(what.contains("token 5"), "{what}");

    // Too few types, an id outside the vocabulary (a second key of the same name is
    // not allowed in a file, so this one replaces the id of the vocabulary), a bos
    // that the file asks for but lacks.
    assert!(invalid(tiny().types(vec![1, 1]), "fewtypes").contains("token_type"));
    assert!(
        invalid(tiny().u32("tokenizer.ggml.bos_token_id", 5000), "badid").contains("bos_token_id")
    );
    assert!(
        invalid(tiny().bool("tokenizer.ggml.add_bos_token", true), "nobos")
            .contains("bos_token_id")
    );
}

#[test]
fn an_empty_token_gets_a_name_and_does_not_break_the_rest() {
    let mut texts = token_texts();
    texts[270] = String::new();
    let t = tiny().tokens(texts).load("empty").unwrap();
    assert_eq!(t.token_text(270).unwrap(), b"[EMPTY_270]");
    assert_eq!(t.encode("hello", plain()), [HELLO]);
}
