//! Code point classes and UTF-8 decoding exactly as llama.cpp does them.
//!
//! Two details matter for token-for-token agreement. First, llama.cpp's UTF-8
//! decoder is not the standard one: it accepts the surrogates U+D800 to U+DFFF
//! and, on a malformed sequence, turns only the first byte into U+FFFD and
//! tries again at the next byte. Second, its idea of a letter, a number, a
//! combining mark and white space comes from its own tables (see
//! `tools/gen_unicode.py`), not from the Unicode version of this machine.

use std::borrow::Cow;
use std::sync::LazyLock;

use crate::unicode_table::RANGES;

/// The class of a code point, one of [`crate::unicode_table`]'s constants.
pub(crate) type Class = u8;

/// Code point classes in two stages: the block of 256 code points a code point
/// is in, and the classes inside that block. Most blocks are shared, so the
/// whole table is a few kilobytes and stays in the cache.
pub(crate) struct Classes {
    block_of: Box<[u16]>,
    blocks: Box<[[u8; 256]]>,
    /// The classes of ASCII, which is most of the text, one load away.
    ascii: [u8; 128],
}

static CLASSES: LazyLock<Classes> = LazyLock::new(Classes::build);

/// The table of code point classes. Built on first use.
pub(crate) fn classes() -> &'static Classes {
    &CLASSES
}

impl Classes {
    fn build() -> Self {
        let mut flat = vec![0u8; 0x110000];
        for pair in RANGES.windows(2) {
            let ((start, class), (end, _)) = (pair[0], pair[1]);
            flat[start as usize..end as usize].fill(class);
        }
        let mut blocks: Vec<[u8; 256]> = Vec::new();
        let mut block_of = Vec::with_capacity(0x1100);
        for chunk in flat.as_chunks::<256>().0 {
            let chunk = *chunk;
            let index = match blocks.iter().position(|b| *b == chunk) {
                Some(index) => index,
                None => {
                    blocks.push(chunk);
                    blocks.len() - 1
                }
            };
            block_of.push(u16::try_from(index).expect("fewer than 65536 distinct blocks"));
        }
        let ascii = std::array::from_fn(|c| flat[c]);
        Self { block_of: block_of.into(), blocks: blocks.into(), ascii }
    }

    /// The class of `cpt`. Code points above U+10FFFF cannot come out of the
    /// decoder; they are classed as `OTHER` like llama.cpp's undefined ones.
    #[inline(always)]
    pub(crate) fn class(&self, cpt: u32) -> Class {
        if cpt < 128 {
            return self.ascii[cpt as usize];
        }
        match self.block_of.get((cpt >> 8) as usize) {
            Some(block) => self.blocks[usize::from(*block)][(cpt & 0xff) as usize],
            None => crate::unicode_table::OTHER,
        }
    }
}

/// The U+FFFD that stands for every byte that does not start a valid sequence.
const REPLACEMENT: [u8; 3] = [0xEF, 0xBF, 0xBD];

/// Decodes the code point at the start of `bytes` by llama.cpp's rules.
///
/// Returns the code point and the number of bytes it takes, or `None` when the
/// bytes do not start a valid sequence: a continuation byte, a byte above
/// 0xF7, a truncated or overlong sequence, or a value above U+10FFFF. Unlike
/// the standard rules, a three-byte sequence for a surrogate is accepted.
pub(crate) fn decode(bytes: &[u8]) -> Option<(u32, usize)> {
    let b0 = *bytes.first()?;
    let cont = |i: usize| bytes.get(i).copied().filter(|b| b & 0xC0 == 0x80);
    if b0 < 0x80 {
        return Some((u32::from(b0), 1));
    }
    if b0 & 0x40 == 0 {
        return None;
    }
    if b0 & 0x20 == 0 {
        let cpt = (u32::from(b0 & 0x1F) << 6) | u32::from(cont(1)? & 0x3F);
        return (cpt >= 0x80).then_some((cpt, 2));
    }
    if b0 & 0x10 == 0 {
        let cpt = (u32::from(b0 & 0x0F) << 12)
            | (u32::from(cont(1)? & 0x3F) << 6)
            | u32::from(cont(2)? & 0x3F);
        return (cpt >= 0x800).then_some((cpt, 3));
    }
    if b0 & 0x08 == 0 {
        let cpt = (u32::from(b0 & 0x07) << 18)
            | (u32::from(cont(1)? & 0x3F) << 12)
            | (u32::from(cont(2)? & 0x3F) << 6)
            | u32::from(cont(3)? & 0x3F);
        return (0x10000..=0x10FFFF).contains(&cpt).then_some((cpt, 4));
    }
    None
}

/// The text as llama.cpp sees it: every byte that does not start a valid
/// sequence becomes U+FFFD. Valid text is returned as it is, without a copy.
pub(crate) fn sanitize(text: &[u8]) -> Cow<'_, [u8]> {
    // The standard rules accept less than llama.cpp's, so text that passes them
    // needs no work.
    if std::str::from_utf8(text).is_ok() {
        return Cow::Borrowed(text);
    }
    let mut out = Vec::with_capacity(text.len() + 16);
    let mut i = 0;
    while i < text.len() {
        match decode(&text[i..]) {
            Some((_, len)) => {
                out.extend_from_slice(&text[i..i + len]);
                i += len;
            }
            None => {
                out.extend_from_slice(&REPLACEMENT);
                i += 1;
            }
        }
    }
    Cow::Owned(out)
}

/// The code point at `i` of text that [`sanitize`] has accepted, and its length.
///
/// # Panics
///
/// Panics when `i` is not the start of a character of valid text.
#[inline(always)]
pub(crate) fn cpt_at(text: &[u8], i: usize) -> (u32, usize) {
    let b0 = text[i];
    if b0 < 0x80 {
        (u32::from(b0), 1)
    } else if b0 < 0xE0 {
        ((u32::from(b0 & 0x1F) << 6) | u32::from(text[i + 1] & 0x3F), 2)
    } else if b0 < 0xF0 {
        (
            (u32::from(b0 & 0x0F) << 12)
                | (u32::from(text[i + 1] & 0x3F) << 6)
                | u32::from(text[i + 2] & 0x3F),
            3,
        )
    } else {
        (
            (u32::from(b0 & 0x07) << 18)
                | (u32::from(text[i + 1] & 0x3F) << 12)
                | (u32::from(text[i + 2] & 0x3F) << 6)
                | u32::from(text[i + 3] & 0x3F),
            4,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unicode_table::{LETTER, MARK, NUMBER, OTHER, SPACE};

    #[test]
    fn classifies_the_characters_pre_tokenization_cares_about() {
        let c = classes();
        assert_eq!(c.class('a' as u32), LETTER);
        assert_eq!(c.class('Z' as u32), LETTER);
        assert_eq!(c.class('ß' as u32), LETTER);
        assert_eq!(c.class('漢' as u32), LETTER);
        assert_eq!(c.class('7' as u32), NUMBER);
        assert_eq!(c.class('٣' as u32), NUMBER);
        assert_eq!(c.class('½' as u32), NUMBER);
        assert_eq!(c.class(0x0301), MARK);
        assert_eq!(c.class('!' as u32), OTHER);
        assert_eq!(c.class('€' as u32), OTHER);
        assert_eq!(c.class(0x1F600), OTHER);
        for space in [0x09, 0x0A, 0x0D, 0x20, 0x85, 0xA0, 0x2003, 0x3000] {
            assert_eq!(c.class(space), SPACE, "{space:#x}");
        }
        // llama.cpp does not count these as white space.
        for not_space in [0x1C, 0x200B, 0x180E, 0xFEFF] {
            assert_ne!(c.class(not_space), SPACE, "{not_space:#x}");
        }
        assert_eq!(c.class(0x110000), OTHER);
    }

    #[test]
    fn decodes_valid_sequences_of_every_length() {
        assert_eq!(decode(b"a"), Some((0x61, 1)));
        assert_eq!(decode("ß".as_bytes()), Some((0xDF, 2)));
        assert_eq!(decode("€".as_bytes()), Some((0x20AC, 3)));
        assert_eq!(decode("😀".as_bytes()), Some((0x1F600, 4)));
        // A surrogate is accepted although Rust refuses it.
        assert_eq!(decode(&[0xED, 0xA0, 0x80]), Some((0xD800, 3)));
        assert_eq!(decode(b""), None);
    }

    #[test]
    fn refuses_malformed_sequences_as_llama_cpp_does() {
        for bad in [
            &[0x80][..],         // a continuation byte
            &[0xBF],             // the last continuation byte
            &[0xC0, 0x80],       // overlong NUL
            &[0xC1, 0xBF],       // overlong DEL
            &[0xE0, 0x80, 0x80], // overlong
            &[0xF0, 0x80, 0x80, 0x80],
            &[0xF4, 0x90, 0x80, 0x80], // above U+10FFFF
            &[0xF5, 0x80, 0x80, 0x80],
            &[0xF8, 0x80, 0x80, 0x80, 0x80],
            &[0xFF],
            &[0xC3], // truncated
            &[0xE2, 0x82],
            &[0xF0, 0x9F, 0x98],
            &[0xC3, 0x28], // not a continuation byte
        ] {
            assert_eq!(decode(bad), None, "{bad:02x?}");
        }
    }

    #[test]
    fn replaces_each_byte_that_starts_nothing_valid_by_one_replacement() {
        assert!(matches!(sanitize(b"plain"), Cow::Borrowed(_)));
        assert!(matches!(sanitize("héllo 😀".as_bytes()), Cow::Borrowed(_)));
        // A lone continuation byte, and every byte of a truncated sequence.
        assert_eq!(sanitize(b"a\x80b").as_ref(), "a\u{FFFD}b".as_bytes());
        assert_eq!(sanitize(b"x\xE2\x82").as_ref(), "x\u{FFFD}\u{FFFD}".as_bytes());
        assert_eq!(sanitize(b"\xF0\x9F\x98(").as_ref(), "\u{FFFD}\u{FFFD}\u{FFFD}(".as_bytes());
        // Surrogates survive as the three bytes they came in.
        assert_eq!(sanitize(&[0xED, 0xA0, 0x80]).as_ref(), &[0xED, 0xA0, 0x80]);
        assert_eq!(sanitize(&[0xFF, 0xFE]).as_ref(), "\u{FFFD}\u{FFFD}".as_bytes());
    }

    #[test]
    fn reads_code_points_of_sanitized_text() {
        let text = "a\u{DF}\u{20AC}\u{1F600}".as_bytes();
        assert_eq!(cpt_at(text, 0), (0x61, 1));
        assert_eq!(cpt_at(text, 1), (0xDF, 2));
        assert_eq!(cpt_at(text, 3), (0x20AC, 3));
        assert_eq!(cpt_at(text, 6), (0x1F600, 4));
    }
}
