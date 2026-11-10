//! The pre-tokenizer `qwen35`: text into the words that BPE merges inside.
//!
//! It is this regular expression, from the model's `tokenizer.json`:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//! | \p{N}
//! | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! llama.cpp does not run the expression; it runs a hand-written scanner
//! (`unicode_regex_split_custom_qwen35`) that follows it alternative by
//! alternative, and this is that scanner on UTF-8 bytes. Where the scanner and
//! the expression could differ, the scanner is what the engine reproduces, and
//! the tests compare with llama.cpp itself.

use crate::unicode::{Classes, cpt_at};
use crate::unicode_table::{LETTER, MARK, NUMBER, OTHER, SPACE};

/// Cuts `text` into words and calls `emit` with each, in order. The words
/// cover the text without gaps. `text` must be valid UTF-8 by llama.cpp's
/// rules, that is, the result of [`crate::unicode::sanitize`].
pub(crate) fn split<'a>(text: &'a [u8], classes: &Classes, mut emit: impl FnMut(&'a [u8])) {
    let mut start = 0;
    while start < text.len() {
        let end = word_end(text, start, classes);
        emit(&text[start..end]);
        start = end;
    }
}

/// Whether `c` is the letter `lower` or its capital, for the ASCII letters the
/// contractions are made of. llama.cpp lower-cases with its own table, which
/// maps no other code point to one of these letters.
#[inline(always)]
fn is_ci(c: u32, lower: u8) -> bool {
    c == u32::from(lower) || c == u32::from(lower - 32)
}

/// Where the word that starts at `p` ends. `p` is the start of a character.
fn word_end(text: &[u8], p: usize, classes: &Classes) -> usize {
    let n = text.len();
    let (c0, l0) = cpt_at(text, p);
    let k0 = classes.class(c0);
    let p1 = p + l0;

    // 's 't 're 've 'm 'll 'd, in either case
    if c0 == u32::from(b'\'') && p1 < n {
        let (c1, l1) = cpt_at(text, p1);
        if is_ci(c1, b's') || is_ci(c1, b't') || is_ci(c1, b'm') || is_ci(c1, b'd') {
            return p1 + l1;
        }
        let p2 = p1 + l1;
        if p2 < n {
            let (c2, l2) = cpt_at(text, p2);
            if (is_ci(c1, b'r') && is_ci(c2, b'e'))
                || (is_ci(c1, b'v') && is_ci(c2, b'e'))
                || (is_ci(c1, b'l') && is_ci(c2, b'l'))
            {
                return p2 + l2;
            }
        }
    }

    // One character that is not a line break or a number, then letters and
    // marks. The first character may itself be the first letter.
    if c0 != u32::from(b'\r') && c0 != u32::from(b'\n') && k0 != NUMBER {
        let starts_run = |k: u8| k == LETTER || k == MARK;
        let next_starts_run = p1 < n && starts_run(classes.class(cpt_at(text, p1).0));
        if starts_run(k0) || next_starts_run {
            let mut q = p1;
            while q < n {
                let (c, l) = cpt_at(text, q);
                if !starts_run(classes.class(c)) {
                    break;
                }
                q += l;
            }
            return q;
        }
    }

    // A single number character; the model reads digits one at a time.
    if k0 == NUMBER {
        return p1;
    }

    // An optional space, then characters that are none of white space, letter,
    // mark and number, then any line breaks. After a space the run starts at
    // the next character, and when that one is a letter, mark, number or white
    // space the space stands alone (below).
    let after_space = if c0 == u32::from(b' ') { Some(p1) } else { None };
    let first_k = match after_space {
        Some(q) if q < n => classes.class(cpt_at(text, q).0),
        Some(_) => OTHER,
        None => k0,
    };
    if first_k == OTHER {
        let mut q = after_space.unwrap_or(p);
        while q < n {
            let (c, l) = cpt_at(text, q);
            if classes.class(c) != OTHER {
                break;
            }
            q += l;
        }
        while q < n && (text[q] == b'\r' || text[q] == b'\n') {
            q += 1;
        }
        return q;
    }

    // What is left starts with white space. Line breaks end a word: the word
    // reaches the last one of the white space run.
    let mut q = p;
    let mut last_start = p;
    let mut count = 0usize;
    let mut after_last_break = 0;
    while q < n {
        let (c, l) = cpt_at(text, q);
        if classes.class(c) != SPACE {
            break;
        }
        last_start = q;
        q += l;
        count += 1;
        if c == u32::from(b'\r') || c == u32::from(b'\n') {
            after_last_break = q;
        }
    }
    if after_last_break > 0 {
        return after_last_break;
    }
    // A run of white space that is followed by something else leaves its last
    // character to start the next word ("  x" is " " and " x"); at the end of
    // the text it is taken whole.
    if count > 1 && q < n {
        return last_start;
    }
    q.max(p1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unicode::classes;

    fn words(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        split(text.as_bytes(), classes(), |w| out.push(std::str::from_utf8(w).unwrap()));
        out
    }

    #[test]
    fn splits_english_like_the_regular_expression() {
        assert_eq!(words("Hello world"), ["Hello", " world"]);
        assert_eq!(words("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(
            words("don't I'LL we're THEY'VE it's"),
            ["don", "'t", " I", "'LL", " we", "'re", " THEY", "'VE", " it", "'s"]
        );
        assert_eq!(words("'x 'ab '"), ["'x", " '", "ab", " '"]);
        // A contraction ends the word, in any case; without it the letters after
        // the apostrophe would belong to it.
        assert_eq!(
            words("it'Sx we'REx I'LLx"),
            ["it", "'S", "x", " we", "'RE", "x", " I", "'LL", "x"]
        );
    }

    #[test]
    fn reads_numbers_one_digit_at_a_time() {
        assert_eq!(
            words("in 2024 or 3.14"),
            ["in", " ", "2", "0", "2", "4", " or", " ", "3", ".", "1", "4"]
        );
        assert_eq!(words("x٣½"), ["x", "٣", "½"]);
    }

    #[test]
    fn keeps_combining_marks_with_their_letters() {
        assert_eq!(words("cafe\u{0301} ok"), ["cafe\u{0301}", " ok"]);
        // A mark after a space starts a word, as a letter would.
        assert_eq!(words(" \u{0301}a"), [" \u{0301}a"]);
        assert_eq!(words("\u{0301}\u{0301}"), ["\u{0301}\u{0301}"]);
    }

    #[test]
    fn groups_white_space_so_that_the_last_space_joins_the_next_word() {
        assert_eq!(words("a  b"), ["a", " ", " b"]);
        assert_eq!(words("a     b"), ["a", "    ", " b"]);
        assert_eq!(words("a   "), ["a", "   "]);
        assert_eq!(words("a\tb"), ["a", "\tb"]);
        assert_eq!(words("a\t\tb"), ["a", "\t", "\tb"]);
        assert_eq!(words("  "), ["  "]);
        assert_eq!(words(" "), [" "]);
    }

    #[test]
    fn ends_a_word_at_the_last_line_break_of_a_white_space_run() {
        assert_eq!(words("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(words("a \n \nb"), ["a", " \n \n", "b"]);
        assert_eq!(words("a\n  b"), ["a", "\n", " ", " b"]);
        assert_eq!(words("a\r\nb"), ["a", "\r\n", "b"]);
        assert_eq!(words("\n"), ["\n"]);
        assert_eq!(words("a\n\t b"), ["a", "\n", "\t", " b"]);
    }

    #[test]
    fn takes_punctuation_with_one_leading_space_and_trailing_line_breaks() {
        assert_eq!(words("a !! b"), ["a", " !!", " b"]);
        assert_eq!(words("a!\n\nb"), ["a", "!\n\n", "b"]);
        assert_eq!(words("x = (y);\n"), ["x", " =", " (", "y", ");\n"]);
        assert_eq!(words("a  !"), ["a", " ", " !"]);
        assert_eq!(words("é€£"), ["é", "€£"]);
        // Only one space joins the run; the other characters of the run are
        // not white space.
        assert_eq!(words("a $€$"), ["a", " $€$"]);
    }

    #[test]
    fn lets_a_single_character_before_letters_join_them() {
        assert_eq!(words("$foo-bar"), ["$foo", "-bar"]);
        // The punctuation alternative comes before the letter one, so it takes
        // the whole run, and a space or a second mark never reaches the letters.
        assert_eq!(words("a #bar @@x"), ["a", " #", "bar", " @@", "x"]);
        assert_eq!(words("\"quoted\""), ["\"quoted", "\""]);
        // A line break is never a prefix.
        assert_eq!(words("\nx"), ["\n", "x"]);
        assert_eq!(words("\rx"), ["\r", "x"]);
    }

    #[test]
    fn handles_non_ascii_text() {
        assert_eq!(words("größer ß"), ["größer", " ß"]);
        assert_eq!(words("日本語のテキスト"), ["日本語のテキスト"]);
        assert_eq!(words("a😀b"), ["a", "😀b"]);
        assert_eq!(words("😀😀 😀"), ["😀😀", " 😀"]);
        assert_eq!(words("a\u{00A0}b"), ["a", "\u{00A0}b"]);
        assert_eq!(words("a\u{3000}\u{3000}b"), ["a", "\u{3000}", "\u{3000}b"]);
    }

    #[test]
    fn cuts_the_whole_text_without_gaps() {
        for text in ["", "x", "  \n\t x'  'y\r\n\r\n  ", "a😀 b\u{0301}\u{0301}'sx 12 \n"] {
            let joined: String = words(text).concat();
            assert_eq!(joined, text);
        }
    }
}
