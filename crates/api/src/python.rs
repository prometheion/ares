//! JSON values written as Python writes them, for prompts that must be the
//! ones Flide (and vLLM) give a model: `json.dumps(value, ensure_ascii=False)`
//! and `str(value)` of the value Python's `json` module reads.
//!
//! Flide renders a decision's state with `json.dumps` and its instructions,
//! descriptions and levels with `str()` (`flide/formats/next_token.py`,
//! <https://github.com/OmniJev/Flide>, commit
//! caffe660df88db747d6da9f95f7d6759fea38698). The same request gives the same
//! prompt here only if the text is the same byte for byte.

use std::fmt::Write;

use crate::Json;

/// `json.dumps(value, ensure_ascii=False)`: `", "` between items, `": "` after
/// a key, text other than ASCII as it is.
pub(crate) fn dumps(value: &Json) -> String {
    let mut out = String::new();
    dump(value, &mut out);
    out
}

fn dump(value: &Json, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Int(i) => write!(out, "{i}").expect("a string takes any text"),
        Json::Uint(u) => write!(out, "{u}").expect("a string takes any text"),
        Json::Float(f) => out.push_str(&float_repr(*f)),
        Json::String(s) => dump_string(s, out),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump(item, out);
            }
            out.push(']');
        }
        Json::Object(members) => {
            out.push('{');
            for (i, (key, item)) in kept(members).into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_string(key, out);
                out.push_str(": ");
                dump(item, out);
            }
            out.push('}');
        }
    }
}

/// The members of an object as the dict Python's `json` module reads: the
/// last value of a repeated key, at the place of the first.
fn kept(members: &[(String, Json)]) -> Vec<(&str, &Json)> {
    let mut kept: Vec<(&str, &Json)> = Vec::with_capacity(members.len());
    for (key, item) in members {
        match kept.iter_mut().find(|(k, _)| *k == key.as_str()) {
            Some(entry) => entry.1 = item,
            None => kept.push((key, item)),
        }
    }
    kept
}

/// A JSON string as `json.dumps` writes it with `ensure_ascii=False`.
fn dump_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).expect("a string takes any text")
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `str(value)` of the value Python's `json` module reads from `value`: text
/// as it is, `None`, `True`, `False`, numbers as Python writes them, and a list
/// or a dict as Python prints it, with the `repr` of everything inside.
pub(crate) fn str_of(value: &Json) -> String {
    match value {
        Json::String(s) => s.clone(),
        other => {
            let mut out = String::new();
            repr(other, &mut out);
            out
        }
    }
}

fn repr(value: &Json, out: &mut String) {
    match value {
        Json::Null => out.push_str("None"),
        Json::Bool(true) => out.push_str("True"),
        Json::Bool(false) => out.push_str("False"),
        Json::Int(i) => write!(out, "{i}").expect("a string takes any text"),
        Json::Uint(u) => write!(out, "{u}").expect("a string takes any text"),
        Json::Float(f) => out.push_str(&float_repr(*f)),
        Json::String(s) => repr_string(s, out),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr(item, out);
            }
            out.push(']');
        }
        Json::Object(members) => {
            out.push('{');
            for (i, (key, item)) in kept(members).into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr_string(key, out);
                out.push_str(": ");
                repr(item, out);
            }
            out.push('}');
        }
    }
}

/// `repr()` of a string: single quotes, unless the text holds a single quote
/// and no double one; the escapes of Python for the backslash, the quote, the
/// line breaks, tabs and the characters Python does not print.
fn repr_string(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !printable(c) => {
                let code = c as u32;
                if code <= 0xff {
                    write!(out, "\\x{code:02x}")
                } else if code <= 0xffff {
                    write!(out, "\\u{code:04x}")
                } else {
                    write!(out, "\\U{code:08x}")
                }
                .expect("a string takes any text")
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// Whether Python prints `c` as it is in a `repr`: not a control character, a
/// format character, or a separator other than the space. The common ones;
/// Python's table of every Unicode category is not repeated here.
fn printable(c: char) -> bool {
    !(c.is_control()
        || matches!(
            c,
            '\u{a0}'
                | '\u{ad}'
                | '\u{600}'..='\u{605}'
                | '\u{61c}'
                | '\u{6dd}'
                | '\u{70f}'
                | '\u{1680}'
                | '\u{180e}'
                | '\u{2000}'..='\u{200f}'
                | '\u{2028}'..='\u{202f}'
                | '\u{205f}'..='\u{2064}'
                | '\u{2066}'..='\u{206f}'
                | '\u{3000}'
                | '\u{e000}'..='\u{f8ff}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
        ))
}

/// `repr()` of a float: the shortest digits that read back as the same value,
/// in positional notation from 1e-4 up to below 1e16 (with `.0` for a whole
/// number) and in scientific notation outside (`1e-05`, `1.5e+16`).
pub(crate) fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` gives the shortest digits that round-trip: `d.ddde±x`.
    let sci = format!("{:e}", f.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("a float in scientific notation");
    let exponent: i32 = exponent.parse().expect("an exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // The value is 0.<digits> times ten to `point`.
    let point = exponent + 1;
    let mut out = String::new();
    if f < 0.0 {
        out.push('-');
    }
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        write!(out, "e{sign}{:02}", exponent.abs()).expect("a string takes any text");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> Json {
        Json::String(text.into())
    }

    #[test]
    fn floats_are_written_as_python_writes_them() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-2.25, "-2.25"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (123456.789, "123456.789"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e20, "1.2345e+20"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0 / 3.0, "0.3333333333333333"),
        ];
        for (value, text) in cases {
            assert_eq!(float_repr(value), text, "{value}");
        }
    }

    #[test]
    fn json_is_dumped_with_pythons_separators_and_escapes() {
        let value = Json::Object(vec![
            ("a".into(), Json::Array(vec![Json::Int(1), Json::Bool(true), Json::Null])),
            ("ünï".into(), s("tab\there \"q\" \\ \u{1} é")),
            ("e".into(), Json::Object(vec![])),
            ("f".into(), Json::Float(2.0)),
        ]);
        assert_eq!(
            dumps(&value),
            "{\"a\": [1, true, null], \"ünï\": \"tab\\there \\\"q\\\" \\\\ \\u0001 é\", \"e\": {}, \
             \"f\": 2.0}"
        );
    }

    #[test]
    fn str_of_a_value_is_what_python_prints() {
        assert_eq!(str_of(&s("as it is")), "as it is");
        assert_eq!(str_of(&Json::Null), "None");
        assert_eq!(str_of(&Json::Bool(false)), "False");
        assert_eq!(str_of(&Json::Float(1e-5)), "1e-05");
        let value = Json::Object(vec![
            ("k".into(), s("it's")),
            ("l".into(), Json::Array(vec![s("a\nb"), Json::Null, Json::Float(0.5)])),
            ("q".into(), s("both ' and \"")),
            ("k".into(), Json::Int(2)),
        ]);
        assert_eq!(str_of(&value), "{'k': 2, 'l': ['a\\nb', None, 0.5], 'q': 'both \\' and \"'}");
        assert_eq!(str_of(&Json::Array(vec![s("it's")])), "[\"it's\"]");
        assert_eq!(str_of(&Json::Array(vec![s("\u{7f}\u{a0}é")])), "['\\x7f\\xa0é']");
    }
}
