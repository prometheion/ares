//! JSON values with the order of their keys.
//!
//! A chat template sees the tools and the arguments of tool calls as values it
//! can walk, print and turn back into JSON, and the prompt then holds the keys
//! in the order the client wrote them. A map that sorts its keys, as
//! `serde_json::Value` does by default, would change the prompt, so the values
//! the chat module takes are these.

use std::fmt::Write as _;

/// A JSON value. Objects keep their keys in the order they were written in.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` or `false`
    Bool(bool),
    /// A number without a fraction or exponent that fits in 64 bits.
    Int(i64),
    /// Any other number.
    Float(f64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, as its (key, value) pairs in order. A key appears once.
    Object(Vec<(String, Json)>),
}

/// Why a text is not JSON that [`Json::parse`] accepts.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid JSON at byte {offset}: {what}")]
pub struct JsonError {
    /// Where in the text.
    pub offset: usize,
    /// What is wrong.
    pub what: &'static str,
}

/// Arrays and objects nested deeper than this are refused instead of being
/// followed down the stack.
const MAX_DEPTH: usize = 128;

impl Json {
    /// Parses JSON text.
    ///
    /// Numbers are integers when they have no fraction and no exponent and fit
    /// in 64 bits, and floats otherwise. An integer above `i64::MAX` is read as
    /// its two's complement, a negative number, because that is the number
    /// llama.cpp's templates see when a client sends one (its JSON reader keeps
    /// it unsigned and the conversion to a template value wraps it). An object
    /// that repeats a key keeps the first position and takes the last value.
    ///
    /// # Errors
    ///
    /// The text is not valid JSON: anything after the value, a number written
    /// as `01` or `.5` or `NaN`, a control character in a string, an unpaired
    /// surrogate escape, or nesting deeper than 128 levels.
    pub fn parse(text: &str) -> Result<Json, JsonError> {
        let mut parser = Parser { bytes: text.as_bytes(), pos: 0 };
        parser.skip_space();
        let value = parser.value(0)?;
        parser.skip_space();
        if parser.pos != parser.bytes.len() {
            return Err(parser.error("unexpected text after the value"));
        }
        Ok(value)
    }

    /// The value as compact JSON text: no spaces, keys in their order, only the
    /// characters JSON requires escaped.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Json::Float(f) => out.push_str(&float_json(*f)),
            Json::String(s) => write_string(out, s),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Object(pairs) => {
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(out, key);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Appends `s` as a JSON string. Quote, backslash and the control characters are
/// escaped (`\b \f \n \r \t` by letter, the others as `\u00xx` in lower case);
/// everything else, non-ASCII text included, is written as it is.
pub(crate) fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float as Python's `repr` writes it: the shortest digits that read back as
/// the same value, in fixed notation from 1e-4 up to 1e16 and in scientific
/// notation (`1e-05`, `1.5e+20`) outside.
pub(crate) fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.into();
    }
    // Rust's `{:e}` gives the shortest digits that read back, as "d.ddde-5".
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let n = digits.len() as i32;
    let point = exponent + 1; // digits before the decimal point
    if point > -4 && point <= 16 {
        if point <= 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
        } else if point >= n {
            format!("{sign}{digits}{}.0", "0".repeat((point - n) as usize))
        } else {
            format!("{sign}{}.{}", &digits[..point as usize], &digits[point as usize..])
        }
    } else {
        let tail = if n > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{}{tail}e{exp_sign}{:02}", &digits[..1], exponent.abs())
    }
}

/// A float as `json.dumps` writes it: like `repr`, with NaN and the infinities
/// as `NaN`, `Infinity` and `-Infinity` (which are not valid JSON).
pub(crate) fn float_json(value: f64) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value.is_infinite() {
        if value < 0.0 { "-Infinity" } else { "Infinity" }.into()
    } else {
        float_repr(value)
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, what: &'static str) -> JsonError {
        JsonError { offset: self.pos, what }
    }

    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.bytes.get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn literal(&mut self, word: &'static str, value: Json) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error("not a JSON value"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nested too deeply"));
        }
        match self.bytes.get(self.pos) {
            None => Err(self.error("the text ends where a value should be")),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_space();
                if self.eat(b']') {
                    return Ok(Json::Array(items));
                }
                loop {
                    self.skip_space();
                    items.push(self.value(depth + 1)?);
                    self.skip_space();
                    if self.eat(b']') {
                        return Ok(Json::Array(items));
                    }
                    if !self.eat(b',') {
                        return Err(self.error("expected ',' or ']'"));
                    }
                }
            }
            Some(b'{') => {
                self.pos += 1;
                let mut pairs: Vec<(String, Json)> = Vec::new();
                self.skip_space();
                if self.eat(b'}') {
                    return Ok(Json::Object(pairs));
                }
                loop {
                    self.skip_space();
                    if self.bytes.get(self.pos) != Some(&b'"') {
                        return Err(self.error("expected a string as key"));
                    }
                    let key = self.string()?;
                    self.skip_space();
                    if !self.eat(b':') {
                        return Err(self.error("expected ':'"));
                    }
                    self.skip_space();
                    let value = self.value(depth + 1)?;
                    match pairs.iter_mut().find(|(k, _)| *k == key) {
                        Some(pair) => pair.1 = value,
                        None => pairs.push((key, value)),
                    }
                    self.skip_space();
                    if self.eat(b'}') {
                        return Ok(Json::Object(pairs));
                    }
                    if !self.eat(b',') {
                        return Err(self.error("expected ',' or '}'"));
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("not a JSON value")),
        }
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        self.eat(b'-');
        match self.bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.error("a number needs a digit")),
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            is_float = true;
            self.pos += 1;
            if !self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                return Err(self.error("a fraction needs a digit"));
            }
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                return Err(self.error("an exponent needs a digit"));
            }
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).expect("ASCII digits");
        if !is_float {
            if let Ok(int) = text.parse::<i64>() {
                return Ok(Json::Int(int));
            }
            if let Ok(unsigned) = text.parse::<u64>() {
                return Ok(Json::Int(unsigned as i64));
            }
        }
        match text.parse::<f64>() {
            Ok(f) if f.is_finite() => Ok(Json::Float(f)),
            _ => Err(JsonError { offset: start, what: "a number out of range" }),
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.pos += 1; // the opening quote
        let mut out = String::new();
        loop {
            // Copy the run of plain characters up to the next quote or escape.
            let run = self.bytes[self.pos..]
                .iter()
                .position(|b| *b == b'"' || *b == b'\\' || *b < 0x20)
                .ok_or_else(|| self.error("the text ends inside a string"))?;
            out.push_str(
                std::str::from_utf8(&self.bytes[self.pos..self.pos + run])
                    .expect("the text is a &str and the run ends at ASCII"),
            );
            self.pos += run;
            match self.bytes[self.pos] {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let escape =
                        *self.bytes.get(self.pos).ok_or_else(|| self.error("bad escape"))?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&unit) {
                                if !(self.eat(b'\\') && self.eat(b'u')) {
                                    return Err(self.error("a high surrogate needs a low one"));
                                }
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(self.error("a high surrogate needs a low one"));
                                }
                                0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&unit) {
                                return Err(self.error("a low surrogate without a high one"));
                            } else {
                                unit
                            };
                            out.push(char::from_u32(code).expect("not a surrogate"));
                        }
                        _ => return Err(self.error("bad escape")),
                    }
                }
                _ => return Err(self.error("a control character in a string")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let digits =
            self.bytes.get(self.pos..self.pos + 4).ok_or_else(|| self.error("bad \\u escape"))?;
        let mut unit = 0;
        for d in digits {
            let digit = (*d as char).to_digit(16).ok_or_else(|| self.error("bad \\u escape"))?;
            unit = unit * 16 + digit;
        }
        self.pos += 4;
        Ok(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_kind_of_value_and_keeps_the_order_of_keys() {
        let json =
            Json::parse(r#" {"b": [1, -2, 3.5, 1e3, true, null], "a": "x", "c": {}} "#).unwrap();
        let Json::Object(pairs) = &json else { panic!("{json:?}") };
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);
        assert_eq!(
            pairs[0].1,
            Json::Array(vec![
                Json::Int(1),
                Json::Int(-2),
                Json::Float(3.5),
                Json::Float(1000.0),
                Json::Bool(true),
                Json::Null
            ])
        );
    }

    #[test]
    fn repeated_keys_keep_their_first_place_and_last_value() {
        let json = Json::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(json.dump(), r#"{"a":3,"b":2}"#);
    }

    #[test]
    fn reads_escapes_and_surrogate_pairs() {
        let json = Json::parse(r#""a\n\t\"\\\/\u00e4\ud83d\ude00\u0000""#).unwrap();
        assert_eq!(json, Json::String("a\n\t\"\\/ä😀\0".into()));
    }

    #[test]
    fn refuses_what_is_not_json() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "-",
            "1.",
            ".5",
            "+1",
            "1e",
            "NaN",
            "tru",
            "'a'",
            "[1 2]",
            "{\"a\" 1}",
            "{1: 2}",
            "\"a",
            "\"\\x\"",
            "\"\\ud800\"",
            "\"\\udc00\"",
            "\"\n\"",
            "1 2",
            "[]]",
            "1e999",
        ] {
            assert!(Json::parse(bad).is_err(), "{bad:?}");
        }
        let deep = "[".repeat(200) + &"]".repeat(200);
        assert_eq!(Json::parse(&deep).unwrap_err().what, "nested too deeply");
    }

    #[test]
    fn integers_above_i64_wrap_like_llama_cpp_and_the_rest_become_floats() {
        assert_eq!(Json::parse("9223372036854775807").unwrap(), Json::Int(i64::MAX));
        assert_eq!(Json::parse("9223372036854775808").unwrap(), Json::Int(i64::MIN));
        assert_eq!(Json::parse("12345678901234567890").unwrap(), Json::Int(-6101065172474983726));
        assert_eq!(
            Json::parse("18446744073709551616").unwrap(),
            Json::Float(1.8446744073709552e19)
        );
        assert_eq!(
            Json::parse("-9223372036854775809").unwrap(),
            Json::Float(-9.223372036854776e18)
        );
        assert_eq!(Json::parse("-0").unwrap(), Json::Int(0));
    }

    #[test]
    fn dumps_compact_text_and_parses_it_back() {
        let text = r#"{"k":[1,2.5,"a\"b\\c\n\u0001é😀",null,true,{"x":[]}],"e":{}}"#;
        let json = Json::parse(text).unwrap();
        assert_eq!(json.dump(), text);
        assert_eq!(Json::parse(&json.dump()).unwrap(), json);
    }

    #[test]
    fn writes_floats_like_python() {
        for (value, text) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.5, "0.5"),
            (1.23456, "1.23456"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e20, "1.5e+20"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.25e-7, "1.25e-07"),
            (-2.5e-10, "-2.5e-10"),
            (1e100, "1e+100"),
            (0.1 + 0.2, "0.30000000000000004"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(float_repr(value), text, "{value:e}");
        }
        assert_eq!(float_json(f64::NAN), "NaN");
        assert_eq!(float_json(f64::NEG_INFINITY), "-Infinity");
    }
}
