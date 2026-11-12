//! GBNF, the grammar syntax of llama.cpp, turned into the syntax the engine
//! reads (Lark).
//!
//! The reading follows `llama-grammar.cpp`: a rule is `name ::= alternatives`;
//! a literal in double quotes, a class in brackets (`[a-z]`, `[^"\\]`), `.` for
//! any character, `<[id]>` and `!<[id]>` for a token by its number and
//! `<text>` for a token by its text, groups in parentheses, `*`, `+`, `?` and
//! `{m}`, `{m,}`, `{m,n}` after an item, `#` comments. A rule ends at a line
//! end unless it is inside parentheses or follows `::=` or `|`. The grammar
//! starts at the rule named `root`.
//!
//! Left recursion is allowed here, which llama.cpp refuses: the engine parses
//! with an Earley parser.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::{Error, Vocabulary};

/// The most times llama.cpp lets an item repeat; above it `{m,n}` has no upper
/// limit, and a minimum above it is an error.
const REPEAT_LIMIT: u32 = 2000;

#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// Characters, in order.
    Literal(String),
    /// A class of characters, as ranges of code points.
    Class {
        negated: bool,
        ranges: Vec<(u32, u32)>,
    },
    Any,
    Rule(usize),
    Token(u32),
    NotToken(u32),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Repeat {
        item: Box<Node>,
        min: u32,
        max: Option<u32>,
    },
}

struct Parser<'a> {
    text: &'a [char],
    at: usize,
    vocab: &'a Vocabulary,
    ids: HashMap<String, usize>,
    names: Vec<String>,
    /// The first place each rule was used, for the error of a rule never defined.
    first_use: Vec<usize>,
    rules: Vec<Option<Node>>,
}

/// The Lark text for the GBNF grammar in `text`.
pub(crate) fn to_lark(text: &str, vocab: &Vocabulary) -> Result<String, Error> {
    let chars: Vec<char> = text.chars().collect();
    let mut parser = Parser {
        text: &chars,
        at: 0,
        vocab,
        ids: HashMap::new(),
        names: Vec::new(),
        first_use: Vec::new(),
        rules: Vec::new(),
    };
    parser.grammar()?;
    parser.emit()
}

impl Parser<'_> {
    fn error(&self, at: usize, what: impl std::fmt::Display) -> Error {
        let before = &self.text[..at.min(self.text.len())];
        let line = before.iter().filter(|&&c| c == '\n').count() + 1;
        let column = before.iter().rev().take_while(|&&c| c != '\n').count() + 1;
        Error::Gbnf(format!("grammar, line {line}, column {column}: {what}"))
    }

    fn peek(&self) -> Option<char> {
        self.text.get(self.at).copied()
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.text.get(self.at + n).copied()
    }

    /// Skips spaces, tabs and comments, and line ends where they do not end the rule.
    fn space(&mut self, newline_ok: bool) {
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' => self.at += 1,
                '\r' | '\n' if newline_ok => self.at += 1,
                '#' => {
                    while self.peek().is_some_and(|c| c != '\r' && c != '\n') {
                        self.at += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn is_word(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '-'
    }

    fn name(&mut self) -> Result<String, Error> {
        let start = self.at;
        while self.peek().is_some_and(Self::is_word) {
            self.at += 1;
        }
        if start == self.at {
            return Err(self.error(start, "expecting a rule name"));
        }
        Ok(self.text[start..self.at].iter().collect())
    }

    fn int(&mut self) -> Result<u32, Error> {
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
        if start == self.at {
            return Err(self.error(start, "expecting a number"));
        }
        let digits: String = self.text[start..self.at].iter().collect();
        // A number too big for a repetition is too big for anything.
        Ok(digits.parse().unwrap_or(u32::MAX))
    }

    fn rule_id(&mut self, name: &str, at: usize) -> usize {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len();
        self.ids.insert(name.to_owned(), id);
        self.names.push(name.to_owned());
        self.first_use.push(at);
        self.rules.push(None);
        id
    }

    fn grammar(&mut self) -> Result<(), Error> {
        self.space(true);
        while self.peek().is_some() {
            self.rule()?;
        }
        if let Some(id) = self.rules.iter().position(Option::is_none) {
            return Err(self.error(
                self.first_use[id],
                format_args!("the rule '{}' is used and never defined", self.names[id]),
            ));
        }
        if !self.ids.contains_key("root") {
            return Err(Error::Gbnf("grammar: there is no rule named 'root'".into()));
        }
        Ok(())
    }

    fn rule(&mut self) -> Result<(), Error> {
        let start = self.at;
        let name = self.name()?;
        self.space(false);
        if !(self.peek() == Some(':')
            && self.peek_at(1) == Some(':')
            && self.peek_at(2) == Some('='))
        {
            return Err(self.error(self.at, "expecting '::='"));
        }
        self.at += 3;
        self.space(true);
        let id = self.rule_id(&name, start);
        let body = self.alternatives(false)?;
        if self.rules[id].is_some() {
            return Err(self.error(start, format_args!("the rule '{name}' is defined twice")));
        }
        self.rules[id] = Some(body);
        match self.peek() {
            Some('\r') => self.at += if self.peek_at(1) == Some('\n') { 2 } else { 1 },
            Some('\n') => self.at += 1,
            Some(_) => {
                return Err(self.error(self.at, "expecting a line end or the end of the grammar"));
            }
            None => {}
        }
        self.space(true);
        Ok(())
    }

    fn alternatives(&mut self, nested: bool) -> Result<Node, Error> {
        let mut alts = vec![self.sequence(nested)?];
        while self.peek() == Some('|') {
            self.at += 1;
            self.space(true);
            alts.push(self.sequence(nested)?);
        }
        Ok(if alts.len() == 1 { alts.pop().expect("one") } else { Node::Alt(alts) })
    }

    fn sequence(&mut self, nested: bool) -> Result<Node, Error> {
        let mut items: Vec<Node> = Vec::new();
        while let Some(c) = self.peek() {
            let item_start = self.at;
            match c {
                '"' => {
                    self.at += 1;
                    let mut literal = String::new();
                    loop {
                        match self.peek() {
                            None => return Err(self.error(item_start, "a string is not closed")),
                            Some('"') => break,
                            Some(_) => literal.push(self.character()?),
                        }
                    }
                    self.at += 1;
                    self.space(nested);
                    items.push(Node::Literal(literal));
                }
                '[' => {
                    self.at += 1;
                    let negated = self.peek() == Some('^');
                    if negated {
                        self.at += 1;
                    }
                    let mut ranges = Vec::new();
                    loop {
                        match self.peek() {
                            None => {
                                return Err(
                                    self.error(item_start, "a character class is not closed")
                                );
                            }
                            Some(']') => break,
                            Some(_) => {
                                let low = self.character()? as u32;
                                let high = if self.peek() == Some('-')
                                    && self.peek_at(1) != Some(']')
                                {
                                    self.at += 1;
                                    if self.peek().is_none() {
                                        return Err(self
                                            .error(item_start, "a character class is not closed"));
                                    }
                                    self.character()? as u32
                                } else {
                                    low
                                };
                                if high < low {
                                    return Err(self
                                        .error(self.at, "the end of a range is before its start"));
                                }
                                ranges.push((low, high));
                            }
                        }
                    }
                    if ranges.is_empty() {
                        return Err(self.error(item_start, "a character class has no characters"));
                    }
                    self.at += 1;
                    self.space(nested);
                    items.push(Node::Class { negated, ranges });
                }
                '<' | '!' => {
                    let inverse = c == '!';
                    if inverse {
                        self.at += 1;
                    }
                    let token = self.token()?;
                    self.space(nested);
                    items.push(if inverse { Node::NotToken(token) } else { Node::Token(token) });
                }
                c if Self::is_word(c) => {
                    let name = self.name()?;
                    self.space(nested);
                    let id = self.rule_id(&name, item_start);
                    items.push(Node::Rule(id));
                }
                '(' => {
                    self.at += 1;
                    self.space(true);
                    let inner = self.alternatives(true)?;
                    if self.peek() != Some(')') {
                        return Err(self.error(self.at, "expecting ')'"));
                    }
                    self.at += 1;
                    self.space(nested);
                    items.push(inner);
                }
                '.' => {
                    self.at += 1;
                    self.space(nested);
                    items.push(Node::Any);
                }
                '*' | '+' | '?' | '{' => {
                    let (min, max) = self.repetition(c, nested)?;
                    let Some(item) = items.pop() else {
                        return Err(
                            self.error(item_start, "expecting an item before '*', '+', '?' or '{'")
                        );
                    };
                    items.push(Node::Repeat { item: Box::new(item), min, max });
                }
                _ => break,
            }
        }
        Ok(if items.len() == 1 { items.pop().expect("one") } else { Node::Seq(items) })
    }

    /// The `*`, `+`, `?` or `{...}` at the cursor.
    fn repetition(&mut self, c: char, nested: bool) -> Result<(u32, Option<u32>), Error> {
        self.at += 1;
        self.space(nested);
        let (min, max) = match c {
            '*' => (0, None),
            '+' => (1, None),
            '?' => (0, Some(1)),
            _ => {
                if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return Err(self.error(self.at, "expecting a number after '{'"));
                }
                let min = self.int()?;
                self.space(nested);
                let max = match self.peek() {
                    Some('}') => Some(min),
                    Some(',') => {
                        self.at += 1;
                        self.space(nested);
                        let max = if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                            let max = self.int()?;
                            self.space(nested);
                            Some(max)
                        } else {
                            None
                        };
                        if self.peek() != Some('}') {
                            return Err(self.error(self.at, "expecting '}'"));
                        }
                        max
                    }
                    _ => return Err(self.error(self.at, "expecting ',' or '}'")),
                };
                self.at += 1;
                self.space(nested);
                if min > REPEAT_LIMIT {
                    return Err(self.error(
                        self.at,
                        format_args!(
                            "a repetition of more than {REPEAT_LIMIT} times is not served"
                        ),
                    ));
                }
                if max.is_some_and(|m| m < min) {
                    return Err(self.error(self.at, "the most is less than the least"));
                }
                // llama.cpp reads a most above the limit as no limit.
                (min, max.filter(|&m| m <= REPEAT_LIMIT))
            }
        };
        Ok((min, max))
    }

    /// One character of a literal or a class, with its escapes.
    fn character(&mut self) -> Result<char, Error> {
        let start = self.at;
        let Some(c) = self.peek() else {
            return Err(self.error(start, "the grammar ends inside a string or a class"));
        };
        self.at += 1;
        if c != '\\' {
            return Ok(c);
        }
        let Some(escape) = self.peek() else {
            return Err(self.error(start, "the grammar ends inside an escape"));
        };
        self.at += 1;
        let digits = match escape {
            'x' => 2,
            'u' => 4,
            'U' => 8,
            't' => return Ok('\t'),
            'r' => return Ok('\r'),
            'n' => return Ok('\n'),
            '\\' | '"' | '[' | ']' | '-' => return Ok(escape),
            _ => return Err(self.error(start, format_args!("unknown escape '\\{escape}'"))),
        };
        let mut value = 0u32;
        for _ in 0..digits {
            let Some(digit) = self.peek().and_then(|c| c.to_digit(16)) else {
                return Err(self.error(start, format_args!("expecting {digits} hex digits")));
            };
            self.at += 1;
            value = value << 4 | digit;
        }
        char::from_u32(value).ok_or_else(|| {
            self.error(
                start,
                format_args!("{value:#x} is not a character (a surrogate or past U+10FFFF)"),
            )
        })
    }

    /// `<[id]>` or `<text>` at the cursor.
    fn token(&mut self) -> Result<u32, Error> {
        let start = self.at;
        self.at += 1; // '<'
        if self.peek() == Some('[') {
            self.at += 1;
            let id = self.int()?;
            if self.peek() != Some(']') || self.peek_at(1) != Some('>') {
                return Err(self.error(self.at, "expecting ']>'"));
            }
            self.at += 2;
            if id >= self.vocab.len() {
                return Err(self.error(
                    start,
                    format_args!(
                        "token {id} is not in a vocabulary of {} tokens",
                        self.vocab.len()
                    ),
                ));
            }
            return Ok(id);
        }
        let mut text = String::from("<");
        loop {
            match self.peek() {
                None => return Err(self.error(start, "a token is not closed")),
                Some('>') => break,
                Some(c) => {
                    text.push(c);
                    self.at += 1;
                }
            }
        }
        self.at += 1;
        text.push('>');
        self.vocab.find(text.as_bytes()).ok_or_else(|| {
            self.error(
                start,
                format_args!(
                    "{text} is not a token with that text (a token that is not text is named by its number, <[id]>)"
                ),
            )
        })
    }
}

impl Parser<'_> {
    /// The Lark grammar with the same language.
    fn emit(&self) -> Result<String, Error> {
        let mut out = String::new();
        let root = self.ids["root"];
        let _ = writeln!(out, "start: {}", self.rule_name(root));
        for (id, body) in self.rules.iter().enumerate() {
            let body = body.as_ref().expect("checked: every rule is defined");
            let text = self.emit_alternatives(body)?;
            let _ = writeln!(out, "{}: {}", self.rule_name(id), text);
        }
        Ok(out)
    }

    fn rule_name(&self, id: usize) -> String {
        let name: String = self.names[id]
            .chars()
            .map(|c| if c == '-' { '_' } else { c.to_ascii_lowercase() })
            .collect();
        format!("r{id}_{name}")
    }

    /// `node` as the alternatives of a rule or a group.
    fn emit_alternatives(&self, node: &Node) -> Result<String, Error> {
        match node {
            Node::Alt(alts) => {
                let parts: Result<Vec<_>, _> =
                    alts.iter().map(|alt| self.emit_sequence(alt)).collect();
                Ok(parts?.join(" | "))
            }
            other => self.emit_sequence(other),
        }
    }

    /// `node` as a sequence of items.
    fn emit_sequence(&self, node: &Node) -> Result<String, Error> {
        match node {
            Node::Seq(items) => {
                let parts: Result<Vec<_>, _> =
                    items.iter().map(|item| self.emit_item(item)).collect();
                Ok(parts?.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" "))
            }
            other => self.emit_item(other),
        }
    }

    /// `node` as one item: something a repetition can follow. Empty when it
    /// matches nothing but the empty text.
    fn emit_item(&self, node: &Node) -> Result<String, Error> {
        Ok(match node {
            Node::Literal(text) => literal(text),
            Node::Class { negated, ranges } => class(*negated, ranges),
            Node::Any => "/(?s:.)/".to_owned(),
            Node::Rule(id) => self.rule_name(*id),
            Node::Token(id) => format!("<[{id}]>"),
            Node::NotToken(id) => {
                let last = self.vocab.len() - 1;
                let mut parts = Vec::new();
                if *id > 0 {
                    parts.push(format!("<[0-{}]>", id - 1));
                }
                if *id < last {
                    parts.push(format!("<[{}-{last}]>", id + 1));
                }
                if parts.is_empty() {
                    return Err(Error::Gbnf(
                        "grammar: a vocabulary of one token has no token other than that one"
                            .into(),
                    ));
                }
                format!("({})", parts.join(" | "))
            }
            Node::Seq(_) | Node::Alt(_) => {
                let inner = self.emit_alternatives(node)?;
                if inner.is_empty() { String::new() } else { format!("({inner})") }
            }
            Node::Repeat { item, min, max } => {
                let inner = self.emit_item(item)?;
                if inner.is_empty() || (*min == 0 && *max == Some(0)) {
                    return Ok(String::new());
                }
                // A repetition of a sequence or of another repetition names
                // the thing it repeats.
                let atom = if matches!(**item, Node::Seq(_) | Node::Alt(_) | Node::Repeat { .. })
                    && !inner.starts_with('(')
                {
                    format!("({inner})")
                } else {
                    inner
                };
                match (min, max) {
                    (0, None) => format!("{atom}*"),
                    (1, None) => format!("{atom}+"),
                    (0, Some(1)) => format!("{atom}?"),
                    (m, None) => format!("{atom}{{{m}}} {atom}*"),
                    (m, Some(n)) if m == n => format!("{atom}{{{m}}}"),
                    (m, Some(n)) => format!("{atom}{{{m},{n}}}"),
                }
            }
        })
    }
}

/// A literal as an item: the quoted text if it is plain, and the regular
/// expression of its characters if it is not.
fn literal(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.chars().all(|c| (' '..='~').contains(&c) && c != '"' && c != '\\') {
        return format!("\"{text}\"");
    }
    let mut out = String::from("/");
    for c in text.chars() {
        let _ = write!(out, "\\x{{{:x}}}", c as u32);
    }
    out.push('/');
    out
}

fn class(negated: bool, ranges: &[(u32, u32)]) -> String {
    let mut out = String::from(if negated { "/[^" } else { "/[" });
    for &(low, high) in ranges {
        let _ = write!(out, "\\x{{{low:x}}}");
        if high != low {
            let _ = write!(out, "-\\x{{{high:x}}}");
        }
    }
    out.push_str("]/");
    out
}
