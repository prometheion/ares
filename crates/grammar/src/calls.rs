//! The grammar of a reply that calls tools.
//!
//! A model writes a call as its chat template writes the calls of earlier
//! turns ([`ToolFormat`]): the name of the tool in a tag, and each argument
//! between a tag that names it and a tag that closes it. [`CallRule::lark`]
//! writes the grammar of such calls for the tools of a request, after the
//! grammar llama-server builds for the same format:
//!
//! - the name is one of the tools';
//! - the arguments are the properties of the tool's schema: the required ones
//!   first, each once, in any order (in the order of the schema when there are
//!   more than [`MAX_PERMUTED`]), then the others in any order;
//! - a value whose schema allows no string is JSON that the schema accepts; a
//!   value that can only be a string is text up to the closing tag, or one of
//!   the strings of its `enum` or `const`; a value that may be a string or
//!   something else is text up to the closing tag;
//! - a call may be followed by white space and, when several calls are
//!   allowed, by another call; then the reply ends.
//!
//! A string is written as it is, so the schema's `pattern`, `minLength`,
//! `maxLength` and `format` of a string are not enforced.

use ares_api::ToolFormat;
use serde_json::{Map, Value, json};

use crate::Error;
use crate::schema::with_options;

/// Up to this many required arguments are accepted in any order; more are
/// accepted in the order of the schema (llama.cpp's limit).
pub const MAX_PERMUTED: usize = 6;

/// llama.cpp's `space`: nothing, one space, or line breaks and an indent.
const SPACE: &str = r"/ |\n{1,2}[ \t]{0,20}/";

/// A tool a reply may call.
#[derive(Debug, Clone, Copy)]
pub struct Callable<'a> {
    pub name: &'a str,
    /// The JSON schema of the arguments (`parameters`), if the tool has one.
    pub parameters: Option<&'a Value>,
}

/// What a reply that calls tools may be.
#[derive(Debug, Clone, Copy)]
pub struct CallRule<'a> {
    pub format: &'a ToolFormat,
    pub tools: &'a [Callable<'a>],
    /// Whether a reply may hold several calls.
    pub parallel: bool,
    /// Whether the grammar starts after the format's trigger (the reply was
    /// free up to it); otherwise the reply is calls from its first token, after
    /// the white space a template writes before them.
    pub triggered: bool,
}

/// The kinds of JSON value a schema allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueTypes(u8);

impl ValueTypes {
    const STRING: u8 = 1;
    const NUMBER: u8 = 2;
    const BOOLEAN: u8 = 4;
    const NULL: u8 = 8;
    const OBJECT: u8 = 16;
    const ARRAY: u8 = 32;
    const ALL: u8 = 63;

    /// The kinds the property `key` of the schema `parameters` allows; every
    /// kind for a property the schema does not describe.
    pub fn of(parameters: Option<&Value>, key: &str) -> Self {
        let Some(root) = parameters else { return Self(Self::ALL) };
        match property(root, key) {
            Some(schema) => Self::of_schema(schema, root),
            None => Self(Self::ALL),
        }
    }

    /// The kinds `schema` allows, inside the tool's schema `root`. A schema
    /// that contradicts itself allows no kind and no value; it is read as
    /// one that says nothing, so that the grammar and the reader of the reply
    /// agree on how its value is written.
    fn of_schema(schema: &Value, root: &Value) -> Self {
        match types(schema, root, 0) {
            0 => Self(Self::ALL),
            kinds => Self(kinds),
        }
    }

    /// Whether a value can only be a string.
    pub fn only_string(self) -> bool {
        self.0 == Self::STRING
    }

    /// Whether a value can be a string.
    pub fn has_string(self) -> bool {
        self.0 & Self::STRING != 0
    }

    /// Whether `value` is of one of the kinds.
    pub fn allows(self, value: &Value) -> bool {
        self.0 & kind(value) != 0
    }
}

fn kind(value: &Value) -> u8 {
    match value {
        Value::String(_) => ValueTypes::STRING,
        Value::Number(_) => ValueTypes::NUMBER,
        Value::Bool(_) => ValueTypes::BOOLEAN,
        Value::Null => ValueTypes::NULL,
        Value::Object(_) => ValueTypes::OBJECT,
        Value::Array(_) => ValueTypes::ARRAY,
    }
}

/// The schema of the property `key`.
fn property<'a>(root: &'a Value, key: &str) -> Option<&'a Value> {
    resolve(root, root, 0)?.get("properties")?.get(key)
}

/// `schema` with a chain of local `$ref` followed.
fn resolve<'a>(schema: &'a Value, root: &'a Value, depth: u8) -> Option<&'a Value> {
    match schema.get("$ref").and_then(Value::as_str) {
        Some(_) if depth > 16 => None,
        Some(reference) => {
            let target = root.pointer(reference.strip_prefix('#')?)?;
            resolve(target, root, depth + 1)
        }
        None => Some(schema),
    }
}

/// The kinds of value `schema` allows, as llama.cpp works them out: from
/// `const`, `enum`, `type`, a `$ref`, the union of `anyOf` and `oneOf` and the
/// intersection of `allOf`. A schema that says none of it allows every kind; a
/// reference that cannot be followed allows none.
fn types(schema: &Value, root: &Value, depth: u8) -> u8 {
    let Value::Object(members) = schema else { return ValueTypes::ALL };
    if depth > 16 {
        return 0;
    }
    let mut allowed = ValueTypes::ALL;
    if let Some(value) = members.get("const") {
        allowed &= kind(value);
    }
    if let Some(Value::Array(values)) = members.get("enum") {
        allowed &= values.iter().fold(0, |all, value| all | kind(value));
    }
    let named = |name: &Value| match name.as_str() {
        Some("string") => ValueTypes::STRING,
        Some("number" | "integer") => ValueTypes::NUMBER,
        Some("boolean") => ValueTypes::BOOLEAN,
        Some("null") => ValueTypes::NULL,
        Some("object") => ValueTypes::OBJECT,
        Some("array") => ValueTypes::ARRAY,
        _ => 0,
    };
    match members.get("type") {
        Some(Value::Array(names)) => allowed &= names.iter().fold(0, |all, n| all | named(n)),
        Some(name) => allowed &= named(name),
        None => {}
    }
    if let Some(reference) = members.get("$ref").and_then(Value::as_str) {
        allowed &= reference
            .strip_prefix('#')
            .and_then(|pointer| root.pointer(pointer))
            .map_or(0, |target| types(target, root, depth + 1));
    }
    for union in ["anyOf", "oneOf"] {
        if let Some(Value::Array(children)) = members.get(union) {
            allowed &= children.iter().fold(0, |all, c| all | types(c, root, depth + 1));
        }
    }
    if let Some(Value::Array(children)) = members.get("allOf") {
        allowed &= children.iter().fold(ValueTypes::ALL, |all, c| all & types(c, root, depth + 1));
    }
    allowed
}

/// The strings a string-only schema is limited to by `enum` or `const`.
fn choices(schema: &Value, root: &Value) -> Option<Vec<String>> {
    let schema = resolve(schema, root, 0)?;
    let strings = |values: &[Value]| -> Option<Vec<String>> {
        values.iter().map(|v| v.as_str().map(str::to_owned)).collect()
    };
    match (schema.get("const"), schema.get("enum")) {
        (Some(value), _) => strings(std::slice::from_ref(value)),
        (None, Some(Value::Array(values))) => strings(values),
        _ => None,
    }
}

/// A string literal of the grammar.
fn literal(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}

impl CallRule<'_> {
    /// The grammar in the Lark syntax of the engine, starting at `start`.
    ///
    /// # Errors
    ///
    /// [`Error::Schema`] when a tool cannot be written in the format: a name or
    /// a key that holds what the format ends it with, a string of an `enum`
    /// that holds the tag that closes a value.
    pub fn lark(&self) -> Result<String, Error> {
        let format = self.format;
        if self.tools.is_empty() {
            return Err(Error::Schema("there is no tool to call".into()));
        }
        let mut rules = Vec::new();
        let mut uses_text = false;
        let (mut first, mut next) = (Vec::new(), Vec::new());
        for (index, tool) in self.tools.iter().enumerate() {
            let name = tool.name;
            if name.is_empty() || name.contains(&format.name_close) {
                return Err(Error::Schema(format!(
                    "the tool name {name:?} cannot be written in the model's format for calls"
                )));
            }
            let body = self.body(index, tool, &mut rules, &mut uses_text)?;
            rules.push(format!("b{index}: {body}{}", literal(&format.call_close)));
            let header = |open: &str| literal(&format!("{open}{name}{}", format.name_close));
            first.push(format!("{} b{index}", header(format.after_trigger())));
            next.push(format!("{} b{index}", header(&format.call_open)));
        }
        let mut lark = String::new();
        let more = if self.parallel { "(WS? next)* WS?" } else { "WS?" };
        if self.triggered {
            lark.push_str(&format!("start: first more\nfirst: {}\n", first.join(" | ")));
        } else {
            lark.push_str("start: WS? next more\n");
        }
        lark.push_str(&format!("more: {more}\nWS: {SPACE}\n"));
        if self.parallel || !self.triggered {
            lark.push_str(&format!("next: {}\n", next.join(" | ")));
        }
        if uses_text {
            lark.push_str(&format!("text[suffix={}]: /(?s:.*)/\n", literal(&format.param_close)));
        }
        for rule in rules {
            lark.push_str(&rule);
            lark.push('\n');
        }
        Ok(lark)
    }

    /// The arguments of tool `index` as an expression (empty for none, or
    /// ending in a blank), with the rules it uses added to `rules`.
    fn body(
        &self,
        index: usize,
        tool: &Callable<'_>,
        rules: &mut Vec<String>,
        uses_text: &mut bool,
    ) -> Result<String, Error> {
        let format = self.format;
        let root = tool.parameters.unwrap_or(&Value::Null);
        let object = resolve(root, root, 0);
        let empty = Map::new();
        let properties =
            object.and_then(|o| o.get("properties")).and_then(Value::as_object).unwrap_or(&empty);
        let is_required = |key: &str| {
            object
                .and_then(|o| o.get("required"))
                .and_then(Value::as_array)
                .is_some_and(|names| names.iter().any(|name| name.as_str() == Some(key)))
        };
        let (mut required, mut optional) = (Vec::new(), Vec::new());
        for (at, (key, schema)) in properties.iter().enumerate() {
            if key.is_empty() || key.contains(&format.key_close) {
                return Err(Error::Schema(format!(
                    "the argument {key:?} of the tool {:?} cannot be written in the model's \
                     format for calls",
                    tool.name
                )));
            }
            let open = literal(&format!("{}{key}{}", format.param_open, format.key_close));
            let kinds = ValueTypes::of_schema(schema, root);
            let value = if !kinds.has_string() {
                rules.push(format!("j{index}_{at}: %json {}", embedded(schema, root)?));
                format!("j{index}_{at} {}", literal(&format.param_close))
            } else if let Some(strings) = choices(schema, root).filter(|_| kinds.only_string()) {
                if let Some(bad) = strings.iter().find(|s| s.contains(&format.param_close)) {
                    return Err(Error::Schema(format!(
                        "the value {bad:?} of the argument {key:?} of the tool {:?} holds the \
                         tag that closes a value",
                        tool.name
                    )));
                }
                let closed = |s: &String| literal(&format!("{s}{}", format.param_close));
                format!("({})", strings.iter().map(closed).collect::<Vec<_>>().join(" | "))
            } else {
                *uses_text = true;
                "text".to_owned()
            };
            rules.push(format!("p{index}_{at}: {open} {value}"));
            let rule = format!("p{index}_{at}");
            if is_required(key) { required.push(rule) } else { optional.push(rule) }
        }

        let mut body = String::new();
        match required.len() {
            0 => {}
            n if n == 1 || n > MAX_PERMUTED => {
                body.push_str(&required.join(" "));
                body.push(' ');
            }
            n => {
                // One rule for every set of arguments still to come.
                for left in 1u32..(1 << n) {
                    let alternatives: Vec<String> = (0..n)
                        .filter(|bit| left >> bit & 1 != 0)
                        .map(|bit| match left & !(1 << bit) {
                            0 => required[bit].clone(),
                            rest => format!("{} r{index}_{rest}", required[bit]),
                        })
                        .collect();
                    rules.push(format!("r{index}_{left}: {}", alternatives.join(" | ")));
                }
                body.push_str(&format!("r{index}_{} ", (1u32 << n) - 1));
            }
        }
        if !optional.is_empty() {
            body.push_str(&format!("({})* ", optional.join(" | ")));
        }
        Ok(body)
    }
}

/// The schema of one argument as the engine reads it inside a grammar: with
/// the definitions of the tool's schema, which its references point into, and
/// with the options every schema of a request is read with.
fn embedded(schema: &Value, root: &Value) -> Result<Value, Error> {
    let mut schema = schema.clone();
    if let Value::Object(members) = &mut schema {
        for definitions in ["$defs", "definitions"] {
            if let Some(shared) = root.get(definitions)
                && !members.contains_key(definitions)
            {
                members.insert(definitions.to_owned(), shared.clone());
            }
        }
    } else if schema == Value::Bool(true) {
        schema = json!({});
    }
    with_options(schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kinds_of_a_value_follow_the_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "n": {"type": "integer"},
                "either": {"type": ["string", "null"]},
                "choice": {"enum": ["a", "b"]},
                "mixed": {"enum": ["a", 1]},
                "ref": {"$ref": "#/$defs/count"},
                "any": {"description": "anything"},
                "union": {"anyOf": [{"type": "boolean"}, {"type": "array"}]},
                "both": {"allOf": [{"type": ["string", "number"]}, {"type": "number"}]},
            },
            "$defs": {"count": {"type": "number"}},
        });
        let of = |key| ValueTypes::of(Some(&schema), key);
        assert!(of("s").only_string());
        assert!(!of("n").has_string());
        assert!(of("either").has_string() && !of("either").only_string());
        assert!(of("either").allows(&Value::Null) && !of("either").allows(&json!(1)));
        assert!(of("choice").only_string());
        assert!(of("mixed").has_string() && of("mixed").allows(&json!(1)));
        assert!(!of("ref").has_string() && of("ref").allows(&json!(1.5)));
        assert_eq!(of("any"), ValueTypes(ValueTypes::ALL));
        assert_eq!(of("not there"), ValueTypes(ValueTypes::ALL));
        assert!(of("union").allows(&json!(true)) && of("union").allows(&json!([])));
        assert!(!of("both").has_string() && of("both").allows(&json!(2)));
        // A schema that allows nothing is read as one that says nothing.
        let never = json!({"properties": {"x": {"type": "string", "enum": [1]}}});
        assert_eq!(ValueTypes::of(Some(&never), "x"), ValueTypes(ValueTypes::ALL));
        assert_eq!(ValueTypes::of(None, "s"), ValueTypes(ValueTypes::ALL));
    }
}
