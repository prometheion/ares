//! JSON schemas.

use serde_json::{Value, json};

use crate::Error;

/// The schema the engine compiles for the schema in `text`.
///
/// llama-server reads a schema loosely: keywords it does not know are left
/// out, and what it knows it follows. The engine's own default refuses a
/// schema with a keyword it cannot serve; `lenient` makes it leave them out as
/// llama.cpp does. The white space between the parts of the JSON is the one
/// llama.cpp's grammars allow: one space, or a line break with some indent.
pub(crate) fn prepare(text: &str) -> Result<Value, Error> {
    let mut schema: Value = serde_json::from_str(text)
        .map_err(|e| Error::Schema(format!("the JSON schema is not valid JSON: {e}")))?;
    match &mut schema {
        Value::Object(members) => {
            let options = json!({
                "lenient": true,
                "whitespace_flexible": true,
                "whitespace_pattern": WHITESPACE,
            });
            members.entry("x-guidance").or_insert(options);
        }
        Value::Bool(_) => {}
        other => {
            return Err(Error::Schema(format!(
                "a JSON schema is an object or a boolean, got {}",
                kind(other)
            )));
        }
    }
    Ok(schema)
}

/// llama.cpp's `space ::= | " " | "\n"{1,2} [ \t]{0,20}`.
const WHITESPACE: &str = r"( |\n{1,2}[ \t]{0,20})?";

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}
