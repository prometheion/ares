//! Structured output in requests: `grammar`, `json_schema` and `response_format`.
//!
//! The fields are llama-server's, with its meaning where OpenAI's API and
//! llama-server differ:
//!
//! | field | here |
//! |---|---|
//! | `grammar` | A grammar in GBNF; an empty string asks for nothing. |
//! | `json_schema` | A JSON schema (llama.cpp's own field). An empty schema is any object, as llama-server reads it. |
//! | `response_format` | `text` asks for nothing. `json_object` is `{"type":"object"}`, or the `schema` it carries (llama-server's extension). `json_schema` takes the schema from `json_schema.schema`; `name`, `description` and `strict` are accepted and say nothing more: the schema is always enforced, which is what `strict: true` asks for. |
//!
//! Two of them together are refused, where llama-server's chat endpoint
//! refuses `json_schema` with `grammar` and takes the schema over the grammar
//! otherwise, and its completions endpoint takes `grammar` over `json_schema`.
//! A request must not get an answer that follows only part of it.
//!
//! The same fields work on chat and on completions. `response_format` on
//! completions is this server's addition: llama-server ignores it there.

use ares_api::{Json, Shape, Structure};

use crate::error::ApiError;
use crate::openai::request::{Fields, wrong_type};

/// The members of a request body that ask for structured output.
pub(crate) const FIELDS: &[&str] = &["grammar", "json_schema", "response_format"];

/// What the fields of a request ask of the reply.
pub(crate) fn asked(fields: &Fields<'_>) -> Result<Option<Structure>, ApiError> {
    let grammar = fields.string("grammar")?.filter(|g| !g.is_empty());
    let mut schema = schema_member(fields.get("json_schema"), "json_schema")?;
    if grammar.is_some() && schema.is_some() {
        return Err(both("json_schema", "grammar"));
    }
    if let Some(format) = fields.get("response_format") {
        if !matches!(format, Json::Object(_)) {
            return Err(wrong_type("response_format", "an object", format));
        }
        match format.get("type") {
            None | Some(Json::Null) => {}
            Some(Json::String(kind)) => match kind.as_str() {
                "" | "text" => {}
                "json_object" => {
                    if grammar.is_some() {
                        return Err(both("response_format", "grammar"));
                    }
                    let given = schema_member(format.get("schema"), "response_format.schema")?;
                    // llama-server: the format's own schema wins; without one
                    // the schema field stays, and without that it is any object.
                    schema = given.or(schema).or(Some(Json::Object(Vec::new())));
                }
                "json_schema" => {
                    if grammar.is_some() {
                        return Err(both("response_format", "grammar"));
                    }
                    let wrapper = match format.get("json_schema") {
                        None | Some(Json::Null) => None,
                        Some(wrapper @ Json::Object(_)) => Some(wrapper),
                        Some(other) => {
                            return Err(wrong_type(
                                "response_format.json_schema",
                                "an object",
                                other,
                            ));
                        }
                    };
                    let given = wrapper.map(|w| w.get("schema"));
                    schema = schema_member(given.flatten(), "response_format.json_schema.schema")?
                        .or(Some(Json::Object(Vec::new())));
                }
                other => {
                    return Err(ApiError::invalid_param(
                        "response_format",
                        format!(
                            "response_format type must be one of \"text\", \"json_object\" or \
                             \"json_schema\", but got: {other}"
                        ),
                    ));
                }
            },
            Some(other) => return Err(wrong_type("response_format.type", "a string", other)),
        }
    }
    Ok(match (grammar, schema) {
        (Some(grammar), _) => Some(Structure::new(Shape::Gbnf(grammar.to_owned()))),
        (None, Some(schema)) => Some(Structure::new(Shape::JsonSchema(schema_text(&schema)?))),
        (None, None) => None,
    })
}

fn both(first: &str, second: &str) -> ApiError {
    ApiError::invalid_param(first, format!("{first} and {second} are both set; a request has one"))
}

/// A schema given in a field: an object, or `true` or `false`; `None` when
/// the field is absent or `null`.
fn schema_member(value: Option<&Json>, at: &str) -> Result<Option<Json>, ApiError> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(schema @ (Json::Object(_) | Json::Bool(_))) => Ok(Some(schema.clone())),
        Some(other) => Err(wrong_type(at, "an object", other)),
    }
}

/// The schema as the JSON text the engine reads. A schema with no members
/// is any object, as llama-server reads it.
fn schema_text(schema: &Json) -> Result<String, ApiError> {
    match schema {
        Json::Object(members) if members.is_empty() => Ok(r#"{"type":"object"}"#.to_owned()),
        other => serde_json::to_string(other).map_err(|e| {
            ApiError::invalid_param("json_schema", format!("the schema cannot be read: {e}"))
        }),
    }
}
