//! What the readers of the Messages and the Responses bodies share: both
//! write a body of their API over into the body of a chat request, which
//! `wire::from_json` then reads as it reads every chat request.

use ares_api::Json;

use crate::error::ApiError;
use crate::wire::Fields;

/// What a body is read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// The route that generates a reply.
    Reply,
    /// The route that counts the prompt: `stream` says nothing, and on
    /// `/v1/messages/count_tokens` `max_tokens` may be missing.
    Count,
}

/// The members of the chat body as they are written.
pub(crate) type Chat = Vec<(&'static str, Json)>;

pub(crate) fn refuse(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, message)
}

pub(crate) fn object(members: Vec<(&str, Json)>) -> Json {
    Json::Object(members.into_iter().map(|(name, value)| (name.to_owned(), value)).collect())
}

pub(crate) fn string(text: impl Into<String>) -> Json {
    Json::String(text.into())
}

/// The members of an object of the body, which may have the members `names`
/// and no others. A `null` is a member that is not set.
pub(crate) fn known<'a>(at: &str, value: &'a Json, names: &[&str]) -> Result<Fields<'a>, ApiError> {
    let Json::Object(members) = value else {
        return Err(refuse(at, format!("{at} must be an object, got {}", value.kind())));
    };
    for (name, member) in members {
        if !member.is_null() && !names.contains(&name.as_str()) {
            return Err(refuse(at, format!("{at}.{name} is not supported")));
        }
    }
    Ok(Fields(members))
}

/// A member of an object that must be a string, if it is there.
pub(crate) fn optional_text<'a>(
    at: &str,
    fields: &Fields<'a>,
    name: &str,
) -> Result<Option<&'a str>, ApiError> {
    match fields.get(name) {
        None => Ok(None),
        Some(Json::String(text)) => Ok(Some(text)),
        Some(other) => {
            Err(refuse(at, format!("{at}.{name} must be a string, got {}", other.kind())))
        }
    }
}

/// A member of an object that must be a string.
pub(crate) fn text_member<'a>(
    at: &str,
    fields: &Fields<'a>,
    name: &str,
) -> Result<&'a str, ApiError> {
    optional_text(at, fields, name)?.ok_or_else(|| refuse(at, format!("{at}.{name} is required")))
}

/// The `type` of an object, if it says one.
pub(crate) fn kind<'a>(at: &str, value: &'a Json) -> Result<Option<&'a str>, ApiError> {
    match value {
        Json::Object(_) => match value.get("type") {
            None | Some(Json::Null) => Ok(None),
            Some(Json::String(kind)) => Ok(Some(kind)),
            Some(other) => {
                Err(refuse(at, format!("{at}.type must be a string, got {}", other.kind())))
            }
        },
        other => Err(refuse(at, format!("{at} must be an object, got {}", other.kind()))),
    }
}

/// Appends `text` to `whole`, after `between` if there is text already.
pub(crate) fn join(whole: &mut Option<String>, between: &str, text: &str) {
    match whole {
        Some(whole) => {
            whole.push_str(between);
            whole.push_str(text);
        }
        None => *whole = Some(text.to_owned()),
    }
}

/// An error of the reader of the chat body under the names the fields have
/// in the API: `names` pairs a chat field with its name there, and a
/// parameter inside the field (`response_format.json_schema`) is renamed too.
pub(crate) fn renamed(mut error: ApiError, names: &[(&str, &str)]) -> ApiError {
    for (chat, here) in names {
        if error.param.as_deref().is_some_and(|param| param.starts_with(chat)) {
            error.message = error.message.replace(chat, here);
            error.param = error.param.take().map(|param| param.replacen(chat, here, 1));
        }
    }
    error
}
