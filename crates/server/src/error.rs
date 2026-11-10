//! Errors in the shape OpenAI clients parse.

use ares_api::{Engine, EngineError};
use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// A failed request, ready to be sent as `{"error": {...}}`.
///
/// Clients branch on `code` (the OpenAI SDKs raise a dedicated exception for
/// `context_length_exceeded`, for example), so codes are part of the interface
/// and stay stable; `message` is for people and says what to change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub code: &'static str,
    /// The request field the error is about.
    pub param: Option<String>,
    /// Seconds after which a retry is worth trying.
    pub retry_after: Option<u32>,
}

const INVALID_REQUEST: &str = "invalid_request_error";
const SERVER_ERROR: &str = "server_error";

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: INVALID_REQUEST,
            code: "invalid_request",
            param: None,
            retry_after: None,
        }
    }

    pub fn invalid_param(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self { param: Some(param.into()), ..Self::invalid(message) }
    }

    pub fn too_large(limit: usize) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: format!("the request body is larger than the limit of {limit} bytes"),
            code: "request_too_large",
            ..Self::invalid("")
        }
    }

    pub fn not_found(what: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!("there is no endpoint '{what}'"),
            code: "not_found",
            ..Self::invalid("")
        }
    }

    pub fn method_not_allowed(method: &str, path: &str) -> Self {
        Self {
            status: StatusCode::METHOD_NOT_ALLOWED,
            message: format!("{method} is not allowed on '{path}'"),
            code: "method_not_allowed",
            ..Self::invalid("")
        }
    }

    /// The request is fine but this build cannot serve it. Retrying does not
    /// help.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "not_implemented",
            ..Self::internal(message)
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: SERVER_ERROR,
            code: "internal_error",
            param: None,
            retry_after: None,
        }
    }

    /// The engine stopped the request while its client was still waiting.
    pub fn cancelled() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "the engine cancelled the request before it finished (it may be shutting \
                      down); retry shortly"
                .into(),
            code: "cancelled",
            retry_after: Some(1),
            ..Self::internal("")
        }
    }

    /// The bounded event queue of the request overflowed.
    pub fn slow_client() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "the response was not read fast enough, so the request was cancelled \
                      to keep the engine free for others"
                .into(),
            code: "slow_client",
            ..Self::internal("")
        }
    }

    /// The engine dropped its sink without a terminal event.
    pub fn engine_vanished() -> Self {
        Self::internal("the engine stopped working on the request without finishing it")
    }

    /// Maps what the engine reported to what a client should see.
    ///
    /// `engine` is only asked for its model list, to make "unknown model"
    /// actionable.
    pub fn from_engine(engine: &dyn Engine, error: EngineError) -> Self {
        let message = error.to_string();
        match error {
            EngineError::ModelNotFound(model) => {
                let available: Vec<_> = engine.models().into_iter().map(|m| m.id).collect();
                let hint = if available.is_empty() {
                    "no model is served".to_owned()
                } else {
                    format!("available models: {}", available.join(", "))
                };
                Self {
                    status: StatusCode::NOT_FOUND,
                    message: format!("The model '{model}' does not exist ({hint})"),
                    code: "model_not_found",
                    param: Some("model".into()),
                    ..Self::invalid("")
                }
            }
            EngineError::InvalidRequest(_) => Self::invalid(message),
            // OpenAI answers 400 here, and its SDKs key on the code.
            EngineError::ContextOverflow { .. } => {
                Self { code: "context_length_exceeded", ..Self::invalid(message) }
            }
            EngineError::Overloaded => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "overloaded",
                retry_after: Some(1),
                ..Self::internal(message)
            },
            EngineError::Unsupported(_) => Self::unsupported(message),
            EngineError::Internal(_) => Self::internal(message),
        }
    }

    pub fn body(&self) -> ErrorEnvelope<'_> {
        ErrorEnvelope {
            error: ErrorBody {
                message: &self.message,
                kind: self.kind,
                param: self.param.as_deref(),
                code: self.code,
            },
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    param: Option<&'a str>,
    code: &'a str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::warn!(status = self.status.as_u16(), code = self.code, "{}", self.message);
        }
        let mut response = (self.status, Json(self.body())).into_response();
        if let Some(seconds) = self.retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        response
    }
}
