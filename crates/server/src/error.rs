//! Errors in the shape OpenAI clients parse, and in Anthropic's for its routes.

use ares_api::{Engine, EngineError, RequestId};
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
    /// The request the engine had when it failed, for the log.
    pub request: Option<RequestId>,
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
            request: None,
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

    /// The body of the request did not arrive in time.
    pub fn too_slow(timeout: std::time::Duration) -> Self {
        Self {
            status: StatusCode::REQUEST_TIMEOUT,
            message: format!(
                "the request body did not arrive within {} s (the server's --timeout)",
                timeout.as_secs()
            ),
            code: "request_timeout",
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

    /// The request carries none of the server's keys.
    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "this server wants an API key: send one of its keys as 'Authorization: \
                      Bearer <key>' or as 'x-api-key: <key>'"
                .into(),
            code: "invalid_api_key",
            ..Self::invalid("")
        }
    }

    /// The request was sent to a name the server does not answer to.
    pub fn forbidden_host(host: &str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: format!(
                "this server does not answer to the name '{host}': it listens on a local address \
                 and answers to its local names (localhost, addresses); the operator can add \
                 names with --allowed-hosts"
            ),
            code: "forbidden_host",
            ..Self::invalid("")
        }
    }

    /// A browser sent the request from a page of another site.
    pub fn forbidden_origin(origin: &str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: format!(
                "pages of '{origin}' may not call this server from a browser; the operator can \
                 allow the origin with --cors-origins"
            ),
            code: "forbidden_origin",
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
            request: None,
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

    /// A request named a model that is not served; `available` are the ids
    /// that are.
    pub fn unknown_model(model: &str, available: &[String]) -> Self {
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

    /// Maps what the engine reported to what a client should see.
    ///
    /// `engine` is only asked for its model list, to make "unknown model"
    /// actionable.
    pub fn from_engine(engine: &dyn Engine, error: EngineError) -> Self {
        let message = error.to_string();
        match error {
            EngineError::ModelNotFound(model) => {
                let available: Vec<_> = engine.models().into_iter().map(|m| m.id).collect();
                Self::unknown_model(&model, &available)
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
            // The request found no room in the engine's memory limit (`--ram`) and
            // ended; the model serves on.
            EngineError::NoMemory(_) => Self {
                status: StatusCode::INSUFFICIENT_STORAGE,
                code: "insufficient_memory",
                retry_after: Some(1),
                ..Self::internal(message)
            },
            EngineError::Internal(_) => Self::internal(message),
        }
    }

    /// The error of the request `id`, for the log.
    pub fn of_request(self, id: RequestId) -> Self {
        Self { request: Some(id), ..self }
    }

    /// Writes the error into the log as its client gets it: a refusal, or a
    /// failure in the middle of a stream (`where`, the method and path of the
    /// route, or what failed). A failure of the server is an error; a refusal
    /// of the request, and a service that is not there for now (503), a
    /// warning.
    pub fn tell(&self, at: &str) {
        let request = self.request.map(|id| format!(" (request {id})")).unwrap_or_default();
        let status = self.status.as_u16();
        if self.status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!("{at}: {status} {}{request}: {}", self.code, self.message);
        } else {
            tracing::warn!("{at}: {status} {}{request}: {}", self.code, self.message);
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

impl From<crate::models::AcquireError> for ApiError {
    /// Why a request did not get its model.
    fn from(error: crate::models::AcquireError) -> Self {
        use crate::models::AcquireError;
        match error {
            AcquireError::NotFound { model, available } => Self::unknown_model(&model, &available),
            // The request is fine and the server cannot serve it now: the
            // reason says whether trying again can help.
            AcquireError::Failed { model, reason } => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "model_load_failed",
                param: Some("model".into()),
                ..Self::internal(format!("the model '{model}' could not be loaded: {reason}"))
            },
            AcquireError::NotLoaded { model } => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "model_not_loaded",
                param: Some("model".into()),
                ..Self::internal(format!("the model '{model}' is not loaded"))
            },
            AcquireError::Stopping => Self::cancelled(),
        }
    }
}

impl ApiError {
    /// The error as Anthropic's API writes one: `{"type": "error", "error":
    /// {"type", "message"}}`. The status is the same as on the other routes.
    pub fn anthropic(&self) -> AnthropicEnvelope<'_> {
        let kind = match (self.status, self.code) {
            (_, "overloaded") => "overloaded_error",
            (StatusCode::UNAUTHORIZED, _) => "authentication_error",
            (StatusCode::FORBIDDEN, _) => "permission_error",
            (StatusCode::NOT_FOUND, _) => "not_found_error",
            (StatusCode::PAYLOAD_TOO_LARGE, _) => "request_too_large",
            (StatusCode::TOO_MANY_REQUESTS, _) => "rate_limit_error",
            (status, _) if status.is_server_error() => "api_error",
            _ => "invalid_request_error",
        };
        AnthropicEnvelope { kind: "error", error: AnthropicBody { kind, message: &self.message } }
    }
}

/// An [`ApiError`] on a route of Anthropic's API.
#[derive(Debug)]
pub(crate) struct Anthropic(pub ApiError);

impl IntoResponse for Anthropic {
    fn into_response(self) -> Response {
        let error = self.0;
        let mut response = (error.status, Json(error.anthropic())).into_response();
        if let Some(seconds) = error.retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        response.extensions_mut().insert(Refused(error));
        response
    }
}

/// The error a response carries, for the layer that writes refusals into the
/// log with their route ([`crate::routes::tell_refusals`]).
#[derive(Debug, Clone)]
pub(crate) struct Refused(pub ApiError);

#[derive(Serialize)]
pub(crate) struct AnthropicEnvelope<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    error: AnthropicBody<'a>,
}

#[derive(Serialize)]
struct AnthropicBody<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    message: &'a str,
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
        let mut response = (self.status, Json(self.body())).into_response();
        if let Some(seconds) = self.retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response.extensions_mut().insert(Refused(self));
        response
    }
}
