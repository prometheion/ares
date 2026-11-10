//! Why the engine refused or abandoned a request.

/// A request the engine cannot serve.
///
/// It is returned from [`Engine::submit`](crate::Engine::submit) when the
/// problem is known at once, and arrives as [`Event::Error`](crate::Event::Error)
/// when it only shows later (the prompt has to be tokenized before its length
/// is known, and tokenizing is the engine's work). The front end handles both
/// the same way, and every variant says which side must change something.
///
/// The enum is deliberately not `#[non_exhaustive]`: the server maps each
/// variant to a status, and a new variant should fail to compile there until
/// somebody decides what it means to a client.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    /// The client named a model the engine does not serve.
    #[error("model '{0}' is not served by this engine")]
    ModelNotFound(String),
    /// The request is malformed or asks for something impossible. The client
    /// can fix it; the message says how.
    #[error("{0}")]
    InvalidRequest(String),
    /// The prompt alone is longer than the model can attend to.
    #[error(
        "the prompt has {prompt_tokens} tokens but the model's context holds {limit}: \
         shorten the conversation or the prompt"
    )]
    ContextOverflow { prompt_tokens: u32, limit: u32 },
    /// The engine has no room for another request right now. Retrying later
    /// can succeed.
    #[error("the engine is at capacity: retry shortly")]
    Overloaded,
    /// The request is fine but cannot be served by this build or this model
    /// (an architecture without an implementation, a feature the model lacks).
    /// Retrying does not help; the message says what is missing.
    #[error("{0}")]
    Unsupported(String),
    /// A bug or a failed device. The client did nothing wrong.
    #[error("internal engine error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_numbers_a_user_needs() {
        let error = EngineError::ContextOverflow { prompt_tokens: 9000, limit: 8192 };
        let message = error.to_string();
        assert!(message.contains("9000") && message.contains("8192"), "{message}");
        assert!(EngineError::ModelNotFound("qwen".into()).to_string().contains("'qwen'"));
        assert_eq!(
            EngineError::InvalidRequest("max_tokens must be at least 1".into()).to_string(),
            "max_tokens must be at least 1"
        );
    }
}
