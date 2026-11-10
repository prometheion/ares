//! The engine of a build that cannot run any model yet.

use ares_api::{
    DecisionRequest, DecisionSink, Engine, EngineError, EventSink, GenerateRequest, ModelCard,
    RequestId,
};

/// An [`Engine`] that serves no model.
///
/// The server binary starts with it until a runtime exists. It lets the HTTP
/// side be run, probed and monitored today: `/health` answers, `/v1/models`
/// is empty, and a completion or a decision request gets a `501` that says
/// why, instead of a connection that hangs or a model name that is "not found"
/// although the operator named a file.
#[derive(Debug, Clone, Default)]
pub struct NoModelEngine {
    architecture: Option<String>,
}

impl NoModelEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remembers the architecture of the model file the operator asked for, to
    /// name it in the refusal.
    pub fn with_architecture(mut self, architecture: Option<String>) -> Self {
        self.architecture = architecture;
        self
    }

    fn refusal(&self) -> EngineError {
        EngineError::Unsupported(match &self.architecture {
            Some(architecture) => format!(
                "the model architecture '{architecture}' is not implemented yet: this build \
                 can open GGUF files but cannot run any model"
            ),
            None => "no model architecture is implemented yet: this build cannot run any model"
                .to_owned(),
        })
    }
}

impl Engine for NoModelEngine {
    fn models(&self) -> Vec<ModelCard> {
        Vec::new()
    }

    fn submit(&self, _request: GenerateRequest, _events: EventSink) -> Result<(), EngineError> {
        Err(self.refusal())
    }

    fn decide(&self, _request: DecisionRequest, _outcome: DecisionSink) -> Result<(), EngineError> {
        Err(self.refusal())
    }

    fn cancel(&self, _id: RequestId) {}
}
