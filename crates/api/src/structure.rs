//! What the text of a reply must look like.

use serde::{Deserialize, Serialize};

/// A rule the text of a reply has to follow, and where in the reply it starts to
/// bind. The engine turns it into the set of tokens that may come next, at
/// every token, so a reply that follows it from its first token to its last is
/// the only reply there can be. A rule that cannot be turned into such a set
/// makes the request invalid (`EngineError::InvalidRequest`, which names the
/// problem).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Structure {
    pub shape: Shape,
    pub start: Start,
}

impl Structure {
    /// A rule that binds the reply from its first token.
    pub fn new(shape: Shape) -> Self {
        Self { shape, start: Start::Reply }
    }
}

/// The rule itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    /// A grammar in GBNF, as llama.cpp defines it; it starts at the rule
    /// `root`.
    Gbnf(String),
    /// A JSON schema, as JSON text: the reply is a JSON value that the schema
    /// accepts.
    JsonSchema(String),
}

/// Where the rule starts to bind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Start {
    /// At the first token of the reply.
    Reply,
    /// After the reasoning: the reply thinks first, and the rule binds the text
    /// that follows the closing tag. The tokens of the reasoning are free.
    AfterReasoning {
        /// The tag that opens the reasoning, for a reply that may begin with
        /// it (the prompt did not write it): the reply either starts with this
        /// tag and thinks, or starts with the rule. `None` when the prompt has
        /// opened the block already, so that the reply is reasoning from its
        /// first token.
        open: Option<String>,
        /// The tag that closes the reasoning.
        close: String,
    },
}
