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
    /// A text that the rule waits for. `None`: the rule binds from `start`.
    /// With a trigger the reply is free from `start` on, until its text
    /// contains the trigger; the rule then binds what follows the trigger, and
    /// a reply that never writes it stays free to its end. The rule is checked
    /// from the token after the one that completes the trigger: a tokenizer
    /// writes such a tag as a token of its own.
    pub trigger: Option<String>,
    /// What ends the reasoning of a reply besides its closing tag, and the
    /// rule for what follows. `None`: the reasoning ends at its closing tag
    /// and nowhere else.
    pub interrupt: Option<Interrupt>,
}

impl Structure {
    /// A rule that binds the reply from its first token.
    pub fn new(shape: Shape) -> Self {
        Self { shape, start: Start::Reply, trigger: None, interrupt: None }
    }
}

/// A text that ends the reasoning of a reply where it stands, and the rule
/// for what follows it.
///
/// A model that thinks may write what belongs after its reasoning without
/// closing the reasoning first: the marker that opens the call of a tool, for
/// one. With an interrupt such a reply has ended its reasoning at the text,
/// as if the closing tag stood before it, and `shape` binds what follows the
/// text, from the token after the one that completes it. The reasoning ends
/// at whichever comes first, the closing tag or the text; after the closing
/// tag the structure's own rule (and its trigger) applies as without an
/// interrupt. A reply that does not think ([`Start::Reply`]) has no
/// reasoning to end, and the interrupt says nothing about it.
///
/// A structure with an interrupt and without a trigger binds every reply: it
/// either closes its reasoning and follows the structure's rule, or writes
/// the text and follows this one. Such a reply may not end while it thinks:
/// the end tokens are not allowed in its reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interrupt {
    /// The text; it is not empty.
    pub text: String,
    /// The rule that binds what follows the text.
    pub shape: Shape,
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
    /// A grammar in the Lark syntax of the grammar engine (llguidance); it
    /// starts at the rule `start`. The front end writes these for the calls
    /// of tools (`ares_grammar::calls`).
    Lark(String),
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
