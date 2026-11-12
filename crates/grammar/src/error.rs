/// Why a grammar, a schema or a vocabulary cannot be used.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// The GBNF grammar does not parse or does not make sense. The message says
    /// where (line and column) and what.
    #[error("{0}")]
    Gbnf(String),
    /// The JSON schema is not valid JSON, or asks for something that cannot be
    /// turned into a grammar.
    #[error("{0}")]
    Schema(String),
    /// The grammar engine could not build the grammar the request describes
    /// (for example a regular expression it cannot handle, or one that is too
    /// big).
    #[error("{0}")]
    Compile(String),
    /// The vocabulary cannot be used: no tokens, or an end token past its end.
    #[error("{0}")]
    Vocabulary(String),
}
