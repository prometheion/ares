/// Why a tokenizer could not be built from a model file, or a token id could
/// not be decoded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A key the tokenizer cannot work without is not in the file.
    #[error("the model has no '{key}', which the tokenizer needs")]
    MissingKey {
        /// The metadata key.
        key: &'static str,
    },
    /// A key is there but has another type than the format defines.
    #[error("'{key}' is {found}, expected {expected}")]
    WrongType {
        /// The metadata key.
        key: &'static str,
        /// What the key must be.
        expected: &'static str,
        /// What it is.
        found: String,
    },
    /// `tokenizer.ggml.model` names a vocabulary type this engine does not implement.
    #[error(
        "tokenizer model '{model}' is not supported (only 'gpt2', byte-level BPE, is); \
         the engine implements the tokenizers of the models it runs and nothing older"
    )]
    UnsupportedModel {
        /// The value of `tokenizer.ggml.model`.
        model: String,
    },
    /// `tokenizer.ggml.pre` names a pre-tokenizer this engine does not implement.
    #[error(
        "pre-tokenizer '{pre}' is not supported (only 'qwen35' is); \
         the engine implements the pre-tokenizers of the models it runs and nothing older"
    )]
    UnsupportedPre {
        /// The value of `tokenizer.ggml.pre`.
        pre: String,
    },
    /// A tokenizer option that changes the text the vocabulary sees is set to a
    /// value the engine does not implement.
    #[error("'{key}' is set, which the tokenizer does not support for this vocabulary")]
    UnsupportedOption {
        /// The metadata key.
        key: &'static str,
    },
    /// The vocabulary is damaged or does not have the shape the algorithm relies on.
    #[error("invalid vocabulary: {0}")]
    Invalid(String),
    /// A token id that is not in the vocabulary was decoded.
    #[error("token id {id} is outside the vocabulary of {n_tokens} tokens")]
    TokenOutOfRange {
        /// The id.
        id: u32,
        /// The number of tokens the vocabulary has.
        n_tokens: usize,
    },
}

/// A result with the tokenizer's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
