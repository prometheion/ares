use ares_core::GgmlType;

/// Why a `qwen35` model could not be configured, loaded or run.
///
/// Every variant names the key or tensor that is wrong, so that a damaged or
/// unexpected file can be diagnosed from the message alone.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is a model of another architecture.
    #[error("general.architecture is {found}, this is the qwen35 implementation")]
    Architecture {
        /// The architecture the file names.
        found: String,
    },
    /// A metadata key the architecture needs is not in the file.
    #[error("metadata key '{key}' is missing")]
    MissingKey {
        /// The full name of the key.
        key: String,
    },
    /// A metadata key has a value of the wrong kind or an impossible value.
    #[error("metadata key '{key}' {problem}")]
    BadKey {
        /// The full name of the key.
        key: String,
        /// What is wrong with the value, as the rest of a sentence.
        problem: String,
    },
    /// A `qwen35.*` key that this implementation does not know. Such a key
    /// could change the computation, so it is refused instead of ignored.
    #[error(
        "metadata key '{key}' is not a key of the qwen35 architecture that this implementation knows"
    )]
    UnknownKey {
        /// The full name of the key.
        key: String,
    },
    /// The keys are individually fine but do not describe a model.
    #[error("inconsistent configuration: {what}")]
    Inconsistent {
        /// Which relation between the values does not hold.
        what: String,
    },
    /// A tensor the configuration needs is not in the file.
    #[error("tensor '{name}' is missing")]
    MissingTensor {
        /// The name of the tensor.
        name: String,
    },
    /// A tensor has other dimensions than the configuration gives it.
    #[error("tensor '{name}' has dims {found:?}, the configuration needs {expected:?}")]
    TensorShape {
        /// The name of the tensor.
        name: String,
        /// The dimensions in the file, in GGUF order.
        found: Vec<u64>,
        /// The dimensions the configuration gives it.
        expected: Vec<u64>,
    },
    /// A file given as the MTP block that is none.
    #[error(
        "the file is no MTP block of a qwen35 model: its architecture is '{found}', not \
         'qwen35-mtp'"
    )]
    NotAnMtpFile {
        /// The architecture the file names.
        found: String,
    },
    /// An MTP block of another model.
    #[error(
        "the MTP block is of a model of {blocks} blocks (the block included) and width {width}; \
         this model has {n_layer} blocks and width {n_embd}"
    )]
    MtpFileShape { blocks: u64, width: u64, n_layer: usize, n_embd: usize },
    /// A tensor in the file that the architecture has no use for.
    #[error("tensor '{name}' is in the file but is not part of the qwen35 architecture")]
    UnexpectedTensor {
        /// The name of the tensor.
        name: String,
    },
    /// A tensor of an element type that cannot be turned into `f32`.
    #[error("tensor '{name}' is stored as {ty}, which cannot be decoded")]
    TensorType {
        /// The name of the tensor.
        name: String,
        /// Its element type.
        ty: GgmlType,
        /// Why the decoder refused it.
        #[source]
        source: ares_quant::Error,
    },
    /// A token id that is not in the vocabulary.
    #[error("token {token} is outside the vocabulary of {n_vocab}")]
    Token {
        /// The id.
        token: u32,
        /// The size of the vocabulary.
        n_vocab: usize,
    },
    /// A window that would run past the context the model was trained for.
    #[error("positions up to {end} are past the trained context of {limit}")]
    Context {
        /// The position after the last row of the window.
        end: usize,
        /// The trained context.
        limit: usize,
    },
    /// A window without rows.
    #[error("a window needs at least one token")]
    NoTokens,
    /// A commit of more rows than the window has.
    #[error("{accepted} rows cannot be accepted from a window of {rows}")]
    Accepted {
        /// The number of rows asked for.
        accepted: usize,
        /// The number of rows the window has.
        rows: usize,
    },
    /// A window committed to a state that is not where the window started.
    #[error("the window starts at position {window}, the state is at {state}")]
    WindowPosition {
        /// The position of the window's first row.
        window: usize,
        /// The position of the state.
        state: usize,
    },
}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, Error>;
