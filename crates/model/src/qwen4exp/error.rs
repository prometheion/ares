use ares_core::GgmlType;

/// Why a `qwen4exp` model could not be configured, loaded or run.
///
/// Every variant names the key or tensor that is wrong, so that a damaged or
/// unexpected file can be diagnosed from the message alone.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is a model of another architecture.
    #[error("general.architecture is {found}, this is the qwen4exp implementation")]
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
    /// A `qwen4exp.*` key that this implementation does not know. Such a key
    /// could change the computation, so it is refused instead of ignored.
    #[error(
        "metadata key '{key}' is not a key of the qwen4exp architecture that this implementation \
         knows"
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
    /// A file that was named as the model's draft block is something else.
    #[error("general.architecture is {found}; the file of this model's draft block has {expected}")]
    NotADraftBlock {
        /// The architecture the file names.
        found: String,
        /// The architecture of a draft block's file.
        expected: &'static str,
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
    /// A tensor in the file that the architecture has no use for.
    #[error("tensor '{name}' is in the file but is not part of the qwen4exp architecture")]
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
    /// A tensor whose rows cannot be addressed one by one: a row is not a
    /// whole number of blocks of its type.
    #[error("tensor '{name}' is stored as {ty}, whose blocks do not divide a row of {cols} values")]
    TensorRows {
        /// The name of the tensor.
        name: String,
        /// Its element type.
        ty: GgmlType,
        /// The values of one row.
        cols: usize,
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
    #[error("cells up to {end} are past the trained context of {limit}")]
    Context {
        /// The cell after the last row of the window.
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
    #[error("the window starts at cell {window}, the state is at {state}")]
    WindowPosition {
        /// The cell of the window's first row.
        window: usize,
        /// The cell count of the state.
        state: usize,
    },
}

/// The result type of this module.
pub type Result<T> = std::result::Result<T, Error>;
