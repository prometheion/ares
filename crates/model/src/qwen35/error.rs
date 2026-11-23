use crate::files::FileError;

/// Why a `qwen35` model could not be configured, loaded or run.
///
/// Every variant names the key or tensor that is wrong, so that a damaged or
/// unexpected file can be diagnosed from the message alone.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file does not hold what the architecture reads: a key or a
    /// tensor, named.
    #[error(transparent)]
    File(#[from] FileError),
    /// The keys are individually fine but do not describe a model.
    #[error("inconsistent configuration: {what}")]
    Inconsistent {
        /// Which relation between the values does not hold.
        what: String,
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
