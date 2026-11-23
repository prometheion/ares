use crate::files::FileError;

/// Why a `qwen4exp` model could not be configured, loaded or run.
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
    /// A file that was named as the model's MTP block is something else.
    #[error("general.architecture is {found}; an MTP block is a file of {expected} or the model's")]
    NotAnMtpBlock {
        /// The architecture the file names.
        found: String,
        /// The architecture of an MTP block's file.
        expected: &'static str,
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
