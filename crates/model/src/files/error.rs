use ares_core::GgmlType;

/// Why a model's file does not hold what its architecture reads.
///
/// Every variant names the key or tensor that is wrong, so that a damaged or
/// unexpected file can be diagnosed from the message alone. Each model's error
/// wraps it.
#[derive(Debug, thiserror::Error)]
pub enum FileError {
    /// The file is a model of another architecture.
    #[error("general.architecture is {found}, this is the {expected} implementation")]
    Architecture {
        /// The architecture the file names.
        found: String,
        /// The architecture of the implementation that read it.
        expected: &'static str,
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
    /// A key under the architecture's prefix that this implementation does
    /// not know. Such a key could change the computation, so it is refused
    /// instead of ignored.
    #[error("metadata key '{key}' is not a key of {of} that this implementation knows")]
    UnknownKey {
        /// The full name of the key.
        key: String,
        /// Whose keys the implementation knows, as a noun ("the qwen35
        /// architecture").
        of: &'static str,
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
    /// A tensor in the file that the architecture has no use for.
    #[error("tensor '{name}' is in the file but is not part of {of}")]
    UnexpectedTensor {
        /// The name of the tensor.
        name: String,
        /// What it would have to be part of, as a noun ("the qwen35
        /// architecture").
        of: &'static str,
    },
}
