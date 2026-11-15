use ares_core::GgmlType;

/// Why a projector could not be configured, loaded or run.
///
/// Every variant names the key, tensor or size that is wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is not a projector of the kind implemented here.
    #[error("{key} is {found}, the qwen3vl implementation needs {expected}")]
    Kind {
        /// The key that tells the kinds apart.
        key: &'static str,
        /// What the file says.
        found: String,
        /// What it has to say.
        expected: &'static str,
    },
    /// A metadata key the projector needs is not in the file.
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
    /// A `clip.*` key that this implementation does not know. Such a key could
    /// change the computation, so it is refused instead of ignored.
    #[error("metadata key '{key}' is not a key of the projector that this implementation knows")]
    UnknownKey {
        /// The full name of the key.
        key: String,
    },
    /// The keys are individually fine but do not describe a projector.
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
    /// A tensor in the file that the projector has no use for.
    #[error("tensor '{name}' is in the file but is not part of the projector")]
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
    /// Pixels that the tower cannot take.
    #[error("an image of {width} x {height} pixels {problem}")]
    Image {
        /// The width in pixels.
        width: usize,
        /// The height in pixels.
        height: usize,
        /// What is wrong with it, as the rest of a sentence.
        problem: String,
    },
}

/// The result of configuring, loading or running a projector.
pub type Result<T> = std::result::Result<T, Error>;
