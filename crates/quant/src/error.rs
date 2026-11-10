use ares_core::GgmlType;

use crate::kv::KvFormat;

/// What a call was asked to do, for error messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Block bytes to `f32`.
    Dequantize,
    /// `f32` to block bytes.
    Quantize,
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Operation::Dequantize => "dequantize",
            Operation::Quantize => "quantize",
        })
    }
}

/// Why a conversion was refused. Nothing is written to the output in any of
/// these cases.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The number of values is not a whole number of blocks, so there is no
    /// byte length that could match it.
    #[error("{ty}: {values} values are not a whole number of {block_values}-value blocks")]
    PartialBlock { ty: GgmlType, values: usize, block_values: u32 },
    /// The byte slice is not exactly as long as the blocks of the values.
    #[error("{ty}: {values} values take {expected} bytes, but the byte slice has {actual}")]
    ByteLength { ty: GgmlType, values: usize, expected: u64, actual: usize },
    /// The type has no conversion in this direction.
    #[error("{ty}: cannot {operation}: {reason}")]
    Unsupported { ty: GgmlType, operation: Operation, reason: &'static str },
    /// The number of values of a KV row is not a whole number of groups.
    #[error(
        "{values} values are not a whole number of {}-value KV groups",
        crate::kv::GROUP_VALUES
    )]
    KvPartialGroup { values: usize },
    /// The byte slice of a KV row is not exactly as long as the row.
    #[error("{format}: {values} values take {expected} bytes, but the byte slice has {actual}")]
    KvByteLength { format: KvFormat, values: usize, expected: usize, actual: usize },
    /// A value to store is NaN or infinite; `position` is its index in the row
    /// (in the values of the tile, for a tile) and `kind` says which of the three
    /// it is.
    #[error("the KV value at position {position} is {kind}, only finite values can be stored")]
    KvNonFinite { position: usize, kind: &'static str },
    /// Rotating a group overflows `f32`, so it has no rotated values to quantize;
    /// `group` is its index in the row (in the groups of the tile, for a tile).
    #[error("KV group {group}: its values are so large that the rotation overflows f32")]
    KvRotationOverflow { group: usize },
    /// The values of a KV tile are not one or more whole rows of `row` values.
    #[error("{values} values are not one or more whole KV rows of {row} values")]
    KvTileRows { values: usize, row: usize },
    /// The byte slice of the mean of a KV tile is not as long as the format has
    /// it for rows of `values` values (nothing for a format without a mean).
    #[error(
        "{format}: the mean of a tile of rows of {values} values takes {expected} bytes, but the byte slice has {actual}"
    )]
    KvMeanLength { format: KvFormat, values: usize, expected: usize, actual: usize },
    /// A KV format name that is not on the ladder.
    #[error("unknown KV format {name:?}, the formats are {}", crate::kv::FormatNames)]
    KvFormatName { name: String },
}
