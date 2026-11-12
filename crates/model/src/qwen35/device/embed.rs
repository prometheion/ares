//! Where the rows of the token embedding come from.
//!
//! The token embedding table (`token_embd`: a row of `n_embd` values in the
//! file's quantized form for every token of the vocabulary) is large, 313 MiB
//! and 521 MiB in two files of the 27B, and a step needs the rows of its own
//! tokens only.
//! So the table does not lie on the device. The device that embeds the step's
//! tokens gets their rows from two places:
//!
//! - **The host.** The whole table lies in page-locked host memory
//!   ([`HostRows`]), counted in the engine's one RAM limit as required memory
//!   ([`ares_runtime::Model::pinned_ram`]). A step copies the rows of its
//!   tokens, in the order of its rows, into a staging buffer and uploads them
//!   on its stream beside the step record, in one copy. The gather decodes
//!   them where they lie, as it decodes any table.
//! - **The device itself.** The tokens the draft block's chain picks on the
//!   device, inside the step, are not known to the host. The device therefore
//!   keeps the rows of every token the chain can pick: the rows of the draft
//!   head (`KeptW` in `load.rs`), which are those of the draft vocabulary, or
//!   without one the first rows of the head on the device. A device without a
//!   draft block keeps no row, and a draft block that can pick every token
//!   keeps the whole table and needs nothing from the host.
//!
//! A tied model (a file without `output.weight`, whose head is the table)
//! has no rows of its own to keep for the chain: the head's rows on the
//! draft block's device are laid out in the file's layout, and the rows of
//! the draft head, which are the first ones, serve the gather as well.

use ares_core::GgmlType;
use ares_cuda::{Device, PinnedBuffer};
use ares_runtime::ModelError;

use super::load::read_matrix;
use crate::qwen35::weights::Matrix;

/// The table as the file stores it: `n_vocab` rows of `n_embd` values of
/// type `ty`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TableShape {
    pub ty: GgmlType,
    pub n_vocab: u32,
    pub n_embd: u32,
}

impl TableShape {
    pub fn of(m: &Matrix<'_>) -> Self {
        Self { ty: m.ty(), n_vocab: m.rows() as u32, n_embd: m.cols() as u32 }
    }

    /// The bytes of one row.
    pub fn row_bytes(&self) -> u64 {
        self.ty.bytes_for(u64::from(self.n_embd)).expect("whole blocks")
    }

    /// The bytes of the whole table.
    pub fn bytes(&self) -> u64 {
        u64::from(self.n_vocab) * self.row_bytes()
    }
}

/// What the embedding of a part takes and where it lies, for the report at
/// start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmbeddingReport {
    /// Whether the part embeds the step's tokens (the first device does).
    pub embeds: bool,
    /// The table in the file's form.
    pub table_bytes: u64,
    /// Page-locked host memory the part holds for it: the table and the rows
    /// of a step; none when the device keeps the whole table.
    pub host_bytes: u64,
    /// Rows of the table that lie on the device, for the draft chain.
    pub kept_rows: u32,
    /// What they take on the device (in the weights): none when they are the
    /// rows of a tied head, which it holds anyway.
    pub kept_bytes: u64,
    /// Whether the kept rows are the first rows of the head.
    pub shared_with_head: bool,
    /// The room in the step's scratch for the rows the host stages.
    pub staged_bytes: u64,
}

/// The table in page-locked host memory, and the buffer in which the rows of
/// a step are gathered for the upload.
pub(super) struct HostRows {
    shape: TableShape,
    table: PinnedBuffer,
    stage: PinnedBuffer,
    /// Rows the stage holds.
    capacity: u32,
}

impl std::fmt::Debug for HostRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostRows").field("shape", &self.shape).finish_non_exhaustive()
    }
}

impl HostRows {
    /// The page-locked bytes of the table and of a stage of `rows` rows.
    pub fn locked_bytes(shape: &TableShape, rows: u32) -> u64 {
        shape.bytes() + u64::from(rows) * shape.row_bytes()
    }

    /// Reads the table `m` into page-locked memory on `device`, with a stage
    /// for the rows of a step of at most `rows` rows.
    pub fn open(device: &Device, m: &Matrix<'_>, rows: u32) -> Result<Self, ModelError> {
        let shape = TableShape::of(m);
        let mut table = PinnedBuffer::new(device, shape.bytes() as usize)?;
        read_matrix(m, &mut table);
        let stage = PinnedBuffer::new(device, (u64::from(rows) * shape.row_bytes()) as usize)?;
        Ok(Self { shape, table, stage, capacity: rows })
    }

    /// The bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.shape.row_bytes() as usize
    }

    /// The page-locked bytes this holds.
    pub fn locked(&self) -> u64 {
        (self.table.len() + self.stage.len()) as u64
    }

    /// The row of `token` in the file's form; none past the table.
    pub fn row(&self, token: u32) -> Option<&[u8]> {
        let size = self.shape.row_bytes() as usize;
        (token < self.shape.n_vocab).then(|| &self.table[token as usize * size..][..size])
    }

    /// Copies the row of each of `tokens` into the stage, row `r` for
    /// `tokens[r]` (a token past the table leaves its row as it was: the
    /// gather gives zeros for it). Returns the bytes to upload and where they
    /// are.
    pub fn stage(&mut self, tokens: &[u32]) -> Result<(*const u8, usize), ModelError> {
        if tokens.len() > self.capacity as usize {
            return Err(ModelError::Invalid(format!(
                "a step of {} rows; the stage of the embedding holds {}",
                tokens.len(),
                self.capacity
            )));
        }
        let size = self.shape.row_bytes() as usize;
        for (r, &token) in tokens.iter().enumerate() {
            if token < self.shape.n_vocab {
                let from = token as usize * size;
                self.stage[r * size..(r + 1) * size]
                    .copy_from_slice(&self.table[from..from + size]);
            }
        }
        Ok((self.stage.as_ptr(), tokens.len() * size))
    }
}
