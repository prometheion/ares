//! Token rows: where the rows of a token embedding table come from.
//!
//! A table (`token_embd`: a row of `n_embd` values in the file's quantized
//! form for every token of the vocabulary) is large, 313 MiB and 521 MiB in two
//! files of the 27B and 337 MiB in Flash-Next's, and a step needs the rows of
//! its own tokens only. So the table does not lie on the device. A device that
//! needs rows gets them from two places, which the gather
//! ([`ares_cuda::LayerKernels::embed_gather`]) reads side by side and decodes
//! where they lie:
//!
//! - **The host.** The table's home ([`HostTable`]) is page-locked host memory,
//!   read once at load and counted in the engine's one RAM limit (the 27B), or
//!   the mapped model file, read where it lies (Flash-Next, whose RAM is the
//!   tier of its experts). A step copies the rows of its tokens, in the order
//!   of its rows, into a stage ([`Stage`]) and uploads them beside its record
//!   in one copy.
//! - **The device.** The tokens a drafter picks on the device, inside a step,
//!   are not known to the host. The device therefore keeps the rows of every
//!   token a drafter can pick: the rows of the head view's tokens (the draft
//!   vocabulary, or the head's rows on that device; see [`super::head`]). A
//!   device that keeps every row needs nothing from the host.
//!
//! A step's rows may also come from a **second source**: rows of the model's
//! width that another computation made, such as the image tower's for the
//! cells of an image ([`splice`]), on the device or in page-locked host memory
//! (the runtime's image stage). The gather fills every row first (an image
//! cell gathers its pad token's row) and the splice copies the other rows over
//! theirs, in the step's order, before the first layer reads them.

use std::sync::Arc;

use ares_core::GgmlType;
use ares_cuda::{Device, DevicePtr, EmbedRows, KeptRows, PinnedBuffer, Stream};
use ares_gguf::Gguf;
use ares_runtime::ModelError;

use super::invalid;

/// The name of the token embedding table in a model file.
pub(crate) const TOKEN_EMBD: &str = "token_embd.weight";

/// The table as the file stores it: `n_vocab` rows of `n_embd` values of
/// type `ty`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableShape {
    pub ty: GgmlType,
    pub n_vocab: u32,
    pub n_embd: u32,
}

impl TableShape {
    /// The bytes of one row.
    pub fn row_bytes(&self) -> u64 {
        self.ty.bytes_for(u64::from(self.n_embd)).expect("whole blocks")
    }

    /// The bytes of the whole table.
    pub fn bytes(&self) -> u64 {
        u64::from(self.n_vocab) * self.row_bytes()
    }
}

/// Where a table lies on the host.
enum Home {
    /// Page-locked memory the table was read into.
    Pinned(PinnedBuffer),
    /// The mapping of the model file: the bytes of tensor `tensor`.
    Mapped { gguf: Arc<Gguf>, tensor: usize },
}

/// A token embedding table on the host, which the parts of one model share.
pub struct HostTable {
    shape: TableShape,
    home: Home,
}

impl std::fmt::Debug for HostTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostTable").field("shape", &self.shape).finish_non_exhaustive()
    }
}

impl HostTable {
    /// Reads a table of `shape` into page-locked memory on `device` with
    /// `read`, which fills the buffer with the table's bytes.
    pub fn pinned(
        device: &Device,
        shape: TableShape,
        read: impl FnOnce(&mut [u8]),
    ) -> Result<Self, ModelError> {
        let mut table = PinnedBuffer::new(device, shape.bytes() as usize)?;
        read(&mut table);
        Ok(Self { shape, home: Home::Pinned(table) })
    }

    /// The table that is tensor `name` of `gguf`, read where the file is
    /// mapped.
    pub fn mapped(gguf: Arc<Gguf>, name: &str) -> Result<Self, ModelError> {
        let tensor = gguf
            .tensors()
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| invalid(format!("the model has no tensor {name}")))?;
        let info = &gguf.tensors()[tensor];
        let dims = info.dims.as_slice();
        let (&n_embd, rest) =
            dims.split_first().ok_or_else(|| invalid(format!("{name} has no dimensions")))?;
        let n_vocab: u64 = rest.iter().product();
        let shape = TableShape { ty: info.ty, n_vocab: n_vocab as u32, n_embd: n_embd as u32 };
        if !n_embd.is_multiple_of(u64::from(info.ty.block_values()))
            || gguf.data(info).len() as u64 != shape.bytes()
        {
            return Err(invalid(format!("{name} is no table of whole {} rows", info.ty)));
        }
        Ok(Self { shape, home: Home::Mapped { gguf, tensor } })
    }

    /// The table's type and size.
    pub fn shape(&self) -> TableShape {
        self.shape
    }

    /// The page-locked bytes the table holds: none when it is mapped.
    pub fn locked(&self) -> u64 {
        match &self.home {
            Home::Pinned(table) => table.len() as u64,
            Home::Mapped { .. } => 0,
        }
    }

    fn bytes(&self) -> &[u8] {
        match &self.home {
            Home::Pinned(table) => table,
            Home::Mapped { gguf, tensor } => gguf.data(&gguf.tensors()[*tensor]),
        }
    }

    /// The row of `token` in the file's form; none past the table.
    pub fn row(&self, token: u32) -> Option<&[u8]> {
        let size = self.shape.row_bytes() as usize;
        (token < self.shape.n_vocab).then(|| &self.bytes()[token as usize * size..][..size])
    }

    /// Copies the row of each of `tokens` into `out`, row `r` for `tokens[r]`
    /// (a token past the table leaves its row as it was: the gather gives
    /// zeros for it). Returns the bytes of the rows.
    pub fn stage(&self, tokens: &[u32], out: &mut [u8]) -> Result<usize, ModelError> {
        let size = self.shape.row_bytes() as usize;
        if tokens.len() * size > out.len() {
            return Err(invalid(format!(
                "{} rows of {size} bytes; the stage holds {} bytes",
                tokens.len(),
                out.len()
            )));
        }
        for (r, &token) in tokens.iter().enumerate() {
            if let Some(row) = self.row(token) {
                out[r * size..(r + 1) * size].copy_from_slice(row);
            }
        }
        Ok(tokens.len() * size)
    }

    /// The row of `token` decoded into `out` (`n_embd` values): zeros for a
    /// token past the table, as the gather gives it. For the host's own use of
    /// a row (a drafter's constant input), not for a step's rows.
    pub fn decode(&self, token: u32, out: &mut [f32]) -> Result<(), ModelError> {
        match self.row(token) {
            Some(row) => ares_quant::dequantize(self.shape.ty, row, out)
                .map_err(|e| invalid(format!("the embedding of token {token}: {e}"))),
            None => {
                out.fill(0.0);
                Ok(())
            }
        }
    }
}

/// A stage: the rows of a step's tokens, copied from the host table in the
/// order of the rows for one upload.
pub(crate) struct Stage {
    table: Arc<HostTable>,
    buffer: PinnedBuffer,
    /// Rows the stage holds.
    capacity: u32,
}

impl std::fmt::Debug for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage").field("capacity", &self.capacity).finish_non_exhaustive()
    }
}

impl Stage {
    /// The page-locked bytes of a stage of `rows` rows of `shape`.
    pub fn locked_bytes(shape: &TableShape, rows: u32) -> u64 {
        u64::from(rows) * shape.row_bytes()
    }

    /// A stage of `rows` rows of `table` on `device`.
    pub fn new(device: &Device, table: Arc<HostTable>, rows: u32) -> Result<Self, ModelError> {
        let bytes = Self::locked_bytes(&table.shape(), rows) as usize;
        Ok(Self { buffer: PinnedBuffer::new(device, bytes)?, table, capacity: rows })
    }

    /// The table the rows come from.
    pub fn table(&self) -> &Arc<HostTable> {
        &self.table
    }

    /// The bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.table.shape().row_bytes() as usize
    }

    /// The page-locked bytes of the stage.
    pub fn locked(&self) -> u64 {
        self.buffer.len() as u64
    }

    /// Copies the row of each of `tokens` into the stage, row `r` for
    /// `tokens[r]` (a token past the table leaves its row as it was: the
    /// gather gives zeros for it). Returns where the bytes to upload are and
    /// how many.
    pub fn stage(&mut self, tokens: &[u32]) -> Result<(*const u8, usize), ModelError> {
        if tokens.len() > self.capacity as usize {
            return Err(invalid(format!(
                "a step of {} rows; the stage of the embedding holds {}",
                tokens.len(),
                self.capacity
            )));
        }
        let bytes = self.table.stage(tokens, &mut self.buffer)?;
        Ok((self.buffer.as_ptr(), bytes))
    }

    /// Enqueues the upload of the rows of `tokens` to `to` on `stream`. The
    /// upload of the step before has run: the stage is written again here.
    ///
    /// # Safety
    ///
    /// `to` must have room for a row of the table's form per token, on the
    /// stream's device.
    pub unsafe fn upload(
        &mut self,
        stream: &Stream,
        to: DevicePtr,
        tokens: &[u32],
    ) -> Result<(), ModelError> {
        let (from, bytes) = self.stage(tokens)?;
        // SAFETY: the pinned stage is not written again before the next
        // step, which is enqueued after this one has run; `to` is the
        // caller's.
        unsafe { stream.copy_from_host(to, from, bytes)? };
        Ok(())
    }
}

/// What the token rows of a part take and where they lie, for the report at
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
    /// Rows of the table that lie on the device, for the drafters that pick
    /// on the device.
    pub kept_rows: u32,
    /// What they take on the device (in the weights): none when they are the
    /// rows of a tied head, which it holds anyway.
    pub kept_bytes: u64,
    /// Whether the kept rows are the first rows of the head.
    pub shared_with_head: bool,
    /// The room in the step's scratch for the rows the host stages.
    pub staged_bytes: u64,
}

/// Where a gather finds the rows of `shape`: the rows a device keeps and the
/// rows staged at `staged` (null for none).
pub(crate) fn source(shape: &TableShape, kept: KeptRows, staged: DevicePtr) -> EmbedRows {
    EmbedRows { ty: shape.ty, n_vocab: shape.n_vocab, n_embd: shape.n_embd, kept, staged }
}

/// Where the rows of a [`Splice`] are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpliceFrom {
    /// On the step's device.
    Device(DevicePtr),
    /// In page-locked host memory: the runtime's image stage.
    Host(HostRows),
}

/// The address of rows in page-locked host memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostRows(pub *const f32);

// SAFETY: an address the runtime keeps valid and unchanged until the step
// that reads it has run; nothing is read through it on the host.
unsafe impl Send for HostRows {}

/// Rows `row..row + rows` of a step take the `rows * n_embd` f32 at `from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Splice {
    pub row: u32,
    pub rows: u32,
    pub from: SpliceFrom,
}

/// The splices of a step's image runs: the rows the runtime staged for its
/// image cells, read from the page-locked stage at `stage`.
pub(crate) fn image_splices(
    record: &ares_runtime::RecordView<'_>,
    stage: *const f32,
    n_embd: u32,
) -> Vec<Splice> {
    if stage.is_null() {
        return Vec::new();
    }
    (0..record.image_runs())
        .map(|i| {
            let run = record.image_run(i);
            // The stage holds a row of `n_embd` values per staged row.
            let from = stage.wrapping_add(run.stage as usize * n_embd as usize);
            Splice { row: run.row, rows: run.rows, from: SpliceFrom::Host(HostRows(from)) }
        })
        .collect()
}

/// The second row source: copies rows that another computation made over
/// gathered rows. `out` holds the step's rows (`n_embd` f32 each). The image
/// tower's rows enter a step here, at the cells of an image.
///
/// # Safety
///
/// `out` must hold every row the splices name, and each source its rows of
/// `n_embd` f32 values (on the stream's device, or in page-locked host
/// memory), alive and unchanged until the copies have run.
pub(crate) unsafe fn splice(
    stream: &Stream,
    out: DevicePtr,
    n_embd: u32,
    splices: &[Splice],
) -> Result<(), ModelError> {
    let bytes = u64::from(n_embd) * 4;
    for s in splices {
        let to = out.offset(u64::from(s.row) * bytes);
        let len = (u64::from(s.rows) * bytes) as usize;
        // SAFETY: the caller's guarantee.
        unsafe {
            match s.from {
                SpliceFrom::Device(from) => stream.copy(to, from, len)?,
                SpliceFrom::Host(HostRows(from)) => {
                    stream.copy_from_host(to, from.cast::<u8>(), len)?
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_of_q8_rows_has_rows_of_34_bytes_per_block() {
        let shape = TableShape { ty: GgmlType::Q8_0, n_vocab: 10, n_embd: 64 };
        assert_eq!(shape.row_bytes(), 68);
        assert_eq!(shape.bytes(), 680);
        assert_eq!(Stage::locked_bytes(&shape, 3), 204);
    }

    #[test]
    fn a_mapped_table_stages_the_rows_of_its_tokens_and_decodes_them_as_the_file_does() {
        // Five tokens of 32 values in q8_0: a scale of 0.5 and the values
        // `token + j` as codes.
        let row = |token: u32| -> Vec<u8> {
            let mut bytes = 0x3800u16.to_le_bytes().to_vec(); // 0.5 in f16
            bytes.extend((0..32).map(|j| (token as i8 + j as i8) as u8));
            bytes
        };
        let data: Vec<u8> = (0..5).flat_map(row).collect();
        let dir = std::env::temp_dir().join(format!("ares-rows-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("table.gguf");
        ares_gguf::GgufBuilder::new()
            .str("general.architecture", "test")
            .tensor(TOKEN_EMBD, &[32, 5], GgmlType::Q8_0, data)
            .write(&path)
            .unwrap();
        let table = HostTable::mapped(Arc::new(Gguf::open(&path).unwrap()), TOKEN_EMBD).unwrap();
        assert_eq!(table.shape(), TableShape { ty: GgmlType::Q8_0, n_vocab: 5, n_embd: 32 });
        assert_eq!(table.locked(), 0);
        let mut staged = vec![0xffu8; 3 * 34];
        // A token past the table leaves its row as it was.
        assert_eq!(table.stage(&[3, 9, 1], &mut staged).unwrap(), 3 * 34);
        assert_eq!(&staged[..34], row(3).as_slice());
        assert!(staged[34..68].iter().all(|b| *b == 0xff));
        assert_eq!(&staged[68..], row(1).as_slice());
        assert!(table.stage(&[0; 4], &mut staged).is_err(), "four rows do not fit three");
        let mut values = vec![0f32; 32];
        table.decode(4, &mut values).unwrap();
        assert_eq!(values, (0..32).map(|j| 0.5 * (4 + j) as f32).collect::<Vec<_>>());
        table.decode(5, &mut values).unwrap();
        assert!(values.iter().all(|v| *v == 0.0), "zeros past the table");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
