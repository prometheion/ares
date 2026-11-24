//! A device's arena of weights, the same for every model: every tensor at a
//! multiple of [`ALIGN`], its bytes from the model file as they are, rows or
//! columns of a matrix gathered into a matrix of their own (the same bytes per
//! value, never decoded), or values the host made.
//!
//! The bytes are read from the model file with ordinary reads and uploaded
//! from the loader's memory ([`MappedFiles`]).

use std::ops::Range;

use ares_core::GgmlType;
use ares_cuda::{DeviceBuffer, WeightRange};
use ares_runtime::ModelError;

use crate::files::Matrix;

/// Every tensor starts at a multiple of this in the arena: more than any
/// weight kernel asks for (16 bytes).
pub(crate) const ALIGN: u64 = 64;

/// Where the bytes of a tensor on the device come from.
pub(crate) enum Source<'g> {
    /// Stored bytes as they are.
    Bytes(&'g [u8]),
    /// Rows of a matrix, range after range.
    Rows(Matrix<'g>, Vec<Range<usize>>),
    /// The same columns of every row of a matrix, range after range. The
    /// ranges are whole blocks of the matrix's type.
    Columns(Matrix<'g>, Vec<Range<usize>>),
    /// f32 values.
    Floats(Vec<f32>),
    /// u32 values.
    Words(Vec<u32>),
}

pub(crate) fn row_bytes(m: &Matrix<'_>) -> usize {
    m.bytes().len() / m.rows().max(1)
}

impl Source<'_> {
    pub(crate) fn len(&self) -> u64 {
        (match self {
            Source::Bytes(b) => b.len(),
            Source::Rows(m, ranges) => ranges.iter().map(|r| r.len()).sum::<usize>() * row_bytes(m),
            Source::Columns(m, ranges) => {
                let (bv, bb) = (m.ty().block_values() as usize, m.ty().block_bytes() as usize);
                m.rows() * ranges.iter().map(|r| r.len() / bv * bb).sum::<usize>()
            }
            Source::Floats(f) => f.len() * 4,
            Source::Words(w) => w.len() * 4,
        }) as u64
    }

    /// Writes the bytes at offset `at` of `arena`. The upload has reached
    /// the device when this returns.
    pub(crate) fn write(
        &self,
        arena: &DeviceBuffer,
        at: u64,
        reader: &mut Reader,
    ) -> Result<(), ModelError> {
        match self {
            Source::Bytes(b) => reader.upload(arena, at, b)?,
            Source::Rows(m, ranges) => {
                let size = row_bytes(m);
                let mut at = at;
                for r in ranges {
                    let bytes = &m.bytes()[r.start * size..r.end * size];
                    reader.upload(arena, at, bytes)?;
                    at += bytes.len() as u64;
                }
            }
            Source::Columns(m, ranges) => {
                let (bv, bb) = (m.ty().block_values() as usize, m.ty().block_bytes() as usize);
                let size = row_bytes(m);
                let Reader { files, staging, gathered } = reader;
                gathered.clear();
                // The rows of the matrix, a piece at a time, and of each row
                // the columns' blocks.
                let per_piece = (PIECE / size.max(1)).max(1) * size;
                for piece in m.bytes().chunks(per_piece) {
                    let rows = files.read(piece, staging);
                    for row in rows.chunks_exact(size) {
                        for r in ranges {
                            gathered.extend_from_slice(&row[r.start / bv * bb..r.end / bv * bb]);
                        }
                    }
                }
                arena.write(at, gathered)?;
            }
            Source::Floats(f) => {
                let bytes = &mut reader.gathered;
                bytes.clear();
                bytes.extend(f.iter().flat_map(|v| v.to_le_bytes()));
                arena.write(at, bytes)?;
            }
            Source::Words(w) => {
                let bytes = &mut reader.gathered;
                bytes.clear();
                bytes.extend(w.iter().flat_map(|v| v.to_le_bytes()));
                arena.write(at, bytes)?;
            }
        }
        Ok(())
    }
}

/// The bytes read from a file at a time.
const PIECE: usize = 16 << 20;

/// The files behind the mapped bytes of a model, for reading them with
/// ordinary reads.
///
/// The stored bytes of a matrix are a slice of a mapping of the model file.
/// Uploading from the mapping makes the driver fault the file in page by
/// page, and the kernel stops reading ahead on a mapping whose faults keep
/// missing the page cache: on a machine short of memory a load of
/// Qwen3.8-27B then read 38 MB per second from a drive that gives 900 to an
/// ordinary read. So the loader reads the same bytes from the file itself, a
/// piece at a time into its own memory, and uploads from there; reads of a
/// file are read ahead whatever the page cache does. Which file and where is
/// taken from the process's table of mappings, so nothing above the loader
/// needs to know; bytes that are not a file's (a model built in memory) are
/// uploaded from where they are.
#[derive(Default)]
struct MappedFiles {
    /// The file mappings of the process: start and end address, the file's
    /// offset at the start, and its path.
    regions: Vec<(usize, usize, u64, std::path::PathBuf)>,
    /// The files opened so far, by index of their first region.
    open: Vec<(std::path::PathBuf, std::fs::File)>,
}

impl MappedFiles {
    /// The file mappings of the process as they are now.
    fn of_process() -> Self {
        let mut regions = Vec::new();
        // start-end perms offset dev inode path
        for line in std::fs::read_to_string("/proc/self/maps").unwrap_or_default().lines() {
            let mut fields = line.splitn(6, ' ');
            let (Some(range), Some(perms), Some(offset)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let path = fields.nth(2).unwrap_or("").trim_start();
            let Some((start, end)) = range.split_once('-') else { continue };
            let parsed = (
                usize::from_str_radix(start, 16),
                usize::from_str_radix(end, 16),
                u64::from_str_radix(offset, 16),
            );
            if let (Ok(start), Ok(end), Ok(offset)) = parsed
                && perms.starts_with('r')
                && path.starts_with('/')
            {
                regions.push((start, end, offset, path.into()));
            }
        }
        Self { regions, open: Vec::new() }
    }

    /// The bytes of `mapped`: read from its file into `staging` when it lies
    /// in a file mapping and the read succeeds, `mapped` itself otherwise.
    fn read<'a>(&mut self, mapped: &'a [u8], staging: &'a mut Vec<u8>) -> &'a [u8] {
        use std::os::unix::fs::FileExt;
        let at = mapped.as_ptr() as usize;
        let Some((start, _, offset, path)) =
            self.regions.iter().find(|(start, end, ..)| at >= *start && at + mapped.len() <= *end)
        else {
            return mapped;
        };
        let known = self.open.iter().position(|(p, _)| p == path);
        let file = match known {
            Some(i) => &self.open[i].1,
            None => match std::fs::File::open(path) {
                Ok(file) => {
                    self.open.push((path.clone(), file));
                    &self.open.last().expect("pushed above").1
                }
                Err(_) => return mapped,
            },
        };
        staging.resize(mapped.len(), 0);
        match file.read_exact_at(staging, offset + (at - start) as u64) {
            Ok(()) => staging,
            Err(_) => mapped,
        }
    }
}

/// Reads the stored bytes of `m` into `into`, which has their length: with
/// ordinary reads, a piece at a time, as the uploads do.
pub(crate) fn read_matrix(m: &Matrix<'_>, into: &mut [u8]) {
    let mut files = MappedFiles::of_process();
    let mut staging = Vec::new();
    for (piece, to) in m.bytes().chunks(PIECE).zip(into.chunks_mut(PIECE)) {
        to.copy_from_slice(files.read(piece, &mut staging));
    }
}

/// What the uploads of a load share: the files behind the model's bytes and
/// the memory they are read and gathered into.
#[derive(Default)]
pub(crate) struct Reader {
    files: MappedFiles,
    staging: Vec<u8>,
    gathered: Vec<u8>,
}

impl Reader {
    /// The uploads of a load, reading through the file mappings the process
    /// has now.
    pub(crate) fn new() -> Self {
        Self { files: MappedFiles::of_process(), ..Self::default() }
    }

    /// Uploads `bytes` to offset `at` of `arena`, a piece at a time.
    pub(crate) fn upload(
        &mut self,
        arena: &DeviceBuffer,
        at: u64,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        for (i, piece) in bytes.chunks(PIECE).enumerate() {
            let read = self.files.read(piece, &mut self.staging);
            arena.write(at + (i * PIECE) as u64, read)?;
        }
        Ok(())
    }
}

/// The ranges of the float matrices of a load, by the address of their stored
/// bytes: a matrix is read once for it, with ordinary reads as the uploads
/// do, whatever the number of layouts and parts it is placed in. The range is
/// that of the whole stored matrix, also where a device holds a part of it.
#[derive(Default)]
pub(crate) struct Ranges {
    known: Vec<(usize, WeightRange)>,
    files: Option<MappedFiles>,
    staging: Vec<u8>,
}

impl Ranges {
    pub(crate) fn of(&mut self, m: &Matrix<'_>) -> WeightRange {
        if !matches!(m.ty(), GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return WeightRange::Any;
        }
        let at = m.bytes().as_ptr() as usize;
        if let Some((_, range)) = self.known.iter().find(|(known, _)| *known == at) {
            return *range;
        }
        let files = self.files.get_or_insert_with(MappedFiles::of_process);
        let mut range = WeightRange::Any;
        for piece in m.bytes().chunks(PIECE) {
            range = range.and(WeightRange::of(m.ty(), files.read(piece, &mut self.staging)));
        }
        self.known.push((at, range));
        range
    }
}

/// The number of values of `columns` of `m` gathered into rows of their own:
/// refused unless they are whole blocks of the matrix's type.
pub(crate) fn columns_in(m: &Matrix<'_>, columns: &[Range<usize>]) -> Result<u32, ModelError> {
    let bv = m.ty().block_values() as usize;
    if columns.iter().any(|r| !r.start.is_multiple_of(bv) || !r.end.is_multiple_of(bv)) {
        return Err(ModelError::Invalid(format!(
            "columns {columns:?} of a {} matrix are not whole blocks of {bv} values",
            m.ty()
        )));
    }
    Ok(columns.iter().map(Range::len).sum::<usize>() as u32)
}

/// The ranges of the value heads of a gated delta rule that read the key
/// heads `keys` of `n_k_heads`, of `n_v_heads` in all, each head `per_head`
/// items wide: value head `h` reads key head `h mod n_k_heads`.
pub(crate) fn value_ranges(
    n_k_heads: usize,
    n_v_heads: usize,
    keys: &Range<usize>,
    per_head: usize,
) -> Vec<Range<usize>> {
    (0..n_v_heads / n_k_heads)
        .map(|tile| {
            (tile * n_k_heads + keys.start) * per_head..(tile * n_k_heads + keys.end) * per_head
        })
        .collect()
}

/// The values of `ranges` of `values`, range after range.
pub(crate) fn gather(values: &[f32], ranges: &[Range<usize>]) -> Vec<f32> {
    ranges.iter().flat_map(|r| values[r.clone()].iter().copied()).collect()
}
