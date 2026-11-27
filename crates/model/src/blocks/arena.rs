//! A device's arena of weights, the same for every model: every tensor at a
//! multiple of [`ALIGN`], its bytes from the model file as they are, rows or
//! columns of a matrix gathered into a matrix of their own (the same bytes per
//! value, never decoded), or values the host made.
//!
//! The bytes of the model file are uploaded from its mapping where its pages
//! are in the page cache; the others are read from the file with ordinary
//! reads into the loader's memory and uploaded from there ([`MappedFiles`]).

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
/// ordinary reads where that is faster than the mapping.
///
/// The stored bytes of a matrix are a slice of a mapping of the model file.
/// Uploading from the mapping makes the driver fault the file in page by
/// page, and Linux reads around a fault only while the faults of that file
/// mostly find the page cache: every fault that finds nothing counts a miss,
/// every hit takes one off, and past a hundred misses (`MMAP_LOTSAMISS` in
/// `mm/filemap.c`) every fault reads one page. A hit on a page that was read
/// again soon after the kernel took it back (a refault, the folio marked
/// `workingset`) takes nothing off. So a load of Qwen3.8-27B read 38 MB per
/// second from a drive that gives 900 to an ordinary read on a machine short
/// of memory, and 5 to 20 MB per second (659 s for the weights of one device
/// instead of 17) when it was loaded again in the same process soon after
/// the kernel had taken the file's pages back from the first engine (GitHub
/// issue #66, `docs/dev/experiments/load-and-stop.md`); a copy from a cold
/// mapping without read-around gave 2 MB per second there, against 390 MB
/// per second for ordinary reads and for the mapping with read-around. An
/// ordinary read is read ahead by its own size and counts no misses, so bytes
/// whose pages are not all in the page cache are read from the file itself, a
/// piece at a time into the loader's memory, and uploaded from there; bytes
/// that are all there are uploaded from the mapping, which saves that copy. Which file and where is
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

    /// The bytes of `mapped`: `mapped` itself when its pages are in memory,
    /// otherwise read from its file into `staging` when it lies in a file
    /// mapping and the read succeeds (and `mapped` when it does not).
    fn read<'a>(&mut self, mapped: &'a [u8], staging: &'a mut Vec<u8>) -> &'a [u8] {
        use std::os::unix::fs::FileExt;
        if resident(mapped) {
            return mapped;
        }
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

/// Whether every page of `bytes` is in memory: for bytes of a file mapping,
/// in the page cache (one `mincore` over their pages).
fn resident(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    // SAFETY: `sysconf` reads a constant of the system.
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096);
    let start = bytes.as_ptr() as usize / page * page;
    let end = (bytes.as_ptr() as usize + bytes.len()).next_multiple_of(page);
    let mut pages = vec![0u8; (end - start) / page];
    // SAFETY: `start..end` are the whole pages that `bytes` lies in, all of
    // them mapped while `bytes` is borrowed; `mincore` writes one byte per page
    // into `pages`, which has that many.
    let done =
        unsafe { libc::mincore(start as *mut libc::c_void, end - start, pages.as_mut_ptr()) };
    done == 0 && pages.iter().all(|page| page & 1 == 1)
}

/// Reads the stored bytes of `m` into `into`, which has their length: a
/// piece at a time, from the mapping or with ordinary reads as the uploads do.
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

    /// Uploads `bytes` to offset `at` of `arena`: at once from where they are
    /// when their pages are in memory, otherwise a piece at a time.
    pub(crate) fn upload(
        &mut self,
        arena: &DeviceBuffer,
        at: u64,
        bytes: &[u8],
    ) -> Result<(), ModelError> {
        if resident(bytes) {
            arena.write(at, bytes)?;
            return Ok(());
        }
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

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::{MappedFiles, resident};

    /// Bytes of a file mapping whose pages are not in the page cache are read
    /// from the file with an ordinary read, however much memory the machine
    /// has (a fault of the mapping may read one page at a time); bytes whose
    /// pages are all there come from the mapping itself.
    #[test]
    fn bytes_not_in_the_page_cache_are_read_from_the_file() {
        // Beside the test binary: the build's disk, where the page cache can
        // give pages back (a tmpfs cannot).
        let exe = std::env::current_exe().expect("the test binary");
        let path = exe.with_file_name(format!("arena-read-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..4usize << 20).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &bytes).expect("the build directory is writable");
        let file = std::fs::File::open(&path).expect("just written");
        file.sync_all().expect("written to the disk");
        // SAFETY: a shared read-only mapping of the whole file, unmapped below
        // after its last use.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes.len(),
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(map, libc::MAP_FAILED);
        // SAFETY: the mapping holds `bytes.len()` readable bytes until the
        // `munmap` below.
        let mapped = unsafe { std::slice::from_raw_parts(map as *const u8, bytes.len()) };
        // SAFETY: plain call on an open descriptor.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        let evicted = !resident(mapped);

        let mut files = MappedFiles::of_process();
        let mut staging = Vec::new();
        let read = files.read(mapped, &mut staging);
        assert!(read == &bytes[..], "the file's bytes");
        let from_file = read.as_ptr() != mapped.as_ptr();
        if evicted {
            assert!(from_file, "pages outside the page cache are read with an ordinary read");
        }
        // The read brought them into the page cache: now from the mapping.
        assert!(resident(mapped));
        let again = files.read(mapped, &mut staging);
        assert_eq!(again.as_ptr(), mapped.as_ptr(), "pages in the page cache come from the map");

        // SAFETY: the mapping made above, not used any more.
        unsafe { libc::munmap(map, bytes.len()) };
        let _ = std::fs::remove_file(&path);
    }
}
