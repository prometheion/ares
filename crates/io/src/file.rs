//! Opening files for direct I/O, and the alignment arithmetic that goes with it.

use std::fmt;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Names one of the files a [`Reader`](crate::Reader) was opened with. For a
/// reader opened on a GGUF model ([`Reader::open_gguf`](crate::Reader::open_gguf))
/// it is the number of the part, which is what `TensorInfo::part` holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(pub u16);

impl fmt::Display for FileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How a file is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoMode {
    /// `O_DIRECT`: the page cache is bypassed and reads go to the device.
    ///
    /// The filesystem has the last word: btrfs, for one, quietly reads
    /// compressed extents through the page cache (a model file written to a
    /// `compress=zstd` volume has a few of them, mostly where tensors
    /// compress well), and the reads are still correct, only not uncached.
    Direct,
    /// Ordinary cached reads, for the stated reason.
    Buffered(BufferedReason),
}

/// Why a file is read through the page cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferedReason {
    /// [`Config::direct`](crate::Config::direct) is off.
    Disabled,
    /// The file is not on a block device whose block size can be found (tmpfs
    /// and network filesystems are not), so there is nothing to bypass or
    /// nothing to align to.
    Unsupported,
    /// The filesystem advertises direct I/O but refused to switch it on; the
    /// number is the `errno`.
    Refused(i32),
}

impl fmt::Display for IoMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoMode::Direct => f.write_str("direct I/O"),
            IoMode::Buffered(BufferedReason::Disabled) => {
                f.write_str("buffered reads (direct I/O is switched off)")
            }
            IoMode::Buffered(BufferedReason::Unsupported) => {
                f.write_str("buffered reads (the file is not on a block device)")
            }
            IoMode::Buffered(BufferedReason::Refused(errno)) => write!(
                f,
                "buffered reads (the filesystem refused direct I/O: {})",
                std::io::Error::from_raw_os_error(*errno)
            ),
        }
    }
}

/// What direct reads on a file require.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectAlign {
    /// File offsets and lengths must be multiples of this: the logical block
    /// size of the device.
    pub offset: u32,
    /// Memory addresses must be multiples of this.
    pub memory: u32,
    pub from: AlignSource,
}

/// Where the alignment came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignSource {
    /// The filesystem told the kernel's `statx` (ext4, xfs, f2fs).
    Statx,
    /// The filesystem says nothing (btrfs). This is the larger of the logical
    /// block size of the device it is on (from sysfs) and the filesystem's own
    /// block size.
    BlockSizes,
}

/// One file of a reader, as it was opened.
#[derive(Clone, Debug)]
pub struct FileInfo {
    pub path: PathBuf,
    pub size: u64,
    /// How the file is read. This is the one place a fallback to cached reads
    /// is reported.
    pub mode: IoMode,
    /// What the kernel reported for direct I/O on this file, if anything.
    pub direct_align: Option<DirectAlign>,
    /// The granule reads are widened to: file offsets, lengths and buffer
    /// addresses are multiples of it. It is 1 when the file is read
    /// buffered.
    pub block: u64,
}

pub(crate) struct OpenFile {
    pub(crate) file: File,
    pub(crate) info: FileInfo,
}

/// Opens `path`, with `O_DIRECT` where the filesystem supports it.
pub(crate) fn open(path: &Path, direct: bool) -> Result<OpenFile> {
    let open_err = |source| Error::Open { path: path.to_owned(), source };
    let file = File::open(path).map_err(open_err)?;
    let size = file.metadata().map_err(open_err)?.len();

    let direct_align = dio_alignment(&file);
    let mut info = FileInfo {
        path: path.to_owned(),
        size,
        mode: IoMode::Buffered(BufferedReason::Disabled),
        direct_align,
        block: 1,
    };
    if !direct {
        return Ok(OpenFile { file, info });
    }
    let Some(align) = direct_align else {
        info.mode = IoMode::Buffered(BufferedReason::Unsupported);
        return Ok(OpenFile { file, info });
    };
    // Switching the flag on the descriptor instead of reopening the path
    // keeps the file and its alignment report the same object.
    match set_direct(&file) {
        Ok(()) => {
            info.mode = IoMode::Direct;
            // One granule for everything keeps the arithmetic simple; the
            // two numbers are equal on every device seen so far (the memory
            // one is smaller, if anything).
            info.block = u64::from(align.offset.max(align.memory));
        }
        Err(errno) => info.mode = IoMode::Buffered(BufferedReason::Refused(errno)),
    }
    Ok(OpenFile { file, info })
}

fn set_direct(file: &File) -> std::result::Result<(), i32> {
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a valid open descriptor owned by `file`; F_GETFL and
    // F_SETFL take and return plain integers.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags < 0 || libc::fcntl(fd, libc::F_SETFL, flags | libc::O_DIRECT) < 0 {
            return Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EINVAL));
        }
    }
    Ok(())
}

/// The direct I/O alignment of the file, or `None` when direct I/O cannot be
/// used on it.
///
/// The kernel can say what it requires (`statx` with `STATX_DIOALIGN`), but
/// not every filesystem does: btrfs, for one, does not. The requirement is
/// then the larger of the logical block size of the device under the
/// filesystem (read from sysfs) and the filesystem's block size. A filesystem that is not on a block device (tmpfs, a
/// network share) has none, and a file on it is read through the page cache:
/// there is no cache to bypass on tmpfs, and the others make no promise.
fn dio_alignment(file: &File) -> Option<DirectAlign> {
    // SAFETY: an all-zero `statx` is a valid value (plain integers).
    let mut st: libc::statx = unsafe { std::mem::zeroed() };
    // SAFETY: `st` is a valid out-pointer; with AT_EMPTY_PATH the empty path
    // names the descriptor itself.
    let rc = unsafe {
        libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            libc::STATX_DIOALIGN | libc::STATX_MNT_ID,
            &mut st,
        )
    };
    if rc != 0 {
        return None;
    }
    let valid = |n: u32| n != 0 && n.is_power_of_two();
    if st.stx_mask & libc::STATX_DIOALIGN != 0
        && valid(st.stx_dio_offset_align)
        && valid(st.stx_dio_mem_align)
    {
        return Some(DirectAlign {
            offset: st.stx_dio_offset_align,
            memory: st.stx_dio_mem_align,
            from: AlignSource::Statx,
        });
    }
    let mnt_id = (st.stx_mask & libc::STATX_MNT_ID != 0).then_some(st.stx_mnt_id);
    let device =
        device_of(&st, mnt_id).and_then(|(major, minor)| logical_block_size(major, minor))?;
    // A filesystem that does not report its requirement may need more than
    // the device does: btrfs reads directly only when offset, length and
    // address are multiples of its sector size (4096) and otherwise quietly
    // serves the read through the page cache. Its `st_blksize` is that size.
    let fs_block = st.stx_blksize;
    let block = if fs_block.is_power_of_two() { device.max(fs_block) } else { device };
    Some(DirectAlign { offset: block, memory: block, from: AlignSource::BlockSizes })
}

/// The block device the file's filesystem lives on, as (major, minor): the
/// file's own device if that is a block device (ext4, xfs), else the source
/// of its mount (btrfs reports an anonymous device number per subvolume).
fn device_of(st: &libc::statx, mnt_id: Option<u64>) -> Option<(u32, u32)> {
    if logical_block_size(st.stx_dev_major, st.stx_dev_minor).is_some() {
        return Some((st.stx_dev_major, st.stx_dev_minor));
    }
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let line = mountinfo
        .lines()
        .find(|l| l.split(' ').next().and_then(|id| id.parse::<u64>().ok()) == mnt_id)?;
    // `36 35 98:0 /root /mnt rw,noatime shared:1 - btrfs /dev/nvme0n1p2 rw`
    let (_, after) = line.split_once(" - ")?;
    let source = after.split(' ').nth(1)?;
    if !source.starts_with("/dev/") {
        return None;
    }
    let meta = std::fs::metadata(source).ok()?;
    if !meta.file_type().is_block_device() {
        return None;
    }
    let rdev = meta.rdev();
    // glibc's encoding of a device number.
    let major = (((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff)) as u32;
    let minor = ((rdev & 0xff) | ((rdev >> 12) & !0xff)) as u32;
    Some((major, minor))
}

/// The logical block size of a block device from sysfs. A partition has no
/// queue of its own; its disk's is one level up.
fn logical_block_size(major: u32, minor: u32) -> Option<u32> {
    let dir = format!("/sys/dev/block/{major}:{minor}");
    let text = std::fs::read_to_string(format!("{dir}/queue/logical_block_size"))
        .or_else(|_| std::fs::read_to_string(format!("{dir}/../queue/logical_block_size")))
        .ok()?;
    let size: u32 = text.trim().parse().ok()?;
    (size >= 512 && size.is_power_of_two()).then_some(size)
}

/// A byte range widened to whole blocks, and where the wanted bytes sit in
/// it.
///
/// GGUF aligns tensor data to 32 bytes while direct I/O works in the
/// device's blocks (512 or 4096 bytes), so an expert or a row rarely starts
/// on a block. The read covers the surrounding blocks and the data is used
/// where it lands, at `skip`: nothing is copied or moved afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// File offset the read starts at: a multiple of the block size at or
    /// before the wanted range.
    pub start: u64,
    /// Bytes the read covers, a multiple of the block size. The buffer must
    /// be at least this long. The file may end inside the last block; the
    /// read is then short, which is fine as long as the wanted bytes are
    /// there.
    pub len: usize,
    /// Where the first wanted byte is in the buffer.
    pub skip: usize,
    /// The number of wanted bytes.
    pub wanted: usize,
}

impl Span {
    /// The span of `len` bytes at `offset` for reads in blocks of `block`
    /// bytes (a power of two; 1 means no widening). `None` when it does not
    /// fit in the address space.
    pub fn new(offset: u64, len: usize, block: u64) -> Option<Self> {
        debug_assert!(block.is_power_of_two());
        if len == 0 {
            return Some(Self { start: offset, len: 0, skip: 0, wanted: 0 });
        }
        let start = offset & !(block - 1);
        let end = offset.checked_add(len as u64)?;
        let end = end.checked_add(block - 1)? & !(block - 1);
        // io_uring takes the offset as signed and reads -1 as "the current
        // position", so nothing past `i64::MAX` is addressable.
        if end > i64::MAX as u64 {
            return None;
        }
        let span_len = usize::try_from(end - start).ok()?;
        Some(Self { start, len: span_len, skip: (offset - start) as usize, wanted: len })
    }
}

/// The span of a read of `file` among `files`.
pub(crate) fn span_in(files: &[FileInfo], file: FileId, offset: u64, len: usize) -> Result<Span> {
    let info = files.get(usize::from(file.0)).ok_or(Error::UnknownFile(file, files.len()))?;
    Span::new(offset, len, info.block).ok_or(Error::RangeOverflow { offset, len })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_widens_to_blocks_and_locates_the_data() {
        // Block 4096, 100 bytes at 5000: block 1 only.
        let s = Span::new(5000, 100, 4096).unwrap();
        assert_eq!(s, Span { start: 4096, len: 4096, skip: 904, wanted: 100 });
        // Straddling a block boundary needs both blocks.
        let s = Span::new(4090, 100, 4096).unwrap();
        assert_eq!(s, Span { start: 0, len: 8192, skip: 4090, wanted: 100 });
        // Exactly one block, aligned, needs no widening.
        let s = Span::new(8192, 4096, 4096).unwrap();
        assert_eq!(s, Span { start: 8192, len: 4096, skip: 0, wanted: 4096 });
        // Ending exactly on a boundary does not take the next block.
        let s = Span::new(4096 - 32, 32, 4096).unwrap();
        assert_eq!(s, Span { start: 0, len: 4096, skip: 4064, wanted: 32 });
        // Block 1 is the buffered case: nothing is widened.
        let s = Span::new(1234, 56, 1).unwrap();
        assert_eq!(s, Span { start: 1234, len: 56, skip: 0, wanted: 56 });
    }

    #[test]
    fn empty_and_overflowing_spans() {
        assert_eq!(Span::new(77, 0, 4096).unwrap().len, 0);
        assert_eq!(Span::new(u64::MAX - 10, 100, 4096), None);
        assert_eq!(Span::new(u64::MAX - 10, 5, 4096), None, "rounding the end up overflows");
        assert_eq!(Span::new(i64::MAX as u64 - 4096, 5000, 4096), None, "offsets are signed");
        assert!(Span::new(1 << 40, 100, 4096).is_some(), "a terabyte offset is fine");
    }

    #[test]
    fn span_covers_the_range_for_every_offset() {
        for block in [512u64, 4096] {
            for offset in 0..2 * block + 3 {
                for len in [1usize, 2, 31, 32, 33, 511, 512, 513, 4095, 4096, 4097, 10_000] {
                    let s = Span::new(offset, len, block).unwrap();
                    assert_eq!(s.start % block, 0);
                    assert_eq!(s.len as u64 % block, 0);
                    assert_eq!(s.start + s.skip as u64, offset);
                    assert!(s.skip + len <= s.len, "data must lie inside the buffer");
                    assert!(s.skip < block as usize, "no whole block is read for nothing");
                    assert!(s.len - s.skip - len < block as usize, "no whole block after it");
                }
            }
        }
    }
}
