//! Memory the kernel reads into.

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::fmt;
use std::ptr::NonNull;

/// A block of memory handed to the reader for one read, and handed back with
/// the completion.
///
/// Direct I/O needs the memory to be aligned (to the file's block size, see
/// [`Reader::buffer_alignment`](crate::Reader::buffer_alignment)), and the
/// engine wants the bytes to land where they are used: in the page-locked
/// staging buffers of a GPU backend, not in a bounce buffer that is copied
/// out of afterwards. A `Buffer` is therefore a plain pointer and length with
/// one of two origins:
///
/// - [`Buffer::aligned`] allocates and owns aligned, zeroed memory. It is for
///   tests, for rows that are consumed on the CPU, and for anything else that
///   needs no special memory.
/// - [`Buffer::from_raw`] lends memory the caller owns (a slot of a pinned
///   ring, say). Nothing is freed when such a buffer is dropped.
///
/// While a buffer is inside the reader nobody else touches the memory, which
/// is why submitting moves it. The reader never allocates memory for a read.
pub struct Buffer {
    ptr: NonNull<u8>,
    len: usize,
    /// The layout to free with; `None` for memory that is only lent.
    owned: Option<Layout>,
}

// SAFETY: a `Buffer` is exclusive access to its memory (allocated by it, or
// lent under the contract of `from_raw`), and moving that access to another
// thread is what the reader does with it.
unsafe impl Send for Buffer {}

impl Buffer {
    /// Allocates `len` zeroed bytes aligned to `align`, which must be a
    /// power of two.
    ///
    /// # Panics
    ///
    /// Panics when `align` is not a power of two or `len` rounded up to it
    /// overflows `isize`, and aborts when memory runs out.
    pub fn aligned(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), align)
            .expect("align must be a power of two and len must fit in isize");
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(ptr) else { handle_alloc_error(layout) };
        Self { ptr, len, owned: Some(layout) }
    }

    /// Lends `len` bytes at `ptr` to the reader.
    ///
    /// # Safety
    ///
    /// The memory must be valid for reads and writes of `len` bytes and
    /// initialised, and it must stay valid, and untouched by anyone else,
    /// until the buffer is returned in a completion or dropped. A buffer that
    /// is still inside the reader when the [`Reader`](crate::Reader) is
    /// dropped is released after the last read into it has finished, so the
    /// memory may be freed once the reader is gone.
    pub unsafe fn from_raw(ptr: *mut u8, len: usize) -> Self {
        Self {
            ptr: NonNull::new(ptr).expect("a buffer needs a non-null pointer"),
            len,
            owned: None,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// The bytes of the buffer. After a completion only the range the
    /// completion names holds data of the read.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the memory is valid and initialised for `len` bytes
        // (`aligned` zeroes it; `from_raw` requires it) and a `Buffer` has
        // exclusive access to it while it exists.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Whether the address is a multiple of `align` (a power of two).
    pub(crate) fn is_aligned_to(&self, align: u64) -> bool {
        (self.ptr.as_ptr() as usize as u64) & (align - 1) == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(layout) = self.owned {
            // SAFETY: `ptr` came from `alloc_zeroed` with exactly this layout
            // and is freed only here.
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("owned", &self.owned.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_memory_is_aligned_zeroed_and_sized() {
        for align in [1, 8, 512, 4096, 1 << 16] {
            let buf = Buffer::aligned(10_000, align);
            assert_eq!(buf.len(), 10_000);
            assert!(buf.is_aligned_to(align as u64), "alignment {align}");
            assert!(buf.as_slice().iter().all(|b| *b == 0));
        }
    }

    #[test]
    fn empty_buffers_are_allowed() {
        let buf = Buffer::aligned(0, 4096);
        assert!(buf.is_empty());
        assert_eq!(buf.as_slice(), &[] as &[u8]);
    }

    #[test]
    fn lent_memory_is_not_freed_by_the_buffer() {
        let mut backing = vec![7u8; 64];
        // SAFETY: `backing` outlives the buffer and nothing else touches it
        // meanwhile.
        let mut buf = unsafe { Buffer::from_raw(backing.as_mut_ptr(), backing.len()) };
        buf.as_mut_slice()[0] = 9;
        drop(buf);
        assert_eq!(backing[0], 9);
        assert_eq!(backing[1], 7);
    }
}
