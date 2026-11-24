//! Images in a prompt: their cells, the rotary positions they give, and the
//! rows the image tower computed for them.
//!
//! An image of `nx x ny` rows takes `nx * ny` cells of its sequence, one per
//! row, in row-major order, between the tokens the chat template writes
//! around it. Its rows are not token embeddings: a model's image tower
//! computes them from the image's pixels ([`Model::vision`](crate::Model)), and
//! the conductor hands them to the step that runs the cells, through the
//! executor's image stage (see the record's module, "Images"). The token of an
//! image cell is the model's pad token, so that whatever reads tokens (the
//! drafter's corpus, the n-gram embedding of Flash-Next, the counts) sees one
//! token there.
//!
//! **Rotary positions** follow llama.cpp's `mtmd` for Qwen3-VL's images
//! (M-RoPE): a text cell at rotary position `p` has `(p, p, p)`; row `j` of an
//! image that starts at rotary position `p` has `(p, p + j / nx, p + j % nx)`,
//! and the cell after the image is at `p + max(nx, ny)`. The rotary position
//! of a text cell is therefore its cell less, for every image before it,
//! `nx * ny - max(nx, ny)`.
//!
//! **The store** keeps the rows of the images the tower has computed, by the
//! image's name, so that a prompt that sends the same image again (every turn
//! of a conversation does) and a replay of cached cells do not run the tower
//! again. It holds at most a number of bytes and lets the least recently used
//! images go.

use std::collections::HashMap;
use std::sync::Arc;

/// The name of an image: a digest of the pixels the tower is given (the
/// resized image) with their width and height. Two files that resize to the
/// same pixels are one image.
pub type ImageName = [u8; 32];

/// An image as the tower takes it.
#[derive(Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub name: ImageName,
    /// The resized image: `width * height` pixels of three bytes (red, green,
    /// blue), row by row.
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    /// Its rows: `nx` across, `ny` down.
    pub nx: u32,
    pub ny: u32,
}

impl std::fmt::Debug for ImageInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageInput")
            .field("name", &short_name(&self.name))
            .field("width", &self.width)
            .field("height", &self.height)
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .finish_non_exhaustive()
    }
}

/// The name of the cell that holds row `row` of the image `name`, for the
/// prefix cache: the top bit set (a text cell's name is its token, which never
/// has it) and 63 bits mixed from the image's name and the row. The image's
/// name covers its pixels, its size and the projector, so a cell's name
/// changes when anything that changes its row does, and the cells after an
/// image change with its size, which is in the name too.
pub fn cell_name(name: &ImageName, row: u32) -> u64 {
    let word = |i: usize| u64::from_le_bytes(name[i..i + 8].try_into().expect("8 bytes"));
    let mut h = word(0) ^ word(8).rotate_left(17) ^ word(16).rotate_left(31) ^ word(24);
    h ^= u64::from(row).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    // The finaliser of SplitMix64.
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h | 1 << 63
}

/// The first eight bytes of a name, as hex: enough to tell images apart in a
/// log.
pub(crate) fn short_name(name: &ImageName) -> String {
    name[..8].iter().map(|b| format!("{b:02x}")).collect()
}

impl ImageInput {
    /// The image's rows, which is the cells it takes.
    pub fn rows(&self) -> u32 {
        self.nx * self.ny
    }

    /// What the image advances the rotary position by.
    pub fn advance(&self) -> u32 {
        self.nx.max(self.ny)
    }
}

/// An image in a prompt: its rows take the cells `first..first + rows`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    pub first: u32,
    pub image: Arc<ImageInput>,
}

impl ImageSpan {
    /// The cell after the image.
    pub fn end(&self) -> u32 {
        self.first + self.image.rows()
    }
}

/// The images of a sequence, in the order of their cells, and the rotary
/// positions they give.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spans {
    spans: Vec<ImageSpan>,
    /// Per span, what the images before it took from the rotary position:
    /// the rotary position of its first cell is `first - shift`.
    shifts: Vec<u32>,
}

impl Spans {
    /// The spans of a prompt. They must be ascending and must not overlap.
    pub fn new(spans: Vec<ImageSpan>) -> Self {
        let mut shifts = Vec::with_capacity(spans.len());
        let mut shift = 0;
        let mut end = 0;
        for span in &spans {
            assert!(span.first >= end, "image spans overlap or are out of order");
            shifts.push(shift);
            shift += span.image.rows() - span.image.advance();
            end = span.end();
        }
        Self { spans, shifts }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn spans(&self) -> &[ImageSpan] {
        &self.spans
    }

    /// The span that holds `cell`, and the row of its image the cell is.
    pub fn at(&self, cell: u32) -> Option<(usize, u32)> {
        let i = self.spans.partition_point(|s| s.end() <= cell);
        let span = self.spans.get(i)?;
        (span.first <= cell).then(|| (i, cell - span.first))
    }

    /// The first span that ends after `cell`: the image `cell` is in, or the
    /// next one.
    pub fn from(&self, cell: u32) -> Option<usize> {
        let i = self.spans.partition_point(|s| s.end() <= cell);
        (i < self.spans.len()).then_some(i)
    }

    /// The rotary position `(time, height, width)` of `cell`.
    pub fn rotary(&self, cell: u32) -> [u32; 3] {
        let i = self.spans.partition_point(|s| s.end() <= cell);
        match self.spans.get(i) {
            Some(span) if span.first <= cell => {
                let p = span.first - self.shifts[i];
                let j = cell - span.first;
                let nx = span.image.nx;
                [p, p + j / nx, p + j % nx]
            }
            _ => {
                let shift = match i.checked_sub(1) {
                    Some(last) => {
                        let image = &self.spans[last].image;
                        self.shifts[last] + image.rows() - image.advance()
                    }
                    None => 0,
                };
                let p = cell - shift;
                [p, p, p]
            }
        }
    }

    /// What the images before `cell` took from the rotary position of a text
    /// cell there: its cell less its rotary position.
    pub fn shift(&self, cell: u32) -> u32 {
        let [t, ..] = self.rotary(cell);
        let i = self.spans.partition_point(|s| s.end() <= cell);
        match self.spans.get(i) {
            Some(span) if span.first <= cell => self.shifts[i],
            _ => cell - t,
        }
    }
}

/// The rows of one image, one after the other: as many as the image has
/// cells, each as wide as the model's rows.
#[derive(Debug)]
pub(crate) struct ImageRows {
    pub values: Vec<f32>,
}

impl ImageRows {
    pub fn bytes(&self) -> u64 {
        self.values.len() as u64 * 4
    }
}

/// The rows of the images the tower has computed, by name, within a limit of
/// bytes: the least recently used go first. An image whose rows are asked for
/// and gone is computed again.
#[derive(Debug)]
pub(crate) struct ImageStore {
    limit: u64,
    held: u64,
    clock: u64,
    images: HashMap<ImageName, (Arc<ImageRows>, u64)>,
}

impl ImageStore {
    /// A store of at most `limit` bytes. It keeps the last image it was given
    /// even when that one alone is larger.
    pub fn new(limit: u64) -> Self {
        Self { limit, held: 0, clock: 0, images: HashMap::new() }
    }

    /// The rows of `name`, if the store has them.
    pub fn get(&mut self, name: &ImageName) -> Option<Arc<ImageRows>> {
        self.clock += 1;
        let clock = self.clock;
        self.images.get_mut(name).map(|(rows, used)| {
            *used = clock;
            rows.clone()
        })
    }

    pub fn contains(&self, name: &ImageName) -> bool {
        self.images.contains_key(name)
    }

    /// Keeps `rows` under `name`, and lets the least recently used images go
    /// until the store is within its limit again.
    pub fn put(&mut self, name: ImageName, rows: Arc<ImageRows>) {
        self.clock += 1;
        self.held += rows.bytes();
        if let Some((old, _)) = self.images.insert(name, (rows, self.clock)) {
            self.held -= old.bytes();
        }
        while self.held > self.limit && self.images.len() > 1 {
            let oldest = self
                .images
                .iter()
                .filter(|(n, _)| **n != name)
                .min_by_key(|(_, (_, used))| *used)
                .map(|(n, _)| *n)
                .expect("another image");
            let (gone, _) = self.images.remove(&oldest).expect("held");
            self.held -= gone.bytes();
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.images.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(nx: u32, ny: u32, seed: u8) -> Arc<ImageInput> {
        Arc::new(ImageInput {
            name: [seed; 32],
            width: nx * 32,
            height: ny * 32,
            pixels: Vec::new(),
            nx,
            ny,
        })
    }

    /// The rule of `mtmd` written out cell by cell, as the design gives it.
    fn by_hand(spans: &[(u32, u32, u32)], cells: u32) -> Vec<[u32; 3]> {
        let mut out = Vec::new();
        let mut p = 0;
        let mut cell = 0;
        let mut next = spans.iter().peekable();
        while cell < cells {
            match next.peek() {
                Some(&&(first, nx, ny)) if first == cell => {
                    for j in 0..nx * ny {
                        out.push([p, p + j / nx, p + j % nx]);
                    }
                    p += nx.max(ny);
                    cell += nx * ny;
                    next.next();
                }
                _ => {
                    out.push([p, p, p]);
                    p += 1;
                    cell += 1;
                }
            }
        }
        out.truncate(cells as usize);
        out
    }

    #[test]
    fn rotary_positions_follow_mtmd_through_wide_tall_and_square_images() {
        let layout = [(3, 4, 2, 1), (20, 2, 5, 2), (30, 3, 3, 3)];
        let spans: Vec<ImageSpan> = layout
            .iter()
            .map(|&(first, nx, ny, seed)| ImageSpan { first, image: image(nx, ny, seed) })
            .collect();
        let spans = Spans::new(spans);
        let expected =
            by_hand(&layout.iter().map(|&(f, nx, ny, _)| (f, nx, ny)).collect::<Vec<_>>(), 50);
        for (cell, want) in expected.iter().enumerate() {
            assert_eq!(spans.rotary(cell as u32), *want, "cell {cell}");
        }
        // Text after the three images: 8 - 4, 10 - 5 and 9 - 3 cells less.
        assert_eq!(spans.shift(45), 4 + 5 + 6);
        assert_eq!(spans.rotary(45), [45 - 15; 3]);
        assert_eq!(spans.at(4), Some((0, 1)));
        assert_eq!(spans.at(11), None);
        assert_eq!(spans.from(11), Some(1));
        assert_eq!(spans.from(39), None);
    }

    #[test]
    fn image_cells_have_names_no_token_has() {
        let a = cell_name(&[1; 32], 0);
        assert!(a >> 63 == 1);
        assert_ne!(a, cell_name(&[1; 32], 1));
        assert_ne!(a, cell_name(&[2; 32], 0));
    }

    #[test]
    fn a_sequence_without_images_has_its_cells_as_rotary_positions() {
        let spans = Spans::default();
        assert_eq!(spans.rotary(17), [17, 17, 17]);
        assert_eq!(spans.shift(17), 0);
    }

    #[test]
    fn the_store_lets_the_least_recently_used_images_go() {
        let rows = |n: usize| Arc::new(ImageRows { values: vec![0.0; n] });
        let mut store = ImageStore::new(100);
        store.put([1; 32], rows(10));
        store.put([2; 32], rows(10));
        assert!(store.get(&[1; 32]).is_some());
        store.put([3; 32], rows(10));
        assert_eq!(store.len(), 2, "the second image, used least recently, went");
        assert!(store.contains(&[1; 32]) && store.contains(&[3; 32]));
        store.put([4; 32], rows(50));
        assert_eq!(store.len(), 1, "an image larger than the limit is kept alone");
    }
}
