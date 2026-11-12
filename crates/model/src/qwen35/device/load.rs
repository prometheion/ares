//! The weights of one device: which bytes of the model file it holds and
//! where they are in its arena.
//!
//! All weights of a device live in one allocation (the driver rounds every
//! allocation up, and a model has hundreds of tensors). A device that holds a
//! slice of a matrix gets the rows or the columns of its slice, gathered at
//! load into a matrix of the same type: the same bytes per value, never
//! decoded.
//!
//! A matrix lies in the arena in one of two layouts ([`Mat::layout`]): as the
//! file stores it, or in its device layout, the same bytes rearranged for the
//! tensor cores with the rows rounded up to 16. Which one is chosen per matrix
//! by timing both on the device (`choose.rs`). A matrix that something else
//! than a product reads keeps the file's layout: the rows of the embedding
//! table that the draft chain's gather decodes (see `embed.rs`: the rest of
//! the table lies in host memory) and a head whose rows they are. A slice is cut
//! in the file's layout and then rearranged, because the device layout mixes
//! 16 rows; for the same reason a head whose first rows the draft block reads
//! as a matrix of their own is laid out in two parts when a part takes the
//! device layout ([`HeadW`]).
//!
//! The bytes are read from the model file with ordinary reads and uploaded
//! from the loader's memory ([`MappedFiles`]); a matrix in its device layout
//! is uploaded in the file's to a place that is free for the moment and
//! rearranged by a kernel into its own.

use std::ops::Range;
use std::time::{Duration, Instant};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, EmbedRows, KeptRows, Kernels, LayerKernels, Precision,
    RecurrentShape, Stream, WeightLayout, WeightRange,
};
use ares_runtime::ModelError;

use super::choose::{Balance, Chooser, LayoutReport, Layouts, Prep, Shape, Usage};
use super::embed::TableShape;
use super::placement::{BlockSlice, DevicePlan, DraftPlan};
use crate::qwen35::Config;
use crate::qwen35::weights::{Matrix, Mixer, Weights};

/// Every tensor starts at a multiple of this in the arena: more than any
/// weight kernel asks for (16 bytes).
const ALIGN: u64 = 64;

/// A matrix on a device: `n_out` rows of `n_in` values of type `ty`, at
/// offset `at` of the arena, in `layout`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mat {
    pub ty: GgmlType,
    pub n_in: u32,
    pub n_out: u32,
    pub at: u64,
    pub layout: WeightLayout,
    /// What the size of its weights leaves of the tile kernels (a float
    /// matrix; any other takes them all).
    pub range: WeightRange,
}

impl Mat {
    /// The bytes of the matrix in the arena.
    pub fn bytes(&self) -> u64 {
        Kernels::layout_bytes(self.ty, self.layout, self.n_out, self.n_in).unwrap_or(0)
    }
}

/// A device's slice of a recurrent block.
#[derive(Clone, Debug)]
pub(crate) struct RecurrentW {
    /// The geometry of the slice: its key heads and their value heads.
    pub shape: RecurrentShape,
    pub qkv: Mat,
    pub z: Mat,
    pub beta: Mat,
    pub alpha: Mat,
    pub out: Mat,
    /// Offsets of f32 vectors: the taps of the convolution, the decay bias and
    /// factor per value head, the weight of the gated norm.
    pub taps: u64,
    pub dt_bias: u64,
    pub ssm_a: u64,
    pub ssm_norm: u64,
}

/// A device's slice of an attention block.
#[derive(Clone, Debug)]
pub(crate) struct AttentionW {
    /// Query heads and KV heads of the slice.
    pub n_head: u32,
    pub n_kv: u32,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub out: Mat,
    pub q_norm: u64,
    pub k_norm: u64,
}

#[derive(Clone, Debug)]
pub(crate) enum MixerW {
    Recurrent(RecurrentW),
    Attention(AttentionW),
    /// The device holds no head of this block: its partial result is zero.
    Empty,
}

#[derive(Clone, Debug)]
pub(crate) struct FfnW {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

/// A device's slice of a block.
#[derive(Clone, Debug)]
pub(crate) struct BlockW {
    /// The block's index in the model.
    pub layer: usize,
    pub attn_norm: u64,
    pub post_norm: u64,
    pub mixer: MixerW,
    /// `None` when the device holds no row of the feed-forward part.
    pub ffn: Option<FfnW>,
}

/// The model's draft block (its next-token-prediction block) on the device
/// that holds it, whole.
#[derive(Clone, Debug)]
pub(crate) struct DraftW {
    /// The projection of the embedding and the hidden state, as the two
    /// matrices its columns make: `x = proj_e e' + proj_h h'`.
    pub proj_e: Mat,
    pub proj_h: Mat,
    /// Offsets of the norm weights of the embedding, the hidden state and the
    /// block's output.
    pub enorm: u64,
    pub hnorm: u64,
    pub head_norm: u64,
    /// The block itself: an attention block with all its heads.
    pub block: BlockW,
    /// The draft head: the first rows of the output head on this device.
    pub head: Mat,
}

/// Where the bytes of a tensor on the device come from.
enum Source<'g> {
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

fn row_bytes(m: &Matrix<'_>) -> usize {
    m.bytes().len() / m.rows().max(1)
}

impl Source<'_> {
    fn len(&self) -> u64 {
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
    fn write(&self, arena: &DeviceBuffer, at: u64, reader: &mut Reader) -> Result<(), ModelError> {
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
pub(super) fn read_matrix(m: &Matrix<'_>, into: &mut [u8]) {
    let mut files = MappedFiles::of_process();
    let mut staging = Vec::new();
    for (piece, to) in m.bytes().chunks(PIECE).zip(into.chunks_mut(PIECE)) {
        to.copy_from_slice(files.read(piece, &mut staging));
    }
}

/// What the uploads of a load share: the files behind the model's bytes and
/// the memory they are read and gathered into.
#[derive(Default)]
struct Reader {
    files: MappedFiles,
    staging: Vec<u8>,
    gathered: Vec<u8>,
}

impl Reader {
    /// Uploads `bytes` to offset `at` of `arena`, a piece at a time.
    fn upload(&mut self, arena: &DeviceBuffer, at: u64, bytes: &[u8]) -> Result<(), ModelError> {
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
struct Ranges {
    known: Vec<(usize, WeightRange)>,
    files: Option<MappedFiles>,
    staging: Vec<u8>,
}

impl Ranges {
    fn of(&mut self, m: &Matrix<'_>) -> WeightRange {
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

/// One tensor of the arena.
struct Item<'g> {
    at: u64,
    source: Source<'g>,
    /// For a matrix in its device layout: its type, rows and columns. The
    /// source is the matrix in the file's layout.
    rearrange: Option<(GgmlType, u32, u32)>,
}

/// The arena being laid out.
struct Layout<'g, 'c> {
    items: Vec<Item<'g>>,
    len: u64,
    chooser: Chooser<'c>,
    /// What the matrices of the model's blocks are used for.
    model: Usage,
    /// How the norm in front of a half block is prepared for its products.
    normed: Prep,
    ranges: Ranges,
}

impl<'g> Layout<'g, '_> {
    fn push(&mut self, source: Source<'g>) -> u64 {
        let at = self.len.next_multiple_of(ALIGN);
        self.len = at + source.len();
        self.items.push(Item { at, source, rearrange: None });
        at
    }

    fn floats(&mut self, values: &[f32]) -> u64 {
        self.push(Source::Floats(values.to_vec()))
    }

    /// A matrix of `n_out` rows of `n_in` values of `ty` whose bytes in the
    /// file's layout are `source`, placed in `layout`.
    fn matrix(
        &mut self,
        ty: GgmlType,
        n_out: u32,
        n_in: u32,
        source: Source<'g>,
        layout: WeightLayout,
        range: WeightRange,
    ) -> Mat {
        self.chooser.placed(ty, layout, n_out, n_in);
        let at = self.len.next_multiple_of(ALIGN);
        let mat = Mat { ty, n_in, n_out, at, layout, range };
        let rearrange = (layout == WeightLayout::Device).then_some((ty, n_out, n_in));
        self.len = at + mat.bytes().max(source.len());
        self.items.push(Item { at, source, rearrange });
        mat
    }

    /// The layouts for matrices of `shapes` that read the same input; `half`
    /// is the half block of the model they belong to, if any (the mixer of
    /// block 0 is half 0).
    fn choose(
        &mut self,
        shapes: &[Shape],
        usage: Usage,
        prep: Prep,
        half: Option<u32>,
    ) -> Result<Vec<WeightLayout>, ModelError> {
        self.chooser.group(shapes, usage, prep, half)
    }

    /// A matrix that keeps the file's layout: something else than a product
    /// reads it.
    fn kept(&mut self, m: &Matrix<'g>, rows: Vec<Range<usize>>) -> Mat {
        self.rows(m, rows, WeightLayout::File)
    }

    /// Rows `rows` of `m` (range after range) as a matrix of its own.
    fn rows(&mut self, m: &Matrix<'g>, rows: Vec<Range<usize>>, layout: WeightLayout) -> Mat {
        let (ty, n_out, n_in) = rows_shape(m, &rows);
        let source = match rows.as_slice() {
            [one] => {
                let size = row_bytes(m);
                Source::Bytes(&m.bytes()[one.start * size..one.end * size])
            }
            _ => Source::Rows(*m, rows),
        };
        let range = self.ranges.of(m);
        self.matrix(ty, n_out, n_in, source, layout, range)
    }

    /// Columns `columns` of every row of `m` (range after range, whole blocks
    /// of the matrix's type: [`columns_shape`] checks it) as a matrix of its
    /// own.
    fn columns(&mut self, m: &Matrix<'g>, columns: Vec<Range<usize>>, layout: WeightLayout) -> Mat {
        let n_in = columns.iter().map(|r| r.len()).sum::<usize>() as u32;
        let source = match columns.as_slice() {
            [one] if one.start == 0 && one.end == m.cols() => Source::Bytes(m.bytes()),
            _ => Source::Columns(*m, columns),
        };
        let range = self.ranges.of(m);
        self.matrix(m.ty(), m.rows() as u32, n_in, source, layout, range)
    }
}

/// One range as a list of ranges.
fn one(range: Range<usize>) -> Vec<Range<usize>> {
    std::iter::once(range).collect()
}

/// The shape of rows `rows` of `m` as a matrix of their own.
fn rows_shape(m: &Matrix<'_>, rows: &[Range<usize>]) -> Shape {
    (m.ty(), rows.iter().map(|r| r.len()).sum::<usize>() as u32, m.cols() as u32)
}

/// The shape of columns `columns` of `m` as a matrix of their own; they must
/// be whole blocks of the matrix's type.
fn columns_shape(m: &Matrix<'_>, columns: &[Range<usize>]) -> Result<Shape, ModelError> {
    let bv = m.ty().block_values() as usize;
    if columns.iter().any(|r| !r.start.is_multiple_of(bv) || !r.end.is_multiple_of(bv)) {
        return Err(ModelError::Invalid(format!(
            "columns {columns:?} of a {} matrix are not whole blocks of {bv} values",
            m.ty()
        )));
    }
    Ok((m.ty(), m.rows() as u32, columns.iter().map(|r| r.len()).sum::<usize>() as u32))
}

/// The value-head ranges that belong to key heads `keys`, each head
/// `per_head` items wide: value head `h` reads key head `h mod n_k_heads`.
fn value_ranges(c: &Config, keys: &Range<u32>, per_head: usize) -> Vec<Range<usize>> {
    let r = &c.recurrent;
    let (k0, k1) = (keys.start as usize, keys.end as usize);
    (0..r.n_v_heads / r.n_k_heads)
        .map(|tile| (tile * r.n_k_heads + k0) * per_head..(tile * r.n_k_heads + k1) * per_head)
        .collect()
}

fn gather(values: &[f32], ranges: &[Range<usize>]) -> Vec<f32> {
    ranges.iter().flat_map(|r| values[r.clone()].iter().copied()).collect()
}

/// The rows of the output head on a device.
#[derive(Clone, Debug)]
pub(crate) struct HeadW {
    /// The rows as matrices, in order: one, or two when the draft block
    /// reads the first rows as a matrix of their own and a part lies in its
    /// device layout, of which a product cannot read the first rows alone:
    /// the draft head, then the rest.
    pub parts: Vec<Mat>,
    /// All rows.
    pub n_out: u32,
    /// The token of the first row.
    pub first_token: u32,
}

/// What a load needs to know of the model's options: how the layouts are
/// chosen and which rows the products will run with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoadOptions {
    pub layouts: Layouts,
    pub precision: Precision,
    /// The rows of the products of a decode step of the model's blocks.
    pub decode_rows: u32,
    /// The rows prompts are read in at a time.
    pub prompt_rows: u32,
    /// Whether the norm in front of each half block is written by the merge
    /// of a tensor split, in the forms the half's products read.
    pub merged: bool,
}

/// The token embedding on the device that embeds the step's tokens: the
/// table's shape, and the rows of it that lie on the device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbedW {
    pub table: TableShape,
    /// The rows the draft chain can pick; none without a draft block here.
    pub kept: Option<KeptW>,
}

/// The rows of the embedding table that lie on a device: the rows of the
/// draft head, for the tokens the chain picks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeptW {
    /// The rows, in the file's layout: the draft head's, in its order.
    pub rows: Mat,
    /// The tokens of the rows (u32 each, ascending): the offset of the list
    /// in the arena. `None`: the rows are those of the first tokens.
    pub tokens: Option<u64>,
    /// Whether the rows are the first rows of the head, which a tied model
    /// holds anyway.
    pub shared_with_head: bool,
}

impl EmbedW {
    /// Whether the host stages rows for the gather: the device keeps rows
    /// for fewer than all tokens.
    pub fn stages(&self) -> bool {
        self.kept.is_none_or(|k| k.rows.n_out < self.table.n_vocab)
    }
}

/// The tables of a device's tensors: where each is in the arena.
struct Tables {
    blocks: Vec<BlockW>,
    embedding: Option<EmbedW>,
    output_norm: u64,
    head: Option<HeadW>,
    head_tokens: Option<u64>,
    draft: Option<DraftW>,
}

/// The weights of one device.
#[derive(Debug)]
pub(crate) struct DeviceWeights {
    arena: DeviceBuffer,
    pub blocks: Vec<BlockW>,
    /// The token embedding, on the device that embeds the step's tokens.
    pub embedding: Option<EmbedW>,
    pub output_norm: u64,
    /// The rows of the output head on this device.
    pub head: Option<HeadW>,
    /// The tokens of those rows (u32 each) when a draft vocabulary reorders
    /// the head: the offset of the list in the arena.
    pub head_tokens: Option<u64>,
    /// The draft block, on the device that holds it.
    pub draft: Option<DraftW>,
    /// How long the upload took, the rearranging of the matrices in their
    /// device layout included.
    pub upload: Duration,
    /// Which matrices got which layout, and what choosing took.
    pub layouts: LayoutReport,
}

impl DeviceWeights {
    /// The device address of offset `at` of the arena.
    pub fn at(&self, at: u64) -> DevicePtr {
        self.arena.ptr().offset(at)
    }

    /// The address of a matrix.
    pub fn mat(&self, m: &Mat) -> DevicePtr {
        self.at(m.at)
    }

    /// The bytes of the arena.
    pub fn bytes(&self) -> u64 {
        self.arena.len()
    }

    /// Where the embedding gather finds its rows: the rows this device keeps
    /// and the rows staged at `staged` (null for none). `None` on a device
    /// that does not embed.
    pub fn embed_rows(&self, staged: DevicePtr) -> Option<EmbedRows> {
        let EmbedW { table, kept } = self.embedding?;
        let kept = kept.map_or(KeptRows::NONE, |k| KeptRows {
            rows: self.mat(&k.rows),
            tokens: k.tokens.map_or(DevicePtr::NULL, |at| self.at(at)),
            count: k.rows.n_out,
        });
        Some(EmbedRows { ty: table.ty, n_vocab: table.n_vocab, n_embd: table.n_embd, kept, staged })
    }

    /// Lays out and uploads what `plan` gives device `index`; `draft` is the
    /// placement's draft block, whose device holds it and whose order of the
    /// head's rows every device follows.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        kernels: &Kernels,
        layers: &LayerKernels,
        ledger: &mut Ledger,
        c: &Config,
        w: &Weights<'_>,
        plan: &DevicePlan,
        draft: Option<&DraftPlan>,
        index: usize,
        opts: LoadOptions,
        balance: Option<&Balance>,
    ) -> Result<Self, ModelError> {
        let chooser =
            Chooser::new(device, kernels, layers, opts.layouts, opts.precision, opts.prompt_rows);
        let model = Usage { decode_rows: opts.decode_rows.max(1), prompt: true };
        let normed = if opts.merged { Prep::Merge } else { Prep::Launch };
        // The tensors are laid out twice: once for the chooser to see every
        // matrix and what the weights take in all, and, when it has probed
        // and decided, once more with the layouts it chose.
        let ranges = Ranges::default();
        let mut layout = Layout { items: Vec::new(), len: 0, chooser, model, normed, ranges };
        tables(&mut layout, c, w, plan, draft, index)?;
        let Layout { len, mut chooser, ranges, .. } = layout;
        chooser.decide(len, balance.map(|b| (b, index)))?;
        let mut layout = Layout { items: Vec::new(), len: 0, chooser, model, normed, ranges };
        let Tables { blocks, embedding, output_norm, head, head_tokens, draft } =
            tables(&mut layout, c, w, plan, draft, index)?;
        let Layout { items, len, chooser, .. } = layout;
        let layouts = chooser.finish();

        let bytes = len.next_multiple_of(ALIGN).max(ALIGN);
        ledger.reserve("qwen35 weights", bytes)?;
        let arena = DeviceBuffer::new(device, bytes)?;
        let begun = Instant::now();
        // A matrix in its device layout is uploaded in the file's and
        // rearranged by a kernel, so its file form needs a place for a
        // moment: the end of the arena, which the tensors written later
        // overwrite, or for the last matrices, which reach into that end
        // themselves, a scratch buffer of their size.
        let tail_of = |item: &Item<'_>| {
            let at = bytes.checked_sub(item.source.len())? / ALIGN * ALIGN;
            let end = item.at
                + Kernels::layout_bytes(
                    item.rearrange?.0,
                    WeightLayout::Device,
                    item.rearrange?.1,
                    item.rearrange?.2,
                )?;
            (at >= end).then_some(at)
        };
        let scratch_bytes = items
            .iter()
            .filter(|item| item.rearrange.is_some() && tail_of(item).is_none())
            .map(|item| item.source.len())
            .max();
        let scratch = scratch_bytes.map(|b| DeviceBuffer::new(device, b.max(ALIGN))).transpose()?;
        let stream = Stream::new(device, 0)?;
        let mut reader = Reader { files: MappedFiles::of_process(), ..Reader::default() };
        for item in &items {
            let Some((ty, n_out, n_in)) = item.rearrange else {
                item.source.write(&arena, item.at, &mut reader)?;
                continue;
            };
            let from = match (tail_of(item), &scratch) {
                (Some(at), _) => {
                    item.source.write(&arena, at, &mut reader)?;
                    arena.ptr().offset(at)
                }
                (None, Some(scratch)) => {
                    item.source.write(scratch, 0, &mut reader)?;
                    scratch.ptr()
                }
                (None, None) => unreachable!("the scratch was sized for this matrix"),
            };
            // SAFETY: `from` holds the matrix in the file's layout (the write
            // above has reached the device) and the arena at `item.at` has
            // the bytes of its device layout, which do not overlap `from`;
            // both are aligned to `ALIGN`.
            unsafe {
                kernels.rearrange(&stream, ty, from, arena.ptr().offset(item.at), n_out, n_in)?;
            }
            // The place of the file form is written again by the next tensor.
            stream.synchronize()?;
        }
        Ok(Self {
            arena,
            blocks,
            embedding,
            output_norm,
            head,
            head_tokens,
            draft,
            upload: begun.elapsed(),
            layouts,
        })
    }
}

/// The rows of the tokens `tokens` as runs of consecutive tokens.
fn runs_of(tokens: &[u32]) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for &token in tokens {
        match runs.last_mut() {
            Some(run) if run.end == token as usize => run.end += 1,
            _ => runs.push(token as usize..token as usize + 1),
        }
    }
    runs
}

/// Lays out what `plan` gives device `index`: the rows of the embedding table
/// the draft chain picks and the head first (the largest matrices find room
/// behind them in the arena to be rearranged from), then the blocks, then
/// the draft block.
fn tables<'g>(
    layout: &mut Layout<'g, '_>,
    c: &Config,
    w: &Weights<'g>,
    plan: &DevicePlan,
    draft: Option<&DraftPlan>,
    index: usize,
) -> Result<Tables, ModelError> {
    let model = layout.model;
    let own_draft = draft.filter(|d| d.device == index);
    let order = draft.and_then(|d| d.head_order.as_deref());
    // The embedding table stays on the host (`embed.rs`); the device that
    // embeds keeps the rows of the draft head, the tokens its chain picks.
    // A model with an head of its own has them as a matrix of their own,
    // laid out first like the head. A tied model has them in its head
    // (below).
    let tied = w.output.is_none();
    if own_draft.is_some() && !plan.embedding {
        return Err(ModelError::Invalid("the draft block's device does not embed".into()));
    }
    let chain_tokens = own_draft.map(|d| match order {
        Some(order) => runs_of(&order[..d.head_rows as usize]),
        None => one(0..d.head_rows as usize),
    });
    let own_rows = match (&chain_tokens, tied) {
        (Some(rows), false) => Some(layout.kept(&w.token_embd, rows.clone())),
        _ => None,
    };
    // The head's logits are wanted for the rows of a decode step and for
    // one row of a prompt. The draft block reads the head's first rows as
    // a matrix of their own (the draft vocabulary): where those are not
    // all rows of the head here, they are laid out apart from the rest.
    let head_usage = Usage { decode_rows: model.decode_rows, prompt: false };
    let draft_rows = own_draft.map(|d| d.head_rows as usize);
    // A tied head with a draft block here is one matrix in the file's
    // layout, so that its first rows are the rows of the draft head and of
    // the embedding the chain reads.
    let in_file = tied && own_draft.is_some();
    let mut head_tokens = None;
    let head = if plan.head.is_empty() {
        None
    } else if let Some(order) = order {
        // The rows of this device's tokens in the head's order, as runs of
        // consecutive tokens.
        let tokens = &order[plan.head.start as usize..plan.head.end as usize];
        head_tokens = Some(layout.push(Source::Words(tokens.to_vec())));
        let parts = match draft_rows {
            Some(first) if first > 0 && first < tokens.len() => {
                vec![runs_of(&tokens[..first]), runs_of(&tokens[first..])]
            }
            _ => vec![runs_of(tokens)],
        };
        Some(head_rows(layout, w.head(), parts, head_usage, plan.head.start, in_file)?)
    } else {
        let rows = plan.head.start as usize..plan.head.end as usize;
        let parts = match draft_rows {
            Some(first) if first > 0 && first < rows.len() => {
                vec![one(rows.start..rows.start + first), one(rows.start + first..rows.end)]
            }
            _ => vec![one(rows)],
        };
        Some(head_rows(layout, w.head(), parts, head_usage, plan.head.start, in_file)?)
    };
    let kept = match (own_draft, &head) {
        (Some(d), Some(head)) => Some(match own_rows {
            Some(rows) => KeptW { rows, tokens: head_tokens, shared_with_head: false },
            None => KeptW {
                rows: Mat { n_out: d.head_rows, ..head.parts[0] },
                tokens: head_tokens,
                shared_with_head: true,
            },
        }),
        (Some(_), None) => {
            return Err(ModelError::Invalid(
                "the draft block's device holds no row of the output head".into(),
            ));
        }
        (None, _) => None,
    };
    let embedding = plan.embedding.then(|| EmbedW { table: TableShape::of(&w.token_embd), kept });
    let mut blocks = Vec::new();
    for (layer, slice) in plan.blocks.iter().enumerate() {
        let Some(slice) = slice else { continue };
        blocks.push(block(layout, c, w, layer, slice)?);
    }
    let output_norm = layout.floats(&w.output_norm);
    let draft = match (own_draft, &head) {
        (Some(plan), Some(head)) => Some(draft_block(layout, c, w, plan, head)?),
        _ => None,
    };
    Ok(Tables { blocks, embedding, output_norm, head, head_tokens, draft })
}

/// Lays out the rows of the output head `m` that a device holds ([`head_parts`]),
/// or, with `in_file`, as one matrix in the file's layout whatever is faster.
fn head_rows<'g>(
    layout: &mut Layout<'g, '_>,
    m: &Matrix<'g>,
    parts: Vec<Vec<Range<usize>>>,
    usage: Usage,
    first_token: u32,
    in_file: bool,
) -> Result<HeadW, ModelError> {
    if !in_file {
        return head_parts(layout, m, parts, usage, first_token);
    }
    let n_out = parts.iter().flatten().map(|r| r.len()).sum::<usize>() as u32;
    let part = layout.rows(m, parts.into_iter().flatten().collect(), WeightLayout::File);
    Ok(HeadW { parts: vec![part], n_out, first_token })
}

/// Lays out the rows of the output head `m` that a device holds, given as the
/// row ranges of one part or of two (the draft head, then the rest), each in
/// the layout that is faster for its product; the parts read one input, which
/// a launch of its own prepares. Two parts that both keep the file's layout
/// are laid out as one matrix, of which the first rows can be read alone.
fn head_parts<'g>(
    layout: &mut Layout<'g, '_>,
    m: &Matrix<'g>,
    parts: Vec<Vec<Range<usize>>>,
    usage: Usage,
    first_token: u32,
) -> Result<HeadW, ModelError> {
    let shapes: Vec<Shape> = parts.iter().map(|rows| rows_shape(m, rows)).collect();
    let n_out = shapes.iter().map(|s| s.1).sum();
    let chosen = layout.choose(&shapes, usage, Prep::Launch, None)?;
    let parts = if chosen.iter().all(|l| *l == WeightLayout::File) {
        vec![layout.rows(m, parts.into_iter().flatten().collect(), WeightLayout::File)]
    } else {
        parts.into_iter().zip(chosen).map(|(rows, l)| layout.rows(m, rows, l)).collect()
    };
    Ok(HeadW { parts, n_out, first_token })
}

/// Lays out the draft block: the model's next-token-prediction block with all
/// its heads, and the draft head as the first rows of `head`.
fn draft_block<'g>(
    layout: &mut Layout<'g, '_>,
    c: &Config,
    w: &Weights<'g>,
    plan: &DraftPlan,
    head: &HeadW,
) -> Result<DraftW, ModelError> {
    let n = w.nextn.as_ref().ok_or_else(|| {
        ModelError::Invalid("the model has no next-token-prediction block".into())
    })?;
    if n.embed_tokens.is_some() || n.shared_head_head.is_some() {
        return Err(ModelError::Invalid(
            "a next-token-prediction block with an embedding table or a head of its own is not \
             implemented"
                .into(),
        ));
    }
    if plan.head_rows == 0 || plan.head_rows > head.n_out {
        return Err(ModelError::Invalid(format!(
            "a draft head of {} rows; the device holds {} rows of the head",
            plan.head_rows, head.n_out
        )));
    }
    // The draft head: the first part of a head in two parts, or the first
    // rows of a head in one, which the file's layout lets a product read
    // alone (a head in its device layout is in one part only when the draft
    // head is all of it).
    let first = head.parts[0];
    let draft_head = match (head.parts.len(), first.layout) {
        (2, _) | (_, WeightLayout::File) => Mat { n_out: plan.head_rows, ..first },
        _ if plan.head_rows == first.n_out => first,
        _ => {
            return Err(ModelError::Invalid(
                "the draft head is a part of a head that lies in its device layout".into(),
            ));
        }
    };
    if head.parts.len() == 2 && plan.head_rows != first.n_out {
        return Err(ModelError::Invalid("the head was not cut at the draft head".into()));
    }
    let (d, ac) = (c.n_embd, &c.attention);
    let (a, b) = (&n.attention, &n.block);
    // A link of a chain is one row per sequence that drafts. The projections
    // and the keys and values also follow the model through every row of a
    // step, a prompt's included; the rest runs in the chains only. Every
    // input but SwiGLU's result is prepared by a launch of its own.
    let follows = Usage { decode_rows: 1, prompt: true };
    let chain = Usage { decode_rows: 1, prompt: false };
    let whole = |m: &Matrix<'g>| one(0..m.rows());
    let all = |m: &Matrix<'g>| one(0..m.cols());
    // Queries, keys and values read one input; in the chains all three, when
    // the block follows the model only the keys and the values.
    let qkv = layout.choose(
        &[
            rows_shape(&a.q, &whole(&a.q)),
            rows_shape(&a.k, &whole(&a.k)),
            rows_shape(&a.v, &whole(&a.v)),
        ],
        chain,
        Prep::Launch,
        None,
    )?;
    let out = layout.choose(&[columns_shape(&a.out, &all(&a.out))?], chain, Prep::Launch, None)?;
    let attention = AttentionW {
        n_head: ac.n_head as u32,
        n_kv: ac.n_head_kv as u32,
        q: layout.rows(&a.q, whole(&a.q), qkv[0]),
        k: layout.rows(&a.k, whole(&a.k), qkv[1]),
        v: layout.rows(&a.v, whole(&a.v), qkv[2]),
        out: layout.columns(&a.out, all(&a.out), out[0]),
        q_norm: layout.floats(&a.q_norm),
        k_norm: layout.floats(&a.k_norm),
    };
    let (gate, up) = (&b.ffn_gate, &b.ffn_up);
    let inner = layout.choose(
        &[rows_shape(gate, &whole(gate)), rows_shape(up, &whole(up))],
        chain,
        Prep::Launch,
        None,
    )?;
    let down_shape = columns_shape(&b.ffn_down, &all(&b.ffn_down))?;
    let down = layout.choose(&[down_shape], chain, Prep::Swiglu, None)?;
    let ffn = FfnW {
        gate: layout.rows(gate, whole(gate), inner[0]),
        up: layout.rows(up, whole(up), inner[1]),
        down: layout.columns(&b.ffn_down, all(&b.ffn_down), down[0]),
    };
    let halves =
        [columns_shape(&n.eh_proj, &one(0..d))?, columns_shape(&n.eh_proj, &one(d..2 * d))?];
    let proj_e = layout.choose(&halves[..1], follows, Prep::Launch, None)?;
    let proj_h = layout.choose(&halves[1..], follows, Prep::Launch, None)?;
    Ok(DraftW {
        proj_e: layout.columns(&n.eh_proj, one(0..d), proj_e[0]),
        proj_h: layout.columns(&n.eh_proj, one(d..2 * d), proj_h[0]),
        enorm: layout.floats(&n.enorm),
        hnorm: layout.floats(&n.hnorm),
        head_norm: layout.floats(n.shared_head_norm.as_deref().unwrap_or(&w.output_norm)),
        block: BlockW {
            layer: c.n_layer,
            attn_norm: layout.floats(&b.attn_norm),
            post_norm: layout.floats(&b.post_attention_norm),
            mixer: MixerW::Attention(attention),
            ffn: Some(ffn),
        },
        head: draft_head,
    })
}

fn block<'g>(
    layout: &mut Layout<'g, '_>,
    c: &Config,
    w: &Weights<'g>,
    layer: usize,
    slice: &BlockSlice,
) -> Result<BlockW, ModelError> {
    let l = &w.layers[layer];
    let (usage, normed) = (layout.model, layout.normed);
    // The half blocks of this block: the devices of a tensor split meet
    // after each.
    let (mixer_half, ffn_half) = (Some(2 * layer as u32), Some(2 * layer as u32 + 1));
    let attn_norm = layout.floats(&l.block.attn_norm);
    let post_norm = layout.floats(&l.block.post_attention_norm);
    let heads = slice.heads.start as usize..slice.heads.end as usize;
    let mixer = match &l.mixer {
        _ if heads.is_empty() => MixerW::Empty,
        Mixer::Recurrent(r) => {
            let rc = &c.recurrent;
            let (dk, kd) = (rc.head_k_dim, rc.key_dim());
            let values = value_ranges(c, &slice.heads, rc.head_v_dim);
            let per_head = value_ranges(c, &slice.heads, 1);
            // q of the key heads, k of the key heads, v of their value heads.
            let mut channels = Vec::with_capacity(2 + values.len());
            channels.push(heads.start * dk..heads.end * dk);
            channels.push(kd + heads.start * dk..kd + heads.end * dk);
            channels.extend(values.iter().map(|v| 2 * kd + v.start..2 * kd + v.end));
            let k = rc.conv_kernel;
            let taps: Vec<f32> =
                channels.iter().flat_map(|r| r.start * k..r.end * k).map(|i| r.conv[i]).collect();
            let n_k = heads.len() as u32;
            // The four projections read the norm of the residual stream; the
            // output projection reads what the gated norm writes for it.
            let read = layout.choose(
                &[
                    rows_shape(&r.qkv, &channels),
                    rows_shape(&r.gate, &values),
                    rows_shape(&r.beta, &per_head),
                    rows_shape(&r.alpha, &per_head),
                ],
                usage,
                normed,
                mixer_half,
            )?;
            let gated = Prep::GatedNorm {
                heads: n_k * (rc.n_v_heads / rc.n_k_heads) as u32,
                width: dk as u32,
            };
            let out =
                layout.choose(&[columns_shape(&r.out, &values)?], usage, gated, mixer_half)?;
            MixerW::Recurrent(RecurrentW {
                shape: RecurrentShape {
                    n_k_heads: n_k,
                    n_v_heads: n_k * (rc.n_v_heads / rc.n_k_heads) as u32,
                    head_dim: dk as u32,
                    conv_kernel: k as u32,
                },
                qkv: layout.rows(&r.qkv, channels, read[0]),
                z: layout.rows(&r.gate, values.clone(), read[1]),
                beta: layout.rows(&r.beta, per_head.clone(), read[2]),
                alpha: layout.rows(&r.alpha, per_head.clone(), read[3]),
                out: layout.columns(&r.out, values, out[0]),
                taps: layout.floats(&taps),
                dt_bias: layout.floats(&gather(&r.dt_bias, &per_head)),
                ssm_a: layout.floats(&gather(&r.a, &per_head)),
                ssm_norm: layout.floats(&r.norm),
            })
        }
        Mixer::Attention(a) => {
            let ac = &c.attention;
            let (h, per_kv) = (ac.head_dim, ac.n_head / ac.n_head_kv);
            let q_heads = heads.start * per_kv..heads.end * per_kv;
            let q_rows = one(q_heads.start * 2 * h..q_heads.end * 2 * h);
            let kv_rows = one(heads.start * h..heads.end * h);
            let out_columns = one(q_heads.start * h..q_heads.end * h);
            // Queries, keys and values read the norm of the residual stream;
            // the output projection reads what attention computed, prepared
            // by a launch of its own.
            let read = layout.choose(
                &[
                    rows_shape(&a.q, &q_rows),
                    rows_shape(&a.k, &kv_rows),
                    rows_shape(&a.v, &kv_rows),
                ],
                usage,
                normed,
                mixer_half,
            )?;
            let out_shape = columns_shape(&a.out, &out_columns)?;
            let out = layout.choose(&[out_shape], usage, Prep::Launch, mixer_half)?;
            MixerW::Attention(AttentionW {
                n_head: q_heads.len() as u32,
                n_kv: heads.len() as u32,
                q: layout.rows(&a.q, q_rows, read[0]),
                k: layout.rows(&a.k, kv_rows.clone(), read[1]),
                v: layout.rows(&a.v, kv_rows, read[2]),
                out: layout.columns(&a.out, out_columns, out[0]),
                q_norm: layout.floats(&a.q_norm),
                k_norm: layout.floats(&a.k_norm),
            })
        }
    };
    let rows = slice.ffn.start as usize..slice.ffn.end as usize;
    let ffn = if rows.is_empty() {
        None
    } else {
        let (gate, up, down) = (&l.block.ffn_gate, &l.block.ffn_up, &l.block.ffn_down);
        let rows = one(rows);
        // Gate and up read the norm of the residual stream; down reads what
        // SwiGLU writes for it.
        let inner = [rows_shape(gate, &rows), rows_shape(up, &rows)];
        let read = layout.choose(&inner, usage, normed, ffn_half)?;
        let fused = layout.choose(&[columns_shape(down, &rows)?], usage, Prep::Swiglu, ffn_half)?;
        Some(FfnW {
            gate: layout.rows(gate, rows.clone(), read[0]),
            up: layout.rows(up, rows.clone(), read[1]),
            down: layout.columns(down, rows, fused[0]),
        })
    };
    Ok(BlockW { layer, attn_norm, post_norm, mixer, ffn })
}
