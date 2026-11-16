//! Where the tower's weights and activations lie, worked out from the
//! configuration and the types of the matrices alone, without a device.
//!
//! **The weights** are one image of bytes in the device's layout, cut into
//! *units* that are uploaded and used one after the other: the stem (the patch
//! matrix, its bias and the position table), each block, and the merger's two
//! matrices in parts of whole output rows. In VRAM the image lies as it is and a
//! unit is read where it lies. In page-locked RAM the image stays on the host
//! and each unit is copied, for every image, into one of two slots on the
//! device, the next unit's copy running while the current one computes; a slot
//! is as large as the largest unit, which is a block, so the merger's matrices
//! are cut into parts that fit one. The passes of the two homes launch the same
//! kernels on the same units, so their rows are the same bit for bit.
//!
//! In the image a matrix is stored as the file stores it, except that a matrix
//! whose inputs are not a multiple of 32 (`ffn_down`, 4304 inputs) gets zero
//! columns up to the next one, which the tile kernels need; and the patch
//! matrix is the sum of the file's two patch kernels, formed once in `f32`.
//!
//! **The activations** of one image lie in a scratch sized for the largest
//! image it takes. The block's work before and after its attention is row by
//! row, so it is done for [`GROUP_PATCHES`] patches at a time, which bounds the
//! scratch: what grows with the image is the residual, the attention's input
//! (query, key and value of every patch) and its output.

use std::ops::Range;

use ares_core::GgmlType;
use ares_cuda::{HeadShape, Kernels, PrefillPath};

use crate::qwen3vl::config::Config;
use crate::qwen3vl::error::{Error, Result};
use crate::qwen3vl::weights::{Matrix, Weights};

/// Every tensor starts at a multiple of this in its unit: a line of the
/// devices' caches, more than any kernel asks for.
const TENSOR_ALIGN: u64 = 256;
/// Every unit starts at a multiple of this in the image (a page of the host).
const UNIT_ALIGN: u64 = 4096;
/// Every region of the scratch starts at a multiple of this.
const SCRATCH_ALIGN: u64 = 256;
/// The rows of a merger part are a multiple of this: the tile of the
/// products' large shape.
const PART_ROWS: u32 = 128;

/// The patches whose row-by-row work of a block runs at a time. It bounds the
/// scratch (about 34 KiB a patch for the 27B's tower) and not the speed: the
/// products of 2048 rows run at the rate of the tile kernels.
pub const GROUP_PATCHES: u32 = 2048;

/// The type of every matrix of a projector: what the image's layout follows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixTypes {
    /// Per block: query-key-value, attention output, up, down.
    pub blocks: Vec<[GgmlType; 4]>,
    /// The merger's two matrices.
    pub mm0: GgmlType,
    pub mm2: GgmlType,
}

impl MatrixTypes {
    /// The types of a file's matrices.
    pub fn of(weights: &Weights<'_>) -> Self {
        Self {
            blocks: weights
                .blocks
                .iter()
                .map(|b| {
                    [b.qkv.weight.ty(), b.out.weight.ty(), b.up.weight.ty(), b.down.weight.ty()]
                })
                .collect(),
            mm0: weights.mm0.weight.ty(),
            mm2: weights.mm2.weight.ty(),
        }
    }

    /// Every matrix of the same type.
    pub fn all(config: &Config, ty: GgmlType) -> Self {
        Self { blocks: vec![[ty; 4]; config.n_layer], mm0: ty, mm2: ty }
    }
}

/// A matrix in a unit: `n_out` rows of `n_in` values of `ty` (`n_in` padded
/// with zero columns where the file's is not a multiple of 32), at `at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixAt {
    pub ty: GgmlType,
    pub n_in: u32,
    pub n_out: u32,
    pub at: u64,
}

impl MatrixAt {
    /// The bytes of one row.
    pub fn row_bytes(&self) -> u64 {
        self.ty.bytes_for(u64::from(self.n_in)).expect("checked when planned")
    }

    pub fn bytes(&self) -> u64 {
        self.row_bytes() * u64::from(self.n_out)
    }
}

/// A layer norm's weight and bias in a unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormAt {
    pub weight: u64,
    pub bias: u64,
}

/// The stem's tensors: the patch matrix (`f32`), the patch bias and the
/// learned position table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StemAt {
    pub patch: MatrixAt,
    pub bias: u64,
    pub table: u64,
}

/// A block's tensors; a product's bias follows its matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockAt {
    pub ln1: NormAt,
    pub qkv: MatrixAt,
    pub qkv_bias: u64,
    pub out: MatrixAt,
    pub out_bias: u64,
    pub ln2: NormAt,
    pub up: MatrixAt,
    pub up_bias: u64,
    pub down: MatrixAt,
    pub down_bias: u64,
}

/// A part of one of the merger's matrices: its output rows `rows`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartAt {
    pub rows: Range<u32>,
    pub matrix: MatrixAt,
    /// `mm.0`: in its first part, the last norm (`v.post_ln`), which runs
    /// before it; in its last part, the whole bias, which the GELU after the
    /// last part adds. `mm.2`: the bias of this part's rows, in every part.
    pub post_ln: Option<NormAt>,
    pub bias: Option<u64>,
}

/// What a unit holds and therefore what its work is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitKind {
    Stem(StemAt),
    Block(usize, BlockAt),
    /// A part of `mm.0`, the merger's first matrix.
    Hidden(PartAt),
    /// A part of `mm.2`, the merger's second matrix.
    Output(PartAt),
}

/// A unit: its tensors and its bytes at `at` of the image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub kind: UnitKind,
    pub at: u64,
    pub bytes: u64,
}

/// The weights' layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    units: Vec<Unit>,
    image: u64,
    slot: u64,
}

/// Places tensors one after the other in a unit.
struct Cursor(u64);

impl Cursor {
    fn take(&mut self, bytes: u64) -> u64 {
        let at = self.0.next_multiple_of(TENSOR_ALIGN);
        self.0 = at + bytes;
        at
    }

    fn vector(&mut self, values: usize) -> u64 {
        self.take(values as u64 * 4)
    }

    fn norm(&mut self, n: usize) -> NormAt {
        NormAt { weight: self.vector(n), bias: self.vector(n) }
    }

    fn matrix(&mut self, ty: GgmlType, n_in: u32, n_out: u32) -> MatrixAt {
        let mut matrix = MatrixAt { ty, n_in, n_out, at: 0 };
        matrix.at = self.take(matrix.bytes());
        matrix
    }

    fn end(&self) -> u64 {
        self.0.next_multiple_of(TENSOR_ALIGN)
    }
}

/// The inputs a matrix of `n_in` columns of `ty` has on the device: `n_in`, or
/// for a float matrix the next multiple of 32 (the tile kernels take inputs in
/// steps of 32). A quantized matrix's blocks cannot be padded.
fn device_inputs(name: &str, ty: GgmlType, n_in: usize) -> Result<u32> {
    let unsupported = |problem: String| Error::Device { what: name.to_owned(), problem };
    let n_in = u32::try_from(n_in).map_err(|_| unsupported(format!("{n_in} inputs")))?;
    if n_in.is_multiple_of(32) {
        Ok(n_in)
    } else if matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        Ok(n_in.next_multiple_of(32))
    } else {
        Err(unsupported(format!(
            "{n_in} inputs of {ty} are not a multiple of 32, and only a float matrix can be padded"
        )))
    }
}

impl Plan {
    /// The layout of a projector of `config` whose matrices have `types`.
    pub fn new(config: &Config, types: &MatrixTypes) -> Result<Self> {
        let (e, group) = (config.n_embd, 4 * config.n_embd);
        if types.blocks.len() != config.n_layer {
            return Err(Error::Inconsistent {
                what: format!(
                    "{} blocks of types for {} blocks",
                    types.blocks.len(),
                    config.n_layer
                ),
            });
        }
        let dim = |n: usize, what: &str| {
            u32::try_from(n)
                .map_err(|_| Error::Device { what: what.to_owned(), problem: format!("{n} rows") })
        };
        let mut units = Vec::new();

        let mut c = Cursor(0);
        let patch_in = device_inputs("the patch matrix", GgmlType::F32, config.patch_len())?;
        if patch_in as usize != config.patch_len() {
            return Err(Error::Device {
                what: "the patch matrix".to_owned(),
                problem: format!("{} inputs are not a multiple of 32", config.patch_len()),
            });
        }
        let stem = StemAt {
            patch: c.matrix(GgmlType::F32, patch_in, dim(e, "the patch matrix")?),
            bias: c.vector(e),
            table: c.vector(config.table_side * config.table_side * e),
        };
        units.push((UnitKind::Stem(stem), c.end()));

        for (l, [qkv, out, up, down]) in types.blocks.iter().copied().enumerate() {
            let name = |part: &str| format!("v.blk.{l}.{part}");
            let mut c = Cursor(0);
            let ln1 = c.norm(e);
            let qkv = c.matrix(qkv, device_inputs(&name("attn_qkv"), qkv, e)?, dim(3 * e, "qkv")?);
            let qkv_bias = c.vector(3 * e);
            let out = c.matrix(out, device_inputs(&name("attn_out"), out, e)?, dim(e, "out")?);
            let out_bias = c.vector(e);
            let ln2 = c.norm(e);
            let up = c.matrix(up, device_inputs(&name("ffn_up"), up, e)?, dim(config.n_ff, "up")?);
            let up_bias = c.vector(config.n_ff);
            let down = c.matrix(
                down,
                device_inputs(&name("ffn_down"), down, config.n_ff)?,
                dim(e, "down")?,
            );
            let down_bias = c.vector(e);
            let block =
                BlockAt { ln1, qkv, qkv_bias, out, out_bias, ln2, up, up_bias, down, down_bias };
            units.push((UnitKind::Block(l, block), c.end()));
        }

        // The slot holds the largest of the stem and the blocks; the merger's
        // matrices are cut into as few parts of whole tiles as fit it.
        let slot = units.iter().map(|(_, bytes)| *bytes).max().unwrap_or(0);
        let mm0_in = device_inputs("mm.0", types.mm0, group)?;
        let mm2_in = device_inputs("mm.2", types.mm2, group)?;
        let parts = |ty: GgmlType, n_in: u32, n_out: u32, extra: u64| -> Result<Vec<Range<u32>>> {
            let row = ty.bytes_for(u64::from(n_in)).expect("whole blocks");
            for count in 1..=n_out.div_ceil(PART_ROWS) {
                let rows = n_out.div_ceil(count).next_multiple_of(PART_ROWS).min(n_out);
                // Room for the part's rows, its tensors besides and the alignment.
                if u64::from(rows) * row + extra + 4 * TENSOR_ALIGN <= slot {
                    let starts = (0..n_out).step_by(rows as usize);
                    return Ok(starts.map(|r| r..(r + rows).min(n_out)).collect());
                }
            }
            Err(Error::Device {
                what: "the merger".to_owned(),
                problem: format!("a part of {PART_ROWS} rows does not fit a slot of {slot} bytes"),
            })
        };
        let vectors = 4 * (2 * e as u64 + group as u64);
        let mm0_out = dim(group, "mm.0")?;
        let mm0_parts = parts(types.mm0, mm0_in, mm0_out, vectors)?;
        for (i, rows) in mm0_parts.iter().enumerate() {
            let mut c = Cursor(0);
            let post_ln = (i == 0).then(|| c.norm(e));
            let matrix = c.matrix(types.mm0, mm0_in, rows.end - rows.start);
            let bias = (i + 1 == mm0_parts.len()).then(|| c.vector(group));
            let part = PartAt { rows: rows.clone(), matrix, post_ln, bias };
            units.push((UnitKind::Hidden(part), c.end()));
        }
        let mm2_out = dim(config.n_out, "mm.2")?;
        for rows in parts(types.mm2, mm2_in, mm2_out, 4 * u64::from(mm2_out))? {
            let mut c = Cursor(0);
            let matrix = c.matrix(types.mm2, mm2_in, rows.end - rows.start);
            let bias = Some(c.vector((rows.end - rows.start) as usize));
            units.push((UnitKind::Output(PartAt { rows, matrix, post_ln: None, bias }), c.end()));
        }

        let mut image = 0;
        let units: Vec<Unit> = units
            .into_iter()
            .map(|(kind, bytes)| {
                let at = image;
                image = (at + bytes).next_multiple_of(UNIT_ALIGN);
                Unit { kind, at, bytes }
            })
            .collect();
        let slot = units.iter().map(|u| u.bytes).max().unwrap_or(0);
        Ok(Self { units, image, slot })
    }

    /// The units in the order the pass uses them.
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// The bytes of the whole image: what VRAM holds in the VRAM home, and
    /// page-locked RAM in the RAM home.
    pub fn image_bytes(&self) -> u64 {
        self.image
    }

    /// The bytes of one slot of the RAM home: the largest unit.
    pub fn slot_bytes(&self) -> u64 {
        self.slot
    }

    /// Writes the bytes of `unit` of `weights` into `out` (`unit.bytes` long);
    /// what lies between the tensors is zero.
    pub fn write_unit(&self, weights: &Weights<'_>, unit: &Unit, out: &mut [u8]) {
        assert_eq!(out.len() as u64, unit.bytes, "a buffer of the unit's size");
        out.fill(0);
        let vector = |out: &mut [u8], at: u64, values: &[f32]| {
            let at = at as usize;
            for (dst, v) in
                out[at..at + 4 * values.len()].as_chunks_mut::<4>().0.iter_mut().zip(values)
            {
                *dst = v.to_le_bytes();
            }
        };
        let matrix = |out: &mut [u8], m: &MatrixAt, source: &Matrix<'_>, rows: Range<u32>| {
            let file_row = source.bytes().len() / source.rows();
            let row = m.row_bytes() as usize;
            let bytes =
                &source.bytes()[rows.start as usize * file_row..rows.end as usize * file_row];
            let dst = &mut out[m.at as usize..(m.at + m.bytes()) as usize];
            for (dst, src) in dst.chunks_exact_mut(row).zip(bytes.chunks_exact(file_row)) {
                dst[..file_row].copy_from_slice(src);
            }
        };
        match &unit.kind {
            UnitKind::Stem(s) => {
                let sum: Vec<f32> =
                    weights.patch[0].iter().zip(&weights.patch[1]).map(|(a, b)| a + b).collect();
                vector(out, s.patch.at, &sum);
                vector(out, s.bias, &weights.patch_bias);
                vector(out, s.table, &weights.position);
            }
            UnitKind::Block(l, b) => {
                let w = &weights.blocks[*l];
                vector(out, b.ln1.weight, &w.ln1.weight);
                vector(out, b.ln1.bias, &w.ln1.bias);
                matrix(out, &b.qkv, &w.qkv.weight, 0..b.qkv.n_out);
                vector(out, b.qkv_bias, &w.qkv.bias);
                matrix(out, &b.out, &w.out.weight, 0..b.out.n_out);
                vector(out, b.out_bias, &w.out.bias);
                vector(out, b.ln2.weight, &w.ln2.weight);
                vector(out, b.ln2.bias, &w.ln2.bias);
                matrix(out, &b.up, &w.up.weight, 0..b.up.n_out);
                vector(out, b.up_bias, &w.up.bias);
                matrix(out, &b.down, &w.down.weight, 0..b.down.n_out);
                vector(out, b.down_bias, &w.down.bias);
            }
            UnitKind::Hidden(p) => {
                if let Some(norm) = p.post_ln {
                    vector(out, norm.weight, &weights.post_ln.weight);
                    vector(out, norm.bias, &weights.post_ln.bias);
                }
                matrix(out, &p.matrix, &weights.mm0.weight, p.rows.clone());
                if let Some(bias) = p.bias {
                    vector(out, bias, &weights.mm0.bias);
                }
            }
            UnitKind::Output(p) => {
                matrix(out, &p.matrix, &weights.mm2.weight, p.rows.clone());
                let rows = p.rows.start as usize..p.rows.end as usize;
                vector(
                    out,
                    p.bias.expect("every part of mm.2 has its bias"),
                    &weights.mm2.bias[rows],
                );
            }
        }
    }
}

/// Where the activations of one image lie in a scratch for images of up to
/// `max_rows` rows (`4 * max_rows` patches), as byte ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchPlan {
    pub max_rows: u32,
    /// The residual, `f32` rows of the tower's width. After the last norm it
    /// holds the input of the merger's second matrix.
    pub x: Range<u64>,
    /// Query, key and value of every patch in the attention's form. Before the
    /// first block it holds the patch rows prepared for the patch matrix; in
    /// the merger the input of its first matrix and the output of a part.
    pub heads: Range<u64>,
    /// The attention's output, `f32` rows of the tower's width. Before the
    /// first block it holds the image's patch rows (the input of a pass); in
    /// the merger the output of its first matrix.
    pub attended: Range<u64>,
    /// A group's input of a product of the tower's width, prepared.
    pub act: Range<u64>,
    /// A group's output of query-key-value or of up, `f32`.
    pub wide: Range<u64>,
    /// A group's input of down, prepared.
    pub act_ff: Range<u64>,
    /// A group's output of a product of the tower's width, `f32`.
    pub y: Range<u64>,
    /// Where the output of a merger part starts in `heads`, from its start.
    pub part_out: u64,
    pub bytes: u64,
}

impl ScratchPlan {
    /// The scratch of a projector of `config` with `plan`'s weights for images
    /// of up to `max_rows` rows.
    pub fn new(config: &Config, plan: &Plan, max_rows: u32) -> Self {
        let f16 = |n_in: usize, rows: u64| {
            Kernels::prefill_activation_bytes(PrefillPath::F16, n_in as u32, rows as u32)
                .next_multiple_of(SCRATCH_ALIGN)
        };
        let (e, group) = (config.n_embd as u64, 4 * config.n_embd);
        let rows = u64::from(max_rows);
        let patches = 4 * rows;
        let g = patches.min(u64::from(GROUP_PATCHES));
        let ff_pad = config.n_ff.next_multiple_of(32);
        let shape = HeadShape {
            n_head: config.n_head as u32,
            head_dim: config.head_dim() as u32,
            n_patches: patches as u32,
        };
        let widest_part = plan
            .units()
            .iter()
            .filter_map(|u| match &u.kind {
                UnitKind::Hidden(p) | UnitKind::Output(p) => Some(u64::from(p.matrix.n_out)),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let part_out = f16(group, rows);
        let sizes = [
            (patches * e * 4).max(f16(group, rows)),
            shape
                .bytes()
                .max(f16(config.patch_len(), patches))
                .max(part_out + rows * widest_part * 4),
            (patches * e * 4)
                .max(patches * config.patch_len() as u64 * 4)
                .max(rows * group as u64 * 4),
            f16(config.n_embd, g),
            g * (3 * e).max(config.n_ff as u64) * 4,
            f16(ff_pad, g),
            g * e * 4,
        ];
        let mut at = 0;
        let ranges: Vec<Range<u64>> = sizes
            .iter()
            .map(|bytes| {
                let start = at;
                at = (at + bytes).next_multiple_of(SCRATCH_ALIGN);
                start..start + bytes
            })
            .collect();
        Self {
            max_rows,
            x: ranges[0].clone(),
            heads: ranges[1].clone(),
            attended: ranges[2].clone(),
            act: ranges[3].clone(),
            wide: ranges[4].clone(),
            act_ff: ranges[5].clone(),
            y: ranges[6].clone(),
            part_out,
            bytes: at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real projectors' shape; `n_out` is the 27B's.
    fn real(n_out: usize) -> Config {
        Config {
            patch: 16,
            merge: 2,
            n_embd: 1152,
            n_ff: 4304,
            n_layer: 27,
            n_head: 16,
            eps: 1e-6,
            mean: [0.5; 3],
            std: [0.5; 3],
            n_out,
            table_side: 48,
        }
    }

    fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
        a.start < b.end && b.start < a.end
    }

    /// The tensors of a unit as byte ranges.
    fn tensors(unit: &Unit) -> Vec<Range<u64>> {
        let v = |at: u64, values: u64| at..at + 4 * values;
        let m = |m: &MatrixAt| m.at..m.at + m.bytes();
        let (e, ff) = (1152, 4304);
        match &unit.kind {
            UnitKind::Stem(s) => vec![m(&s.patch), v(s.bias, e), v(s.table, 48 * 48 * e)],
            UnitKind::Block(_, b) => vec![
                v(b.ln1.weight, e),
                v(b.ln1.bias, e),
                m(&b.qkv),
                v(b.qkv_bias, 3 * e),
                m(&b.out),
                v(b.out_bias, e),
                v(b.ln2.weight, e),
                v(b.ln2.bias, e),
                m(&b.up),
                v(b.up_bias, ff),
                m(&b.down),
                v(b.down_bias, e),
            ],
            UnitKind::Hidden(p) => {
                let mut t = vec![m(&p.matrix)];
                t.extend(p.post_ln.iter().flat_map(|n| [v(n.weight, e), v(n.bias, e)]));
                t.extend(p.bias.map(|b| v(b, 4 * e)));
                t
            }
            UnitKind::Output(p) => {
                vec![m(&p.matrix), v(p.bias.unwrap(), u64::from(p.rows.end - p.rows.start))]
            }
        }
    }

    /// The RAM home of both real projectors: the units come in the order of the
    /// pass, every tensor is aligned and inside its unit and no two overlap, the
    /// slot is one block and every unit fits it, the merger's parts cover their
    /// matrices' rows in whole tiles, and the image is the file's matrices plus
    /// what the padding of `ffn_down` and the alignment add.
    #[test]
    fn the_ram_home_streams_units_no_larger_than_a_block() {
        for (n_out, mm2_parts) in [(5120, 2), (2560, 1)] {
            let config = real(n_out);
            let plan = Plan::new(&config, &MatrixTypes::all(&config, GgmlType::BF16)).unwrap();
            let units = plan.units();
            assert_eq!(units.len(), 1 + 27 + 2 + mm2_parts, "{n_out}");
            assert!(matches!(units[0].kind, UnitKind::Stem(_)));
            for (l, unit) in units[1..28].iter().enumerate() {
                assert!(matches!(unit.kind, UnitKind::Block(i, _) if i == l));
            }
            // A block: the four matrices (down padded from 4304 to 4320 inputs) and
            // the vectors, 29.1 MiB.
            let block = units[1].bytes;
            let matrices = 2 * 1152 * (3456 + 1152 + 4304) + 2 * 4320 * 1152;
            assert!(block >= matrices && block < matrices + 64 * 1024, "{block}");
            assert_eq!(plan.slot_bytes(), block, "the slot is a block");
            let mut end = 0;
            for unit in units {
                assert!(unit.bytes <= plan.slot_bytes(), "{:?} exceeds the slot", unit.kind);
                assert_eq!(unit.at % UNIT_ALIGN, 0);
                assert!(unit.at >= end, "units overlap");
                end = unit.at + unit.bytes;
                let t = tensors(unit);
                for (i, a) in t.iter().enumerate() {
                    assert_eq!(a.start % TENSOR_ALIGN, 0, "{:?}", unit.kind);
                    assert!(a.end <= unit.bytes, "{:?}: a tensor outside its unit", unit.kind);
                    assert!(t[i + 1..].iter().all(|b| !overlaps(a, b)), "{:?}", unit.kind);
                }
            }
            assert_eq!(plan.image_bytes(), end.next_multiple_of(UNIT_ALIGN));
            for (kind, rows) in [(0usize, 4608u32), (1, n_out as u32)] {
                let parts: Vec<&PartAt> = units
                    .iter()
                    .filter_map(|u| match (&u.kind, kind) {
                        (UnitKind::Hidden(p), 0) | (UnitKind::Output(p), 1) => Some(p),
                        _ => None,
                    })
                    .collect();
                let mut next = 0;
                for p in &parts {
                    assert_eq!(p.rows.start, next, "parts in order, without a gap");
                    assert_eq!(p.matrix.n_out, p.rows.end - p.rows.start);
                    assert!(p.rows.end == rows || (p.rows.end - p.rows.start) % PART_ROWS == 0);
                    next = p.rows.end;
                }
                assert_eq!(next, rows);
                if kind == 0 {
                    assert!(
                        parts[0].post_ln.is_some()
                            && parts[1..].iter().all(|p| p.post_ln.is_none())
                    );
                    let last = parts.len() - 1;
                    assert!(parts.iter().enumerate().all(|(i, p)| p.bias.is_some() == (i == last)));
                }
            }
            // The file's tensors are 888.0 and 865.5 MiB; the image drops the second
            // patch kernel (3.4 MiB) and adds the padding of down (0.6 MiB) and the
            // alignment.
            let file = if n_out == 5120 { 931_126_208u64 } else { 907_523_008 };
            let image = plan.image_bytes();
            assert!(image > file - 3_538_944 && image < file + 2 * 1024 * 1024, "{image} {file}");
            // VRAM the RAM home holds between images: two slots, under 61 MiB.
            assert!(2 * plan.slot_bytes() < 61 << 20);
        }
    }

    /// A quantized `ffn_down` of 4304 inputs cannot be laid out; a float one can,
    /// and so can a quantized matrix whose inputs are whole blocks.
    #[test]
    fn only_a_float_matrix_is_padded() {
        let config = real(2560);
        let mut types = MatrixTypes::all(&config, GgmlType::BF16);
        types.blocks[3][3] = GgmlType::Q8_0;
        let error = Plan::new(&config, &types).unwrap_err().to_string();
        assert!(error.contains("v.blk.3.ffn_down") && error.contains("4304"), "{error}");
        types.blocks[3][3] = GgmlType::F16;
        types.blocks[3][2] = GgmlType::Q8_0;
        let plan = Plan::new(&config, &types).unwrap();
        let UnitKind::Block(_, b) = &plan.units()[4].kind else { panic!() };
        assert_eq!(
            (b.up.ty, b.up.n_in, b.down.ty, b.down.n_in),
            (GgmlType::Q8_0, 1152, GgmlType::F16, 4320)
        );
    }

    /// The scratch: no two regions overlap; the merger's reuse fits the regions
    /// it takes; what grows with the image grows with it and the groups do not.
    #[test]
    fn the_scratch_grows_with_the_image_and_not_with_the_groups() {
        let config = real(5120);
        let plan = Plan::new(&config, &MatrixTypes::all(&config, GgmlType::BF16)).unwrap();
        // 512 rows are one group of 2048 patches.
        let small = ScratchPlan::new(&config, &plan, 512);
        let large = ScratchPlan::new(&config, &plan, 4096);
        for s in [&small, &large] {
            let regions = [&s.x, &s.heads, &s.attended, &s.act, &s.wide, &s.act_ff, &s.y];
            for (i, a) in regions.iter().enumerate() {
                assert_eq!(a.start % SCRATCH_ALIGN, 0);
                assert!(a.end <= s.bytes);
                assert!(regions[i + 1..].iter().all(|b| !overlaps(a, b)));
            }
            assert!(s.part_out < s.heads.end - s.heads.start);
        }
        // Per patch: the residual, query, key and value, the attention's output.
        let per_patch = 1152 * 4 + 3 * 16 * 80 * 2 + 1152 * 4;
        let growth = large.bytes - small.bytes;
        let patches = 4 * (4096 - 512);
        assert!(growth >= patches * per_patch && growth < patches * (per_patch + 64), "{growth}");
        assert_eq!(large.wide.end - large.wide.start, small.wide.end - small.wide.start);
        // 4096 rows of the 27B's tower: about 330 MiB.
        assert!(large.bytes < 340 << 20, "{}", large.bytes);
    }
}
