//! The engine's kernels, loaded on a device, with one typed launch per kernel.
//!
//! The launches are `unsafe` because the device trusts the addresses it is
//! given. The layers above hold the allocations and are the ones that can
//! promise the ranges are right; this layer turns shapes into grid sizes and
//! keeps the argument lists in one place next to the kernel names.

use ares_core::GgmlType;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Function, Module};
use crate::stream::{Event, Stream};

mod layout;

pub use layout::{
    ActivationForm, FEW_ROWS, FewPlan, LayoutKernel, LayoutRows, LayoutTable, WeightLayout,
};

/// The kernel module `build.rs` compiled from `kernels/module.cu`: every GPU
/// the toolkit builds for can load it.
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.fatbin"));

/// The module of the weight kernels that multiply unquantized activations with a quantized
/// matrix, compiled from `kernels/mmv_exact.cu`.
static EXACT_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mmv_exact.fatbin"));

/// The prefill module, compiled from `kernels/prefill.cu` for compute
/// capability 8.0 and newer (the tensor-core weight kernels).
static PREFILL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/prefill.fatbin"));

/// The second prefill module, from `kernels/prefill_more.cu`: the tensor-core weight
/// kernels of the types the first has none for, in the two f16 variants.
static PREFILL_MORE_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/prefill_more.fatbin"));

/// The most activation rows one launch of a weight kernel handles
/// (`ARES_MAX_BATCH` in `kernels/common.cuh`).
pub const MAX_BATCH: u32 = 8;

/// The blocks of a peer sum launch, each with its own flag
/// (`ARES_SUM_PEER_BLOCKS` in `kernels/sum_peer.cuh`).
pub(crate) const SUM_PEER_BLOCKS: u32 = 8;

/// Threads per block for kernels that reduce over a block.
const REDUCE_THREADS: u32 = 256;

/// The launch shape of the fast weight kernels (`ARES_MMVQ_WARPS` and
/// `ARES_MMVQ_ROWS` in `kernels/mmvq.cuh`): a thread block of 4 warps computes 8
/// output rows.
const MMVQ_THREADS: u32 = 128;
const MMVQ_ROWS_PER_BLOCK: u32 = 8;

/// The weight types that have a decode kernel (`kernels/mmvq_*.cuh`): the type,
/// the kernel's name in the module, and the alignment in bytes the kernel
/// needs of the matrix (the widest load it uses).
const FAST_TYPES: [(GgmlType, &str, u64); 28] = [
    (GgmlType::Q4_K, "ares_mmvq_q4_K", 16),
    (GgmlType::Q5_K, "ares_mmvq_q5_K", 16),
    (GgmlType::Q6_K, "ares_mmvq_q6_K", 2),
    (GgmlType::Q3_K, "ares_mmvq_q3_K", 2),
    (GgmlType::IQ4_XS, "ares_mmvq_iq4_xs", 8),
    (GgmlType::IQ4_NL, "ares_mmvq_iq4_nl", 2),
    (GgmlType::IQ3_S, "ares_mmvq_iq3_s", 2),
    (GgmlType::IQ3_XXS, "ares_mmvq_iq3_xxs", 2),
    (GgmlType::IQ2_S, "ares_mmvq_iq2_s", 2),
    (GgmlType::IQ2_XS, "ares_mmvq_iq2_xs", 2),
    (GgmlType::Q8_0, "ares_mmvq_q8_0", 2),
    (GgmlType::F32, "ares_mmvq_f32", 4),
    (GgmlType::F16, "ares_mmvq_f16", 2),
    (GgmlType::BF16, "ares_mmvq_bf16", 2),
    (GgmlType::Q4_0, "ares_mmvq_q4_0", 2),
    (GgmlType::Q4_1, "ares_mmvq_q4_1", 4),
    (GgmlType::Q5_0, "ares_mmvq_q5_0", 2),
    (GgmlType::Q5_1, "ares_mmvq_q5_1", 4),
    (GgmlType::Q2_K, "ares_mmvq_q2_K", 4),
    (GgmlType::IQ1_S, "ares_mmvq_iq1_s", 2),
    (GgmlType::IQ1_M, "ares_mmvq_iq1_m", 8),
    (GgmlType::IQ2_XXS, "ares_mmvq_iq2_xxs", 2),
    (GgmlType::TQ1_0, "ares_mmvq_tq1_0", 2),
    (GgmlType::TQ2_0, "ares_mmvq_tq2_0", 2),
    (GgmlType::MXFP4, "ares_mmvq_mxfp4", 1),
    (GgmlType::NVFP4, "ares_mmvq_nvfp4", 4),
    (GgmlType::Q1_0, "ares_mmvq_q1_0", 2),
    (GgmlType::Q2_0, "ares_mmvq_q2_0", 2),
];

/// The quantized weight types and their kernel of [`Kernels::matvec_exact`]
/// (`kernels/mmv_exact.cu`), with the alignment in bytes it needs of the matrix (its
/// decoders read bytes and the f16 scales of the blocks).
const EXACT_TYPES: [(GgmlType, &str, u64); 25] = [
    (GgmlType::Q4_0, "ares_mmv_exact_q4_0", 2),
    (GgmlType::Q4_1, "ares_mmv_exact_q4_1", 2),
    (GgmlType::Q5_0, "ares_mmv_exact_q5_0", 2),
    (GgmlType::Q5_1, "ares_mmv_exact_q5_1", 2),
    (GgmlType::Q8_0, "ares_mmv_exact_q8_0", 2),
    (GgmlType::Q2_K, "ares_mmv_exact_q2_K", 2),
    (GgmlType::Q3_K, "ares_mmv_exact_q3_K", 2),
    (GgmlType::Q4_K, "ares_mmv_exact_q4_K", 2),
    (GgmlType::Q5_K, "ares_mmv_exact_q5_K", 2),
    (GgmlType::Q6_K, "ares_mmv_exact_q6_K", 2),
    (GgmlType::IQ2_XXS, "ares_mmv_exact_iq2_xxs", 2),
    (GgmlType::IQ2_XS, "ares_mmv_exact_iq2_xs", 2),
    (GgmlType::IQ3_XXS, "ares_mmv_exact_iq3_xxs", 2),
    (GgmlType::IQ1_S, "ares_mmv_exact_iq1_s", 2),
    (GgmlType::IQ4_NL, "ares_mmv_exact_iq4_nl", 2),
    (GgmlType::IQ3_S, "ares_mmv_exact_iq3_s", 2),
    (GgmlType::IQ2_S, "ares_mmv_exact_iq2_s", 2),
    (GgmlType::IQ4_XS, "ares_mmv_exact_iq4_xs", 2),
    (GgmlType::IQ1_M, "ares_mmv_exact_iq1_m", 2),
    (GgmlType::TQ1_0, "ares_mmv_exact_tq1_0", 2),
    (GgmlType::TQ2_0, "ares_mmv_exact_tq2_0", 2),
    (GgmlType::MXFP4, "ares_mmv_exact_mxfp4", 1),
    (GgmlType::NVFP4, "ares_mmv_exact_nvfp4", 2),
    (GgmlType::Q1_0, "ares_mmv_exact_q1_0", 2),
    (GgmlType::Q2_0, "ares_mmv_exact_q2_0", 2),
];

/// The decode types whose codebook is large (16 KiB: IQ1_S and IQ1_M). A thread
/// block copies it once and then walks several tiles of rows, so their launches
/// have fewer blocks than the matrix has tiles ([`DECODE_TABLE_BLOCKS_PER_SM`] per
/// multiprocessor). The other codebooks (1 to 8 KiB) are cheap enough to copy
/// for every tile, and a block per tile keeps more warps resident: persistent
/// launches of those lost 10% of the speed of IQ2_XS and IQ3_S.
const DECODE_TABLE_TYPES: [GgmlType; 2] = [GgmlType::IQ1_S, GgmlType::IQ1_M];
const DECODE_TABLE_BLOCKS_PER_SM: u32 = 12;

/// The weight types that have a prefill kernel (`kernels/mm_types.cuh`): the type,
/// the kernels' name suffix (`ares_mm_{i8,i8m,i8s,f16,f16m,f16s}_<suffix>`), the
/// alignment in bytes it needs of the matrix (the same as the decode kernel's) and
/// the bytes of shared memory its codebook takes (`SMEM_BYTES` of the type).
const PREFILL_TYPES: [(GgmlType, &str, u64, u32); 11] = [
    (GgmlType::Q4_K, "q4_K", 16, 0),
    (GgmlType::Q5_K, "q5_K", 16, 0),
    (GgmlType::Q6_K, "q6_K", 2, 0),
    (GgmlType::Q3_K, "q3_K", 2, 0),
    (GgmlType::IQ4_XS, "iq4_xs", 8, 0),
    (GgmlType::IQ4_NL, "iq4_nl", 2, 0),
    (GgmlType::IQ3_S, "iq3_s", 2, 512 * 4),
    (GgmlType::IQ3_XXS, "iq3_xxs", 2, 256 * 4),
    (GgmlType::IQ2_S, "iq2_s", 2, 1024 * 8),
    (GgmlType::IQ2_XS, "iq2_xs", 2, 512 * 8),
    (GgmlType::Q8_0, "q8_0", 2, 0),
];

/// The weight types of the second prefill module (`kernels/mm_types_more.cuh`), laid out as
/// [`PREFILL_TYPES`]. They have the f16 and the f16 flush variants and no int8 one
/// ([`Kernels::prefill_path_for`] takes the first on a card that runs it at full rate and
/// the second on a capped card).
const PREFILL_MORE_TYPES: [(GgmlType, &str, u64, u32); 17] = [
    (GgmlType::IQ2_XXS, "iq2_xxs", 2, 2048 + 4096),
    (GgmlType::IQ1_S, "iq1_s", 2, 16384),
    (GgmlType::IQ1_M, "iq1_m", 8, 16384),
    (GgmlType::Q2_0, "q2_0", 2, 0),
    (GgmlType::Q2_K, "q2_K", 4, 0),
    (GgmlType::Q4_0, "q4_0", 2, 0),
    (GgmlType::Q4_1, "q4_1", 4, 0),
    (GgmlType::Q5_0, "q5_0", 2, 0),
    (GgmlType::Q5_1, "q5_1", 4, 0),
    (GgmlType::TQ1_0, "tq1_0", 2, 0),
    (GgmlType::TQ2_0, "tq2_0", 2, 0),
    (GgmlType::MXFP4, "mxfp4", 1, 0),
    (GgmlType::NVFP4, "nvfp4", 4, 0),
    (GgmlType::Q1_0, "q1_0", 2, 0),
    (GgmlType::F32, "f32", 16, 0),
    (GgmlType::F16, "f16", 16, 0),
    (GgmlType::BF16, "bf16", 16, 0),
];

/// One shape of the prefill kernels (`kernels/prefill.cu`): the weight rows and
/// activation rows a thread block computes, and its threads.
#[derive(Clone, Copy, Debug)]
struct Shape {
    weight_rows: u32,
    batch_rows: u32,
    threads: u32,
}

impl Shape {
    /// The dynamic shared memory of a block (`AresMmLayout` in `kernels/mm.cuh`): two
    /// stages of a weight tile and an activation tile of 64 values per row, each row
    /// 16 bytes longer than its data so that `ldmatrix` finds no bank conflicts, with
    /// the (d, s) scales of both tiles in the int8 variant, then the type's table.
    /// `elem` is the bytes of a value in the tile: 1 for int8, 2 for f16.
    const fn shared_bytes(self, elem: u32, table: u32) -> u32 {
        let rows = self.weight_rows + self.batch_rows;
        let tiles = rows * (64 * elem + 16);
        let scales = if elem == 1 { 2 * rows * 8 } else { 0 };
        2 * (tiles + scales) + table
    }
}

/// The shapes, `[int8, f16, f16 with f16 accumulation][large, medium, small]`, and the numbers of activation
/// rows they take: up to [`MM_SMALL_ROWS`] the small one, up to [`MM_MEDIUM_ROWS`]
/// the medium one, more the large one. The large shape is a block of 16 warps with
/// the whole SM to itself, which is what lets the decoding of a weight tile be shared
/// by 128 activation rows; the others are blocks of 4 warps, three (int8) or two (f16)
/// to an SM, for the rows where there are not enough tiles to fill the large ones.
const MM_SMALL_ROWS: u32 = 32;
const MM_MEDIUM_ROWS: u32 = 64;
const SHAPES: [[Shape; 3]; 3] = [
    [
        Shape { weight_rows: 128, batch_rows: 128, threads: 512 },
        Shape { weight_rows: 64, batch_rows: 64, threads: 128 },
        Shape { weight_rows: 64, batch_rows: 32, threads: 128 },
    ],
    [
        Shape { weight_rows: 128, batch_rows: 128, threads: 512 },
        Shape { weight_rows: 64, batch_rows: 64, threads: 128 },
        Shape { weight_rows: 64, batch_rows: 32, threads: 128 },
    ],
    [
        Shape { weight_rows: 128, batch_rows: 128, threads: 512 },
        Shape { weight_rows: 64, batch_rows: 64, threads: 128 },
        Shape { weight_rows: 64, batch_rows: 32, threads: 128 },
    ],
];

/// The kernels' name part for each shape.
const SHAPE_SUFFIX: [&str; 3] = ["", "m", "s"];
/// What a launch gets without asking for more shared memory.
const DEFAULT_SHARED: u32 = 48 << 10;

/// For each weight type, up to how many activation rows the decode kernels (in
/// launches of [`MAX_BATCH`] rows) are the faster family; above it the tensor-core
/// kernels are. See [`Kernels::decode_kernel_rows`].
const DECODE_ROWS: [(GgmlType, u32); 16] = [
    (GgmlType::Q4_K, 16),
    (GgmlType::Q5_K, 16),
    (GgmlType::Q6_K, 8),
    (GgmlType::Q3_K, 8),
    (GgmlType::IQ4_XS, 12),
    (GgmlType::IQ4_NL, 12),
    (GgmlType::IQ3_S, 16),
    (GgmlType::IQ3_XXS, 16),
    (GgmlType::IQ2_S, 12),
    (GgmlType::IQ2_XS, 12),
    (GgmlType::Q8_0, 8),
    (GgmlType::IQ2_XXS, 16),
    (GgmlType::IQ1_S, 12),
    (GgmlType::IQ1_M, 12),
    (GgmlType::Q2_0, 16),
    (GgmlType::Q4_0, 16),
];

/// The types for which [`PrefillPath::F16Flush`] is as fast as int8 on a capped card
/// ([`Kernels::prefill_path_for`]).
const FLUSH_AS_FAST: [GgmlType; 3] = [GgmlType::Q4_K, GgmlType::IQ2_S, GgmlType::IQ2_XS];

/// The activation bytes the blocks that run together should be able to keep in
/// L2: the grid is ordered in groups of activation tiles that take every weight
/// tile in turn (see `kernels/mm.cuh`), and a group is as many tiles as fit in this
/// many bytes, at least 2 (a group of 1 costs the RTX 4070 10% on the 128-row
/// shape) and at most 8. The RTX 3060, whose L2 is 3 MiB, does not care.
const MM_GROUP_BYTES: u64 = 4 << 20;
const MM_GROUP_MIN: u64 = 2;
const MM_GROUP_MAX: u64 = 8;

/// The smallest L2 cache, in bytes, for which the small and medium shapes ask L2 to
/// prefetch the next group of four steps of every row they read (see
/// `kernels/mm.cuh`). On an RTX 4070 (36 MiB) it takes the small shape from 0.34 to
/// 0.19 ms at 32 rows; on an RTX 3060 (3 MiB) it costs about 10%.
const MM_PREFETCH_L2_BYTES: u32 = 16 << 20;

/// How the prefill family reads its activations and which of its two kernels it
/// runs. See [`Kernels::prefill_path`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPath {
    /// Integer `mma` over the weights' integer codes and activations quantized
    /// to int8 ([`Kernels::quantize_activations`]'s format, the decode kernels'
    /// too): the arithmetic of the decode kernels on the tensor cores.
    Int8,
    /// f16 `mma` with f32 accumulation over the dequantised weights and f16
    /// activations: the direct product, which rounds nothing but the f16
    /// inputs.
    F16,
    /// The f16 path with f16 accumulation, flushed to f32 every four `mma` (64
    /// values of K): the same tiles and activations as [`PrefillPath::F16`], at
    /// the rate GeForce cards give f16 accumulation (twice that of f32
    /// accumulation). 0.04 % rms from the exact product on the test data, against
    /// 1.1 % for [`PrefillPath::Int8`]. Partial sums are at most 64 products of a
    /// weight and a value of magnitude below 1 (the activations are divided per
    /// row by a power of two, see [`Kernels::prepare_prefill_activations`]), so
    /// they cannot overflow for weights up to [`WeightRange::F16_SUMS_MAX`], and
    /// they are rounded to f16 after every 8 products: a partial sum below
    /// 1.2e-4 of the row's factor to a multiple of 2^-24 of it, which an output
    /// of very small weights feels (`docs/ARCHITECTURE.md`, "Float matrices").
    F16Flush,
}

/// What the size of a float matrix's weights (F32, F16, BF16) leaves of the tile kernels
/// ([`Kernels::matmul`]), which hold these weights as f16 without a scale. The decode kernels
/// ([`Kernels::matvec`]) multiply the weights as stored, in f32, whatever their size.
///
/// A loader asks once per matrix ([`WeightRange::of`], a pass over its bytes) and takes the
/// path [`WeightRange::path`] leaves, so that a product is never infinite where the exact one
/// is finite. A matrix of another type is always [`WeightRange::Any`]: its scales are f16
/// in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightRange {
    /// Every path: no weight is larger than [`WeightRange::F16_SUMS_MAX`].
    #[default]
    Any,
    /// Not [`PrefillPath::F16Flush`]: a weight is larger than [`WeightRange::F16_SUMS_MAX`],
    /// so an f16 sum of 64 products could overflow. [`PrefillPath::F16`] sums in f32.
    F32Sums,
    /// No tile kernel: a weight is larger than the largest f16 (65504), infinite or not a
    /// number, and would be an infinity in the tile.
    Beyond,
}

impl WeightRange {
    /// The largest weight the path with f16 accumulation takes: 2^9. Its partial sums are 64
    /// products of a weight and an activation of at most 1 (the rows are divided by a power
    /// of two above their largest value), so they stay below 2^15, half the range of f16,
    /// and the roundings on the way (at most a spacing of f16 each) cannot reach the rest.
    pub const F16_SUMS_MAX: f32 = 512.0;

    /// The range of a matrix of `ty` stored in `bytes`: the largest magnitude decides.
    /// Bytes beyond the last whole value are ignored.
    pub fn of(ty: GgmlType, bytes: &[u8]) -> Self {
        // The bits of a float without its sign order as its magnitude does, and those of an
        // infinity or a NaN are above every finite number's.
        let largest = match ty {
            GgmlType::F32 => {
                let bits = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b));
                f32::from_bits(bits.fold(0, |m, b| m.max(b & 0x7FFF_FFFF)))
            }
            GgmlType::F16 => {
                let bits = bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b));
                match bits.fold(0, |m, b| m.max(b & 0x7FFF)) {
                    0x7C00.. => f32::INFINITY,
                    // A subnormal: far below any limit.
                    bits @ ..0x0400 => f32::from(bits) * 2f32.powi(-24),
                    bits => {
                        f32::from(0x0400 | (bits & 0x03FF)) * 2f32.powi(i32::from(bits >> 10) - 25)
                    }
                }
            }
            GgmlType::BF16 => {
                let bits = bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b));
                f32::from_bits(u32::from(bits.fold(0, |m, b| m.max(b & 0x7FFF))) << 16)
            }
            _ => return Self::Any,
        };
        Self::of_largest(largest)
    }

    /// The range of a float matrix whose largest magnitude is `largest`.
    pub fn of_largest(largest: f32) -> Self {
        if largest <= Self::F16_SUMS_MAX {
            Self::Any
        } else if largest <= 65504.0 {
            Self::F32Sums
        } else {
            Self::Beyond
        }
    }

    /// The more restrictive of two ranges: that of a matrix made of both.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Beyond, _) | (_, Self::Beyond) => Self::Beyond,
            (Self::F32Sums, _) | (_, Self::F32Sums) => Self::F32Sums,
            _ => Self::Any,
        }
    }

    /// The path a matrix of this range takes where one of [`WeightRange::Any`] takes `path`:
    /// `None` is the decode kernels.
    pub fn path(self, path: Option<PrefillPath>) -> Option<PrefillPath> {
        match (self, path) {
            (Self::Beyond, _) => None,
            (Self::F32Sums, Some(PrefillPath::F16Flush)) => Some(PrefillPath::F16),
            _ => path,
        }
    }
}

/// How one shape of the prefill kernels runs on a device
/// ([`Kernels::prefill_shapes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillShape {
    /// The weight rows and the activation rows one thread block computes.
    pub weight_rows: u32,
    pub batch_rows: u32,
    pub threads: u32,
    /// Dynamic shared memory per block, in bytes.
    pub shared_bytes: u32,
    /// Registers per thread, as the compiler allocated them.
    pub registers: u32,
    /// Blocks of this kernel one multiprocessor holds at once (registers and shared
    /// memory permitting).
    pub blocks_per_sm: u32,
    /// The most activation rows this shape is used for (`u32::MAX` for the last).
    pub up_to_rows: u32,
}

/// What a caller of the prefill family asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// The path this device takes by default ([`Kernels::prefill_path`]): f16 where
    /// the tensor cores run it at full rate, since the direct product is then
    /// both the more exact and, in principle, as fast, and on GeForce cards with
    /// the cap the fastest path per type ([`Kernels::prefill_path_for`]): int8, or
    /// f16 with f16 accumulation where that is as fast.
    ///
    /// The rate of an uncapped card could not be measured on the machine the
    /// kernels were written on (both its GPUs are capped).
    #[default]
    Fast,
    /// Always the f16 path with f32 accumulation. More exact than int8, and on a
    /// GeForce card whose tensor cores are capped (the case until the cap is
    /// lifted) slower.
    Exact,
}

/// The kernels of one type: the alignment, the bytes of its table and the kernels indexed
/// `[path][shape]`; a type has none for a path it has no variant of.
type PrefillType = (GgmlType, u64, u32, [Option<[Function; 3]>; 3]);

/// The kernels of the prefill modules on one device.
#[derive(Debug, Clone)]
struct Prefill {
    /// Per type: the alignment, the bytes of its table and the kernels, indexed
    /// `[path][shape]`.
    types: Vec<PrefillType>,
    /// The kernels over the device layouts and their tables.
    layouts: layout::Layouts,
    convert_f16: Function,
    /// Whether `mma` with f32 accumulation runs as fast as with f16
    /// accumulation (measured at load).
    full_rate: bool,
}

/// The kernels of the engine on one device.
#[derive(Debug, Clone)]
pub struct Kernels {
    device: Device,
    add_f32: Function,
    add_to_f32: Function,
    sum_peer_f32: Function,
    rms_norm_f32: Function,
    sum32_f32: Function,
    matvec_q4_k: Function,
    quantize_act: Function,
    fast: Vec<(GgmlType, u64, Function)>,
    /// The kernels of [`Kernels::matvec_exact`] for the quantized types.
    exact: Vec<(GgmlType, u64, Function)>,
    /// `None` on a GPU below compute capability 8.0.
    prefill: Option<Prefill>,
}

impl Prefill {
    fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, PREFILL_IMAGE)?;
        let more = Module::load(device, PREFILL_MORE_IMAGE)?;
        let limit = device.info().max_shared_per_block_optin;
        let mut types = Self::load_types(&module, &PREFILL_TYPES, true, limit)?;
        types.extend(Self::load_types(&more, &PREFILL_MORE_TYPES, false, limit)?);
        Ok(Self {
            types,
            layouts: layout::Layouts::load(&module, limit)?,
            convert_f16: module.function("ares_convert_f16")?,
            full_rate: probe_full_rate(device, &module)?,
        })
    }

    /// The kernels of `list` in `module`, with the int8 variant or without it.
    fn load_types(
        module: &Module,
        list: &[(GgmlType, &str, u64, u32)],
        int8: bool,
        limit: u32,
    ) -> Result<Vec<PrefillType>> {
        list.iter()
            .map(|(ty, suffix, align, table)| {
                let mut paths = Vec::new();
                for (path, (variant, elem)) in
                    [("i8", 1u32), ("f16", 2u32), ("h", 2u32)].into_iter().enumerate()
                {
                    if path == 0 && !int8 {
                        paths.push(None);
                        continue;
                    }
                    let mut shapes = Vec::new();
                    for (shape, part) in SHAPES[path].iter().zip(SHAPE_SUFFIX) {
                        let function =
                            module.function(&format!("ares_mm_{variant}{part}_{suffix}"))?;
                        let bytes = shape.shared_bytes(elem, *table);
                        if bytes > limit {
                            return Err(Error::Invalid {
                                what: "prefill kernel",
                                detail: format!(
                                    "{ty} needs {bytes} bytes of shared memory per block, the device allows {limit}"
                                ),
                            });
                        }
                        if bytes > DEFAULT_SHARED {
                            function.allow_dynamic_shared(bytes)?;
                        }
                        shapes.push(function);
                    }
                    paths.push(Some(<[Function; 3]>::try_from(shapes).expect("three shapes")));
                }
                let kernels =
                    <[Option<[Function; 3]>; 3]>::try_from(paths).expect("three paths");
                Ok((*ty, *align, *table, kernels))
            })
            .collect()
    }
}

/// Whether the tensor cores of `device` run f16 `mma` with f32 accumulation as
/// fast as with f16 accumulation. NVIDIA caps the first at half the rate of the
/// second on GeForce cards (unless the cap is lifted, for example with
/// gpu-unlock); professional cards run both at the same rate. The two kernels of
/// the module (`ares_probe_mma_*`) issue the same number of `mma` and are timed
/// alternately a few times; the fastest run of each is compared, which keeps the
/// clocks of a card that is still waking up out of the result.
fn probe_full_rate(device: &Device, module: &Module) -> Result<bool> {
    /// Rounds of 8 `mma` per warp: about a millisecond on the cards measured.
    const ITERS: u32 = 1024;
    /// Warps per SM: enough to keep the tensor cores busy.
    const BLOCKS_PER_SM: u32 = 4;
    const THREADS: u32 = 256;
    /// A capped card measures about 2.0, an uncapped one 1.0.
    const CAPPED_ABOVE: f32 = 1.5;
    const ROUNDS: usize = 4;

    let f32_acc = module.function("ares_probe_mma_f32")?;
    let f16_acc = module.function("ares_probe_mma_f16")?;
    let stream = Stream::new(device, 0)?;
    let blocks = device.info().multiprocessors * BLOCKS_PER_SM;
    let out = DeviceBuffer::new(device, u64::from(blocks) * u64::from(THREADS) * 4)?;
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);
    let run = |function: &Function| -> Result<f32> {
        let args = Args::new().ptr(out.ptr()).u32(ITERS);
        start.record(&stream)?;
        // SAFETY: `out` has one f32 per thread of the grid, as the probe kernels write.
        unsafe { function.launch(&stream, [blocks, 1, 1], [THREADS, 1, 1], 0, args)? };
        end.record(&stream)?;
        end.synchronize()?;
        end.elapsed_ms_since(&start)
    };
    run(&f16_acc)?;
    run(&f32_acc)?;
    let (mut best_f32, mut best_f16) = (f32::INFINITY, f32::INFINITY);
    for _ in 0..ROUNDS {
        best_f16 = best_f16.min(run(&f16_acc)?);
        best_f32 = best_f32.min(run(&f32_acc)?);
    }
    Ok(best_f32 < best_f16 * CAPPED_ABOVE)
}

impl Kernels {
    /// Loads the kernel modules on `device`. On a GPU generation without
    /// precompiled code the driver compiles the modules here, once; it caches
    /// the result on disk. On a GPU of compute capability 8.0 or newer this
    /// also measures, in a few milliseconds, whether its tensor cores run `mma`
    /// with f32 accumulation at full rate ([`Kernels::tensor_cores_full_rate`]).
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        let prefill = if device.info().compute_capability.0 >= 8 {
            Some(Prefill::load(device)?)
        } else {
            None
        };
        let fast = FAST_TYPES
            .iter()
            .map(|(ty, name, align)| Ok((*ty, *align, module.function(name)?)))
            .collect::<Result<Vec<_>>>()?;
        let exact_module = Module::load(device, EXACT_IMAGE)?;
        let exact = EXACT_TYPES
            .iter()
            .map(|(ty, name, align)| Ok((*ty, *align, exact_module.function(name)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: device.clone(),
            add_f32: module.function("ares_add_f32")?,
            add_to_f32: module.function("ares_add_to_f32")?,
            sum_peer_f32: module.function("ares_sum_peer_f32")?,
            rms_norm_f32: module.function("ares_rms_norm_f32")?,
            sum32_f32: module.function("ares_sum32_f32")?,
            matvec_q4_k: module.function("ares_matvec_q4_K")?,
            quantize_act: module.function("ares_quantize_act")?,
            fast,
            exact,
            prefill,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `y[i] += x[i]` for `n` f32 values.
    ///
    /// # Safety
    ///
    /// `y` and `x` must each address `n` f32 values on the stream's device,
    /// alive until the launch has run.
    pub unsafe fn add_f32(
        &self,
        stream: &Stream,
        y: DevicePtr,
        x: DevicePtr,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(y).ptr(x).u32(n);
        // SAFETY: the argument list matches ares_add_f32; the ranges are the
        // caller's guarantee.
        unsafe { self.add_f32.launch(stream, [n.div_ceil(256), 1, 1], [256, 1, 1], 0, args) }
    }

    /// `out[i] = a[i] + b[i]` for `n` f32 values. `out` may be `a` or `b`.
    ///
    /// # Safety
    ///
    /// `out`, `a` and `b` must each address `n` f32 values on the stream's
    /// device, alive until the launch has run.
    pub unsafe fn add_to_f32(
        &self,
        stream: &Stream,
        out: DevicePtr,
        a: DevicePtr,
        b: DevicePtr,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(out).ptr(a).ptr(b).u32(n);
        // SAFETY: the argument list matches ares_add_to_f32; the ranges are
        // the caller's guarantee.
        unsafe { self.add_to_f32.launch(stream, [n.div_ceil(256), 1, 1], [256, 1, 1], 0, args) }
    }

    /// One device's half of a sum over peer access: writes `data` into the
    /// peer's receive buffer and `token` into the peer's flags, waits for the
    /// peer's token in its own flags, and writes `data + received` to `out`.
    /// See `kernels/sum_peer.cuh`. [`Mesh`](crate::Mesh) is the caller.
    ///
    /// # Safety
    ///
    /// `data`, `out` and `own_recv` must address `n` f32 values and
    /// `own_flags` [`SUM_PEER_BLOCKS`] u32 values on the stream's device;
    /// `peer_recv` and `peer_flags` the same on a device this one has peer
    /// access to. The peer must launch its half with the same `token`, or
    /// this kernel waits until it gives up and writes NaN.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sum_peer_f32(
        &self,
        stream: &Stream,
        data: DevicePtr,
        out: DevicePtr,
        peer_recv: DevicePtr,
        peer_flags: DevicePtr,
        own_recv: DevicePtr,
        own_flags: DevicePtr,
        n: u32,
        token: u32,
    ) -> Result<()> {
        let args = Args::new()
            .ptr(data)
            .ptr(out)
            .ptr(peer_recv)
            .ptr(peer_flags)
            .ptr(own_recv)
            .ptr(own_flags)
            .u32(n)
            .u32(token);
        // SAFETY: the argument list matches ares_sum_peer_f32.
        unsafe { self.sum_peer_f32.launch(stream, [SUM_PEER_BLOCKS, 1, 1], [256, 1, 1], 0, args) }
    }

    /// RMS normalisation of `rows` rows of `n` values:
    /// `y = x / sqrt(mean(x^2) + eps) * weight`. `weight` holds `n` values or
    /// is [`DevicePtr::NULL`] for no weight. `y` may be `x`.
    ///
    /// # Safety
    ///
    /// `x` and `y` must each address `rows * n` f32 values and `weight`, when
    /// not null, `n` values, on the stream's device, alive until the launch
    /// has run.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rms_norm_f32(
        &self,
        stream: &Stream,
        x: DevicePtr,
        weight: DevicePtr,
        y: DevicePtr,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<()> {
        if n == 0 || rows == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(x).ptr(weight).ptr(y).u32(n).f32(eps);
        // SAFETY: the argument list matches ares_rms_norm_f32.
        unsafe { self.rms_norm_f32.launch(stream, [rows, 1, 1], [REDUCE_THREADS, 1, 1], 0, args) }
    }

    /// `y[b] = W x[b]` for a Q4_K matrix `w` of `n_out` rows and `n_in`
    /// columns and `n_batch` activation rows.
    ///
    /// `scratch` receives `n_batch * n_in / 32` f32 values (the sums of the
    /// activations per 32 values, which the type's offsets need).
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until both launches have run: `w`
    /// must address the matrix as GGUF stores it, 16-byte aligned; `x`
    /// `n_batch * n_in` f32 values; `scratch` `n_batch * n_in / 32` f32
    /// values; `y` `n_batch * n_out` f32 values. `y` and `scratch` must not
    /// overlap the inputs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec_q4_k(
        &self,
        stream: &Stream,
        w: DevicePtr,
        x: DevicePtr,
        scratch: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_batch: u32,
    ) -> Result<()> {
        if !n_in.is_multiple_of(256) || n_in == 0 {
            return Err(Error::Invalid {
                what: "Q4_K matrix",
                detail: format!("{n_in} columns are not a whole number of 256-value blocks"),
            });
        }
        if n_batch == 0 || n_batch > MAX_BATCH {
            return Err(Error::Invalid {
                what: "weight kernel batch",
                detail: format!("{n_batch} activation rows; one launch takes 1 to {MAX_BATCH}"),
            });
        }
        if n_out == 0 {
            return Ok(());
        }
        let n_groups = n_batch * (n_in / 32);
        let args = Args::new().ptr(x).ptr(scratch).u32(n_groups);
        // SAFETY: the argument list matches ares_sum32_f32.
        unsafe {
            self.sum32_f32.launch(stream, [n_groups.div_ceil(256), 1, 1], [256, 1, 1], 0, args)?;
        }
        // One thread per unit of 64 values, in whole warps, up to the reduce width.
        let threads = (n_in / 64).next_multiple_of(32).min(REDUCE_THREADS);
        let args = Args::new().ptr(w).ptr(x).ptr(scratch).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
        // SAFETY: the argument list matches ares_matvec_q4_K.
        unsafe { self.matvec_q4_k.launch(stream, [n_out, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// The bytes that [`Kernels::quantize_activations`] writes for `n_batch`
    /// rows of `n_in` activations: the int8 values, then (from the next
    /// multiple of 16 bytes) one `(scale, sum)` pair of f32 per group of 32
    /// values, then (again from the next multiple of 16) the activations
    /// themselves as f32, which the float weight types multiply exactly. See
    /// `kernels/quantize_act.cuh` for the format.
    pub fn quantized_activation_bytes(n_in: u32, n_batch: u32) -> u64 {
        let values = u64::from(n_in) * u64::from(n_batch);
        let pairs_end = values.next_multiple_of(16) + values.div_ceil(32) * 8;
        pairs_end.next_multiple_of(16) + values * 4
    }

    /// Whether [`Kernels::matvec`] has a kernel for weights of type `ty`.
    pub fn has_fast_matvec(ty: GgmlType) -> bool {
        FAST_TYPES.iter().any(|(t, _, _)| *t == ty)
    }

    /// Quantizes `n_batch` rows of `n_in` f32 activations to the int8 format
    /// the fast weight kernels read ([`Kernels::matvec`]). One quantization
    /// serves every matrix that multiplies the same activations, whatever
    /// their types.
    ///
    /// The quantized weight types need `n_in` to be a multiple of their block (32 or
    /// 256 values; [`Kernels::matvec`] checks it); the float types take any `n_in`.
    /// `q` receives
    /// [`Kernels::quantized_activation_bytes`] bytes.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` must
    /// address `n_batch * n_in` f32 values and `q` the bytes above, 16-byte
    /// aligned, not overlapping `x`.
    pub unsafe fn quantize_activations(
        &self,
        stream: &Stream,
        x: DevicePtr,
        q: DevicePtr,
        n_in: u32,
        n_batch: u32,
    ) -> Result<()> {
        if n_in == 0 {
            return Err(Error::Invalid {
                what: "activation quantization",
                detail: "rows of 0 values".into(),
            });
        }
        check_batch(n_batch)?;
        // SAFETY: the caller's guarantee is the one of the function it forwards to.
        unsafe { self.quantize_rows(stream, x, q, n_in, n_batch) }
    }

    /// The quantization of [`Kernels::quantize_activations`] for any number of
    /// rows (`n_in * n_batch` values must fit in 32 bits).
    unsafe fn quantize_rows(
        &self,
        stream: &Stream,
        x: DevicePtr,
        q: DevicePtr,
        n_in: u32,
        n_batch: u32,
    ) -> Result<()> {
        check_alignment("quantized activations", q, 16)?;
        let n_values = n_in.checked_mul(n_batch).ok_or(Error::Invalid {
            what: "activation quantization",
            detail: "size overflows u32".into(),
        })?;
        let args = Args::new().ptr(x).ptr(q).u32(n_values);
        // SAFETY: the argument list matches ares_quantize_act; the ranges are the
        // caller's guarantee.
        unsafe {
            self.quantize_act.launch(stream, [n_values.div_ceil(256), 1, 1], [256, 1, 1], 0, args)
        }
    }

    /// `y[b] = W x[b]` for a matrix `w` of type `ty` with `n_out` rows and
    /// `n_in` columns and `n_batch` (1 to [`MAX_BATCH`]) rows of activations
    /// that [`Kernels::quantize_activations`] has quantized into `q`. The
    /// weights are read once however many rows there are.
    ///
    /// This is the decode kernel, for every weight type a GGUF file can carry
    /// except the integer and double types and the activation formats Q8_1 and
    /// Q8_K. The quantized types compute with integer dot products over the
    /// quantized activations, so their results differ from the exact product by
    /// about 1% rms (the rounding of the activations; the weights are exact);
    /// the float types (F32, F16, BF16) multiply the activations as they are, and
    /// differ from it by f32 rounding only. The exact f32 kernel
    /// ([`Kernels::matvec_q4_k`] for Q4_K) stays the reference for Q4_K. A type
    /// without a fast kernel (see [`Kernels::has_fast_matvec`]) is refused with
    /// an error.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `w` must
    /// address the matrix as GGUF stores it (`n_out` rows of `n_in` values in
    /// whole blocks of `ty`), aligned to 16 bytes for Q4_K and Q5_K, to 8 for
    /// IQ4_XS and IQ1_M, to 4 for F32, Q4_1, Q5_1, Q2_K and NVFP4, to 1 for MXFP4
    /// and to 2 for the other types (checked); `q` the quantized activations of
    /// this `n_in` and `n_batch`, 16-byte aligned (checked); `y`
    /// `n_batch * n_out` f32 values. `y` must not overlap the inputs. A matrix of
    /// MXFP4 (17-byte blocks) is read in aligned words, so the word that holds its
    /// last byte may reach up to 3 bytes past the matrix.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec(
        &self,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        q: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_batch: u32,
    ) -> Result<()> {
        let Some((_, align, function)) = self.fast.iter().find(|(t, _, _)| *t == ty) else {
            return Err(Error::Invalid {
                what: "weight kernel type",
                detail: format!(
                    "{ty} has no fast weight kernel (it has one for {})",
                    FAST_TYPES.map(|(t, _, _)| t.name()).join(", ")
                ),
            });
        };
        let block_values = ty.block_values();
        if n_in == 0 || !n_in.is_multiple_of(block_values) {
            return Err(Error::Invalid {
                what: "weight matrix",
                detail: format!(
                    "{n_in} columns are not a whole number of {ty} blocks of {block_values} values"
                ),
            });
        }
        check_batch(n_batch)?;
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", w, *align)?;
        check_alignment("quantized activations", q, 16)?;
        let tiles = n_out.div_ceil(MMVQ_ROWS_PER_BLOCK);
        let blocks = if DECODE_TABLE_TYPES.contains(&ty) {
            tiles.min(DECODE_TABLE_BLOCKS_PER_SM * self.device.info().multiprocessors)
        } else {
            tiles
        };
        let args = Args::new().ptr(w).ptr(q).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
        // SAFETY: the argument list matches the ares_mmvq_* kernels; the ranges are the
        // caller's guarantee.
        unsafe { function.launch(stream, [blocks, 1, 1], [MMVQ_THREADS, 1, 1], 0, args) }
    }

    /// `y[b] = W x[b]` as [`Kernels::matvec`], with the activations as they are: the f32
    /// section of the prepared rows at `q` (the layout of
    /// [`Kernels::quantize_activations`]; its int8 codes are not read and need not be
    /// written). The float types take their kernel of [`Kernels::matvec`], which reads the
    /// same section; a quantized type takes a kernel that decodes each weight as ggml does
    /// and sums in f32 in the order of the float kernel, so its product is that of its
    /// decoded copy in F32 up to the last bit of a weight (a decoder may fuse a multiply and
    /// an add where the CPU rounds twice). It is for the matrices the engine multiplies
    /// exactly, whatever type the file stores them in: the stream mixers and the n-gram
    /// block of qwen4exp. Every weight type has such a kernel.
    ///
    /// # Safety
    ///
    /// As [`Kernels::matvec`], except that only the f32 section of `q` is read (16-byte
    /// aligned, at the offset the layout of `n_in` and `n_batch` gives it).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matvec_exact(
        &self,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        q: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_batch: u32,
    ) -> Result<()> {
        let Some((_, align, function)) = self.exact.iter().find(|(t, _, _)| *t == ty) else {
            // SAFETY: the caller's guarantee covers the float kernels, which read the f32
            // section only.
            return unsafe { self.matvec(stream, ty, w, q, y, n_in, n_out, n_batch) };
        };
        let block_values = ty.block_values();
        if n_in == 0 || !n_in.is_multiple_of(block_values) {
            return Err(Error::Invalid {
                what: "weight matrix",
                detail: format!(
                    "{n_in} columns are not a whole number of {ty} blocks of {block_values} values"
                ),
            });
        }
        check_batch(n_batch)?;
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", w, *align)?;
        check_alignment("activations", q, 16)?;
        let blocks = n_out.div_ceil(MMVQ_ROWS_PER_BLOCK);
        let args = Args::new().ptr(w).ptr(q).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
        // SAFETY: the argument list matches the ares_mmv_exact_* kernels; the ranges are the
        // caller's guarantee.
        unsafe { function.launch(stream, [blocks, 1, 1], [MMVQ_THREADS, 1, 1], 0, args) }
    }

    /// Whether the tensor cores of this device run f16 `mma` with f32
    /// accumulation at the same rate as with f16 accumulation, measured when the
    /// kernels were loaded. GeForce cards halve the first (and not the integer
    /// `mma`) unless the cap has been lifted; there the int8 variant of the
    /// prefill kernels is the fast one. `false` also on a GPU without prefill
    /// kernels.
    pub fn tensor_cores_full_rate(&self) -> bool {
        self.prefill.as_ref().is_some_and(|p| p.full_rate)
    }

    /// The rule for which weight kernel family to use at which number of
    /// activation rows: up to this many rows the decode kernels
    /// ([`Kernels::matvec`], in launches of [`MAX_BATCH`] rows, activations
    /// quantized with [`Kernels::quantize_activations`]) are the faster family
    /// for weights of type `ty`; with more rows the tensor-core kernels
    /// ([`Kernels::matmul`]) are. `None` for a type that has no kernels.
    ///
    /// The decode kernels read the weights once per launch, so their time grows
    /// by a launch with every 8 rows; the tensor-core kernels take about the same
    /// time (0.2 to 0.3 ms for a 17408 x 5120 matrix on an RTX 4070, 0.45 to 0.75 on
    /// an RTX 3060) for any number of rows up to 48, since the time of the small
    /// shapes is that of reading and decoding the weights. They cross at 12 to 20
    /// rows (the first row count, of 8, 12, 16, 20 ..., at which the tensor-core
    /// kernels are faster in `examples/prefill_bench.rs`): 12 or 16 for the types
    /// whose decode kernels reach the memory bandwidth (the K-quants of 4 and 5 bits,
    /// IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS), 8 for Q6_K, Q3_K and Q8_0, whose decode
    /// kernels turn arithmetic-bound at a few rows, and 12 for IQ2_S and IQ2_XS. The
    /// table holds, per type, the row count with the least total loss over both GPUs
    /// and the row counts 8, 12, ..., 32: following it costs at most 13% against
    /// the faster family at any of them. It was measured with the int8 path, the one
    /// both GPUs take; the f16 path of a device with full-rate tensor cores has its
    /// own crossover, not measured. The types of the second prefill module have no
    /// int8 path: theirs were measured against the f16 path with f16 accumulation (the
    /// same rule, rows 8 to 32 on both GPUs; at most 6 % lost in total at the row count
    /// given: IQ2_XXS 16, IQ1_S 12, IQ1_M 12, Q2_0 16, Q4_0 16). Q2_K, whose decode
    /// kernel loses from 9 rows on, and the types not measured have no entry: above
    /// eight rows they take the tile kernels.
    pub fn decode_kernel_rows(ty: GgmlType) -> Option<u32> {
        DECODE_ROWS.iter().find(|(t, _)| *t == ty).map(|(_, rows)| *rows)
    }

    /// The bytes of shared memory the tile kernels' decoder of `ty` keeps its tables in
    /// (`SMEM_BYTES` of the type in `kernels/mm_types*.cuh`); `None` for a type without a
    /// tile kernel. The many-row expert kernels, which decode with the same decoders, lay
    /// out their shared memory with it.
    pub fn tile_table_bytes(ty: GgmlType) -> Option<u32> {
        PREFILL_TYPES
            .iter()
            .chain(PREFILL_MORE_TYPES.iter())
            .find(|(t, _, _, _)| *t == ty)
            .map(|(_, _, _, table)| *table)
    }

    /// Whether this device has the prefill kernels at all: they need compute
    /// capability 8.0 (Ampere) or newer.
    pub fn has_prefill_kernels(&self) -> bool {
        self.prefill.is_some()
    }

    /// Whether [`Kernels::matmul`] has a kernel of `path` for weights of type `ty` on this
    /// device: every type of [`Kernels::has_fast_matvec`] has the two f16 variants, those
    /// that are not in the first module (the float types among them) have no int8 one. `false`
    /// on a GPU without prefill kernels.
    pub fn has_prefill_path(&self, ty: GgmlType, path: PrefillPath) -> bool {
        let index = match path {
            PrefillPath::Int8 => 0,
            PrefillPath::F16 => 1,
            PrefillPath::F16Flush => 2,
        };
        self.prefill.as_ref().is_some_and(|p| {
            p.types.iter().any(|(t, .., paths)| *t == ty && paths[index].is_some())
        })
    }

    /// The path the prefill family takes on this device for the requested
    /// precision. [`Precision::Exact`] is always [`PrefillPath::F16`].
    /// [`Precision::Fast`] is f16 where the tensor cores run it at full rate,
    /// since the direct product is then both fast and the more exact, and int8
    /// where they do not (measured: [`Kernels::tensor_cores_full_rate`]).
    ///
    /// The path also decides the format the activations are prepared in
    /// ([`Kernels::prepare_prefill_activations`]), so a caller that wants one
    /// kernel family for several matrices asks once and passes the result on.
    /// Refused on a GPU below compute capability 8.0.
    pub fn prefill_path(&self, precision: Precision) -> Result<PrefillPath> {
        let prefill = self.prefill_kernels()?;
        Ok(match precision {
            Precision::Exact => PrefillPath::F16,
            Precision::Fast if prefill.full_rate => PrefillPath::F16,
            Precision::Fast => PrefillPath::Int8,
        })
    }

    /// The path to use for a matrix of `ty` under `precision`: [`Kernels::prefill_path`],
    /// except that on a capped card [`Precision::Fast`] takes [`PrefillPath::F16Flush`]
    /// for the types where it is not slower than int8 (measured at 1024 rows on an RTX
    /// 4070 and an RTX 3060: Q4_K 1.04 and 1.02 times, IQ2_S 1.18 and 1.05, IQ2_XS
    /// 1.08 and 1.06; for the others 0.74 to 0.91): the same speed with 0.05 % instead
    /// of 1.1 % rms error. The activations are prepared per path
    /// ([`Kernels::prepare_prefill_activations`]), so a caller with matrices of several
    /// types prepares an input in each format it uses.
    pub fn prefill_path_for(&self, ty: GgmlType, precision: Precision) -> Result<PrefillPath> {
        let path = self.prefill_path(precision)?;
        let has_int8 = self
            .prefill_kernels()?
            .types
            .iter()
            .find(|(t, ..)| *t == ty)
            .is_none_or(|(.., paths)| paths[0].is_some());
        Ok(match (precision, path) {
            (Precision::Fast, PrefillPath::Int8) if FLUSH_AS_FAST.contains(&ty) || !has_int8 => {
                PrefillPath::F16Flush
            }
            _ => path,
        })
    }

    fn prefill_kernels(&self) -> Result<&Prefill> {
        self.prefill.as_ref().ok_or_else(|| Error::Invalid {
            what: "prefill kernel",
            detail: format!(
                "the tensor-core weight kernels need compute capability 8.0 or newer, {} has {}.{}",
                self.device.info().name,
                self.device.info().compute_capability.0,
                self.device.info().compute_capability.1
            ),
        })
    }

    /// The shapes the prefill kernels of `ty` come in for `path`, largest first: what
    /// each needs and how it sits on this device, and up to how many activation rows
    /// [`Kernels::matmul`] uses it.
    pub fn prefill_shapes(&self, ty: GgmlType, path: PrefillPath) -> Result<Vec<PrefillShape>> {
        let prefill = self.prefill_kernels()?;
        let Some((_, _, table, kernels)) = prefill.types.iter().find(|(t, ..)| *t == ty) else {
            return Err(Error::Invalid {
                what: "prefill kernel type",
                detail: format!("{ty} has no prefill kernel"),
            });
        };
        let (path_index, elem) = match path {
            PrefillPath::Int8 => (0, 1),
            PrefillPath::F16 => (1, 2),
            PrefillPath::F16Flush => (2, 2),
        };
        let up_to = [u32::MAX, MM_MEDIUM_ROWS, MM_SMALL_ROWS];
        (0..3)
            .map(|shape| {
                let info = SHAPES[path_index][shape];
                let function =
                    &kernels[path_index].as_ref().ok_or_else(|| no_variant(ty, path))?[shape];
                let shared_bytes = info.shared_bytes(elem, *table);
                let (registers, _) = function.resources()?;
                Ok(PrefillShape {
                    weight_rows: info.weight_rows,
                    batch_rows: info.batch_rows,
                    threads: info.threads,
                    shared_bytes,
                    registers,
                    blocks_per_sm: function
                        .blocks_per_multiprocessor(info.threads, shared_bytes)?,
                    up_to_rows: up_to[shape],
                })
            })
            .collect()
    }

    /// The bytes of a line of the devices' caches, and where a buffer of f16
    /// activations ([`PrefillPath::F16`], [`PrefillPath::F16Flush`]) should start: at a
    /// multiple of it. The rows of that form lie a whole number of lines apart, counted
    /// from the start of the buffer, and a step of the tile kernels reads 64 values of a
    /// row, which is one line. From a start that is 64 bytes off a line every step reads
    /// two, and the products are slower (docs/ARCHITECTURE.md, "The token embedding of
    /// `qwen35`": 1 % of a prompt on an RTX 3060). 16 bytes are all the kernels ask for;
    /// the 8-bit form, of which a step reads 64 bytes of a row, is as fast from either.
    pub const ACTIVATION_LINE: u64 = 128;

    /// The bytes [`Kernels::prepare_prefill_activations`] writes for `n_rows`
    /// rows of `n_in` activations in the format of `path`.
    pub fn prefill_activation_bytes(path: PrefillPath, n_in: u32, n_rows: u32) -> u64 {
        match path {
            PrefillPath::Int8 => Self::quantized_activation_bytes(n_in, n_rows),
            PrefillPath::F16 | PrefillPath::F16Flush => {
                // The rows lie a whole, odd number of lines of 128 bytes apart
                // (`ares_act_f16_pitch` in `kernels/mm.cuh`), so that the same place of
                // many rows does not fall into one set of the caches.
                let line = Self::ACTIVATION_LINE;
                let pitch = ((u64::from(n_in) * 2).div_ceil(line) + 1) | 1;
                let values = pitch * line * u64::from(n_rows);
                // Then, at the next multiple of 16 bytes, one f32 per row: the power of
                // two the row was divided by.
                values.next_multiple_of(16) + 4 * u64::from(n_rows)
            }
        }
    }

    /// Prepares `n_rows` rows of `n_in` f32 activations for [`Kernels::matmul`]:
    /// quantizes them to int8 ([`PrefillPath::Int8`], the format of
    /// [`Kernels::quantize_activations`], so the same bytes serve the decode
    /// kernels) or converts them to f16 ([`PrefillPath::F16`]). One preparation
    /// serves every matrix that multiplies the same activations.
    ///
    /// `n_in` must be a multiple of 32, `n_in * n_rows` must fit in 32 bits.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` must
    /// address `n_rows * n_in` f32 values and `out` the
    /// [`Kernels::prefill_activation_bytes`] of the path, 16-byte aligned
    /// (checked), not overlapping `x`. The f16 paths are fastest when `out`
    /// starts on a line ([`Kernels::ACTIVATION_LINE`]).
    pub unsafe fn prepare_prefill_activations(
        &self,
        stream: &Stream,
        path: PrefillPath,
        x: DevicePtr,
        out: DevicePtr,
        n_in: u32,
        n_rows: u32,
    ) -> Result<()> {
        let prefill = self.prefill_kernels()?;
        if n_in == 0 || !n_in.is_multiple_of(32) || n_rows == 0 {
            return Err(Error::Invalid {
                what: "prefill activations",
                detail: format!(
                    "{n_rows} rows of {n_in} values: rows must be a whole number of groups of 32 and not empty"
                ),
            });
        }
        match path {
            // SAFETY: the caller's guarantee is the one of the function it forwards to.
            PrefillPath::Int8 => unsafe { self.quantize_rows(stream, x, out, n_in, n_rows) },
            PrefillPath::F16 | PrefillPath::F16Flush => {
                check_alignment("prefill activations", out, 16)?;
                n_in.checked_mul(n_rows).ok_or(Error::Invalid {
                    what: "prefill activations",
                    detail: "size overflows u32".into(),
                })?;
                let args = Args::new().ptr(x).ptr(out).u32(n_in).u32(n_rows);
                // SAFETY: the argument list matches ares_convert_f16; the ranges are the
                // caller's guarantee.
                unsafe { prefill.convert_f16.launch(stream, [n_rows, 1, 1], [1024, 1, 1], 0, args) }
            }
        }
    }

    /// `y[b] = W x[b]` for a matrix `w` of type `ty` with `n_out` rows and
    /// `n_in` columns and `n_rows` rows of activations that
    /// [`Kernels::prepare_prefill_activations`] has prepared in the format of
    /// `path` into `act`. Any `n_rows` from 1 up: this is the tensor-core
    /// kernel for prompt processing, which dequantises a tile of the weights per
    /// step and uses it for 16 to 64 rows of activations (see
    /// `kernels/mm.cuh`). It is faster than the decode kernels
    /// ([`Kernels::matvec`]) from [`Kernels::decode_kernel_rows`] + 1 rows on.
    ///
    /// Int8 and f16 accumulate in f32 and agree with the decode kernels up to the
    /// rounding of what they compute with: int8 with the decode kernels'
    /// quantization of the activations, f16 with the rounding of weights and
    /// activations to f16; [`PrefillPath::F16Flush`] also rounds partial sums of
    /// 64 products to f16. The float types (F32, F16, BF16) have the two f16 paths
    /// only, with their weights as f16 operands without a scale: F16 weights are
    /// exact, BF16 weights exact within the range of normal f16 numbers, F32 weights
    /// rounded to 11 bits; a weight beyond 65504 is an infinity in the tile, so the
    /// outputs that read it are infinite or NaN, and one below 6.1e-5 keeps what an f16
    /// subnormal holds (error at most 2^-25); [`WeightRange`] tells a loader which
    /// path holds the weights of a matrix. A type without a prefill kernel is
    /// refused with an error, as is a device below compute capability 8.0.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `w` must
    /// address the matrix as GGUF stores it (`n_out` rows of `n_in` values in
    /// whole blocks of `ty`; `n_in` a multiple of 32 as well), aligned to 16 bytes for Q4_K,
    /// Q5_K and the float types, to 8 for IQ4_XS and to 2 for the other types (checked);
    /// `act` the activations
    /// prepared for this `path`, `n_in` and `n_rows`, 16-byte aligned
    /// (checked); `y` `n_rows * n_out` f32 values. `y` must not overlap the
    /// inputs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul(
        &self,
        stream: &Stream,
        ty: GgmlType,
        path: PrefillPath,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        let prefill = self.prefill_kernels()?;
        let Some((_, align, table, kernels)) = prefill.types.iter().find(|(t, ..)| *t == ty) else {
            return Err(Error::Invalid {
                what: "prefill kernel type",
                detail: format!(
                    "{ty} has no prefill kernel (it has one for {})",
                    PREFILL_TYPES
                        .iter()
                        .chain(&PREFILL_MORE_TYPES)
                        .map(|(t, ..)| t.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        };
        // The kernels take the weights in items of 32 values; a float type has blocks of one.
        let block_values = ty.block_values().max(32);
        if n_in == 0 || !n_in.is_multiple_of(block_values) {
            return Err(Error::Invalid {
                what: "weight matrix",
                detail: format!(
                    "{n_in} columns are not a whole number of {ty} blocks of {block_values} values"
                ),
            });
        }
        if n_rows == 0 || n_in.checked_mul(n_rows).is_none() {
            return Err(Error::Invalid {
                what: "prefill kernel batch",
                detail: format!("{n_rows} rows of {n_in} values: not 1 or more, or too many"),
            });
        }
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", w, *align)?;
        check_alignment("prepared activations", act, 16)?;
        let (path_index, elem, act_row_bytes) = match path {
            PrefillPath::Int8 => (0, 1, u64::from(n_in) * 9 / 8),
            PrefillPath::F16 => (1, 2, u64::from(n_in) * 2),
            PrefillPath::F16Flush => (2, 2, u64::from(n_in) * 2),
        };
        let shape = shape_index(n_rows);
        let function = &kernels[path_index].as_ref().ok_or_else(|| no_variant(ty, path))?[shape];
        let shape_info = SHAPES[path_index][shape];
        let Shape { weight_rows, batch_rows, threads } = shape_info;
        let shared = shape_info.shared_bytes(elem, *table);
        let n_weight_tiles = n_out.div_ceil(weight_rows);
        let n_batch_tiles = n_rows.div_ceil(batch_rows);
        let group = (MM_GROUP_BYTES / (u64::from(batch_rows) * act_row_bytes))
            .clamp(MM_GROUP_MIN, MM_GROUP_MAX);
        let group = u32::try_from(group).unwrap_or(2).min(n_batch_tiles);
        let blocks = u64::from(n_weight_tiles) * u64::from(n_batch_tiles);
        let blocks = u32::try_from(blocks).map_err(|_| Error::Invalid {
            what: "prefill kernel batch",
            detail: format!("{n_out} x {n_rows} needs more blocks than a grid has"),
        })?;
        let prefetch =
            u32::from(shape != 0 && self.device.info().l2_cache_bytes >= MM_PREFETCH_L2_BYTES);
        let args = Args::new()
            .ptr(w)
            .ptr(act)
            .ptr(y)
            .u32(n_in)
            .u32(n_out)
            .u32(n_rows)
            .u32(group)
            .u32(prefetch);
        // SAFETY: the argument list matches the ares_mm_* kernels; the ranges are the
        // caller's guarantee.
        unsafe { function.launch(stream, [blocks, 1, 1], [threads, 1, 1], shared, args) }
    }
}

/// The error for a path a type has no variant of (the types of the second module have
/// no int8 one).
fn no_variant(ty: GgmlType, path: PrefillPath) -> Error {
    Error::Invalid {
        what: "prefill kernel path",
        detail: format!("{ty} has no prefill kernel for the {path:?} path"),
    }
}

/// Which of the shapes takes `n_rows` rows of activations.
const fn shape_index(n_rows: u32) -> usize {
    if n_rows <= MM_SMALL_ROWS {
        2
    } else if n_rows <= MM_MEDIUM_ROWS {
        1
    } else {
        0
    }
}

fn check_batch(n_batch: u32) -> Result<()> {
    if n_batch == 0 || n_batch > MAX_BATCH {
        return Err(Error::Invalid {
            what: "weight kernel batch",
            detail: format!("{n_batch} activation rows; one launch takes 1 to {MAX_BATCH}"),
        });
    }
    Ok(())
}

fn check_alignment(what: &'static str, ptr: DevicePtr, align: u64) -> Result<()> {
    if !ptr.as_raw().is_multiple_of(align) {
        return Err(Error::Invalid {
            what,
            detail: format!("address {:#x} is not aligned to {align} bytes", ptr.as_raw()),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ares_core::GgmlType;
    use ares_quant::{f32_to_bf16, f32_to_f16};

    use super::{PrefillPath, WeightRange};

    fn bytes(ty: GgmlType, values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| match ty {
                GgmlType::F32 => v.to_le_bytes().to_vec(),
                GgmlType::F16 => f32_to_f16(*v).to_le_bytes().to_vec(),
                _ => f32_to_bf16(*v).to_le_bytes().to_vec(),
            })
            .collect()
    }

    /// The largest magnitude decides, of either sign and wherever it is; the limits belong
    /// to the range below them. The values are ones every float type holds.
    #[test]
    fn the_range_of_a_float_matrix_is_that_of_its_largest_weight() {
        for ty in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let range = |values: &[f32]| WeightRange::of(ty, &bytes(ty, values));
            assert_eq!(range(&[]), WeightRange::Any, "{ty}");
            assert_eq!(range(&[0.0, -0.0, 3.0e-7, 0.25, -8.5]), WeightRange::Any, "{ty}");
            assert_eq!(range(&[0.25, -512.0, 3.0]), WeightRange::Any, "{ty}");
            assert_eq!(range(&[0.25, 3.0, -520.0]), WeightRange::F32Sums, "{ty}");
            assert_eq!(range(&[65280.0, 0.25]), WeightRange::F32Sums, "{ty}");
            assert_eq!(range(&[0.25, f32::INFINITY, 1.0]), WeightRange::Beyond, "{ty}");
            assert_eq!(range(&[f32::NEG_INFINITY]), WeightRange::Beyond, "{ty}");
            assert_eq!(range(&[1.0, f32::NAN]), WeightRange::Beyond, "{ty}");
        }
        // The largest f16 is within f16; the next bf16 and f32 are not.
        assert_eq!(WeightRange::of(GgmlType::F16, &0x7BFFu16.to_le_bytes()), WeightRange::F32Sums);
        assert_eq!(WeightRange::of(GgmlType::F32, &65504f32.to_le_bytes()), WeightRange::F32Sums);
        assert_eq!(WeightRange::of(GgmlType::F32, &65505f32.to_le_bytes()), WeightRange::Beyond);
        for ty in [GgmlType::F32, GgmlType::BF16] {
            assert_eq!(WeightRange::of(ty, &bytes(ty, &[1.0, -7.0e4])), WeightRange::Beyond);
            assert_eq!(WeightRange::of(ty, &bytes(ty, &[1.0e30, 2.0])), WeightRange::Beyond);
        }
        // The scales of the other types are f16 in the file: any bytes are in range.
        assert_eq!(WeightRange::of(GgmlType::Q4_K, &[0xFF; 144]), WeightRange::Any);
    }

    #[test]
    fn a_range_leaves_the_paths_that_hold_its_weights() {
        use PrefillPath::{F16, F16Flush, Int8};
        for path in [None, Some(Int8), Some(F16), Some(F16Flush)] {
            assert_eq!(WeightRange::Any.path(path), path);
            assert_eq!(WeightRange::Beyond.path(path), None);
        }
        assert_eq!(WeightRange::F32Sums.path(Some(F16Flush)), Some(F16));
        assert_eq!(WeightRange::F32Sums.path(Some(F16)), Some(F16));
        assert_eq!(WeightRange::F32Sums.path(None), None);
        let all = [WeightRange::Any, WeightRange::F32Sums, WeightRange::Beyond];
        for (i, a) in all.into_iter().enumerate() {
            for (j, b) in all.into_iter().enumerate() {
                assert_eq!(a.and(b), all[i.max(j)]);
            }
        }
    }
}
