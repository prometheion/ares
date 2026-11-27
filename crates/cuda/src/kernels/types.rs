//! What the kernels of this crate are and need for each weight type, in one table
//! ([`TYPES`]): every list of types the crate keeps is read out of it. The kernels are named
//! after the type as ggml spells it ([`GgmlType::name`]: `ares_mmvq_q4_K`, `ares_mm_f16_iq2_s`,
//! `ares_er_down_q2_0_n1` ...), as the kernel sources instantiate them, so the table holds no
//! names.
//!
//! What a kernel can tell about itself is not in the table: whether a decode kernel walks
//! several tiles of rows per thread block is read from the kernel when it is loaded (its
//! codebook in shared memory, see [`DECODE_WALK_TABLE_BYTES`](super::DECODE_WALK_TABLE_BYTES)),
//! and which module of the many-row expert kernels holds a type's kernels is found by asking
//! each module for them (`ExpertRowKernels::load`).

use ares_core::GgmlType::{self, *};

/// The kernels of one weight type and what they need.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TypeInfo {
    pub ty: GgmlType,
    /// The alignment in bytes the decode kernel (`ares_mmvq_<type>`, `kernels/mmvq_*.cuh`)
    /// needs of a matrix: the widest load it uses.
    pub decode_align: u64,
    /// The alignment the exact kernel of [`Kernels::matvec_exact`](super::Kernels::matvec_exact)
    /// (`ares_mmv_exact_<type>`, `kernels/mmv_exact.cu`) needs; its decoders read bytes and the
    /// f16 scales of the blocks. `None` for the float types, which take their decode kernel.
    pub exact_align: Option<u64>,
    /// The tile kernels (`ares_mm_{i8,f16,h}{,m,s}_<type>`, `kernels/mm_types*.cuh`).
    pub tile: Tile,
    /// Up to how many activation rows the decode kernels are the faster family
    /// ([`Kernels::decode_kernel_rows`](super::Kernels::decode_kernel_rows)); `None` for a
    /// type that was not measured.
    pub decode_rows: Option<u32>,
    /// Whether [`PrefillPath::F16Flush`](super::PrefillPath::F16Flush) was as fast as int8 on
    /// the two capped cards the kernels were written on: such a type takes an f16 path where it
    /// was not measured, on every card ([`Kernels::prefill_path_for`](super::Kernels::prefill_path_for)).
    pub flush_as_fast: bool,
    /// The bytes of the tables of the gate and up decoder written for the many-row expert
    /// kernels (`kernels/expertrows_types.cuh`, and IQ1_S's in `mm_types_more.cuh`); `None`
    /// for a type whose many-row kernels decode with the item decoder of its tile kernels.
    pub rows_gate_up_table: Option<u32>,
    /// Whether the fused read of qwen4exp's stream mixer has its decoder inlined for the type
    /// (`kernels/q4_reads.cu`): the quantized types files use for the mixers. The others are
    /// read by the kernels that decode by the type at run time.
    pub inlined_read: bool,
}

/// The tile kernels of a type (`kernels/mm_types*.cuh`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tile {
    /// Whether the type has the int8 variant. Those that do are in the first prefill module
    /// (`kernels/prefill.cu`), the others in the second (`prefill_more.cu`), with the two
    /// f16 variants only.
    pub int8: bool,
    /// The alignment in bytes the kernels need of a matrix.
    pub align: u64,
    /// The bytes of shared memory its decoder keeps its tables in (`SMEM_BYTES` of the type).
    pub table: u32,
}

const fn row(ty: GgmlType, decode_align: u64, exact_align: u64, tile: Tile) -> TypeInfo {
    TypeInfo {
        ty,
        decode_align,
        exact_align: if exact_align == 0 { None } else { Some(exact_align) },
        tile,
        decode_rows: None,
        flush_as_fast: false,
        rows_gate_up_table: None,
        inlined_read: false,
    }
}

const fn int8(align: u64, table: u32) -> Tile {
    Tile { int8: true, align, table }
}

const fn f16(align: u64, table: u32) -> Tile {
    Tile { int8: false, align, table }
}

impl TypeInfo {
    const fn decode_rows(mut self, rows: u32) -> Self {
        self.decode_rows = Some(rows);
        self
    }

    const fn flush_as_fast(mut self) -> Self {
        self.flush_as_fast = true;
        self
    }

    const fn rows_gate_up_table(mut self, bytes: u32) -> Self {
        self.rows_gate_up_table = Some(bytes);
        self
    }

    const fn inlined_read(mut self) -> Self {
        self.inlined_read = true;
        self
    }
}

/// Every weight type a model file can hold (the types of [`GgmlType::is_weight`]): each has
/// a decode kernel, tile kernels, the decode and the many-row expert kernels for gate and up
/// and for down, and a fused read. The columns: the type, the alignment of its decode kernel,
/// that of its exact kernel (0 for none) and its tile kernels (`int8` or `f16` only, their
/// alignment and table bytes), then what holds of some types only.
///
/// The order is that of [`ExpertShape::types`](crate::ExpertShape::types).
///
/// The decode rows: the row counts with the least total loss over the two GPUs and the row
/// counts 8, 12, ..., 32 ([`Kernels::decode_kernel_rows`](super::Kernels::decode_kernel_rows)).
#[rustfmt::skip]
pub(crate) const TYPES: [TypeInfo; 28] = [
    row(Q4_K,    16, 2, int8(16, 0)).decode_rows(16).flush_as_fast().inlined_read(),
    row(Q5_K,    16, 2, int8(16, 0)).decode_rows(16).inlined_read(),
    row(Q6_K,     2, 2, int8(2, 0)).decode_rows(8).inlined_read(),
    row(Q3_K,     2, 2, int8(2, 0)).decode_rows(8),
    row(Q2_K,     4, 2, f16(4, 0)),
    row(IQ4_XS,   8, 2, int8(8, 0)).decode_rows(12),
    row(IQ4_NL,   2, 2, int8(2, 0)).decode_rows(12).inlined_read(),
    row(IQ3_S,    2, 2, int8(2, 512 * 4)).decode_rows(16),
    row(IQ3_XXS,  2, 2, int8(2, 256 * 4)).decode_rows(16),
    row(IQ2_S,    2, 2, int8(2, 1024 * 8)).decode_rows(12).flush_as_fast()
        .rows_gate_up_table(8192 + 4096),
    row(IQ2_XS,   2, 2, int8(2, 512 * 8)).decode_rows(12).flush_as_fast(),
    row(IQ2_XXS,  2, 2, f16(2, 2048 + 4096)).decode_rows(16).rows_gate_up_table(2048 + 4096),
    row(IQ1_S,    2, 2, f16(2, 16384)).decode_rows(12).rows_gate_up_table(16384),
    row(IQ1_M,    8, 2, f16(8, 16384)).decode_rows(12).rows_gate_up_table(16384),
    row(Q8_0,     2, 2, int8(2, 0)).decode_rows(8).inlined_read(),
    row(Q4_0,     2, 2, f16(2, 0)).decode_rows(16).inlined_read(),
    row(Q4_1,     4, 2, f16(4, 0)).inlined_read(),
    row(Q5_0,     2, 2, f16(2, 0)).inlined_read(),
    row(Q5_1,     4, 2, f16(4, 0)).inlined_read(),
    row(TQ1_0,    2, 2, f16(2, 0)),
    row(TQ2_0,    2, 2, f16(2, 0)),
    row(MXFP4,    1, 1, f16(1, 0)),
    row(NVFP4,    4, 2, f16(4, 0)),
    row(Q1_0,     2, 2, f16(2, 0)),
    row(Q2_0,     2, 2, f16(2, 0)).decode_rows(16),
    row(F32,      4, 0, f16(16, 0)),
    row(F16,      2, 0, f16(16, 0)),
    row(BF16,     2, 0, f16(16, 0)),
];

/// The index of `ty` in [`TYPES`]: `None` for a type no weight is stored in (the integer
/// types, F64 and the two activation formats).
pub(crate) fn index(ty: GgmlType) -> Option<usize> {
    TYPES.iter().position(|t| t.ty == ty)
}

/// The row of `ty` in [`TYPES`].
pub(crate) fn info(ty: GgmlType) -> Option<&'static TypeInfo> {
    index(ty).map(|i| &TYPES[i])
}

/// The names of the types of [`TYPES`] that `keep` keeps, for messages.
pub(crate) fn names(keep: impl Fn(&TypeInfo) -> bool) -> String {
    TYPES.iter().filter(|t| keep(t)).map(|t| t.ty.name()).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use ares_core::ggml_type::ALL_TYPES;

    use super::*;

    #[test]
    fn every_weight_type_has_one_row() {
        for ty in ALL_TYPES {
            let rows = TYPES.iter().filter(|t| t.ty == ty).count();
            assert_eq!(rows, usize::from(ty.is_weight()), "{ty}");
        }
    }

    #[test]
    fn the_float_types_take_their_decode_kernel_as_the_exact_one() {
        for t in TYPES {
            assert_eq!(t.exact_align.is_none(), !t.ty.is_quantized(), "{}", t.ty);
        }
    }
}
