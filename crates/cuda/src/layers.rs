//! The layer kernels: everything a model needs between the matrix products and
//! attention, from `kernels/layers.cu`.
//!
//! They take their sizes as arguments and read what varies from step to step
//! (which rows belong to which sequence, their tokens and positions, which
//! slot of recurrent state a sequence owns) from the **step record**
//! ([`StepRecord`]): one small buffer that the conductor writes into pinned
//! memory and uploads once per step. A step carries rows of any mix of
//! sequences, one [`Segment`] of consecutive rows per sequence.
//!
//! What the conductor puts in the record, per step:
//!
//! - per segment: the first row, the number of rows (the length of the window),
//!   the sequence's slot in the recurrent state arrays, the position of the
//!   first row, whether the segment is *direct* (every row is certain to be
//!   accepted: a slice of a prompt, a generation row without drafts) and, after
//!   verification and before the commit kernels run, the number of accepted
//!   rows;
//! - per row: the token (embedding gather) and four positions (time, height,
//!   width, extra; a text token has `(p, p, p, 0)`), from which the rotary table
//!   is built;
//! - the rows whose logits are wanted.
//!
//! Recurrent state is addressed by slot: `base + slot * slot_stride` (in
//! elements) per layer, in the layout of `ares_model`'s `SequenceState`: the
//! convolution window `[channel][K-1]` oldest first, the delta-rule matrices
//! `[value head][value index][key index]`.
//!
//! # A recurrent block in a step
//!
//! ```text
//! rms_norm, prepare activations, products: qkv, z, beta_in, alpha     (existing kernels)
//! conv_window      qkv -> q, kv        reads the windows, never writes them
//! delta_window     q, kv, alpha, beta_in -> o     writes the matrices of direct segments only
//! conv_commit(Direct)                 direct segments: windows advance now, while qkv is there
//! gated_norm_quant o, z -> activations for ssm_out                    (then the product, residual)
//! ... end of the step, after verification wrote `accepted` into the record:
//! commit           conv windows and matrices of the deferred segments, from the kept
//!                  qkv, kv, alpha, beta_in
//! ```
//!
//! What a block keeps for the commit of deferred segments: `qkv` (`conv_dim` values per
//! row), `kv` (`key_dim + value_dim`), `alpha` and `beta_in` (`n_v_heads` each): 18 528
//! values per row for the 27B. Direct segments need none of it after their own launches, so
//! a step without drafts can reuse one set of buffers for all blocks.
//!
//! All launches are `unsafe`: the device trusts the addresses it is given.

use ares_core::GgmlType;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DevicePtr;
use crate::mesh::{PUSH_MAX_ROWS, PushSlot};
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/layers.fatbin"));

const SEG_BYTES: usize = 32;
const HEADER_BYTES: usize = 32;
const FLAG_DIRECT: u32 = 1;

/// One sequence's rows in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The first row of the segment among the rows of the step.
    pub row0: u32,
    /// The number of rows: the length of the window.
    pub n_rows: u32,
    /// The sequence's slot in the recurrent state arrays.
    pub slot: u32,
    /// The position of the first row.
    pub pos0: i32,
    /// All rows will be accepted: the delta-rule window kernel writes the state
    /// itself, and the deferred commit skips the segment.
    pub direct: bool,
}

/// The sizes of a step that decide grid sizes. The host knows them without
/// reading the record back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepDims {
    pub n_rows: u32,
    pub n_segs: u32,
    /// The length of the longest segment.
    pub max_seg_rows: u32,
    pub n_out: u32,
}

/// The step record: header, segments, tokens, positions and output rows in the
/// layout `kernels/layer_step.cuh` documents. Upload [`StepRecord::as_bytes`]
/// to a device buffer (16-byte aligned) and pass its address to the kernels.
#[derive(Clone, Debug)]
pub struct StepRecord {
    bytes: Vec<u8>,
    dims: StepDims,
}

impl StepRecord {
    /// Builds a record. `tokens` has one id per row. `positions` has four
    /// positions per row, or is `None` for plain text: row `t` of a segment gets
    /// `(pos0 + t, pos0 + t, pos0 + t, 0)`. `out_rows` lists the rows whose
    /// logits are wanted.
    pub fn new(
        segments: &[Segment],
        tokens: &[u32],
        positions: Option<&[[i32; 4]]>,
        out_rows: &[u32],
    ) -> Result<Self> {
        let n_rows = tokens.len();
        let bad = |detail: String| Error::Invalid { what: "step record", detail };
        if let Some(p) = positions
            && p.len() != n_rows
        {
            return Err(bad(format!("{} position sets for {n_rows} rows", p.len())));
        }
        let mut covered = vec![false; n_rows];
        for (i, s) in segments.iter().enumerate() {
            let end = s.row0 as usize + s.n_rows as usize;
            if end > n_rows {
                return Err(bad(format!("segment {i} ends at row {end} of {n_rows}")));
            }
            for (row, seen) in covered.iter_mut().enumerate().take(end).skip(s.row0 as usize) {
                if std::mem::replace(seen, true) {
                    return Err(bad(format!("segment {i} overlaps another at row {row}")));
                }
            }
        }
        if let Some(&r) = out_rows.iter().find(|&&r| r as usize >= n_rows) {
            return Err(bad(format!("output row {r} of {n_rows}")));
        }
        let align = |n: usize| n.next_multiple_of(16);
        let off_segs = HEADER_BYTES;
        let off_tokens = align(off_segs + segments.len() * SEG_BYTES);
        let off_pos = align(off_tokens + n_rows * 4);
        let off_out = align(off_pos + n_rows * 16);
        let total = align(off_out + out_rows.len() * 4).max(HEADER_BYTES + 16);
        let mut bytes = vec![0u8; total];
        let put = |bytes: &mut [u8], at: usize, v: u32| {
            bytes[at..at + 4].copy_from_slice(&v.to_le_bytes())
        };
        put(&mut bytes, 0, n_rows as u32);
        put(&mut bytes, 4, segments.len() as u32);
        put(&mut bytes, 8, out_rows.len() as u32);
        put(&mut bytes, 16, off_segs as u32);
        put(&mut bytes, 20, off_tokens as u32);
        put(&mut bytes, 24, off_pos as u32);
        put(&mut bytes, 28, off_out as u32);
        let mut max_seg_rows = 0;
        for (i, s) in segments.iter().enumerate() {
            let at = off_segs + i * SEG_BYTES;
            put(&mut bytes, at, s.row0);
            put(&mut bytes, at + 4, s.n_rows);
            put(&mut bytes, at + 8, s.slot);
            put(&mut bytes, at + 12, s.n_rows); // accepted: all, until the verifier says otherwise
            put(&mut bytes, at + 16, s.pos0 as u32);
            put(&mut bytes, at + 20, if s.direct { FLAG_DIRECT } else { 0 });
            max_seg_rows = max_seg_rows.max(s.n_rows);
            if positions.is_none() {
                for t in 0..s.n_rows as usize {
                    let p = s.pos0 + t as i32;
                    let row = s.row0 as usize + t;
                    for (k, v) in [p, p, p, 0].into_iter().enumerate() {
                        put(&mut bytes, off_pos + row * 16 + k * 4, v as u32);
                    }
                }
            }
        }
        for (row, &token) in tokens.iter().enumerate() {
            put(&mut bytes, off_tokens + row * 4, token);
        }
        if let Some(p) = positions {
            for (row, set) in p.iter().enumerate() {
                for (k, v) in set.iter().enumerate() {
                    put(&mut bytes, off_pos + row * 16 + k * 4, *v as u32);
                }
            }
        }
        for (i, &r) in out_rows.iter().enumerate() {
            put(&mut bytes, off_out + i * 4, r);
        }
        let dims = StepDims {
            n_rows: n_rows as u32,
            n_segs: segments.len() as u32,
            max_seg_rows,
            n_out: out_rows.len() as u32,
        };
        Ok(Self { bytes, dims })
    }

    /// The bytes to upload.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The sizes that decide grids.
    pub fn dims(&self) -> StepDims {
        self.dims
    }

    /// Sets the number of accepted rows of segment `segment` (at most its rows).
    /// In the engine the verifier writes this field on the device after the
    /// window; the host can also rewrite the record and upload again.
    pub fn set_accepted(&mut self, segment: usize, accepted: u32) -> Result<()> {
        let at = HEADER_BYTES + segment * SEG_BYTES;
        if segment >= self.dims.n_segs as usize {
            return Err(Error::Invalid {
                what: "step record",
                detail: format!("segment {segment} of {}", self.dims.n_segs),
            });
        }
        let n_rows = u32::from_le_bytes(self.bytes[at + 4..at + 8].try_into().unwrap());
        if accepted > n_rows {
            return Err(Error::Invalid {
                what: "step record",
                detail: format!("{accepted} accepted rows of {n_rows}"),
            });
        }
        self.bytes[at + 12..at + 16].copy_from_slice(&accepted.to_le_bytes());
        Ok(())
    }

    /// The byte offset of the `accepted` field of segment `segment` in the
    /// record, for a kernel that writes it on the device.
    pub fn accepted_offset(segment: usize) -> u64 {
        (HEADER_BYTES + segment * SEG_BYTES + 12) as u64
    }
}

/// The geometry of a recurrent block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentShape {
    /// Query/key heads.
    pub n_k_heads: u32,
    /// Value heads (a multiple of `n_k_heads`; value head `h` uses key head `h % n_k_heads`).
    pub n_v_heads: u32,
    /// The width of a key head and of a value head (128).
    pub head_dim: u32,
    /// The width of the causal convolution (4).
    pub conv_kernel: u32,
}

impl RecurrentShape {
    pub fn key_dim(&self) -> u32 {
        self.n_k_heads * self.head_dim
    }
    pub fn value_dim(&self) -> u32 {
        self.n_v_heads * self.head_dim
    }
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }
    /// Values of the convolution window of one sequence and block.
    pub fn conv_state_len(&self) -> u64 {
        u64::from(self.conv_dim()) * u64::from(self.conv_kernel - 1)
    }
    /// Values of the delta-rule matrices of one sequence and block.
    pub fn delta_state_len(&self) -> u64 {
        u64::from(self.n_v_heads) * u64::from(self.head_dim) * u64::from(self.head_dim)
    }

    fn check(&self) -> Result<()> {
        let bad = |detail: String| Error::Invalid { what: "recurrent shape", detail };
        if self.head_dim != 128 {
            return Err(bad(format!(
                "head width {} (the kernels are built for 128)",
                self.head_dim
            )));
        }
        if self.n_k_heads == 0
            || self.n_v_heads == 0
            || !self.n_v_heads.is_multiple_of(self.n_k_heads)
        {
            return Err(bad(format!(
                "{} value heads over {} key heads",
                self.n_v_heads, self.n_k_heads
            )));
        }
        if self.conv_kernel < 2 || self.conv_kernel > 8 {
            return Err(bad(format!("convolution width {} (2 to 8)", self.conv_kernel)));
        }
        Ok(())
    }
}

/// How the delta-rule matrices are stored. The arithmetic is f32 either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateFormat {
    F32,
    F16,
}

impl StateFormat {
    pub fn bytes(self) -> u64 {
        match self {
            StateFormat::F32 => 4,
            StateFormat::F16 => 2,
        }
    }
}

/// What a norm kernel divides by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// `x / sqrt(mean(x^2) + eps) * weight`
    Rms,
    /// `x / sqrt(sum(x^2) + eps) * weight`: the q/k norm of the delta rule.
    L2,
}

/// The gate of the gated output norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputGate {
    /// `silu(z)`: Qwen3.5 and Qwen3.8.
    Silu,
    /// `sigmoid(z)`: qwen4exp.
    Sigmoid,
}

/// The rotary embedding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    /// The number of rotated values of a head (64).
    pub dim: u32,
    pub freq_base: f32,
    /// The interleaved sections of the pairs over (time, height, width, extra).
    pub sections: [u32; 4],
}

/// The shapes of attention's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape {
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
}

/// One compiled shape of the delta-rule kernels.
#[derive(Clone, Debug)]
struct DeltaKernels {
    /// Rows of the matrix per block: the grid has `head_dim / rows_per_block` blocks per head.
    rows_per_block: u32,
    threads: u32,
    window_f32: Function,
    window_f16: Function,
    commit_f32: Function,
    commit_f16: Function,
}

/// The compiled shapes of the delta rule: suffix, lanes per row, rows per block.
const DELTA_VARIANTS: [(&str, u32, u32); 10] = [
    ("d128", 2, 64),
    ("lpr4_rb64", 4, 64),
    ("lpr8_rb64", 8, 64),
    ("lpr4_rb32", 4, 32),
    ("lpr4_rb128", 4, 128),
    ("lpr2_rb32", 2, 32),
    ("lpr8_rb32", 8, 32),
    ("lpr1_rb64", 1, 64),
    ("lpr2_rb64_tc8", 2, 64),
    ("lpr2_rb16", 2, 16),
];

/// The layer kernels on one device.
#[derive(Clone, Debug)]
pub struct LayerKernels {
    device: Device,
    embed_gather: Function,
    head_norm: Function,
    gather_rows: Function,
    rope_table: Function,
    attn_prep: Function,
    gated_norm_quant: Function,
    swiglu_quant: Function,
    gated_norm_f16: Function,
    swiglu_f16: Function,
    place_rows: Function,
    partial_to_wire: Function,
    merge_wire: Function,
    merge_push: Function,
    conv_window: Function,
    conv_commit: Function,
    delta_fork: Function,
    delta: Vec<(&'static str, DeltaKernels)>,
    delta_selected: usize,
}

fn check_align(what: &'static str, ptr: DevicePtr, align: u64) -> Result<()> {
    if !ptr.as_raw().is_multiple_of(align) {
        return Err(Error::Invalid {
            what,
            detail: format!("address {:#x} is not aligned to {align} bytes", ptr.as_raw()),
        });
    }
    Ok(())
}

fn invalid(what: &'static str, detail: String) -> Error {
    Error::Invalid { what, detail }
}

/// The bytes of the quantized activations the fused kernels write for
/// `n_values` values: the layout of `Kernels::quantize_activations`.
pub fn quantized_activation_bytes(n_values: u64) -> u64 {
    let pairs_end = n_values.next_multiple_of(16) + n_values.div_ceil(32) * 8;
    pairs_end.next_multiple_of(16) + n_values * 4
}

impl LayerKernels {
    /// Loads the layer kernels on `device`.
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        let delta = DELTA_VARIANTS
            .iter()
            .map(|&(suffix, lpr, rb)| {
                Ok((
                    suffix,
                    DeltaKernels {
                        rows_per_block: rb,
                        threads: rb * lpr,
                        window_f32: module.function(&format!("ares_delta_window_f32_{suffix}"))?,
                        window_f16: module.function(&format!("ares_delta_window_f16_{suffix}"))?,
                        commit_f32: module.function(&format!("ares_delta_commit_f32_{suffix}"))?,
                        commit_f16: module.function(&format!("ares_delta_commit_f16_{suffix}"))?,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: device.clone(),
            embed_gather: module.function("ares_embed_gather")?,
            head_norm: module.function("ares_head_norm")?,
            gather_rows: module.function("ares_gather_rows")?,
            rope_table: module.function("ares_rope_table")?,
            attn_prep: module.function("ares_attn_prep")?,
            gated_norm_quant: module.function("ares_gated_norm_quant")?,
            swiglu_quant: module.function("ares_swiglu_quant")?,
            gated_norm_f16: module.function("ares_gated_norm_f16")?,
            swiglu_f16: module.function("ares_swiglu_f16")?,
            place_rows: module.function("ares_place_rows")?,
            partial_to_wire: module.function("ares_partial_to_wire")?,
            merge_wire: module.function("ares_merge_wire")?,
            merge_push: module.function("ares_merge_push")?,
            conv_window: module.function("ares_conv_window")?,
            conv_commit: module.function("ares_conv_commit")?,
            delta_fork: module.function("ares_delta_window_fork_f32")?,
            delta,
            delta_selected: 0,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The names of the compiled shapes of the delta rule (for tuning); the
    /// first is the default.
    pub fn delta_variants() -> impl Iterator<Item = &'static str> {
        DELTA_VARIANTS.iter().map(|v| v.0)
    }

    /// Selects a compiled shape of the delta rule by name.
    pub fn select_delta_variant(&mut self, name: &str) -> Result<()> {
        let at = self
            .delta
            .iter()
            .position(|(n, _)| *n == name)
            .ok_or_else(|| invalid("delta variant", format!("no variant '{name}'")))?;
        self.delta_selected = at;
        Ok(())
    }

    /// Whether [`LayerKernels::embed_gather`] can decode rows of type `ty`.
    pub fn can_gather(ty: GgmlType) -> bool {
        use GgmlType::*;
        matches!(
            ty,
            F32 | F16
                | BF16
                | Q4_0
                | Q4_1
                | Q5_0
                | Q5_1
                | Q8_0
                | Q2_K
                | Q3_K
                | Q4_K
                | Q5_K
                | Q6_K
                | IQ4_NL
                | IQ4_XS
        )
    }

    /// Embedding gather: `out[r] = table[token_r]` decoded to f32 for the
    /// `dims.n_rows` rows of the step. `table` is the tensor as GGUF stores it
    /// (`n_vocab` rows of `n_embd` values of type `ty`), aligned to 4 bytes.
    /// A token past `n_vocab` gives a row of zeros.
    ///
    /// # Safety
    ///
    /// `step` must address the uploaded record, `table` the whole table and
    /// `out` `n_rows * n_embd` f32 values, on the stream's device, alive until
    /// the launch has run.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn embed_gather(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        table: DevicePtr,
        ty: GgmlType,
        n_vocab: u32,
        n_embd: u32,
        out: DevicePtr,
    ) -> Result<()> {
        if !Self::can_gather(ty) {
            return Err(invalid("embedding gather", format!("no decoder for {ty}")));
        }
        if !n_embd.is_multiple_of(ty.block_values()) {
            return Err(invalid(
                "embedding gather",
                format!("{n_embd} values are not whole {ty} blocks of {}", ty.block_values()),
            ));
        }
        check_align("embedding table", table, 4)?;
        if dims.n_rows == 0 {
            return Ok(());
        }
        let row_bytes = ty.bytes_for(u64::from(n_embd)).expect("whole blocks");
        let args = Args::new()
            .ptr(step)
            .ptr(table)
            .u32(ty.id())
            .u64(row_bytes)
            .u32(n_vocab)
            .u32(n_embd)
            .ptr(out);
        // SAFETY: the argument list matches ares_embed_gather; ranges are the caller's guarantee.
        unsafe { self.embed_gather.launch(stream, [dims.n_rows, 1, 1], [256, 1, 1], 0, args) }
    }

    /// A norm of `rows` rows of `width` values with a weight of `width` values
    /// shared by all rows (`None` for none): the per-head norms of q, k and the
    /// delta rule's L2 norm. In place (`y == x`) is allowed. `width` is a
    /// multiple of 4 and at most 1024.
    ///
    /// # Safety
    ///
    /// `x` and `y` must address `rows * width` f32 values, `weight` `width`
    /// values, all 16-byte aligned, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn head_norm(
        &self,
        stream: &Stream,
        kind: NormKind,
        x: DevicePtr,
        weight: Option<DevicePtr>,
        y: DevicePtr,
        rows: u32,
        width: u32,
        eps: f32,
    ) -> Result<()> {
        if width == 0 || !width.is_multiple_of(4) || width > 1024 {
            return Err(invalid(
                "head norm",
                format!("width {width} (a multiple of 4, at most 1024)"),
            ));
        }
        for (what, p) in [("x", x), ("y", y)] {
            check_align(if what == "x" { "head norm input" } else { "head norm output" }, p, 16)?;
        }
        if let Some(w) = weight {
            check_align("head norm weight", w, 16)?;
        }
        if rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(x)
            .ptr(weight.unwrap_or(DevicePtr::NULL))
            .ptr(y)
            .u32(rows)
            .u32(width)
            .f32(eps)
            .u32(u32::from(kind == NormKind::Rms));
        // SAFETY: the argument list matches ares_head_norm.
        unsafe { self.head_norm.launch(stream, [rows.div_ceil(8), 1, 1], [256, 1, 1], 0, args) }
    }

    /// The rows whose logits are wanted: `y[i] = x[out_rows[i]]` for `i <
    /// dims.n_out`, normalised by `weight` (an RMS norm over `d` values) when
    /// given, copied otherwise.
    ///
    /// # Safety
    ///
    /// `x` must address all rows of the step (`n_rows * d` f32 values), `y`
    /// `n_out * d`, `weight` `d` values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather_rows(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        x: DevicePtr,
        weight: Option<DevicePtr>,
        y: DevicePtr,
        d: u32,
        eps: f32,
    ) -> Result<()> {
        if dims.n_out == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(step)
            .ptr(x)
            .ptr(weight.unwrap_or(DevicePtr::NULL))
            .ptr(y)
            .u32(d)
            .f32(eps);
        // SAFETY: the argument list matches ares_gather_rows.
        unsafe { self.gather_rows.launch(stream, [dims.n_out, 1, 1], [256, 1, 1], 0, args) }
    }

    /// The rotary table of a step: `(cos, sin)` per row and pair, `n_rows *
    /// rope.dim / 2` pairs of f32, computed in f64. Built once per step; every
    /// attention layer reads it.
    ///
    /// # Safety
    ///
    /// `table` must address `n_rows * rope.dim / 2 * 8` bytes.
    pub unsafe fn rope_table(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        table: DevicePtr,
        rope: RopeParams,
    ) -> Result<()> {
        let half = rope.dim / 2;
        let total: u32 = rope.sections.iter().sum();
        if rope.dim == 0 || !rope.dim.is_multiple_of(2) || half > 1024 || total == 0 {
            return Err(invalid("rope", format!("{rope:?}")));
        }
        if dims.n_rows == 0 {
            return Ok(());
        }
        let [s0, s1, s2, s3] = rope.sections;
        let args = Args::new()
            .ptr(step)
            .ptr(table)
            .u32(rope.dim)
            .f32(rope.freq_base)
            .u32(s0)
            .u32(s1)
            .u32(s2)
            .u32(s3);
        // SAFETY: the argument list matches ares_rope_table.
        unsafe {
            self.rope_table.launch(
                stream,
                [dims.n_rows, 1, 1],
                [half.next_multiple_of(32), 1, 1],
                0,
                args,
            )
        }
    }

    /// Prepares attention's inputs for the rows of a step. From `qg` (per row
    /// `n_head` heads of query then gate, `2 * head_dim` each) and `k` (`n_kv_head`
    /// heads): `q_out` gets the queries with the per-head RMS norm by `q_norm`
    /// and the rotation, `gate_out` the gates (`None` to leave them in `qg`),
    /// `k_out` the keys with `k_norm` and the rotation. `v` is not touched.
    ///
    /// # Safety
    ///
    /// The buffers must hold `n_rows` rows of their shapes (f32), `table` the
    /// rotary table of the step, all on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_prep(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: AttentionShape,
        rope_dim: u32,
        eps: f32,
        qg: DevicePtr,
        k: DevicePtr,
        q_norm: DevicePtr,
        k_norm: DevicePtr,
        table: DevicePtr,
        q_out: DevicePtr,
        gate_out: Option<DevicePtr>,
        k_out: DevicePtr,
    ) -> Result<()> {
        let hd = shape.head_dim;
        if hd == 0
            || !hd.is_multiple_of(32)
            || hd > 256
            || rope_dim > hd
            || !rope_dim.is_multiple_of(2)
        {
            return Err(invalid(
                "attention prep",
                format!("{shape:?} with {rope_dim} rotated values"),
            ));
        }
        for p in [qg, k, q_out, k_out] {
            check_align("attention prep", p, 16)?;
        }
        if dims.n_rows == 0 {
            return Ok(());
        }
        let items = u64::from(dims.n_rows) * u64::from(shape.n_head + shape.n_kv_head);
        let args = Args::new()
            .ptr(step)
            .ptr(qg)
            .ptr(k)
            .ptr(q_norm)
            .ptr(k_norm)
            .ptr(table)
            .ptr(q_out)
            .ptr(gate_out.unwrap_or(DevicePtr::NULL))
            .ptr(k_out)
            .u32(shape.n_head)
            .u32(shape.n_kv_head)
            .u32(hd)
            .u32(rope_dim)
            .f32(eps);
        let blocks = u32::try_from(items.div_ceil(8))
            .map_err(|_| invalid("attention prep", "too many heads".into()))?;
        // SAFETY: the argument list matches ares_attn_prep.
        unsafe { self.attn_prep.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args) }
    }

    /// The causal convolution with SiLU over the rows of the step, from the
    /// committed windows of the sequences' slots (never written here), with the
    /// split into `q_out` (`[rows][key_dim]`) and `kv_out`
    /// (`[rows][key_dim + value_dim]`: keys, then values) and, when
    /// `l2_eps` is `Some`, the L2 normalisation of every query and key head.
    ///
    /// `taps` is `ssm_conv1d.weight` (`[conv_dim][K]` f32), `conv_state` the
    /// windows (`[slot][channel][K-1]`, `conv_slot_stride` f32 values per slot).
    ///
    /// # Safety
    ///
    /// All buffers on the stream's device and big enough for the step; `qkv`
    /// `n_rows * conv_dim` f32 values, 16-byte aligned.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_window(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: RecurrentShape,
        l2_eps: Option<f32>,
        qkv: DevicePtr,
        taps: DevicePtr,
        conv_state: DevicePtr,
        conv_slot_stride: u64,
        q_out: DevicePtr,
        kv_out: DevicePtr,
    ) -> Result<()> {
        shape.check()?;
        for p in [qkv, q_out, kv_out] {
            check_align("convolution", p, 16)?;
        }
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let tiles = shape.conv_dim() / 128;
        // Enough chunks of rows to occupy the device when a segment is long, few
        // enough that a short one is a single warp-iteration.
        let chunk_rows = 64u32;
        let warps = 4;
        let args = Args::new()
            .ptr(step)
            .ptr(qkv)
            .ptr(taps)
            .ptr(conv_state)
            .u64(conv_slot_stride)
            .ptr(q_out)
            .ptr(kv_out)
            .u32(shape.conv_dim())
            .u32(shape.key_dim())
            .u32(shape.conv_kernel)
            .u32(chunk_rows)
            .f32(l2_eps.unwrap_or(-1.0));
        // SAFETY: the argument list matches ares_conv_window.
        unsafe {
            self.conv_window.launch(
                stream,
                [tiles.div_ceil(warps), dims.max_seg_rows.div_ceil(chunk_rows), dims.n_segs],
                [warps * 32, 1, 1],
                0,
                args,
            )
        }
    }

    /// The gated delta rule over the rows of the step from the committed
    /// matrices of the sequences' slots: `out` (`[rows][n_v_heads * head_dim]`)
    /// gets the outputs. The matrices are written only for segments flagged
    /// direct (all rows certain to be accepted); for the others the state is
    /// untouched and [`LayerKernels::delta_commit`] advances it later from what
    /// the window kept.
    ///
    /// Inputs: `q` (`[rows][key_dim]`) and `kv` (`[rows][key_dim + value_dim]`)
    /// from [`LayerKernels::conv_window`]; `alpha` and `beta_in` (`[rows][n_v_heads]`)
    /// the raw outputs of the `ssm_alpha` and `ssm_beta` products; `dt_bias` and
    /// `ssm_a` (`[n_v_heads]`). The decay `exp(softplus(alpha + dt_bias) * a)`
    /// and `sigmoid(beta_in)` are computed inside.
    ///
    /// # Safety
    ///
    /// All buffers on the stream's device and big enough; `state` addresses
    /// `n_slots * state_slot_stride` values of `format`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_window(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: RecurrentShape,
        format: StateFormat,
        q: DevicePtr,
        kv: DevicePtr,
        alpha: DevicePtr,
        beta_in: DevicePtr,
        dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        state: DevicePtr,
        state_slot_stride: u64,
        out: DevicePtr,
    ) -> Result<()> {
        shape.check()?;
        for p in [q, kv, out, state] {
            check_align("delta rule", p, 16)?;
        }
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let k = &self.delta[self.delta_selected].1;
        let function = match format {
            StateFormat::F32 => &k.window_f32,
            StateFormat::F16 => &k.window_f16,
        };
        let args = Args::new()
            .ptr(step)
            .ptr(q)
            .ptr(kv)
            .ptr(alpha)
            .ptr(beta_in)
            .ptr(dt_bias)
            .ptr(ssm_a)
            .ptr(state)
            .u64(state_slot_stride)
            .ptr(out)
            .u32(shape.n_k_heads)
            .u32(shape.n_v_heads);
        let blocks = shape.n_v_heads * (shape.head_dim / k.rows_per_block);
        // SAFETY: the argument list matches ares_delta_window_*.
        unsafe { function.launch(stream, [blocks, dims.n_segs, 1], [k.threads, 1, 1], 0, args) }
    }

    /// The design of the llama.cpp fork's delta-rule kernel (4 columns of the
    /// matrix per warp, 8 lanes per column, `k` and `q` read from global memory at
    /// every row, no staging), ported to this engine's inputs: the baseline
    /// that `examples/layers_bench.rs` times the delta rule against. f32 state
    /// only; writes the state when `write_state`.
    ///
    /// # Safety
    ///
    /// As [`LayerKernels::delta_window`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_window_fork_design(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: RecurrentShape,
        write_state: bool,
        q: DevicePtr,
        kv: DevicePtr,
        alpha: DevicePtr,
        beta_in: DevicePtr,
        dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        state: DevicePtr,
        state_slot_stride: u64,
        out: DevicePtr,
    ) -> Result<()> {
        shape.check()?;
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(step)
            .ptr(q)
            .ptr(kv)
            .ptr(alpha)
            .ptr(beta_in)
            .ptr(dt_bias)
            .ptr(ssm_a)
            .ptr(state)
            .u64(state_slot_stride)
            .ptr(out)
            .u32(shape.n_k_heads)
            .u32(shape.n_v_heads)
            .u32(u32::from(write_state));
        // SAFETY: the argument list matches ares_delta_window_fork_f32.
        unsafe {
            self.delta_fork.launch(stream, [shape.n_v_heads, dims.n_segs, 8], [32, 4, 1], 0, args)
        }
    }

    /// Advances the delta-rule matrices of the segments that are not direct over
    /// their first `accepted` rows, from the kept `kv`, `alpha` and `beta_in`
    /// of the window (the arguments of [`LayerKernels::delta_window`]).
    ///
    /// # Safety
    ///
    /// As [`LayerKernels::delta_window`]; the kept buffers must still hold the
    /// window's values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_commit(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: RecurrentShape,
        format: StateFormat,
        kv: DevicePtr,
        alpha: DevicePtr,
        beta_in: DevicePtr,
        dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        state: DevicePtr,
        state_slot_stride: u64,
    ) -> Result<()> {
        shape.check()?;
        check_align("delta commit", state, 16)?;
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let k = &self.delta[self.delta_selected].1;
        let function = match format {
            StateFormat::F32 => &k.commit_f32,
            StateFormat::F16 => &k.commit_f16,
        };
        let args = Args::new()
            .ptr(step)
            .ptr(kv)
            .ptr(alpha)
            .ptr(beta_in)
            .ptr(dt_bias)
            .ptr(ssm_a)
            .ptr(state)
            .u64(state_slot_stride)
            .u32(shape.n_k_heads)
            .u32(shape.n_v_heads);
        let blocks = shape.n_v_heads * (shape.head_dim / k.rows_per_block);
        // SAFETY: the argument list matches ares_delta_commit_*.
        unsafe { function.launch(stream, [blocks, dims.n_segs, 1], [k.threads, 1, 1], 0, args) }
    }

    /// Advances the convolution windows over the accepted rows. `which`
    /// selects the segments: [`CommitSet::Deferred`] those that are not direct
    /// (with `accepted` rows, at the end of the step), [`CommitSet::Direct`] the
    /// direct ones (with all their rows, right after their window, while the
    /// raw inputs `qkv` are still there), [`CommitSet::All`] both.
    ///
    /// # Safety
    ///
    /// `qkv` must still hold the raw projections of the window.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv_commit(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: RecurrentShape,
        which: CommitSet,
        qkv: DevicePtr,
        conv_state: DevicePtr,
        conv_slot_stride: u64,
    ) -> Result<()> {
        shape.check()?;
        if dims.n_segs == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(step)
            .ptr(qkv)
            .ptr(conv_state)
            .u64(conv_slot_stride)
            .u32(shape.conv_dim())
            .u32(shape.conv_kernel)
            .u32(which as u32);
        // SAFETY: the argument list matches ares_conv_commit.
        unsafe {
            self.conv_commit.launch(
                stream,
                [shape.conv_dim().div_ceil(128), dims.n_segs, 1],
                [128, 1, 1],
                0,
                args,
            )
        }
    }

    /// The commit of a recurrent block: [`LayerKernels::conv_commit`] for the
    /// deferred segments and [`LayerKernels::delta_commit`]. Segments that were
    /// direct are done by their window (and `conv_commit(Direct)`).
    ///
    /// # Safety
    ///
    /// As the two it calls.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn commit(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: RecurrentShape,
        format: StateFormat,
        qkv: DevicePtr,
        kv: DevicePtr,
        alpha: DevicePtr,
        beta_in: DevicePtr,
        dt_bias: DevicePtr,
        ssm_a: DevicePtr,
        conv_state: DevicePtr,
        conv_slot_stride: u64,
        state: DevicePtr,
        state_slot_stride: u64,
    ) -> Result<()> {
        // SAFETY: the callers' guarantees are those of the two functions.
        unsafe {
            self.conv_commit(
                stream,
                step,
                dims,
                shape,
                CommitSet::Deferred,
                qkv,
                conv_state,
                conv_slot_stride,
            )?;
            self.delta_commit(
                stream,
                step,
                dims,
                shape,
                format,
                kv,
                alpha,
                beta_in,
                dt_bias,
                ssm_a,
                state,
                state_slot_stride,
            )
        }
    }

    /// The gated norm of the recurrent block followed by the quantization of
    /// its result for the output projection: for every of `items` (row, head)
    /// pairs of `width` values, `rms_norm(o) * weight * gate(z)`, written as
    /// quantized activations (the layout of `Kernels::quantize_activations`, over
    /// `items * width` values; [`quantized_activation_bytes`] bytes, 16-byte
    /// aligned) that every weight kernel takes. `width` is 64, 128 or 256.
    ///
    /// # Safety
    ///
    /// `o` and `z` must address `items * width` f32 values (16-byte aligned),
    /// `weight` `width` values, `act` the bytes above.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gated_norm_quant(
        &self,
        stream: &Stream,
        gate: OutputGate,
        o: DevicePtr,
        z: DevicePtr,
        weight: DevicePtr,
        act: DevicePtr,
        items: u32,
        width: u32,
        eps: f32,
    ) -> Result<()> {
        if !matches!(width, 64 | 128 | 256) {
            return Err(invalid("gated norm", format!("width {width} (64, 128 or 256)")));
        }
        for p in [o, z, weight, act] {
            check_align("gated norm", p, 16)?;
        }
        if items == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(o)
            .ptr(z)
            .ptr(weight)
            .ptr(act)
            .u32(items)
            .u32(width)
            .f32(eps)
            .u32(gate as u32);
        // SAFETY: the argument list matches ares_gated_norm_quant.
        unsafe {
            self.gated_norm_quant.launch(stream, [items.div_ceil(8), 1, 1], [256, 1, 1], 0, args)
        }
    }

    /// `silu(gate) * up` quantized for the down projection, over `n_values`
    /// values (a multiple of 32): `act` gets the layout of
    /// `Kernels::quantize_activations`.
    ///
    /// # Safety
    ///
    /// `gate` and `up` must address `n_values` f32 values (16-byte aligned),
    /// `act` [`quantized_activation_bytes`] bytes.
    pub unsafe fn swiglu_quant(
        &self,
        stream: &Stream,
        gate: DevicePtr,
        up: DevicePtr,
        act: DevicePtr,
        n_values: u32,
    ) -> Result<()> {
        if n_values == 0 || !n_values.is_multiple_of(32) {
            return Err(invalid("swiglu", format!("{n_values} values (a multiple of 32)")));
        }
        for p in [gate, up, act] {
            check_align("swiglu", p, 16)?;
        }
        let args = Args::new().ptr(gate).ptr(up).ptr(act).u32(n_values);
        let threads = n_values / 4;
        // SAFETY: the argument list matches ares_swiglu_quant.
        unsafe {
            self.swiglu_quant.launch(stream, [threads.div_ceil(256), 1, 1], [256, 1, 1], 0, args)
        }
    }
}

/// float4 chunks a thread of `ares_swiglu_f16` keeps in registers
/// (`ARES_SWIGLU_F16_RESIDENT`), and heads a warp of `ares_gated_norm_f16`
/// keeps (`ARES_GATED_NORM_F16_HEADS`).
const SWIGLU_F16_RESIDENT: u32 = 6;
const GATED_NORM_F16_HEADS: u32 = 2;

/// The fused kernels that write f16 activations: what
/// [`LayerKernels::gated_norm_quant`] and [`LayerKernels::swiglu_quant`] are
/// for a product that reads 8-bit activations, for one that reads f16 (a
/// matrix in its device layout). They write what a conversion of the same
/// values would (`Kernels::prepare_activations` for `ActivationForm::F16`,
/// `Kernels::activation_bytes` bytes: `kernels/layer_act_f16.cuh`), in the
/// launch that computes them.
impl LayerKernels {
    /// The widest row [`LayerKernels::swiglu_f16`] takes: a block of 1024
    /// threads keeps the row in registers.
    pub const SWIGLU_F16_MAX_WIDTH: u32 = 4 * SWIGLU_F16_RESIDENT * 1024;

    /// The most heads of a row [`LayerKernels::gated_norm_f16`] takes: the 32
    /// warps of a block keep the row in registers.
    pub const GATED_NORM_F16_MAX_HEADS: u32 = GATED_NORM_F16_HEADS * 32;

    /// The gated norm of the recurrent block as f16 activations for the
    /// output projection: for each of `n_rows` rows its `n_heads` heads (at
    /// most [`LayerKernels::GATED_NORM_F16_MAX_HEADS`]) of `width` values,
    /// `rms_norm(o) * weight * gate(z)`, written as `n_rows` rows of
    /// `n_heads * width` f16 values. `width` is 64, 128 or 256.
    ///
    /// # Safety
    ///
    /// `o` and `z` must address `n_rows * n_heads * width` f32 values,
    /// `weight` `width` values and `act` the f16 activations of `n_rows` rows
    /// of `n_heads * width` values, all 16-byte aligned, on the stream's
    /// device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gated_norm_f16(
        &self,
        stream: &Stream,
        gate: OutputGate,
        o: DevicePtr,
        z: DevicePtr,
        weight: DevicePtr,
        act: DevicePtr,
        n_rows: u32,
        n_heads: u32,
        width: u32,
        eps: f32,
    ) -> Result<()> {
        if !matches!(width, 64 | 128 | 256) {
            return Err(invalid("gated norm", format!("width {width} (64, 128 or 256)")));
        }
        if n_heads > Self::GATED_NORM_F16_MAX_HEADS {
            return Err(invalid(
                "gated norm",
                format!("{n_heads} heads (at most {})", Self::GATED_NORM_F16_MAX_HEADS),
            ));
        }
        for p in [o, z, weight, act] {
            check_align("gated norm", p, 16)?;
        }
        if n_rows == 0 || n_heads == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(o)
            .ptr(z)
            .ptr(weight)
            .ptr(act)
            .u32(n_rows)
            .u32(n_heads)
            .u32(width)
            .f32(eps)
            .u32(gate as u32);
        // A warp per head, up to the 32 warps of a block; with more heads the
        // warps keep several each.
        let threads = 32 * n_heads.min(32);
        // SAFETY: the argument list matches ares_gated_norm_f16.
        unsafe { self.gated_norm_f16.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// `silu(gate) * up` as f16 activations for the down projection: `n_rows`
    /// rows of `n_in` values (a multiple of 32, at most
    /// [`LayerKernels::SWIGLU_F16_MAX_WIDTH`]).
    ///
    /// # Safety
    ///
    /// `gate` and `up` must address `n_rows * n_in` f32 values and `act` the
    /// f16 activations of `n_rows` rows of `n_in` values, all 16-byte aligned,
    /// on the stream's device.
    pub unsafe fn swiglu_f16(
        &self,
        stream: &Stream,
        gate: DevicePtr,
        up: DevicePtr,
        act: DevicePtr,
        n_rows: u32,
        n_in: u32,
    ) -> Result<()> {
        if n_in == 0 || !n_in.is_multiple_of(32) || n_in > Self::SWIGLU_F16_MAX_WIDTH {
            return Err(invalid(
                "swiglu",
                format!(
                    "rows of {n_in} values (a multiple of 32 up to {})",
                    Self::SWIGLU_F16_MAX_WIDTH
                ),
            ));
        }
        for p in [gate, up, act] {
            check_align("swiglu", p, 16)?;
        }
        if n_rows == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(gate).ptr(up).ptr(act).u32(n_rows).u32(n_in);
        // A thread per four values, up to the 1024 of a block, before a
        // thread takes more: it waits for each of its loads.
        let threads = (n_in / 4).next_multiple_of(32).min(1024);
        // SAFETY: the argument list matches ares_swiglu_f16.
        unsafe { self.swiglu_f16.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// Puts `n_rows` rows of `width` f32 values at `src` into rows of
    /// `dst_width` values at `dst`, from column `at` on: the place of a
    /// result that was computed as some columns of a wider one (the logits of
    /// a head that lies on the device in parts).
    ///
    /// # Safety
    ///
    /// `src` must address `n_rows * width` f32 values and `dst`
    /// `n_rows * dst_width`, on the stream's device, not overlapping.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn place_rows(
        &self,
        stream: &Stream,
        src: DevicePtr,
        dst: DevicePtr,
        n_rows: u32,
        width: u32,
        dst_width: u32,
        at: u32,
    ) -> Result<()> {
        let fits = at.checked_add(width).is_some_and(|end| end <= dst_width);
        let blocks = (u64::from(n_rows) * u64::from(width)).div_ceil(256);
        if !fits || blocks > u64::from(u32::MAX) {
            return Err(invalid(
                "place rows",
                format!("{n_rows} rows of {width} values at column {at} of {dst_width}"),
            ));
        }
        if blocks == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(src).ptr(dst).u32(n_rows).u32(width).u32(dst_width).u32(at);
        // SAFETY: the argument list matches ares_place_rows.
        unsafe { self.place_rows.launch(stream, [blocks as u32, 1, 1], [256, 1, 1], 0, args) }
    }
}

/// What a merge of a tensor split does with the residual stream after it
/// added the devices' partial results to it (`kernels/layer_merge.cuh`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MergeNorm {
    /// Nothing.
    None,
    /// The RMS norm with `weight`, as f32 rows at `out`.
    Rows { weight: DevicePtr, out: DevicePtr },
    /// The RMS norm with `weight`, quantized into `act` for the weight
    /// kernels: the layout of `Kernels::quantize_activations` over all rows
    /// ([`quantized_activation_bytes`] bytes).
    Quantized { weight: DevicePtr, act: DevicePtr },
    /// The RMS norm with `weight`, as f16 activations at `act` for the
    /// products that read those (matrices in their device layout): the layout
    /// of `Kernels::prepare_activations` for `ActivationForm::F16`
    /// (`Kernels::activation_bytes` bytes).
    F16 { weight: DevicePtr, act: DevicePtr },
    /// Both: quantized at `quantized` and as f16 at `f16`, for a half block
    /// whose matrices read one or the other.
    Both { weight: DevicePtr, quantized: DevicePtr, f16: DevicePtr },
}

impl MergeNorm {
    /// The kernels' arguments: weight, rows (or f16 activations), quantized
    /// activations, and the mode with the bit for the wide wire form.
    fn args(self, wire: Wire) -> (DevicePtr, DevicePtr, DevicePtr, u32) {
        let wide = match wire {
            Wire::F16 => 0,
            Wire::F32 => 256,
        };
        match self {
            Self::None => (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, wide),
            Self::Rows { weight, out } => (weight, out, DevicePtr::NULL, 1 | wide),
            Self::Quantized { weight, act } => (weight, DevicePtr::NULL, act, 2 | wide),
            // The f16 activations travel in the argument of the rows, which
            // these modes do not write.
            Self::F16 { weight, act } => (weight, act, DevicePtr::NULL, 3 | wide),
            Self::Both { weight, quantized, f16 } => (weight, f16, quantized, 4 | wide),
        }
    }
}

/// How a partial result crosses between the devices of a tensor split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    /// As f16, each row with a power of two that keeps it in range: half the
    /// bytes, and every value rounded to 11 bits.
    F16,
    /// As f32: the exact sum.
    F32,
}

/// The bytes `n_rows` rows of `d` values take on the wire between the devices
/// of a tensor split: the rows and one f32 factor per row.
pub fn partial_wire_bytes(wire: Wire, n_rows: u32, d: u32) -> u64 {
    let values = u64::from(n_rows) * u64::from(d);
    let rows = match wire {
        Wire::F16 => (values * 2).next_multiple_of(4),
        Wire::F32 => values * 4,
    };
    rows + u64::from(n_rows) * 4
}

/// The widest row the merge kernels take.
pub const MERGE_MAX_WIDTH: u32 = 16 * MERGE_RESIDENT * 1024;

/// float4 chunks a thread of the merge kernels keeps in registers
/// (`ARES_MERGE_RESIDENT`).
const MERGE_RESIDENT: u32 = 6;

/// The threads per row of the merge kernels for rows of `d` values.
fn merge_threads(what: &'static str, d: u32) -> Result<u32> {
    if d == 0 || !d.is_multiple_of(32) || d > MERGE_MAX_WIDTH {
        return Err(invalid(
            what,
            format!("rows of {d} values (a multiple of 32 up to {MERGE_MAX_WIDTH})"),
        ));
    }
    Ok((d / 4).div_ceil(MERGE_RESIDENT).next_multiple_of(32).max(256))
}

/// The merges of a tensor split.
impl LayerKernels {
    /// Writes the wire form `form` of `n_rows` rows of `d` f32 values at `p`
    /// to `wire` ([`partial_wire_bytes`] bytes).
    ///
    /// # Safety
    ///
    /// `p` must address `n_rows * d` f32 values and `wire` the bytes above,
    /// both 16-byte aligned, on the stream's device.
    pub unsafe fn partial_to_wire(
        &self,
        stream: &Stream,
        form: Wire,
        p: DevicePtr,
        wire: DevicePtr,
        n_rows: u32,
        d: u32,
    ) -> Result<()> {
        let threads = merge_threads("partial to wire", d)?;
        check_align("partial to wire", p, 16)?;
        check_align("partial to wire", wire, 16)?;
        if n_rows == 0 {
            return Ok(());
        }
        let args =
            Args::new().ptr(p).ptr(wire).u32(n_rows).u32(d).u32(u32::from(form == Wire::F32));
        // SAFETY: the argument list matches ares_partial_to_wire.
        unsafe { self.partial_to_wire.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// `x += ` the sum of all devices' partials of `n_rows` rows of `d`
    /// values, then what `norm` says. The partials are in wire form `form`: this
    /// device's at `own`, the `n_peers` others' in device order at `peers`,
    /// `peer_stride` bytes apart; `own_rank` is this device's place in that
    /// order. Every device that merges the same partials gets the same bits.
    ///
    /// # Safety
    ///
    /// `x` must address `n_rows * d` f32 values, `own` and every peer region
    /// [`partial_wire_bytes`] bytes, and the buffers of `norm` what it says,
    /// all 16-byte aligned, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn merge_wire(
        &self,
        stream: &Stream,
        form: Wire,
        x: DevicePtr,
        own: DevicePtr,
        peers: DevicePtr,
        peer_stride: u64,
        n_peers: u32,
        own_rank: u32,
        n_rows: u32,
        d: u32,
        norm: MergeNorm,
        eps: f32,
    ) -> Result<()> {
        let threads = merge_threads("merge", d)?;
        if n_peers >= 8 || own_rank > n_peers || !peer_stride.is_multiple_of(16) {
            return Err(invalid(
                "merge",
                format!(
                    "device {own_rank} of {} with regions {peer_stride} bytes apart",
                    n_peers + 1
                ),
            ));
        }
        let (weight, xn, act, mode) = norm.args(form);
        for ptr in [x, own, peers, weight, xn, act] {
            check_align("merge", ptr, 16)?;
        }
        if n_rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(x)
            .ptr(own)
            .ptr(peers)
            .u64(peer_stride)
            .u32(n_peers)
            .u32(own_rank)
            .u32(n_rows)
            .u32(d)
            .ptr(weight)
            .ptr(xn)
            .ptr(act)
            .f32(eps)
            .u32(mode);
        // SAFETY: the argument list matches ares_merge_wire.
        unsafe { self.merge_wire.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// `x += p + ` the peer's `p`, both rounded as the wire form `form` rounds,
    /// then what `norm` says: the merge of two devices with peer access in one
    /// kernel per device, for up to [`PUSH_MAX_ROWS`] rows. `slot` is this
    /// device's [`Mesh::push`](crate::Mesh::push) of the sum and `own_first`
    /// whether this device is the first of the two.
    ///
    /// # Safety
    ///
    /// `p` and `x` must address `n_rows * d` f32 values and the buffers of
    /// `norm` what it says, all 16-byte aligned, on the stream's device. The
    /// peer must enqueue the same merge with its push of the same number.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn merge_push(
        &self,
        stream: &Stream,
        form: Wire,
        p: DevicePtr,
        x: DevicePtr,
        slot: PushSlot,
        own_first: bool,
        n_rows: u32,
        d: u32,
        norm: MergeNorm,
        eps: f32,
    ) -> Result<()> {
        let threads = merge_threads("merge push", d)?;
        if n_rows > PUSH_MAX_ROWS || partial_wire_bytes(form, n_rows, d) > slot.bytes {
            return Err(invalid(
                "merge push",
                format!(
                    "{n_rows} rows of {d} values; a push takes {PUSH_MAX_ROWS} rows and {} bytes",
                    slot.bytes
                ),
            ));
        }
        let (weight, xn, act, mode) = norm.args(form);
        for ptr in [p, x, weight, xn, act] {
            check_align("merge push", ptr, 16)?;
        }
        if n_rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(p)
            .ptr(x)
            .ptr(slot.peer_recv)
            .ptr(slot.peer_flags)
            .ptr(slot.own_recv)
            .ptr(slot.own_flags)
            .u32(PUSH_MAX_ROWS)
            .u32(slot.token)
            .u32(u32::from(own_first))
            .u32(n_rows)
            .u32(d)
            .ptr(weight)
            .ptr(xn)
            .ptr(act)
            .f32(eps)
            .u32(mode);
        // SAFETY: the argument list matches ares_merge_push; the slot's
        // pointers are the mesh's buffers on this device and its peer.
        unsafe { self.merge_push.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args) }
    }
}

/// Which segments a convolution commit handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CommitSet {
    /// Segments that are not direct, with their accepted rows.
    Deferred = 0,
    /// Direct segments, with all their rows.
    Direct = 1,
    /// Both.
    All = 2,
}
