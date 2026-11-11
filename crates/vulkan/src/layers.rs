//! The layer kernels: everything a model needs between the matrix products and
//! attention, from `shaders/layer_*.comp`. The counterpart of `ares_cuda::LayerKernels`,
//! with the same step record (byte for byte), the same arguments and the same semantics,
//! so that a runtime drives either backend with the same code.
//!
//! They take their sizes as arguments and read what varies from step to step
//! (which rows belong to which sequence, their tokens and positions, which
//! slot of recurrent state a sequence owns) from the **step record**
//! ([`StepRecord`]): one small buffer that the conductor uploads once per step. A step
//! carries rows of any mix of sequences, one [`Segment`] of consecutive rows per sequence.
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
//! # Against `ares-cuda`
//!
//! - The fused kernels ([`LayerKernels::gated_norm_quant`], [`LayerKernels::swiglu_quant`])
//!   write the quantized activations in this crate's format
//!   ([`crate::Kernels::quantize_activations`]: codes and pairs) and, behind them, the f32
//!   values they computed ([`quantized_activation_copy_offset`]). CUDA's own format carries
//!   that copy inside; here the weight kernels of the float types and the conversion to f16
//!   for the tensor cores are pointed at it.
//! - The rotary table is computed in 64-bit integers, not in f64 (GLSL has no sine of a
//!   double and a device need not have doubles): see `shaders/layer_rope_table.comp`. It
//!   needs `freq_base > 1`.
//! - The delta rule stages 8 rows of a segment at a time, not 16, so that its shared memory
//!   fits the 16 KiB every Vulkan device has.
//! - There is no `delta_window_fork_design` (a benchmark baseline of the CUDA layer) and no
//!   `merge_push` (a kernel that writes into a peer's memory: Vulkan has no peer memory).
//! - The pipelines are made when a kernel is first launched, or by
//!   [`LayerKernels::prepare`].
//!
//! All launches are `unsafe`: the device trusts the addresses it is given.

use std::sync::Arc;

use ares_core::GgmlType;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::{Kernels, grid_1d, parallel_map};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Lazy};
use crate::stream::Stream;
use crate::weights::image;

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
    /// Rows of the step.
    pub n_rows: u32,
    /// Segments (sequences) of the step.
    pub n_segs: u32,
    /// The length of the longest segment.
    pub max_seg_rows: u32,
    /// Rows whose logits are wanted.
    pub n_out: u32,
}

/// The step record: header, segments, tokens, positions and output rows in the
/// layout `shaders/layer_common.glsl` documents (the CUDA layer's, byte for byte). Upload [`StepRecord::as_bytes`]
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
    /// Values of the queries (and of the keys) of a row.
    pub fn key_dim(&self) -> u32 {
        self.n_k_heads * self.head_dim
    }
    /// Values of the values of a row.
    pub fn value_dim(&self) -> u32 {
        self.n_v_heads * self.head_dim
    }
    /// Channels of the convolution: queries, keys and values.
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
    /// As f32.
    F32,
    /// As f16: half the bytes, every value rounded when a launch stores the state.
    F16,
}

impl StateFormat {
    /// Bytes per value.
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
    /// The base of the frequencies: pair `p` turns by `freq_base^(-2p / dim)` per position.
    pub freq_base: f32,
    /// The interleaved sections of the pairs over (time, height, width, extra).
    pub sections: [u32; 4],
}

/// The shapes of attention's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape {
    /// Query heads.
    pub n_head: u32,
    /// Key and value heads.
    pub n_kv_head: u32,
    /// The width of a head.
    pub head_dim: u32,
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

/// One compiled shape of the delta-rule kernels.
#[derive(Clone, Debug)]
struct DeltaKernels {
    /// Rows of the matrix per workgroup: the grid has `head_dim / rows_per_group`
    /// workgroups per head.
    rows_per_group: u32,
    window_f32: Arc<Lazy>,
    window_f16: Arc<Lazy>,
    commit_f32: Arc<Lazy>,
    commit_f16: Arc<Lazy>,
}

/// The compiled shapes of the delta rule: name, shader suffix, rows of the matrix per
/// workgroup. The first is the default: two invocations per row of the matrix (64 values
/// each in registers), 64 rows per workgroup, 8 rows of the segment staged at a time. The
/// others are for tuning (`examples/vk_layer_bench.rs`); `lpr1_rb64` uses no subgroup
/// operation at all.
const DELTA_VARIANTS: [(&str, &str, u32); 6] = [
    ("d128", "", 64),
    ("lpr1_rb64", "_lpr1_rb64", 64),
    ("lpr4_rb64", "_lpr4_rb64", 64),
    ("lpr2_rb32", "_lpr2_rb32", 32),
    ("lpr4_rb32", "_lpr4_rb32", 32),
    ("lpr2_rb64_tc16", "_lpr2_rb64_tc16", 64),
];

/// The layer kernels on one device.
#[derive(Clone, Debug)]
pub struct LayerKernels {
    device: Device,
    /// The weight kernels of the device: the embedding gather reads their codebooks.
    kernels: Kernels,
    embed_gather: Arc<Lazy>,
    head_norm: Arc<Lazy>,
    gather_rows: Arc<Lazy>,
    rope_table: Arc<Lazy>,
    attn_prep: Arc<Lazy>,
    gated_norm_quant: Arc<Lazy>,
    swiglu_quant: Arc<Lazy>,
    partial_to_wire: Arc<Lazy>,
    merge_wire: Arc<Lazy>,
    conv_window: Arc<Lazy>,
    conv_commit: Arc<Lazy>,
    delta: Vec<(&'static str, DeltaKernels)>,
    delta_selected: usize,
}

fn check_align(what: &'static str, ptr: DevicePtr, align: u64) -> Result<()> {
    crate::weights::check_alignment(what, ptr, align)
}

fn invalid(what: &'static str, detail: String) -> Error {
    Error::Invalid { what, detail }
}

/// The byte offset, in the output of a fused kernel over `n_values` values, of the f32
/// values it computed: behind the int8 codes and the `(scale, sum)` pairs, which are the
/// [`crate::Kernels::quantized_activation_bytes`] the weight kernels read.
pub fn quantized_activation_copy_offset(n_values: u64) -> u64 {
    let pairs_end = n_values.next_multiple_of(16) + n_values.div_ceil(32) * 8;
    pairs_end.next_multiple_of(16)
}

/// The bytes of the quantized activations the fused kernels write for `n_values` values:
/// the layout of [`crate::Kernels::quantize_activations`], then (at the next multiple of 16
/// bytes, [`quantized_activation_copy_offset`]) the values in f32.
pub fn quantized_activation_bytes(n_values: u64) -> u64 {
    quantized_activation_copy_offset(n_values) + n_values * 4
}

impl LayerKernels {
    /// Loads the layer kernels on the device of `kernels`, whose codebooks the embedding
    /// gather decodes the i-quants with. No pipeline is made yet: a kernel's is made when it is
    /// first launched, or by [`LayerKernels::prepare`].
    ///
    /// The kernels give rows and heads to clusters of 8 and 32 invocations and reduce over
    /// them with clustered subgroup operations, and the embedding gather reads single bytes:
    /// what the quantized weight kernels need too ([`crate::Kernels::has_quantized_kernels`]).
    /// A device without it is refused here, with the reason.
    pub fn load(kernels: &Kernels) -> Result<Self> {
        let device = kernels.device();
        if let Some(reason) = &device.physical().quantized_unsupported {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!("the layer kernels need what the weight kernels need: {reason}"),
            });
        }
        let lazy = |name: &str| -> Result<Arc<Lazy>> { Ok(Lazy::new(device, image(name)?)) };
        let delta = DELTA_VARIANTS
            .iter()
            .map(|&(name, suffix, rows_per_group)| {
                Ok((
                    name,
                    DeltaKernels {
                        rows_per_group,
                        window_f32: lazy(&format!("layer_delta_window_f32{suffix}"))?,
                        window_f16: lazy(&format!("layer_delta_window_f16{suffix}"))?,
                        commit_f32: lazy(&format!("layer_delta_commit_f32{suffix}"))?,
                        commit_f16: lazy(&format!("layer_delta_commit_f16{suffix}"))?,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: device.clone(),
            kernels: kernels.clone(),
            embed_gather: lazy("layer_embed_gather")?,
            head_norm: lazy("layer_head_norm")?,
            gather_rows: lazy("layer_gather_rows")?,
            rope_table: lazy("layer_rope_table")?,
            attn_prep: lazy("layer_attn_prep")?,
            gated_norm_quant: lazy("layer_gated_norm_quant")?,
            swiglu_quant: lazy("layer_swiglu_quant")?,
            partial_to_wire: lazy("layer_partial_to_wire")?,
            merge_wire: lazy("layer_merge_wire")?,
            conv_window: lazy("layer_conv_window")?,
            conv_commit: lazy("layer_conv_commit")?,
            delta,
            delta_selected: 0,
        })
    }

    /// The device the kernels are on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Makes the pipelines of every kernel now, on all cores (the delta rule in its selected
    /// shape): the runtime calls this when it loads a model, so that no step pays for a
    /// compilation.
    pub fn prepare(&self) -> Result<()> {
        let d = &self.delta[self.delta_selected].1;
        let all = [
            &self.embed_gather,
            &self.head_norm,
            &self.gather_rows,
            &self.rope_table,
            &self.attn_prep,
            &self.gated_norm_quant,
            &self.swiglu_quant,
            &self.partial_to_wire,
            &self.merge_wire,
            &self.conv_window,
            &self.conv_commit,
            &d.window_f32,
            &d.window_f16,
            &d.commit_f32,
            &d.commit_f16,
        ];
        parallel_map(&all, |lazy| lazy.get().map(|_| ())).map(|_| ())
    }

    fn grid(&self, groups: u32) -> [u32; 3] {
        grid_1d(groups, self.device.physical().max_work_group_count[0])
    }

    /// A launch whose grid dimensions mean something to the kernel (segments, chunks of
    /// rows): refused when one of them is more than the device allows.
    unsafe fn launch_3d(
        &self,
        what: &'static str,
        function: &Function,
        stream: &Stream,
        grid: [u32; 3],
        args: Args,
    ) -> Result<()> {
        let limit = self.device.physical().max_work_group_count;
        if grid.iter().zip(limit).any(|(g, l)| *g > l) {
            return Err(invalid(
                what,
                format!("a grid of {grid:?} workgroups; the device allows {limit:?}"),
            ));
        }
        // SAFETY: the caller's guarantee.
        unsafe { function.launch(stream, grid, args) }
    }

    /// The names of the compiled shapes of the delta rule (for tuning); the
    /// first is the default.
    pub fn delta_variants() -> impl Iterator<Item = &'static str> {
        DELTA_VARIANTS.iter().map(|v| v.0)
    }

    /// Selects a compiled shape of the delta rule by name. `lpr2_rb64_tc16` needs 24.6 KiB
    /// of shared memory per workgroup, more than every device has.
    pub fn select_delta_variant(&mut self, name: &str) -> Result<()> {
        let at = self
            .delta
            .iter()
            .position(|(n, _)| *n == name)
            .ok_or_else(|| invalid("delta variant", format!("no variant '{name}'")))?;
        self.delta_selected = at;
        Ok(())
    }

    /// Embedding gather: `out[r] = table[token_r]` decoded to f32 for the
    /// `dims.n_rows` rows of the step. `table` is the tensor as GGUF stores it
    /// (`n_vocab` rows of `n_embd` values of type `ty`, any of the 28 weight types),
    /// aligned to 4 bytes. A token past `n_vocab` gives a row of zeros.
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
        if !ty.is_weight() {
            return Err(invalid("embedding gather", format!("{ty} is not a weight type")));
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
            .u64(row_bytes)
            .ptr(out)
            .ptr(self.kernels.quantized()?.tables())
            .u32(ty.id())
            .u32(n_vocab)
            .u32(n_embd);
        // SAFETY: the argument list matches layer_embed_gather.comp; ranges are the caller's
        // guarantee.
        unsafe { self.embed_gather.get()?.launch(stream, self.grid(dims.n_rows), args) }
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
        check_align("head norm input", x, 16)?;
        check_align("head norm output", y, 16)?;
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
        // SAFETY: the argument list matches layer_head_norm.comp.
        unsafe { self.head_norm.get()?.launch(stream, self.grid(rows.div_ceil(8)), args) }
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
        // SAFETY: the argument list matches layer_gather_rows.comp.
        unsafe { self.gather_rows.get()?.launch(stream, self.grid(dims.n_out), args) }
    }

    /// The rotary table of a step: `(cos, sin)` per row and pair, `n_rows *
    /// rope.dim / 2` pairs of f32, each the correctly rounded value of the exact
    /// angle to about 1e-10 (the kernel computes in 64-bit integers). Built once per
    /// step; every attention layer reads it. `rope.freq_base` must be above 1.
    ///
    /// # Safety
    ///
    /// `table` must address `n_rows * rope.dim / 2 * 8` bytes, 8-byte aligned.
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
        // The frequency of pair p is r^p; the kernel takes r as a fraction of 2^64.
        let r = f64::from(rope.freq_base).powf(-2.0 / f64::from(rope.dim));
        if !(r > 0.0 && r < 1.0) {
            return Err(invalid(
                "rope",
                format!("frequency base {} (it must be above 1)", rope.freq_base),
            ));
        }
        check_align("rope table", table, 8)?;
        if dims.n_rows == 0 {
            return Ok(());
        }
        let r64 = (r * 18_446_744_073_709_551_616.0) as u64;
        let [s0, s1, s2, s3] = rope.sections;
        let args =
            Args::new().ptr(step).ptr(table).u64(r64).u32(rope.dim).u32(s0).u32(s1).u32(s2).u32(s3);
        let pairs = u64::from(dims.n_rows) * u64::from(half);
        let groups = u32::try_from(pairs.div_ceil(256))
            .map_err(|_| invalid("rope", format!("{} rows of {half} pairs", dims.n_rows)))?;
        // SAFETY: the argument list matches layer_rope_table.comp.
        unsafe { self.rope_table.get()?.launch(stream, self.grid(groups), args) }
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
        let groups = u32::try_from(items.div_ceil(8))
            .map_err(|_| invalid("attention prep", "too many heads".into()))?;
        // SAFETY: the argument list matches layer_attn_prep.comp.
        unsafe { self.attn_prep.get()?.launch(stream, self.grid(groups), args) }
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
        // enough that a short one is a single trip of a cluster.
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
        let grid = [tiles.div_ceil(warps), dims.max_seg_rows.div_ceil(chunk_rows), dims.n_segs];
        // SAFETY: the argument list matches layer_conv_window.comp.
        unsafe { self.launch_3d("convolution", self.conv_window.get()?, stream, grid, args) }
    }

    /// The pipeline of the selected shape of the delta rule and its rows per workgroup.
    fn delta_kernel(&self, window: bool, format: StateFormat) -> Result<(&Function, u32)> {
        let k = &self.delta[self.delta_selected].1;
        let lazy = match (window, format) {
            (true, StateFormat::F32) => &k.window_f32,
            (true, StateFormat::F16) => &k.window_f16,
            (false, StateFormat::F32) => &k.commit_f32,
            (false, StateFormat::F16) => &k.commit_f16,
        };
        Ok((lazy.get()?, k.rows_per_group))
    }

    /// The state must be 16-byte aligned and a slot a whole number of 16-byte units from
    /// the next, so that every row of a matrix is.
    fn check_state(format: StateFormat, state: DevicePtr, slot_stride: u64) -> Result<()> {
        check_align("delta rule state", state, 16)?;
        let unit = 16 / format.bytes();
        if !slot_stride.is_multiple_of(unit) {
            return Err(invalid(
                "delta rule state",
                format!("slots {slot_stride} values apart (a multiple of {unit})"),
            ));
        }
        Ok(())
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
        for p in [q, kv, out] {
            check_align("delta rule", p, 16)?;
        }
        Self::check_state(format, state, state_slot_stride)?;
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let (function, rows_per_group) = self.delta_kernel(true, format)?;
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
        let grid = [shape.n_v_heads * (shape.head_dim / rows_per_group), dims.n_segs, 1];
        // SAFETY: the argument list matches layer_delta.glsl.
        unsafe { self.launch_3d("delta rule", function, stream, grid, args) }
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
        check_align("delta commit", kv, 16)?;
        Self::check_state(format, state, state_slot_stride)?;
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let (function, rows_per_group) = self.delta_kernel(false, format)?;
        let args = Args::new()
            .ptr(step)
            .ptr(DevicePtr::NULL)
            .ptr(kv)
            .ptr(alpha)
            .ptr(beta_in)
            .ptr(dt_bias)
            .ptr(ssm_a)
            .ptr(state)
            .u64(state_slot_stride)
            .ptr(DevicePtr::NULL)
            .u32(shape.n_k_heads)
            .u32(shape.n_v_heads);
        let grid = [shape.n_v_heads * (shape.head_dim / rows_per_group), dims.n_segs, 1];
        // SAFETY: the argument list matches layer_delta.glsl.
        unsafe { self.launch_3d("delta commit", function, stream, grid, args) }
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
        let grid = [shape.conv_dim().div_ceil(128), dims.n_segs, 1];
        // SAFETY: the argument list matches layer_conv_commit.comp.
        unsafe { self.launch_3d("convolution commit", self.conv_commit.get()?, stream, grid, args) }
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
    /// quantized activations (the layout of `Kernels::quantize_activations` over
    /// `items * width` values, then the f32 values; [`quantized_activation_bytes`]
    /// bytes, 16-byte aligned). `width` is 64, 128 or 256.
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
        if items.checked_mul(width).is_none() {
            return Err(invalid("gated norm", format!("{items} items of {width} values")));
        }
        if items == 0 {
            return Ok(());
        }
        let args =
            Args::new().ptr(o).ptr(z).ptr(weight).ptr(act).u32(items).u32(width).f32(eps).u32(
                match gate {
                    OutputGate::Silu => 0,
                    OutputGate::Sigmoid => 1,
                },
            );
        // SAFETY: the argument list matches layer_gated_norm_quant.comp.
        unsafe { self.gated_norm_quant.get()?.launch(stream, self.grid(items.div_ceil(8)), args) }
    }

    /// `silu(gate) * up` quantized for the down projection, over `n_values`
    /// values (a multiple of 32): `act` gets the layout of
    /// `Kernels::quantize_activations`, then the f32 values
    /// ([`quantized_activation_bytes`] bytes).
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
        // SAFETY: the argument list matches layer_swiglu_quant.comp.
        unsafe { self.swiglu_quant.get()?.launch(stream, self.grid(threads.div_ceil(256)), args) }
    }
}

/// What a merge of a tensor split does with the residual stream after it
/// added the devices' partial results to it (`shaders/layer_merge_wire.comp`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MergeNorm {
    /// Nothing.
    None,
    /// The RMS norm with `weight`, as f32 rows at `out`.
    Rows { weight: DevicePtr, out: DevicePtr },
    /// The RMS norm with `weight`, quantized into `act` for the weight
    /// kernels: the layout of `Kernels::quantize_activations` over all rows, then the
    /// f32 values ([`quantized_activation_bytes`] bytes).
    Quantized { weight: DevicePtr, act: DevicePtr },
}

impl MergeNorm {
    /// The kernel's arguments: weight, rows, activations, and the mode with
    /// the bit for the wide wire form.
    fn args(self, wire: Wire) -> (DevicePtr, DevicePtr, DevicePtr, u32) {
        let wide = match wire {
            Wire::F16 => 0,
            Wire::F32 => 256,
        };
        match self {
            Self::None => (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, wide),
            Self::Rows { weight, out } => (weight, out, DevicePtr::NULL, 1 | wide),
            Self::Quantized { weight, act } => (weight, DevicePtr::NULL, act, 2 | wide),
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
/// of a tensor split: the rows and one f32 factor per row. The form is the CUDA
/// layer's, byte for byte.
pub fn partial_wire_bytes(wire: Wire, n_rows: u32, d: u32) -> u64 {
    let values = u64::from(n_rows) * u64::from(d);
    let rows = match wire {
        Wire::F16 => (values * 2).next_multiple_of(4),
        Wire::F32 => values * 4,
    };
    rows + u64::from(n_rows) * 4
}

/// Rows of the merge kernels are whole groups of 32 values (the groups of the quantized
/// activations).
fn check_merge_width(what: &'static str, d: u32) -> Result<()> {
    if d == 0 || !d.is_multiple_of(32) {
        return Err(invalid(what, format!("rows of {d} values (a multiple of 32)")));
    }
    Ok(())
}

/// The merges of a tensor split. A partial crosses between Vulkan devices as a copy
/// ([`crate::Mesh`], [`crate::Lane`]); there is no kernel that writes into a peer's memory
/// (`merge_push` of the CUDA layer), because Vulkan has no peer memory.
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
        check_merge_width("partial to wire", d)?;
        check_align("partial to wire", p, 16)?;
        check_align("partial to wire", wire, 16)?;
        if n_rows == 0 {
            return Ok(());
        }
        let args =
            Args::new().ptr(p).ptr(wire).u32(n_rows).u32(d).u32(u32::from(form == Wire::F32));
        // SAFETY: the argument list matches layer_partial_to_wire.comp.
        unsafe { self.partial_to_wire.get()?.launch(stream, self.grid(n_rows), args) }
    }

    /// `x += ` the sum of all devices' partials of `n_rows` rows of `d`
    /// values, then what `norm` says. The partials are in wire form `form`: this
    /// device's at `own`, the `n_peers` others' in device order at `peers`,
    /// `peer_stride` bytes apart; `own_rank` is this device's place in that
    /// order. Every device that merges the same partials gets the same sum, bit for bit.
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
        check_merge_width("merge", d)?;
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
            .ptr(weight)
            .ptr(xn)
            .ptr(act)
            .u32(n_peers)
            .u32(own_rank)
            .u32(n_rows)
            .u32(d)
            .f32(eps)
            .u32(mode);
        // SAFETY: the argument list matches layer_merge_wire.comp.
        unsafe { self.merge_wire.get()?.launch(stream, self.grid(n_rows), args) }
    }
}
