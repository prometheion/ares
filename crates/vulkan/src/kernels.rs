//! The engine's kernels, loaded on a device, with one typed launch per kernel.
//!
//! The launches are `unsafe` because the device trusts the addresses it is
//! given. The layers above hold the allocations and are the ones that can
//! promise the ranges are right; this layer turns shapes into grid sizes and
//! keeps the argument lists in one place next to the kernel names.
//!
//! The kernels are GLSL compute shaders (`shaders/`), compiled to SPIR-V by
//! `build.rs`. Every pipeline is made in [`Kernels::load`]; a launch never
//! creates one, and, because buffers are passed by device address in push
//! constants, never writes a descriptor either.

use std::sync::Arc;

use ares_core::GgmlType;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::prefill::{Precision, Prefill, PrefillPath};
use crate::stream::Stream;
use crate::weights::{self, Quantized};

mod layout;

pub use layout::{ActivationForm, Layout, LayoutKernel, LayoutRows, LayoutTable};

static ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_f32.spv"));
static RMS_NORM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm_f32.spv"));
static SUM32_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sum32_f32.spv"));
static MATVEC_Q4_K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_q4_K.spv"));

/// The most activation rows one launch of a weight kernel handles
/// (`ARES_MAX_BATCH` in `shaders/common.glsl`).
pub const MAX_BATCH: u32 = 8;

include!(concat!(env!("OUT_DIR"), "/shader_constants.rs"));

/// Values in a Q4_K block, and bytes: 256 and 144.
const Q4K_VALUES: u32 = GgmlType::Q4_K.block_values();

/// The most warps in a workgroup of the matrix kernel, and so the number of
/// workgroup sizes it is compiled for: 32 to 32 * `MATVEC_MAX_WARPS` invocations
/// in steps of a warp. Like the CUDA kernel's block size, the size is chosen per
/// launch from the row length; unlike it, it is fixed in the pipeline, so there
/// is one pipeline per size, all made at load.
const MATVEC_MAX_WARPS: u32 = 4;
const MATVEC_SIZES: usize = MATVEC_MAX_WARPS as usize;

struct Set {
    device: Device,
    add_f32: Function,
    rms_norm_f32: Function,
    sum32_f32: Function,
    /// Index `i` has workgroups of `32 * (i + 1)` invocations.
    matvec_q4_k: Vec<Function>,
    /// The quantized weight kernels, on a device that can run them.
    quantized: Option<Quantized>,
    /// The tensor-core kernels for many rows, on a device that has cooperative matrices.
    prefill: Option<Prefill>,
    /// The kernels over the device layouts, on a device whose cooperative matrices can be
    /// built in registers.
    layouts: Option<layout::Layouts>,
}

/// The kernels of the engine on one device.
#[derive(Clone)]
pub struct Kernels(Arc<Set>);

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernels(on {:?})", self.0.device)
    }
}

/// `f` over `items` on all cores, in order. Making a pipeline compiles a shader for the GPU, which
/// takes milliseconds, and Vulkan allows pipelines to be made on several threads at once; a
/// device has hundreds (a weight kernel for each type and number of activation rows), so
/// loading them one after the other would take seconds.
pub(crate) fn parallel_map<T: Sync, R: Send>(
    items: &[T],
    f: impl Fn(&T) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(items.len());
    let slots: Vec<std::sync::Mutex<Option<Result<R>>>> =
        items.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    *slots[i].lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(f(item));
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .expect("every item was visited")
        })
        .collect()
}

/// A launch of `groups` workgroups along one logical dimension. A device may
/// cap a dispatch's x dimension below what a long vector needs; the excess
/// then goes into y, and the shaders fold the two back into one index.
pub(crate) fn grid_1d(groups: u32, limit_x: u32) -> [u32; 3] {
    if groups <= limit_x { [groups, 1, 1] } else { [limit_x, groups.div_ceil(limit_x), 1] }
}

/// The invocations per workgroup of the matrix kernel for rows of `n_in`
/// values: a warp for each block of 256 values of the row, up to four warps.
/// More warps leave fewer workgroups resident on a multiprocessor for the
/// registers the kernel needs, and measured slower on both test GPUs for the
/// shapes of a 27B model (a row of 5120 or 17408 values) with one and eight
/// activation rows.
pub(crate) fn matvec_threads(n_in: u32) -> u32 {
    32 * (n_in / Q4K_VALUES).clamp(1, MATVEC_MAX_WARPS)
}

impl Kernels {
    /// Compiles the small kernels for `device` (one pipeline per kernel, and one per
    /// workgroup size for the matrix kernel), measures what its tensor cores do (the speed
    /// of f16 accumulation, the element order of an operand: about 50 ms) and uploads the
    /// codebooks. The driver compiles the shaders for the GPU as pipelines are made, which
    /// takes a moment; do it once per device. The weight kernels (hundreds: each type has
    /// a pipeline for each number of activation rows, and the prefill kernels) are made when
    /// they are first launched, or ahead of that by [`Kernels::prepare`].
    pub fn load(device: &Device) -> Result<Self> {
        let main = |image: &[u8]| Module::load(device, image)?.function("main");
        let matvec = Module::load(device, MATVEC_Q4_K)?;
        let matvec_q4_k = (1..=MATVEC_SIZES as u32)
            .map(|i| matvec.function_with("main", &[(0, 32 * i)]))
            .collect::<Result<Vec<_>>>()?;
        let quantized = if device.physical().quantized_unsupported.is_none() {
            Some(Quantized::load(device)?)
        } else {
            None
        };
        let prefill = match &quantized {
            // The prefill kernels take a subgroup for 32 invocations (four to a workgroup).
            Some(q)
                if !device.info().cooperative_matrix.is_empty()
                    && device.info().subgroup_size == 32 =>
            {
                Some(Prefill::load(device, q.tables())?)
            }
            _ => None,
        };
        let layouts = match &prefill {
            Some(p) if p.register_operand() => Some(layout::Layouts::load(device, p.full_rate())?),
            _ => None,
        };
        Ok(Self(Arc::new(Set {
            device: device.clone(),
            add_f32: main(ADD_F32)?,
            rms_norm_f32: main(RMS_NORM_F32)?,
            sum32_f32: main(SUM32_F32)?,
            matvec_q4_k,
            quantized,
            prefill,
            layouts,
        })))
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    fn grid(&self, groups: u32) -> [u32; 3] {
        grid_1d(groups, self.0.device.physical().max_work_group_count[0])
    }

    pub(crate) fn quantized(&self) -> Result<&Quantized> {
        self.0.quantized.as_ref().ok_or_else(|| Error::Unsupported {
            device: self.0.device.info().name.clone(),
            reason: self
                .0
                .device
                .physical()
                .quantized_unsupported
                .clone()
                .unwrap_or_else(|| "the quantized weight kernels are not loaded".into()),
        })
    }

    /// Makes the weight kernels for the types in `types` now, on all cores: the decode kernels
    /// (a pipeline for each number of activation rows) and the prefill kernels. A kernel is
    /// otherwise made when it is first launched, which takes milliseconds (tens for a
    /// prefill kernel) in the step that launches it: the runtime calls this with the types of
    /// the model when it loads it. [`Kernels::load`] itself makes only the small kernels.
    pub fn prepare(&self, types: &[GgmlType]) -> Result<()> {
        if let Some(q) = &self.0.quantized {
            q.prepare(types)?;
        }
        if let Some(p) = &self.0.prefill {
            p.prepare(types)?;
        }
        if let Some(l) = &self.0.layouts {
            l.prepare(types)?;
        }
        Ok(())
    }

    /// Whether this device runs the quantized weight kernels
    /// ([`Kernels::quantize_activations`], [`Kernels::matvec`]): they need integer
    /// dot products, 8- and 16-bit buffer access and 32-wide clustered subgroup
    /// operations, which every current desktop GPU has.
    pub fn has_quantized_kernels(&self) -> bool {
        self.0.quantized.is_some()
    }

    /// The bytes that [`Kernels::quantize_activations`] writes for `n_batch` rows
    /// of `n_in` activations: the int8 values, then (from the next multiple of 16
    /// bytes) one `(scale, sum)` pair of f32 per group of 32 values. The format is
    /// the CUDA layer's (`kernels/quantize_act.cuh`), byte for byte.
    pub fn quantized_activation_bytes(n_in: u32, n_batch: u32) -> u64 {
        weights::quantized_activation_bytes(n_in, n_batch)
    }

    /// Whether [`Kernels::matvec`] has a kernel for weights of type `ty`: every type a
    /// GGUF file can carry as a weight (F32, F16, BF16 and the 25 quantized ones).
    pub fn has_fast_matvec(ty: GgmlType) -> bool {
        weights::has_fast_matvec(ty)
    }

    /// Whether the kernel of `ty` reads activations quantized by
    /// [`Kernels::quantize_activations`] (every type but F32, F16 and BF16, whose
    /// kernels read the f32 activations as they are).
    pub fn needs_quantized_activations(ty: GgmlType) -> bool {
        weights::needs_quantized_activations(ty)
    }

    /// Quantizes `n_batch` rows of `n_in` f32 activations to the int8 format the
    /// fast weight kernels read ([`Kernels::matvec`]). One quantization serves every
    /// matrix that multiplies the same activations, whatever their types. The codes
    /// and scales are exactly those the CUDA kernel produces.
    ///
    /// `n_in` must be a multiple of 32. `q` receives
    /// [`Kernels::quantized_activation_bytes`] bytes.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` must address
    /// `n_batch * n_in` f32 values and `q` the bytes above, 16-byte aligned, not
    /// overlapping `x`.
    pub unsafe fn quantize_activations(
        &self,
        stream: &Stream,
        x: DevicePtr,
        q: DevicePtr,
        n_in: u32,
        n_batch: u32,
    ) -> Result<()> {
        let limit = self.0.device.physical().max_work_group_count[0];
        // SAFETY: the caller's guarantee is the one of the function it forwards to.
        unsafe { self.quantized()?.quantize_activations(stream, x, q, n_in, n_batch, limit) }
    }

    /// Quantizes `n_rows` (any number, at least 1) rows of `n_in` f32 activations to the
    /// format of [`Kernels::quantize_activations`], which the tensor-core kernels
    /// ([`Kernels::matmul`]) read too: one quantization serves both kernel families.
    ///
    /// # Safety
    ///
    /// As [`Kernels::quantize_activations`], for `n_rows` rows.
    pub unsafe fn quantize_activation_rows(
        &self,
        stream: &Stream,
        x: DevicePtr,
        q: DevicePtr,
        n_in: u32,
        n_rows: u32,
    ) -> Result<()> {
        let limit = self.0.device.physical().max_work_group_count[0];
        // SAFETY: the caller's guarantee is the one of the function it forwards to.
        unsafe { self.quantized()?.quantize_rows(stream, x, q, n_in, n_rows, limit) }
    }

    fn prefill(&self) -> Result<&Prefill> {
        self.0.prefill.as_ref().ok_or_else(|| Error::Unsupported {
            device: self.0.device.info().name.clone(),
            reason:
                "the prefill kernels need cooperative matrices and the quantized weight kernels"
                    .into(),
        })
    }

    fn grid_limit(&self) -> u32 {
        self.0.device.physical().max_work_group_count[0]
    }

    /// Whether this device has the prefill kernels of `path` at all: they need
    /// `VK_KHR_cooperative_matrix` with int8 products of 16 x 32 by 32 x 16 into int32
    /// ([`PrefillPath::Int8`]) or f16 products of 16 x 16 by 16 x 16 into f16 and f32
    /// ([`PrefillPath::F16`], [`PrefillPath::F16Flush`]).
    pub fn has_prefill_path(&self, path: PrefillPath) -> bool {
        self.0.prefill.as_ref().is_some_and(|p| p.has_path(path))
    }

    /// Whether [`Kernels::matmul`] has a tensor-core kernel for weights of type `ty` on
    /// `path` on this device.
    pub fn has_matmul(&self, ty: GgmlType, path: PrefillPath) -> bool {
        self.0.prefill.as_ref().is_some_and(|p| p.has_matmul(ty, path))
    }

    /// The bytes [`Kernels::prepare_prefill_activations`] writes for `n_rows` rows of
    /// `n_in` activations in the format of `path`.
    pub fn prefill_activation_bytes(path: PrefillPath, n_in: u32, n_rows: u32) -> u64 {
        Prefill::activation_bytes(path, n_in, n_rows)
    }

    /// Prepares `n_rows` rows of `n_in` f32 activations for the prefill kernels of `path`:
    /// quantizes them to int8 ([`PrefillPath::Int8`], the format of
    /// [`Kernels::quantize_activations`], so the same bytes serve the decode kernels) or
    /// converts them to f16 divided per row by a power of two ([`PrefillPath::F16`] and
    /// [`PrefillPath::F16Flush`]). One preparation serves every matrix that multiplies the
    /// same activations.
    ///
    /// `n_in` must be a multiple of 32 and `n_in * n_rows` must fit in 32 bits.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` must address
    /// `n_rows * n_in` f32 values and `out` the [`Kernels::prefill_activation_bytes`] of
    /// the path, 16-byte aligned (checked), not overlapping `x`.
    pub unsafe fn prepare_prefill_activations(
        &self,
        stream: &Stream,
        path: PrefillPath,
        x: DevicePtr,
        out: DevicePtr,
        n_in: u32,
        n_rows: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee is the one of the functions it forwards to.
        unsafe {
            match path {
                PrefillPath::Int8 => self.quantize_activation_rows(stream, x, out, n_in, n_rows),
                PrefillPath::F16 | PrefillPath::F16Flush => {
                    self.prefill()?.convert_f16(stream, x, out, n_in, n_rows, self.grid_limit())
                }
            }
        }
    }

    /// The path the prefill family takes on this device for the requested precision, chosen
    /// when the kernels were loaded. [`Precision::Exact`] is [`PrefillPath::F16`] where the
    /// device has the f16 kernels (else [`PrefillPath::Int8`]). [`Precision::Fast`] is the
    /// fastest path whose error is no worse than the decode kernels' (about 1 % rms):
    /// [`PrefillPath::F16`] where the tensor cores run f16 products with f32 accumulation
    /// as fast as with f16 accumulation ([`Kernels::tensor_cores_full_rate`]: measured at
    /// load), else [`PrefillPath::F16Flush`], which is as fast as int8 on the cards measured
    /// and 20 times more exact; [`PrefillPath::Int8`] on a device that has only int8
    /// cooperative matrices. Refused on a device without cooperative matrices: the decode
    /// kernels in batches of 8 rows ([`Kernels::matmul_decode`]) are what it has instead.
    pub fn prefill_path(&self, precision: Precision) -> Result<PrefillPath> {
        let prefill = self.prefill()?;
        let f16 = prefill.has_path(PrefillPath::F16);
        if !f16 && !prefill.has_path(PrefillPath::Int8) {
            return Err(Error::Unsupported {
                device: self.0.device.info().name.clone(),
                reason: "no cooperative matrix shape the prefill kernels use".into(),
            });
        }
        Ok(match precision {
            Precision::Exact | Precision::Fast if !f16 => PrefillPath::Int8,
            Precision::Exact => PrefillPath::F16,
            Precision::Fast if prefill.full_rate() => PrefillPath::F16,
            Precision::Fast => PrefillPath::F16Flush,
        })
    }

    /// Whether the tensor cores of this device run f16 products with f32 accumulation as
    /// fast as with f16 accumulation (measured when the kernels were loaded). GeForce cards
    /// run the second at twice the rate of the first.
    pub fn tensor_cores_full_rate(&self) -> bool {
        self.0.prefill.as_ref().is_some_and(Prefill::full_rate)
    }

    /// Whether the elements of a cooperative matrix of use A are held in the order of an
    /// `mma` fragment on this device (checked when the kernels were loaded): the kernels over
    /// the device layouts, which decode the weights straight into the operand, need it
    /// ([`Kernels::has_layout_kernels`]).
    pub fn cooperative_operand_in_registers(&self) -> bool {
        self.0.prefill.as_ref().is_some_and(Prefill::register_operand)
    }

    /// `y[b] = W x[b]` for `n_rows` (any number) rows of f32 activations `x`, with the decode
    /// kernels: the rows are quantized and multiplied in batches of [`MAX_BATCH`], each batch
    /// through `scratch` (at least [`Kernels::quantized_activation_bytes`] of `n_in` and
    /// [`MAX_BATCH`] rows, 16-byte aligned). What a device without cooperative matrices runs
    /// for a prompt, and the reference for the tensor-core paths: the weights are read once
    /// per batch of 8, so it is memory-bound up to 8 rows and not beyond. The float types
    /// read `x` as it is.
    ///
    /// # Safety
    ///
    /// As [`Kernels::matvec`] for the matrix, `x` `n_rows * n_in` f32 values, `y`
    /// `n_rows * n_out` f32 values, `scratch` as above; all alive until the launches have run.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_decode(
        &self,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        x: DevicePtr,
        scratch: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        let float = !Self::needs_quantized_activations(ty);
        let mut done = 0;
        while done < n_rows {
            let n = (n_rows - done).min(MAX_BATCH);
            let x_chunk = x.offset(u64::from(done) * u64::from(n_in) * 4);
            let y_chunk = y.offset(u64::from(done) * u64::from(n_out) * 4);
            // SAFETY: the caller's guarantee covers the rows of this chunk.
            unsafe {
                let act = if float {
                    x_chunk
                } else {
                    self.quantize_activations(stream, x_chunk, scratch, n_in, n)?;
                    scratch
                };
                self.matvec(stream, ty, w, act, y_chunk, n_in, n_out, n)?;
            }
            done += n;
        }
        Ok(())
    }

    /// `y[b] = W x[b]` for a matrix `w` of type `ty` with `n_out` rows and `n_in` columns as
    /// the file stores it, and `n_rows` (1 or more) rows of activations that
    /// [`Kernels::prepare_prefill_activations`] has prepared in the format of `path` into
    /// `act`: the prefill kernel, on the tensor cores. It reads the weights once per 64 rows.
    ///
    /// [`PrefillPath::Int8`] computes with the same arithmetic as [`Kernels::matvec`]
    /// (integer products of the weight codes with int8 activations, scaled per sub-block in
    /// f32), so a row gives the same result in either; Q4_K only so far. The f16 paths
    /// decode the weights to f16 (`shaders/mmd_<type>.glsl`) and multiply f16 activations:
    /// they differ from the exact product by the rounding of weights and activations to f16
    /// (and, for [`PrefillPath::F16Flush`], of partial sums of 64 products). Check
    /// [`Kernels::has_matmul`]; `n_in` must be a whole number of blocks and of 64 values.
    ///
    /// # Safety
    ///
    /// As [`Kernels::matvec`], for `n_rows` rows of activations prepared for this `path`
    /// (16-byte aligned, checked) and `y` of `n_rows * n_out` f32 values; the alignment of
    /// the weights is that of [`Kernels::matvec`] for the int8 path and 16 bytes for Q4_K and
    /// Q5_K and the float types, 4 for Q4_1, Q5_1 and Q2_K and 2 for the others on the f16
    /// paths (checked).
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
        // SAFETY: the caller's guarantee is the one of the function it forwards to.
        unsafe {
            self.prefill()?.matmul(
                stream,
                ty,
                path,
                w,
                act,
                y,
                n_in,
                n_out,
                n_rows,
                self.grid_limit(),
            )
        }
    }

    /// `y[b] = W x[b]` for a matrix `w` of type `ty` with `n_out` rows and `n_in`
    /// columns and `n_batch` (1 to [`MAX_BATCH`]) rows of activations that
    /// [`Kernels::quantize_activations`] has quantized into `q`. The weights are
    /// read once however many rows there are.
    ///
    /// This is the decode kernel: integer dot products over the quantized
    /// activations, on the weight codes, with the arithmetic of the CUDA layer's
    /// `matvec`: the same activations give the same results up to f32 rounding of
    /// the sums. A type without a fast kernel (see [`Kernels::has_fast_matvec`]) is
    /// refused with an error.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `w` must address
    /// the matrix as GGUF stores it (`n_out` rows of `n_in` values in whole blocks
    /// of `ty`), aligned to 16 bytes for Q4_K and Q5_K, to 8 for IQ4_XS and to 2
    /// for the other types (checked); `q` the quantized activations of this `n_in`
    /// and `n_batch`, 16-byte aligned (checked); `y` `n_batch * n_out` f32 values.
    /// `y` must not overlap the inputs.
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
        let limit = self.0.device.physical().max_work_group_count[0];
        // SAFETY: the caller's guarantee is the one of the function it forwards to.
        unsafe { self.quantized()?.matvec(stream, ty, w, q, y, n_in, n_out, n_batch, limit) }
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
        // SAFETY: the argument list matches add_f32.comp; the ranges are the
        // caller's guarantee.
        unsafe { self.0.add_f32.launch(stream, self.grid(n.div_ceil(256)), args) }
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
        let args = Args::new().ptr(x).ptr(weight).ptr(y).u32(n).u32(rows).f32(eps);
        // SAFETY: the argument list matches rms_norm_f32.comp.
        unsafe { self.0.rms_norm_f32.launch(stream, self.grid(rows), args) }
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
    /// `n_batch * n_in` f32 values, 4-byte aligned (16-byte aligned reads
    /// faster: the kernel then loads it four values at a time); `scratch`
    /// `n_batch * n_in / 32` f32 values; `y` `n_batch * n_out` f32 values. `y`
    /// and `scratch` must not overlap the inputs.
    ///
    /// The thread layout is not the CUDA kernel's (see
    /// `shaders/matvec_q4_K.comp` for why), so the sums are added in another
    /// order and agree with it to rounding, not bit for bit.
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
        if !n_in.is_multiple_of(Q4K_VALUES) || n_in == 0 {
            return Err(Error::invalid(
                "Q4_K matrix",
                format!("{n_in} columns are not a whole number of {Q4K_VALUES}-value blocks"),
            ));
        }
        if n_batch == 0 || n_batch > MAX_BATCH {
            return Err(Error::invalid(
                "weight kernel batch",
                format!("{n_batch} activation rows; one launch takes 1 to {MAX_BATCH}"),
            ));
        }
        if n_out == 0 {
            return Ok(());
        }
        let n_groups = n_batch * (n_in / 32);
        let args = Args::new().ptr(x).ptr(scratch).u32(n_groups);
        // SAFETY: the argument list matches sum32_f32.comp.
        unsafe {
            self.0.sum32_f32.launch(stream, self.grid(n_groups.div_ceil(256)), args)?;
        }
        let function = &self.0.matvec_q4_k[(matvec_threads(n_in) / 32 - 1) as usize];
        let args = Args::new().ptr(w).ptr(x).ptr(scratch).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
        // SAFETY: the argument list matches matvec_q4_K.comp.
        unsafe { function.launch(stream, self.grid(n_out.div_ceil(MATVEC_ROWS)), args) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_workgroup_of_the_matrix_kernel_follows_the_row_length() {
        // A warp per block of 256 values, up to four.
        assert_eq!(matvec_threads(256), 32);
        assert_eq!(matvec_threads(512), 64);
        assert_eq!(matvec_threads(2560), 128);
        assert_eq!(matvec_threads(17408), 128);
        assert_eq!(matvec_threads(256 * 1000), 128);
        for n_in in (256..=40960).step_by(256) {
            let threads = matvec_threads(n_in);
            assert!(threads.is_multiple_of(32) && (32..=128).contains(&threads));
            assert!(((threads / 32 - 1) as usize) < MATVEC_SIZES);
        }
    }

    #[test]
    fn long_launches_fold_into_two_dimensions_when_the_device_caps_x() {
        assert_eq!(grid_1d(10, 65535), [10, 1, 1]);
        assert_eq!(grid_1d(65535, 65535), [65535, 1, 1]);
        assert_eq!(grid_1d(65536, 65535), [65535, 2, 1]);
        let [x, y, z] = grid_1d(300_000, 65535);
        assert!(u64::from(x) * u64::from(y) >= 300_000 && z == 1);
        // Less than one full extra row of waste, which the shaders' bounds
        // checks absorb.
        assert!(u64::from(x) * u64::from(y) - 300_000 < u64::from(x));
    }

    /// The batch limit and the row tile exist once in Rust and once in GLSL.
    #[test]
    fn the_limits_agree_with_the_shaders() {
        let common = include_str!("../shaders/common.glsl");
        let define = common
            .lines()
            .find_map(|line| line.strip_prefix("#define ARES_MAX_BATCH "))
            .expect("common.glsl defines ARES_MAX_BATCH");
        assert_eq!(define.trim().parse::<u32>().unwrap(), MAX_BATCH);
        // The row tile is handed to the shader by build.rs; the shader takes
        // it from the command line, not from a literal.
        let matvec = include_str!("../shaders/matvec_q4_K.comp");
        assert!(
            matvec.contains("ARES_MATVEC_ROWS") && !matvec.contains("#define ARES_MATVEC_ROWS")
        );
        const { assert!(MATVEC_ROWS >= 1) };
    }

    #[test]
    fn the_q4_k_geometry_comes_from_the_core_crate() {
        assert_eq!(Q4K_VALUES, 256);
        assert_eq!(GgmlType::Q4_K.block_bytes(), 144);
    }
}
