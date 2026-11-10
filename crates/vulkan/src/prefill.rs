//! The prefill weight kernels: a quantized matrix times many rows of activations on the
//! tensor cores (`VK_KHR_cooperative_matrix`). The counterpart of the CUDA layer's
//! `PrefillPath`, `Kernels::prepare_prefill_activations`, `Kernels::matmul`,
//! `Kernels::rearrange` and `Kernels::matmul_rearranged`, with the same activation formats
//! and the same rearranged form of a Q4_K matrix, so that one upload serves both backends.
//!
//! Three ways to compute the product, chosen by the caller (see [`PrefillPath`]) and, for
//! the common case, by [`Precision`] and what the device does fastest:
//!
//! - int8: integer products of the weight codes with activations quantized to int8 (the
//!   decode kernels' format, `quantize_act.comp`), scaled per sub-block in f32;
//! - f16: weights decoded to f16, f16 activations, f32 accumulation;
//! - f16 with f16 accumulation, flushed to f32 every 64 values of K.
//!
//! `shaders/mm_q4_K.comp` is the int8 kernel, `shaders/mm_f16_core.glsl and mmw_q4_K_r.glsl` the f16 ones.

use ares_core::GgmlType;

use std::sync::Arc;

use crate::device::{ComponentType, Device};
use crate::error::{Error, Result};
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Function, Lazy, Module};
use crate::stream::{Event, Stream};
use crate::weights::{check_alignment, image, quantized_activation_bytes};

macro_rules! spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")) as &[u8]
    };
}

static MM_Q4_K: &[u8] = spv!("mm_q4_K");

/// The types with an f16 prefill kernel for the file's layout (`mm_{f,h}_<name>.comp`), the
/// shader name, and the alignment in bytes the decoder needs of the matrix (its widest
/// load). The decoder of a type is `shaders/mmd_<name>.glsl`; adding a type is writing one.
const GENERIC: [(GgmlType, &str, u64); 28] = [
    (GgmlType::Q4_K, "q4_K", 16),
    (GgmlType::Q5_K, "q5_K", 16),
    (GgmlType::Q6_K, "q6_K", 2),
    (GgmlType::Q3_K, "q3_K", 2),
    (GgmlType::Q2_K, "q2_K", 4),
    (GgmlType::Q8_0, "q8_0", 2),
    (GgmlType::Q4_0, "q4_0", 2),
    (GgmlType::Q4_1, "q4_1", 4),
    (GgmlType::Q5_0, "q5_0", 2),
    (GgmlType::Q5_1, "q5_1", 4),
    (GgmlType::F32, "f32", 16),
    (GgmlType::F16, "f16", 16),
    (GgmlType::BF16, "bf16", 16),
    (GgmlType::IQ4_NL, "iq4_nl", 2),
    (GgmlType::IQ4_XS, "iq4_xs", 8),
    (GgmlType::MXFP4, "mxfp4", 1),
    (GgmlType::NVFP4, "nvfp4", 4),
    (GgmlType::Q2_0, "q2_0", 2),
    (GgmlType::Q1_0, "q1_0", 2),
    (GgmlType::TQ2_0, "tq2_0", 2),
    (GgmlType::TQ1_0, "tq1_0", 2),
    (GgmlType::IQ2_XXS, "iq2_xxs", 2),
    (GgmlType::IQ2_XS, "iq2_xs", 2),
    (GgmlType::IQ2_S, "iq2_s", 2),
    (GgmlType::IQ3_XXS, "iq3_xxs", 2),
    (GgmlType::IQ3_S, "iq3_s", 2),
    (GgmlType::IQ1_S, "iq1_s", 2),
    (GgmlType::IQ1_M, "iq1_m", 4),
];
static CONVERT_F16: &[u8] = spv!("convert_act_f16");
static REPACK_Q4_K: &[u8] = spv!("repack_q4_K");
static MM_F_Q4_K_R: &[u8] = spv!("mm_f_q4_K_r");
static MM_H_Q4_K_R: &[u8] = spv!("mm_h_q4_K_r");

/// The weight rows and activation rows one workgroup of the prefill kernels computes.
const TILE: u32 = 64;

/// How the prefill family reads its activations and which of its kernels it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPath {
    /// Integer products of the weights' codes with activations quantized to int8
    /// ([`crate::Kernels::quantize_activation_rows`]'s format, the decode kernels' too):
    /// the arithmetic of the decode kernels on the tensor cores.
    Int8,
    /// f16 products over the dequantized weights and f16 activations with f32
    /// accumulation: the direct product, which rounds nothing but the f16 inputs.
    F16,
    /// The f16 path with f16 accumulation, flushed to f32 after every 64 values of K: the
    /// same tiles and activations as [`PrefillPath::F16`], at the rate GeForce cards give
    /// f16 accumulation (twice that of f32 accumulation). The activations are divided per
    /// row by a power of two so that the largest is below 1, so a partial sum of 64
    /// products cannot overflow.
    F16Flush,
}

/// What a caller wants of the prefill product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// The direct f16 product with f32 accumulation, on every device.
    Exact,
    /// The fastest kernel whose error is no worse than the decode kernels' (about 1 %
    /// rms): on a device that runs f16 products with f32 accumulation as fast as with f16
    /// accumulation that is [`PrefillPath::F16`], else the fastest of the others.
    Fast,
}

/// The prefill kernels on one device.
#[derive(Debug, Clone)]
pub(crate) struct Prefill {
    int8: Vec<(GgmlType, Function)>,
    convert_f16: Option<Function>,
    rearranged: Option<Rearranged>,
    /// The f16 kernels for the file's layout: type, alignment, f32 and flushed f16
    /// accumulation.
    generic: Vec<(GgmlType, u64, Arc<Lazy>, Arc<Lazy>)>,
    /// The codebooks of the i-quants (`weights::Quantized::tables`).
    tables: DevicePtr,
    /// Whether the tensor cores run f16 products with f32 accumulation as fast as with f16
    /// accumulation (measured when the kernels are loaded; GeForce cards halve the rate of
    /// f32 accumulation).
    full_rate: bool,
    /// Whether the elements of a cooperative matrix of use A are held in the order of an
    /// `mma` fragment (checked when the kernels are loaded), so that a kernel can build the
    /// operand from the weights in registers.
    register_operand: bool,
}

#[derive(Debug, Clone)]
struct Rearranged {
    repack: Arc<Lazy>,
    /// The kernels with the weights in shared memory (64 weight rows per workgroup).
    exact: Arc<Lazy>,
    flush: Arc<Lazy>,
    /// The kernels that build the A operand in registers (128 weight rows per workgroup),
    /// on a device whose cooperative matrices allow it.
    registers: Option<(Arc<Lazy>, Arc<Lazy>)>,
}

/// Whether the device has a cooperative matrix of the given shape and types.
fn has_shape(
    device: &Device,
    shape: (u32, u32, u32),
    a: ComponentType,
    b: ComponentType,
    c: ComponentType,
) -> bool {
    device
        .info()
        .cooperative_matrix
        .iter()
        .any(|p| (p.m, p.n, p.k) == shape && p.a == a && p.b == b && p.c == c && p.result == c)
}

impl Prefill {
    pub(crate) fn load(device: &Device, tables: DevicePtr) -> Result<Self> {
        let main = |image: &[u8]| Module::load(device, image)?.function("main");
        // The int8 kernel multiplies int8 matrices of 16 x 32 by 32 x 16 into int32.
        let int8_product = has_shape(
            device,
            (16, 16, 32),
            ComponentType::I8,
            ComponentType::I8,
            ComponentType::I32,
        );
        let int8 = if int8_product { vec![(GgmlType::Q4_K, main(MM_Q4_K)?)] } else { Vec::new() };
        // The f16 kernels multiply f16 matrices of 16 x 16 by 16 x 16 into f16 or f32.
        let f16_product = has_shape(
            device,
            (16, 16, 16),
            ComponentType::F16,
            ComponentType::F16,
            ComponentType::F16,
        ) && has_shape(
            device,
            (16, 16, 16),
            ComponentType::F16,
            ComponentType::F16,
            ComponentType::F32,
        );
        let (convert_f16, rearranged) = if f16_product {
            (
                Some(main(CONVERT_F16)?),
                Some(Rearranged {
                    repack: Lazy::new(device, REPACK_Q4_K),
                    exact: Lazy::new(device, MM_F_Q4_K_R),
                    flush: Lazy::new(device, MM_H_Q4_K_R),
                    registers: None,
                }),
            )
        } else {
            (None, None)
        };
        let generic = if f16_product {
            GENERIC
                .iter()
                .map(|(ty, name, align)| {
                    Ok((
                        *ty,
                        *align,
                        Lazy::new(device, image(&format!("mm_f_{name}"))?),
                        Lazy::new(device, image(&format!("mm_h_{name}"))?),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let full_rate = f16_product && measure_full_rate(device).unwrap_or(false);
        let register_operand = f16_product && check_register_operand(device).unwrap_or(false);
        // ARES_VK_PREFILL_SHARED=1 keeps the kernels with the weights in shared memory, for
        // comparing the two.
        let rearranged = match rearranged {
            Some(mut re)
                if register_operand && std::env::var_os("ARES_VK_PREFILL_SHARED").is_none() =>
            {
                re.registers = Some((
                    Lazy::new(device, image("mm_f_q4_K_r2")?),
                    Lazy::new(device, image("mm_h_q4_K_r2")?),
                ));
                Some(re)
            }
            other => other,
        };
        Ok(Self { int8, convert_f16, rearranged, generic, tables, full_rate, register_operand })
    }

    /// Makes the f16 prefill kernels of `types` now, on all cores.
    pub(crate) fn prepare(&self, types: &[GgmlType]) -> Result<()> {
        let mut all: Vec<&Arc<Lazy>> = self
            .generic
            .iter()
            .filter(|(ty, ..)| types.contains(ty))
            .flat_map(|(_, _, exact, flush)| [exact, flush])
            .collect();
        if types.contains(&GgmlType::Q4_K)
            && let Some(re) = &self.rearranged
        {
            all.extend([&re.repack, &re.exact, &re.flush]);
            if let Some((exact, flush)) = &re.registers {
                all.extend([exact, flush]);
            }
        }
        crate::kernels::parallel_map(&all, |lazy| lazy.get().map(|_| ())).map(|_| ())
    }

    /// Whether f16 products run as fast with f32 accumulation as with f16 accumulation.
    pub(crate) fn full_rate(&self) -> bool {
        self.full_rate
    }

    /// Whether the matrix operand can be built from registers (see `register_operand`).
    pub(crate) fn register_operand(&self) -> bool {
        self.register_operand
    }

    /// Whether a prefill path is available on this device.
    pub(crate) fn has_path(&self, path: PrefillPath) -> bool {
        match path {
            PrefillPath::Int8 => !self.int8.is_empty(),
            PrefillPath::F16 | PrefillPath::F16Flush => self.rearranged.is_some(),
        }
    }

    /// Whether the kernel of `ty` on `path` is loaded.
    pub(crate) fn has_matmul(&self, ty: GgmlType, path: PrefillPath) -> bool {
        match path {
            PrefillPath::Int8 => self.int8.iter().any(|(t, _)| *t == ty),
            PrefillPath::F16 | PrefillPath::F16Flush => self.generic.iter().any(|(t, ..)| *t == ty),
        }
    }

    pub(crate) fn has_rearranged_form(&self, ty: GgmlType) -> bool {
        ty == GgmlType::Q4_K && self.rearranged.is_some()
    }

    /// The bytes of `prepare_activations` for the path.
    pub(crate) fn activation_bytes(path: PrefillPath, n_in: u32, n_rows: u32) -> u64 {
        match path {
            PrefillPath::Int8 => quantized_activation_bytes(n_in, n_rows),
            PrefillPath::F16 | PrefillPath::F16Flush => {
                let values = u64::from(n_in) * u64::from(n_rows) * 2;
                // The f16 values, then at the next multiple of 16 bytes one f32 per row.
                values.next_multiple_of(16) + 4 * u64::from(n_rows)
            }
        }
    }

    /// Converts `n_rows` rows of f32 activations to the f16 format of the f16 paths.
    ///
    /// # Safety
    ///
    /// As `Kernels::prepare_prefill_activations`.
    pub(crate) unsafe fn convert_f16(
        &self,
        stream: &Stream,
        x: DevicePtr,
        out: DevicePtr,
        n_in: u32,
        n_rows: u32,
        grid_limit: u32,
    ) -> Result<()> {
        let Some(function) = &self.convert_f16 else {
            return Err(Error::Unsupported {
                device: stream.device().info().name.clone(),
                reason: "the f16 prefill kernels need cooperative matrices of f16".into(),
            });
        };
        if n_in == 0 || !n_in.is_multiple_of(32) || n_rows == 0 {
            return Err(Error::invalid(
                "prefill activations",
                format!(
                    "{n_rows} rows of {n_in} values: rows must be a whole number of groups of 32 and not empty"
                ),
            ));
        }
        n_in.checked_mul(n_rows)
            .ok_or_else(|| Error::invalid("prefill activations", "size overflows u32"))?;
        check_alignment("prefill activations", out, 16)?;
        let args = Args::new().ptr(x).ptr(out).u32(n_in).u32(n_rows);
        // SAFETY: the argument list matches convert_act_f16.comp; the ranges are the caller's
        // guarantee.
        unsafe { function.launch(stream, crate::kernels::grid_1d(n_rows, grid_limit), args) }
    }

    /// The bytes of a rearranged matrix.
    pub(crate) fn rearranged_bytes(ty: GgmlType, n_out: u32, n_in: u32) -> Option<u64> {
        (ty == GgmlType::Q4_K)
            .then(|| u64::from(n_out.next_multiple_of(16)) * (u64::from(n_in) / 256) * 144)
    }

    /// # Safety
    ///
    /// As `Kernels::rearrange`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn rearrange(
        &self,
        stream: &Stream,
        ty: GgmlType,
        src: DevicePtr,
        dst: DevicePtr,
        n_out: u32,
        n_in: u32,
        grid_limit: u32,
    ) -> Result<()> {
        let Some(re) = self.rearranged.as_ref().filter(|_| ty == GgmlType::Q4_K) else {
            return Err(Error::invalid("rearranged form", format!("{ty} has none here")));
        };
        if n_in == 0 || !n_in.is_multiple_of(256) {
            return Err(Error::invalid(
                "weight matrix",
                format!("{n_in} columns are not a whole number of Q4_K blocks of 256 values"),
            ));
        }
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", src, 16)?;
        check_alignment("rearranged matrix", dst, 16)?;
        let tiles = u64::from(n_out.div_ceil(16));
        let threads = tiles * (u64::from(n_in) / 64) * 32 + tiles * (u64::from(n_in) / 256) * 16;
        let groups = u32::try_from(threads.div_ceil(256)).map_err(|_| {
            Error::invalid("rearranged form", format!("{n_out} x {n_in} needs too many workgroups"))
        })?;
        let args = Args::new().ptr(src).ptr(dst).u32(n_out).u32(n_in);
        // SAFETY: the argument list matches repack_q4_K.comp; the ranges are the caller's.
        unsafe {
            re.repack.get()?.launch(stream, crate::kernels::grid_1d(groups, grid_limit), args)
        }
    }

    /// # Safety
    ///
    /// As `Kernels::matmul_rearranged`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul_rearranged(
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
        grid_limit: u32,
    ) -> Result<()> {
        let Some(re) = self.rearranged.as_ref().filter(|_| ty == GgmlType::Q4_K) else {
            return Err(Error::invalid("rearranged form", format!("{ty} has none here")));
        };
        let (function, rows_per_group) = match (path, &re.registers) {
            (PrefillPath::F16, Some((exact, _))) => (exact.get()?, 128),
            (PrefillPath::F16Flush, Some((_, flush))) => (flush.get()?, 128),
            (PrefillPath::F16, None) => (re.exact.get()?, TILE),
            (PrefillPath::F16Flush, None) => (re.flush.get()?, TILE),
            (PrefillPath::Int8, _) => {
                return Err(Error::invalid(
                    "rearranged form",
                    "the rearranged kernels take f16 activations",
                ));
            }
        };
        if n_in == 0
            || !n_in.is_multiple_of(256)
            || n_rows == 0
            || n_in.checked_mul(n_rows).is_none()
        {
            return Err(Error::invalid(
                "prefill kernel batch",
                format!(
                    "{n_rows} rows of {n_in} values: not whole blocks of 256, empty or too many"
                ),
            ));
        }
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("rearranged matrix", w, 16)?;
        check_alignment("prepared activations", act, 16)?;
        let n_wt = n_out.div_ceil(rows_per_group);
        let n_bt = n_rows.div_ceil(TILE);
        let group = group_of(n_in, n_bt);
        let groups = u32::try_from(u64::from(n_wt) * u64::from(n_bt)).map_err(|_| {
            Error::invalid(
                "prefill kernel batch",
                format!("{n_out} x {n_rows} needs too many workgroups"),
            )
        })?;
        let args = Args::new()
            .ptr(w)
            .ptr(act)
            .ptr(y)
            .ptr(self.tables)
            .u32(n_in)
            .u32(n_out)
            .u32(n_rows)
            .u32(group);
        // SAFETY: the argument list matches mm_*_q4_K_r.comp; the ranges are the caller's.
        unsafe { function.launch(stream, crate::kernels::grid_1d(groups, grid_limit), args) }
    }

    /// # Safety
    ///
    /// As `Kernels::matmul`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matmul(
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
        grid_limit: u32,
    ) -> Result<()> {
        let (exact, flush) = match self.generic.iter().find(|(t, ..)| *t == ty) {
            Some((_, align, exact, flush)) => {
                if path == PrefillPath::Int8 {
                    // SAFETY: the caller's guarantee is the one of the function it forwards to.
                    return unsafe {
                        self.matmul_int8(stream, ty, w, act, y, n_in, n_out, n_rows, grid_limit)
                    };
                }
                check_alignment("weight matrix", w, *align)?;
                (exact, flush)
            }
            None if path == PrefillPath::Int8 => {
                // SAFETY: as above.
                return unsafe {
                    self.matmul_int8(stream, ty, w, act, y, n_in, n_out, n_rows, grid_limit)
                };
            }
            None => {
                return Err(Error::invalid(
                    "prefill kernel type",
                    format!(
                        "{ty} has no f16 prefill kernel here (it has one for {})",
                        GENERIC.map(|(t, ..)| t.name()).join(", ")
                    ),
                ));
            }
        };
        let function = if path == PrefillPath::F16 { exact.get()? } else { flush.get()? };
        let block_values = ty.block_values().max(64);
        if n_in == 0 || !n_in.is_multiple_of(block_values) {
            return Err(Error::invalid(
                "weight matrix",
                format!(
                    "{n_in} columns are not a whole number of {ty} blocks and steps of 64 values"
                ),
            ));
        }
        if n_rows == 0 || n_in.checked_mul(n_rows).is_none() {
            return Err(Error::invalid(
                "prefill kernel batch",
                format!("{n_rows} rows of {n_in} values: not 1 or more, or too many"),
            ));
        }
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("prepared activations", act, 16)?;
        let n_wt = n_out.div_ceil(TILE);
        let n_bt = n_rows.div_ceil(TILE);
        let group = group_of(n_in, n_bt);
        let groups = u32::try_from(u64::from(n_wt) * u64::from(n_bt)).map_err(|_| {
            Error::invalid(
                "prefill kernel batch",
                format!("{n_out} x {n_rows} needs too many workgroups"),
            )
        })?;
        let args = Args::new()
            .ptr(w)
            .ptr(act)
            .ptr(y)
            .ptr(self.tables)
            .u32(n_in)
            .u32(n_out)
            .u32(n_rows)
            .u32(group);
        // SAFETY: the argument list matches mm_{f,h}_<type>.comp; the ranges are the caller's.
        unsafe { function.launch(stream, crate::kernels::grid_1d(groups, grid_limit), args) }
    }

    /// # Safety
    ///
    /// As `Kernels::matmul` for the int8 path.
    #[allow(clippy::too_many_arguments)]
    unsafe fn matmul_int8(
        &self,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        q: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
        grid_limit: u32,
    ) -> Result<()> {
        let Some((_, function)) = self.int8.iter().find(|(t, _)| *t == ty) else {
            let have: Vec<&str> = self.int8.iter().map(|(t, _)| t.name()).collect();
            return Err(Error::invalid(
                "prefill kernel type",
                format!(
                    "{ty} has no tensor-core kernel on this device (it has one for {})",
                    if have.is_empty() { "no type".to_owned() } else { have.join(", ") }
                ),
            ));
        };
        let block_values = ty.block_values();
        if n_in == 0 || !n_in.is_multiple_of(block_values) {
            return Err(Error::invalid(
                "weight matrix",
                format!(
                    "{n_in} columns are not a whole number of {ty} blocks of {block_values} values"
                ),
            ));
        }
        if n_rows == 0 || n_in.checked_mul(n_rows).is_none() {
            return Err(Error::invalid(
                "prefill kernel batch",
                format!("{n_rows} rows of {n_in} values: not 1 or more, or too many"),
            ));
        }
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", w, 16)?;
        check_alignment("quantized activations", q, 16)?;
        let args = Args::new().ptr(w).ptr(q).ptr(y).u32(n_in).u32(n_out).u32(n_rows);
        let groups = u64::from(n_out.div_ceil(TILE)) * u64::from(n_rows.div_ceil(TILE));
        let groups = u32::try_from(groups).map_err(|_| {
            Error::invalid(
                "prefill kernel batch",
                format!("{n_out} x {n_rows} needs too many workgroups"),
            )
        })?;
        // SAFETY: the argument list matches mm_q4_K.comp; the ranges are the caller's guarantee.
        unsafe { function.launch(stream, crate::kernels::grid_1d(groups, grid_limit), args) }
    }
}

/// Whether a cooperative matrix of use A (f16, 16 x 16) holds its elements in the order of an
/// `mma` m16n8k16 A fragment, which `coop_layout_check.comp` finds out by setting them in
/// that order and multiplying by the identity.
fn check_register_operand(device: &Device) -> Result<bool> {
    let stream = Stream::new(device)?;
    let out = DeviceBuffer::new(device, 256 * 4)?;
    let function = Module::load(device, image("coop_layout_check")?)?.function("main")?;
    // SAFETY: `out` holds the 256 floats the kernel writes.
    unsafe { function.launch(&stream, [1, 1, 1], Args::new().ptr(out.ptr()))? };
    stream.synchronize()?;
    let mut bytes = [0u8; 256 * 4];
    out.read(0, &mut bytes)?;
    Ok((0..256).all(|i| {
        f32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]])
            == i as f32
    }))
}

/// Times a loop of f16 products on the tensor cores (`coop_rate_*.comp`, the same loop
/// `vk_coop_rate` times) with f16 and with f32 accumulation, and says whether the second is
/// within a third of the speed of the first. A warm-up of a few milliseconds lets the clocks
/// of an idle card rise first; then the two alternate and the best of each counts. About 40
/// ms at load.
fn measure_full_rate(device: &Device) -> Result<bool> {
    let stream = Stream::new(device)?;
    let sink = DeviceBuffer::new(device, 1 << 20)?;
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);
    let half = Module::load(device, image("coop_rate_f16f16_16x16x16")?)?.function("main")?;
    let single = Module::load(device, image("coop_rate_f16f32_16x16x16")?)?.function("main")?;
    let (groups, iterations) = (2048u32, 512u32);
    let run = |function: &Function| -> Result<f64> {
        let args = Args::new().ptr(sink.ptr()).u32(iterations);
        start.record(&stream)?;
        // SAFETY: the sink is 1 MiB and the kernel writes to it only for a count of
        // iterations that never occurs.
        unsafe { function.launch(&stream, [groups, 1, 1], args)? };
        end.record(&stream)?;
        end.synchronize()?;
        Ok(f64::from(end.elapsed_ms_since(&start)?))
    };
    for _ in 0..4 {
        run(&single)?;
    }
    let (mut best_half, mut best_single) = (f64::MAX, f64::MAX);
    for _ in 0..5 {
        best_half = best_half.min(run(&half)?);
        best_single = best_single.min(run(&single)?);
    }
    Ok(best_single < best_half * 1.3)
}

/// How many activation tiles run together on the same weight tiles: enough that their
/// activations (64 rows of `n_in` f16 each) fill about 8 MiB of L2, between 2 and 16, and a
/// divisor of the number of tiles where one is near.
fn group_of(n_in: u32, n_bt: u32) -> u32 {
    let tile_bytes = u64::from(TILE) * u64::from(n_in) * 2;
    let group = ((8u64 << 20) / tile_bytes).clamp(2, 16) as u32;
    let group = group.min(n_bt);
    (1..=group).rev().find(|g| n_bt.is_multiple_of(*g)).filter(|g| *g * 2 > group).unwrap_or(group)
}
