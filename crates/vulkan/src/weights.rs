//! The quantized weight kernels: activations quantized to int8, then a quantized
//! matrix times 1 to 8 rows of them with integer dot products on the weight codes.
//! The counterpart of the CUDA layer's `Kernels::quantize_activations` and
//! `Kernels::matvec`, with the same arithmetic; `shaders/mmvq.glsl` and
//! `shaders/mmvq_*.glsl` are the kernels.

use ares_core::GgmlType;

use std::sync::Arc;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::MAX_BATCH;
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Function, Lazy, Module};
use crate::stream::Stream;

// The shape of the launch of every weight kernel is `MMVQ_THREADS` threads and
// `MMVQ_WARPS` times the rows per warp of the shader's `Shape` output rows (build.rs hands
// the same numbers to the shaders): as in CUDA, a workgroup of 4 warps.
include!(concat!(env!("OUT_DIR"), "/shader_constants.rs"));

// The weight kernels, in the variants build.rs made of them (`Shape` there).
include!(concat!(env!("OUT_DIR"), "/shader_images.rs"));

/// The SPIR-V of the shader `name` (the first variant, for the shaders that have several).
pub(crate) fn image(name: &str) -> Result<&'static [u8]> {
    IMAGES
        .iter()
        .find(|i| i.name == name)
        .and_then(|i| i.variants.first().copied())
        .ok_or_else(|| Error::NoSuchKernel(name.to_owned()))
}

macro_rules! spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")) as &[u8]
    };
}

static QUANTIZE_ACT: &[u8] = spv!("quantize_act");

/// The large codebooks of the i-quants, in the layout of `shaders/iq_tables.glsl`
/// (generated from the CPU decoder's tables by `shaders/gen_iq_tables.py`). They are
/// uploaded once per device and the kernels copy what they need to shared memory.
static IQ_TABLES: &[u8] = include_bytes!("../shaders/iq_tables.bin");

/// How a kernel reads the activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Acts {
    /// int8 with a scale per 32 values (`quantize_activations`).
    Quantized,
    /// f32 as they are (the float weight types).
    Float,
}

/// The weight types that have a kernel: the type, the name of its shader, the alignment in bytes the
/// kernel needs of the matrix (the widest load it uses) and how it reads the activations.
/// Every type a GGUF file can carry (the 28 that `ares-quant` decodes) but the integer
/// ones and f64, which are not weights.
const FAST_TYPES: [(GgmlType, &str, u64, Acts); 28] = [
    (GgmlType::Q4_K, "mmvq_q4_K", 16, Acts::Quantized),
    (GgmlType::Q5_K, "mmvq_q5_K", 16, Acts::Quantized),
    (GgmlType::Q6_K, "mmvq_q6_K", 2, Acts::Quantized),
    (GgmlType::Q3_K, "mmvq_q3_K", 2, Acts::Quantized),
    (GgmlType::Q2_K, "mmvq_q2_K", 4, Acts::Quantized),
    (GgmlType::IQ4_XS, "mmvq_iq4_xs", 8, Acts::Quantized),
    (GgmlType::IQ4_NL, "mmvq_iq4_nl", 2, Acts::Quantized),
    (GgmlType::IQ3_S, "mmvq_iq3_s", 2, Acts::Quantized),
    (GgmlType::IQ3_XXS, "mmvq_iq3_xxs", 2, Acts::Quantized),
    (GgmlType::IQ2_S, "mmvq_iq2_s", 2, Acts::Quantized),
    (GgmlType::IQ2_XS, "mmvq_iq2_xs", 2, Acts::Quantized),
    (GgmlType::IQ2_XXS, "mmvq_iq2_xxs", 2, Acts::Quantized),
    (GgmlType::IQ1_S, "mmvq_iq1_s", 2, Acts::Quantized),
    (GgmlType::IQ1_M, "mmvq_iq1_m", 4, Acts::Quantized),
    (GgmlType::Q8_0, "mmvq_q8_0", 2, Acts::Quantized),
    (GgmlType::Q5_1, "mmvq_q5_1", 4, Acts::Quantized),
    (GgmlType::Q5_0, "mmvq_q5_0", 2, Acts::Quantized),
    (GgmlType::Q4_1, "mmvq_q4_1", 4, Acts::Quantized),
    (GgmlType::Q4_0, "mmvq_q4_0", 2, Acts::Quantized),
    (GgmlType::TQ1_0, "mmvq_tq1_0", 2, Acts::Quantized),
    (GgmlType::TQ2_0, "mmvq_tq2_0", 2, Acts::Quantized),
    (GgmlType::MXFP4, "mmvq_mxfp4", 1, Acts::Quantized),
    (GgmlType::NVFP4, "mmvq_nvfp4", 4, Acts::Quantized),
    (GgmlType::Q1_0, "mmvq_q1_0", 2, Acts::Quantized),
    (GgmlType::Q2_0, "mmvq_q2_0", 2, Acts::Quantized),
    (GgmlType::F32, "mmvf_f32", 16, Acts::Float),
    (GgmlType::F16, "mmvf_f16", 16, Acts::Float),
    (GgmlType::BF16, "mmvf_bf16", 16, Acts::Float),
];

/// The kernel of one weight type on one device.
#[derive(Debug, Clone)]
struct Weight {
    ty: GgmlType,
    /// The alignment in bytes the kernel needs of the matrix.
    align: u64,
    acts: Acts,
    /// Output rows one workgroup computes.
    rows_per_group: u32,
    /// One pipeline, or one for each number of activation rows (1 to 8).
    functions: Vec<Arc<Lazy>>,
}

/// The kernels of this module on one device.
#[derive(Debug, Clone)]
pub(crate) struct Quantized {
    quantize_act: Function,
    types: Vec<Weight>,
    /// The codebooks (`IQ_TABLES`) in device memory.
    tables: Arc<DeviceBuffer>,
}

/// The bytes that `quantize_activations` writes for `n_batch` rows of `n_in`
/// activations: the int8 values, then (from the next multiple of 16 bytes) one
/// `(scale, sum)` pair of f32 per group of 32 values.
pub(crate) fn quantized_activation_bytes(n_in: u32, n_batch: u32) -> u64 {
    let values = u64::from(n_in) * u64::from(n_batch);
    values.next_multiple_of(16) + values / 32 * 8
}

pub(crate) fn has_fast_matvec(ty: GgmlType) -> bool {
    FAST_TYPES.iter().any(|(t, ..)| *t == ty)
}

/// Whether the kernel of `ty` reads quantized activations (every type but the float ones).
pub(crate) fn needs_quantized_activations(ty: GgmlType) -> bool {
    FAST_TYPES.iter().any(|(t, _, _, acts)| *t == ty && *acts == Acts::Quantized)
}

fn check_batch(n_batch: u32) -> Result<()> {
    if n_batch == 0 || n_batch > MAX_BATCH {
        return Err(Error::invalid(
            "weight kernel batch",
            format!("{n_batch} activation rows; one launch takes 1 to {MAX_BATCH}"),
        ));
    }
    Ok(())
}

pub(crate) fn check_alignment(what: &'static str, ptr: DevicePtr, align: u64) -> Result<()> {
    if !ptr.as_raw().is_multiple_of(align) {
        return Err(Error::invalid(
            what,
            format!("address {:#x} is not aligned to {align} bytes", ptr.as_raw()),
        ));
    }
    Ok(())
}

impl Quantized {
    pub(crate) fn load(device: &Device) -> Result<Self> {
        let main = |image: &[u8]| Module::load(device, image)?.function("main");
        // The weight kernels are made on first use (`Lazy`), or ahead of it by `prepare`.
        let types = FAST_TYPES
            .iter()
            .map(|(ty, name, align, acts)| {
                let image = IMAGES.iter().find(|i| i.name == *name).expect("a shader per type");
                Weight {
                    ty: *ty,
                    align: *align,
                    acts: *acts,
                    rows_per_group: MMVQ_WARPS * image.rows,
                    functions: image.variants.iter().map(|v| Lazy::new(device, v)).collect(),
                }
            })
            .collect();
        let tables = DeviceBuffer::new(device, IQ_TABLES.len() as u64)?;
        tables.write(0, IQ_TABLES)?;
        Ok(Self { quantize_act: main(QUANTIZE_ACT)?, types, tables: Arc::new(tables) })
    }

    /// Makes the decode kernels of `types` now, on all cores, so that no step pays for it.
    pub(crate) fn prepare(&self, types: &[GgmlType]) -> Result<()> {
        let wanted: Vec<&Weight> = self.types.iter().filter(|w| types.contains(&w.ty)).collect();
        let all: Vec<&Arc<Lazy>> = wanted.iter().flat_map(|w| w.functions.iter()).collect();
        crate::kernels::parallel_map(&all, |lazy| lazy.get().map(|_| ())).map(|_| ())
    }

    /// The device address of the codebooks.
    pub(crate) fn tables(&self) -> DevicePtr {
        self.tables.ptr()
    }

    /// # Safety
    ///
    /// As `Kernels::quantize_activations`.
    pub(crate) unsafe fn quantize_activations(
        &self,
        stream: &Stream,
        x: DevicePtr,
        q: DevicePtr,
        n_in: u32,
        n_batch: u32,
        grid_limit: u32,
    ) -> Result<()> {
        check_batch(n_batch)?;
        // SAFETY: the caller's guarantee is the one of the function it forwards to.
        unsafe { self.quantize_rows(stream, x, q, n_in, n_batch, grid_limit) }
    }

    /// The quantization for any number of rows.
    ///
    /// # Safety
    ///
    /// As `Kernels::quantize_activation_rows`.
    pub(crate) unsafe fn quantize_rows(
        &self,
        stream: &Stream,
        x: DevicePtr,
        q: DevicePtr,
        n_in: u32,
        n_batch: u32,
        grid_limit: u32,
    ) -> Result<()> {
        if n_batch == 0 {
            return Err(Error::invalid("activation quantization", "no rows"));
        }
        if n_in == 0 || !n_in.is_multiple_of(32) {
            return Err(Error::invalid(
                "activation quantization",
                format!("{n_in} values per row are not a whole number of groups of 32"),
            ));
        }
        check_alignment("quantized activations", q, 16)?;
        let n_values = n_in
            .checked_mul(n_batch)
            .ok_or_else(|| Error::invalid("activation quantization", "size overflows u32"))?;
        let args = Args::new().ptr(x).ptr(q).u32(n_values);
        // One invocation per 4 values, 256 to a workgroup.
        let groups = n_values.div_ceil(4).div_ceil(256);
        // SAFETY: the argument list matches quantize_act.comp; the ranges are the
        // caller's guarantee.
        unsafe {
            self.quantize_act.launch(stream, crate::kernels::grid_1d(groups, grid_limit), args)
        }
    }

    /// # Safety
    ///
    /// As `Kernels::matvec`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn matvec(
        &self,
        stream: &Stream,
        ty: GgmlType,
        w: DevicePtr,
        q: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_batch: u32,
        grid_limit: u32,
    ) -> Result<()> {
        let Some(Weight { align, acts, rows_per_group, functions, .. }) =
            self.types.iter().find(|w| w.ty == ty)
        else {
            return Err(Error::invalid(
                "weight kernel type",
                format!(
                    "{ty} has no fast weight kernel (it has one for {})",
                    FAST_TYPES.map(|(t, ..)| t.name()).join(", ")
                ),
            ));
        };
        // A float matrix has blocks of one value; its kernel reads 8 at a time.
        let block_values = if *acts == Acts::Float { 8 } else { ty.block_values() };
        if n_in == 0 || !n_in.is_multiple_of(block_values) {
            return Err(Error::invalid(
                "weight matrix",
                format!(
                    "{n_in} columns are not a whole number of {ty} blocks of {block_values} values"
                ),
            ));
        }
        check_batch(n_batch)?;
        let function =
            functions[if functions.len() > 1 { n_batch as usize - 1 } else { 0 }].get()?;
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", w, *align)?;
        check_alignment("quantized activations", q, 16)?;
        // The float kernels need no codebook.
        let args = if *acts == Acts::Float {
            Args::new().ptr(w).ptr(q).ptr(y)
        } else {
            Args::new().ptr(w).ptr(q).ptr(y).ptr(self.tables.ptr())
        };
        let args = args.u32(n_in).u32(n_out).u32(n_batch);
        let groups = n_out.div_ceil(*rows_per_group);
        // SAFETY: the argument list matches the mmvq_* shaders; the ranges are the
        // caller's guarantee.
        unsafe { function.launch(stream, crate::kernels::grid_1d(groups, grid_limit), args) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers of the table `name` of `ares-quant`, read from its source.
    fn table(source: &str, name: &str) -> Vec<u64> {
        let start = source.find(&format!("static {name}:")).unwrap_or_else(|| panic!("{name}"));
        let body = &source[start..];
        let body = &body[body.find("= [").unwrap() + 3..];
        let body = &body[..body.find("];").unwrap()];
        body.split(|c: char| !c.is_ascii_hexdigit() && c != 'x')
            .filter(|t| t.starts_with("0x"))
            .map(|t| u64::from_str_radix(&t[2..], 16).unwrap())
            .collect()
    }

    /// The uploaded codebooks are the CPU decoder's, so a change of `tables.rs` without
    /// a run of `gen_iq_tables.py` fails here.
    #[test]
    fn the_codebooks_are_those_of_the_cpu_decoder() {
        let source = include_str!("../../quant/src/tables.rs");
        let mut at = 0usize;
        for (name, width) in [
            ("IQ2XXS_GRID", 8usize),
            ("IQ2XS_GRID", 8),
            ("IQ2S_GRID", 8),
            ("IQ1S_GRID", 2),
            ("IQ3XXS_GRID", 4),
            ("IQ3S_GRID", 4),
        ] {
            let values = table(source, name);
            for (i, v) in values.iter().enumerate() {
                let bytes = &IQ_TABLES[at + i * width..][..width];
                let mut le = [0u8; 8];
                le[..width].copy_from_slice(bytes);
                let stored = u64::from_le_bytes(le);
                if name == "IQ1S_GRID" {
                    // 2 bits per value: the value + 1 (the bytes are -1, 0, 1).
                    for j in 0..8 {
                        let byte = (*v >> (8 * j)) as u8 as i8;
                        assert_eq!(
                            i64::from(byte) + 1,
                            ((stored >> (2 * j)) & 3) as i64,
                            "{name}[{i}][{j}]"
                        );
                    }
                } else {
                    assert_eq!(stored, *v, "{name}[{i}]");
                }
            }
            at = (at + values.len() * width).next_multiple_of(16);
        }
        assert_eq!(at, IQ_TABLES.len());
        // The offsets the shaders use.
        let glsl = include_str!("../shaders/iq_tables.glsl");
        assert!(glsl.contains("TBL_IQ2XS = 2048;") && glsl.contains("TBL_IQ3S = 19456;"));
    }

    #[test]
    fn the_activation_format_has_the_size_of_the_cuda_one() {
        // values rounded up to 16, then 8 bytes per group of 32.
        assert_eq!(quantized_activation_bytes(32, 1), 32 + 8);
        assert_eq!(quantized_activation_bytes(256, 3), 768 + 24 * 8);
        assert_eq!(quantized_activation_bytes(5120, 8), 40960 + 1280 * 8);
        // 96 values: 96 is a multiple of 16; the pairs start at 96.
        assert_eq!(quantized_activation_bytes(96, 1), 96 + 3 * 8);
    }

    #[test]
    fn the_types_are_those_of_the_cuda_layer() {
        let names: Vec<&str> = FAST_TYPES.iter().map(|(t, ..)| t.name()).collect();
        assert_eq!(names.len(), 28);
        for ty in [
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q3_K,
            GgmlType::IQ4_XS,
            GgmlType::IQ4_NL,
            GgmlType::IQ3_S,
            GgmlType::IQ3_XXS,
            GgmlType::IQ2_S,
            GgmlType::IQ2_XS,
            GgmlType::Q8_0,
        ] {
            assert!(has_fast_matvec(ty), "{ty}");
        }
        assert!(!has_fast_matvec(GgmlType::F64));
        assert!(!has_fast_matvec(GgmlType::Q8_K));
        assert!(needs_quantized_activations(GgmlType::Q4_K));
        assert!(!needs_quantized_activations(GgmlType::F32));
        // The widest load of each type fits the block geometry the kernel assumes.
        for (ty, _, align, acts) in FAST_TYPES {
            let block = u64::from(ty.block_bytes());
            assert!(align <= 16 && (acts == Acts::Float || block % align == 0), "{ty}");
        }
        // Every weight type of a GGUF file that is not an integer or f64.
        for ty in ares_core::ggml_type::ALL_TYPES {
            assert_eq!(has_fast_matvec(ty), ty.is_weight(), "{ty}");
        }
    }
}
