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

/// The kernel module `build.rs` compiled from `kernels/module.cu`: every GPU
/// the toolkit builds for can load it.
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.fatbin"));

/// The prefill module, compiled from `kernels/prefill.cu` for compute
/// capability 8.0 and newer (the tensor-core weight kernels).
static PREFILL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/prefill.fatbin"));

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

/// The weight types that have a fast kernel (`kernels/mmvq_*.cuh`): the type,
/// the kernel's name in the module, and the alignment in bytes the kernel
/// needs of the matrix (the widest load it uses).
const FAST_TYPES: [(GgmlType, &str, u64); 11] = [
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
];

/// The weight types that have a prefill kernel (`kernels/mm_types.cuh`): the type,
/// the kernels' name suffix (`ares_mm_{i8,i8s,f16,f16s}_<suffix>`) and the
/// alignment in bytes it needs of the matrix (the same as the decode kernel's).
const PREFILL_TYPES: [(GgmlType, &str, u64); 11] = [
    (GgmlType::Q4_K, "q4_K", 16),
    (GgmlType::Q5_K, "q5_K", 16),
    (GgmlType::Q6_K, "q6_K", 2),
    (GgmlType::Q3_K, "q3_K", 2),
    (GgmlType::IQ4_XS, "iq4_xs", 8),
    (GgmlType::IQ4_NL, "iq4_nl", 2),
    (GgmlType::IQ3_S, "iq3_s", 2),
    (GgmlType::IQ3_XXS, "iq3_xxs", 2),
    (GgmlType::IQ2_S, "iq2_s", 2),
    (GgmlType::IQ2_XS, "iq2_xs", 2),
    (GgmlType::Q8_0, "q8_0", 2),
];

/// The shapes of the prefill kernels (`ARES_MM_*_KERNEL` in `kernels/mm.cuh`):
/// weight rows and activation rows per thread block, and threads. The small
/// shape takes up to [`MM_SMALL_ROWS`] activation rows, the large one more.
#[derive(Clone, Copy, Debug)]
struct Shape {
    weight_rows: u32,
    batch_rows: u32,
    threads: u32,
}

const MM_SMALL_ROWS: u32 = 32;
const SHAPES: [[Shape; 2]; 2] = [
    // int8: large, small
    [
        Shape { weight_rows: 128, batch_rows: 64, threads: 256 },
        Shape { weight_rows: 64, batch_rows: 16, threads: 128 },
    ],
    // f16: large, small
    [
        Shape { weight_rows: 64, batch_rows: 64, threads: 128 },
        Shape { weight_rows: 32, batch_rows: 32, threads: 64 },
    ],
];

/// For each weight type, up to how many activation rows the decode kernels (in
/// launches of [`MAX_BATCH`] rows) are the faster family; above it the tensor-core
/// kernels are. See [`Kernels::decode_kernel_rows`].
const DECODE_ROWS: [(GgmlType, u32); 11] = [
    (GgmlType::Q4_K, 16),
    (GgmlType::Q5_K, 16),
    (GgmlType::Q6_K, 8),
    (GgmlType::Q3_K, 6),
    (GgmlType::IQ4_XS, 20),
    (GgmlType::IQ4_NL, 16),
    (GgmlType::IQ3_S, 20),
    (GgmlType::IQ3_XXS, 20),
    (GgmlType::IQ2_S, 8),
    (GgmlType::IQ2_XS, 8),
    (GgmlType::Q8_0, 8),
];

/// The activation bytes the blocks that run together should be able to keep in
/// L2: the grid is ordered in groups of activation tiles that take every weight
/// tile in turn (see `kernels/mm.cuh`), and a group is as many tiles as fit in
/// this many bytes (at least one, at most 16). An RTX 3060 has 3 MiB of L2.
const MM_GROUP_BYTES: u64 = 2 << 20;

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
}

/// What a caller of the prefill family asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// The path this device takes by default ([`Kernels::prefill_path`]): f16 where
    /// the tensor cores run it at full rate, since the direct product is then
    /// both the more exact and, in principle, as fast, and int8 where they do
    /// not (GeForce cards with the cap, where int8 is the faster of the two).
    ///
    /// The rate of an uncapped card could not be measured on the machine the
    /// kernels were written on (both its GPUs are capped). The f16 kernels were
    /// timed with f16 accumulation instead, which runs at the rate an uncapped
    /// card has for f32 accumulation: they take the same time as with f32
    /// accumulation, so the cap is not what limits them, and int8 stays 20 to 30%
    /// faster from 256 rows on (an RTX 4070, 17408 x 5120 Q4_K) while f16 is
    /// about 10% faster at 64 rows. Whoever prefers speed on such a card
    /// can take [`PrefillPath::Int8`] by choosing the path explicitly.
    #[default]
    Fast,
    /// Always the f16 path with f32 accumulation. More exact than int8, and on a
    /// GeForce card whose tensor cores are capped (the case until the cap is
    /// lifted) slower.
    Exact,
}

/// The kernels of the prefill module on one device.
#[derive(Debug, Clone)]
struct Prefill {
    /// Per type: the alignment and the kernels, indexed `[path][small]`.
    types: Vec<(GgmlType, u64, [[Function; 2]; 2])>,
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
    /// `None` on a GPU below compute capability 8.0.
    prefill: Option<Prefill>,
}

impl Prefill {
    fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, PREFILL_IMAGE)?;
        let types = PREFILL_TYPES
            .iter()
            .map(|(ty, suffix, align)| {
                let function =
                    |variant: &str| module.function(&format!("ares_mm_{variant}_{suffix}"));
                let kernels =
                    [[function("i8")?, function("i8s")?], [function("f16")?, function("f16s")?]];
                Ok((*ty, *align, kernels))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            types,
            convert_f16: module.function("ares_convert_f16")?,
            full_rate: probe_full_rate(device, &module)?,
        })
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
    /// values. See `kernels/quantize_act.cuh` for the format.
    pub fn quantized_activation_bytes(n_in: u32, n_batch: u32) -> u64 {
        let values = u64::from(n_in) * u64::from(n_batch);
        values.next_multiple_of(16) + values / 32 * 8
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
    /// `n_in` must be a multiple of 32. `q` receives
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
        if n_in == 0 || !n_in.is_multiple_of(32) {
            return Err(Error::Invalid {
                what: "activation quantization",
                detail: format!("{n_in} values per row are not a whole number of groups of 32"),
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
    /// This is the decode kernel: it computes with integer dot products over
    /// the quantized activations, so its results differ slightly from the
    /// exact f32 kernel ([`Kernels::matvec_q4_k`] for Q4_K, which stays the
    /// reference). A type without a fast kernel (see
    /// [`Kernels::has_fast_matvec`]) is refused with an error.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `w` must
    /// address the matrix as GGUF stores it (`n_out` rows of `n_in` values in
    /// whole blocks of `ty`), aligned to 16 bytes for Q4_K and Q5_K, to 8 for
    /// IQ4_XS and to 2 for the other types (checked); `q` the quantized
    /// activations of this `n_in` and `n_batch`, 16-byte aligned (checked);
    /// `y` `n_batch * n_out` f32 values. `y` must not overlap the inputs.
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
        let args = Args::new().ptr(w).ptr(q).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
        // SAFETY: the argument list matches the ares_mmvq_* kernels; the ranges are the
        // caller's guarantee.
        unsafe {
            function.launch(
                stream,
                [n_out.div_ceil(MMVQ_ROWS_PER_BLOCK), 1, 1],
                [MMVQ_THREADS, 1, 1],
                0,
                args,
            )
        }
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
    /// by a launch with every 8 rows; the tensor-core kernels take about the
    /// same time for any number of rows up to 64, since they read the weights once
    /// and the time is the weights' decoding. They cross at 16 to 20 rows for the
    /// types whose decode kernels reach the memory bandwidth (Q4_K, Q5_K, IQ4_XS,
    /// IQ4_NL, IQ3_S, IQ3_XXS: 12 to 24 on an RTX 4070 and 12 to 28 on an RTX
    /// 3060, as the first row count at which the tensor-core kernels are faster
    /// in `examples/prefill_bench.rs`) and at 8 or fewer for the types whose
    /// decode kernels are arithmetic-bound already at a few rows (Q3_K, Q6_K,
    /// IQ2_S, IQ2_XS) and for Q8_0 (the 3060 crosses at 12, the 4070 at 20). The
    /// table holds, per type, the row count that gives the least total loss over
    /// both GPUs and 8, 12, ..., 32 rows: following it costs at most 27% against
    /// the faster family at any of those row counts (IQ4_NL at 16 rows), at most
    /// 15% for the other types. It was measured with the int8 path of the
    /// prefill kernels, the one both GPUs take; the f16 path of a device whose
    /// tensor cores run at full rate has its own crossover, not measured.
    pub fn decode_kernel_rows(ty: GgmlType) -> Option<u32> {
        DECODE_ROWS.iter().find(|(t, _)| *t == ty).map(|(_, rows)| *rows)
    }

    /// Whether this device has the prefill kernels at all: they need compute
    /// capability 8.0 (Ampere) or newer.
    pub fn has_prefill_kernels(&self) -> bool {
        self.prefill.is_some()
    }

    /// Whether [`Kernels::matmul`] has a kernel for weights of type `ty` (the
    /// types of [`Kernels::has_fast_matvec`]; the device must also have the
    /// prefill kernels, see [`Kernels::has_prefill_kernels`]).
    pub fn has_prefill_matmul(ty: GgmlType) -> bool {
        PREFILL_TYPES.iter().any(|(t, ..)| *t == ty)
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

    /// The bytes [`Kernels::prepare_prefill_activations`] writes for `n_rows`
    /// rows of `n_in` activations in the format of `path`.
    pub fn prefill_activation_bytes(path: PrefillPath, n_in: u32, n_rows: u32) -> u64 {
        match path {
            PrefillPath::Int8 => Self::quantized_activation_bytes(n_in, n_rows),
            PrefillPath::F16 => u64::from(n_in) * u64::from(n_rows) * 2,
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
    /// (checked), not overlapping `x`.
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
            PrefillPath::F16 => {
                check_alignment("prefill activations", out, 16)?;
                let n_values = n_in.checked_mul(n_rows).ok_or(Error::Invalid {
                    what: "prefill activations",
                    detail: "size overflows u32".into(),
                })?;
                let args = Args::new().ptr(x).ptr(out).u32(n_values);
                // SAFETY: the argument list matches ares_convert_f16; the ranges are the
                // caller's guarantee.
                unsafe {
                    prefill.convert_f16.launch(
                        stream,
                        [n_values.div_ceil(256), 1, 1],
                        [256, 1, 1],
                        0,
                        args,
                    )
                }
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
    /// Both paths accumulate in f32 and agree with the decode kernels up to the
    /// rounding of what they compute with: int8 with the decode kernels'
    /// quantization of the activations, f16 with the rounding of weights and
    /// activations to f16. A type without a prefill kernel is refused with an
    /// error, as is a device below compute capability 8.0.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `w` must
    /// address the matrix as GGUF stores it (`n_out` rows of `n_in` values in
    /// whole blocks of `ty`), aligned to 16 bytes for Q4_K and Q5_K, to 8 for
    /// IQ4_XS and to 2 for the other types (checked); `act` the activations
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
        let Some((_, align, kernels)) = prefill.types.iter().find(|(t, ..)| *t == ty) else {
            return Err(Error::Invalid {
                what: "prefill kernel type",
                detail: format!(
                    "{ty} has no prefill kernel (it has one for {})",
                    PREFILL_TYPES.map(|(t, ..)| t.name()).join(", ")
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
        let (path_index, act_row_bytes) = match path {
            PrefillPath::Int8 => (0, u64::from(n_in) * 9 / 8),
            PrefillPath::F16 => (1, u64::from(n_in) * 2),
        };
        let small = usize::from(n_rows <= MM_SMALL_ROWS);
        let Shape { weight_rows, batch_rows, threads } = SHAPES[path_index][small];
        let function = &kernels[path_index][small];
        let n_weight_tiles = n_out.div_ceil(weight_rows);
        let n_batch_tiles = n_rows.div_ceil(batch_rows);
        let group = (MM_GROUP_BYTES / (u64::from(batch_rows) * act_row_bytes)).clamp(1, 16);
        let group = u32::try_from(group).unwrap_or(16).min(n_batch_tiles);
        let blocks = u64::from(n_weight_tiles) * u64::from(n_batch_tiles);
        let blocks = u32::try_from(blocks).map_err(|_| Error::Invalid {
            what: "prefill kernel batch",
            detail: format!("{n_out} x {n_rows} needs more blocks than a grid has"),
        })?;
        let args = Args::new().ptr(w).ptr(act).ptr(y).u32(n_in).u32(n_out).u32(n_rows).u32(group);
        // SAFETY: the argument list matches the ares_mm_* kernels; the ranges are the
        // caller's guarantee.
        unsafe { function.launch(stream, [blocks, 1, 1], [threads, 1, 1], 0, args) }
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
