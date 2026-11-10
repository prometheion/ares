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
use crate::stream::Stream;

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
}

/// The kernels of the engine on one device.
#[derive(Clone)]
pub struct Kernels(Arc<Set>);

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernels(on {:?})", self.0.device)
    }
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
    /// Compiles the kernels for `device`: one pipeline per kernel, and one per
    /// workgroup size for the matrix kernel. The driver compiles the shaders
    /// for the GPU here, which takes a moment; do it once per device.
    pub fn load(device: &Device) -> Result<Self> {
        let main = |image: &[u8]| Module::load(device, image)?.function("main");
        let matvec = Module::load(device, MATVEC_Q4_K)?;
        let matvec_q4_k = (1..=MATVEC_SIZES as u32)
            .map(|i| matvec.function_with("main", &[(0, 32 * i)]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(Arc::new(Set {
            device: device.clone(),
            add_f32: main(ADD_F32)?,
            rms_norm_f32: main(RMS_NORM_F32)?,
            sum32_f32: main(SUM32_F32)?,
            matvec_q4_k,
        })))
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    fn grid(&self, groups: u32) -> [u32; 3] {
        grid_1d(groups, self.0.device.physical().max_work_group_count[0])
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
