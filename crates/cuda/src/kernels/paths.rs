//! Which path of the tile kernels ([`Kernels::matmul`])
//! [`Precision::Fast`](super::Precision::Fast) takes for each weight type, measured per type
//! and device when a model is loaded ([`Kernels::measure_prefill_path`]).
//!
//! The paths read the same weights in the file's layout and differ in what the tensor
//! cores multiply: int8 codes with 8-bit activations, or the dequantised weights as f16
//! with f16 activations, summed in f32 (the exact product of the f16 operands) or in f16
//! flushed to f32 every 64 values. On a GeForce card with the cap on f32 accumulation the
//! exact f16 path runs at half the rate of the other two, and of those int8 is the fast one
//! for most types; where the cap is lifted (or on a professional card) the f16 path costs
//! the tensor cores no more than int8, but it moves twice the bytes through the load unit
//! (its tiles and activations hold two bytes a value), which the loads of the file's layout
//! keep busy (its blocks lie a row apart, so a load of a warp touches 16 lines): the types
//! with many load instructions per item fall behind on it (an RTX 4060 Ti at 1024 rows:
//! IQ3_S 23 %, IQ3_XXS 15 %, Q8_0 12 %, IQ4_NL 10 %, IQ4_XS 9 %), the others are as fast or
//! faster (`kernels/mm.cuh`). Which is which depends on the type and the card, capped or
//! not, so it is measured. A type that was not measured takes the rule kept from the cards
//! the kernels were written on ([`Kernels::prefill_path_for`]).

use std::sync::{Arc, Mutex};

use ares_core::GgmlType;

use super::{Kernels, PrefillPath};
use crate::error::{Error, Result};
use crate::memory::DeviceBuffer;
use crate::stream::{Event, Stream};

/// The paths in the order of their precision: the exact product of the f16 operands, the
/// f16 sums flushed to f32 (0.05 % rms from the exact product), the 8-bit activations
/// (1.1 %). Of the paths that are as fast as the fastest the first is taken.
const BY_PRECISION: [PrefillPath; 3] = [PrefillPath::F16, PrefillPath::F16Flush, PrefillPath::Int8];

/// A path within this factor of the fastest is as fast: the probe's noise is a few
/// percent, and a difference of that size is not worth the precision.
const AS_FAST_WITHIN: f32 = 1.05;

/// Which path [`Precision::Fast`](super::Precision::Fast) takes for one weight type on one
/// device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefillChoice {
    pub ty: GgmlType,
    /// Whether the path was measured on this device ([`Kernels::measure_prefill_path`]);
    /// otherwise it is the kept rule of [`Kernels::prefill_path_for`], for a type that was
    /// not measured.
    pub measured: bool,
    pub path: PrefillPath,
    /// The time of the probe product with each path, `[int8, f16, f16 flushed]`, in
    /// milliseconds: infinite for a path the type has no kernel of, 0 where nothing was
    /// measured.
    pub probe_ms: [f32; 3],
}

/// The choices measured so far on one device.
pub(super) type Choices = Arc<Mutex<Vec<PrefillChoice>>>;

impl Kernels {
    /// The path [`Precision::Fast`](super::Precision::Fast) takes for matrices of `ty` in the
    /// file's layout: the one [`Kernels::measure_prefill_path`] measured, or the kept rule of
    /// [`Kernels::prefill_path_for`].
    pub fn prefill_choice(&self, ty: GgmlType) -> Result<PrefillChoice> {
        let prefill = self.prefill_kernels()?;
        let choices = prefill.choices.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(choice) = choices.iter().find(|c| c.ty == ty) {
            return Ok(*choice);
        }
        drop(choices);
        Ok(PrefillChoice { ty, measured: false, path: self.kept_path(ty)?, probe_ms: [0.0; 3] })
    }

    /// Measures which path of the tile kernels is fastest for weights of `ty` on this
    /// device and keeps it for [`Kernels::prefill_path_for`]: of the paths within 5 % of
    /// the fastest the most exact. A loader calls it once per type its model multiplies
    /// in the file's layout, before the weights are placed; it returns the choice, and
    /// the same without measuring again if it has been measured. It takes a probe matrix
    /// with rows of 4096 values larger than the L2 cache (up to 96 MiB of the device's
    /// memory) and up to 60 MiB for activations and products, freed on return, and 50 to
    /// 150 ms.
    ///
    /// Every card is measured, capped ([`Kernels::tensor_cores_full_rate`]) or not: on a
    /// capped card the exact f16 path runs at half the rate, and which of the other two is
    /// as fast depends on the type and the card as the f16 path does on a card without the
    /// cap.
    ///
    /// The device should be idle: a product of another context that runs at the same
    /// time is measured into the choice.
    pub fn measure_prefill_path(&self, ty: GgmlType) -> Result<PrefillChoice> {
        let known = self.prefill_choice(ty)?;
        if known.measured {
            return Ok(known);
        }
        let probe_ms = self.probe_paths(ty)?;
        let fastest = probe_ms.iter().copied().fold(f32::INFINITY, f32::min);
        let index = |path: PrefillPath| match path {
            PrefillPath::Int8 => 0,
            PrefillPath::F16 => 1,
            PrefillPath::F16Flush => 2,
        };
        let path = BY_PRECISION
            .into_iter()
            .find(|p| probe_ms[index(*p)] <= fastest * AS_FAST_WITHIN)
            .unwrap_or(known.path);
        let choice = PrefillChoice { ty, measured: true, path, probe_ms };
        let prefill = self.prefill_kernels()?;
        let mut choices = prefill.choices.lock().unwrap_or_else(|e| e.into_inner());
        choices.retain(|c| c.ty != ty);
        choices.push(choice);
        Ok(choice)
    }

    /// The time of a product of the probe matrix of `ty` with each path,
    /// `[int8, f16, f16 flushed]`: infinite where the type has no kernel.
    fn probe_paths(&self, ty: GgmlType) -> Result<[f32; 3]> {
        /// The values of a row of the probe matrix: the width of a model's matrices
        /// (2560 to 6144 for most of them). It sets the bytes of an activation tile and so
        /// how many tiles a group of the grid holds; with rows of 14080 values the groups
        /// of the f16 path were half those of the int8 path, and its Q5_K 17 % behind
        /// int8 on an RTX 4060 Ti, where with 5120 it is as fast.
        const COLUMNS: u32 = 4096;
        /// The probe matrix is larger than L2, so that the kernels read their weights
        /// from memory as they do from a model's matrices.
        const MIN_BYTES: u64 = 16 << 20;
        const MAX_BYTES: u64 = 96 << 20;
        /// The most units of weight rows (below) the probe matrix has, which bounds its
        /// products: for the types of the fewest bits it stays below L2 then (4 x 34 x 128
        /// rows on an RTX 4060 Ti, as many as Qwen3.8-27B's largest matrix).
        const MAX_UNITS: u64 = 4;
        /// The rows of activations: a prompt step of the engine's default, four tiles of
        /// the large shape that run in a group. (With two tiles the f16 path of Q5_K and
        /// Q6_K was 4 to 9 % behind int8 on an RTX 4060 Ti, from four on as fast.)
        const ROWS: u32 = 512;
        /// Every path is timed this many times, in turn with the others, as a run of
        /// `BURST` launches; its fastest run counts. The first round warms the clocks
        /// and the caches up and is not counted.
        const ROUNDS: usize = 4;
        const BURST: u32 = 4;
        /// The bytes of random data the weights are made of, repeated.
        const PATTERN: usize = 1 << 20;

        let device = &self.device;
        let info = device.info();
        let paths: Vec<PrefillPath> =
            BY_PRECISION.into_iter().filter(|p| self.has_prefill_path(ty, *p)).collect();
        if paths.is_empty() {
            return Err(Error::Invalid {
                what: "prefill kernel type",
                detail: format!("{ty} has no prefill kernel"),
            });
        }
        // The weight rows: enough that the matrix is larger than L2, in whole waves of
        // blocks of the large shape, so that the probe times the kernel and not a last wave
        // of a few blocks.
        let mut held = 1;
        for path in &paths {
            let large = self.prefill_shapes(ty, *path)?[0];
            held = held.max(large.blocks_per_sm);
        }
        let n_in = COLUMNS;
        let row_bytes = |n_in: u32| ty.bytes_for(u64::from(n_in)).unwrap_or(u64::MAX);
        let want = (u64::from(info.l2_cache_bytes) * 5 / 4).clamp(MIN_BYTES, MAX_BYTES);
        // A unit is a tile of 128 weight rows per block the device holds at once: with any
        // number of activation tiles, the blocks are whole waves.
        let unit = info.multiprocessors * held * 128;
        let units = want.div_ceil(u64::from(unit) * row_bytes(n_in)).clamp(1, MAX_UNITS);
        let n_out = unit * units as u32;
        let w_bytes = u64::from(n_out) * row_bytes(n_in);
        let w = DeviceBuffer::new(device, w_bytes)?;
        let stream = Stream::new(device, 0)?;
        // Random bytes: the codebook types look their values up in shared memory, where
        // equal indices would meet no bank conflicts, and data that is all alike draws
        // less power than weights do (the clocks of a card at its power limit follow).
        // Scales may be infinite or not numbers; no kernel takes longer for that.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let pattern: Vec<u8> = (0..PATTERN.min(w_bytes as usize))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect();
        w.write(0, &pattern)?;
        let mut filled = pattern.len() as u64;
        while filled < w_bytes {
            let len = filled.min(w_bytes - filled);
            // SAFETY: both ranges are inside `w` and do not overlap.
            unsafe { stream.copy(w.ptr().offset(filled), w.ptr(), len as usize)? };
            filled += len;
        }
        let y = DeviceBuffer::new(device, u64::from(ROWS) * u64::from(n_out) * 4)?;
        let mut acts = Vec::with_capacity(paths.len());
        for path in &paths {
            let bytes = Self::prefill_activation_bytes(*path, n_in, ROWS);
            let act = DeviceBuffer::new(device, bytes)?;
            // Numbers in either form, all of the byte 0x3c: f16 values of 1.06, int8 codes
            // of 60, f32 factors and scales of 0.0115. No kernel's time depends on them.
            // SAFETY: the whole buffer of this device.
            unsafe { stream.fill(act.ptr(), 0x3c, bytes as usize)? };
            acts.push(act);
        }
        stream.synchronize()?;
        let (start, end) = (Event::timed(device)?, Event::timed(device)?);
        let mut best = [f32::INFINITY; 3];
        for round in 0..=ROUNDS {
            for (path, act) in paths.iter().zip(&acts) {
                start.record(&stream)?;
                for _ in 0..BURST {
                    // SAFETY: `w` holds a matrix of n_out rows of n_in values of `ty`,
                    // `act` ROWS rows in the form of `path` and `y` room for the products.
                    unsafe {
                        self.matmul(
                            &stream,
                            ty,
                            *path,
                            w.ptr(),
                            act.ptr(),
                            y.ptr(),
                            n_in,
                            n_out,
                            ROWS,
                        )?
                    };
                }
                end.record(&stream)?;
                end.synchronize()?;
                if round > 0 {
                    let slot = match path {
                        PrefillPath::Int8 => 0,
                        PrefillPath::F16 => 1,
                        PrefillPath::F16Flush => 2,
                    };
                    best[slot] = best[slot].min(end.elapsed_ms_since(&start)? / BURST as f32);
                }
            }
        }
        Ok(best)
    }

    /// The path of a type that was not measured, by the rule kept from the cards the kernels
    /// were written on: f16 where the tensor cores run it at full rate, and on a capped card
    /// f16 with f16 accumulation for the types where that was as fast as int8 there or that
    /// have no int8 kernel, int8 for the others. A model's loader measures every type it
    /// multiplies in the file's layout, so this serves tests and benches.
    fn kept_path(&self, ty: GgmlType) -> Result<PrefillPath> {
        let prefill = self.prefill_kernels()?;
        if prefill.full_rate {
            return Ok(PrefillPath::F16);
        }
        let has_int8 = prefill
            .types
            .iter()
            .find(|(t, ..)| *t == ty)
            .is_none_or(|(.., paths)| paths[0].is_some());
        Ok(if super::FLUSH_AS_FAST.contains(&ty) || !has_int8 {
            PrefillPath::F16Flush
        } else {
            PrefillPath::Int8
        })
    }
}
