//! Times an expert computed on the CPU for every weight type: a job of one expert on the
//! engine's worker pool (`LayerJob`, `CpuPool` as the engine places it), with every type once
//! as the type of gate and up (down as Q2_0) and once as the type of down (gate and up as
//! IQ2_S), next to the types that have kernels written for them.
//!
//! The layer has the widths of Qwen3.8-Flash-Next (2560 and 640; 768 for a type of down whose
//! block does not divide 640). A job is timed from its creation to its last task, which is
//! what the cost model of the store times (`CostModel::record_job`); the experts are eight
//! that take turns, so that a job reads bytes the one before did not leave in the cache.
//!
//! ```sh
//! cargo run --release -p ares-runtime --example expert_cpu_types -- [workers]
//! ```
//!
//! Without an argument the workers are placed as the engine places them; with a number,
//! that many unpinned workers.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::ExpertShape;
use ares_runtime::expert_store::cpu::{
    CpuExpert, CpuPool, LayerJob, MAX_ROWS, Placement, Topology, quantize_activations,
};

/// Bytes of a matrix of `ty` that vary and decode to finite values of moderate size, as the
/// store's own calibration makes them: every byte is below 0x40, which keeps an f16 scale
/// small wherever it lies, and the bytes that hold the exponent of a float or of a block of
/// MXFP4 are set so that no value is subnormal or near it (the products of such weights are
/// subnormal, which a CPU computes many times slower, and no real weight is).
fn fill(ty: GgmlType, len: usize, seed: usize) -> Vec<u8> {
    let mut bytes: Vec<u8> =
        (0..len).map(|i| ((i * 31 + seed * 17) ^ (i >> 7)) as u8 & 0x3f).collect();
    let (width, at, exponent, mask) = match ty {
        GgmlType::F32 => (4, 3, 0x38, 0x07),
        GgmlType::F16 | GgmlType::BF16 => (2, 1, 0x38, 0x07),
        GgmlType::MXFP4 => (ty.block_bytes() as usize, 0, 0x78, 0x03),
        _ => return bytes,
    };
    for value in bytes.chunks_exact_mut(width) {
        value[at] = exponent | (value[at] & mask);
    }
    bytes
}

/// The median time of a job of one expert of `shape` that `rows` rows use, in microseconds.
fn time(pool: &CpuPool, shape: &ExpertShape, rows: usize) -> Result<f64, String> {
    let gate_bytes = shape.gate_bytes().ok_or("gate size")? as usize;
    let down_bytes = shape.down_bytes().ok_or("down size")? as usize;
    let experts: Vec<[Vec<u8>; 3]> = (0..8)
        .map(|e| {
            [
                fill(shape.gate_ty, gate_bytes, e),
                fill(shape.gate_ty, gate_bytes, e + 8),
                fill(shape.down_ty, down_bytes, e + 16),
            ]
        })
        .collect();
    let n_in = shape.n_embd as usize;
    let x: Vec<f32> = (0..rows * n_in).map(|i| ((i * 37 % 255) as f32 - 127.0) / 64.0).collect();
    let act = quantize_activations(&x, n_in);
    let mut times = Vec::new();
    for round in 0..40 {
        let m = &experts[round % experts.len()];
        let expert = CpuExpert {
            gate: m[0].as_ptr(),
            up: m[1].as_ptr(),
            down: m[2].as_ptr(),
            rows: (1u32 << rows) - 1,
            weight: [0.1; MAX_ROWS],
        };
        // SAFETY: the matrices have the sizes of the shape and outlive the job, which is
        // waited for below.
        let job = unsafe { LayerJob::new(pool.features(), shape, rows, &act, &[expert]) }
            .map_err(|e| e.to_string())?;
        let job = Arc::new(job);
        pool.submit(Arc::clone(&job));
        let start = Instant::now();
        while !job.is_done() {
            if start.elapsed() > Duration::from_secs(60) {
                return Err("the workers do not finish a job".into());
            }
            std::hint::spin_loop();
        }
        let sums = job.sums().ok_or("a job without sums")?;
        if !sums.iter().all(|v| v.is_finite()) {
            return Err(format!("{} and {}: a sum is not a number", shape.gate_ty, shape.down_ty));
        }
        // The first rounds wake the workers and warm the tables.
        if let (true, Some(time)) = (round >= 8, job.elapsed()) {
            times.push(time.as_secs_f64() * 1e6);
        }
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

fn main() -> Result<(), String> {
    let placement = match std::env::args().nth(1).and_then(|a| a.parse().ok()) {
        Some(workers) => Placement::unpinned(workers),
        None => Placement::for_machine(&Topology::read(), 1),
    };
    let pool = CpuPool::new(&placement, Duration::from_millis(50)).map_err(|e| e.to_string())?;
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("{} workers; load average {}", pool.workers(), load.trim());
    println!("| type | as | bits a weight | 1 row, us | 2 rows | 4 rows | 8 rows |");
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: |");
    for ty in ExpertShape::types() {
        let n_ff = if 640 % ty.block_values() == 0 { 640 } else { 768 };
        let shape = |n_ff, gate_ty, down_ty| ExpertShape {
            n_embd: 2560,
            n_ff,
            n_experts: 64,
            top_k: 8,
            gate_ty,
            down_ty,
        };
        for (role, shape) in [
            ("gate and up", shape(640, ty, GgmlType::Q2_0)),
            ("down", shape(n_ff, GgmlType::IQ2_S, ty)),
        ] {
            let bits = 8.0 * f64::from(ty.block_bytes()) / f64::from(ty.block_values());
            let mut line = format!("| {ty} | {role} ({} x {}) | {bits:.2} |", 2560, shape.n_ff);
            for rows in [1, 2, 4, 8] {
                line.push_str(&format!(" {:.0} |", time(&pool, &shape, rows)?));
            }
            println!("{line}");
        }
    }
    Ok(())
}
