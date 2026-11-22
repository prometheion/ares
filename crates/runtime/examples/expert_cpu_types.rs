//! Times an expert computed on the CPU for every weight type: a job of one expert on the
//! engine's worker pool (`LayerJob`, `CpuPool` as the engine places it), with every type once
//! as the type of gate and up (down as Q2_0) and once as the type of down (gate and up as
//! IQ2_S), next to the types that have kernels written for them; then the pairs of types of
//! the Flash-Next files, on the pool and on one thread alone.
//!
//! The layer has the widths of Qwen3.8-Flash-Next (2560 and 640; 768 for a type of down whose
//! block does not divide 640). A job is timed from its creation to its last task, which is
//! what the cost model of the store times (`CostModel::record_job`); the experts are eight
//! that take turns, so that a job reads bytes the one before did not leave in the cache.
//!
//! ```sh
//! cargo run --release -p ares-runtime --example expert_cpu_types -- [workers] [pairs]
//! ```
//!
//! Without a number the workers are placed as the engine places them; with one, that many
//! unpinned workers. With `pairs` only the second part runs: the pairs of types of the
//! Flash-Next files, as the pool computes them and as one thread does (the main thread, which
//! runs the job's tasks itself; its time over the job's bytes is the kernels' rate).

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::ExpertShape;
use ares_runtime::expert_store::cpu::{
    CpuExpert, CpuPool, Features, LayerJob, MAX_ROWS, Placement, Topology, quantize_activations,
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

/// Who runs a job's tasks.
#[derive(Clone, Copy)]
enum Runner<'a> {
    Pool(&'a CpuPool),
    /// The calling thread alone.
    Alone,
}

/// The median time of a job of one expert of `shape` that `rows` rows use, in microseconds.
fn time(runner: Runner<'_>, shape: &ExpertShape, rows: usize) -> Result<f64, String> {
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
    let features = match runner {
        Runner::Pool(pool) => pool.features(),
        Runner::Alone => Features::detect(),
    };
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
        let job = unsafe { LayerJob::new(features, shape, rows, &act, &[expert]) }
            .map_err(|e| e.to_string())?;
        let job = Arc::new(job);
        match runner {
            Runner::Pool(pool) => pool.submit(Arc::clone(&job)),
            Runner::Alone => job.work(),
        }
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

fn shape(n_ff: u32, gate_ty: GgmlType, down_ty: GgmlType) -> ExpertShape {
    ExpertShape { n_embd: 2560, n_ff, n_experts: 64, top_k: 8, gate_ty, down_ty }
}

/// Every weight type as gate and up and as down, on the pool.
fn every_type(pool: &CpuPool) -> Result<(), String> {
    println!("| type | as | bits a weight | 1 row, us | 2 rows | 4 rows | 8 rows |");
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: |");
    for ty in ExpertShape::types() {
        let n_ff = if 640 % ty.block_values() == 0 { 640 } else { 768 };
        for (role, shape) in [
            ("gate and up", shape(640, ty, GgmlType::Q2_0)),
            ("down", shape(n_ff, GgmlType::IQ2_S, ty)),
        ] {
            let bits = 8.0 * f64::from(ty.block_bytes()) / f64::from(ty.block_values());
            let mut line = format!("| {ty} | {role} ({} x {}) | {bits:.2} |", 2560, shape.n_ff);
            for rows in [1, 2, 4, 8] {
                line.push_str(&format!(" {:.0} |", time(Runner::Pool(pool), &shape, rows)?));
            }
            println!("{line}");
        }
    }
    Ok(())
}

/// The pairs of types of the Flash-Next files (UD-Q3_K_XL, IQ2_XS, UD-Q4_K_XL), on the pool
/// and on one thread; for one thread also the rate over the expert's bytes.
fn file_pairs(pool: &CpuPool) -> Result<(), String> {
    use GgmlType::*;
    let pairs = [
        (IQ3_XXS, Q5_0),
        (IQ3_XXS, IQ4_NL),
        (IQ4_XS, IQ4_NL),
        (IQ4_XS, Q5_0),
        (IQ2_S, Q5_0),
        (IQ2_S, IQ4_NL),
        (IQ2_S, Q2_0),
        (IQ2_XXS, Q2_0),
        (IQ1_M, Q2_0),
        (Q4_K, Q5_0),
        (Q4_K, Q8_0),
        (Q5_K, Q5_1),
        (Q6_K, Q8_0),
    ];
    println!(
        "| gate and up / down | pool, 1 row, us | 2 rows | 4 rows | 8 rows | one thread, 1 row, us | GB/s | 8 rows, us |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (gate_ty, down_ty) in pairs {
        let shape = shape(640, gate_ty, down_ty);
        let gate = shape.gate_bytes().ok_or("gate size")?;
        let bytes = (2 * gate + shape.down_bytes().ok_or("down size")?) as f64;
        let mut line = format!("| {gate_ty} / {down_ty} |");
        for rows in [1, 2, 4, 8] {
            line.push_str(&format!(" {:.0} |", time(Runner::Pool(pool), &shape, rows)?));
        }
        let alone = time(Runner::Alone, &shape, 1)?;
        line.push_str(&format!(" {alone:.0} | {:.1} |", bytes / alone / 1e3));
        line.push_str(&format!(" {:.0} |", time(Runner::Alone, &shape, 8)?));
        println!("{line}");
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let placement = match args.iter().find_map(|a| a.parse().ok()) {
        Some(workers) => Placement::unpinned(workers),
        None => Placement::for_machine(&Topology::read(), 1),
    };
    let pool = CpuPool::new(&placement, Duration::from_millis(50)).map_err(|e| e.to_string())?;
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("{} workers, {:?}; load average {}", pool.workers(), pool.features(), load.trim());
    if !args.iter().any(|a| a == "pairs") {
        every_type(&pool)?;
        println!();
    }
    file_pairs(&pool)
}
