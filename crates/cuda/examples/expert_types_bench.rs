//! Times the expert kernels of every weight type on one device: the decode kernels
//! (`ExpertKernels::experts`, 1 and 8 rows) and the many-row kernels
//! (`ExpertRowKernels`, plan and experts with the atomic sum, 8, 64 and 512 rows), with
//! every type once as the type of gate and up (down as Q2_0) and once as the type of down
//! (gate and up as IQ2_S).
//!
//! The layer has the widths of Qwen3.8-Flash-Next (2560 and 640; 768 for a type of down
//! whose block does not divide 640), 64 experts of random bytes and rows that pick eight
//! experts each, no two rows of a step of eight the same ones. The time of a kernel does
//! not depend on the values, only on the bytes it reads and the products it computes; the
//! columns say both.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example expert_types_bench -- [device]
//! ```

use ares_core::GgmlType;
use ares_cuda::{
    Device, DeviceBuffer, Event, ExpertKernels, ExpertRowKernels, ExpertShape, Kernels, MAX_BATCH,
    Result, Stream, Wait,
};

const EXPERTS: u32 = 64;
const TOP_K: u32 = 8;
const MANY_ROWS: [u32; 3] = [8, 64, 512];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The best of three runs of `reps` calls of `f`, in microseconds a call.
fn time(
    stream: &Stream,
    start: &Event,
    end: &Event,
    reps: u32,
    f: &mut dyn FnMut() -> Result<()>,
) -> Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        start.record(stream)?;
        for _ in 0..reps {
            f()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        best = best.min(f64::from(end.elapsed_ms_since(start)?) * 1e3 / f64::from(reps));
    }
    Ok(best)
}

/// The times of one shape: the decode kernels at 1 and 8 rows, the many-row kernels at
/// [`MANY_ROWS`].
fn measure(
    device: &Device,
    decode: &ExpertKernels,
    many: &ExpertRowKernels,
    shape: &ExpertShape,
) -> Result<([f64; 2], [f64; 3])> {
    shape.check_rows()?;
    let stream = Stream::new(device, 0)?;
    let stride = shape.slot_bytes().expect("a shape that was checked");
    let pool = DeviceBuffer::new(device, u64::from(EXPERTS) * stride)?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ stride);
    let slot: Vec<u8> = (0..stride).map(|_| (rng.next() >> 32) as u8).collect();
    for e in 0..u64::from(EXPERTS) {
        pool.write(e * stride, &slot)?;
    }
    let buffer = |bytes: u64| DeviceBuffer::new(device, bytes.max(16));
    let addresses: Vec<u8> = (0..u64::from(EXPERTS))
        .flat_map(|e| (pool.ptr().as_raw() + e * stride).to_le_bytes())
        .collect();
    let slots: Vec<i32> = (0..EXPERTS as i32).collect();
    let table = buffer(addresses.len() as u64)?;
    table.write(0, &addresses)?;
    let slot_table = buffer(4 * u64::from(EXPERTS))?;
    slot_table.write(0, &slots.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;
    let most = MANY_ROWS[2];
    let n = u64::from(shape.n_embd);
    let decode_act = buffer(Kernels::quantized_activation_bytes(shape.n_embd, MAX_BATCH))?;
    let decode_hidden = buffer(shape.hidden_bytes(MAX_BATCH))?;
    let act = buffer(shape.rows_activation_bytes(most))?;
    let hidden = buffer(shape.rows_hidden_bytes(most))?;
    let out = buffer(4 * n * u64::from(most))?;
    // SAFETY: every buffer is live, and the length is its own.
    unsafe {
        for buf in [&decode_act, &decode_hidden, &act, &hidden, &out] {
            stream.fill(buf.ptr(), 0, buf.len() as usize)?;
        }
    }
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);
    // Row r picks the experts r * top_k ...: the rows of a step of eight share none.
    let picks = |rows: u32| -> Vec<Vec<u32>> {
        (0..rows).map(|r| (0..TOP_K).map(|j| (r * TOP_K + j) % EXPERTS).collect()).collect()
    };

    let mut decode_us = [0.0; 2];
    for (us, rows) in decode_us.iter_mut().zip([1, MAX_BATCH]) {
        let record = shape.host_record(&picks(rows), &slots);
        let record_buf = buffer(record.len() as u64)?;
        record_buf.write(0, &record)?;
        // SAFETY: the buffers are sized for the shape and the rows; every picked expert
        // has a slot.
        *us = time(&stream, &start, &end, 20, &mut || unsafe {
            decode.experts(
                &stream,
                shape,
                pool.ptr(),
                slot_table.ptr(),
                record_buf.ptr(),
                decode_act.ptr(),
                decode_hidden.ptr(),
                out.ptr(),
                None,
                rows,
                None,
            )
        })?;
    }
    let mut many_us = [0.0; 3];
    for (us, rows) in many_us.iter_mut().zip(MANY_ROWS) {
        let tile = ExpertRowKernels::tile_for(shape, rows);
        let ids = picks(rows);
        let mut record = vec![0u8; 16];
        record.extend(ids.iter().flatten().flat_map(|id| id.to_le_bytes()));
        record.extend(ids.iter().flatten().flat_map(|_| (1.0 / TOP_K as f32).to_le_bytes()));
        let record_buf = buffer(record.len() as u64)?;
        record_buf.write(0, &record)?;
        let plan = buffer(shape.rows_plan_bytes(rows, tile))?;
        // SAFETY: as above; every expert has an address.
        *us = time(&stream, &start, &end, 5, &mut || unsafe {
            many.plan_rows(&stream, shape, rows, tile, record_buf.ptr(), plan.ptr(), 0)?;
            many.rows_experts(
                &stream,
                shape,
                rows,
                tile,
                table.ptr(),
                plan.ptr(),
                act.ptr(),
                hidden.ptr(),
                out.ptr(),
            )
        })?;
    }
    Ok((decode_us, many_us))
}

fn main() -> Result<()> {
    let ordinal = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(0);
    let device = Device::open(ordinal, Wait::Block)?;
    let decode = ExpertKernels::load(&device)?;
    let many = ExpertRowKernels::load(&device)?;
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("{}; load average {}", device.info().name, load.trim());
    println!(
        "| type | as | bits a weight | KiB an expert | decode 1 row, us | 8 rows, us | GB/s at 1 row \
         | many rows: 8, us | 64, us | 512, us | TFLOPS at 512 |"
    );
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for ty in ExpertShape::types() {
        let n_ff = if 640 % ty.block_values() == 0 { 640 } else { 768 };
        let shapes = [
            (
                "gate and up",
                ExpertShape {
                    n_embd: 2560,
                    n_ff: 640,
                    n_experts: EXPERTS,
                    top_k: TOP_K,
                    gate_ty: ty,
                    down_ty: GgmlType::Q2_0,
                },
            ),
            (
                "down",
                ExpertShape {
                    n_embd: 2560,
                    n_ff,
                    n_experts: EXPERTS,
                    top_k: TOP_K,
                    gate_ty: GgmlType::IQ2_S,
                    down_ty: ty,
                },
            ),
        ];
        for (role, shape) in shapes {
            let (decode_us, many_us) = measure(&device, &decode, &many, &shape)?;
            let expert = shape.expert_bytes().expect("a shape that was checked") as f64;
            let bits = 8.0 * ty.block_bytes() as f64 / ty.block_values() as f64;
            // A step of one row reads eight experts once.
            let gbs = f64::from(TOP_K) * expert / decode_us[0] / 1e3;
            let flops = 2.0
                * 3.0
                * f64::from(shape.n_embd)
                * f64::from(shape.n_ff)
                * f64::from(MANY_ROWS[2] * TOP_K);
            println!(
                "| {ty} | {role} ({} x {}) | {bits:.2} | {:.0} | {:.0} | {:.0} | {gbs:.0} | {:.0} | {:.0} \
                 | {:.0} | {:.1} |",
                shape.n_embd,
                shape.n_ff,
                expert / 1024.0,
                decode_us[0],
                decode_us[1],
                many_us[0],
                many_us[1],
                many_us[2],
                flops / many_us[2] / 1e6,
            );
        }
    }
    Ok(())
}
