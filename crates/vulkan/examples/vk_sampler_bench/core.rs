// The sampler benchmark for one backend. Included twice by `main.rs`, once inside a module that
// imports `ares_vulkan` and once `ares_cuda`; the two functions the modules define (`open`,
// `new_stream`) are the only places where the crates differ.

use super::{Setting, VOCAB};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.unit() + 1e-7;
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Rows as a model's: most tokens far below a few likely ones.
fn peaked_row(rng: &mut Rng) -> Vec<f32> {
    let mut logits: Vec<f32> = (0..VOCAB).map(|_| rng.normal() * 2.5 - 4.0).collect();
    for _ in 0..12 {
        let at = (rng.next() % VOCAB as u64) as usize;
        logits[at] = 8.0 + rng.normal() * 3.0;
    }
    logits
}

/// Microseconds of one pick of `n_rows` rows with this setting (the median of a few runs of 20
/// picks, timed on the device) and the launches it takes.
pub fn time(ordinal: u32, setting: &Setting, n_rows: usize) -> Result<(f64, usize)> {
    let device = open(ordinal)?;
    let kernels = SamplerKernels::load(&device)?;
    let stream = new_stream(&device)?;
    let mut rng = Rng(0x5eed_1234_abcd_0003);
    let flat: Vec<u8> = (0..n_rows)
        .flat_map(|_| peaked_row(&mut rng))
        .flat_map(f32::to_le_bytes)
        .collect();
    let logits = DeviceBuffer::new(&device, flat.len() as u64)?;
    logits.write(0, &flat)?;
    let params: Vec<SampleRow> = (0..n_rows)
        .map(|r| {
            SampleParams {
                temperature: setting.temperature,
                top_k: setting.top_k,
                top_p: setting.top_p,
                min_p: 0.0,
                seed: 9,
            }
            .row(r as u32)
        })
        .collect();
    let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_bytes()).collect();
    let params_buffer = DeviceBuffer::new(&device, bytes.len() as u64)?;
    params_buffer.write(0, &bytes)?;
    let out = DeviceBuffer::new(&device, (n_rows * SamplePick::BYTES) as u64)?;
    let n = n_rows as u32;
    let scratch = DeviceBuffer::new(&device, SamplerKernels::scratch_bytes(n))?;
    let exchange =
        DeviceBuffer::new(&device, u64::from(SamplerKernels::exchange_values(n, 1)) * 4)?;
    // SAFETY: the scratch buffer has the size the sampler asks for.
    unsafe { kernels.clear(&stream, scratch.ptr(), n)? };
    let batch = SampleBatch {
        logits: logits.ptr(),
        stride: VOCAB as u32,
        n_rows: n,
        capacity: n,
        n_local: VOCAB as u32,
        tokens: DevicePtr::NULL,
        first: 0,
        params: params_buffer.ptr(),
        out: out.ptr(),
        scratch: scratch.ptr(),
        exchange: exchange.ptr(),
        devices: 1,
        rank: 0,
    };
    let plan = SamplePlan::of(&params);
    let run = |picks: usize| -> Result<()> {
        for _ in 0..picks {
            // SAFETY: the buffers have the sizes the batch documents.
            unsafe { kernels.sample(&stream, &batch, plan, |_, _| Ok::<(), Error>(()))? };
        }
        Ok(())
    };
    // Warm the pipelines and the clocks.
    for _ in 0..5 {
        run(10)?;
        stream.synchronize()?;
    }
    let (start, stop) = (Event::timed(&device)?, Event::timed(&device)?);
    let mut times = Vec::new();
    for _ in 0..7 {
        start.record(&stream)?;
        run(20)?;
        stop.record(&stream)?;
        stop.synchronize()?;
        times.push(f64::from(stop.elapsed_ms_since(&start)?) * 1000.0 / 20.0);
    }
    times.sort_by(f64::total_cmp);
    Ok((times[times.len() / 2], plan.steps(1).len()))
}
