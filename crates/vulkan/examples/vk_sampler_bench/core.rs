// The sampler benchmark for one backend. Included twice by `main.rs`, once inside a module that
// imports `ares_vulkan` and once `ares_cuda`; the two functions the modules define (`open`,
// `new_stream`) are the only places where the crates differ.

use std::time::Instant;

use ares_sampler::{Entry, LogprobOut, SamplePick, SampleRow, TokenMask};

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

fn buffer(device: &Device, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64)?;
    buffer.write(0, bytes)?;
    Ok(buffer)
}

/// Microseconds of one pick of `n_rows` rows with this setting (the median of 40 picks, each
/// timed from its launches to the end of the stream) and the launches it takes.
pub fn time(ordinal: u32, setting: &Setting, n_rows: usize) -> Result<(f64, usize)> {
    let device = open(ordinal)?;
    let kernels = SamplerKernels::load(&device)?;
    let stream = new_stream(&device)?;
    let mut rng = Rng(0x5eed_1234_abcd_0003);
    let host: Vec<u8> = (0..n_rows)
        .flat_map(|_| peaked_row(&mut rng))
        .flat_map(f32::to_le_bytes)
        .collect();
    let logits = DeviceBuffer::new(&device, host.len() as u64)?;
    let words = TokenMask::words_for(VOCAB as u32);
    let mask = TokenMask::only(VOCAB as u32, (0..VOCAB as u32).step_by(3));
    let mut rows: Vec<SampleRow> = Vec::new();
    let (mut entries, mut masks) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        let mut row = setting.params.row(r as u32);
        if setting.entries {
            // As a penalty over the last 64 tokens takes.
            row = row.with_entries(entries.len() as u32, 64);
            entries.extend((0..64).map(|i| Entry {
                rep: 1.1,
                sub: 0.3,
                ..Entry::new(i * 3571 % VOCAB as u32)
            }));
        }
        if setting.mask {
            row = row.with_mask((masks.len() / words) as u32);
            masks.extend_from_slice(mask.words());
        }
        rows.push(row);
    }
    let params_bytes: Vec<u8> = rows.iter().flat_map(|p| p.to_bytes()).collect();
    let entry_bytes: Vec<u8> = entries.iter().flat_map(Entry::to_bytes).collect();
    let mask_bytes: Vec<u8> = masks.iter().flat_map(|w| w.to_le_bytes()).collect();
    let params = buffer(&device, &params_bytes)?;
    let entries_buffer = buffer(&device, &entry_bytes)?;
    let masks_buffer = buffer(&device, &mask_bytes)?;
    let out = DeviceBuffer::new(&device, (n_rows * SamplePick::BYTES) as u64)?;
    let logprobs = DeviceBuffer::new(&device, (n_rows * LogprobOut::BYTES) as u64)?;
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
        inverse: DevicePtr::NULL,
        vocab: VOCAB as u32,
        params: params.ptr(),
        entries: if setting.entries { entries_buffer.ptr() } else { DevicePtr::NULL },
        masks: if setting.mask { masks_buffer.ptr() } else { DevicePtr::NULL },
        mask_words: words as u32,
        out: out.ptr(),
        logprobs: logprobs.ptr(),
        scratch: scratch.ptr(),
        exchange: exchange.ptr(),
        devices: 1,
        rank: 0,
    };
    let plan = SamplePlan::of(&rows);
    let mut times = Vec::new();
    for round in 0..45 {
        // The corrections change the logits in place, so every round starts from a fresh copy.
        logits.write(0, &host)?;
        if setting.entries {
            entries_buffer.write(0, &entry_bytes)?;
        }
        stream.synchronize()?;
        let begun = Instant::now();
        // SAFETY: the buffers have the sizes the batch documents.
        unsafe { kernels.sample(&stream, &batch, plan, |_, _| Ok::<(), Error>(()))? };
        stream.synchronize()?;
        // The first rounds make the pipelines and warm the clocks.
        if round >= 5 {
            times.push(begun.elapsed().as_secs_f64() * 1e6);
        }
    }
    times.sort_by(f64::total_cmp);
    Ok((times[times.len() / 2], plan.steps(1).len()))
}
