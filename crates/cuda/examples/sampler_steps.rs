//! Where the time of a pick goes: the device time of every part of the sampler
//! ([`SampleStep`]) for 1, 4 and 16 rows of the vocabulary of Qwen (248320
//! logits) and the settings of `the_time_of_a_pick` in `tests/sampler.rs`.
//!
//! For every pick the logits (and the corrections) are written fresh and the
//! stream is drained. A timed event is recorded before the first part and
//! after each part, so a part's time is the device time from the end of the
//! part before it to its own end, the gap between kernels included. The wall
//! time is that of the launches of the pick to the end of the stream, as the
//! sampler benchmark of `ares-vulkan` measures it. Medians of 40 picks. Parts of the same kind
//! are added up; `--each` prints every part.
//!
//! ```sh
//! flock <lock> cargo run --release -p ares-cuda --example sampler_steps -- [--device N] [--rows N] [--each]
//! ```

use std::time::Instant;

use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Error, Event, Result, SampleBatch, SamplePlan, SamplerKernels,
    Stream, Wait, devices,
};
use ares_sampler::{Entry, LogprobOut, Logprobs, SampleParams, SamplePick, SampleRow, TokenMask};

/// The vocabulary of Qwen3.8.
const VOCAB: usize = 248_320;
const ROUNDS: usize = 45;
const WARMUP: usize = 5;

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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// What the rows of a batch ask for: the setting, whether the rows have 64
/// corrections (as a penalty over the last 64 tokens gives) and whether they
/// have a mask that allows a third of the tokens.
type Setting = (&'static str, SampleParams, bool, bool);

fn settings() -> Vec<Setting> {
    let sampled =
        SampleParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 9, ..SampleParams::GREEDY };
    let lp = |after_chain| Logprobs { top: 5, chosen: true, after_chain };
    vec![
        ("greedy", SampleParams::GREEDY, false, false),
        ("top-k 20, top-p 0.95", sampled, false, false),
        ("top-p 0.95", SampleParams { top_k: 0, ..sampled }, false, false),
        ("greedy with penalties", SampleParams::GREEDY, true, false),
        ("greedy with a mask", SampleParams::GREEDY, false, true),
        ("top-k, top-p, penalties and a mask", sampled, true, true),
        (
            "greedy with logprobs",
            SampleParams { logprobs: lp(false), ..SampleParams::GREEDY },
            false,
            false,
        ),
        (
            "top-k, top-p with logprobs after the chain",
            SampleParams { logprobs: lp(true), ..sampled },
            false,
            false,
        ),
        (
            "top-n-sigma 2, typical 0.9, top-k, top-p",
            SampleParams { typical_p: 0.9, top_n_sigma: 2.0, ..sampled },
            false,
            false,
        ),
    ]
}

fn time(device: &Device, setting: &Setting, n_rows: usize, each: bool) -> Result<()> {
    let (name, sample, with_entries, with_mask) = setting;
    let kernels = SamplerKernels::load(device)?;
    let stream = Stream::new(device, 0)?;
    let mut rng = Rng(0x5eed_1234_abcd_0003);
    let host: Vec<u8> =
        (0..n_rows).flat_map(|_| peaked_row(&mut rng)).flat_map(f32::to_le_bytes).collect();
    let logits = DeviceBuffer::new(device, host.len() as u64)?;
    let words = TokenMask::words_for(VOCAB as u32);
    let mask = TokenMask::only(VOCAB as u32, (0..VOCAB as u32).step_by(3));
    let mut rows: Vec<SampleRow> = Vec::new();
    let (mut entries, mut masks) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        let mut row = sample.row(r as u32);
        if *with_entries {
            row = row.with_entries(entries.len() as u32, 64);
            entries.extend((0..64).map(|i| Entry {
                rep: 1.1,
                sub: 0.3,
                ..Entry::new(i * 3571 % VOCAB as u32)
            }));
        }
        if *with_mask {
            row = row.with_mask((masks.len() / words) as u32);
            masks.extend_from_slice(mask.words());
        }
        rows.push(row);
    }
    let params_bytes: Vec<u8> = rows.iter().flat_map(|p| p.to_bytes()).collect();
    let entry_bytes: Vec<u8> = entries.iter().flat_map(Entry::to_bytes).collect();
    let mask_bytes: Vec<u8> = masks.iter().flat_map(|w| w.to_le_bytes()).collect();
    let params = buffer(device, &params_bytes)?;
    let entries_buffer = buffer(device, &entry_bytes)?;
    let masks_buffer = buffer(device, &mask_bytes)?;
    let out = DeviceBuffer::new(device, (n_rows * SamplePick::BYTES) as u64)?;
    let logprobs = DeviceBuffer::new(device, (n_rows * LogprobOut::BYTES) as u64)?;
    let n = n_rows as u32;
    let scratch = DeviceBuffer::new(device, SamplerKernels::scratch_bytes(n))?;
    let exchange = DeviceBuffer::new(device, u64::from(SamplerKernels::exchange_values(n, 1)) * 4)?;
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
        entries: if *with_entries { entries_buffer.ptr() } else { DevicePtr::NULL },
        masks: if *with_mask { masks_buffer.ptr() } else { DevicePtr::NULL },
        mask_words: words as u32,
        out: out.ptr(),
        logprobs: logprobs.ptr(),
        scratch: scratch.ptr(),
        exchange: exchange.ptr(),
        devices: 1,
        rank: 0,
    };
    let plan = SamplePlan::of(&rows);
    let steps = plan.steps(1);
    let events: Vec<Event> =
        (0..=steps.len()).map(|_| Event::timed(device)).collect::<Result<_>>()?;
    let mut wall = Vec::new();
    let mut device_total = Vec::new();
    let mut per_step: Vec<Vec<f64>> = vec![Vec::new(); steps.len()];
    for round in 0..2 * ROUNDS {
        // The corrections change the logits in place, so every round starts from a fresh copy.
        logits.write(0, &host)?;
        if *with_entries {
            entries_buffer.write(0, &entry_bytes)?;
        }
        stream.synchronize()?;
        // Odd rounds time the whole pick as a caller sees it, even rounds its parts.
        if round % 2 == 1 {
            let begun = Instant::now();
            // SAFETY: the buffers have the sizes the batch documents.
            unsafe { kernels.sample(&stream, &batch, plan, |_, _| Ok::<(), Error>(()))? };
            stream.synchronize()?;
            if round / 2 >= WARMUP {
                wall.push(begun.elapsed().as_secs_f64() * 1e6);
            }
            continue;
        }
        events[0].record(&stream)?;
        for (i, step) in steps.iter().enumerate() {
            // SAFETY: as above.
            unsafe { kernels.enqueue(&stream, &batch, *step)? };
            events[i + 1].record(&stream)?;
        }
        stream.synchronize()?;
        if round / 2 >= WARMUP {
            for i in 0..steps.len() {
                per_step[i].push(f64::from(events[i + 1].elapsed_ms_since(&events[i])?) * 1e3);
            }
            device_total.push(f64::from(events[steps.len()].elapsed_ms_since(&events[0])?) * 1e3);
        }
    }
    println!(
        "{name}, {n_rows} rows: wall {:.0} us, device {:.0} us, {} parts",
        median(&mut wall),
        median(&mut device_total),
        steps.len()
    );
    if each {
        for (step, times) in steps.iter().zip(&mut per_step) {
            println!("    {:<28} {:>7.1} us", format!("{step:?}"), median(times));
        }
        return Ok(());
    }
    // Parts of the same kind (the levels of a descent) are added up.
    let mut kinds: Vec<(String, f64, usize)> = Vec::new();
    for (step, times) in steps.iter().zip(&mut per_step) {
        let text = format!("{step:?}");
        let kind = text.split('(').next().unwrap_or(&text).to_string();
        let us = median(times);
        match kinds.iter_mut().find(|(k, _, _)| *k == kind) {
            Some(entry) => {
                entry.1 += us;
                entry.2 += 1;
            }
            None => kinds.push((kind, us, 1)),
        }
    }
    for (kind, us, count) in kinds {
        println!("    {kind:<10} x{count:<2} {us:>7.1} us");
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let only = arg("--device").map(|d| d.parse::<u32>().expect("--device N"));
    let only_rows = arg("--rows").map(|d| d.parse::<usize>().expect("--rows N"));
    let each = args.iter().any(|a| a == "--each");
    for info in devices().expect("the CUDA driver") {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        println!("\n## {}\n", info.name);
        let device = Device::open(info.ordinal, Wait::Block).expect("the device");
        for setting in &settings() {
            for rows in [1usize, 4, 16] {
                if only_rows.is_some_and(|r| r != rows) {
                    continue;
                }
                time(&device, setting, rows, each).expect("the sampler");
            }
        }
    }
}
