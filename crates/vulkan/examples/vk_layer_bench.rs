//! Times the layer kernels (`LayerKernels`) with the shapes of Qwen3.8-27B on each
//! device: width 5120, 48 value heads and 16 key heads of 128, convolution over
//! 10240 channels, feed-forward width 17408, attention of 24 query heads and 4
//! key/value heads of 256. The twin of `crates/cuda/examples/layers_bench.rs`, with the
//! same shapes, configurations and counting, so that the two tables can be laid side by
//! side.
//!
//! Each kernel is launched many times in a row between two events, and the
//! number printed is the time per launch on the device. The kernels that read
//! recurrent state rotate over 48 copies of it (the blocks of the model), so
//! that the state comes from memory and not from the L2 cache, as it does in a
//! step. `GB/s` counts the bytes the kernel must move (inputs read once,
//! outputs written once, the state once or twice).
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_layer_bench [device] [decode|prompt|variants|all]
//! ```
//!
//! `variants` times the delta rule in every compiled shape (`LAYERS_VARIANTS=a,b` for some).

// Every launch below works on buffers that `run` allocates for the largest configuration and
// that outlive it; the sizes of a launch never exceed them.
#![allow(clippy::undocumented_unsafe_blocks, clippy::redundant_closure)]

use ares_core::GgmlType;
use ares_vulkan::{
    AttentionShape, CommitSet, Device, DeviceBuffer, Event, Kernels, LayerKernels, NormKind,
    OutputGate, RecurrentShape, Result, RopeParams, Segment, StateFormat, StepDims, StepRecord,
    Stream, devices, quantized_activation_bytes,
};

const D: u32 = 5120;
const FF: u32 = 17408;
const SHAPE: RecurrentShape =
    RecurrentShape { n_k_heads: 16, n_v_heads: 48, head_dim: 128, conv_kernel: 4 };
const ATTN: AttentionShape = AttentionShape { n_head: 24, n_kv_head: 4, head_dim: 256 };
const LAYERS: usize = 48;

struct Rig {
    device: Device,
    stream: Stream,
    layers: LayerKernels,
    start: Event,
    stop: Event,
}

impl Rig {
    fn buf(&self, bytes: u64, fill: u8) -> DeviceBuffer {
        let b = DeviceBuffer::new(&self.device, bytes.max(16)).unwrap();
        // SAFETY: the whole buffer, on this device.
        unsafe { self.stream.fill(b.ptr(), fill, bytes as usize).unwrap() };
        // The fill is asynchronous; a later synchronous write must not be overwritten by it.
        self.stream.synchronize().unwrap();
        b
    }

    /// Floats of a small magnitude (0x3C = 0.0117 as the first byte pattern of
    /// each f32 would be fragile; use 0x3F000000 words: 0.5).
    fn half_floats(&self, n: u64) -> DeviceBuffer {
        let data: Vec<u8> =
            (0..n).flat_map(|i| (0.25 + (i % 13) as f32 * 0.03).to_le_bytes()).collect();
        let b = DeviceBuffer::new(&self.device, n * 4).unwrap();
        b.write(0, &data).unwrap();
        b
    }

    /// Microseconds per launch of `launch(i)`: the median of five batches of
    /// `iters` launches between two events, after a warm-up.
    fn time(&self, iters: u32, mut launch: impl FnMut(u32)) -> f64 {
        for i in 0..iters.min(4) {
            launch(i);
        }
        self.stream.synchronize().unwrap();
        let mut batches: Vec<f64> = (0..5)
            .map(|_| {
                let wall = std::time::Instant::now();
                self.start.record(&self.stream).unwrap();
                for i in 0..iters {
                    launch(i);
                }
                self.stop.record(&self.stream).unwrap();
                self.stop.synchronize().unwrap();
                let wall_us = wall.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
                let event_us = f64::from(self.stop.elapsed_ms_since(&self.start).unwrap()) * 1000.0
                    / f64::from(iters);
                // The events are the device's own clock; the wall clock includes the launch of
                // the first kernel and the wake-up, so it is never smaller. On a device shared
                // with other processes the events sometimes report far too little: then the
                // wall clock, which is at most a few microseconds per batch too long, is used.
                if event_us < 0.9 * wall_us { wall_us } else { event_us }
            })
            .collect();
        batches.sort_by(f64::total_cmp);
        batches[2]
    }
}

struct Config {
    label: String,
    seqs: u32,
    rows_per_seq: u32,
}

impl Config {
    fn rows(&self) -> u32 {
        self.seqs * self.rows_per_seq
    }
}

fn record(cfg: &Config, direct: bool) -> StepRecord {
    let segments: Vec<Segment> = (0..cfg.seqs)
        .map(|s| Segment {
            row0: s * cfg.rows_per_seq,
            n_rows: cfg.rows_per_seq,
            slot: s,
            pos0: 1000,
            direct,
        })
        .collect();
    let out_rows: Vec<u32> = (0..cfg.seqs).map(|s| (s + 1) * cfg.rows_per_seq - 1).collect();
    StepRecord::new(&segments, &vec![5; cfg.rows() as usize], None, &out_rows).unwrap()
}

struct Line {
    name: String,
    cells: Vec<(f64, f64)>, // microseconds, GB/s
}

fn print_table(title: &str, configs: &[Config], lines: &[Line]) {
    println!("\n{title}");
    print!("{:<34}", "kernel (us per launch, GB/s)");
    for c in configs {
        print!("{:>20}", c.label);
    }
    println!();
    for l in lines {
        print!("{:<34}", l.name);
        for (us, gbs) in &l.cells {
            if us.is_nan() {
                print!("{:>20}", "-");
            } else {
                print!("{:>20}", format!("{us:8.1} {gbs:6.0}"));
            }
        }
        println!();
    }
}

/// Times every kernel for each configuration.
fn run(rig: &Rig, configs: &[Config], title: &str, with_state_rotation: bool, only: Option<&str>) {
    let max_rows = configs.iter().map(Config::rows).max().unwrap();
    let max_seqs = configs.iter().map(|c| c.seqs).max().unwrap();
    let vd = SHAPE.value_dim();
    let kd = SHAPE.key_dim();
    let cd = SHAPE.conv_dim();
    let (conv_len, delta_len) = (SHAPE.conv_state_len(), SHAPE.delta_state_len());
    let rows = u64::from(max_rows);

    // Activations. Values only need to be finite and of moderate size.
    let qkv = rig.half_floats(rows * u64::from(cd));
    let taps = rig.half_floats(u64::from(cd) * 4);
    let q_out = rig.buf(rows * u64::from(kd) * 4, 0);
    let kv_out = rig.buf(rows * u64::from(kd + vd) * 4, 0);
    let alpha = rig.half_floats(rows * 48);
    let beta = rig.half_floats(rows * 48);
    let dt = rig.half_floats(48);
    let a: Vec<u8> = (0..48).flat_map(|i| (-0.5 - i as f32 * 0.05).to_le_bytes()).collect();
    let ssm_a = DeviceBuffer::new(&rig.device, 48 * 4).unwrap();
    ssm_a.write(0, &a).unwrap();
    let out = rig.buf(rows * u64::from(vd) * 4, 0);
    let z = rig.half_floats(rows * u64::from(vd));
    let norm_w = rig.half_floats(256);
    let act_o = rig.buf(quantized_activation_bytes(rows * u64::from(vd)), 0);
    let layers_n = if with_state_rotation { LAYERS } else { 1 };
    let conv_state = rig.buf(layers_n as u64 * u64::from(max_seqs) * (conv_len + 16) * 4, 0);
    let delta32 = rig.buf(layers_n as u64 * u64::from(max_seqs) * (delta_len + 16) * 4, 0);
    let delta16 = rig.buf(layers_n as u64 * u64::from(max_seqs) * (delta_len + 16) * 2, 0);
    let (conv_stride, delta_stride) = (conv_len + 16, delta_len + 16);
    // Embedding table of the 27B's type (q3_K), a thousand rows.
    let table_rows = 1000u64;
    let table = rig.buf(table_rows * (u64::from(D) / 256) * 110, 0x11);
    let emb_out = rig.buf(rows * u64::from(D) * 4, 0);
    // Attention preparation.
    let qg = rig.half_floats(rows * u64::from(ATTN.n_head) * 512);
    let kin = rig.half_floats(rows * u64::from(ATTN.n_kv_head) * 256);
    let qn = rig.half_floats(256);
    let kn = rig.half_floats(256);
    let rope = RopeParams { dim: 64, freq_base: 1e7, sections: [11, 11, 10, 0] };
    let table_rope = rig.buf(rows * 32 * 8, 0);
    let q_att = rig.buf(rows * u64::from(ATTN.n_head) * 256 * 4, 0);
    let g_att = rig.buf(rows * u64::from(ATTN.n_head) * 256 * 4, 0);
    let k_att = rig.buf(rows * u64::from(ATTN.n_kv_head) * 256 * 4, 0);
    // Norm of heads.
    let hn_x = rig.half_floats(rows * u64::from(ATTN.n_head) * 256);
    let hn_y = rig.buf(rows * u64::from(ATTN.n_head) * 256 * 4, 0);
    // Feed-forward.
    let gate = rig.half_floats(rows * u64::from(FF));
    let up = rig.half_floats(rows * u64::from(FF));
    let act_ff = rig.buf(quantized_activation_bytes(rows * u64::from(FF)), 0);
    // Output rows.
    let hidden = rig.half_floats(rows * u64::from(D));
    let head_in = rig.buf(rows * u64::from(D) * 4, 0);
    let norm_d = rig.half_floats(u64::from(D));
    rig.stream.synchronize().unwrap();

    let mut lines: Vec<Line> = Vec::new();
    let mut add = |name: &str, f: &mut dyn FnMut(&Config) -> (f64, f64)| {
        if only.is_some_and(|prefix| !name.starts_with(prefix)) {
            return;
        }
        let cells = configs.iter().map(|c| f(c)).collect();
        lines.push(Line { name: name.to_owned(), cells });
    };
    let iters = |c: &Config| if c.rows() >= 1024 { 20 } else { 200 };
    let gbs = |bytes: f64, us: f64| bytes / (us * 1e-6) / 1e9;

    add("embed_gather (q3_K)", &mut |c| {
        let rec = record(c, false);
        let step = rig.buf(rec.as_bytes().len() as u64, 0);
        step.write(0, rec.as_bytes()).unwrap();
        let dims: StepDims = rec.dims();
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers
                .embed_gather(
                    &rig.stream,
                    step.ptr(),
                    dims,
                    table.ptr(),
                    GgmlType::Q3_K,
                    table_rows as u32,
                    D,
                    emb_out.ptr(),
                )
                .unwrap()
        });
        let bytes = f64::from(c.rows()) * f64::from(D) * (4.0 + 0.43);
        (us, gbs(bytes, us))
    });
    add("head_norm q (rows*24 x 256)", &mut |c| {
        let n = c.rows() * ATTN.n_head;
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers
                .head_norm(
                    &rig.stream,
                    NormKind::Rms,
                    hn_x.ptr(),
                    Some(qn.ptr()),
                    hn_y.ptr(),
                    n,
                    256,
                    1e-6,
                )
                .unwrap()
        });
        (us, gbs(f64::from(n) * 256.0 * 8.0, us))
    });
    add("conv_window + l2 norm", &mut |c| {
        let rec = record(c, false);
        let step = rig.buf(rec.as_bytes().len() as u64, 0);
        step.write(0, rec.as_bytes()).unwrap();
        let us = rig.time(iters(c), |i| unsafe {
            let off = if with_state_rotation {
                u64::from(i) % LAYERS as u64 * u64::from(max_seqs) * conv_stride * 4
            } else {
                0
            };
            rig.layers
                .conv_window(
                    &rig.stream,
                    step.ptr(),
                    rec.dims(),
                    SHAPE,
                    Some(1e-6),
                    qkv.ptr(),
                    taps.ptr(),
                    conv_state.ptr().offset(off),
                    conv_stride,
                    q_out.ptr(),
                    kv_out.ptr(),
                )
                .unwrap()
        });
        let bytes =
            f64::from(c.rows()) * f64::from(cd) * 8.0 + f64::from(c.seqs) * conv_len as f64 * 4.0;
        (us, gbs(bytes, us))
    });
    add("conv_commit (deferred)", &mut |c| {
        let rec = record(c, false);
        let step = rig.buf(rec.as_bytes().len() as u64, 0);
        step.write(0, rec.as_bytes()).unwrap();
        let us = rig.time(iters(c), |i| unsafe {
            let off = if with_state_rotation {
                u64::from(i) % LAYERS as u64 * u64::from(max_seqs) * conv_stride * 4
            } else {
                0
            };
            rig.layers
                .conv_commit(
                    &rig.stream,
                    step.ptr(),
                    rec.dims(),
                    SHAPE,
                    CommitSet::Deferred,
                    qkv.ptr(),
                    conv_state.ptr().offset(off),
                    conv_stride,
                )
                .unwrap()
        });
        let bytes = f64::from(c.seqs) * conv_len as f64 * 8.0;
        (us, gbs(bytes, us))
    });
    for (name, format, direct) in [
        ("delta_window f32, deferred", StateFormat::F32, false),
        ("delta_window f32, direct", StateFormat::F32, true),
        ("delta_window f16, deferred", StateFormat::F16, false),
        ("delta_window f16, direct", StateFormat::F16, true),
    ] {
        add(name, &mut |c| {
            let rec = record(c, direct);
            let step = rig.buf(rec.as_bytes().len() as u64, 0);
            step.write(0, rec.as_bytes()).unwrap();
            let (state, stride_bytes) = match format {
                StateFormat::F32 => (&delta32, 4u64),
                StateFormat::F16 => (&delta16, 2u64),
            };
            let us = rig.time(iters(c), |i| unsafe {
                let off = if with_state_rotation {
                    u64::from(i) % LAYERS as u64 * u64::from(max_seqs) * delta_stride * stride_bytes
                } else {
                    0
                };
                rig.layers
                    .delta_window(
                        &rig.stream,
                        step.ptr(),
                        rec.dims(),
                        SHAPE,
                        format,
                        q_out.ptr(),
                        kv_out.ptr(),
                        alpha.ptr(),
                        beta.ptr(),
                        dt.ptr(),
                        ssm_a.ptr(),
                        state.ptr().offset(off),
                        delta_stride,
                        out.ptr(),
                    )
                    .unwrap()
            });
            let state_bytes = f64::from(c.seqs)
                * delta_len as f64
                * stride_bytes as f64
                * if direct { 2.0 } else { 1.0 };
            let io = f64::from(c.rows()) * f64::from(vd) * 4.0
                + f64::from(c.rows()) * f64::from(kd + vd) * 4.0 * 0.5;
            (us, gbs(state_bytes + io, us))
        });
    }
    for (name, format) in [
        ("delta_commit f32 (all rows)", StateFormat::F32),
        ("delta_commit f16 (all rows)", StateFormat::F16),
    ] {
        add(name, &mut |c| {
            let rec = record(c, false);
            let step = rig.buf(rec.as_bytes().len() as u64, 0);
            step.write(0, rec.as_bytes()).unwrap();
            let (state, stride_bytes) = match format {
                StateFormat::F32 => (&delta32, 4u64),
                StateFormat::F16 => (&delta16, 2u64),
            };
            let us = rig.time(iters(c), |i| unsafe {
                let off = if with_state_rotation {
                    u64::from(i) % LAYERS as u64 * u64::from(max_seqs) * delta_stride * stride_bytes
                } else {
                    0
                };
                rig.layers
                    .delta_commit(
                        &rig.stream,
                        step.ptr(),
                        rec.dims(),
                        SHAPE,
                        format,
                        kv_out.ptr(),
                        alpha.ptr(),
                        beta.ptr(),
                        dt.ptr(),
                        ssm_a.ptr(),
                        state.ptr().offset(off),
                        delta_stride,
                    )
                    .unwrap()
            });
            let bytes = f64::from(c.seqs) * delta_len as f64 * stride_bytes as f64 * 2.0;
            (us, gbs(bytes, us))
        });
    }
    add("gated_norm_quant (silu)", &mut |c| {
        let items = c.rows() * SHAPE.n_v_heads;
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers
                .gated_norm_quant(
                    &rig.stream,
                    OutputGate::Silu,
                    out.ptr(),
                    z.ptr(),
                    norm_w.ptr(),
                    act_o.ptr(),
                    items,
                    128,
                    1e-6,
                )
                .unwrap()
        });
        let n = f64::from(items) * 128.0;
        (us, gbs(n * 8.0 + n * 5.25, us))
    });
    add("rope_table (once per step)", &mut |c| {
        let rec = record(c, false);
        let step = rig.buf(rec.as_bytes().len() as u64, 0);
        step.write(0, rec.as_bytes()).unwrap();
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers
                .rope_table(&rig.stream, step.ptr(), rec.dims(), table_rope.ptr(), rope)
                .unwrap()
        });
        (us, 0.0)
    });
    add("attn_prep (24+4 heads x 256)", &mut |c| {
        let rec = record(c, false);
        let step = rig.buf(rec.as_bytes().len() as u64, 0);
        step.write(0, rec.as_bytes()).unwrap();
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers
                .attn_prep(
                    &rig.stream,
                    step.ptr(),
                    rec.dims(),
                    ATTN,
                    64,
                    1e-6,
                    qg.ptr(),
                    kin.ptr(),
                    qn.ptr(),
                    kn.ptr(),
                    table_rope.ptr(),
                    q_att.ptr(),
                    Some(g_att.ptr()),
                    k_att.ptr(),
                )
                .unwrap()
        });
        let r = f64::from(c.rows());
        let bytes = r
            * (f64::from(ATTN.n_head) * 512.0 * 4.0 * 2.0
                + f64::from(ATTN.n_kv_head) * 256.0 * 8.0);
        (us, gbs(bytes, us))
    });
    add("swiglu_quant (rows x 17408)", &mut |c| {
        let n = c.rows() * FF;
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers.swiglu_quant(&rig.stream, gate.ptr(), up.ptr(), act_ff.ptr(), n).unwrap()
        });
        (us, gbs(f64::from(n) * (8.0 + 5.25), us))
    });
    add("gather_rows + norm (5120)", &mut |c| {
        let rec = record(c, false);
        let step = rig.buf(rec.as_bytes().len() as u64, 0);
        step.write(0, rec.as_bytes()).unwrap();
        let us = rig.time(iters(c), |_| unsafe {
            rig.layers
                .gather_rows(
                    &rig.stream,
                    step.ptr(),
                    rec.dims(),
                    hidden.ptr(),
                    Some(norm_d.ptr()),
                    head_in.ptr(),
                    D,
                    1e-6,
                )
                .unwrap()
        });
        (us, gbs(f64::from(c.seqs) * f64::from(D) * 8.0, us))
    });
    print_table(title, configs, &lines);
}

fn decode_configs() -> Vec<Config> {
    vec![
        Config { label: "1 seq x 1 row".into(), seqs: 1, rows_per_seq: 1 },
        Config { label: "1 seq x 4 rows".into(), seqs: 1, rows_per_seq: 4 },
        Config { label: "1 seq x 8 rows".into(), seqs: 1, rows_per_seq: 8 },
        Config { label: "4 seqs x 1 row".into(), seqs: 4, rows_per_seq: 1 },
        Config { label: "4 seqs x 2 rows".into(), seqs: 4, rows_per_seq: 2 },
    ]
}

fn prompt_configs() -> Vec<Config> {
    vec![
        Config { label: "256 rows".into(), seqs: 1, rows_per_seq: 256 },
        Config { label: "1024 rows".into(), seqs: 1, rows_per_seq: 1024 },
        Config { label: "2048 rows".into(), seqs: 1, rows_per_seq: 2048 },
    ]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let only_device: Option<u32> = args.first().and_then(|a| a.parse().ok());
    let which = args.get(1).map_or("all", String::as_str);
    for info in devices()? {
        if only_device.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal)?;
        let kernels = Kernels::load(&device)?;
        let mut rig = Rig {
            stream: Stream::new(&device)?,
            layers: LayerKernels::load(&kernels)?,
            start: Event::timed(&device)?,
            stop: Event::timed(&device)?,
            device: device.clone(),
        };
        println!("\n=== Vulkan{} {}", info.ordinal, info.name);
        rig.layers.prepare()?;
        if which == "all" || which == "decode" {
            run(&rig, &decode_configs(), "Decode steps", true, None);
        }
        if which == "all" || which == "prompt" {
            run(&rig, &prompt_configs(), "Prompt rows of one sequence", false, None);
        }
        if which == "variants" {
            // Only the delta rule, for every compiled shape of it.
            let chosen = std::env::var("LAYERS_VARIANTS").ok();
            for variant in LayerKernels::delta_variants() {
                if chosen.as_ref().is_some_and(|list| !list.split(',').any(|v| v == variant)) {
                    continue;
                }
                rig.layers.select_delta_variant(variant).unwrap();
                run(
                    &rig,
                    &decode_configs(),
                    &format!("delta rule, shape {variant}: decode"),
                    true,
                    Some("delta_window f32"),
                );
                run(
                    &rig,
                    &prompt_configs(),
                    &format!("delta rule, shape {variant}: prompt"),
                    false,
                    Some("delta_window f32"),
                );
            }
        }
    }
    Ok(())
}
