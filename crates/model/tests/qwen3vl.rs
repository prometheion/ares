//! The projector's pixel work and its reference pass on a small made-up
//! projector. Nothing here needs a file.

mod common;

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_model::qwen3vl::{
    Arithmetic, Config, Error, Limits, Reference, Stage, Weights, normalize, patch_position,
    patch_rows, resize_bicubic, target_size,
};
use common::TempDir;

/// The real projectors' shape as far as the size rule needs it.
fn real_shape() -> Config {
    Config {
        patch: 16,
        merge: 2,
        n_embd: 1152,
        n_ff: 4304,
        n_layer: 27,
        n_head: 16,
        eps: 1e-6,
        mean: [0.5; 3],
        std: [0.5; 3],
        n_out: 5120,
        table_side: 48,
    }
}

// --- pixels ----------------------------------------------------------------

#[test]
fn the_size_rule_gives_the_sizes_of_llama_cpp() {
    let c = real_shape();
    let size = |w, h, max_rows| target_size(&c, Limits { min_rows: 8, max_rows }, w, h).unwrap();
    // Rounded to the nearest multiple of 32, a tie to the even multiple.
    assert_eq!(size(512, 512, 4096), (512, 512));
    assert_eq!(size(640, 480, 4096), (640, 480));
    assert_eq!(size(800, 600, 4096), (800, 608));
    assert_eq!(size(1280, 720, 4096), (1280, 704));
    assert_eq!(size(1920, 1080, 4096), (1920, 1088));
    assert_eq!(size(3000, 200, 4096), (3008, 192));
    assert_eq!(size(1170, 2532, 4096), (1184, 2528));
    // 1.5 and 2.5 units both go to 2.
    assert_eq!(size(48, 512, 4096), (64, 512));
    assert_eq!(size(80, 512, 4096), (64, 512));
    assert_eq!(size(112, 512, 4096), (128, 512));
    // Too few rows: enlarged, each side rounded up.
    assert_eq!(size(16, 16, 4096), (96, 96));
    assert_eq!(size(64, 64, 4096), (96, 96));
    assert_eq!(size(100, 100, 4096), (96, 96));
    assert_eq!(size(1, 1, 4096), (96, 96));
    assert_eq!(size(32, 256, 4096), (32, 256));
    // Too many: shrunk by the square root of the excess, each side rounded down.
    assert_eq!(size(3840, 2160, 4096), (2720, 1536));
    assert_eq!(size(4032, 3024, 4096), (2336, 1760));
    assert_eq!(size(2048, 2048, 4096), (2048, 2048));
    assert_eq!(size(5000, 20, 4096), (4992, 32));
    assert_eq!(size(1920, 1080, 1024), (1344, 768));
    assert_eq!(size(4032, 3024, 1024), (1152, 864));
    assert_eq!(size(1170, 2532, 1024), (672, 1504));
    assert_eq!(size(1024, 1024, 300), (544, 544));
    for (w, h) in [(1usize, 9000usize), (7777, 3), (4096, 4096), (333, 555)] {
        let (tw, th) = size(w, h, 4096);
        assert!(
            tw.is_multiple_of(32) && th.is_multiple_of(32) && tw > 0 && th > 0,
            "{w} x {h} -> {tw} x {th}"
        );
    }
    assert!(matches!(
        target_size(&c, Limits::default(), 0, 5),
        Err(Error::Image { width: 0, height: 5, .. })
    ));
}

/// One gray value per pixel as RGB.
fn gray(values: &[u8]) -> Vec<u8> {
    values.iter().flat_map(|v| [*v; 3]).collect()
}

#[test]
fn the_resize_is_the_bicubic_filter_of_pillow() {
    // Shrinking 4 to 2 widens the filter to two input pixels a unit. The
    // first output pixel has its centre at 1.0 and reads all four inputs at
    // the distances 0.25, 0.25, 0.75 and 1.25 units: weights 0.8671875,
    // 0.8671875, 0.2265625 and -0.0703125, which sum to 1.890625. For the
    // inputs 0, 0, 255, 255 that is 255 * 0.15625 / 1.890625 = 21.07, and the
    // mirror image gives 255 * 1.734375 / 1.890625 = 233.93.
    assert_eq!(resize_bicubic(&gray(&[0, 0, 255, 255]), 4, 1, 2, 1), gray(&[21, 234]));
    assert_eq!(resize_bicubic(&gray(&[0, 0, 255, 255]), 1, 4, 1, 2), gray(&[21, 234]));
    // Enlarging 2 to 4 keeps the filter. Output 0 (centre 0.25) reads the
    // inputs at 0.25 and 1.25: 0.8671875 and -0.0703125, sum 0.796875, so
    // 100 and 200 give (86.71875 - 14.0625) / 0.796875 = 91.18. Output 1
    // (centre 0.75) reads them at 0.25 and 0.75: 0.8671875 and 0.2265625,
    // sum 1.09375, giving (86.71875 + 45.3125) / 1.09375 = 120.71.
    assert_eq!(resize_bicubic(&gray(&[100, 200]), 2, 1, 4, 1), gray(&[91, 121, 179, 209]));
    assert_eq!(resize_bicubic(&gray(&[100, 200]), 1, 2, 1, 4), gray(&[91, 121, 179, 209]));
    // Overshoot is cut to the range of a byte.
    assert_eq!(resize_bicubic(&gray(&[0, 255]), 2, 1, 4, 1), gray(&[0, 53, 202, 255]));
    // The channels are filtered apart, a flat image stays flat, and the same
    // size is a copy.
    let pixels: Vec<u8> = (0..5 * 3 * 3).map(|i| (i * 37 % 251) as u8).collect();
    assert_eq!(resize_bicubic(&pixels, 5, 3, 5, 3), pixels);
    let flat: Vec<u8> = [9u8, 130, 250].repeat(7 * 5);
    assert_eq!(resize_bicubic(&flat, 7, 5, 64, 96), [9u8, 130, 250].repeat(64 * 96));
    let red_only = resize_bicubic(&[10, 0, 0, 200, 0, 0], 2, 1, 4, 1);
    assert_eq!(red_only.iter().step_by(3).count(), 4);
    assert!(red_only.chunks(3).all(|p| p[1] == 0 && p[2] == 0));
}

#[test]
fn pixels_are_normalised_in_the_order_of_llama_cpp() {
    let c = real_shape();
    assert_eq!(normalize(&c, &[0, 255, 128]), [-1.0, 1.0, (128.0f32 / 255.0 - 0.5) / 0.5]);
}

#[test]
fn patch_rows_cut_the_image_in_the_mergers_order() {
    let c = real_shape();
    // Four patches by two: one line of two groups.
    let (width, height) = (64, 32);
    let pixels: Vec<f32> = (0..width * height * 3).map(|i| i as f32).collect();
    let (rows, nx, ny) = patch_rows(&c, &pixels, width, height).unwrap();
    assert_eq!((nx, ny, rows.len()), (2, 1, 8 * 768));
    let order = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)];
    for (i, (row, &(py, px))) in rows.as_chunks::<768>().0.iter().zip(&order).enumerate() {
        assert_eq!(patch_position(i, nx), (py, px));
        // Channel by channel, in a channel the patch's pixel rows.
        for (j, v) in row.iter().enumerate() {
            let (channel, ky, kx) = (j / 256, j / 16 % 16, j % 16);
            assert_eq!(
                *v,
                pixels[((py * 16 + ky) * width + px * 16 + kx) * 3 + channel],
                "{i} {j}"
            );
        }
    }
    assert!(patch_rows(&c, &pixels[..48 * 32 * 3], 48, 32).is_err());
    assert!(patch_rows(&c, &pixels[1..], width, height).is_err());
}

// --- a made-up projector ---------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn floats(&mut self, n: u64, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// Patches of 2 x 2 pixels, width 8, two heads of four values, two blocks, a
/// position table of 4 x 4 and rows of six values. `mm.0` is stored as BF16.
fn small() -> GgufBuilder {
    small_with("qwen3vl_merger", true, 2)
}

/// The same with other values for three keys.
fn small_with(projector: &str, use_gelu: bool, merge: u32) -> GgufBuilder {
    let (e, ff, out, patch) = (8u64, 12u64, 6u64, 2u64);
    let mut rng = Lcg(7);
    let mut b = GgufBuilder::new()
        .str("general.architecture", "clip")
        .str("clip.projector_type", projector)
        .bool("clip.use_gelu", use_gelu)
        .bool("clip.has_vision_encoder", true)
        .u32("clip.vision.projection_dim", out as u32)
        .u32("clip.vision.image_size", 8)
        .u32("clip.vision.patch_size", patch as u32)
        .u32("clip.vision.embedding_length", e as u32)
        .u32("clip.vision.feed_forward_length", ff as u32)
        .u32("clip.vision.block_count", 2)
        .u32("clip.vision.attention.head_count", 2)
        .f32("clip.vision.attention.layer_norm_epsilon", 1e-6)
        .f32_array("clip.vision.image_mean", &[0.5, 0.4, 0.3])
        .f32_array("clip.vision.image_std", &[0.5, 0.25, 0.2])
        .u32("clip.vision.spatial_merge_size", merge);
    let mut vector = |b: GgufBuilder, name: &str, dims: &[u64], scale: f32| {
        let values = rng.floats(dims.iter().product(), scale);
        b.tensor_f32(name, dims, &values)
    };
    b = vector(b, "v.patch_embd.weight", &[patch, patch, 3, e], 1.0);
    b = vector(b, "v.patch_embd.weight.1", &[patch, patch, 3, e], 1.0);
    b = vector(b, "v.patch_embd.bias", &[e], 1.0);
    b = vector(b, "v.position_embd.weight", &[e, 16], 2.0);
    for i in 0..2 {
        for (part, cols, rows) in
            [("attn_qkv", e, 3 * e), ("attn_out", e, e), ("ffn_up", e, ff), ("ffn_down", ff, e)]
        {
            b = vector(b, &format!("v.blk.{i}.{part}.weight"), &[cols, rows], 1.5);
            b = vector(b, &format!("v.blk.{i}.{part}.bias"), &[rows], 1.0);
        }
        for norm in ["ln1", "ln2"] {
            b = vector(b, &format!("v.blk.{i}.{norm}.weight"), &[e], 3.0);
            b = vector(b, &format!("v.blk.{i}.{norm}.bias"), &[e], 1.0);
        }
    }
    b = vector(b, "v.post_ln.weight", &[e], 3.0);
    b = vector(b, "v.post_ln.bias", &[e], 1.0);
    let mm0: Vec<u8> = Lcg(11)
        .floats(4 * e * 4 * e, 1.0)
        .iter()
        .flat_map(|v| ares_quant::f32_to_bf16(*v).to_le_bytes())
        .collect();
    b = b.tensor("mm.0.weight", &[4 * e, 4 * e], GgmlType::BF16, mm0);
    b = vector(b, "mm.0.bias", &[4 * e], 1.0);
    b = vector(b, "mm.2.weight", &[4 * e, out], 1.0);
    vector(b, "mm.2.bias", &[out], 1.0)
}

fn with_small<R>(builder: &GgufBuilder, run: impl FnOnce(&Config, &Weights<'_>) -> R) -> R {
    let dir = TempDir::new("qwen3vl");
    let gguf = Gguf::open(dir.file("mmproj.gguf", &builder.build())).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    run(&config, &weights)
}

fn load_error(builder: &GgufBuilder) -> Error {
    let dir = TempDir::new("qwen3vl-error");
    let gguf = Gguf::open(dir.file("mmproj.gguf", &builder.build())).unwrap();
    match Config::from_gguf(&gguf).and_then(|c| Weights::load(&gguf, &c).map(|_| ())) {
        Ok(()) => panic!("the projector should have been refused"),
        Err(e) => e,
    }
}

/// Normalised pixels of a `width` x `height` image with no symmetry.
fn pixels(width: usize, height: usize) -> Vec<f32> {
    Lcg(3).floats((width * height * 3) as u64, 2.0)
}

#[test]
fn the_configuration_is_read_and_strict() {
    with_small(&small(), |c, w| {
        assert_eq!((c.patch, c.merge, c.n_embd, c.n_ff, c.n_layer, c.n_head), (2, 2, 8, 12, 2, 2));
        assert_eq!(
            (c.n_out, c.table_side, c.head_dim(), c.align(), c.patch_len()),
            (6, 4, 4, 4, 12)
        );
        assert_eq!((c.mean, c.std, c.eps), ([0.5, 0.4, 0.3], [0.5, 0.25, 0.2], 1e-6));
        assert_eq!(
            (w.blocks.len(), w.mm0.weight.ty(), w.mm2.weight.rows()),
            (2, GgmlType::BF16, 6)
        );
    });
    let unknown = load_error(&small().u32("clip.vision.n_wa_pattern", 8));
    assert!(matches!(unknown, Error::UnknownKey { key } if key == "clip.vision.n_wa_pattern"));
    let other = load_error(&small_with("pixtral", true, 2));
    assert!(matches!(other, Error::Kind { key: "clip.projector_type", .. }));
    let relu = load_error(&small_with("qwen3vl_merger", false, 2));
    assert!(matches!(relu, Error::BadKey { key, .. } if key == "clip.use_gelu"));
    let merge = load_error(&small_with("qwen3vl_merger", true, 3));
    assert!(matches!(merge, Error::Inconsistent { .. }));
    let extra = load_error(&small().tensor_f32("v.blk.0.deepstack.0.weight", &[2], &[0.0; 2]));
    assert!(matches!(extra, Error::UnexpectedTensor { name } if name.contains("deepstack")));
    let shape = load_error(&small().tensor_f32("v.pre_ln.weight", &[8], &[0.0; 8]));
    assert!(matches!(shape, Error::UnexpectedTensor { .. }));
}

#[test]
fn an_image_that_is_not_made_of_merged_patches_is_refused() {
    with_small(&small(), |c, w| {
        let reference = Reference::new(c, w);
        assert!(matches!(reference.encode(&pixels(6, 8), 6, 8), Err(Error::Image { .. })));
        assert!(matches!(reference.encode(&pixels(8, 8)[1..], 8, 8), Err(Error::Image { .. })));
        let rows = reference.encode(&pixels(12, 8), 12, 8).unwrap();
        assert_eq!((rows.nx, rows.ny, rows.width, rows.data.len()), (3, 2, 6, 36));
        assert!(rows.data.iter().all(|v| v.is_finite()));
        // From bytes: 13 x 7 pixels become 12 x 8, and too few rows are enlarged.
        let bytes: Vec<u8> = (0..13 * 7 * 3).map(|i| (i * 29 % 256) as u8).collect();
        let limits = Limits { min_rows: 1, max_rows: 64 };
        let small_rows = reference.encode_rgb(&bytes, 13, 7, limits).unwrap();
        assert_eq!((small_rows.nx, small_rows.ny), (3, 2));
        let more = reference.encode_rgb(&bytes, 13, 7, Limits { min_rows: 20, max_rows: 64 });
        assert!(more.unwrap().data.len() >= 20 * 6);
    });
}

/// The stages of one pass, by stage.
fn trace(reference: &Reference<'_>, width: usize, height: usize) -> Vec<(Stage, Vec<f32>)> {
    let mut stages = Vec::new();
    let rows = reference
        .encode_traced(&pixels(width, height), width, height, &mut |s, v| {
            stages.push((s, v.to_vec()))
        })
        .unwrap();
    assert_eq!(stages.last().unwrap().1, rows.data);
    stages
}

fn stage(stages: &[(Stage, Vec<f32>)], which: Stage) -> &[f32] {
    &stages.iter().find(|(s, _)| *s == which).unwrap().1
}

#[test]
fn the_rows_do_not_depend_on_the_threads() {
    with_small(&small(), |c, w| {
        let one = trace(&Reference::new(c, w).with_threads(1), 12, 16);
        for threads in [2, 5, 64] {
            assert_eq!(trace(&Reference::new(c, w).with_threads(threads), 12, 16), one);
        }
    });
}

/// Where patch `i` of the merger's order lies: (row, column) of the grid of
/// `pw` patches a line.
fn place(i: usize, pw: usize) -> (usize, usize) {
    let (group, dy, dx) = (i / 4, i % 4 / 2, i % 2);
    (2 * (group / (pw / 2)) + dy, 2 * (group % (pw / 2)) + dx)
}

#[test]
fn patches_come_in_the_mergers_order_with_their_positions() {
    with_small(&small(), |c, w| {
        // 12 x 16 pixels: a grid of 6 x 8 patches, not the table's 4 x 4.
        let (width, height, pw, ph) = (12usize, 16usize, 6usize, 8usize);
        let stages = trace(&Reference::new(c, w), width, height);
        let (patches, positioned) =
            (stage(&stages, Stage::Patches), stage(&stages, Stage::Positions));
        let image = pixels(width, height);
        let table = |y: usize, x: usize, ch: usize| f64::from(w.position[(y * 4 + x) * 8 + ch]);
        for i in 0..pw * ph {
            let (row, col) = place(i, pw);
            for ch in 0..8 {
                // The patch as channel, row, column against both kernels.
                let mut sum = 0.0f64;
                for k in 0..12 {
                    let (channel, ky, kx) = (k / 4, k % 4 / 2, k % 2);
                    let pixel = image[((row * 2 + ky) * width + col * 2 + kx) * 3 + channel];
                    let weight = w.patch[0][ch * 12 + k] + w.patch[1][ch * 12 + k];
                    sum += f64::from(pixel) * f64::from(weight);
                }
                let patch = sum as f32 + w.patch_bias[ch];
                assert_eq!(patches[i * 8 + ch], patch, "patch {i} value {ch}");
                // The table read at row * 3 / 7 and col * 3 / 5, bilinearly.
                let (y, x) =
                    (row as f64 * 3.0 / (ph - 1) as f64, col as f64 * 3.0 / (pw - 1) as f64);
                let (y0, x0) = ((y.floor() as usize).min(3), (x.floor() as usize).min(3));
                let (y1, x1) = ((y0 + 1).min(3), (x0 + 1).min(3));
                let (dy, dx) = (y - y0 as f64, x - x0 as f64);
                let position = table(y0, x0, ch) * (1.0 - dx) * (1.0 - dy)
                    + table(y0, x1, ch) * dx * (1.0 - dy)
                    + table(y1, x0, ch) * (1.0 - dx) * dy
                    + table(y1, x1, ch) * dx * dy;
                let got = f64::from(positioned[i * 8 + ch]) - f64::from(patch);
                assert!(
                    (got - position).abs() < 2e-6,
                    "position of patch {i}: {got} vs {position}"
                );
            }
        }
        // A grid of the table's size reads the table itself.
        let stages = trace(&Reference::new(c, w), 8, 8);
        let (patches, positioned) =
            (stage(&stages, Stage::Patches), stage(&stages, Stage::Positions));
        for i in 0..16 {
            let (row, col) = place(i, 4);
            for ch in 0..8 {
                let expected = patches[i * 8 + ch] + w.position[(row * 4 + col) * 8 + ch];
                assert_eq!(positioned[i * 8 + ch], expected);
            }
        }
    });
}

#[test]
fn queries_and_keys_turn_by_row_and_column() {
    with_small(&small(), |c, w| {
        let pw = 6;
        let stages = trace(&Reference::new(c, w), 12, 16);
        let qkv = stage(&stages, Stage::Qkv(1));
        for (rotated, offset) in [(Stage::QueryRotated(1), 0), (Stage::KeyRotated(1), 8)] {
            let rotated = stage(&stages, rotated);
            for i in 0..48 {
                let (row, col) = place(i, pw);
                for head in 0..2 {
                    // A head of four values: pair (0, 2) turns by the row and
                    // pair (1, 3) by the column, both at the first frequency.
                    for (pair, position) in [(0, row), (1, col)] {
                        let at = |k: usize| f64::from(qkv[i * 24 + offset + head * 4 + k]);
                        let (a, b) = (at(pair), at(pair + 2));
                        let (sin, cos) = (position as f64).sin_cos();
                        let got = |k: usize| f64::from(rotated[i * 8 + head * 4 + k]);
                        assert!((got(pair) - (a * cos - b * sin)).abs() < 1e-5, "patch {i}");
                        assert!((got(pair + 2) - (a * sin + b * cos)).abs() < 1e-5, "patch {i}");
                    }
                }
            }
        }
    });
}

#[test]
fn the_fork_arithmetic_rounds_where_llama_cpp_rounds() {
    use ares_model::qwen3vl::reference::{gelu, gelu_fork};
    // The exact GELU against values of the definition.
    assert_eq!(gelu(0.0), 0.0);
    assert!((f64::from(gelu(1.0)) - 0.841_191_990_608_3).abs() < 1e-7);
    assert!((f64::from(gelu(-1.0)) + 0.158_808_009_391_7).abs() < 1e-7);
    assert!((f64::from(gelu(3.0)) - 2.996_362_634_335).abs() < 3e-7);
    // The table: an f16 in, an f16 out, cut at -10 and 10.
    assert_eq!(
        (gelu_fork(-10.0), gelu_fork(-37.5), gelu_fork(10.0), gelu_fork(12.25)),
        (0.0, 0.0, 10.0, 12.25)
    );
    for x in [-9.99f32, -2.3, -0.1, 1e-5, 0.3337, 1.0, 4.4, 9.99] {
        let table = gelu_fork(x);
        assert_eq!(ares_quant::f16_to_f32(ares_quant::f32_to_f16(table)), table);
        let rounded = ares_quant::f16_to_f32(ares_quant::f32_to_f16(x));
        assert!((table - gelu(rounded)).abs() <= 1e-3 * gelu(rounded).abs() + 1e-7, "{x}");
    }
    with_small(&small(), |c, w| {
        let exact = trace(&Reference::new(c, w), 8, 8);
        let fork = trace(&Reference::new(c, w).with_arithmetic(Arithmetic::ForkCpu), 8, 8);
        // Close, and not the same: the pixels, the GELU and the input of the
        // BF16 matrix are rounded.
        let (a, b) = (stage(&exact, Stage::Output), stage(&fork, Stage::Output));
        assert_ne!(a, b);
        let norm = a.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        let diff = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>().sqrt();
        assert!(diff / norm < 0.05, "relative difference {}", diff / norm);
        let hidden = stage(&fork, Stage::Gelu(0));
        assert!(hidden.iter().all(|v| ares_quant::f16_to_f32(ares_quant::f32_to_f16(*v)) == *v));
    });
}
