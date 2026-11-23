//! The image tower on a device (`ares_model::qwen3vl::DeviceTower`) against the
//! exact CPU reference, for both projector files, in both homes of the weights.
//! These tests need files that are not in the repository and say so when they
//! skip:
//!
//! - `ARES_TEST_QWEN3VL_27B`, `ARES_TEST_QWEN3VL_FLASH`: the two projectors;
//! - `ARES_TEST_QWEN3VL_REF`: the reference dumps of llama.cpp, of which the
//!   eleven square patterns are read (`pixels/<pattern>-<side>.f32`);
//! - `ARES_TEST_QWEN3VL_PILLOW`: own images as bytes (`manifest.txt`,
//!   `<name>.src.rgb`), wide, tall and up to 4060 rows, which are resized and
//!   normalised with `qwen3vl::image` here;
//! - `ARES_TEST_QWEN3VL_ROWS`: a directory where the reference's rows of every
//!   case are kept (`<27b|flash>/<case>.<exact|half>.f32`), computed when
//!   missing (the reference took 21 to 36 minutes on 12 threads, under other
//!   load, for an image of 4060 rows); delete it when the reference changes;
//! - `ARES_TEST_QWEN3VL_MAX_ROWS`: the largest image to run, in rows, 1024
//!   unless set (4096 for all);
//! - `ARES_TEST_QWEN3VL_THREADS`: threads of the reference, all unless set;
//! - `ARES_TEST_CUDA=1` for the tests that open a device, and
//!   `ARES_TEST_QWEN3VL_DEVICES` (`0,1`) for the devices they run on, all unless
//!   set.
//!
//! The tolerance is measured per image, as the reference's own tests measure
//! theirs: how far the roundings of the device's products carry this image is
//! the distance between the reference with them (`Arithmetic::HalfOperands`)
//! and without (`Arithmetic::Exact`); the device may be a few times that far
//! from the exact rows.

use std::time::Instant;

use ares_cuda::{Device, Kernels, Stream};
use ares_gguf::Gguf;
use ares_model::qwen3vl::{
    Arithmetic, Config, DeviceTower, Home, Limits, Reference, Stage, Weights, normalize,
    resize_bicubic, target_size,
};
use common::gpu::gpus;
use common::helpers::{env_path, read_f32};

mod common;

/// One image of the tests: normalised pixels whose sides are multiples of 32.
struct Case {
    name: String,
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

impl Case {
    fn rows(&self) -> usize {
        (self.width / 32) * (self.height / 32)
    }
}

/// The largest image to run, in rows.
fn max_rows() -> usize {
    std::env::var("ARES_TEST_QWEN3VL_MAX_ROWS").map_or(1024, |s| s.parse().unwrap())
}

/// The eleven square patterns of llama.cpp's debug program and the own images
/// of the Pillow directory, smallest first, up to [`max_rows`].
fn cases(config: &Config) -> Option<Vec<Case>> {
    let (Some(reference), Some(pillow)) =
        (env_path("ARES_TEST_QWEN3VL_REF"), env_path("ARES_TEST_QWEN3VL_PILLOW"))
    else {
        return None;
    };
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(reference.join("pixels")).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        let Some(stem) = name.strip_suffix(".f32") else { continue };
        let side: usize = stem.rsplit_once('-').unwrap().1.parse().unwrap();
        let pixels = read_f32(&reference.join("pixels").join(&name));
        assert_eq!(pixels.len(), side * side * 3, "{name}");
        cases.push(Case { name: stem.to_owned(), width: side, height: side, pixels });
    }
    let manifest = std::fs::read_to_string(pillow.join("manifest.txt")).unwrap();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let (width, height): (usize, usize) =
            (fields[1].parse().unwrap(), fields[2].parse().unwrap());
        let source = std::fs::read(pillow.join(format!("{}.src.rgb", fields[0]))).unwrap();
        let (w, h) = target_size(config, Limits::default(), width, height).unwrap();
        let pixels = normalize(config, &resize_bicubic(&source, width, height, w, h));
        cases.push(Case { name: fields[0].to_owned(), width: w, height: h, pixels });
    }
    cases.retain(|case| case.rows() <= max_rows());
    cases.sort_by_key(|case| (case.rows(), case.name.clone()));
    assert!(!cases.is_empty());
    Some(cases)
}

/// A projector file, its tag in the directory of rows, and the width of its rows.
struct Projector {
    tag: &'static str,
    gguf: Gguf,
}

impl Projector {
    fn open(variable: &str, tag: &'static str) -> Option<Self> {
        Some(Self { tag, gguf: Gguf::open(env_path(variable)?).unwrap() })
    }
}

/// The reference's rows of a case in `arithmetic`, from the directory of rows or
/// computed and stored there.
fn reference_rows(
    projector: &Projector,
    config: &Config,
    weights: &Weights<'_>,
    case: &Case,
    arithmetic: Arithmetic,
) -> Vec<f32> {
    let dir =
        env_path("ARES_TEST_QWEN3VL_ROWS").expect("ARES_TEST_QWEN3VL_ROWS").join(projector.tag);
    let kind = match arithmetic {
        Arithmetic::Exact => "exact",
        Arithmetic::HalfOperands => "half",
        Arithmetic::ForkCpu => "fork",
    };
    let path = dir.join(format!("{}.{kind}.f32", case.name));
    let values = case.rows() * config.n_out;
    if path.exists() {
        let rows = read_f32(&path);
        assert_eq!(rows.len(), values, "{}: a file of rows of another size", path.display());
        return rows;
    }
    let threads = std::env::var("ARES_TEST_QWEN3VL_THREADS")
        .map_or(std::thread::available_parallelism().map_or(1, usize::from), |s| {
            s.parse().unwrap()
        });
    let started = Instant::now();
    let rows = Reference::new(config, weights)
        .with_threads(threads)
        .with_arithmetic(arithmetic)
        .encode(&case.pixels, case.width, case.height)
        .unwrap();
    eprintln!(
        "{} {}: {} rows of the reference ({kind}) in {:.1} s",
        projector.tag,
        case.name,
        case.rows(),
        started.elapsed().as_secs_f64()
    );
    assert!(rows.data.iter().all(|v| v.is_finite()), "{}: not finite", case.name);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes: Vec<u8> = rows.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).unwrap();
    std::fs::rename(&partial, &path).unwrap();
    rows.data
}

/// The root of the summed squared difference over that of `b`'s squares, and
/// the largest difference.
fn distance(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut d, mut n, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let diff = f64::from(*x) - f64::from(*y);
        d += diff * diff;
        n += f64::from(*y).powi(2);
        worst = worst.max(diff.abs());
    }
    ((d / n).sqrt(), worst)
}

/// The reference's rows of every case in both arithmetics, computed into
/// `ARES_TEST_QWEN3VL_ROWS` where they are missing: the exact ones of both
/// projectors first. Needs no device; the device tests compute what they miss
/// themselves.
#[test]
fn the_reference_rows_of_the_device_tests() {
    let projectors: Vec<Projector> = [
        Projector::open("ARES_TEST_QWEN3VL_27B", "27b"),
        Projector::open("ARES_TEST_QWEN3VL_FLASH", "flash"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if projectors.is_empty() || env_path("ARES_TEST_QWEN3VL_ROWS").is_none() {
        return;
    }
    for arithmetic in [Arithmetic::Exact, Arithmetic::HalfOperands] {
        for projector in &projectors {
            let config = Config::from_gguf(&projector.gguf).unwrap();
            let weights = Weights::load(&projector.gguf, &config).unwrap();
            let Some(cases) = cases(&config) else { return };
            for case in &cases {
                reference_rows(projector, &config, &weights, case, arithmetic);
            }
        }
    }
    for projector in &projectors {
        let config = Config::from_gguf(&projector.gguf).unwrap();
        let weights = Weights::load(&projector.gguf, &config).unwrap();
        for case in &cases(&config).unwrap() {
            let exact = reference_rows(projector, &config, &weights, case, Arithmetic::Exact);
            let half = reference_rows(projector, &config, &weights, case, Arithmetic::HalfOperands);
            let (rel, max) = distance(&half, &exact);
            eprintln!(
                "{} {}: {} rows, the reference with f16 operands against the exact one: rel \
                 {rel:.2e} max {max:.2e}",
                projector.tag,
                case.name,
                case.rows()
            );
        }
    }
}

/// The residual after the first and the second block on the device against the
/// exact reference's, on a few small images of the 27B's projector: up to there
/// the roundings of f16 operands have not been carried far, so the device must be
/// close to the reference value by value, not only as a whole. Then every
/// device, both homes.
#[test]
fn the_first_blocks_on_the_device_match_the_reference() {
    let (Some(devices), Some(projector)) =
        (gpus(), Projector::open("ARES_TEST_QWEN3VL_27B", "27b"))
    else {
        return;
    };
    let config = Config::from_gguf(&projector.gguf).unwrap();
    let weights = Weights::load(&projector.gguf, &config).unwrap();
    let Some(cases) = cases(&config) else { return };
    let chosen: Vec<&Case> = cases
        .iter()
        .filter(|c| {
            ["gray-64", "gradient-32x256", "rainbow-224", "text-large-768x256"]
                .contains(&c.name.as_str())
        })
        .collect();
    assert!(!chosen.is_empty());
    for case in chosen {
        let mut taps: Vec<Vec<f32>> = Vec::new();
        Reference::new(&config, &weights)
            .encode_traced(&case.pixels, case.width, case.height, &mut |stage, values| {
                if matches!(stage, Stage::Positions | Stage::Block(0) | Stage::Block(1)) {
                    taps.push(values.to_vec());
                }
            })
            .unwrap();
        for device in picked(devices) {
            let kernels = Kernels::load(device).unwrap();
            let stream = Stream::new(device, 0).unwrap();
            for home in [Home::Vram, Home::Ram] {
                let mut tower =
                    DeviceTower::load(device, &kernels, &config, &weights, home).unwrap();
                let scratch = tower.scratch(case.rows() as u32).unwrap();
                for (blocks, reference) in taps.iter().enumerate() {
                    let ours = tower
                        .residual_after(
                            &stream,
                            &scratch,
                            &case.pixels,
                            case.width,
                            case.height,
                            blocks,
                        )
                        .unwrap();
                    let (rel, max) = distance(&ours, reference);
                    eprintln!(
                        "{} on {} ({home:?}): after {blocks} blocks rel {rel:.2e} max {max:.2e}",
                        case.name,
                        device.info().name
                    );
                    assert!(ours.iter().all(|v| v.is_finite()));
                    assert!(rel < 2e-3, "{}: after {blocks} blocks rel {rel:.2e}", case.name);
                }
            }
        }
    }
}

/// How many times the distance that the f16 operands of the products carry an
/// image (the reference with them against the exact one) the device may be from
/// the exact rows: it makes those roundings, in another order.
const ROUNDING_NOISE: f64 = 3.0;
/// And this much besides, for images on which the roundings change little.
const FLOOR: f64 = 1e-3;

/// The devices to run on: `ARES_TEST_QWEN3VL_DEVICES` (`0,1`), all unless set.
fn picked(devices: &[Device]) -> Vec<&Device> {
    match std::env::var("ARES_TEST_QWEN3VL_DEVICES") {
        Ok(list) => list.split(',').map(|i| &devices[i.trim().parse::<usize>().unwrap()]).collect(),
        Err(_) => devices.iter().collect(),
    }
}

/// Every case on every device in both homes: the two homes give the same rows
/// bit for bit, and those are within the image's own tolerance of the exact
/// reference.
fn against_the_reference(variable: &str, tag: &'static str, n_out: usize) {
    let (Some(devices), Some(projector)) = (gpus(), Projector::open(variable, tag)) else {
        return;
    };
    let config = Config::from_gguf(&projector.gguf).unwrap();
    assert_eq!(config.n_out, n_out);
    let weights = Weights::load(&projector.gguf, &config).unwrap();
    let Some(cases) = cases(&config) else { return };
    if env_path("ARES_TEST_QWEN3VL_ROWS").is_none() {
        return;
    }
    let references: Vec<(Vec<f32>, f64)> = cases
        .iter()
        .map(|case| {
            let exact = reference_rows(&projector, &config, &weights, case, Arithmetic::Exact);
            let half =
                reference_rows(&projector, &config, &weights, case, Arithmetic::HalfOperands);
            let between = distance(&half, &exact).0;
            (exact, between)
        })
        .collect();
    let largest = cases.iter().map(Case::rows).max().unwrap() as u32;
    for device in picked(devices) {
        let kernels = Kernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut vram = DeviceTower::load(device, &kernels, &config, &weights, Home::Vram).unwrap();
        let mut ram = DeviceTower::load(device, &kernels, &config, &weights, Home::Ram).unwrap();
        let scratch = vram.scratch(largest).unwrap();
        let mut worst = (0.0f64, String::new());
        for (case, (exact, between)) in cases.iter().zip(&references) {
            let started = Instant::now();
            let resident =
                vram.encode(&stream, &scratch, &case.pixels, case.width, case.height).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let streamed =
                ram.encode(&stream, &scratch, &case.pixels, case.width, case.height).unwrap();
            assert_eq!(
                (resident.nx * resident.ny, resident.width),
                (case.rows(), n_out),
                "{}",
                case.name
            );
            assert!(resident.data.iter().all(|v| v.is_finite()), "{}: not finite", case.name);
            let differing = resident
                .data
                .iter()
                .zip(&streamed.data)
                .filter(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(differing.count(), 0, "{}: the homes differ", case.name);
            let (rel, max) = distance(&resident.data, exact);
            let allowed = ROUNDING_NOISE * between + FLOOR;
            eprintln!(
                "{tag} {} on {}: {} rows, rel {rel:.2e} max {max:.2e} from the exact reference \
                 (f16 operands carry it {between:.2e}, allowed {allowed:.2e}), {:.0} ms",
                case.name,
                device.info().name,
                case.rows(),
                seconds * 1000.0
            );
            if rel / allowed > worst.0 {
                worst = (rel / allowed, case.name.clone());
            }
            assert!(rel < allowed, "{tag} {}: rel {rel:.2e} of {allowed:.2e}", case.name);
        }
        eprintln!(
            "{tag} on {}: the closest to its tolerance is {} at {:.0} % of it",
            device.info().name,
            worst.1,
            100.0 * worst.0
        );
    }
}

#[test]
fn the_27b_tower_on_the_device_gives_the_reference_rows() {
    against_the_reference("ARES_TEST_QWEN3VL_27B", "27b", 5120);
}

#[test]
fn the_flash_next_tower_on_the_device_gives_the_reference_rows() {
    against_the_reference("ARES_TEST_QWEN3VL_FLASH", "flash", 2560);
}

/// The tower as the runtime drives it ([`ares_model::qwen3vl::TowerAt`]): an
/// image's pass in ranges of units, one job after the other as the conductor
/// plans them, gives the rows of a whole pass ([`DeviceTower::encode`]) bit
/// for bit, in both homes; a second image after a pass that was left half done
/// starts afresh.
#[test]
fn a_pass_in_ranges_of_units_gives_the_rows_of_a_whole_pass() {
    use std::sync::Arc;

    use ares_model::qwen3vl::Projector;
    use ares_runtime::TowerJob;

    let (Some(gpus), Some(path)) = (gpus(), env_path("ARES_TEST_QWEN3VL_27B")) else { return };
    let device = &gpus[0];
    let kernels = Kernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let limits = Limits { min_rows: 8, max_rows: 300 };
    // An image of 640 x 480 pixels with no symmetry: 20 x 15 rows.
    let rgb: Vec<u8> =
        (0..640 * 480 * 3).map(|i: u32| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
    for home in [Home::Ram, Home::Vram] {
        let projector = Projector::open(Gguf::open(&path).unwrap(), limits, home).unwrap();
        let image = Arc::new(projector.prepare(&rgb, 640, 480).unwrap());
        assert_eq!((image.nx, image.ny), (20, 15));
        // The whole pass.
        let config = projector.config().clone();
        let weights_file = Gguf::open(&path).unwrap();
        let weights = Weights::load(&weights_file, &config).unwrap();
        let mut whole = DeviceTower::load(device, &kernels, &config, &weights, home).unwrap();
        let scratch = whole.scratch(300).unwrap();
        let pixels = normalize(&config, &image.pixels);
        let expected = whole.encode(&stream, &scratch, &pixels, 640, 480).unwrap().data;
        drop((whole, scratch));
        // In ranges, after half of another image's pass.
        let mut tower = projector.load(device, &kernels).unwrap();
        let units = projector.units();
        let other = Arc::new(projector.prepare(&rgb[..320 * 240 * 3], 320, 240).unwrap());
        let mut half = TowerJob { image: other, units: 0..units / 2, rows: Vec::new(), ms: 0.0 };
        tower.run(&stream, &mut half).unwrap();
        let mut rows = Vec::new();
        let mut from = 0;
        for size in [1, 3, 7, units] {
            let to = (from + size).min(units);
            let mut job = TowerJob { image: image.clone(), units: from..to, rows, ms: 0.0 };
            tower.run(&stream, &mut job).unwrap();
            rows = job.rows;
            from = to;
            if from == units {
                break;
            }
        }
        assert_eq!(rows.len(), expected.len());
        let same = rows.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "{home:?}: the rows of a pass in ranges differ from a whole pass");
    }
}
