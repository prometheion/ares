//! The runtime on real devices: the same scenarios as `tests/conductor.rs`,
//! with kernels computing the picks, for one device, a tensor split and a
//! layer split.
//!
//! Opening a device takes VRAM that a running model may need, so these run
//! only with `ARES_TEST_CUDA=1`, one at a time:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-runtime --test gpu -- --test-threads=1
//! ```
//!
//! `ARES_LIBCUDA=<path>/libcuda.so.1` selects the patched driver library, with
//! which the two-device placements use peer access and without which they go
//! through host memory.
//!
//! What they prove, beyond the conductor's logic: the next token of a
//! sequence is the same whatever else runs in the step, bit for bit, on each
//! placement (the placements differ from each other, which is expected, as
//! their sums add in a different order), and the threads, streams and
//! exchanges between devices neither deadlock nor lose a step.

mod common;

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ares_cuda::{Device, Wait, device_count};
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use common::*;

fn gpus() -> Option<&'static [Device]> {
    static DEVICES: OnceLock<Option<Vec<Device>>> = OnceLock::new();
    DEVICES
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_CUDA").is_none() {
                eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
                return None;
            }
            let count = device_count().expect("the CUDA driver");
            assert!(count > 0, "ARES_TEST_CUDA is set but the driver reports no device");
            Some((0..count).map(|i| Device::open(i, Wait::Yield).expect("open device")).collect())
        })
        .as_deref()
}

/// The placements to test with the devices there are.
fn placements() -> Vec<(Split, usize)> {
    let count = gpus().map_or(0, <[Device]>::len);
    let mut all = vec![(Split::Single, 1)];
    if count >= 2 {
        all.push((Split::Tensor, 2));
        all.push((Split::Layer, 2));
    }
    all
}

struct Setup {
    model: SyntheticConfig,
    runtime: RuntimeConfig,
}

fn setup(split: Split, devices: usize) -> Setup {
    let mut model = SyntheticConfig::small(split, devices);
    model.context_length = 1024;
    Setup {
        model,
        runtime: RuntimeConfig {
            pages: 4096,
            cells_per_page: 16,
            max_slots: 64,
            ..RuntimeConfig::default()
        },
    }
}

fn start(setup: Setup) -> Runtime {
    let devices = gpus().expect("enabled");
    let used = setup.model.devices;
    let model = Arc::new(SyntheticModel::new(setup.model).unwrap());
    let gpu = GpuConfig::new(&setup.runtime, &*model);
    Runtime::start_gpu(model, devices[..used].to_vec(), setup.runtime, gpu)
        .expect("the runtime starts")
}

fn on_every_placement(check: impl Fn(Split, usize)) {
    if gpus().is_none() {
        return;
    }
    for (split, devices) in placements() {
        eprintln!("-- {split:?} over {devices} device(s)");
        check(split, devices);
    }
}

#[test]
fn output_is_independent_of_the_batch_on_every_placement() {
    on_every_placement(|split, devices| {
        for stagger in [Duration::ZERO, Duration::from_micros(300)] {
            let rt = start(setup(split, devices));
            output_is_independent_of_the_batch(&rt, stagger);
        }
    });
}

#[test]
fn the_text_depends_on_the_whole_history_on_every_placement() {
    on_every_placement(|split, devices| {
        let rt = start(setup(split, devices));
        // The same last token at the same position, a different first one.
        let a = alone(&rt, "aXz", Some(40)).text;
        let b = alone(&rt, "bXz", Some(40)).text;
        assert_ne!(a, b, "the history reaches the kernels");
        let distinct: std::collections::HashSet<char> = a.chars().collect();
        assert!(distinct.len() > 10, "not a degenerate output: {a}");
        // And the same twice.
        assert_eq!(alone(&rt, "aXz", Some(40)).text, a);
    });
}

#[test]
fn cancelling_frees_the_pages_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 4096;
        cancel_frees_pages_and_spares_the_others(&start(s));
    });
}

#[test]
fn more_requests_than_the_pool_holds_wait_in_line_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 256;
        // A request of up to 40 prompt tokens and 100 of output is charged
        // 3 pages and 7: three of them fit in 32 pages.
        s.runtime.pages = 32;
        more_requests_than_fit_wait_and_say_the_same(&start(s), 3);
    });
}

#[test]
fn a_request_that_gives_way_comes_back_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 256;
        s.runtime.pages = 24;
        s.runtime.output_ratio = 0.05;
        a_request_that_gives_way_comes_back_and_says_the_same(&start(s));
    });
}

#[test]
fn shutdown_with_requests_in_flight_joins_every_thread_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 4096;
        s.runtime.pages = 64 * 256;
        shutdown_ends_everything(&start(s));
        assert!(engine_threads().is_empty(), "threads left: {:?}", engine_threads());
    });
}

#[test]
fn requests_end_for_the_right_reason_on_every_placement() {
    on_every_placement(|split, devices| {
        let rt = start(setup(split, devices));
        requests_end_for_the_right_reason(&rt, 1024);
        bad_requests_are_refused_at_once(&rt, 1024);
    });
}

#[test]
fn the_end_token_ends_a_request_where_it_first_comes_up_on_every_placement() {
    on_every_placement(|split, devices| {
        the_end_token_ends_a_request_where_it_first_comes_up(&|eos| {
            let mut s = setup(split, devices);
            s.model.eos = eos;
            start(s)
        });
    });
}

#[test]
fn a_model_that_does_not_fit_is_refused_at_load_with_the_account() {
    let Some(devices) = gpus() else { return };
    let mut config = SyntheticConfig::small(Split::Single, 1);
    // Weights of about 40 GB.
    config.layers = 1;
    config.distinct_layers = 1;
    config.width = 51_200;
    config.attn_inner = 51_200;
    config.ffn_inner = 409_600;
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let runtime = RuntimeConfig { max_slots: 64, ..RuntimeConfig::default() };
    let gpu = GpuConfig::new(&runtime, &*model);
    let error = Runtime::start_gpu(model, devices[..1].to_vec(), runtime, gpu).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("synthetic weights") && text.contains("CUDA0"), "{text}");
}

#[test]
fn state_pages_are_reserved_before_the_model_is_built_and_do_not_change_the_text() {
    let Some(devices) = gpus() else { return };
    // A model that declares a kilobyte of state per cell: the pool's pages exist
    // on the device and the text is what it is without them.
    let plain = {
        let rt = start(setup(Split::Single, 1));
        outputs_of(&rt, 8, 3)
    };
    let mut s = setup(Split::Single, 1);
    s.model.state_bytes_per_cell = 1024;
    let rt = start(s);
    assert_eq!(outputs_of(&rt, 8, 3), plain);
    drop(rt);

    // A kilobyte more than the device holds is refused at load, and the
    // account names the pages.
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.state_bytes_per_cell = 1 << 30;
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let runtime = RuntimeConfig { max_slots: 64, ..RuntimeConfig::default() };
    let gpu = GpuConfig::new(&runtime, &*model);
    let error = Runtime::start_gpu(model, devices[..1].to_vec(), runtime, gpu).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("state pages") && text.contains("CUDA0"), "{text}");
}
