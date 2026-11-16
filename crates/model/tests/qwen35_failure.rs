//! A device that fails reaches the caller as an error that names the device
//! and the cause, and the runtime neither hangs in its start nor in a step.
//!
//! The real Qwen3.5-0.8B split over two GPUs, with one part that fails on
//! purpose ([`ares_runtime::testing::Faulty`]). The tensor split is the one
//! that can hang: its merge kernels are enqueued by one thread per device and
//! each waits on its device for the other's flag, so a thread that fails
//! before it has enqueued its half leaves the kernel of the other waiting for
//! a flag nobody sets, and the step, the start and the shutdown with it.
//!
//! Needs `ARES_TEST_CUDA=1`, `ARES_TEST_QWEN35_MODEL` and two GPUs
//! (`ARES_LIBCUDA` with the patched driver library makes them exchange
//! directly, which is the path that hangs).

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ares_api::{
    Engine, EngineError, Event, EventSink, GenerateRequest, Message, Prompt, RequestId, Role,
    Sampling,
};
use ares_cuda::{Device, DeviceBuffer, Precision, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::testing::{Failure, Fault, Faulty};
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig, RuntimeError};

/// How long a failure may take to arrive. A hang is forever; nothing here
/// takes more than a few seconds when the runtime works.
const PATIENCE: Duration = Duration::from_secs(60);

fn model_path() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    if device_count().expect("the CUDA driver") < 2 {
        eprintln!("skipped: the tensor split needs two devices");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN35_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to run this test");
            None
        }
    }
}

fn model(path: &PathBuf, fault: Fault) -> Faulty<Qwen35> {
    let options = Options {
        precision: Precision::Fast,
        max_rows: 64,
        logit_rows: 4,
        window_rows: 0,
        ..Options::default()
    };
    let inner =
        Qwen35::new(Gguf::open(path).unwrap(), "q", &Spread::Tensor(vec![0.5, 0.5]), options, 1024)
            .unwrap();
    Faulty::new(inner, fault)
}

/// Starts the runtime on a thread, so that a start that hangs is a failed
/// test and not a test that never ends.
fn start(path: &PathBuf, fault: Fault) -> Result<Runtime, RuntimeError> {
    let model = std::sync::Arc::new(model(path, fault));
    let (tx, rx) = mpsc::channel();
    let begun = Instant::now();
    std::thread::spawn(move || {
        let config = RuntimeConfig {
            pages: 16,
            cells_per_page: 128,
            max_slots: 2,
            ..RuntimeConfig::default()
        };
        let gpu = GpuConfig::new(&config, &*model);
        let devices =
            vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
        let _ = tx.send(Runtime::start_gpu(model, devices, config, gpu));
    });
    let started = rx.recv_timeout(PATIENCE).unwrap_or_else(|_| {
        panic!("the runtime did not start or fail within {PATIENCE:?}: it hangs ({fault:?})")
    });
    eprintln!("RES {fault:?}: the start ended after {:?}", begun.elapsed());
    started
}

fn say(runtime: &Runtime, text: &str) -> Result<String, EngineError> {
    let (tx, rx) = mpsc::channel();
    let mut answer = String::new();
    let sink = EventSink::new(move |event| {
        let _ = tx.send(event);
        true
    });
    runtime.submit(
        GenerateRequest {
            model: "q".into(),
            prompt: Prompt::Chat {
                messages: vec![Message::new(Role::User, text)],
                tools: None,
                variables: Vec::new(),
                continue_final: false,
            },
            sampling: Sampling::default(),
            max_tokens: Some(20),
            stop: Vec::new(),
            draft: true,
            structure: None,
            id: RequestId(1),
        },
        sink,
    )?;
    loop {
        match rx.recv_timeout(PATIENCE).unwrap_or_else(|_| {
            panic!("no event within {PATIENCE:?} after the request: the step hangs")
        }) {
            Event::Text(piece) => answer.push_str(&piece),
            Event::Logprobs(_) => {}
            Event::Done { .. } => return Ok(answer),
            Event::Error(error) => return Err(error),
        }
    }
}

fn names(error: &RuntimeError, device: usize, cause: &str) {
    let text = error.to_string();
    assert!(text.starts_with(&format!("device {device}: ")), "{text}");
    assert!(text.contains(&format!("(CUDA {device})")), "the device's name and number: {text}");
    assert!(text.contains(cause), "the cause: {text}");
}

#[test]
fn a_part_that_cannot_be_built_is_reported_with_its_device_and_the_start_ends() {
    let Some(path) = model_path() else { return };
    for device in [0, 1] {
        let fault = Fault { device, step: 0, how: Failure::Open };
        let error = start(&path, fault).map(|_| ()).unwrap_err();
        names(&error, device, "injected failure while opening the part");
    }
}

#[test]
fn a_device_that_runs_out_of_memory_while_loading_is_named_and_the_others_do_not_wait_for_it() {
    let Some(path) = model_path() else { return };
    // The other device measures its weight products and waits for this one's
    // times, which never come: it used to wait for two minutes.
    let device = Device::open(0, Wait::Block).unwrap();
    let ballast = DeviceBuffer::new(&device, device.mem_info().unwrap().free - (64 << 20)).unwrap();
    let nothing = Fault { device: 99, step: 0, how: Failure::Error };
    let error = start(&path, nothing).map(|_| ()).unwrap_err();
    drop(ballast);
    names(&error, 0, "memory");
}

#[test]
fn a_device_that_fails_in_the_warm_up_step_is_named_and_the_start_ends() {
    let Some(path) = model_path() else { return };
    // The other device has enqueued its half of the step and its kernels
    // wait for the one that failed.
    for device in [0, 1] {
        let fault = Fault { device, step: 1, how: Failure::Error };
        let error = start(&path, fault).map(|_| ()).unwrap_err();
        names(&error, device, "injected failure in step 1");
    }
}

#[test]
fn a_device_that_panics_in_a_step_is_named_and_the_start_ends() {
    let Some(path) = model_path() else { return };
    let fault = Fault { device: 1, step: 2, how: Failure::Panic };
    let error = start(&path, fault).map(|_| ()).unwrap_err();
    names(&error, 1, "panicked in a step: injected panic in step 2");
}

/// The steps the warm-up of the runtime takes with this model. It measures the
/// price of a step for every row count the model's windows and pieces of prompt
/// can have, so the number depends on the model's limits; a failure meant for
/// a request's own step has to come after them.
fn warm_up_steps(path: &PathBuf) -> u64 {
    let nothing = Fault { device: 99, step: 0, how: Failure::Error };
    let runtime = start(path, nothing).expect("a start without a failure");
    let calibration = runtime.snapshot().and_then(|snapshot| snapshot.calibration);
    runtime.shutdown();
    u64::from(calibration.expect("the warm-up measured the price of a step").steps)
}

#[test]
fn a_device_that_fails_while_serving_ends_the_request_with_an_error_and_the_runtime_stops() {
    let Some(path) = model_path() else { return };
    // The request's own steps come after the warm-up's.
    let step = warm_up_steps(&path) + 3;
    for device in [0, 1] {
        let fault = Fault { device, step, how: Failure::Error };
        let runtime = start(&path, fault).expect("the warm-up steps do not fail");
        let error = say(&runtime, "Say hello.").unwrap_err();
        let EngineError::Internal(text) = &error else { panic!("{error:?}") };
        assert!(text.starts_with(&format!("device {device}: ")), "{text}");
        assert!(text.contains(&format!("injected failure in step {step}")), "{text}");
        // Nothing else is accepted, and the shutdown ends.
        assert!(matches!(say(&runtime, "Again."), Err(EngineError::Internal(_))));
        let begun = Instant::now();
        runtime.shutdown();
        assert!(
            begun.elapsed() < Duration::from_secs(20),
            "the shutdown took {:?}",
            begun.elapsed()
        );
        eprintln!("RES device {device}: the shutdown took {:?}", begun.elapsed());
    }
}
