//! What the tests that open GPUs share: the devices they open, only with
//! `ARES_TEST_CUDA` set and only one test at a time, and the copies of values
//! to and from a device.

use std::cell::RefCell;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use ares_cuda::{Device, DeviceBuffer, Wait, device_count};

/// The cards, which one test of a binary has at a time.
///
/// A test binary runs its tests on several threads unless it is told
/// otherwise, and tests that use the cards at once get in each other's way:
/// they share the driver's primary context of each device, the process's gate
/// of page-locked memory (`ares_cuda::gate`), and the VRAM each of them sizes
/// its model by, and an engine that one test left working takes the devices
/// from the next. So every test that opens a GPU holds this lock: [`enabled`]
/// takes it, and every function here that opens devices calls that. It is
/// held until the test's thread ends, which is after the test has dropped
/// its engines, sessions and devices, also when it fails. A test is the
/// thread the test harness runs it on: a thread the test starts itself must
/// not call these functions.
static CARDS: Mutex<()> = Mutex::new(());

thread_local! {
    /// The lock of the cards, held by the test that runs on this thread.
    static HELD: RefCell<Option<MutexGuard<'static, ()>>> = const { RefCell::new(None) };
}

/// Waits until no other test of the binary has the cards, and holds them for
/// the test on this thread until it ends. A test that failed while it held
/// them gave them up all the same.
fn hold_the_cards() {
    HELD.with_borrow_mut(|held| {
        if held.is_none() {
            *held = Some(CARDS.lock().unwrap_or_else(PoisonError::into_inner));
        }
    });
}

/// Whether the tests that open GPUs run: `ARES_TEST_CUDA` is set. Says so when
/// they do not. When they do, the test on this thread has the cards until it
/// ends (see [`CARDS`]).
pub fn enabled() -> bool {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return false;
    }
    hold_the_cards();
    true
}

/// The device `ARES_TEST_DEVICE` names (0 by default), opened once for all the
/// tests of a file.
pub fn gpu() -> Option<&'static Device> {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    if !enabled() {
        return None;
    }
    Some(DEVICE.get_or_init(|| {
        let ordinal =
            std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        Device::open(ordinal, Wait::Block).expect("open device")
    }))
}

/// Every device of the machine, opened once for all the tests of a file.
pub fn gpus() -> Option<&'static [Device]> {
    static DEVICES: OnceLock<Vec<Device>> = OnceLock::new();
    if !enabled() {
        return None;
    }
    Some(DEVICES.get_or_init(|| {
        let count = device_count().expect("the CUDA driver");
        assert!(count > 0, "ARES_TEST_CUDA is set but the driver reports no device");
        (0..count).map(|i| Device::open(i, Wait::Block).expect("open device")).collect()
    }))
}

/// The devices `ARES_TEST_DEVICES` names (a list of numbers, `0` by default),
/// opened for one test, with waits that spin.
pub fn devices() -> Option<Vec<Device>> {
    if !enabled() {
        return None;
    }
    let wanted = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0".into());
    let count = device_count().expect("the CUDA driver");
    Some(
        wanted
            .split(',')
            .map(|d| {
                let ordinal: u32 = d.trim().parse().expect("ARES_TEST_DEVICES: a list of numbers");
                assert!(ordinal < count, "device {ordinal} of {count}");
                Device::open(ordinal, Wait::Spin).expect("open device")
            })
            .collect(),
    )
}

/// `values` in a new buffer on `device`.
pub fn upload(device: &Device, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = DeviceBuffer::new(device, bytes.len().max(64) as u64).unwrap();
    buffer.write(0, &bytes).unwrap();
    buffer
}

/// The first `bytes` bytes of `buffer`.
pub fn download(buffer: &DeviceBuffer, bytes: u64) -> Vec<u8> {
    let mut raw = vec![0u8; bytes as usize];
    buffer.read(0, &mut raw).unwrap();
    raw
}
