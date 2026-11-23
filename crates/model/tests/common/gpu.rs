//! What the tests that open GPUs share: the devices they open, only with
//! `ARES_TEST_CUDA` set, and the copies of values to and from a device.

use std::sync::OnceLock;

use ares_cuda::{Device, DeviceBuffer, Wait, device_count};

/// Whether the tests that open GPUs run: `ARES_TEST_CUDA` is set. Says so when
/// they do not.
pub fn enabled() -> bool {
    let enabled = std::env::var_os("ARES_TEST_CUDA").is_some();
    if !enabled {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
    }
    enabled
}

/// The device `ARES_TEST_DEVICE` names (0 by default), opened once for all the
/// tests of a file.
pub fn gpu() -> Option<&'static Device> {
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            if !enabled() {
                return None;
            }
            let ordinal =
                std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            Some(Device::open(ordinal, Wait::Block).expect("open device"))
        })
        .as_ref()
}

/// Every device of the machine, opened once for all the tests of a file.
pub fn gpus() -> Option<&'static [Device]> {
    static DEVICES: OnceLock<Option<Vec<Device>>> = OnceLock::new();
    DEVICES
        .get_or_init(|| {
            if !enabled() {
                return None;
            }
            let count = device_count().expect("the CUDA driver");
            assert!(count > 0, "ARES_TEST_CUDA is set but the driver reports no device");
            Some((0..count).map(|i| Device::open(i, Wait::Block).expect("open device")).collect())
        })
        .as_deref()
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
