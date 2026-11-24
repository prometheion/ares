//! The merges of a tensor split on real devices: the slots and the push of the
//! mesh with the merge kernels, against the sum computed on the host.
//!
//! Needs two GPUs and `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test mesh_merge
//! ```

use ares_cuda::{
    Device, DeviceBuffer, Kernels, LayerKernels, MergeNorm, Mesh, Route, Stream, Wait, Wire,
    device_count, partial_wire_bytes, peer_access,
};
use ares_quant::act;
use common::{f32_bytes, read_f32, upload_f32};

mod common;

const D: u32 = 1024;
const EPS: f32 = 1e-6;

fn gpus() -> Option<Vec<Device>> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    if device_count().expect("the CUDA driver") < 2 {
        eprintln!("skipped: the merges of a tensor split need two devices");
        return None;
    }
    Some((0..2).map(|i| Device::open(i, Wait::Block).expect("open device")).collect())
}

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// `n` rows of values around `scale`; row 1 is large enough to overflow
    /// f16 and row 2 is tiny.
    fn rows(&mut self, n: u32, scale: f32) -> Vec<f32> {
        (0..n * D)
            .map(|i| {
                let row = i / D;
                let factor = match row {
                    1 => 3.0e5,
                    2 => 1.0e-6,
                    _ => 1.0,
                };
                self.next() * scale * factor
            })
            .collect()
    }
}

/// One device's buffers of a merge.
struct Side {
    stream: Stream,
    kernels: Kernels,
    layers: LayerKernels,
    p: DeviceBuffer,
    x: DeviceBuffer,
    xn: DeviceBuffer,
    act: DeviceBuffer,
    act_ref: DeviceBuffer,
    weight: DeviceBuffer,
}

impl Side {
    /// The buffers of a device for merges of up to `max_rows` rows.
    fn new(device: &Device, max_rows: u32, weight: &[f32]) -> Self {
        let values = u64::from(max_rows * D) * 4;
        let act_bytes = Kernels::quantized_activation_bytes(D, max_rows);
        Side {
            stream: Stream::new(device, 0).unwrap(),
            kernels: Kernels::load(device).unwrap(),
            layers: LayerKernels::load(device).unwrap(),
            p: DeviceBuffer::new(device, values).unwrap(),
            x: DeviceBuffer::new(device, values).unwrap(),
            xn: DeviceBuffer::new(device, values).unwrap(),
            act: DeviceBuffer::new(device, act_bytes).unwrap(),
            act_ref: DeviceBuffer::new(device, act_bytes).unwrap(),
            weight: upload_f32(device, weight),
        }
    }
}

/// Enqueues one merge of `n` rows on both devices: pushed when `push`, through
/// slot `slot` otherwise.
fn merge(
    mesh: &Mesh,
    sides: &[Side],
    wire: Wire,
    n: u32,
    slot: u32,
    push: bool,
    norm: impl Fn(&Side) -> MergeNorm,
) {
    if push {
        for (rank, side) in sides.iter().enumerate() {
            let target = mesh.push(rank).expect("a push path");
            // SAFETY: the buffers hold `n` rows on the side's device and both
            // devices enqueue the merge of the same push.
            unsafe {
                side.layers
                    .merge_push(
                        &side.stream,
                        wire,
                        side.p.ptr(),
                        side.x.ptr(),
                        target,
                        rank == 0,
                        n,
                        D,
                        norm(side),
                        EPS,
                    )
                    .unwrap();
            }
        }
        return;
    }
    let size = partial_wire_bytes(wire, n, D);
    let mut own = Vec::new();
    for (rank, side) in sides.iter().enumerate() {
        let outbox = mesh.outbox(rank, &side.stream, slot).unwrap();
        // SAFETY: `p` holds `n` rows, the outbox their wire form; every
        // device's use of the slot's last parcels has ended (the loop below
        // ran for the merge before).
        unsafe {
            side.layers.partial_to_wire(&side.stream, wire, side.p.ptr(), outbox, n, D).unwrap();
            mesh.post(rank, &side.stream, slot, size).unwrap();
        }
        own.push(outbox);
    }
    for (rank, side) in sides.iter().enumerate() {
        // SAFETY: both devices have posted; the regions hold the wire form of
        // `n` rows.
        unsafe {
            let peers = mesh.arrive(rank, &side.stream, slot).unwrap();
            side.layers
                .merge_wire(
                    &side.stream,
                    wire,
                    side.x.ptr(),
                    own[rank],
                    peers,
                    mesh.slot_bytes(),
                    1,
                    rank as u32,
                    n,
                    D,
                    norm(side),
                    EPS,
                )
                .unwrap();
        }
        mesh.consumed(rank, &side.stream, slot).unwrap();
    }
}

#[test]
fn merges_give_every_device_the_same_sum_and_its_norm() {
    let Some(gpus) = gpus() else { return };
    let peers = peer_access(0, 1).unwrap().0 && peer_access(1, 0).unwrap().0;
    let mut routes = vec![Route::Host];
    if peers {
        routes.push(Route::Peer);
    } else {
        eprintln!("peer access is not available between these devices; host route only");
    }
    const MAX_ROWS: u32 = 40;
    let mut rng = Rng(7);
    let weight: Vec<f32> = (0..D).map(|_| 1.0 + 0.5 * rng.next()).collect();
    for route in routes {
        for wire in [Wire::F16, Wire::F32] {
            let mesh = Mesh::with_route(&gpus, 16 * D, route)
                .unwrap()
                .with_slots(2, partial_wire_bytes(wire, MAX_ROWS, D))
                .unwrap();
            let sides: Vec<Side> =
                gpus.iter().map(|device| Side::new(device, MAX_ROWS, &weight)).collect();
            let sync = || sides.iter().for_each(|s| s.stream.synchronize().unwrap());

            // Rounds of different sizes on alternating slots, so that slots,
            // tokens and flags are used again: a few rows are pushed where
            // the devices can, many rows take the slots.
            for (round, n) in [3u32, MAX_ROWS, 1, 8, MAX_ROWS, 5].into_iter().enumerate() {
                let push = mesh.push_bytes() > 0 && n <= 8;
                let slot = round as u32 % 2;
                let count = (n * D) as usize;
                let x0 = rng.rows(n, 4.0);
                let partials = [rng.rows(n, 1.0), rng.rows(n, 0.5)];
                let load = || {
                    for (side, partial) in sides.iter().zip(&partials) {
                        side.x.write(0, &f32_bytes(&x0)).unwrap();
                        side.p.write(0, &f32_bytes(partial)).unwrap();
                    }
                };

                // The sum and its norm as rows.
                load();
                merge(&mesh, &sides, wire, n, slot, push, |s| MergeNorm::Rows {
                    weight: s.weight.ptr(),
                    out: s.xn.ptr(),
                });
                sync();
                let got: Vec<Vec<f32>> = sides.iter().map(|s| read_f32(&s.x, count)).collect();
                let what = format!("{route:?} {wire:?} round {round} ({n} rows, push {push})");
                assert!(
                    got[0].iter().zip(&got[1]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: the devices hold different sums"
                );
                // The smallest step of f16 next to zero (2^-24), times the
                // power of two that brings a row into range.
                let floor = |partial: &[f32], row: usize| {
                    let at = row * D as usize;
                    let largest =
                        partial[at..at + D as usize].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    6.0e-8 * (largest / 16384.0).max(1.0)
                };
                for (i, value) in got[0].iter().enumerate() {
                    let (a, b) = (partials[0][i], partials[1][i]);
                    let exact = x0[i] + (a + b);
                    match wire {
                        Wire::F32 => {
                            assert_eq!(value.to_bits(), exact.to_bits(), "{what}: value {i}")
                        }
                        // Each partial is rounded to 11 bits, or to the
                        // smallest step next to zero; the factor of a row is
                        // a power of two and costs nothing.
                        Wire::F16 => {
                            let row = i / D as usize;
                            let bound = (a.abs() + b.abs()) / 2048.0
                                + floor(&partials[0], row)
                                + floor(&partials[1], row)
                                + exact.abs() * 1e-6;
                            assert!(
                                (value - exact).abs() <= bound,
                                "{what}: value {i} is {value}, the sum is {exact}"
                            );
                        }
                    }
                }
                let normed = read_f32(&sides[0].xn, count);
                for row in 0..n as usize {
                    let at = row * D as usize;
                    let xs = &got[0][at..at + D as usize];
                    let mean = xs.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / f64::from(D);
                    let scale = 1.0 / (mean + f64::from(EPS)).sqrt();
                    for (i, v) in xs.iter().enumerate() {
                        let want = f64::from(*v) * scale * f64::from(weight[i]);
                        let have = f64::from(normed[at + i]);
                        assert!(
                            (have - want).abs() <= want.abs() * 1e-4 + 1e-6,
                            "{what}: norm of row {row} value {i} is {have}, not {want}"
                        );
                    }
                }

                // The same merge with the norm as quantized activations: the
                // bytes that quantizing the normalised rows gives.
                load();
                merge(&mesh, &sides, wire, n, slot, push, |s| MergeNorm::Quantized {
                    weight: s.weight.ptr(),
                    act: s.act.ptr(),
                });
                for side in &sides {
                    // SAFETY: `xn` holds the `n` rows of the merge before and
                    // the buffer the quantized form of the most rows.
                    unsafe {
                        if n <= ares_cuda::MAX_BATCH {
                            side.kernels
                                .quantize_activations(
                                    &side.stream,
                                    side.xn.ptr(),
                                    side.act_ref.ptr(),
                                    D,
                                    n,
                                )
                                .unwrap();
                        }
                    }
                }
                sync();
                if n <= ares_cuda::MAX_BATCH {
                    let len = act::Layout::of(u64::from(n * D)).bytes as usize;
                    for side in &sides {
                        let (mut have, mut want) = (vec![0u8; len], vec![0u8; len]);
                        side.act.read(0, &mut have).unwrap();
                        side.act_ref.read(0, &mut want).unwrap();
                        let differing = have.iter().zip(&want).filter(|(a, b)| a != b).count();
                        assert_eq!(
                            differing, 0,
                            "{what}: quantized norm differs in {differing} bytes"
                        );
                    }
                }
            }
        }
    }
}

/// A merge is enqueued by one thread per device, and each device's kernel
/// waits for the other's flag. When one thread fails before it has enqueued
/// its half, the stream of the other holds a kernel that waits for nobody,
/// and so does everything behind it: the end of the step, the synchronise
/// that comes before the device's buffers are freed. The kernel gives up
/// after about four billion polls. `release_waiters` ends the wait from the
/// host without waiting for the stream.
#[test]
fn a_merge_that_waits_for_a_device_that_never_comes_is_released() {
    let Some(gpus) = gpus() else { return };
    if !(peer_access(0, 1).unwrap().0 && peer_access(1, 0).unwrap().0) {
        eprintln!("skipped: the push path needs peer access between the devices");
        return;
    }
    let mesh = Mesh::with_route(&gpus, 16 * D, Route::Peer).unwrap();
    assert!(mesh.push_bytes() > 0, "two devices with peer access have the push path");
    let weight = vec![1.0f32; D as usize];
    let side = Side::new(&gpus[0], 8, &weight);
    let enqueue = |rank: usize| {
        let target = mesh.push(rank).expect("a push path");
        // SAFETY: the buffers hold 3 rows of `D` values on the side's device.
        // The other device's half is never enqueued: that is the point.
        unsafe {
            side.layers
                .merge_push(
                    &side.stream,
                    Wire::F32,
                    side.p.ptr(),
                    side.x.ptr(),
                    target,
                    true,
                    3,
                    D,
                    MergeNorm::Rows { weight: side.weight.ptr(), out: side.xn.ptr() },
                    EPS,
                )
                .unwrap();
        }
    };
    enqueue(0);
    std::thread::sleep(std::time::Duration::from_millis(400));
    assert!(!side.stream.is_idle().unwrap(), "the kernel waits for a peer that is not there");

    let released = std::time::Instant::now();
    mesh.release_waiters().unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !side.stream.is_idle().unwrap() {
        assert!(std::time::Instant::now() < until, "the released kernel did not end in 10 s");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    eprintln!("RES the stream was idle {:?} after the release", released.elapsed());

    // Merges enqueued after the release do not wait either.
    enqueue(0);
    enqueue(0);
    side.stream.synchronize().unwrap();
}

/// The pushes the failed device had queued before it failed still run after
/// the release, and each writes the other device's flag with its own token,
/// which can be behind the kernels that wait for a later one. One release is
/// not always enough for the thread that failed; `release_waiters_until_idle`
/// repeats it until the stream of that thread is idle.
#[test]
fn a_release_is_repeated_while_the_pushes_the_failed_device_queued_still_write_flags() {
    let Some(gpus) = gpus() else { return };
    if !(peer_access(0, 1).unwrap().0 && peer_access(1, 0).unwrap().0) {
        eprintln!("skipped: the push path needs peer access between the devices");
        return;
    }
    let weight = vec![1.0f32; D as usize];
    let sides = [Side::new(&gpus[0], 8, &weight), Side::new(&gpus[1], 8, &weight)];
    let queue = |mesh: &Mesh, rank: usize, pushes: usize| {
        for _ in 0..pushes {
            let side = &sides[rank];
            let target = mesh.push(rank).expect("a push path");
            // SAFETY: the buffers hold 3 rows of `D` values on the side's
            // device; the other device's matching pushes are not queued (or
            // fewer): that is the point.
            unsafe {
                side.layers
                    .merge_push(
                        &side.stream,
                        Wire::F32,
                        side.p.ptr(),
                        side.x.ptr(),
                        target,
                        rank == 0,
                        3,
                        D,
                        MergeNorm::Rows { weight: side.weight.ptr(), out: side.xn.ptr() },
                        EPS,
                    )
                    .unwrap();
            }
        }
    };
    let wait_idle = |rank: usize, limit: std::time::Duration| {
        let until = std::time::Instant::now() + limit;
        while !sides[rank].stream.is_idle().unwrap() {
            if std::time::Instant::now() >= until {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        true
    };
    let patience = std::time::Duration::from_secs(10);

    // One device queued three pushes and failed. A single release lets them
    // run, and they write tokens 1 to 3 into the flags of the other device,
    // over the release of those flags if the release wrote them first (it
    // goes by rank, so that is what happens when the device that failed is
    // the last one): a push that the other device queues after that, a fourth,
    // waits for a token nobody writes. Repeated until the stream of the device
    // that failed is idle, the last release is answered by nothing, and the
    // flags stay ahead for good: no push of the other device waits, whichever
    // device failed.
    for failed in [0usize, 1] {
        let other = 1 - failed;
        // What a single release leaves is reported and not asserted.
        let mesh = Mesh::with_route(&gpus, 16 * D, Route::Peer).unwrap();
        queue(&mesh, failed, 3);
        std::thread::sleep(std::time::Duration::from_millis(200));
        mesh.release_waiters().unwrap();
        assert!(wait_idle(failed, patience), "device {failed} went on");
        queue(&mesh, other, 4);
        let waits = !wait_idle(other, std::time::Duration::from_millis(500));
        eprintln!(
            "RES one release, device {failed} failed: the fourth push of device {other} {}",
            if waits { "waits" } else { "goes on" }
        );
        // (Let it go, so that it does not spin until the process ends.)
        mesh.release_waiters_until_idle(&sides[other].stream, patience).unwrap();
        for round in 0..20 {
            let mesh = Mesh::with_route(&gpus, 16 * D, Route::Peer).unwrap();
            queue(&mesh, failed, 3);
            std::thread::sleep(std::time::Duration::from_millis(5 * (round % 4)));
            let idle = mesh.release_waiters_until_idle(&sides[failed].stream, patience).unwrap();
            assert!(idle, "device {failed}, round {round}: the stream did not end");
            queue(&mesh, other, 4);
            let ended = wait_idle(other, patience);
            assert!(ended, "device {failed}, round {round}: a push of device {other} waits");
        }
    }
}
