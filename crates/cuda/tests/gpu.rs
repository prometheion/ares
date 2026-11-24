//! Tests on real devices.
//!
//! Opening a device takes a few hundred MiB of its memory, which is not
//! acceptable next to a production model that fills the cards. So these tests
//! run only when `ARES_TEST_CUDA=1` is set and report themselves as skipped
//! otherwise:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test gpu
//! ```

use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, Kernels, PinnedBuffer, Stream, Wait, device_count,
    devices, driver_version, peer_access,
};
use common::{Rng, f32_bytes, gpus, read_f32, upload_f32};

mod common;

#[test]
fn describes_devices_without_opening_them() {
    if gpus().is_none() {
        return;
    }
    let (major, _) = driver_version().unwrap();
    assert!(major >= 13, "the engine needs a driver for CUDA 13 or newer, found {major}");
    let infos = devices().unwrap();
    assert_eq!(infos.len() as u32, device_count().unwrap());
    for info in &infos {
        assert!(!info.name.is_empty());
        assert!(info.total_memory > 1 << 30, "{info:?}");
        // A class this build runs: the test opened every device.
        info.check_class().unwrap();
        assert!(info.multiprocessors > 0, "{info:?}");
        assert_ne!(info.uuid, [0; 16]);
        // A device can always reach itself; the question is about peers.
        peer_access(info.ordinal, info.ordinal).unwrap();
    }
    let missing = devices().unwrap().len() as u32;
    assert!(Device::open(missing, Wait::Block).is_err());
}

#[test]
fn buffers_round_trip_and_check_their_bounds() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let info = device.mem_info().unwrap();
        assert!(info.free <= info.total);
        assert_eq!(info.total, device.info().total_memory);

        let buffer = DeviceBuffer::new(device, 1 << 20).unwrap();
        let data: Vec<u8> = (0..1usize << 20).map(|i| (i * 31 % 251) as u8).collect();
        buffer.write(0, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        buffer.read(0, &mut back).unwrap();
        assert_eq!(back, data);

        // A window in the middle.
        buffer.write(1000, &[1, 2, 3, 4]).unwrap();
        let mut window = [0u8; 6];
        buffer.read(999, &mut window).unwrap();
        assert_eq!(window, [data[999], 1, 2, 3, 4, data[1004]]);

        assert!(buffer.write((1 << 20) - 3, &[0; 4]).is_err());
        assert!(buffer.read(1 << 20, &mut [0; 1]).is_err());
        assert!(buffer.slice(u64::MAX, 2).is_err());
        assert!(DeviceBuffer::new(device, 0).is_err());
        assert!(PinnedBuffer::new(device, 0).is_err());
    }
}

#[test]
fn an_allocation_that_cannot_fit_is_reported_as_out_of_memory() {
    let Some(gpus) = gpus() else { return };
    let error = DeviceBuffer::new(&gpus[0], 1 << 40).unwrap_err();
    assert!(error.is_out_of_memory(), "{error}");
}

#[test]
fn a_stream_runs_its_work_in_order() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let stream = Stream::new(device, 0).unwrap();
        let len = 4usize << 20;
        let a = DeviceBuffer::new(device, len as u64).unwrap();
        let b = DeviceBuffer::new(device, len as u64).unwrap();
        let mut pinned_in = PinnedBuffer::new(device, len).unwrap();
        let mut pinned_out = PinnedBuffer::new(device, len).unwrap();
        for (i, byte) in pinned_in.iter_mut().enumerate() {
            *byte = (i % 253) as u8;
        }
        let done = Event::new(device).unwrap();

        // SAFETY: all ranges are whole buffers of `len` bytes on this device
        // that outlive the synchronisation below; the pinned buffers are not
        // touched until then.
        unsafe {
            stream.fill(a.ptr(), 7, len).unwrap();
            stream.copy_from_host(a.ptr(), pinned_in.as_ptr(), len).unwrap();
            stream.copy(b.ptr(), a.ptr(), len).unwrap();
            stream.fill(a.ptr(), 9, len).unwrap();
            stream.copy_to_host(pinned_out.as_mut_ptr(), b.ptr(), len).unwrap();
        }
        done.record(&stream).unwrap();
        done.synchronize().unwrap();
        assert!(done.is_done().unwrap());
        stream.synchronize().unwrap();
        assert!(stream.is_idle().unwrap());

        // The host data went through `a` into `b` before `a` was overwritten.
        assert_eq!(&pinned_out[..], &pinned_in[..]);
        let mut first = [0u8; 16];
        a.read(0, &mut first).unwrap();
        assert_eq!(first, [9; 16]);
    }
}

#[test]
fn add_and_rms_norm_match_the_cpu() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();

        // Sizes around the block and warp boundaries.
        for n in [1u32, 31, 32, 255, 256, 257, 2560, 10240] {
            let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            let y: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            let (dx, dy) = (upload_f32(device, &x), upload_f32(device, &y));
            // SAFETY: both buffers hold `n` values on this device.
            unsafe { kernels.add_f32(&stream, dy.ptr(), dx.ptr(), n).unwrap() };
            stream.synchronize().unwrap();
            let got = read_f32(&dy, n as usize);
            let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
            assert_eq!(got, want, "add, n = {n}");
        }

        for (n, rows) in [(32u32, 1u32), (160, 3), (2560, 4), (10240, 2), (1000, 5)] {
            let total = (n * rows) as usize;
            let x: Vec<f32> = (0..total).map(|_| rng.unit() * 3.0).collect();
            let weight: Vec<f32> = (0..n).map(|_| rng.unit() + 1.5).collect();
            let (dx, dw) = (upload_f32(device, &x), upload_f32(device, &weight));
            let dy = DeviceBuffer::new(device, total as u64 * 4).unwrap();
            let eps = 1e-6f32;
            for with_weight in [true, false] {
                let w = if with_weight { dw.ptr() } else { DevicePtr::NULL };
                // SAFETY: the buffers hold `rows * n` and `n` values on this device.
                unsafe {
                    kernels.rms_norm_f32(&stream, dx.ptr(), w, dy.ptr(), n, rows, eps).unwrap();
                }
                stream.synchronize().unwrap();
                let got = read_f32(&dy, total);
                for row in 0..rows as usize {
                    let xr = &x[row * n as usize..][..n as usize];
                    let mean: f64 =
                        xr.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / f64::from(n);
                    let scale = 1.0 / (mean + f64::from(eps)).sqrt();
                    for (i, value) in xr.iter().enumerate() {
                        let w = if with_weight { f64::from(weight[i]) } else { 1.0 };
                        let want = f64::from(*value) * scale * w;
                        let have = f64::from(got[row * n as usize + i]);
                        assert!(
                            (have - want).abs() <= 2e-5 * want.abs().max(1.0),
                            "rms_norm n = {n}, row {row}, value {i}: {have} != {want}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn data_moves_between_devices_in_stream_order() {
    let Some(gpus) = gpus() else { return };
    let [first, second, ..] = gpus else {
        eprintln!("skipped: needs two devices");
        return;
    };
    let len = 1usize << 20;
    let data: Vec<u8> = (0..len).map(|i| (i * 7 % 241) as u8).collect();

    let run = |label: &str| {
        for (src_device, dst_device) in [(first, second), (second, first)] {
            let src = DeviceBuffer::new(src_device, len as u64).unwrap();
            let dst = DeviceBuffer::new(dst_device, len as u64).unwrap();
            let src_stream = Stream::new(src_device, 0).unwrap();
            let dst_stream = Stream::new(dst_device, 0).unwrap();
            let filled = Event::new(src_device).unwrap();
            let mut pinned = PinnedBuffer::new(src_device, len).unwrap();
            pinned.copy_from_slice(&data);

            // The source is filled on its own stream; the destination's
            // stream waits for that before it pulls the data over, and
            // overwrites nothing until then.
            // SAFETY: whole buffers of `len` bytes on the named devices,
            // alive until both streams are synchronised below.
            unsafe {
                dst_stream.fill(dst.ptr(), 0xee, len).unwrap();
                src_stream.copy_from_host(src.ptr(), pinned.as_ptr(), len).unwrap();
                filled.record(&src_stream).unwrap();
                dst_stream.wait_event(&filled).unwrap();
                dst_stream.copy_peer(dst_device, dst.ptr(), src_device, src.ptr(), len).unwrap();
            }
            dst_stream.synchronize().unwrap();
            src_stream.synchronize().unwrap();
            let mut back = vec![0u8; len];
            dst.read(0, &mut back).unwrap();
            assert!(back == data, "{label}: {src_device:?} -> {dst_device:?}");
        }
    };

    run("without peer access");
    let (forward, _) = peer_access(first.ordinal(), second.ordinal()).unwrap();
    let (backward, _) = peer_access(second.ordinal(), first.ordinal()).unwrap();
    if forward && backward {
        first.enable_peer_access(second).unwrap();
        second.enable_peer_access(first).unwrap();
        // Enabling twice is not an error.
        first.enable_peer_access(second).unwrap();
        run("with peer access");
    } else {
        eprintln!("peer access is not available between these devices; host route only");
    }
}

/// `DeviceBuffer::write` copies from ordinary memory, for which the driver
/// returns as soon as the data is in its staging buffer. A kernel launched
/// right after, on a stream, must still see the new data in full; before
/// `write` waited for the device it read stale values in the tail of large
/// uploads.
#[test]
fn a_kernel_launched_right_after_a_write_sees_all_of_it() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let n = 8u32 << 20;
        let bytes = n as usize * 4;
        let x = DeviceBuffer::new(device, bytes as u64).unwrap();
        let y = DeviceBuffer::new(device, bytes as u64).unwrap();
        let ones = f32_bytes(&vec![1.0f32; n as usize]);
        for round in 0..4 {
            // SAFETY: whole buffers of `bytes` bytes on this device.
            unsafe {
                stream.fill(x.ptr(), 0, bytes).unwrap();
                stream.fill(y.ptr(), 0, bytes).unwrap();
            }
            stream.synchronize().unwrap();
            x.write(0, &ones).unwrap();
            // SAFETY: both buffers hold `n` values on this device.
            unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), n).unwrap() };
            stream.synchronize().unwrap();
            let got = read_f32(&y, n as usize);
            let stale = got.iter().filter(|v| **v != 1.0).count();
            assert_eq!(
                stale, 0,
                "round {round}: {stale} of {n} values were read before the upload"
            );
        }
    }
}

/// A chain of sums across the devices, enqueued in one go: each round every
/// device scales nothing, adds the sum of all partials into its own, and the
/// next round builds on that. After `ROUNDS` rounds each device must hold
/// `initial sum * devices^(ROUNDS-1)` in every value, which only comes out
/// right if every exchange saw exactly the values of its own round (no device
/// read a partial too early or too late, on either route).
#[test]
fn sums_across_devices_stay_in_order_without_host_waits() {
    let Some(gpus) = gpus() else { return };
    if gpus.len() < 2 {
        eprintln!("skipped: needs two devices");
        return;
    }
    const ROUNDS: u32 = 12;
    // A size the push kernel takes with peer access, and one that goes to the
    // copy engine.
    for n in [40_960u32, 200_000] {
        sums_chain(gpus, n, ROUNDS);
    }
}

fn sums_chain(gpus: &[Device], n: u32, rounds: u32) {
    use ares_cuda::{Mesh, Route};
    #[allow(non_snake_case)]
    let (N, ROUNDS) = (n, rounds);
    let peers = gpus.iter().all(|a| {
        gpus.iter()
            .all(|b| a.ordinal() == b.ordinal() || peer_access(a.ordinal(), b.ordinal()).unwrap().0)
    });
    let mut routes = vec![Route::Host];
    if peers {
        routes.push(Route::Peer);
    } else {
        eprintln!("peer access is not available between these devices; host route only");
        assert!(Mesh::with_route(gpus, N, Route::Peer).is_err());
    }
    assert_eq!(Mesh::new(gpus, N).unwrap().route(), if peers { Route::Peer } else { Route::Host });

    for route in routes {
        let mesh = Mesh::with_route(gpus, N, route).unwrap();
        let streams: Vec<Stream> = gpus.iter().map(|d| Stream::new(d, 0).unwrap()).collect();
        let stream_refs: Vec<&Stream> = streams.iter().collect();
        // Device i starts with i + 1 everywhere.
        let partials: Vec<DeviceBuffer> = gpus
            .iter()
            .enumerate()
            .map(|(i, d)| upload_f32(d, &vec![(i + 1) as f32; N as usize]))
            .collect();
        let pointers: Vec<DevicePtr> = partials.iter().map(|p| p.ptr()).collect();

        for _ in 0..ROUNDS {
            // SAFETY: each buffer holds N values on its device and outlives
            // the synchronisation below; the streams belong to the devices.
            unsafe { mesh.sum_f32(&stream_refs, &pointers, N).unwrap() };
        }
        for stream in &streams {
            stream.synchronize().unwrap();
        }

        let count = gpus.len() as f32;
        let initial: f32 = (1..=gpus.len()).map(|i| i as f32).sum();
        let want = initial * count.powi(ROUNDS as i32 - 1);
        for (device, partial) in gpus.iter().zip(&partials) {
            let got = read_f32(partial, N as usize);
            let wrong = got.iter().filter(|v| **v != want).count();
            assert_eq!(
                wrong, 0,
                "{route:?}, {device:?}: {wrong} values differ from {want} (first {})",
                got[0]
            );
        }

        // The same chain in the form that writes the sum elsewhere, going
        // back and forth between two sets of buffers.
        let others: Vec<DeviceBuffer> =
            gpus.iter().map(|d| DeviceBuffer::new(d, u64::from(N) * 4).unwrap()).collect();
        let other_pointers: Vec<DevicePtr> = others.iter().map(|p| p.ptr()).collect();
        for (i, (device, partial)) in gpus.iter().zip(&partials).enumerate() {
            partial.write(0, &f32_bytes(&vec![(i + 1) as f32; N as usize])).unwrap();
            let _ = device;
        }
        for round in 0..ROUNDS {
            let (from, to) = if round % 2 == 0 {
                (&pointers, &other_pointers)
            } else {
                (&other_pointers, &pointers)
            };
            // SAFETY: as above; the two sets of buffers do not overlap.
            unsafe { mesh.sum_f32_to(&stream_refs, from, to, N).unwrap() };
        }
        for stream in &streams {
            stream.synchronize().unwrap();
        }
        // ROUNDS is even, so the last sums were written to the first set.
        for (device, partial) in gpus.iter().zip(&partials) {
            let got = read_f32(partial, N as usize);
            let wrong = got.iter().filter(|v| **v != want).count();
            assert_eq!(wrong, 0, "{route:?} to, {device:?}: {wrong} values differ from {want}");
        }

        // Sizes the mesh was not built for are refused.
        // SAFETY: refused before anything is enqueued.
        unsafe {
            assert!(mesh.sum_f32(&stream_refs, &pointers, N + 1).is_err());
            assert!(mesh.sum_f32(&stream_refs[..1], &pointers, N).is_err());
            assert!(mesh.sum_f32_to(&stream_refs, &pointers, &other_pointers[..1], N).is_err());
        }
    }
}

/// A pipeline across the devices, enqueued in one go: in each step the
/// sender increments its values and sends them, and the receiver adds what it
/// received to a running total. The total only comes out right if every
/// parcel was read exactly once, complete, and before its slot was reused.
/// The ring has fewer slots than there are steps, so slots are reused many
/// times, and the sender runs ahead of the slower receiver.
#[test]
fn a_lane_delivers_every_parcel_once_while_the_sender_runs_ahead() {
    use ares_cuda::{Lane, Mesh, Route};
    let Some(gpus) = gpus() else { return };
    if gpus.len() < 2 {
        eprintln!("skipped: needs two devices");
        return;
    }
    const N: u32 = 81_920;
    const STEPS: u32 = 64;
    const SLOTS: u32 = 3;
    let peers = peer_access(0, 1).unwrap().0 && peer_access(1, 0).unwrap().0;
    let routes = if peers { vec![Route::Host, Route::Peer] } else { vec![Route::Host] };
    for route in routes {
        let mesh = Mesh::with_route(&gpus[..2], N, route).unwrap();
        for (from, to) in [(0usize, 1usize), (1, 0)] {
            let lane = Lane::new(&mesh, from, to, SLOTS, N).unwrap();
            let (sender_dev, receiver_dev) = (&gpus[from], &gpus[to]);
            let (send_k, recv_k) =
                (Kernels::load(sender_dev).unwrap(), Kernels::load(receiver_dev).unwrap());
            let (sender, receiver) =
                (Stream::new(sender_dev, 0).unwrap(), Stream::new(receiver_dev, 0).unwrap());
            let values = upload_f32(sender_dev, &vec![1.0f32; N as usize]);
            let ones = upload_f32(sender_dev, &vec![1.0f32; N as usize]);
            let total = upload_f32(receiver_dev, &vec![0.0f32; N as usize]);

            for _ in 0..STEPS {
                // SAFETY: all buffers hold N values on the device of the
                // stream they are used with and outlive the synchronisation.
                unsafe {
                    send_k.add_f32(&sender, values.ptr(), ones.ptr(), N).unwrap();
                    let parcel = lane.send_f32(&sender, values.ptr(), N).unwrap();
                    let received = lane.receive(&receiver, &parcel).unwrap();
                    recv_k.add_f32(&receiver, total.ptr(), received, N).unwrap();
                    lane.release(&receiver, parcel).unwrap();
                }
            }
            sender.synchronize().unwrap();
            receiver.synchronize().unwrap();

            // Step k sends k + 1 (starting at 2): the total is the sum of 2..=STEPS+1.
            let want = (2..=STEPS + 1).sum::<u32>() as f32;
            let got = read_f32(&total, N as usize);
            let wrong = got.iter().filter(|v| **v != want).count();
            assert_eq!(
                wrong, 0,
                "{route:?} {from}->{to}: {wrong} values differ from {want} (first {})",
                got[0]
            );

            // More parcels in flight than slots are refused before anything
            // is enqueued, and parcels are released in order.
            // SAFETY: as above.
            unsafe {
                let parcels: Vec<_> =
                    (0..SLOTS).map(|_| lane.send_f32(&sender, values.ptr(), N).unwrap()).collect();
                assert!(lane.send_f32(&sender, values.ptr(), N).is_err());
                let mut parcels = parcels.into_iter();
                let (first, second) = (parcels.next().unwrap(), parcels.next().unwrap());
                assert!(lane.release(&receiver, second).is_err());
                lane.receive(&receiver, &first).unwrap();
                lane.release(&receiver, first).unwrap();
                assert!(lane.send_f32(&sender, values.ptr(), N + 1).is_err());
            }
            sender.synchronize().unwrap();
            receiver.synchronize().unwrap();
        }
        assert!(Lane::new(&mesh, 0, 0, SLOTS, N).is_err());
        assert!(Lane::new(&mesh, 0, 1, 0, N).is_err());
    }
}
