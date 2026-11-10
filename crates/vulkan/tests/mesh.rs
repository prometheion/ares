//! Tests of several devices working together: sums and lanes over every route
//! that works on this machine.
//!
//! They open every GPU, so they run only when `ARES_TEST_VULKAN=1` is set (and
//! need at least two devices):
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test mesh
//! ```
//!
//! Which routes exist depends on the driver; the tests run on all that the
//! mesh builds and verifies, and say which in their output (`--nocapture`).
//! The tests take a lock because collectives of different meshes over the same
//! devices must be enqueued in one order on all of them: two test threads
//! enqueuing sums on the same GPUs at once could leave each GPU spinning in
//! the kernel of a different sum.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ares_vulkan::{
    Device, DeviceBuffer, DevicePtr, HostPages, Kernels, Lane, Mesh, Route, SharedTimeline, Stream,
    Wait, device_count, device_groups,
};

/// Two devices, opened to spin while they wait, shared by all tests; `None`
/// when the tests are not enabled or the machine has fewer than two.
fn pair() -> Option<(&'static [Device], MutexGuard<'static, ()>)> {
    static DEVICES: OnceLock<Option<Vec<Device>>> = OnceLock::new();
    static LOCK: Mutex<()> = Mutex::new(());
    let devices = DEVICES
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_VULKAN").is_none() {
                eprintln!("skipped: set ARES_TEST_VULKAN=1 to run the tests that open GPUs");
                return None;
            }
            if device_count().expect("the Vulkan loader") < 2 {
                eprintln!("skipped: needs two devices");
                return None;
            }
            Some((0..2).map(|i| Device::open_with(i, Wait::Spin).expect("open device")).collect())
        })
        .as_deref()?;
    Some((devices, LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)))
}

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buffer.read(0, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn upload_f32(device: &Device, values: &[f32]) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, values.len() as u64 * 4).unwrap();
    buffer.write(0, &f32_bytes(values)).unwrap();
    buffer
}

/// The sets of devices the exchanges are tested on: both GPUs, and the first
/// GPU twice (two logical devices of one GPU). The second set is what a direct
/// route can be tested on where the GPUs of the machine cannot share memory:
/// devices of one GPU can, and everything built on that (the memory exported by
/// one and imported by the other, kernels that write into it and poll it) is
/// the same code.
fn device_sets(gpus: &[Device]) -> Vec<(&'static str, Vec<Device>)> {
    vec![
        ("both GPUs", gpus.to_vec()),
        ("the first GPU twice", vec![gpus[0].clone(), Device::open_with(0, Wait::Spin).unwrap()]),
    ]
}

/// Every route that the mesh builds and verifies on these devices. The host route
/// and the staged one must be among them: both need nothing that a desktop driver
/// of the last years lacks, so a machine where they fail is a machine on which the
/// mesh is broken, and the tests say so instead of passing on the routes that are
/// left.
fn routes_that_work(devices: &[Device]) -> Vec<Route> {
    let mut working = Vec::new();
    for route in [Route::Peer, Route::Host, Route::Staged] {
        match Mesh::with_route(devices, 4096, route) {
            Ok(_) => working.push(route),
            Err(error) => {
                eprintln!("  {route:?} does not work on these devices: {error}");
                assert_eq!(route, Route::Peer, "the {route:?} route must work: {error}");
            }
        }
    }
    working
}

#[test]
fn a_mesh_takes_the_best_route_that_works_and_says_why_it_did_not_take_the_others() {
    let Some((gpus, _guard)) = pair() else { return };
    let mesh = Mesh::new(gpus, 4096).unwrap();
    eprintln!("mesh: {mesh:?}, peer kind {:?}, notes {:?}", mesh.peer_kind(), mesh.notes());
    for (route, why) in mesh.rejected() {
        eprintln!("  not {route:?}: {why}");
    }
    let working = routes_that_work(gpus);
    eprintln!("routes that work: {working:?}");
    // The staged route needs nothing but host memory, and the mesh takes the
    // first route in the order Peer, Host, Staged that works.
    assert!(working.contains(&Route::Staged));
    assert_eq!(mesh.route(), working[0]);
    // What it did not take it rejected with a reason, and a rejected route is
    // refused when asked for, as an unsupported device and not as a mistake.
    for route in [Route::Peer, Route::Host] {
        let refused = Mesh::with_route(gpus, 4096, route);
        let listed = mesh.rejected().iter().any(|(r, why)| *r == route && !why.is_empty());
        assert_eq!(
            refused.is_err(),
            listed || (route != mesh.route() && !working.contains(&route))
        );
        if let Err(error) = refused {
            assert!(
                matches!(error, ares_vulkan::Error::Unsupported { .. }),
                "{route:?} was refused with {error}"
            );
        }
    }
    // Wrong requests are mistakes.
    assert!(Mesh::new(&[], 16).is_err());
    assert!(Mesh::new(gpus, 0).is_err());
    assert!(Mesh::new(&[gpus[0].clone(), gpus[0].clone()], 16).is_err());
    for group in device_groups().unwrap() {
        assert!(group.len() > 1);
    }
}

/// The sum of `partials` on the CPU, in device order: what every device must
/// hold, bit for bit.
fn expected_sum(partials: &[Vec<f32>]) -> Vec<f32> {
    (0..partials[0].len())
        .map(|i| partials[1..].iter().fold(partials[0][i], |sum, p| sum + p[i]))
        .collect()
}

/// Sums of random values, in place and into other buffers, for sizes from one
/// value to several MiB, on one route; every device must hold the bitwise sum
/// in device order.
fn exact_sums(devices: &[Device], route: Route, sizes: &[u32]) {
    let max = *sizes.iter().max().unwrap();
    let mesh = Mesh::with_route(devices, max, route).unwrap();
    let streams: Vec<Stream> = devices.iter().map(|d| Stream::new(d).unwrap()).collect();
    let stream_refs: Vec<&Stream> = streams.iter().collect();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for &n in sizes {
        let partials: Vec<Vec<f32>> =
            (0..devices.len()).map(|_| rng.values(n as usize, 3.0)).collect();
        let want = expected_sum(&partials);

        // In place.
        let buffers: Vec<DeviceBuffer> =
            devices.iter().zip(&partials).map(|(d, p)| upload_f32(d, p)).collect();
        let pointers: Vec<DevicePtr> = buffers.iter().map(DeviceBuffer::ptr).collect();
        // SAFETY: whole buffers of `n` values on their devices, alive past the
        // synchronisation below.
        unsafe { mesh.sum_f32(&stream_refs, &pointers, n).unwrap() };
        // The sum into other buffers, the partials left alone.
        let outs: Vec<DeviceBuffer> =
            devices.iter().map(|d| DeviceBuffer::new(d, u64::from(n) * 4).unwrap()).collect();
        let out_pointers: Vec<DevicePtr> = outs.iter().map(DeviceBuffer::ptr).collect();
        let fresh: Vec<DeviceBuffer> =
            devices.iter().zip(&partials).map(|(d, p)| upload_f32(d, p)).collect();
        let fresh_pointers: Vec<DevicePtr> = fresh.iter().map(DeviceBuffer::ptr).collect();
        // SAFETY: as above; the output buffers do not overlap the partials.
        unsafe { mesh.sum_f32_to(&stream_refs, &fresh_pointers, &out_pointers, n).unwrap() };
        for stream in &streams {
            stream.synchronize().unwrap();
        }
        for (d, buffer) in buffers.iter().enumerate() {
            let got = read_f32(buffer, n as usize);
            let wrong = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
            assert_eq!(wrong, 0, "{route:?}, {n} values, in place on device {d}: {wrong} differ");
        }
        for (d, (out, partial)) in outs.iter().zip(&fresh).enumerate() {
            let got = read_f32(out, n as usize);
            let wrong = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
            assert_eq!(
                wrong, 0,
                "{route:?}, {n} values, into another buffer on device {d}: {wrong} differ"
            );
            // The partial was left as it was.
            let kept = read_f32(partial, n as usize);
            assert_eq!(kept, partials[d], "{route:?}: the partial of device {d} changed");
        }
    }
}

#[test]
fn every_device_gets_the_bitwise_same_sum_from_one_value_to_several_mib() {
    let Some((gpus, _guard)) = pair() else { return };
    // A single value; sizes around the push kernels' limit; and several MiB.
    let sizes = [1, 7, 255, 2560, 65_537, 131_072, 131_073, 400_000, 2_097_152 + 5];
    for (label, devices) in device_sets(gpus) {
        for route in routes_that_work(&devices) {
            eprintln!("exact sums over {route:?} on {label}");
            exact_sums(&devices, route, &sizes);
        }
    }
}

#[test]
fn three_devices_get_the_same_bits_as_two() {
    let Some((gpus, _guard)) = pair() else { return };
    // A third logical device on the first GPU: the copy path takes any number
    // of devices, and the order in which partials are added must not depend
    // on which device does the adding.
    let mut devices = gpus.to_vec();
    devices.push(Device::open_with(0, Wait::Spin).unwrap());
    for route in [Route::Host, Route::Staged] {
        if Mesh::with_route(&devices, 1024, route).is_err() {
            eprintln!("{route:?} does not work for three devices here");
            assert_eq!(route, Route::Host, "the staged route works with any devices");
            continue;
        }
        eprintln!("three devices over {route:?}");
        exact_sums(&devices, route, &[1, 1000, 70_000, 300_000]);
    }
}

/// A chain of sums across the devices, enqueued in one go: each round adds the
/// sum of all partials into every device's, and the next round builds on that.
/// After `rounds` rounds each device must hold the initial sum times
/// `devices ^ (rounds - 1)` in every value, which only comes out right if every
/// exchange saw exactly the values of its own round (no device read a partial
/// too early or too late).
fn sums_chain(devices: &[Device], route: Route, n: u32, rounds: u32) {
    let mesh = Mesh::with_route(devices, n, route).unwrap();
    let streams: Vec<Stream> = devices.iter().map(|d| Stream::new(d).unwrap()).collect();
    let stream_refs: Vec<&Stream> = streams.iter().collect();
    // Device i starts with i + 1 everywhere.
    let partials: Vec<DeviceBuffer> = devices
        .iter()
        .enumerate()
        .map(|(i, d)| upload_f32(d, &vec![(i + 1) as f32; n as usize]))
        .collect();
    let pointers: Vec<DevicePtr> = partials.iter().map(DeviceBuffer::ptr).collect();
    // Busy work for one device at a time before its part of a sum, so that the
    // other device is ahead of it by about a millisecond and has to wait: a
    // device that did not wait for its partner's partial would add a stale one.
    let lag: Vec<(DeviceBuffer, DeviceBuffer)> = devices
        .iter()
        .map(|d| (DeviceBuffer::new(d, 16 << 20).unwrap(), DeviceBuffer::new(d, 16 << 20).unwrap()))
        .collect();
    for round in 0..rounds {
        let slow = round as usize % devices.len();
        for _ in 0..8 {
            // SAFETY: two separate buffers of 16 MiB on the slow device.
            unsafe { streams[slow].copy(lag[slow].1.ptr(), lag[slow].0.ptr(), 16 << 20).unwrap() };
        }
        // SAFETY: each buffer holds `n` values on its device and outlives the
        // synchronisation below; the streams belong to the devices.
        unsafe { mesh.sum_f32(&stream_refs, &pointers, n).unwrap() };
    }
    // Waiting for the first stream first must not hang because the second
    // has not been submitted.
    for stream in &streams {
        stream.synchronize().unwrap();
    }
    let count = devices.len() as f32;
    let initial: f32 = (1..=devices.len()).map(|i| i as f32).sum();
    let want = initial * count.powi(rounds as i32 - 1);
    for (d, partial) in partials.iter().enumerate() {
        let got = read_f32(partial, n as usize);
        let wrong = got.iter().filter(|v| **v != want).count();
        assert_eq!(
            wrong, 0,
            "{route:?}, device {d}: {wrong} values differ from {want} (first {})",
            got[0]
        );
    }

    // The same chain in the form that writes the sum elsewhere, going back and
    // forth between two sets of buffers.
    let others: Vec<DeviceBuffer> =
        devices.iter().map(|d| DeviceBuffer::new(d, u64::from(n) * 4).unwrap()).collect();
    let other_pointers: Vec<DevicePtr> = others.iter().map(DeviceBuffer::ptr).collect();
    for (i, partial) in partials.iter().enumerate() {
        partial.write(0, &f32_bytes(&vec![(i + 1) as f32; n as usize])).unwrap();
    }
    for round in 0..rounds {
        let (from, to) = if round % 2 == 0 {
            (&pointers, &other_pointers)
        } else {
            (&other_pointers, &pointers)
        };
        // SAFETY: as above; the two sets of buffers do not overlap.
        unsafe { mesh.sum_f32_to(&stream_refs, from, to, n).unwrap() };
    }
    for stream in &streams {
        stream.synchronize().unwrap();
    }
    // `rounds` is even, so the last sums were written to the first set.
    for (d, partial) in partials.iter().enumerate() {
        let got = read_f32(partial, n as usize);
        let wrong = got.iter().filter(|v| **v != want).count();
        assert_eq!(wrong, 0, "{route:?} to, device {d}: {wrong} values differ from {want}");
    }

    // Sizes the mesh was not built for, and mismatched arguments, are refused.
    // SAFETY: refused before anything is enqueued.
    unsafe {
        assert!(mesh.sum_f32(&stream_refs, &pointers, n + 1).is_err());
        assert!(mesh.sum_f32(&stream_refs[..1], &pointers, n).is_err());
        assert!(mesh.sum_f32_to(&stream_refs, &pointers, &other_pointers[..1], n).is_err());
        // Streams in the wrong order are on the wrong devices.
        assert!(mesh.sum_f32(&[stream_refs[1], stream_refs[0]], &pointers, n).is_err());
    }
}

#[test]
fn sums_across_devices_stay_in_order_without_host_waits() {
    let Some((gpus, _guard)) = pair() else { return };
    // A size the push kernels take where the route has them, and one that
    // goes to the copy engine.
    for (label, devices) in device_sets(gpus) {
        for route in routes_that_work(&devices) {
            for n in [40_960u32, 200_000] {
                eprintln!("chain of sums over {route:?} on {label}, {n} values");
                sums_chain(&devices, route, n, 12);
            }
        }
    }
}

#[test]
fn a_chain_that_mixes_push_and_copy_sums_stays_in_order() {
    let Some((gpus, _guard)) = pair() else { return };
    for (label, devices) in device_sets(gpus) {
        for route in routes_that_work(&devices) {
            eprintln!("mixed sizes over {route:?} on {label}");
            let mesh = Mesh::with_route(&devices, 300_000, route).unwrap();
            let streams: Vec<Stream> = devices.iter().map(|d| Stream::new(d).unwrap()).collect();
            let stream_refs: Vec<&Stream> = streams.iter().collect();
            let n = 300_000u32;
            let partials: Vec<DeviceBuffer> = devices
                .iter()
                .enumerate()
                .map(|(i, d)| upload_f32(d, &vec![(i + 1) as f32; n as usize]))
                .collect();
            let pointers: Vec<DevicePtr> = partials.iter().map(DeviceBuffer::ptr).collect();
            let sizes = [100u32, 300_000, 5000, 140_000, 1, 131_072, 131_073, 64];
            for round in 0..24 {
                // SAFETY: as in `sums_chain`.
                unsafe {
                    mesh.sum_f32(&stream_refs, &pointers, sizes[round % sizes.len()]).unwrap()
                };
            }
            for stream in &streams {
                stream.synchronize().unwrap();
            }
            // Only the first `size` values of a round take part; the others keep
            // their values. The reference is the same rounds on the CPU.
            let mut cpu = [vec![1.0f32; n as usize], vec![2.0f32; n as usize]];
            for round in 0..24 {
                let size = sizes[round % sizes.len()] as usize;
                let [first, second] = &mut cpu;
                for (a, b) in first[..size].iter_mut().zip(&mut second[..size]) {
                    *a += *b;
                    *b = *a;
                }
            }
            for (d, partial) in partials.iter().enumerate() {
                let got = read_f32(partial, n as usize);
                assert_eq!(
                    got, cpu[d],
                    "{route:?} on {label}: device {d} differs after mixed sums"
                );
            }
        }
    }
}

/// A pipeline across the devices, enqueued in one go: in each step the sender
/// increments its values and sends them, and the receiver adds what it
/// received to a running total. The total only comes out right if every parcel
/// was read exactly once, complete, and before its slot was reused. The ring
/// has fewer slots than there are steps, so slots are reused many times, and
/// the sender runs ahead of the receiver; with `slow` the receiver is held up
/// by a long run of copies at its start, so the sender gets a whole ring ahead
/// and must be held back on the device.
fn lane_pipeline(devices: &[Device], route: Route, slow: bool) {
    const N: u32 = 81_920;
    const STEPS: u32 = 64;
    const SLOTS: u32 = 3;
    let mesh = Mesh::with_route(devices, N, route).unwrap();
    for (from, to) in [(0usize, 1usize), (1, 0)] {
        let lane = Lane::new(&mesh, from, to, SLOTS, N).unwrap();
        let (sender_dev, receiver_dev) = (&devices[from], &devices[to]);
        let (send_k, recv_k) =
            (Kernels::load(sender_dev).unwrap(), Kernels::load(receiver_dev).unwrap());
        let (sender, receiver) =
            (Stream::new(sender_dev).unwrap(), Stream::new(receiver_dev).unwrap());
        let values = upload_f32(sender_dev, &vec![1.0f32; N as usize]);
        let ones = upload_f32(sender_dev, &vec![1.0f32; N as usize]);
        let total = upload_f32(receiver_dev, &vec![0.0f32; N as usize]);
        let (big_a, big_b) = (
            DeviceBuffer::new(receiver_dev, 32 << 20).unwrap(),
            DeviceBuffer::new(receiver_dev, 32 << 20).unwrap(),
        );
        if slow {
            for _ in 0..8 {
                // SAFETY: two separate buffers of 32 MiB on the receiver.
                unsafe { receiver.copy(big_b.ptr(), big_a.ptr(), 32 << 20).unwrap() };
            }
        }
        for _ in 0..STEPS {
            // SAFETY: all buffers hold N values on the device of the stream they are
            // used with and outlive the synchronisation.
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

        // More parcels in flight than slots are refused before anything is
        // enqueued, and parcels are released in order.
        // SAFETY: as above.
        unsafe {
            let parcels: Vec<_> =
                (0..SLOTS).map(|_| lane.send_f32(&sender, values.ptr(), N).unwrap()).collect();
            assert!(lane.send_f32(&sender, values.ptr(), N).is_err());
            let mut parcels = parcels.into_iter();
            let (first, second, third) =
                (parcels.next().unwrap(), parcels.next().unwrap(), parcels.next().unwrap());
            assert!(lane.release(&receiver, second).is_err());
            lane.receive(&receiver, &first).unwrap();
            lane.release(&receiver, first).unwrap();
            assert!(lane.send_f32(&sender, values.ptr(), N + 1).is_err());
            let _ = third;
        }
        sender.synchronize().unwrap();
        receiver.synchronize().unwrap();
    }
    assert!(Lane::new(&mesh, 0, 0, SLOTS, N).is_err());
    assert!(Lane::new(&mesh, 0, 1, 0, N).is_err());
    assert!(Lane::new(&mesh, 0, 2, SLOTS, N).is_err());
}

#[test]
fn a_lane_delivers_every_parcel_once_while_the_sender_runs_ahead() {
    let Some((gpus, _guard)) = pair() else { return };
    for (label, devices) in device_sets(gpus) {
        for route in routes_that_work(&devices) {
            eprintln!("lane over {route:?} on {label}");
            lane_pipeline(&devices, route, false);
        }
    }
}

#[test]
fn a_sender_a_whole_ring_ahead_is_held_back_on_the_device() {
    let Some((gpus, _guard)) = pair() else { return };
    for (label, devices) in device_sets(gpus) {
        for route in routes_that_work(&devices) {
            // The Staged lane's receiving side waits for the sender on the host,
            // so the receiver cannot be held up behind its own queue without
            // blocking the test thread; it is covered by the plain test.
            if route == Route::Staged {
                continue;
            }
            eprintln!("slow receiver over {route:?} on {label}");
            lane_pipeline(&devices, route, true);
        }
    }
}

#[test]
fn a_shared_timeline_orders_work_between_devices() {
    let Some((gpus, _guard)) = pair() else { return };
    let timeline = SharedTimeline::new(gpus).unwrap();
    let streams: Vec<Stream> = gpus.iter().map(|d| Stream::new(d).unwrap()).collect();
    streams[0].link(&streams[1]);
    const N: usize = 1 << 20;
    let kernels: Vec<Kernels> = gpus.iter().map(|d| Kernels::load(d).unwrap()).collect();
    // Both devices import the same host pages. One device accumulates for a
    // while, copies its result into the pages and sets the counter; the other
    // waits for the counter and copies the pages. It sees the result only if
    // its copy ran after the counter was set, on the devices and not by luck
    // of the host's timing.
    let refs: Vec<&Device> = gpus.iter().collect();
    let pages = std::sync::Arc::new(HostPages::new(&refs, N * 4).unwrap());
    let views: Vec<DeviceBuffer> =
        gpus.iter().map(|d| DeviceBuffer::import_host(d, &pages).unwrap()).collect();
    let local: Vec<DeviceBuffer> = gpus.iter().map(|d| upload_f32(d, &vec![1.0f32; N])).collect();
    let scratch: Vec<DeviceBuffer> = gpus.iter().map(|d| upload_f32(d, &vec![0.0f32; N])).collect();
    for (w, r) in [(0usize, 1usize), (1, 0)] {
        let base = timeline.value(&gpus[0]).unwrap();
        // A long chain of work before the signal (in-place adds), then the data.
        for _ in 0..100 {
            // SAFETY: separate buffers of N values on the device.
            unsafe {
                kernels[w].add_f32(&streams[w], scratch[w].ptr(), local[w].ptr(), N as u32).unwrap()
            };
        }
        // SAFETY: whole buffers; the pages are 4 MiB.
        unsafe { streams[w].copy(views[w].ptr(), scratch[w].ptr(), N * 4).unwrap() };
        timeline.signal(&streams[w], base + 1).unwrap();
        timeline.wait(&streams[r], base + 1).unwrap();
        // SAFETY: as above.
        unsafe { streams[r].copy(local[r].ptr(), views[r].ptr(), N * 4).unwrap() };
        streams[r].synchronize().unwrap();
        streams[w].synchronize().unwrap();
        // The reader holds what the writer had accumulated (100 times 1), not
        // what the pages held before.
        let got = read_f32(&local[r], N);
        assert!(got.iter().all(|v| *v == 100.0), "{w}->{r}: first {} of {}", got[0], got.len());
        assert_eq!(timeline.value(&gpus[0]).unwrap(), base + 1);
        assert_eq!(timeline.value(&gpus[1]).unwrap(), base + 1);
        // Restore for the next round.
        for d in 0..2 {
            local[d].write(0, &f32_bytes(&vec![1.0f32; N])).unwrap();
            scratch[d].write(0, &f32_bytes(&vec![0.0f32; N])).unwrap();
        }
    }
    // A stream that is not of a device of the timeline is refused.
    let third = Device::open(0).unwrap();
    let stream = Stream::new(&third).unwrap();
    assert!(timeline.signal(&stream, 99).is_err());
    assert!(timeline.wait(&stream, 99).is_err());
}

#[test]
fn devices_of_one_gpu_share_memory_directly() {
    let Some((gpus, _guard)) = pair() else { return };
    let twice = vec![gpus[0].clone(), Device::open_with(0, Wait::Spin).unwrap()];
    let mesh = Mesh::new(&twice, 1 << 17).unwrap();
    eprintln!("{mesh:?}: rejected {:?}, notes {:?}", mesh.rejected(), mesh.notes());
    // Memory exported by one logical device is the same memory in the other
    // (unlike memory of a different GPU, whose import the driver accepts and does not
    // honour), so the direct route works.
    assert_eq!(mesh.route(), Route::Peer);
    assert!(mesh.peer_kind().is_some());
    assert!(mesh.rejected().is_empty());
    exact_sums(&twice, Route::Peer, &[1, 255, 4096, 131_072]);
    // Two contexts of one GPU are time-sliced and a kernel that spins is not
    // preempted, so the push kernels, which need both partners to run at once,
    // cannot be tested here: the mesh finds out and sends every sum through the copy
    // engine.
    eprintln!("push kernels: {} values", mesh.push_values());
}
