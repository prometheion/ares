//! Which element of a cooperative matrix of use A each invocation holds: tags element `e` of
//! lane `l` with `l * 8 + e`, stores the matrix row-major and prints, for the first lanes,
//! where their elements went. The kernels that build the A operand from registers
//! (`mm_*_r2`) assume the order of an `mma` fragment; `Kernels::cooperative_operand_in_registers`
//! says whether a device keeps it.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_coop_layout [device]
//! ```

use ares_vulkan::{Args, Device, DeviceBuffer, Module, Result, Stream, devices};

fn main() -> Result<()> {
    let only: Option<u32> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal)?;
        println!("Vulkan{} {}", info.ordinal, info.name);
        let stream = Stream::new(&device)?;
        let out = DeviceBuffer::new(&device, 256 * 4)?;
        let module = Module::load(
            &device,
            include_bytes!(concat!(env!("OUT_DIR"), "/coop_layout_probe.spv")),
        )?;
        let function = module.function("main")?;
        // SAFETY: `out` holds the 256 floats the kernel writes.
        unsafe { function.launch(&stream, [1, 1, 1], Args::new().ptr(out.ptr()))? };
        stream.synchronize()?;
        let mut bytes = [0u8; 256 * 4];
        out.read(0, &mut bytes)?;
        let value = |row: usize, col: usize| {
            let at = 4 * (row * 16 + col);
            f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
        };
        // position of (lane, element)
        let mut position = vec![[(0usize, 0usize); 8]; 32];
        for row in 0..16 {
            for col in 0..16 {
                let v = value(row, col);
                position[v / 8][v % 8] = (row, col);
            }
        }
        for lane in [0usize, 1, 2, 3, 4, 5, 31] {
            let cells: Vec<String> =
                position[lane].iter().map(|(r, c)| format!("({r},{c})")).collect();
            println!("  lane {lane:>2}: {}", cells.join(" "));
        }
        println!(
            "  the mma fragment order puts lane 0 at (0,0) (0,1) (8,0) (8,1) (0,8) (0,9) (8,8) (8,9)"
        );
    }
    Ok(())
}
