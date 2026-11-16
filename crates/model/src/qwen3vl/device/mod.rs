//! The tower on one device: an image's patch rows in, the merged rows of the
//! language model's width out, in a buffer on the device.
//!
//! The pass is the reference's ([`crate::qwen3vl::reference`]) on the kernels
//! of `ares-cuda`: the products are the tile kernels over f16 operands with f32
//! sums ([`Kernels::matmul`] on the f16 path), everything else is
//! [`TowerKernels`]: the norms with their bias and the GELU written as the next
//! product's f16 input, the residual in f32 with the product's bias, the
//! position table resized on the device, query, key and value with their bias
//! and rotation as f16 rows per head, and attention of every patch over every
//! patch. A block's work before and after its attention runs for
//! [`GROUP_PATCHES`] patches at a time (see [`plan`]).
//!
//! **Two homes for the weights** ([`Home`]), as llama.cpp's `mmproj` with and
//! without `no-mmproj-offload`, both computed on the device:
//!
//! - [`Home::Vram`]: the weights are uploaded once and stay (888 MiB for the
//!   27B's projector).
//! - [`Home::Ram`]: the weights lie in page-locked RAM, counted in the engine's
//!   RAM limit by the model that carries the tower
//!   ([`DeviceTower::pinned_bytes`]), and every image uploads them unit by unit
//!   (the stem, each block, the merger in parts) into two slots of VRAM on a
//!   stream of their own: the upload of the next unit runs while the current one
//!   computes, a slot is overwritten once the unit in it is done, and nothing
//!   but the two slots (a block each, 58 MiB together) is held between images.
//!
//! The two homes launch the same kernels on the same units, so they give the
//! same rows bit for bit.

pub mod plan;

use std::time::Duration;

use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, HeadShape, Kernels, PinnedBuffer, PositionGrid,
    PrefillPath, Stream, TowerKernels, WeightRange,
};

use crate::qwen3vl::config::Config;
use crate::qwen3vl::error::{Error, Result};
use crate::qwen3vl::image::patch_rows;
use crate::qwen3vl::reference::{Rows, attention_scale, position_factor, rotation_step};
use crate::qwen3vl::weights::Weights;

pub use plan::{
    BlockAt, GROUP_PATCHES, MatrixAt, MatrixTypes, NormAt, PartAt, Plan, ScratchPlan, StemAt, Unit,
    UnitKind,
};

/// Where the tower's weights are kept between images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Home {
    /// In VRAM, uploaded once.
    Vram,
    /// In page-locked RAM, uploaded unit by unit for every image into two
    /// slots of VRAM. The default: the tower is used a fraction of a second per
    /// new image, and its VRAM is worth more as K/V cells or expert slots (the
    /// owner's preset sets `no-mmproj-offload`).
    #[default]
    Ram,
}

/// The activations of one image, on the device, for images of up to
/// [`Scratch::max_rows`] rows. The patch rows of an image go into
/// [`Scratch::input`] before its pass.
#[derive(Debug)]
pub struct Scratch {
    buffer: DeviceBuffer,
    plan: ScratchPlan,
}

impl Scratch {
    /// The largest image, in output rows.
    pub fn max_rows(&self) -> u32 {
        self.plan.max_rows
    }

    /// Its bytes on the device.
    pub fn bytes(&self) -> u64 {
        self.buffer.len()
    }

    /// Where a pass reads the patch rows of its image: `4 * nx * ny` rows of
    /// [`Config::patch_len`] `f32` values in the merger's order
    /// ([`crate::qwen3vl::patch_rows`]).
    pub fn input(&self) -> DevicePtr {
        self.at(&self.plan.attended)
    }

    /// The buffer itself, for a caller that writes the input with a copy of its
    /// own (at the offset of [`Scratch::input`]).
    pub fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    fn at(&self, range: &std::ops::Range<u64>) -> DevicePtr {
        self.buffer.ptr().offset(range.start)
    }
}

/// The weights in page-locked RAM and what streams them to the device.
struct Streamed {
    host: PinnedBuffer,
    slots: [DeviceBuffer; 2],
    copy: Stream,
    /// Recorded on the copy stream when a unit is in its slot.
    uploaded: [Event; 2],
    /// Recorded on the pass's stream when the unit in a slot is done with.
    consumed: [Event; 2],
}

enum Placed {
    Vram(DeviceBuffer),
    Ram(Streamed),
}

/// The geometry of one pass.
struct Pass<'a> {
    scratch: &'a Scratch,
    nx: u32,
    ny: u32,
    rows: u32,
    patches: u32,
    out: DevicePtr,
    /// How many units of the plan run, from the first.
    units: usize,
}

/// The tower of one projector on one device.
pub struct DeviceTower {
    device: Device,
    config: Config,
    kernels: Kernels,
    tower: TowerKernels,
    plan: Plan,
    placed: Placed,
    /// The factor between the angles of neighbouring pairs of the rotation.
    step: f32,
    /// What the attention's scores are multiplied by.
    scale: f32,
}

impl std::fmt::Debug for DeviceTower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTower")
            .field("device", &self.device)
            .field("home", &self.home())
            .field("image_bytes", &self.plan.image_bytes())
            .finish()
    }
}

impl Drop for DeviceTower {
    /// In the RAM home the uploads read the page-locked image and the passes
    /// read the slots: both wait for what is under way before they are freed.
    fn drop(&mut self) {
        if let Placed::Ram(s) = &self.placed {
            let _ = s.copy.synchronize();
            for event in &s.consumed {
                let _ = event.synchronize();
            }
        }
    }
}

/// The groups of a block's row-by-row work: (first patch, patches).
fn groups(patches: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..patches)
        .step_by(GROUP_PATCHES as usize)
        .map(move |p0| (p0, GROUP_PATCHES.min(patches - p0)))
}

impl DeviceTower {
    /// Lays the weights of a projector out on `device` in `home`: uploads them
    /// to VRAM, or writes them into page-locked RAM and takes the two slots.
    /// `kernels` are the engine's kernels on the device (the tile kernels of
    /// the products); the tower's own are loaded here. Refused for a matrix
    /// whose type has no tile kernel, whose weights a tile does not hold, or a
    /// head size the attention has no kernel for.
    pub fn load(
        device: &Device,
        kernels: &Kernels,
        config: &Config,
        weights: &Weights<'_>,
        home: Home,
    ) -> Result<Self> {
        let tower = TowerKernels::load(device)?;
        let unsupported =
            |what: &str, problem: String| Error::Device { what: what.to_owned(), problem };
        if !TowerKernels::has_attention(config.head_dim() as u32) {
            return Err(unsupported(
                "the attention",
                format!("heads of {} values have no kernel", config.head_dim()),
            ));
        }
        let mut matrices = vec![("mm.0", weights.mm0.weight), ("mm.2", weights.mm2.weight)];
        for b in &weights.blocks {
            matrices.extend([
                ("attn_qkv", b.qkv.weight),
                ("attn_out", b.out.weight),
                ("ffn_up", b.up.weight),
                ("ffn_down", b.down.weight),
            ]);
        }
        for (name, m) in &matrices {
            if !kernels.has_prefill_path(m.ty(), PrefillPath::F16) {
                return Err(unsupported(name, format!("{} has no tile kernel", m.ty())));
            }
            if WeightRange::of(m.ty(), m.bytes()).path(Some(PrefillPath::F16)).is_none() {
                return Err(unsupported(name, "a weight is beyond the range of f16".to_owned()));
            }
        }
        let plan = Plan::new(config, &MatrixTypes::of(weights))?;
        let placed = match home {
            Home::Vram => {
                let arena = DeviceBuffer::new(device, plan.image_bytes())?;
                let mut staging = vec![0u8; plan.slot_bytes() as usize];
                for unit in plan.units() {
                    let bytes = &mut staging[..unit.bytes as usize];
                    plan.write_unit(weights, unit, bytes);
                    arena.write(unit.at, bytes)?;
                }
                Placed::Vram(arena)
            }
            Home::Ram => {
                let mut host = PinnedBuffer::new(device, plan.image_bytes() as usize)?;
                for unit in plan.units() {
                    let (start, end) = (unit.at as usize, (unit.at + unit.bytes) as usize);
                    plan.write_unit(weights, unit, &mut host[start..end]);
                }
                let slot = || DeviceBuffer::new(device, plan.slot_bytes());
                Placed::Ram(Streamed {
                    host,
                    slots: [slot()?, slot()?],
                    copy: Stream::new(device, 0)?,
                    uploaded: [Event::new(device)?, Event::new(device)?],
                    consumed: [Event::new(device)?, Event::new(device)?],
                })
            }
        };
        Ok(Self {
            device: device.clone(),
            config: config.clone(),
            kernels: kernels.clone(),
            tower,
            plan,
            placed,
            step: rotation_step(config),
            scale: attention_scale(config),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn home(&self) -> Home {
        match self.placed {
            Placed::Vram(_) => Home::Vram,
            Placed::Ram(_) => Home::Ram,
        }
    }

    /// The layout of the weights.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The VRAM the weights hold between images: the whole image in VRAM, the
    /// two slots in RAM.
    pub fn vram_bytes(&self) -> u64 {
        match &self.placed {
            Placed::Vram(arena) => arena.len(),
            Placed::Ram(s) => s.slots.iter().map(DeviceBuffer::len).sum(),
        }
    }

    /// The page-locked RAM the weights hold: the whole image in RAM, nothing in
    /// VRAM. The model that carries the tower adds it to what it books as
    /// required (`ares_runtime::Model::pinned_ram`), so that it counts in the
    /// engine's RAM limit.
    pub fn pinned_bytes(&self) -> u64 {
        match &self.placed {
            Placed::Vram(_) => 0,
            Placed::Ram(s) => s.host.len() as u64,
        }
    }

    /// The layout of a scratch for images of up to `max_rows` rows.
    pub fn scratch_plan(&self, max_rows: u32) -> ScratchPlan {
        ScratchPlan::new(&self.config, &self.plan, max_rows)
    }

    /// A scratch for images of up to `max_rows` rows. It serves every tower of
    /// the same projector on this device (the two homes of one projector share
    /// one).
    pub fn scratch(&self, max_rows: u32) -> Result<Scratch> {
        let plan = self.scratch_plan(max_rows);
        Ok(Scratch { buffer: DeviceBuffer::new(&self.device, plan.bytes)?, plan })
    }

    /// Enqueues the pass of one image of `nx * ny` rows on `stream`: reads the
    /// patch rows in `scratch.input()`, writes `nx * ny` rows of
    /// [`Config::n_out`] `f32` values to `out`, and does not wait. In the RAM
    /// home the weights' uploads run on the tower's own stream, ordered against
    /// `stream` by events.
    ///
    /// One pass at a time per tower: the next may be enqueued at once on the
    /// same stream (the slots are guarded), not on another.
    ///
    /// # Safety
    ///
    /// `out` must address `nx * ny * n_out` `f32` values on this device, alive
    /// until the pass has run, and the scratch must stay alive and untouched
    /// until then.
    pub unsafe fn enqueue(
        &mut self,
        stream: &Stream,
        scratch: &Scratch,
        nx: u32,
        ny: u32,
        out: DevicePtr,
    ) -> Result<()> {
        let units = self.plan.units().len();
        // SAFETY: the caller's guarantee.
        unsafe { self.enqueue_units(stream, scratch, nx, ny, out, units) }
    }

    /// The pass up to `units` units of the plan (the stem is one, then the
    /// blocks): for tests that compare the residual after a block.
    ///
    /// # Safety
    ///
    /// As [`DeviceTower::enqueue`].
    pub unsafe fn enqueue_units(
        &mut self,
        stream: &Stream,
        scratch: &Scratch,
        nx: u32,
        ny: u32,
        out: DevicePtr,
        units: usize,
    ) -> Result<()> {
        let rows = nx.checked_mul(ny).filter(|r| *r > 0 && *r <= scratch.max_rows());
        let Some(rows) = rows else {
            return Err(Error::Device {
                what: "an image".to_owned(),
                problem: format!(
                    "{nx} x {ny} rows; the scratch takes 1 to {} rows",
                    scratch.max_rows()
                ),
            });
        };
        let same = scratch.buffer.device().ordinal() == self.device.ordinal()
            && scratch.plan == self.scratch_plan(scratch.max_rows());
        if !same {
            return Err(Error::Device {
                what: "a scratch".to_owned(),
                problem: "it was made for another device or another projector".to_owned(),
            });
        }
        let pass = Pass { scratch, nx, ny, rows, patches: 4 * rows, out, units };
        match &self.placed {
            Placed::Vram(arena) => {
                for unit in self.plan.units().iter().take(pass.units) {
                    self.run_unit(stream, &pass, unit, arena.ptr().offset(unit.at))?;
                }
            }
            Placed::Ram(s) => {
                let units = &self.plan.units()[..pass.units.min(self.plan.units().len())];
                let upload = |u: usize| -> Result<()> {
                    let slot = u % 2;
                    let unit = &units[u];
                    s.copy.wait_event(&s.consumed[slot])?;
                    // SAFETY: the unit lies inside the pinned image, which lives as long as
                    // the tower, and fits the slot (the plan's slot is its largest unit); the
                    // slot is free once the unit before in it is done (the wait above).
                    unsafe {
                        s.copy.copy_from_host(
                            s.slots[slot].ptr(),
                            s.host.as_ptr().add(unit.at as usize),
                            unit.bytes as usize,
                        )?;
                    }
                    s.uploaded[slot].record(&s.copy)?;
                    Ok(())
                };
                for u in 0..units.len().min(2) {
                    upload(u)?;
                }
                for (u, unit) in units.iter().enumerate() {
                    let slot = u % 2;
                    stream.wait_event(&s.uploaded[slot])?;
                    // A unit that fails half enqueued still marks its slot, so that the
                    // next upload into it waits for the launches it did enqueue.
                    let ran = self.run_unit(stream, &pass, unit, s.slots[slot].ptr());
                    s.consumed[slot].record(stream)?;
                    ran?;
                    if u + 2 < units.len() {
                        upload(u + 2)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The pass of one image of normalised pixels, waited for: the patch rows
    /// uploaded, the pass, the rows read back. For tests and measurements.
    pub fn encode(
        &mut self,
        stream: &Stream,
        scratch: &Scratch,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Rows> {
        let (rows, nx, ny) = patch_rows(&self.config, pixels, width, height)?;
        let n_out = self.config.n_out;
        if nx * ny > scratch.max_rows() as usize {
            return Err(Error::Image {
                width,
                height,
                problem: format!(
                    "gives {} rows, the scratch takes {}",
                    nx * ny,
                    scratch.max_rows()
                ),
            });
        }
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        scratch.buffer.write(scratch.plan.attended.start, &bytes)?;
        let out = DeviceBuffer::new(&self.device, (nx * ny * n_out * 4) as u64)?;
        // SAFETY: `out` holds the rows and outlives the pass, which is waited for below.
        unsafe { self.enqueue(stream, scratch, nx as u32, ny as u32, out.ptr())? };
        stream.synchronize()?;
        let mut bytes = vec![0u8; nx * ny * n_out * 4];
        out.read(0, &mut bytes)?;
        let data = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok(Rows { nx, ny, width: n_out, data })
    }

    /// The residual after the first `blocks` blocks of the pass of an image, as
    /// rows of the tower's width in the merger's patch order. For tests.
    pub fn residual_after(
        &mut self,
        stream: &Stream,
        scratch: &Scratch,
        pixels: &[f32],
        width: usize,
        height: usize,
        blocks: usize,
    ) -> Result<Vec<f32>> {
        let (rows, nx, ny) = patch_rows(&self.config, pixels, width, height)?;
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        scratch.buffer.write(scratch.plan.attended.start, &bytes)?;
        // SAFETY: the units of the stem and the blocks write nothing to `out`.
        unsafe {
            self.enqueue_units(stream, scratch, nx as u32, ny as u32, DevicePtr::NULL, 1 + blocks)?
        };
        stream.synchronize()?;
        let mut bytes = vec![0u8; 4 * nx * ny * self.config.n_embd * 4];
        scratch.buffer.read(scratch.plan.x.start, &mut bytes)?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    /// The time the uploads of all units take in the RAM home, alone on the
    /// tower's copy stream (`None` in the VRAM home). For measurements.
    pub fn upload_time(&mut self) -> Result<Option<Duration>> {
        let Placed::Ram(s) = &self.placed else { return Ok(None) };
        let (start, end) = (Event::timed(&self.device)?, Event::timed(&self.device)?);
        s.copy.wait_event(&s.consumed[0])?;
        s.copy.wait_event(&s.consumed[1])?;
        start.record(&s.copy)?;
        for (u, unit) in self.plan.units().iter().enumerate() {
            // SAFETY: as in `enqueue_units`; no pass uses the slots meanwhile (the waits
            // above, and `&mut self`), and the slots are written again by the next pass's
            // uploads before it reads them.
            unsafe {
                s.copy.copy_from_host(
                    s.slots[u % 2].ptr(),
                    s.host.as_ptr().add(unit.at as usize),
                    unit.bytes as usize,
                )?;
            }
        }
        end.record(&s.copy)?;
        end.synchronize()?;
        let ms = end.elapsed_ms_since(&start)?;
        Ok(Some(Duration::from_secs_f64(f64::from(ms) / 1000.0)))
    }

    fn heads(&self, patches: u32) -> HeadShape {
        HeadShape {
            n_head: self.config.n_head as u32,
            head_dim: self.config.head_dim() as u32,
            n_patches: patches,
        }
    }

    /// # Safety
    ///
    /// `w`, `act` and `y` as [`Kernels::matmul`] wants them for `m` and `n_rows`.
    unsafe fn matmul(
        &self,
        stream: &Stream,
        m: &MatrixAt,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_rows: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe {
            self.kernels.matmul(
                stream,
                m.ty,
                PrefillPath::F16,
                w,
                act,
                y,
                m.n_in,
                m.n_out,
                n_rows,
            )?;
        }
        Ok(())
    }

    /// Enqueues the work of one unit whose weights lie at `base`.
    fn run_unit(
        &self,
        stream: &Stream,
        pass: &Pass<'_>,
        unit: &Unit,
        base: DevicePtr,
    ) -> Result<()> {
        let c = &self.config;
        let s = pass.scratch;
        let (e, eps) = (c.n_embd as u32, c.eps);
        let w = |at: u64| base.offset(at);
        let (x, heads, attended) = (s.at(&s.plan.x), s.at(&s.plan.heads), s.at(&s.plan.attended));
        let (act, wide, act_ff, y) =
            (s.at(&s.plan.act), s.at(&s.plan.wide), s.at(&s.plan.act_ff), s.at(&s.plan.y));
        let row = |p0: u32| u64::from(p0) * u64::from(e) * 4;
        // Every launch below reads and writes regions of the scratch, which its plan
        // sized for `pass.rows <= scratch.max_rows()` rows, weights at `base`, where the
        // plan laid the unit out, and `pass.out`, which holds the rows (the caller's
        // guarantee).
        match &unit.kind {
            // SAFETY: as said above.
            UnitKind::Stem(st) => unsafe {
                let n = c.patch_len() as u32;
                self.kernels.prepare_prefill_activations(
                    stream,
                    PrefillPath::F16,
                    attended,
                    heads,
                    n,
                    pass.patches,
                )?;
                self.matmul(stream, &st.patch, w(st.patch.at), heads, x, pass.patches)?;
                let (pw, ph) = (2 * pass.nx, 2 * pass.ny);
                let side = c.table_side;
                let grid = PositionGrid {
                    n: e,
                    side: side as u32,
                    pw,
                    ph,
                    sfx: position_factor(pw as usize, side),
                    sfy: position_factor(ph as usize, side),
                };
                self.tower.positions(stream, x, w(st.bias), w(st.table), grid)?;
            },
            // SAFETY: as said above.
            UnitKind::Block(_, b) => unsafe {
                let shape = self.heads(pass.patches);
                for (p0, n) in groups(pass.patches) {
                    let xg = x.offset(row(p0));
                    self.tower.norm_f16(
                        stream,
                        xg,
                        w(b.ln1.weight),
                        w(b.ln1.bias),
                        act,
                        e,
                        1,
                        n,
                        eps,
                    )?;
                    self.matmul(stream, &b.qkv, w(b.qkv.at), act, wide, n)?;
                    self.tower.rotate_heads(
                        stream,
                        wide,
                        w(b.qkv_bias),
                        heads,
                        shape,
                        p0,
                        n,
                        pass.nx,
                        self.step,
                    )?;
                }
                self.tower.attention(stream, heads, attended, shape, self.scale)?;
                for (p0, n) in groups(pass.patches) {
                    let xg = x.offset(row(p0));
                    let f16 = PrefillPath::F16;
                    self.kernels.prepare_prefill_activations(
                        stream,
                        f16,
                        attended.offset(row(p0)),
                        act,
                        e,
                        n,
                    )?;
                    self.matmul(stream, &b.out, w(b.out.at), act, y, n)?;
                    self.tower.residual(stream, xg, y, w(b.out_bias), e, n)?;
                    self.tower.norm_f16(
                        stream,
                        xg,
                        w(b.ln2.weight),
                        w(b.ln2.bias),
                        act,
                        e,
                        1,
                        n,
                        eps,
                    )?;
                    self.matmul(stream, &b.up, w(b.up.at), act, wide, n)?;
                    let ff = c.n_ff as u32;
                    self.tower.gelu_f16(stream, wide, w(b.up_bias), act_ff, ff, b.down.n_in, n)?;
                    self.matmul(stream, &b.down, w(b.down.at), act_ff, y, n)?;
                    self.tower.residual(stream, xg, y, w(b.down_bias), e, n)?;
                }
            },
            // SAFETY: as said above.
            UnitKind::Hidden(p) => unsafe {
                // The merger's input in `heads`, a part's output behind it, the hidden
                // layer in `attended`, its GELU (the input of mm.2) in `x`.
                let (r, width) = (pass.rows, 4 * e);
                let part_out = heads.offset(s.plan.part_out);
                if let Some(norm) = p.post_ln {
                    self.tower.norm_f16(
                        stream,
                        x,
                        w(norm.weight),
                        w(norm.bias),
                        heads,
                        e,
                        4,
                        r,
                        eps,
                    )?;
                }
                self.matmul(stream, &p.matrix, w(p.matrix.at), heads, part_out, r)?;
                let n = p.matrix.n_out;
                self.tower.place(
                    stream,
                    part_out,
                    attended,
                    DevicePtr::NULL,
                    r,
                    n,
                    width,
                    p.rows.start,
                )?;
                if let Some(bias) = p.bias {
                    self.tower.gelu_f16(stream, attended, w(bias), x, width, width, r)?;
                }
            },
            // SAFETY: as said above.
            UnitKind::Output(p) => unsafe {
                let r = pass.rows;
                let part_out = heads.offset(s.plan.part_out);
                self.matmul(stream, &p.matrix, w(p.matrix.at), x, part_out, r)?;
                let bias = w(p.bias.expect("every part of mm.2 has its bias"));
                let (n, n_out) = (p.matrix.n_out, c.n_out as u32);
                self.tower.place(stream, part_out, pass.out, bias, r, n, n_out, p.rows.start)?;
            },
        }
        Ok(())
    }
}
