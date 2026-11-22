//! A projector as a language model serves it: the file, its configuration and
//! limits, where its weights live, the name it gives an image, and the tower
//! on the device that runs it.
//!
//! A language model that reads images holds one ([`Projector`]): it turns an
//! image's pixels into what the runtime takes ([`ImageInput`]: the pixels
//! resized by the size rule, the grid of rows and a name), counts the rows of
//! an image without resizing it, and builds the tower on the device part the
//! server chose for it ([`TowerAt`]), which runs the jobs the conductor plans
//! ([`ares_runtime::TowerJob`]).
//!
//! **The name** of an image is the SHA-256 of what the tower is given and of
//! what it is given to: the projector file's identity, the resized image's
//! width and height, and its pixels. Two files that resize to the same pixels
//! are one image; the same pixels through another projector are another. The
//! prefix cache names an image's cells from it (`ares_runtime::cell_name`), so
//! a conversation that sends its image again with every turn resumes from the
//! cache, and the store of image rows keeps one copy of the rows.

use std::sync::Arc;

use ares_cuda::{Device, Kernels, Stream};
use ares_gguf::Gguf;
use ares_runtime::{ImageInput, ModelError, TowerJob};
use sha2::{Digest, Sha256};

use crate::qwen3vl::config::Config;
use crate::qwen3vl::device::{DeviceTower, Home, MatrixTypes, Plan, Scratch, ScratchPlan};
use crate::qwen3vl::error::{Error, Result};
use crate::qwen3vl::image::{Limits, normalize, patch_rows, resize_bicubic, target_size};
use crate::qwen3vl::weights::Weights;

/// A projector file opened for serving.
pub struct Projector {
    gguf: Gguf,
    config: Config,
    limits: Limits,
    home: Home,
    plan: Plan,
    /// The file's identity, which every image's name covers.
    identity: Vec<u8>,
}

impl std::fmt::Debug for Projector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Projector")
            .field("path", &self.gguf.path(0))
            .field("limits", &self.limits)
            .field("home", &self.home)
            .finish_non_exhaustive()
    }
}

impl Projector {
    /// Opens a projector: its configuration and weights are read strictly
    /// (see [`Config::from_gguf`]), and the layout of its weights on a device
    /// is planned. `limits` bounds the rows of one image; `home` is where the
    /// weights stay between images.
    pub fn open(gguf: Gguf, limits: Limits, home: Home) -> Result<Self> {
        let config = Config::from_gguf(&gguf)?;
        if limits.min_rows == 0 || limits.min_rows > limits.max_rows {
            return Err(Error::Image {
                width: 0,
                height: 0,
                problem: format!(
                    "the limits of {} to {} rows per image are not a range",
                    limits.min_rows, limits.max_rows
                ),
            });
        }
        let plan = {
            let weights = Weights::load(&gguf, &config)?;
            Plan::new(&config, &MatrixTypes::of(&weights))?
        };
        let mut identity = b"qwen3vl projector\0".to_vec();
        for part in 0..gguf.n_parts() {
            crate::files::identity(gguf.path(part as u16), &mut identity);
        }
        Ok(Self { gguf, config, limits, home, plan, identity })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    pub fn home(&self) -> Home {
        self.home
    }

    /// The width of a row it gives, which must be the language model's.
    pub fn n_out(&self) -> u32 {
        self.config.n_out as u32
    }

    /// The units of a pass ([`Plan::units`]), which a job runs ranges of.
    pub fn units(&self) -> u32 {
        self.plan.units().len() as u32
    }

    /// The scratch of an image of the most rows, on the device.
    pub fn scratch_plan(&self) -> ScratchPlan {
        ScratchPlan::new(&self.config, &self.plan, self.limits.max_rows as u32)
    }

    /// The VRAM the tower holds on its device: its weights (all of them, or
    /// the two slots they pass through) and the scratch of the largest image.
    pub fn vram_bytes(&self) -> u64 {
        let weights = match self.home {
            Home::Vram => self.plan.image_bytes(),
            Home::Ram => 2 * self.plan.slot_bytes(),
        };
        weights + self.scratch_plan().bytes
    }

    /// The page-locked RAM the tower holds: its weights in the RAM home.
    pub fn pinned_bytes(&self) -> u64 {
        match self.home {
            Home::Vram => 0,
            Home::Ram => self.plan.image_bytes(),
        }
    }

    /// The rows an image of `width` x `height` pixels becomes: `(nx, ny)`.
    pub fn grid(&self, width: u32, height: u32) -> Result<(u32, u32)> {
        let (w, h) = target_size(&self.config, self.limits, width as usize, height as usize)?;
        let align = self.config.align();
        Ok(((w / align) as u32, (h / align) as u32))
    }

    /// An image of `width` x `height` RGB pixels as the tower takes it:
    /// resized by the size rule (copied when it has the size already) and
    /// named.
    pub fn prepare(&self, rgb: &[u8], width: u32, height: u32) -> Result<ImageInput> {
        let (w, h) = (width as usize, height as usize);
        if rgb.len() != w * h * 3 {
            return Err(Error::Image {
                width: w,
                height: h,
                problem: format!("{} bytes are not its pixels of three bytes", rgb.len()),
            });
        }
        let (tw, th) = target_size(&self.config, self.limits, w, h)?;
        let pixels =
            if (tw, th) == (w, h) { rgb.to_vec() } else { resize_bicubic(rgb, w, h, tw, th) };
        let align = self.config.align();
        let mut digest = Sha256::new();
        digest.update(b"ares image 1\0");
        digest.update(&self.identity);
        digest.update((tw as u64).to_le_bytes());
        digest.update((th as u64).to_le_bytes());
        digest.update(&pixels);
        Ok(ImageInput {
            name: digest.finalize().into(),
            width: tw as u32,
            height: th as u32,
            pixels,
            nx: (tw / align) as u32,
            ny: (th / align) as u32,
        })
    }

    /// The tower on `device`, with the scratch of the largest image.
    pub fn load(&self, device: &Device, kernels: &Kernels) -> Result<TowerAt> {
        let weights = Weights::load(&self.gguf, &self.config)?;
        let tower = DeviceTower::load(device, kernels, &self.config, &weights, self.home)?;
        let scratch = tower.scratch(self.limits.max_rows as u32)?;
        Ok(TowerAt { tower, scratch, units: self.units() })
    }
}

/// The tower of a projector on the device part that runs it, with its
/// scratch: what a part does with the jobs of the image tower.
#[derive(Debug)]
pub struct TowerAt {
    tower: DeviceTower,
    scratch: Scratch,
    units: u32,
}

impl TowerAt {
    /// The VRAM it holds: its weights or their slots, and the scratch.
    pub fn vram_bytes(&self) -> u64 {
        self.tower.vram_bytes() + self.scratch.bytes()
    }

    /// The page-locked RAM it holds.
    pub fn pinned_bytes(&self) -> u64 {
        self.tower.pinned_bytes()
    }

    /// Runs a job of the conductor on `stream` and waits for it: the units it
    /// names, after the image's patch rows went into the scratch when it is the
    /// first of its pass, and the rows read back when it is the last.
    pub fn run(
        &mut self,
        stream: &Stream,
        job: &mut TowerJob,
    ) -> std::result::Result<(), ModelError> {
        let failed = |e: Error| ModelError::Invalid(e.to_string());
        let image: &Arc<ImageInput> = &job.image;
        let config = self.tower.config().clone();
        if job.units.start == 0 {
            let pixels = normalize(&config, &image.pixels);
            let (rows, nx, ny) =
                patch_rows(&config, &pixels, image.width as usize, image.height as usize)
                    .map_err(failed)?;
            if (nx as u32, ny as u32) != (image.nx, image.ny) {
                return Err(ModelError::Invalid(format!(
                    "an image of {} x {} pixels has {nx} x {ny} rows, not the {} x {} it came with",
                    image.width, image.height, image.nx, image.ny
                )));
            }
            let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            // The device has run everything before this job: the conductor waits
            // for each step and each job.
            stream.synchronize()?;
            self.scratch.buffer().write(self.scratch.input_offset(), &bytes)?;
        }
        let range = job.units.start as usize..job.units.end as usize;
        // SAFETY: the output lies in the scratch, which this value owns and
        // keeps; the passes of this tower run on this one stream.
        unsafe {
            self.tower
                .enqueue_range(
                    stream,
                    &self.scratch,
                    image.nx,
                    image.ny,
                    self.scratch.output(),
                    range,
                )
                .map_err(failed)?;
        }
        if job.units.end >= self.units {
            stream.synchronize()?;
            let values = image.rows() as usize * config.n_out;
            let mut bytes = vec![0u8; values * 4];
            self.scratch.buffer().read(self.scratch.output_offset(), &mut bytes)?;
            job.rows.clear();
            job.rows.extend(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        }
        Ok(())
    }
}
