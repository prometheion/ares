//! DFlash2, a block drafter from a file of its own that reads taps of the
//! model it drafts for: its shape and tensors ([`config`]), the CPU reference
//! ([`reference`](mod@reference)) and the drafter on a device ([`device`]).
//!
//! The drafter is a kind the runtime knows ([`ares_runtime::DrafterKind`]):
//! every model with a DFlash2 file gets it, and gives only what is its own
//! (where its taps are, its embedding of a token, its draft head).

pub mod config;
pub mod device;
pub mod reference;

pub use config::{ARCHITECTURE, Config, Error, Weights};
pub use device::{Drafter, DrafterBytes, DrafterSetup, PassSeq, bytes_of};
pub use reference::{Cells, Lattice, Reference};

/// A drafter file, opened and its shape read: what a model is given to draft
/// with. The weights are read from the mapped file when a device part opens.
pub struct File {
    pub gguf: ares_gguf::Gguf,
    pub config: Config,
}

impl File {
    /// Opens `path` and reads its shape; a variant the kind does not run is
    /// refused by its name.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let gguf = ares_gguf::Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let config = Config::from_gguf(&gguf).map_err(|e| format!("{}: {e}", path.display()))?;
        // The tensors are checked now, so a damaged file is refused at once.
        Weights::load(&gguf, &config).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { gguf, config })
    }

    /// The tensors, borrowed from the mapped file.
    pub fn weights(&self) -> Result<Weights<'_>, Error> {
        Weights::load(&self.gguf, &self.config)
    }

    /// The bytes of the drafter's tensors.
    pub fn bytes(&self) -> u64 {
        self.weights().map_or(0, |w| w.bytes())
    }

    /// Opens the drafter on `device` for a model of `n_embd` values a row,
    /// `n_layer` layers and `n_vocab` tokens, whose embedding of the mask token
    /// is `mask_row`: refused when the drafter does not fit the model.
    #[allow(clippy::too_many_arguments)]
    pub fn open_drafter(
        &self,
        n_embd: usize,
        n_layer: usize,
        n_vocab: usize,
        device: &ares_cuda::Device,
        ledger: &mut ares_core::Ledger,
        setup: DrafterSetup,
        mask_row: Vec<f32>,
    ) -> Result<Drafter, ares_runtime::ModelError> {
        let invalid = |e: String| ares_runtime::ModelError::Invalid(e);
        self.config.fits(n_embd, n_layer, n_vocab).map_err(|e| invalid(e.to_string()))?;
        let w = self.weights().map_err(|e| invalid(e.to_string()))?;
        Drafter::open(&self.config, &w, device, ledger, setup, mask_row)
    }
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dflash::File")
            .field("path", &self.gguf.path(0))
            .field("config", &self.config)
            .finish()
    }
}
