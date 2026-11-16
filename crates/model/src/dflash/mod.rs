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
}

impl std::fmt::Debug for File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("dflash::File")
            .field("path", &self.gguf.path(0))
            .field("config", &self.config)
            .finish()
    }
}
