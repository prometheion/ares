//! What every model reads from its GGUF files the same way: the metadata keys
//! of its architecture (`Keys`, strict: a key that is missing, of the wrong
//! kind or not known is an error), the tensors of its weight table (`Table`,
//! exact: a tensor that is missing, of other dimensions, not decodable or not
//! taken is an error), the stored [`Matrix`], the [`FileError`] that names the
//! key or tensor, and what a file is for the fingerprints of what is kept of
//! it on disk (`identity`: the prompt cache's state, the usage profile of the
//! experts). Each model keeps its own list of keys and tensors and its own
//! checks of how the values fit together. Besides the GGUF files, a draft
//! vocabulary ([`read_draft_vocabulary`]).

mod error;
mod keys;
mod table;

use std::path::Path;

pub use error::FileError;
pub(crate) use keys::Keys;
pub use table::Matrix;
pub(crate) use table::{Table, vocabulary};

/// The token ids of a draft vocabulary file: one per line, blank lines and
/// lines that start with `#` left out. Why not, with the line, when a line is
/// no token id.
pub fn read_draft_vocabulary(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read it: {e}"))?;
    let mut tokens = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let token = line.parse::<u32>();
        tokens.push(token.map_err(|_| format!("line {}: '{line}' is not a token id", number + 1))?);
    }
    Ok(tokens)
}

/// What of a model file goes into a fingerprint: its size, its first bytes
/// (the header, with the model's settings) and samples spread over the rest
/// (the weights). A file that was copied or moved is the same file; reading
/// all of it at every load would take as long as loading it.
pub(crate) fn identity(path: &Path, out: &mut Vec<u8>) {
    use std::os::unix::fs::FileExt;
    const HEAD: usize = 64 << 10;
    const SAMPLE: usize = 4096;
    const SAMPLES: u64 = 16;
    let Ok(file) = std::fs::File::open(path) else { return };
    let len = file.metadata().map_or(0, |meta| meta.len());
    out.extend_from_slice(&len.to_le_bytes());
    let mut buf = vec![0u8; HEAD];
    let got = file.read_at(&mut buf, 0).unwrap_or(0);
    out.extend_from_slice(&buf[..got]);
    for i in 1..=SAMPLES {
        let at = len / (SAMPLES + 1) * i;
        let got = file.read_at(&mut buf[..SAMPLE], at).unwrap_or(0);
        out.extend_from_slice(&buf[..got]);
    }
}
