//! The usage profile of a model's experts on disk: the heat of every expert
//! of every layer, so that the tiers of the next start can be filled from it.
//!
//! A profile is a small file named by the fingerprint of the model it was
//! made for ([`ProfileKey`]). It holds the heat as the store had it, a count
//! of the picks in which recent ones weigh more, per layer of the model and
//! expert. The file says what it is made for, and a checksum closes it, so
//! that a profile of another model file, a damaged one or one in another
//! format is recognised and ignored ([`Ignored`]). It is written atomically:
//! a temporary file in the same directory, made durable, then renamed over
//! the old one, so that a reader finds the old profile or the new one and
//! never a part of either.
//!
//! The layout, little endian:
//!
//! ```text
//! "ARESHEAT"  format (u32)  layers (u32)  experts (u32)  fingerprint (16 bytes)
//! heat (f32), layer by layer, expert by expert
//! checksum (16 bytes): ChainHash::root of everything before it
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ares_state::ChainHash;
use ares_store::Fingerprint;

/// The first bytes of every profile.
const MAGIC: &[u8; 8] = b"ARESHEAT";

/// The format this engine writes and reads. Format 2 (2026-11-22): a wide
/// step counts each expert it picks once, as a decode step does; in format 1
/// it counted every row, so the prompts outweighed everything generated, and
/// such a profile is not used.
pub const PROFILE_FORMAT: u32 = 2;

/// Bytes before the heat, and after it.
const HEADER: usize = 8 + 4 + 4 + 4 + 16;
const CHECKSUM: usize = 16;

/// What a profile is made for: a model, by the fingerprint of its files, and
/// the shape of its experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileKey {
    pub fingerprint: [u8; 16],
    pub layers: u32,
    pub experts: u32,
}

impl ProfileKey {
    /// The key of a model whose files are described by `identity` (whatever
    /// changes when the weights do: sizes, headers, samples of the data),
    /// with `layers` layers of `experts` experts.
    pub fn new(identity: &[u8], layers: u32, experts: u32) -> Self {
        let fingerprint = Fingerprint::new(&[
            b"ares experts",
            identity,
            &layers.to_le_bytes(),
            &experts.to_le_bytes(),
        ]);
        Self { fingerprint: fingerprint.bytes(), layers, experts }
    }

    /// The name of the profile's file.
    pub fn file_name(&self) -> String {
        self.named("experts")
    }

    /// Where the profile lives in `dir`.
    pub fn path(&self, dir: &Path) -> PathBuf {
        dir.join(self.file_name())
    }

    /// The name of a file of this key with `extension`: the fingerprint in
    /// hex.
    pub(super) fn named(&self, extension: &str) -> String {
        let hex: String = self.fingerprint.iter().map(|b| format!("{b:02x}")).collect();
        format!("{hex}.{extension}")
    }

    /// The start of a file of this key: `magic`, the `format`, the shape,
    /// `more` words of the file's own, then the fingerprint.
    pub(super) fn header(&self, magic: &[u8; 8], format: u32, more: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(magic);
        for word in [format, self.layers, self.experts].iter().chain(more) {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&self.fingerprint);
        out
    }

    /// The values of a file of this key in `bytes` (after a header of
    /// `header` bytes that [`ProfileKey::header`] wrote with `magic` and
    /// `format`, before the checksum), and the words of the header from
    /// offset 12 on, if the file is one made for this key.
    ///
    /// # Errors
    ///
    /// Why it is not: see [`Ignored`].
    pub(super) fn open<'a>(
        &self,
        bytes: &'a [u8],
        magic: &[u8; 8],
        format: u32,
        header: usize,
    ) -> Result<(&'a [u8], Vec<u32>), Ignored> {
        if bytes.len() < header + CHECKSUM || &bytes[..8] != magic {
            return Err(Ignored::NotAProfile);
        }
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
        let found = word(8);
        if found < format {
            return Err(Ignored::OlderFormat { found });
        }
        if found > format {
            return Err(Ignored::UnknownFormat { found });
        }
        let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM);
        if ChainHash::root(body).to_bytes() != checksum {
            return Err(Ignored::Damaged("its checksum does not match"));
        }
        if bytes[header - 16..header] != self.fingerprint {
            return Err(Ignored::AnotherFile);
        }
        let (layers, experts) = (word(12), word(16));
        if (layers, experts) != (self.layers, self.experts) {
            return Err(Ignored::Shape {
                layers,
                experts,
                want_layers: self.layers,
                want_experts: self.experts,
            });
        }
        let words = (12..header - 16).step_by(4).map(word).collect();
        Ok((&body[header..], words))
    }
}

/// `bytes` with their checksum after them: what a file of a key holds.
pub(super) fn sealed(mut bytes: Vec<u8>) -> Vec<u8> {
    let checksum = ChainHash::root(&bytes).to_bytes();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The bytes of the file at `path`: `Ok(None)` when there is none.
///
/// # Errors
///
/// A file that is there and cannot be read.
pub(super) fn read_file(path: &Path) -> Result<Option<Vec<u8>>, Ignored> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Ignored::Unreadable(error)),
    }
}

/// Writes `bytes` into `dir` (made if it is missing) as the file `name`,
/// atomically: a reader finds the file that was there or this one, whole.
/// Returns the path.
///
/// # Errors
///
/// What the file system said; the file that was there stays.
pub(super) fn write_atomically(dir: &Path, name: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::fs::create_dir_all(dir)?;
    let path = dir.join(name);
    let temporary = dir.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut file = std::fs::File::create_new(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, &path)
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    // The rename is durable with the directory.
    if let Ok(directory) = std::fs::File::open(dir) {
        let _ = directory.sync_all();
    }
    Ok(path)
}

/// Why a file that is there was not taken as the profile.
#[derive(Debug, thiserror::Error)]
pub enum Ignored {
    #[error("it is no profile of experts")]
    NotAProfile,
    #[error("it is in format {found}, an older one than the {PROFILE_FORMAT} this engine reads")]
    OlderFormat { found: u32 },
    #[error("it is in format {found}, which this engine does not know")]
    UnknownFormat { found: u32 },
    #[error("it was made for another model file")]
    AnotherFile,
    #[error(
        "it has {layers} layers of {experts} experts, the model {want_layers} of {want_experts}"
    )]
    Shape { layers: u32, experts: u32, want_layers: u32, want_experts: u32 },
    #[error("it is damaged ({0})")]
    Damaged(&'static str),
    #[error("it cannot be read: {0}")]
    Unreadable(std::io::Error),
}

/// The heat of every expert of every layer of a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertProfile {
    key: ProfileKey,
    heat: Vec<f32>,
}

impl ExpertProfile {
    /// A profile for `key` in which no expert has any heat.
    pub fn new(key: ProfileKey) -> Self {
        Self { key, heat: vec![0.0; key.layers as usize * key.experts as usize] }
    }

    pub fn key(&self) -> &ProfileKey {
        &self.key
    }

    /// The heat of each expert of `layer`.
    pub fn layer(&self, layer: usize) -> &[f32] {
        let n = self.key.experts as usize;
        &self.heat[layer * n..][..n]
    }

    pub fn layer_mut(&mut self, layer: usize) -> &mut [f32] {
        let n = self.key.experts as usize;
        &mut self.heat[layer * n..][..n]
    }

    /// The experts that have any heat, over all layers.
    pub fn known(&self) -> usize {
        self.heat.iter().filter(|h| **h > 0.0).count()
    }

    /// The bytes of the file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.key.header(MAGIC, PROFILE_FORMAT, &[]);
        for heat in &self.heat {
            out.extend_from_slice(&heat.to_le_bytes());
        }
        sealed(out)
    }

    /// The profile in `bytes`, if it is one made for `key`.
    ///
    /// # Errors
    ///
    /// Why it is not: see [`Ignored`].
    pub fn from_bytes(bytes: &[u8], key: &ProfileKey) -> Result<Self, Ignored> {
        let (values, _) = key.open(bytes, MAGIC, PROFILE_FORMAT, HEADER)?;
        if values.len() != 4 * key.layers as usize * key.experts as usize {
            return Err(Ignored::Damaged("its length does not fit its shape"));
        }
        let heat: Vec<f32> =
            values.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect();
        if heat.iter().any(|h| !h.is_finite() || *h < 0.0) {
            return Err(Ignored::Damaged("a heat that is no count"));
        }
        Ok(Self { key: *key, heat })
    }

    /// The profile for `key` in `dir`: `Ok(None)` when there is none.
    ///
    /// # Errors
    ///
    /// A file that is there and is not taken, and why.
    pub fn read(dir: &Path, key: &ProfileKey) -> Result<Option<Self>, Ignored> {
        read_file(&key.path(dir))?.map(|bytes| Self::from_bytes(&bytes, key)).transpose()
    }

    /// Writes the profile into `dir` (made if it is missing) under its key's
    /// name, atomically: a reader finds the profile that was there or this
    /// one, whole. Returns the path.
    ///
    /// # Errors
    ///
    /// What the file system said; the profile that was there stays.
    pub fn write(&self, dir: &Path) -> std::io::Result<PathBuf> {
        write_atomically(dir, &self.key.file_name(), &self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    /// A directory that goes away with the test.
    struct Dir(PathBuf);

    impl Dir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "ares-profile-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn key() -> ProfileKey {
        ProfileKey::new(b"model one: its header and samples", 3, 8)
    }

    /// A profile in which expert `e` of layer `l` has heat `l * 10 + e`.
    fn profile(key: ProfileKey) -> ExpertProfile {
        let mut profile = ExpertProfile::new(key);
        for layer in 0..key.layers as usize {
            for (expert, heat) in profile.layer_mut(layer).iter_mut().enumerate() {
                *heat = (layer * 10 + expert) as f32 * 0.5;
            }
        }
        profile
    }

    #[test]
    fn a_profile_comes_back_as_it_was_written() {
        let dir = Dir::new("back");
        let written = profile(key());
        let path = written.write(&dir.0).unwrap();
        assert_eq!(path, key().path(&dir.0));
        assert!(path.file_name().unwrap().to_str().unwrap().ends_with(".experts"));
        let read = ExpertProfile::read(&dir.0, &key()).unwrap().expect("it is there");
        assert_eq!(read, written);
        assert_eq!(read.layer(2)[3], 11.5);
        // Expert 0 of layer 0 has no heat.
        assert_eq!(read.known(), 3 * 8 - 1);
        // The file is the header, a value per expert and the checksum.
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), HEADER + 4 * 24 + CHECKSUM);
        assert_eq!(&bytes[..8], b"ARESHEAT");
        // Nothing of the write is left beside it.
        let names: Vec<_> = std::fs::read_dir(&dir.0).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(names, [path]);
    }

    #[test]
    fn no_file_is_no_profile() {
        let dir = Dir::new("none");
        assert!(ExpertProfile::read(&dir.0, &key()).unwrap().is_none());
    }

    #[test]
    fn the_key_names_the_model_and_its_shape() {
        let other = ProfileKey::new(b"model two: another header", 3, 8);
        assert_ne!(key().fingerprint, other.fingerprint);
        assert_ne!(key().file_name(), other.file_name());
        // The same file with another shape is another key.
        assert_ne!(
            key().fingerprint,
            ProfileKey::new(b"model one: its header and samples", 3, 9).fingerprint
        );
        assert_eq!(key(), ProfileKey::new(b"model one: its header and samples", 3, 8));
    }

    #[test]
    fn a_profile_of_another_file_is_ignored() {
        let dir = Dir::new("other");
        let other = ProfileKey::new(b"model two: another header", 3, 8);
        // A profile of the other model put where this model's would be, as a
        // copied or renamed file would be.
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(key().path(&dir.0), profile(other).to_bytes()).unwrap();
        let why = ExpertProfile::read(&dir.0, &key()).unwrap_err();
        assert!(matches!(why, Ignored::AnotherFile), "{why}");
        assert_eq!(why.to_string(), "it was made for another model file");
    }

    #[test]
    fn a_profile_of_another_shape_is_ignored() {
        // The fingerprint covers the shape; a key that claims another shape
        // with the same fingerprint stands for a file that says it wrongly.
        let bytes = profile(key()).to_bytes();
        let wrong = ProfileKey { layers: 4, ..key() };
        let why = ExpertProfile::from_bytes(&bytes, &wrong).unwrap_err();
        assert!(matches!(why, Ignored::Shape { layers: 3, experts: 8, .. }), "{why}");
    }

    #[test]
    fn a_damaged_profile_is_ignored() {
        let bytes = profile(key()).to_bytes();
        // A bit of a heat flipped.
        let mut flipped = bytes.clone();
        flipped[HEADER + 13] ^= 0x10;
        let why = ExpertProfile::from_bytes(&flipped, &key()).unwrap_err();
        assert!(matches!(why, Ignored::Damaged(_)), "{why}");
        // Cut short, as a write that was not atomic would leave it.
        for len in [bytes.len() - 1, bytes.len() / 2, HEADER + CHECKSUM, 20, 0] {
            let why = ExpertProfile::from_bytes(&bytes[..len], &key()).unwrap_err();
            assert!(matches!(why, Ignored::Damaged(_) | Ignored::NotAProfile), "{len}: {why}");
        }
        // Something else altogether.
        let why = ExpertProfile::from_bytes(&[7u8; 200], &key()).unwrap_err();
        assert!(matches!(why, Ignored::NotAProfile), "{why}");
        // A value that is no count, under a checksum that fits it.
        let mut odd = profile(key());
        odd.layer_mut(1)[2] = f32::NAN;
        let why = ExpertProfile::from_bytes(&odd.to_bytes(), &key()).unwrap_err();
        assert!(matches!(why, Ignored::Damaged(_)), "{why}");
        odd.layer_mut(1)[2] = -1.0;
        let why = ExpertProfile::from_bytes(&odd.to_bytes(), &key()).unwrap_err();
        assert!(matches!(why, Ignored::Damaged(_)), "{why}");
    }

    #[test]
    fn a_profile_of_an_older_or_newer_format_is_ignored() {
        let mut bytes = profile(key()).to_bytes();
        for (format, older) in [(0u32, true), (PROFILE_FORMAT + 1, false)] {
            bytes[8..12].copy_from_slice(&format.to_le_bytes());
            let body = bytes.len() - CHECKSUM;
            let checksum = ChainHash::root(&bytes[..body]).to_bytes();
            bytes[body..].copy_from_slice(&checksum);
            let why = ExpertProfile::from_bytes(&bytes, &key()).unwrap_err();
            if older {
                assert!(matches!(why, Ignored::OlderFormat { found: 0 }), "{why}");
            } else {
                assert!(matches!(why, Ignored::UnknownFormat { .. }), "{why}");
            }
        }
    }

    #[test]
    fn a_reader_finds_a_whole_profile_while_others_write() {
        let dir = Dir::new("atomic");
        let (first, mut second) = (profile(key()), profile(key()));
        second.layer_mut(0)[0] = 1000.0;
        first.write(&dir.0).unwrap();
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for written in [&first, &second] {
                let done = &done;
                let dir = &dir.0;
                scope.spawn(move || {
                    for _ in 0..200 {
                        written.write(dir).unwrap();
                    }
                    done.store(true, Ordering::SeqCst);
                });
            }
            let mut reads = 0;
            while !done.load(Ordering::SeqCst) || reads < 100 {
                let read = ExpertProfile::read(&dir.0, &key()).unwrap().expect("always there");
                assert!(read == first || read == second, "a whole profile, one of the two");
                reads += 1;
            }
        });
        // No temporary file is left.
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
    }

    #[test]
    fn a_write_that_fails_leaves_the_profile_that_was_there() {
        use std::os::unix::fs::PermissionsExt;
        let dir = Dir::new("failed");
        let old = profile(key());
        old.write(&dir.0).unwrap();
        // A directory nobody may write to: the temporary file cannot be made.
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o555)).unwrap();
        let mut new = profile(key());
        new.layer_mut(2)[7] = 99.0;
        let failed = new.write(&dir.0);
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        if failed.is_ok() {
            // Run as a user whom permissions do not stop: nothing to see.
            return;
        }
        assert_eq!(ExpertProfile::read(&dir.0, &key()).unwrap(), Some(old));
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
    }
}
