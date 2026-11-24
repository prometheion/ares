//! The experts most like each expert of a model, on disk: for every expert of
//! every layer, the other experts of the layer whose outputs are nearest to
//! its own, the nearest first. The router's pruning stands one of them in for
//! a pick that only the file holds (`ares_cuda::PruneRule::Similar`).
//!
//! The table is made once per model file (it takes the model's experts and a
//! text to measure them on) and kept like the usage profile
//! ([`super::ExpertProfile`]): a file named by the fingerprint of the model,
//! that says again what it was made for and ends in a checksum, written
//! atomically. The key is the profile's ([`ProfileKey`]).
//!
//! The layout, little endian:
//!
//! ```text
//! "ARESNEAR"  format (u32)  layers (u32)  experts (u32)  count (u32)  fingerprint (16 bytes)
//! entries, layer by layer, expert by expert, `count` each: expert (u16), closeness (f32), scale (f32)
//! checksum (16 bytes): ChainHash::root of everything before it
//! ```

use std::path::{Path, PathBuf};

use super::profile::{Ignored, ProfileKey, read_file, sealed, write_atomically};

/// The first bytes of every table.
const MAGIC: &[u8; 8] = b"ARESNEAR";

/// The format this engine writes and reads.
pub const NEAR_FORMAT: u32 = 1;

/// Bytes before the entries.
const HEADER: usize = 8 + 4 + 4 + 4 + 4 + 16;
const ENTRY: usize = 2 + 4 + 4;

/// One expert that is like another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Near {
    pub expert: u16,
    /// How alike the two experts' outputs are, 0 (nothing in common) to 1
    /// (the same up to `scale`): the share of the other's output that this
    /// one gives, measured on rows the other is picked for.
    pub closeness: f32,
    /// The factor that brings this expert's output nearest to the other's.
    pub scale: f32,
}

/// The experts most like each expert of every layer of a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertNeighbours {
    key: ProfileKey,
    count: usize,
    entries: Vec<Near>,
}

impl ExpertNeighbours {
    /// A table for `key` with `count` entries per expert, every one expert 0
    /// at closeness 0 until it is set.
    pub fn new(key: ProfileKey, count: usize) -> Self {
        let n = key.layers as usize * key.experts as usize * count;
        Self { key, count, entries: vec![Near { expert: 0, closeness: 0.0, scale: 1.0 }; n] }
    }

    pub fn key(&self) -> &ProfileKey {
        &self.key
    }

    /// The entries per expert.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The experts most like `expert` of `layer`, the nearest first.
    pub fn of(&self, layer: usize, expert: usize) -> &[Near] {
        let at = (layer * self.key.experts as usize + expert) * self.count;
        &self.entries[at..at + self.count]
    }

    pub fn of_mut(&mut self, layer: usize, expert: usize) -> &mut [Near] {
        let at = (layer * self.key.experts as usize + expert) * self.count;
        &mut self.entries[at..at + self.count]
    }

    /// The name of the table's file.
    pub fn file_name(key: &ProfileKey) -> String {
        key.named("near")
    }

    /// Where the table of `key` lives in `dir`.
    pub fn path(key: &ProfileKey, dir: &Path) -> PathBuf {
        dir.join(Self::file_name(key))
    }

    /// The bytes of the file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.key.header(MAGIC, NEAR_FORMAT, &[self.count as u32]);
        for near in &self.entries {
            out.extend_from_slice(&near.expert.to_le_bytes());
            out.extend_from_slice(&near.closeness.to_le_bytes());
            out.extend_from_slice(&near.scale.to_le_bytes());
        }
        sealed(out)
    }

    /// The table in `bytes`, if it is one made for `key`.
    ///
    /// # Errors
    ///
    /// Why it is not: see [`Ignored`].
    pub fn from_bytes(bytes: &[u8], key: &ProfileKey) -> Result<Self, Ignored> {
        let (values, words) = key.open(bytes, MAGIC, NEAR_FORMAT, HEADER)?;
        let (experts, count) = (key.experts, words[2] as usize);
        if values.len() != ENTRY * key.layers as usize * experts as usize * count {
            return Err(Ignored::Damaged("its length does not fit its shape"));
        }
        let entries: Vec<Near> = values
            .as_chunks::<ENTRY>()
            .0
            .iter()
            .map(|e| Near {
                expert: u16::from_le_bytes([e[0], e[1]]),
                closeness: f32::from_le_bytes(e[2..6].try_into().expect("4 bytes")),
                scale: f32::from_le_bytes(e[6..10].try_into().expect("4 bytes")),
            })
            .collect();
        if entries.iter().any(|n| {
            u32::from(n.expert) >= experts
                || !(0.0..=1.0).contains(&n.closeness)
                || !n.scale.is_finite()
        }) {
            return Err(Ignored::Damaged("an entry that names no expert of the layer"));
        }
        Ok(Self { key: *key, count, entries })
    }

    /// The table for `key` in `dir`: `Ok(None)` when there is none.
    ///
    /// # Errors
    ///
    /// A file that is there and is not taken, and why.
    pub fn read(dir: &Path, key: &ProfileKey) -> Result<Option<Self>, Ignored> {
        read_file(&Self::path(key, dir))?.map(|bytes| Self::from_bytes(&bytes, key)).transpose()
    }

    /// Writes the table into `dir` (made if it is missing), atomically: a
    /// reader finds the table that was there or this one, whole. Returns the
    /// path.
    ///
    /// # Errors
    ///
    /// What the file system said; the table that was there stays.
    pub fn write(&self, dir: &Path) -> std::io::Result<PathBuf> {
        write_atomically(dir, &Self::file_name(&self.key), &self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ProfileKey {
        ProfileKey::new(b"model one: its header and samples", 2, 6)
    }

    fn table(key: ProfileKey) -> ExpertNeighbours {
        let mut table = ExpertNeighbours::new(key, 3);
        for layer in 0..2 {
            for expert in 0..6 {
                for (i, near) in table.of_mut(layer, expert).iter_mut().enumerate() {
                    *near = Near {
                        expert: ((expert + i + 1) % 6) as u16,
                        closeness: 0.9 - 0.2 * i as f32,
                        scale: 1.0 + layer as f32 * 0.25,
                    };
                }
            }
        }
        table
    }

    #[test]
    fn a_table_comes_back_as_it_was_written() {
        let dir = std::env::temp_dir().join(format!("ares-near-back-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let written = table(key());
        let path = written.write(&dir).unwrap();
        assert!(path.to_str().unwrap().ends_with(".near"));
        let read = ExpertNeighbours::read(&dir, &key()).unwrap().expect("it is there");
        assert_eq!(read, written);
        assert_eq!(read.of(1, 5)[0], Near { expert: 0, closeness: 0.9, scale: 1.25 });
        assert_eq!(std::fs::read(&path).unwrap().len(), HEADER + ENTRY * 2 * 6 * 3 + 16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_table_of_another_file_or_a_damaged_one_is_ignored() {
        let other = ProfileKey::new(b"model two", 2, 6);
        let why = ExpertNeighbours::from_bytes(&table(other).to_bytes(), &key()).unwrap_err();
        assert!(matches!(why, Ignored::AnotherFile), "{why}");
        let mut bytes = table(key()).to_bytes();
        bytes[HEADER + 3] ^= 0x40;
        let why = ExpertNeighbours::from_bytes(&bytes, &key()).unwrap_err();
        assert!(matches!(why, Ignored::Damaged(_)), "{why}");
        // An entry that names an expert the layer does not have, under a
        // checksum that fits it.
        let mut odd = table(key());
        odd.of_mut(0, 0)[0].expert = 6;
        let why = ExpertNeighbours::from_bytes(&odd.to_bytes(), &key()).unwrap_err();
        assert!(matches!(why, Ignored::Damaged(_)), "{why}");
        let dir = std::env::temp_dir().join(format!("ares-near-none-{}", std::process::id()));
        assert!(ExpertNeighbours::read(&dir, &key()).unwrap().is_none());
    }
}
