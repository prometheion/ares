//! The metadata keys of one architecture in a GGUF file, read strictly: a
//! missing key or a value of the wrong kind is an error that names the key.

use ares_gguf::{Gguf, Value};

use super::FileError;

/// The keys under one prefix (`qwen35.`, `clip.`) of a file.
pub(crate) struct Keys<'g> {
    gguf: &'g Gguf,
    prefix: String,
}

impl<'g> Keys<'g> {
    /// The keys of `architecture` in `gguf`, a file that must be of that
    /// architecture (`general.architecture`).
    pub(crate) fn of(gguf: &'g Gguf, architecture: &'static str) -> Result<Self, FileError> {
        let found = gguf.architecture().unwrap_or("(none)");
        if found != architecture {
            return Err(FileError::Architecture {
                found: found.to_owned(),
                expected: architecture,
            });
        }
        Ok(Self::under(gguf, architecture))
    }

    /// The keys under `prefix` (without its dot) in `gguf`, whatever the
    /// file's architecture.
    pub(crate) fn under(gguf: &'g Gguf, prefix: &str) -> Self {
        Self { gguf, prefix: format!("{prefix}.") }
    }

    /// Every key under the prefix: its full name and the part after the
    /// prefix.
    pub(crate) fn suffixes(&self) -> impl Iterator<Item = (&'g str, &'g str)> + '_ {
        self.gguf
            .metadata()
            .filter_map(|(key, _)| Some((key, key.strip_prefix(self.prefix.as_str())?)))
    }

    /// Refuses a key under the prefix that is not in `known`: it could change
    /// the computation and would be ignored silently otherwise. `of` names
    /// whose keys `known` are, for the message.
    pub(crate) fn only_known(&self, known: &[&str], of: &'static str) -> Result<(), FileError> {
        match self.suffixes().find(|(_, rest)| !known.contains(rest)) {
            Some((key, _)) => Err(FileError::UnknownKey { key: key.to_owned(), of }),
            None => Ok(()),
        }
    }

    /// The full name of the key `suffix`.
    pub(crate) fn name(&self, suffix: &str) -> String {
        format!("{}{suffix}", self.prefix)
    }

    /// The value of `suffix`, `None` when the file has no such key.
    pub(crate) fn get(&self, suffix: &str) -> Option<Value<'g>> {
        self.gguf.get(&self.name(suffix))
    }

    /// The value of `suffix`, which the file must have.
    pub(crate) fn value(&self, suffix: &str) -> Result<Value<'g>, FileError> {
        self.get(suffix).ok_or_else(|| FileError::MissingKey { key: self.name(suffix) })
    }

    /// The error of a value of `suffix` that is wrong: `problem` is the rest
    /// of a sentence that starts with the key.
    pub(crate) fn bad(&self, suffix: &str, problem: impl Into<String>) -> FileError {
        FileError::BadKey { key: self.name(suffix), problem: problem.into() }
    }

    /// A count that may be zero.
    pub(crate) fn uint0(&self, suffix: &str) -> Result<usize, FileError> {
        let value = self.value(suffix)?;
        value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| self.bad(suffix, format!("is {value}, not a count")))
    }

    /// A count of at least one.
    pub(crate) fn uint(&self, suffix: &str) -> Result<usize, FileError> {
        match self.uint0(suffix)? {
            0 => Err(self.bad(suffix, "is 0, it must be at least 1")),
            n => Ok(n),
        }
    }

    /// A finite float above zero.
    pub(crate) fn float(&self, suffix: &str) -> Result<f32, FileError> {
        let value = self.value(suffix)?;
        let x =
            value.as_f64().ok_or_else(|| self.bad(suffix, format!("is {value}, not a float")))?;
        if !(x.is_finite() && x > 0.0) {
            return Err(self.bad(suffix, format!("is {x}, it must be positive")));
        }
        Ok(x as f32)
    }

    /// An array of unsigned integers, `None` when the key is absent.
    pub(crate) fn uints(&self, suffix: &str) -> Result<Option<Vec<u64>>, FileError> {
        self.array(suffix, "a count", |item| item.as_u64())
    }

    /// An array of unsigned integers that the file must have.
    pub(crate) fn required_uints(&self, suffix: &str) -> Result<Vec<u64>, FileError> {
        self.uints(suffix)?.ok_or_else(|| FileError::MissingKey { key: self.name(suffix) })
    }

    /// An array of booleans, `None` when the key is absent.
    pub(crate) fn bools(&self, suffix: &str) -> Result<Option<Vec<bool>>, FileError> {
        self.array(suffix, "a boolean", |item| item.as_bool())
    }

    /// An array whose every item `read` takes, `None` when the key is absent;
    /// `kind` names what an item must be.
    fn array<T>(
        &self,
        suffix: &str,
        kind: &str,
        read: impl Fn(&Value<'g>) -> Option<T>,
    ) -> Result<Option<Vec<T>>, FileError> {
        let Some(value) = self.get(suffix) else { return Ok(None) };
        let array = value
            .as_array()
            .ok_or_else(|| self.bad(suffix, format!("is {value}, not an array")))?;
        let mut out = Vec::with_capacity(array.len() as usize);
        for item in array.iter() {
            out.push(
                read(&item).ok_or_else(|| self.bad(suffix, format!("holds {item}, not {kind}")))?,
            );
        }
        Ok(Some(out))
    }
}
