//! Memory accounting.
//!
//! VRAM is the binding limit of the machines this engine targets, and running
//! out of it in the middle of a request is the failure to design out. So no
//! component allocates device or pinned memory on its own: it asks a
//! [`Ledger`] for a named reservation first. A ledger has a fixed capacity
//! and refuses a reservation that does not fit, which turns "out of memory
//! under load" into "refused at load time, with the list of who holds what".

use std::fmt;

use crate::units::Bytes;

/// Why a reservation was refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LedgerError {
    #[error("{ledger}: cannot reserve {requested} for '{name}': {free} free of {capacity}")]
    Exhausted { ledger: String, name: String, requested: Bytes, free: Bytes, capacity: Bytes },
    #[error("{ledger}: no reservation named '{name}'")]
    Unknown { ledger: String, name: String },
    #[error("{ledger}: a reservation named '{name}' exists already")]
    Duplicate { ledger: String, name: String },
}

/// A fixed budget of bytes with named reservations.
#[derive(Debug, Clone)]
pub struct Ledger {
    name: String,
    capacity: u64,
    reserved: u64,
    entries: Vec<(String, u64)>,
}

impl Ledger {
    /// A ledger over `capacity` bytes. `name` identifies it in errors and
    /// reports (for example `"CUDA0"` or `"pinned RAM"`).
    pub fn new(name: impl Into<String>, capacity: u64) -> Self {
        Self { name: name.into(), capacity, reserved: 0, entries: Vec::new() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    pub fn free(&self) -> u64 {
        self.capacity - self.reserved
    }

    /// Reserves `bytes` under `name`. Names are unique within a ledger.
    pub fn reserve(&mut self, name: impl Into<String>, bytes: u64) -> Result<(), LedgerError> {
        let name = name.into();
        if self.entries.iter().any(|(n, _)| *n == name) {
            return Err(LedgerError::Duplicate { ledger: self.name.clone(), name });
        }
        if bytes > self.free() {
            return Err(LedgerError::Exhausted {
                ledger: self.name.clone(),
                name,
                requested: Bytes(bytes),
                free: Bytes(self.free()),
                capacity: Bytes(self.capacity),
            });
        }
        self.reserved += bytes;
        self.entries.push((name, bytes));
        Ok(())
    }

    /// Releases the reservation `name` and returns its size.
    pub fn release(&mut self, name: &str) -> Result<u64, LedgerError> {
        let index = self.entries.iter().position(|(n, _)| n == name).ok_or_else(|| {
            LedgerError::Unknown { ledger: self.name.clone(), name: name.to_owned() }
        })?;
        let (_, bytes) = self.entries.remove(index);
        self.reserved -= bytes;
        Ok(bytes)
    }

    /// The reservations in the order they were made.
    pub fn entries(&self) -> impl Iterator<Item = (&str, u64)> {
        self.entries.iter().map(|(n, b)| (n.as_str(), *b))
    }
}

impl fmt::Display for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: {} of {} reserved, {} free",
            self.name,
            Bytes(self.reserved),
            Bytes(self.capacity),
            Bytes(self.free())
        )?;
        for (name, bytes) in &self.entries {
            writeln!(f, "  {:>12}  {name}", Bytes(*bytes).to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_what_does_not_fit_and_keeps_the_books() {
        let mut ledger = Ledger::new("CUDA0", 1000);
        ledger.reserve("weights", 700).unwrap();
        ledger.reserve("kv pages", 200).unwrap();
        let refused = ledger.reserve("scratch", 101).unwrap_err();
        assert!(matches!(refused, LedgerError::Exhausted { .. }));
        assert_eq!(ledger.free(), 100);
        ledger.reserve("scratch", 100).unwrap();
        assert_eq!(ledger.free(), 0);
        assert_eq!(ledger.release("kv pages"), Ok(200));
        assert_eq!(ledger.free(), 200);
        assert!(matches!(ledger.release("kv pages"), Err(LedgerError::Unknown { .. })));
        assert!(matches!(ledger.reserve("weights", 1), Err(LedgerError::Duplicate { .. })));
        assert_eq!(ledger.entries().map(|(n, _)| n).collect::<Vec<_>>(), ["weights", "scratch"]);
    }
}
