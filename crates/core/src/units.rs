//! Byte quantities.

use std::fmt;

pub const KIB: u64 = 1 << 10;
pub const MIB: u64 = 1 << 20;
pub const GIB: u64 = 1 << 30;

/// A byte count that prints in binary units (`"11.2 GiB"`, `"310 MiB"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0;
        if n >= GIB {
            write!(f, "{:.2} GiB", n as f64 / GIB as f64)
        } else if n >= MIB {
            write!(f, "{:.1} MiB", n as f64 / MIB as f64)
        } else if n >= KIB {
            write!(f, "{:.1} KiB", n as f64 / KIB as f64)
        } else {
            write!(f, "{n} B")
        }
    }
}

/// `value` rounded up to a multiple of `align`, which must be a power of two.
/// `None` on overflow.
pub fn align_up(value: u64, align: u64) -> Option<u64> {
    debug_assert!(align.is_power_of_two());
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_binary_units() {
        assert_eq!(Bytes(512).to_string(), "512 B");
        assert_eq!(Bytes(1536).to_string(), "1.5 KiB");
        assert_eq!(Bytes(310 * MIB).to_string(), "310.0 MiB");
        assert_eq!(Bytes(12 * GIB).to_string(), "12.00 GiB");
    }

    #[test]
    fn aligns_up() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(u64::MAX, 32), None);
    }
}
