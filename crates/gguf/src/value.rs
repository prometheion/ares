//! Metadata values.
//!
//! Values borrow from the mapped file. Nothing is copied when a file is
//! opened: a vocabulary of 250 000 token strings stays in the page cache and
//! is decoded while it is iterated.

use std::fmt;

/// The type tag of a metadata value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub(crate) fn from_id(id: u32) -> Option<Self> {
        use ValueType::*;
        Some(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }

    /// Bytes of one value, for the types with a fixed size.
    pub(crate) fn fixed_size(self) -> Option<usize> {
        use ValueType::*;
        match self {
            U8 | I8 | Bool => Some(1),
            U16 | I16 => Some(2),
            U32 | I32 | F32 => Some(4),
            U64 | I64 | F64 => Some(8),
            String | Array => None,
        }
    }
}

/// One metadata value.
#[derive(Clone, Copy, Debug)]
pub enum Value<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    /// The bytes of a string. GGUF strings are meant to be UTF-8, but token
    /// texts are arbitrary byte sequences, so the raw bytes are kept.
    String(&'a [u8]),
    Array(Array<'a>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl<'a> Value<'a> {
    /// The value as an unsigned integer, for any integer type holding a
    /// non-negative value. Booleans are not integers.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as a signed integer, for any integer type whose value fits.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::I8(v) => Some(v.into()),
            Value::I16(v) => Some(v.into()),
            Value::I32(v) => Some(v.into()),
            Value::I64(v) => Some(v),
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => i64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as a float, for `f32` and `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// The raw bytes of a string value.
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        match *self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    /// A string value that is valid UTF-8.
    pub fn as_str(&self) -> Option<&'a str> {
        self.as_bytes().and_then(|b| std::str::from_utf8(b).ok())
    }

    pub fn as_array(&self) -> Option<Array<'a>> {
        match *self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn value_type(&self) -> ValueType {
        match self {
            Value::U8(_) => ValueType::U8,
            Value::I8(_) => ValueType::I8,
            Value::U16(_) => ValueType::U16,
            Value::I16(_) => ValueType::I16,
            Value::U32(_) => ValueType::U32,
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::Array(_) => ValueType::Array,
            Value::U64(_) => ValueType::U64,
            Value::I64(_) => ValueType::I64,
            Value::F64(_) => ValueType::F64,
        }
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "{:?}", String::from_utf8_lossy(v)),
            Value::Array(a) => {
                write!(f, "[")?;
                for (i, item) in a.iter().take(6).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                if a.len() > 6 {
                    write!(f, ", ... {} items", a.len())?;
                }
                write!(f, "]")
            }
        }
    }
}

/// An array value. Elements are decoded while iterating.
///
/// The bytes were validated when the file was opened, so iteration cannot
/// run past the end of the array.
#[derive(Clone, Copy)]
pub struct Array<'a> {
    pub(crate) elem: ValueType,
    pub(crate) len: u64,
    pub(crate) data: &'a [u8],
}

impl fmt::Debug for Array<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array").field("elem", &self.elem).field("len", &self.len).finish()
    }
}

impl<'a> Array<'a> {
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn elem_type(&self) -> ValueType {
        self.elem
    }

    pub fn iter(&self) -> ArrayIter<'a> {
        ArrayIter { elem: self.elem, left: self.len, data: self.data }
    }

    /// The raw bytes of each element of a string array. Empty for arrays of
    /// another type.
    pub fn strings(&self) -> impl Iterator<Item = &'a [u8]> + use<'a> {
        let iter = if self.elem == ValueType::String { self.iter() } else { ArrayIter::empty() };
        iter.filter_map(|v| v.as_bytes())
    }

    /// The elements as `u32`, when every element is a non-negative integer
    /// that fits.
    pub fn to_u32_vec(&self) -> Option<Vec<u32>> {
        self.iter().map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok())).collect()
    }

    /// The elements as `i32`, when every element is an integer that fits.
    pub fn to_i32_vec(&self) -> Option<Vec<i32>> {
        self.iter().map(|v| v.as_i64().and_then(|n| i32::try_from(n).ok())).collect()
    }

    /// The elements as `f32`, for float arrays.
    pub fn to_f32_vec(&self) -> Option<Vec<f32>> {
        self.iter().map(|v| v.as_f64().map(|n| n as f32)).collect()
    }
}

/// Iterator over the elements of an [`Array`].
#[derive(Debug)]
pub struct ArrayIter<'a> {
    elem: ValueType,
    left: u64,
    data: &'a [u8],
}

impl<'a> ArrayIter<'a> {
    fn empty() -> Self {
        Self { elem: ValueType::U8, left: 0, data: &[] }
    }
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = Value<'a>;

    fn next(&mut self) -> Option<Value<'a>> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let (value, used) = decode(self.elem, self.data)?;
        self.data = &self.data[used..];
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = usize::try_from(self.left).unwrap_or(usize::MAX);
        (left, Some(left))
    }
}

/// Arrays may nest; a file that nests deeper than this is refused instead of
/// being followed down the stack.
const MAX_ARRAY_DEPTH: u32 = 4;

/// Decodes one value of type `ty` from the start of `data` and returns it
/// with the number of bytes it took. `None` when `data` is too short or
/// malformed, which cannot happen for bytes that passed validation at open.
pub(crate) fn decode(ty: ValueType, data: &[u8]) -> Option<(Value<'_>, usize)> {
    decode_at(ty, data, 0)
}

fn decode_at(ty: ValueType, data: &[u8], depth: u32) -> Option<(Value<'_>, usize)> {
    fn arr<const N: usize>(data: &[u8]) -> Option<[u8; N]> {
        data.get(..N)?.try_into().ok()
    }
    Some(match ty {
        ValueType::U8 => (Value::U8(*data.first()?), 1),
        ValueType::I8 => (Value::I8(*data.first()? as i8), 1),
        ValueType::Bool => (Value::Bool(*data.first()? != 0), 1),
        ValueType::U16 => (Value::U16(u16::from_le_bytes(arr(data)?)), 2),
        ValueType::I16 => (Value::I16(i16::from_le_bytes(arr(data)?)), 2),
        ValueType::U32 => (Value::U32(u32::from_le_bytes(arr(data)?)), 4),
        ValueType::I32 => (Value::I32(i32::from_le_bytes(arr(data)?)), 4),
        ValueType::F32 => (Value::F32(f32::from_le_bytes(arr(data)?)), 4),
        ValueType::U64 => (Value::U64(u64::from_le_bytes(arr(data)?)), 8),
        ValueType::I64 => (Value::I64(i64::from_le_bytes(arr(data)?)), 8),
        ValueType::F64 => (Value::F64(f64::from_le_bytes(arr(data)?)), 8),
        ValueType::String => {
            let len = usize::try_from(u64::from_le_bytes(arr(data)?)).ok()?;
            let end = len.checked_add(8)?;
            (Value::String(data.get(8..end)?), end)
        }
        ValueType::Array => {
            if depth >= MAX_ARRAY_DEPTH {
                return None;
            }
            let elem = ValueType::from_id(u32::from_le_bytes(arr(data)?))?;
            let len = u64::from_le_bytes(arr(data.get(4..)?)?);
            let body = data.get(12..)?;
            let size = match elem.fixed_size() {
                Some(size) => {
                    let total = usize::try_from(len).ok()?.checked_mul(size)?;
                    (total <= body.len()).then_some(total)?
                }
                None => {
                    let mut pos = 0usize;
                    for _ in 0..len {
                        let (_, used) = decode_at(elem, body.get(pos..)?, depth + 1)?;
                        pos = pos.checked_add(used)?;
                    }
                    pos
                }
            };
            (Value::Array(Array { elem, len, data: &body[..size] }), 12 + size)
        }
    })
}
