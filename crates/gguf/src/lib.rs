//! Reader for GGUF model files.
//!
//! A model is opened once and mapped into memory. Opening validates the whole
//! header (every key, every array, every tensor's bounds), so later accessors
//! cannot fail and return plain borrowed data. Tensor data is never copied:
//! [`Gguf::data`] is a slice of the mapping, and [`TensorInfo`] also carries
//! the absolute file offset for readers that bypass the page cache
//! (direct I/O streaming from SSD).
//!
//! Split models (`name-00001-of-00003.gguf`) are opened as one [`Gguf`]: the
//! metadata comes from the first part and the tensor directory spans all
//! parts.

mod builder;
mod error;
mod value;

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ares_core::GgmlType;
use ares_core::units::align_up;
use memmap2::Mmap;

pub use builder::GgufBuilder;
pub use error::{Error, Result};
pub use value::{Array, ArrayIter, Value, ValueType};

/// The most dimensions a tensor can have (ggml's `GGML_MAX_DIMS`).
pub const MAX_DIMS: usize = 4;

const DEFAULT_ALIGNMENT: u64 = 32;

/// The shape of a tensor in ggml order: `dims()[0]` is the innermost
/// dimension. A weight matrix that maps `n_in` inputs to `n_out` outputs has
/// dims `[n_in, n_out]`, so one row of quantized blocks spans `dims()[0]`
/// values.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dims {
    len: u8,
    dims: [u64; MAX_DIMS],
}

impl Dims {
    pub fn new(dims: &[u64]) -> Option<Self> {
        if dims.len() > MAX_DIMS {
            return None;
        }
        let mut out = [1u64; MAX_DIMS];
        out[..dims.len()].copy_from_slice(dims);
        Some(Self { len: dims.len() as u8, dims: out })
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.dims[..usize::from(self.len)]
    }

    /// Dimension `i`, or 1 past the last dimension.
    pub fn get(&self, i: usize) -> u64 {
        self.dims.get(i).copied().unwrap_or(1)
    }

    /// The number of values, or `None` on overflow.
    pub fn n_values(&self) -> Option<u64> {
        self.as_slice().iter().try_fold(1u64, |acc, d| acc.checked_mul(*d))
    }
}

impl std::fmt::Debug for Dims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

/// One entry of the tensor directory.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Dims,
    pub ty: GgmlType,
    /// Which part of a split model holds the data (0 for a single file).
    pub part: u16,
    /// Offset of the data from the start of that part's file.
    pub offset: u64,
    /// Size of the data in bytes.
    pub size: u64,
}

struct Part {
    path: PathBuf,
    map: Mmap,
}

struct MetaEntry {
    key: String,
    ty: ValueType,
    range: Range<usize>,
}

/// An opened GGUF model.
pub struct Gguf {
    parts: Vec<Part>,
    version: u32,
    alignment: u64,
    meta: Vec<MetaEntry>,
    meta_index: HashMap<String, usize>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.parts[0].path)
            .field("parts", &self.parts.len())
            .field("version", &self.version)
            .field("metadata", &self.meta.len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl Gguf {
    /// Opens a model. `path` may be a single file or any part of a split
    /// model; the other parts are found next to it.
    ///
    /// The files are mapped read-only and must not be truncated or rewritten
    /// while the model is open.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let first = match split_name(path) {
            Some(split) if split.part != 1 => split.path_of(1),
            _ => path.to_owned(),
        };

        let part0 = map_file(&first)?;
        let parsed = parse(&part0, 0)?;

        let mut gguf = Self {
            parts: Vec::new(),
            version: parsed.version,
            alignment: parsed.alignment,
            meta: Vec::new(),
            meta_index: HashMap::new(),
            tensors: Vec::new(),
            tensor_index: HashMap::new(),
        };
        for (i, entry) in parsed.meta.into_iter().enumerate() {
            if gguf.meta_index.insert(entry.key.clone(), i).is_some() {
                return Err(invalid(&first, format!("metadata key '{}' appears twice", entry.key)));
            }
            gguf.meta.push(entry);
        }
        gguf.parts.push(part0);
        gguf.add_tensors(parsed.tensors)?;

        let count = gguf.u64("split.count").unwrap_or(0);
        if count > 1 {
            let count = u32::try_from(count)
                .map_err(|_| invalid(&first, format!("split.count {count} is not plausible")))?;
            let split = split_name(&first).filter(|s| s.count == count).ok_or_else(|| {
                invalid(
                    &first,
                    format!(
                        "the file says it is part of a model split in {count}, \
                         but its name does not end in -00001-of-{count:05}.gguf"
                    ),
                )
            })?;
            for part in 2..=count {
                let expected = split.path_of(part);
                if !expected.is_file() {
                    return Err(Error::MissingSplit { path: first.clone(), part, count, expected });
                }
                let mapped = map_file(&expected)?;
                let index = u16::try_from(gguf.parts.len())
                    .map_err(|_| invalid(&first, "too many parts".into()))?;
                let parsed = parse(&mapped, index)?;
                gguf.parts.push(mapped);
                gguf.add_tensors(parsed.tensors)?;
            }
        }
        Ok(gguf)
    }

    fn add_tensors(&mut self, tensors: Vec<TensorInfo>) -> Result<()> {
        for tensor in tensors {
            let index = self.tensors.len();
            if self.tensor_index.insert(tensor.name.clone(), index).is_some() {
                let path = &self.parts[usize::from(tensor.part)].path;
                return Err(invalid(path, format!("tensor '{}' appears twice", tensor.name)));
            }
            self.tensors.push(tensor);
        }
        Ok(())
    }

    /// The GGUF format version of the file.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The alignment of tensor data in the file, in bytes.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// The path of part `part` (0 for a single file).
    pub fn path(&self, part: u16) -> &Path {
        &self.parts[usize::from(part)].path
    }

    /// The number of files the model is stored in.
    pub fn n_parts(&self) -> usize {
        self.parts.len()
    }

    /// The total size of all parts in bytes.
    pub fn file_size(&self) -> u64 {
        self.parts.iter().map(|p| p.map.len() as u64).sum()
    }

    /// The metadata value stored under `key`.
    pub fn get(&self, key: &str) -> Option<Value<'_>> {
        let entry = &self.meta[*self.meta_index.get(key)?];
        self.decode_entry(entry)
    }

    fn decode_entry(&self, entry: &MetaEntry) -> Option<Value<'_>> {
        let bytes = &self.parts[0].map[entry.range.clone()];
        value::decode(entry.ty, bytes).map(|(value, _)| value)
    }

    /// All metadata in file order.
    pub fn metadata(&self) -> impl Iterator<Item = (&str, Value<'_>)> {
        self.meta.iter().filter_map(|e| Some((e.key.as_str(), self.decode_entry(e)?)))
    }

    /// An integer value. Any integer type is accepted when the value is not
    /// negative.
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// A float value (`f32` or `f64`).
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    /// A string value that is valid UTF-8.
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn array(&self, key: &str) -> Option<Array<'_>> {
        self.get(key)?.as_array()
    }

    /// `general.architecture`, the key that selects the model implementation.
    pub fn architecture(&self) -> Option<&str> {
        self.str("general.architecture")
    }

    /// The tensor directory, in file order, first part first.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|i| &self.tensors[*i])
    }

    /// The data of a tensor of this model, as stored in the file.
    ///
    /// # Panics
    ///
    /// Panics when `tensor` does not come from this model's directory.
    pub fn data(&self, tensor: &TensorInfo) -> &[u8] {
        let map = &self.parts[usize::from(tensor.part)].map;
        &map[tensor.offset as usize..(tensor.offset + tensor.size) as usize]
    }
}

fn invalid(path: &Path, what: String) -> Error {
    Error::Invalid { path: path.to_owned(), what }
}

fn map_file(path: &Path) -> Result<Part> {
    let io = |source| Error::Io { path: path.to_owned(), source };
    let file = File::open(path).map_err(io)?;
    // SAFETY: the mapping is read-only and private to this process. The
    // contract of `Gguf::open` is that the file is not truncated or rewritten
    // while it is open; model files are written once and then only read.
    let map = unsafe { Mmap::map(&file) }.map_err(io)?;
    Ok(Part { path: path.to_owned(), map })
}

/// `prefix-00002-of-00005.gguf` taken apart.
struct SplitName {
    dir: PathBuf,
    prefix: String,
    part: u32,
    count: u32,
}

impl SplitName {
    fn path_of(&self, part: u32) -> PathBuf {
        self.dir.join(format!("{}-{:05}-of-{:05}.gguf", self.prefix, part, self.count))
    }
}

fn split_name(path: &Path) -> Option<SplitName> {
    let stem = path.file_name()?.to_str()?.strip_suffix(".gguf")?;
    let (rest, count) = stem.rsplit_once("-of-")?;
    let (prefix, part) = rest.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(count) || !digits(part) {
        return None;
    }
    let (part, count) = (part.parse().ok()?, count.parse().ok()?);
    if part == 0 || part > count {
        return None;
    }
    Some(SplitName {
        dir: path.parent().unwrap_or(Path::new("")).to_owned(),
        prefix: prefix.to_owned(),
        part,
        count,
    })
}

struct Parsed {
    version: u32,
    alignment: u64,
    meta: Vec<MetaEntry>,
    tensors: Vec<TensorInfo>,
}

/// A bounds-checked reader over the mapped header.
struct Cursor<'a> {
    path: &'a Path,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|end| *end <= self.buf.len()).ok_or(
            Error::Truncated { path: self.path.to_owned(), what, offset: self.pos as u64 },
        )?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }

    fn string(&mut self, what: &'static str) -> Result<&'a str> {
        let len = self.u64(what)?;
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        let bytes = self.take(len, what)?;
        std::str::from_utf8(bytes)
            .map_err(|_| invalid(self.path, format!("{what} is not valid UTF-8")))
    }
}

fn parse(part: &Part, part_index: u16) -> Result<Parsed> {
    let path = part.path.as_path();
    let buf = &part.map[..];
    let mut cur = Cursor { path, buf, pos: 0 };

    if buf.len() < 4 || &buf[..4] != b"GGUF" {
        return Err(Error::NotGguf { path: path.to_owned() });
    }
    cur.pos = 4;
    let version = cur.u32("the header")?;
    if version != 3 {
        return Err(Error::Version { path: path.to_owned(), version });
    }
    let n_tensors = cur.u64("the header")?;
    let n_meta = cur.u64("the header")?;

    // Each entry takes at least 12 bytes (metadata) or 24 bytes (tensor), so
    // counts beyond the file size are refused before anything is reserved.
    let plausible = |count: u64, min_size: u64| count <= buf.len() as u64 / min_size;
    if !plausible(n_meta, 12) || !plausible(n_tensors, 24) {
        return Err(invalid(
            path,
            format!("header counts ({n_meta} metadata, {n_tensors} tensors) exceed the file size"),
        ));
    }

    let mut meta = Vec::with_capacity(n_meta as usize);
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..n_meta {
        let key = cur.string("a metadata key")?.to_owned();
        let type_id = cur.u32("a metadata value")?;
        let ty = ValueType::from_id(type_id).ok_or_else(|| {
            invalid(path, format!("metadata key '{key}' has unknown value type {type_id}"))
        })?;
        let (value, used) = value::decode(ty, &buf[cur.pos..]).ok_or(Error::Truncated {
            path: path.to_owned(),
            what: "a metadata value",
            offset: cur.pos as u64,
        })?;
        if key == "general.alignment" {
            alignment = value
                .as_u64()
                .filter(|a| a.is_power_of_two())
                .ok_or_else(|| invalid(path, "general.alignment is not a power of two".into()))?;
        }
        meta.push(MetaEntry { key, ty, range: cur.pos..cur.pos + used });
        cur.pos += used;
    }

    struct Raw {
        name: String,
        dims: Dims,
        type_id: u32,
        rel_offset: u64,
    }
    let mut raw = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = cur.string("a tensor name")?.to_owned();
        let n_dims = cur.u32("a tensor entry")? as usize;
        if n_dims > MAX_DIMS {
            return Err(invalid(path, format!("tensor '{name}' has {n_dims} dimensions")));
        }
        let mut dims = [0u64; MAX_DIMS];
        for dim in dims.iter_mut().take(n_dims) {
            *dim = cur.u64("a tensor entry")?;
        }
        let dims = Dims::new(&dims[..n_dims]).expect("at most MAX_DIMS");
        let type_id = cur.u32("a tensor entry")?;
        let rel_offset = cur.u64("a tensor entry")?;
        raw.push(Raw { name, dims, type_id, rel_offset });
    }

    let data_start = align_up(cur.pos as u64, alignment)
        .ok_or_else(|| invalid(path, "header size overflows".into()))?;

    let mut tensors = Vec::with_capacity(raw.len());
    for Raw { name, dims, type_id, rel_offset } in raw {
        let ty = GgmlType::from_id(type_id).ok_or_else(|| Error::UnknownTensorType {
            path: path.to_owned(),
            tensor: name.clone(),
            type_id,
        })?;
        // Quantized blocks never straddle rows: the innermost dimension must
        // hold a whole number of blocks.
        if !dims.get(0).is_multiple_of(u64::from(ty.block_values())) {
            return Err(invalid(
                path,
                format!(
                    "tensor '{name}': row length {} is not a multiple of the {ty} block of {} values",
                    dims.get(0),
                    ty.block_values()
                ),
            ));
        }
        let size = dims
            .n_values()
            .and_then(|n| ty.bytes_for(n))
            .ok_or_else(|| invalid(path, format!("tensor '{name}': size overflows")))?;
        if !rel_offset.is_multiple_of(alignment) {
            return Err(invalid(
                path,
                format!("tensor '{name}': offset {rel_offset} is not aligned to {alignment}"),
            ));
        }
        let offset = data_start
            .checked_add(rel_offset)
            .filter(|start| start.checked_add(size).is_some_and(|end| end <= buf.len() as u64))
            .ok_or(Error::Truncated {
                path: path.to_owned(),
                what: "tensor data",
                offset: buf.len() as u64,
            })?;
        tensors.push(TensorInfo { name, dims, ty, part: part_index, offset, size });
    }

    Ok(Parsed { version, alignment, meta, tensors })
}
