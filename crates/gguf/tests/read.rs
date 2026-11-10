//! Reading files written by `GgufBuilder`, and refusing damaged ones.

use std::path::PathBuf;

use ares_core::GgmlType;
use ares_gguf::{Error, Gguf, GgufBuilder, ValueType};

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ares-gguf-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sample() -> GgufBuilder {
    let q4k_row = vec![0x5a; 144 * 2];
    GgufBuilder::new()
        .str("general.architecture", "testarch")
        .u32("testarch.block_count", 3)
        .f32("testarch.rope.freq_base", 10000.0)
        .bool("testarch.flag", true)
        .u64("testarch.big", 1 << 40)
        .i32("testarch.negative", -7)
        .str_array("tokenizer.ggml.tokens", &[&b"<s>"[..], b"hello", b"\xff\xfe"])
        .i32_array("tokenizer.ggml.token_type", &[3, 1, 6])
        .f32_array("tokenizer.ggml.scores", &[0.0, -1.5, -2.5])
        .tensor_f32("norm.weight", &[4], &[1.0, 2.0, 3.0, 4.0])
        .tensor("blk.0.ffn.weight", &[512, 1], GgmlType::Q4_K, q4k_row)
        .tensor_f32("matrix", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

#[test]
fn reads_metadata_and_tensors() {
    let dir = TempDir::new("read");
    let gguf = Gguf::open(dir.file("model.gguf", &sample().build())).unwrap();

    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.architecture(), Some("testarch"));
    assert_eq!(gguf.u64("testarch.block_count"), Some(3));
    assert_eq!(gguf.f64("testarch.rope.freq_base"), Some(10000.0));
    assert_eq!(gguf.bool("testarch.flag"), Some(true));
    assert_eq!(gguf.u64("testarch.big"), Some(1 << 40));
    assert_eq!(gguf.get("testarch.negative").unwrap().as_i64(), Some(-7));
    // A negative value is not an unsigned count, and a key is not a default.
    assert_eq!(gguf.u64("testarch.negative"), None);
    assert_eq!(gguf.u64("testarch.missing"), None);
    assert_eq!(gguf.str("testarch.block_count"), None);

    let tokens = gguf.array("tokenizer.ggml.tokens").unwrap();
    assert_eq!(tokens.elem_type(), ValueType::String);
    assert_eq!(tokens.len(), 3);
    let texts: Vec<&[u8]> = tokens.strings().collect();
    assert_eq!(texts, [&b"<s>"[..], b"hello", b"\xff\xfe"]);
    assert_eq!(gguf.array("tokenizer.ggml.token_type").unwrap().to_i32_vec(), Some(vec![3, 1, 6]));
    assert_eq!(
        gguf.array("tokenizer.ggml.scores").unwrap().to_f32_vec(),
        Some(vec![0.0, -1.5, -2.5])
    );

    let keys: Vec<&str> = gguf.metadata().map(|(k, _)| k).collect();
    assert_eq!(keys.len(), 9);
    assert_eq!(keys[0], "general.architecture");

    assert_eq!(gguf.tensors().len(), 3);
    let norm = gguf.tensor("norm.weight").unwrap();
    assert_eq!(norm.dims.as_slice(), [4]);
    assert_eq!(norm.ty, GgmlType::F32);
    let values: Vec<f32> =
        gguf.data(norm).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);

    let ffn = gguf.tensor("blk.0.ffn.weight").unwrap();
    assert_eq!(ffn.size, 288);
    assert_eq!(ffn.offset % gguf.alignment(), 0);
    assert!(gguf.data(ffn).iter().all(|b| *b == 0x5a));

    let matrix = gguf.tensor("matrix").unwrap();
    assert_eq!(matrix.dims.as_slice(), [2, 3]);
    assert_eq!(matrix.dims.get(3), 1);
    assert_eq!(matrix.dims.n_values(), Some(6));
    assert!(gguf.tensor("absent").is_none());
}

#[test]
fn refuses_files_that_are_not_gguf() {
    let dir = TempDir::new("notgguf");
    assert!(matches!(Gguf::open(dir.file("a.gguf", b"GGML....")), Err(Error::NotGguf { .. })));
    assert!(matches!(Gguf::open(dir.file("b.gguf", b"")), Err(Error::NotGguf { .. })));

    let mut old = sample().build();
    old[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert!(matches!(Gguf::open(dir.file("c.gguf", &old)), Err(Error::Version { version: 1, .. })));

    assert!(matches!(Gguf::open(dir.0.join("absent.gguf")), Err(Error::Io { .. })));
}

/// Every truncation of a valid file is either refused or, when only
/// trailing padding is missing, still fully readable. None may panic.
#[test]
fn truncated_files_are_refused_without_panicking() {
    let dir = TempDir::new("trunc");
    let bytes = sample().build();
    let mut refused = 0;
    for len in 0..bytes.len() {
        match Gguf::open(dir.file("t.gguf", &bytes[..len])) {
            Ok(gguf) => {
                for tensor in gguf.tensors() {
                    assert_eq!(gguf.data(tensor).len() as u64, tensor.size);
                }
            }
            Err(_) => refused += 1,
        }
    }
    // The last tensor ends less than one alignment unit before the end.
    assert!(refused >= bytes.len() - 32, "only {refused} of {} refused", bytes.len());
}

/// Flipping any single header byte must never panic or read out of bounds.
#[test]
fn corrupted_headers_do_not_panic() {
    let dir = TempDir::new("corrupt");
    let bytes = sample().build();
    let header_len = bytes.len() - 288 - 64;
    for i in 0..header_len {
        for flip in [0x01u8, 0x80, 0xff] {
            let mut damaged = bytes.clone();
            damaged[i] ^= flip;
            if let Ok(gguf) = Gguf::open(dir.file("c.gguf", &damaged)) {
                for (_, value) in gguf.metadata() {
                    let _ = value.to_string();
                }
                for tensor in gguf.tensors() {
                    let _ = gguf.data(tensor);
                }
            }
        }
    }
}

#[test]
fn refuses_inconsistent_tensor_entries() {
    let dir = TempDir::new("tensors");

    // A quantized row that is not a whole number of blocks.
    let bytes = GgufBuilder::new().tensor("w", &[256], GgmlType::Q4_K, vec![0; 144]).build();
    // The entry is: name ("w" with its length), dimension count, dimensions,
    // type id, offset.
    let name = [1, 0, 0, 0, 0, 0, 0, 0, b'w'];
    let at = bytes.windows(9).position(|w| w == name).unwrap() + 9 + 4;
    assert_eq!(bytes[at..at + 8], 256u64.to_le_bytes());
    let mut bad = bytes.clone();
    bad[at..at + 8].copy_from_slice(&200u64.to_le_bytes());
    let err = Gguf::open(dir.file("blocks.gguf", &bad)).unwrap_err();
    assert!(err.to_string().contains("not a multiple"), "{err}");

    // An element type id that does not exist.
    let mut bad = bytes.clone();
    bad[at + 8..at + 12].copy_from_slice(&4u32.to_le_bytes());
    assert!(matches!(
        Gguf::open(dir.file("type.gguf", &bad)),
        Err(Error::UnknownTensorType { type_id: 4, .. })
    ));

    // The same tensor name twice.
    let twice = GgufBuilder::new().tensor_f32("w", &[1], &[1.0]).tensor_f32("w", &[1], &[2.0]);
    let err = Gguf::open(dir.file("twice.gguf", &twice.build())).unwrap_err();
    assert!(err.to_string().contains("appears twice"), "{err}");

    // The same metadata key twice.
    let twice = GgufBuilder::new().u32("k", 1).u32("k", 2);
    let err = Gguf::open(dir.file("keys.gguf", &twice.build())).unwrap_err();
    assert!(err.to_string().contains("appears twice"), "{err}");
}

#[test]
fn opens_split_models_from_any_part() {
    let dir = TempDir::new("split");
    let part1 = GgufBuilder::new()
        .str("general.architecture", "testarch")
        .u16("split.no", 0)
        .u16("split.count", 3)
        .tensor_f32("a", &[2], &[1.0, 2.0]);
    let part2 =
        GgufBuilder::new().u16("split.no", 1).u16("split.count", 3).tensor_f32("b", &[1], &[3.0]);
    let part3 =
        GgufBuilder::new().u16("split.no", 2).u16("split.count", 3).tensor_f32("c", &[1], &[4.0]);
    let p1 = dir.file("m-00001-of-00003.gguf", &part1.build());
    let p2 = dir.file("m-00002-of-00003.gguf", &part2.build());
    let p3 = dir.0.join("m-00003-of-00003.gguf");

    match Gguf::open(&p1) {
        Err(Error::MissingSplit { part: 3, count: 3, expected, .. }) => assert_eq!(expected, p3),
        other => panic!("expected a missing part, got {other:?}"),
    }

    std::fs::write(&p3, part3.build()).unwrap();
    for opened in [&p1, &p2, &p3] {
        let gguf = Gguf::open(opened).unwrap();
        assert_eq!(gguf.n_parts(), 3);
        assert_eq!(gguf.architecture(), Some("testarch"));
        let names: Vec<&str> = gguf.tensors().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        let c = gguf.tensor("c").unwrap();
        assert_eq!(c.part, 2);
        assert_eq!(gguf.path(c.part), p3);
        assert_eq!(gguf.data(c), 4.0f32.to_le_bytes());
    }

    // A file that claims to be split but is not named like a part.
    let renamed = dir.file("renamed.gguf", &part1.build());
    let err = Gguf::open(renamed).unwrap_err();
    assert!(err.to_string().contains("does not end in"), "{err}");
}
