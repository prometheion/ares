//! Expert addressing against synthetic models, including a split one, and
//! reading an expert through the reader.

mod common;

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Class, Config, ExpertMap, ExpertRange, FileId, LayoutError, Read, Reader};
use common::*;

const EXPERTS: u64 = 8;
const NAMES: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

/// The matrices of one MoE layer: (name part, dims, type). The three have
/// different sizes, as gate/up and down do not match in real models.
fn matrices(experts: u64) -> [(&'static str, Vec<u64>, GgmlType); 3] {
    [
        ("ffn_gate_exps", vec![256, 2, experts], GgmlType::Q4_K), // 288 bytes per expert
        ("ffn_up_exps", vec![16, 4, experts], GgmlType::F32),     // 256
        ("ffn_down_exps", vec![256, 1, experts], GgmlType::Q6_K), // 210
    ]
}

fn expert_bytes(layer: usize, matrix: usize, expert: u64, len: usize) -> Vec<u8> {
    pattern(len, (layer * 100 + matrix * 10) as u64 + expert + 1)
}

fn add_layer(mut b: GgufBuilder, layer: usize, experts: u64) -> GgufBuilder {
    for (m, (name, dims, ty)) in matrices(experts).iter().enumerate() {
        let stride = ty.bytes_for(dims[0] * dims[1]).unwrap() as usize;
        let data: Vec<u8> = (0..experts).flat_map(|e| expert_bytes(layer, m, e, stride)).collect();
        b = b.tensor(&format!("blk.{layer}.{name}.weight"), dims, *ty, data);
    }
    b
}

/// Layer 0 is dense; layers 1 to 3 have experts. Single file, or split with
/// layers 0 and 1 in the first part and 2 and 3 in the second.
fn write_model(dir: &TestDir, split: bool) -> std::path::PathBuf {
    let dense = |b: GgufBuilder| b.tensor_f32("blk.0.ffn_up.weight", &[4, 4], &[1.0; 16]);
    let head = |count: u16| {
        GgufBuilder::new().str("general.architecture", "testmoe").u16("split.count", count)
    };
    if !split {
        let b = (1..=3).fold(dense(head(1)), |b, l| add_layer(b, l, EXPERTS));
        return dir.file("model.gguf", &b.build());
    }
    let first = add_layer(dense(head(2)), 1, EXPERTS);
    let second = (2..=3).fold(GgufBuilder::new(), |b, l| add_layer(b, l, EXPERTS));
    let path = dir.file("model-00001-of-00002.gguf", &first.build());
    dir.file("model-00002-of-00002.gguf", &second.build());
    path
}

#[test]
fn ranges_match_the_gguf_data_of_every_expert() {
    for setup in setups("expert") {
        for split in [false, true] {
            let path = write_model(&setup.dir, split);
            let gguf = Gguf::open(&path).unwrap();
            assert_eq!(gguf.n_parts(), if split { 2 } else { 1 });
            let map = ExpertMap::discover(&gguf, &NAMES).unwrap();

            assert_eq!(map.moe_layers().collect::<Vec<_>>(), [1, 2, 3]);
            assert_eq!(map.n_experts(0), None, "layer 0 is dense");
            assert_eq!(map.n_experts(2), Some(EXPERTS as usize));
            assert_eq!(map.n_experts(4), None);
            assert!(map.ranges(0, 0).is_none() && map.ranges(1, 8).is_none());

            let parts: Vec<Vec<u8>> =
                (0..gguf.n_parts()).map(|p| std::fs::read(gguf.path(p as u16)).unwrap()).collect();
            for layer in 1..=3 {
                assert_eq!(map.expert_bytes(layer), Some(288 + 256 + 210));
                for expert in 0..EXPERTS as usize {
                    let ranges: Vec<ExpertRange> = map.ranges(layer, expert).unwrap().collect();
                    assert_eq!(ranges.len(), 3, "one range per tensor");
                    assert_eq!(ranges.iter().map(|r| r.len as u64).sum::<u64>(), 754);
                    for (m, range) in ranges.iter().enumerate() {
                        let info =
                            gguf.tensor(&format!("blk.{layer}.{}.weight", NAMES[m])).unwrap();
                        let whole = gguf.data(info);
                        let stride = whole.len() / EXPERTS as usize;
                        let want = &whole[expert * stride..(expert + 1) * stride];
                        assert_eq!(range.file, FileId(info.part));
                        assert_eq!(range.len, stride);
                        // The range, taken from the file itself, is what
                        // Gguf::data holds, and what the test wrote.
                        let file = &parts[usize::from(info.part)];
                        assert_eq!(&file[range.offset as usize..][..range.len], want);
                        assert_eq!(want, &expert_bytes(layer, m, expert as u64, stride)[..]);
                    }
                }
            }
            if split {
                let part_of = |layer: usize| map.ranges(layer, 0).unwrap().next().unwrap().file;
                assert_eq!((part_of(1), part_of(2), part_of(3)), (FileId(0), FileId(1), FileId(1)));
            }
        }
    }
}

#[test]
fn an_expert_is_read_with_one_request_per_tensor() {
    for setup in setups("expert-read") {
        for split in [false, true] {
            let path = write_model(&setup.dir, split);
            let gguf = Gguf::open(&path).unwrap();
            let map = ExpertMap::discover(&gguf, &NAMES).unwrap();
            let reader = Reader::open_gguf(&gguf, setup.config()).unwrap();
            report(&format!(
                "expert-read: {} split={split} -> {}",
                setup.name(),
                reader.files()[0].mode
            ));

            let mut key = 0;
            for layer in [1usize, 3] {
                for expert in [0usize, 3, 7] {
                    let ranges: Vec<_> = map.ranges(layer, expert).unwrap().collect();
                    let keys: Vec<u64> = (key..key + ranges.len() as u64).collect();
                    key += ranges.len() as u64;
                    for (k, r) in keys.iter().zip(&ranges) {
                        let span = reader.span(r.file, r.offset, r.len).unwrap();
                        reader
                            .submit(Read {
                                key: *k,
                                file: r.file,
                                offset: r.offset,
                                len: r.len,
                                class: Class::Demand,
                                buffer: ares_io::Buffer::aligned(
                                    span.len,
                                    reader.buffer_alignment(),
                                ),
                            })
                            .unwrap();
                    }
                    for (c, (m, _)) in collect(&reader, &keys).iter().zip(ranges.iter().enumerate())
                    {
                        let info =
                            gguf.tensor(&format!("blk.{layer}.{}.weight", NAMES[m])).unwrap();
                        let whole = gguf.data(info);
                        let stride = whole.len() / EXPERTS as usize;
                        assert_eq!(c.data(), &whole[expert * stride..(expert + 1) * stride]);
                    }
                }
            }
        }
    }
}

#[test]
fn an_explicit_list_of_names_gives_the_same_map() {
    let setup = &setups("expert-names")[0];
    let gguf = Gguf::open(write_model(&setup.dir, true)).unwrap();
    let by_pattern = ExpertMap::discover(&gguf, &NAMES).unwrap();
    let names: Vec<Vec<String>> = (0..4)
        .map(|l| {
            if l == 0 {
                vec![]
            } else {
                NAMES.iter().map(|n| format!("blk.{l}.{n}.weight")).collect()
            }
        })
        .collect();
    let by_name = ExpertMap::new(&gguf, &names).unwrap();
    for layer in 0..4 {
        for expert in 0..EXPERTS as usize {
            let a: Option<Vec<_>> = by_pattern.ranges(layer, expert).map(Iterator::collect);
            let b: Option<Vec<_>> = by_name.ranges(layer, expert).map(Iterator::collect);
            assert_eq!(a, b, "layer {layer} expert {expert}");
        }
    }
}

#[test]
fn layers_that_disagree_or_are_incomplete_are_refused_with_the_names() {
    let setup = &setups("expert-bad")[0];

    // A layer whose up matrix holds 4 experts where gate holds 8.
    let b = GgufBuilder::new()
        .tensor("blk.0.ffn_gate_exps.weight", &[16, 2, 8], GgmlType::F32, vec![0; 16 * 2 * 8 * 4])
        .tensor("blk.0.ffn_up_exps.weight", &[16, 2, 4], GgmlType::F32, vec![0; 16 * 2 * 4 * 4]);
    let gguf = Gguf::open(setup.dir.file("mismatch.gguf", &b.build())).unwrap();
    let err = ExpertMap::discover(&gguf, &NAMES[..2]).unwrap_err();
    match &err {
        LayoutError::ExpertCountMismatch { layer: 0, experts: 4, first_experts: 8, .. } => {}
        other => panic!("{other:?}"),
    }
    let message = err.to_string();
    assert!(message.contains("ffn_up_exps") && message.contains("ffn_gate_exps"), "{message}");

    // A layer with gate and up but no down.
    let err = ExpertMap::discover(&gguf, &NAMES).unwrap_err();
    assert!(
        matches!(&err, LayoutError::IncompleteLayer { layer: 0, missing } if missing == "ffn_down_exps")
    );

    // A name that is not in the model, and a matrix that is not stacked.
    let err = ExpertMap::new(&gguf, &[vec!["blk.0.nothing.weight"]]).unwrap_err();
    assert!(matches!(err, LayoutError::UnknownTensor(n) if n == "blk.0.nothing.weight"));
    let flat = GgufBuilder::new().tensor_f32("blk.0.ffn_up_exps.weight", &[4, 4], &[0.0; 16]);
    let gguf = Gguf::open(setup.dir.file("flat.gguf", &flat.build())).unwrap();
    let err = ExpertMap::new(&gguf, &[vec!["blk.0.ffn_up_exps.weight"]]).unwrap_err();
    assert!(matches!(err, LayoutError::NotStacked { dims: 2, .. }), "{err}");

    // Nothing matches, and a pattern that matches two tensors of one layer.
    let err = ExpertMap::discover(&gguf, &NAMES).unwrap_err();
    assert!(matches!(err, LayoutError::NoMatch { .. }), "{err}");
    let twice = GgufBuilder::new()
        .tensor("blk.0.ffn_up_exps.weight", &[16, 2, 2], GgmlType::F32, vec![0; 16 * 2 * 2 * 4])
        .tensor("blk.0.ffn_up_exps.scale", &[16, 2, 2], GgmlType::F32, vec![0; 16 * 2 * 2 * 4]);
    let gguf = Gguf::open(setup.dir.file("twice.gguf", &twice.build())).unwrap();
    let err = ExpertMap::discover(&gguf, &["ffn_up_exps"]).unwrap_err();
    assert!(matches!(err, LayoutError::AmbiguousMatch { layer: 0, .. }), "{err}");
}

#[test]
fn the_default_config_reader_opens_a_split_model_part_by_part() {
    let setup = &setups("expert-split")[0];
    let path = write_model(&setup.dir, true);
    let gguf = Gguf::open(&path).unwrap();
    let reader = Reader::open_gguf(&gguf, Config::default()).unwrap();
    assert_eq!(reader.files().len(), 2);
    assert!(reader.files()[1].path.ends_with("model-00002-of-00002.gguf"));
    // Opening via the second part finds the same model.
    let again = Gguf::open(setup.dir.path.join("model-00002-of-00002.gguf")).unwrap();
    assert_eq!(again.n_parts(), 2);
}
