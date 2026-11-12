//! The rows of a step that a file's types ask for (`step_rows`), on made-up
//! models: no GPU is opened.

mod common;

use ares_core::GgmlType;
use ares_cuda::Precision;
use ares_gguf::Gguf;
use ares_model::qwen35::device::{BATCHED_ROWS, Options, step_rows};
use ares_model::qwen35::{Config, Weights};
use common::{Spec, Synth, TempDir};

fn rows(matrix_type: GgmlType, precision: Precision) -> u32 {
    let synth = Synth::new(Spec { matrix_type, ..Spec::default() });
    let dir = TempDir::new("rows");
    let gguf = Gguf::open(dir.file("model.gguf", &synth.builder().build())).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    step_rows(&weights, precision)
}

#[test]
fn a_file_in_types_with_tensor_core_kernels_gets_the_wide_step() {
    assert_eq!(rows(GgmlType::Q8_0, Precision::Fast), Options::default().max_rows);
}

#[test]
fn a_file_in_types_without_one_gets_the_narrow_step() {
    assert_eq!(rows(GgmlType::F32, Precision::Fast), BATCHED_ROWS);
    assert!(BATCHED_ROWS < Options::default().max_rows);
}

#[test]
fn exact_products_multiply_with_the_tensor_cores_everywhere() {
    assert_eq!(rows(GgmlType::F32, Precision::Exact), Options::default().max_rows);
}
