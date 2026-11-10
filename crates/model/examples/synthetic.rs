//! Writes the made-up model of the tests (`tests/common`) to a file, and a
//! prompt for it, so that llama.cpp can run the same weights.
//!
//! ```sh
//! cargo run --release -p ares-model --example synthetic -- OUT_DIR [matrix type: f32|q8_0]
//! ```
//!
//! The model is `Spec::default()`: eight blocks of width 64, three value heads
//! per key head, two query heads per key/value head. With `f32` matrices
//! llama.cpp quantizes no activations, so what is left between the two is
//! rounding.

#[path = "../tests/common/mod.rs"]
mod common;

use ares_core::GgmlType;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: synthetic OUT_DIR [f32|q8_0]");
    let matrix_type = match args.next().as_deref() {
        None | Some("f32") => GgmlType::F32,
        Some("q8_0") => GgmlType::Q8_0,
        Some(other) => panic!("unknown matrix type {other}"),
    };
    let synth = common::Synth::new(common::Spec { matrix_type, ..common::Spec::default() });
    let dir = std::path::Path::new(&dir);
    synth.builder().write(dir.join("synthetic.gguf")).unwrap();
    let prompt: Vec<String> = (0..48).map(|i| ((i * 37 + 11) % 128).to_string()).collect();
    std::fs::write(dir.join("prompt.tokens"), prompt.join(" ") + "\n").unwrap();
}
