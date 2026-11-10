//! Prints what a GGUF model contains: metadata, tensor types and sizes.
//!
//! ```sh
//! cargo run --release -p ares-gguf --example inspect -- model.gguf [--tensors]
//! ```

use std::collections::BTreeMap;

use ares_core::units::Bytes;
use ares_gguf::Gguf;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: inspect <model.gguf> [--tensors]");
        std::process::exit(2);
    };
    let list_tensors = args.any(|a| a == "--tensors");

    let started = std::time::Instant::now();
    let gguf = match Gguf::open(&path) {
        Ok(gguf) => gguf,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    let opened = started.elapsed();

    println!(
        "GGUF v{}, {} part(s), {}, opened in {:.1} ms",
        gguf.version(),
        gguf.n_parts(),
        Bytes(gguf.file_size()),
        opened.as_secs_f64() * 1e3
    );
    println!("\nmetadata ({} keys):", gguf.metadata().count());
    for (key, value) in gguf.metadata() {
        let text = value.to_string();
        let text: String = if text.chars().count() > 100 {
            text.chars().take(100).chain("...".chars()).collect()
        } else {
            text
        };
        println!("  {key} = {text}");
    }

    let mut by_type: BTreeMap<&str, (usize, u64, u64)> = BTreeMap::new();
    for tensor in gguf.tensors() {
        let entry = by_type.entry(tensor.ty.name()).or_default();
        entry.0 += 1;
        entry.1 += tensor.size;
        entry.2 += tensor.dims.n_values().unwrap_or(0);
    }
    println!("\ntensors ({}):", gguf.tensors().len());
    for (name, (count, bytes, values)) in &by_type {
        println!(
            "  {name:>8}  {count:>5} tensors  {:>10}  {:.2} bits/value",
            Bytes(*bytes).to_string(),
            *bytes as f64 * 8.0 / *values as f64
        );
    }
    if list_tensors {
        println!();
        for tensor in gguf.tensors() {
            println!(
                "  {:<60} {:>8} {:>10}  {:?}",
                tensor.name,
                tensor.ty.name(),
                Bytes(tensor.size).to_string(),
                tensor.dims
            );
        }
    }
}
