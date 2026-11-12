//! Writes the made-up `qwen4exp` model of `ares_model::qwen4exp::synthetic`
//! and a token sequence for it, for the comparison with the llama.cpp fork
//! (`tools/q4capture/README.md`).
//!
//! ```sh
//! cargo run --release -p ares-model --example q4synth -- OUT_DIR [f32|shipped|quantized] [SEED] [TOKENS]
//! ```
//!
//! writes `OUT_DIR/synthetic-<types>.gguf` and `OUT_DIR/prompt.tokens` (ids
//! separated by blanks). The defaults are `f32`, seed 1 and 48 tokens.

use std::path::PathBuf;

use ares_model::qwen4exp::synthetic::{Synthetic, Types};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first().map(PathBuf::from) else {
        return Err("usage: q4synth OUT_DIR [f32|shipped|quantized] [SEED] [TOKENS]".into());
    };
    let kind = args.get(1).map_or("f32", String::as_str);
    let types = match kind {
        "f32" => Types::F32,
        "shipped" => Types::Shipped,
        "quantized" => Types::Quantized,
        other => return Err(format!("unknown tensor types '{other}'").into()),
    };
    let seed: u64 = args.get(2).map_or(Ok(1), |s| s.parse())?;
    let n: usize = args.get(3).map_or(Ok(48), |s| s.parse())?;

    std::fs::create_dir_all(&out)?;
    let model = Synthetic::small(seed).types(types);
    let path = out.join(format!("synthetic-{kind}.gguf"));
    model.write(&path)?;
    let tokens = model.shape.tokens(n, seed);
    let text: Vec<String> = tokens.iter().map(u32::to_string).collect();
    std::fs::write(out.join("prompt.tokens"), text.join(" ") + "\n")?;
    println!("{} and {n} tokens (seed {seed})", path.display());
    Ok(())
}
