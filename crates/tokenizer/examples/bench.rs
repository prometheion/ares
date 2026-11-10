//! Measures how fast a model's tokenizer encodes and decodes a text file.
//!
//! ```text
//! cargo run --release -p ares-tokenizer --example bench -- MODEL.gguf TEXT_FILE [REPEATS]
//! ```
//!
//! Prints the load time, the encode and decode speed in MB/s of text (best and
//! median of the repeats) and the number of tokens. To compare with llama.cpp,
//! run `llama-tokenize -m MODEL.gguf -f TEXT_FILE --no-escape --ids --log-disable`
//! and time it; it also loads the vocabulary, so subtract that: its `--show-count`
//! run on a tiny file gives the load time.

use std::time::Instant;

use ares_gguf::Gguf;
use ares_tokenizer::{EncodeOptions, Tokenizer};

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(model), Some(file)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: bench MODEL.gguf TEXT_FILE [REPEATS]");
        std::process::exit(2);
    };
    let repeats: usize = args.get(3).map_or(15, |s| s.parse().expect("REPEATS is a number"));
    let text = std::fs::read(file).expect("read the text file");
    let mb = text.len() as f64 / 1e6;

    let gguf = Gguf::open(model).expect("open the model");
    let start = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("build the tokenizer");
    println!(
        "load: {:.1} ms ({} tokens)",
        start.elapsed().as_secs_f64() * 1e3,
        tokenizer.n_tokens()
    );

    let options = EncodeOptions { add_special: false, parse_special: true };
    // The encoder keeps a cache of words in each thread. A prompt that arrives
    // on a thread that has not seen text yet meets it empty: "cold". The same
    // thread encoding the next prompt finds it filled: "warm".
    let mut ids = Vec::new();
    let mut cold = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        ids = std::thread::scope(|scope| {
            scope.spawn(|| tokenizer.encode(&text, options)).join().unwrap()
        });
        cold.push(start.elapsed().as_secs_f64());
    }
    let mut warm = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        ids = tokenizer.encode(&text, options);
        warm.push(start.elapsed().as_secs_f64());
    }
    let best = |times: &[f64]| times.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "encode: {} bytes -> {} tokens ({:.2} bytes per token)",
        text.len(),
        ids.len(),
        text.len() as f64 / ids.len() as f64
    );
    println!(
        "  cold cache (new thread): best {:.1} MB/s, median {:.1} MB/s",
        mb / best(&cold),
        mb / median(&mut cold)
    );
    println!(
        "  warm cache (same thread): best {:.1} MB/s, median {:.1} MB/s",
        mb / best(&warm),
        mb / median(&mut warm)
    );

    let serial = ids.clone();
    for threads in [2, 4, 8] {
        let mut times = Vec::new();
        for _ in 0..repeats {
            let start = Instant::now();
            ids = tokenizer.encode_parallel(&text, options, threads);
            times.push(start.elapsed().as_secs_f64());
        }
        assert_eq!(ids, serial, "{threads} threads give the same tokens");
        println!(
            "  {threads} threads: best {:.1} MB/s, median {:.1} MB/s",
            mb / best(&times),
            mb / median(&mut times)
        );
    }

    let mut decode = Vec::new();
    let mut bytes = Vec::new();
    for _ in 0..repeats {
        let start = Instant::now();
        bytes = tokenizer.decode(&ids, true).expect("decode");
        decode.push(start.elapsed().as_secs_f64());
    }
    let best = decode.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "decode: {} tokens -> {} bytes; best {:.1} MB/s, median {:.1} MB/s ({:.1} M tokens/s best)",
        ids.len(),
        bytes.len(),
        mb / best,
        mb / median(&mut decode),
        ids.len() as f64 / best / 1e6
    );
    // Malformed UTF-8 decodes to U+FFFD, so only a valid text comes back as it went in.
    assert!(
        bytes == text || std::str::from_utf8(&text).is_err(),
        "decoding gives a UTF-8 text back"
    );
}
