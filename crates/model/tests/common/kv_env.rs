//! The K/V formats a GPU test runs with. By default those of
//! `Options::default()`; `ARES_TEST_KV=k,v` (or one name for both) and
//! `ARES_TEST_DRAFT_KV` change them, so that the same tests can be run with
//! the formats that keep exact cells (`q7` to `q4`):
//!
//! ```sh
//! ARES_TEST_KV=q5,q4 ARES_TEST_DRAFT_KV=q4 ARES_TEST_CUDA=1 cargo test --release ...
//! ```

#![allow(dead_code)]

use ares_cuda::CacheFormat;
use ares_model::qwen35::device::Options;

fn pair(var: &str, default: (CacheFormat, CacheFormat)) -> (CacheFormat, CacheFormat) {
    let name = |name: &str| {
        CacheFormat::ALL
            .into_iter()
            .find(|f| format!("{f:?}").eq_ignore_ascii_case(name.trim()))
            .unwrap_or_else(|| panic!("{var}: a K/V format (f16, q8 to q4), not {name}"))
    };
    match std::env::var(var) {
        Ok(value) => match value.split_once(',') {
            Some((k, v)) => (name(k), name(v)),
            None => (name(&value), name(&value)),
        },
        Err(_) => default,
    }
}

/// `options` with the formats of the environment.
pub fn with_env(options: Options) -> Options {
    Options {
        kv: pair("ARES_TEST_KV", options.kv),
        draft_kv: pair("ARES_TEST_DRAFT_KV", options.draft_kv),
        ..options
    }
}
