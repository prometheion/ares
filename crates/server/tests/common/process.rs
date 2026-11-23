//! The `ares` binary as the tests start it.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// The binary with the environment every test gives it: the user's cache
/// directory moved into the build's, so that no test reads or writes
/// `~/.cache/ares` (where Flash-Next keeps the usage profiles of its
/// experts), no `RUST_LOG` of the caller, and no input.
pub fn ares() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ares"));
    command.env("XDG_CACHE_HOME", cache_home()).env_remove("RUST_LOG").stdin(Stdio::null());
    command
}

/// The user's cache directory as the servers of the tests see it: one of the
/// build's, shared by the tests of one build directory.
fn cache_home() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cache-home")
}
