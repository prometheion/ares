//! The `ares` binary as an operator runs it, as far as that needs no GPU:
//! flags, and the refusals of a start that cannot work. `tests/real.rs`
//! serves a model.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

use ares_gguf::GgufBuilder;

const BIN: &str = env!("CARGO_BIN_EXE_ares");

fn ares(args: &[&str]) -> Output {
    Command::new(BIN).args(args).env_remove("RUST_LOG").stdin(Stdio::null()).output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("binary-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A model file of an architecture nobody implements.
fn toy_model(dir: &str) -> PathBuf {
    let path = scratch(dir).join("toy.gguf");
    GgufBuilder::new()
        .str("general.architecture", "toy")
        .tensor_f32("w", &[4], &[1.0, 2.0, 3.0, 4.0])
        .write(&path)
        .unwrap();
    path
}

#[test]
fn help_prints_every_option_and_exits_cleanly() {
    let output = ares(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for flag in [
        "--host",
        "--port",
        "--model",
        "--devices",
        "--split",
        "--shares",
        "--context",
        "--parallel",
        "--kv",
        "--precision",
        "--drafters",
        "--window",
        "--draft-vocab",
        "--chat-var",
        "--reasoning",
        "--reasoning-format",
        "--reasoning-effort",
        "--reasoning-preserve",
        "--help",
    ] {
        assert!(stdout.contains(flag), "{flag} is documented:\n{stdout}");
    }
    assert!(output.stderr.is_empty(), "no log noise for --help");
}

#[test]
fn a_mistyped_flag_is_named_and_exits_with_2() {
    let output = ares(&["--prot", "80"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = stderr(&output);
    assert!(stderr.contains("unknown option '--prot'") && stderr.contains("--help"), "{stderr}");
    assert!(output.stdout.is_empty());

    let output = ares(&["--model", "m.gguf", "--split", "diagonal"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(self::stderr(&output).contains("'diagonal' is not single, tensor or layer"));
}

#[test]
fn without_a_model_there_is_nothing_to_serve() {
    let output = ares(&["--port", "0"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = stderr(&output);
    assert!(stderr.contains("nothing to serve") && stderr.contains("--model"), "{stderr}");
}

#[test]
fn a_model_that_cannot_be_opened_stops_the_start_with_the_reason() {
    let dir = scratch("bad-model");
    let missing = dir.join("missing.gguf");
    let output = ares(&["--port", "0", "--model", missing.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = stderr(&output);
    assert!(
        stderr.contains("cannot open the model") && stderr.contains("missing.gguf"),
        "{stderr}"
    );

    let garbage = dir.join("garbage.gguf");
    std::fs::write(&garbage, b"this is not a model file at all").unwrap();
    let output = ares(&["--port", "0", "--model", garbage.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(self::stderr(&output).contains("not a GGUF file"));
}

#[test]
fn an_architecture_without_an_implementation_is_refused_by_name() {
    let path = toy_model("toy-model");
    let output = ares(&["--port", "0", "--model", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = stderr(&output);
    // The file is described first, which proves it was readable.
    let opened =
        stderr.lines().find(|l| l.contains("opened the model file")).expect("the model is logged");
    assert!(opened.contains(r#"architecture="toy""#), "{opened}");
    assert!(opened.contains("tensors=1") && opened.contains("toy.gguf"), "{opened}");
    assert!(
        stderr.contains("the model architecture 'toy' has no implementation on the devices"),
        "{stderr}"
    );
    assert!(stderr.contains("this build runs qwen35"), "{stderr}");
    assert!(!stderr.contains("listening"), "nothing is served: {stderr}");

    let nameless = scratch("nameless-model").join("nameless.gguf");
    GgufBuilder::new().tensor_f32("w", &[1], &[1.0]).write(&nameless).unwrap();
    let output = ares(&["--port", "0", "--model", nameless.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(self::stderr(&output).contains("declares no architecture"));
}

#[test]
fn a_port_in_use_is_reported_before_the_model_is_looked_at() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port().to_string();
    let path = toy_model("port-in-use");
    let output = ares(&["--port", &port, "--model", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = stderr(&output);
    assert!(stderr.contains("cannot listen on 127.0.0.1"), "{stderr}");
    assert!(!stderr.contains("opened the model file"), "{stderr}");
}
