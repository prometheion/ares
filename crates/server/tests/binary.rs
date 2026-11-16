//! The `ares` binary as an operator runs it, as far as that needs no GPU:
//! flags, and the refusals of a start that cannot work. `tests/real.rs`
//! serves a model.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
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
        "--temperature",
        "--top-k",
        "--top-p",
        "--min-p",
        "--typical-p",
        "--top-n-sigma",
        "--xtc-probability",
        "--xtc-threshold",
        "--dynatemp-range",
        "--dynatemp-exponent",
        "--repeat-penalty",
        "--presence-penalty",
        "--frequency-penalty",
        "--repeat-last-n",
        "--dry-multiplier",
        "--dry-base",
        "--dry-allowed-length",
        "--dry-penalty-last-n",
        "--dry-sequence-breakers",
        "--mirostat",
        "--mirostat-tau",
        "--mirostat-eta",
        "--samplers",
        "--seed",
        "--max-tokens",
        "--max-tokens-limit",
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

#[test]
fn the_defaults_of_requests_are_checked_before_anything_is_started() {
    let output = ares(&["--model", "m.gguf", "--top-p", "1.5"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stderr(&output).contains("option '--top-p': '1.5' is not a number above 0 and at most 1"),
        "{}",
        stderr(&output)
    );
    let output = ares(&["--model", "m.gguf", "--max-tokens", "9", "--max-tokens-limit", "8"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("max-tokens = 9 is more than max-tokens-limit = 8"));

    // A preset file: the file, the line and the reason, and the name Ares
    // has for an option of llama-server.
    let dir = scratch("defaults-refused");
    let model = toy_model("defaults-refused-model");
    let preset = dir.join("models.ini");
    for (keys, line, reason) in [
        ("temperature = 1.0\ntop-p = 2\n", 4, "top-p = '2' is not a number above 0"),
        ("temp = 1.0\n", 3, "Ares has no option 'temp': the temperature of a request"),
        ("n-predict = 96000\n", 3, "Ares has no option 'n-predict': the limit of tokens"),
        ("max-tokens = 9\nmax-tokens-limit = 8\n", 4, "[toy]: max-tokens = 9 is more than"),
    ] {
        std::fs::write(&preset, format!("[toy]\nmodel = {}\n{keys}", model.display())).unwrap();
        let output = ares(&["--port", "0", "--models-preset", preset.to_str().unwrap()]);
        assert_eq!(output.status.code(), Some(2), "{keys}");
        let stderr = stderr(&output);
        let at = format!("{}:{line}: ", preset.display());
        assert!(stderr.contains(&at) && stderr.contains(reason), "{keys}: {stderr}");
        assert!(!stderr.contains("listening"), "{stderr}");
    }
}

/// A server of a preset file loads no model before a request names one, so
/// it starts without a device: its log and its model list are looked at, and
/// it is stopped.
#[test]
fn a_preset_file_is_served_with_its_defaults_in_the_log_and_in_the_model_list() {
    let dir = scratch("defaults-served");
    let model = toy_model("defaults-served-model");
    let preset = dir.join("models.ini");
    let text = format!(
        "[*]\ntemperature = 1.0\ntop-p = 0.95\ntop-k = 20\nmax-tokens = 96000\n\
         [toy]\nmodel = {0}\npresence-penalty = 0.0\n\
         [short]\nmodel = {0}\ntemperature = 0.7\nmax-tokens-limit = 512\n\
         [plain]\nmodel = {0}\n",
        model.display()
    );
    std::fs::write(&preset, text).unwrap();
    let mut child = Command::new(BIN)
        .args(["--port", "0", "--models-preset", preset.to_str().unwrap()])
        .args(["--max-tokens", "500"])
        .env_remove("RUST_LOG")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // The log up to the line that says the server listens, which has the port.
    let mut log = Vec::new();
    for line in BufReader::new(child.stderr.take().unwrap()).lines() {
        let line = line.unwrap();
        let listening = line.contains("listening");
        log.push(line);
        if listening {
            break;
        }
    }
    let said = log.join("\n");
    let listed = log.last().and_then(|line| {
        let address = line.split("address=").nth(1)?.split_whitespace().next()?;
        let mut stream = TcpStream::connect(address).ok()?;
        stream
            .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .ok()?;
        let mut answer = String::new();
        stream.read_to_string(&mut answer).ok()?;
        Some(answer)
    });
    child.kill().unwrap();
    child.wait().unwrap();

    // The command line's option holds over the file's for every model.
    for line in [
        "a field a request leaves out: temperature=1.0 top_k=20 top_p=0.95 presence_penalty=0.0 \
         max_tokens=500 model=\"toy\"",
        "a field a request leaves out: temperature=0.7 top_k=20 top_p=0.95 max_tokens=500; no \
         reply is longer than 512 tokens model=\"short\"",
    ] {
        assert!(said.contains(line), "{line}:\n{said}");
    }
    assert_eq!(said.matches("a field a request leaves out").count(), 3, "once a model:\n{said}");
    let listed = listed.unwrap_or_else(|| panic!("the server did not answer:\n{said}"));
    assert!(listed.starts_with("HTTP/1.1 200"), "{listed}");
    let body = listed.split_once("\r\n\r\n").expect("a body").1;
    let list: serde_json::Value = serde_json::from_str(body).expect("JSON");
    let json = serde_json::json!([
        {
            "id": "toy", "object": "model", "owned_by": "ares", "aliases": [],
            "status": {"value": "unloaded"},
            "defaults": {"temperature": 1.0, "top_k": 20, "top_p": 0.95,
                         "presence_penalty": 0.0, "max_tokens": 500}
        },
        {
            "id": "short", "object": "model", "owned_by": "ares", "aliases": [],
            "status": {"value": "unloaded"},
            "defaults": {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "max_tokens": 500},
            "max_tokens_limit": 512
        },
        {
            "id": "plain", "object": "model", "owned_by": "ares", "aliases": [],
            "status": {"value": "unloaded"},
            "defaults": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "max_tokens": 500}
        }
    ]);
    assert_eq!(list["data"], json, "{body}");
}
