//! The `ares` binary as an operator runs it, as far as that needs no GPU:
//! flags, the refusals of a start that cannot work, and a preset file served
//! before any model is loaded. `tests/real.rs` serves a model.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Output;
use std::time::Duration;

use ares_gguf::GgufBuilder;
use common::process::Server;
use serde_json::json;

mod common;

/// How long a server may take for anything before a test fails.
const LIMIT: Duration = Duration::from_secs(60);

fn ares(args: &[&str]) -> Output {
    common::process::ares().args(args).output().unwrap()
}

/// `ares` with `args`, on a port of its own, once it listens.
fn serve(args: &[&str]) -> Server {
    let mut command = common::process::ares();
    command.args(["--port", "0"]).args(args);
    Server::start(command, LIMIT)
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn scratch(name: &str) -> PathBuf {
    common::scratch(&format!("binary-{name}"))
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
    let mut server = serve(&["--models-preset", preset.to_str().unwrap(), "--max-tokens", "500"]);
    let list = server.get("/v1/models");
    let said = server.logged().join("\n");
    server.stop();

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
    assert_eq!(list.status, 200, "{}", list.body);
    let expected = json!([
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
    assert_eq!(list.json()["data"], expected, "{}", list.body);
}

#[test]
fn a_preset_file_that_cannot_be_served_is_refused_with_its_line() {
    let dir = scratch("refused");
    let model = dir.join("m.gguf");
    std::fs::write(&model, b"not a model").unwrap();
    let ini = dir.join("models.ini");
    let run = |text: &str, flags: &[&str]| {
        std::fs::write(&ini, text.replace("MODEL", model.to_str().unwrap())).unwrap();
        let mut args = vec!["--port", "0", "--models-preset", ini.to_str().unwrap()];
        args.extend(flags);
        let output = ares(&args);
        (output.status.code(), stderr(&output))
    };
    let (code, said) = run("[m]\nmodel = MODEL\nctx-size = 4096\n", &[]);
    assert_eq!(code, Some(2), "{said}");
    let wanted =
        format!("error: {}:3: Ares has no option 'ctx-size': the context is", ini.display());
    assert!(said.starts_with(&wanted), "{said}");
    assert!(!said.contains("listening"), "nothing is served: {said}");

    let (code, said) = run("[m]\nmodel = /nowhere/m.gguf\n", &[]);
    assert_eq!(code, Some(2));
    assert!(
        said.contains("models.ini:2: the model file '/nowhere/m.gguf' does not exist"),
        "{said}"
    );
    // A model on the command line and a preset file exclude each other.
    let (code, said) = run("[m]\nmodel = MODEL\n", &["--model", "x.gguf"]);
    assert_eq!(code, Some(2));
    assert!(said.contains("--model and --models-preset are both given"), "{said}");
    let (code, said) = run("[m]\nmodel = MODEL\n", &["--context", "none"]);
    assert_eq!(code, Some(2));
    assert!(said.contains("option '--context': 'none' is not a number of tokens"), "{said}");
}

/// A model of a preset file that fails to load is listed with its reason and
/// leaves the server serving; nothing loaded, the server stops in order.
#[test]
fn a_preset_file_is_served_with_a_failed_load_listed_and_stops_in_order() {
    let dir = scratch("served");
    let model = dir.join("garbage.gguf");
    std::fs::write(&model, b"this is not a model file at all").unwrap();
    let ini = dir.join("models.ini");
    let text =
        format!("[*]\nidle-unload = 10m\n[broken]\nmodel = {}\naliases = b\n", model.display());
    std::fs::write(&ini, text).unwrap();
    // The server listens at once: nothing is loaded before a request.
    let server = serve(&["--models-preset", ini.to_str().unwrap()]);
    let list = server.get("/v1/models").body;
    assert!(list.contains(r#""status":{"value":"unloaded"}"#), "{list}");
    let answer = server.post("/v1/completions", &json!({"model": "b", "prompt": "hi"}));
    assert_eq!(answer.status, 503, "{}", answer.body);
    let error = &answer.body;
    assert!(error.contains("model_load_failed") && error.contains("not a GGUF file"), "{error}");
    let list = server.get("/v1/models").body;
    assert!(list.contains(r#""value":"failed""#) && list.contains("not a GGUF file"), "{list}");
    server.stop();
}
