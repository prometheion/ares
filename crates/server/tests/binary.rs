//! The `ares` binary as an operator runs it: flags, startup log, a model
//! file, signals. Binds 127.0.0.1:0 only.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use ares_gguf::GgufBuilder;

const BIN: &str = env!("CARGO_BIN_EXE_ares");

fn ares(args: &[&str]) -> Command {
    let mut command = Command::new(BIN);
    command.args(args).env_remove("RUST_LOG").stdin(Stdio::null());
    command
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("binary-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn help_prints_the_usage_and_exits_cleanly() {
    let output = ares(&["--help"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for flag in ["--host", "--port", "--model", "--help"] {
        assert!(stdout.contains(flag), "{flag} is documented:\n{stdout}");
    }
    assert!(output.stderr.is_empty(), "no log noise for --help");
}

#[test]
fn a_mistyped_flag_is_named_and_exits_with_2() {
    let output = ares(&["--prot", "80"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("unknown option '--prot'") && stderr.contains("--help"), "{stderr}");
    assert!(output.stdout.is_empty());
}

#[test]
fn a_model_that_cannot_be_opened_stops_the_start_with_the_reason() {
    let dir = scratch("bad-model");
    let missing = dir.join("missing.gguf");
    let output = ares(&["--port", "0", "--model", missing.to_str().unwrap()]).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("cannot open the model") && stderr.contains("missing.gguf"),
        "{stderr}"
    );

    let garbage = dir.join("garbage.gguf");
    std::fs::write(&garbage, b"this is not a model file at all").unwrap();
    let output = ares(&["--port", "0", "--model", garbage.to_str().unwrap()]).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8(output.stderr).unwrap().contains("not a GGUF file"));
}

/// A running server process. Killed when dropped so a failing test leaves nothing behind.
struct Running {
    child: Child,
    lines: mpsc::Receiver<String>,
    log: Vec<String>,
    address: String,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Running {
    fn start(args: &[&str]) -> Self {
        let mut child = ares(args).stderr(Stdio::piped()).spawn().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        // The reader keeps draining after startup so that the child can never
        // block on a full pipe.
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut running = Self { child, lines, log: Vec::new(), address: String::new() };
        let line = running.wait_for_log("listening");
        let address = line.split("address=").nth(1).expect("the log names the address");
        running.address = address.split_whitespace().next().unwrap().to_owned();
        running
    }

    /// Reads log lines until one contains `needle`.
    fn wait_for_log(&mut self, needle: &str) -> String {
        loop {
            let line = self.lines.recv_timeout(Duration::from_secs(10)).unwrap_or_else(|e| {
                panic!("no log line with {needle:?} ({e}); got {:#?}", self.log)
            });
            self.log.push(line.clone());
            if line.contains(needle) {
                return line;
            }
        }
    }

    fn request(&self, method: &str, path: &str, body: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(&self.address).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut reply = String::new();
        stream.read_to_string(&mut reply).unwrap();
        let status = reply.split(' ').nth(1).unwrap().parse().unwrap();
        (status, reply.split("\r\n\r\n").nth(1).unwrap_or("").to_owned())
    }

    fn signal(&self, name: &str) {
        let status = Command::new("kill")
            .args([&format!("-{name}"), &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn wait_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit after the signal");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

#[test]
fn starts_without_a_model_answers_health_and_refuses_completions_with_501() {
    let mut server = Running::start(&["--host", "127.0.0.1", "--port", "0"]);

    assert_eq!(server.request("GET", "/health", ""), (200, r#"{"status":"ok"}"#.to_owned()));
    let (status, models) = server.request("GET", "/v1/models", "");
    assert_eq!(status, 200);
    assert!(models.contains(r#""data":[]"#), "{models}");

    let (status, error) = server.request(
        "POST",
        "/v1/chat/completions",
        r#"{"model":"any","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(status, 501);
    assert!(error.contains("no model architecture is implemented yet"), "{error}");

    server.signal("TERM");
    let status = server.wait_exit();
    assert!(status.success(), "SIGTERM is a clean stop: {status:?}");
}

#[test]
fn refuses_decisions_with_501_like_completions_and_says_so_at_startup() {
    let mut server = Running::start(&["--host", "127.0.0.1", "--port", "0"]);
    // The warning follows the line `start` waited for.
    server.wait_for_log("completions and decisions answer 501");

    // As llama.cpp's clients send it: no `model`.
    let body = r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"x"}}}"#;
    let (status, error) = server.request("POST", "/v1/systemone", body);
    assert_eq!(status, 501);
    assert!(error.contains("no model architecture is implemented yet"), "{error}");

    server.signal("TERM");
    assert!(server.wait_exit().success());
}

#[test]
fn opens_the_model_file_logs_what_it_is_and_names_its_architecture_in_the_refusal() {
    let dir = scratch("toy-model");
    let path = dir.join("toy.gguf");
    GgufBuilder::new()
        .str("general.architecture", "toy")
        .tensor_f32("w", &[4], &[1.0, 2.0, 3.0, 4.0])
        .write(&path)
        .unwrap();

    let mut server = Running::start(&["--port", "0", "--model", path.to_str().unwrap()]);
    let opened = server
        .log
        .iter()
        .find(|l| l.contains("opened the model file"))
        .expect("the model is logged");
    assert!(opened.contains(r#"architecture="toy""#), "{opened}");
    assert!(opened.contains("tensors=1"), "{opened}");
    assert!(opened.contains("size="), "{opened}");
    assert!(opened.contains("toy.gguf"), "{opened}");

    let (status, error) =
        server.request("POST", "/v1/completions", r#"{"model":"toy","prompt":"x"}"#);
    assert_eq!(status, 501);
    assert!(error.contains("'toy' is not implemented yet"), "{error}");

    server.signal("INT");
    assert!(server.wait_exit().success(), "SIGINT is a clean stop too");
}

#[test]
fn a_port_in_use_is_reported_and_exits_with_1() {
    let first = Running::start(&["--port", "0"]);
    let port = first.address.rsplit(':').next().unwrap().to_owned();
    let output = ares(&["--port", &port]).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("cannot listen on 127.0.0.1"), "{stderr}");
}
