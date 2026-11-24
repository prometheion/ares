//! The `ares` binary as the tests start it, and the client of its port.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::socket::{Reply, request_bytes};

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

/// The port `offset` of a suite whose ports start at `ARES_TEST_PORT`, or at
/// `base` when that is not set.
pub fn port(base: u16, offset: u16) -> u16 {
    let base = std::env::var("ARES_TEST_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(base);
    base + offset
}

/// What the suites that serve a real model ask it.
pub const GERMAN: &str = "Erkl\u{e4}re in drei S\u{e4}tzen, warum der Himmel blau ist.";
pub const ENGLISH: &str = "Explain in three sentences why the sky is blue.";

/// How long anything of a server of one small model may take before a test
/// fails instead of hanging.
pub const LIMIT: Duration = Duration::from_secs(60);

/// `ares --model <model>` started on `port` with exactly `flags`; nothing it
/// does may take longer than `limit`.
pub fn serve_model(model: &Path, port: u16, flags: &[&str], limit: Duration) -> Server {
    let mut command = ares();
    command.arg("--model").arg(model).args(["--port", &port.to_string()]).args(flags);
    Server::start(command, limit)
}

/// [`serve_model`] as the suites of one model start it: on the test device,
/// with exact products, so that a text does not depend on what else is in a
/// step, and a context of 8192 tokens, unless `flags` name `--devices`,
/// `--precision` or `--context`; a minute for anything ([`LIMIT`]).
pub fn serve(model: &Path, port: u16, flags: &[&str]) -> Server {
    let device = super::test_device().to_string();
    let mut all: Vec<&str> = Vec::new();
    if !flags.contains(&"--devices") {
        all.extend(["--devices", device.as_str()]);
    }
    if !flags.contains(&"--precision") {
        all.extend(["--precision", "exact"]);
    }
    if !flags.contains(&"--context") {
        all.extend(["--context", "8192"]);
    }
    all.extend(flags);
    serve_model(model, port, &all, LIMIT)
}

/// Polls until `condition` holds; fails the test after [`LIMIT`].
pub fn eventually(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + LIMIT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(20));
    }
}

/// A running `ares`. Killed when dropped, so that a failing test leaves
/// nothing on the devices.
pub struct Server {
    child: Child,
    lines: mpsc::Receiver<String>,
    log: Vec<String>,
    client: Client,
}

/// Where a server listens and what it serves: all a request needs, and
/// something threads can share.
pub struct Client {
    pub address: String,
    /// The first model the server lists: the one it serves, when it serves one.
    pub model: String,
    /// How long the server may take for anything before a test fails.
    pub limit: Duration,
}

impl Deref for Server {
    type Target = Client;

    fn deref(&self) -> &Client {
        &self.client
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    /// Starts `command` (from [`ares`], with its flags) and waits until it
    /// listens, at the address its log names; nothing it does may take longer
    /// than `limit`.
    pub fn start(command: Command, limit: Duration) -> Self {
        let mut server = Self::spawn(command, limit);
        let listening = server.wait_for_log("listening");
        server.client.address = listening
            .split("address=")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .unwrap_or_else(|| panic!("the log names the address: {listening}"))
            .to_owned();
        let models = server.get("/v1/models").json();
        server.client.model = models["data"][0]["id"].as_str().unwrap_or_default().to_owned();
        server
    }

    /// Starts `command` and returns at once: for a start that is to fail.
    pub fn spawn(mut command: Command, limit: Duration) -> Self {
        let mut child = command.stdout(Stdio::null()).stderr(Stdio::piped()).spawn().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        // The reader keeps draining after the start, so that the child can
        // never block on a full pipe.
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let client = Client { address: String::new(), model: String::new(), limit };
        Self { child, lines, log: Vec::new(), client }
    }

    /// Reads log lines until one contains `needle`.
    pub fn wait_for_log(&mut self, needle: &str) -> String {
        loop {
            let line = self
                .lines
                .recv_timeout(self.limit)
                .unwrap_or_else(|e| panic!("no log line with {needle:?} ({e}): {:#?}", self.log));
            self.log.push(line.clone());
            if line.contains(needle) {
                return line;
            }
        }
    }

    /// Everything the server has logged so far.
    pub fn logged(&mut self) -> &[String] {
        while let Ok(line) = self.lines.try_recv() {
            self.log.push(line);
        }
        &self.log
    }

    pub fn signal(&self, name: &str) {
        let status = Command::new("kill")
            .args([&format!("-{name}"), &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// The exit status, once the process has ended.
    pub fn exited(&mut self) -> Option<ExitStatus> {
        self.child.try_wait().unwrap()
    }

    /// Waits for the process to end, and for the last of its log.
    pub fn wait_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + self.limit;
        loop {
            if let Some(status) = self.exited() {
                // The reader ends with the pipe.
                while let Ok(line) = self.lines.recv_timeout(self.limit) {
                    self.log.push(line);
                }
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit: {:#?}", self.logged());
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Stops the server with one signal and checks that it left in order: a
    /// clean exit after everything it loaded was released. Returns its log.
    pub fn stop(mut self) -> Vec<String> {
        self.signal("TERM");
        let status = self.wait_exit();
        let log = std::mem::take(&mut self.log);
        assert!(status.success(), "a clean stop: {status:?}\n{log:#?}");
        assert!(log.last().is_some_and(|line| line.trim_end().ends_with("stopped")), "{log:#?}");
        log
    }

    /// Waits until `model`, of a server of presets, is in `state`.
    pub fn until(&mut self, model: &str, state: &str) {
        let deadline = Instant::now() + self.limit;
        while self.state(model) != state {
            assert!(
                Instant::now() < deadline,
                "{model} did not become {state}: {:#?}",
                self.logged()
            );
            thread::sleep(Duration::from_millis(50));
        }
    }

    /// The bytes of the server's process that are in RAM.
    pub fn resident(&self) -> u64 {
        let status = std::fs::read_to_string(format!("/proc/{}/status", self.child.id())).unwrap();
        let line = status.lines().find_map(|line| line.strip_prefix("VmRSS:")).expect("VmRSS");
        line.trim().trim_end_matches("kB").trim().parse::<u64>().unwrap() * 1024
    }
}

impl Client {
    pub fn connect(&self) -> std::io::Result<TcpStream> {
        let stream = TcpStream::connect(&self.address)?;
        stream.set_read_timeout(Some(self.limit))?;
        Ok(stream)
    }

    /// Sends a request and leaves the answer unread.
    pub fn send(&self, method: &str, path: &str, body: &str) -> TcpStream {
        let mut stream = self.connect().expect("the server accepts connections");
        stream.write_all(&request_bytes(method, path, body)).unwrap();
        stream
    }

    pub fn request(&self, method: &str, path: &str, body: &str) -> Reply {
        let mut raw = Vec::new();
        self.send(method, path, body).read_to_end(&mut raw).unwrap();
        Reply::parse(&raw)
    }

    pub fn get(&self, path: &str) -> Reply {
        self.request("GET", path, "")
    }

    pub fn post(&self, path: &str, body: &Value) -> Reply {
        self.request("POST", path, &body.to_string())
    }

    /// A chat request body for the server's model: `extra` on top of one
    /// user message.
    pub fn chat(&self, question: &str, extra: Value) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
        });
        for (key, value) in extra.as_object().expect("an object") {
            body[key] = value.clone();
        }
        body
    }

    /// One value of `GET /metrics`, of a server with one model loaded.
    pub fn metric(&self, name: &str) -> u64 {
        let reply = self.get("/metrics");
        assert_eq!(reply.status, 200, "{}", reply.body);
        let value = reply
            .body
            .lines()
            .find_map(|line| line.strip_prefix(&format!("ares_{name} ")))
            .unwrap_or_else(|| panic!("no metric {name}: {}", reply.body));
        value.trim().parse::<f64>().unwrap_or_else(|e| panic!("{name} = {value}: {e}")) as u64
    }

    /// The state of every model of a server of presets, by id.
    pub fn states(&self) -> Vec<(String, String)> {
        let list = self.get("/v1/models").json();
        let data = list["data"].as_array().expect("a list").iter();
        data.map(|m| {
            let text = |v: &Value| v.as_str().expect("a string").to_owned();
            (text(&m["id"]), text(&m["status"]["value"]))
        })
        .collect()
    }

    pub fn state(&self, model: &str) -> String {
        let states = self.states();
        states.into_iter().find(|(id, _)| id == model).expect("the model is listed").1
    }
}
