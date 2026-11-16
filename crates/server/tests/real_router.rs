//! Router mode on real devices: `ares --models-preset` loads models when
//! requests name them, keeps those loaded that fit side by side, unloads one
//! for another that needs its room and one that was idle for its time, and
//! gives the memory back, by what the devices report.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (a small `qwen35`
//! file, Qwen3.5-0.8B). `ARES_TEST_DEVICE` names the device of the small
//! models (default 0) and `ARES_TEST_PORT` the first of the ports on
//! 127.0.0.1 the tests listen on (default 18951). The test that replaces a
//! model needs two devices, `ARES_TEST_QWEN35_27B` (a file both devices
//! hold together and one alone does not, Qwen3.8-27B Q4_K_S) and
//! `ARES_TEST_QWEN35_27B_SMALL` (a file one device holds that leaves the two
//! too little room for the other, Qwen3.8-27B IQ2_XXS).
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test real_router -- --test-threads=1 --nocapture
//! ```
//!
//! The times of the loads and unloads are printed, as the server's log has
//! them. Time here is the machine's: the binary has no other clock, so the
//! idle time is a few seconds and the tests wait for a state, not for a
//! duration (`tests/models.rs` has the state machine on a clock it sets).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use ares_cuda::{Device, Wait};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ares");

/// How long anything may take before a test fails instead of hanging: the
/// load of the large file over a busy machine included.
const LIMIT: Duration = Duration::from_secs(300);

/// What a device may report differently before and after for reasons that
/// are not a model: the granularity of its allocations.
const SLACK: u64 = 48 << 20;

fn file(variable: &str, what: &str) -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    let path = std::env::var_os(variable).map(PathBuf::from);
    if path.is_none() {
        eprintln!("skipped: set {variable} to {what}");
    }
    path
}

fn port(offset: u16) -> u16 {
    let base = std::env::var("ARES_TEST_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(18951);
    base + offset
}

fn test_device() -> u32 {
    std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0)
}

/// Free memory of a device as its driver reports it, to every process alike.
fn free(device: &Device) -> u64 {
    device.mem_info().expect("the device reports its memory").free
}

struct Server {
    child: Child,
    lines: mpsc::Receiver<String>,
    log: Vec<String>,
    address: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        // A failing test leaves nothing on the devices.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Reply {
    status: u16,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }
}

fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(end) = raw.windows(2).position(|w| w == b"\r\n") {
        let size = std::str::from_utf8(&raw[..end])
            .ok()
            .and_then(|hex| usize::from_str_radix(hex.trim(), 16).ok())
            .unwrap_or(0);
        let start = end + 2;
        if size == 0 || raw.len() < start + size {
            break;
        }
        out.extend_from_slice(&raw[start..start + size]);
        raw = &raw[(start + size + 2).min(raw.len())..];
    }
    out
}

impl Server {
    /// Starts `ares` over the preset file `ini` and waits until it listens.
    fn start(ini: &Path, port: u16) -> Self {
        let mut child = Command::new(BIN)
            .arg("--models-preset")
            .arg(ini)
            .args(["--port", &port.to_string()])
            .env_remove("RUST_LOG")
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        // The reader keeps draining, so that the child never blocks on a full pipe.
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut server =
            Self { child, lines, log: Vec::new(), address: format!("127.0.0.1:{port}") };
        server.wait_for_log("listening");
        server
    }

    fn wait_for_log(&mut self, needle: &str) -> String {
        loop {
            let line = self
                .lines
                .recv_timeout(LIMIT)
                .unwrap_or_else(|e| panic!("no log line with {needle:?} ({e}): {:#?}", self.log));
            self.log.push(line.clone());
            if line.contains(needle) {
                return line;
            }
        }
    }

    fn logged(&mut self) -> &[String] {
        while let Ok(line) = self.lines.try_recv() {
            self.log.push(line);
        }
        &self.log
    }

    fn request(&self, method: &str, path: &str, body: &str) -> Reply {
        let mut stream = TcpStream::connect(&self.address).expect("the server accepts");
        stream.set_read_timeout(Some(LIMIT)).unwrap();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).expect("an answer");
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("a complete header");
        let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
        let status = head.split(' ').nth(1).and_then(|s| s.parse().ok()).expect("a status line");
        let body = &raw[split + 4..];
        let body =
            if head.contains("transfer-encoding: chunked") { dechunk(body) } else { body.to_vec() };
        Reply { status, body: String::from_utf8(body).expect("the body is valid UTF-8") }
    }

    /// A short completion by `model`, which is loaded for it if it is not.
    fn complete(&self, model: &str) -> Reply {
        let body = json!({ "model": model, "prompt": "The capital of France is", "max_tokens": 8 });
        self.request("POST", "/v1/completions", &body.to_string())
    }

    /// The state of every model, by id.
    fn states(&self) -> Vec<(String, String)> {
        let list = self.request("GET", "/v1/models", "").json();
        let data = list["data"].as_array().expect("a list").iter();
        data.map(|m| {
            let text = |v: &Value| v.as_str().expect("a string").to_owned();
            (text(&m["id"]), text(&m["status"]["value"]))
        })
        .collect()
    }

    fn state(&self, model: &str) -> String {
        let states = self.states();
        states.into_iter().find(|(id, _)| id == model).expect("the model is listed").1
    }

    /// Waits until `model` is in `state`.
    fn until(&mut self, model: &str, state: &str) {
        let deadline = Instant::now() + LIMIT;
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
    fn resident(&self) -> u64 {
        let status = std::fs::read_to_string(format!("/proc/{}/status", self.child.id())).unwrap();
        let line = status.lines().find_map(|line| line.strip_prefix("VmRSS:")).expect("VmRSS");
        line.trim().trim_end_matches("kB").trim().parse::<u64>().unwrap() * 1024
    }

    /// Stops the server with one signal and checks that it left in order.
    fn stop(mut self) -> Vec<String> {
        let pid = self.child.id().to_string();
        assert!(Command::new("kill").args(["-TERM", &pid]).status().unwrap().success());
        let deadline = Instant::now() + LIMIT;
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "the process did not exit: {:#?}", self.logged());
            thread::sleep(Duration::from_millis(10));
        };
        let log = self.logged().to_vec();
        assert!(status.success(), "a clean stop: {status:?}\n{log:#?}");
        assert!(log.last().is_some_and(|line| line.trim_end().ends_with("stopped")), "{log:#?}");
        log
    }
}

/// Prints the lines of the log that say how long loads and unloads took.
fn print_times(what: &str, log: &[String]) {
    for line in log {
        let said = ["loaded in ", "the engine has stopped in ", "released of the "];
        if said.iter().any(|needle| line.contains(needle)) {
            eprintln!("{what}: {}", line.split("INFO").last().unwrap_or(line).trim());
        }
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn two_small_models_share_a_card_and_an_idle_one_gives_its_memory_back() {
    let Some(small) = file("ARES_TEST_QWEN35_MODEL", "a small qwen35 model file") else { return };
    let ordinal = test_device();
    let dir = scratch("router-small");
    let garbage = dir.join("garbage.gguf");
    std::fs::write(&garbage, b"this is not a model file at all").unwrap();
    let ini = dir.join("models.ini");
    let text = format!(
        "[*]\ndevices = {ordinal}\ncontext = 8192\nprecision = exact\nram = 4G\n\n\
         [stays]\nmodel = {small}\naliases = first\nram = 1G\n\n\
         [idles]\nmodel = {small}\nram = 1G\nidle-unload = 3\n\n\
         [broken]\nmodel = {garbage}\n",
        small = small.display(),
        garbage = garbage.display(),
    );
    std::fs::write(&ini, text).unwrap();

    let device = Device::open(ordinal, Wait::Block).expect("the test device opens");
    let before = free(&device);
    let mut server = Server::start(&ini, port(0));
    let all = |state: &str| -> Vec<(String, String)> {
        ["stays", "idles", "broken"].iter().map(|id| ((*id).to_owned(), state.to_owned())).collect()
    };
    assert_eq!(server.states(), all("unloaded"));
    assert!(free(&device) + SLACK > before, "a server that has loaded nothing holds no device");

    // The first request loads its model, by an alias.
    let reply = server.complete("first");
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.json()["model"], "stays");
    assert_eq!(server.state("stays"), "loaded");
    assert_eq!(server.state("idles"), "unloaded");
    let with_one = free(&device);
    assert!(with_one + SLACK < before, "the model is on the device");

    // The second model fits next to it: both are loaded, on one card.
    let reply = server.complete("idles");
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(server.state("stays"), "loaded", "nothing made way: there was room");
    assert_eq!(server.state("idles"), "loaded");
    let with_two = free(&device);
    assert!(with_two + SLACK < with_one, "the second model is on the device as well");
    let resident_two = server.resident();

    // A model whose file is not one fails, says why, and leaves the others serving.
    let reply = server.complete("broken");
    assert_eq!(reply.status, 503, "{}", reply.body);
    let error = reply.json();
    assert_eq!(error["error"]["code"], "model_load_failed");
    assert!(error["error"]["message"].as_str().unwrap().contains("not a GGUF file"), "{error}");
    let list = server.request("GET", "/v1/models", "").json();
    assert_eq!(list["data"][2]["status"]["value"], "failed");
    assert!(list["data"][2]["status"]["reason"].as_str().unwrap().contains("not a GGUF file"));
    assert_eq!(server.complete("stays").status, 200);

    // The idle one is unloaded after its time, and what it held is back.
    server.until("idles", "unloaded");
    assert_eq!(server.state("stays"), "loaded");
    let after_idle = free(&device);
    assert!(
        after_idle.abs_diff(with_one) < SLACK,
        "the device has free what it had with one model: {after_idle} against {with_one} \
         ({with_two} with two)"
    );
    let resident_one = server.resident();
    assert!(
        resident_one < resident_two,
        "the process holds less RAM: {resident_one} against {resident_two}"
    );
    let log = server.logged().join("\n");
    assert!(log.contains("unloading (idle for"), "{log}");
    assert!(!log.contains("did not come back"), "{log}");

    // The next request loads it again.
    let reply = server.complete("idles");
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(server.state("idles"), "loaded");

    // The endpoint unloads the other one, which says so in the events' words.
    let reply = server.request("POST", "/models/unload", r#"{"model": "stays"}"#);
    assert_eq!((reply.status, reply.body.as_str()), (200, r#"{"success":true}"#));
    server.until("stays", "unloaded");
    let reply = server.request("POST", "/models/load", r#"{"model": "stays"}"#);
    assert_eq!(reply.status, 200);
    server.until("stays", "loaded");

    // A clean stop unloads everything.
    let log = server.stop();
    print_times("0.8B", &log);
    let released = log.iter().filter(|line| line.contains("threads joined, devices released"));
    assert!(released.count() >= 4, "every load was unloaded: {log:#?}");
    let after = free(&device);
    assert!(after.abs_diff(before) < SLACK, "everything is back: {after} against {before}");
}

#[test]
fn a_model_that_needs_both_cards_takes_the_place_of_an_idle_one() {
    let Some(large) = file("ARES_TEST_QWEN35_27B", "a qwen35 file that needs two devices") else {
        return;
    };
    let Some(medium) =
        file("ARES_TEST_QWEN35_27B_SMALL", "a qwen35 file that fills most of one device")
    else {
        return;
    };
    if ares_cuda::devices().map_or(0, |devices| devices.len()) < 2 {
        eprintln!("skipped: needs two devices");
        return;
    }
    let ini = scratch("router-replace").join("models.ini");
    // The first is held to one device; the second takes what it needs.
    let text = format!(
        "[one-card]\nmodel = {}\ndevices = 0\ncontext = 8192\n\n[both-cards]\nmodel = {}\n",
        medium.display(),
        large.display()
    );
    std::fs::write(&ini, text).unwrap();
    let devices = [0, 1].map(|ordinal| Device::open(ordinal, Wait::Block).expect("a device"));
    let before = devices.each_ref().map(free);
    let mut server = Server::start(&ini, port(1));

    let reply = server.complete("one-card");
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(server.state("one-card"), "loaded");
    let with_one = devices.each_ref().map(free);
    assert!(with_one[0] + SLACK < before[0], "the first model is on device 0");
    assert!(with_one[1] + SLACK > before[1], "and not on device 1");

    // The second does not fit next to it, and the first is idle: it makes way.
    let reply = server.complete("both-cards");
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(server.state("one-card"), "unloaded");
    assert_eq!(server.state("both-cards"), "loaded");
    let with_both = devices.each_ref().map(free);
    assert!(with_both[1] + SLACK < before[1], "the second model is on device 1 as well");
    let log = server.logged().join("\n");
    assert!(log.contains("unloading (to make room for both-cards)"), "{log}");
    assert!(!log.contains("did not come back"), "{log}");

    // And the other way round: the first comes back when a request names it.
    let reply = server.complete("one-card");
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(server.state("one-card"), "loaded");
    assert_eq!(server.state("both-cards"), "unloaded");
    let again = devices.each_ref().map(free);
    assert!(again[1] + SLACK > before[1], "device 1 is free again: {again:?} against {before:?}");

    let log = server.stop();
    print_times("27B", &log);
    let after = devices.each_ref().map(free);
    for (after, before) in after.iter().zip(before) {
        assert!(after.abs_diff(before) < SLACK, "everything is back: {after} against {before}");
    }
}

/// Flash-Next in router mode: loaded, unloaded and loaded again in one
/// process, then made to give way to the 27B (both need both cards) and
/// loaded once more. Every load locks the tier of experts in a new range of
/// addresses, often where the last one was: a lock that outlived its model
/// would make the next load fail (`cuMemHostRegister`). Needs
/// `ARES_TEST_QWEN4EXP_MODEL` (the first part of Flash-Next),
/// `ARES_TEST_QWEN35_27B` and two devices; `ARES_TEST_FLASH_RAM` is the limit
/// of Flash-Next's RAM (default `8G`).
#[test]
fn flash_next_is_loaded_again_after_an_unload_and_after_the_27b() {
    let Some(flash) = file("ARES_TEST_QWEN4EXP_MODEL", "the first part of Flash-Next") else {
        return;
    };
    let Some(large) = file("ARES_TEST_QWEN35_27B", "a qwen35 file that needs two devices") else {
        return;
    };
    if ares_cuda::devices().map_or(0, |devices| devices.len()) < 2 {
        eprintln!("skipped: needs two devices");
        return;
    }
    let ram = std::env::var("ARES_TEST_FLASH_RAM").unwrap_or_else(|_| "8G".to_owned());
    let ini = scratch("router-flash").join("models.ini");
    let text = format!(
        "[*]\ndevices = 0,1\ncontext = 8192\n\n[flash]\nmodel = {}\nram = {ram}\n\n\
         [both-cards]\nmodel = {}\n",
        flash.display(),
        large.display()
    );
    std::fs::write(&ini, text).unwrap();
    let mut server = Server::start(&ini, port(2));
    let resident_before = server.resident();
    let answer = |server: &Server, model: &str| {
        let reply = server.complete(model);
        assert_eq!(reply.status, 200, "{model}: {}", reply.body);
        assert_eq!(server.state(model), "loaded");
    };

    answer(&server, "flash");
    let resident_loaded = server.resident();
    let reply = server.request("POST", "/models/unload", r#"{"model": "flash"}"#);
    assert_eq!(reply.status, 200, "{}", reply.body);
    server.until("flash", "unloaded");
    let resident_unloaded = server.resident();
    eprintln!(
        "RES resident: {} MiB before, {} with Flash-Next, {} after its unload",
        resident_before >> 20,
        resident_loaded >> 20,
        resident_unloaded >> 20
    );
    assert!(resident_unloaded < resident_loaded, "the unload gave the tier back");
    answer(&server, "flash");

    // The 27B does not fit next to it: Flash-Next makes way, and comes back.
    answer(&server, "both-cards");
    assert_eq!(server.state("flash"), "unloaded");
    answer(&server, "flash");
    assert_eq!(server.state("both-cards"), "unloaded");

    let log = server.stop();
    print_times("router Flash-Next", &log);
    let text = log.join("\n");
    assert!(!text.contains("cuMemHostRegister"), "no lock of the tier failed: {text}");
    assert!(!text.contains("did not come back"), "{text}");
    let loads = log.iter().filter(|line| line.contains("threads joined, devices released"));
    assert!(loads.count() >= 4, "every load was unloaded: {log:#?}");
}
