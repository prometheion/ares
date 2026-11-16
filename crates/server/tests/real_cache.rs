//! The prompt cache in RAM and on SSD on real devices: what leaves the
//! devices is kept in RAM and comes back from there; `ares --cache <dir>`
//! writes what is worth keeping when the server stops or the model is
//! unloaded (and what leaves RAM), and a prompt that starts with it resumes
//! from the drive after a restart of the server, and after the model was
//! unloaded and loaded again in router mode, with the reply of the first run.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (a small `qwen35`
//! file, Qwen3.5-0.8B). `ARES_TEST_DEVICE` names the device of the small model
//! (default 0), `ARES_TEST_PORT` the first of the ports on 127.0.0.1 the tests
//! listen on (default 18970) and `ARES_TEST_CACHE_DIR` the directory the
//! stores are made in (default: the tests' directory under the build's). The
//! test over two devices needs `ARES_TEST_QWEN35_27B_SMALL` (Qwen3.8-27B
//! IQ2_XXS), which it splits over devices 0 and 1 with K and V in `q5`, the
//! formats that keep new pages in another form than old ones.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test real_cache -- --test-threads=1 --nocapture
//! ```
//!
//! The prompts are lists of facts and the replies copy from them, so that the
//! tokens follow with a wide margin: what is compared is whether the state
//! that came back is the state that was computed, not how two ways of
//! rounding break a near tie.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ares");

/// How long anything may take before a test fails instead of hanging.
const LIMIT: Duration = Duration::from_secs(300);

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
    let base = std::env::var("ARES_TEST_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(18970);
    base + offset
}

fn test_device() -> u32 {
    std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0)
}

/// A fresh directory for a test's store and files.
fn scratch(name: &str) -> PathBuf {
    let base = std::env::var_os("ARES_TEST_CACHE_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
    let dir = base.join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

/// What a completion said, and how much of its prompt came from the cache.
#[derive(Debug, PartialEq)]
struct Said {
    text: String,
    prompt_tokens: u64,
    cached_tokens: u64,
}

impl Server {
    /// Starts `ares` with `args` on `port` and waits until it listens.
    fn start(args: &[String], port: u16) -> Self {
        let mut child = Command::new(BIN)
            .args(args)
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

    fn request(&self, method: &str, path: &str, body: &str) -> (u16, String) {
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
        (status, String::from_utf8(body).expect("the body is valid UTF-8"))
    }

    /// The most probable continuation of `prompt` by `model`, `max_tokens`
    /// long at most.
    fn complete(&self, model: &str, prompt: &str, max_tokens: u32) -> Said {
        let body = json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stop": ["\n"],
        });
        let (status, body) = self.request("POST", "/v1/completions", &body.to_string());
        assert_eq!(status, 200, "{body}");
        let reply: Value = serde_json::from_str(&body).expect("JSON");
        let usage = &reply["usage"];
        Said {
            text: reply["choices"][0]["text"].as_str().expect("a text").to_owned(),
            prompt_tokens: usage["prompt_tokens"].as_u64().expect("prompt tokens"),
            cached_tokens: usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
        }
    }

    /// The value of a line of `GET /metrics`, for a server of one model.
    fn metric(&self, name: &str) -> u64 {
        let (status, body) = self.request("GET", "/metrics", "");
        assert_eq!(status, 200, "{body}");
        body.lines()
            .filter(|line| !line.starts_with('#'))
            .find_map(|line| {
                let (key, value) = line.rsplit_once(' ')?;
                key.ends_with(name).then(|| value.trim().parse::<f64>().ok())?
            })
            .unwrap_or_else(|| panic!("no metric {name} in:\n{body}")) as u64
    }

    /// Waits until nothing waits for the writer. Returns the pages written.
    fn settled(&self) -> u64 {
        let deadline = Instant::now() + LIMIT;
        loop {
            if self.metric("cache_ssd_pending") == 0 {
                return self.metric("cache_ssd_pages_written_total");
            }
            assert!(Instant::now() < deadline, "the store never came to rest");
            thread::sleep(Duration::from_millis(100));
        }
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
        log
    }
}

/// A list of `lines` facts that only `seed` decides, and a question about
/// one of them that the list answers.
fn facts(seed: u64, lines: u32) -> String {
    let mut text = String::from("The codes of the stations along the line:\n");
    for i in 0..lines {
        let code = (u64::from(i) * 7919 + seed * 104_729) % 9000 + 1000;
        text.push_str(&format!("Station {} has the code {code}.\n", i + 1));
    }
    text
}

fn code_of(seed: u64, station: u32) -> u64 {
    (u64::from(station - 1) * 7919 + seed * 104_729) % 9000 + 1000
}

fn question(station: u32) -> String {
    format!(
        "Question: which code does station {station} have?\nAnswer: Station {station} has the code"
    )
}

fn args(model: &Path, rest: &[&str]) -> Vec<String> {
    let mut args = vec!["--model".to_owned(), model.display().to_string()];
    args.extend(rest.iter().map(|arg| (*arg).to_owned()));
    args
}

/// The two turns of a conversation over `seed`'s facts: the first prompt,
/// and the second, which goes on from the first reply.
fn turns(server: &Server, model: &str, seed: u64, lines: u32) -> (String, Said, String, Said) {
    let first = format!("{}{}", facts(seed, lines), question(7));
    let one = server.complete(model, &first, 12);
    assert!(one.text.contains(&code_of(seed, 7).to_string()), "{one:?}");
    let second = format!("{first}{}\n{}", one.text, question(lines - 3));
    let two = server.complete(model, &second, 12);
    assert!(two.text.contains(&code_of(seed, lines - 3).to_string()), "{two:?}");
    (first, one, second, two)
}

fn name_of(model: &Path) -> String {
    model.file_stem().unwrap().to_string_lossy().into_owned()
}

/// One model on its devices with `flags`: a conversation is computed, which
/// RAM and the devices hold, so nothing is written; the server stops, which
/// writes it, and the second turn resumes from the drive with the reply of
/// the first run; then other conversations take the pages of the pool, and
/// the first one comes back from RAM, still without a write.
fn resumes_after_a_restart_and_after_eviction(model: &Path, flags: &[&str], tag: &str, at: u16) {
    let dir = scratch(tag);
    let cache = dir.display().to_string();
    let name = name_of(model);
    let lines = 330;
    let with_cache: Vec<&str> = flags.iter().copied().chain(["--cache", &cache]).collect();

    // What a server without the store says: the reference for every reply.
    let plain = Server::start(&args(model, flags), port(at));
    let (_, _, second, reference) = turns(&plain, &name, 1, lines);
    assert_eq!(plain.metric("cache_ssd_pages_written_total"), 0);
    plain.stop();

    let server = Server::start(&args(model, &with_cache), port(at));
    let (_, one, _, two) = turns(&server, &name, 1, lines);
    assert_eq!(one.cached_tokens, 0);
    assert!(two.cached_tokens > 2048, "the second turn starts from the first: {two:?}");
    assert_eq!(two.text, reference.text);
    // Nothing is written while the devices and RAM hold the conversation.
    assert_eq!(server.settled(), 0);
    assert_eq!(server.metric("cache_ssd_written_bytes_total"), 0);
    assert_eq!(server.metric("cache_ssd_pages_read_total"), 0);
    let log = server.stop();
    assert!(log.iter().any(|line| line.contains("prompt cache on SSD")), "{log:#?}");

    // A new process: the devices hold nothing of the conversation, and the
    // drive what the last one wrote when it stopped.
    let mut server = Server::start(&args(model, &with_cache), port(at));
    let stored = server.metric("cache_ssd_bytes");
    assert!(stored >= u64::from(lines) * 8 / 128 * 4096, "{stored} bytes on the drive");
    eprintln!("{tag}: {} prompt tokens, {stored} bytes written at the stop", two.prompt_tokens);
    let began = Instant::now();
    let again = server.complete(&name, &second, 12);
    let took = began.elapsed();
    assert_eq!(again.text, reference.text, "the reply after a restart");
    assert!(again.cached_tokens > 2048, "{again:?}");
    let read = server.metric("cache_ssd_pages_read_total");
    assert!(read > 0 && server.metric("cache_ssd_checkpoints_read_total") >= 1);
    eprintln!(
        "{tag}: after a restart {} of {} prompt tokens came from the drive ({read} pages, {} \
         bytes) and the reply took {took:.2?}",
        again.cached_tokens,
        again.prompt_tokens,
        server.metric("cache_ssd_read_bytes_total")
    );

    // Other conversations until the pool has given up the first one's pages,
    // which RAM keeps: they come back from there, and nothing is written.
    for seed in 2..6 {
        turns(&server, &name, seed, lines);
    }
    let before = server.metric("cache_ram_pages_read_total");
    let back = server.complete(&name, &second, 12);
    assert_eq!(back.text, reference.text, "the reply after eviction");
    assert!(back.cached_tokens > 2048, "{back:?}");
    let read = server.metric("cache_ram_pages_read_total") - before;
    assert!(read > 0, "the pages came back from RAM: {:#?}", server.logged());
    assert!(server.metric("cache_ram_pages_in_total") > 0);
    assert_eq!(server.settled(), 0, "nothing was written while RAM held it");
    eprintln!("{tag}: after eviction {read} pages came back from RAM");
    server.stop();
}

#[test]
fn a_conversation_resumes_from_ssd_after_a_restart_and_after_eviction() {
    let Some(small) = file("ARES_TEST_QWEN35_MODEL", "a small qwen35 model file") else { return };
    let device = test_device().to_string();
    // A context that holds two of the conversations and not five.
    let flags = ["--devices", &device, "--context", "10240", "--parallel", "2"];
    resumes_after_a_restart_and_after_eviction(&small, &flags, "cache-small", 0);
}

#[test]
fn pages_in_two_forms_on_two_devices_resume_from_ssd() {
    let Some(model) = file("ARES_TEST_QWEN35_27B_SMALL", "a qwen35 file two devices split") else {
        return;
    };
    let flags = [
        "--devices",
        "0,1",
        "--split",
        "tensor",
        "--context",
        "10240",
        "--parallel",
        "2",
        "--kv",
        "q5",
    ];
    resumes_after_a_restart_and_after_eviction(&model, &flags, "cache-two-forms", 1);
}

#[test]
fn a_model_that_was_unloaded_resumes_from_ssd_when_it_is_loaded_again() {
    let Some(small) = file("ARES_TEST_QWEN35_MODEL", "a small qwen35 model file") else { return };
    let dir = scratch("cache-router");
    let ini = dir.join("models.ini");
    let text = format!(
        "[*]\ndevices = {}\ncontext = 10240\nram = 4G\ncache = {}\ncache-size = 2G\n\n\
         [talker]\nmodel = {}\nidle-unload = 3\n",
        test_device(),
        dir.join("store").display(),
        small.display(),
    );
    std::fs::write(&ini, text).unwrap();
    let preset = ["--models-preset".to_owned(), ini.display().to_string()];
    let mut server = Server::start(&preset, port(2));
    let (_, _, second, two) = turns(&server, "talker", 1, 330);
    assert!(two.cached_tokens > 2048, "{two:?}");

    // Idle for its time, the model is unloaded, and its caches with it.
    let state = |server: &Server| {
        let (_, body) = server.request("GET", "/v1/models", "");
        let list: Value = serde_json::from_str(&body).expect("JSON");
        list["data"][0]["status"]["value"].as_str().expect("a state").to_owned()
    };
    let deadline = Instant::now() + LIMIT;
    while state(&server) != "unloaded" {
        assert!(Instant::now() < deadline, "the model was never unloaded: {:#?}", server.logged());
        thread::sleep(Duration::from_millis(100));
    }

    // The next request loads it again; the conversation comes from the drive.
    let again = server.complete("talker", &second, 12);
    assert_eq!(again.text, two.text, "the reply after an unload and a reload");
    assert!(again.cached_tokens > 2048, "{again:?}");
    let log = server.stop();
    let stores = log.iter().filter(|line| line.contains("prompt cache on SSD")).count();
    assert_eq!(stores, 2, "the store was opened by both loads: {log:#?}");
}
