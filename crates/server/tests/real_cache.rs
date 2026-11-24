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
//! formats that keep new pages in another form than old ones. The test of
//! Qwen3.8-Flash-Next needs `ARES_TEST_QWEN4EXP_MODEL` (the first of its two
//! files) and both devices.
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

use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use common::fresh;
use common::process::{Client, Server, ares};
use serde_json::json;

mod common;

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
    common::process::port(18970, offset)
}

fn test_device() -> u32 {
    std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0)
}

/// A fresh directory for a test's store and files.
fn scratch(name: &str) -> PathBuf {
    let base = std::env::var_os("ARES_TEST_CACHE_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
    fresh(base.join(name))
}

/// `ares` with `args`, started on `port`.
fn serve(args: &[String], port: u16) -> Server {
    let mut command = ares();
    command.args(args).args(["--port", &port.to_string()]);
    Server::start(command, LIMIT)
}

/// What a completion said, how much of its prompt came from the cache, and
/// the milliseconds from the request's arrival to its first token.
#[derive(Debug)]
struct Said {
    text: String,
    prompt_tokens: u64,
    cached_tokens: u64,
    first_ms: f64,
}

/// The most probable continuation of `prompt` by `model`, `max_tokens` long
/// at most.
fn complete(server: &Client, model: &str, prompt: &str, max_tokens: u32) -> Said {
    let body = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stop": ["\n"],
    });
    let reply = server.post("/v1/completions", &body);
    assert_eq!(reply.status, 200, "{}", reply.body);
    let reply = reply.json();
    let usage = &reply["usage"];
    Said {
        text: reply["choices"][0]["text"].as_str().expect("a text").to_owned(),
        prompt_tokens: usage["prompt_tokens"].as_u64().expect("prompt tokens"),
        cached_tokens: usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
        first_ms: reply["timings"]["prompt_ms"].as_f64().unwrap_or(f64::NAN),
    }
}

/// Waits until nothing waits for the writer. Returns the pages written.
fn settled(server: &Client) -> u64 {
    let deadline = Instant::now() + LIMIT;
    loop {
        if server.metric("cache_ssd_pending") == 0 {
            return server.metric("cache_ssd_pages_written_total");
        }
        assert!(Instant::now() < deadline, "the store never came to rest");
        thread::sleep(Duration::from_millis(100));
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
fn turns(server: &Client, model: &str, seed: u64, lines: u32) -> (String, Said, String, Said) {
    let first = format!("{}{}", facts(seed, lines), question(7));
    let one = complete(server, model, &first, 12);
    assert!(one.text.contains(&code_of(seed, 7).to_string()), "{one:?}");
    let second = format!("{first}{}\n{}", one.text, question(lines - 3));
    let two = complete(server, model, &second, 12);
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
    let plain = serve(&args(model, flags), port(at));
    let (_, _, second, reference) = turns(&plain, &name, 1, lines);
    assert_eq!(plain.metric("cache_ssd_pages_written_total"), 0);
    plain.stop();

    let server = serve(&args(model, &with_cache), port(at));
    let (_, one, _, two) = turns(&server, &name, 1, lines);
    assert_eq!(one.cached_tokens, 0);
    assert!(two.cached_tokens > 2048, "the second turn starts from the first: {two:?}");
    assert_eq!(two.text, reference.text);
    // Nothing is written while the devices and RAM hold the conversation.
    assert_eq!(settled(&server), 0);
    assert_eq!(server.metric("cache_ssd_written_bytes_total"), 0);
    assert_eq!(server.metric("cache_ssd_pages_read_total"), 0);
    let log = server.stop();
    assert!(log.iter().any(|line| line.contains("prompt cache on SSD")), "{log:#?}");

    // A new process: the devices hold nothing of the conversation, and the
    // drive what the last one wrote when it stopped.
    let mut server = serve(&args(model, &with_cache), port(at));
    let stored = server.metric("cache_ssd_bytes");
    assert!(stored >= u64::from(lines) * 8 / 128 * 4096, "{stored} bytes on the drive");
    eprintln!("{tag}: {} prompt tokens, {stored} bytes written at the stop", two.prompt_tokens);
    let began = Instant::now();
    let again = complete(&server, &name, &second, 12);
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
    let back = complete(&server, &name, &second, 12);
    assert_eq!(back.text, reference.text, "the reply after eviction");
    assert!(back.cached_tokens > 2048, "{back:?}");
    let read = server.metric("cache_ram_pages_read_total") - before;
    assert!(read > 0, "the pages came back from RAM: {:#?}", server.logged());
    assert!(server.metric("cache_ram_pages_in_total") > 0);
    assert_eq!(settled(&server), 0, "nothing was written while RAM held it");
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
    let mut server = serve(&preset, port(2));
    let (_, _, second, two) = turns(&server, "talker", 1, 330);
    assert!(two.cached_tokens > 2048, "{two:?}");

    // Idle for its time, the model is unloaded, and its caches with it.
    server.until("talker", "unloaded");

    // The next request loads it again; the conversation comes from the drive.
    let again = complete(&server, "talker", &second, 12);
    assert_eq!(again.text, two.text, "the reply after an unload and a reload");
    assert!(again.cached_tokens > 2048, "{again:?}");
    let log = server.stop();
    let stores = log.iter().filter(|line| line.contains("prompt cache on SSD")).count();
    assert_eq!(stores, 2, "the store was opened by both loads: {log:#?}");
}

/// Qwen3.8-Flash-Next on two devices with a RAM limit of 8 GiB, most of which
/// the tier of experts takes: a conversation of about 8k tokens is computed
/// from nothing, goes on from the cards, comes back after other conversations
/// took the pool (from RAM where the tier leaves the prompt cache room, from
/// the drive otherwise), and after a restart from the drive. Every reply of
/// the second turn is the one a server that never saw the conversation gives
/// (a list of facts: the tokens follow with a wide margin). The time to the
/// first token of each is reported.
#[test]
fn flash_next_resumes_a_conversation_from_the_cards_from_ram_and_from_ssd() {
    let Some(model) = file("ARES_TEST_QWEN4EXP_MODEL", "the first file of Qwen3.8-Flash-Next")
    else {
        return;
    };
    let dir = scratch("cache-flash");
    let cache = dir.display().to_string();
    let name = name_of(&model);
    // About 8k tokens of facts; a context that holds two such conversations.
    let lines = 660;
    let flags = ["--devices", "0,1", "--ram", "8G", "--context", "20480", "--parallel", "2"];
    let with_cache: Vec<&str> = flags.iter().copied().chain(["--cache", &cache]).collect();
    let first = format!("{}{}", facts(1, lines), question(7));
    let report = |what: &str, said: &Said| {
        eprintln!(
            "RES flash {what}: {} of {} prompt tokens from the cache, first token after {:.0} ms",
            said.cached_tokens, said.prompt_tokens, said.first_ms
        );
    };

    // A server that never saw the conversation: the second turn computed
    // from nothing is the reference.
    let plain = serve(&args(&model, &flags), port(3));
    let one = complete(&plain, &name, &first, 12);
    assert!(one.text.contains(&code_of(1, 7).to_string()), "{one:?}");
    plain.stop();
    let second = format!("{first}{}\n{}", one.text, question(lines - 3));
    let plain = serve(&args(&model, &flags), port(3));
    let cold = complete(&plain, &name, &second, 12);
    assert!(cold.text.contains(&code_of(1, lines - 3).to_string()), "{cold:?}");
    assert_eq!(cold.cached_tokens, 0);
    report("cold", &cold);
    plain.stop();

    let mut server = serve(&args(&model, &with_cache), port(3));
    let plan = server.logged().iter().find(|line| line.contains("plan: RAM")).cloned();
    eprintln!("{plan:?}");
    let again = complete(&server, &name, &first, 12);
    assert_eq!(again.text, one.text, "the first turn");
    let replayed = server.metric("replayed_rows_total");
    let cards = complete(&server, &name, &second, 12);
    let replayed = server.metric("replayed_rows_total") - replayed;
    report(&format!("from the cards ({replayed} rows replayed)"), &cards);
    assert_eq!(cards.text, cold.text, "the second turn from the cards");
    assert!(cards.cached_tokens > 4096, "{cards:?}");
    // It goes on from the first turn's anchor (the last page boundary of its prompt), which a
    // prompt read layer by layer saves inside its pass: at most a page and the reply are
    // replayed, not the rows from an earlier checkpoint on.
    assert!(replayed < 256, "{replayed} rows replayed for the second turn");

    // Two other conversations take the pool.
    for seed in 2..4 {
        let other = format!("{}{}", facts(seed, lines), question(5));
        let said = complete(&server, &name, &other, 12);
        assert!(said.text.contains(&code_of(seed, 5).to_string()), "{said:?}");
    }
    let (ram_before, ssd_before) =
        (server.metric("cache_ram_pages_read_total"), server.metric("cache_ssd_pages_read_total"));
    let back = complete(&server, &name, &second, 12);
    let from_ram = server.metric("cache_ram_pages_read_total") - ram_before;
    let from_ssd = server.metric("cache_ssd_pages_read_total") - ssd_before;
    report(&format!("after eviction ({from_ram} pages from RAM, {from_ssd} from SSD)"), &back);
    eprintln!(
        "RES flash: checkpoints in RAM {} bytes, pages in RAM {} bytes",
        server.metric("checkpoint_ram_bytes"),
        server.metric("cache_ram_bytes")
    );
    assert_eq!(back.text, cold.text, "the second turn after the pool was taken");
    assert!(back.cached_tokens > 4096, "{back:?}");
    server.stop();

    // A new process: the drive has what the last one wrote when it stopped.
    let server = serve(&args(&model, &with_cache), port(3));
    let restart = complete(&server, &name, &second, 12);
    report("after a restart", &restart);
    assert_eq!(restart.text, cold.text, "the second turn after a restart");
    assert!(restart.cached_tokens > 4096, "{restart:?}");
    assert!(server.metric("cache_ssd_checkpoints_read_total") >= 1);
    server.stop();
}
