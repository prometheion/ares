//! The `ares` binary serving Qwen3.8-Flash-Next over real sockets: it answers
//! in both placements and on one card, drafts with its MTP block, and starts
//! from the usage profile of its experts.
//!
//! Needs `ARES_TEST_CUDA=1`, `ARES_TEST_FLASH_NEXT` (the first part of its
//! split file) and two devices (`ARES_TEST_DEVICES`, default `0,1`); the test
//! of drafts needs its MTP block, `ARES_TEST_FLASH_NEXT_MTP`. `ARES_TEST_FLASH_RAM`
//! is the RAM limit of its servers (default `8G`), `ARES_TEST_PORT` the first
//! of the ports on 127.0.0.1 the tests listen on (default 18951). The tests are
//! slow: a server loads in about 15 seconds, and the first answer comes from
//! experts that are still being fetched.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_FLASH_NEXT=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
//!     cargo test --release -p ares-server --test real_flash_next -- --test-threads=1 --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use common::model_file;
use common::process::{Client, ENGLISH, GERMAN, Server, eventually, serve_model};
use common::socket::{Reply, read_events};
use serde_json::json;

mod common;

fn port(offset: u16) -> u16 {
    common::process::port(18951, offset)
}

/// How long anything may take with the real Flash-Next: a server loads in
/// about 15 seconds, and an answer from experts that are still being fetched
/// can take a minute.
const SLOW: Duration = Duration::from_secs(900);

/// The first part of the split file of Qwen3.8-Flash-Next, or `None` (and a
/// note) when the tests of it are not enabled.
fn flash_model() -> Option<PathBuf> {
    model_file("ARES_TEST_FLASH_NEXT", "the first part of Flash-Next")
}

/// The RAM limit of the servers of Flash-Next (`ARES_TEST_FLASH_RAM`, default
/// `8G`). Without a limit the tier of experts takes what the machine has
/// available less a small headroom, which is the server's default and not
/// what a test that shares the machine with other work should do.
fn flash_ram() -> String {
    std::env::var("ARES_TEST_FLASH_RAM").unwrap_or_else(|_| "8G".to_owned())
}

/// A server on two devices (`ARES_TEST_DEVICES`, default `0,1`) that spread
/// the model as `split` says.
fn flash_server(model: &Path, port: u16, split: &str) -> Server {
    let devices = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0,1".to_owned());
    let ram = flash_ram();
    let chat = "enable_thinking=false";
    serve_model(
        model,
        port,
        &["--devices", &devices, "--split", split, "--ram", &ram, "--chat-var", chat],
        SLOW,
    )
}

/// The text of a chat answer, checked to be one.
fn answer(reply: &Reply) -> String {
    assert_eq!(reply.status, 200, "{}", reply.body);
    reply.json()["choices"][0]["message"]["content"].as_str().expect("a text").to_owned()
}

/// Whether `text` says one of `words`, in any case.
fn mentions(text: &str, words: &[&str]) -> bool {
    let text = text.to_lowercase();
    words.iter().any(|w| text.contains(w))
}

const HASH_TABLE: &str = "Explain in three sentences how a hash table works.";
const GERMAN_WORDS: &[&str] = &["licht", "streu", "rayleigh", "wellenl", "atmosph"];
const ENGLISH_WORDS: &[&str] = &["light", "scatter", "rayleigh", "wavelength", "atmosphere"];

/// Flash-Next through the binary on both devices, in both placements: it
/// answers in German and in English, two requests at once and a request that
/// is cancelled while it streams leave the server able to answer, and the
/// counters say how the experts were found. Both placements need two devices:
/// with one named (`ARES_TEST_DEVICES`) the test is skipped.
#[test]
fn flash_next_answers_in_both_placements() {
    let Some(model) = flash_model() else { return };
    if std::env::var("ARES_TEST_DEVICES").is_ok_and(|d| !d.contains(',')) {
        eprintln!("skipped: both placements need two devices, ARES_TEST_DEVICES names one");
        return;
    }
    for (offset, split, placement) in
        [(0, "tensor", "both devices on every layer"), (1, "layer", "layers 0..")]
    {
        let mut server = flash_server(&model, port(offset), split);
        let log = server.logged().join("\n");
        assert!(log.contains(placement), "{split}: the plan says where the layers are: {log}");
        assert!(log.contains("plan: RAM"), "{split}: the plan says where the RAM goes: {log}");
        let client: &Client = &server;
        assert_eq!(client.get("/health").body, r#"{"status":"ok"}"#);

        let ask = |question: &str| {
            let body = client.chat(question, json!({"max_tokens": 80, "temperature": 0}));
            client.post("/v1/chat/completions", &body)
        };
        let german = ask(GERMAN);
        let text = answer(&german);
        eprintln!("RES {split}: German answer {text:?}");
        assert!(mentions(&text, GERMAN_WORDS), "{split}: a German answer about light: {text:?}");
        let timings = &german.json()["timings"];
        assert!(timings["predicted_n"].as_u64().unwrap_or(0) >= 10, "{split}: {timings}");
        let text = answer(&ask(ENGLISH));
        eprintln!("RES {split}: English answer {text:?}");
        assert!(mentions(&text, ENGLISH_WORDS), "{split}: an English answer about light: {text:?}");
        // And the answer to another question is the CPU reference's, as far
        // as its first sentence (the reference's greedy tokens are in
        // `/var/tmp/ares-qwen4exp-trace/english.txt`).
        let text = answer(&ask(HASH_TABLE));
        assert!(
            text.starts_with(
                "A hash table stores data by mapping keys to specific indices in an array using a \
                 hash function."
            ),
            "{split}: the reference's first sentence: {text:?}"
        );
        // A request that turns thinking on gets the reasoning apart from the
        // content, and neither has the tags that divide them.
        let body = client.chat(
            ENGLISH,
            json!({
                "max_tokens": 160,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": true},
            }),
        );
        let reply = client.post("/v1/chat/completions", &body);
        assert_eq!(reply.status, 200, "{}", reply.body);
        let message = &reply.json()["choices"][0]["message"];
        let reasoning = message["reasoning_content"].as_str().unwrap_or("");
        let content = message["content"].as_str().unwrap_or("");
        eprintln!("RES {split}: reasoning {reasoning:?}, content {content:?}");
        assert!(reasoning.len() > 40, "{split}: the thinking is the reasoning: {message}");
        assert!(content.len() > 40, "{split}: and the answer is the content: {message}");
        assert!(!reasoning.contains("think>") && !content.contains("think>"), "{split}: {message}");

        // Two requests at once share the steps.
        let together: Vec<String> = thread::scope(|scope| {
            let handles: Vec<_> = [GERMAN, ENGLISH]
                .into_iter()
                .map(|question| scope.spawn(move || answer(&ask(question))))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(mentions(&together[0], GERMAN_WORDS), "{split}: {:?}", together[0]);
        assert!(mentions(&together[1], ENGLISH_WORDS), "{split}: {:?}", together[1]);

        // A client that goes away in the middle of an answer frees its place.
        let body = client.chat(ENGLISH, json!({"max_tokens": 400, "stream": true}));
        let mut stream = client.send("POST", "/v1/chat/completions", &body.to_string());
        read_events(&mut stream, 3);
        drop(stream);
        eventually("the cancelled request to be counted", || {
            client.metric("requests_cancelled_total") == 1
        });
        let text = answer(&ask(GERMAN));
        assert!(mentions(&text, GERMAN_WORDS), "{split}: after a cancelled request: {text:?}");

        // How the experts were found.
        let (picks, hits) =
            (client.metric("expert_picks_total"), client.metric("expert_hits_total"));
        let misses = client.metric("expert_misses_ram_total")
            + client.metric("expert_misses_disk_total")
            + client.metric("expert_misses_on_their_way_total");
        eprintln!(
            "RES {split}: {picks} picks, {hits} in a slot, {misses} not; {} computed on the CPU, {} uploaded",
            client.metric("expert_cpu_total"),
            client.metric("expert_uploads_total"),
        );
        assert!(picks > 1000 && hits > 0 && hits <= picks, "{split}: {picks} picks, {hits} hits");
        assert_eq!(picks, hits + misses, "{split}: every pick was a hit or a miss");
        assert!(client.metric("steps_total") > 100);
        server.stop();
    }
}

/// Flash-Next with its MTP block (`ARES_TEST_FLASH_NEXT_MTP` names the
/// block's file; without it the test is skipped): the plan says that the
/// block drafts and what it took, the answer to the question the reference
/// answered starts as the reference's does, drafts are verified and some are
/// accepted, and a request that says no drafts gets none.
#[test]
fn flash_next_drafts_with_its_mtp_block() {
    let Some(model) = flash_model() else { return };
    let Some(block) = std::env::var_os("ARES_TEST_FLASH_NEXT_MTP") else {
        eprintln!("skipped: set ARES_TEST_FLASH_NEXT_MTP to the file of Flash-Next's MTP block");
        return;
    };
    let block = block.to_str().expect("a path in UTF-8").to_owned();
    let devices = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0,1".to_owned());
    let ram = flash_ram();
    let chat = "enable_thinking=false";
    let mut server = serve_model(
        &model,
        port(3),
        &["--devices", &devices, "--ram", &ram, "--chat-var", chat, "--drafters", &block],
        SLOW,
    );
    let log = server.logged().join("\n");
    assert!(log.contains("MTP block") && log.contains("drafts"), "the plan names it: {log}");
    assert!(log.contains("experts share the tiers"), "and what it took: {log}");
    let client: &Client = &server;
    let ask = |question: &str, draft: bool| {
        let body =
            client.chat(question, json!({"max_tokens": 80, "temperature": 0, "draft": draft}));
        client.post("/v1/chat/completions", &body)
    };
    let first = answer(&ask(HASH_TABLE, true));
    eprintln!("RES drafts: {first:?}");
    assert!(
        first.starts_with(
            "A hash table stores data by mapping keys to specific indices in an array using a \
             hash function."
        ),
        "the reference's first sentence: {first:?}"
    );
    let text = answer(&ask(GERMAN, true));
    assert!(mentions(&text, GERMAN_WORDS), "a German answer about light: {text:?}");
    // The same question again: the n-gram drafter has the answer.
    let again = answer(&ask(HASH_TABLE, true));
    assert!(again.starts_with("A hash table stores data"), "the second time: {again:?}");
    let (proposed, accepted) =
        (client.metric("drafts_proposed_total"), client.metric("drafts_accepted_total"));
    eprintln!("RES drafts: {proposed} verified, {accepted} accepted");
    assert!(proposed > 0 && accepted > 0, "{proposed} drafts verified, {accepted} accepted");
    assert!(accepted <= proposed);
    // A request that says no gets none, and the same kind of answer.
    let plain = answer(&ask(HASH_TABLE, false));
    assert_eq!(client.metric("drafts_proposed_total"), proposed, "a request without drafts");
    assert!(plain.starts_with("A hash table stores data"), "without drafts: {plain:?}");
    let (picks, hits) = (client.metric("expert_picks_total"), client.metric("expert_hits_total"));
    let misses = client.metric("expert_misses_ram_total")
        + client.metric("expert_misses_disk_total")
        + client.metric("expert_misses_on_their_way_total");
    assert_eq!(picks, hits + misses, "every pick, the block's included, was a hit or a miss");
    server.stop();
}

/// Flash-Next through the binary on one device, where nearly half of the
/// experts are in no tier and come from the file while it answers: the
/// answers are the ones of two devices, the reference's first sentence
/// included, and the counters say that experts were read from the file.
#[test]
fn flash_next_answers_on_one_card() {
    let Some(model) = flash_model() else { return };
    let devices = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0".to_owned());
    let device = devices.split(',').next().unwrap_or("0").to_owned();
    let ram = flash_ram();
    let mut server = serve_model(
        &model,
        port(2),
        &["--devices", &device, "--ram", &ram, "--chat-var", "enable_thinking=false"],
        SLOW,
    );
    let log = server.logged().join("\n");
    assert!(log.contains("plan: RAM"), "the plan says where the RAM goes: {log}");
    let client: &Client = &server;
    assert_eq!(client.get("/health").body, r#"{"status":"ok"}"#);
    let ask = |question: &str| {
        let body = client.chat(question, json!({"max_tokens": 80, "temperature": 0}));
        client.post("/v1/chat/completions", &body)
    };
    let german = ask(GERMAN);
    let text = answer(&german);
    eprintln!("RES one card: German answer {text:?}; timings {}", german.json()["timings"]);
    assert!(mentions(&text, GERMAN_WORDS), "a German answer about light: {text:?}");
    let text = answer(&ask(ENGLISH));
    eprintln!("RES one card: English answer {text:?}");
    assert!(mentions(&text, ENGLISH_WORDS), "an English answer about light: {text:?}");
    let text = answer(&ask(HASH_TABLE));
    assert!(
        text.starts_with(
            "A hash table stores data by mapping keys to specific indices in an array using a \
             hash function."
        ),
        "the reference's first sentence: {text:?}"
    );
    // The same question again finds its experts in the tiers.
    let disk_before = client.metric("expert_misses_disk_total");
    let text = answer(&ask(HASH_TABLE));
    assert!(text.starts_with("A hash table stores data"), "the second time: {text:?}");
    let disk_again = client.metric("expert_misses_disk_total") - disk_before;

    let (picks, hits) = (client.metric("expert_picks_total"), client.metric("expert_hits_total"));
    let (ram, disk, coming) = (
        client.metric("expert_misses_ram_total"),
        client.metric("expert_misses_disk_total"),
        client.metric("expert_misses_on_their_way_total"),
    );
    eprintln!(
        "RES one card: {picks} picks, {hits} in a slot, {ram} from RAM, {disk} from the file \
         ({disk_again} of them when the last question was asked again), {coming} on their way"
    );
    assert!(picks > 1000 && hits > 0 && hits <= picks, "{picks} picks, {hits} hits");
    assert_eq!(picks, hits + ram + disk + coming, "every pick was a hit or a miss");
    assert!(disk > 0, "one card does not hold the model: experts came from the file");
    assert!(client.metric("steps_total") > 100);
    server.stop();
}

/// Flash-Next keeps a usage profile of its experts in the `--cache` directory:
/// the first start has none, reads the warm-up text and writes the profile;
/// the next start fills the tiers from it instead of the warm-up and answers
/// as the reference does; a damaged profile is ignored and written anew.
#[test]
fn flash_next_starts_from_the_usage_profile_of_its_experts() {
    let Some(model) = flash_model() else { return };
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("flash-profile");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let devices = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0,1".to_owned());
    let ram = flash_ram();
    let start = || {
        let flags = [
            "--devices",
            devices.as_str(),
            "--ram",
            ram.as_str(),
            "--chat-var",
            "enable_thinking=false",
            "--cache",
            dir.to_str().expect("a path in UTF-8"),
        ];
        let begun = Instant::now();
        let mut server = serve_model(&model, port(4), &flags, SLOW);
        let line = server
            .logged()
            .iter()
            .find(|line| line.contains("expert profile"))
            .cloned()
            .unwrap_or_else(|| panic!("a line about the profile: {:#?}", server.logged()));
        eprintln!("RES profile: listening after {:.1} s; {line}", begun.elapsed().as_secs_f64());
        (server, line)
    };
    let profile = || -> Vec<PathBuf> {
        std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|e| e == "experts"))
            .collect()
    };
    let hash_table = |server: &Server| {
        let body = server.chat(HASH_TABLE, json!({"max_tokens": 80, "temperature": 0}));
        let reply = server.post("/v1/chat/completions", &body);
        let text = answer(&reply);
        eprintln!("RES profile: {text:?}; timings {}", reply.json()["timings"]);
        assert!(
            text.starts_with(
                "A hash table stores data by mapping keys to specific indices in an array using \
                 a hash function."
            ),
            "the reference's first sentence: {text:?}"
        );
    };

    // No profile: the warm-up fills the tiers, and the profile is written.
    let (server, line) = start();
    assert!(line.contains("none for this model file") && line.contains("written now"), "{line}");
    let written = profile();
    assert_eq!(written.len(), 1, "one profile, in the directory of --cache");
    let first = std::fs::read(&written[0]).unwrap();
    hash_table(&server);
    // A clean stop writes what the requests used.
    server.stop();
    let stopped = std::fs::read(&written[0]).unwrap();
    assert_eq!(stopped.len(), first.len());
    assert_ne!(stopped, first, "the profile has the heat of the request now");

    // The profile fills the tiers, and nothing is warmed up.
    let (server, line) = start();
    assert!(line.contains("found") && line.contains("no warm-up"), "{line}");
    hash_table(&server);
    let (picks, hits) = (server.metric("expert_picks_total"), server.metric("expert_hits_total"));
    eprintln!("RES profile: {picks} picks, {hits} in a slot");
    server.stop();

    // A damaged profile is ignored with a line that says so, and written anew.
    let mut damaged = std::fs::read(&written[0]).unwrap();
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0x5a;
    std::fs::write(&written[0], &damaged).unwrap();
    let (server, line) = start();
    assert!(line.contains("ignored: it is damaged") && line.contains("written now"), "{line}");
    hash_table(&server);
    server.stop();
    assert_ne!(std::fs::read(&written[0]).unwrap(), damaged);
    let _ = std::fs::remove_dir_all(&dir);
}
