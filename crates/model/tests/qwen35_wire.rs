//! The conductor's bookkeeping on the real models: which drafter is worth
//! what, and requests that wait in line.
//!
//! Needs GPUs (`ARES_TEST_CUDA=1`) and the models (`ARES_TEST_QWEN35_0_8B`
//! for the small one on one device, `ARES_TEST_QWEN38_27B` for the large one
//! on two).
//!
//! The measurement of the drafters runs only when `ARES_WIRE_CASE` names
//! one of its cases (`none`, `w4`, `w8`, `w16`): each
//! loads the 27B once and takes about a minute. `ARES_TEST_DRAFT_VOCAB` names
//! a draft vocabulary (a token id per line) and `ARES_WIRE_OUT` a directory
//! for the texts.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use ares_api::{
    Engine, Event, EventSink, GenerateRequest, Message, Prompt, RequestId, Role, Sampling, Usage,
};
use ares_cuda::{Device, Precision, Wait, device_count};
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig};
use ares_tokenizer::ChatOptions;
use common::helpers::env_path;

mod common;

fn enabled() -> bool {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return false;
    }
    true
}

/// What a request came to.
struct Answer {
    text: String,
    usage: Usage,
    /// Seconds from the submission to the first text.
    first: f64,
    /// Seconds from the first text to the end.
    rest: f64,
}

fn ask(runtime: &Runtime, id: u64, question: &str, sampling: Sampling, max_tokens: u32) -> Answer {
    let (tx, rx) = mpsc::channel();
    let state = Arc::new(Mutex::new((String::new(), None::<Instant>)));
    let sink = {
        let state = state.clone();
        EventSink::new(move |event| {
            match event {
                Event::Logprobs(_) | Event::Progress(_) | Event::Generated(_) => {}
                Event::Text(piece) => {
                    let mut state = state.lock().unwrap();
                    state.0.push_str(&piece);
                    state.1.get_or_insert_with(Instant::now);
                }
                Event::Done { usage, .. } => {
                    let _ = tx.send(Ok((usage, Instant::now())));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            true
        })
    };
    let begun = Instant::now();
    runtime
        .submit(
            GenerateRequest {
                model: "q".into(),
                prompt: Prompt::Chat {
                    messages: vec![Message::new(Role::User, question)],
                    tools: None,
                    variables: Vec::new(),
                    continue_final: false,
                },
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft: true,
                structure: None,
                priority: 0,
                id: RequestId(id),
            },
            sink,
        )
        .expect("nothing is refused for load");
    // A request that takes long says where the engine stands, so that a run
    // that hangs shows whether steps still run and who waits.
    let watching = std::sync::atomic::AtomicBool::new(true);
    let result = std::thread::scope(|scope| {
        let watching = &watching;
        scope.spawn(move || {
            let mut waited = 0u32;
            while watching.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                waited += 1;
                if waited.is_multiple_of(200) {
                    match runtime.snapshot() {
                        Some(s) => eprintln!(
                            "RES request {id} after {} s: {} running, {} in line, {:?}",
                            waited / 10,
                            s.active,
                            s.waiting,
                            s.counters
                        ),
                        None => eprintln!(
                            "RES request {id} after {} s: the conductor does not answer",
                            waited / 10
                        ),
                    }
                }
            }
        });
        let result = rx.recv();
        watching.store(false, std::sync::atomic::Ordering::Relaxed);
        result
    });
    let (usage, done) = result.unwrap().unwrap();
    let (text, first) = std::mem::take(&mut *state.lock().unwrap());
    let first = first.unwrap_or(done);
    Answer { text, usage, first: (first - begun).as_secs_f64(), rest: (done - first).as_secs_f64() }
}

/// A German text of everyday prose, for a request that writes its prompt
/// again with edits.
const GERMAN_TEXT: &str = "\
Die Stadtbibliothek am Marktplatz wurde 1987 eröffnet und war von Anfang an mehr als ein Ort, an \
dem man Bücher ausleiht. Im Erdgeschoss liegt der große Lesesaal mit seinen hohen Fenstern, durch \
die am Vormittag die Sonne auf die langen Holztische fällt. Wer früh kommt, findet dort fast immer \
einen ruhigen Platz, denn die meisten Besucher treffen erst nach dem Mittagessen ein.

Im ersten Stock stehen die Regale mit den Sachbüchern, geordnet nach Themen: Geschichte, Technik, \
Naturwissenschaften, Reisen und Kochen. Die Bibliothek besitzt heute rund achtzigtausend Bände, \
dazu Zeitungen, Zeitschriften, Hörbücher und eine kleine Sammlung alter Stadtpläne. Viele dieser \
Pläne wurden von Bürgerinnen und Bürgern gespendet, die sie auf dem Dachboden gefunden hatten.

Für Kinder gibt es im Untergeschoss einen eigenen Bereich mit niedrigen Regalen, bunten Kissen \
und einer Bühne, auf der jeden Mittwoch um vier Uhr vorgelesen wird. Die Vorlesestunde ist so \
beliebt, dass die Bibliothek im vergangenen Jahr eine zweite am Samstagvormittag eingeführt hat. \
Eltern schätzen besonders, dass sie in dieser Zeit in Ruhe selbst nach einem Buch suchen können.

Seit dem Umbau vor drei Jahren gibt es außerdem zwanzig Arbeitsplätze mit Strom und schnellem \
Internet. Schülerinnen und Schüler bereiten sich dort auf Prüfungen vor, Studierende schreiben \
ihre Hausarbeiten, und ältere Menschen lassen sich von den Mitarbeitern zeigen, wie man ein \
Formular online ausfüllt. Die Bibliothek ist damit zu einem Treffpunkt für alle Generationen \
geworden.

Die Leiterin der Bibliothek, Frau Dr. Hoffmann, sagt, dass die Zahl der Ausleihen zwar leicht \
zurückgeht, die Zahl der Besuche aber jedes Jahr steigt. Die Menschen kämen nicht mehr nur wegen \
der Bücher, sondern weil sie einen Ort suchten, an dem man nichts kaufen muss und trotzdem \
willkommen ist. Für das kommende Jahr plant die Stadt, die Öffnungszeiten der Bibliothek zu \
verlängern: Sie soll dann auch am Sonntagnachmittag geöffnet sein.";

/// An English text of the same kind.
const ENGLISH_TEXT: &str = "\
The footbridge over the river was built in 1932, when the town on the north bank had grown too \
large for the old ferry. It is a narrow steel bridge with a wooden deck, just wide enough for \
two people to pass each other, and for almost a century it has been the shortest way from the \
railway station to the market square.

Every morning the bridge carries a steady stream of commuters. Cyclists are asked to dismount, \
although few of them do, and the planks rattle under their wheels. In summer, children lean over \
the railing to watch the fish that gather in the shade below, and in winter the town council \
spreads sand on the deck before the first frost so that nobody slips.

The bridge has been repaired many times. The wooden deck was replaced after the flood of 1954, \
the steel was repainted in its original dark green in the seventies, and new lamps were added \
when the riverside path was opened. Engineers inspect the bridge every two years; their last \
report found the steel sound but warned that the bearings at the southern end would need to be \
renewed within a decade.

Not everyone is fond of the old structure. Shopkeepers on the south bank have long argued that a \
wider bridge, open to delivery vans, would bring more customers to their street. Others reply \
that the narrow bridge is exactly what keeps the square quiet, and that a town with a second \
road across the river would soon need a third.

Last spring the council put the question to a vote. A clear majority chose to keep the bridge \
as it is and to spend the money on the bearings, the lamps and a fresh coat of paint. The work \
is due to begin next year and should take about four months, during which a temporary ferry \
will once again carry people across the river, as it did before the bridge was built.";

/// The four requests of the measurement: a name, the question, the most
/// tokens it may write.
fn requests() -> Vec<(&'static str, String, u32)> {
    vec![
        (
            "german new",
            "Schreibe einen ausf\u{fc}hrlichen Aufsatz \u{fc}ber die Geschichte der Eisenbahn in \
             Deutschland, von den Anf\u{e4}ngen im 19. Jahrhundert bis heute."
                .to_owned(),
            400,
        ),
        (
            "english new",
            "Write a detailed essay about the history of the printing press and how it changed \
             Europe."
                .to_owned(),
            400,
        ),
        (
            "german edit",
            format!(
                "Schreibe den folgenden Text vollst\u{e4}ndig ab. \u{c4}ndere dabei nur zweierlei: \
                 Ersetze \u{fc}berall das Wort \u{201e}Bibliothek\u{201c} durch \
                 \u{201e}B\u{fc}cherei\u{201c} und die Jahreszahl 1987 durch 1978. Gib nur den \
                 ge\u{e4}nderten Text aus, ohne Einleitung und ohne Anmerkungen.\n\n{GERMAN_TEXT}"
            ),
            900,
        ),
        (
            "english edit",
            format!(
                "Copy the following text in full. Change only two things: replace the word \
                 \"bridge\" by \"crossing\" everywhere and the year 1932 by 1923. Output only the \
                 changed text, without an introduction and without remarks.\n\n{ENGLISH_TEXT}"
            ),
            900,
        ),
    ]
}

fn draft_vocabulary() -> Option<Arc<[u32]>> {
    std::env::var_os("ARES_TEST_DRAFT_VOCAB")
        .map(|path| ares_model::files::read_draft_vocabulary(path.as_ref()).unwrap().into())
}

/// Tokens per second of a request that writes from nothing and of one that
/// writes its prompt again with edits, in German and in English, with the
/// window `ARES_WIRE_CASE` names (none, or w4, w8, w16: the rows of a
/// sequence's window, the model's MTP block and the n-gram drafter
/// as the policy chooses them for every step). The model runs as it is
/// served: a tensor split over both devices, the fast kernels, the most
/// probable token.
#[test]
fn what_each_drafter_is_worth_on_the_27b() {
    let Some(case) = std::env::var("ARES_WIRE_CASE").ok() else {
        eprintln!("skipped: set ARES_WIRE_CASE to one of none, w4, w8, w16");
        return;
    };
    if !enabled() {
        return;
    }
    let Some(path) = env_path("ARES_TEST_QWEN38_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let window = match case.as_str() {
        "none" => 0,
        "w4" => 4,
        "w8" => 8,
        "w16" => 16,
        other => panic!("no case {other}"),
    };
    let options = Options {
        precision: Precision::Fast,
        logit_rows: 32,
        window_rows: 2 * window,
        draft_vocabulary: draft_vocabulary(),
        ..Options::default()
    };
    let context = 4096;
    let (gguf, tokenizer) = common::opened(&path);
    let model = Arc::new(
        Qwen35::new(gguf, tokenizer, "q", &Spread::Tensor(vec![0.625, 0.375]), options, context)
            .unwrap()
            .with_chat_options(ChatOptions { enable_thinking: false, ..ChatOptions::default() }),
    );
    let config = RuntimeConfig {
        pages: 2 * context.div_ceil(128) + 2,
        cells_per_page: 128,
        max_slots: 2,
        window_rows: window.max(1),
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let devices =
        vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
    let begun = Instant::now();
    let runtime = Runtime::start_gpu(model, devices, config, gpu).unwrap();
    eprintln!("RES {case} | loaded in {:.1} s", begun.elapsed().as_secs_f64());
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    // The first request pays for whatever the first steps of a process pay.
    let _ = ask(&runtime, 1, "Say hello.", greedy.clone(), 8);
    let out = std::env::var_os("ARES_WIRE_OUT").map(PathBuf::from);
    for (i, (name, question, max_tokens)) in requests().into_iter().enumerate() {
        let _ = runtime.take_snapshot();
        let answer = ask(&runtime, 10 + i as u64, &question, greedy.clone(), max_tokens);
        let c = runtime.snapshot().expect("the engine runs").counters;
        let tokens = answer.usage.completion_tokens;
        let by_source = |source: usize| {
            let (tested, accepted) = (c.drafts_tested_at[source], c.drafts_accepted_at[source]);
            let all: u64 = accepted.iter().sum();
            format!("{all} accepted, by position {:?} of {:?}", &accepted[..4], &tested[..4])
        };
        eprintln!(
            "RES {case} | {name}: {tokens} tokens after a prompt of {}, first after {:.2} s, then \
             {:.1} tokens/s; {} steps with {:.2} tokens each, {} draft rows ({} cut again); ngram \
             {}; own {} in {} links of {:.2} ms",
            answer.usage.prompt_tokens,
            answer.first,
            f64::from(tokens.saturating_sub(1)) / answer.rest.max(1e-9),
            c.window_rows,
            c.tokens as f64 / c.window_rows.max(1) as f64,
            c.draft_rows,
            c.drafts_rejected,
            by_source(0),
            by_source(1),
            c.model_draft_links,
            c.model_draft_us as f64 / 1e3 / c.model_draft_links.max(1) as f64,
        );
        if let Some(dir) = &out {
            std::fs::create_dir_all(dir).unwrap();
            let file = format!("{case}-{}.txt", name.replace(' ', "-"));
            std::fs::write(dir.join(file), &answer.text).unwrap();
        }
    }
    runtime.shutdown();
}

/// More requests than the pool holds, on the small model: none is refused,
/// the ones that do not fit wait in line, and each answers as it does alone
/// (as far as a sequence does next to others: the kernels of a step of
/// several sequences round differently from those of one).
#[test]
fn more_requests_than_fit_wait_in_line_and_all_answer() {
    let (true, Some(path)) = (enabled(), env_path("ARES_TEST_QWEN35_0_8B")) else {
        return;
    };
    let options = Options {
        precision: Precision::Exact,
        max_rows: 64,
        logit_rows: 16,
        window_rows: 7,
        ..Options::default()
    };
    let (gguf, tokenizer) = common::opened(&path);
    let model =
        Arc::new(Qwen35::new(gguf, tokenizer, "q", &Spread::Single, options, 1024).unwrap());
    // Six pages of 128 cells. A request is charged for its prompt (a page)
    // and for the 120 tokens it may write (a page): three fit, six ask.
    let config = RuntimeConfig {
        pages: 6,
        cells_per_page: 128,
        max_slots: 4,
        window_rows: 7,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let device = Device::open(0, Wait::Block).unwrap();
    let runtime = Runtime::start_gpu(model, vec![device], config, gpu).unwrap();
    let questions = [
        "Z\u{e4}hle von eins bis zwanzig, jede Zahl als Wort in einer eigenen Zeile.",
        "Write the line 'the quick brown fox jumps over the lazy dog' four times, numbered.",
        "Nenne die Wochentage und die Monate des Jahres, jeweils in einer Zeile.",
        "List the first fifteen prime numbers, one per line.",
        "Schreibe drei kurze S\u{e4}tze \u{fc}ber den Herbst.",
        "Spell the word 'engine' letter by letter, then spell it backwards.",
    ];
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    let alone: Vec<String> = questions
        .iter()
        .enumerate()
        .map(|(i, question)| ask(&runtime, 1 + i as u64, question, greedy.clone(), 120).text)
        .collect();
    let before = runtime.take_snapshot().expect("the engine runs").counters;
    let together: Vec<Answer> = std::thread::scope(|scope| {
        let handles: Vec<_> = questions
            .iter()
            .enumerate()
            .map(|(i, question)| {
                let (runtime, greedy) = (&runtime, greedy.clone());
                scope.spawn(move || ask(runtime, 100 + i as u64, question, greedy, 120))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let snapshot = runtime.snapshot().expect("the engine runs");
    let c = snapshot.counters;
    eprintln!(
        "RES six requests in a pool for three: {} waited, {} gave way, {} draft rows cut, {} \
         steps, {} tokens; first text after {:?} s",
        c.requests_waited,
        c.requests_suspended,
        c.draft_rows_cut,
        c.steps,
        c.tokens,
        together.iter().map(|a| (a.first * 100.0).round() / 100.0).collect::<Vec<_>>(),
    );
    assert_eq!(before.requests_waited, 0, "alone nobody waits");
    assert!(c.requests_waited >= 2, "{c:?}");
    assert_eq!((snapshot.active, snapshot.waiting), (0, 0));
    assert_eq!(
        snapshot.pages_free + snapshot.cached_pages,
        snapshot.pages_capacity,
        "every page is back: free, or held by the prefix cache"
    );
    for (i, (alone, shared)) in alone.iter().zip(&together).enumerate() {
        let same = alone.chars().zip(shared.text.chars()).take_while(|(a, b)| a == b).count();
        eprintln!(
            "RES request {i}: {} tokens, {same} of {} characters equal its text alone",
            shared.usage.completion_tokens,
            alone.chars().count()
        );
        assert!(shared.usage.completion_tokens > 0, "request {i} wrote nothing");
        assert!(
            same >= 40.min(alone.chars().count()),
            "request {i} says something else next to the others: {:?}",
            shared.text
        );
    }
    runtime.shutdown();
}
