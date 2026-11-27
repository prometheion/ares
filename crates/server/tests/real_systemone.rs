//! `POST /v1/systemone` with a real model on a GPU: a request as a client of
//! Flide sends it, answered, and every question's choice checked against a chat
//! completion of that question alone (the same prompt, written out here as
//! Flide's `next_token` format writes it, greedy, one token): where the chat
//! reply starts with a label of the question, the decision chose that option.
//! Several requests in one body give the answers they give one at a time.
//!
//! The server runs in this process over the real engine (requests go through
//! the router without a socket). Needs `ARES_TEST_CUDA=1` and
//! `ARES_TEST_QWEN35_0_8B` (the 0.8B); with `ARES_TEST_API_MODEL` the tests run
//! on that file instead, loaded as `ares --model` loads it (the 27B,
//! Flash-Next: see `gpu/mod.rs`). Lines that start with `RES` are the
//! measurements.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_0_8B=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test real_systemone -- --test-threads=1 --nocapture
//! ```

use std::time::Instant;

use serde_json::{Value, json};

mod common;
mod gpu;

const SYSTEMONE: &str = "/v1/systemone";
const CHAT: &str = "/v1/chat/completions";

const STATE: &str = "From: Erika Mustermann\nSubject: Charged twice!!\n\nHello, I ordered \
                     yesterday and today I see that the amount was taken from my account twice. \
                     This is the second time this month. I expect this to be fixed at once, or I \
                     will close my account. Order 4711-0815.";

/// A question's id, the question as Flide's `next_token` format (style
/// `vllm`) writes it, and its labels in the order of the options.
type Written = (&'static str, String, Vec<&'static str>);

/// The questions of the request, and each as written.
fn questions() -> (Value, Vec<Written>) {
    let body = json!({
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this email?",
            "criteria": {"billing": "payments, refunds, invoices", "shipping": null,
                         "technical": "the website and the app"}
        },
        "angry": {"type": "noul", "instructions": "Is the sender angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this email?",
            "criteria": ["can wait", "this week", "today", "right now"]
        }
    });
    let written = vec![
        (
            "route",
            "Question: Which team should handle this email?\nA: billing - payments, refunds, \
             invoices\nB: shipping\nC: technical - the website and the app\nAnswer with the \
             letter of one option only."
                .to_owned(),
            vec!["A", "B", "C"],
        ),
        (
            "angry",
            "Question: Is the sender angry?\nyes\nno\nAnswer with yes or no only.".to_owned(),
            vec!["yes", "no"],
        ),
        (
            "urgency",
            "Question: How urgent is this email?\n0: can wait\n1: this week\n2: today\n3: right \
             now\nAnswer with the number of one level only."
                .to_owned(),
            vec!["0", "1", "2", "3"],
        ),
    ];
    (body, written)
}

/// The index of the option an answer chose.
fn chosen(answer: &Value, labels: usize) -> usize {
    match answer["type"].as_str().unwrap() {
        "choice" => {
            let keys: Vec<&String> = answer["probabilities"].as_object().unwrap().keys().collect();
            let choice = answer["choice"].as_str().unwrap();
            keys.iter().position(|k| *k == choice).unwrap()
        }
        "noul" => usize::from(answer["noul"].as_f64().unwrap() < 0.5),
        "score" => {
            let p: Vec<f64> = (0..labels)
                .map(|i| answer["probabilities"][i.to_string()].as_f64().unwrap())
                .collect();
            (0..labels).fold(0, |best, i| if p[i] > p[best] { i } else { best })
        }
        other => panic!("an answer of type {other}"),
    }
}

async fn post(served: &gpu::Served, path: &str, body: &Value) -> Value {
    let (status, text) = served.ask(path, body).await;
    assert_eq!(status, 200, "{text}");
    serde_json::from_str(&text).unwrap()
}

#[tokio::test]
async fn a_decision_chooses_what_each_question_alone_starts_its_reply_with() {
    let Some(served) = gpu::Served::any(None).await else { return };
    let (questions, written) = questions();
    let body = json!({"model": "q", "state": STATE, "questions": questions});
    let begun = Instant::now();
    let answer = post(&served, SYSTEMONE, &body).await;
    eprintln!(
        "RES decision of 3 questions: {:.1} ms; usage {}, meta {}",
        begun.elapsed().as_secs_f64() * 1e3,
        answer["usage"],
        answer["meta"]
    );
    assert_eq!(answer["model"], "q");
    assert_eq!(answer["usage"]["output_tokens"], 3);
    let mut checked = 0;
    for (id, question, labels) in &written {
        let decided = &answer["answers"][id];
        // The same question alone, as a chat request: thinking off, greedy,
        // one token.
        let chat = post(
            &served,
            CHAT,
            &json!({
                "model": "q",
                "messages": [{"role": "user", "content": format!("{STATE}\n\n{question}")}],
                "chat_template_kwargs": {"enable_thinking": false},
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": true,
            }),
        )
        .await;
        let first = chat["choices"][0]["logprobs"]["content"][0]["token"].as_str().unwrap();
        let at = labels.iter().position(|l| *l == first);
        eprintln!("RES question {id}: {decided}; alone the reply starts with {first:?}");
        if let Some(at) = at {
            assert_eq!(chosen(decided, labels.len()), at, "question {id}: {decided}");
            checked += 1;
        }
    }
    eprintln!("RES {checked} of 3 questions start their reply with a label alone");
    assert!(checked >= 2, "the model answers most of these with a label");

    // Several requests in one body: each answered as alone.
    let other = "Hi, can you tell me when my parcel arrives? No hurry. Thanks!";
    let batch = json!({"model": "q", "requests": [
        {"state": STATE, "questions": questions},
        {"state": other, "questions": questions},
    ]});
    let begun = Instant::now();
    let both = post(&served, SYSTEMONE, &batch).await;
    eprintln!("RES two requests in one body: {:.1} ms", begun.elapsed().as_secs_f64() * 1e3);
    let alone = post(&served, SYSTEMONE, &json!({"state": other, "questions": questions})).await;
    for (id, _, labels) in &written {
        let first = &both["results"][0]["answers"][id];
        let second = &both["results"][1]["answers"][id];
        assert_eq!(chosen(first, labels.len()), chosen(&answer["answers"][id], labels.len()));
        assert_eq!(chosen(second, labels.len()), chosen(&alone["answers"][id], labels.len()));
    }
}
