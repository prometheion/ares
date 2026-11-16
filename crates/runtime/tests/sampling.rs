//! What a request's sampling settings do to the step record, and what the
//! conductor refuses, against an executor that computes picks on the host.
//!
//! The records are what the devices read, so what they say about the rows of a
//! window that verifies drafts is what the sampler on the devices works from:
//! these tests read the record of every step and check each wanted row against
//! the definition in `ares-sampler`. That the devices then pick the same
//! tokens with drafts and without is `tests/gpu.rs`'s business.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{BiasKey, LogitBias, SamplerKind, Sampling};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{
    Model, RecordView, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel, TokenFilter,
};
use ares_sampler::penalties::{Corrections, PenaltyParams, entries_for};
use ares_sampler::{Entry, TokenMask};
use common::*;

fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 512;
    // Text that loops, so that drafts from earlier text come true, and a draft
    // block of the model's own that gets the first links right more often
    // than the last (the lookup in earlier text takes part only next to one).
    config.context.window = 1;
    config.context.epoch = 40;
    config.context.buckets = 512;
    config.own_drafts = 7;
    config.own_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
    Arc::new(SyntheticModel::new(config).unwrap())
}

/// A runtime that hands every step's record to `observer`.
fn runtime(
    model: &Arc<SyntheticModel>,
    speculation: bool,
    observer: impl FnMut(&RecordView<'_>) + Send + 'static,
) -> Runtime {
    let config = RuntimeConfig {
        pages: 4096,
        cells_per_page: 16,
        max_slots: 64,
        // A window of one row is no drafts at all.
        window_rows: if speculation { 8 } else { 1 },
        warm_up: false,
        ..RuntimeConfig::default()
    };
    // A step costs what a device's would (a pass over the weights and a little
    // for each row), so that the policy sees what drafts cost and are worth.
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_row_cost(Duration::from_micros(3000), Duration::from_micros(30))
        .with_observer(observer);
    Runtime::start(model.clone(), Box::new(host), config).unwrap()
}

/// Calls `f` with the wanted rows of the record, in order: `(row, segment)`.
fn wanted_rows(record: &RecordView<'_>) -> Vec<(u32, u32)> {
    let segment_of = |row: u32| {
        (0..record.segments())
            .find(|&i| {
                let s = record.segment(i);
                (s.row0..s.row0 + s.n_rows).contains(&row)
            })
            .expect("a segment holds the row")
    };
    (0..record.runs())
        .flat_map(|run| {
            let (first, count) = record.run(run);
            first..first + count
        })
        .map(|row| (row, segment_of(row)))
        .collect()
}

#[test]
fn the_rows_of_a_window_carry_the_corrections_of_their_own_paths() {
    // A penalty over the last three tokens: a row three tokens into a window
    // is corrected by the window's own tokens alone, which the record shows.
    let corrections = Corrections {
        penalties: PenaltyParams { last_n: 3, repeat: 1.2, frequency: 0.5, presence: 0.25 },
        ..Corrections::none()
    };
    let verified = Arc::new(Mutex::new(0usize));
    let seen = verified.clone();
    let wanted_corrections = corrections.clone();
    let rt = runtime(&model(), true, move |record| {
        // A step that only drafts verifies nothing: its links pick without the
        // corrections of the drafts before them, which are not made yet.
        if record.drafts_only() {
            return;
        }
        let wanted = wanted_rows(record);
        for (index, &(row, segment)) in wanted.iter().enumerate() {
            let s = record.segment(segment);
            let k = (row - s.row0) as usize;
            if s.direct || s.n_rows < 3 || k < 2 {
                continue;
            }
            // The window's tokens up to this row.
            let path: Vec<u32> = (0..=k as u32).map(|i| record.row(s.row0 + i).token).collect();
            let want = entries_for(&path, &wanted_corrections);
            let (first, count) = record.sampler(index as u32).entries();
            let got: Vec<Entry> = (first..first + count).map(|i| record.correction(i)).collect();
            assert_eq!(got, want, "row {k} of a window of {}", s.n_rows);
            *seen.lock().unwrap() += 1;
        }
    });
    let sampling = Sampling {
        temperature: Some(0.8),
        repeat_penalty: Some(1.2),
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.25),
        repeat_last_n: Some(3),
        seed: Some(5),
        ..Sampling::default()
    };
    let mut request = request("a prompt that repeats itself, a prompt that repeats", Some(300));
    request.sampling = sampling;
    let run = run_one(&rt, request);
    assert_eq!(run.usage.completion_tokens, 300);
    let checked = *verified.lock().unwrap();
    assert!(checked > 20, "only {checked} rows of windows were checked");
}

#[test]
fn a_request_without_corrections_has_none_in_its_records() {
    let steps = Arc::new(Mutex::new((0usize, 0usize)));
    let counts = steps.clone();
    let rt = runtime(&model(), true, move |record| {
        let mut c = counts.lock().unwrap();
        c.0 += 1;
        c.1 += (record.corrections() + record.masks().0) as usize;
    });
    let mut request = request("nothing to correct here", Some(80));
    request.sampling.temperature = Some(0.7);
    run_one(&rt, request);
    let (steps, corrections) = *steps.lock().unwrap();
    assert!(steps > 5 && corrections == 0, "{steps} steps, {corrections} corrections");
}

#[test]
fn a_bias_is_the_same_for_every_row_and_a_ban_reaches_the_record() {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let seen = rows.clone();
    let rt = runtime(&model(), true, move |record| {
        for index in 0..record.picked_rows() {
            let (first, count) = record.sampler(index).entries();
            let entries: Vec<Entry> =
                (first..first + count).map(|i| record.correction(i)).collect();
            seen.lock().unwrap().push(entries);
        }
    });
    let mut request = request("bias", Some(60));
    request.sampling.logit_bias = vec![
        LogitBias { key: BiasKey::Token(65), bias: Some(3.0) },
        LogitBias { key: BiasKey::Token(66), bias: None },
    ];
    run_one(&rt, request);
    let rows = rows.lock().unwrap();
    assert!(rows.len() > 30);
    let want = vec![
        Entry { bias: 3.0, ..Entry::new(65) },
        Entry { bias: f32::NEG_INFINITY, ..Entry::new(66) },
    ];
    assert!(rows.iter().all(|entries| *entries == want), "{:?}", rows.iter().find(|e| **e != want));
}

#[test]
fn a_mirostat_request_verifies_no_drafts_and_another_still_does() {
    let widest = Arc::new(Mutex::new(0u32));
    let seen = widest.clone();
    let rt = runtime(&model(), true, move |record| {
        let mut w = seen.lock().unwrap();
        for i in 0..record.segments() {
            *w = (*w).max(record.segment(i).n_rows);
        }
    });
    let mut plain = request("loops and loops and loops", Some(200));
    plain.sampling.temperature = Some(0.8);
    plain.sampling.seed = Some(1);
    run_one(&rt, plain.clone());
    assert!(*widest.lock().unwrap() > 1, "drafts are verified for an ordinary request");

    *widest.lock().unwrap() = 0;
    let mut miro = plain;
    miro.id = fresh_id();
    miro.sampling.mirostat = Some(2);
    miro.prompt = ares_api::Prompt::Text("a prompt of one row at a time".into());
    let draft_rows = |rt: &Runtime| rt.snapshot().expect("the engine runs").counters.draft_rows;
    let before = draft_rows(&rt);
    run_one(&rt, miro);
    let w = *widest.lock().unwrap();
    // The prompt is read in one segment; every segment of the reply is one row.
    assert!(w <= "a prompt of one row at a time".len() as u32);
    let again = {
        let mut r = request("loops and loops and loops", Some(200));
        r.sampling.mirostat = Some(1);
        run_one(&rt, r)
    };
    assert_eq!(again.usage.completion_tokens, 200);
    assert_eq!(draft_rows(&rt), before, "a request with mirostat verified drafts");
}

/// Allows only the tokens whose value is the number of tokens the reply and
/// the drafts have, modulo 3, plus 100.
/// The calls a filter has had: the tokens of the reply, the number of drafts
/// before the row, and those drafts.
type Calls = Arc<Mutex<Vec<(usize, usize, Vec<u32>)>>>;

struct Counting {
    calls: Calls,
}

impl TokenFilter for Counting {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        self.calls.lock().unwrap().push((reply.len(), drafts.len(), drafts.to_vec()));
        mask.fill(false);
        mask.set(100 + ((reply.len() + drafts.len()) % 3) as u32, true);
        true
    }
}

#[test]
fn a_filter_makes_a_mask_for_every_wanted_row_and_the_record_carries_it() {
    let model = model();
    let words = TokenMask::words_for(model.vocab());
    let masks = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
    let seen = masks.clone();
    let rt = runtime(&model, true, move |record| {
        let (count, w) = record.masks();
        let mut m = seen.lock().unwrap();
        m.0 += record.picked_rows() as usize;
        m.1 += count as usize;
        assert!(count == 0 || w as usize == words);
        for index in 0..record.picked_rows() {
            let slot = record.sampler(index).mask().expect("every wanted row has a mask");
            // Exactly one token is allowed.
            let allowed: u32 = (0..w).map(|i| record.mask_word(slot, i).count_ones()).sum();
            assert_eq!(allowed, 1);
            m.2 += 1;
        }
    });
    let calls = Arc::new(Mutex::new(Vec::new()));
    // The filter lets one token through at each place, so that the reply
    // repeats at once and the lookup in earlier text has drafts to offer (the
    // filter binds every row, so the request takes no drafts from the model's
    // own block).
    let mut r = request("filtered", Some(200));
    r.sampling.temperature = Some(0.8);
    let run = start_filtered(&rt, r, Counting { calls: calls.clone() }).unwrap().finish();
    assert_eq!(run.usage.completion_tokens, 200);
    let (wanted, held, checked) = *masks.lock().unwrap();
    assert!(wanted >= 200 && held == wanted && checked == wanted, "{wanted} {held} {checked}");
    // The filter saw the reply grow by one token at a time, and the drafts of
    // a window before each of its rows.
    let calls = calls.lock().unwrap();
    assert!(calls.iter().any(|(_, drafts, _)| *drafts > 0), "no window with drafts");
    assert!(
        calls.iter().all(|(reply, drafts, list)| *drafts == list.len() && reply + drafts <= 201)
    );
}

#[test]
fn what_the_sampler_cannot_serve_is_refused_by_name() {
    let rt = runtime(&model(), false, |_| {});
    // The ranges of the API are refused when the request is handed over; what
    // only the engine knows, when the conductor looks at it.
    let refused = |sampling: Sampling| -> String {
        let mut r = request("refuse", Some(5));
        r.sampling = sampling;
        match start(&rt, r) {
            Err(error) => error.to_string(),
            Ok(handle) => handle.failure().to_string(),
        }
    };
    let late = refused(Sampling {
        samplers: Some(vec![SamplerKind::TopK, SamplerKind::Penalties]),
        ..Sampling::default()
    });
    assert!(late.contains("penalties") && late.contains("before"), "{late}");
    let twice = refused(Sampling {
        samplers: Some(vec![SamplerKind::Dry, SamplerKind::Dry]),
        ..Sampling::default()
    });
    assert!(twice.contains("once each"), "{twice}");
    let outside = refused(Sampling {
        logit_bias: vec![LogitBias { key: BiasKey::Token(100_000), bias: Some(1.0) }],
        ..Sampling::default()
    });
    assert!(outside.contains("logit_bias") && outside.contains("100000"), "{outside}");
    let long = refused(Sampling { mirostat: Some(3), ..Sampling::default() });
    assert!(long.contains("mirostat"), "{long}");
    // The range checks of the API reach the engine's submit.
    let mut r = request("refuse", Some(5));
    r.sampling.repeat_last_n = Some(100_000);
    assert!(start(&rt, r).err().unwrap().to_string().contains("repeat_last_n"));
}
