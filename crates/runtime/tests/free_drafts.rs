//! A request with a rule takes the model's own drafts while the rule is not
//! bound, and stops taking what a window gives at the token that makes it bind.
//! On the synthetic model with a host executor.
//!
//! The model's text never repeats here (its next token depends on the
//! position), so the lookup in earlier text has nothing to offer inside a
//! request and every draft is the draft block's. The executor states what a
//! step costs and takes no time for it ([`HostExecutor::without_sleeping`]),
//! so the windows the policy chooses are the same on a machine however busy.
//!
//! The synthetic model has one token for each byte, and `~` ends a reply.

mod common;

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{FinishReason, GenerateRequest, Shape, Start, Structure, ToolFormat};
use ares_grammar::calls::{CallRule, Callable};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{
    RecordView, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel, TokenFilter,
    unprintable,
};
use ares_sampler::TokenMask;
use common::*;
use serde_json::json;

const END: u8 = b'~';

/// The rows of a window: the newest token and seven drafts.
const WINDOW: u32 = 8;

/// A draft block that gets the first links right more often than the last.
const FALLIBLE: &[f64] = &[0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];

/// A model whose text never repeats and whose draft block is right this
/// often (empty: always).
fn model(accuracy: &[f64]) -> Arc<SyntheticModel> {
    model_ending(accuracy, Some(u32::from(END)))
}

/// The same with `eos` as the token that ends a reply, or with none.
fn model_ending(accuracy: &[f64], eos: Option<u32>) -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.eos = eos;
    config.context.window = 1;
    config.context.epoch = 1;
    config.context.buckets = 512;
    config.own_drafts = WINDOW - 1;
    config.own_accuracy = accuracy.to_vec();
    Arc::new(SyntheticModel::new(config).unwrap())
}

/// A window of a step, as its record has it when the devices read it.
#[derive(Clone, Debug)]
struct Window {
    /// Where the newest token is in the request's tokens, the prompt first.
    at: usize,
    /// The tokens of the rows: the newest token, then the drafts.
    tokens: Vec<u32>,
    /// The drafts are the draft block's.
    own: bool,
    /// Whether each row carries a mask.
    masked: Vec<bool>,
}

type Windows = Arc<Mutex<Vec<Window>>>;

/// A runtime on a device whose step costs 3 ms and 0.03 ms a row, and whose
/// draft block takes 0.3 ms a link. It writes down the windows of the request
/// whose prompt has `prompt_len` tokens.
fn runtime(model: &Arc<SyntheticModel>, windows: &Windows, prompt_len: usize) -> Runtime {
    let config = RuntimeConfig {
        pages: 4096,
        cells_per_page: 16,
        max_slots: 8,
        window_rows: WINDOW,
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let windows = windows.clone();
    let observer = move |record: &RecordView<'_>| {
        // The sampler row of a wanted row is its place among the wanted rows.
        let mut sampler_of = HashMap::new();
        for run in 0..record.runs() {
            let (first, count) = record.run(run);
            for row in first..first + count {
                sampler_of.insert(row, sampler_of.len() as u32);
            }
        }
        for i in 0..record.segments() {
            let s = record.segment(i);
            let at = record.row(s.row0).position as usize;
            if at < prompt_len {
                continue;
            }
            let rows = s.row0..s.row0 + s.n_rows;
            windows.lock().unwrap().push(Window {
                at,
                tokens: rows.clone().map(|row| record.row(row).token).collect(),
                own: record.own_drafts(i),
                masked: rows.map(|row| record.sampler(sampler_of[&row]).mask().is_some()).collect(),
            });
        }
    };
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_row_cost(Duration::from_micros(3000), Duration::from_micros(30))
        .with_link_cost(Duration::from_micros(300))
        .without_sleeping()
        .with_observer(observer);
    Runtime::start(model.clone(), Box::new(host), config).unwrap()
}

/// A runtime nobody watches.
fn quiet(model: &Arc<SyntheticModel>) -> Runtime {
    runtime(model, &Windows::default(), usize::MAX)
}

/// The drafts of the model's own block that became tokens.
fn own_accepted(rt: &Runtime) -> u64 {
    snapshot(rt).counters.drafts_accepted_at[1].iter().sum()
}

fn tokens_of(text: &str) -> Vec<u32> {
    text.chars().map(|c| u32::from(unprintable(c).expect("a token of the model"))).collect()
}

/// A prompt whose reply, with no rule, is at least `tokens` tokens before the
/// model ends it, and that reply: the free text of every request below.
fn free_text(model: &Arc<SyntheticModel>, tokens: usize) -> (String, Vec<char>) {
    let rt = quiet(model);
    for i in 0..200 {
        let prompt = format!("free text number {i}");
        let mut plain = request(&prompt, Some(tokens as u32));
        plain.draft = false;
        let run = run_one(&rt, plain);
        if run.finish == FinishReason::Length {
            return (prompt, run.text.chars().collect());
        }
    }
    panic!("no prompt gives {tokens} tokens of free text");
}

/// The two tokens of `text` that end at `end`, as a tag: `None` when the pair
/// is in the text before, or holds a blank (a trigger has none).
fn tag_at(text: &[char], end: usize) -> Option<String> {
    let pair = [text[end - 1], text[end]];
    let earlier = text[..end].windows(2).any(|w| w == pair);
    (!earlier && !pair.contains(&' ')).then(|| pair.iter().collect())
}

/// The format of Qwen3.8's template with blanks for its line breaks, and
/// with `trigger` where it has `<tool_call>`.
fn format(trigger: &str) -> ToolFormat {
    ToolFormat {
        call_open: format!("{trigger} <function="),
        name_close: "> ".into(),
        param_open: "<parameter=".into(),
        key_close: "> ".into(),
        param_close: " </parameter> ".into(),
        call_close: "</function> </tool_call>".into(),
        before_calls: "  ".into(),
        between_calls: " ".into(),
    }
}

/// The two kinds of rule that leave a part of the reply free.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// A grammar that binds after the tag that closes the reasoning.
    Reasoning,
    /// One call of a tool after a trigger, as `tool_choice: "auto"` asks.
    Trigger,
}

/// The rule of `kind` that binds after `tag`.
fn rule(kind: Kind, tag: &str) -> Structure {
    match kind {
        Kind::Reasoning => Structure {
            shape: Shape::Gbnf("root ::= [x-z]{5}".into()),
            start: Start::AfterReasoning { open: None, close: tag.to_owned() },
            trigger: None,
            interrupt: None,
        },
        Kind::Trigger => {
            let format = format(tag);
            let level = json!({
                "type": "object",
                "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 99}},
                "required": ["level"],
            });
            let tools = [
                Callable { name: "set_volume", parameters: Some(&level) },
                Callable { name: "mute", parameters: None },
            ];
            let rule =
                CallRule { format: &format, tools: &tools, parallel: false, triggered: true };
            Structure {
                shape: Shape::Lark(rule.lark().unwrap()),
                start: Start::Reply,
                trigger: Some(format.trigger().to_owned()),
                interrupt: None,
            }
        }
    }
}

fn ruled(prompt: &str, structure: &Structure, draft: bool) -> GenerateRequest {
    let mut r = request(prompt, Some(700));
    r.structure = Some(structure.clone());
    r.draft = draft;
    r
}

/// What the window that gave the binding token did.
#[derive(Debug)]
struct Binding {
    /// The window's rows, and which of its picks the binding token was: 0 is
    /// the pick of the newest token's row, `k` the pick of the row of draft
    /// `k - 1`.
    rows: usize,
    pick: usize,
    /// The window's drafts were the draft block's.
    own: bool,
    /// The draft block proposed the binding token itself, and was right in
    /// everything before it: the row after the binding token was verified
    /// with the token it should have, and without the mask it should have.
    proposed: bool,
}

/// Checks the windows of a reply whose binding token is the token at `at` of
/// the request's `tokens`, the second token of `tag`, and returns what the
/// window that gave it did.
fn check_windows(windows: &[Window], tokens: &[u32], tag: &[u32], at: usize) -> Binding {
    // The window that gave the binding token is the last that began before it.
    let giving = windows.iter().rfind(|w| w.at < at).expect("a window before the binding token");
    let pick = at - giving.at - 1;
    assert!(pick < giving.tokens.len(), "the binding token came from {giving:?}");
    // Never beyond: the binding token is the newest token of a window, whose
    // first row has the mask the rule gives it.
    let next = windows.iter().find(|w| w.at == at).unwrap_or_else(|| {
        panic!("no window began at the binding token {at}: a window was accepted across it")
    });
    assert!(next.masked[0], "the row of the binding token has no mask: {next:?}");
    for w in windows {
        if w.at >= at {
            // The rule is bound: the drafts are the host's, and every row has
            // the mask of its path.
            assert!(!w.own && w.masked.iter().all(|m| *m), "bound, at {at}: {w:?}");
        } else {
            // Free rows have no mask. With the host's drafts a row whose path
            // (the reply and the drafts before the row) has the tag has its
            // mask; with the model's own drafts no row has one.
            for (k, masked) in w.masked.iter().enumerate() {
                let path = [&tokens[..=w.at], &w.tokens[1..=k]].concat();
                let bound = path.windows(2).any(|pair| pair == tag);
                assert_eq!(*masked, bound && !w.own, "row {k}, binding at {at}: {w:?}");
            }
        }
    }
    let right = |w: &Window, k: usize| tokens.get(w.at + k) == Some(&w.tokens[k]);
    Binding {
        rows: giving.tokens.len(),
        pick,
        own: giving.own,
        proposed: giving.own
            && pick + 1 < giving.tokens.len()
            && (1..=pick + 1).all(|k| right(giving, k)),
    }
}

/// The reply with the rule of `kind` bound after `tag`, which the free text
/// ends at `end`: without drafts, with them, and a second time, when the
/// lookup in earlier text knows the whole reply. Returns what the window
/// that gave the binding token did in the first run with drafts.
fn bind_at(
    model: &Arc<SyntheticModel>,
    kind: Kind,
    prompt: &str,
    text: &[char],
    tag: &str,
    end: usize,
) -> Binding {
    let structure = rule(kind, tag);
    let expected = run_one(&quiet(model), ruled(prompt, &structure, false));
    let free: String = text[..=end].iter().collect();
    let answer = expected.text.strip_prefix(&free).unwrap_or_else(|| {
        panic!("the reply does not begin with the free text: {}", expected.text)
    });
    match kind {
        Kind::Reasoning => {
            assert_eq!(expected.finish, FinishReason::Stop, "{expected:?}");
            assert!(
                answer.len() == 5 && answer.chars().all(|c| ('x'..='z').contains(&c)),
                "{answer}"
            );
        }
        Kind::Trigger => {
            assert!(
                answer.starts_with(" <function=set_volume> ")
                    || answer.starts_with(" <function=mute> "),
                "{answer}"
            );
        }
    }

    let windows = Windows::default();
    let rt = runtime(model, &windows, prompt.len());
    let drafted = run_one(&rt, ruled(prompt, &structure, true));
    assert_eq!(drafted, expected, "{kind:?} bound after {end}: drafts changed the reply");
    let tokens: Vec<u32> = prompt.bytes().map(u32::from).chain(tokens_of(&drafted.text)).collect();
    let binding =
        check_windows(&windows.lock().unwrap(), &tokens, &tokens_of(tag), prompt.len() + end);
    // Again: the lookup has the reply now, and offers it next to the block.
    let again = run_one(&rt, ruled(prompt, &structure, true));
    assert_eq!(again, expected, "{kind:?} bound after {end}: the second time");
    binding
}

#[test]
fn the_binding_token_at_every_place_of_a_window_ends_what_the_window_gives() {
    // The free text ends at each of 48 places in turn, so the token that binds
    // the rule is each pick of a window in turn.
    let ends = 40..88;
    for kind in [Kind::Reasoning, Kind::Trigger] {
        for accuracy in [&[][..], FALLIBLE] {
            let model = model(accuracy);
            let (prompt, text) = free_text(&model, ends.end + 2);
            let mut picks = BTreeSet::new();
            let (mut tried, mut proposed, mut cut) = (0, 0, 0);
            for end in ends.clone() {
                let Some(tag) = tag_at(&text, end) else { continue };
                let binding = bind_at(&model, kind, &prompt, &text, &tag, end);
                tried += 1;
                if binding.own {
                    picks.insert(binding.pick);
                    proposed += usize::from(binding.proposed);
                    cut += usize::from(binding.pick + 1 < binding.rows);
                }
            }
            assert!(tried >= 40, "{kind:?}: only {tried} places had a tag of their own");
            assert!(cut > 10, "{kind:?}: only {cut} windows had rows after the binding token");
            assert!(
                proposed > 5,
                "{kind:?}: the draft block proposed the binding token {proposed} times"
            );
            if accuracy.is_empty() {
                // A block that is always right has every window whole, so the
                // places in turn are the picks of a window in turn.
                let all: BTreeSet<usize> = (0..WINDOW as usize).collect();
                assert_eq!(picks, all, "{kind:?}: the picks the binding token was");
            }
        }
    }
}

#[test]
fn a_draft_of_the_binding_token_is_right_and_the_row_after_it_is_not_taken() {
    // One case spelled out. The draft block is always right, so a window that
    // begins three tokens before the end of the tag drafts the tag's last
    // token and the free text after it, and the model, asked without a mask,
    // confirms all of it. Only the tokens up to the tag may be taken.
    let model = model(&[]);
    let (prompt, text) = free_text(&model, 100);
    let letter = |c: char| ('x'..='z').contains(&c);
    let found =
        (40..90).filter(|end| !letter(text[end + 1])).find_map(|end| {
            let tag = tag_at(&text, end)?;
            let structure = rule(Kind::Reasoning, &tag);
            let windows = Windows::default();
            let rt = runtime(&model, &windows, prompt.len());
            let run = run_one(&rt, ruled(&prompt, &structure, true));
            let at = prompt.len() + end;
            let windows = windows.lock().unwrap();
            let giving = windows.iter().rfind(|w| w.at < at)?.clone();
            (giving.own && giving.tokens.len() == WINDOW as usize && at - giving.at == 3)
                .then_some((end, giving, run, own_accepted(&rt)))
        });
    let (end, giving, run, accepted) = found.expect("a window that begins three before the tag");
    let free: Vec<u32> = tokens_of(&text.iter().collect::<String>());
    let at = prompt.len() + end;
    // The window's drafts are the free text, through the tag and past it.
    let drafted = &giving.tokens[1..];
    let from = giving.at - prompt.len() + 1;
    assert_eq!(drafted, &free[from..from + drafted.len()], "the block drafted the free text");
    assert_eq!(giving.at + 3, at, "its third draft is the tag's last token");
    assert!(giving.masked.iter().all(|m| !m), "no row of it has a mask");
    // The reply has the tag and then the answer, not the free text.
    let (reasoning, answer) = run.text.split_at(text[..=end].iter().collect::<String>().len());
    assert_eq!(reasoning.chars().collect::<Vec<_>>(), text[..=end]);
    assert!(answer.len() == 5 && answer.chars().all(letter), "{answer}");
    assert!(accepted > 20, "{accepted} drafts of the block were accepted");
}

/// Binds the tokens of the reply whose place, counted in stretches of
/// `period`, is one of the last `bound`: those are letters from x to z. The
/// rest of the reply is free, again and again.
struct Stretches {
    period: usize,
    bound: usize,
}

impl Stretches {
    fn binds(&self, place: usize) -> bool {
        place % self.period >= self.period - self.bound
    }
}

impl TokenFilter for Stretches {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        if !self.binds(reply.len() + drafts.len()) {
            return false;
        }
        mask.fill(false);
        for token in b'x'..=b'z' {
            mask.set(u32::from(token), true);
        }
        true
    }

    fn free_rows(&mut self, reply: &[u32], tokens: &[u32]) -> usize {
        (0..=tokens.len()).take_while(|k| !self.binds(reply.len() + k)).count()
    }
}

#[test]
fn the_models_drafts_come_back_when_a_rule_is_free_again() {
    // Free stretches of every length from 6 to 24 tokens between bound ones of
    // 3, so that the free stretches end at every place of a window.
    for accuracy in [&[][..], FALLIBLE] {
        // No token ends a reply here: every reply is 200 tokens.
        let model = model_ending(accuracy, None);
        let prompt = "free and bound in turn";
        for period in 9..=27 {
            let filter = || Stretches { period, bound: 3 };
            let mut plain = request(prompt, Some(200));
            plain.draft = false;
            let expected = start_filtered(&quiet(&model), plain, filter()).unwrap().finish();
            let letters = expected.text.chars().filter(|c| ('x'..='z').contains(c)).count();
            assert!(letters >= 3 * (expected.text.chars().count() / period), "{}", expected.text);

            let windows = Windows::default();
            let rt = runtime(&model, &windows, prompt.len());
            let drafted =
                start_filtered(&rt, request(prompt, Some(200)), filter()).unwrap().finish();
            assert_eq!(drafted, expected, "stretches of {period}");
            // The block drafted in the free stretches after the first bound
            // one, and never for a window whose first row is bound.
            let windows = windows.lock().unwrap();
            let place = |w: &Window| w.at + 1 - prompt.len();
            let later = windows.iter().filter(|w| w.own && place(w) > period).count();
            assert!(
                later >= 3,
                "stretches of {period}: {later} windows of the block after the first"
            );
            for w in windows.iter() {
                let bound = filter().binds(place(w));
                assert_eq!(w.masked[0], bound, "stretches of {period}: {w:?}");
                assert!(!(w.own && bound), "stretches of {period}: {w:?}");
            }
            assert!(own_accepted(&rt) > 20, "stretches of {period}: {}", own_accepted(&rt));
        }
    }
}

#[test]
fn a_reply_with_a_long_free_part_takes_as_many_of_the_models_drafts_as_one_without_a_rule() {
    // 300 tokens of reasoning or of text before a call, then a few bound
    // tokens: what most replies with tools are.
    let model = model(&[0.9]);
    let (prompt, text) = free_text(&model, 320);
    let end = (280..310).rev().find(|end| tag_at(&text, *end).is_some()).expect("a tag");
    let tag = tag_at(&text, end).unwrap();
    // The same free text without a rule: the reply cut after the tag.
    let plain = quiet(&model);
    let run = run_one(&plain, request(&prompt, Some(end as u32 + 1)));
    assert_eq!(run.text.chars().collect::<Vec<_>>(), text[..=end]);
    let (without, steps_without) = (own_accepted(&plain), snapshot(&plain).counters.steps);
    assert!(without > 180, "the block drafts most of the free text: {without} of {end} tokens");
    for kind in [Kind::Reasoning, Kind::Trigger] {
        let structure = rule(kind, &tag);
        let slow = quiet(&model);
        let expected = run_one(&slow, ruled(&prompt, &structure, false));
        let rt = quiet(&model);
        let drafted = run_one(&rt, ruled(&prompt, &structure, true));
        assert_eq!(drafted, expected, "{kind:?}");
        let with = own_accepted(&rt);
        let counters = snapshot(&rt).counters;
        let bound = u64::from(drafted.usage.completion_tokens) - (end as u64 + 1);
        eprintln!(
            "{kind:?}: {} free tokens and {bound} bound ones. Drafts of the block accepted: \
             {with} with the rule, {without} without one. Steps: {} with the rule ({} without \
             drafts), {steps_without} for the free text without a rule",
            end + 1,
            counters.steps,
            snapshot(&slow).counters.steps,
        );
        // The windows are the same up to the one that holds the tag, which
        // the request without a rule cuts short and the one with it verifies
        // whole: a window's worth of difference at most.
        assert!(
            with + u64::from(WINDOW) >= without && with <= without + u64::from(WINDOW),
            "{kind:?}: {with} drafts of the block accepted with the rule, {without} without"
        );
        // And the reply takes about the steps of its free text, not a step a token.
        assert!(
            counters.steps <= steps_without + bound + 2,
            "{kind:?}: {} steps for {bound} bound tokens after {steps_without}",
            counters.steps
        );
    }
}
