//! What a client asks the engine to decide, and how a chat model is asked it.
//!
//! A decision answers typed questions about a state by choosing among fixed
//! options, with no generation ("Jev mode"). The request is the System One API
//! as Flide serves it on `POST /v1/systemone`: a `state` (any JSON) and
//! questions of three types (`choice`, `score`, `noul`), each with its
//! instructions and criteria (any JSON). Everything here is independent of
//! HTTP and of JSON parsing, so another front end could use it; the server's
//! parsing is the only place that knows the wire spelling.
//!
//! A model that was not trained to decide is asked the way Flide's
//! `next_token` format asks a chat model, in its default style (`vllm`, the
//! rendering of vLLM's `/v1/systemone`): every question is a prompt of its own,
//! one user turn holding the state, a blank line and the question with its
//! options labeled, thinking off, and the answer is read at the first token of
//! the reply, over the tokens of the question's labels
//! ([`Question::render`], [`Question::answer`]). The questions of a request
//! share the state as the start of their prompts, so the engine computes it
//! once and forks it into the questions.
//!
//! Ported from Flide (<https://github.com/OmniJev/Flide>, Apache-2.0), commit
//! caffe660df88db747d6da9f95f7d6759fea38698: the limits and checks of
//! `flide/api/schema.py`, and the rendering and the labels of
//! `flide/formats/next_token.py` (`NextToken._question_vllm`,
//! `NextToken._render`, `VLLM_LABELS`, `VLLM_REPLY`, `NextToken._pairs`, the
//! limits `max_branch_tokens` and `max_request_tokens`).

use serde::{Deserialize, Serialize};

use crate::python;
use crate::{EngineError, Json, RequestId};

/// The most options a [`QuestionKind::Choice`] question and the most levels a
/// [`QuestionKind::Score`] question can have (Flide's `MAX_CHOICE_OPTIONS` and
/// `MAX_SCORE_LEVELS`). A model may read fewer: a chat model reads as many as
/// it has labels for ([`Labels::most`]).
pub const MAX_OPTIONS: usize = 256;

/// The most tokens of the shared start of a request's prompts and the longest
/// question's rest together (Flide's `max_branch_tokens`).
pub const MAX_BRANCH_TOKENS: usize = 32_768;

/// The most tokens of the shared start and every question's rest together
/// (Flide's `max_request_tokens`).
pub const MAX_REQUEST_TOKENS: usize = 65_536;

/// One request to answer questions by choosing options.
///
/// Every model can serve it: the engine reads the options' labels from the
/// logits after each question's prompt and generates nothing. See
/// [`Engine::decide`](crate::Engine::decide).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Which model to run, as listed by [`Engine::models`](crate::Engine::models).
    pub model: String,
    /// What the questions are about, as the client sent it: text, any other
    /// JSON value, or [`Json::Null`] when the client sent none.
    pub state: Json,
    /// In the order the client wrote them. Questions are answered
    /// independently: no answer depends on another question.
    pub questions: Vec<Question>,
    /// Who goes first when the engine is busy, as for a generation
    /// ([`GenerateRequest::priority`](crate::GenerateRequest::priority)).
    #[serde(default)]
    pub priority: i32,
    pub id: RequestId,
}

/// One question about the state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// Names the question in the answer. Unique within a request.
    pub id: String,
    /// What is asked, as the client sent it; [`Json::Null`] when it sent
    /// none.
    pub instructions: Json,
    pub kind: QuestionKind,
}

/// What kind of answer a [`Question`] wants, and the options it has.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuestionKind {
    /// Pick one of the options (`choice`): 1 to [`MAX_OPTIONS`], in the order
    /// the client wrote them.
    Choice { options: Vec<ChoiceOption> },
    /// Rate on an ordered scale (`score`): 1 to [`MAX_OPTIONS`] levels,
    /// lowest first, each as the client described it.
    Score { levels: Vec<Json> },
    /// Say whether something is true (`noul`). The descriptions of the two
    /// outcomes, [`Json::Null`] when the client gave none.
    Noul { when_true: Json, when_false: Json },
}

/// One option of a [`QuestionKind::Choice`] question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    /// What the option is called: the answer names it.
    pub key: String,
    /// What the option means, for the model to read; [`Json::Null`] for none.
    pub description: Json,
}

impl Question {
    /// How many options the question has: what the engine reads a label for.
    pub fn n_options(&self) -> usize {
        match &self.kind {
            QuestionKind::Choice { options } => options.len(),
            QuestionKind::Score { levels } => levels.len(),
            QuestionKind::Noul { .. } => 2,
        }
    }

    /// The name of the question's type on the wire.
    pub fn type_name(&self) -> &'static str {
        match &self.kind {
            QuestionKind::Choice { .. } => "choice",
            QuestionKind::Score { .. } => "score",
            QuestionKind::Noul { .. } => "noul",
        }
    }
}

/// A decision request that cannot be served as written.
///
/// `at` names the place in the request the way the client wrote it
/// (`["questions", "route", "criteria"]`), so the request can be fixed without
/// reading the engine's source; the message is Flide's.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}: {message}", at.join("."))]
pub struct DecisionError {
    pub at: Vec<String>,
    pub message: String,
}

impl DecisionError {
    pub fn new(at: &[&str], message: impl Into<String>) -> Self {
        Self { at: at.iter().map(|s| (*s).to_owned()).collect(), message: message.into() }
    }
}

impl From<DecisionError> for EngineError {
    fn from(error: DecisionError) -> Self {
        EngineError::InvalidRequest(error.to_string())
    }
}

impl DecisionRequest {
    /// Checks what Flide's schema refuses: no questions, a choice without
    /// options or with more than [`MAX_OPTIONS`], a score without levels or
    /// with more than [`MAX_OPTIONS`].
    pub fn validate(&self) -> Result<(), DecisionError> {
        if self.questions.is_empty() {
            return Err(DecisionError::new(&["questions"], "at least one question is required"));
        }
        for question in &self.questions {
            let at = ["questions", question.id.as_str(), "criteria"];
            let (n, what, unit) = match &question.kind {
                QuestionKind::Choice { options } => (options.len(), "choice", "option"),
                QuestionKind::Score { levels } => (levels.len(), "score", "level"),
                QuestionKind::Noul { .. } => continue,
            };
            if n == 0 {
                return Err(DecisionError::new(
                    &at,
                    format!("{what} question needs at least one {unit} in criteria"),
                ));
            }
            if n > MAX_OPTIONS {
                return Err(DecisionError::new(
                    &at,
                    format!("{what} question has {n} {unit}s, maximum is {MAX_OPTIONS}"),
                ));
            }
        }
        Ok(())
    }
}

/// The text a question is labeled with: A to Z for the options of a choice,
/// `yes` and `no` for a noul, 0 to 9 for the levels of a score, in this order
/// ([`Labels::singles`]). Flide's `VLLM_LABELS`.
pub const SINGLE_LABELS: [&str; 38] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z", "yes", "no", "0", "1", "2", "3", "4", "5", "6", "7", "8",
    "9",
];

/// Where the letters, `yes` and the digits start in [`SINGLE_LABELS`].
const LETTERS: usize = 26;
const YES: usize = 26;
const DIGITS: usize = 28;

/// The labels a chat model reads the options of a question by, with the token
/// each one is where the reply starts (after the chat template's generation
/// prompt), as Flide's `next_token` format finds them in a tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labels {
    /// The token of each of [`SINGLE_LABELS`], in that order: every one is a
    /// single token there, and no two are the same.
    pub singles: Vec<u32>,
    /// The labels after Z for a question with more options than letters:
    /// `AA`, `AB`, ... in that order, those that are one token where the
    /// reply starts and differ from every other label's, up to
    /// [`Labels::MAX_PAIRS`].
    pub pairs: Vec<(String, u32)>,
}

impl Labels {
    /// The most pairs a model keeps: 256 labels with the letters.
    pub const MAX_PAIRS: usize = MAX_OPTIONS - LETTERS;

    /// The two-letter labels in the order they are tried, as `NextToken._pairs`
    /// walks them.
    pub fn pair_candidates() -> impl Iterator<Item = String> {
        let letters = || (b'A'..=b'Z').map(char::from);
        letters().flat_map(move |a| letters().map(move |b| format!("{a}{b}")))
    }

    /// The most options a choice (and levels a score) can have with these
    /// labels.
    pub fn most(&self) -> usize {
        (LETTERS + self.pairs.len()).min(MAX_OPTIONS)
    }

    /// The labels of the letters and then the pairs: what a question with more
    /// options than its single labels reads.
    fn wide(&self) -> impl Iterator<Item = (&str, u32)> {
        SINGLE_LABELS[..LETTERS]
            .iter()
            .copied()
            .zip(self.singles[..LETTERS].iter().copied())
            .chain(self.pairs.iter().map(|(label, token)| (label.as_str(), *token)))
    }
}

/// A question as a chat model is asked it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    /// The question's part of the user turn, after the state and a blank line.
    pub text: String,
    /// The token of each option's label, in the order of the options
    /// ([`Question::answer`] takes their logits in this order): `[yes, no]`
    /// for a noul.
    pub tokens: Vec<u32>,
}

impl Question {
    /// The question as Flide's `next_token` format writes it for a chat model
    /// (`NextToken._question_vllm`, style `vllm`): "Question: " and the
    /// instructions when there are any, a line per option ("A: key -
    /// description", "A: key" without one; "yes: description" or "yes" for a
    /// noul; "0: level" for the levels of a score), and the instruction to
    /// answer with a label. A choice of more than 26 options, and a score of
    /// more than 10 levels, read letters and then two-letter labels.
    ///
    /// Refuses a question with more options than `labels` has labels for.
    pub fn render(&self, labels: &Labels) -> Result<Rendered, DecisionError> {
        // The option names and, for a choice and a noul, their descriptions.
        let (names, descriptions): (Vec<String>, Vec<String>) = match &self.kind {
            QuestionKind::Noul { when_true, when_false } => {
                (vec!["yes".into(), "no".into()], vec![plain(when_true), plain(when_false)])
            }
            QuestionKind::Choice { options } => (
                options.iter().map(|o| o.key.clone()).collect(),
                options.iter().map(|o| plain(&o.description)).collect(),
            ),
            // A level is shown as Python's `str()` writes it (`None` for
            // null), as vLLM does; it has no description.
            QuestionKind::Score { levels } => {
                (levels.iter().map(python::str_of).collect(), Vec::new())
            }
        };
        let n = names.len();
        let (singles, mut reply) = match &self.kind {
            QuestionKind::Choice { .. } => {
                (0..LETTERS, "Answer with the letter of one option only.".to_owned())
            }
            QuestionKind::Noul { .. } => (YES..YES + 2, "Answer with yes or no only.".to_owned()),
            QuestionKind::Score { .. } => {
                (DIGITS..DIGITS + 10, "Answer with the number of one level only.".to_owned())
            }
        };
        let noul = matches!(self.kind, QuestionKind::Noul { .. });
        let wide = !noul && n > singles.len();
        let chosen: Vec<(&str, u32)> = if wide {
            let word =
                if matches!(self.kind, QuestionKind::Choice { .. }) { "option" } else { "level" };
            let letter = if n <= LETTERS { "letter" } else { "label" };
            reply = format!("Answer with the {letter} of one {word} only.");
            labels.wide().take(n).collect()
        } else {
            singles.clone().map(|i| (SINGLE_LABELS[i], labels.singles[i])).collect()
        };
        if chosen.len() < n {
            let kind = self.type_name();
            return Err(DecisionError::new(
                &["questions", &self.id, "criteria"],
                format!("{n} options; a {kind} reads at most {}", chosen.len()),
            ));
        }
        let instructions = plain(&self.instructions);
        let mut lines: Vec<String> = Vec::with_capacity(n + 2);
        if !instructions.is_empty() {
            lines.push(format!("Question: {instructions}"));
        }
        for (i, (label, _)) in chosen.iter().take(n).enumerate() {
            let name = &names[i];
            let description = descriptions.get(i).map_or("", String::as_str);
            lines.push(match (noul, description.is_empty()) {
                (true, true) => (*label).to_owned(),
                (true, false) => format!("{label}: {description}"),
                (false, true) => format!("{label}: {name}"),
                (false, false) => format!("{label}: {name} - {description}"),
            });
        }
        lines.push(reply);
        Ok(Rendered {
            text: lines.join("\n"),
            tokens: chosen.iter().take(n).map(|(_, token)| *token).collect(),
        })
    }
}

/// An instruction or a description as Flide's `NextToken._plain` writes it in
/// style `vllm`: nothing for null, else Python's `str()` of the value.
fn plain(value: &Json) -> String {
    if value.is_null() { String::new() } else { python::str_of(value) }
}

/// The state as the user turn shows it (`NextToken._render`, style `vllm`):
/// text as it is, any other value (null included) as Python's
/// `json.dumps(state, ensure_ascii=False)` writes it.
pub fn state_text(state: &Json) -> String {
    match state {
        Json::String(text) => text.clone(),
        other => python::dumps(other),
    }
}

/// The user turn of a question's prompt: the state, a blank line, the
/// question ([`Question::render`]).
pub fn user_turn(state: &Json, question: &str) -> String {
    let mut text = state_text(state);
    text.push_str("\n\n");
    text.push_str(question);
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        Labels {
            singles: (0..38).map(|i| 100 + i).collect(),
            pairs: Labels::pair_candidates()
                .take(Labels::MAX_PAIRS)
                .enumerate()
                .map(|(i, p)| (p, 1000 + i as u32))
                .collect(),
        }
    }

    fn text(s: &str) -> Json {
        Json::String(s.into())
    }

    fn choice(id: &str, keys: &[(&str, Json)]) -> Question {
        Question {
            id: id.into(),
            instructions: text("Which team?"),
            kind: QuestionKind::Choice {
                options: keys
                    .iter()
                    .map(|(k, d)| ChoiceOption { key: (*k).into(), description: d.clone() })
                    .collect(),
            },
        }
    }

    fn keys(n: usize) -> Vec<(String, Json)> {
        (0..n).map(|i| (format!("k{i}"), Json::Null)).collect()
    }

    fn choice_of(id: &str, keys: &[(String, Json)]) -> Question {
        let keys: Vec<(&str, Json)> = keys.iter().map(|(k, d)| (k.as_str(), d.clone())).collect();
        choice(id, &keys)
    }

    fn score(id: &str, levels: Vec<Json>) -> Question {
        Question {
            id: id.into(),
            instructions: text("How urgent?"),
            kind: QuestionKind::Score { levels },
        }
    }

    fn noul(id: &str, yes: Json, no: Json) -> Question {
        Question {
            id: id.into(),
            instructions: text("Is the sender angry?"),
            kind: QuestionKind::Noul { when_true: yes, when_false: no },
        }
    }

    fn request(questions: Vec<Question>) -> DecisionRequest {
        DecisionRequest {
            model: "m".into(),
            state: text("state"),
            questions,
            priority: 0,
            id: RequestId(1),
        }
    }

    #[test]
    fn a_choice_is_written_as_vllm_writes_it() {
        let q = choice("route", &[("billing", text("money")), ("shipping", Json::Null)]);
        let r = q.render(&labels()).unwrap();
        assert_eq!(
            r.text,
            "Question: Which team?\nA: billing - money\nB: shipping\nAnswer with the letter of one \
             option only."
        );
        assert_eq!(r.tokens, [100, 101]);
    }

    #[test]
    fn a_noul_lists_yes_and_no_and_reads_them_in_that_order() {
        let r = noul("angry", text("they are"), Json::Null).render(&labels()).unwrap();
        assert_eq!(
            r.text,
            "Question: Is the sender angry?\nyes: they are\nno\nAnswer with yes or no only."
        );
        assert_eq!(r.tokens, [126, 127]);
    }

    #[test]
    fn a_score_shows_its_levels_as_python_writes_them() {
        let levels = vec![text("low"), Json::Null, Json::Int(3), Json::Bool(true)];
        let r = score("urgency", levels).render(&labels()).unwrap();
        assert_eq!(
            r.text,
            "Question: How urgent?\n0: low\n1: None\n2: 3\n3: True\nAnswer with the number of \
             one level only."
        );
        assert_eq!(r.tokens, [128, 129, 130, 131]);
    }

    #[test]
    fn wide_questions_read_letters_then_pairs() {
        let r = score("s", vec![Json::Int(1); 11]).render(&labels()).unwrap();
        assert!(r.text.ends_with("K: 1\nAnswer with the letter of one level only."), "{}", r.text);
        assert_eq!(r.tokens, (100..111).collect::<Vec<_>>());
        let r = choice_of("c", &keys(30)).render(&labels()).unwrap();
        assert!(r.text.contains("\nZ: k25\nAA: k26\nAB: k27\n"), "{}", r.text);
        assert!(r.text.ends_with("Answer with the label of one option only."));
        assert_eq!(r.tokens[26..], [1000, 1001, 1002, 1003]);
        assert_eq!(choice_of("c", &keys(256)).render(&labels()).unwrap().tokens.len(), 256);
    }

    #[test]
    fn a_question_without_instructions_has_no_question_line() {
        let mut q = noul("n", Json::Null, Json::Null);
        q.instructions = Json::Null;
        assert_eq!(q.render(&labels()).unwrap().text, "yes\nno\nAnswer with yes or no only.");
    }

    #[test]
    fn more_options_than_labels_are_refused() {
        let few = Labels { singles: labels().singles, pairs: vec![("AA".into(), 999)] };
        let error = choice_of("big", &keys(28)).render(&few).unwrap_err();
        assert_eq!(
            error.to_string(),
            "questions.big.criteria: 28 options; a choice reads at most 27"
        );
        assert_eq!(few.most(), 27);
        assert_eq!(labels().most(), 256);
    }

    #[test]
    fn the_user_turn_holds_the_state_as_python_dumps_it() {
        let state = Json::Object(vec![
            ("from".into(), text("Zoë \"x\"\n")),
            ("n".into(), Json::Array(vec![Json::Int(1), Json::Float(0.5), Json::Null])),
        ]);
        assert_eq!(
            user_turn(&state, "Q"),
            "{\"from\": \"Zoë \\\"x\\\"\\n\", \"n\": [1, 0.5, null]}\n\nQ"
        );
        assert_eq!(user_turn(&Json::Null, "Q"), "null\n\nQ");
        assert_eq!(user_turn(&text("plain"), "Q"), "plain\n\nQ");
    }

    #[test]
    fn flides_schema_checks_are_kept() {
        let cases = [
            (request(vec![]), "questions: at least one question is required"),
            (
                request(vec![choice("a", &[])]),
                "questions.a.criteria: choice question needs at least one option in criteria",
            ),
            (
                request(vec![score("s", vec![])]),
                "questions.s.criteria: score question needs at least one level in criteria",
            ),
            (
                request(vec![score("s", vec![Json::Null; 257])]),
                "questions.s.criteria: score question has 257 levels, maximum is 256",
            ),
            (
                request(vec![choice_of("c", &keys(257))]),
                "questions.c.criteria: choice question has 257 options, maximum is 256",
            ),
        ];
        for (request, message) in cases {
            assert_eq!(request.validate().unwrap_err().to_string(), message);
        }
        let fine = request(vec![
            choice("one", &[("", Json::Null)]),
            score("s", vec![Json::Null]),
            noul("n", Json::Null, Json::Null),
        ]);
        assert_eq!(fine.validate(), Ok(()));
        assert_eq!(
            EngineError::from(request(vec![]).validate().unwrap_err()),
            EngineError::InvalidRequest("questions: at least one question is required".into())
        );
    }

    #[test]
    fn a_request_survives_serialization() {
        let request = request(vec![
            choice("route", &[("billing", Json::Null), ("shipping", text("parcels"))]),
            score("urgency", vec![text("low"), text("high")]),
            noul("angry", Json::Null, text("calm")),
        ]);
        let text = serde_json::to_string(&request).unwrap();
        assert_eq!(serde_json::from_str::<DecisionRequest>(&text).unwrap(), request);
    }
}
