//! What a client asks the engine to decide.
//!
//! A decision answers typed questions about a text by choosing among fixed
//! options, with no generation. The shape follows the System One API of
//! TypeSafe, as llama.cpp serves it on `POST /v1/systemone`: a `state`, and
//! questions of three types (`choice`, `score`, `noul`), each with its own
//! criteria. Everything here is independent of HTTP and of JSON, so another
//! front end could use it, and the server's parsing is the only place that
//! knows the wire spelling.

use serde::{Deserialize, Serialize};

use crate::{EngineError, Prompt, RequestId};

/// The fewest levels a [`QuestionKind::Score`] question can have.
pub const MIN_SCORE_LEVELS: usize = 2;
/// The most levels a [`QuestionKind::Score`] question can have. The TypeSafe
/// API fixes this for every model.
pub const MAX_SCORE_LEVELS: usize = 10;

/// One request to answer questions by choosing options.
///
/// Every model can serve it ("Jev mode"): the engine scores the options from
/// the logits after the prompt and generates nothing. See
/// [`Engine::decide`](crate::Engine::decide).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Which model to run, as listed by [`Engine::models`](crate::Engine::models).
    pub model: String,
    /// What the questions are about (`state` on the wire). A string state is
    /// [`Prompt::Text`] as it is; the server turns an object or an array into
    /// [`Prompt::Text`] holding its compact JSON, which is what the wire says
    /// the model reads. [`Prompt::Chat`] is for a caller that has a real
    /// conversation: the engine renders it with the chat template and puts
    /// the questions after it.
    pub prompt: Prompt,
    /// In the order the client wrote them. Questions are answered
    /// independently: no answer depends on another question.
    pub questions: Vec<Question>,
    pub id: RequestId,
}

/// One question about the prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Question {
    /// Names the question in the answer. Unique within a request.
    pub id: String,
    /// What is asked. The server turns an object or an array into its compact
    /// JSON, so this is always text.
    pub instructions: String,
    pub kind: QuestionKind,
}

/// What kind of answer a [`Question`] wants, and the options it has.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuestionKind {
    /// Pick one of the options (`choice`).
    Choice {
        /// In the order the client wrote them. At least one; keys are
        /// unique and not empty.
        options: Vec<ChoiceOption>,
    },
    /// Rate on an ordered scale (`score`): [`MIN_SCORE_LEVELS`] to
    /// [`MAX_SCORE_LEVELS`] levels, lowest first. Level `i` is the option
    /// `"i"`. The description of a level is `None` when the client sent
    /// `null`.
    Score { levels: Vec<Option<String>> },
    /// Say whether something is true (`noul`). The two options are `"false"`
    /// and `"true"`, and the client may describe either.
    Noul { when_false: Option<String>, when_true: Option<String> },
}

/// One option of a [`QuestionKind::Choice`] question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    /// What the option is called. The engine scores this text, so a key is
    /// never empty.
    pub key: String,
    /// What the option means, for the model to read. `None` when the client
    /// sent `null` or nothing.
    pub description: Option<String>,
}

impl Question {
    /// The text each option is scored as, in the order of the options:
    /// the keys of a choice, `"0"`, `"1"`, ... for the levels of a score, and
    /// `"false"`, `"true"` for a noul question.
    ///
    /// [`answer`](Question::answer) takes one score per entry, in this order.
    pub fn option_texts(&self) -> Vec<String> {
        match &self.kind {
            QuestionKind::Choice { options } => options.iter().map(|o| o.key.clone()).collect(),
            QuestionKind::Score { levels } => (0..levels.len()).map(|i| i.to_string()).collect(),
            QuestionKind::Noul { .. } => vec!["false".to_owned(), "true".to_owned()],
        }
    }

    /// How many options the question has, which is how many rows the engine
    /// has to score for it.
    pub fn n_options(&self) -> usize {
        match &self.kind {
            QuestionKind::Choice { options } => options.len(),
            QuestionKind::Score { levels } => levels.len(),
            QuestionKind::Noul { .. } => 2,
        }
    }
}

/// A decision request that cannot be served as written.
///
/// It names the field the way the client wrote it (`questions.route.criteria`)
/// so the request can be fixed without reading the engine's source.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field} {problem}")]
pub struct DecisionError {
    pub field: String,
    /// Starts with a verb ("must", "lists") so that it reads after the field.
    pub problem: String,
}

impl DecisionError {
    fn new(field: impl Into<String>, problem: impl Into<String>) -> Self {
        Self { field: field.into(), problem: problem.into() }
    }
}

impl From<DecisionError> for EngineError {
    fn from(error: DecisionError) -> Self {
        EngineError::InvalidRequest(error.to_string())
    }
}

impl DecisionRequest {
    /// Checks what no engine can answer: no questions, a repeated question id,
    /// a choice without options or with an empty or repeated key, a score
    /// scale outside [`MIN_SCORE_LEVELS`] to [`MAX_SCORE_LEVELS`].
    ///
    /// Repeated ids and keys can only reach here from a client that wrote
    /// the same member twice. Most parsers keep one of them silently; refusing
    /// is better than answering a question the client did not mean.
    pub fn validate(&self) -> Result<(), DecisionError> {
        if self.questions.is_empty() {
            return Err(DecisionError::new("questions", "must not be empty"));
        }
        for (i, question) in self.questions.iter().enumerate() {
            let id = &question.id;
            if self.questions[..i].iter().any(|earlier| earlier.id == *id) {
                return Err(DecisionError::new(
                    format!("questions.{id}"),
                    "is listed twice: question ids must be unique",
                ));
            }
            let criteria = format!("questions.{id}.criteria");
            match &question.kind {
                QuestionKind::Choice { options } => check_options(criteria, options)?,
                QuestionKind::Score { levels } => {
                    if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&levels.len()) {
                        return Err(DecisionError::new(
                            criteria,
                            format!(
                                "must have {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, \
                                 got {}",
                                levels.len()
                            ),
                        ));
                    }
                }
                QuestionKind::Noul { .. } => {}
            }
        }
        Ok(())
    }
}

fn check_options(criteria: String, options: &[ChoiceOption]) -> Result<(), DecisionError> {
    if options.is_empty() {
        return Err(DecisionError::new(criteria, "must list at least one option"));
    }
    for (i, option) in options.iter().enumerate() {
        if option.key.is_empty() {
            return Err(DecisionError::new(
                criteria,
                format!(
                    "has an empty option key (option {} of {}): an option is scored by its \
                     text, so every key needs some",
                    i + 1,
                    options.len()
                ),
            ));
        }
        if options[..i].iter().any(|earlier| earlier.key == option.key) {
            return Err(DecisionError::new(
                criteria,
                format!("lists the option '{}' twice: option keys must be unique", option.key),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(id: &str, keys: &[&str]) -> Question {
        Question {
            id: id.into(),
            instructions: "which?".into(),
            kind: QuestionKind::Choice {
                options: keys
                    .iter()
                    .map(|k| ChoiceOption { key: (*k).into(), description: None })
                    .collect(),
            },
        }
    }

    fn score(id: &str, levels: usize) -> Question {
        Question {
            id: id.into(),
            instructions: "how much?".into(),
            kind: QuestionKind::Score { levels: vec![None; levels] },
        }
    }

    fn noul(id: &str) -> Question {
        Question {
            id: id.into(),
            instructions: "is it?".into(),
            kind: QuestionKind::Noul { when_false: None, when_true: Some("yes, it is".into()) },
        }
    }

    fn request(questions: Vec<Question>) -> DecisionRequest {
        DecisionRequest {
            model: "m".into(),
            prompt: Prompt::Text("state".into()),
            questions,
            id: RequestId(1),
        }
    }

    #[test]
    fn each_kind_of_question_is_valid_at_its_edges() {
        let valid = request(vec![
            choice("one", &["only"]),
            choice("many", &["a", "b", "c"]),
            score("least", MIN_SCORE_LEVELS),
            score("most", MAX_SCORE_LEVELS),
            noul("flag"),
        ]);
        assert_eq!(valid.validate(), Ok(()));
    }

    #[test]
    fn every_bad_request_is_refused_by_naming_its_field() {
        let cases = [
            ("questions", request(vec![]), "must not be empty"),
            ("questions.a", request(vec![choice("a", &["x"]), noul("a")]), "is listed twice"),
            ("questions.a.criteria", request(vec![choice("a", &[])]), "at least one option"),
            (
                "questions.a.criteria",
                request(vec![choice("a", &["x", "", "z"])]),
                "empty option key (option 2 of 3)",
            ),
            (
                "questions.a.criteria",
                request(vec![choice("a", &["x", "y", "x"])]),
                "lists the option 'x' twice",
            ),
            (
                "questions.a.criteria",
                request(vec![score("a", MIN_SCORE_LEVELS - 1)]),
                "2 to 10 levels, got 1",
            ),
            (
                "questions.a.criteria",
                request(vec![score("a", MAX_SCORE_LEVELS + 1)]),
                "2 to 10 levels, got 11",
            ),
        ];
        for (field, request, words) in cases {
            let error = request.validate().expect_err(field);
            assert_eq!(error.field, field, "{error}");
            assert!(error.to_string().starts_with(field), "{error}");
            assert!(error.problem.contains(words), "'{}' should contain '{words}'", error.problem);
        }
    }

    #[test]
    fn the_message_reads_as_a_sentence_and_becomes_an_invalid_request() {
        let error = request(vec![score("urgency", 11)]).validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "questions.urgency.criteria must have 2 to 10 levels, got 11"
        );
        assert_eq!(
            EngineError::from(error.clone()),
            EngineError::InvalidRequest(error.to_string())
        );
    }

    #[test]
    fn the_first_bad_question_is_reported() {
        let bad = request(vec![choice("first", &[]), score("second", 1)]);
        assert_eq!(bad.validate().unwrap_err().field, "questions.first.criteria");
    }

    #[test]
    fn options_are_scored_as_their_text_in_order() {
        assert_eq!(choice("a", &["billing", "shipping"]).option_texts(), ["billing", "shipping"]);
        assert_eq!(score("a", 4).option_texts(), ["0", "1", "2", "3"]);
        assert_eq!(noul("a").option_texts(), ["false", "true"]);
        for question in [choice("a", &["x", "y", "z"]), score("b", 5), noul("c")] {
            assert_eq!(question.n_options(), question.option_texts().len());
        }
    }

    #[test]
    fn a_request_survives_serialization() {
        let request = request(vec![
            Question {
                id: "route".into(),
                instructions: "Which team?".into(),
                kind: QuestionKind::Choice {
                    options: vec![
                        ChoiceOption { key: "billing".into(), description: None },
                        ChoiceOption {
                            key: "shipping".into(),
                            description: Some("parcels".into()),
                        },
                    ],
                },
            },
            score("urgency", 4),
            noul("angry"),
        ]);
        let text = serde_json::to_string(&request).unwrap();
        assert_eq!(serde_json::from_str::<DecisionRequest>(&text).unwrap(), request);
    }
}
