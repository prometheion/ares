//! The System One wire format: `POST /v1/systemone`.
//!
//! It is the TypeSafe API as llama.cpp serves it (`tools/server/README.md`,
//! "TypeSafe-compatible System One API"), so a client written for llama.cpp
//! works unchanged: a `state` and a map of `questions`, answered with a
//! map of `answers` and the token count of the prompt. Like the OpenAI side
//! this reads the request from a generic [`Json`] value, to name the field in
//! every error (`questions.route.criteria must be an object ...`).
//!
//! Any model answers here ("Jev mode"), so the options that only make sense
//! for native decision models are accepted and handled as follows:
//!
//! * **No model is needed in the request.** llama.cpp serves one model and
//!   ignores `model` (a router reads it). Here `model` selects the model as
//!   on the other endpoints, and when it is left out the only served model is
//!   used, so a single-model client works without change.
//! * **`files`, `images`, image and audio parts of the state, `videos`** are
//!   media. The engine interface carries text, so a request with images or
//!   audio is refused with `501`, as llama.cpp does for a model without image
//!   input, and `videos` with `400` as llama.cpp does.
//! * **Limits that belong to a model type** (the number of options a model
//!   can name, `strict_payload`, a state that may be `null`) do not apply.
//!   What applies is what scoring needs: a choice has at least one option,
//!   keys are not empty (the engine scores the key's text) and not repeated,
//!   and a score has 2 to 10 levels, as the TypeSafe API says for every model.
//! * **Temperatures, score biases and the order of options** are stored in
//!   the file of a native model. A language model has none: probabilities are
//!   the softmax of the options' log-likelihoods, options keep the order of
//!   the request, and `noul` options are `[false, true]`.
//! * **`noul` descriptions** are read under `"false"` and `"true"`, and under
//!   `"no"` and `"yes"` as lfm2-d1-omni and h2o-lightning also do.
//!
//! A `state`, `instructions` or description that is not text is given to the
//! model as its compact JSON, which is what the API says.

use ares_api::{
    ChoiceOption, DecisionRequest, DecisionResult, Json, ModelCard, Prompt, Question, QuestionKind,
    RequestId, Verdict,
};

use crate::error::ApiError;
use crate::wire::{self, Fields, expect_str, wrong_type};

/// A request body, understood.
#[derive(Debug)]
pub(crate) struct Parsed {
    pub request: DecisionRequest,
    /// What the response needs from the request besides the engine's answers.
    pub reply: Reply,
}

/// Parses and checks the body of a System One request.
///
/// `models` is what the engine serves, to default `model`.
pub(crate) fn parse(body: &[u8], id: RequestId, models: &[ModelCard]) -> Result<Parsed, ApiError> {
    let json = wire::read_json(body)?;
    let fields = Fields(wire::members(&json)?);

    let model = model_to_ask(&fields, models)?;
    let state = read_state(&fields)?;
    let (questions, legends) = read_questions(&fields)?;
    let has_media = media(&fields, &state)?;

    let state = match &state {
        Json::String(text) => text.clone(),
        structured => compact(structured),
    };
    let reply = Reply::new(&model, &questions, legends);
    let request = DecisionRequest { model, prompt: Prompt::Text(state), questions, id };
    request.validate().map_err(|e| {
        let message = e.to_string();
        ApiError::invalid_param(e.field, message)
    })?;

    // After the validation: a request that is wrong in other ways is told so
    // first, as llama.cpp does, and what is left is only unsupported.
    if has_media {
        return Err(ApiError::unsupported(
            "decisions with images or audio are not supported: this engine decides on text only. \
             Remove 'files', 'images' and the image_url and input_audio parts of 'state'",
        ));
    }
    Ok(Parsed { request, reply })
}

/// The model to ask. Named, or the only one that is served.
fn model_to_ask(fields: &Fields<'_>, models: &[ModelCard]) -> Result<String, ApiError> {
    if let Some(name) = fields.string("model")?.filter(|m| !m.is_empty()) {
        return Ok(name.to_owned());
    }
    match models {
        [only] => Ok(only.id.clone()),
        // Nothing is served. The engine is asked all the same: it knows why,
        // and says so (this build cannot run any model, say), where a
        // complaint about a field the client could not have filled in
        // would hide it.
        [] => Ok(String::new()),
        several => {
            let names: Vec<_> = several.iter().map(|m| m.id.as_str()).collect();
            Err(ApiError::invalid_param(
                "model",
                format!(
                    "model is required: this server serves {} models ({}), name one",
                    several.len(),
                    names.join(", ")
                ),
            ))
        }
    }
}

fn read_state(fields: &Fields<'_>) -> Result<Json, ApiError> {
    match fields.get("state") {
        None => Err(ApiError::invalid_param("state", "state is required")),
        Some(state @ (Json::String(_) | Json::Object(_) | Json::Array(_))) => Ok(state.clone()),
        Some(other) => Err(wrong_type("state", "text, an object or an array", other)),
    }
}

/// For each question, the legend of a score question: its level descriptions
/// as the client sent them.
type Legends = Vec<Option<Vec<Json>>>;

/// The questions in the order of the request, and their legends.
fn read_questions(fields: &Fields<'_>) -> Result<(Vec<Question>, Legends), ApiError> {
    match fields.get("questions") {
        None => Err(ApiError::invalid_param("questions", "questions is required")),
        Some(Json::Object(members)) => {
            let parsed: Vec<_> = members
                .iter()
                .map(|(id, question)| question_from(id, question))
                .collect::<Result<_, _>>()?;
            Ok(parsed.into_iter().unzip())
        }
        Some(other) => {
            Err(wrong_type("questions", "an object that maps a question id to a question", other))
        }
    }
}

fn question_from(id: &str, value: &Json) -> Result<(Question, Option<Vec<Json>>), ApiError> {
    let at = format!("questions.{id}");
    let Json::Object(members) = value else {
        return Err(ApiError::invalid_param(
            at.clone(),
            format!("{at} must be an object, got {}", value.kind()),
        ));
    };
    let fields = Fields(members);

    let instructions = match fields.get("instructions") {
        Some(value) => payload(&format!("{at}.instructions"), value)?,
        None => {
            let field = format!("{at}.instructions");
            return Err(ApiError::invalid_param(field.clone(), format!("{field} is required")));
        }
    };

    let criteria = format!("{at}.criteria");
    let type_field = format!("{at}.type");
    let Some(type_name) = fields.get("type") else {
        return Err(ApiError::invalid_param(
            type_field.clone(),
            format!("{type_field} is required: one of choice, score, noul"),
        ));
    };
    let (kind, legend) = match expect_str(&type_field, type_name)? {
        "choice" => {
            (QuestionKind::Choice { options: options(&criteria, fields.get("criteria"))? }, None)
        }
        "score" => {
            let (levels, legend) = levels(&criteria, fields.get("criteria"))?;
            (QuestionKind::Score { levels }, Some(legend))
        }
        "noul" => (noul(&criteria, fields.get("criteria"))?, None),
        other => {
            return Err(ApiError::invalid_param(
                type_field.clone(),
                format!("{type_field} must be one of: choice, score, noul, got '{other}'"),
            ));
        }
    };
    Ok((Question { id: id.to_owned(), instructions, kind }, legend))
}

fn options(criteria: &str, value: Option<&Json>) -> Result<Vec<ChoiceOption>, ApiError> {
    match value {
        None => Err(ApiError::invalid_param(criteria, format!("{criteria} is required"))),
        Some(Json::Object(members)) => members
            .iter()
            .map(|(key, description)| {
                Ok(ChoiceOption {
                    key: key.clone(),
                    description: description_of(&format!("{criteria}.{key}"), description)?,
                })
            })
            .collect(),
        Some(other) => {
            Err(wrong_type(criteria, "an object that maps each option to its description", other))
        }
    }
}

/// The levels as text for the model, and as the client sent them.
fn levels(
    criteria: &str,
    value: Option<&Json>,
) -> Result<(Vec<Option<String>>, Vec<Json>), ApiError> {
    match value {
        None => Err(ApiError::invalid_param(criteria, format!("{criteria} is required"))),
        Some(Json::Array(levels)) => {
            let text = levels
                .iter()
                .enumerate()
                .map(|(i, description)| description_of(&format!("{criteria}[{i}]"), description))
                .collect::<Result<_, _>>()?;
            Ok((text, levels.clone()))
        }
        Some(other) => {
            Err(wrong_type(criteria, "an array of 2 to 10 level descriptions, lowest first", other))
        }
    }
}

fn noul(criteria: &str, value: Option<&Json>) -> Result<QuestionKind, ApiError> {
    match value {
        None => Ok(QuestionKind::Noul { when_false: None, when_true: None }),
        Some(Json::Object(members)) => {
            let described = Fields(members);
            // The description of one side, under its name or its alias.
            let side = |name: &str, alias: &str| -> Result<Option<String>, ApiError> {
                let (field, value) = match described.get(name) {
                    Some(value) => (name, Some(value)),
                    None => (alias, described.get(alias)),
                };
                value.map(|value| payload(&format!("{criteria}.{field}"), value)).transpose()
            };
            Ok(QuestionKind::Noul {
                when_false: side("false", "no")?,
                when_true: side("true", "yes")?,
            })
        }
        Some(other) => {
            Err(wrong_type(criteria, "an object that describes \"true\" and \"false\"", other))
        }
    }
}

/// Text for the model: a string as it is, an object or an array as compact
/// JSON. Anything else is not a thing the API says can be read.
fn payload(field: &str, value: &Json) -> Result<String, ApiError> {
    match value {
        Json::String(text) => Ok(text.clone()),
        Json::Object(_) | Json::Array(_) => Ok(compact(value)),
        other => Err(wrong_type(field, "text, an object or an array", other)),
    }
}

/// A description, which `null` leaves out.
fn description_of(field: &str, value: &Json) -> Result<Option<String>, ApiError> {
    if value.is_null() { Ok(None) } else { payload(field, value).map(Some) }
}

fn compact(value: &Json) -> String {
    serde_json::to_string(value).expect("a JSON value serializes")
}

/// Checks the media fields, and says whether the request carries any.
fn media(fields: &Fields<'_>, state: &Json) -> Result<bool, ApiError> {
    match fields.get("videos") {
        None => {}
        Some(Json::Array(videos)) if videos.is_empty() => {}
        Some(Json::Object(videos)) if videos.is_empty() => {}
        Some(_) => return Err(ApiError::invalid_param("videos", "videos is not supported")),
    }
    let mut found = state_has_media(state);
    // `images` is an alias of `files`.
    for name in ["files", "images"] {
        match fields.get(name) {
            None => {}
            Some(Json::Array(urls)) => {
                for (i, url) in urls.iter().enumerate() {
                    if !url.as_str().is_some_and(|url| url.starts_with("data:")) {
                        return Err(ApiError::invalid_param(
                            format!("{name}[{i}]"),
                            format!(
                                "{name}[{i}] must be a data URL (data:image/...;base64,...), \
                                 got {}",
                                url.kind()
                            ),
                        ));
                    }
                }
                found |= !urls.is_empty();
            }
            Some(other) => return Err(wrong_type(name, "an array", other)),
        }
    }
    Ok(found)
}

/// Whether a state made of chat messages (an array of messages, or an object
/// with a `messages` array) has image or audio parts in its content.
fn state_has_media(state: &Json) -> bool {
    let messages = match state {
        Json::Array(messages) => Some(messages.as_slice()),
        Json::Object(_) => state.get("messages").and_then(Json::as_array),
        _ => None,
    };
    messages.into_iter().flatten().any(|message| {
        message
            .get("content")
            .and_then(Json::as_array)
            .is_some_and(|parts| parts.iter().any(is_media_part))
    })
}

fn is_media_part(part: &Json) -> bool {
    match part.get("type").and_then(Json::as_str) {
        Some("image_url") => part.get("image_url").is_some(),
        Some("input_audio") => part.get("input_audio").is_some(),
        _ => false,
    }
}

// --- the response -------------------------------------------------------------

/// What the response repeats from the request.
#[derive(Debug)]
pub(crate) struct Reply {
    model: String,
    questions: Vec<Asked>,
}

#[derive(Debug)]
struct Asked {
    id: String,
    expects: Expects,
}

/// The kind of answer a question gets, with what the response needs of the
/// request to write it.
#[derive(Debug)]
enum Expects {
    Choice {
        n_options: usize,
    },
    /// The level descriptions as the client sent them, for `legend`.
    Score {
        legend: Vec<Json>,
    },
    Noul,
}

impl Reply {
    fn new(model: &str, questions: &[Question], legends: Legends) -> Self {
        let questions = questions
            .iter()
            .zip(legends)
            .map(|(question, legend)| Asked {
                id: question.id.clone(),
                expects: match (&question.kind, legend) {
                    (QuestionKind::Score { .. }, Some(legend)) => Expects::Score { legend },
                    (QuestionKind::Noul { .. }, _) => Expects::Noul,
                    _ => Expects::Choice { n_options: question.n_options() },
                },
            })
            .collect();
        Self { model: model.to_owned(), questions }
    }

    /// The body of the response, in the order of the request.
    ///
    /// The engine's result is checked against what was asked: an engine that
    /// answers other questions, or with another kind of answer, has a bug,
    /// and the client is told that and not handed a response that cannot be
    /// right.
    pub fn render(&self, result: &DecisionResult) -> Result<String, ApiError> {
        if result.answers.len() != self.questions.len() {
            return Err(mismatch(format!(
                "{} answers for {} questions",
                result.answers.len(),
                self.questions.len()
            )));
        }
        let mut answers = Vec::with_capacity(result.answers.len());
        for (asked, answer) in self.questions.iter().zip(&result.answers) {
            if answer.question != asked.id {
                return Err(mismatch(format!(
                    "the answer to '{}' where '{}' was asked",
                    answer.question, asked.id
                )));
            }
            answers.push((asked.id.clone(), body_of(asked, &answer.verdict)?));
        }
        let body = Json::Object(vec![
            ("model".into(), Json::String(self.model.clone())),
            ("answers".into(), Json::Object(answers)),
            (
                "usage".into(),
                Json::Object(vec![
                    ("input_tokens".into(), Json::Int(i64::from(result.usage.prompt_tokens))),
                    // Nothing is generated.
                    ("output_tokens".into(), Json::Int(0)),
                ]),
            ),
        ]);
        Ok(serde_json::to_string(&body).expect("response objects serialize"))
    }
}

fn mismatch(what: String) -> ApiError {
    tracing::error!("the engine's decision does not match the request: {what}");
    ApiError::internal(format!("the engine's answer does not match the questions: {what}"))
}

/// One answer, with the fields and in the order llama.cpp writes them.
fn body_of(asked: &Asked, verdict: &Verdict) -> Result<Json, ApiError> {
    let text = |s: &str| Json::String(s.to_owned());
    let count = |got: usize, want: usize| {
        if got == want {
            return Ok(());
        }
        Err(mismatch(format!("{got} probabilities for the {want} options of '{}'", asked.id)))
    };
    match (&asked.expects, verdict) {
        (Expects::Choice { n_options }, Verdict::Choice { choice, probabilities, confidence }) => {
            count(probabilities.len(), *n_options)?;
            Ok(Json::Object(vec![
                ("type".into(), text("choice")),
                ("choice".into(), text(choice)),
                (
                    "probabilities".into(),
                    Json::Object(
                        probabilities.iter().map(|(k, p)| (k.clone(), Json::Float(*p))).collect(),
                    ),
                ),
                ("confidence".into(), Json::Float(*confidence)),
            ]))
        }
        (Expects::Score { legend }, Verdict::Score { score, probabilities, confidence }) => {
            count(probabilities.len(), legend.len())?;
            let by_level = |values: Vec<Json>| {
                Json::Object(
                    values.into_iter().enumerate().map(|(i, v)| (i.to_string(), v)).collect(),
                )
            };
            Ok(Json::Object(vec![
                ("type".into(), text("score")),
                ("score".into(), Json::Float(*score)),
                ("legend".into(), by_level(legend.clone())),
                (
                    "probabilities".into(),
                    by_level(probabilities.iter().map(|p| Json::Float(*p)).collect()),
                ),
                ("confidence".into(), Json::Float(*confidence)),
            ]))
        }
        (Expects::Noul, Verdict::Noul { probability }) => Ok(Json::Object(vec![
            ("type".into(), text("noul")),
            ("noul".into(), Json::Float(*probability)),
        ])),
        _ => Err(mismatch(format!("a verdict of the wrong kind for '{}'", asked.id))),
    }
}
