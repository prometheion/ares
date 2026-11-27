//! The System One wire format: `POST /v1/systemone`, as Flide serves it.
//!
//! Ported from Flide (<https://github.com/OmniJev/Flide>, Apache-2.0), commit
//! caffe660df88db747d6da9f95f7d6759fea38698: the request and response of
//! `flide/api/schema.py` and the route of `flide/api/server.py`, so a client
//! written for Flide (or for TypeSafe's API, whose shapes Flide follows) works
//! unchanged:
//!
//! * **A request** is `state` (any JSON, null when left out), `model` (the
//!   served model's id, or `jev-latest`, the API's default name, for the one
//!   model the server serves) and a map of `questions`, each with a `type`
//!   (`choice`, `score` or `noul`), `instructions` (any JSON, optional) and
//!   `criteria`: a map of option to description for a choice (1 to 256), a
//!   list of levels for a score (1 to 256), `{"true", "false"}` descriptions
//!   for a noul (optional). Keys this route does not read are let through.
//! * **Several requests in one body**: `{"requests": [...], "model"}`, at most
//!   1024, every one checked before any runs; they run together, and one that
//!   fails fails them all. The answer is `{"model", "results": [...]}`, each
//!   result as a single request's answer without `model`.
//! * **The answer**: `model`, a map of `answers` (`choice` with the chosen key,
//!   the probability of every option and the confidence; `score` with the
//!   expected level, the `legend` as the client wrote it, the probability of
//!   every level and the confidence; `noul` with the probability of yes),
//!   `usage` (`input_tokens`, `output_tokens`), and `meta`: how the engine read
//!   it (`format`, the shared `prefix_tokens`, the `branches`, the
//!   `cached_tokens` of the prefix the prefix cache held).
//! * **Errors** are Flide's: 422 with `{"detail": [{"loc", "msg", "type"}]}`
//!   for a request that cannot be served as written (the messages of Flide's
//!   checks, the place in the body), 404 for a model that is not served, 429
//!   when the line is full, and `{"detail": "..."}` with the status of the
//!   other routes for everything else. The query takes Flide's `debias` (a
//!   whole number) and `debug` (a boolean), checked and not used: a chat model
//!   has no calibration.
//! * **Images** (`images`, `files`) are refused with 422 as Flide refuses them
//!   for a model that reads text only: this route reads text.
//!
//! There is no streaming: the API has none.

use std::collections::HashMap;

use ares_api::{
    ChoiceOption, DecisionRequest, DecisionResult, Json, Question, QuestionKind, Verdict,
};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::bridge::DecisionRun;
use crate::error::ApiError;
use crate::routes::{App, model_of, next_request_id, read_body};

/// The API's default model name, which names the served model whatever it is
/// (Flide's `jev-latest`).
pub(crate) const DEFAULT_MODEL: &str = "jev-latest";

/// The most requests in one body (Flide's `MAX_BATCH`).
pub(crate) const MAX_BATCH: usize = 1024;

/// The header that carries the request's id, as TypeSafe's API names it.
const REQUEST_ID: &str = "x-typesafe-request-id";

// --- errors ---------------------------------------------------------------------

/// One item of a 422's `detail`: where in the body, what is wrong, and the
/// kind of error as pydantic names it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Problem {
    pub loc: Vec<Json>,
    pub msg: String,
    pub kind: &'static str,
}

/// A refused request, written as Flide writes it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Refusal {
    pub status: StatusCode,
    /// `None`: the detail is `message` as text; else the list of problems.
    pub problems: Option<Vec<Problem>>,
    pub message: String,
    pub retry_after: Option<u32>,
}

impl Refusal {
    fn invalid(loc: Vec<Json>, msg: impl Into<String>, kind: &'static str) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            problems: Some(vec![Problem { loc, msg: msg.into(), kind }]),
            message: String::new(),
            retry_after: None,
        }
    }

    /// A check of Flide's (a value error of pydantic, or one its code raises).
    fn value(loc: Vec<Json>, msg: impl Into<String>) -> Self {
        Self::invalid(loc, msg, "value_error")
    }

    fn text(status: StatusCode, message: impl Into<String>) -> Self {
        Self { status, problems: None, message: message.into(), retry_after: None }
    }

    /// What the rest of the server says, in this route's shape: a request
    /// the engine refused as written is a 422 at the body (Flide's value
    /// errors), a full line a 429, everything else keeps its status.
    fn from_api(error: ApiError, loc: Vec<Json>) -> Self {
        match (error.status, error.code) {
            (_, "overloaded") => Self {
                status: StatusCode::TOO_MANY_REQUESTS,
                problems: None,
                message: "request queue is full".into(),
                retry_after: error.retry_after,
            },
            (StatusCode::BAD_REQUEST, _) => Self::value(loc, error.message),
            (status, _) => {
                Self { retry_after: error.retry_after, ..Self::text(status, error.message) }
            }
        }
    }

    fn body(&self) -> Json {
        let detail = match &self.problems {
            None => Json::String(self.message.clone()),
            Some(problems) => Json::Array(
                problems
                    .iter()
                    .map(|p| {
                        Json::Object(vec![
                            ("loc".into(), Json::Array(p.loc.clone())),
                            ("msg".into(), Json::String(p.msg.clone())),
                            ("type".into(), Json::String(p.kind.into())),
                        ])
                    })
                    .collect(),
            ),
        };
        Json::Object(vec![("detail".into(), detail)])
    }
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::warn!(status = self.status.as_u16(), "{}", self.message);
        }
        let body = serde_json::to_string(&self.body()).expect("JSON serializes");
        let mut response = (
            self.status,
            [(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
            body,
        )
            .into_response();
        if let Some(seconds) = self.retry_after {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        response
    }
}

fn at(loc: &[Json], more: &[&str]) -> Vec<Json> {
    let mut out = loc.to_vec();
    out.extend(more.iter().map(|s| Json::String((*s).to_owned())));
    out
}

fn missing(loc: Vec<Json>) -> Refusal {
    Refusal::invalid(loc, "Field required", "missing")
}

// --- the request ----------------------------------------------------------------

/// A request body as Python's `json` module reads an object: the last value
/// of a repeated key, at the place of the first.
fn dict(members: &[(String, Json)]) -> Vec<(&str, &Json)> {
    let mut kept: Vec<(&str, &Json)> = Vec::with_capacity(members.len());
    for (key, value) in members {
        match kept.iter_mut().find(|(k, _)| *k == key.as_str()) {
            Some(entry) => entry.1 = value,
            None => kept.push((key, value)),
        }
    }
    kept
}

fn field<'a>(members: &[(&str, &'a Json)], name: &str) -> Option<&'a Json> {
    members.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
}

/// One request of a body, understood: the decision without its id and its
/// model, the model it names, and the levels of its score questions as the
/// client wrote them (for the legends).
#[derive(Debug)]
pub(crate) struct Asked {
    pub model: Option<String>,
    pub state: Json,
    pub questions: Vec<Question>,
}

/// Reads one request (`loc` is where it is in the body) as Flide's schema
/// reads it.
pub(crate) fn read_request(value: &Json, loc: &[Json]) -> Result<Asked, Refusal> {
    let Json::Object(members) = value else {
        return Err(Refusal::invalid(
            loc.to_vec(),
            "Input should be a valid dictionary or object to extract fields from",
            "model_type",
        ));
    };
    let members = dict(members);
    let model = match field(&members, "model") {
        None | Some(Json::Null) => None,
        Some(Json::String(name)) => Some(name.clone()),
        Some(_) => {
            return Err(Refusal::invalid(
                at(loc, &["model"]),
                "Input should be a valid string",
                "string_type",
            ));
        }
    };
    for name in ["images", "files"] {
        match field(&members, name) {
            None | Some(Json::Null) => {}
            Some(Json::Array(items)) if items.is_empty() => {}
            Some(_) => {
                return Err(Refusal::value(at(loc, &[]), "images: this model reads text only"));
            }
        }
    }
    let state = field(&members, "state").cloned().unwrap_or(Json::Null);
    let questions = match field(&members, "questions") {
        None => return Err(missing(at(loc, &["questions"]))),
        Some(Json::Object(questions)) => dict(questions)
            .into_iter()
            .map(|(id, question)| read_question(id, question, &at(loc, &["questions", id])))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err(Refusal::invalid(
                at(loc, &["questions"]),
                "Input should be a valid dictionary",
                "dict_type",
            ));
        }
    };
    if questions.is_empty() {
        return Err(Refusal::value(loc.to_vec(), "Value error, at least one question is required"));
    }
    Ok(Asked { model, state, questions })
}

fn read_question(id: &str, value: &Json, loc: &[Json]) -> Result<Question, Refusal> {
    let Json::Object(members) = value else {
        return Err(Refusal::invalid(
            loc.to_vec(),
            "Input should be a valid dictionary or object to extract fields from",
            "model_type",
        ));
    };
    let members = dict(members);
    let instructions = field(&members, "instructions").cloned().unwrap_or(Json::Null);
    let criteria = field(&members, "criteria");
    let kind = match field(&members, "type") {
        None => return Err(missing(at(loc, &["type"]))),
        Some(Json::String(name)) if name == "choice" => match criteria {
            None => return Err(missing(at(loc, &["criteria"]))),
            Some(Json::Object(options)) => {
                let options: Vec<ChoiceOption> = dict(options)
                    .into_iter()
                    .map(|(key, description)| ChoiceOption {
                        key: key.to_owned(),
                        description: description.clone(),
                    })
                    .collect();
                if options.is_empty() {
                    return Err(Refusal::value(
                        loc.to_vec(),
                        "Value error, choice question needs at least one option in criteria",
                    ));
                }
                if options.len() > ares_api::MAX_OPTIONS {
                    return Err(Refusal::value(
                        loc.to_vec(),
                        format!(
                            "Value error, choice question has {} options, maximum is {}",
                            options.len(),
                            ares_api::MAX_OPTIONS
                        ),
                    ));
                }
                QuestionKind::Choice { options }
            }
            Some(_) => {
                return Err(Refusal::invalid(
                    at(loc, &["criteria"]),
                    "Input should be a valid dictionary",
                    "dict_type",
                ));
            }
        },
        Some(Json::String(name)) if name == "score" => match criteria {
            None => return Err(missing(at(loc, &["criteria"]))),
            Some(Json::Array(levels)) => {
                if levels.is_empty() {
                    return Err(Refusal::value(
                        loc.to_vec(),
                        "Value error, score question needs at least one level in criteria",
                    ));
                }
                if levels.len() > ares_api::MAX_OPTIONS {
                    return Err(Refusal::value(
                        loc.to_vec(),
                        format!(
                            "Value error, score question has {} levels, maximum is {}",
                            levels.len(),
                            ares_api::MAX_OPTIONS
                        ),
                    ));
                }
                QuestionKind::Score { levels: levels.clone() }
            }
            Some(_) => {
                return Err(Refusal::invalid(
                    at(loc, &["criteria"]),
                    "Input should be a valid list",
                    "list_type",
                ));
            }
        },
        Some(Json::String(name)) if name == "noul" => match criteria {
            None | Some(Json::Null) => {
                QuestionKind::Noul { when_true: Json::Null, when_false: Json::Null }
            }
            Some(Json::Object(sides)) => {
                let sides = dict(sides);
                let side = |name| field(&sides, name).cloned().unwrap_or(Json::Null);
                QuestionKind::Noul { when_true: side("true"), when_false: side("false") }
            }
            Some(_) => {
                return Err(Refusal::invalid(
                    at(loc, &["criteria"]),
                    "Input should be a valid dictionary or object to extract fields from",
                    "model_type",
                ));
            }
        },
        Some(_) => {
            return Err(Refusal::invalid(
                at(loc, &["type"]),
                "Input should be 'noul', 'choice' or 'score'",
                "literal_error",
            ));
        }
    };
    Ok(Question { id: id.to_owned(), instructions, kind })
}

/// Flide's query options: `debias`, a whole number, and `debug`, a boolean.
/// A chat model has no calibration, so they are checked and not used.
fn check_query(query: Option<&str>) -> Result<(), Refusal> {
    let refuse = |message: String| Refusal::value(vec![Json::String("query".into())], message);
    for pair in query.unwrap_or("").split('&').filter(|p| !p.is_empty()) {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        match name {
            "debug" => {
                let flag = value.to_ascii_lowercase();
                if !["1", "true", "yes", "on", "0", "false", "no", "off", ""]
                    .contains(&flag.as_str())
                {
                    return Err(refuse(format!("debug is true or false, not '{value}'")));
                }
            }
            "debias" if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) => {
                return Err(refuse(format!("debias is a whole number, not '{value}'")));
            }
            _ => {}
        }
    }
    Ok(())
}

// --- the answer -----------------------------------------------------------------

/// The body of one request's answer: `answers`, `usage` and `meta`, and
/// `model` first when it is not one of several results.
pub(crate) fn render(
    model: Option<&str>,
    questions: &[Question],
    result: &DecisionResult,
) -> Result<Json, Refusal> {
    if result.answers.len() != questions.len() {
        return Err(mismatch(format!(
            "{} answers for {} questions",
            result.answers.len(),
            questions.len()
        )));
    }
    let mut answers = Vec::with_capacity(questions.len());
    for (question, answer) in questions.iter().zip(&result.answers) {
        if answer.question != question.id {
            return Err(mismatch(format!(
                "the answer to '{}' where '{}' was asked",
                answer.question, question.id
            )));
        }
        answers.push((question.id.clone(), answer_body(question, &answer.verdict)?));
    }
    let count = |n: u32| Json::Int(i64::from(n));
    let mut body = Vec::with_capacity(4);
    if let Some(model) = model {
        body.push(("model".into(), Json::String(model.to_owned())));
    }
    body.push(("answers".into(), Json::Object(answers)));
    body.push((
        "usage".into(),
        Json::Object(vec![
            ("input_tokens".into(), count(result.usage.prompt_tokens)),
            ("output_tokens".into(), count(result.usage.completion_tokens)),
        ]),
    ));
    body.push((
        "meta".into(),
        Json::Object(vec![
            ("format".into(), Json::String("next_token".into())),
            ("prefix_tokens".into(), count(result.prefix_tokens)),
            ("branches".into(), Json::Int(questions.len() as i64)),
            ("cached_tokens".into(), count(result.usage.cached_prompt_tokens)),
        ]),
    ));
    Ok(Json::Object(body))
}

fn mismatch(what: String) -> Refusal {
    tracing::error!("the engine's decision does not match the request: {what}");
    Refusal::text(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("the engine's answer does not match the questions: {what}"),
    )
}

/// One answer, with the fields and in the order Flide writes them.
fn answer_body(question: &Question, verdict: &Verdict) -> Result<Json, Refusal> {
    let text = |s: &str| Json::String(s.to_owned());
    let count = |got: usize| {
        if got == question.n_options() {
            return Ok(());
        }
        Err(mismatch(format!(
            "{got} probabilities for the {} options of '{}'",
            question.n_options(),
            question.id
        )))
    };
    let by_level = |values: Vec<Json>| {
        Json::Object(values.into_iter().enumerate().map(|(i, v)| (i.to_string(), v)).collect())
    };
    match (&question.kind, verdict) {
        (QuestionKind::Choice { .. }, Verdict::Choice { choice, probabilities, confidence }) => {
            count(probabilities.len())?;
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
        (QuestionKind::Score { levels }, Verdict::Score { score, probabilities, confidence }) => {
            count(probabilities.len())?;
            Ok(Json::Object(vec![
                ("type".into(), text("score")),
                ("score".into(), Json::Float(*score)),
                ("legend".into(), by_level(levels.clone())),
                (
                    "probabilities".into(),
                    by_level(probabilities.iter().map(|p| Json::Float(*p)).collect()),
                ),
                ("confidence".into(), Json::Float(*confidence)),
            ]))
        }
        (QuestionKind::Noul { .. }, Verdict::Noul { probability }) => Ok(Json::Object(vec![
            ("type".into(), text("noul")),
            ("noul".into(), Json::Float(*probability)),
        ])),
        _ => Err(mismatch(format!("a verdict of the wrong kind for '{}'", question.id))),
    }
}

// --- the route ------------------------------------------------------------------

/// `POST /v1/systemone`.
pub(crate) async fn decide(State(app): State<App>, request: Request) -> Response {
    let id = request.headers().get(REQUEST_ID).cloned().unwrap_or_else(|| {
        HeaderValue::from_str(&format!("ares-{}", next_request_id())).expect("hex is a header")
    });
    let mut response = match try_decide(app, request).await {
        Ok(response) => response,
        Err(refusal) => refusal.into_response(),
    };
    response.headers_mut().insert(REQUEST_ID, id);
    response
}

/// What the requests of a body resolve to: the served model each runs on.
async fn try_decide(app: App, request: Request) -> Result<Response, Refusal> {
    let query = request.uri().query().map(str::to_owned);
    let body =
        read_body(request, &app.config).await.map_err(|e| Refusal::from_api(e, body_loc()))?;
    let json: Json = serde_json::from_slice(&body).map_err(|_| {
        Refusal::invalid(vec![Json::String("body".into())], "invalid JSON", "json_invalid")
    })?;
    let batch = json.as_object().map(dict).and_then(|members| {
        field(&members, "requests").map(|items| {
            let mixed =
                field(&members, "state").is_some() || field(&members, "questions").is_some();
            (items.clone(), field(&members, "model").cloned(), mixed)
        })
    });
    match batch {
        Some((items, model, mixed)) => {
            if mixed {
                return Err(Refusal::value(
                    body_loc(),
                    "a batch cannot also contain state or questions",
                ));
            }
            let loc = at(&body_loc(), &["requests"]);
            let items = match items {
                Json::Array(items) if !items.is_empty() => items,
                _ => return Err(Refusal::value(loc, "requests must be a non-empty list")),
            };
            if items.len() > MAX_BATCH {
                return Err(Refusal::value(loc, format!("requests exceeds {MAX_BATCH} items")));
            }
            let model = match model {
                None | Some(Json::Null) => None,
                Some(Json::String(name)) if name.is_empty() => None,
                Some(Json::String(name)) => Some(name),
                Some(_) => {
                    return Err(Refusal::invalid(
                        at(&body_loc(), &["model"]),
                        "Input should be a valid string",
                        "string_type",
                    ));
                }
            };
            check_query(query.as_deref())?;
            let mut asked = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let mut loc = loc.clone();
                loc.push(Json::Int(i as i64));
                asked.push(read_request(item, &loc)?);
            }
            run_batch(&app, model, asked).await
        }
        None => {
            let asked = read_request(&json, &body_loc())?;
            check_query(query.as_deref())?;
            let (model, mut results) = run(&app, None, vec![asked]).await?;
            let (questions, result) = results.remove(0);
            let body = render(Some(&model), &questions, &result)?;
            Ok(json_body(&body))
        }
    }
}

fn body_loc() -> Vec<Json> {
    vec![Json::String("body".into())]
}

fn json_body(body: &Json) -> Response {
    let text = serde_json::to_string(body).expect("JSON serializes");
    ([(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))], text).into_response()
}

async fn run_batch(
    app: &App,
    model: Option<String>,
    asked: Vec<Asked>,
) -> Result<Response, Refusal> {
    let (model, results) = run(app, model, asked).await?;
    let mut items = Vec::with_capacity(results.len());
    for (questions, result) in &results {
        items.push(render(None, questions, result)?);
    }
    Ok(json_body(&Json::Object(vec![
        ("model".into(), Json::String(model)),
        ("results".into(), Json::Array(items)),
    ])))
}

/// The served id of the model `asked` names: the API's default name and an
/// empty one name the one model the server serves.
fn resolve(app: &App, asked: Option<&str>) -> Result<String, Refusal> {
    let offered = app.models.offered();
    match asked.filter(|name| !name.is_empty() && *name != DEFAULT_MODEL) {
        Some(name) => Ok(name.to_owned()),
        None => match offered.as_slice() {
            [only] => Ok(only.clone()),
            _ => Err(not_served(asked.unwrap_or(DEFAULT_MODEL), &offered)),
        },
    }
}

/// Flide's 404 for a model that is not served.
fn not_served(asked: &str, offered: &[String]) -> Refusal {
    let available = if offered.is_empty() { "none".to_owned() } else { offered.join(", ") };
    Refusal::text(
        StatusCode::NOT_FOUND,
        format!(
            "model '{asked}' is not served; available: {available} plus aliases \
             ['{DEFAULT_MODEL}']"
        ),
    )
}

/// Runs the requests together and waits for all of them. Each runs on the
/// model it names, else on `model` (the body's), else on the one served. The
/// first one (in the order of the body) that cannot start or fails fails the
/// whole: the ones that run are cancelled. Returns the served id of the
/// model the body names and each request's questions with their answers.
async fn run(
    app: &App,
    model: Option<String>,
    asked: Vec<Asked>,
) -> Result<(String, Vec<(Vec<Question>, DecisionResult)>), Refusal> {
    let batch = asked.len() > 1 || model.is_some();
    let loc = |i: usize| {
        if batch {
            vec![Json::String("body".into()), Json::String("requests".into()), Json::Int(i as i64)]
        } else {
            body_loc()
        }
    };
    // Every request's model is resolved and held before any of them runs: a
    // body with a model that is not served runs nothing.
    let main = || resolve(app, model.as_deref());
    let mut leases: HashMap<String, crate::models::Lease> = HashMap::new();
    let mut named = Vec::with_capacity(asked.len());
    let mut main_id = None;
    for (i, request) in asked.iter().enumerate() {
        let requested = match &request.model {
            Some(name) => resolve(app, Some(name))?,
            None => main()?,
        };
        if !leases.contains_key(&requested) {
            let mut name = requested.clone();
            let lease = model_of(app, &mut name).await.map_err(|e| match e.status {
                StatusCode::NOT_FOUND => not_served(&requested, &app.models.offered()),
                _ => Refusal::from_api(e, loc(i)),
            })?;
            leases.insert(requested.clone(), lease);
        }
        if main_id.is_none() && (request.model.is_none() || asked.len() == 1) {
            main_id = Some(leases[&requested].id().to_owned());
        }
        named.push(requested);
    }
    let main_id = match main_id {
        Some(id) => id,
        None => main().unwrap_or_else(|_| named[0].clone()),
    };
    let mut runs = Vec::with_capacity(asked.len());
    for (i, (request, requested)) in asked.into_iter().zip(named).enumerate() {
        let lease = &leases[&requested];
        let decision = DecisionRequest {
            model: lease.id().to_owned(),
            state: request.state,
            questions: request.questions,
            priority: 0,
            id: next_request_id(),
        };
        tracing::info!(
            id = %decision.id,
            model = %decision.model,
            questions = decision.questions.len(),
            "decision request"
        );
        let questions = decision.questions.clone();
        let run = DecisionRun::start(lease.engine(), decision, app.stop.clone())
            .map_err(|e| Refusal::from_api(e, loc(i)))?;
        runs.push((questions, run));
    }
    let (questions, runs): (Vec<_>, Vec<_>) = runs.into_iter().unzip();
    let outcomes = futures_util::future::join_all(runs.into_iter().map(DecisionRun::finish)).await;
    let mut results = Vec::with_capacity(outcomes.len());
    for (i, (questions, outcome)) in questions.into_iter().zip(outcomes).enumerate() {
        results.push((questions, outcome.map_err(|e| Refusal::from_api(e, loc(i)))?));
    }
    drop(leases);
    Ok((main_id, results))
}
