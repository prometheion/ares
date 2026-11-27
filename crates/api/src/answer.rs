//! What the engine reports for a decision.
//!
//! The answers are made as Flide makes them for a chat model
//! (<https://github.com/OmniJev/Flide>, Apache-2.0, commit
//! caffe660df88db747d6da9f95f7d6759fea38698): the softmax of the labels'
//! logits (`NextToken.finish` in `flide/formats/next_token.py`) and the typed
//! answers of `flide/api/answers.py`, whose two confidence functions Flide
//! ports from TypeSafe's `system-one-adapter-python` (MIT License, Copyright
//! (c) 2026 TypeSafe AI; see THIRD-PARTY-NOTICES). Every number of an answer is
//! rounded to six places, as Flide rounds it.

use crate::{EngineError, Question, QuestionKind, Usage};

/// How a decision ended.
///
/// A decision has no stream: it ends once, with one of these. That is the
/// reason it has its own [`DecisionSink`](crate::DecisionSink) and does not
/// share [`EventSink`](crate::EventSink) with generation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionOutcome {
    /// Every question is answered.
    Decided(DecisionResult),
    /// [`Engine::cancel`](crate::Engine::cancel) took effect, or the engine
    /// is shutting down. A front end whose client is still connected treats
    /// this as a failure, as it does [`FinishReason::Cancelled`](crate::FinishReason::Cancelled).
    Cancelled,
    /// The request failed; no question has an answer. A decision is answered
    /// whole or not at all.
    Failed(EngineError),
}

/// The answers to a [`DecisionRequest`](crate::DecisionRequest).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionResult {
    /// One per question, in the order the request listed them.
    pub answers: Vec<Answer>,
    /// `prompt_tokens` counts the tokens the request put through the model as
    /// Flide counts them: the shared start of the questions' prompts once and
    /// the rest of every question's prompt, cached ones included;
    /// `cached_prompt_tokens` the part of the shared start that came from the
    /// prefix cache. `completion_tokens` is the number of questions: one
    /// token read for each.
    pub usage: Usage,
    /// The tokens the questions' prompts share, which the engine computed
    /// once.
    pub prefix_tokens: u32,
}

/// The answer to one [`Question`].
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The [`Question::id`] this answers.
    pub question: String,
    pub verdict: Verdict,
}

/// What the model decided, with the numbers behind it. The fields are the
/// ones the System One API returns.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The answer to a [`QuestionKind::Choice`] question.
    Choice {
        /// The key of the most probable option; the first of equals.
        choice: String,
        /// The probability of each option in the order of the request.
        probabilities: Vec<(String, f64)>,
        /// 0 when all options are equally likely, 1 when one is certain.
        confidence: f64,
    },
    /// The answer to a [`QuestionKind::Score`] question.
    Score {
        /// The expected level index, weighted by probability. It can lie
        /// between two levels.
        score: f64,
        /// The probability of each level, lowest first.
        probabilities: Vec<f64>,
        /// From 0 to 1; high when the probability sits near one level.
        confidence: f64,
    },
    /// The answer to a [`QuestionKind::Noul`] question: how likely it is
    /// that the answer is true.
    Noul { probability: f64 },
}

impl Question {
    /// Makes the answer to this question from the logits of its labels at the
    /// first token of the reply, in the order of
    /// [`Rendered::tokens`](crate::Rendered::tokens) (`[yes, no]` for a noul).
    ///
    /// This is the one place that turns logits into the numbers the API
    /// returns, so that every engine answers the same way. The probabilities
    /// are the softmax of the logits over the question's labels. A logit of
    /// `-inf` is an option the model rules out; NaN, `+inf`, or a question
    /// whose every label is ruled out means the engine computed nonsense and
    /// is an internal error that names the question.
    pub fn answer(&self, logits: &[f32]) -> Result<Answer, EngineError> {
        let id = &self.id;
        if logits.len() != self.n_options() {
            return Err(EngineError::Internal(format!(
                "question '{id}' has {} options but the engine read {} logits",
                self.n_options(),
                logits.len()
            )));
        }
        let logits: Vec<f64> = logits.iter().map(|l| f64::from(*l)).collect();
        let probabilities = softmax(&logits).ok_or_else(|| {
            EngineError::Internal(format!(
                "the logits of question '{id}' cannot be turned into probabilities (NaN, an \
                 infinite logit, or every option ruled out): the model could not evaluate the \
                 decision"
            ))
        })?;
        let verdict = match &self.kind {
            QuestionKind::Choice { options } => {
                let p = normalize(&probabilities);
                Verdict::Choice {
                    choice: options[first_max(&p)].key.clone(),
                    probabilities: options
                        .iter()
                        .map(|o| o.key.clone())
                        .zip(p.iter().map(|p| round6(*p)))
                        .collect(),
                    confidence: round6(choice_confidence(&p)),
                }
            }
            QuestionKind::Score { .. } => {
                let p = normalize(&probabilities);
                Verdict::Score {
                    score: round6(p.iter().enumerate().map(|(i, p)| i as f64 * p).sum()),
                    confidence: round6(score_confidence(&p)),
                    probabilities: p.iter().map(|p| round6(*p)).collect(),
                }
            }
            // The labels are [yes, no].
            QuestionKind::Noul { .. } => {
                Verdict::Noul { probability: round6(probabilities[0].clamp(0.0, 1.0)) }
            }
        };
        Ok(Answer { question: id.clone(), verdict })
    }
}

/// The softmax as `NextToken.finish` computes it; `None` when no distribution
/// follows from the logits.
fn softmax(logits: &[f64]) -> Option<Vec<f64>> {
    if logits.iter().any(|l| l.is_nan()) {
        return None;
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return None;
    }
    // Relative to the maximum, so a large logit cannot overflow, and a label
    // at -inf comes out as exactly 0.
    let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    Some(weights.into_iter().map(|w| w / sum).collect())
}

/// `_normalize` of `answers.py`: the probabilities divided by their sum, or a
/// uniform distribution when the sum is not positive.
fn normalize(probs: &[f64]) -> Vec<f64> {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / probs.len() as f64; probs.len()];
    }
    probs.iter().map(|p| p / total).collect()
}

/// The index of the first largest value, as Python's `max(range(n),
/// key=p.__getitem__)` finds it.
fn first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// TypeSafe's confidence of a choice: the peak probability rescaled from a
/// uniform guess (0) to certainty (1).
fn choice_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let uniform = 1.0 / p.len() as f64;
    (p[first_max(p)] - uniform) / (1.0 - uniform)
}

/// TypeSafe's confidence of a score: one minus the probability-weighted mean
/// distance from the modal level, over the mean distance of a uniform
/// distribution from its center.
fn score_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let k = p.len() as f64;
    let mode = first_max(p) as f64;
    let distance: f64 = p.iter().enumerate().map(|(i, p)| p * (i as f64 - mode).abs()).sum();
    let center = (k - 1.0) / 2.0;
    let uniform: f64 = (0..p.len()).map(|i| (i as f64 - center).abs()).sum::<f64>() / k;
    (1.0 - distance / uniform).max(0.0)
}

/// Python's `round(x, 6)`: the nearest value with six decimal places.
fn round6(x: f64) -> f64 {
    format!("{x:.6}").parse().unwrap_or(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChoiceOption, Json};

    fn choice(keys: &[&str]) -> Question {
        Question {
            id: "route".into(),
            instructions: Json::Null,
            kind: QuestionKind::Choice {
                options: keys
                    .iter()
                    .map(|k| ChoiceOption { key: (*k).into(), description: Json::Null })
                    .collect(),
            },
        }
    }

    fn score(levels: usize) -> Question {
        Question {
            id: "urgency".into(),
            instructions: Json::Null,
            kind: QuestionKind::Score { levels: vec![Json::Null; levels] },
        }
    }

    fn noul() -> Question {
        Question {
            id: "angry".into(),
            instructions: Json::Null,
            kind: QuestionKind::Noul { when_true: Json::Null, when_false: Json::Null },
        }
    }

    fn ln(p: f64) -> f32 {
        p.ln() as f32
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 2e-6
    }

    #[test]
    fn a_choice_names_the_most_probable_option_and_how_sure_that_is() {
        let logits = [ln(0.5), ln(0.25), ln(0.25)];
        let answer = choice(&["billing", "shipping", "technical"]).answer(&logits).unwrap();
        assert_eq!(answer.question, "route");
        let Verdict::Choice { choice: chosen, probabilities, confidence } = answer.verdict else {
            panic!("a choice answers with a choice");
        };
        assert_eq!(chosen, "billing");
        let keys: Vec<_> = probabilities.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["billing", "shipping", "technical"], "in the order of the request");
        let values: Vec<_> = probabilities.iter().map(|(_, p)| *p).collect();
        assert!(close(values[0], 0.5) && close(values[1], 0.25) && close(values[2], 0.25));
        // (0.5 - 1/3) / (1 - 1/3)
        assert!(close(confidence, 0.25), "{confidence}");
    }

    #[test]
    fn numbers_are_rounded_to_six_places_as_flide_rounds_them() {
        let Verdict::Choice { probabilities, confidence, .. } =
            choice(&["a", "b", "c"]).answer(&[0.0, 0.0, 0.0]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!(probabilities[0].1, 0.333333);
        assert_eq!(confidence, 0.0);
        assert_eq!(round6(0.1234565), 0.123456, "0.1234565 is below the half in binary");
        assert_eq!(round6(2.0 / 3.0), 0.666667);
    }

    #[test]
    fn equal_options_are_a_tie_that_goes_to_the_first() {
        let Verdict::Choice { choice: chosen, .. } =
            choice(&["a", "b", "c"]).answer(&[-3.0, -3.0, -3.0]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!(chosen, "a");
    }

    #[test]
    fn a_single_option_is_certain_and_an_option_ruled_out_has_probability_zero() {
        let Verdict::Choice { choice: chosen, probabilities, confidence } =
            choice(&["only"]).answer(&[-12.5]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!((chosen.as_str(), probabilities[0].1, confidence), ("only", 1.0, 1.0));
        let Verdict::Choice { choice: chosen, probabilities, confidence } =
            choice(&["a", "b"]).answer(&[f32::NEG_INFINITY, -1.0]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!(chosen, "b");
        assert_eq!((probabilities[0].1, probabilities[1].1, confidence), (0.0, 1.0, 1.0));
    }

    #[test]
    fn a_score_is_the_expected_level_and_its_confidence_follows_the_spread() {
        // Probabilities 0.1, 0.2, 0.6, 0.1: the mode is level 2.
        let logits = [ln(0.1), ln(0.2), ln(0.6), ln(0.1)];
        let Verdict::Score { score, probabilities, confidence } =
            score(4).answer(&logits).unwrap().verdict
        else {
            panic!("a score answers with a score");
        };
        assert!(close(score, 0.2 + 1.2 + 0.3), "{score}");
        assert_eq!(probabilities.len(), 4);
        // distance = 0.1*2 + 0.2*1 + 0.1*1 = 0.5; uniform = (1.5+0.5+0.5+1.5)/4 = 1.
        assert!(close(confidence, 0.5), "{confidence}");
        let Verdict::Score { score, confidence, .. } =
            self::score(5).answer(&[0.0; 5]).unwrap().verdict
        else {
            panic!()
        };
        assert!(close(score, 2.0) && confidence == 0.0, "{score} {confidence}");
        let Verdict::Score { score, confidence, .. } =
            self::score(1).answer(&[4.0]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!((score, confidence), (0.0, 1.0));
    }

    #[test]
    fn a_noul_answer_is_the_probability_of_yes() {
        // [yes, no] with yes three times as likely.
        let Verdict::Noul { probability } = noul().answer(&[ln(0.75), ln(0.25)]).unwrap().verdict
        else {
            panic!("a noul answers with a probability");
        };
        assert!(close(probability, 0.75), "{probability}");
    }

    #[test]
    fn logits_that_mean_nothing_are_an_internal_error_that_names_the_question() {
        let cases: [(Question, Vec<f32>, &str); 6] = [
            (choice(&["a", "b"]), vec![0.0], "has 2 options but the engine read 1"),
            (choice(&["a", "b"]), vec![0.0, f32::NAN], "cannot be turned into probabilities"),
            (choice(&["a", "b"]), vec![0.0, f32::INFINITY], "cannot be turned into probabilities"),
            (choice(&["a", "b"]), vec![f32::NEG_INFINITY; 2], "every option ruled out"),
            (score(3), vec![0.0; 4], "has 3 options but the engine read 4"),
            (noul(), vec![], "has 2 options but the engine read 0"),
        ];
        for (question, logits, words) in cases {
            let EngineError::Internal(message) = question.answer(&logits).unwrap_err() else {
                panic!("nonsense from the engine is an internal error");
            };
            assert!(message.contains(&format!("'{}'", question.id)), "{message}");
            assert!(message.contains(words), "'{message}' should contain '{words}'");
        }
    }
}
