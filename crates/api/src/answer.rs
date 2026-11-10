//! What the engine reports for a decision.

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
    /// `prompt_tokens` counts every token the request put through the model
    /// (the prompt, and the option tokens that were scored), cached ones
    /// included. `completion_tokens` is 0: nothing is generated.
    pub usage: Usage,
}

/// The answer to one [`Question`].
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The [`Question::id`] this answers.
    pub question: String,
    pub verdict: Verdict,
}

/// What the model decided, with the numbers behind it. The fields are the
/// ones the TypeSafe API returns.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The answer to a [`QuestionKind::Choice`] question.
    Choice {
        /// The key of the most probable option; the first of equals.
        choice: String,
        /// The probability of each option in the order of the request. They
        /// sum to 1.
        probabilities: Vec<(String, f64)>,
        /// 0 when all options are equally likely, 1 when one is certain.
        confidence: f64,
    },
    /// The answer to a [`QuestionKind::Score`] question.
    Score {
        /// The expected level index, weighted by probability. It can lie
        /// between two levels.
        score: f64,
        /// The probability of each level, lowest first. They sum to 1.
        probabilities: Vec<f64>,
        /// From 0 to 1; high when the probability sits near one level.
        confidence: f64,
    },
    /// The answer to a [`QuestionKind::Noul`] question: how likely it is
    /// that the answer is true.
    Noul { probability: f64 },
}

impl Question {
    /// Makes the answer to this question from the score of each of its
    /// options.
    ///
    /// This is the one place that turns scores into the numbers the API
    /// returns, so that every engine answers the same way. `scores` holds one
    /// natural-log likelihood per option, in the order of
    /// [`option_texts`](Question::option_texts): the log-probability of the
    /// option's text after the prompt, summed over its tokens. Only the
    /// differences matter. `f64::NEG_INFINITY` is a valid score (an option
    /// the model rules out); NaN, an infinity above all, or a question where
    /// every option is ruled out is not, and is reported as an internal
    /// error because it means the engine computed nonsense.
    ///
    /// The probabilities are the softmax of the scores at temperature 1.
    /// Native decision models store a calibration temperature in their file;
    /// a language model has none, so what comes out is the model's own
    /// likelihood of the options, which is not calibrated for any data.
    pub fn answer(&self, scores: &[f64]) -> Result<Answer, EngineError> {
        let id = &self.id;
        if scores.len() != self.n_options() {
            return Err(EngineError::Internal(format!(
                "question '{id}' has {} options but the engine scored {}",
                self.n_options(),
                scores.len()
            )));
        }
        let probabilities = softmax(scores).ok_or_else(|| {
            EngineError::Internal(format!(
                "the scores of question '{id}' cannot be turned into probabilities (NaN, an \
                 infinite score, or every option ruled out): the model could not evaluate \
                 the decision"
            ))
        })?;

        let verdict = match &self.kind {
            QuestionKind::Choice { options } => {
                let best = first_max(&probabilities);
                Verdict::Choice {
                    choice: options[best].key.clone(),
                    probabilities: options
                        .iter()
                        .map(|o| o.key.clone())
                        .zip(probabilities.iter().copied())
                        .collect(),
                    confidence: confidence_choice(&probabilities),
                }
            }
            QuestionKind::Score { .. } => Verdict::Score {
                score: probabilities.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
                confidence: confidence_score(&probabilities),
                probabilities,
            },
            // The options are [false, true].
            QuestionKind::Noul { .. } => Verdict::Noul { probability: probabilities[1] },
        };
        Ok(Answer { question: id.clone(), verdict })
    }
}

/// `None` when no distribution follows from the scores.
fn softmax(scores: &[f64]) -> Option<Vec<f64>> {
    if scores.iter().any(|s| s.is_nan()) {
        return None;
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return None;
    }
    // Relative to the maximum, so a large score cannot overflow, and an
    // option at -inf comes out as exactly 0.
    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    Some(weights.into_iter().map(|w| w / sum).collect())
}

/// The index of the first largest value.
fn first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// How far the most probable option is above a uniform guess, scaled so that
/// certainty is 1. The formula TypeSafe publishes.
fn confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / probs.len() as f64;
    let p_max = probs[first_max(probs)];
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

/// How tightly the probability sits around its mode, against a uniform
/// distribution over the levels: the mean distance to the mode, relative to
/// the one of a uniform distribution around its center. The formula TypeSafe
/// publishes.
fn confidence_score(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let n = probs.len() as f64;
    let mode = first_max(probs) as f64;
    let mut dist = 0.0;
    let mut dist_uniform = 0.0;
    for (i, p) in probs.iter().enumerate() {
        let i = i as f64;
        dist += p * (i - mode).abs();
        dist_uniform += (i - (n - 1.0) / 2.0).abs() / n;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChoiceOption;

    fn choice(keys: &[&str]) -> Question {
        Question {
            id: "route".into(),
            instructions: "which?".into(),
            kind: QuestionKind::Choice {
                options: keys
                    .iter()
                    .map(|k| ChoiceOption { key: (*k).into(), description: None })
                    .collect(),
            },
        }
    }

    fn score(levels: usize) -> Question {
        Question {
            id: "urgency".into(),
            instructions: "how urgent?".into(),
            kind: QuestionKind::Score { levels: vec![None; levels] },
        }
    }

    fn noul() -> Question {
        Question {
            id: "angry".into(),
            instructions: "angry?".into(),
            kind: QuestionKind::Noul { when_false: None, when_true: None },
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn a_choice_names_the_most_probable_option_and_how_sure_that_is() {
        // Log-likelihoods of 0.5, 0.25, 0.25.
        let scores = [0.5f64.ln(), 0.25f64.ln(), 0.25f64.ln()];
        let answer = choice(&["billing", "shipping", "technical"]).answer(&scores).unwrap();
        assert_eq!(answer.question, "route");
        let Verdict::Choice { choice: chosen, probabilities, confidence } = answer.verdict else {
            panic!("a choice answers with a choice");
        };
        assert_eq!(chosen, "billing");
        let keys: Vec<_> = probabilities.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["billing", "shipping", "technical"], "in the order of the request");
        let values: Vec<_> = probabilities.iter().map(|(_, p)| *p).collect();
        assert!(close(values[0], 0.5) && close(values[1], 0.25) && close(values[2], 0.25));
        assert!(close(values.iter().sum::<f64>(), 1.0));
        // (0.5 - 1/3) / (1 - 1/3)
        assert!(close(confidence, 0.25), "{confidence}");
    }

    #[test]
    fn equal_options_are_a_tie_that_goes_to_the_first_with_confidence_zero() {
        let Verdict::Choice { choice: chosen, confidence, .. } =
            choice(&["a", "b", "c"]).answer(&[-3.0, -3.0, -3.0]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!(chosen, "a");
        assert_eq!(confidence, 0.0);
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
            choice(&["a", "b"]).answer(&[f64::NEG_INFINITY, -1.0]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!(chosen, "b");
        assert_eq!((probabilities[0].1, probabilities[1].1, confidence), (0.0, 1.0, 1.0));
    }

    #[test]
    fn only_the_differences_between_scores_matter() {
        let near_zero = choice(&["a", "b"]).answer(&[-0.1, -1.3]).unwrap();
        let far_below = choice(&["a", "b"]).answer(&[-900.1, -901.3]).unwrap();
        let (Verdict::Choice { probabilities: p, .. }, Verdict::Choice { probabilities: q, .. }) =
            (near_zero.verdict, far_below.verdict)
        else {
            panic!()
        };
        assert!(close(p[0].1, q[0].1) && close(p[1].1, q[1].1), "{p:?} vs {q:?}");
        assert!(q[0].1.is_finite() && q[1].1.is_finite(), "no overflow and no underflow to NaN");
    }

    #[test]
    fn a_score_is_the_expected_level_and_its_confidence_follows_the_spread() {
        // Probabilities 0.1, 0.2, 0.6, 0.1: the mode is level 2.
        let scores = [0.1f64.ln(), 0.2f64.ln(), 0.6f64.ln(), 0.1f64.ln()];
        let Verdict::Score { score, probabilities, confidence } =
            score(4).answer(&scores).unwrap().verdict
        else {
            panic!("a score answers with a score");
        };
        assert!(close(score, 0.2 + 1.2 + 0.3), "{score}");
        assert_eq!(probabilities.len(), 4);
        // dist = 0.1*2 + 0.2*1 + 0.1*1 = 0.5; dist_uniform = (1.5+0.5+0.5+1.5)/4 = 1.
        assert!(close(confidence, 0.5), "{confidence}");

        // All the probability on one level: certain; spread evenly: no confidence.
        let Verdict::Score { score, confidence, .. } =
            self::score(3).answer(&[f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY]).unwrap().verdict
        else {
            panic!()
        };
        assert_eq!((score, confidence), (1.0, 1.0));
        let Verdict::Score { score, confidence, .. } =
            self::score(5).answer(&[0.0; 5]).unwrap().verdict
        else {
            panic!()
        };
        assert!(close(score, 2.0) && confidence == 0.0, "{score} {confidence}");
    }

    #[test]
    fn a_noul_answer_is_the_probability_of_true() {
        // [false, true] with true three times as likely.
        let Verdict::Noul { probability } =
            noul().answer(&[0.25f64.ln(), 0.75f64.ln()]).unwrap().verdict
        else {
            panic!("a noul answers with a probability");
        };
        assert!(close(probability, 0.75), "{probability}");
    }

    #[test]
    fn scores_that_mean_nothing_are_an_internal_error_that_names_the_question() {
        let cases: [(Question, Vec<f64>, &str); 6] = [
            (choice(&["a", "b"]), vec![0.0], "has 2 options but the engine scored 1"),
            (choice(&["a", "b"]), vec![0.0, f64::NAN], "cannot be turned into probabilities"),
            (choice(&["a", "b"]), vec![0.0, f64::INFINITY], "cannot be turned into probabilities"),
            (choice(&["a", "b"]), vec![f64::NEG_INFINITY; 2], "every option ruled out"),
            (score(3), vec![0.0; 4], "has 3 options but the engine scored 4"),
            (noul(), vec![], "has 2 options but the engine scored 0"),
        ];
        for (question, scores, words) in cases {
            let EngineError::Internal(message) = question.answer(&scores).unwrap_err() else {
                panic!("nonsense from the engine is an internal error");
            };
            assert!(message.contains(&format!("'{}'", question.id)), "{message}");
            assert!(message.contains(words), "'{message}' should contain '{words}'");
        }
    }
}
