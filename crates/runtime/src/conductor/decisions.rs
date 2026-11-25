//! Decisions ("Jev mode"): questions about a state answered from the logits of
//! their options' labels, with no generation.
//!
//! The algorithm is Flide's exact state forking (<https://github.com/OmniJev/Flide>,
//! Apache-2.0, commit caffe660df88db747d6da9f95f7d6759fea38698,
//! `flide/runtime` and `flide/backbones/qwen35/decoder.py`), fitted to the
//! conductor's structures. Each question is a prompt of its own
//! ([`ares_api::Question::render`]); the prompts share their start (the
//! state), and the questions are answered from one computation of it:
//!
//! 1. **The shared start** runs as a prompt of its own, the *parent*: a request
//!    that reads its prompt and ends, as every request does, by leaving its
//!    pages, its partly used last page and the state of its recurrent layers in
//!    the prefix cache ([`Conductor::keep`]). It is looked up in the cache as
//!    any prompt, so a state the engine has read before costs a row.
//! 2. **The forks**: each question is then a request whose prompt is the whole
//!    question, which starts from the parent's cached pages and restores the
//!    checkpoint of its recurrent state (an exact copy of the state at the end
//!    of the shared start: the fork of Flide's delta-rule layers) and computes
//!    only its own rows. They are prompts like any other, so they share steps
//!    with each other and with generation, as many at once as there are slots.
//! 3. **The read**: the last row of a fork is wanted, and the step hands back
//!    the logits of the question's label tokens there (the record's scores)
//!    before the sampler touches them. The fork ends with that row: it keeps
//!    nothing in the cache but the full pages it published.
//!
//! When every fork has its logits, [`ares_api::Question::answer`] makes the
//! answers. A decision of one question has no parent: its one fork computes
//! everything. When the cache cannot keep what the parent left (no room, a
//! model without a prefix cache), a fork computes what it misses, from an
//! earlier checkpoint or from its first token: the answers are the same, only
//! the work is not shared.
//!
//! The parent and the forks are requests the conductor makes itself
//! ([`Role`]): their ids have the top bit set, which the server's ids never
//! have (they count from the time of day in nanoseconds), they send their
//! events nowhere, and the accounting of what requests produce does not count
//! them.

use std::sync::atomic::{AtomicU64, Ordering};

use ares_api::{DecisionOutcome, DecisionResult, DecisionSink, Question, Sampling};

use super::*;

/// The ids of the requests the conductor makes for decisions: the top bit set.
const INTERNAL: u64 = 1 << 63;

/// What a request the conductor holds is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// A client's generation.
    Request,
    /// The shared start of decision `decision`'s questions.
    Parent { decision: RequestId },
    /// Question `question` of decision `decision`, read at its last row over
    /// the tokens of its labels.
    Fork { decision: RequestId, question: usize, labels: Vec<u32> },
}

/// A decision, checked and turned into tokens ([`crate::Runtime`]'s
/// `Engine::decide`).
pub(crate) struct DecisionSubmission {
    pub id: RequestId,
    pub questions: Vec<Question>,
    /// Each question's whole prompt.
    pub prompts: Vec<Vec<u32>>,
    /// Each question's label tokens, in the order of its options.
    pub labels: Vec<Vec<u32>>,
    /// The tokens every prompt starts with; each prompt is longer.
    pub shared: usize,
    pub priority: i32,
    pub sink: DecisionSink,
}

/// A decision the conductor holds.
pub(crate) struct Decision {
    questions: Vec<Question>,
    prompts: Vec<Vec<u32>>,
    labels: Vec<Vec<u32>>,
    shared: usize,
    priority: i32,
    sink: DecisionSink,
    /// The requests of the decision that are held (waiting or running).
    parts: Vec<RequestId>,
    /// The logits of each question's labels, once its fork has read them.
    logits: Vec<Option<Vec<f32>>>,
    /// What the prefix cache held of the shared start.
    cached: u32,
}

impl Decision {
    fn answered(&self) -> bool {
        self.logits.iter().all(Option::is_some)
    }

    /// Ends the decision with `error`, as a conductor that stops does.
    pub(super) fn fail(self, error: EngineError) {
        self.sink.send(DecisionOutcome::Failed(error));
    }
}

/// The next id of a request the conductor makes.
fn internal_id() -> RequestId {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    RequestId(INTERNAL | NEXT.fetch_add(1, Ordering::Relaxed))
}

impl Conductor {
    /// Takes a decision in: its parent goes in line, or with one question its
    /// one fork.
    pub(super) fn take_decision(&mut self, submission: DecisionSubmission) {
        let DecisionSubmission { id, questions, prompts, labels, shared, priority, sink } =
            submission;
        if let Some(why) = &self.failed {
            sink.send(DecisionOutcome::Failed(EngineError::Internal(why.clone())));
            return;
        }
        if self.decisions.contains_key(&id) || self.by_id.contains_key(&id) {
            let error = EngineError::InvalidRequest(format!("request id {id} is in use"));
            sink.send(DecisionOutcome::Failed(error));
            return;
        }
        let n = questions.len();
        let decision = Decision {
            questions,
            prompts,
            labels,
            shared,
            priority,
            sink,
            parts: Vec::new(),
            logits: vec![None; n],
            cached: 0,
        };
        self.decisions.insert(id, decision);
        if n > 1 && shared > 0 {
            let prompt = self.decisions[&id].prompts[0][..shared].to_vec();
            self.enqueue_part(id, prompt, Role::Parent { decision: id });
        } else {
            self.enqueue_forks(id);
        }
    }

    /// Puts the forks of decision `id` in line.
    fn enqueue_forks(&mut self, id: RequestId) {
        let Some(decision) = self.decisions.get(&id) else { return };
        let forks: Vec<(Vec<u32>, Role)> = (0..decision.questions.len())
            .map(|question| {
                let role = Role::Fork {
                    decision: id,
                    question,
                    labels: decision.labels[question].clone(),
                };
                (decision.prompts[question].clone(), role)
            })
            .collect();
        for (prompt, role) in forks {
            self.enqueue_part(id, prompt, role);
        }
    }

    /// Puts a request of decision `id` in line: `prompt`, read and not
    /// continued.
    fn enqueue_part(&mut self, id: RequestId, prompt: Vec<u32>, role: Role) {
        let part = internal_id();
        let priority = self.decisions.get(&id).map_or(0, |d| d.priority);
        let errors = self.part_errors.clone();
        let sink = EventSink::new(move |event| {
            if let Event::Error(error) = event {
                errors.lock().unwrap_or_else(|e| e.into_inner()).push((part, error));
            }
            true
        });
        if let Some(decision) = self.decisions.get_mut(&id) {
            decision.parts.push(part);
        }
        self.enqueue(Submission {
            id: part,
            prompt,
            boundaries: Vec::new(),
            images: Vec::new(),
            max_tokens: Some(1),
            stop: Watch::new(&[], None),
            sampling: Sampling::default(),
            draft: false,
            filter: None,
            priority,
            displace: false,
            sink,
            role,
        });
        // A request the line refused at once (it does not fit the pool) has
        // ended already.
        if !self.by_id.contains_key(&part) {
            self.fail_decision(id, part);
        }
    }

    /// What the logits of the step mean for a fork whose last row ran: they
    /// are its question's, in the order of its labels. Called from the commit
    /// with the scores of the part, which ends there.
    pub(super) fn fork_read(&mut self, decision: RequestId, question: usize, logits: &[f32]) {
        if let Some(d) = self.decisions.get_mut(&decision)
            && let Some(slot) = d.logits.get_mut(question)
        {
            *slot = Some(logits.to_vec());
        }
    }

    /// A request of a decision ended (`role` is what it was for; `id` its id):
    /// the parent's end lets the forks start; the last fork's end answers the
    /// decision. One that ended without its work (cancelled, failed, or the
    /// engine going down) fails or cancels the decision.
    pub(super) fn part_ended(&mut self, id: RequestId, role: &Role, finish: FinishReason) {
        let decision = match role {
            Role::Request => return,
            Role::Parent { decision } | Role::Fork { decision, .. } => *decision,
        };
        let Some(d) = self.decisions.get_mut(&decision) else { return };
        d.parts.retain(|part| *part != id);
        match (role, finish) {
            (Role::Parent { .. }, FinishReason::Stop) => self.enqueue_forks(decision),
            // The last fork to end answers: the forks that read their
            // labels in the same step end one after the other.
            (Role::Fork { question, .. }, FinishReason::Stop)
                if d.logits.get(*question).is_some_and(Option::is_some) =>
            {
                if d.answered() && d.parts.is_empty() {
                    self.answer(decision);
                }
            }
            _ => self.fail_decision(decision, id),
        }
    }

    /// Notes what the prefix cache held of the parent's prompt.
    pub(super) fn parent_cached(&mut self, decision: RequestId, cached: u32) {
        if let Some(d) = self.decisions.get_mut(&decision) {
            d.cached = cached;
        }
    }

    /// Ends decision `decision` because its request `part` ended without its
    /// work: with the error that request reported, or as cancelled.
    fn fail_decision(&mut self, decision: RequestId, part: RequestId) {
        let error = {
            let mut errors = self.part_errors.lock().unwrap_or_else(|e| e.into_inner());
            let found = errors.iter().position(|(id, _)| *id == part);
            found.map(|i| errors.swap_remove(i).1)
        };
        let outcome = match (error, &self.failed) {
            (Some(error), _) => DecisionOutcome::Failed(error),
            (None, Some(why)) => DecisionOutcome::Failed(EngineError::Internal(why.clone())),
            (None, None) => DecisionOutcome::Cancelled,
        };
        self.end_decision(decision, outcome);
    }

    /// Ends decision `decision` with `outcome`, and every request of it that
    /// is still held.
    pub(super) fn end_decision(&mut self, decision: RequestId, outcome: DecisionOutcome) {
        let Some(d) = self.decisions.remove(&decision) else { return };
        for part in &d.parts {
            match self.by_id.get(part) {
                Some(Place::Slot(slot)) => self.retire_with(*slot, FinishReason::Cancelled, false),
                Some(Place::Waiting) => {
                    if let Some(i) = self.waiting.iter().position(|a| a.id == *part) {
                        self.dismiss(i, FinishReason::Cancelled);
                    }
                }
                None => {}
            }
        }
        self.part_errors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(id, _)| !d.parts.contains(id));
        d.sink.send(outcome);
    }

    /// Answers decision `decision`, whose every fork has read its logits.
    fn answer(&mut self, decision: RequestId) {
        let Some(d) = self.decisions.get(&decision) else { return };
        let answers: Result<Vec<_>, EngineError> = d
            .questions
            .iter()
            .zip(&d.logits)
            .map(|(question, logits)| question.answer(logits.as_deref().unwrap_or(&[])))
            .collect();
        let outcome = match answers {
            Ok(answers) => {
                let rest: usize = d.prompts.iter().map(|p| p.len() - d.shared).sum();
                DecisionOutcome::Decided(DecisionResult {
                    answers,
                    usage: Usage {
                        prompt_tokens: (d.shared + rest) as u32,
                        completion_tokens: d.questions.len() as u32,
                        cached_prompt_tokens: d.cached,
                        ..Usage::default()
                    },
                    prefix_tokens: d.shared as u32,
                })
            }
            Err(error) => DecisionOutcome::Failed(error),
        };
        self.stats.counters.decisions += 1;
        self.end_decision(decision, outcome);
    }

    /// Ends every decision, as the engine goes down or fails.
    pub(super) fn end_decisions(&mut self) {
        let ids: Vec<RequestId> = self.decisions.keys().copied().collect();
        for id in ids {
            let outcome = match &self.failed {
                Some(why) => DecisionOutcome::Failed(EngineError::Internal(why.clone())),
                None => DecisionOutcome::Cancelled,
            };
            self.end_decision(id, outcome);
        }
    }
}
