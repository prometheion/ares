//! What keeps a reply's thinking within bounds, and a reply out of a loop.
//!
//! [`Guard`] is a [`TokenFilter`] around the request's own rule (a grammar, or
//! none). It follows the reply token by token, as text: where the thinking
//! opens and closes (the tags the model's template declares, followed by the
//! [`Tracker`] that also counts the thinking's tokens for the reply's usage),
//! how many tokens it has taken, and whether the last tokens go round in a
//! cycle.
//!
//! - **The budget** (llama.cpp's `reasoning_budget_tokens`): once the thinking
//!   has taken that many tokens, the next rows may each pick one token only,
//!   the tokens of the budget's message and of the end of the thinking, which
//!   the model would otherwise not write there; then the answer follows. As in
//!   llama.cpp the end waits for a character the budget cut in two, and the
//!   tokens of the closing tag are not counted.
//! - **A loop in the thinking**: the last [`LOOP_TOKENS`] tokens of the
//!   thinking are a cycle of at most [`LOOP_PERIOD`] tokens repeated (at least
//!   eight times): the thinking is ended the same way.
//! - **A loop in the answer**: one token [`RUN_TOKENS`] times in a row
//!   (Strata's guard against a model that writes one token for ever); the
//!   reply ends there as a limit ends it ([`TokenFilter::cut`]). A cycle of
//!   several tokens is not cut in the answer: a matrix of zeros is one.
//!
//! A forced row has a mask of one token, so its pick is that token whatever
//! the sampler draws; a draft that is not that token is cut from the window,
//! and the rows before the forcing are free (a model's drafts go on there).

use std::sync::Arc;

use ares_sampler::TokenMask;

use crate::sampling::TokenFilter;
use crate::watch::{Place, Thinking, Tracker};

/// The tokens of thinking that, all one cycle, end the thinking.
pub const LOOP_TOKENS: usize = 512;
/// The longest cycle that counts as a loop.
pub const LOOP_PERIOD: usize = 64;
/// The times one token in a row that end a reply.
pub const RUN_TOKENS: usize = 256;

/// What the guard does about the thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The thinking opens and ends where the text says ([`Tracker`]).
    Following,
    /// The thinking is to end, after the character that is cut in two.
    Waiting,
    /// Token `at` of the forced end is next.
    Forcing { at: usize },
    /// The thinking is over, or the reply has none.
    After,
}

/// What the guard knows after a number of tokens of the reply.
#[derive(Debug, Clone)]
struct State {
    phase: Phase,
    /// Where the thinking is, for a model that thinks.
    thinking: Option<Tracker>,
    /// The bytes of a character that is not whole yet.
    pending: Vec<u8>,
    /// The last tokens, the newest last, and for each period `p` how many
    /// tokens in a row equal the one `p` before them (index `p - 1`).
    recent: Vec<u32>,
    runs: [u32; LOOP_PERIOD],
    /// The reply is to end here.
    cut: bool,
}

/// The bytes a token writes (`Model::piece`).
pub(crate) type Pieces = Arc<dyn Fn(u32, &mut Vec<u8>) + Send + Sync>;

/// See the module documentation.
pub(crate) struct Guard {
    pieces: Pieces,
    inner: Option<Box<dyn TokenFilter>>,
    /// The model's thinking, if it thinks.
    tags: Option<Thinking>,
    budget: Option<u32>,
    /// The tokens that end the thinking: the budget's message, then the end
    /// of the thinking as the template writes it.
    forced: Vec<u32>,
    /// The state after the first `read` tokens of the reply.
    state: State,
    read: usize,
    piece: Vec<u8>,
}

impl Guard {
    /// A guard for a reply whose tokens write `pieces` and whose thinking
    /// `tags` describe (none for a model that does not think), with a budget
    /// of `budget` tokens of thinking; `forced` are the tokens that end the
    /// thinking. `inner` is the request's own rule.
    pub fn new(
        pieces: Pieces,
        inner: Option<Box<dyn TokenFilter>>,
        tags: Option<Thinking>,
        budget: Option<u32>,
        forced: Vec<u32>,
    ) -> Self {
        let state = start(tags.as_ref(), budget);
        Self { pieces, inner, tags, budget, forced, state, read: 0, piece: Vec::new() }
    }

    /// The state after all of `reply`, kept for the next call.
    fn at(&mut self, reply: &[u32]) -> State {
        if reply.len() < self.read {
            // The reply does not shrink; start again if it did.
            self.state = start(self.tags.as_ref(), self.budget);
            self.read = 0;
        }
        let mut state = std::mem::replace(&mut self.state, start(None, None));
        for &token in &reply[self.read..] {
            self.advance(&mut state, token);
        }
        self.read = reply.len();
        self.state = state.clone();
        state
    }

    /// The state after `state` and `tokens`, without keeping it.
    fn after(&mut self, mut state: State, tokens: &[u32]) -> State {
        for &token in tokens {
            self.advance(&mut state, token);
        }
        state
    }

    /// Takes one token of the reply into `state`.
    fn advance(&mut self, state: &mut State, token: u32) {
        if state.cut {
            return;
        }
        self.piece.clear();
        (self.pieces)(token, &mut self.piece);
        // The loop: for every period, the tokens in a row that repeat.
        for p in 1..=LOOP_PERIOD {
            let same = state.recent.len() >= p && state.recent[state.recent.len() - p] == token;
            state.runs[p - 1] = if same { state.runs[p - 1] + 1 } else { 0 };
        }
        if state.recent.len() == LOOP_PERIOD {
            state.recent.remove(0);
        }
        state.recent.push(token);
        // Whether the text ends with a whole character.
        state.pending.extend_from_slice(&self.piece);
        let whole = match std::str::from_utf8(&state.pending) {
            Ok(_) => true,
            Err(error) => error.error_len().is_some(),
        };
        if whole {
            state.pending.clear();
        }
        let (Some(tags), Some(thinking)) = (&self.tags, &mut state.thinking) else {
            state.cut = state.runs[0] as usize + 1 >= RUN_TOKENS;
            return;
        };
        let was = thinking.place();
        thinking.token(tags, &self.piece);
        let (place, used) = (thinking.place(), thinking.used());
        let ended = matches!(place, Place::Done(_));
        match state.phase {
            Phase::Following if ended => state.phase = Phase::After,
            Phase::Following if was == Place::Before => {
                if matches!(place, Place::Inside { .. }) {
                    // The model opened its thinking with this token.
                    state.runs = [0; LOOP_PERIOD];
                    if self.budget == Some(0) {
                        state.phase = Phase::Forcing { at: 0 };
                    }
                }
            }
            Phase::Following => {
                let used = used.unwrap_or(0);
                let spent = self.budget.is_some_and(|budget| used >= budget);
                let looped = used as usize >= LOOP_TOKENS
                    && (1..=LOOP_PERIOD).any(|p| state.runs[p - 1] as usize + p >= LOOP_TOKENS);
                if spent || looped {
                    state.phase = if whole { Phase::Forcing { at: 0 } } else { Phase::Waiting };
                }
            }
            Phase::Waiting => {
                state.phase = match (ended, whole) {
                    (true, _) => Phase::After,
                    (false, true) => Phase::Forcing { at: 0 },
                    (false, false) => Phase::Waiting,
                };
            }
            Phase::Forcing { at } => {
                state.phase = if at + 1 >= self.forced.len() {
                    state.runs = [0; LOOP_PERIOD];
                    Phase::After
                } else {
                    Phase::Forcing { at: at + 1 }
                };
            }
            Phase::After => state.cut = state.runs[0] as usize + 1 >= RUN_TOKENS,
        }
        if self.forced.is_empty() && matches!(state.phase, Phase::Forcing { .. } | Phase::Waiting) {
            state.phase = Phase::After;
        }
    }

    /// The token a row after `state` must pick, if one is forced.
    fn forced(&self, state: &State) -> Option<u32> {
        match state.phase {
            Phase::Forcing { at } => self.forced.get(at).copied(),
            _ => None,
        }
    }
}

/// The state before the first token of a reply whose thinking `tags`
/// describe, with a budget of `budget` tokens of thinking.
fn start(tags: Option<&Thinking>, budget: Option<u32>) -> State {
    let thinking = tags.map(Tracker::new);
    let phase = match &thinking {
        Some(t) if budget == Some(0) && matches!(t.place(), Place::Inside { .. }) => {
            Phase::Forcing { at: 0 }
        }
        Some(_) => Phase::Following,
        None => Phase::After,
    };
    State {
        phase,
        thinking,
        pending: Vec::new(),
        recent: Vec::new(),
        runs: [0; LOOP_PERIOD],
        cut: false,
    }
}

impl TokenFilter for Guard {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        let state = self.at(reply);
        let state = self.after(state, drafts);
        if let Some(token) = self.forced(&state) {
            mask.fill(false);
            mask.set(token, true);
            return true;
        }
        match &mut self.inner {
            Some(inner) => inner.mask(reply, drafts, mask),
            None => false,
        }
    }

    fn admitted(&mut self, reply: &[u32], drafts: &[u32]) -> usize {
        let mut state = self.at(reply);
        let mut allowed = drafts.len();
        for (k, &draft) in drafts.iter().enumerate() {
            if state.cut || self.forced(&state).is_some_and(|token| token != draft) {
                allowed = k;
                break;
            }
            self.advance(&mut state, draft);
        }
        match &mut self.inner {
            Some(inner) => allowed.min(inner.admitted(reply, drafts)),
            None => allowed,
        }
    }

    fn free_rows(&mut self, reply: &[u32], tokens: &[u32]) -> usize {
        let mut state = self.at(reply);
        let mut free = 0;
        for k in 0..=tokens.len() {
            if self.forced(&state).is_some() {
                break;
            }
            free += 1;
            if let Some(&token) = tokens.get(k) {
                self.advance(&mut state, token);
            }
        }
        match &mut self.inner {
            Some(inner) => free.min(inner.free_rows(reply, tokens)),
            None => free,
        }
    }

    fn cut(&mut self, reply: &[u32]) -> bool {
        self.at(reply).cut
    }

    fn failure(&mut self) -> Option<String> {
        self.inner.as_mut().and_then(|inner| inner.failure())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens below 256 write their byte; 1000 and 1001 the two bytes of `é`.
    fn pieces() -> Pieces {
        Arc::new(|token, text: &mut Vec<u8>| match token {
            1000 => text.push(0xc3),
            1001 => text.push(0xa9),
            byte => text.push(byte as u8),
        })
    }

    fn tokens(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn tags(inside: bool) -> Thinking {
        Thinking {
            inside,
            open: "<think>".into(),
            close: "</think>".into(),
            interrupt: Some("<tool_call>".into()),
        }
    }

    fn guard(inside: bool, budget: Option<u32>) -> Guard {
        Guard::new(pieces(), None, Some(tags(inside)), budget, tokens("\n</think>\n\n"))
    }

    /// The token a row after `reply` is held to, if one.
    fn held(guard: &mut Guard, reply: &[u32]) -> Option<u32> {
        let mut mask = TokenMask::all(1100);
        guard.mask(reply, &[], &mut mask).then(|| {
            assert_eq!(mask.count(), 1);
            (0..1100).find(|&t| mask.allowed(t)).unwrap()
        })
    }

    #[test]
    fn a_spent_budget_ends_the_thinking_with_the_tokens_of_its_end() {
        let mut g = guard(true, Some(5));
        assert_eq!(held(&mut g, &tokens("abcd")), None);
        let end = tokens("\n</think>\n\n");
        let mut reply = tokens("abcde");
        for &token in &end {
            assert_eq!(held(&mut g, &reply), Some(token));
            reply.push(token);
        }
        // The answer is free, and a long one is not cut.
        assert_eq!(held(&mut g, &reply), None);
        reply.extend(tokens(&"the answer ".repeat(100)));
        assert_eq!(held(&mut g, &reply), None);
        assert!(!g.cut(&reply));

        // Drafts: the free rows end where the forcing starts, and a draft
        // that is not the forced token is cut.
        let mut g = guard(true, Some(5));
        assert_eq!(g.free_rows(&tokens("abc"), &tokens("de\n")), 2);
        assert_eq!(g.admitted(&tokens("abcde"), &tokens("\n<x")), 2);
        assert_eq!(g.admitted(&tokens("ab"), &tokens("cdefg")), 3);
    }

    #[test]
    fn a_thinking_that_ends_itself_is_left_alone_and_the_tags_count_not() {
        let mut g = guard(true, Some(8));
        let reply = tokens("abc</think>\n\nanswer that is long");
        assert_eq!(held(&mut g, &reply), None);
        // The thinking opened by the model: the tokens after its tag count.
        let mut g = guard(false, Some(3));
        assert_eq!(held(&mut g, &tokens("  <think>ab")), None);
        assert_eq!(held(&mut g, &tokens("  <think>abc")), Some(u32::from(b'\n')));
        // An answer without thinking is never bound.
        let mut g = guard(false, Some(1));
        assert_eq!(held(&mut g, &tokens("Hello <think>")), None);
        // The marker of a call ends the thinking too.
        let mut g = guard(true, Some(12));
        assert_eq!(held(&mut g, &tokens("ab<tool_call>xyz and more")), None);
    }

    #[test]
    fn the_end_waits_for_a_character_the_budget_cut_in_two() {
        let mut g = guard(true, Some(1));
        assert_eq!(held(&mut g, &[1000]), None);
        assert_eq!(held(&mut g, &[1000, 1001]), Some(u32::from(b'\n')));
        // A budget of 0 ends the thinking at once.
        let mut g = guard(true, Some(0));
        assert_eq!(held(&mut g, &[]), Some(u32::from(b'\n')));
    }

    #[test]
    fn a_loop_ends_the_thinking_and_a_run_ends_the_answer() {
        let mut g = guard(true, None);
        let looping = tokens(&"ab".repeat(400));
        let first = (1..=looping.len()).find(|&n| held(&mut g, &looping[..n]).is_some());
        assert_eq!(first, Some(LOOP_TOKENS), "the thinking is ended after its loop");
        // Text that changes does not end it.
        let mut g = guard(true, None);
        let mut seed = 7u64;
        let text: String = (0..2000)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                char::from(b'a' + (seed >> 59) as u8 % 26)
            })
            .collect();
        let changing = tokens(&text);
        assert!((1..=changing.len()).all(|n| held(&mut g, &changing[..n]).is_none()));

        // In the answer a cycle of several tokens is no loop, a run of one is.
        let mut g = Guard::new(pieces(), None, None, None, Vec::new());
        assert!(!g.cut(&tokens(&"0, ".repeat(400))));
        let run = tokens(&"!".repeat(RUN_TOKENS));
        assert!(!g.cut(&run[..RUN_TOKENS - 1]));
        let mut g = Guard::new(pieces(), None, None, None, Vec::new());
        assert!(g.cut(&run));
    }
}
