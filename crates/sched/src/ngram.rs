//! The n-gram drafter: drafts from everything the engine has read and written.
//!
//! Models copy a lot: they quote the prompt, rewrite a file with one line
//! changed, repeat a function signature, and agents send the same documents
//! in request after request. Whenever the text being generated has an earlier
//! occurrence, the tokens that followed it are a draft that costs a walk in an automaton.
//!
//! [`NgramCorpus`] is one suffix automaton over the text of all sequences,
//! and [`NgramDrafter`] is one sequence's place in it. For the current end
//! of a sequence it finds the **longest** suffix that occurred before,
//! anywhere, in this request or another, and proposes what followed its most
//! recent occurrence. A match of the last `n` tokens of one sequence, which
//! this replaced, cannot tell two places apart that end in the same `n`
//! tokens, and it knows nothing of other requests.
//!
//! # Why an automaton
//!
//! The code read for this (TensorRT-LLM's `suffixAutomaton.h`, SGLang's
//! `suffix_automaton.cpp` and `trie.cpp`) has both structures. SGLang's trie
//! is incremental and evicts, but it is bounded in depth: it stores every run
//! of up to `d` tokens, which is up to `d` nodes and `d` lookups per token,
//! and it never matches more than `d`. A suffix automaton has at most two
//! states and three transitions per token whatever the match length, an
//! insertion is amortised constant work, and the longest match of the
//! current end is not searched for at all: it is where the suffix links of
//! the sequence's own state lead (TensorRT-LLM's `lookup`). What an
//! automaton cannot do is forget. That is solved outside it: the corpus is
//! two automata, a young one that grows and an old one that is only read;
//! when the young one is full the old one is dropped and the young one
//! takes its place. Text therefore lives for one to two generations, and
//! memory is bounded by construction.
//!
//! # What is kept per state
//!
//! Besides the automaton itself (length, suffix link, transitions), a state
//! keeps where its most recent occurrence went on (`cont`) and how many of
//! its occurrences went on at all (`seen`). Both are written when a token is
//! appended, for the states of the text before that token, along at most
//! [`NgramConfig::max_hops`] suffix links. The longest suffixes come first
//! on that walk, so the cap leaves out only short ones.
//!
//! # Confidence
//!
//! Every proposed token comes with a probability that it is accepted, given
//! that the ones before it were, which is what
//! [`choose`](crate::choose) takes. It is read from a small
//! table by two things the automaton knows: how long the matched context is
//! at that token, and how the earlier occurrences of that context went on
//! (it occurred once; several times and always like this; mostly; seldom).
//! The table starts from acceptance measured by replaying real text and
//! follows what verification reports ([`NgramDrafter::observe`]), so it
//! stays right for the model and the sampling in use.

/// No state, no position.
const NONE: u32 = u32::MAX;
/// The start state of an automaton.
const ROOT: u32 = 0;
/// An unused slot of the transition table.
const EMPTY: u64 = u64::MAX;
/// How many tokens back a drafter can be cut ([`NgramDrafter::truncate`]).
const HISTORY: usize = 32;
/// The longest draft a drafter remembers the table cells of.
const MAX_DRAFT: usize = 64;

/// The settings of a [`NgramCorpus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramConfig {
    /// The most memory the corpus takes, for both automata (never less than
    /// two automata of 64 tokens, about 14 KiB). A token costs
    /// 112 to 176 bytes (8 for the text, 40 for its two states, 64 to 128 for
    /// the transition table), so 64 MiB hold 380 000 to 600 000 tokens.
    pub max_bytes: usize,
    /// The shortest match that is drafted from. Short matches are frequent
    /// and often wrong, which their confidence says; with the draft length
    /// chosen by confidence, 2 gave more tokens per second than 3 on English
    /// and on code, and 1 a little more again on those and less on German.
    pub min_match: u32,
    /// How many suffix links a walk follows at most. On real text 2 and 96
    /// gave the same drafts; the cap is there for text that repeats one
    /// token, where a walk is as long as the run.
    pub max_hops: u32,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self { max_bytes: 64 << 20, min_match: 2, max_hops: 8 }
    }
}

#[derive(Clone, Copy, Debug)]
struct State {
    /// The length of the longest string of the state.
    len: u32,
    link: u32,
    /// The position of the token that followed the most recent occurrence
    /// that went on.
    cont: u32,
    /// How many occurrences went on.
    seen: u32,
    /// The first slot of this state's transitions.
    edges: u32,
}

/// One suffix automaton over the text of many sequences.
#[derive(Debug)]
struct Automaton {
    states: Vec<State>,
    /// The transition table: `(state, token)` to the target state, open
    /// addressing; `chain` links the slots of one state.
    keys: Vec<u64>,
    targets: Vec<u32>,
    chain: Vec<u32>,
    entries: usize,
    /// The tokens of all sequences in the order they came, and for each the
    /// position of the next token of the same sequence.
    text: Vec<u32>,
    next: Vec<u32>,
    /// The number of tokens this automaton takes before it is full.
    capacity: usize,
}

impl Automaton {
    /// Slots of the transition table for `tokens` tokens: at most three
    /// transitions per token, at most three quarters full.
    fn slots(tokens: usize) -> usize {
        (tokens * 4).next_power_of_two().max(16)
    }

    /// The bytes an automaton for `tokens` tokens takes.
    fn bytes(tokens: usize) -> usize {
        tokens * 8 + (2 * tokens + 2) * size_of::<State>() + Self::slots(tokens) * 16
    }

    fn with_capacity(tokens: usize) -> Self {
        let slots = Self::slots(tokens);
        let mut states = Vec::with_capacity(2 * tokens + 2);
        states.push(State { len: 0, link: NONE, cont: NONE, seen: 0, edges: NONE });
        Self {
            states,
            keys: vec![EMPTY; slots],
            targets: vec![NONE; slots],
            chain: vec![NONE; slots],
            entries: 0,
            text: Vec::with_capacity(tokens),
            next: Vec::with_capacity(tokens),
            capacity: tokens,
        }
    }

    fn is_full(&self) -> bool {
        self.text.len() >= self.capacity
    }

    fn slot(&self, state: u32, token: u32) -> (usize, bool) {
        let key = u64::from(state) << 32 | u64::from(token);
        let mask = self.keys.len() - 1;
        let mut at = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & mask;
        loop {
            match self.keys[at] {
                found if found == key => return (at, true),
                EMPTY => return (at, false),
                _ => at = (at + 1) & mask,
            }
        }
    }

    fn step(&self, state: u32, token: u32) -> Option<u32> {
        let (at, found) = self.slot(state, token);
        found.then(|| self.targets[at])
    }

    /// Sets the transition of `state` by `token`.
    fn set(&mut self, state: u32, token: u32, target: u32) {
        let (at, found) = self.slot(state, token);
        if !found {
            // An automaton never has more than three transitions per token,
            // so the table cannot fill; a full table would only lose matches.
            if self.entries + 1 >= self.keys.len() {
                return;
            }
            self.keys[at] = u64::from(state) << 32 | u64::from(token);
            self.chain[at] = self.states[state as usize].edges;
            self.states[state as usize].edges = at as u32;
            self.entries += 1;
        }
        self.targets[at] = target;
    }

    /// A copy of `of` that takes over its strings up to `len` long.
    fn clone_state(&mut self, of: u32, len: u32) -> u32 {
        let clone = self.states.len() as u32;
        let source = self.states[of as usize];
        self.states.push(State { len, edges: NONE, ..source });
        let mut at = source.edges;
        while at != NONE {
            let token = self.keys[at as usize] as u32;
            self.set(clone, token, self.targets[at as usize]);
            at = self.chain[at as usize];
        }
        clone
    }

    /// The state of a text after `token`, from the state `last` before it.
    /// This is the construction of a suffix automaton over several strings:
    /// each sequence extends from its own state.
    fn extend(&mut self, last: u32, token: u32) -> u32 {
        let len = self.states[last as usize].len + 1;
        if let Some(q) = self.step(last, token) {
            if self.states[q as usize].len == len {
                return q;
            }
            // The text exists as part of longer strings: split them off.
            let clone = self.clone_state(q, len);
            let mut p = last;
            while p != NONE && self.step(p, token) == Some(q) {
                self.set(p, token, clone);
                p = self.states[p as usize].link;
            }
            self.states[q as usize].link = clone;
            return clone;
        }
        let current = self.states.len() as u32;
        self.states.push(State { len, link: ROOT, cont: NONE, seen: 0, edges: NONE });
        let mut p = last;
        let q = loop {
            if p == NONE {
                break None;
            }
            if let Some(q) = self.step(p, token) {
                break Some(q);
            }
            self.set(p, token, current);
            p = self.states[p as usize].link;
        };
        if let Some(q) = q {
            let shorter = self.states[p as usize].len + 1;
            if self.states[q as usize].len == shorter {
                self.states[current as usize].link = q;
            } else {
                let clone = self.clone_state(q, shorter);
                while p != NONE && self.step(p, token) == Some(q) {
                    self.set(p, token, clone);
                    p = self.states[p as usize].link;
                }
                self.states[q as usize].link = clone;
                self.states[current as usize].link = clone;
            }
        }
        current
    }

    /// Appends `token` to the sequence whose state is `last` and whose newest
    /// token is at `tail`. Returns the new state and the token's position.
    fn push(&mut self, last: u32, tail: u32, token: u32, max_hops: u32) -> (u32, u32) {
        let position = self.text.len() as u32;
        self.text.push(token);
        self.next.push(NONE);
        if tail != NONE {
            self.next[tail as usize] = position;
        }
        // The text before this token went on: its suffixes, the longest
        // first, have a continuation here.
        let mut state = last;
        for _ in 0..max_hops {
            if state == ROOT || state == NONE {
                break;
            }
            let s = &mut self.states[state as usize];
            s.cont = position;
            s.seen = s.seen.saturating_add(1);
            state = s.link;
        }
        (self.extend(last, token), position)
    }

    /// Follows `token` from a match of `matched` tokens in `state`, for a
    /// text that is not part of this automaton: the longest suffix of the
    /// text so far that occurs in it.
    fn follow(&self, mut state: u32, mut matched: u32, token: u32) -> (u32, u32) {
        loop {
            if let Some(next) = self.step(state, token) {
                return (next, matched + 1);
            }
            if state == ROOT {
                return (ROOT, 0);
            }
            state = self.states[state as usize].link;
            matched = self.states[state as usize].len;
        }
    }

    /// The longest match of at least `min_match` tokens that went on
    /// somewhere other than at `blocked`, starting from a match of `matched`
    /// tokens in `state`: the state and the length.
    fn longest(
        &self,
        mut state: u32,
        mut matched: u32,
        min_match: u32,
        blocked: u32,
        max_hops: u32,
    ) -> Option<(u32, u32)> {
        for _ in 0..=max_hops {
            if state == ROOT || state == NONE || matched < min_match.max(1) {
                return None;
            }
            let s = &self.states[state as usize];
            if s.cont != NONE && s.cont != blocked {
                return Some((state, matched));
            }
            state = s.link;
            matched = self.states.get(state as usize).map_or(0, |s| s.len);
        }
        None
    }
}

/// How the earlier occurrences of a context went on, as far as a draft token
/// is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Agreement {
    /// The context occurred once before.
    Once,
    /// Several times, and every time it went on with this token.
    Always,
    /// At least half of the times.
    Mostly,
    /// Less than half of the times.
    Seldom,
}

const LENGTHS: usize = 10;
const AGREEMENTS: usize = 4;

/// The table cell of a context of `len` tokens whose `occurrences` earlier
/// occurrences went on with the proposed token `agreeing` times.
fn cell(len: u32, occurrences: u32, agreeing: u32) -> u8 {
    let length = match len {
        0..=4 => len.saturating_sub(1) as usize,
        5..=6 => 4,
        7..=8 => 5,
        9..=12 => 6,
        13..=16 => 7,
        17..=32 => 8,
        _ => 9,
    };
    let agreement = if occurrences <= 1 {
        Agreement::Once
    } else if agreeing >= occurrences {
        Agreement::Always
    } else if agreeing.saturating_mul(2) >= occurrences {
        Agreement::Mostly
    } else {
        Agreement::Seldom
    };
    (length * AGREEMENTS + agreement as usize) as u8
}

/// Acceptance by context length (rows: 1, 2, 3, 4, 5-6, 7-8, 9-12, 13-16,
/// 17-32, more) and agreement (columns: once, always, mostly, seldom), as
/// measured by replaying the text of this repository (English prose, Rust
/// and a little German) with the vocabulary of Qwen3.5, plain and as edits
/// of a prompt: `examples/ngram_bench.rs` prints this table. A context that
/// went on in several ways is worth about the same at any length; one that
/// always went on the same way, or occurred once, is worth more the longer
/// it is.
const PRIOR: [[f32; AGREEMENTS]; LENGTHS] = [
    [0.24, 0.51, 0.39, 0.14],
    [0.32, 0.66, 0.37, 0.19],
    [0.45, 0.75, 0.43, 0.21],
    [0.56, 0.80, 0.44, 0.22],
    [0.65, 0.84, 0.42, 0.23],
    [0.78, 0.88, 0.41, 0.24],
    [0.88, 0.92, 0.38, 0.26],
    [0.94, 0.95, 0.40, 0.28],
    [0.99, 0.98, 0.39, 0.38],
    [0.99, 0.99, 0.57, 0.50],
];

/// What the automaton knows about a drafter's current match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramMatch {
    /// The length of the matched suffix in tokens.
    pub len: u32,
    /// How many earlier occurrences of it went on.
    pub occurrences: u32,
    /// How many of them went on with the token that is proposed first.
    pub agreeing: u32,
}

/// The text of all sequences, searchable: one suffix automaton that every
/// [`NgramDrafter`] reads and extends.
///
/// The corpus is two automata, a young one that grows and an old one that is
/// only read. When the young one is full the old one is dropped and the
/// young one takes its place, so text is found for one to two automata's
/// worth of tokens after it was read, and memory never exceeds
/// [`NgramConfig::max_bytes`]. It also holds the table from which the
/// confidence of a proposed token is read (by the length of the matched
/// context and by how its earlier occurrences went on), which starts from
/// measured acceptance and follows what [`NgramDrafter::observe`] reports.
#[derive(Debug)]
pub struct NgramCorpus {
    config: NgramConfig,
    young: Automaton,
    old: Option<Automaton>,
    /// The number of the young automaton. A drafter that was last used with
    /// another one finds out by it.
    generation: u64,
    /// Tokens per automaton.
    capacity: usize,
    /// Acceptance by table cell.
    rate: [[f32; AGREEMENTS]; LENGTHS],
}

impl NgramCorpus {
    /// A corpus within `config.max_bytes`. The memory is taken as the text
    /// comes in, never more than that.
    pub fn new(config: NgramConfig) -> Self {
        // The largest automaton of which two fit; a small one at the least.
        let (mut low, mut high) = (64usize, 64usize);
        while 2 * Automaton::bytes(high * 2) <= config.max_bytes && high < (u32::MAX / 8) as usize {
            high *= 2;
        }
        low = low.max(high);
        let mut step = high / 2;
        while step > 0 {
            if 2 * Automaton::bytes(low + step) <= config.max_bytes {
                low += step;
            }
            step /= 2;
        }
        Self {
            config,
            young: Automaton::with_capacity(low),
            old: None,
            generation: 0,
            capacity: low,
            rate: PRIOR,
        }
    }

    /// The memory the corpus takes when both automata are full.
    pub fn bytes(&self) -> usize {
        2 * Automaton::bytes(self.capacity)
    }

    /// The number of tokens the corpus holds now.
    pub fn tokens(&self) -> usize {
        self.young.text.len() + self.old.as_ref().map_or(0, |old| old.text.len())
    }

    /// How many tokens an automaton takes before the older text is dropped.
    /// Text is findable for at least this many tokens after it was read and
    /// for at most twice as many.
    pub fn tokens_per_generation(&self) -> usize {
        self.capacity
    }

    /// The acceptance the corpus expects for a context of `len` tokens whose
    /// `occurrences` earlier occurrences went on with the proposed token
    /// `agreeing` times.
    pub fn expected_acceptance(&self, len: u32, occurrences: u32, agreeing: u32) -> f64 {
        let cell = cell(len, occurrences, agreeing) as usize;
        f64::from(self.rate[cell / AGREEMENTS][cell % AGREEMENTS])
    }

    /// The acceptance table as it stands: a row per context length (1, 2, 3,
    /// 4, 5-6, 7-8, 9-12, 13-16, 17-32, more) and a column per agreement
    /// (once, always, mostly, seldom).
    pub fn table(&self) -> [[f32; AGREEMENTS]; LENGTHS] {
        self.rate
    }

    fn rotate(&mut self) {
        let fresh = Automaton::with_capacity(self.capacity);
        self.old = Some(std::mem::replace(&mut self.young, fresh));
        self.generation += 1;
    }
}

/// One sequence's place in a [`NgramCorpus`], with the corpus passed to
/// every call.
#[derive(Clone, Debug)]
pub struct NgramDrafter {
    len: usize,
    /// The automaton `state` and `tail` belong to; `u64::MAX` before the
    /// first token.
    generation: u64,
    /// The state of this sequence's text in the young automaton and the
    /// position of its newest token there.
    state: u32,
    tail: u32,
    /// The longest suffix of this sequence's text in the old automaton.
    old_state: u32,
    old_matched: u32,
    /// The four numbers above after each of the last tokens, by length
    /// modulo [`HISTORY`], and how many of them are valid.
    history: [[u32; 4]; HISTORY],
    valid: usize,
    /// A continuation this sequence was cut away from, as a position in the
    /// automaton `blocked_generation`.
    blocked: u32,
    blocked_generation: u64,
    /// The table cells of the tokens of the last draft.
    cells: [u8; MAX_DRAFT],
    proposed: usize,
}

impl Default for NgramDrafter {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramDrafter {
    /// The drafter of a sequence that has no tokens yet.
    pub fn new() -> Self {
        Self {
            len: 0,
            generation: u64::MAX,
            state: ROOT,
            tail: NONE,
            old_state: ROOT,
            old_matched: 0,
            history: [[0; 4]; HISTORY],
            valid: 0,
            blocked: NONE,
            blocked_generation: u64::MAX,
            cells: [0; MAX_DRAFT],
            proposed: 0,
        }
    }

    /// The number of tokens seen.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence has no tokens yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Catches up with a corpus that started a new automaton since this
    /// drafter was last used: what was its own state is a match in the old
    /// automaton now, and its text starts anew in the young one.
    fn sync(&mut self, corpus: &NgramCorpus) {
        if self.generation == corpus.generation {
            return;
        }
        let previous = self.generation.wrapping_add(1) == corpus.generation;
        match (&corpus.old, previous && self.generation != u64::MAX) {
            (Some(old), true) => {
                self.old_state = self.state;
                self.old_matched = old.states[self.state as usize].len;
            }
            _ => (self.old_state, self.old_matched) = (ROOT, 0),
        }
        (self.state, self.tail) = (ROOT, NONE);
        self.generation = corpus.generation;
        self.valid = 0;
    }

    /// Adds a token of the sequence (prompt or accepted output) to the
    /// corpus.
    pub fn push(&mut self, corpus: &mut NgramCorpus, token: u32) {
        if corpus.young.is_full() {
            corpus.rotate();
        }
        self.sync(corpus);
        let hops = corpus.config.max_hops;
        (self.state, self.tail) = corpus.young.push(self.state, self.tail, token, hops);
        if let Some(old) = &corpus.old {
            (self.old_state, self.old_matched) =
                old.follow(self.old_state, self.old_matched, token);
        }
        self.len += 1;
        self.history[self.len % HISTORY] =
            [self.state, self.tail, self.old_state, self.old_matched];
        self.valid = (self.valid + 1).min(HISTORY);
    }

    /// Adds several tokens, as [`NgramDrafter::push`] does one.
    pub fn extend(&mut self, corpus: &mut NgramCorpus, tokens: &[u32]) {
        for token in tokens {
            self.push(corpus, *token);
        }
    }

    /// Cuts the sequence back to `len` tokens, after a rollback. The removed
    /// tokens stay in the corpus as text that was written once, but this
    /// drafter does not propose them again as the continuation of the place
    /// it was cut at: there it proposes nothing until the sequence has gone
    /// on, and from then on what the sequence wrote instead.
    ///
    /// The last 31 tokens can be cut exactly. A deeper cut keeps the count
    /// and loses the context: the drafter matches again from the tokens that
    /// follow.
    pub fn truncate(&mut self, corpus: &NgramCorpus, len: usize) {
        if len >= self.len {
            return;
        }
        self.sync(corpus);
        let back = self.len - len;
        if back < self.valid {
            let removed = self.history[(len + 1) % HISTORY];
            (self.blocked, self.blocked_generation) = (removed[1], self.generation);
            let [state, tail, old_state, old_matched] = self.history[len % HISTORY];
            (self.state, self.tail, self.old_state, self.old_matched) =
                (state, tail, old_state, old_matched);
            self.valid -= back;
        } else {
            (self.state, self.tail, self.old_state, self.old_matched) = (ROOT, NONE, ROOT, 0);
            self.valid = 0;
        }
        self.len = len;
        self.proposed = 0;
    }

    /// The match to draft from: the automaton, the state and the length.
    fn find<'a>(&mut self, corpus: &'a NgramCorpus) -> Option<(&'a Automaton, u32, u32)> {
        self.sync(corpus);
        let config = &corpus.config;
        let blocked = |generation: u64| {
            if self.blocked_generation == generation { self.blocked } else { NONE }
        };
        let young = &corpus.young;
        let own = young.states[self.state as usize].len;
        let in_young = young
            .longest(self.state, own, config.min_match, blocked(self.generation), config.max_hops)
            .map(|(state, matched)| (young, state, matched));
        let in_old = corpus.old.as_ref().and_then(|old| {
            let blocked = blocked(self.generation.wrapping_sub(1));
            old.longest(
                self.old_state,
                self.old_matched,
                config.min_match,
                blocked,
                config.max_hops,
            )
            .map(|(state, matched)| (old, state, matched))
        });
        match (in_young, in_old) {
            (Some(young), Some(old)) => Some(if old.2 > young.2 { old } else { young }),
            (young, old) => young.or(old),
        }
    }

    /// Proposes up to `max` tokens that followed the most recent earlier
    /// occurrence of the longest suffix of the sequence that has one, of at
    /// least [`NgramConfig::min_match`] tokens. `tokens` receives the draft
    /// and `confidence`, for each token, the probability that it is accepted
    /// given that the ones before it were. Returns the length of the match;
    /// 0 with an empty draft when there is none.
    pub fn draft(
        &mut self,
        corpus: &NgramCorpus,
        max: usize,
        tokens: &mut Vec<u32>,
        confidence: &mut Vec<f64>,
    ) -> usize {
        tokens.clear();
        confidence.clear();
        self.proposed = 0;
        let Some((automaton, mut state, matched)) = self.find(corpus) else { return 0 };
        let mut len = matched;
        let mut at = automaton.states[state as usize].cont;
        while tokens.len() < max && at != NONE {
            let token = automaton.text[at as usize];
            let Some(next) = automaton.step(state, token) else { break };
            let occurrences = automaton.states[state as usize].seen;
            let agreeing = automaton.states[next as usize].seen;
            let cell = cell(len, occurrences, agreeing);
            if tokens.len() < MAX_DRAFT {
                self.cells[tokens.len()] = cell;
            }
            let (row, column) = (cell as usize / AGREEMENTS, cell as usize % AGREEMENTS);
            confidence.push(f64::from(corpus.rate[row][column]));
            tokens.push(token);
            (state, len, at) = (next, len + 1, automaton.next[at as usize]);
        }
        self.proposed = tokens.len().min(MAX_DRAFT);
        if tokens.is_empty() { 0 } else { matched as usize }
    }

    /// The match [`NgramDrafter::draft`] would draft from, with how often
    /// its earlier occurrences went on the way the draft starts.
    pub fn matched(&mut self, corpus: &NgramCorpus) -> Option<NgramMatch> {
        let (automaton, state, len) = self.find(corpus)?;
        let s = &automaton.states[state as usize];
        let first = automaton.text[s.cont as usize];
        let agreeing =
            automaton.step(state, first).map_or(0, |next| automaton.states[next as usize].seen);
        Some(NgramMatch { len, occurrences: s.seen, agreeing: agreeing.min(s.seen) })
    }

    /// The table cell of each token of the last draft, as row times four plus
    /// column of [`NgramCorpus::table`]. This is what a calibration run
    /// counts acceptance by.
    pub fn cells(&self) -> &[u8] {
        &self.cells[..self.proposed]
    }

    /// Records the outcome of the last draft: its first `verified` tokens
    /// were verified and the first `accepted` of them were right. The corpus
    /// learns from it what to expect of such matches.
    pub fn observe(&mut self, corpus: &mut NgramCorpus, verified: usize, accepted: usize) {
        const STEP: f32 = 0.02;
        for (i, cell) in self.cells.iter().enumerate().take(verified.min(self.proposed)) {
            let rate = &mut corpus.rate[*cell as usize / AGREEMENTS][*cell as usize % AGREEMENTS];
            if i < accepted {
                *rate += STEP * (1.0 - *rate);
            } else {
                // The first rejected token; the ones after it were not tested.
                *rate -= STEP * *rate;
                break;
            }
        }
        self.proposed = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> NgramCorpus {
        NgramCorpus::new(NgramConfig { max_bytes: 1 << 20, min_match: 3, max_hops: 24 })
    }

    fn draft(drafter: &mut NgramDrafter, corpus: &NgramCorpus, max: usize) -> Vec<u32> {
        let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
        drafter.draft(corpus, max, &mut tokens, &mut confidence);
        assert_eq!(tokens.len(), confidence.len());
        assert!(confidence.iter().all(|p| (0.0..=1.0).contains(p)));
        tokens
    }

    #[test]
    fn continues_an_earlier_occurrence() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        // A "file" in the prompt, then the model starts to write it again.
        drafter.extend(&mut corpus, &[10, 11, 12, 13, 14, 15, 16, 99, 98]);
        assert!(draft(&mut drafter, &corpus, 8).is_empty(), "nothing repeats yet");
        drafter.extend(&mut corpus, &[10, 11]);
        assert!(draft(&mut drafter, &corpus, 8).is_empty(), "two tokens are not a match of three");
        drafter.push(&mut corpus, 12);
        assert_eq!(draft(&mut drafter, &corpus, 3), [13, 14, 15]);
        assert_eq!(draft(&mut drafter, &corpus, 100), [13, 14, 15, 16, 99, 98, 10, 11, 12]);
        assert_eq!(drafter.matched(&corpus).map(|m| m.len), Some(3));

        // The drafts are accepted: the next draft continues further, and
        // the match is as long as what was copied so far.
        drafter.extend(&mut corpus, &[13, 14]);
        assert_eq!(draft(&mut drafter, &corpus, 2), [15, 16]);
        assert_eq!(drafter.matched(&corpus).map(|m| m.len), Some(5));
        assert_eq!(drafter.len(), 14);
    }

    #[test]
    fn the_longest_match_decides_where_the_last_tokens_alone_cannot() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        // Two functions that end in the same three tokens and go on
        // differently. The text then repeats the first one.
        let first = [1, 2, 3, 7, 8, 9, 100, 101];
        let second = [4, 5, 6, 7, 8, 9, 200, 201];
        drafter.extend(&mut corpus, &first);
        drafter.extend(&mut corpus, &second);
        drafter.extend(&mut corpus, &[50, 1, 2, 3, 7, 8, 9]);
        // The last three tokens were last followed by 200. Six tokens match
        // the first function, which went on with 100.
        assert_eq!(draft(&mut drafter, &corpus, 2), [100, 101]);
        assert_eq!(drafter.matched(&corpus).map(|m| m.len), Some(6));
    }

    #[test]
    fn prefers_the_most_recent_occurrence_of_the_longest_match() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        drafter.extend(&mut corpus, &[1, 2, 3, 30, 0, 1, 2, 3, 40, 0, 1, 2, 3]);
        assert_eq!(draft(&mut drafter, &corpus, 1), [40]);
        // Longer matches exist here: "0 1 2 3" occurred once before.
        assert_eq!(
            drafter.matched(&corpus),
            Some(NgramMatch { len: 4, occurrences: 1, agreeing: 1 })
        );
    }

    #[test]
    fn a_repeating_pattern_drafts_up_to_the_end_of_what_exists() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        drafter.extend(&mut corpus, &[7, 8, 9, 7, 8, 9, 7, 8]);
        // "7 8 9 7 8" matched at its first occurrence; what followed is
        // "9 7 8", up to the sequence's own end.
        assert_eq!(draft(&mut drafter, &corpus, 10), [9, 7, 8]);
    }

    #[test]
    fn short_sequences_and_new_text_have_no_drafts() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        assert!(drafter.is_empty());
        assert!(draft(&mut drafter, &corpus, 4).is_empty());
        drafter.extend(&mut corpus, &[1, 1, 1]);
        assert!(draft(&mut drafter, &corpus, 4).is_empty());
        // Four equal tokens: the last three occurred before, followed by 1.
        drafter.push(&mut corpus, 1);
        assert_eq!(draft(&mut drafter, &corpus, 4), [1]);
        drafter.extend(&mut corpus, &[2, 3, 4, 5]);
        assert!(draft(&mut drafter, &corpus, 4).is_empty());
    }

    #[test]
    fn one_request_drafts_from_the_text_of_another() {
        let mut corpus = corpus();
        // An agent's first request: a document and an answer.
        let document: Vec<u32> = (100..140).collect();
        let answer: Vec<u32> = (500..520).collect();
        let mut first = NgramDrafter::new();
        first.extend(&mut corpus, &document);
        first.extend(&mut corpus, &answer);

        // A second request quotes a part of the document in other words.
        let mut second = NgramDrafter::new();
        second.extend(&mut corpus, &[1, 2, 3, 110, 111, 112, 113]);
        assert_eq!(draft(&mut second, &corpus, 4), [114, 115, 116, 117]);
        assert_eq!(second.matched(&corpus).map(|m| m.len), Some(4));

        // A third sends the same document again: the whole of it matches,
        // and the draft is the first request's answer.
        let mut third = NgramDrafter::new();
        third.extend(&mut corpus, &document);
        assert_eq!(draft(&mut third, &corpus, 5), answer[..5]);
        assert_eq!(third.matched(&corpus).map(|m| m.len), Some(40));

        // Requests that run at the same time keep their texts apart: each
        // continues its own, token by token, in any interleaving.
        let (mut a, mut b) = (NgramDrafter::new(), NgramDrafter::new());
        for i in 0..12 {
            a.push(&mut corpus, 100 + i);
            b.push(&mut corpus, 500 + i);
        }
        assert_eq!(draft(&mut a, &corpus, 3), [112, 113, 114]);
        assert_eq!(draft(&mut b, &corpus, 3), [512, 513, 514]);
    }

    #[test]
    fn confidence_grows_with_the_match_and_with_agreement() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        // "1 2 3" went on with 4 five times and with 9 once (the last time).
        for _ in 0..5 {
            drafter.extend(&mut corpus, &[1, 2, 3, 4, 70, 71]);
        }
        drafter.extend(&mut corpus, &[60, 1, 2, 3, 9, 80, 81, 61, 1, 2, 3]);
        let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
        assert_eq!(drafter.draft(&corpus, 1, &mut tokens, &mut confidence), 3);
        assert_eq!(tokens, [9], "the most recent continuation is proposed");
        let found = drafter.matched(&corpus).unwrap();
        assert_eq!((found.len, found.occurrences, found.agreeing), (3, 6, 1));
        let seldom = confidence[0];
        assert_eq!(seldom, corpus.expected_acceptance(3, 6, 1));

        // The same three tokens when they always went on the same way, and
        // when they were seen once.
        let always = corpus.expected_acceptance(3, 6, 6);
        let once = corpus.expected_acceptance(3, 1, 1);
        assert!(always > once && once > seldom, "{always} {once} {seldom}");
        // A longer match is worth more when its context went on one way
        // only. (When it went on in several ways the length says little: the
        // measured table is flat there.)
        for (occurrences, agreeing) in [(1, 1), (6, 6)] {
            let mut last = 0.0;
            for len in [1, 2, 3, 4, 6, 8, 12, 16, 32, 100] {
                let p = corpus.expected_acceptance(len, occurrences, agreeing);
                assert!(p >= last && p > 0.2, "{len} tokens, {agreeing} of {occurrences}: {p}");
                last = p;
            }
            assert!(last > 0.95);
        }
        assert!(corpus.expected_acceptance(40, 6, 1) < 0.6);
        // Inside one draft the context grows with every token.
        drafter.extend(&mut corpus, &[9, 80]);
        drafter.draft(&corpus, 4, &mut tokens, &mut confidence);
        assert_eq!(tokens, [81, 61, 1, 2]);
        assert!(confidence[1] >= confidence[0], "{confidence:?}");
    }

    #[test]
    fn the_table_follows_what_verification_reports() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        drafter.extend(&mut corpus, &[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]);
        let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
        drafter.draft(&corpus, 3, &mut tokens, &mut confidence);
        let before = confidence.clone();
        // The first draft token is accepted, the second is not, and the
        // third was never tested, 40 times.
        for _ in 0..40 {
            drafter.draft(&corpus, 3, &mut tokens, &mut confidence);
            drafter.observe(&mut corpus, 3, 1);
        }
        drafter.draft(&corpus, 3, &mut tokens, &mut confidence);
        assert!(confidence[0] > before[0] + 0.2, "{before:?} -> {confidence:?}");
        assert!(confidence[1] < before[1] - 0.2, "{before:?} -> {confidence:?}");
        assert_eq!(confidence[2], before[2], "a token that is never reached teaches nothing");
        // Only what was verified counts.
        drafter.observe(&mut corpus, 0, 0);
        let mut again = Vec::new();
        drafter.draft(&corpus, 3, &mut tokens, &mut again);
        assert_eq!(again, confidence);
    }

    #[test]
    fn a_rollback_does_not_propose_what_was_removed() {
        let mut corpus = corpus();
        let mut drafter = NgramDrafter::new();
        drafter.extend(&mut corpus, &[1, 2, 3, 4, 5, 6, 0, 1, 2, 3]);
        assert_eq!(draft(&mut drafter, &corpus, 2), [4, 5]);
        // The model writes "4 5 6", which is rolled back (a banned string).
        drafter.extend(&mut corpus, &[4, 5, 6]);
        drafter.truncate(&corpus, 10);
        assert_eq!(drafter.len(), 10);
        // What followed this place most recently is the text that was just
        // removed. It is not proposed again.
        assert_eq!(drafter.matched(&corpus), None);
        assert!(draft(&mut drafter, &corpus, 2).is_empty());
        // The sequence goes on differently, and what it writes now is what
        // follows this place from here on.
        drafter.extend(&mut corpus, &[9, 9, 0, 1, 2, 3]);
        assert_eq!(draft(&mut drafter, &corpus, 2), [9, 9]);

        // A cut deeper than the drafter remembers keeps the count and starts
        // matching anew.
        let mut long = NgramDrafter::new();
        let text: Vec<u32> = (1000..1100).collect();
        long.extend(&mut corpus, &text);
        long.truncate(&corpus, 20);
        assert_eq!(long.len(), 20);
        assert!(draft(&mut long, &corpus, 2).is_empty());
        long.extend(&mut corpus, &text[20..24]);
        assert_eq!(draft(&mut long, &corpus, 2), text[24..26]);
        // Cutting to a length it does not have changes nothing.
        long.truncate(&corpus, 500);
        assert_eq!(long.len(), 24);
    }

    #[test]
    fn equal_matches_in_both_automata_take_the_newer_text() {
        let config = NgramConfig { max_bytes: 256 << 10, min_match: 3, max_hops: 24 };
        let mut corpus = NgramCorpus::new(config);
        let passage = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut first = NgramDrafter::new();
        first.extend(&mut corpus, &passage);
        first.push(&mut corpus, 100);
        // Enough other text for one turnover: the passage is in the old
        // automaton now.
        let mut other = NgramDrafter::new();
        let mut token = 1000;
        while corpus.generation == 0 {
            other.push(&mut corpus, token);
            token += 1;
        }
        // It is written again and goes on differently this time.
        let mut second = NgramDrafter::new();
        second.extend(&mut corpus, &passage);
        second.push(&mut corpus, 200);
        let mut reader = NgramDrafter::new();
        reader.extend(&mut corpus, &passage);
        assert_eq!(reader.matched(&corpus).map(|m| m.len), Some(8));
        assert_eq!(draft(&mut reader, &corpus, 1), [200]);
        assert_eq!(reader.cells().len(), 1);
    }

    #[test]
    fn memory_is_bounded_and_old_text_is_forgotten() {
        let config = NgramConfig { max_bytes: 256 << 10, min_match: 3, max_hops: 24 };
        let mut corpus = NgramCorpus::new(config);
        assert!(corpus.bytes() <= config.max_bytes, "{}", corpus.bytes());
        let per_generation = corpus.tokens_per_generation();
        assert!(per_generation >= 512, "{per_generation}");

        // A passage, then far more text than the corpus holds, from several
        // requests; one of them lives through all of it.
        let passage: Vec<u32> = (1_000_000..1_000_050).collect();
        let mut writer = NgramDrafter::new();
        writer.extend(&mut corpus, &passage);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 50_000) as u32
        };
        let mut pushed = passage.len();
        while pushed < 10 * per_generation {
            let mut other = NgramDrafter::new();
            for _ in 0..300 {
                other.push(&mut corpus, noise());
                writer.push(&mut corpus, noise());
                pushed += 2;
            }
            assert!(corpus.tokens() <= 2 * per_generation);
        }
        assert_eq!(writer.len(), 50 + (pushed - 50) / 2);

        // The passage is long gone.
        let mut reader = NgramDrafter::new();
        reader.extend(&mut corpus, &passage[..10]);
        assert!(draft(&mut reader, &corpus, 4).is_empty());
        // Recent text is found, also by the request that lived through the
        // turnovers, and also right after one: what it wrote before the last
        // turnover is in the old automaton.
        let recent: Vec<u32> = (2_000_000..2_000_040).collect();
        writer.extend(&mut corpus, &recent);
        writer.extend(&mut corpus, &recent[..5]);
        assert_eq!(draft(&mut writer, &corpus, 3), recent[5..8]);
        let before = corpus.generation;
        while corpus.generation == before {
            reader.push(&mut corpus, noise());
        }
        assert_eq!(draft(&mut writer, &corpus, 3), recent[5..8], "found in the old automaton");
        writer.extend(&mut corpus, &recent[5..7]);
        assert_eq!(draft(&mut writer, &corpus, 3), recent[7..10], "and followed from there");
    }
}
