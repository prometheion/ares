//! The corrections that depend on the tokens before a row: the repetition,
//! presence and frequency penalties, DRY, and the logit bias.
//!
//! They are definitions of llama.cpp's samplers of the same names. The host
//! works them out for every wanted row from the row's own path, the tokens
//! of the prompt and the reply up to and including the row's input token, which
//! for a draft row includes the drafts before it. That is what makes a window
//! that verifies drafts pick what the same sequence picks one token at a time:
//! each row is penalised for the text it would continue.
//!
//! The result is a list of [`Entry`], one per token that changes, in order of
//! token; the device applies them to the logits before it looks at anything
//! else.

use std::collections::HashMap;
use std::sync::Arc;

use crate::row::{ENTRIES_PER_ROW, Entry};

/// The most tokens of history the penalties and DRY look at, so that the
/// corrections of a row stay below [`ENTRIES_PER_ROW`] with a logit bias.
pub const WINDOW_MAX: u32 = 1024;
/// The most tokens a logit bias names.
pub const BIAS_MAX: usize = 1024;

/// llama.cpp's repetition, presence and frequency penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PenaltyParams {
    /// How many of the last tokens of the path count; 0 is off.
    pub last_n: u32,
    /// Divides a positive logit and multiplies a negative one of a token in
    /// the window; 1 is off.
    pub repeat: f32,
    /// Subtracted once per occurrence in the window; 0 is off.
    pub frequency: f32,
    /// Subtracted once for a token that is in the window; 0 is off.
    pub presence: f32,
}

impl PenaltyParams {
    /// llama.cpp's defaults, which leave the penalties off.
    pub const OFF: PenaltyParams =
        PenaltyParams { last_n: 64, repeat: 1.0, frequency: 0.0, presence: 0.0 };

    pub fn is_on(&self) -> bool {
        self.last_n > 0 && (self.repeat != 1.0 || self.frequency != 0.0 || self.presence != 0.0)
    }
}

/// DRY, "don't repeat yourself": a token that would continue a repetition of
/// the text pays `multiplier * base^(length - allowed_length)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DryParams {
    /// 0 is off.
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: u32,
    pub last_n: u32,
}

impl DryParams {
    /// llama.cpp's defaults, which leave DRY off.
    pub const OFF: DryParams =
        DryParams { multiplier: 0.0, base: 1.75, allowed_length: 2, last_n: 64 };

    pub fn is_on(&self) -> bool {
        self.multiplier != 0.0 && self.base >= 1.0 && self.last_n > 0
    }
}

/// The sequences that end a repetition for DRY: a token that is one (or the
/// head of one whose tail follows) stops DRY from looking further back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Breakers {
    /// A head token and the tokens that complete the sequence (none for a
    /// breaker of one token).
    sequences: HashMap<u32, Vec<Vec<u32>>>,
}

/// llama.cpp cuts a breaker at this many bytes and its tail at this many
/// tokens.
const BREAKER_BYTES: usize = 40;
const BREAKER_TAIL: usize = 20;

impl Breakers {
    /// The breakers made of `strings` for a vocabulary of `n_tokens` tokens:
    /// every token that contains a string, and every token that ends with the
    /// start of one, with the tokens of the rest as its tail. `piece` is the
    /// text of a token and `tokenize` the tokens of a text.
    pub fn build(
        strings: &[String],
        n_tokens: u32,
        piece: impl Fn(u32) -> Vec<u8>,
        tokenize: impl Fn(&[u8]) -> Vec<u32>,
    ) -> Breakers {
        let mut sequences: HashMap<u32, Vec<Vec<u32>>> = HashMap::new();
        // The text of every token, once for all the strings.
        let words: Vec<Vec<u8>> = (0..n_tokens).map(&piece).collect();
        for string in strings {
            let mut bytes = string.as_bytes();
            if bytes.is_empty() {
                continue;
            }
            bytes = &bytes[..bytes.len().min(BREAKER_BYTES)];
            for (token, word) in (0..n_tokens).zip(&words) {
                if contains(word, bytes) {
                    sequences.entry(token).or_default().push(Vec::new());
                    continue;
                }
                let mut from = 0;
                while let Some(offset) = word[from..].iter().position(|b| *b == bytes[0]) {
                    let pos = from + offset;
                    from = pos + 1;
                    let mut matched = 1;
                    while matched < bytes.len()
                        && matched + pos < word.len()
                        && word[pos + matched] == bytes[matched]
                    {
                        matched += 1;
                    }
                    // The whole string is not in the word (that case is
                    // above), so a match is of a start of the string that
                    // reaches the end of the word.
                    if matched + pos == word.len() {
                        let mut tail = tokenize(&bytes[matched..]);
                        tail.truncate(BREAKER_TAIL);
                        let tails = sequences.entry(token).or_default();
                        if !tails.contains(&tail) {
                            tails.push(tail);
                        }
                    }
                }
            }
        }
        Breakers { sequences }
    }

    /// Breakers that are given as tokens, the head first.
    #[cfg(test)]
    pub(crate) fn of_tokens(sequences: &[&[u32]]) -> Breakers {
        let mut map: HashMap<u32, Vec<Vec<u32>>> = HashMap::new();
        for sequence in sequences {
            if let Some((head, tail)) = sequence.split_first() {
                map.entry(*head).or_default().push(tail.to_vec());
            }
        }
        Breakers { sequences: map }
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    fn tails(&self, head: u32) -> &[Vec<u32>] {
        self.sequences.get(&head).map_or(&[], Vec::as_slice)
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// What a request asks the host to correct.
#[derive(Clone, Debug)]
pub struct Corrections {
    pub penalties: PenaltyParams,
    pub dry: DryParams,
    pub breakers: Arc<Breakers>,
    /// A bias per token, added to the logit. Minus infinity bans the token.
    pub bias: Vec<(u32, f32)>,
}

impl Corrections {
    pub fn none() -> Corrections {
        Corrections {
            penalties: PenaltyParams::OFF,
            dry: DryParams::OFF,
            breakers: Arc::default(),
            bias: Vec::new(),
        }
    }

    /// Whether any row has anything to correct.
    pub fn is_on(&self) -> bool {
        self.penalties.is_on() || self.dry.is_on() || !self.bias.is_empty()
    }
}

/// The corrections of the row whose path is `path` (the prompt and the reply
/// so far, and any drafts before the row), one entry per token, by token.
pub fn entries_for(path: &[u32], c: &Corrections) -> Vec<Entry> {
    let mut entries: HashMap<u32, Entry> = HashMap::new();
    for &(token, bias) in &c.bias {
        entries.entry(token).or_insert_with(|| Entry::new(token)).bias += bias;
    }
    if c.penalties.is_on() {
        let window = &path[path.len().saturating_sub(c.penalties.last_n as usize)..];
        let mut sorted = window.to_vec();
        sorted.sort_unstable();
        for run in sorted.chunk_by(|a, b| a == b) {
            let token = run[0];
            let count = run.len() as f32;
            let entry = entries.entry(token).or_insert_with(|| Entry::new(token));
            entry.rep = c.penalties.repeat;
            entry.sub += count * c.penalties.frequency + 1.0 * c.penalties.presence;
        }
    }
    if c.dry.is_on() {
        for (token, penalty) in dry_penalties(path, &c.dry, &c.breakers) {
            entries.entry(token).or_insert_with(|| Entry::new(token)).sub += penalty;
        }
    }
    let mut entries: Vec<Entry> = entries.into_values().collect();
    entries.sort_unstable_by_key(|e| e.token);
    assert!(entries.len() <= ENTRIES_PER_ROW, "{} corrections in a row", entries.len());
    entries
}

/// The penalty of each token that would continue a repetition, by llama.cpp's
/// algorithm (the Z algorithm on the reversed window), by token.
pub fn dry_penalties(path: &[u32], dry: &DryParams, breakers: &Breakers) -> Vec<(u32, f32)> {
    let last = &path[path.len().saturating_sub(dry.last_n as usize)..];
    let n = last.len();
    let allowed = dry.allowed_length as usize;
    if n <= allowed {
        return Vec::new();
    }
    // The i-th token from the end.
    let rat = |i: usize| last[n - 1 - i];

    // Step 1: a restart sequence limits how far back a repetition counts.
    let mut rep_limit = n;
    for i in 0..n {
        let longest = breakers
            .tails(rat(i))
            .iter()
            .filter(|tail| {
                tail.len() <= i
                    && tail.iter().enumerate().all(|(o, token)| *token == rat(i - o - 1))
            })
            .map(Vec::len)
            .max();
        if let Some(longest) = longest {
            rep_limit = i - longest;
            break;
        }
    }
    if rep_limit < allowed {
        return Vec::new();
    }

    // Step 2: how long a suffix of the window repeats ending at each place.
    let mut count = vec![0usize; n];
    let end = n - 1;
    let (mut rt, mut lt) = (0usize, 0usize);
    for k in 1..n {
        if k > rt {
            let mut m = 0;
            while m + k < n && rat(m) == rat(m + k) {
                m += 1;
            }
            count[end - k] = m.min(rep_limit);
            if m > 0 {
                lt = k;
                rt = k + m - 1;
            }
        } else {
            let p = k - lt;
            let right = rt - k + 1;
            if count[end - p] < right {
                count[end - k] = count[end - p].min(rep_limit);
            } else {
                let mut i = rt + 1;
                while i < n && rat(i) == rat(i - k) {
                    i += 1;
                }
                count[end - k] = (i - k).min(rep_limit);
                lt = k;
                rt = i - 1;
            }
        }
    }

    // Step 3: the longest repetition each token would continue.
    let mut longest: HashMap<u32, usize> = HashMap::new();
    for (i, &repeat) in count.iter().enumerate().take(n - 1) {
        if repeat >= allowed {
            let entry = longest.entry(rat(n - 2 - i)).or_insert(0);
            *entry = (*entry).max(repeat);
        }
    }

    // Step 4: the penalties, for tokens that are not one-token breakers. The
    // exponent is held where `base^exponent` would overflow a float.
    let max_exponent: i32 =
        if dry.base > 1.000_001 { (88.722_84 / dry.base.ln()) as i32 } else { 0 };
    let mut out: Vec<(u32, f32)> = longest
        .into_iter()
        .filter(|(token, _)| !breakers.tails(*token).iter().any(Vec::is_empty))
        .map(|(token, repeat)| {
            let mut exponent = repeat as i32 - allowed as i32;
            if max_exponent > 0 && exponent > max_exponent {
                exponent = max_exponent;
            }
            let penalty = f64::from(dry.multiplier) * f64::from(dry.base).powi(exponent);
            (token, penalty as f32)
        })
        .collect();
    out.sort_unstable_by_key(|(token, _)| *token);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same penalties from the definition, one place at a time.
    fn dry_by_definition(path: &[u32], dry: &DryParams, breakers: &Breakers) -> Vec<(u32, f32)> {
        let last = &path[path.len().saturating_sub(dry.last_n as usize)..];
        let n = last.len();
        let allowed = dry.allowed_length as usize;
        if n <= allowed {
            return Vec::new();
        }
        // The head of the nearest breaker, `i` tokens from the end, whose
        // tail follows it in the text.
        let mut limit = n;
        for i in 0..n {
            let at = n - 1 - i;
            let longest = breakers
                .tails(last[at])
                .iter()
                .filter(|t| {
                    t.len() <= i && t.iter().enumerate().all(|(o, tok)| *tok == last[at + 1 + o])
                })
                .map(Vec::len)
                .max();
            if let Some(longest) = longest {
                limit = i - longest;
                break;
            }
        }
        if limit < allowed {
            return Vec::new();
        }
        // At every place `i` but the last: how many tokens ending there are
        // the tokens the window ends with.
        let mut longest: HashMap<u32, usize> = HashMap::new();
        for i in 0..n - 1 {
            let mut l = 0;
            while l <= i && last[i - l] == last[n - 1 - l] {
                l += 1;
            }
            let l = l.min(limit);
            if l >= allowed {
                let entry = longest.entry(last[i + 1]).or_insert(0);
                *entry = (*entry).max(l);
            }
        }
        let mut out: Vec<(u32, f32)> = longest
            .into_iter()
            .filter(|(token, _)| !breakers.tails(*token).iter().any(Vec::is_empty))
            .map(|(token, l)| {
                let exponent = (l - allowed) as i32;
                (token, (f64::from(dry.multiplier) * f64::from(dry.base).powi(exponent)) as f32)
            })
            .collect();
        out.sort_unstable_by_key(|(token, _)| *token);
        out
    }

    #[test]
    fn the_penalties_of_llama_cpp_are_counted_over_the_window() {
        let mut c = Corrections::none();
        c.penalties = PenaltyParams { last_n: 4, repeat: 1.5, frequency: 0.25, presence: 0.5 };
        // The window is the last four tokens: 7 7 8 9 (the 5 has left it).
        let entries = entries_for(&[5, 7, 7, 8, 9], &c);
        let tokens: Vec<u32> = entries.iter().map(|e| e.token).collect();
        assert_eq!(tokens, [7, 8, 9]);
        assert_eq!(entries[0], Entry { token: 7, bias: 0.0, rep: 1.5, sub: 2.0 * 0.25 + 0.5 });
        assert_eq!(entries[1].sub, 0.25 + 0.5);
        // Nothing is penalised with the penalties off or an empty window.
        assert!(entries_for(&[5, 7], &Corrections::none()).is_empty());
        c.penalties.last_n = 0;
        assert!(entries_for(&[5, 7], &c).is_empty());
    }

    #[test]
    fn a_draft_row_is_penalised_for_its_own_path() {
        let mut c = Corrections::none();
        c.penalties = PenaltyParams { last_n: 3, repeat: 1.0, frequency: 1.0, presence: 0.0 };
        let reply = [1, 2, 3];
        let drafts = [3u32, 4];
        // Row 0 continues 1 2 3; row 1 continues 1 2 3 3; row 2 continues 1 2 3 3 4.
        let row = |k: usize| {
            let mut path = reply.to_vec();
            path.extend_from_slice(&drafts[..k]);
            entries_for(&path, &c)
        };
        let sub =
            |entries: &[Entry], token| entries.iter().find(|e| e.token == token).map(|e| e.sub);
        assert_eq!((sub(&row(0), 3), sub(&row(0), 1)), (Some(1.0), Some(1.0)));
        assert_eq!((sub(&row(1), 3), sub(&row(1), 1)), (Some(2.0), None));
        assert_eq!((sub(&row(2), 3), sub(&row(2), 4)), (Some(2.0), Some(1.0)));
        // The same path gives the same entries whether it was reached by
        // drafts or one token at a time.
        assert_eq!(row(2), entries_for(&[1, 2, 3, 3, 4], &c));
    }

    #[test]
    fn a_bias_is_added_and_repeats_add_up() {
        let mut c = Corrections::none();
        c.bias = vec![(10, 1.5), (11, f32::NEG_INFINITY), (10, 0.5)];
        c.penalties = PenaltyParams { last_n: 8, repeat: 2.0, frequency: 0.0, presence: 0.0 };
        let entries = entries_for(&[10], &c);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], Entry { token: 10, bias: 2.0, rep: 2.0, sub: 0.0 });
        assert_eq!(entries[1].bias, f32::NEG_INFINITY);
        // The bias comes first, then the repetition: (3 + 2) / 2.
        assert_eq!(entries[0].apply(3.0), 2.5);
    }

    #[test]
    fn dry_penalises_a_token_that_continues_a_repetition() {
        let dry = DryParams { multiplier: 0.5, base: 2.0, allowed_length: 2, last_n: 64 };
        let breakers = Breakers::default();
        // "1 2 3" occurred before and the text now ends "1 2 3": the token
        // that followed it last time (7) would extend a repetition of 3.
        let path = [1, 2, 3, 7, 9, 1, 2, 3];
        assert_eq!(dry_penalties(&path, &dry, &breakers), [(7, 0.5 * 2.0)]);
        // A repetition shorter than the allowed length: nothing.
        assert!(dry_penalties(&[1, 2, 3, 7, 9, 1], &dry, &breakers).is_empty());
        // A breaker in the repeated text stops the repetition at it.
        let breakers = Breakers::of_tokens(&[&[2]]);
        assert!(dry_penalties(&path, &dry, &breakers).is_empty());
        // The breaker token itself is never penalised.
        let penalties = dry_penalties(&[5, 2, 5, 2, 5], &dry, &Breakers::of_tokens(&[&[2]]));
        assert!(penalties.iter().all(|(token, _)| *token != 2));
    }

    #[test]
    fn dry_equals_its_definition_on_random_text() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: u32| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % u64::from(n)) as u32
        };
        for round in 0..2000 {
            let alphabet = 2 + next(5);
            let len = 3 + next(60) as usize;
            let path: Vec<u32> = (0..len).map(|_| next(alphabet)).collect();
            let dry = DryParams {
                multiplier: 0.8,
                base: 1.75,
                allowed_length: 1 + next(3),
                last_n: 4 + next(80),
            };
            let breakers = match round % 3 {
                0 => Breakers::default(),
                1 => Breakers::of_tokens(&[&[next(alphabet)]]),
                _ => Breakers::of_tokens(&[&[next(alphabet), next(alphabet)], &[next(alphabet)]]),
            };
            assert_eq!(
                dry_penalties(&path, &dry, &breakers),
                dry_by_definition(&path, &dry, &breakers),
                "{path:?} {dry:?} {breakers:?}"
            );
        }
    }

    #[test]
    fn breakers_are_found_in_the_pieces_of_the_vocabulary() {
        // Tokens: 0 "a", 1 "ab", 2 "b", 3 "x:", 4 ":", 5 "a:b".
        let pieces: [&[u8]; 6] = [b"a", b"ab", b"b", b"x:", b":", b"a:b"];
        let breakers = Breakers::build(
            &[":".to_owned(), "a:b".to_owned()],
            6,
            |t| pieces[t as usize].to_vec(),
            |text| text.iter().map(|b| u32::from(*b)).collect(),
        );
        // ":" is inside tokens 3, 4 and 5.
        for token in [3, 4, 5] {
            assert!(breakers.tails(token).iter().any(Vec::is_empty), "token {token}");
        }
        // "a:b" starts at the end of token 0, "a", and goes on with ":b".
        assert_eq!(breakers.tails(0), [vec![u32::from(b':'), u32::from(b'b')]]);
        // "ab" has an "a" but goes on with another letter; "b" has none.
        assert!(breakers.tails(1).is_empty() && breakers.tails(2).is_empty());
    }
}
