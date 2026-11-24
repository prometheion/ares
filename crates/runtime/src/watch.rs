//! What the conductor tells about a reply besides its text: which stop
//! string ended it, how many of its tokens the model spent thinking, and how
//! long the request waited in line.
//!
//! The conductor feeds a [`Watch`] what it already has in hand: every token
//! it accepts ([`Watch::token`]) and every character of their text
//! ([`Watch::push`], which is the stop strings' matcher). The request's
//! [`Usage`](ares_api::Usage) and its finish reason are read off it at the
//! end. Everything a watch decides is in this file, so that the conductor
//! carries it without knowing what it looks for.
//!
//! # The tokens of the thinking
//!
//! [`Thinking`] says where the thinking of a reply is: the reply begins
//! inside it (the prompt ends with the opening tag, as the template of
//! Qwen3.8 writes it), or may open it with its first text (a template that
//! leaves the tag to the model). It ends at the closing tag, whose tokens are
//! the thinking's, or at the start of a call of a tool that the model opens
//! inside its thinking, whose tokens are not. [`Tracker`] follows that over
//! the bytes of the tokens, for this count and for the guard that bounds the
//! thinking (`guard.rs`). The count is exact: tokens are numbered as they are
//! taken, and every byte of their text carries the number of the token it
//! came with.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ares_api::FinishReason;

use crate::text::StopMatcher;

/// Where the thinking of a reply is, as the request's prompt leaves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thinking {
    /// The reply begins inside the thinking; else it may open it with its
    /// first text, `open`.
    pub inside: bool,
    /// The opening tag (`<think>`).
    pub open: String,
    /// The closing tag (`</think>`).
    pub close: String,
    /// Text that ends the thinking without being part of it: the start of a
    /// call of a tool the model may open while it thinks (`<tool_call>`).
    pub interrupt: Option<String>,
}

/// Where the reply is with respect to its thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Place {
    /// Nothing but white space so far; the thinking may still open.
    Before,
    /// Thinking since the token after `from`; the opening tag ended in token
    /// `opened` (0: the reply began inside).
    Inside { from: u32, opened: u32 },
    /// The thinking took this many tokens (0: it never opened).
    Done(u32),
}

/// Follows the tags of a reply's thinking ([`Thinking`]) over the bytes of
/// its tokens, a token at a time.
#[derive(Debug, Clone)]
pub(crate) struct Tracker {
    place: Place,
    /// The last bytes of the text, each with the number of its token, as many
    /// as the longest tag has.
    recent: VecDeque<(u8, u32)>,
    /// Tokens taken so far; the token being read is number `tokens`.
    tokens: u32,
}

impl Tracker {
    pub fn new(thinking: &Thinking) -> Self {
        let place =
            if thinking.inside { Place::Inside { from: 0, opened: 0 } } else { Place::Before };
        Self { place, recent: VecDeque::new(), tokens: 0 }
    }

    pub fn place(&self) -> Place {
        self.place
    }

    /// The tokens of the thinking so far, its tags included.
    pub fn count(&self) -> u32 {
        match self.place {
            Place::Before => 0,
            Place::Inside { from, .. } => self.tokens - from,
            Place::Done(tokens) => tokens,
        }
    }

    /// The tokens since the opening tag while the thinking is open: what a
    /// budget of the thinking counts.
    pub fn used(&self) -> Option<u32> {
        match self.place {
            Place::Inside { opened, .. } => Some(self.tokens - opened),
            _ => None,
        }
    }

    /// Takes the next token, whose text is `piece` (empty for a token that
    /// writes nothing).
    pub fn token(&mut self, thinking: &Thinking, piece: &[u8]) {
        self.tokens += 1;
        for &byte in piece {
            if matches!(self.place, Place::Done(_)) {
                return;
            }
            self.byte(thinking, byte);
        }
    }

    /// Follows the thinking through one more byte.
    fn byte(&mut self, thinking: &Thinking, byte: u8) {
        self.recent.push_back((byte, self.tokens));
        let longest = [Some(&thinking.open), Some(&thinking.close), thinking.interrupt.as_ref()]
            .into_iter()
            .flatten()
            .map(String::len)
            .max()
            .unwrap_or(0);
        while self.recent.len() > longest.max(1) {
            self.recent.pop_front();
        }
        // The token the last bytes began with, if they are `tag`.
        let ends_with = |recent: &VecDeque<(u8, u32)>, tag: &str| {
            let tag = tag.as_bytes();
            if tag.is_empty() || recent.len() < tag.len() {
                return None;
            }
            let start = recent.len() - tag.len();
            let same = recent.iter().skip(start).map(|(b, _)| *b).eq(tag.iter().copied());
            same.then(|| recent[start].1)
        };
        match self.place {
            Place::Before => {
                if let Some(first) = ends_with(&self.recent, &thinking.open) {
                    self.place = Place::Inside { from: first - 1, opened: self.tokens };
                    self.recent.clear();
                    return;
                }
                // White space, or the beginning of the opening tag, may still
                // come before the tag; anything else is an answer without
                // thinking. A character that is not whole yet is waited for;
                // the end of one that left the bytes kept is white space.
                let bytes: Vec<u8> =
                    self.recent.iter().map(|(b, _)| *b).skip_while(|b| b & 0xc0 == 0x80).collect();
                let text = match std::str::from_utf8(&bytes) {
                    Ok(text) => text,
                    Err(error) if error.error_len().is_none() => {
                        // SAFETY: `valid_up_to` bytes are valid UTF-8 by its contract.
                        unsafe { std::str::from_utf8_unchecked(&bytes[..error.valid_up_to()]) }
                    }
                    Err(_) => {
                        self.place = Place::Done(0);
                        return;
                    }
                };
                let trimmed = text.trim_start();
                if !trimmed.is_empty() && !thinking.open.starts_with(trimmed) {
                    self.place = Place::Done(0);
                }
            }
            Place::Inside { from, .. } => {
                if let Some(first) =
                    thinking.interrupt.as_deref().and_then(|t| ends_with(&self.recent, t))
                {
                    self.place = Place::Done(first - 1 - from);
                } else if ends_with(&self.recent, &thinking.close).is_some() {
                    self.place = Place::Done(self.tokens - from);
                }
            }
            Place::Done(_) => {}
        }
    }
}

/// See the module documentation.
#[derive(Debug)]
pub struct Watch {
    stop: StopMatcher,
    thinking: Option<(Thinking, Tracker)>,
    submitted: Instant,
    started: Option<Instant>,
}

impl Watch {
    /// A watch for a reply that ends at the first of `stops` (empty ones
    /// are never matched) and whose thinking is where `thinking` says, if the
    /// model thinks. The request is in line from now on.
    pub fn new(stops: &[String], thinking: Option<Thinking>) -> Self {
        let thinking = thinking.map(|thinking| {
            let tracker = Tracker::new(&thinking);
            (thinking, tracker)
        });
        Self { stop: StopMatcher::new(stops), thinking, submitted: Instant::now(), started: None }
    }

    /// The request leaves the line and starts; only the first time counts.
    pub fn started(&mut self) {
        self.started.get_or_insert_with(Instant::now);
    }

    /// The next token was taken; it writes `piece`, whose characters follow
    /// ([`Watch::push`]).
    pub fn token(&mut self, piece: &[u8]) {
        if let Some((thinking, tracker)) = &mut self.thinking {
            tracker.token(thinking, piece);
        }
    }

    /// Feeds one character of the text: see [`StopMatcher::push`], whose
    /// answer this is.
    pub fn push(&mut self, ch: char, out: &mut String) -> bool {
        self.stop.push(ch, out)
    }

    /// Releases the text held for the stop strings: see
    /// [`StopMatcher::flush`].
    pub fn flush(&mut self, out: &mut String) {
        self.stop.flush(out);
    }

    /// Why a reply ended that a stop string ended: which one it was.
    pub fn finish(&self) -> FinishReason {
        match self.stop.matched() {
            Some(index) => FinishReason::StopString(u32::try_from(index).unwrap_or(u32::MAX)),
            None => FinishReason::Stop,
        }
    }

    /// The tokens of the reply's thinking so far, for a model that thinks.
    pub fn reasoning_tokens(&self) -> Option<u32> {
        self.thinking.as_ref().map(|(_, tracker)| tracker.count())
    }

    /// When the request was put in line.
    pub fn submitted(&self) -> Instant {
        self.submitted
    }

    /// How long the request waited before it first started.
    pub fn queued(&self) -> Option<Duration> {
        self.started.map(|started| started.duration_since(self.submitted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thinking(inside: bool, interrupt: Option<&str>) -> Thinking {
        Thinking {
            inside,
            open: "<think>".into(),
            close: "</think>".into(),
            interrupt: interrupt.map(str::to_owned),
        }
    }

    /// Feeds `tokens` (the text of each) and returns the count of the
    /// thinking and the text that went out.
    fn feed(watch: &mut Watch, tokens: &[&str]) -> (Option<u32>, String) {
        let mut out = String::new();
        for token in tokens {
            watch.token(token.as_bytes());
            for ch in token.chars() {
                if watch.push(ch, &mut out) {
                    return (watch.reasoning_tokens(), out);
                }
            }
        }
        watch.flush(&mut out);
        (watch.reasoning_tokens(), out)
    }

    #[test]
    fn a_reply_that_begins_inside_counts_to_the_closing_tag() {
        let mut watch = Watch::new(&[], Some(thinking(true, None)));
        let reply = ["Let", " me", " think", ".", "\n", "</think>", "\n\n", "Four", "."];
        assert_eq!(feed(&mut watch, &reply).0, Some(6));
        // A tag that a token cuts in two is found as well.
        let mut watch = Watch::new(&[], Some(thinking(true, None)));
        assert_eq!(feed(&mut watch, &["Hm", "</th", "ink>\n\n", "Yes"]).0, Some(3));
        // A reply the limit cut inside the thinking is all thinking.
        let mut watch = Watch::new(&[], Some(thinking(true, None)));
        assert_eq!(feed(&mut watch, &["a", "b", "c"]).0, Some(3));
    }

    #[test]
    fn a_reply_that_opens_its_thinking_counts_from_the_tag() {
        let mut watch = Watch::new(&[], Some(thinking(false, None)));
        assert_eq!(feed(&mut watch, &["\n", "<think>", "x", "</think>", "a"]).0, Some(3));
        // An answer that begins without the tag has no thinking.
        let mut watch = Watch::new(&[], Some(thinking(false, None)));
        assert_eq!(feed(&mut watch, &["Four", " <think>", "</think>"]).0, Some(0));
        // The model does not think: nothing is counted.
        assert_eq!(feed(&mut Watch::new(&[], None), &["a"]).0, None);
    }

    #[test]
    fn a_call_inside_the_thinking_ends_it_without_its_marker() {
        let mut watch = Watch::new(&[], Some(thinking(true, Some("<tool_call>"))));
        let reply = ["I", " call", "\n", "<tool_call>", "\n<function=f>"];
        assert_eq!(feed(&mut watch, &reply).0, Some(3));
        // A marker that begins in the middle of a token leaves that token to
        // the thinking: the marker began there.
        let mut watch = Watch::new(&[], Some(thinking(true, Some("<tool_call>"))));
        assert_eq!(feed(&mut watch, &["I", " call\n<tool", "_call>"]).0, Some(1));
    }

    #[test]
    fn the_tags_are_found_in_bytes_whatever_the_tokens_cut() {
        // A tag and its end in one token; a character of white space split
        // over two tokens before the tag; text before the tag in its token.
        let mut tracker = Tracker::new(&thinking(false, None));
        let tags = thinking(false, None);
        tracker.token(&tags, b"<think>x</think>");
        assert_eq!((tracker.place(), tracker.count()), (Place::Done(1), 1));
        let mut tracker = Tracker::new(&tags);
        let space = "\u{3000}".as_bytes();
        tracker.token(&tags, &space[..1]);
        tracker.token(&tags, &space[1..]);
        tracker.token(&tags, b"<think>");
        assert_eq!(tracker.place(), Place::Inside { from: 2, opened: 3 });
        tracker.token(&tags, b"a");
        assert_eq!((tracker.count(), tracker.used()), (2, Some(1)));
        let mut tracker = Tracker::new(&tags);
        tracker.token(&tags, b"a<think>");
        assert_eq!(tracker.place(), Place::Done(0), "an answer that names the tag");
    }

    #[test]
    fn the_stop_string_that_ended_a_reply_is_named() {
        let stops = ["END".to_owned(), String::new(), "\n\n".to_owned()];
        let mut watch = Watch::new(&stops, None);
        let (_, out) = feed(&mut watch, &["a", "b\n", "\nc"]);
        assert_eq!(out, "ab");
        assert_eq!(watch.finish(), FinishReason::StopString(2));
        let mut watch = Watch::new(&stops, None);
        feed(&mut watch, &["xEN", "D"]);
        assert_eq!(watch.finish(), FinishReason::StopString(0));
        let mut watch = Watch::new(&stops, None);
        feed(&mut watch, &["no stop"]);
        assert_eq!(watch.finish(), FinishReason::Stop);
    }

    #[test]
    fn the_wait_in_line_ends_with_the_first_start() {
        let mut watch = Watch::new(&[], None);
        assert_eq!(watch.queued(), None);
        std::thread::sleep(Duration::from_millis(2));
        watch.started();
        let queued = watch.queued().unwrap();
        assert!(queued >= Duration::from_millis(2));
        watch.started();
        assert_eq!(watch.queued(), Some(queued), "a second start does not count");
    }
}
