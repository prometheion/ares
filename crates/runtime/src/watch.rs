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
//! inside its thinking, whose tokens are not. The count is exact: tokens are
//! numbered as they are accepted, and every character of their text carries
//! the number of the token it came with.

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
enum Phase {
    /// Nothing but white space so far; the thinking may still open.
    Before,
    /// Thinking since the token after `from`.
    Inside { from: u32 },
    /// The thinking took this many tokens (0: it never opened).
    Done(u32),
}

/// See the module documentation.
#[derive(Debug)]
pub struct Watch {
    stop: StopMatcher,
    thinking: Option<(Thinking, Phase)>,
    /// The last characters of the text, each with the number of its token,
    /// as many as the longest tag has.
    recent: VecDeque<(char, u32)>,
    /// Tokens accepted so far; the token being read is number `tokens`.
    tokens: u32,
    submitted: Instant,
    started: Option<Instant>,
}

impl Watch {
    /// A watch for a reply that ends at the first of `stops` (empty ones
    /// are never matched) and whose thinking is where `thinking` says, if the
    /// model thinks. The request is in line from now on.
    pub fn new(stops: &[String], thinking: Option<Thinking>) -> Self {
        let thinking = thinking.map(|thinking| {
            let phase = if thinking.inside { Phase::Inside { from: 0 } } else { Phase::Before };
            (thinking, phase)
        });
        Self {
            stop: StopMatcher::new(stops),
            thinking,
            recent: VecDeque::new(),
            tokens: 0,
            submitted: Instant::now(),
            started: None,
        }
    }

    /// The request leaves the line and starts; only the first time counts.
    pub fn started(&mut self) {
        self.started.get_or_insert_with(Instant::now);
    }

    /// The next token's text follows.
    pub fn token(&mut self) {
        self.tokens += 1;
    }

    /// Feeds one character of the text: see [`StopMatcher::push`], whose
    /// answer this is.
    pub fn push(&mut self, ch: char, out: &mut String) -> bool {
        self.look(ch);
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
        let (_, phase) = self.thinking.as_ref()?;
        Some(match *phase {
            Phase::Before => 0,
            Phase::Inside { from } => self.tokens - from,
            Phase::Done(tokens) => tokens,
        })
    }

    /// When the request was put in line.
    pub fn submitted(&self) -> Instant {
        self.submitted
    }

    /// How long the request waited before it first started.
    pub fn queued(&self) -> Option<Duration> {
        self.started.map(|started| started.duration_since(self.submitted))
    }

    /// Follows the thinking through one more character.
    fn look(&mut self, ch: char) {
        let Some((thinking, phase)) = &mut self.thinking else { return };
        if matches!(phase, Phase::Done(_)) {
            return;
        }
        self.recent.push_back((ch, self.tokens));
        let longest = [Some(&thinking.open), Some(&thinking.close), thinking.interrupt.as_ref()]
            .into_iter()
            .flatten()
            .map(|tag| tag.chars().count())
            .max()
            .unwrap_or(0);
        while self.recent.len() > longest.max(1) {
            self.recent.pop_front();
        }
        // The token the last `count` characters began with, if they are `tag`.
        let ends_with = |recent: &VecDeque<(char, u32)>, tag: &str| {
            let count = tag.chars().count();
            if count == 0 || recent.len() < count {
                return None;
            }
            let start = recent.len() - count;
            let same = recent.iter().skip(start).map(|(c, _)| *c).eq(tag.chars());
            same.then(|| recent[start].1)
        };
        match *phase {
            Phase::Before => {
                if let Some(first) = ends_with(&self.recent, &thinking.open) {
                    *phase = Phase::Inside { from: first - 1 };
                    self.recent.clear();
                    return;
                }
                // White space, or the beginning of the opening tag, may still
                // come before the tag; anything else is an answer without
                // thinking.
                let text: String = self.recent.iter().map(|(c, _)| *c).collect();
                let trimmed = text.trim_start();
                if !trimmed.is_empty() && !thinking.open.starts_with(trimmed) {
                    *phase = Phase::Done(0);
                }
            }
            Phase::Inside { from } => {
                if let Some(first) =
                    thinking.interrupt.as_deref().and_then(|t| ends_with(&self.recent, t))
                {
                    *phase = Phase::Done(first - 1 - from);
                } else if ends_with(&self.recent, &thinking.close).is_some() {
                    *phase = Phase::Done(self.tokens - from);
                }
            }
            Phase::Done(_) => {}
        }
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
            watch.token();
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
