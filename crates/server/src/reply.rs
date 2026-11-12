//! The text of a reply, sorted as it comes: reasoning, content and the calls
//! of tools.

use crate::reasoning::{Split, Splitter};
use crate::tools::{CallSplitter, Part};

/// What a sorter found in a piece of a reply, in the order it stands there.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Reasoning and content.
    Text(Split),
    /// A call begins.
    Call { index: usize, id: String, name: String },
    /// The next fragment of the arguments of the call `index`.
    Arguments { index: usize, json: String },
}

/// Sorts the text of a reply: the reasoning's splitter first, and the
/// splitter of the calls on what that one calls content.
///
/// A reply that may call tools ends its reasoning at the marker of a call
/// too, so the two splitters agree with the engine's rule on where the calls
/// start: at the first marker, inside the thinking or after it.
#[derive(Debug, Clone)]
pub(crate) struct Sorter {
    thinking: Splitter,
    calls: Option<CallSplitter>,
}

impl Sorter {
    pub fn new(thinking: Splitter, calls: Option<CallSplitter>) -> Self {
        let thinking = match &calls {
            Some(calls) => thinking.ends_at(calls.trigger()),
            None => thinking,
        };
        Self { thinking, calls }
    }

    /// The same sorter for another reply to the same request, whose calls
    /// have ids that start with `id`.
    pub fn with_call_ids(&self, id: String) -> Self {
        Self {
            thinking: self.thinking.clone(),
            calls: self.calls.as_ref().map(|calls| calls.with_id(id)),
        }
    }

    /// Takes the next piece of the reply.
    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        let split = self.thinking.push(text);
        self.sorted(split, false)
    }

    /// Takes the end of the reply: what was held back is sorted as it is.
    pub fn finish(&mut self) -> Vec<Piece> {
        let split = self.thinking.finish();
        self.sorted(split, true)
    }

    fn sorted(&mut self, split: Split, last: bool) -> Vec<Piece> {
        let Some(calls) = &mut self.calls else {
            return if split.is_empty() { Vec::new() } else { vec![Piece::Text(split)] };
        };
        let mut parts = calls.push(&split.content);
        if last {
            parts.extend(calls.finish());
        }
        let mut pieces = Vec::new();
        if !split.reasoning.is_empty() {
            pieces.push(Piece::Text(Split { reasoning: split.reasoning, content: String::new() }));
        }
        for part in parts {
            match (part, pieces.last_mut()) {
                (Part::Content(text), Some(Piece::Text(split))) => split.content.push_str(&text),
                (Part::Content(text), _) => {
                    pieces.push(Piece::Text(Split { reasoning: String::new(), content: text }));
                }
                (Part::Call { index, id, name }, _) => {
                    pieces.push(Piece::Call { index, id, name });
                }
                (Part::Arguments { index, json }, _) => {
                    pieces.push(Piece::Arguments { index, json });
                }
            }
        }
        pieces
    }
}

#[cfg(test)]
mod tests;
