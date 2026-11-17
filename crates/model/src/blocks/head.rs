//! The head view: the output head's rows ordered by a draft vocabulary.
//!
//! Every drafter that projects to the vocabulary (an MTP block, a DFlash
//! drafter) reads the draft head: the first rows of the output head on the
//! drafters' device. A **draft vocabulary** (`--draft-vocab`, the tokens a
//! drafter may propose) puts the rows of its tokens first, in ascending order,
//! and the other rows behind them, so that the draft head is those rows alone:
//! a smaller product per link or pass, and no weights twice. The model's own
//! head is the same matrix, so its logits come in the view's order too, and a
//! list on the device names the token of each column (the sampler maps the
//! columns back to tokens).
//!
//! Without a vocabulary the view is the head as it is, and the draft head is
//! every row of it on the drafters' device. The tokens a drafter can pick on
//! the device are the draft head's: the device keeps their rows of the
//! embedding table (`blocks/rows.rs`).

use std::ops::Range;
use std::sync::Arc;

/// The order of the output head's rows and the rows the drafters read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadView {
    /// Row `r` of the head is token `order[r]`; `None`: row `r` is token `r`.
    pub order: Option<Arc<[u32]>>,
    /// The rows of the draft head: the first rows of the head on the
    /// drafters' device.
    pub rows: u32,
}

impl HeadView {
    /// The view of a head of `n_vocab` rows of which the first `on_device`
    /// are on the drafters' device. `vocabulary` lists the tokens drafters may
    /// propose (in any order, each once); without one the draft head is the
    /// rows on that device.
    pub fn new(n_vocab: u32, on_device: u32, vocabulary: Option<&[u32]>) -> Result<Self, String> {
        let Some(list) = vocabulary else {
            return Ok(Self { order: None, rows: on_device });
        };
        let mut listed = vec![false; n_vocab as usize];
        for &token in list {
            if token >= n_vocab || std::mem::replace(&mut listed[token as usize], true) {
                return Err(format!(
                    "draft vocabulary: token {token} is listed twice or is not one of the \
                     {n_vocab} tokens"
                ));
            }
        }
        if list.is_empty() || list.len() as u32 > on_device {
            return Err(format!(
                "a draft vocabulary of {} tokens; the drafters' device holds {on_device} rows of \
                 the head",
                list.len()
            ));
        }
        // The listed tokens in ascending order, then the others.
        let mut order: Vec<u32> = (0..n_vocab).filter(|t| listed[*t as usize]).collect();
        order.extend((0..n_vocab).filter(|t| !listed[*t as usize]));
        Ok(Self { order: Some(Arc::from(order)), rows: list.len() as u32 })
    }

    /// The token of row `row` of the head.
    pub fn token(&self, row: u32) -> u32 {
        match &self.order {
            Some(order) => order[row as usize],
            None => row,
        }
    }

    /// The tokens of the head's rows `rows`, in the view's order.
    pub fn tokens(&self, rows: Range<u32>) -> Vec<u32> {
        rows.map(|row| self.token(row)).collect()
    }

    /// The tokens of the draft head (ascending), whose rows of the embedding
    /// table the drafters' device keeps.
    pub fn draft_tokens(&self) -> Vec<u32> {
        self.tokens(0..self.rows)
    }
}

/// The rows of the tokens `tokens` as runs of consecutive tokens.
pub(crate) fn runs_of(tokens: &[u32]) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for &token in tokens {
        match runs.last_mut() {
            Some(run) if run.end == token as usize => run.end += 1,
            _ => runs.push(token as usize..token as usize + 1),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vocabulary_puts_its_tokens_first_in_ascending_order() {
        let view = HeadView::new(10, 10, Some(&[7, 2, 5])).unwrap();
        assert_eq!(view.rows, 3);
        assert_eq!(view.draft_tokens(), [2, 5, 7]);
        assert_eq!(view.tokens(0..10), [2, 5, 7, 0, 1, 3, 4, 6, 8, 9]);
        assert_eq!(runs_of(&view.tokens(0..10)), [2..3, 5..6, 7..8, 0..2, 3..5, 6..7, 8..10]);
        let plain = HeadView::new(10, 6, None).unwrap();
        assert_eq!((plain.rows, plain.token(4)), (6, 4));
        assert!(HeadView::new(10, 10, Some(&[1, 1])).is_err(), "twice");
        assert!(HeadView::new(10, 10, Some(&[10])).is_err(), "past the vocabulary");
        assert!(HeadView::new(10, 2, Some(&[1, 2, 3])).is_err(), "more than the device holds");
    }
}
