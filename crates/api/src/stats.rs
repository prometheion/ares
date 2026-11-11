//! What an engine has done so far, for whoever monitors it.

use serde::{Deserialize, Serialize};

/// Totals of an engine since it started, and what it holds right now.
///
/// The counts only grow, so the difference of two copies says what happened
/// between them. A front end uses that for the draft counts of a request
/// that ran alone, until the engine reports them per request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineStats {
    /// Steps run.
    pub steps: u64,
    /// Prompt tokens computed.
    pub prompt_tokens: u64,
    /// Tokens delivered to requests.
    pub generated_tokens: u64,
    /// Draft tokens that were put to the test in a step.
    pub drafts_proposed: u64,
    /// Draft tokens that turned out to be what the model picks.
    pub drafts_accepted: u64,
    /// Requests that ran to their end.
    pub requests_finished: u64,
    /// Requests that were cancelled, by their client or by a shutdown.
    pub requests_cancelled: u64,
    /// Requests refused for lack of room.
    pub requests_refused: u64,
    /// Starts of a request from cells the prefix cache held, and the cells
    /// those starts took from it instead of computing them (a request that
    /// gave way and came back counts again).
    pub prefix_hits: u64,
    pub cached_cells: u64,
    /// Rows run again to rebuild recurrent state from cached cells.
    pub replayed_rows: u64,
    /// Checkpoints of recurrent state taken, restored by a start of a request,
    /// and dropped again.
    pub checkpoints_saved: u64,
    pub checkpoints_restored: u64,
    pub checkpoints_dropped: u64,
    /// Pages copied before a sequence wrote into a page that others read.
    pub page_copies: u64,
    /// Sequences running when the copy was made.
    pub sequences: u32,
    /// Pages of sequence state that are free.
    pub pages_free: u32,
    /// All pages of sequence state.
    pub pages: u32,
    /// Pages and tails the prefix cache holds.
    pub cached_pages: u32,
    /// Bytes of page-locked RAM held for checkpoints of recurrent state.
    pub checkpoint_ram: u64,
}

impl EngineStats {
    /// The draft tokens proposed and accepted between `earlier` and `self`.
    pub fn drafts_since(&self, earlier: &EngineStats) -> (u64, u64) {
        (
            self.drafts_proposed.saturating_sub(earlier.drafts_proposed),
            self.drafts_accepted.saturating_sub(earlier.drafts_accepted),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_difference_of_two_copies_is_what_happened_between_them() {
        let earlier =
            EngineStats { drafts_proposed: 10, drafts_accepted: 7, ..EngineStats::default() };
        let later =
            EngineStats { drafts_proposed: 25, drafts_accepted: 16, ..EngineStats::default() };
        assert_eq!(later.drafts_since(&earlier), (15, 9));
        // Counters that were reset in between give nothing instead of wrapping.
        assert_eq!(earlier.drafts_since(&later), (0, 0));
    }
}
