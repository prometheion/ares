//! Where checkpoints of recurrent state go, which are dropped, and what a
//! prompt that starts from the cache replays.
//!
//! The state of recurrent layers is one blob that every token rewrites, so a
//! prompt can resume from cached pages only where a copy of that blob was
//! kept. A copy is large (75 MiB as `f16` for Qwen3.8-27B), so where copies
//! are kept decides whether a prefix hit is worth anything. This module
//! decides; storing the copies is someone else's part.
//!
//! **Where they go** ([`CheckpointPolicy::next_capture`]). Only at page
//! boundaries, where the index has a name to attach them to, and at the end
//! of a tail. A prompt of a conversation names the places where its parts
//! end ([`Boundary`]: the chat template knows them), because a later prompt
//! that changes the conversation (an edited message, an older tool result
//! cut, a new conversation with the same system prompt) shares the text up to
//! such a place:
//!
//! - at the **roots**, the page boundary at or before the end of the system
//!   prompt and of the tool list, always: every conversation of an agent
//!   starts there (llama.cpp fork's `--checkpoint-root`);
//! - at **every message boundary**: the page boundary at or before it, when
//!   it lies at least [`CheckpointPolicy::boundary_interval`] cells after the
//!   last checkpoint, so that a conversation of many short messages does not
//!   copy the state at every page. A prompt step that ends a little before the
//!   boundary (at most [`CheckpointPolicy::boundary_reach`] cells, a page
//!   boundary) takes it where it ends, so that no step is cut for it: a step
//!   that is cut costs a pass over the weights. Two such checkpoints are
//!   therefore the reach apart at least, too;
//! - inside a prompt at least [`CheckpointPolicy::prompt_interval`] cells
//!   apart (the fixed distance), while generating at least
//!   [`CheckpointPolicy::generation_interval`] apart: measured from the last
//!   checkpoint on the sequence's own path and not on a fixed grid, so a
//!   sequence that resumed from a checkpoint continues the spacing from there;
//! - always at the prompt's anchor, the last page boundary that is at least
//!   [`CheckpointPolicy::volatile_rows`] before the prompt's end (ExLlamaV3's
//!   last page boundary, LMDeploy's skip of the volatile end): the next turn
//!   of a conversation shares the prompt up to about there and replays from
//!   the anchor less than a page of rows;
//! - at the end of a request the live state is kept as the checkpoint of the
//!   tail the request publishes, which costs no computation (LMDeploy adopts
//!   the terminal state the same way). The caller does that; no rule decides
//!   it.
//!
//! The commit emits a checkpoint when it crosses one of these positions
//! (SGLang): the caller advances the recurrent state to the position, takes
//! the copy, and goes on.
//!
//! **What is replayed** ([`CheckpointPolicy::resume`]). The K and V of a
//! prompt are taken from every cached page and from a tail, however far the
//! last checkpoint is behind them. The rows between that checkpoint and the
//! end of the cached cells are replayed: they run through the model to
//! rebuild the recurrent state, they read K and V, and they write none
//! (LMDeploy's read-only blocks). That keeps the pages shared and saves the
//! scale search of the low-bit formats; it does not save the rows' time.
//!
//! **Which are dropped** ([`PrefixIndex::checkpoint_ranks`], the order in which
//! the runtime's cache gives them up). There is a budget in bytes. When it is
//! full:
//!
//! 1. Aged checkpoints go before fresh ones. The checkpoints form a tree by
//!    their text; a checkpoint on one branch is aged when more than
//!    [`CheckpointPolicy::fresh_requests`] later requests have published
//!    checkpoints below it (TokenSpeed), or when at least
//!    [`CheckpointPolicy::fresh_checkpoints`] checkpoints lie below it on
//!    that branch (the llama.cpp fork keeps the newest three of a cached
//!    prompt). The newest checkpoints of a conversation are where its next
//!    turn resumes; old ones are passed, and a long prompt with many message
//!    boundaries does not push the newest checkpoints of other conversations
//!    out. A checkpoint with checkpoints on two or more branches below it has
//!    been resumed from more than once and is never aged, and the checkpoints
//!    one request publishes are no step apart for the count of requests, so a
//!    request's anchor is as fresh as its end.
//! 2. Roots go after every other checkpoint and never age: every new
//!    conversation of an agent starts from them.
//! 3. Within those, the checkpoint with the least distance to the next
//!    checkpoint below it (toward the start of the text) goes first: that
//!    distance is the rows a resume replays more when it is gone. An end
//!    state right after an anchor goes before a checkpoint that stands alone
//!    for 30 000 rows (LMDeploy demotes such a checkpoint the same way), and
//!    as a dense run thins out, the distances of the ones that are left grow
//!    (the fork's `--checkpoint-spread`).
//! 4. Then the least recently used.
//!
//! Dropping a checkpoint never takes a K/V page with it, so on entries inside
//! a chain the checkpoint goes before the page (SGLang's order); a page at
//! the end of a chain that is evicted takes its checkpoint along
//! ([`Evicted::checkpoint`](crate::Evicted)).

use std::collections::HashMap;
use std::ops::Range;

use crate::prefix::{ChainHash, CheckpointId, Owner, PrefixIndex, PrefixMatch};

/// A place in a prompt where a part of a conversation ends: what a later
/// prompt that changes the conversation after it still shares with this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Boundary {
    /// The number of the prompt's tokens before it.
    pub at: u64,
    /// The end of the system prompt or of the tool list: the start that the
    /// conversations of an agent share.
    pub root: bool,
}

/// The numbers of the checkpoint policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointPolicy {
    /// Cells per page.
    pub cells_per_page: u32,
    /// The least distance in cells between checkpoints inside a prompt.
    pub prompt_interval: u32,
    /// The least distance in cells between checkpoints while generating.
    pub generation_interval: u32,
    /// The least distance in cells from the last checkpoint to a message
    /// boundary that gets one.
    pub boundary_interval: u32,
    /// How far in cells before a message boundary its checkpoint may lie: a
    /// prompt step that ends this close before it, on a page boundary, takes
    /// the checkpoint there instead of a step that is cut at the boundary.
    /// The size of a piece of prompt is the right reach; 0 cuts at every
    /// boundary.
    pub boundary_reach: u32,
    /// How many of a prompt's last rows the next turn of the conversation is
    /// not expected to share (what a chat template puts after the last
    /// message and then renders differently). The anchor is at least this far
    /// before the prompt's end.
    pub volatile_rows: u32,
    /// The size of one checkpoint.
    pub bytes_per_checkpoint: u64,
    /// The most bytes all checkpoints together may take.
    pub budget_bytes: u64,
    /// A checkpoint stays fresh while at most this many later requests have
    /// published checkpoints below it. TokenSpeed's default is 0: only the
    /// newest request's checkpoints on a branch are fresh.
    pub fresh_requests: u32,
    /// A checkpoint stays fresh while fewer than this many checkpoints lie
    /// below it on its branch: the newest ones of a conversation.
    pub fresh_checkpoints: u32,
}

/// Distances beyond this many requests rank alike (TokenSpeed caps at 5 too).
const MAX_REQUESTS_BELOW: u32 = 5;

impl CheckpointPolicy {
    /// The cells between checkpoints in a prompt: the llama.cpp fork's
    /// spacing (ExLlamaV3 has 32768).
    pub const PROMPT_INTERVAL: u32 = 8192;
    /// The cells between checkpoints while generating (ExLlamaV3).
    pub const GENERATION_INTERVAL: u32 = 2048;
    /// The fewest cells between the checkpoints of two message boundaries.
    pub const BOUNDARY_INTERVAL: u32 = 1024;

    /// Checkpoints at the intervals above, LMDeploy's one volatile row, the
    /// newest three checkpoints of a conversation fresh (the fork), for
    /// checkpoints of `bytes_per_checkpoint` within `budget_bytes`.
    ///
    /// # Panics
    ///
    /// Panics when `cells_per_page` is 0.
    pub fn new(cells_per_page: u32, bytes_per_checkpoint: u64, budget_bytes: u64) -> Self {
        assert!(cells_per_page > 0, "a page holds at least one cell");
        Self {
            cells_per_page,
            prompt_interval: Self::PROMPT_INTERVAL,
            generation_interval: Self::GENERATION_INTERVAL,
            boundary_interval: Self::BOUNDARY_INTERVAL,
            boundary_reach: 0,
            volatile_rows: 1,
            bytes_per_checkpoint,
            budget_bytes,
            fresh_requests: 0,
            fresh_checkpoints: 3,
        }
    }

    /// How many checkpoints fit in the budget.
    pub fn capacity(&self) -> usize {
        match self.budget_bytes.checked_div(self.bytes_per_checkpoint) {
            Some(n) => usize::try_from(n).unwrap_or(usize::MAX),
            // A model without recurrent state has checkpoints of no size and
            // needs none.
            None => 0,
        }
    }

    /// The anchor of a prompt of `prompt_len` cells: the last page boundary
    /// at least [`CheckpointPolicy::volatile_rows`] before its end. 0 when
    /// there is none, which is the empty state that needs no checkpoint.
    pub fn anchor(&self, prompt_len: u64) -> u64 {
        let per_page = u64::from(self.cells_per_page);
        prompt_len.saturating_sub(u64::from(self.volatile_rows)) / per_page * per_page
    }

    /// The first position in `(from, to]` at which a checkpoint is due, for a
    /// sequence whose prompt is `prompt_len` cells long with the parts that
    /// end at `boundaries` (ascending), and whose last checkpoint (restored or
    /// taken) is at `last` (0: none).
    ///
    /// `from` is where the recurrent state is now and `to` where the rows
    /// about to be committed (or replayed) take it. The caller advances the
    /// state to the returned position, takes the checkpoint unless the index
    /// has one there already, and asks again with `from` and `last` at that
    /// position. `None`: the rows run through. A step that ends at `to` may
    /// be told `to` for a message boundary a little after it
    /// ([`CheckpointPolicy::boundary_reach`]); a step cut shorter is asked
    /// again with its own end.
    pub fn next_capture(
        &self,
        prompt_len: u64,
        boundaries: &[Boundary],
        last: u64,
        from: u64,
        to: u64,
    ) -> Option<u64> {
        if self.capacity() == 0 {
            return None;
        }
        let per_page = u64::from(self.cells_per_page);
        let floor = |x: u64| x / per_page * per_page;
        let boundary_after = |x: u64| (x / per_page + 1) * per_page;
        let boundary_at_or_after = |x: u64| x.div_ceil(per_page) * per_page;
        let anchor = self.anchor(prompt_len);

        // Sparse inside the prompt, before the anchor.
        let sparse =
            boundary_after(from).max(boundary_at_or_after(last + u64::from(self.prompt_interval)));
        let sparse = (sparse < anchor && sparse <= to).then_some(sparse);
        // Always at the anchor.
        let at_anchor = (anchor > from && anchor > last && anchor <= to).then_some(anchor);
        // Dense after it.
        let dense = boundary_after(from.max(anchor))
            .max(boundary_at_or_after(last + u64::from(self.generation_interval)));
        let dense = (dense <= to).then_some(dense);
        // The roots, always; the message boundaries far enough from the last
        // checkpoint, at the end of a step that ends a little before one, or
        // at the boundary.
        let reach = floor(u64::from(self.boundary_reach));
        let step_end = to.is_multiple_of(per_page).then_some(to);
        let parts = boundaries.iter().filter_map(|boundary| {
            let at = floor(boundary.at);
            if at <= from.max(last) || at >= anchor {
                return None;
            }
            if boundary.root {
                return (at <= to).then_some(at);
            }
            // Too close to the last checkpoint, or within reach of it: a
            // step that ended there served the boundary already.
            if at < last + u64::from(self.boundary_interval).max(reach) {
                return None;
            }
            match step_end {
                Some(end) if end > from && end > last && end <= at && end + reach >= at => {
                    Some(end)
                }
                _ => (at <= to).then_some(at),
            }
        });
        [sparse, at_anchor, dense].into_iter().flatten().chain(parts).min()
    }

    /// Whether a checkpoint at `at` is at a root of the prompt with the parts
    /// that end at `boundaries`: the page boundary at or before the end of
    /// its system prompt or of its tool list.
    pub fn at_root(&self, boundaries: &[Boundary], at: u64) -> bool {
        let per_page = u64::from(self.cells_per_page);
        at > 0 && boundaries.iter().any(|b| b.root && b.at / per_page * per_page == at)
    }

    /// What a prompt of `prompt_len` cells does with the match `found`.
    ///
    /// The match must come from a lookup of the prompt without its last
    /// token ([`PrefixIndex::lookup`]), so at least one row is computed.
    pub fn resume(&self, found: &PrefixMatch, prompt_len: u64) -> Resume {
        let per_page = u64::from(self.cells_per_page);
        let cached = found.cells(self.cells_per_page).min(prompt_len);
        let (mut state_at, mut restore) = match found.last_checkpoint() {
            Some((pages, id)) => (pages as u64 * per_page, Some(id)),
            None => (0, None),
        };
        // The end of a tail that the prompt shares whole is the deepest
        // place there is.
        if let Some(id) = found.tail.and_then(|tail| tail.checkpoint) {
            (state_at, restore) = (cached, Some(id));
        }
        Resume {
            cached_cells: cached,
            restore,
            replay: state_at..cached,
            compute: cached..prompt_len,
        }
    }
}

/// How a prompt starts from the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resume {
    /// The cells whose K and V are taken from cached pages and a tail. The
    /// sequence starts with this length; these cells cost no pages.
    pub cached_cells: u64,
    /// The checkpoint to restore before anything runs. `None`: the recurrent
    /// state starts empty, at position 0.
    pub restore: Option<CheckpointId>,
    /// The rows that are replayed, from the restored state to the end of the
    /// cached cells: they rebuild the recurrent state, read K and V and write
    /// none.
    pub replay: Range<u64>,
    /// The rows that are computed and written: the rest of the prompt.
    pub compute: Range<u64>,
}

impl Resume {
    /// The rows the device runs for the prompt: replayed and computed ones.
    /// This is what the prompt costs in time; `compute` is what it costs in
    /// pages.
    pub fn rows(&self) -> u64 {
        (self.replay.end - self.replay.start) + (self.compute.end - self.compute.start)
    }
}

/// How a checkpoint ranks when the budget is full. The first in the order of
/// [`PrefixIndex::checkpoint_ranks`] is dropped first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointRank {
    /// The entry or tail that carries the checkpoint.
    pub hash: ChainHash,
    /// The checkpoint.
    pub checkpoint: CheckpointId,
    /// Who published it, when that is known.
    pub by: Option<Owner>,
    /// The position in cells of the state it holds.
    pub position: u64,
    /// On its one branch, more than the allowed number of later requests
    /// have published checkpoints below it, or more checkpoints than the
    /// newest that stay fresh lie below it.
    pub aged: bool,
    /// It is at a root ([`PrefixIndex::mark_root`]): it goes last.
    pub root: bool,
    /// The distance in cells to the next checkpoint toward the start of the
    /// text (or to the start): the rows a resume replays more without it.
    pub rows_saved: u64,
    /// When its entry was last used.
    pub last_used: u64,
}

struct Node {
    rank: CheckpointRank,
    by: Option<Owner>,
    parent: Option<usize>,
    children: u32,
    /// The least number of later requests down to a checkpoint with none
    /// below it.
    requests_below: u32,
    /// The least number of checkpoints down to one with none below it.
    checkpoints_below: u32,
}

impl Node {
    /// Whether later requests or newer checkpoints on its one branch have
    /// left it behind under `policy`. A root and a checkpoint where texts
    /// branch never are.
    fn aged(&self, policy: &CheckpointPolicy) -> bool {
        self.children == 1
            && !self.rank.root
            && (self.requests_below > policy.fresh_requests
                || self.checkpoints_below >= policy.fresh_checkpoints)
    }
}

impl PrefixIndex {
    /// The number of checkpoints attached to entries and tails.
    pub fn checkpoints(&self) -> usize {
        self.checkpoints
    }

    /// Attaches the checkpoint `id`, published by `by`, to the entry or tail
    /// `hash`. Returns the checkpoint that was attached before, which the
    /// caller owns again; `Err` gives `id` back when there is no such entry.
    ///
    /// The caller makes room first ([`PrefixIndex::checkpoint_ranks`] says
    /// which go).
    pub fn attach_checkpoint(
        &mut self,
        hash: ChainHash,
        id: CheckpointId,
        by: Owner,
    ) -> Result<Option<CheckpointId>, CheckpointId> {
        self.replace_checkpoint(hash, Some(id), Some(by)).map_err(|_| id)
    }

    /// The entry or tail with a checkpoint that lies above `child`, a page
    /// under `prior`: a tail of `prior` that `child` continues, or the
    /// nearest entry up the chain.
    fn checkpoint_above(
        &self,
        mut child: ChainHash,
        mut prior: ChainHash,
        tails: bool,
    ) -> Option<ChainHash> {
        loop {
            let tail = self.tails.get(&prior).filter(|_| tails).and_then(|tails| {
                tails
                    .iter()
                    .filter(|tail| tail.checkpoint.is_some() && tail.extended_by.contains(&child))
                    .max_by_key(|tail| tail.tokens.len())
            });
            if let Some(tail) = tail {
                return Some(tail.hash);
            }
            let entry = self.entries.get(&prior)?;
            if entry.checkpoint.is_some() {
                return Some(prior);
            }
            (child, prior) = (prior, entry.parent);
        }
    }

    /// The tree of checkpoints by their text: of entries and tails, or of
    /// entries only without `tails`.
    fn checkpoint_nodes(&self, cells_per_page: u32, tails: bool) -> Vec<Node> {
        let per_page = u64::from(cells_per_page);
        let mut nodes: Vec<Node> = Vec::with_capacity(self.checkpoints);
        let mut above: Vec<Option<ChainHash>> = Vec::with_capacity(self.checkpoints);
        let node = |hash, checkpoint, position, last_used, by, root| Node {
            rank: CheckpointRank {
                hash,
                checkpoint,
                by,
                position,
                aged: false,
                root,
                rows_saved: position,
                last_used,
            },
            by,
            parent: None,
            children: 0,
            requests_below: 0,
            checkpoints_below: 0,
        };
        for (hash, entry) in &self.entries {
            let Some(checkpoint) = entry.checkpoint else { continue };
            let position = u64::from(entry.depth) * per_page;
            nodes.push(node(
                *hash,
                checkpoint,
                position,
                entry.last_used,
                entry.checkpoint_by,
                entry.checkpoint_root,
            ));
            above.push(self.checkpoint_above(*hash, entry.parent, tails));
        }
        for (prior, tails) in self.tails.iter().filter(|_| tails) {
            let depth = self.entries.get(prior).map_or(0, |entry| entry.depth);
            for tail in tails {
                let Some(checkpoint) = tail.checkpoint else { continue };
                let position = u64::from(depth) * per_page + tail.tokens.len() as u64;
                nodes.push(node(
                    tail.hash,
                    checkpoint,
                    position,
                    tail.last_used,
                    tail.checkpoint_by,
                    false,
                ));
                // A shorter tail of the same page whose text this one
                // continues is above it; else the page before, or what is
                // above that.
                let shorter = tails
                    .iter()
                    .filter(|other| other.checkpoint.is_some())
                    .filter(|other| other.tokens.len() < tail.tokens.len())
                    .filter(|other| tail.tokens.starts_with(&other.tokens))
                    .max_by_key(|other| other.tokens.len())
                    .map(|other| other.hash);
                above.push(shorter.or_else(|| match self.entries.get(prior) {
                    Some(entry) if entry.checkpoint.is_some() => Some(*prior),
                    Some(entry) => self.checkpoint_above(*prior, entry.parent, true),
                    None => None,
                }));
            }
        }
        let by_hash: HashMap<ChainHash, usize> =
            nodes.iter().enumerate().map(|(i, node)| (node.rank.hash, i)).collect();
        for (i, above) in above.into_iter().enumerate() {
            let Some(parent) = above.and_then(|hash| by_hash.get(&hash).copied()) else { continue };
            nodes[i].parent = Some(parent);
            nodes[i].rank.rows_saved = nodes[i].rank.position - nodes[parent].rank.position;
        }

        // From the deepest up, so that a checkpoint is finished before the
        // one above it reads it.
        let mut order: Vec<usize> = (0..nodes.len()).collect();
        order.sort_by_key(|i| std::cmp::Reverse(nodes[*i].rank.position));
        for i in &order {
            if nodes[*i].children == 0 {
                nodes[*i].requests_below = 0;
                nodes[*i].checkpoints_below = 0;
            }
            let Some(parent) = nodes[*i].parent else { continue };
            // The checkpoints one request publishes are no step apart.
            let same_request = nodes[*i].by.is_some() && nodes[*i].by == nodes[parent].by;
            let seen =
                (nodes[*i].requests_below + u32::from(!same_request)).min(MAX_REQUESTS_BELOW);
            let below = nodes[*i].checkpoints_below.saturating_add(1);
            let first = nodes[parent].children == 0;
            nodes[parent].children += 1;
            let parent = &mut nodes[parent];
            (parent.requests_below, parent.checkpoints_below) = if first {
                (seen, below)
            } else {
                (parent.requests_below.min(seen), parent.checkpoints_below.min(below))
            };
        }
        nodes
    }

    /// Every checkpoint with its rank under `policy`, the one that is
    /// dropped first at the front.
    pub fn checkpoint_ranks(&self, policy: &CheckpointPolicy) -> Vec<CheckpointRank> {
        let mut ranks: Vec<CheckpointRank> = self
            .checkpoint_nodes(policy.cells_per_page, true)
            .into_iter()
            .map(|node| CheckpointRank { aged: node.aged(policy), ..node.rank })
            .collect();
        ranks.sort_by_key(|rank| {
            (!rank.aged, rank.root, rank.rows_saved, rank.last_used, rank.hash)
        });
        ranks
    }

    /// The checkpoints at the ends of pages that later requests have left
    /// behind, counting only checkpoints at the ends of pages: on their one
    /// branch, more than [`CheckpointPolicy::fresh_requests`] later requests
    /// have checkpoints at deeper page boundaries. For a level that keeps no
    /// tails (and no state at their ends), where a prompt resumes from the
    /// deepest checkpoint at a page boundary: a checkpoint that later turns
    /// left behind only with the state at the end of their tails is not left
    /// behind there. A root never is: a new conversation starts from it.
    pub fn left_behind_on_pages(&self, policy: &CheckpointPolicy) -> Vec<ChainHash> {
        self.checkpoint_nodes(policy.cells_per_page, false)
            .into_iter()
            .filter(|node| {
                node.children == 1 && !node.rank.root && node.requests_below > policy.fresh_requests
            })
            .map(|node| node.rank.hash)
            .collect()
    }

    /// Detaches checkpoints until `incoming` more fit in the policy's budget,
    /// in the order of [`PrefixIndex::checkpoint_ranks`], and returns them.
    /// The caller frees their storage. K/V pages are not touched.
    ///
    /// When the budget holds fewer than `incoming` checkpoints, every
    /// checkpoint is returned and the caller keeps as many of the new ones as
    /// [`CheckpointPolicy::capacity`] allows. For the tests of the order.
    #[cfg(test)]
    pub(crate) fn make_room_for_checkpoints(
        &mut self,
        policy: &CheckpointPolicy,
        incoming: usize,
    ) -> Vec<CheckpointId> {
        let mut dropped = Vec::new();
        while self.checkpoints > 0 && self.checkpoints + incoming > policy.capacity() {
            // Ranks change with every checkpoint that goes: the distance of
            // the one below the gap grows, and branches merge.
            let Some(first) = self.checkpoint_ranks(policy).first().copied() else { break };
            if let Ok(Some(id)) = self.set_checkpoint(first.hash, None) {
                dropped.push(id);
            }
        }
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PagePool, Priority, Sequence};

    const MIB: u64 = 1 << 20;

    /// Qwen3.8-27B: pages of one tile, checkpoints of 75 MiB, 3 GiB for them,
    /// ExLlamaV3's sparse interval of 32768 in a prompt (the tests below were
    /// written for it).
    fn policy() -> CheckpointPolicy {
        CheckpointPolicy {
            prompt_interval: 32_768,
            ..CheckpointPolicy::new(128, 75 * MIB, 3072 * MIB)
        }
    }

    /// The positions at which a sequence takes checkpoints when it runs from
    /// `from` to `to` in steps of `step` rows.
    fn captures(
        policy: &CheckpointPolicy,
        prompt_len: u64,
        last: u64,
        from: u64,
        to: u64,
        step: u64,
    ) -> Vec<u64> {
        captures_at(policy, prompt_len, &[], last, from, to, step)
    }

    /// The same for a prompt whose parts end at `boundaries`, as the
    /// conductor runs it: a step of `step` rows ends where a checkpoint is
    /// due, and the commit takes it when the step ends there.
    fn captures_at(
        policy: &CheckpointPolicy,
        prompt_len: u64,
        boundaries: &[Boundary],
        mut last: u64,
        from: u64,
        to: u64,
        step: u64,
    ) -> Vec<u64> {
        let per_page = u64::from(policy.cells_per_page);
        let mut taken = Vec::new();
        let mut at = from;
        while at < to {
            let mut end = (at + step).min(to);
            if let Some(due) = policy.next_capture(prompt_len, boundaries, last, at, end) {
                assert!(due > at && due <= end && due % per_page == 0);
                end = due;
            }
            if policy.next_capture(prompt_len, boundaries, last, at, end) == Some(end) {
                taken.push(end);
                last = end;
            }
            at = end;
        }
        taken
    }

    #[test]
    fn checkpoints_are_sparse_in_a_prompt_dense_in_generation_and_always_at_the_anchor() {
        let p = policy();
        assert_eq!(p.capacity(), 40);
        // A prompt of 70 000 tokens read in chunks of 2048 rows, then 6000
        // generated tokens, one per step.
        let prompt = 70_000;
        assert_eq!(p.anchor(prompt), 69_888);
        let taken = captures(&p, prompt, 0, 0, prompt, 2048);
        assert_eq!(taken, [32_768, 65_536, 69_888], "two sparse ones and the anchor");
        let taken = captures(&p, prompt, 69_888, prompt, prompt + 6000, 1);
        assert_eq!(taken, [71_936, 73_984], "every 2048 cells from the anchor on");

        // The chunks do not matter: one pass over the prompt, or steps of 100.
        assert_eq!(captures(&p, prompt, 0, 0, prompt, prompt), [32_768, 65_536, 69_888]);
        assert_eq!(captures(&p, prompt, 0, 0, prompt, 100), [32_768, 65_536, 69_888]);

        // A short prompt has the anchor and nothing else.
        assert_eq!(captures(&p, 1000, 0, 0, 1000, 256), [896]);
        // A prompt shorter than a page has no boundary to anchor at; the
        // empty state before it needs no checkpoint.
        assert_eq!(p.anchor(100), 0);
        assert_eq!(captures(&p, 100, 0, 0, 100, 100), []);
        assert_eq!(captures(&p, 100, 0, 100, 4200, 1), [2048, 4096]);
    }

    #[test]
    fn the_anchor_is_a_little_before_the_prompts_end() {
        let mut p = policy();
        // A prompt that ends exactly on a page boundary: its last token is
        // volatile, so the anchor is the boundary before.
        assert_eq!(p.anchor(1280), 1152);
        assert_eq!(p.anchor(1281), 1280);
        // A template that ends every prompt with three tokens the next turn
        // does not share.
        p.volatile_rows = 3;
        assert_eq!(p.anchor(1282), 1152);
        assert_eq!(p.anchor(1283), 1280);
        // The boundary between the anchor and the end is not skipped for
        // ever: it counts as generation and gets a checkpoint when the dense
        // interval is due.
        assert_eq!(captures(&p, 1282, 1152, 1152, 1282 + 5000, 7), [3200, 5248]);
    }

    #[test]
    fn spacing_continues_from_a_restored_checkpoint() {
        let p = policy();
        // Resumed at 40 960 (a checkpoint of an earlier request there), the
        // prompt goes on to 120 000: the next sparse one is 32 768 later.
        let taken = captures(&p, 120_000, 40_960, 40_960, 120_000, 2048);
        assert_eq!(taken, [73_728, 106_496, 119_936]);
        // Replaying from a checkpoint far behind: the replayed rows pass the
        // positions where checkpoints are due and take them.
        let taken = captures(&p, 120_000, 0, 0, 100_000, 4096);
        assert_eq!(taken, [32_768, 65_536, 98_304]);
        // No budget, no checkpoints.
        let none = CheckpointPolicy { budget_bytes: 74 * MIB, ..p };
        assert_eq!(none.capacity(), 0);
        assert_eq!(captures(&none, 120_000, 0, 0, 120_000, 2048), []);
    }

    fn boundary(at: u64) -> Boundary {
        Boundary { at, root: false }
    }

    fn root(at: u64) -> Boundary {
        Boundary { at, root: true }
    }

    #[test]
    fn a_conversation_has_checkpoints_at_its_roots_its_message_boundaries_and_fixed_distances() {
        // The defaults: 8192 in a prompt, message boundaries 1024 apart.
        let p = CheckpointPolicy::new(128, 75 * MIB, 3072 * MIB);
        // A system prompt whose tools end at 3000 and whose text ends at
        // 12 000, then messages; the prompt is 30 000 tokens long.
        let parts = [
            root(3000),
            root(12_000),
            boundary(12_010),
            boundary(12_500),
            boundary(13_100),
            boundary(13_300),
            boundary(15_000),
            boundary(29_990),
        ];
        let prompt = 30_000;
        // Read in steps of 512 rows from nothing.
        let taken = captures_at(&p, prompt, &parts, 0, 0, prompt, 512);
        assert_eq!(
            taken,
            [
                2944,   // the end of the tool list
                11_136, // 8192 after it
                11_904, // the end of the system prompt (12 010 is on the same page)
                13_056, // 13 100: 1024 past the root (12 500 is too close)
                14_976, // 15 000
                23_168, // 8192 after the last checkpoint
                29_952, // 29 990, which is also the anchor
            ]
        );
        let roots: Vec<u64> = taken.iter().copied().filter(|at| p.at_root(&parts, *at)).collect();
        assert_eq!(roots, [2944, 11_904]);
        // Without boundaries only the fixed distance and the anchor.
        assert_eq!(captures(&p, prompt, 0, 0, prompt, 512), [8192, 16_384, 24_576, 29_952]);
        // A prompt that resumes after the system prompt (a new conversation of
        // the same agent) goes on from its root.
        let taken = captures_at(&p, prompt, &parts, 11_904, 11_904, prompt, 512);
        assert_eq!(taken, [13_056, 14_976, 23_168, 29_952]);
    }

    #[test]
    fn a_step_that_ends_a_little_before_a_message_boundary_takes_its_checkpoint() {
        // Steps of 256 rows reach a page boundary 256 rows before the next
        // message boundary at most.
        let p =
            CheckpointPolicy { boundary_reach: 256, ..CheckpointPolicy::new(128, MIB, 64 * MIB) };
        let parts = [boundary(1500), boundary(2100), boundary(2700), boundary(5000)];
        // From nothing in steps of 256: the steps end on page boundaries, and
        // one ends within reach before each boundary that is due, so no step
        // is cut: 1280 for 1500 (page 1408), 2560 for 2700 (2688); 2100 is
        // too close to 1280.
        let steps = |taken: &[u64]| taken.iter().all(|at| at % 256 == 0);
        let taken = captures_at(&p, 6000, &parts, 0, 0, 6000, 256);
        assert_eq!(taken, [1280, 2560, 4864, 5888]);
        assert!(steps(&taken), "{taken:?}");
        // Steps that start off a page boundary (a prompt resumed from a
        // tail) are cut at the boundary's page, and from there on they end
        // on page boundaries again.
        let taken = captures_at(&p, 6000, &parts, 0, 1000, 6000, 256);
        assert_eq!(taken, [1408, 2432, 4736, 5888]);
        // Without reach every boundary that is due cuts its step.
        let cut = CheckpointPolicy { boundary_reach: 0, ..p };
        assert_eq!(captures_at(&cut, 6000, &parts, 0, 0, 6000, 256), [1408, 2688, 4992, 5888]);
        // A boundary after the anchor counts as generation; one in the first
        // page has no page boundary before it.
        let late = [boundary(100), boundary(5990)];
        assert_eq!(captures_at(&cut, 6000, &late, 0, 0, 6000, 256), [5888]);
        // No budget, no checkpoints.
        let none = CheckpointPolicy { budget_bytes: 0, ..p };
        assert_eq!(captures_at(&none, 6000, &parts, 0, 0, 6000, 256), []);
        // Pieces larger than the interval: a boundary is served once, at the
        // end of the piece before it, and the next boundary that is due lies
        // a piece further at least; no step is cut.
        let wide = CheckpointPolicy { boundary_reach: 2048, ..p };
        let many: Vec<Boundary> = (1..40).map(|i| boundary(i * 300)).collect();
        let taken = captures_at(&wide, 12_000, &many, 0, 0, 12_000, 2048);
        assert_eq!(taken, [2048, 4096, 6144, 8192, 10_240, 11_904]);
    }

    const PER_PAGE: u32 = 4;

    fn small() -> CheckpointPolicy {
        CheckpointPolicy {
            cells_per_page: PER_PAGE,
            prompt_interval: 16,
            generation_interval: 8,
            boundary_interval: 8,
            boundary_reach: 0,
            volatile_rows: 1,
            bytes_per_checkpoint: 1,
            budget_bytes: 100,
            fresh_requests: 0,
            fresh_checkpoints: 3,
        }
    }

    /// Runs a sequence over `tokens` from the cache or from nothing, and
    /// publishes its pages. Returns the sequence and its chain.
    fn run(
        pool: &mut PagePool,
        index: &mut PrefixIndex,
        root: ChainHash,
        tokens: &[u64],
    ) -> (Sequence, crate::Chain) {
        let found = index.lookup(root, tokens, PER_PAGE);
        let mut seq = Sequence::from_shared(pool, &found.table(), found.cells(PER_PAGE));
        let mut chain = crate::Chain::resumed(root, PER_PAGE, &found);
        seq.append(pool, tokens.len() as u64 - seq.len()).unwrap();
        index.publish(pool, &mut seq, &mut chain, tokens, tokens.len() as u64, Priority::NORMAL);
        (seq, chain)
    }

    #[test]
    fn a_prompt_replays_from_the_last_checkpoint_to_the_end_of_the_cached_cells() {
        let p = small();
        let mut pool = PagePool::new(64, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let text: Vec<u64> = (0..23).collect();
        let (seq, chain) = run(&mut pool, &mut index, root, &text);
        // 5 full pages and a tail of 3 cells; a checkpoint after page 2.
        let tail = seq.pages()[5];
        index.insert_tail(&mut pool, chain.hashes()[4], &text[20..], tail, Priority::NORMAL);
        index.attach_checkpoint(chain.hashes()[1], CheckpointId(8), Owner(1)).unwrap();
        seq.release(&mut pool);

        // A prompt that continues the whole text: looked up without its last
        // token, it takes 5 pages and the 3 cells of the tail (23 cells), and
        // replays 8..23 because the tail has no checkpoint.
        let mut prompt = text.clone();
        prompt.extend([100, 101, 102]);
        let lookup = |index: &mut PrefixIndex, prompt: &[u64]| {
            index.lookup(root, &prompt[..prompt.len() - 1], PER_PAGE)
        };
        let found = lookup(&mut index, &prompt);
        let resume = p.resume(&found, prompt.len() as u64);
        assert_eq!(
            resume,
            Resume {
                cached_cells: 23,
                restore: Some(CheckpointId(8)),
                replay: 8..23,
                compute: 23..26
            }
        );
        assert_eq!(resume.rows(), 18);

        // With the end state kept at the tail nothing is replayed.
        let tail_hash = PrefixIndex::tail_hash(chain.hashes()[4], &text[20..]);
        index.attach_checkpoint(tail_hash, CheckpointId(9), Owner(1)).unwrap();
        let found = lookup(&mut index, &prompt);
        let resume = p.resume(&found, prompt.len() as u64);
        assert_eq!(
            (resume.restore, resume.replay, resume.compute),
            (Some(CheckpointId(9)), 23..23, 23..26)
        );

        // A prompt that leaves the text inside the tail takes the cells it
        // shares (22) and cannot use the tail's checkpoint, which is the
        // state one cell later.
        let mut other = text[..22].to_vec();
        other.extend([7, 7, 7]);
        let found = lookup(&mut index, &other);
        let resume = p.resume(&found, other.len() as u64);
        assert_eq!(
            (resume.cached_cells, resume.restore, resume.replay),
            (22, Some(CheckpointId(8)), 8..22)
        );

        // The same text again: everything is cached, and still the last row
        // is computed. The tail's checkpoint is not the state before it.
        let found = lookup(&mut index, &text);
        let resume = p.resume(&found, text.len() as u64);
        assert_eq!(
            (resume.cached_cells, resume.restore, resume.compute),
            (22, Some(CheckpointId(8)), 22..23)
        );

        // No checkpoint on the path: the cached cells are still taken, and
        // all of them are replayed.
        index.set_checkpoint(chain.hashes()[1], None).unwrap();
        let found = lookup(&mut index, &other);
        let resume = p.resume(&found, other.len() as u64);
        assert_eq!((resume.cached_cells, resume.restore, resume.replay), (22, None, 0..22));
        // Nothing cached: nothing replayed.
        let found = index.lookup(ChainHash::root(b"other"), &other, PER_PAGE);
        assert_eq!(
            p.resume(&found, other.len() as u64),
            Resume { cached_cells: 0, restore: None, replay: 0..0, compute: 0..25 }
        );
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 64);
    }

    /// Builds one conversation: every turn extends the text and leaves
    /// checkpoints at the given positions (page boundaries).
    fn conversation(
        pool: &mut PagePool,
        index: &mut PrefixIndex,
        root: ChainHash,
        turns: &[(u64, std::ops::Range<u32>, &[u64])],
        next_id: &mut u64,
    ) -> Vec<u64> {
        let mut text: Vec<u64> = Vec::new();
        for (request, tokens, checkpoints) in turns {
            text.extend(tokens.clone().map(u64::from));
            let (seq, chain) = run(pool, index, root, &text);
            for position in *checkpoints {
                *next_id += 1;
                let hash = chain.hashes()[(*position / u64::from(PER_PAGE)) as usize - 1];
                index.attach_checkpoint(hash, CheckpointId(*next_id), Owner(*request)).unwrap();
            }
            seq.release(pool);
        }
        text
    }

    fn positions(index: &PrefixIndex, p: &CheckpointPolicy) -> Vec<(u64, bool, u64)> {
        index.checkpoint_ranks(p).iter().map(|r| (r.position, r.aged, r.rows_saved)).collect()
    }

    #[test]
    fn old_turns_go_first_and_among_them_the_ones_that_save_the_fewest_rows() {
        let p = small();
        let mut pool = PagePool::new(64, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let mut id = 0;
        // Three turns of one conversation. Request 1 leaves checkpoints at 8
        // and 12, request 2 at 24 and 28, request 3 at 40 and 48.
        conversation(
            &mut pool,
            &mut index,
            root,
            &[(1, 0..14, &[8, 12]), (2, 14..30, &[24, 28]), (3, 30..50, &[40, 48])],
            &mut id,
        );
        assert_eq!(index.checkpoints(), 6);
        // Request 3's are fresh, both: the checkpoints of one request are no
        // step apart. The others have a later request below them. Among the
        // aged, the least distance to the checkpoint below goes first, and
        // among the fresh too.
        assert_eq!(
            positions(&index, &p),
            [
                (12, true, 4),
                (28, true, 4),
                (8, true, 8),
                (24, true, 12),
                (48, false, 8),
                (40, false, 12),
            ]
        );
        // With one later request allowed, request 2's stay fresh as well when
        // more than three newer checkpoints do not age them.
        let lenient = CheckpointPolicy { fresh_requests: 1, fresh_checkpoints: 5, ..p };
        let aged: Vec<u64> = index
            .checkpoint_ranks(&lenient)
            .iter()
            .filter(|r| r.aged)
            .map(|r| r.position)
            .collect();
        assert_eq!(aged, [12, 8]);

        // A budget of 3: three go, and the distances of the ones that stay
        // grow as their neighbours go.
        let tight = CheckpointPolicy { budget_bytes: 3, ..p };
        let dropped = index.make_room_for_checkpoints(&tight, 0);
        assert_eq!(dropped, [CheckpointId(2), CheckpointId(4), CheckpointId(1)]);
        assert_eq!(positions(&index, &p), [(24, true, 24), (48, false, 8), (40, false, 16)]);
        // Room for one more: the last aged one goes.
        assert_eq!(index.make_room_for_checkpoints(&tight, 1), [CheckpointId(3)]);
        assert_eq!(index.checkpoints(), 2);
        // No K/V page went with them.
        assert_eq!(index.len(), 12);
        // More coming in than the budget holds: all go.
        assert_eq!(index.make_room_for_checkpoints(&tight, 5).len(), 2);
        assert_eq!(index.checkpoints(), 0);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 64);
    }

    #[test]
    fn a_conversation_keeps_its_newest_three_and_its_roots_longest() {
        let p = small();
        let mut pool = PagePool::new(64, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let mut id = 0;
        // One request leaves checkpoints at 8 (its root), 12, 20, 24, 32, 36.
        conversation(&mut pool, &mut index, root, &[(1, 0..40, &[8, 12, 20, 24, 32, 36])], &mut id);
        let hashes = index.lookup(root, &(0..40).collect::<Vec<u64>>(), PER_PAGE).hashes;
        assert!(index.mark_root(hashes[1]));
        assert!(!index.mark_root(hashes[0]), "no checkpoint there");
        // The newest three are fresh; the root never ages and goes last.
        assert_eq!(
            positions(&index, &p),
            [
                (12, true, 4),
                (20, true, 8),
                (24, false, 4),
                (36, false, 4),
                (32, false, 8),
                (8, false, 8)
            ]
        );
        // Room for three: the aged go first (20 is aged once 12 is gone),
        // then the one that saves the fewest rows; the root stays.
        let tight = CheckpointPolicy { budget_bytes: 3, ..p };
        index.make_room_for_checkpoints(&tight, 0);
        let left: Vec<u64> = index.checkpoint_ranks(&p).iter().map(|r| r.position).collect();
        assert_eq!(left, [32, 24, 8]);
        // A later turn leaves the root behind on its branch: it is still not
        // aged, and not left behind for a level without tails.
        let text: Vec<u64> = (0..48).collect();
        let (seq, chain) = run(&mut pool, &mut index, root, &text);
        index.attach_checkpoint(chain.hashes()[10], CheckpointId(90), Owner(2)).unwrap();
        seq.release(&mut pool);
        let ranks = index.checkpoint_ranks(&p);
        assert!(ranks.iter().any(|r| r.position == 8 && r.root && !r.aged), "{ranks:?}");
        assert!(!index.left_behind_on_pages(&p).contains(&hashes[1]));
        assert!(index.left_behind_on_pages(&p).contains(&chain.hashes()[7]), "32 is");
        // Dropping the checkpoint takes the mark with it.
        index.set_checkpoint(hashes[1], None).unwrap();
        assert!(!index.is_root(hashes[1]));
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 64);
    }

    #[test]
    fn a_page_checkpoint_is_left_behind_on_pages_only_by_a_deeper_page_checkpoint() {
        let p = small();
        let mut pool = PagePool::new(64, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let mut id = 0;
        // Request 1 leaves an anchor at 20; requests 2 and 3 go on in the
        // same page and leave only the state at the ends of their tails.
        let text = conversation(&mut pool, &mut index, root, &[(1, 0..22, &[20])], &mut id);
        let found = index.lookup(root, &text, PER_PAGE);
        let prior = *found.hashes.last().unwrap();
        for (request, len) in [(2u64, 22u32), (3, 23)] {
            let mut seq = Sequence::from_shared(&mut pool, &found.pages, 20);
            seq.append(&mut pool, u64::from(len - 20)).unwrap();
            let tokens: Vec<u64> = (20..u64::from(len)).collect();
            index.insert_tail(&mut pool, prior, &tokens, seq.pages()[5], Priority::NORMAL);
            let tail = PrefixIndex::tail_hash(prior, &tokens);
            index.attach_checkpoint(tail, CheckpointId(60 + request), Owner(request)).unwrap();
            seq.release(&mut pool);
        }
        // Among all checkpoints the anchor is aged; among those at page
        // boundaries, which a level without tails keeps, it is the deepest.
        let anchor = index.checkpoint_ranks(&p).into_iter().find(|r| r.position == 20).unwrap();
        assert!(anchor.aged);
        assert!(index.left_behind_on_pages(&p).is_empty());
        // A later request's checkpoint at a deeper page boundary leaves it
        // behind there too.
        let mut longer = text.clone();
        longer.extend(300..310);
        let (seq, chain) = run(&mut pool, &mut index, root, &longer);
        index.attach_checkpoint(chain.hashes()[6], CheckpointId(70), Owner(4)).unwrap();
        seq.release(&mut pool);
        assert_eq!(index.left_behind_on_pages(&p), [chain.hashes()[4]]);
        index.clear(&mut pool);
    }

    #[test]
    fn a_checkpoint_where_texts_branch_keeps_its_rank() {
        let p = small();
        let mut pool = PagePool::new(64, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let mut id = 0;
        // A shared document with a checkpoint at 8; two requests continue it
        // differently and leave checkpoints of their own.
        conversation(&mut pool, &mut index, root, &[(1, 0..9, &[8])], &mut id);
        conversation(&mut pool, &mut index, root, &[(2, 0..8, &[]), (2, 100..110, &[16])], &mut id);
        assert_eq!(positions(&index, &p), [(8, true, 8), (16, false, 8)], "one branch: it ages");
        conversation(&mut pool, &mut index, root, &[(3, 0..8, &[]), (3, 200..210, &[16])], &mut id);
        // Two branches below 8: it was resumed from twice and is not aged,
        // though both branches are later requests.
        let ranks = index.checkpoint_ranks(&p);
        assert!(ranks.iter().all(|r| !r.aged), "{ranks:?}");
        // All three save 8 rows, so the least recently used goes: the first
        // branch's. After that the checkpoint at 8 has one branch below it
        // again, and is aged again.
        let tight = CheckpointPolicy { budget_bytes: 2, ..p };
        assert_eq!(index.make_room_for_checkpoints(&tight, 0), [CheckpointId(2)]);
        assert_eq!(positions(&index, &p), [(8, true, 8), (16, false, 8)]);
        index.clear(&mut pool);
    }

    #[test]
    fn an_end_state_right_after_another_checkpoint_goes_before_one_that_stands_alone() {
        let p = small();
        let mut pool = PagePool::new(64, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let mut id = 0;
        // One request: an anchor at 20 and its end state at a tail, 2 cells
        // later. Another conversation has one checkpoint at 24.
        let text = conversation(&mut pool, &mut index, root, &[(1, 0..22, &[20])], &mut id);
        let found = index.lookup(root, &text, PER_PAGE);
        let mut seq = Sequence::from_shared(&mut pool, &found.pages, 20);
        seq.append(&mut pool, 2).unwrap();
        let prior = *found.hashes.last().unwrap();
        index.insert_tail(&mut pool, prior, &text[20..], seq.pages()[5], Priority::NORMAL);
        let tail = PrefixIndex::tail_hash(prior, &text[20..]);
        index.attach_checkpoint(tail, CheckpointId(50), Owner(1)).unwrap();
        seq.release(&mut pool);
        let other = ChainHash::root(b"other");
        conversation(&mut pool, &mut index, other, &[(2, 0..25, &[24])], &mut id);

        // All three are fresh. The end state saves 2 rows, the anchor 20, the
        // other conversation's 24.
        assert_eq!(positions(&index, &p), [(22, false, 2), (20, false, 20), (24, false, 24)]);
        let tight = CheckpointPolicy { budget_bytes: 2, ..p };
        assert_eq!(index.make_room_for_checkpoints(&tight, 0), [CheckpointId(50)]);

        // The next turn continues the text past the tail and fills its page:
        // the tail's checkpoint, attached again, now has a later request's
        // checkpoint below it and is aged, the anchor with it.
        index.attach_checkpoint(tail, CheckpointId(51), Owner(1)).unwrap();
        let mut longer = text.clone();
        longer.extend(300..310);
        let (seq, chain) = run(&mut pool, &mut index, root, &longer);
        index.attach_checkpoint(chain.hashes()[6], CheckpointId(52), Owner(3)).unwrap();
        seq.release(&mut pool);
        assert_eq!(
            positions(&index, &p),
            [(22, true, 2), (20, true, 20), (28, false, 6), (24, false, 24)]
        );
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 64);
    }
}
