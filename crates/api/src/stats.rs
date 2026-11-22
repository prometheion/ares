//! What an engine has done so far, for whoever monitors it.

use serde::{Deserialize, Serialize};

/// What a model whose experts do not all fit in VRAM counted since the
/// engine started: how the experts the routers picked were found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertStats {
    /// Experts the routers picked (one per layer and distinct expert of a
    /// step), and how many of them were in a VRAM slot when the router's
    /// record reached the host.
    pub picks: u64,
    pub hits: u64,
    /// The misses, by where the expert's bytes were: in the RAM tier, only in
    /// the file, or already on their way.
    pub from_ram: u64,
    pub from_disk: u64,
    pub on_their_way: u64,
    /// Experts the CPU computed instead of the device, and experts uploaded
    /// to a VRAM slot.
    pub cpu_experts: u64,
    pub uploads: u64,
    /// Bytes read from the model file for experts.
    pub read_bytes: u64,
    /// Microseconds layers waited for what they missed, summed over layers.
    pub wait_us: u64,
    /// The read-ahead of predicted picks: experts the routers of later layers, applied
    /// early, named for a layer before its own router ran, and of them those it picked;
    /// reads of predicted experts that no tier held, started before the layer's router,
    /// and their bytes; of those, the experts the router picked and those it did not.
    pub predicted: u64,
    pub predicted_picked: u64,
    pub ahead_reads: u64,
    pub ahead_read_bytes: u64,
    pub ahead_used: u64,
    pub ahead_wasted: u64,
    /// Bytes of page-locked RAM the tier of experts holds.
    pub ram_tier_bytes: u64,
    /// Experts in VRAM slots now, and in the RAM tier (copies of experts that
    /// VRAM holds among them), over the model's layers.
    pub vram_experts: u64,
    pub ram_experts: u64,
}

/// What the level of the prompt cache on SSD has done since the engine
/// started, and what it holds; all zero for an engine without one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SsdStats {
    /// Pages and checkpoints written, the bytes they take on the drive, and
    /// the nanoseconds the writer spent writing and syncing them.
    pub pages_written: u64,
    pub checkpoints_written: u64,
    pub bytes_written: u64,
    pub write_ns: u64,
    /// Pages and checkpoints read back, their bytes, and the nanoseconds the
    /// readers spent on them, summed over the readers.
    pub pages_read: u64,
    pub checkpoints_read: u64,
    pub bytes_read: u64,
    pub read_ns: u64,
    /// Nanoseconds the engine's step loop waited while state moved between
    /// the devices and host memory.
    pub transfer_ns: u64,
    /// Bytes the model's store takes on the drive now.
    pub store_bytes: u64,
    /// Pages and checkpoints that wait to be written or are being written
    /// now.
    pub pending: u64,
    /// Checkpoints that were due on the drive and were not written: the
    /// writer held every buffer when one had to leave its place, or the
    /// checkpoint was given up (host memory ran short) before the step that
    /// saves it had run.
    pub checkpoints_skipped: u64,
}

/// What the level of the prompt cache in RAM has done since the engine
/// started, and what it holds; all zero for an engine without one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamCacheStats {
    /// Pages it holds now, and the bytes of their page-locked buffers (those
    /// that hold nothing at the moment included).
    pub pages: u64,
    pub bytes: u64,
    /// Pages that left the devices and were copied into it.
    pub pages_in: u64,
    /// Pages a prompt took back to the devices from it.
    pub pages_read: u64,
    /// Pages given up without being written although they were worth
    /// keeping: no room in RAM, or no level on SSD below it.
    pub pages_dropped: u64,
    /// Nanoseconds the engine's step loop waited while pages moved between
    /// the devices and RAM.
    pub transfer_ns: u64,
}

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
    /// Times a running request gave way to a request of a better priority
    /// (it comes back when there is room).
    pub requests_outranked: u64,
    /// Requests in a full line refused as overloaded because a request of a
    /// better priority came.
    pub requests_displaced: u64,
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
    /// Bytes of page-locked RAM held for checkpoints of recurrent state (the
    /// pages of the prompt cache in RAM are in `ram`).
    pub checkpoint_ram: u64,
    /// How the experts were found, for a model that streams them.
    pub experts: ExpertStats,
    /// What the prompt cache did in RAM.
    #[serde(default)]
    pub ram: RamCacheStats,
    /// What the prompt cache did on SSD.
    #[serde(default)]
    pub ssd: SsdStats,
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
