//! What is decided before a model is loaded: how the weights are shared out
//! among the devices and how much sequence state the memory they report
//! holds.
//!
//! Everything here is arithmetic over numbers the devices and the model file
//! give. Nothing touches a device, so the rules can be tested without one.

use std::path::PathBuf;

use ares_core::units::Bytes;
use ares_cuda::{CacheFormat, Precision, PruneRule};
use ares_tokenizer::Json;

/// The cells of a page of sequence state: what the attention kernels store
/// per page-table entry.
pub const CELLS_PER_PAGE: u32 = ares_cuda::ENTRY_CELLS;

/// The most rows one sequence's window has when the operator names no number:
/// the rows one launch of the few-row weight kernels takes. Up to that many
/// rows a step reads the weights once; a ninth row is a second launch of
/// them, and the price of that is measured on the devices at load and decides
/// whether a longer window pays (it did not in the measurements of the model's
/// own MTP block). Rows of windows are VRAM, which is why the room for them
/// is a choice: each takes the inputs the model keeps for the commit, its
/// logits and a link of the MTP block's chain, about 5 MiB of the 27B
/// together. The window is not how long a draft is: that is chosen for every
/// step from the measured price and acceptance.
pub const DEFAULT_WINDOW: u32 = ares_cuda::MAX_BATCH;

/// The sequences that run at once when the operator names no number: one
/// conversation and one request next to it. Every sequence takes memory for
/// its recurrent state on the devices; further requests wait in line.
pub const DEFAULT_PARALLEL: u32 = 2;

/// A single device is chosen for a model when it holds the weights and at
/// least this much context (or the model's own context, if that is less).
/// Below that a split over more devices is the better default.
pub const SINGLE_DEVICE_MIN_CONTEXT: u32 = 8192;

/// How a model is spread over the devices, as an operator names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitKind {
    /// Everything on one device.
    Single,
    /// Every device holds a slice of every block.
    Tensor,
    /// Every device holds a range of blocks.
    Layer,
}

impl SplitKind {
    /// The name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            SplitKind::Single => "single",
            SplitKind::Tensor => "tensor",
            SplitKind::Layer => "layer",
        }
    }
}

/// A model drafter, as the operator names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drafter {
    /// `mtp`: the MTP block (multi-token prediction) in the model file.
    Mtp,
    /// A drafter from a file of its own: a DFlash2 drafter (architecture
    /// `dflash`), or the MTP block of a model whose file carries none.
    File(PathBuf),
}

/// What a mixture of experts does with a pick whose expert only the SSD holds
/// (`--expert-prune RULE:THRESHOLD`): `rule` for the picks from the
/// `from_rank`-th of a row on (1 the strongest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPrune {
    pub rule: PruneRule,
    pub from_rank: u32,
}

/// The most bytes the prompt cache takes on SSD when `--cache-size` does not
/// say: room for the state of about a million tokens of the largest model the
/// engine runs today, and little enough to be no surprise on a drive whose
/// size the operator did not state.
pub const DEFAULT_CACHE_SIZE: u64 = 32 << 30;

/// What the operator asked for. `None` means "decide from the devices and
/// the model".
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    /// The GGUF file (any part of a split model).
    pub model: PathBuf,
    /// The id the model is served under: the name of its section in a preset
    /// file. `None`: the file's name without its extension.
    pub name: Option<String>,
    /// The devices to use, by the driver's index.
    pub devices: Option<Vec<u32>>,
    pub split: Option<SplitKind>,
    /// Each device's share of the weights under a split.
    pub shares: Option<Vec<f64>>,
    /// Where a tensor split puts the KV heads of the attention blocks, with
    /// their K and V and their attention work; the rest of the model follows
    /// `shares`. The engine's way is [`AttentionShares::ByContext`], which
    /// `--context` steers (the owner, 2026-11-22: no option of its own);
    /// shares given are for tests and measurements.
    pub attention: AttentionShares,
    /// Tokens of sequence state the engine holds for all sequences together.
    pub context: Option<u32>,
    /// Sequences that run at once.
    pub parallel: Option<u32>,
    /// The storage formats of K and of V.
    pub kv: Option<(CacheFormat, CacheFormat)>,
    /// The most host RAM the engine may use, in bytes.
    pub ram: Option<u64>,
    /// The directory in which the prompt cache keeps state on SSD. `None`:
    /// the cache lives in the devices and in RAM alone.
    pub cache: Option<PathBuf>,
    /// The most bytes everything in that directory may take
    /// ([`DEFAULT_CACHE_SIZE`]).
    pub cache_size: Option<u64>,
    /// How the matrix products round.
    pub precision: Precision,
    /// The model drafters that take part. `None`: every one that is
    /// available (the model's MTP block, if the file has one). An empty
    /// list: none, and with none the engine does not speculate at all. The
    /// n-gram drafter takes part with them and never alone.
    pub drafters: Option<Vec<Drafter>>,
    /// The most rows of one sequence's window ([`DEFAULT_WINDOW`]).
    pub window: Option<u32>,
    /// The most rows of the step a prompt read alone takes (llama.cpp's
    /// logical batch): above `ubatch` only for a mixture of experts, whose
    /// expert slots lend the VRAM of such a step for its time. `None`: the
    /// model's default.
    pub batch: Option<u32>,
    /// The most rows of a step in the memory the model keeps for good
    /// (llama.cpp's physical batch), and of a piece of prompt while another
    /// request waits. `None`: what the devices' memory allows.
    pub ubatch: Option<u32>,
    /// A file with the token ids the model's MTP block may propose.
    pub draft_vocabulary: Option<PathBuf>,
    /// Variables of the chat template.
    pub chat_variables: Vec<(String, Json)>,
    /// A file with a chat template that takes the place of the model file's.
    pub chat_template: Option<PathBuf>,
    /// The most requests that may wait for a sequence while every sequence
    /// runs. `None`: a request waits until it runs.
    pub queue: Option<u32>,
    /// The rate in tokens per second every generating sequence keeps while
    /// prompts are read (`RuntimeConfig::generation_floor`). `None`: the
    /// runtime's default.
    pub generation_floor: Option<f64>,
    /// Pruning of the picks of a mixture of experts that only the SSD holds.
    /// `None`: none, the output as the model gives it.
    pub expert_prune: Option<ExpertPrune>,
    /// The projector the model reads images through (`--mmproj`): the image
    /// tower's GGUF beside the model's. `None`: the model reads no images.
    pub mmproj: Option<PathBuf>,
    /// Where the projector's weights stay between images.
    pub mmproj_weights: ProjectorWeights,
    /// The most rows (tokens) one image becomes; a larger image is shrunk to
    /// fit ([`IMAGE_MAX_TOKENS`] by default, as llama.cpp).
    pub image_max_tokens: u32,
}

/// The fewest and the most rows an image becomes: llama.cpp's limits for this
/// projector. `--image-max-tokens` lowers the most.
pub const IMAGE_MIN_TOKENS: u32 = 8;
pub const IMAGE_MAX_TOKENS: u32 = 4096;

/// Where a projector's weights stay between images (`--mmproj-weights`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProjectorWeights {
    /// In page-locked RAM, inside `--ram`, uploaded block by block for every
    /// new image (the owner's choice, 2026-11-16: VRAM is worth more as
    /// context).
    #[default]
    Ram,
    /// In VRAM, which the context gives up.
    Vram,
}

impl LoadOptions {
    /// Everything left to the devices and the model.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            name: None,
            devices: None,
            split: None,
            shares: None,
            attention: AttentionShares::ByContext,
            context: None,
            parallel: None,
            kv: None,
            ram: None,
            cache: None,
            cache_size: None,
            precision: Precision::Fast,
            drafters: None,
            window: None,
            batch: None,
            ubatch: None,
            draft_vocabulary: None,
            chat_variables: Vec::new(),
            chat_template: None,
            queue: None,
            generation_floor: None,
            expert_prune: None,
            mmproj: None,
            mmproj_weights: ProjectorWeights::Ram,
            image_max_tokens: IMAGE_MAX_TOKENS,
        }
    }
}

/// Where a tensor split puts the KV heads of the attention blocks.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionShares {
    /// Each device's share, as given.
    Given(Vec<f64>),
    /// Where the context finds room: the heads follow the weights as far as
    /// the context wanted (the one asked for, or the model's own) fits that
    /// way, and move to the devices with room as far as it needs
    /// ([`attention_for_context`]); when it cannot fit at all, the shares with
    /// which the devices hold the most ([`attention_by_room`]).
    ByContext,
}

/// Each device's share of the weights when the operator names none: in
/// proportion to the bound on its memory bandwidth that the device reports
/// (the width of its memory interface times its clock).
///
/// A generating sequence is bound by reading the weights, and the devices of
/// a tensor split meet at every sum, so each should take as long for its
/// slice as the others. The reported bound is the one fact about that speed
/// a device gives without a measurement; shares measured at load would be
/// better and are not built. A device that reports nothing gets the mean of
/// the others, and equal shares result when none reports anything.
pub fn shares_by_bandwidth(bandwidths: &[f64]) -> Vec<f64> {
    let known: Vec<f64> =
        bandwidths.iter().copied().filter(|b| b.is_finite() && *b > 0.0).collect();
    let mean = if known.is_empty() { 1.0 } else { known.iter().sum::<f64>() / known.len() as f64 };
    let filled: Vec<f64> =
        bandwidths.iter().map(|b| if b.is_finite() && *b > 0.0 { *b } else { mean }).collect();
    let total: f64 = filled.iter().sum();
    filled.iter().map(|b| b / total).collect()
}

/// Lowers the shares of devices that cannot hold theirs and gives the rest
/// to the others in proportion to what they asked for. `most[i]` is the
/// largest share device `i` has memory for. `None` when the devices together
/// cannot hold the model.
pub fn fit_shares(wanted: &[f64], most: &[f64]) -> Option<Vec<f64>> {
    let mut shares = wanted.to_vec();
    let mut full = vec![false; shares.len()];
    // Each round fills at least one device, so the loop ends.
    for _ in 0..=shares.len() {
        let mut over = 0.0;
        for i in 0..shares.len() {
            if !full[i] && shares[i] > most[i] {
                over += shares[i] - most[i];
                shares[i] = most[i].max(0.0);
                full[i] = true;
            }
        }
        if over <= 1e-9 {
            return Some(shares);
        }
        let open: f64 = (0..shares.len()).filter(|i| !full[*i]).map(|i| shares[i]).sum();
        if open <= 0.0 {
            return None;
        }
        for i in 0..shares.len() {
            if !full[i] {
                shares[i] += over * shares[i] / open;
            }
        }
    }
    None
}

/// Each device's share of the attention heads that lets the devices hold the
/// most context, from `room[i]`: the bytes device `i` has left once it holds
/// everything of the model but the attention heads, their weights and their
/// K and V.
///
/// A head's weights and its K and V take the same bytes on every device, so
/// with `a[i]` of the heads, `w` bytes of all heads' weights and `p` of all
/// heads' K and V per page, `n` pages fit when `a[i] * (w + p * n) <= room[i]`
/// on every device. The most pages fit when the heads follow the rooms. `None`
/// when no device has room.
pub fn attention_by_room(room: &[u64]) -> Option<Vec<f64>> {
    let total: u64 = room.iter().sum();
    (total > 0).then(|| room.iter().map(|r| *r as f64 / total as f64).collect())
}

/// Each device's share of the attention heads for `pages` pages of context:
/// `wanted` (the shares of the weights) lowered where a device's `room` (as
/// for [`attention_by_room`]) cannot hold its heads' weights, `heads` bytes for
/// all heads, and their K and V, `page` bytes a page for all heads, and the
/// rest given to the others ([`fit_shares`]). The heads stay with the weights
/// whenever the context fits that way: attention work on a device that is not
/// the fastest makes a long sequence slower. `None` when the devices cannot
/// hold that many pages.
pub fn attention_for_context(
    wanted: &[f64],
    room: &[u64],
    heads: u64,
    page: u64,
    pages: u32,
) -> Option<Vec<f64>> {
    let all = heads as f64 + page as f64 * f64::from(pages);
    let most: Vec<f64> = room.iter().map(|r| (*r as f64 / all.max(1.0)).min(1.0)).collect();
    fit_shares(wanted, &most)
}

/// What a device has to give and what a model takes of it, in bytes: the
/// input of [`pages_that_fit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Room {
    /// Free memory of the device, less what is left unreserved.
    pub usable: u64,
    /// What the model takes whatever the context is: weights and the scratch
    /// of a step.
    pub fixed: u64,
    /// Recurrent state of one sequence.
    pub per_sequence: u64,
    /// One page of K and V.
    pub per_page: u64,
}

/// The pages of sequence state that fit on every device next to the model
/// and the state of `parallel` sequences. 0 when a device cannot even hold
/// the model.
pub fn pages_that_fit(rooms: &[Room], parallel: u32) -> u32 {
    rooms
        .iter()
        .map(|room| {
            let taken = room.fixed + room.per_sequence * u64::from(parallel);
            match (room.usable.checked_sub(taken), room.per_page) {
                (None, _) => 0,
                // A device without attention blocks holds no pages.
                (Some(_), 0) => u32::MAX,
                (Some(left), per_page) => u32::try_from(left / per_page).unwrap_or(u32::MAX),
            }
        })
        .min()
        .unwrap_or(0)
}

/// The pages of K/V and keys that fit beside the fewest expert slots each
/// device must keep: `slot_bytes[d]` is what device `d` has for expert slots
/// with no context, `floor_bytes[d]` the slots its layers cannot do without,
/// `page_bytes[d]` a page's bytes there. Every page of K and V is slots fewer
/// for experts, and a step that misses an expert waits for RAM or the drive;
/// the owner's rule (2026-11-22) is that the context is the model's own all
/// the same, so a mixture of experts gives its slots up to that floor. A
/// device without attention blocks (0 bytes a page) does not bound the pool.
pub fn pages_beside_floors(slot_bytes: &[u64], floor_bytes: &[u64], page_bytes: &[u64]) -> u32 {
    slot_bytes
        .iter()
        .zip(floor_bytes)
        .zip(page_bytes)
        .filter(|(_, per_page)| **per_page > 0)
        .map(|((slots, floor), per_page)| {
            u32::try_from(slots.saturating_sub(*floor) / per_page).unwrap_or(u32::MAX)
        })
        .min()
        .unwrap_or(0)
}

/// The pages a pool needs to hold `tokens` tokens of state in `parallel`
/// sequences: the last page of every sequence may be partly used.
pub fn pool_pages(tokens: u32, parallel: u32) -> u32 {
    tokens.div_ceil(CELLS_PER_PAGE).saturating_add(parallel)
}

/// The tokens of state a pool of `pages` pages is sure to hold in `parallel`
/// sequences: the inverse of [`pool_pages`].
pub fn pool_tokens(pages: u32, parallel: u32) -> u32 {
    pages.saturating_sub(parallel).saturating_mul(CELLS_PER_PAGE)
}

/// The context a model is served with: what was wanted and how much of it
/// the devices hold in VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextPlan {
    /// The tokens of state for all sequences together that were wanted: the
    /// operator's `--context`, or the model's own context (its file's context
    /// length, the owner's rule of 2026-11-22) when none was named.
    pub wanted: u32,
    /// The tokens of it whose K and V lie in VRAM. Less than `wanted` only
    /// when no context was named and the model's own does not fit; the rest
    /// is to overflow into RAM, which is not built, so for now the model is
    /// served with this much.
    pub in_vram: u32,
    /// Whether the operator named the context.
    pub asked: bool,
}

impl ContextPlan {
    /// The plan for `asked` tokens (`None`: the model's own context,
    /// `model_context`) when `pages_fit` pages fit in VRAM beside the model
    /// with `parallel` sequences. A context that was asked for is all in VRAM:
    /// one that does not fit is refused as it was asked. What fits of the
    /// model's own is rounded down to whole thousands of tokens (1024) when it
    /// is not all of it, so that the estimate behind it has a little slack.
    pub fn new(asked: Option<u32>, model_context: u32, pages_fit: u32, parallel: u32) -> Self {
        match asked {
            Some(context) => Self { wanted: context, in_vram: context, asked: true },
            None => {
                let fits = pool_tokens(pages_fit, parallel);
                let in_vram = if fits >= model_context {
                    model_context
                } else if fits >= 2048 {
                    fits / 1024 * 1024
                } else {
                    fits
                };
                Self { wanted: model_context, in_vram, asked: false }
            }
        }
    }

    /// The tokens the pool is made for now: what lies in VRAM.
    pub fn context(&self) -> u32 {
        self.in_vram
    }

    /// Why the context is what it is, for the plan's line in the log.
    pub fn why(&self) -> String {
        match (self.asked, self.in_vram == self.wanted) {
            (true, _) => "as asked".into(),
            (false, true) => "the model's own context, all of it in VRAM".into(),
            (false, false) => format!(
                "the model's own context is {} tokens and the devices' VRAM holds {} of them; \
                 K and V beyond VRAM are not moved to RAM yet, so the context is what VRAM holds",
                self.wanted, self.in_vram
            ),
        }
    }
}

/// The most host RAM the engine may ever use when the operator names no
/// limit: all the machine has (`total`, its `MemTotal`). Below it, as below a
/// limit that was named, the engine holds what the machine has available less
/// a headroom and gives memory back when other programs need it
/// (`ares_runtime::Machine`). Nothing when the machine reports nothing (the
/// engine then keeps to what it needs to run).
pub fn default_ram(total: Option<u64>) -> u64 {
    total.unwrap_or(0)
}

/// What the server says when the RAM limit `ram` is less than what the engine
/// must hold to run at all. `needs` is that memory by what it is for:
/// the message names every share and the whole in bytes, the number that
/// `--ram` takes as it stands.
pub fn too_little_ram(ram: u64, needs: &[(&str, u64)]) -> String {
    let required: u64 = needs.iter().map(|(_, bytes)| bytes).sum();
    let shares: Vec<String> =
        needs.iter().map(|(what, bytes)| format!("{} for {what}", Bytes(*bytes))).collect();
    format!(
        "--ram {} is less than the {} the engine needs to run ({required} bytes): {}",
        Bytes(ram),
        Bytes(required),
        shares.join(", ")
    )
}

/// The bytes of memory available for new programs in the text of
/// `/proc/meminfo` (`MemAvailable`), or `None` when it is not there.
pub fn parse_available_ram(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find_map(|line| line.strip_prefix("MemAvailable:"))?;
    let kib: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kib.checked_mul(1024)
}

/// Parses a size on the command line: a number of bytes, or a number with a
/// binary unit (`512M`, `16G`, `1.5G`; K, M, G and T, upper or lower case,
/// with an optional `iB` or `B` after it).
pub fn parse_size(text: &str) -> Option<u64> {
    let text = text.trim();
    let unit_at = text.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(text.len());
    let (number, unit) = text.split_at(unit_at);
    let scale: u64 = match unit.to_ascii_lowercase().trim_end_matches("ib").trim_end_matches('b') {
        "" => 1,
        "k" => 1 << 10,
        "m" => 1 << 20,
        "g" => 1 << 30,
        "t" => 1 << 40,
        _ => return None,
    };
    if number.is_empty() || number.starts_with('.') || number.ends_with('.') {
        return None;
    }
    let value: f64 = number.parse().ok()?;
    let bytes = value * scale as f64;
    (bytes.is_finite() && bytes < u64::MAX as f64).then_some(bytes as u64)
}

/// Parses a K/V format name.
pub fn cache_format(name: &str) -> Option<CacheFormat> {
    CacheFormat::ALL.into_iter().find(|format| format_name(*format) == name)
}

/// The name of a K/V format on the command line and in the log.
pub fn format_name(format: CacheFormat) -> &'static str {
    match format {
        CacheFormat::F16 => "f16",
        CacheFormat::Q8 => "q8",
        CacheFormat::Q7 => "q7",
        CacheFormat::Q6 => "q6",
        CacheFormat::Q5 => "q5",
        CacheFormat::Q4 => "q4",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn sizes_are_numbers_with_a_binary_unit() {
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("4096"), Some(4096));
        assert_eq!(parse_size("512M"), Some(512 << 20));
        assert_eq!(parse_size("16G"), Some(16 << 30));
        assert_eq!(parse_size("16GiB"), Some(16 << 30));
        assert_eq!(parse_size("1.5g"), Some(3 << 29));
        assert_eq!(parse_size("2kB"), Some(2048));
        assert_eq!(parse_size(" 1T "), Some(1 << 40));
        for bad in ["", "G", "-1", "1.", ".5G", "1x", "1 000", "nan", "1GB2", "99999999999T"] {
            assert_eq!(parse_size(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_ram_default_is_half_of_what_the_machine_has_available() {
        let meminfo = "MemTotal:       32768000 kB\nMemFree:         1000000 kB\n\
                       MemAvailable:   20000000 kB\nBuffers:           10000 kB\n";
        assert_eq!(parse_available_ram(meminfo), Some(20_000_000 * 1024));
        assert_eq!(default_ram(Some(32_768_000 * 1024)), 32_768_000 * 1024);
        // A machine that says nothing: no spare RAM.
        assert_eq!(parse_available_ram("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_available_ram("MemAvailable: lots\n"), None);
        assert_eq!(default_ram(None), 0);
    }

    #[test]
    fn a_ram_limit_that_is_too_small_is_answered_with_every_share_and_the_whole_in_bytes() {
        let needs = [
            ("its step buffers", 3 << 20),
            ("the token embedding table and the rows of a step", 200 << 20),
        ];
        let said = too_little_ram(4 << 20, &needs);
        assert_eq!(
            said,
            "--ram 4.0 MiB is less than the 203.0 MiB the engine needs to run (212860928 bytes): \
             3.0 MiB for its step buffers, 200.0 MiB for the token embedding table and the rows \
             of a step"
        );
        // The number of bytes it names is a limit the option takes.
        assert_eq!(parse_size("212860928"), Some(203 << 20));
        assert_eq!(
            too_little_ram(0, &[("its step buffers", 1024)]),
            "--ram 0 B is less than the 1.0 KiB the engine needs to run (1024 bytes): 1.0 KiB \
             for its step buffers"
        );
    }

    #[test]
    fn attention_heads_follow_the_room_and_that_holds_the_most_pages() {
        // Two devices with 2 and 6 GiB left; the heads' weights take 1 GiB,
        // their K and V 1 MiB a page.
        let (w, p) = (1u64 << 30, 1u64 << 20);
        let room = [2u64 << 30, 6 << 30];
        let shares = attention_by_room(&room).unwrap();
        assert_eq!(shares, vec![0.25, 0.75]);
        let pages = |a: &[f64]| {
            (0..2)
                .map(|i| ((room[i] as f64 / a[i] - w as f64) / p as f64).floor() as u64)
                .min()
                .unwrap()
        };
        // (8 GiB - 1 GiB) / 1 MiB: every byte of both devices holds context.
        assert_eq!(pages(&shares), 7 << 10);
        for other in [[0.5, 0.5], [0.2, 0.8], [0.3, 0.7]] {
            assert!(pages(&other) < pages(&shares), "{other:?}");
        }
        assert_eq!(attention_by_room(&[0, 0]), None);
        assert_eq!(attention_by_room(&[0, 5]), Some(vec![0.0, 1.0]));
    }

    #[test]
    fn attention_heads_stay_with_the_weights_until_the_context_needs_room() {
        // Heads of 1 GiB of weights and 1 MiB of K and V a page; 2 and 6 GiB
        // of room; the weights 0.625 and 0.375.
        let (heads, page) = (1u64 << 30, 1u64 << 20);
        let (room, wanted) = ([2u64 << 30, 6 << 30], [0.625, 0.375]);
        let fit = |pages| attention_for_context(&wanted, &room, heads, page, pages);
        // 2 GiB hold 0.625 of 1 GiB + 2048 pages: the heads stay.
        assert_eq!(fit(2048), Some(wanted.to_vec()));
        // 4096 pages: the first device holds 2/5 of the heads, the rest moves.
        let moved = fit(4096).unwrap();
        assert!((moved[0] - 0.4).abs() < 1e-9 && (moved[1] - 0.6).abs() < 1e-9, "{moved:?}");
        // The most the devices hold, (8 - 1) GiB / 1 MiB, still fits; one
        // page more does not.
        let most = fit(7 << 10).unwrap();
        assert!((most[0] - 0.25).abs() < 1e-9, "{most:?}");
        assert_eq!(fit((7 << 10) + 1), None);
    }

    #[test]
    fn shares_follow_the_reported_bandwidth() {
        assert!(close(&shares_by_bandwidth(&[3.0, 1.0]), &[0.75, 0.25]));
        assert!(close(&shares_by_bandwidth(&[5.0]), &[1.0]));
        // Nothing reported: equal shares. One device silent: the mean of the others.
        assert!(close(&shares_by_bandwidth(&[0.0, 0.0]), &[0.5, 0.5]));
        assert!(close(&shares_by_bandwidth(&[4.0, 0.0, 2.0]), &[4.0 / 9.0, 3.0 / 9.0, 2.0 / 9.0]));
        assert!(close(&shares_by_bandwidth(&[f64::NAN, 2.0]), &[0.5, 0.5]));
    }

    #[test]
    fn a_device_that_is_too_small_gives_its_excess_to_the_others() {
        // Everything fits: nothing changes.
        assert!(close(&fit_shares(&[0.6, 0.4], &[0.7, 0.5]).unwrap(), &[0.6, 0.4]));
        // The first can hold half: the second takes the rest.
        assert!(close(&fit_shares(&[0.6, 0.4], &[0.5, 0.9]).unwrap(), &[0.5, 0.5]));
        // The excess is split in proportion, and may fill a second device.
        let shares = fit_shares(&[0.5, 0.3, 0.2], &[0.2, 0.4, 1.0]).unwrap();
        assert!(close(&shares, &[0.2, 0.4, 0.4]), "{shares:?}");
        assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // Together they are too small.
        assert_eq!(fit_shares(&[0.5, 0.5], &[0.3, 0.3]), None);
        assert_eq!(fit_shares(&[1.0], &[0.9]), None);
    }

    #[test]
    fn pages_are_bound_by_the_fullest_device() {
        let big = Room { usable: 1000, fixed: 100, per_sequence: 50, per_page: 10 };
        let small = Room { usable: 500, fixed: 100, per_sequence: 50, per_page: 10 };
        assert_eq!(pages_that_fit(&[big], 2), 80);
        assert_eq!(pages_that_fit(&[big, small], 2), 30);
        assert_eq!(pages_that_fit(&[big, small], 4), 20);
        // A device that cannot hold the model holds no page.
        assert_eq!(pages_that_fit(&[big, Room { usable: 120, ..small }], 1), 0);
        // A device without attention blocks does not bound the pool.
        assert_eq!(pages_that_fit(&[big, Room { per_page: 0, ..small }], 2), 80);
        assert_eq!(pages_that_fit(&[], 1), 0);
    }

    #[test]
    fn the_context_of_a_model_with_experts_leaves_the_slots_the_store_needs() {
        let gib = 1u64 << 30;
        // 19 GiB of slots on two devices, a floor of 1 GiB on each, pages of
        // 1.7 MB on one of them.
        let floors = [gib, gib];
        let pages = pages_beside_floors(&[8 * gib, 11 * gib], &floors, &[1_800_000, 0]);
        assert_eq!(pages, (7 * gib / 1_800_000) as u32);
        // The tighter device decides when both hold pages.
        assert_eq!(
            pages_beside_floors(&[8 * gib, 11 * gib], &floors, &[1_000_000, 4_000_000]),
            (10 * gib / 4_000_000) as u32
        );
        // Slots below the floor hold no page.
        assert_eq!(pages_beside_floors(&[gib / 2], &[gib], &[1_000]), 0);
        // No attention anywhere, or no device: nothing bounds, nothing fits.
        assert_eq!(pages_beside_floors(&[gib], &[0], &[0]), 0);
        assert_eq!(pages_beside_floors(&[], &[], &[]), 0);
    }

    #[test]
    fn a_pool_holds_what_it_was_sized_for() {
        for (tokens, parallel) in [(1, 1), (128, 1), (129, 2), (4096, 2), (100_000, 4)] {
            let pages = pool_pages(tokens, parallel);
            assert!(pool_tokens(pages, parallel) >= tokens, "{tokens} tokens, {parallel}");
            assert!(pool_tokens(pages - 1, parallel) < tokens + CELLS_PER_PAGE);
        }
        assert_eq!(pool_tokens(1, 2), 0);
    }

    #[test]
    fn the_default_context_is_the_models_own_or_what_of_it_fits() {
        let plan = |pages| ContextPlan::new(None, 262_144, pages, 2);
        // 700 pages for two sequences: 698 * 128 = 89344 tokens, rounded down,
        // of the model's 262144.
        assert_eq!(plan(700), ContextPlan { wanted: 262_144, in_vram: 89_088, asked: false });
        assert!(plan(700).why().contains("262144 tokens and the devices' VRAM holds 89088"));
        // Room for more than the model's own context: the model's own, all in VRAM.
        assert_eq!(plan(10_000).context(), 262_144);
        assert_eq!(plan(2050).context(), 262_144);
        assert_eq!(plan(2049).context(), 261_120);
        // Too little to round.
        assert_eq!(plan(10).context(), 1024);
        assert_eq!(plan(5).context(), 384);
        assert_eq!(plan(1).context(), 0);
        // A context that was asked for is what was asked, whatever fits.
        let asked = ContextPlan::new(Some(500_000), 262_144, 10, 2);
        assert_eq!(
            (asked.wanted, asked.in_vram, asked.why().as_str()),
            (500_000, 500_000, "as asked")
        );
    }

    #[test]
    fn format_names_go_both_ways() {
        for format in CacheFormat::ALL {
            assert_eq!(cache_format(format_name(format)), Some(format));
        }
        assert_eq!(cache_format("q3"), None);
        assert_eq!(cache_format("Q8"), None);
    }
}
