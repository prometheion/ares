//! What is decided before a model is loaded: how the weights are shared out
//! among the devices and how much sequence state the memory they report
//! holds.
//!
//! Everything here is arithmetic over numbers the devices and the model file
//! give. Nothing touches a device, so the rules can be tested without one.

use std::path::PathBuf;

use ares_cuda::{CacheFormat, Precision};
use ares_tokenizer::Json;

/// The cells of a page of sequence state: what the attention kernels store
/// per page-table entry.
pub const CELLS_PER_PAGE: u32 = ares_cuda::ENTRY_CELLS;

/// The most draft tokens a sequence verifies in a step when the operator
/// names no number. The conductor picks the length of every draft from the
/// acceptance it measures, so this is a cap, and a cap of two to six made no
/// difference where it was measured (Qwen3.8-27B with its own draft block).
pub const DEFAULT_DRAFTS: u32 = 3;

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

/// What the operator asked for. `None` means "decide from the devices and
/// the model".
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    /// The GGUF file (any part of a split model).
    pub model: PathBuf,
    /// The devices to use, by the driver's index.
    pub devices: Option<Vec<u32>>,
    pub split: Option<SplitKind>,
    /// Each device's share of the weights under a split.
    pub shares: Option<Vec<f64>>,
    /// Tokens of sequence state the engine holds for all sequences together.
    pub context: Option<u32>,
    /// Sequences that run at once.
    pub parallel: Option<u32>,
    /// The storage formats of K and of V.
    pub kv: Option<(CacheFormat, CacheFormat)>,
    /// How the matrix products round.
    pub precision: Precision,
    /// The most draft tokens a sequence verifies in a step; 0 for none.
    pub drafts: Option<u32>,
    /// A file with the token ids the model's draft block may propose.
    pub draft_vocabulary: Option<PathBuf>,
    /// Variables of the chat template.
    pub chat_variables: Vec<(String, Json)>,
}

impl LoadOptions {
    /// Everything left to the devices and the model.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            devices: None,
            split: None,
            shares: None,
            context: None,
            parallel: None,
            kv: None,
            precision: Precision::Fast,
            drafts: None,
            draft_vocabulary: None,
            chat_variables: Vec::new(),
        }
    }
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

/// The context served when the operator names none: what fits, at most what
/// `parallel` sequences of the model's own context could use, rounded down
/// to whole thousands of tokens (1024) so that the estimate behind it has a
/// little slack.
pub fn default_context(pages_fit: u32, parallel: u32, model_context: u32) -> u32 {
    let fits = pool_tokens(pages_fit, parallel);
    let useful = model_context.saturating_mul(parallel);
    let context = fits.min(useful);
    if context >= 2048 { context / 1024 * 1024 } else { context }
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
    fn a_pool_holds_what_it_was_sized_for() {
        for (tokens, parallel) in [(1, 1), (128, 1), (129, 2), (4096, 2), (100_000, 4)] {
            let pages = pool_pages(tokens, parallel);
            assert!(pool_tokens(pages, parallel) >= tokens, "{tokens} tokens, {parallel}");
            assert!(pool_tokens(pages - 1, parallel) < tokens + CELLS_PER_PAGE);
        }
        assert_eq!(pool_tokens(1, 2), 0);
    }

    #[test]
    fn the_default_context_is_what_fits_and_no_more_than_is_useful() {
        // 700 pages for two sequences: 698 * 128 = 89344 tokens, rounded down.
        assert_eq!(default_context(700, 2, 262_144), 89_088);
        // More room than two sequences of the model's context can use.
        assert_eq!(default_context(10_000, 2, 4096), 8192);
        // Too little to round.
        assert_eq!(default_context(10, 2, 262_144), 1024);
        assert_eq!(default_context(5, 2, 262_144), 384);
        assert_eq!(default_context(1, 2, 262_144), 0);
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
