//! How long requests took, as histograms for `/metrics`.
//!
//! Every request that ends with a reply adds one observation to each
//! histogram of its model, all measured from the hand-over to the engine
//! (the server's own wait for a model to load is not in them):
//!
//! | histogram | what |
//! | --- | --- |
//! | `ares_request_queue_seconds` | the wait in the engine's line, before the request first ran |
//! | `ares_request_prompt_seconds` | from leaving the line to the first token: the prompt |
//! | `ares_time_to_first_token_seconds` | from the hand-over to the first token |
//! | `ares_time_per_output_token_seconds` | the time of the tokens after the first, over their number |
//! | `ares_request_seconds` | from the hand-over to the end |
//!
//! The buckets are fixed and the same for every model, so that the
//! histograms of several servers add up.

use std::sync::Mutex;
use std::time::Duration;

/// The upper bounds of the buckets of the times that are whole requests or
/// parts of them, in seconds.
const SPANS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    1800.0,
];

/// The upper bounds of the buckets of the time of one token, in seconds.
const STEPS: &[f64] =
    &[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0];

/// One histogram: how many observations fell into each bucket, and their
/// sum.
#[derive(Debug)]
struct Histogram {
    name: &'static str,
    help: &'static str,
    bounds: &'static [f64],
    /// The observations at or below each bound, and above the last.
    counts: Vec<u64>,
    sum: f64,
}

impl Histogram {
    fn new(name: &'static str, help: &'static str, bounds: &'static [f64]) -> Self {
        Self { name, help, bounds, counts: vec![0; bounds.len() + 1], sum: 0.0 }
    }

    fn observe(&mut self, seconds: f64) {
        let bucket = self.bounds.iter().position(|bound| seconds <= *bound);
        self.counts[bucket.unwrap_or(self.bounds.len())] += 1;
        self.sum += seconds;
    }

    /// The histogram in the text format monitoring systems scrape: the
    /// buckets count what is at or below their bound.
    fn render(&self, text: &mut String) {
        let name = self.name;
        text.push_str(&format!("# HELP {name} {}\n# TYPE {name} histogram\n", self.help));
        let mut below = 0;
        for (bound, count) in self.bounds.iter().zip(&self.counts) {
            below += count;
            text.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {below}\n"));
        }
        let total: u64 = self.counts.iter().sum();
        text.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {total}\n"));
        text.push_str(&format!("{name}_sum {}\n{name}_count {total}\n", self.sum));
    }
}

/// What one request took, as the histograms are fed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Took {
    /// The wait in the engine's line, if the engine says.
    pub queued: Option<Duration>,
    /// From the hand-over to the first token.
    pub first: Duration,
    /// From the hand-over to the end.
    pub whole: Duration,
    /// The tokens generated.
    pub tokens: u32,
}

/// The histograms of one model.
#[derive(Debug)]
pub(crate) struct Latency(Mutex<[Histogram; 5]>);

impl Default for Latency {
    fn default() -> Self {
        Self(Mutex::new([
            Histogram::new(
                "ares_request_queue_seconds",
                "Time requests waited in the engine's line before they first ran",
                SPANS,
            ),
            Histogram::new(
                "ares_request_prompt_seconds",
                "Time from leaving the line to the first token: the prompt",
                SPANS,
            ),
            Histogram::new(
                "ares_time_to_first_token_seconds",
                "Time from the hand-over to the engine to the first token",
                SPANS,
            ),
            Histogram::new(
                "ares_time_per_output_token_seconds",
                "Time of the tokens after the first, over their number, per request",
                STEPS,
            ),
            Histogram::new(
                "ares_request_seconds",
                "Time from the hand-over to the engine to the end of the reply",
                SPANS,
            ),
        ]))
    }
}

impl Latency {
    /// Adds what a request that ended with a reply took.
    pub fn observe(&self, took: Took) {
        let mut histograms = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let [queue, prompt, first, per_token, whole] = &mut *histograms;
        if let Some(queued) = took.queued {
            queue.observe(queued.as_secs_f64());
            prompt.observe(took.first.saturating_sub(queued).as_secs_f64());
        }
        first.observe(took.first.as_secs_f64());
        if took.tokens > 1 {
            let after = took.whole.saturating_sub(took.first);
            per_token.observe(after.as_secs_f64() / f64::from(took.tokens - 1));
        }
        whole.observe(took.whole.as_secs_f64());
    }

    /// The histograms in the text format of `/metrics`.
    pub fn render(&self) -> String {
        let histograms = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let mut text = String::new();
        for histogram in histograms.iter() {
            histogram.render(&mut text);
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_one_observation_in_every_histogram() {
        let latency = Latency::default();
        latency.observe(Took {
            queued: Some(Duration::from_millis(30)),
            first: Duration::from_millis(250),
            whole: Duration::from_millis(1250),
            tokens: 101,
        });
        latency.observe(Took {
            queued: None,
            first: Duration::from_secs(2),
            whole: Duration::from_secs(2),
            tokens: 1,
        });
        let text = latency.render();
        // 30 ms waited, in the bucket of 50 ms and above.
        assert!(text.contains("ares_request_queue_seconds_bucket{le=\"0.025\"} 0\n"), "{text}");
        assert!(text.contains("ares_request_queue_seconds_bucket{le=\"0.05\"} 1\n"));
        assert!(text.contains("ares_request_queue_seconds_count 1\n"));
        // The prompt: 220 ms after the line.
        assert!(text.contains("ares_request_prompt_seconds_bucket{le=\"0.1\"} 0\n"));
        assert!(text.contains("ares_request_prompt_seconds_bucket{le=\"0.25\"} 1\n"));
        // Both have a first token; only one has tokens after it, 10 ms each.
        assert!(text.contains("ares_time_to_first_token_seconds_count 2\n"));
        assert!(text.contains("ares_time_to_first_token_seconds_bucket{le=\"+Inf\"} 2\n"));
        assert!(text.contains("ares_time_per_output_token_seconds_bucket{le=\"0.0075\"} 0\n"));
        assert!(text.contains("ares_time_per_output_token_seconds_bucket{le=\"0.01\"} 1\n"));
        assert!(text.contains("ares_time_per_output_token_seconds_count 1\n"));
        assert!(text.contains("ares_request_seconds_sum 3.25\n"), "{text}");
        assert!(text.contains("# TYPE ares_request_seconds histogram\n"));
    }
}
