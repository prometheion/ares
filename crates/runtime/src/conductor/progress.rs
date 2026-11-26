//! How far the prompt of a request has got: the progress events a request
//! gets while its prompt is read ([`Event::Progress`]), in the terms of
//! llama.cpp's `prompt_progress`, and when it sends them (after
//! `send_partial_response` and its callers in `tools/server/server-context.cpp`
//! of <https://github.com/ggml-org/llama.cpp>, commit d92ea9b, MIT; see
//! THIRD-PARTY-NOTICES).
//!
//! A request hears where its prompt begins when it first starts (the tokens the
//! cache gave are done), then after every step that ran rows of its prompt, up
//! to the step that ends it, whose progress comes before its first text; never
//! after that. What is left is counted in the rows the engine still runs for the
//! prompt, the replayed rows included (their cells came from the cache, but the
//! recurrent state is rebuilt over them), and spread over the tokens the cache did
//! not give, so that the count rises from the cached tokens to all of them as the
//! work is done. While a long step runs, a model that reads it in pieces says
//! how far it has got ([`Executor::step_share`]), and the conductor reports that
//! share of the step's rows as done every [`LOOK`] (`run_on_devices`): a prompt
//! read layer by layer as one pass moves through the pass, where a count of rows
//! done would stand still until its end.

use super::*;

/// How often the conductor tells a request whose prompt runs in a long step how
/// far the step has got.
pub(super) const LOOK: Duration = Duration::from_secs(1);

/// What a request's progress events said, and where the count of its work began.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PromptProgress {
    /// The tokens done that the last event said.
    sent: u32,
    /// At the request's last start: the tokens done, the rows of the prompt that
    /// were left to run (replayed and computed), and the rows it had run before.
    base: u32,
    work: u32,
    ran: u32,
    /// The first event went out.
    begun: bool,
    /// The first token went out: no event follows.
    over: bool,
    /// The rows replayed when the first token came, `None` before.
    replayed: Option<u32>,
}

impl PromptProgress {
    /// The first token was taken, with `replayed` rows replayed so far.
    pub(super) fn read(&mut self, replayed: u32) {
        self.replayed.get_or_insert(replayed);
    }

    /// The rows replayed for the prompt: up to the first token, or `replayed`
    /// (all so far) for a request that ended before it.
    pub(super) fn replayed(&self, replayed: u32) -> u32 {
        self.replayed.unwrap_or(replayed)
    }
}

impl Active {
    /// The request has started (again): the work of its prompt is counted from
    /// here, and at its first start it hears where its prompt begins.
    pub(super) fn progress_starts(&mut self) {
        if self.progress.over {
            return;
        }
        if !self.progress.begun {
            self.progress.sent = self.cached_prompt;
        }
        self.progress.base = self.progress.sent;
        self.progress.work = self.prefill_left();
        self.progress.ran = self.replayed + self.computed;
        if !self.progress.begun {
            self.progress.begun = true;
            let _ = self.sink.send(Event::Progress(self.progress_now(self.progress.sent)));
        }
    }

    /// Whether the request still hears how far its prompt has got.
    pub(super) fn reports_progress(&self) -> bool {
        self.progress.begun && !self.progress.over && self.completion == 0
    }

    /// Tells the request how far its prompt has got, with `in_flight` rows of the
    /// step that runs now done, if it is further than the last event said.
    /// Returns `false` when the sink refused the event: nobody wants the rest.
    pub(super) fn report_progress(&mut self, in_flight: f64) -> bool {
        if !self.progress.begun || self.progress.over {
            return true;
        }
        let total = self.prompt_len;
        let p = self.progress;
        let processed = if self.prefill_left() == 0 && in_flight == 0.0 {
            total
        } else {
            let done = f64::from((self.replayed + self.computed).saturating_sub(p.ran)) + in_flight;
            let share = if p.work == 0 { 1.0 } else { (done / f64::from(p.work)).min(1.0) };
            let counted = p.base + (f64::from(total.saturating_sub(p.base)) * share).round() as u32;
            // The last token is done when the prompt is read, not before.
            counted.min(total.saturating_sub(1))
        };
        if self.completion > 0 {
            self.progress.over = true;
        }
        if processed <= p.sent {
            return true;
        }
        self.progress.sent = processed;
        self.sink.send(Event::Progress(self.progress_now(processed)))
    }

    fn progress_now(&self, processed: u32) -> ares_api::Progress {
        ares_api::Progress {
            total: self.prompt_len,
            cached: self.cached_prompt,
            processed,
            elapsed: self.stop.since_start().unwrap_or_default(),
        }
    }
}

impl Conductor {
    /// The request of the step in flight whose prompt is the step's only part and
    /// still hears how far it has got, with the part's rows: whose progress the
    /// model's share of the step tells.
    pub(super) fn reporting_step(&self) -> Option<(u32, u32)> {
        let [part] = self.parts.as_slice() else { return None };
        if !matches!(part.kind, PartKind::Prefill { .. }) {
            return None;
        }
        let a = self.slots[part.slot as usize].as_ref()?;
        (a.seq == part.seq && a.reports_progress()).then_some((part.slot, part.rows))
    }

    /// Tells the request of the step in flight how far the step has got, by the
    /// model's share of it, if the model says.
    pub(super) fn report_step_share(&mut self, slot: u32, rows: u32) {
        let Some(share) = self.exec.step_share() else { return };
        if let Some(a) = self.slots[slot as usize].as_mut() {
            // A sink that refuses is found when the step's events go out.
            let _ = a.report_progress(share.clamp(0.0, 1.0) * f64::from(rows));
        }
    }
}
