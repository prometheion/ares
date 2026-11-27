//! The warm-up: what a step costs by its rows and cells, and what a chain of
//! a model's drafter takes, measured before the first request.

use super::*;

impl Conductor {
    /// Runs steps before the first request: a few of one row, so lazily
    /// loaded kernels and first-use costs are paid, and then steps of every
    /// row count a window can have and of the sizes a prompt's piece comes
    /// in, which measure what a step costs on these devices (and, for a model
    /// with an MTP block, what each length of its chain costs). The
    /// measurements are the table that drafts and pieces are decided from;
    /// every step that runs corrects it afterwards. The steps work in slot 0
    /// on the first pages of the pool, which the first request overwrites.
    pub(crate) fn warm_up(&mut self) -> Result<(), ExecError> {
        let (begun, first_step) = (Instant::now(), self.step_no);
        for _ in 0..3 {
            self.probe(1, Probe::Prompt)?;
        }
        let window = if self.drafters.iter().any(|d| d.drafts > 0) {
            self.model.draft_rows().min(self.max_rows)
        } else {
            0
        };
        // The rows of the windows and of the rounds without drafts that the
        // sequences make up together, then the doublings up to a piece of
        // prompt. Prompts are read a piece at a time, so nothing above is run.
        let piece = self.model.prompt_rows().min(self.max_rows);
        let capacity = self.config.pages.saturating_mul(self.config.cells_per_page);
        let top = piece.min(capacity).min(self.context_length);
        let mut sizes: Vec<u32> = (1..=window.max(self.config.max_slots.min(8)).min(top)).collect();
        let mut rows = 16;
        while rows < top {
            if rows > *sizes.last().expect("one row at least") {
                sizes.push(rows);
            }
            rows *= 2;
        }
        if top > *sizes.last().expect("one row at least") {
            sizes.push(top);
        }
        // The fastest of two for the steps the policy chooses between, which
        // are short and many; the long ones are measured once. What other work
        // on the machine adds is not the price of a step, and the steps that
        // run correct the table anyway.
        for rows in sizes {
            let reps = if rows <= 24 { 2 } else { 1 };
            let shape = if rows <= window { Probe::Window } else { Probe::Prompt };
            let mut fastest = f64::INFINITY;
            for _ in 0..reps {
                fastest = fastest.min(self.probe(rows, shape)?.0);
            }
            self.prices.measured(rows, fastest);
        }
        // Each drafter of the model: steps that only draft, for windows of 2
        // rows to the longest one sequence has, whose first row is the newest
        // token and the others the drafts the drafter makes.
        for d in 0..self.drafters.len() {
            let longest =
                self.config.window_rows.min(window).min(top).min(self.drafters[d].drafts + 1);
            for links in 1..longest {
                let mut fastest = f64::INFINITY;
                for _ in 0..2 {
                    fastest = fastest.min(self.probe(links + 1, Probe::Chain(d))?.0);
                }
                self.drafting.chains[d].set(links, fastest);
            }
        }
        // The last probe left a window that nothing has committed.
        self.probe(1, Probe::Prompt)?;
        self.calibration = Some(Calibration {
            steps: (self.step_no - first_step) as u32,
            seconds: begun.elapsed().as_secs_f64(),
            step_ms: self.prices.points().collect(),
            chain_ms: self.drafting.chains.iter().map(|c| c.points().collect()).collect(),
        });
        self.stats.reset();
        self.model.measured();
        Ok(())
    }

    /// One step of `rows` rows of the sequence in slot 0, from position 0,
    /// shaped as `shape` says (a chain is a step that only drafts, for a
    /// window of `rows` rows). Returns the time of the step in milliseconds
    /// and what the model's own draft chain took of it.
    pub(super) fn probe(&mut self, rows: u32, shape: Probe) -> Result<(f64, f64), ExecError> {
        let cells = self.config.cells_per_page;
        let used = {
            let pages = rows.div_ceil(cells);
            let entries: Vec<u32> = (0..pages).map(|page| self.model.page_entry(page)).collect();
            let mut writer = RecordWriter::new(self.exec.record(), self.step_no + 1, 1, rows)
                .expect("room for the rows");
            // A window of one row is certain, as it is in a round.
            let deferred = shape != Probe::Prompt && rows > 1;
            writer.segment(0, 0, rows, !deferred).expect("room for a segment");
            if let (Probe::Chain(drafter), true) = (shape, deferred) {
                writer.drafts_only();
                writer.model_drafts(drafter as u32 + 1).expect("room for the flag");
            }
            for position in 0..rows {
                writer
                    .row(Row { slot: 0, position, token: 0, context: 0 })
                    .expect("room for a row");
            }
            if deferred {
                writer.runs(&[(0, rows)]).expect("room for a run");
                writer
                    .samplers((0..rows).map(|position| SampleParams::GREEDY.row(position)))
                    .expect("room for samplers");
            } else {
                writer.runs(&[(rows - 1, 1)]).expect("room for a run");
                writer.samplers([SampleParams::GREEDY.row(rows - 1)]).expect("room for a sampler");
            }
            // The window of the step before is committed by this one.
            let previous: Vec<Commit> = self.pending_probe.take().into_iter().collect();
            writer.commits(&previous).expect("room for a commit");
            writer
                .updates(
                    [TableUpdate { slot: 0, first: 0, count: pages }].into_iter(),
                    entries.into_iter(),
                )
                .expect("room for the page-table entries");
            writer.finish().expect("room for the header")
        };
        // A window leaves its rows for the next record to commit; a step that
        // only drafts leaves none.
        if shape == Probe::Window && rows > 1 {
            self.pending_probe = Some(Commit { row0: 0, rows, slot: 0, accepted: 1 });
        }
        // The devices waited for the warm-up and not for the host between its steps.
        self.after_idle = true;
        let ms = self.run_on_devices(used)?;
        Ok((ms, self.output.draft_ms))
    }
}
