//! The image tower: the passes that compute an image's rows before its prompt
//! reads them, and the rows a step stages.

use super::*;

impl Conductor {
    /// Runs the image tower for the prompts that wait for an image's rows:
    /// the pass under way, or a new one for the first such image of the
    /// shortest of them. While sequences generate it runs as many units as
    /// fit the time of a piece (by what the units took before), otherwise the
    /// rest of the pass. A pass that ends leaves its rows in the store.
    /// Returns what the tower took, in milliseconds (0 when it had nothing to
    /// do).
    pub(super) fn run_tower(&mut self, generating: u32) -> Result<f64, String> {
        let Some(vision) = self.vision else { return Ok(0.0) };
        if self.pass.is_none() {
            let mut waiting: Vec<&Active> =
                self.slots.iter().flatten().filter(|a| !a.generating()).collect();
            waiting.sort_by_key(|a| a.prefill_left());
            // From where the state is; for the paused step's sequence after its
            // rows, whose images it holds.
            let paused = self.paused;
            let from = |a: &Active| match paused {
                Some(p) if p.seq == a.seq => (p.from + u64::from(p.rows)) as u32,
                _ => a.state_pos() as u32,
            };
            let Some(image) = waiting.iter().find_map(|a| a.missing_image(&self.images, from(a)))
            else {
                return Ok(0.0);
            };
            self.pass = Some(TowerPass { image, next: 0, rows: Vec::new(), ms: 0.0 });
        }
        let pass = self.pass.as_mut().expect("a pass");
        let left = vision.units.saturating_sub(pass.next).max(1);
        let rows = f64::from(pass.image.rows().max(1));
        let units = if generating == 0 {
            left
        } else {
            // What a unit takes: of this pass so far, or by the rows of the
            // last one; four units (the stem and three blocks) to learn it.
            let per_unit = if pass.next > 0 {
                Some(pass.ms / f64::from(pass.next))
            } else {
                self.unit_ms_per_row.map(|per_row| per_row * rows)
            };
            match per_unit.filter(|ms| *ms > 0.0) {
                Some(ms) => ((self.config.piece_ms / ms) as u32).clamp(1, left),
                None => left.min(4),
            }
        };
        let mut job = TowerJob {
            image: pass.image.clone(),
            units: pass.next..pass.next + units,
            rows: std::mem::take(&mut pass.rows),
            ms: 0.0,
        };
        // The tower's job is a wait on a device like a step's: the watchdog sees it.
        self.heartbeat.begin();
        let ran = self.exec.tower(&mut job);
        self.heartbeat.end();
        ran.map_err(|error| error.to_string())?;
        pass.next += units;
        pass.ms += job.ms;
        self.unit_ms_per_row = Some(pass.ms / f64::from(pass.next) / rows);
        let counters = &mut self.stats.counters;
        counters.tower_units += u64::from(units);
        counters.tower_us += (job.ms * 1e3) as u64;
        if pass.next >= vision.units {
            let image = pass.image.clone();
            self.pass = None;
            let rows = ImageRows { values: job.rows };
            if rows.values.len() != (image.rows() * vision.width) as usize {
                return Err(format!(
                    "the image tower gave {} values for an image of {} rows of {}",
                    rows.values.len(),
                    image.rows(),
                    vision.width
                ));
            }
            self.images.put(image.name, Arc::new(rows));
            self.stats.counters.images_computed += 1;
        } else {
            pass.rows = job.rows;
        }
        Ok(job.ms)
    }

    /// Copies the rows of the step's image runs into the image stage, in the
    /// order of the runs.
    pub(super) fn fill_stage(&mut self) {
        if self.staged.is_empty() {
            return;
        }
        let width = self.vision.map_or(0, |v| v.width) as usize;
        let stage = self.exec.image_stage();
        let mut at = 0;
        for (rows, first, count) in self.staged.drain(..) {
            let values = &rows.values[first as usize * width..(first + count) as usize * width];
            stage[at..at + values.len()].copy_from_slice(values);
            at += values.len();
        }
    }

    /// Holds the rows of the images among `part`'s rows (the step the model paused)
    /// until the step has run to its end: those it held when it paused before (the store
    /// may have let them go since), else the store's.
    pub(super) fn hold_images(&mut self, part: StepPart) {
        let held = std::mem::take(&mut self.paused_images);
        let Some(a) = &self.slots[part.slot as usize] else { return };
        let (from, end) = (part.from as u32, part.from as u32 + part.rows);
        let Some(first) = a.spans.from(from) else { return };
        for span in a.spans.spans()[first..].iter().take_while(|s| s.first < end) {
            let name = span.image.name;
            if self.paused_images.iter().any(|(n, _)| *n == name) {
                continue;
            }
            let rows = match held.iter().find(|(n, _)| *n == name) {
                Some((_, rows)) => Some(rows.clone()),
                None => self.images.get(&name),
            };
            if let Some(rows) = rows {
                self.paused_images.push((name, rows));
            }
        }
    }
}
