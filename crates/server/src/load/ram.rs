//! Host RAM: one limit for the engine, whichever model it runs (`--ram`, or
//! all the machine has), and below it the machine's memory, which the engine
//! follows.
//!
//! What the engine needs to run at all is taken from the limit first: its
//! step buffers and what a model or the prompt cache cannot do without (the
//! image tower's weights in RAM, a token embedding table in page-locked
//! memory, one checkpoint place, the pages and checkpoints on their way to
//! the prompt cache on SSD). A limit below their sum is refused, and the
//! refusal names every share. The rest goes to the tier of experts first,
//! where the model has one, and to the prompt cache in RAM.

use std::sync::{Arc, OnceLock};

use ares_core::units::Bytes;
use ares_runtime::{Machine, Meminfo, Model, SsdConfig, required_ram};

use super::{LoadError, refuse, short};
use crate::models::Site;
use crate::plan::{self, CELLS_PER_PAGE, LoadOptions};

static MACHINE: OnceLock<Arc<Machine>> = OnceLock::new();

/// The watch over the machine's memory that the engines of this process
/// follow: the machine itself ([`Machine::system`]), unless [`follow`] named
/// another before the first model was loaded.
pub fn machine() -> Arc<Machine> {
    MACHINE.get_or_init(Machine::system).clone()
}

/// Makes the engines of this process follow `machine`. For a test or a tool
/// that gives the engine a memory of its own to follow (a machine with less
/// than this one has, say); the server never calls it. Returns false, and
/// changes nothing, once a watch is in use.
pub fn follow(machine: Arc<Machine>) -> bool {
    MACHINE.set(machine).is_ok()
}

/// What the limit leaves, beyond what the engine needs to run, is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rest {
    /// The prompt cache in RAM: a model without experts.
    Cache,
    /// The tier of experts first, then the prompt cache in RAM.
    Experts,
}

/// The engine's host RAM, decided: the limit and what is taken from it.
pub(crate) struct Budget {
    /// The most RAM the engine may use.
    pub ram: u64,
    /// Why the limit is what it is, for the log.
    why: String,
    /// What the engine page-locks to run ([`required_ram`]): its step
    /// buffers, `table` and `tower` included.
    pub required: u64,
    /// The token embedding table and the rows of a step, where they lie in
    /// page-locked RAM.
    table: u64,
    /// The image tower's weights, where they lie in page-locked RAM.
    tower: u64,
    /// One checkpoint of recurrent state over all devices.
    checkpoint: u64,
    /// What the prompt cache keeps whatever the machine has: a checkpoint
    /// place, and with a store on SSD the pages that state passes through.
    pub floor: u64,
    /// Ordinary memory for checkpoints in transit to the store on SSD and
    /// back, at their most.
    transit: u64,
    /// The prompt cache's level on SSD, when `--cache` names a directory.
    pub ssd: Option<SsdConfig>,
    /// Whether the store on SSD takes this model's state.
    durable: bool,
    rest: Rest,
}

impl Budget {
    /// The budget of `model` for `parallel` sequences at `site`, with the
    /// limit and the prompt cache's directory of `options`. Of the model's
    /// page-locked memory, `table` is its token embedding table and `tower`
    /// the image tower's weights (0 where they are not there). Refused when
    /// the limit is below what the engine needs to run.
    pub(crate) fn of(
        model: &dyn Model,
        options: &LoadOptions,
        site: &Site,
        parallel: u32,
        (table, tower): (u64, u64),
        rest: Rest,
    ) -> Result<Self, LoadError> {
        let required = required_ram(model, parallel);
        let checkpoint = model.prefix_cache().map_or(0, |cache| cache.bytes());
        // The prompt cache on SSD moves state through RAM on its way to the
        // drive and back: with a directory for it, a few pages join the
        // checkpoint place in the floor. And it copies checkpoints out of
        // their places into buffers of ordinary memory, which the engine
        // holds once they are in use: they are not page-locked, so they are
        // not in the books of page-locked memory, and they count against the
        // limit here.
        let ssd = options
            .cache
            .as_ref()
            .map(|dir| SsdConfig::new(dir, options.cache_size.unwrap_or(plan::DEFAULT_CACHE_SIZE)));
        let durable = ssd.as_ref().and_then(|_| model.durable(CELLS_PER_PAGE));
        let transit = durable.as_ref().map_or(0, SsdConfig::transit_ram);
        let floor = checkpoint + durable.as_ref().map_or(0, SsdConfig::floor_pages);
        let mut budget = Self {
            ram: 0,
            why: String::new(),
            required,
            table,
            tower,
            checkpoint,
            floor,
            transit,
            ssd,
            durable: durable.is_some(),
            rest,
        };
        let (ram, why) = ram_limit_of(options, site, &budget.needs(), budget.nothing())?;
        (budget.ram, budget.why) = (ram, why);
        Ok(budget)
    }

    /// What the rest of the limit is for.
    pub(crate) fn rest(&self) -> Rest {
        self.rest
    }

    /// What the engine may page-lock: the limit less the buffers in transit.
    pub(crate) fn lockable(&self) -> u64 {
        self.ram - self.transit
    }

    /// What the limit leaves the tier of experts and the prompt cache in RAM.
    pub(crate) fn cache_ram(&self) -> u64 {
        self.lockable() - self.required
    }

    /// The engine's own step buffers.
    fn step_buffers(&self) -> u64 {
        self.required - self.table - self.tower
    }

    /// What the engine needs to run, by what it is for.
    fn needs(&self) -> Vec<(&'static str, u64)> {
        let mut needs = vec![("its step buffers", self.step_buffers())];
        if self.tower > 0 {
            needs.push((
                "the image tower's weights, which lie in page-locked RAM (--mmproj-weights ram)",
                self.tower,
            ));
        }
        if self.table > 0 {
            needs.push((
                "the token embedding table and the rows of a step, which lie in page-locked RAM \
                 and not on a device",
                self.table,
            ));
        }
        match (self.floor, self.durable, self.rest) {
            (0, _, _) => {}
            (_, true, _) => needs.push((
                "one checkpoint of recurrent state and the pages on their way to the prompt cache \
                 on SSD or back (page-locked)",
                self.floor,
            )),
            (_, false, Rest::Cache) => needs.push((
                "one checkpoint of recurrent state, which the prompt cache in RAM keeps whatever \
                 the machine has (page-locked)",
                self.floor,
            )),
            (_, false, Rest::Experts) => needs.push((
                "one checkpoint of recurrent state, which the prompt cache in RAM keeps whatever \
                 the tier of experts holds (page-locked)",
                self.floor,
            )),
        }
        if self.transit > 0 {
            needs.push((
                "the checkpoints in transit to the prompt cache on SSD and back, at their most \
                 (ordinary memory, taken as it is first used)",
                self.transit,
            ));
        }
        needs
    }

    /// What is not spared when the machine reports no memory.
    fn nothing(&self) -> &'static str {
        match self.rest {
            Rest::Cache => "none is spared for the prompt cache in RAM",
            Rest::Experts => "the tier of experts gets nothing beyond the step buffers",
        }
    }

    /// The plan's line of the RAM in the log.
    pub(crate) fn log(&self) {
        let transit = match self.transit {
            0 => String::new(),
            bytes => format!(
                ", at most {} for checkpoints in transit between RAM and the prompt cache's \
                 store on SSD (ordinary memory, taken as it is first used)",
                Bytes(bytes)
            ),
        };
        match self.rest {
            Rest::Cache => tracing::info!(
                "plan: RAM, at most {} for the engine ({}): {} for its step buffers{}{transit}, \
                 the rest for the prompt cache in RAM, the pages that left the devices and \
                 checkpoints of recurrent state, {} each over all devices, {}; the memory is \
                 page-locked as pages leave the devices and checkpoints are taken, and the \
                 coldest are given up when the machine needs it{}",
                Bytes(self.ram),
                self.why,
                Bytes(self.step_buffers()),
                match self.table {
                    0 => String::new(),
                    bytes => format!(
                        ", {} for the token embedding table and the rows of a step (locked at \
                         load)",
                        Bytes(bytes)
                    ),
                },
                Bytes(self.checkpoint),
                match self.checkpoint {
                    0 => "none for this model".to_owned(),
                    bytes => format!("room for {}", self.cache_ram() / bytes),
                },
                match (self.floor, self.durable) {
                    (0, _) => String::new(),
                    (_, true) => format!(
                        ", down to {} that state passes through on its way to the prompt cache's \
                         store on SSD and back, a checkpoint place and pages, which are kept \
                         whatever the machine has; only what leaves RAM is written there, and \
                         what is worth keeping when the model is unloaded",
                        Bytes(self.floor)
                    ),
                    (_, false) => format!(
                        ", down to {}, one checkpoint place, which is kept whatever the machine \
                         has",
                        Bytes(self.floor)
                    ),
                }
            ),
            Rest::Experts => tracing::info!(
                "plan: RAM, at most {} for the engine ({}): {} for its step buffers{transit}, the \
                 rest for the tier of experts first, which holds the experts that have no slot in \
                 VRAM as far as the memory goes, gives pieces back when the machine needs them, \
                 the coldest experts of every layer with them, and takes them again when the \
                 memory is free; the prompt cache in RAM has what the tier leaves, for pages \
                 that left the devices and checkpoints of recurrent state of {} each over all \
                 devices{}",
                Bytes(self.ram),
                self.why,
                Bytes(self.step_buffers()),
                Bytes(self.checkpoint),
                match (self.floor, self.durable) {
                    (0, _) => String::new(),
                    (bytes, true) => format!(
                        ", and {} that state passes through on its way to the prompt cache's \
                         store on SSD and back, a checkpoint place and pages, which the cache \
                         keeps whatever the tier holds",
                        Bytes(bytes)
                    ),
                    (bytes, false) => format!(
                        ", and {} for one checkpoint place, which the cache keeps whatever the \
                         tier holds",
                        Bytes(bytes)
                    ),
                },
            ),
        }
    }
}

/// The most host RAM the engine may use: what the operator named, or all the
/// machine has. Below that limit the engine follows the machine
/// ([`Machine`]): its elastic memory is what is available less a headroom.
/// `needs` is what the engine must page-lock to run at all (its step buffers,
/// and what else a model or the prompt cache cannot do without), by what it
/// is for: a limit below their sum is refused, and the refusal names every
/// share. `nothing` says what is not spared when the machine reports no
/// memory.
fn ram_limit_of(
    options: &LoadOptions,
    site: &Site,
    needs: &[(&str, u64)],
    nothing: &str,
) -> Result<(u64, String), LoadError> {
    let required: u64 = needs.iter().map(|(_, bytes)| bytes).sum();
    let machine = machine();
    // What the engine really holds below its limit, as it stands now.
    let follows = match machine.available() {
        Some(available) => format!(
            "below it the engine holds what the machine has available, {} now, less a headroom \
             of {}, and gives memory back when other programs need it",
            Bytes(available),
            Bytes(machine.headroom())
        ),
        None => "the machine does not report what it has available, so the engine holds no \
                 more than it needs to run"
            .to_owned(),
    };
    let (ram, why_ram) = match options.ram {
        Some(ram) if site.ram_is_rest => {
            (ram, format!("what the server's limit has left next to the loaded models; {follows}"))
        }
        Some(ram) => (ram, format!("as asked; {follows}")),
        None => match Meminfo::total() {
            Some(total) => (
                total.max(required),
                format!("all the machine has; {follows}; --ram sets a lower limit"),
            ),
            None => (
                required,
                format!("the machine reports no memory, so {nothing}; --ram sets the limit"),
            ),
        },
    };
    if ram < required {
        let error = refuse(plan::too_little_ram(ram, needs));
        // What is left of the server's limit grows when a model is unloaded;
        // a number the operator named for this model does not.
        return Err(if site.ram_is_rest { short(site, error) } else { error });
    }
    Ok((ram, why_ram))
}
