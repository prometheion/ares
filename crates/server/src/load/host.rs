//! The model a test of the binary serves without a GPU: the runtime's host
//! executor over its synthetic model (`ares_runtime::testing`), whose text is a
//! fixed function of the tokens and means nothing. A file whose architecture
//! is [`ARCHITECTURE`] is one; it needs no tensor. It lets the tests run the
//! binary itself on a machine without a device: its stops, the watchdog and
//! the restart in place that follows a step that does not come back
//! (`ares_runtime::HOLD_STEPS_WHILE` holds the steps).

use std::sync::Arc;
use std::time::Instant;

use ares_runtime::testing::HostExecutor;
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};

use super::{LoadError, LoadOptions, Loaded};

/// The architecture of the host's model.
pub(crate) const ARCHITECTURE: &str = "ares-host";

/// Starts the runtime of the host's model for `options`.
pub(super) fn load(options: &LoadOptions, begun: Instant) -> Result<Loaded, LoadError> {
    let name = options.name.clone().unwrap_or_else(|| crate::preset::id_of_file(&options.model));
    let config = SyntheticConfig { name: name.clone(), ..SyntheticConfig::small(Split::Single, 1) };
    let model = SyntheticModel::new(config).map_err(|e| LoadError::Model(e.to_string()))?;
    let model = Arc::new(model);
    let slots = options.parallel.unwrap_or(4);
    let executor = HostExecutor::new(model.clone(), model.config().max_rows, slots);
    let config = RuntimeConfig {
        pages: 256,
        cells_per_page: 16,
        max_slots: slots,
        warm_up: false,
        watchdog: match options.watchdog {
            None => RuntimeConfig::default().watchdog,
            Some(0) => None,
            Some(seconds) => Some(std::time::Duration::from_secs(seconds)),
        },
        ..RuntimeConfig::default()
    };
    let runtime = Runtime::start(model, Box::new(executor), config)
        .map_err(|e| LoadError::Start(e.to_string()))?;
    tracing::info!(
        "the host's model (its text means nothing) started in {:.1} s",
        begun.elapsed().as_secs_f64()
    );
    Ok(Loaded::new(runtime, name, Vec::new(), 0))
}
