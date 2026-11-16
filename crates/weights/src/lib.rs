//! Where weights live.
//!
//! A dense model's weights are placed once. A mixture-of-experts model is
//! different: each token uses a handful of the hundreds of experts per layer,
//! which ones changes slowly with the text, and the whole set is often larger
//! than VRAM, sometimes larger than RAM. So the experts are spread over
//! tiers (VRAM, pinned RAM, SSD), and which expert sits where is decided
//! continuously from what the router picks.
//!
//! This crate holds those decisions as plain bookkeeping:
//!
//! - [`ExpertHeat`] counts routed picks per layer and expert, with decay.
//! - [`SlotSet`] is one tier of one layer: a fixed number of slots, filled
//!   with the hottest experts and re-elected as the heat moves, or fed by
//!   reads like a cache in front of the SSD, with swaps that only take effect
//!   once their data has arrived. A tier fed by reads protects the experts the
//!   current step needs, counts a guess nobody used as worth nothing, and gives
//!   slots back to the memory broker and takes them again.
//! - [`GuessQueue`] holds the experts that a prediction or the token table
//!   named, until an I/O lane is free; a guess takes its slot only then.
//! - [`split_slots`] divides a tier's memory over the layers by what the next
//!   slot gains, and moves as few slots as it can when the profile or the
//!   budget changes.
//! - [`TokenTable`] maps the last one or two token ids to the experts the first
//!   MoE layers will route to, which no router prediction reaches, and is
//!   stored with the model.
//! - [`Lookahead`] chooses how many layers ahead to read, from how long reads
//!   and layers take and how often the prediction was right.
//! - [`ExpertWaves`] divides the experts a layer needs for a whole prompt into
//!   waves that fit the space for them, so that a prompt fetches each expert
//!   once.
//! - [`SlotSet::mark_present`] says which experts a tier holds or has on
//!   their way, which the router's pruning reads (dynamic expert pruning,
//!   `ares_cuda::Prune`): a pick that only the file holds may be left out or
//!   stood in for by an expert that is there, trading a bounded loss of
//!   quality for not waiting for the file.
//!
//! Reading the experts (direct I/O from the GGUF, staging, uploads) and the
//! kernels that compute with slots belong to the layers that own devices and
//! files; they act on the [`Swap`]s decided here.
//!
//! # What a runtime calls, and when
//!
//! Layers are numbered the same way everywhere (the MoE layers, in order), and
//! a step is a token or a window of drafts: any number that grows.
//!
//! - **A token was sampled.** [`TokenTable::lookup`] with it and the token
//!   before it, and for each layer of the entry [`GuessQueue::guess`] with the
//!   experts it names, for the next step.
//! - **The router of a layer has run.** [`ExpertHeat::record`] and
//!   [`SlotSet::routed`] with its picks, [`GuessQueue::forget_layer`] (the
//!   guesses for it are over), and [`SlotSet::start_read`] with
//!   [`Want::Demand`] for each pick that is not in the tier. Then, for the
//!   layer [`Lookahead::distance`] ahead, [`GuessQueue::guess`] with what the
//!   predictor names; for every distance up to the largest, compare the
//!   prediction with what that layer then routes and say so with
//!   [`Lookahead::observe_recall`].
//! - **An I/O lane is free and no demand read waits.**
//!   [`GuessQueue::start`], until it returns nothing. Read into the slot it
//!   names.
//! - **A read has arrived,** or failed: [`SlotSet::commit`], or
//!   [`SlotSet::abort`]. Its time goes to [`Lookahead::observe_read`].
//! - **Between steps, every few tokens.** [`ExpertHeat::decay`], and for the
//!   tiers that are elected, [`SlotSet::elect`].
//! - **The memory broker asks for RAM back, or gives it,** and every few hundred
//!   tokens as the profile moves: [`split_slots`] with the split in force,
//!   then [`SlotSet::give_back`] or [`SlotSet::take`] in each layer by the
//!   difference. The memory of a slot given back is freed by the caller, who then
//!   tells the broker.
//!
//! The first two, as code:
//!
//! ```
//! use ares_weights::{
//!     ExpertHeat, GuessQueue, SlotSet, Start, TableBuilder, TableConfig, Want,
//! };
//!
//! const LAYERS: usize = 6;
//! const EXPERTS: usize = 16;
//! let mut heat = ExpertHeat::new(LAYERS, EXPERTS, 0.5);
//! let mut tiers: Vec<SlotSet> = (0..LAYERS).map(|_| SlotSet::new(EXPERTS, 4)).collect();
//! let mut guesses = GuessQueue::new();
//!
//! // A table over the first two layers, built from a recording of the model:
//! // token 9 after token 7 routed to these experts, three times.
//! let mut builder = TableBuilder::new(TableConfig::new(2, 4), EXPERTS);
//! for _ in 0..3 {
//!     builder.record(Some(7), 9, &[[1, 2, 3, 4], [5, 6, 7, 8]]);
//! }
//! let table = builder.build();
//!
//! // Step 3: token 9 was sampled after token 7. Its first layers are guessed.
//! let step = 3;
//! let entry = table.lookup(Some(7), 9).expect("the table knows the token");
//! for layer in 0..entry.layers() {
//!     let experts: Vec<u32> = entry.layer(layer).map(|(expert, _)| expert).collect();
//!     guesses.guess(&mut tiers, layer, &experts, step);
//! }
//! // A lane is free: the likeliest guess takes a slot and is read.
//! let started = guesses.start(&mut tiers, &heat, step).expect("a guess waits");
//! assert_eq!((started.layer, started.swap.load), (0, 1));
//! // ... the read into the slot is made, and has arrived:
//! tiers[started.layer].commit(&started.swap);
//!
//! // The router of layer 0 has run and picked experts 1 (guessed) and 12.
//! let routed = [1, 12];
//! heat.record(0, &routed);
//! tiers[0].routed(&routed, step);
//! guesses.forget_layer(0);
//! for expert in routed {
//!     match tiers[0].start_read(expert, Want::Demand, step, heat.layer(0)) {
//!         Start::Present => assert_eq!(expert, 1, "the guess arrived in time"),
//!         Start::Slot(swap) => {
//!             // ... a demand read into the slot, and the model waits for it:
//!             tiers[0].commit(&swap);
//!         }
//!         Start::Loading | Start::NoRoom => unreachable!("room, and nothing on its way"),
//!     }
//! }
//! assert!(tiers[0].contains(1) && tiers[0].contains(12));
//! ```

mod expert;
mod guess;
mod lookahead;
mod split;
mod table;
mod waves;

pub use expert::{ExpertHeat, Given, SPARE_WORTH, SlotSet, Start, Swap, SwapRule, Want};
pub use guess::{GuessQueue, Started};
pub use lookahead::{Lookahead, LookaheadRule};
pub use split::{LayerNeed, SplitRule, split_slots};
pub use table::{Entry, TableBuilder, TableConfig, TableError, TokenTable};
pub use waves::{ExpertWaves, Place, Wave, recency_weight};
