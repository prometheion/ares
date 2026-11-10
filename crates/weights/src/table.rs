//! A table from the last tokens to the experts of the first layers.

use std::collections::HashMap;
use std::fmt;

/// Stands for "no token before this one" in a key, and so is not a token id.
const NO_TOKEN: u32 = u32::MAX;
/// Fills the end of a layer's list in an entry that has fewer experts than
/// `width`, and so is not an expert number.
const NO_EXPERT: u16 = u16::MAX;

const MAGIC: [u8; 4] = *b"ARTT";
const VERSION: u16 = 1;
/// Magic, version, layers, width, experts per layer (all `u16`), entries (`u32`).
const HEADER: usize = 16;

/// What a table covers and how large it may get.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableConfig {
    /// How many layers the table covers: the first MoE layers of the model, in
    /// order. These are the layers the router prediction cannot reach, since
    /// the layers before them have no router; a few are enough.
    pub layers: usize,
    /// How many experts an entry names per layer. The number the router
    /// picks per token is the natural width: more reads more guesses that
    /// are wrong for the one that is right.
    pub width: usize,
    /// The most entries the table keeps. Together with `layers` and `width`
    /// it fixes the size: [`TableConfig::entry_bytes`] per entry.
    pub capacity: usize,
    /// An expert that was routed in less than this share of the occurrences
    /// of a context is not kept in its entry. A weak guess costs a read and
    /// a slot for little chance; the shares are stored, so a runtime can
    /// also ask for more at lookup.
    pub min_share: f32,
}

impl TableConfig {
    /// A table over the first `layers` MoE layers with `width` experts per
    /// layer, of at most 65 536 entries, keeping experts routed in at least a
    /// tenth of a context's occurrences.
    pub fn new(layers: usize, width: usize) -> Self {
        Self { layers, width, capacity: 65_536, min_share: 0.1 }
    }

    /// The bytes one entry takes, in memory and in the stored form: the key
    /// of 8 bytes, and per layer and expert 2 bytes for its number and 1 for
    /// its share. Four layers of eight experts: 104 bytes.
    pub fn entry_bytes(&self) -> usize {
        8 + self.layers * self.width * 3
    }
}

/// The key of a context: the token that was just sampled and, when known, the
/// one before it. A single token is the key with no token before it.
fn key(previous: Option<u32>, token: u32) -> u64 {
    (u64::from(previous.unwrap_or(NO_TOKEN)) << 32) | u64::from(token)
}

/// The key of the token alone, for a key of a pair.
fn single(key: u64) -> u64 {
    (u64::from(NO_TOKEN) << 32) | (key & 0xffff_ffff)
}

/// What was routed after one context, counted.
#[derive(Debug, Default)]
struct Context {
    seen: u32,
    /// `layer << 16 | expert` -> how often.
    counts: HashMap<u32, u32>,
}

/// The experts of one context as an entry keeps them: `layers * width`
/// numbers and shares, each layer's list most likely first.
struct Lists {
    experts: Vec<u16>,
    shares: Vec<u8>,
    /// Per kept expert, how many occurrences routed to it. Same shape.
    counts: Vec<u32>,
}

impl Lists {
    fn of(context: &Context, config: &TableConfig) -> Self {
        let slots = config.layers * config.width;
        let mut lists = Self {
            experts: vec![NO_EXPERT; slots],
            shares: vec![0; slots],
            counts: vec![0; slots],
        };
        let mut per_layer: Vec<Vec<(u32, u16)>> = vec![Vec::new(); config.layers];
        for (packed, count) in &context.counts {
            per_layer[(packed >> 16) as usize].push((*count, (packed & 0xffff) as u16));
        }
        for (layer, list) in per_layer.iter_mut().enumerate() {
            // The most often routed first; ties go to the lower number.
            list.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let wanted = list.iter().take(config.width);
            for (i, (count, expert)) in wanted.enumerate() {
                let share = *count as f32 / context.seen as f32;
                if share < config.min_share {
                    break;
                }
                let at = layer * config.width + i;
                lists.experts[at] = *expert;
                lists.shares[at] = (share * 255.0).round().clamp(1.0, 255.0) as u8;
                lists.counts[at] = *count;
            }
        }
        lists
    }

    /// The hits the lists would have had on occurrences that repeat one
    /// another, not counting the first occurrence of each expert: an expert
    /// that was routed once is not known to be routed again. This leaves a
    /// context that was seen once with nothing, as it should be.
    fn repeated_hits(&self) -> u64 {
        self.counts.iter().map(|c| u64::from(c.saturating_sub(1))).sum()
    }

    /// The hits these lists would have on the occurrences `context` counted.
    fn hits_on(&self, context: &Context, config: &TableConfig) -> u64 {
        let mut hits = 0;
        for (at, expert) in self.experts.iter().enumerate() {
            if *expert == NO_EXPERT {
                continue;
            }
            let layer = at / config.width;
            let packed = (layer as u32) << 16 | u32::from(*expert);
            hits += u64::from(context.counts.get(&packed).copied().unwrap_or(0));
        }
        hits
    }
}

/// Counts what the first layers routed after each context, from a recording
/// of the model on text, and builds the [`TokenTable`] from it.
///
/// This is an offline tool for building the model's routing profile: it keeps
/// a count for every context it sees, so its memory grows with the recording
/// (the table that comes out is bounded).
#[derive(Debug)]
pub struct TableBuilder {
    config: TableConfig,
    n_experts: usize,
    contexts: HashMap<u64, Context>,
}

impl TableBuilder {
    /// A builder for a model whose layers each have `n_experts` experts
    /// (below 65 535), for a table as `config` says.
    ///
    /// # Panics
    ///
    /// Panics on a table of no layers or width, of more than 65 535 layers
    /// or width, or on a number of experts that does not fit.
    pub fn new(config: TableConfig, n_experts: usize) -> Self {
        assert!((1..=0xffff).contains(&config.layers), "layers must be 1 to 65535");
        assert!((1..=0xffff).contains(&config.width), "width must be 1 to 65535");
        assert!((1..usize::from(NO_EXPERT)).contains(&n_experts), "experts must be 1 to 65534");
        Self { config, n_experts, contexts: HashMap::new() }
    }

    /// Records one forward pass of the model: it was fed `token` after
    /// `previous` (`None` at the start of a text), and the router of each of
    /// the first MoE layers picked `routed[layer]`. More layers than the
    /// table covers may be given; the rest are ignored. Token ids are below
    /// `u32::MAX`.
    ///
    /// Counts for the token alone and, with `previous`, for the pair.
    pub fn record(&mut self, previous: Option<u32>, token: u32, routed: &[impl AsRef<[u32]>]) {
        assert!(routed.len() >= self.config.layers, "the routing of every table layer");
        assert!(token != NO_TOKEN && previous != Some(NO_TOKEN), "u32::MAX is not a token id");
        let keys = [Some(key(None, token)), previous.map(|p| key(Some(p), token))];
        for key in keys.into_iter().flatten() {
            let context = self.contexts.entry(key).or_default();
            context.seen += 1;
            for (layer, experts) in routed[..self.config.layers].iter().enumerate() {
                for expert in experts.as_ref() {
                    assert!((*expert as usize) < self.n_experts, "expert {expert} out of range");
                    *context.counts.entry((layer as u32) << 16 | *expert).or_insert(0) += 1;
                }
            }
        }
    }

    /// Builds the table: the `config.capacity` contexts whose entries gain
    /// most.
    ///
    /// What an entry gains is how often its guesses would have been right on
    /// the recording, not counting the first time each expert was routed
    /// after the context (a context seen once, or an expert routed once,
    /// gains nothing: the recording cannot say it will happen again). For a
    /// pair the guesses of the token alone are what it would otherwise fall
    /// back on, so it gains only what it adds to them: a pair that routes as
    /// its token does gets no entry, one that routes differently does.
    pub fn build(self) -> TokenTable {
        let Self { config, n_experts, contexts } = self;
        let lists: HashMap<u64, Lists> =
            contexts.iter().map(|(key, context)| (*key, Lists::of(context, &config))).collect();

        let mut chosen: Vec<(u64, u64)> = Vec::new();
        for (key, lists_of) in &lists {
            let own = lists_of.repeated_hits();
            let is_pair = (key >> 32) as u32 != NO_TOKEN;
            let gain = if is_pair {
                let fallback = lists.get(&single(*key));
                let fallback_hits = fallback.map_or(0, |f| f.hits_on(&contexts[key], &config));
                own.saturating_sub(fallback_hits)
            } else {
                own
            };
            if gain > 0 {
                chosen.push((*key, gain));
            }
        }
        // The largest gain first, ties by key; then the keys in order.
        chosen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        chosen.truncate(config.capacity);
        chosen.sort_by_key(|(key, _)| *key);

        let slots = config.layers * config.width;
        let mut table = TokenTable {
            layers: config.layers,
            width: config.width,
            n_experts,
            keys: Vec::with_capacity(chosen.len()),
            experts: Vec::with_capacity(chosen.len() * slots),
            shares: Vec::with_capacity(chosen.len() * slots),
        };
        for (key, _) in chosen {
            let entry = &lists[&key];
            table.keys.push(key);
            table.experts.extend_from_slice(&entry.experts);
            table.shares.extend_from_slice(&entry.shares);
        }
        table
    }
}

/// The experts an entry names for the layers of the table.
#[derive(Clone, Copy, Debug)]
pub struct Entry<'a> {
    width: usize,
    experts: &'a [u16],
    shares: &'a [u8],
}

impl<'a> Entry<'a> {
    /// The number of layers the entry covers.
    pub fn layers(&self) -> usize {
        self.experts.len() / self.width
    }

    /// The experts guessed for `layer` (0 is the first MoE layer), most
    /// likely first, each with the share of the context's occurrences that
    /// routed to it in that layer: the chance that the guess is right, as
    /// recorded.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is not below [`Entry::layers`].
    pub fn layer(&self, layer: usize) -> impl Iterator<Item = (u32, f32)> + 'a {
        let range = layer * self.width..(layer + 1) * self.width;
        self.experts[range.clone()]
            .iter()
            .zip(&self.shares[range])
            .take_while(|(expert, _)| **expert != NO_EXPERT)
            .map(|(expert, share)| (u32::from(*expert), f32::from(*share) / 255.0))
    }
}

/// From the last one or two token ids to the experts the first MoE layers
/// will route to for the token just sampled.
///
/// The router prediction works from the hidden state of an earlier layer, and
/// the first MoE layers of a token have no earlier router to predict them. But
/// their routing depends mostly on the token itself (it is the embedding that
/// goes in), and the token is known the moment it is sampled, before its
/// forward pass starts. So a table from the token (and the one before it, for
/// the contexts where that changes the routing) to the experts those layers
/// picked on a recording of the model gives guesses for them at the moment of
/// sampling, to be queued like predictions ([`GuessQueue`](crate::GuessQueue)).
/// The same idea reads the rows of a hashed embedding table when the token is
/// known.
///
/// # Size
///
/// A table is `capacity` entries at most. An entry costs
/// [`TableConfig::entry_bytes`]: with 4 layers of 8 experts 104 bytes, so a
/// full table of 65 536 entries is 6.8 MB, and with 8 layers 13 MB. The
/// builder keeps the entries that gain most, which are those of the tokens
/// and pairs that come up often and route the same way.
///
/// # Stored with the model
///
/// The table is a profile of one model, built once from a recording and kept
/// next to its weights as a file of its own. It is only valid for the model it
/// was recorded on; the stored form checks its own shape and the range of the
/// expert numbers, and whoever stores it must tie it to the model (by the
/// fingerprint of the weights, say). [`TokenTable::to_bytes`] is its form,
/// little-endian throughout:
///
/// | Bytes | What |
/// | --- | --- |
/// | 0..4 | `ARTT` |
/// | 4..6 | version, 1 (`u16`) |
/// | 6..8 | layers (`u16`) |
/// | 8..10 | width (`u16`) |
/// | 10..12 | experts per layer of the model (`u16`; ids are below it) |
/// | 12..16 | entries (`u32`) |
/// | then, per entry, by ascending key | key (`u64`: the previous token in the upper 32 bits, `u32::MAX` for none, the token in the lower), then `layers * width` expert numbers (`u16`, layer after layer, most likely first, `0xffff` fills the end of a layer), then `layers * width` shares (`u8`, 0 for a filler, otherwise 1 to 255 for the share of 255) |
///
/// The length is exactly `16 + entries * entry_bytes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenTable {
    layers: usize,
    width: usize,
    n_experts: usize,
    /// Ascending, so a lookup is a binary search.
    keys: Vec<u64>,
    /// `layers * width` per entry.
    experts: Vec<u16>,
    shares: Vec<u8>,
}

impl TokenTable {
    /// The number of entries.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The layers an entry covers.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The experts an entry names per layer, at most.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The bytes of one entry, in memory and stored.
    pub fn entry_bytes(&self) -> usize {
        8 + self.layers * self.width * 3
    }

    /// The bytes of the whole table as [`TokenTable::to_bytes`] writes it.
    pub fn byte_size(&self) -> usize {
        HEADER + self.len() * self.entry_bytes()
    }

    fn entry(&self, at: usize) -> Entry<'_> {
        let slots = self.layers * self.width;
        Entry {
            width: self.width,
            experts: &self.experts[at * slots..(at + 1) * slots],
            shares: &self.shares[at * slots..(at + 1) * slots],
        }
    }

    /// The guesses for the first layers of the pass that is fed `token`
    /// after `previous` (`None` at the start of a text): the entry of the
    /// pair if the table has one, else that of the token, else `None`.
    ///
    /// A binary search over the keys, done once per sampled token.
    pub fn lookup(&self, previous: Option<u32>, token: u32) -> Option<Entry<'_>> {
        let find = |key| self.keys.binary_search(&key).ok().map(|at| self.entry(at));
        previous.and_then(|p| find(key(Some(p), token))).or_else(|| find(key(None, token)))
    }

    /// The table in the form it is stored in.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_size());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.layers as u16).to_le_bytes());
        out.extend_from_slice(&(self.width as u16).to_le_bytes());
        out.extend_from_slice(&(self.n_experts as u16).to_le_bytes());
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());
        let slots = self.layers * self.width;
        for (at, key) in self.keys.iter().enumerate() {
            out.extend_from_slice(&key.to_le_bytes());
            for expert in &self.experts[at * slots..(at + 1) * slots] {
                out.extend_from_slice(&expert.to_le_bytes());
            }
            out.extend_from_slice(&self.shares[at * slots..(at + 1) * slots]);
        }
        out
    }

    /// Reads a table written by [`TokenTable::to_bytes`], checking all of it.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TableError> {
        let Some(header) = bytes.get(..HEADER) else { return Err(TableError::Truncated) };
        if header[..4] != MAGIC {
            return Err(TableError::NotATable);
        }
        let u16_at = |at: usize| usize::from(u16::from_le_bytes([header[at], header[at + 1]]));
        let version = u16_at(4) as u16;
        if version != VERSION {
            return Err(TableError::Version(version));
        }
        let (layers, width, n_experts) = (u16_at(6), u16_at(8), u16_at(10));
        let entries = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
        if layers == 0 || width == 0 || n_experts == 0 || n_experts >= usize::from(NO_EXPERT) {
            return Err(TableError::Shape);
        }
        let slots = layers * width;
        let entry_bytes = 8 + slots * 3;
        let expected = HEADER + entries * entry_bytes;
        if bytes.len() != expected {
            return Err(TableError::Length { expected, found: bytes.len() });
        }

        let mut table = Self {
            layers,
            width,
            n_experts,
            keys: Vec::with_capacity(entries),
            experts: Vec::with_capacity(entries * slots),
            shares: Vec::with_capacity(entries * slots),
        };
        for (at, entry) in bytes[HEADER..].chunks_exact(entry_bytes).enumerate() {
            let key = u64::from_le_bytes(entry[..8].try_into().expect("8 bytes"));
            if table.keys.last().is_some_and(|last| *last >= key) {
                return Err(TableError::KeysNotAscending { entry: at });
            }
            let (experts, shares) = entry[8..].split_at(slots * 2);
            for (i, (expert, share)) in experts.as_chunks::<2>().0.iter().zip(shares).enumerate() {
                let expert = u16::from_le_bytes(*expert);
                let filler = expert == NO_EXPERT;
                // A filler has no share, an expert has one, and fillers only
                // end a layer's list.
                let after_filler = i % width > 0 && table.experts.last() == Some(&NO_EXPERT);
                if (!filler && usize::from(expert) >= n_experts)
                    || filler != (*share == 0)
                    || (after_filler && !filler)
                {
                    return Err(TableError::BadEntry { entry: at });
                }
                table.experts.push(expert);
            }
            table.shares.extend_from_slice(shares);
            table.keys.push(key);
        }
        Ok(table)
    }
}

/// Why bytes are not a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    /// Shorter than the header.
    Truncated,
    /// Does not begin with the table's magic.
    NotATable,
    /// A version this code does not read.
    Version(u16),
    /// A header with no layers, no width, or numbers of experts that cannot be.
    Shape,
    /// The length is not that of the header's entries.
    Length { expected: usize, found: usize },
    /// Keys must ascend, or a lookup would not find them.
    KeysNotAscending { entry: usize },
    /// An expert number out of range, a share that does not fit its expert,
    /// or fillers in front of an expert.
    BadEntry { entry: usize },
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "token table: shorter than its header"),
            Self::NotATable => write!(f, "token table: wrong magic, this is not a token table"),
            Self::Version(v) => write!(f, "token table: version {v} is not one this reads"),
            Self::Shape => write!(f, "token table: layers, width or experts of 0 or too large"),
            Self::Length { expected, found } => {
                write!(f, "token table: {found} bytes, its header says {expected}")
            }
            Self::KeysNotAscending { entry } => {
                write!(f, "token table: the key of entry {entry} does not ascend")
            }
            Self::BadEntry { entry } => write!(f, "token table: entry {entry} is malformed"),
        }
    }
}

impl std::error::Error for TableError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder(layers: usize, width: usize) -> TableBuilder {
        TableBuilder::new(TableConfig::new(layers, width), 64)
    }

    fn experts(entry: &Entry<'_>, layer: usize) -> Vec<u32> {
        entry.layer(layer).map(|(expert, _)| expert).collect()
    }

    #[test]
    fn a_token_names_the_experts_each_layer_routed_to_most_often_after_it() {
        let mut builder = builder(2, 2);
        // Token 7: layer 0 routes to 1 and 2 (1 more often), layer 1 to 40.
        for _ in 0..6 {
            builder.record(None, 7, &[[1, 2], [40, 41]]);
        }
        for _ in 0..3 {
            builder.record(None, 7, &[[1, 3], [40, 42]]);
        }
        let table = builder.build();
        assert_eq!(table.len(), 1);
        let entry = table.lookup(None, 7).unwrap();
        assert_eq!(entry.layers(), 2);
        // Layer 0: expert 1 in 9 of 9 passes, 2 in 6 of 9, and 3 (3 of 9) is the
        // third and does not fit the width.
        assert_eq!(experts(&entry, 0), [1, 2]);
        assert_eq!(experts(&entry, 1), [40, 41]);
        let shares: Vec<f32> = entry.layer(0).map(|(_, share)| share).collect();
        assert!((shares[0] - 1.0).abs() < 0.01 && (shares[1] - 6.0 / 9.0).abs() < 0.01);
        assert!(table.lookup(None, 8).is_none(), "an unknown token has no entry");
        // The pair of an unknown pair falls back on the token.
        assert!(table.lookup(Some(3), 7).is_some());
    }

    #[test]
    fn a_context_seen_once_or_an_expert_routed_once_gains_nothing() {
        let mut builder = builder(1, 4);
        builder.record(Some(1), 5, &[[1, 2, 3, 4]]);
        builder.record(None, 6, &[[1, 2]]);
        builder.record(None, 6, &[[1, 9]]);
        let table = builder.build();
        assert!(table.lookup(None, 5).is_none(), "seen once");
        // Token 6 routed to 1 twice: that is worth an entry. Experts 2 and 9
        // were routed once and a tenth of the occurrences passes the share,
        // but they are not known to repeat.
        assert_eq!(table.len(), 1);
        let entry = table.lookup(None, 6).unwrap();
        assert_eq!(experts(&entry, 0), [1, 2, 9]);
    }

    #[test]
    fn a_pair_gets_an_entry_only_where_it_routes_differently_from_its_token() {
        let mut builder = builder(1, 2);
        // After 1 the token 5 routes to 10 and 11, after 2 to 20 and 21.
        for _ in 0..10 {
            builder.record(Some(1), 5, &[[10, 11]]);
            builder.record(Some(2), 5, &[[20, 21]]);
        }
        // Token 6 routes to 30 and 31 whatever comes before it.
        for before in [1, 2, 3, 4] {
            for _ in 0..5 {
                builder.record(Some(before), 6, &[[30, 31]]);
            }
        }
        let table = builder.build();
        let pairs = [(Some(1), 5), (Some(2), 5)];
        let guesses: Vec<Vec<u32>> =
            pairs.iter().map(|(p, t)| experts(&table.lookup(*p, *t).unwrap(), 0)).collect();
        assert_eq!(guesses, [vec![10, 11], vec![20, 21]]);
        // Token 5 alone: the two pairs' experts are equally often, two of
        // the four are named.
        assert_eq!(experts(&table.lookup(None, 5).unwrap(), 0).len(), 2);
        // Token 6 has its token entry only; its pairs add nothing.
        assert_eq!(experts(&table.lookup(Some(3), 6).unwrap(), 0), [30, 31]);
        // The pair (1, 5) routes as the token does, so it has no entry.
        assert_eq!(table.len(), 3, "tokens 5 and 6, and the pair (2, 5)");
    }

    #[test]
    fn a_bounded_table_keeps_the_entries_that_gain_most() {
        let mut builder =
            TableBuilder::new(TableConfig { capacity: 2, ..TableConfig::new(1, 1) }, 8);
        for (token, times) in [(1, 3), (2, 20), (3, 9), (4, 2)] {
            for _ in 0..times {
                builder.record(None, token, &[[token]]);
            }
        }
        let table = builder.build();
        assert_eq!(table.len(), 2);
        assert!(table.lookup(None, 2).is_some() && table.lookup(None, 3).is_some());
        assert!(table.lookup(None, 1).is_none() && table.lookup(None, 4).is_none());
    }

    #[test]
    fn experts_below_the_share_are_left_out_and_short_lists_are_padded() {
        let config = TableConfig { min_share: 0.3, ..TableConfig::new(2, 3) };
        let mut builder = TableBuilder::new(config, 64);
        for i in 0..10 {
            // Layer 0: expert 1 always, expert 2 in 2 of 10. Layer 1: nothing
            // is routed twice, so nothing passes the share but the first.
            let second = if i < 2 { 2 } else { 3 + i };
            builder.record(None, 9, &[vec![1, second], vec![20 + i]]);
        }
        let table = builder.build();
        let entry = table.lookup(None, 9).unwrap();
        assert_eq!(experts(&entry, 0), [1]);
        assert_eq!(experts(&entry, 1), [] as [u32; 0]);
        // Through the stored form as well.
        let again = TokenTable::from_bytes(&table.to_bytes()).unwrap();
        assert_eq!(experts(&again.lookup(None, 9).unwrap(), 0), [1]);
    }

    #[test]
    fn a_table_costs_the_same_bytes_per_entry_whatever_it_holds_and_reads_back_as_written() {
        let mut builder = TableBuilder::new(TableConfig::new(4, 8), 256);
        for token in 0..50u32 {
            for rep in 0..4u32 {
                let routed: Vec<Vec<u32>> = (0..4)
                    .map(|l| (0..8).map(|i| (token * 3 + l * 5 + i + rep % 2) % 256).collect())
                    .collect();
                builder.record(Some(token + 1), token, &routed);
            }
        }
        let table = builder.build();
        assert!(table.len() >= 50);
        assert_eq!(table.entry_bytes(), 104, "key 8, 32 experts of 2 bytes, 32 shares of 1");
        let bytes = table.to_bytes();
        assert_eq!(bytes.len(), table.byte_size());
        assert_eq!(bytes.len(), 16 + table.len() * 104);
        assert_eq!(&bytes[..4], b"ARTT");
        let again = TokenTable::from_bytes(&bytes).unwrap();
        assert_eq!(again, table);
        assert_eq!(again.to_bytes(), bytes);
    }

    #[test]
    fn bytes_that_are_not_a_table_are_refused() {
        let mut builder = builder(1, 2);
        for token in 1..4 {
            for _ in 0..3 {
                builder.record(None, token, &[[token, token + 1]]);
            }
        }
        let good = builder.build().to_bytes();
        assert!(TokenTable::from_bytes(&good).is_ok());

        let broken = |edit: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = good.clone();
            edit(&mut bytes);
            TokenTable::from_bytes(&bytes).unwrap_err()
        };
        assert_eq!(TokenTable::from_bytes(&good[..10]), Err(TableError::Truncated));
        assert_eq!(broken(&|b| b[0] = b'X'), TableError::NotATable);
        assert_eq!(broken(&|b| b[4] = 9), TableError::Version(9));
        assert_eq!(broken(&|b| b[6] = 0), TableError::Shape);
        assert!(matches!(broken(&|b| b.truncate(b.len() - 1)), TableError::Length { .. }));
        assert!(matches!(broken(&|b| b.push(0)), TableError::Length { .. }));
        // Entry 0 gets the key of entry 1.
        assert_eq!(
            broken(&|b| {
                let second: Vec<u8> = b[16 + 14..16 + 22].to_vec();
                b[16..24].copy_from_slice(&second);
            }),
            TableError::KeysNotAscending { entry: 1 }
        );
        // An expert number beyond the model's experts.
        assert_eq!(broken(&|b| b[24] = 200), TableError::BadEntry { entry: 0 });
        // A filler with a share.
        assert_eq!(
            broken(&|b| {
                b[24] = 0xff;
                b[25] = 0xff;
            }),
            TableError::BadEntry { entry: 0 }
        );
    }
}
