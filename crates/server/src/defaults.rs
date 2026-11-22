//! The server's own defaults for the requests of a model: what a field of the
//! sampler means when a request leaves it out, and how long a reply may be.
//!
//! Each is an option of a model ([`crate::options`]), on the command line and
//! as a key of a preset file. An option that gives a field of a request its
//! default has the name of the field, with dashes: `--top-p` for `top_p`,
//! `--repeat-last-n` for `repeat_last_n`, `--max-tokens` for `max_tokens`
//! (`SETTINGS` in this module is the list). One option is not a default but a
//! bound: `--max-tokens-limit`, the most tokens of any reply.
//!
//! # What a request gets
//!
//! A request's own value always holds. A field it leaves out, or writes as
//! `null`, takes the model's default, and without a default the engine's
//! built-in value, as it did before there were defaults. The defaults are
//! filled in where every route that generates takes its model
//! (`routes::prepare`), so they hold on `/v1/chat/completions`,
//! `/v1/completions`, `/v1/messages` and `/v1/responses` alike, also for the
//! fields the API of a route has no member for (`min_p` on `/v1/messages`).
//! `seed: -1`, llama.cpp's word for a seed of the request's own draw, is a
//! value and not a field left out: such a request does not get the model's
//! seed.
//!
//! There is one set of defaults for a model, not one for replies that think
//! and one for those that do not, as Strata and TensorFold keep them: the
//! production setup this server replaces has one, and the one value Qwen's
//! card gives differently for the two (the presence penalty) is 0 there for
//! both. A second set would be more keys of the same kind, applied in the
//! same place, where the server also reads what a request says about
//! thinking.
//!
//! # The length of a reply
//!
//! `--max-tokens` is the limit of a reply whose request names none: no
//! `max_tokens`, `max_completion_tokens`, `max_output_tokens` or `n_predict`,
//! or llama.cpp's `-1`, which llama-server reads as "the server's" too. It is
//! what llama-server's `--n-predict` is. `--max-tokens-limit` is the most any
//! request gets: one that asks for more gets this many and is not refused,
//! and one that names no limit gets `--max-tokens` or, without it, this.
//!
//! A limit is an upper bound and not an order for that many tokens, so a
//! reply the server's limit cut is a reply the request allows, as one that the
//! end of the context cut is. A refusal would lock out the clients that send
//! the same large number with every request (the Messages API requires the
//! member). The request can see what happened: the reply ends with `length`
//! (`max_tokens` on `/v1/messages`, `incomplete` on `/v1/responses`) and its
//! usage has the count, the response has the header [`LIMIT_HEADER`] with the
//! limit that took the place of the request's, and a response object of
//! `/v1/responses` says the limit in effect in `max_output_tokens`.

use ares_api::Sampling;

use crate::options::Invalid;
use crate::wire::Parsed;

/// The header of a response whose request asked for more tokens than the
/// model's `--max-tokens-limit`: its value is the limit the reply got.
pub const LIMIT_HEADER: &str = "x-ares-max-tokens-limit";

/// The most stages of a sampler's program that cut or scale, which is the
/// engine's limit (`ares_sampler::MAX_STAGES`).
pub const STAGES_MAX: usize = 16;

/// What the options of a model say about its requests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Defaults {
    /// The sampler: a field that is set is the default of the request's field
    /// of that name. Only the fields that an option of a model sets are read;
    /// a bias, `ignore_eos` and the probabilities to report have no default.
    pub sampling: Sampling,
    /// The limit of a reply whose request names none (`--max-tokens`).
    pub max_tokens: Option<u32>,
    /// The most tokens of any reply (`--max-tokens-limit`).
    pub max_tokens_limit: Option<u32>,
}

/// Two options that cannot hold together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    /// The two, by the names [`crate::options::lookup`] gives.
    pub flags: [&'static str; 2],
    pub reason: String,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

/// A field of the sampler that the options of a model give a default.
pub(crate) struct Setting {
    /// The option (`--top-p`); the key of a preset file is the name without
    /// the two dashes.
    pub flag: &'static str,
    /// The field of a request it is the default of (`top_p`).
    pub field: &'static str,
    /// What the option takes, for the error of a value it does not take.
    pub expected: &'static str,
    /// Reads the value of the option into the field; `false` for a text that
    /// is not a value of its kind.
    read: fn(&mut Sampling, &str) -> bool,
    /// The field as a request writes it, if it is set.
    show: fn(&Sampling) -> Option<serde_json::Value>,
    /// Gives the first the field of the second where it has none.
    fill: fn(&mut Sampling, &Sampling),
}

/// `value` as the decimal number that was written for it: `0.95` as an `f32`
/// is 0.949999988 as an `f64`, and a listing should say 0.95.
pub(crate) fn decimal(value: f32) -> f64 {
    value.to_string().parse().unwrap_or(f64::from(value))
}

/// A number with a fraction.
mod number {
    pub fn read(text: &str) -> Option<f32> {
        text.trim().parse().ok()
    }

    pub fn show(value: f32) -> serde_json::Value {
        super::decimal(value).into()
    }
}

/// A number of tokens.
mod count {
    pub fn read(text: &str) -> Option<u32> {
        text.trim().parse().ok()
    }

    pub fn show(value: u32) -> serde_json::Value {
        value.into()
    }
}

/// One of a few numbered versions.
mod version {
    pub fn read(text: &str) -> Option<u8> {
        text.trim().parse().ok()
    }

    pub fn show(value: u8) -> serde_json::Value {
        value.into()
    }
}

/// A seed.
mod seed {
    pub fn read(text: &str) -> Option<u64> {
        text.trim().parse().ok()
    }

    pub fn show(value: u64) -> serde_json::Value {
        value.into()
    }
}

/// Strings, written as a JSON array: they may hold anything, a line break
/// and a comma included.
mod strings {
    pub fn read(text: &str) -> Option<Vec<String>> {
        serde_json::from_str(text).ok()
    }

    pub fn show(value: Vec<String>) -> serde_json::Value {
        value.into()
    }
}

/// A text, as it is written.
mod text {
    pub fn read(text: &str) -> Option<String> {
        Some(text.to_owned())
    }

    pub fn show(value: String) -> serde_json::Value {
        value.into()
    }
}

/// The samplers in the order they apply in, by their names.
mod order {
    use ares_api::SamplerKind;

    pub fn read(text: &str) -> Option<Vec<SamplerKind>> {
        let kinds = text
            .split(',')
            .map(|name| SamplerKind::from_name(name.trim()))
            .collect::<Option<Vec<_>>>()?;
        // The engine's rule: the penalties and DRY correct the logits, so
        // they stand before the stages that cut or scale, once each, and a
        // program holds no more than `STAGES_MAX` of those.
        let corrects =
            |kind: &SamplerKind| matches!(kind, SamplerKind::Penalties | SamplerKind::Dry);
        let first = kinds.iter().take_while(|kind| corrects(kind)).count();
        let twice = first == 2 && kinds[0] == kinds[1];
        let stages = &kinds[first..];
        let valid = first <= 2
            && !twice
            && !stages.iter().any(corrects)
            && stages.len() <= super::STAGES_MAX;
        valid.then_some(kinds)
    }

    pub fn show(value: Vec<SamplerKind>) -> serde_json::Value {
        value.iter().map(|kind| kind.name()).collect::<Vec<_>>().into()
    }
}

macro_rules! setting {
    ($flag:literal, $field:ident, $kind:ident, $expected:literal) => {
        Setting {
            flag: $flag,
            field: stringify!($field),
            expected: $expected,
            read: |sampling, text| match $kind::read(text) {
                Some(value) => {
                    sampling.$field = Some(value);
                    true
                }
                None => false,
            },
            show: |sampling| sampling.$field.clone().map($kind::show),
            fill: |request, default| {
                if request.$field.is_none() {
                    request.$field.clone_from(&default.$field);
                }
            },
        }
    };
}

/// Every field of the sampler with a default, in the order the help and a
/// listing have them. The ranges are those a request is held to
/// ([`Sampling::validate`]).
pub(crate) const SETTINGS: &[Setting] = &[
    setting!("--temperature", temperature, number, "a number from 0 on (0 picks the best token)"),
    setting!("--top-k", top_k, count, "a number of tokens, 0 for no cut"),
    setting!("--top-p", top_p, number, "a number above 0 and at most 1 (1 cuts nothing)"),
    setting!("--min-p", min_p, number, "a number from 0 to 1 (0 cuts nothing)"),
    setting!("--typical-p", typical_p, number, "a number above 0 and at most 1 (1 cuts nothing)"),
    setting!("--top-n-sigma", top_n_sigma, number, "a number (0 or less cuts nothing)"),
    setting!("--xtc-probability", xtc_probability, number, "a number from 0 to 1 (0 is off)"),
    setting!("--xtc-threshold", xtc_threshold, number, "a number from 0 to 1"),
    setting!("--dynatemp-range", dynatemp_range, number, "a number from 0 on (0 is off)"),
    setting!("--dynatemp-exponent", dynatemp_exponent, number, "a number from 0 on"),
    setting!("--repeat-penalty", repeat_penalty, number, "a number above 0 (1 changes nothing)"),
    setting!("--presence-penalty", presence_penalty, number, "a number from -2 to 2"),
    setting!("--frequency-penalty", frequency_penalty, number, "a number from -2 to 2"),
    setting!(
        "--repeat-last-n",
        repeat_last_n,
        count,
        "a number of tokens from 0 to 1024 (0 switches the three penalties off)"
    ),
    setting!("--dry-multiplier", dry_multiplier, number, "a number from 0 on (0 is off)"),
    setting!("--dry-base", dry_base, number, "a number from 0 on"),
    setting!("--dry-allowed-length", dry_allowed_length, count, "a number of tokens up to 1024"),
    setting!("--dry-penalty-last-n", dry_penalty_last_n, count, "a number of tokens up to 1024"),
    setting!(
        "--dry-sequence-breakers",
        dry_sequence_breakers,
        strings,
        "a JSON array of strings, such as [\"\\n\", \":\"]"
    ),
    setting!("--mirostat", mirostat, version, "0 (off), 1 or 2"),
    setting!("--mirostat-tau", mirostat_tau, number, "a number above 0"),
    setting!("--mirostat-eta", mirostat_eta, number, "a number from 0 on"),
    setting!(
        "--samplers",
        samplers,
        order,
        "names of samplers separated by commas, penalties and dry before the others and once \
         each, at most 16 others (penalties,dry,top_n_sigma,top_k,typ_p,top_p,min_p,xtc,\
         temperature)"
    ),
    setting!(
        "--seed",
        seed,
        seed,
        "a whole number from 0 on (without the option every request draws its own)"
    ),
    setting!(
        "--reasoning-budget-tokens",
        reasoning_budget_tokens,
        count,
        "a number of tokens of thinking, after which the thinking is ended"
    ),
    setting!(
        "--reasoning-budget-message",
        reasoning_budget_message,
        text,
        "a text written before the end of the thinking when its budget is spent"
    ),
];

const MAX_TOKENS: &str = "--max-tokens";
const MAX_TOKENS_LIMIT: &str = "--max-tokens-limit";

/// The options of this module under their one names.
pub(crate) fn flags() -> impl Iterator<Item = &'static str> {
    [MAX_TOKENS, MAX_TOKENS_LIMIT].into_iter().chain(SETTINGS.iter().map(|setting| setting.flag))
}

/// The option that is written `name` on the command line, if it is one of
/// this module: its one name, under which [`set`] takes it.
pub(crate) fn lookup(name: &str) -> Option<&'static str> {
    flags().find(|flag| *flag == name)
}

/// Sets the option `flag` (a name [`lookup`] gave) to `value`.
pub(crate) fn set(defaults: &mut Defaults, flag: &'static str, value: &str) -> Result<(), Invalid> {
    let invalid = |expected| Invalid { flag, value: value.to_owned(), expected };
    if flag == MAX_TOKENS || flag == MAX_TOKENS_LIMIT {
        let expected = "a number of tokens, at least 1";
        let tokens = value.trim().parse::<u32>().ok().filter(|tokens| *tokens >= 1);
        let tokens = Some(tokens.ok_or_else(|| invalid(expected))?);
        if flag == MAX_TOKENS {
            defaults.max_tokens = tokens;
        } else {
            defaults.max_tokens_limit = tokens;
        }
        return Ok(());
    }
    let setting = SETTINGS
        .iter()
        .find(|setting| setting.flag == flag)
        .unwrap_or_else(|| unreachable!("'{flag}' is not a name lookup gives"));
    // Read alone first: the range of a value is the one a request is held to.
    let mut alone = Sampling::default();
    if !(setting.read)(&mut alone, value) || alone.validate().is_err() {
        return Err(invalid(setting.expected));
    }
    (setting.read)(&mut defaults.sampling, value);
    Ok(())
}

impl Defaults {
    /// Whether the options hold together.
    pub fn check(&self) -> Result<(), Conflict> {
        match (self.max_tokens, self.max_tokens_limit) {
            (Some(default), Some(limit)) if default > limit => Err(Conflict {
                flags: [MAX_TOKENS, MAX_TOKENS_LIMIT],
                reason: format!(
                    "max-tokens = {default} is more than max-tokens-limit = {limit}: the length \
                     of a reply whose request names none cannot be above the most a request gets"
                ),
            }),
            _ => Ok(()),
        }
    }

    /// The limit of a reply whose request asked for `asked` tokens at most
    /// (`None`: it named no limit), and whether that is less than it asked for.
    pub fn length(&self, asked: Option<u32>) -> (Option<u32>, bool) {
        match (asked, self.max_tokens_limit) {
            (Some(asked), Some(limit)) if asked > limit => (Some(limit), true),
            (Some(asked), _) => (Some(asked), false),
            (None, limit) => (self.max_tokens.or(limit), false),
        }
    }

    /// Gives a request what it leaves out: the sampler's fields, and the
    /// limit of its reply, which is also held to the most a request gets.
    pub(crate) fn apply(&self, parsed: &mut Parsed) {
        let request = &mut parsed.request;
        for setting in SETTINGS {
            (setting.fill)(&mut request.sampling, &self.sampling);
        }
        // A request that asks for a seed of its own draw (`seed: -1`) said
        // something, and the model's seed is for those that say nothing.
        if parsed.fresh_seed {
            request.sampling.seed = None;
        }
        let (limit, lowered) = self.length(request.max_tokens);
        request.max_tokens = limit;
        parsed.lowered = if lowered { limit } else { None };
    }

    /// What a field means that a request leaves out, under the names of the
    /// fields and written as a request writes them: the sampler's fields that
    /// have a default, and `max_tokens`, the limit of a reply whose request
    /// names none. A field that is not there is the engine's own.
    pub fn fields(&self) -> serde_json::Value {
        let mut fields = serde_json::Map::new();
        for setting in SETTINGS {
            if let Some(value) = (setting.show)(&self.sampling) {
                fields.insert(setting.field.to_owned(), value);
            }
        }
        if let (Some(limit), _) = self.length(None) {
            fields.insert("max_tokens".to_owned(), limit.into());
        }
        serde_json::Value::Object(fields)
    }
}

impl std::fmt::Display for Defaults {
    /// One line for the log of a start.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shown = SETTINGS.iter().filter_map(|setting| {
            (setting.show)(&self.sampling).map(|value| format!(" {}={value}", setting.field))
        });
        let length = self.length(None).0.map(|limit| format!(" max_tokens={limit}"));
        let fields: String = shown.chain(length).collect();
        if fields.is_empty() {
            f.write_str("a field a request leaves out is the engine's own")?;
        } else {
            write!(f, "a field a request leaves out:{fields}")?;
        }
        match self.max_tokens_limit {
            Some(limit) => write!(f, "; no reply is longer than {limit} tokens"),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests;
