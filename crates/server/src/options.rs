//! The options of one model: what `ares --model` takes on its command line
//! for the model it serves, and what a section of a preset file holds under
//! the same names. Both go through [`lookup`] and [`set`], so there is one
//! vocabulary and one check of every value.

use std::path::PathBuf;

use ares_cuda::Precision;
use ares_tokenizer::Json;

use crate::ReasoningFormat;
use crate::defaults::{self, Conflict, Defaults};
use crate::plan::{self, Drafter, LoadOptions, SplitKind};

/// What is set for one model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelOptions {
    pub load: LoadOptions,
    /// How the reasoning of a reply is returned when a request does not say.
    pub reasoning_format: ReasoningFormat,
    /// What a request gets for the fields it leaves out, and the most tokens
    /// of a reply.
    pub defaults: Defaults,
}

impl Default for ModelOptions {
    /// The built-in defaults: everything left to the devices and the model,
    /// and the thinking of earlier turns stays in the prompt.
    fn default() -> Self {
        let mut load = LoadOptions::new("");
        set_chat_variable(&mut load, "preserve_reasoning", Json::Bool(true));
        Self { load, reasoning_format: ReasoningFormat::default(), defaults: Defaults::default() }
    }
}

impl ModelOptions {
    /// Whether the options hold together, each being a value its option
    /// takes: [`set`] checks one value, this the combinations.
    pub fn check(&self) -> Result<(), Conflict> {
        self.defaults.check()
    }
}

/// The options of the server itself, which no model has: they stand on the
/// command line (and in its other forms) and not in a section of a preset
/// file. Their names without the dashes.
pub const SERVER: &[&str] = &[
    "host",
    "port",
    "models-preset",
    "config",
    "help",
    "api-key",
    "api-key-file",
    "allowed-hosts",
    "cors-origins",
    "timeout",
    "log-format",
    "log-file",
    "log-text",
];

/// How an option is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `--name value`; in a preset file `name = value`.
    Value,
    /// A switch without a value on the command line, which this spelling
    /// turns on (`true`) or off (`false`); in a preset file the name that
    /// turns it on, with `true` or `false`. [`set`] takes that word.
    Switch(bool),
}

/// A value an option does not take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invalid {
    pub flag: &'static str,
    pub value: String,
    pub expected: &'static str,
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "option '{}': '{}' is not {}", self.flag, self.value, self.expected)
    }
}

/// The option of a model that is written `name` on the command line
/// (`--context`, `-rea`): its one name, under which [`set`] takes it, and how
/// it is written. `None` for anything else.
pub fn lookup(name: &str) -> Option<(&'static str, Kind)> {
    Some(match name {
        "--model" => ("--model", Kind::Value),
        "--devices" => ("--devices", Kind::Value),
        "--split" => ("--split", Kind::Value),
        "--shares" => ("--shares", Kind::Value),
        "--context" => ("--context", Kind::Value),
        "--parallel" => ("--parallel", Kind::Value),
        "--kv" => ("--kv", Kind::Value),
        "--ram" => ("--ram", Kind::Value),
        "--cache" => ("--cache", Kind::Value),
        "--cache-size" => ("--cache-size", Kind::Value),
        "--precision" => ("--precision", Kind::Value),
        "--drafters" => ("--drafters", Kind::Value),
        "--window" => ("--window", Kind::Value),
        "--draft-vocab" => ("--draft-vocab", Kind::Value),
        "--chat-var" => ("--chat-var", Kind::Value),
        "--chat-template-file" => ("--chat-template-file", Kind::Value),
        "--queue" => ("--queue", Kind::Value),
        "-rea" | "--reasoning" => ("--reasoning", Kind::Value),
        "--reasoning-format" => ("--reasoning-format", Kind::Value),
        "--reasoning-effort" => ("--reasoning-effort", Kind::Value),
        "--reasoning-preserve" => ("--reasoning-preserve", Kind::Switch(true)),
        "--no-reasoning-preserve" => ("--reasoning-preserve", Kind::Switch(false)),
        // The defaults of a request's fields, under the names of the fields.
        other => (defaults::lookup(other)?, Kind::Value),
    })
}

/// The options of a model under their one names, as [`lookup`] gives them.
pub fn names() -> impl Iterator<Item = &'static str> {
    [
        "--model",
        "--devices",
        "--split",
        "--shares",
        "--context",
        "--parallel",
        "--kv",
        "--ram",
        "--cache",
        "--cache-size",
        "--precision",
        "--drafters",
        "--window",
        "--draft-vocab",
        "--chat-var",
        "--chat-template-file",
        "--queue",
        "--reasoning",
        "--reasoning-format",
        "--reasoning-effort",
        "--reasoning-preserve",
    ]
    .into_iter()
    .chain(defaults::flags())
}

/// Whether the option may be given several times, each adding to the others.
pub fn repeats(flag: &str) -> bool {
    flag == "--chat-var"
}

fn invalid(flag: &'static str, value: &str, expected: &'static str) -> Invalid {
    Invalid { flag, value: value.to_owned(), expected }
}

fn number<T: std::str::FromStr>(
    flag: &'static str,
    value: &str,
    expected: &'static str,
) -> Result<T, Invalid> {
    value.trim().parse().map_err(|_| invalid(flag, value, expected))
}

fn list<T: std::str::FromStr>(
    flag: &'static str,
    value: &str,
    expected: &'static str,
) -> Result<Vec<T>, Invalid> {
    value.split(',').map(|item| number(flag, item, expected)).collect()
}

/// The model drafters of `--drafters`: names separated by commas. `mtp` is
/// the model's own draft block, `none` (alone) is no drafter, anything else is
/// the file of a block drafter.
fn drafters(value: &str) -> Result<Vec<Drafter>, Invalid> {
    let expected = "mtp, none, or a drafter file; several separated by commas";
    if value.trim() == "none" {
        return Ok(Vec::new());
    }
    let mut list = Vec::new();
    for name in value.split(',').map(str::trim) {
        let drafter = match name {
            "" | "none" => return Err(invalid("--drafters", value, expected)),
            "mtp" => Drafter::Mtp,
            file => Drafter::Block(PathBuf::from(file)),
        };
        if list.contains(&drafter) {
            return Err(invalid("--drafters", value, expected));
        }
        list.push(drafter);
    }
    Ok(list)
}

/// A variable of the chat template: a JSON value where the text is one
/// (`false`, `3`, `"x"`), the text itself otherwise (`medium`).
fn chat_variable(value: &str) -> Result<(String, Json), Invalid> {
    let expected = "NAME=VALUE";
    let (name, value) =
        value.split_once('=').ok_or_else(|| invalid("--chat-var", value, expected))?;
    if name.is_empty() {
        return Err(invalid("--chat-var", value, expected));
    }
    let value = Json::parse(value).unwrap_or_else(|_| Json::String(value.to_owned()));
    Ok((name.to_owned(), value))
}

/// Sets a variable of the chat template; the later setting of a name wins, as
/// on any command line.
fn set_chat_variable(load: &mut LoadOptions, name: &str, value: Json) {
    load.chat_variables.retain(|(known, _)| known != name);
    load.chat_variables.push((name.to_owned(), value));
}

/// Sets the option `flag` (a name [`lookup`] gave) to `value`. A later
/// setting replaces an earlier one, which is what gives the command line its
/// place over a preset's section, and that its place over the global section.
pub fn set(options: &mut ModelOptions, flag: &'static str, value: &str) -> Result<(), Invalid> {
    let load = &mut options.load;
    match flag {
        "--model" => load.model = PathBuf::from(value),
        "--devices" => {
            load.devices = Some(list("--devices", value, "a list of device numbers (0,1)")?);
        }
        "--split" => {
            load.split = Some(match value {
                "single" => SplitKind::Single,
                "tensor" => SplitKind::Tensor,
                "layer" => SplitKind::Layer,
                _ => return Err(invalid("--split", value, "single, tensor or layer")),
            });
        }
        "--shares" => {
            let expected = "a list of positive numbers (0.625,0.375)";
            let shares: Vec<f64> = list("--shares", value, expected)?;
            if shares.iter().any(|share| !share.is_finite() || *share <= 0.0) {
                return Err(invalid("--shares", value, expected));
            }
            load.shares = Some(shares);
        }
        "--context" => {
            let expected = "a number of tokens, at least 1";
            let context: u32 = number("--context", value, expected)?;
            if context == 0 {
                return Err(invalid("--context", value, expected));
            }
            load.context = Some(context);
        }
        "--parallel" => {
            let expected = "a number of sequences, at least 1";
            let parallel: u32 = number("--parallel", value, expected)?;
            if parallel == 0 {
                return Err(invalid("--parallel", value, expected));
            }
            load.parallel = Some(parallel);
        }
        "--kv" => {
            let expected = "a format (f16, q8, q7, q6, q5, q4) or two, for K and V (q8,f16)";
            let formats = value
                .split(',')
                .map(|name| plan::cache_format(name.trim()))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("--kv", value, expected))?;
            load.kv = Some(match formats.as_slice() {
                [both] => (*both, *both),
                [k, v] => (*k, *v),
                _ => return Err(invalid("--kv", value, expected)),
            });
        }
        "--ram" => {
            let expected = "a size such as 16G or 512M";
            load.ram =
                Some(plan::parse_size(value).ok_or_else(|| invalid("--ram", value, expected))?);
        }
        "--cache" => {
            if value.trim().is_empty() {
                return Err(invalid("--cache", value, "a directory"));
            }
            load.cache = Some(PathBuf::from(value));
        }
        "--cache-size" => {
            let expected = "a size such as 64G, at least 1G";
            let size = plan::parse_size(value).filter(|size| *size >= 1 << 30);
            load.cache_size = Some(size.ok_or_else(|| invalid("--cache-size", value, expected))?);
        }
        "--precision" => {
            load.precision = match value {
                "fast" => Precision::Fast,
                "exact" => Precision::Exact,
                _ => return Err(invalid("--precision", value, "fast or exact")),
            };
        }
        "--drafters" => load.drafters = Some(drafters(value)?),
        "--window" => {
            let expected = "a number of rows, at least 2";
            let rows: u32 = number("--window", value, expected)?;
            if rows < 2 {
                return Err(invalid("--window", value, expected));
            }
            load.window = Some(rows);
        }
        "--draft-vocab" => load.draft_vocabulary = Some(PathBuf::from(value)),
        "--chat-template-file" => {
            if value.trim().is_empty() {
                return Err(invalid("--chat-template-file", value, "a file"));
            }
            load.chat_template = Some(PathBuf::from(value));
        }
        "--queue" => {
            load.queue = Some(number("--queue", value, "a number of requests, 0 or more")?);
        }
        "--chat-var" => {
            let (name, value) = chat_variable(value)?;
            set_chat_variable(load, &name, value);
        }
        "--reasoning" => match value {
            "on" => set_chat_variable(load, "enable_thinking", Json::Bool(true)),
            "off" => set_chat_variable(load, "enable_thinking", Json::Bool(false)),
            "auto" => load.chat_variables.retain(|(name, _)| name != "enable_thinking"),
            _ => return Err(invalid("--reasoning", value, "on, off or auto")),
        },
        "--reasoning-format" => {
            options.reasoning_format = ReasoningFormat::from_name(value).ok_or_else(|| {
                invalid("--reasoning-format", value, "none, auto, deepseek or deepseek-legacy")
            })?;
        }
        "--reasoning-effort" => {
            if value.is_empty() {
                return Err(invalid("--reasoning-effort", value, "a level, or default"));
            }
            if value == "default" {
                load.chat_variables.retain(|(name, _)| name != "reasoning_effort");
            } else {
                set_chat_variable(load, "reasoning_effort", Json::String(value.to_owned()));
            }
        }
        "--reasoning-preserve" => {
            let on = match value {
                "true" => true,
                "false" => false,
                _ => return Err(invalid("--reasoning-preserve", value, "true or false")),
            };
            set_chat_variable(load, "preserve_reasoning", Json::Bool(on));
        }
        other => defaults::set(&mut options.defaults, other, value)?,
    }
    Ok(())
}
