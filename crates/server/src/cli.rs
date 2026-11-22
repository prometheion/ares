//! The command line. A dozen flags do not justify a parser library.

use std::ffi::OsString;
use std::path::PathBuf;

use ares_server::ReasoningFormat;
use ares_server::defaults::Defaults;
use ares_server::options::{self, Kind, ModelOptions};
use ares_server::plan::LoadOptions;

pub const USAGE: &str = "\
ares: inference server with an OpenAI-compatible API

USAGE:
    ares --model <PATH> [OPTIONS]
    ares --models-preset <FILE> [OPTIONS]
    ares --config <FILE> [OPTIONS]

Every option is also a variable of the environment, ARES_ and its name in
capitals with underscores (ARES_TOP_P=0.9 for --top-p), and a line of the file
of --config, its name without the dashes (top-p = 0.9; a switch takes true or
false, as in a preset file). The command line holds over the environment, and
the environment over the file; with a preset file all three hold over its
sections. Variables without the prefix are never read.

OPTIONS:
        --config <FILE>        A file of options, one on each line: name = value;
                               lines that begin with # or ; are comments
                               [default: ARES_CONFIG, else none]
        --model <PATH>         GGUF model file to load and serve (any part of a split model)
        --models-preset <FILE> Serve several models from a preset file: an INI file
                               with one section for each model, named by the id
                               requests use, and a section [*] for all of them. A
                               section holds the options below under their names
                               (context = 32768), and aliases, load-on-startup and
                               idle-unload. A model is loaded when a request names
                               it and unloaded when another needs its room or when
                               it was idle for its idle-unload. An option given
                               here on the command line holds for every model
        --host <ADDR>          Address to listen on [default: 127.0.0.1]
        --port <PORT>          Port to listen on, 0 picks a free one [default: 8000]
        --timeout <SECONDS>    How long a client may take to send a request, and to
                               take what one write of an answer sent; then the
                               connection is closed and its request ended. A reply
                               the engine takes long for is never cut [default: 600]

    Who may use the server.
        --api-key <KEYS>       Keys a request must carry one of, separated by commas,
                               as 'Authorization: Bearer <key>' or 'x-api-key: <key>';
                               /health stays open [default: none, no key is asked for]
        --api-key-file <FILE>  A file of more keys, one on each line; lines that
                               begin with # are comments
        --allowed-hosts <LIST> Names in a request's Host that the server answers to
                               besides its local ones (localhost, addresses), separated
                               by commas, or * for any. A server on a local address
                               without keys answers only to its local names, against
                               web pages that rebind a name of theirs to it; with these
                               names the check holds for every address and with keys
                               too [default: the local names on a local address without
                               keys, else any]
        --cors-origins <LIST>  Origins whose web pages may call the server from a browser
                               and read its answers (CORS), such as https://chat.example.org,
                               separated by commas, or * for any. Without keys a request
                               from a page of any other site is refused, unless the
                               page is local (localhost, an address) [default: none]

    Where the model goes. What is left out is decided from the devices and the
    model, and the log says how.
        --devices <LIST>       CUDA devices to use, by the driver's index: 0,1
                               [default: all]
        --split <KIND>         single: one device holds the model
                               tensor: every device holds a slice of every block
                               layer:  every device holds a range of blocks
                               [default: single if one device holds the model, else tensor]
        --shares <LIST>        Each device's share of the weights under a split:
                               0.625,0.375 [default: by the memory bandwidth the
                               devices report, as far as their memory allows]

    Sequence state.
        --context <TOKENS>     Tokens of sequence state held for all sequences
                               together; one request holds at most this or the
                               model's own context [default: what the devices'
                               free memory holds]
        --parallel <N>         Sequences that run at once; each takes memory for
                               its recurrent state, further requests wait
                               [default: 2]
        --kv <FORMAT>          Storage format of K and V: f16, q8, q7, q6, q5 or q4,
                               or two names (K,V). The formats from q7 on store a
                               tile of 128 cells relative to its mean and keep the
                               first tile and the newest cells exact; Qwen3.8-Flash-Next
                               takes f16 and q8 only [default: q8]

    Host memory.
        --ram <SIZE>           The most RAM the whole engine may use, such as 16G or
                               512M, for all loaded models together; ram in a
                               model's section of a preset file is that model's
                               part of it, and a model without one takes what is
                               left. What the engine needs to run is taken from it
                               first (with --cache that includes one checkpoint and
                               the pages and checkpoints state passes through on its
                               way to the drive and back); the rest is for the
                               experts of a mixture of experts that have no slot in
                               VRAM, and then for the prompt cache in RAM: the K and
                               V that left the devices and the checkpoints of
                               recurrent state that let a conversation continue
                               from what was computed. Below the limit the engine
                               holds what the machine has available less a small
                               headroom, and gives memory back when other programs
                               need it, the prompt cache first [default: all the
                               machine has]

    Prompt cache on SSD, below the one in RAM.
        --cache <DIR>          A directory in which the engine keeps the state of
                               what it has computed (K and V, and checkpoints of
                               recurrent state), so that a prompt that starts with
                               it resumes from there after the state has left RAM,
                               after the model was unloaded and after a restart.
                               Texts of 2048 tokens and more are written when they
                               leave RAM, and what RAM and the devices hold when
                               the model is unloaded or the server stops; a store
                               fits one model file with its formats and split, and
                               the directory may be shared by all models. A
                               mixture of experts keeps the usage profile of its
                               experts there, which fills its tiers at the next
                               start [default: none, and nothing is written; the
                               profiles are then kept in ~/.cache/ares]
        --cache-size <SIZE>    The most that directory may take, such as 64G; what
                               was used least recently goes first [default: 32G]

    Speed against exactness.
        --precision <KIND>     fast:  8-bit activations where they are faster
                               exact: f16 products everywhere, same text whatever
                                      else runs in a step
                               [default: fast]
        --drafters <LIST>      The model drafters that take part, such as mtp (the
                               MTP block in the model file) or none; a file
                               name is a drafter from a file of its own, an
                               MTP block or a DFlash2 drafter, by the
                               architecture the file names. The
                               n-gram drafter takes part with them and
                               never alone; with none the
                               server does not speculate. The length of a draft
                               and whose it is are chosen for every step from
                               the price of a step and the acceptance measured
                               [default: all the model has: mtp if the file has
                               an MTP block]
        -b, --batch-size <ROWS>
                               Most rows of prompt in one step while a prompt is
                               read alone. Above --ubatch-size only for a mixture
                               of experts (Qwen3.8-Flash-Next): its expert slots
                               in VRAM lend the step's scratch for its time, and
                               every expert is read once a step, not once per
                               --ubatch-size rows; a request that arrives waits
                               for the step in flight [default: 8192 for a mixture
                               of experts, as far as the slots can lend it, else
                               --ubatch-size]
        -ub, --ubatch-size <ROWS>
                               Most rows of a step in the memory the model keeps
                               for good, and of a piece of prompt while another
                               request waits [default: what a thirty-second of the
                               smaller device's memory holds for a mixture of
                               experts, 512 for the 27B]
        --window <ROWS>        Most rows of one sequence's window, its newest
                               token and the drafts verified with it. The room
                               for them is VRAM (about 5 MiB a row for the 27B,
                               for every sequence) [default: 8, the rows one
                               launch of the few-row weight kernels takes]
        --draft-vocab <FILE>   Token ids (one per line) the model drafters (mtp
                               and dflash) may propose: a shorter list makes a
                               draft cheaper [default: every token whose head
                               row is on the drafters' device]
        --expert-prune <RULE:RANK>
                               For a mixture of experts: what a step does with a
                               pick whose expert only the SSD holds, from the RANK-th
                               pick of a row on (1 the strongest), instead of
                               waiting for the drive. skip leaves it out, router
                               takes the router's next best expert that VRAM or RAM
                               holds, similar the one most like it there (from a
                               table of the model file kept beside the usage
                               profile). Changes the output; similar:8 touches the
                               8th to 10th of ten picks [default: off]

    Images.
        --mmproj <FILE>        The projector the model reads images through: the
                               mmproj GGUF of its image tower beside the model file.
                               Images come in requests as data URLs (image_url,
                               Anthropic image blocks, input_image); remote URLs
                               are not fetched [default: none, images are refused]
        --mmproj-weights <ram|vram>
                               Where the projector's weights stay between images:
                               ram, page-locked inside --ram and uploaded block by
                               block for every new image, or vram, which the
                               context gives up (about 0.9 GiB) [default: ram]
        --image-max-tokens <ROWS>
                               The most rows (tokens) one image becomes, 8 to 4096;
                               a larger image is shrunk to fit. The VRAM of the
                               tower's scratch follows it (about 340 MiB at 4096)
                               [default: 4096, as llama.cpp]

    Prompts.
        --chat-var <NAME=VALUE>  A variable of the model's chat template, such as
                               add_vision_id=true; may be given several times
        --chat-template-file <FILE>
                               A chat template (Jinja) that takes the place of the
                               model file's, for every conversation
                               [default: the model file's]

    The line of requests.
        --queue <N>            The most requests that may wait for a sequence while
                               --parallel run; one more is refused with 503 and
                               Retry-After [default: no limit, a request waits
                               until it runs]

    Thinking, under the names llama-server gives these. A request can change
    each of them for itself: enable_thinking and the other variables in
    chat_template_kwargs, reasoning_effort, reasoning_format.
    -rea, --reasoning <on|off|auto>
                               Whether the model thinks before it answers: auto
                               leaves it to the chat template [default: auto]
        --reasoning-format <FORMAT>
                               none: the thinking stays in content, tags and all
                               auto, deepseek, deepseek-legacy: the thinking goes
                               to reasoning_content, in streamed deltas too
                               [default: auto]
        --reasoning-effort <LEVEL>
                               A level the chat template knows, such as low or
                               medium, or default to leave it to the template
                               [default: default]
        --reasoning-preserve, --no-reasoning-preserve
                               Whether the thinking of earlier turns stays in the
                               prompt, as a client sends it back in
                               reasoning_content: it must for a conversation to
                               continue from what was computed [default: on]
    A budget for the thinking tokens is not built: the engine cannot put the
    tokens that end the thinking into a sequence.

    What a request gets for a field it leaves out. An option has the name of
    the field with dashes (--top-p for top_p), and a request's own value always
    holds. The defaults named here are the engine's own.
        --temperature <T>      0 picks the most probable token [default: 0]
        --top-k <N>            Keep the N most probable tokens, 0 for all
                               [default: 0]
        --top-p <P>            Keep the most probable tokens that hold P of the
                               probability, 1 for all [default: 1]
        --min-p <P>            Drop tokens less than P times as probable as the
                               best, 0 for none [default: 0]
        --typical-p <P>        Locally typical sampling, 1 is off [default: 1]
        --top-n-sigma <N>      Keep the tokens within N standard deviations of
                               the best logit, 0 is off [default: 0]
        --xtc-probability <P>, --xtc-threshold <P>
                               XTC: the chance that the most probable tokens are
                               removed, and the probability from which a token is
                               one of them [default: 0 (off), 0.1]
        --dynatemp-range <R>, --dynatemp-exponent <E>
                               A temperature that moves by R around
                               --temperature with the entropy [default: 0 (off),
                               1]
        --repeat-penalty <N>   Lowers the logits of the tokens that stand in the
                               last --repeat-last-n tokens by this factor, 1 is
                               off [default: 1]
        --presence-penalty <N>, --frequency-penalty <N>
                               Subtracted from the logits of those tokens, once
                               and for every time they stand there [default: 0]
        --repeat-last-n <N>    The tokens the three penalties look at, at most
                               1024, 0 switches them off [default: 64]
        --dry-multiplier <N>, --dry-base <N>, --dry-allowed-length <N>,
        --dry-penalty-last-n <N>
                               DRY, the penalty for a token that would repeat a
                               sequence [default: 0 (off), 1.75, 2, 64]
        --dry-sequence-breakers <JSON>
                               The strings a repeated sequence does not reach
                               across, as a JSON array
                               [default: [\"\\n\", \":\", \"\\\"\", \"*\"]]
        --mirostat <0|1|2>, --mirostat-tau <N>, --mirostat-eta <N>
                               Mirostat in place of the other samplers, its
                               target surprise and its rate [default: 0 (off),
                               5, 0.1]
        --samplers <LIST>      The samplers that apply and their order, by name
                               [default: penalties,dry,top_n_sigma,top_k,typ_p,
                               top_p,min_p,xtc,temperature]
        --seed <N>             The seed of a request that names none, which then
                               draws the same for the same prompt; a request
                               asks for a draw of its own with seed -1
                               [default: a new one for every request]

    The length of a reply.
        --max-tokens <N>       The limit of tokens for a request that names
                               none (max_tokens, max_completion_tokens,
                               max_output_tokens, n_predict) [default: none, a
                               reply ends when the model ends it or the context
                               is full]
        --max-tokens-limit <N> The most tokens of any reply: a request that asks
                               for more gets this many, and its response says so
                               in the header x-ares-max-tokens-limit; a request
                               that names no limit gets --max-tokens or, without
                               it, this [default: none]

    -h, --help                 Print this help

    The log.
        --log-format <text|json>
                               text: a line for people; json: one JSON object for
                               every line, for collectors [default: text]
        --log-file <FILE>      Append the log to this file [default: standard error]
        --log-text             The prompt and the reply of every request go into the
                               log as well; without it the log holds neither

The log level is set with the RUST_LOG environment variable (default: info).
ARES_LIBCUDA names the driver library to load instead of the system's libcuda.so.1.
";

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub host: String,
    pub port: u16,
    /// `--timeout`: how long a client may take to send a request, and to
    /// take what one write of an answer sent.
    pub timeout: std::time::Duration,
    /// Where the log goes and what it holds.
    pub log: LogOptions,
    /// Who may use the server, as given: the keys come with those of the
    /// file, read when the server starts.
    pub access: AccessOptions,
    /// The preset file of `--models-preset`; without one `load.model` is the
    /// one model to serve.
    pub presets: Option<PathBuf>,
    pub load: LoadOptions,
    pub reasoning_format: ReasoningFormat,
    pub defaults: Defaults,
    /// The options of a model as they were given, by their names and in
    /// their order: with a preset file they are set again over every section.
    pub given: Vec<(&'static str, String)>,
}

/// The options of `--log-format`, `--log-file` and `--log-text`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogOptions {
    /// Every line a JSON object.
    pub json: bool,
    /// Appended to this file instead of standard error.
    pub file: Option<PathBuf>,
    /// The prompt and the reply of every request too.
    pub text: bool,
}

/// The options of `--api-key`, `--api-key-file`, `--allowed-hosts` and
/// `--cors-origins`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccessOptions {
    pub keys: Vec<String>,
    pub key_file: Option<PathBuf>,
    /// `None`: not given; `Some(None)`: `*`, any name.
    pub hosts: Option<Option<Vec<String>>>,
    pub origins: Vec<String>,
}

impl AccessOptions {
    /// What the server lets in when it listens on an address that is
    /// `local` or not: the keys of the option and of the file, and the names
    /// it answers to.
    pub fn access(&self, local: bool) -> Result<ares_server::Access, String> {
        let mut keys = self.keys.clone();
        if let Some(file) = &self.key_file {
            let text = std::fs::read_to_string(file).map_err(|e| {
                format!("cannot read the keys of --api-key-file '{}': {e}", file.display())
            })?;
            for (number, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if line.contains(char::is_whitespace) {
                    return Err(format!(
                        "{}:{}: a key has no spaces: write one key on each line",
                        file.display(),
                        number + 1
                    ));
                }
                keys.push(line.to_owned());
            }
            if keys.is_empty() {
                return Err(format!(
                    "--api-key-file '{}' holds no key: a server that is to want keys needs one",
                    file.display()
                ));
            }
        }
        // Without names of its own, a server on a local address answers to
        // its local names, unless keys decide who comes in.
        let hosts = match &self.hosts {
            Some(given) => given.clone(),
            None if local && keys.is_empty() => Some(Vec::new()),
            None => None,
        };
        Ok(ares_server::Access { keys, hosts, origins: self.origins.clone() })
    }
}

/// A list of words separated by commas, none of them empty or holding a
/// space.
fn words(flag: &'static str, value: &str, expected: &'static str) -> Result<Vec<String>, CliError> {
    let words: Vec<String> = value.split(',').map(|word| word.trim().to_owned()).collect();
    if words.iter().any(|word| word.is_empty() || word.contains(char::is_whitespace)) {
        return Err(CliError::InvalidValue { flag, value: value.to_owned(), expected });
    }
    Ok(words)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Run(Box<Options>),
    Help,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum CliError {
    #[error("unknown option '{0}'")]
    UnknownOption(String),
    #[error("option '{0}' needs a value")]
    MissingValue(&'static str),
    #[error("'{0}' is not a valid port: use a number from 0 to 65535")]
    InvalidPort(String),
    #[error("option '{0}' does not take a value")]
    UnexpectedValue(&'static str),
    #[error("'{0}' is not valid UTF-8; options must be text")]
    NotUnicode(String),
    #[error("{place}: {message}")]
    Placed { place: String, message: String },
    #[error("option '{flag}': '{value}' is not {expected}")]
    InvalidValue { flag: &'static str, value: String, expected: &'static str },
    #[error("option '{0}' is given twice")]
    Repeated(&'static str),
    #[error("options '{}' and '{}': {reason}", flags[0], flags[1])]
    Conflict { flags: [&'static str; 2], reason: String },
    #[error(
        "there is nothing to serve: name a model file with --model, or a preset file of several \
         models with --models-preset"
    )]
    NoModel,
    #[error(
        "--model and --models-preset are both given: a preset file names its models in its \
         sections"
    )]
    ModelAndPresets,
}

impl From<options::Invalid> for CliError {
    fn from(e: options::Invalid) -> Self {
        CliError::InvalidValue { flag: e.flag, value: e.value, expected: e.expected }
    }
}

impl From<ares_server::defaults::Conflict> for CliError {
    fn from(e: ares_server::defaults::Conflict) -> Self {
        CliError::Conflict { flags: e.flags, reason: e.reason }
    }
}

/// The text of a value.
fn text(value: OsString) -> Result<String, CliError> {
    value.into_string().map_err(|v| CliError::NotUnicode(v.to_string_lossy().into_owned()))
}

/// The options of the server itself, as [`options::lookup`] has those of a
/// model: their one name and how they are written. `--config` is not among
/// them: it names where the others come from.
fn server_option(name: &str) -> Option<(&'static str, Kind)> {
    let flag = match name {
        "--host" => "--host",
        "--port" => "--port",
        "--models-preset" => "--models-preset",
        "--api-key" => "--api-key",
        "--api-key-file" => "--api-key-file",
        "--allowed-hosts" => "--allowed-hosts",
        "--cors-origins" => "--cors-origins",
        "--timeout" => "--timeout",
        "--log-format" => "--log-format",
        "--log-file" => "--log-file",
        "--log-text" => return Some(("--log-text", Kind::Switch(true))),
        _ => return None,
    };
    Some((flag, Kind::Value))
}

/// Every option under its one name, server's and model's: what the
/// environment is asked for.
fn every_option() -> impl Iterator<Item = &'static str> {
    options::SERVER
        .iter()
        .filter_map(|name| server_option(&format!("--{name}")).map(|(flag, _)| flag))
        .chain(options::names())
}

/// The variable of the environment that gives `flag`: `ARES_` and its name in
/// capitals, with underscores (`--top-p`: `ARES_TOP_P`).
pub fn variable_of(flag: &str) -> String {
    format!("ARES_{}", flag.trim_start_matches('-').replace('-', "_").to_ascii_uppercase())
}

/// What is being read while options are put together.
struct Reading {
    host: String,
    port: u16,
    timeout: u64,
    log: LogOptions,
    access: AccessOptions,
    presets: Option<PathBuf>,
    model: ModelOptions,
    given: Vec<(&'static str, String)>,
}

impl Reading {
    /// Sets `flag` (a name of [`server_option`] or [`options::lookup`]) to
    /// `value`; a switch takes `true` or `false`. A later setting replaces an
    /// earlier one.
    fn set(&mut self, flag: &'static str, value: String) -> Result<(), CliError> {
        let invalid = |expected| CliError::InvalidValue { flag, value: value.clone(), expected };
        match flag {
            "--host" => self.host = value,
            "--port" => {
                self.port = value.trim().parse().map_err(|_| CliError::InvalidPort(value))?;
            }
            "--models-preset" => self.presets = Some(PathBuf::from(value)),
            "--log-format" => {
                self.log.json = match value.as_str() {
                    "text" => false,
                    "json" => true,
                    _ => return Err(invalid("text or json")),
                };
            }
            "--log-file" => {
                if value.trim().is_empty() {
                    return Err(invalid("a file"));
                }
                self.log.file = Some(PathBuf::from(value));
            }
            "--log-text" => {
                self.log.text = match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(invalid("true or false")),
                };
            }
            "--timeout" => {
                self.timeout = value
                    .trim()
                    .parse()
                    .ok()
                    .filter(|seconds| *seconds >= 1)
                    .ok_or_else(|| invalid("a number of seconds, at least 1"))?;
            }
            "--api-key" => self.access.keys = words(flag, &value, "keys separated by commas")?,
            "--api-key-file" => {
                if value.trim().is_empty() {
                    return Err(invalid("a file"));
                }
                self.access.key_file = Some(PathBuf::from(value));
            }
            "--allowed-hosts" => {
                let names = words(flag, &value, "host names separated by commas, or *")?;
                self.access.hosts = Some((!names.iter().any(|name| name == "*")).then_some(names));
            }
            "--cors-origins" => {
                let expected = "origins such as https://chat.example.org separated by commas, or *";
                let origins = words(flag, &value, expected)?;
                if origins.iter().any(|o| o != "*" && o != "null" && !o.contains("://")) {
                    return Err(invalid(expected));
                }
                self.access.origins =
                    origins.into_iter().map(|o| o.trim_end_matches('/').to_owned()).collect();
            }
            model => {
                options::set(&mut self.model, model, &value)?;
                self.given.push((model, value));
            }
        }
        Ok(())
    }
}

/// One option as one of the three places gives it.
struct Given {
    flag: &'static str,
    value: String,
    /// Where it stands, for an error: `None` on the command line.
    place: Option<String>,
}

/// Parses the arguments after the program name, with nothing from the
/// environment. `--name value` and `--name=value` are both accepted.
#[cfg(test)]
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, CliError> {
    parse_with(args, &|_| None)
}

/// Parses the arguments after the program name, the variables of
/// `environment` (asked by name) and the configuration file that one of the
/// two names. The command line holds over the environment, and the
/// environment over the file.
pub fn parse_with(
    args: impl IntoIterator<Item = OsString>,
    environment: &dyn Fn(&str) -> Option<OsString>,
) -> Result<Command, CliError> {
    let mut line = Vec::new();
    let mut config: Option<(PathBuf, Option<String>)> = None;
    let mut seen: Vec<&'static str> = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let whole =
            arg.to_str().ok_or_else(|| CliError::NotUnicode(arg.to_string_lossy().into_owned()))?;
        let (name, inline) = match whole.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(OsString::from(value))),
            _ => (whole, None),
        };
        let mut value = |flag: &'static str| {
            inline.clone().or_else(|| args.next()).ok_or(CliError::MissingValue(flag))
        };
        let (flag, value) = match name {
            "-h" | "--help" => {
                if inline.is_some() {
                    return Err(CliError::UnexpectedValue("--help"));
                }
                return Ok(Command::Help);
            }
            "--config" => {
                if config.is_some() {
                    return Err(CliError::Repeated("--config"));
                }
                config = Some((PathBuf::from(value("--config")?), None));
                continue;
            }
            _ => match server_option(name).or(options::lookup(name)) {
                Some((flag, Kind::Value)) => (flag, text(value(flag)?)?),
                Some((flag, Kind::Switch(on))) => {
                    if inline.is_some() {
                        return Err(CliError::UnexpectedValue(flag));
                    }
                    (flag, on.to_string())
                }
                None => return Err(CliError::UnknownOption(whole.to_owned())),
            },
        };
        // A flag that is given twice is a mistake more often than a choice.
        if !options::repeats(flag) {
            if seen.contains(&flag) {
                return Err(CliError::Repeated(flag));
            }
            seen.push(flag);
        }
        line.push(Given { flag, value, place: None });
    }

    // The environment: a variable for every option, and one for the file.
    let mut environ = Vec::new();
    for flag in every_option() {
        let name = variable_of(flag);
        if let Some(value) = environment(&name) {
            let value = text(value)?;
            environ.push(Given { flag, value, place: Some(name) });
        }
    }
    if config.is_none()
        && let Some(file) = environment("ARES_CONFIG")
    {
        config = Some((PathBuf::from(file), Some("ARES_CONFIG".to_owned())));
    }
    let file = match &config {
        Some((path, _)) => read_config(path)?,
        None => Vec::new(),
    };

    let mut reading = Reading {
        host: "127.0.0.1".to_owned(),
        port: 8000,
        timeout: 600,
        log: LogOptions::default(),
        access: AccessOptions::default(),
        presets: None,
        model: ModelOptions::default(),
        given: Vec::new(),
    };
    for Given { flag, value, place } in file.into_iter().chain(environ).chain(line) {
        reading.set(flag, value).map_err(|error| match place {
            Some(place) => CliError::Placed { place, message: error.to_string() },
            None => error,
        })?;
    }
    let Reading { host, port, timeout, log, access, presets, model, given } = reading;
    let named = !model.load.model.as_os_str().is_empty();
    match (named, &presets) {
        (false, None) => return Err(CliError::NoModel),
        (true, Some(_)) => return Err(CliError::ModelAndPresets),
        _ => {}
    }
    // What is given holds together; with a preset file the sections are
    // checked with it when the file is read.
    model.check()?;
    let ModelOptions { load, reasoning_format, defaults } = model;
    let options = Options {
        host,
        port,
        timeout: std::time::Duration::from_secs(timeout),
        log,
        access,
        presets,
        load,
        reasoning_format,
        defaults,
        given,
    };
    Ok(Command::Run(Box::new(options)))
}

/// The options of a configuration file: lines `name = value` with the names
/// of the command line's options without the dashes, as the keys of a preset
/// file are written (a switch takes `true` or `false`; `#` and `;` begin a
/// comment line). A name stands once, unless its option may be given
/// several times.
fn read_config(path: &std::path::Path) -> Result<Vec<Given>, CliError> {
    let place = |line: usize| match line {
        0 => path.display().to_string(),
        line => format!("{}:{line}", path.display()),
    };
    let fail = |line: usize, message: String| CliError::Placed { place: place(line), message };
    let text = std::fs::read_to_string(path)
        .map_err(|e| fail(0, format!("cannot read the configuration file: {e}")))?;
    let mut given: Vec<Given> = Vec::new();
    let mut lines: Vec<(&'static str, usize)> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let (number, line) = (index + 1, raw.trim());
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            return Err(fail(
                number,
                "a configuration file has no sections: the models of several go in a preset \
                 file (models-preset = <file.ini>)"
                    .into(),
            ));
        }
        let Some((name, value)) = line.split_once('=') else {
            return Err(fail(number, format!("'{line}' is not an option: write name = value")));
        };
        let (name, value) = (name.trim(), value.trim());
        let dashed = format!("--{name}");
        let (flag, value) = match server_option(&dashed).or(options::lookup(&dashed)) {
            Some((flag, Kind::Value)) => (flag, value.to_owned()),
            Some((flag, Kind::Switch(true))) => match value {
                "true" | "false" => (flag, value.to_owned()),
                other => {
                    return Err(fail(number, format!("{name} = '{other}': write true or false")));
                }
            },
            Some((flag, Kind::Switch(false))) => {
                return Err(fail(
                    number,
                    format!(
                        "'{name}' is the command line's way to switch {0} off: write {0} = false",
                        flag.trim_start_matches('-')
                    ),
                ));
            }
            None if name == "config" => {
                return Err(fail(number, "a configuration file does not name another".into()));
            }
            None if name == "help" => {
                return Err(fail(number, "'help' is not an option to keep in a file".into()));
            }
            None => {
                return Err(fail(
                    number,
                    format!("Ares has no option '{name}' ('ares --help' lists the options)"),
                ));
            }
        };
        if !options::repeats(flag)
            && let Some((_, first)) = lines.iter().find(|(known, _)| *known == flag)
        {
            return Err(fail(number, format!("'{name}' stands twice (first at line {first})")));
        }
        lines.push((flag, number));
        given.push(Given { flag, value, place: Some(place(number)) });
    }
    Ok(given)
}

#[cfg(test)]
mod tests {
    use ares_cuda::{CacheFormat, Precision, PruneRule};
    use ares_server::plan::{Drafter, ExpertPrune, SplitKind};
    use ares_tokenizer::Json;

    use super::*;

    fn parse_strs(args: &[&str]) -> Result<Command, CliError> {
        parse(args.iter().map(OsString::from))
    }

    fn run(args: &[&str]) -> Options {
        match parse_strs(args) {
            Ok(Command::Run(options)) => *options,
            other => panic!("{args:?}: {other:?}"),
        }
    }

    #[test]
    fn a_model_alone_leaves_everything_else_to_the_devices() {
        let options = run(&["--model", "m.gguf"]);
        assert_eq!(options.host, "127.0.0.1");
        assert_eq!(options.port, 8000);
        // The thinking of earlier turns stays in the prompt, as in llama-server.
        let mut expected = LoadOptions::new("m.gguf");
        expected.chat_variables = vec![("preserve_reasoning".to_owned(), Json::Bool(true))];
        assert_eq!(options.load, expected);
        assert_eq!(options.reasoning_format, ReasoningFormat::Auto);
        // A field a request leaves out is the engine's own, and no reply has a limit.
        assert_eq!(options.defaults, Defaults::default());
    }

    #[test]
    fn without_a_model_there_is_nothing_to_serve() {
        assert_eq!(parse_strs(&[]), Err(CliError::NoModel));
        assert_eq!(parse_strs(&["--port", "1"]), Err(CliError::NoModel));
        assert!(CliError::NoModel.to_string().contains("--model"));
    }

    #[test]
    fn both_value_forms_are_accepted() {
        let spaced = run(&["--host", "0.0.0.0", "--port", "9000", "--model", "/models/a b.gguf"]);
        let inline = run(&["--host=0.0.0.0", "--port=9000", "--model=/models/a b.gguf"]);
        assert_eq!(spaced, inline);
        assert_eq!(spaced.host, "0.0.0.0");
        assert_eq!(spaced.port, 9000);
        assert_eq!(spaced.load.model, PathBuf::from("/models/a b.gguf"));
    }

    #[test]
    fn every_load_option_is_read() {
        let options = run(&[
            "--model",
            "m.gguf",
            "--devices",
            "1,0",
            "--split",
            "tensor",
            "--shares",
            "0.625,0.375",
            "--context",
            "4096",
            "--parallel",
            "3",
            "--kv",
            "f16",
            "--ram",
            "12G",
            "--precision",
            "exact",
            "--drafters",
            "none",
            "--window",
            "12",
            "--chat-var",
            "enable_thinking=false",
            "--chat-var=reasoning_effort=medium",
            "--chat-var",
            "depth=3",
        ]);
        let load = options.load;
        assert_eq!(load.devices, Some(vec![1, 0]));
        assert_eq!(load.split, Some(SplitKind::Tensor));
        assert_eq!(load.shares, Some(vec![0.625, 0.375]));
        assert_eq!(load.context, Some(4096));
        assert_eq!(load.parallel, Some(3));
        assert_eq!(load.kv, Some((CacheFormat::F16, CacheFormat::F16)));
        assert_eq!(load.ram, Some(12 << 30));
        assert_eq!(load.precision, Precision::Exact);
        assert_eq!(load.drafters, Some(Vec::new()));
        assert_eq!(load.window, Some(12));
        assert_eq!(
            load.chat_variables,
            vec![
                ("preserve_reasoning".to_owned(), Json::Bool(true)),
                ("enable_thinking".to_owned(), Json::Bool(false)),
                ("reasoning_effort".to_owned(), Json::String("medium".into())),
                ("depth".to_owned(), Json::Int(3)),
            ]
        );

        let load = run(&["--model=m", "--kv", "q8,f16", "--draft-vocab", "v.txt"]).load;
        assert_eq!(load.kv, Some((CacheFormat::Q8, CacheFormat::F16)));
        assert_eq!(load.draft_vocabulary, Some(PathBuf::from("v.txt")));
        assert_eq!(run(&["--model=m", "--split=layer"]).load.split, Some(SplitKind::Layer));
        // Pruning of experts only the SSD holds: off unless asked for.
        assert_eq!(run(&["--model=m"]).load.expert_prune, None);
        assert_eq!(run(&["--model=m", "--expert-prune", "off"]).load.expert_prune, None);
        assert_eq!(
            run(&["--model=m", "--expert-prune", "similar:8"]).load.expert_prune,
            Some(ExpertPrune { rule: PruneRule::Similar, from_rank: 8 })
        );
        assert_eq!(
            run(&["--model=m", "--expert-prune=skip:1"]).load.expert_prune,
            Some(ExpertPrune { rule: PruneRule::Skip, from_rank: 1 })
        );
        // The prompt cache on SSD: off without a directory, and a size of its own.
        let load = run(&["--model=m"]).load;
        assert_eq!((load.cache, load.cache_size), (None, None));
        let load = run(&["--model=m", "--cache", "/var/cache/ares", "--cache-size=64G"]).load;
        assert_eq!(load.cache, Some(PathBuf::from("/var/cache/ares")));
        assert_eq!(load.cache_size, Some(64 << 30));
        // The formats that keep exact cells are accepted, for K and V apart.
        let load = run(&["--model=m", "--kv", "q5,q4"]).load;
        assert_eq!(load.kv, Some((CacheFormat::Q5, CacheFormat::Q4)));
        assert_eq!(
            run(&["--model=m", "--kv=q7"]).load.kv,
            Some((CacheFormat::Q7, CacheFormat::Q7))
        );
    }

    #[test]
    fn the_defaults_of_requests_are_options_under_the_names_of_the_fields() {
        let options = run(&[
            "--model=m",
            "--temperature",
            "0.7",
            "--top-p=0.8",
            "--top-k",
            "20",
            "--min-p=0",
            "--presence-penalty",
            "1.5",
            "--repeat-penalty=1.05",
            "--repeat-last-n",
            "256",
            "--dry-sequence-breakers",
            r#"["\n", ":"]"#,
            "--samplers=top_k,top_p,temperature",
            "--mirostat=0",
            "--seed",
            "42",
            "--max-tokens",
            "4096",
            "--max-tokens-limit=8192",
        ]);
        let defaults = options.defaults;
        let sampling = &defaults.sampling;
        assert_eq!((sampling.temperature, sampling.top_p), (Some(0.7), Some(0.8)));
        assert_eq!((sampling.top_k, sampling.min_p), (Some(20), Some(0.0)));
        assert_eq!((sampling.presence_penalty, sampling.repeat_penalty), (Some(1.5), Some(1.05)));
        assert_eq!((sampling.repeat_last_n, sampling.mirostat), (Some(256), Some(0)));
        assert_eq!(sampling.dry_sequence_breakers, Some(vec!["\n".to_owned(), ":".to_owned()]));
        assert_eq!(sampling.samplers.as_ref().map(Vec::len), Some(3));
        assert_eq!(sampling.seed, Some(42));
        assert_eq!((defaults.max_tokens, defaults.max_tokens_limit), (Some(4096), Some(8192)));
        // What nobody named stays open.
        assert_eq!((sampling.frequency_penalty, sampling.typical_p), (None, None));

        // With a preset file they are kept, to hold over every section.
        let options = run(&["--models-preset=m.ini", "--temperature=0.2", "--max-tokens", "64"]);
        assert_eq!(
            options.given,
            vec![("--temperature", "0.2".to_owned()), ("--max-tokens", "64".to_owned())]
        );

        // A value outside its range, and two options that cannot hold together.
        for (args, flag) in [
            (&["--temperature", "-1"][..], "--temperature"),
            (&["--top-p", "0"], "--top-p"),
            (&["--mirostat", "3"], "--mirostat"),
            (&["--repeat-last-n", "2000"], "--repeat-last-n"),
            (&["--samplers", "top_k,penalties"], "--samplers"),
            (&["--dry-sequence-breakers", "none"], "--dry-sequence-breakers"),
            (&["--seed", "-1"], "--seed"),
            (&["--max-tokens", "0"], "--max-tokens"),
            (&["--max-tokens-limit", "-1"], "--max-tokens-limit"),
        ] {
            match parse_strs(args) {
                Err(CliError::InvalidValue { flag: named, .. }) => assert_eq!(named, flag),
                other => panic!("{args:?}: {other:?}"),
            }
        }
        for first in [&["--model=m"][..], &["--models-preset=m.ini"]] {
            let args = [first, &["--max-tokens-limit", "100", "--max-tokens=200"]].concat();
            let error = parse_strs(&args).unwrap_err();
            assert_eq!(
                error.to_string(),
                "options '--max-tokens' and '--max-tokens-limit': max-tokens = 200 is more than \
                 max-tokens-limit = 100: the length of a reply whose request names none cannot \
                 be above the most a request gets"
            );
        }
        assert_eq!(
            parse_strs(&["--model=m", "--top-k=1", "--top-k", "2"]),
            Err(CliError::Repeated("--top-k"))
        );
        // The names of llama-server are not options of Ares.
        for name in ["--temp", "--n-predict", "-n", "--typical", "-s"] {
            assert_eq!(
                parse_strs(&["--model=m", name, "1"]),
                Err(CliError::UnknownOption(name.to_owned()))
            );
        }
    }

    #[test]
    fn drafters_are_named_one_by_one_and_a_file_is_a_block_drafter() {
        let drafters = |list: &str| run(&["--model=m", "--drafters", list]).load.drafters;
        assert_eq!(drafters("mtp"), Some(vec![Drafter::Mtp]));
        assert_eq!(drafters("none"), Some(Vec::new()));
        assert_eq!(
            drafters("mtp, /models/dflash2.gguf"),
            Some(vec![Drafter::Mtp, Drafter::File(PathBuf::from("/models/dflash2.gguf"))])
        );
        assert_eq!(run(&["--model=m"]).load.drafters, None, "all that are available");
        for list in ["", "none,mtp", "mtp,mtp", "mtp,,x"] {
            assert!(
                matches!(
                    parse_strs(&["--model=m", "--drafters", list]),
                    Err(CliError::InvalidValue { flag: "--drafters", .. })
                ),
                "{list:?}"
            );
        }
    }

    #[test]
    fn a_chat_variable_given_again_replaces_the_earlier_one() {
        let load = run(&["--model=m", "--chat-var", "a=1", "--chat-var", "a=two"]).load;
        let a = ("a".to_owned(), Json::String("two".into()));
        assert_eq!(load.chat_variables.last(), Some(&a));
        assert_eq!(load.chat_variables.iter().filter(|(name, _)| name == "a").count(), 1);
    }

    fn variable(load: &LoadOptions, name: &str) -> Option<Json> {
        load.chat_variables.iter().find(|(known, _)| known == name).map(|(_, value)| value.clone())
    }

    #[test]
    fn the_thinking_flags_set_chat_variables_under_the_names_of_llama_server() {
        let off = run(&["--model=m", "--reasoning", "off", "--reasoning-effort", "low"]).load;
        assert_eq!(variable(&off, "enable_thinking"), Some(Json::Bool(false)));
        assert_eq!(variable(&off, "reasoning_effort"), Some(Json::String("low".into())));
        let on = run(&["--model=m", "-rea", "on"]).load;
        assert_eq!(variable(&on, "enable_thinking"), Some(Json::Bool(true)));
        // `auto` and `default` leave it to the template, over an earlier --chat-var.
        let auto = run(&[
            "--model=m",
            "--chat-var",
            "enable_thinking=false",
            "--chat-var",
            "reasoning_effort=high",
            "--reasoning=auto",
            "--reasoning-effort=default",
        ])
        .load;
        assert_eq!(variable(&auto, "enable_thinking"), None);
        assert_eq!(variable(&auto, "reasoning_effort"), None);
        // The default keeps the thinking of earlier turns; the flags say the same both ways.
        assert_eq!(
            variable(&run(&["--model=m"]).load, "preserve_reasoning"),
            Some(Json::Bool(true))
        );
        let dropped = run(&["--model=m", "--no-reasoning-preserve"]).load;
        assert_eq!(variable(&dropped, "preserve_reasoning"), Some(Json::Bool(false)));
        let kept = run(&["--model=m", "--reasoning-preserve"]).load;
        assert_eq!(variable(&kept, "preserve_reasoning"), Some(Json::Bool(true)));
    }

    #[test]
    fn the_reasoning_format_is_one_of_the_four_names() {
        for (name, format) in [
            ("none", ReasoningFormat::None),
            ("auto", ReasoningFormat::Auto),
            ("deepseek", ReasoningFormat::Deepseek),
            ("deepseek-legacy", ReasoningFormat::DeepseekLegacy),
        ] {
            assert_eq!(run(&["--model=m", "--reasoning-format", name]).reasoning_format, format);
        }
        for (args, flag) in [
            (&["--reasoning", "maybe"][..], "--reasoning"),
            (&["--reasoning-format", "think"], "--reasoning-format"),
            (&["--reasoning-effort", ""], "--reasoning-effort"),
        ] {
            match parse_strs(args) {
                Err(CliError::InvalidValue { flag: named, .. }) => assert_eq!(named, flag),
                other => panic!("{args:?}: {other:?}"),
            }
        }
        assert_eq!(
            parse_strs(&["--reasoning-preserve=yes"]),
            Err(CliError::UnexpectedValue("--reasoning-preserve"))
        );
        assert_eq!(
            parse_strs(&["--model=m", "--reasoning-preserve", "--no-reasoning-preserve"]),
            Err(CliError::Repeated("--reasoning-preserve"))
        );
    }

    #[test]
    fn help_wins_wherever_it_stands_before_an_error() {
        assert_eq!(parse_strs(&["--help"]), Ok(Command::Help));
        assert_eq!(parse_strs(&["--port", "1", "-h"]), Ok(Command::Help));
    }

    #[test]
    fn mistakes_say_what_to_change() {
        assert_eq!(parse_strs(&["--prot", "1"]), Err(CliError::UnknownOption("--prot".into())));
        assert_eq!(parse_strs(&["serve"]), Err(CliError::UnknownOption("serve".into())));
        assert_eq!(parse_strs(&["--port"]), Err(CliError::MissingValue("--port")));
        assert_eq!(parse_strs(&["--model"]), Err(CliError::MissingValue("--model")));
        assert_eq!(parse_strs(&["--port", "70000"]), Err(CliError::InvalidPort("70000".into())));
        assert_eq!(parse_strs(&["--port=http"]), Err(CliError::InvalidPort("http".into())));
        assert_eq!(parse_strs(&["--help=yes"]), Err(CliError::UnexpectedValue("--help")));
        assert_eq!(parse_strs(&["--model=m", "--model=n"]), Err(CliError::Repeated("--model")));
        for (args, flag) in [
            (&["--split", "diagonal"][..], "--split"),
            (&["--devices", "0,one"], "--devices"),
            (&["--devices", ""], "--devices"),
            (&["--shares", "0.5,-0.5"], "--shares"),
            (&["--shares", "0.5,nan"], "--shares"),
            (&["--context", "0"], "--context"),
            (&["--context", "lots"], "--context"),
            (&["--parallel", "0"], "--parallel"),
            (&["--window", "1"], "--window"),
            (&["--window", "wide"], "--window"),
            (&["--kv", "q3"], "--kv"),
            (&["--kv", "q8,q8,q8"], "--kv"),
            (&["--ram", "lots"], "--ram"),
            (&["--ram", "-1"], "--ram"),
            (&["--cache", ""], "--cache"),
            (&["--cache-size", "plenty"], "--cache-size"),
            (&["--cache-size", "100M"], "--cache-size"),
            (&["--precision", "sloppy"], "--precision"),
            (&["--drafters", "none,mtp"], "--drafters"),
            (&["--chat-var", "novalue"], "--chat-var"),
            (&["--chat-var", "=x"], "--chat-var"),
            (&["--expert-prune", "similar"], "--expert-prune"),
            (&["--expert-prune", "skip:0"], "--expert-prune"),
            (&["--expert-prune", "drop:8"], "--expert-prune"),
        ] {
            match parse_strs(args) {
                Err(CliError::InvalidValue { flag: named, .. }) => assert_eq!(named, flag),
                other => panic!("{args:?}: {other:?}"),
            }
        }
        let message = parse_strs(&["--split", "diagonal"]).unwrap_err().to_string();
        assert_eq!(message, "option '--split': 'diagonal' is not single, tensor or layer");
    }

    #[test]
    fn a_value_may_look_like_a_flag() {
        // `--model --odd.gguf` is a path, not a missing value: the flag asked for one.
        assert_eq!(run(&["--model", "--odd.gguf"]).load.model, PathBuf::from("--odd.gguf"));
    }

    /// A file of `text` in the temporary directory, named for the test.
    fn file(name: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("ares-cli-{}-{name}", std::process::id()));
        std::fs::write(&path, text).unwrap();
        path
    }

    fn with(args: &[&str], variables: &[(&str, &str)]) -> Result<Command, CliError> {
        let variables: Vec<(String, String)> =
            variables.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
        let environment = move |name: &str| {
            variables.iter().find(|(known, _)| known == name).map(|(_, v)| OsString::from(v))
        };
        parse_with(args.iter().map(OsString::from), &environment)
    }

    fn run_with(args: &[&str], variables: &[(&str, &str)]) -> Options {
        match with(args, variables) {
            Ok(Command::Run(options)) => *options,
            other => panic!("{args:?} {variables:?}: {other:?}"),
        }
    }

    #[test]
    fn every_option_is_also_a_variable_of_the_environment() {
        assert_eq!(variable_of("--top-p"), "ARES_TOP_P");
        assert_eq!(variable_of("--models-preset"), "ARES_MODELS_PRESET");
        let options = run_with(
            &[],
            &[
                ("ARES_MODEL", "/models/m.gguf"),
                ("ARES_CONTEXT", "4096"),
                ("ARES_TOP_P", "0.9"),
                ("ARES_REASONING_PRESERVE", "false"),
                ("ARES_CHAT_VAR", "depth=3"),
                ("ARES_PORT", "9000"),
                ("ARES_HOST", "0.0.0.0"),
                ("ARES_API_KEY", "k1,k2"),
                ("ARES_CORS_ORIGINS", "https://chat.example.org"),
            ],
        );
        assert_eq!(options.load.model, PathBuf::from("/models/m.gguf"));
        assert_eq!(options.load.context, Some(4096));
        assert_eq!(options.defaults.sampling.top_p, Some(0.9));
        assert_eq!(variable(&options.load, "preserve_reasoning"), Some(Json::Bool(false)));
        assert_eq!(variable(&options.load, "depth"), Some(Json::Int(3)));
        assert_eq!((options.host.as_str(), options.port), ("0.0.0.0", 9000));
        assert_eq!(options.access.keys, ["k1", "k2"]);
        assert_eq!(options.access.origins, ["https://chat.example.org"]);
        // A name without the prefix is never read: the shell here has CTX.
        let options = run_with(&["--model=m"], &[("CTX", "4096"), ("CONTEXT", "4096")]);
        assert_eq!(options.load.context, None);
        // A value a variable gives is checked like one of the command line.
        let error = with(&["--model=m"], &[("ARES_CONTEXT", "0")]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "ARES_CONTEXT: option '--context': '0' is not a number of tokens, at least 1"
        );
    }

    #[test]
    fn the_command_line_holds_over_the_environment_and_that_over_the_file() {
        let config = file("order.ini", "model = /m.gguf\ncontext = 1000\ntemperature = 0.5\n");
        let path = config.to_str().unwrap();
        let only_file = run_with(&["--config", path], &[]);
        assert_eq!(only_file.load.context, Some(1000));
        assert_eq!(only_file.defaults.sampling.temperature, Some(0.5));
        let environment = [("ARES_CONTEXT", "2000")];
        assert_eq!(run_with(&["--config", path], &environment).load.context, Some(2000));
        let options = run_with(&["--config", path, "--context", "3000"], &environment);
        assert_eq!(options.load.context, Some(3000));
        assert_eq!(options.defaults.sampling.temperature, Some(0.5), "what only the file says");
        // The file may also be named by the environment.
        let options = run_with(&[], &[("ARES_CONFIG", path)]);
        assert_eq!(options.load.context, Some(1000));
        // With a preset file the options of a model are kept in that order,
        // to hold over its sections.
        let config = file("given.ini", "models-preset = /p.ini\ncontext = 1000\n");
        let options = run_with(
            &["--config", config.to_str().unwrap(), "--context=3000"],
            &[("ARES_CONTEXT", "2000")],
        );
        assert_eq!(options.presets, Some(PathBuf::from("/p.ini")));
        let given = [("--context", "1000"), ("--context", "2000"), ("--context", "3000")];
        let given: Vec<_> = given.iter().map(|(f, v)| (*f, (*v).to_owned())).collect();
        assert_eq!(options.given, given);
    }

    #[test]
    fn a_configuration_file_has_the_keys_of_a_preset_file() {
        let config = file(
            "keys.ini",
            "; a comment\n# another\nmodel = /m.gguf\nhost = 0.0.0.0\nport = 8080\n\
             api-key-file = /etc/ares/keys\nchat-var = a=1\nchat-var = b=2\n\
             reasoning-preserve = false\nmax-tokens = 4096\n",
        );
        let options = run_with(&["--config", config.to_str().unwrap()], &[]);
        assert_eq!((options.host.as_str(), options.port), ("0.0.0.0", 8080));
        assert_eq!(options.access.key_file, Some(PathBuf::from("/etc/ares/keys")));
        assert_eq!(variable(&options.load, "a"), Some(Json::Int(1)));
        assert_eq!(variable(&options.load, "b"), Some(Json::Int(2)));
        assert_eq!(variable(&options.load, "preserve_reasoning"), Some(Json::Bool(false)));
        assert_eq!(options.defaults.max_tokens, Some(4096));

        for (text, says) in [
            ("model = /m.gguf\ncontxt = 1\n", ":2: Ares has no option 'contxt'"),
            (
                "model = /m.gguf\ncontext = 1\ncontext = 2\n",
                ":3: 'context' stands twice (first at line 2)",
            ),
            ("[m]\nmodel = /m.gguf\n", ":1: a configuration file has no sections"),
            ("model /m.gguf\n", ":1: 'model /m.gguf' is not an option"),
            (
                "model = /m.gguf\nno-reasoning-preserve = true\n",
                ":2: 'no-reasoning-preserve' is the command line's way",
            ),
            (
                "model = /m.gguf\nreasoning-preserve = yes\n",
                ":2: reasoning-preserve = 'yes': write true or false",
            ),
            ("config = /other.ini\n", ":1: a configuration file does not name another"),
            (
                "model = /m.gguf\ncontext = 0\n",
                ":2: option '--context': '0' is not a number of tokens",
            ),
            ("model = /m.gguf\nport = http\n", ":2: 'http' is not a valid port"),
        ] {
            let config = file("bad.ini", text);
            let said = with(&["--config", config.to_str().unwrap()], &[]).unwrap_err().to_string();
            assert!(said.starts_with(config.to_str().unwrap()), "{said}");
            assert!(said.contains(says), "{text}: {said}");
        }
        let said = with(&["--config", "/nowhere/ares.ini"], &[]).unwrap_err().to_string();
        assert!(
            said.starts_with("/nowhere/ares.ini: cannot read the configuration file"),
            "{said}"
        );
        assert_eq!(with(&["--config=a", "--config=b"], &[]), Err(CliError::Repeated("--config")));
    }

    #[test]
    fn the_server_s_options_of_access() {
        let options = run(&[
            "--model=m",
            "--api-key",
            "one, two",
            "--allowed-hosts=gpu.lan,ares.example",
            "--cors-origins",
            "https://chat.example.org/,https://chat.example",
        ]);
        let access = &options.access;
        assert_eq!(access.keys, ["one", "two"]);
        assert_eq!(access.hosts, Some(Some(vec!["gpu.lan".to_owned(), "ares.example".to_owned()])));
        assert_eq!(access.origins, ["https://chat.example.org", "https://chat.example"]);
        assert_eq!(run(&["--model=m", "--allowed-hosts", "*"]).access.hosts, Some(None));
        for (args, flag) in [
            (&["--api-key", "a,,b"][..], "--api-key"),
            (&["--api-key", "a b"], "--api-key"),
            (&["--allowed-hosts", ""], "--allowed-hosts"),
            (&["--cors-origins", "chat.example.org"], "--cors-origins"),
            (&["--api-key-file", " "], "--api-key-file"),
        ] {
            match parse_strs(args) {
                Err(CliError::InvalidValue { flag: named, .. }) => assert_eq!(named, flag),
                other => panic!("{args:?}: {other:?}"),
            }
        }

        // What the server lets in: a local address without keys answers to
        // its local names only; keys decide on their own; names that are
        // given hold everywhere.
        let open = AccessOptions::default();
        assert_eq!(open.access(true).unwrap().hosts, Some(Vec::new()));
        assert_eq!(open.access(false).unwrap().hosts, None);
        let keyed = AccessOptions { keys: vec!["k".into()], ..AccessOptions::default() };
        assert_eq!(keyed.access(true).unwrap().hosts, None);
        let named = AccessOptions { hosts: Some(Some(vec!["gpu.lan".into()])), ..keyed.clone() };
        assert_eq!(named.access(false).unwrap().hosts, Some(vec!["gpu.lan".to_owned()]));
        let keys = file("keys.txt", "# the keys\nfrom-file\n\n  second  \n");
        let both = AccessOptions { key_file: Some(keys), ..keyed };
        assert_eq!(both.access(true).unwrap().keys, ["k", "from-file", "second"]);
        let empty = file("empty-keys.txt", "# none\n");
        let error = AccessOptions { key_file: Some(empty), ..AccessOptions::default() }
            .access(true)
            .unwrap_err();
        assert!(error.contains("holds no key"), "{error}");
    }

    #[test]
    fn the_log_and_the_timeout_are_options_of_the_server() {
        let options = run(&["--model=m"]);
        assert_eq!(options.log, LogOptions::default());
        assert_eq!(options.timeout, std::time::Duration::from_secs(600));
        let options = run(&[
            "--model=m",
            "--log-format",
            "json",
            "--log-file=/var/log/ares.log",
            "--log-text",
            "--timeout",
            "30",
        ]);
        let log = LogOptions { json: true, file: Some("/var/log/ares.log".into()), text: true };
        assert_eq!(options.log, log);
        assert_eq!(options.timeout, std::time::Duration::from_secs(30));
        for (args, flag) in [
            (&["--log-format", "yaml"][..], "--log-format"),
            (&["--log-file", ""], "--log-file"),
            (&["--timeout", "0"], "--timeout"),
            (&["--timeout", "soon"], "--timeout"),
        ] {
            match parse_strs(args) {
                Err(CliError::InvalidValue { flag: named, .. }) => assert_eq!(named, flag),
                other => panic!("{args:?}: {other:?}"),
            }
        }
        assert_eq!(parse_strs(&["--log-text=yes"]), Err(CliError::UnexpectedValue("--log-text")));
        // A switch in the environment says true or false.
        let options = run_with(&["--model=m"], &[("ARES_LOG_TEXT", "true")]);
        assert!(options.log.text);
    }

    #[test]
    fn a_preset_file_takes_the_place_of_a_model_and_keeps_what_was_given() {
        let options = run(&["--models-preset", "models.ini", "--context", "4096", "-rea", "off"]);
        assert_eq!(options.presets, Some(PathBuf::from("models.ini")));
        assert_eq!(
            options.given,
            vec![("--context", "4096".to_owned()), ("--reasoning", "off".to_owned())]
        );
        assert_eq!(
            parse_strs(&["--models-preset", "m.ini", "--model", "m.gguf"]),
            Err(CliError::ModelAndPresets)
        );
        assert_eq!(
            run(&["--model=m", "--no-reasoning-preserve"]).given,
            vec![("--model", "m".to_owned()), ("--reasoning-preserve", "false".to_owned())]
        );
    }
}
