//! The command line. A dozen flags do not justify a parser library.

use std::ffi::OsString;
use std::path::PathBuf;

use ares_server::ini;
use ares_server::options::{self, Kind, ModelOptions, OPTIONS, Opt, Scope};

/// What `--help` prints before the options.
const HEAD: &str = "\
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
";

/// What `--help` prints after the options.
const TAIL: &str = "
The log level is set with the RUST_LOG environment variable (default: info).
ARES_LIBCUDA names the driver library to load instead of the system's
libcuda.so.1.
";

/// The column the help of an option starts in, and the width of a line.
const HELP_COLUMN: usize = 31;
const WIDTH: usize = 80;

/// What `ares --help` prints: the forms of the command, then every option of
/// [`OPTIONS`] in its group, with its help.
pub fn usage() -> String {
    let mut text = String::from(HEAD);
    for opt in OPTIONS {
        if let Some(heading) = opt.heading {
            text.push('\n');
            for line in wrap(heading, WIDTH - 4) {
                text.push_str(&format!("    {line}\n"));
            }
        }
        let indent = if opt.short.is_some() { 4 } else { 8 };
        let lead = format!("{}{}", " ".repeat(indent), spelling(opt));
        let mut lines = wrap(opt.help, WIDTH - HELP_COLUMN).into_iter();
        if lead.len() < HELP_COLUMN {
            let first = lines.next().unwrap_or_default();
            text.push_str(&format!("{lead:HELP_COLUMN$}{first}\n"));
        } else {
            text.push_str(&format!("{lead}\n"));
        }
        for line in lines {
            text.push_str(&format!("{:HELP_COLUMN$}{line}\n", ""));
        }
    }
    text.push_str(TAIL);
    text
}

/// How the help writes an option: its spellings and its value.
fn spelling(opt: &Opt) -> String {
    let mut names: Vec<&str> = opt.short.into_iter().chain([opt.flag]).collect();
    names.extend(opt.off);
    let names = names.join(", ");
    match opt.value {
        Some(value) => format!("{names} {value}"),
        None => names,
    }
}

/// `text` in lines of at most `width` characters, where they can be cut
/// between words, and a list wider than a line (a,b,c) after a comma; a `\n`
/// in it starts a line.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for part in text.split('\n') {
        let mut line = String::new();
        for mut word in part.split_whitespace() {
            while word.chars().count() > width {
                let used = if line.is_empty() { 0 } else { line.chars().count() + 1 };
                let room = width.saturating_sub(used);
                let cut = word
                    .match_indices(',')
                    .map(|(at, _)| at + 1)
                    .take_while(|end| word[..*end].chars().count() <= room)
                    .last();
                match cut {
                    Some(end) => {
                        if !line.is_empty() {
                            line.push(' ');
                        }
                        line.push_str(&word[..end]);
                        lines.push(std::mem::take(&mut line));
                        word = &word[end..];
                    }
                    None if !line.is_empty() => lines.push(std::mem::take(&mut line)),
                    // No comma to cut at: the word stands alone, wider than the line.
                    None => break,
                }
            }
            if word.is_empty() {
                continue;
            }
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines
}

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
    /// The preset file of `--models-preset`; without one `model` is the one
    /// model to serve.
    pub presets: Option<PathBuf>,
    pub model: ModelOptions,
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
    /// `--version`: print [`ares_server::BUILD_INFO`].
    Version,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum CliError {
    #[error("unknown option '{name}'{}", .hint.map(|hint| format!(": {hint}")).unwrap_or_default())]
    UnknownOption { name: String, hint: Option<&'static str> },
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

/// An argument that is no option: what Ares has for it, if it is one of
/// llama-server's.
fn unknown(name: &str) -> CliError {
    let hint = options::other_name(name.trim_start_matches('-'));
    CliError::UnknownOption { name: name.to_owned(), hint }
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
    /// Sets `flag` (the name of an option of the server or of a model) to
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
        let Some((opt, kind)) = options::find(name) else { return Err(unknown(whole)) };
        let flag = opt.flag;
        let value = match kind {
            Kind::Value => text(value(flag)?)?,
            Kind::Switch(on) => {
                if inline.is_some() {
                    return Err(CliError::UnexpectedValue(flag));
                }
                on.to_string()
            }
        };
        match flag {
            "--help" => return Ok(Command::Help),
            "--version" => return Ok(Command::Version),
            "--config" => {
                if config.is_some() {
                    return Err(CliError::Repeated("--config"));
                }
                config = Some((PathBuf::from(value), None));
                continue;
            }
            _ => {}
        }
        // A flag that is given twice is a mistake more often than a choice.
        if !opt.repeats {
            if seen.contains(&flag) {
                return Err(CliError::Repeated(flag));
            }
            seen.push(flag);
        }
        line.push(Given { flag, value, place: None });
    }

    // The environment: a variable for every option, and one for the file.
    let mut environ = Vec::new();
    for flag in OPTIONS.iter().filter(|opt| opt.scope != Scope::Command).map(|opt| opt.flag) {
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
    let timeout = std::time::Duration::from_secs(timeout);
    let options = Options { host, port, timeout, log, access, presets, model, given };
    Ok(Command::Run(Box::new(options)))
}

/// The options of a configuration file: lines `name = value` with the names
/// of the command line's options without the dashes, as the keys of a preset
/// file are written (the `ini` module; a switch takes `true` or `false`).
fn read_config(path: &std::path::Path) -> Result<Vec<Given>, CliError> {
    let place = |line: usize| match line {
        0 => path.display().to_string(),
        line => format!("{}:{line}", path.display()),
    };
    let fail = |line: usize, message: String| CliError::Placed { place: place(line), message };
    let text = std::fs::read_to_string(path)
        .map_err(|e| fail(0, format!("cannot read the configuration file: {e}")))?;
    let file = ini::parse(&text).map_err(|(line, message)| fail(line, message))?;
    if let Some(section) = file.sections.first() {
        return Err(fail(
            section.line,
            "a configuration file has no sections: the models of several go in a preset file \
             (models-preset = <file.ini>)"
                .into(),
        ));
    }
    let mut given: Vec<Given> = Vec::new();
    for key in file.keys {
        let opt = options::of_key(&key.name, "'ares --help' lists the options")
            .map_err(|message| fail(key.line, message))?;
        match opt.flag {
            "--config" => {
                return Err(fail(key.line, "a configuration file does not name another".into()));
            }
            "--help" | "--version" => {
                let message = format!("'{}' is not an option to keep in a file", opt.key());
                return Err(fail(key.line, message));
            }
            _ => {}
        }
        let value = opt.value_of_key(&key.value).map_err(|message| fail(key.line, message))?;
        given.push(Given { flag: opt.flag, value, place: Some(place(key.line)) });
    }
    Ok(given)
}

#[cfg(test)]
mod tests {
    use ares_cuda::{CacheFormat, PruneRule};
    use ares_server::ReasoningFormat;
    use ares_server::defaults::Defaults;
    use ares_server::plan::{Drafter, ExpertPrune, LoadOptions, SplitKind};
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
        assert_eq!(options.model.load, expected);
        assert_eq!(options.model.reasoning_format, ReasoningFormat::Auto);
        // A field a request leaves out is the engine's own, and no reply has a limit.
        assert_eq!(options.model.defaults, Defaults::default());
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
        assert_eq!(spaced.model.load.model, PathBuf::from("/models/a b.gguf"));
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
        let load = options.model.load;
        assert_eq!(load.devices, Some(vec![1, 0]));
        assert_eq!(load.split, Some(SplitKind::Tensor));
        assert_eq!(load.shares, Some(vec![0.625, 0.375]));
        assert_eq!(load.context, Some(4096));
        assert_eq!(load.parallel, Some(3));
        assert_eq!(load.kv, Some((CacheFormat::F16, CacheFormat::F16)));
        assert_eq!(load.ram, Some(12 << 30));
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

        let load = run(&["--model=m", "--kv", "q8,f16", "--draft-vocab", "v.txt"]).model.load;
        assert_eq!(load.kv, Some((CacheFormat::Q8, CacheFormat::F16)));
        assert_eq!(load.draft_vocabulary, Some(PathBuf::from("v.txt")));
        assert_eq!(run(&["--model=m", "--split=layer"]).model.load.split, Some(SplitKind::Layer));
        // Pruning of experts only the SSD holds: off unless asked for.
        assert_eq!(run(&["--model=m"]).model.load.expert_prune, None);
        assert_eq!(run(&["--model=m", "--expert-prune", "off"]).model.load.expert_prune, None);
        assert_eq!(
            run(&["--model=m", "--expert-prune", "similar:8"]).model.load.expert_prune,
            Some(ExpertPrune { rule: PruneRule::Similar, from_rank: 8 })
        );
        assert_eq!(
            run(&["--model=m", "--expert-prune=skip:1"]).model.load.expert_prune,
            Some(ExpertPrune { rule: PruneRule::Skip, from_rank: 1 })
        );
        // The prompt cache on SSD: off without a directory, and a size of its own.
        let load = run(&["--model=m"]).model.load;
        assert_eq!((load.cache, load.cache_size), (None, None));
        let load = run(&["--model=m", "--cache", "/var/cache/ares", "--cache-size=64G"]).model.load;
        assert_eq!(load.cache, Some(PathBuf::from("/var/cache/ares")));
        assert_eq!(load.cache_size, Some(64 << 30));
        // The formats that keep exact cells are accepted, for K and V apart.
        let load = run(&["--model=m", "--kv", "q5,q4"]).model.load;
        assert_eq!(load.kv, Some((CacheFormat::Q5, CacheFormat::Q4)));
        assert_eq!(
            run(&["--model=m", "--kv=q7"]).model.load.kv,
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
        let defaults = options.model.defaults;
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
            assert_eq!(parse_strs(&["--model=m", name, "1"]), Err(unknown(name)));
        }
    }

    #[test]
    fn drafters_are_named_one_by_one_and_a_file_is_a_block_drafter() {
        let drafters = |list: &str| run(&["--model=m", "--drafters", list]).model.load.drafters;
        assert_eq!(drafters("mtp"), Some(vec![Drafter::Mtp]));
        assert_eq!(drafters("none"), Some(Vec::new()));
        assert_eq!(
            drafters("mtp, /models/dflash2.gguf"),
            Some(vec![Drafter::Mtp, Drafter::File(PathBuf::from("/models/dflash2.gguf"))])
        );
        assert_eq!(run(&["--model=m"]).model.load.drafters, None, "all that are available");
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
        let load = run(&["--model=m", "--chat-var", "a=1", "--chat-var", "a=two"]).model.load;
        let a = ("a".to_owned(), Json::String("two".into()));
        assert_eq!(load.chat_variables.last(), Some(&a));
        assert_eq!(load.chat_variables.iter().filter(|(name, _)| name == "a").count(), 1);
    }

    fn variable(load: &LoadOptions, name: &str) -> Option<Json> {
        load.chat_variables.iter().find(|(known, _)| known == name).map(|(_, value)| value.clone())
    }

    #[test]
    fn the_thinking_flags_set_chat_variables_under_the_names_of_llama_server() {
        let off = run(&["--model=m", "--reasoning", "off", "--reasoning-effort", "low"]).model.load;
        assert_eq!(variable(&off, "enable_thinking"), Some(Json::Bool(false)));
        assert_eq!(variable(&off, "reasoning_effort"), Some(Json::String("low".into())));
        let on = run(&["--model=m", "-rea", "on"]).model.load;
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
        .model
        .load;
        assert_eq!(variable(&auto, "enable_thinking"), None);
        assert_eq!(variable(&auto, "reasoning_effort"), None);
        // The default keeps the thinking of earlier turns; the flags say the same both ways.
        assert_eq!(
            variable(&run(&["--model=m"]).model.load, "preserve_reasoning"),
            Some(Json::Bool(true))
        );
        let dropped = run(&["--model=m", "--no-reasoning-preserve"]).model.load;
        assert_eq!(variable(&dropped, "preserve_reasoning"), Some(Json::Bool(false)));
        let kept = run(&["--model=m", "--reasoning-preserve"]).model.load;
        assert_eq!(variable(&kept, "preserve_reasoning"), Some(Json::Bool(true)));
    }

    #[test]
    fn the_reasoning_format_is_one_of_the_two_names() {
        for (name, format) in [("none", ReasoningFormat::None), ("auto", ReasoningFormat::Auto)] {
            assert_eq!(
                run(&["--model=m", "--reasoning-format", name]).model.reasoning_format,
                format
            );
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
    fn version_wins_like_help_and_stays_out_of_files() {
        assert_eq!(parse_strs(&["--version"]), Ok(Command::Version));
        assert_eq!(parse_strs(&["--port", "1", "-V"]), Ok(Command::Version));
        assert_eq!(parse_strs(&["--version=yes"]), Err(CliError::UnexpectedValue("--version")));
        // The first of the two that stands first is done.
        assert_eq!(parse_strs(&["-V", "--help"]), Ok(Command::Version));
        // The environment has no variable for it.
        let run = run_with(&["--model", "m.gguf"], &[("ARES_VERSION", "true")]);
        assert_eq!(run.model.load.model, PathBuf::from("m.gguf"));
        let config = file("version.conf", "model = m.gguf\nversion = true\n");
        let said = parse_strs(&["--config", config.to_str().unwrap()]).unwrap_err().to_string();
        assert!(said.ends_with(":2: 'version' is not an option to keep in a file"), "{said}");
        std::fs::remove_file(config).unwrap();
    }

    #[test]
    fn the_build_is_named_by_version_commit_and_date() {
        let info = ares_server::BUILD_INFO;
        let rest = info.strip_prefix(&format!("ares {} (", env!("CARGO_PKG_VERSION"))).unwrap();
        let (commit, date) = rest.strip_suffix(')').unwrap().split_once(", ").unwrap();
        assert!(!commit.is_empty() && !commit.contains(char::is_whitespace), "{info}");
        let parts: Vec<&str> = date.split('-').collect();
        assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), [4, 2, 2], "{info}");
        assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()), "{info}");
    }

    #[test]
    fn mistakes_say_what_to_change() {
        assert_eq!(
            parse_strs(&["--prot", "1"]),
            Err(CliError::UnknownOption { name: "--prot".into(), hint: None })
        );
        assert_eq!(
            parse_strs(&["serve"]),
            Err(CliError::UnknownOption { name: "serve".into(), hint: None })
        );
        // A name of llama-server's is told what Ares has for it.
        let said = parse_strs(&["--model=m", "-c", "4096"]).unwrap_err().to_string();
        assert_eq!(
            said,
            "unknown option '-c': the context is 'context', in tokens for all sequences together"
        );
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
        assert_eq!(run(&["--model", "--odd.gguf"]).model.load.model, PathBuf::from("--odd.gguf"));
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
    fn the_help_names_every_option_and_no_other() {
        let help = usage();
        let named = |flag: &str| {
            help.match_indices(flag).any(|(at, _)| {
                let after = help[at + flag.len()..].chars().next();
                !after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '-')
            })
        };
        for opt in OPTIONS {
            for flag in [Some(opt.flag), opt.short, opt.off].into_iter().flatten() {
                assert!(named(flag), "--help does not name {flag}");
            }
        }
        // What the help mentions of an option is one.
        for (at, _) in help.match_indices("--") {
            let flag: String =
                help[at..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
            assert!(options::find(&flag).is_some(), "--help names {flag}, which is no option");
        }
        for line in help.lines() {
            assert!(line.chars().count() <= WIDTH, "a line wider than {WIDTH}: {line}");
        }
    }

    #[test]
    fn a_list_wider_than_a_line_is_cut_after_a_comma() {
        let list = "[default: aaaa,bbbb,cccc,dddd,eeee]";
        assert_eq!(wrap(list, 20), ["[default: aaaa,bbbb,", "cccc,dddd,eeee]"]);
        let after = "x [default: aaaa,bbbb,cccc,dddd]";
        assert_eq!(wrap(after, 16), ["x [default:", "aaaa,bbbb,cccc,", "dddd]"]);
        // A word without a comma stands alone, wider than the line.
        assert_eq!(wrap("x abcdefghijkl y", 5), ["x", "abcdefghijkl", "y"]);
    }

    #[test]
    fn the_help_has_its_options_in_columns() {
        let help = usage();
        let lines: Vec<&str> = help.lines().collect();
        let at = |start: &str| lines.iter().position(|line| line.starts_with(start)).unwrap();
        let context = at("        --context <TOKENS>");
        assert_eq!(
            &lines[context][HELP_COLUMN..],
            "Tokens of sequence state held for all sequences"
        );
        assert!(lines[context + 1].starts_with(&format!("{:HELP_COLUMN$}together", "")));
        let reasoning = at("    -rea, --reasoning <on|off|auto>");
        assert_eq!(lines[reasoning].trim_end(), "    -rea, --reasoning <on|off|auto>");
        assert!(lines[reasoning + 1].starts_with(&format!("{:HELP_COLUMN$}Whether", "")));
        let split = at("        --split <KIND>");
        assert!(lines[split].ends_with("single: one device holds the model"));
        assert!(lines[split + 1].trim_start().starts_with("tensor: "));
        assert!(help.contains("\n    Who may use the server.\n"), "{help}");
        assert!(help.starts_with(HEAD) && help.ends_with(TAIL));
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
        assert_eq!(options.model.load.model, PathBuf::from("/models/m.gguf"));
        assert_eq!(options.model.load.context, Some(4096));
        assert_eq!(options.model.defaults.sampling.top_p, Some(0.9));
        assert_eq!(variable(&options.model.load, "preserve_reasoning"), Some(Json::Bool(false)));
        assert_eq!(variable(&options.model.load, "depth"), Some(Json::Int(3)));
        assert_eq!((options.host.as_str(), options.port), ("0.0.0.0", 9000));
        assert_eq!(options.access.keys, ["k1", "k2"]);
        assert_eq!(options.access.origins, ["https://chat.example.org"]);
        // A name without the prefix is never read: the shell here has CTX.
        let options = run_with(&["--model=m"], &[("CTX", "4096"), ("CONTEXT", "4096")]);
        assert_eq!(options.model.load.context, None);
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
        assert_eq!(only_file.model.load.context, Some(1000));
        assert_eq!(only_file.model.defaults.sampling.temperature, Some(0.5));
        let environment = [("ARES_CONTEXT", "2000")];
        assert_eq!(run_with(&["--config", path], &environment).model.load.context, Some(2000));
        let options = run_with(&["--config", path, "--context", "3000"], &environment);
        assert_eq!(options.model.load.context, Some(3000));
        assert_eq!(
            options.model.defaults.sampling.temperature,
            Some(0.5),
            "what only the file says"
        );
        // The file may also be named by the environment.
        let options = run_with(&[], &[("ARES_CONFIG", path)]);
        assert_eq!(options.model.load.context, Some(1000));
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
        assert_eq!(variable(&options.model.load, "a"), Some(Json::Int(1)));
        assert_eq!(variable(&options.model.load, "b"), Some(Json::Int(2)));
        assert_eq!(variable(&options.model.load, "preserve_reasoning"), Some(Json::Bool(false)));
        assert_eq!(options.model.defaults.max_tokens, Some(4096));

        for (text, says) in [
            ("model = /m.gguf\ncontxt = 1\n", ":2: Ares has no option 'contxt'"),
            (
                "model = /m.gguf\ncontext = 1\ncontext = 2\n",
                ":3: the key 'context' stands twice (first at line 2)",
            ),
            ("[m]\nmodel = /m.gguf\n", ":1: a configuration file has no sections"),
            ("model /m.gguf\n", ":1: 'model /m.gguf' is not a key"),
            ("model = /m.gguf\nctx-size = 1\n", ":2: Ares has no option 'ctx-size': the context"),
            ("model = /m.gguf\nhelp = true\n", ":2: 'help' is not an option to keep in a file"),
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
