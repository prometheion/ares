//! The command line. A dozen flags do not justify a parser library.

use std::ffi::OsString;
use std::path::PathBuf;

use ares_cuda::Precision;
use ares_server::ReasoningFormat;
use ares_server::plan::{self, Drafter, LoadOptions, SplitKind};
use ares_tokenizer::Json;

pub const USAGE: &str = "\
ares: inference server with an OpenAI-compatible API

USAGE:
    ares --model <PATH> [OPTIONS]

OPTIONS:
        --model <PATH>         GGUF model file to load and serve (any part of a split model)
        --host <ADDR>          Address to listen on [default: 127.0.0.1]
        --port <PORT>          Port to listen on, 0 picks a free one [default: 8000]

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
        --kv <FORMAT>          Storage format of K and V: f16 or q8, or two names
                               (K,V) [default: q8]

    Host memory.
        --ram <SIZE>           The most RAM the whole engine may use, such as 16G or
                               512M. What the engine needs to run is taken from it
                               first; the rest is for the checkpoints of recurrent
                               state that let a conversation continue from what was
                               computed, locked as they are taken [default: half of
                               what the machine has available when the server
                               starts]

    Speed against exactness.
        --precision <KIND>     fast:  8-bit activations where they are faster
                               exact: f16 products everywhere, same text whatever
                                      else runs in a step
                               [default: fast]
        --drafters <LIST>      The model drafters that take part, such as mtp (the
                               model's own draft block) or none; a file name is
                               a block drafter from a file of its own (DFlash2,
                               not built yet). The lookup in earlier text takes
                               part with them and never alone; with none the
                               server does not speculate. The length of a draft
                               and whose it is are chosen for every step from
                               the price of a step and the acceptance measured
                               [default: all the model has: mtp if the file has
                               a draft block]
        --window <ROWS>        Most rows of one sequence's window, its newest
                               token and the drafts verified with it. The room
                               for them is VRAM (about 5 MiB a row for the 27B,
                               for every sequence) [default: 8, the rows one
                               launch of the few-row weight kernels takes]
        --draft-vocab <FILE>   Token ids (one per line) the model's draft block may
                               propose: a shorter list makes a draft cheaper
                               [default: every token whose head row is on the
                               block's device]

    Prompts.
        --chat-var <NAME=VALUE>  A variable of the model's chat template, such as
                               add_vision_id=true; may be given several times

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

    -h, --help                 Print this help

The log level is set with the RUST_LOG environment variable (default: info).
ARES_LIBCUDA names the driver library to load instead of the system's libcuda.so.1.
";

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub host: String,
    pub port: u16,
    pub load: LoadOptions,
    pub reasoning_format: ReasoningFormat,
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
    #[error("option '{flag}': '{value}' is not {expected}")]
    InvalidValue { flag: &'static str, value: String, expected: &'static str },
    #[error("option '{0}' is given twice")]
    Repeated(&'static str),
    #[error(
        "there is nothing to serve: name a model file with --model (loading models on request \
         from a preset file is not built yet)"
    )]
    NoModel,
}

/// The text of a value.
fn text(value: OsString) -> Result<String, CliError> {
    value.into_string().map_err(|v| CliError::NotUnicode(v.to_string_lossy().into_owned()))
}

fn invalid(flag: &'static str, value: &str, expected: &'static str) -> CliError {
    CliError::InvalidValue { flag, value: value.to_owned(), expected }
}

fn number<T: std::str::FromStr>(
    flag: &'static str,
    value: &str,
    expected: &'static str,
) -> Result<T, CliError> {
    value.trim().parse().map_err(|_| invalid(flag, value, expected))
}

fn list<T: std::str::FromStr>(
    flag: &'static str,
    value: &str,
    expected: &'static str,
) -> Result<Vec<T>, CliError> {
    value.split(',').map(|item| number(flag, item, expected)).collect()
}

/// The model drafters of `--drafters`: names separated by commas. `mtp` is
/// the model's own draft block, `none` (alone) is no drafter, anything else is
/// the file of a block drafter.
fn drafters(value: &str) -> Result<Vec<Drafter>, CliError> {
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
fn chat_variable(value: &str) -> Result<(String, Json), CliError> {
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

/// Parses the arguments after the program name. `--name value` and
/// `--name=value` are both accepted.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, CliError> {
    let (mut host, mut port) = ("127.0.0.1".to_owned(), 8000);
    let mut load = LoadOptions::new("");
    let mut reasoning_format = ReasoningFormat::default();
    // The thinking of earlier turns stays in the prompt unless asked otherwise.
    set_chat_variable(&mut load, "preserve_reasoning", Json::Bool(true));
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
        let flag: &'static str = match name {
            "-h" | "--help" => {
                if inline.is_some() {
                    return Err(CliError::UnexpectedValue("--help"));
                }
                return Ok(Command::Help);
            }
            "--host" => {
                host = text(value("--host")?)?;
                "--host"
            }
            "--port" => {
                let value = value("--port")?;
                let value = value.to_string_lossy();
                port = value.parse().map_err(|_| CliError::InvalidPort(value.into_owned()))?;
                "--port"
            }
            "--model" => {
                load.model = PathBuf::from(value("--model")?);
                "--model"
            }
            "--devices" => {
                let value = text(value("--devices")?)?;
                load.devices = Some(list("--devices", &value, "a list of device numbers (0,1)")?);
                "--devices"
            }
            "--split" => {
                let value = text(value("--split")?)?;
                load.split = Some(match value.as_str() {
                    "single" => SplitKind::Single,
                    "tensor" => SplitKind::Tensor,
                    "layer" => SplitKind::Layer,
                    _ => return Err(invalid("--split", &value, "single, tensor or layer")),
                });
                "--split"
            }
            "--shares" => {
                let value = text(value("--shares")?)?;
                let expected = "a list of positive numbers (0.625,0.375)";
                let shares: Vec<f64> = list("--shares", &value, expected)?;
                if shares.iter().any(|share| !share.is_finite() || *share <= 0.0) {
                    return Err(invalid("--shares", &value, expected));
                }
                load.shares = Some(shares);
                "--shares"
            }
            "--context" => {
                let value = text(value("--context")?)?;
                let expected = "a number of tokens, at least 1";
                let context: u32 = number("--context", &value, expected)?;
                if context == 0 {
                    return Err(invalid("--context", &value, expected));
                }
                load.context = Some(context);
                "--context"
            }
            "--parallel" => {
                let value = text(value("--parallel")?)?;
                let expected = "a number of sequences, at least 1";
                let parallel: u32 = number("--parallel", &value, expected)?;
                if parallel == 0 {
                    return Err(invalid("--parallel", &value, expected));
                }
                load.parallel = Some(parallel);
                "--parallel"
            }
            "--kv" => {
                let value = text(value("--kv")?)?;
                let expected = "a format (f16, q8, q7, q6, q5, q4) or two, for K and V (q8,f16)";
                let formats = value
                    .split(',')
                    .map(|name| plan::cache_format(name.trim()))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| invalid("--kv", &value, expected))?;
                load.kv = Some(match formats.as_slice() {
                    [both] => (*both, *both),
                    [k, v] => (*k, *v),
                    _ => return Err(invalid("--kv", &value, expected)),
                });
                "--kv"
            }
            "--ram" => {
                let value = text(value("--ram")?)?;
                let expected = "a size such as 16G or 512M";
                load.ram = Some(
                    plan::parse_size(&value).ok_or_else(|| invalid("--ram", &value, expected))?,
                );
                "--ram"
            }
            "--precision" => {
                let value = text(value("--precision")?)?;
                load.precision = match value.as_str() {
                    "fast" => Precision::Fast,
                    "exact" => Precision::Exact,
                    _ => return Err(invalid("--precision", &value, "fast or exact")),
                };
                "--precision"
            }
            "--drafters" => {
                let value = text(value("--drafters")?)?;
                load.drafters = Some(drafters(&value)?);
                "--drafters"
            }
            "--window" => {
                let value = text(value("--window")?)?;
                let expected = "a number of rows, at least 2";
                let rows: u32 = number("--window", &value, expected)?;
                if rows < 2 {
                    return Err(invalid("--window", &value, expected));
                }
                load.window = Some(rows);
                "--window"
            }
            "--draft-vocab" => {
                load.draft_vocabulary = Some(PathBuf::from(value("--draft-vocab")?));
                "--draft-vocab"
            }
            "--chat-var" => {
                let value = text(value("--chat-var")?)?;
                let (name, value) = chat_variable(&value)?;
                set_chat_variable(&mut load, &name, value);
                continue;
            }
            "-rea" | "--reasoning" => {
                let value = text(value("--reasoning")?)?;
                match value.as_str() {
                    "on" => set_chat_variable(&mut load, "enable_thinking", Json::Bool(true)),
                    "off" => set_chat_variable(&mut load, "enable_thinking", Json::Bool(false)),
                    "auto" => load.chat_variables.retain(|(name, _)| name != "enable_thinking"),
                    _ => return Err(invalid("--reasoning", &value, "on, off or auto")),
                }
                "--reasoning"
            }
            "--reasoning-format" => {
                let value = text(value("--reasoning-format")?)?;
                reasoning_format = ReasoningFormat::from_name(&value).ok_or_else(|| {
                    invalid("--reasoning-format", &value, "none, auto, deepseek or deepseek-legacy")
                })?;
                "--reasoning-format"
            }
            "--reasoning-effort" => {
                let value = text(value("--reasoning-effort")?)?;
                if value.is_empty() {
                    return Err(invalid("--reasoning-effort", &value, "a level, or default"));
                }
                if value == "default" {
                    load.chat_variables.retain(|(name, _)| name != "reasoning_effort");
                } else {
                    set_chat_variable(&mut load, "reasoning_effort", Json::String(value));
                }
                "--reasoning-effort"
            }
            "--reasoning-preserve" | "--no-reasoning-preserve" => {
                if inline.is_some() {
                    return Err(CliError::UnexpectedValue("--reasoning-preserve"));
                }
                let on = name == "--reasoning-preserve";
                set_chat_variable(&mut load, "preserve_reasoning", Json::Bool(on));
                "--reasoning-preserve"
            }
            _ => return Err(CliError::UnknownOption(whole.to_owned())),
        };
        // A flag that is given twice is a mistake more often than a choice.
        if seen.contains(&flag) {
            return Err(CliError::Repeated(flag));
        }
        seen.push(flag);
    }
    if load.model.as_os_str().is_empty() {
        return Err(CliError::NoModel);
    }
    Ok(Command::Run(Box::new(Options { host, port, load, reasoning_format })))
}

#[cfg(test)]
mod tests {
    use ares_cuda::CacheFormat;

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
    }

    #[test]
    fn drafters_are_named_one_by_one_and_a_file_is_a_block_drafter() {
        let drafters = |list: &str| run(&["--model=m", "--drafters", list]).load.drafters;
        assert_eq!(drafters("mtp"), Some(vec![Drafter::Mtp]));
        assert_eq!(drafters("none"), Some(Vec::new()));
        assert_eq!(
            drafters("mtp, /models/dflash2.gguf"),
            Some(vec![Drafter::Mtp, Drafter::Block(PathBuf::from("/models/dflash2.gguf"))])
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
            (&["--precision", "sloppy"], "--precision"),
            (&["--drafters", "none,mtp"], "--drafters"),
            (&["--chat-var", "novalue"], "--chat-var"),
            (&["--chat-var", "=x"], "--chat-var"),
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
}
