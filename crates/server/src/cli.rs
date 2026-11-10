//! The command line. Four flags do not justify a parser library.

use std::ffi::OsString;
use std::path::PathBuf;

pub const USAGE: &str = "\
ares: OpenAI-compatible inference server

USAGE:
    ares [OPTIONS]

OPTIONS:
        --host <ADDR>     Address to listen on [default: 127.0.0.1]
        --port <PORT>     Port to listen on, 0 picks a free one [default: 8000]
        --model <PATH>    GGUF model file to open at startup (any part of a split model)
    -h, --help            Print this help

The log level is set with the RUST_LOG environment variable (default: info).
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub host: String,
    pub port: u16,
    pub model: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self { host: "127.0.0.1".to_owned(), port: 8000, model: None }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Run(Options),
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
}

/// Parses the arguments after the program name. `--name value` and
/// `--name=value` are both accepted.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, CliError> {
    let mut options = Options::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let text =
            arg.to_str().ok_or_else(|| CliError::NotUnicode(arg.to_string_lossy().into_owned()))?;
        let (name, inline) = match text.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(OsString::from(value))),
            _ => (text, None),
        };
        let mut value = |flag: &'static str| {
            inline.clone().or_else(|| args.next()).ok_or(CliError::MissingValue(flag))
        };
        match name {
            "-h" | "--help" => {
                if inline.is_some() {
                    return Err(CliError::UnexpectedValue("--help"));
                }
                return Ok(Command::Help);
            }
            "--host" => {
                let host = value("--host")?;
                options.host = host
                    .into_string()
                    .map_err(|h| CliError::NotUnicode(h.to_string_lossy().into_owned()))?;
            }
            "--port" => {
                let port = value("--port")?;
                let port = port.to_string_lossy();
                options.port =
                    port.parse().map_err(|_| CliError::InvalidPort(port.into_owned()))?;
            }
            "--model" => options.model = Some(PathBuf::from(value("--model")?)),
            _ => return Err(CliError::UnknownOption(text.to_owned())),
        }
    }
    Ok(Command::Run(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_strs(args: &[&str]) -> Result<Command, CliError> {
        parse(args.iter().map(OsString::from))
    }

    #[test]
    fn no_arguments_gives_the_defaults() {
        assert_eq!(parse_strs(&[]), Ok(Command::Run(Options::default())));
    }

    #[test]
    fn both_value_forms_are_accepted() {
        let expected = Command::Run(Options {
            host: "0.0.0.0".into(),
            port: 9000,
            model: Some(PathBuf::from("/models/a b.gguf")),
        });
        assert_eq!(
            parse_strs(&["--host", "0.0.0.0", "--port", "9000", "--model", "/models/a b.gguf"]),
            Ok(expected.clone())
        );
        assert_eq!(
            parse_strs(&["--host=0.0.0.0", "--port=9000", "--model=/models/a b.gguf"]),
            Ok(expected)
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
    }

    #[test]
    fn a_value_may_look_like_a_flag() {
        // `--model --odd.gguf` is a path, not a missing value: the flag asked for one.
        assert_eq!(
            parse_strs(&["--model", "--odd.gguf"]),
            Ok(Command::Run(Options {
                model: Some(PathBuf::from("--odd.gguf")),
                ..Options::default()
            }))
        );
    }
}
