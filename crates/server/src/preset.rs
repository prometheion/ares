//! The preset file: the models a server offers, each with its options.
//!
//! An INI file. The section `[*]` holds what is the same for every model; every
//! other section is one model, and its name is the id requests use:
//!
//! ```ini
//! ; a comment (a line that begins with ; or #)
//! [*]
//! ram = 24G
//! idle-unload = 30m
//!
//! [qwen-27b]
//! model = /models/Qwen3.8-27B-UD-Q4_K_S.gguf
//! aliases = default, gpt-4o
//! reasoning-effort = medium
//! load-on-startup = true
//!
//! [small]
//! model = /models/Qwen3.5-0.8B-Q4_K_M.gguf
//! devices = 0
//! context = 16384
//! ram = 2G
//! temperature = 0.7
//! top-p = 0.8
//! max-tokens = 4096
//! ```
//!
//! A key is an option of the command line under its name without the dashes
//! (`context`, `kv`, `chat-var`, which may stand several times), with the
//! value the option takes; a switch (`reasoning-preserve`) takes `true` or
//! `false`. Among them are the defaults of a model's requests (`temperature`,
//! `top-p`, `max-tokens` and the others of [`crate::defaults`]): what a
//! request gets for a field it leaves out. Three keys exist only here:
//! `aliases` (more ids the model answers to, separated by commas),
//! `load-on-startup` (`true`: loaded when the server starts, not at its first
//! request) and `idle-unload` (how long the model may stay loaded without a
//! request: seconds, or a number with `s`, `m` or `h`; `never`, the default).
//!
//! What holds for a model is, from the strongest: the command line, the
//! model's section, the section `[*]`, the built-in defaults. `ram` is the
//! one key that means something else by where it stands: on the command line
//! or in `[*]` it is the most RAM of the server, for all loaded models
//! together; in a model's section it is that model's part of it.
//!
//! Nothing is ignored: a key the server does not know, a value an option does
//! not take, two keys that cannot hold together (`max-tokens` above
//! `max-tokens-limit`), a section without a model file or with a file that is
//! not there, and an id that two sections answer to are errors with the file
//! and the line.
//! An option of llama-server that Ares has under another name is answered with
//! that name.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::options::{self, Kind, ModelOptions};

/// One model of a preset file.
#[derive(Debug, Clone, PartialEq)]
pub struct Preset {
    /// The id requests name, and the model is listed under.
    pub id: String,
    /// More ids the model answers to.
    pub aliases: Vec<String>,
    /// The options the model is loaded with. `load.ram` is the model's own
    /// part of the server's RAM, if its section names one.
    pub options: ModelOptions,
    /// Loaded when the server starts.
    pub load_on_startup: bool,
    /// How long the model stays loaded without a request; `None`: until its
    /// room is needed.
    pub idle_unload: Option<Duration>,
}

/// What a preset file says.
#[derive(Debug, Clone, PartialEq)]
pub struct Presets {
    /// The most RAM of the server, if the command line or `[*]` names it.
    pub ram: Option<u64>,
    pub models: Vec<Preset>,
}

/// A line of a preset file that cannot be served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresetError {
    pub file: PathBuf,
    /// The line, counted from 1; 0 for the file as a whole.
    pub line: usize,
    pub reason: String,
}

impl std::fmt::Display for PresetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            0 => write!(f, "{}: {}", self.file.display(), self.reason),
            line => write!(f, "{}:{line}: {}", self.file.display(), self.reason),
        }
    }
}

impl std::error::Error for PresetError {}

/// The id a model file is served under when nothing names one: its file name
/// without the extension.
pub fn id_of_file(model: &Path) -> String {
    model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "model".to_owned())
}

impl Presets {
    /// The one model of `ares --model`: loaded at start and never unloaded
    /// for being idle. `--ram` is the server's, and so all of it the model's.
    pub fn single(mut options: ModelOptions) -> Self {
        let ram = options.load.ram.take();
        let id = id_of_file(&options.load.model);
        let model =
            Preset { id, aliases: Vec::new(), options, load_on_startup: true, idle_unload: None };
        Self { ram, models: vec![model] }
    }

    /// Reads the preset file `file`. `given` are the options of a model that
    /// stand on the command line ([`options::lookup`]'s names), which hold
    /// over every section.
    pub fn read(file: &Path, given: &[(&'static str, String)]) -> Result<Self, PresetError> {
        let text = std::fs::read_to_string(file).map_err(|e| PresetError {
            file: file.to_owned(),
            line: 0,
            reason: format!("cannot read the preset file: {e}"),
        })?;
        Self::parse(&text, file, given, &|path| path.is_file())
    }

    /// [`Presets::read`] over the text of the file. `exists` says whether a
    /// model file is there.
    pub fn parse(
        text: &str,
        file: &Path,
        given: &[(&'static str, String)],
        exists: &dyn Fn(&Path) -> bool,
    ) -> Result<Self, PresetError> {
        let at = |line: usize, reason: String| PresetError { file: file.to_owned(), line, reason };
        let sections = sections(text).map_err(|(line, reason)| at(line, reason))?;

        let mut global = Section::default();
        let mut models: Vec<&Section> = Vec::new();
        for section in &sections {
            if section.name == "*" {
                if global.line != 0 {
                    return Err(at(
                        section.line,
                        format!(
                            "the section [*] stands a second time (first at line {})",
                            global.line
                        ),
                    ));
                }
                global = section.clone();
            } else if let Some(first) = models.iter().find(|m| m.name == section.name) {
                return Err(at(
                    section.line,
                    format!(
                        "the section [{}] stands a second time (first at line {})",
                        section.name, first.line
                    ),
                ));
            } else {
                models.push(section);
            }
        }
        if models.is_empty() {
            return Err(at(0, "the file names no model: it has no section but [*]".into()));
        }

        // The command line's RAM is the server's, like the one of `[*]`.
        let mut ram = None;
        let mut on_command_line = Vec::new();
        for (flag, value) in given {
            if *flag == "--ram" {
                let mut scratch = ModelOptions::default();
                options::set(&mut scratch, flag, value).map_err(|e| at(0, e.to_string()))?;
                ram = scratch.load.ram;
            } else {
                on_command_line.push((*flag, value.as_str()));
            }
        }

        let mut presets: Vec<Preset> = Vec::new();
        // Every id and alias, with the line that gave it.
        let mut names: Vec<(String, usize)> = Vec::new();
        for section in models {
            let mut preset = Preset {
                id: section.name.clone(),
                aliases: Vec::new(),
                options: ModelOptions::default(),
                load_on_startup: false,
                idle_unload: None,
            };
            let mut model_line = None;
            for (scope, keys) in [(Scope::Global, &global.keys), (Scope::Model, &section.keys)] {
                for key in keys {
                    let fail = |reason: String| at(key.line, reason);
                    match apply(&mut preset, scope, key).map_err(fail)? {
                        Applied::Ram(bytes) if scope == Scope::Global => {
                            preset.options.load.ram = None;
                            ram = ram.or(Some(bytes));
                        }
                        Applied::Model => model_line = Some(key.line),
                        _ => {}
                    }
                }
            }
            let Some(model_line) = model_line else {
                return Err(at(
                    section.line,
                    format!(
                        "the section [{}] names no model file: add model = <file.gguf>",
                        section.name
                    ),
                ));
            };
            if !exists(&preset.options.load.model) {
                return Err(at(
                    model_line,
                    format!(
                        "the model file '{}' does not exist (a path that is not absolute starts \
                         at the directory the server runs in)",
                        preset.options.load.model.display()
                    ),
                ));
            }
            let own_ram = preset.options.load.ram;
            for (flag, value) in &on_command_line {
                options::set(&mut preset.options, flag, value).map_err(|e| at(0, e.to_string()))?;
            }
            preset.options.load.ram = own_ram;
            // Each value is one its option takes; together they may not hold.
            // The line is that of a key of the two that is in effect: one of
            // the model's own section if there is one, else one of [*], and
            // none if the command line gave both.
            if let Err(conflict) = preset.options.check() {
                let given = |flag: &str| on_command_line.iter().any(|(known, _)| *known == flag);
                let line_in = |keys: &[Key]| {
                    let flags = conflict.flags.iter().filter(|flag| !given(flag));
                    let lines = flags.filter_map(|flag| {
                        let name = flag.trim_start_matches('-');
                        keys.iter().find(|key| key.name == name).map(|key| key.line)
                    });
                    lines.max()
                };
                let line = line_in(&section.keys).or_else(|| line_in(&global.keys)).unwrap_or(0);
                return Err(at(line, format!("[{}]: {}", preset.id, conflict.reason)));
            }
            preset.options.load.name = Some(preset.id.clone());

            let ids = std::iter::once((preset.id.clone(), section.line)).chain(
                preset.aliases.iter().map(|alias| {
                    let line = section.keys.iter().find(|k| k.name == "aliases").map(|k| k.line);
                    (alias.clone(), line.unwrap_or(section.line))
                }),
            );
            for (name, line) in ids {
                if let Some((_, first)) = names.iter().find(|(known, _)| *known == name) {
                    return Err(at(
                        line,
                        format!(
                            "the id '{name}' is taken: line {first} gives it to a model, and a \
                             request can mean only one"
                        ),
                    ));
                }
                names.push((name, line));
            }
            presets.push(preset);
        }
        // A part cannot be more than the whole.
        if let Some(total) = ram {
            for (preset, section) in presets.iter().zip(sections.iter().filter(|s| s.name != "*")) {
                if let Some(own) = preset.options.load.ram.filter(|own| *own > total) {
                    let line = section.keys.iter().find(|k| k.name == "ram").map_or(0, |k| k.line);
                    return Err(at(
                        line,
                        format!(
                            "ram = {own} bytes for [{}] is more than the {total} bytes the server \
                             may use (ram in [*] or --ram)",
                            preset.id
                        ),
                    ));
                }
            }
        }
        Ok(Self { ram, models: presets })
    }
}

#[derive(Debug, Clone, Default)]
struct Section {
    name: String,
    line: usize,
    keys: Vec<Key>,
}

#[derive(Debug, Clone)]
struct Key {
    name: String,
    value: String,
    line: usize,
}

/// Cuts the text into its sections; an error is a line and what is wrong
/// with it.
fn sections(text: &str) -> Result<Vec<Section>, (usize, String)> {
    let mut sections: Vec<Section> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let (number, line) = (index + 1, raw.trim());
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            let Some(name) = rest.strip_suffix(']').map(str::trim) else {
                return Err((number, format!("'{line}' is not a section: write [name]")));
            };
            if name.is_empty() {
                return Err((number, "a section needs a name: the id of its model, or *".into()));
            }
            sections.push(Section { name: name.to_owned(), line: number, keys: Vec::new() });
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            return Err((number, format!("'{line}' is not a key: write name = value")));
        };
        let (name, value) = (name.trim(), value.trim());
        let Some(section) = sections.last_mut() else {
            return Err((
                number,
                format!(
                    "the key '{name}' stands before the first section: put it in [*] or in the \
                     section of a model"
                ),
            ));
        };
        if let Some(first) = section.keys.iter().find(|k| k.name == name)
            && !options::repeats(&format!("--{name}"))
        {
            return Err((
                number,
                format!(
                    "the key '{name}' stands twice in [{}] (first at line {})",
                    section.name, first.line
                ),
            ));
        }
        section.keys.push(Key { name: name.to_owned(), value: value.to_owned(), line: number });
    }
    Ok(sections)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    Global,
    Model,
}

/// What a key did that its caller has to know.
enum Applied {
    Option,
    Model,
    Ram(u64),
}

fn truth(key: &Key) -> Result<bool, String> {
    match key.value.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(format!("{} = '{other}': write true or false", key.name)),
    }
}

/// How long a model may stay idle: `never`, seconds, or a number with a unit.
fn idle(key: &Key) -> Result<Option<Duration>, String> {
    let value = key.value.as_str();
    if value == "never" {
        return Ok(None);
    }
    let (number, unit) = match value.char_indices().find(|(_, c)| !c.is_ascii_digit()) {
        Some((at, _)) => value.split_at(at),
        None => (value, "s"),
    };
    let scale = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => 0,
    };
    match number.parse::<u64>().ok().and_then(|n| n.checked_mul(scale)) {
        Some(seconds) if seconds > 0 => Ok(Some(Duration::from_secs(seconds))),
        _ => Err(format!(
            "idle-unload = '{value}': write never, or a time of at least a second such as 600, \
             90s, 30m or 2h"
        )),
    }
}

/// Sets what `key` says in `preset`.
fn apply(preset: &mut Preset, scope: Scope, key: &Key) -> Result<Applied, String> {
    let name = key.name.as_str();
    match name {
        "aliases" if scope == Scope::Global => {
            return Err("aliases in [*]: an alias is one model's, put it in its section".into());
        }
        "aliases" => {
            let aliases: Vec<String> = key.value.split(',').map(|a| a.trim().to_owned()).collect();
            if aliases.iter().any(|a| a.is_empty() || a == "*") {
                return Err(format!("aliases = '{}': write ids separated by commas", key.value));
            }
            preset.aliases = aliases;
            return Ok(Applied::Option);
        }
        "load-on-startup" => {
            preset.load_on_startup = truth(key)?;
            return Ok(Applied::Option);
        }
        "idle-unload" => {
            preset.idle_unload = idle(key)?;
            return Ok(Applied::Option);
        }
        _ if options::SERVER.contains(&name) => {
            return Err(format!(
                "'{name}' is an option of the server, not of a model: give it on the command \
                 line (--{name})"
            ));
        }
        _ => {}
    }
    let Some((flag, kind)) = options::lookup(&format!("--{name}")) else {
        // The field of a request, written as the request writes it.
        let dashed = name.replace('_', "-");
        if dashed != name && options::lookup(&format!("--{dashed}")).is_some() {
            return Err(format!("Ares has no option '{name}': the key is '{dashed}', with dashes"));
        }
        return Err(match other_name(name) {
            Some(hint) => format!("Ares has no option '{name}': {hint}"),
            None => format!(
                "Ares has no option '{name}' ('ares --help' lists the options; the keys that \
                 exist only in a preset file are aliases, load-on-startup and idle-unload)"
            ),
        });
    };
    if flag == "--model" && scope == Scope::Global {
        return Err("model in [*]: a model file is one model's, put it in its section".into());
    }
    let value = match kind {
        Kind::Value => key.value.clone(),
        Kind::Switch(true) => truth(key)?.to_string(),
        Kind::Switch(false) => {
            return Err(format!(
                "'{name}' is the command line's way to switch {0} off: write {0} = false",
                flag.trim_start_matches('-')
            ));
        }
    };
    options::set(&mut preset.options, flag, &value)
        .map_err(|e| format!("{name} = '{}' is not {}", e.value, e.expected))?;
    Ok(match flag {
        "--model" => Applied::Model,
        "--ram" => Applied::Ram(preset.options.load.ram.unwrap_or(0)),
        _ => Applied::Option,
    })
}

/// What to write instead of an option of llama-server that Ares has under
/// another name, or does not have because it has nothing to choose there.
fn other_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "c" | "ctx-size" => "the context is 'context', in tokens for all sequences together",
        "np" => "the sequences that run at once are 'parallel'",
        "dev" | "device" => "the devices are 'devices', by the driver's numbers (0,1)",
        "sm" | "split-mode" => "the split is 'split' (single, tensor or layer)",
        "ts" | "tensor-split" => "the shares of a split are 'shares'",
        "ctk" | "ctv" | "cache-type-k" | "cache-type-v" => {
            "the formats of K and V are 'kv' (one name, or two for K and V: q5,q5)"
        }
        "chat-template-kwargs" => {
            "a variable of the chat template is 'chat-var' (chat-var = reasoning_effort=medium), \
             once for every variable"
        }
        "spec-type" => "the drafters are 'drafters' (mtp or none)",
        "spec-draft-vocab" => "the draft vocabulary is 'draft-vocab'",
        "spec-draft-n-max" | "draft-max" => {
            "the room for drafts is 'window', the rows of one sequence's window; how long a \
             draft is, is chosen for every step"
        }
        "no-mmproj-offload" | "mmproj-offload" => {
            "where the projector's weights stay is 'mmproj-weights' (ram, the default, or vram)"
        }
        "image-min-tokens" => "an image becomes at least 8 rows, as in llama.cpp",
        "cache-ram" => "the RAM the engine may use, its caches included, is 'ram'",
        "cache-disk" => "the directory of the prompt cache on SSD is 'cache'",
        "cache-disk-size" => "the most the prompt cache takes on SSD is 'cache-size'",
        "ctx-checkpoints" | "checkpoint-root" | "checkpoint-spread" | "cache-save-min" => {
            "the engine places its checkpoints and decides what of them is kept and written"
        }
        "sleep-idle-seconds" => "the time a model may stay idle is 'idle-unload'",
        "a" | "alias" => "more ids of a model are 'aliases'",
        "models-max" => {
            "nothing counts the loaded models: one is unloaded when another needs its memory"
        }
        "stop-timeout" => "an unload writes nothing and waits for nothing but running requests",
        "ngl" | "gpu-layers" | "n-gpu-layers" => "all of a model is on the devices",
        "fa" | "flash-attn" | "jinja" => "that is always on",
        // The defaults of the sampler: a key is the field of a request, with
        // dashes.
        "temp" => "the temperature of a request that names none is 'temperature'",
        "top-nsigma" => "that default is 'top-n-sigma' (0 is off)",
        "typical" => "that default is 'typical-p'",
        "dynatemp-exp" => "that default is 'dynatemp-exponent'",
        "mirostat-lr" => "mirostat's rate is 'mirostat-eta'",
        "mirostat-ent" => "mirostat's target is 'mirostat-tau'",
        "dry-sequence-breaker" => {
            "DRY's breakers are 'dry-sequence-breakers', all of them in one JSON array \
             ([\"\\n\", \":\"])"
        }
        "sampler-seq" | "sampling-seq" => {
            "the order of the samplers is 'samplers', by their names \
             (penalties,dry,top_k,top_p,min_p,temperature)"
        }
        "s" => "the seed of a request that names none is 'seed'",
        "reasoning-budget" => {
            "the budget of thinking tokens of a request that names none is \
             'reasoning-budget-tokens'"
        }
        "n" | "predict" | "n-predict" => {
            "the limit of tokens for a request that names none is 'max-tokens'; \
             'max-tokens-limit' is the most any request gets"
        }
        "ignore-eos" => {
            "a reply that does not end is what a request asks for (ignore_eos), not a default \
             of a model"
        }
        "l" | "logit-bias" => "a logit bias is a request's (logit_bias); a model has no default",
        "adaptive-target" | "adaptive-decay" => "adaptive-p is not built",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use ares_cuda::CacheFormat;
    use ares_tokenizer::Json;

    use super::*;
    use crate::ReasoningFormat;
    use crate::plan::SplitKind;

    fn parse(text: &str, given: &[(&'static str, &str)]) -> Result<Presets, PresetError> {
        let given: Vec<_> =
            given.iter().map(|(flag, value)| (*flag, (*value).to_owned())).collect();
        Presets::parse(text, Path::new("m.ini"), &given, &|path| path != Path::new("/gone.gguf"))
    }

    fn refused(text: &str) -> String {
        parse(text, &[]).expect_err(text).to_string()
    }

    fn variable(preset: &Preset, name: &str) -> Option<Json> {
        let variables = &preset.options.load.chat_variables;
        variables.iter().find(|(known, _)| known == name).map(|(_, value)| value.clone())
    }

    const TWO: &str = "\
; the user's models
[*]
context = 8192
kv = q5
reasoning = on
idle-unload = 30m
ram = 24G

# the big one
[big]
model = /models/big.gguf
aliases = default, gpt-4o
context = 100000
split = tensor
shares = 0.625,0.375
chat-var = reasoning_effort=medium
chat-var = depth=3
reasoning-preserve = false
load-on-startup = true
ram = 16G

[small]
model = /models/small.gguf
devices = 0
reasoning = auto
reasoning-format = none
idle-unload = never
";

    #[test]
    fn a_section_is_a_model_with_the_options_of_the_command_line() {
        let presets = parse(TWO, &[]).unwrap();
        assert_eq!(presets.ram, Some(24 << 30));
        let [big, small] = presets.models.as_slice() else { panic!("{presets:?}") };

        assert_eq!(big.id, "big");
        assert_eq!(big.aliases, ["default", "gpt-4o"]);
        assert!(big.load_on_startup);
        assert_eq!(big.idle_unload, Some(Duration::from_secs(1800)));
        let load = &big.options.load;
        assert_eq!(load.model, PathBuf::from("/models/big.gguf"));
        assert_eq!(load.name.as_deref(), Some("big"));
        assert_eq!(load.split, Some(SplitKind::Tensor));
        assert_eq!(load.shares, Some(vec![0.625, 0.375]));
        assert_eq!(load.ram, Some(16 << 30), "its own part of the server's RAM");
        assert_eq!(variable(big, "reasoning_effort"), Some(Json::String("medium".into())));
        assert_eq!(variable(big, "depth"), Some(Json::Int(3)));
        assert_eq!(variable(big, "preserve_reasoning"), Some(Json::Bool(false)));

        assert_eq!(small.id, "small");
        assert!(small.aliases.is_empty() && !small.load_on_startup);
        assert_eq!(small.idle_unload, None, "never, over the 30 minutes of [*]");
        assert_eq!(small.options.load.devices, Some(vec![0]));
        assert_eq!(small.options.load.ram, None, "the RAM of [*] is the server's, not a model's");
        assert_eq!(small.options.reasoning_format, ReasoningFormat::None);
        // The built-in default, which no section changed.
        assert_eq!(variable(small, "preserve_reasoning"), Some(Json::Bool(true)));
    }

    #[test]
    fn the_command_line_holds_over_a_section_and_a_section_over_the_global_one() {
        let presets = parse(TWO, &[]).unwrap();
        let [big, small] = presets.models.as_slice() else { panic!() };
        // The section over [*].
        assert_eq!(big.options.load.context, Some(100_000));
        assert_eq!(variable(small, "enable_thinking"), None, "auto takes back the on of [*]");
        // [*] over the built-in default.
        assert_eq!(small.options.load.context, Some(8192));
        assert_eq!(small.options.load.kv, Some((CacheFormat::Q5, CacheFormat::Q5)));
        assert_eq!(variable(big, "enable_thinking"), Some(Json::Bool(true)));
        // The built-in default where nothing is said.
        assert_eq!(small.options.load.parallel, None);

        // The command line over both.
        let given = [("--context", "4096"), ("--reasoning", "off"), ("--ram", "8G")];
        let presets = parse(&TWO.replace("ram = 16G", "ram = 6G"), &given).unwrap();
        let [big, small] = presets.models.as_slice() else { panic!() };
        assert_eq!(big.options.load.context, Some(4096));
        assert_eq!(small.options.load.context, Some(4096));
        assert_eq!(variable(big, "enable_thinking"), Some(Json::Bool(false)));
        assert_eq!(variable(small, "enable_thinking"), Some(Json::Bool(false)));
        // What the command line does not name stays the section's.
        assert_eq!(big.options.load.split, Some(SplitKind::Tensor));
        // --ram is the server's, over the one of [*]; a model's part stays its own.
        assert_eq!(presets.ram, Some(8 << 30));
        assert_eq!(big.options.load.ram, Some(6 << 30));
        assert_eq!(small.options.load.ram, None);
    }

    #[test]
    fn one_model_on_the_command_line_is_a_preset_loaded_at_start() {
        let mut options = ModelOptions::default();
        options::set(&mut options, "--model", "/models/Qwen-Q4.gguf").unwrap();
        options::set(&mut options, "--ram", "8G").unwrap();
        let presets = Presets::single(options);
        assert_eq!(presets.ram, Some(8 << 30));
        let [only] = presets.models.as_slice() else { panic!() };
        assert_eq!(only.id, "Qwen-Q4");
        assert!(only.load_on_startup);
        assert_eq!(only.idle_unload, None);
        assert_eq!(only.options.load.name, None, "the loader names it by its file");
    }

    #[test]
    fn every_refusal_names_the_file_the_line_and_the_reason() {
        let model = "[m]\nmodel = /m.gguf\n";
        for (text, line, reason) in [
            ("version = 1\n[m]\nmodel = /m.gguf\n", 1, "stands before the first section"),
            ("[m\nmodel = /m.gguf\n", 1, "is not a section"),
            ("[]\nmodel = /m.gguf\n", 1, "a section needs a name"),
            ("[m]\nmodel /m.gguf\n", 2, "is not a key"),
            ("[*]\ncontext = 1\n", 0, "names no model"),
            (
                "[*]\nkv = q8\n[m]\ncontext = 4096\n\n[n]\nmodel = /n.gguf\n",
                3,
                "[m] names no model file",
            ),
            ("[m]\nmodel = /gone.gguf\n", 2, "'/gone.gguf' does not exist"),
            (
                "[m]\nmodel = /m.gguf\n[m]\nmodel = /n.gguf\n",
                3,
                "[m] stands a second time (first at line 1)",
            ),
            ("[*]\nkv = q8\n[*]\nkv = q5\n[m]\nmodel = /m.gguf\n", 3, "[*] stands a second time"),
            (
                "[m]\nmodel = /m.gguf\ncontext = 1\ncontext = 2\n",
                4,
                "'context' stands twice in [m] (first at line 3)",
            ),
            ("[*]\nmodel = /m.gguf\n[m]\nmodel = /m.gguf\n", 2, "model in [*]"),
            ("[*]\naliases = a\n[m]\nmodel = /m.gguf\n", 2, "aliases in [*]"),
            ("[m]\nmodel = /m.gguf\naliases = a,,b\n", 3, "write ids separated by commas"),
            (
                "[m]\nmodel = /m.gguf\n[n]\nmodel = /n.gguf\naliases = x, m\n",
                5,
                "the id 'm' is taken: line 1",
            ),
            (
                "[m]\nmodel = /m.gguf\naliases = x\n[n]\nmodel = /n.gguf\naliases = x\n",
                6,
                "the id 'x' is taken: line 3",
            ),
            (
                "[m]\nmodel = /m.gguf\ncontext = 0\n",
                3,
                "context = '0' is not a number of tokens, at least 1",
            ),
            (
                "[m]\nmodel = /m.gguf\nsplit = diagonal\n",
                3,
                "split = 'diagonal' is not single, tensor or layer",
            ),
            ("[*]\nkv = q3\n[m]\nmodel = /m.gguf\n", 2, "kv = 'q3' is not a format"),
            ("[m]\nmodel = /m.gguf\nchat-var = novalue\n", 3, "is not NAME=VALUE"),
            ("[m]\nmodel = /m.gguf\nreasoning-preserve = yes\n", 3, "write true or false"),
            (
                "[m]\nmodel = /m.gguf\nno-reasoning-preserve = true\n",
                3,
                "write reasoning-preserve = false",
            ),
            ("[m]\nmodel = /m.gguf\nload-on-startup = 1\n", 3, "write true or false"),
            ("[m]\nmodel = /m.gguf\nidle-unload = soon\n", 3, "write never, or a time"),
            ("[m]\nmodel = /m.gguf\nidle-unload = 0\n", 3, "at least a second"),
            ("[m]\nmodel = /m.gguf\nport = 8000\n", 3, "'port' is an option of the server"),
            ("[*]\nhost = 0.0.0.0\n[m]\nmodel = /m.gguf\n", 2, "'host' is an option of the server"),
            ("[m]\nmodel = /m.gguf\ncontxt = 4096\n", 3, "Ares has no option 'contxt'"),
            (
                "[*]\nram = 4G\n[m]\nmodel = /m.gguf\nram = 8G\n",
                5,
                "is more than the 4294967296 bytes the server may use",
            ),
            // The defaults of a model's requests: the range of each value.
            (
                "[m]\nmodel = /m.gguf\ntop-p = 1.5\n",
                3,
                "top-p = '1.5' is not a number above 0 and at most 1",
            ),
            (
                "[*]\ntemperature = -1\n[m]\nmodel = /m.gguf\n",
                2,
                "temperature = '-1' is not a number from 0 on",
            ),
            ("[m]\nmodel = /m.gguf\nmirostat = 3\n", 3, "mirostat = '3' is not 0 (off), 1 or 2"),
            (
                "[m]\nmodel = /m.gguf\nrepeat-last-n = 4096\n",
                3,
                "repeat-last-n = '4096' is not a number of tokens from 0 to 1024",
            ),
            (
                "[m]\nmodel = /m.gguf\nsamplers = top_k;temperature\n",
                3,
                "is not names of samplers separated by commas",
            ),
            (
                "[m]\nmodel = /m.gguf\ndry-sequence-breakers = \\n\n",
                3,
                "is not a JSON array of strings",
            ),
            ("[m]\nmodel = /m.gguf\nseed = -1\n", 3, "seed = '-1' is not a whole number from 0"),
            (
                "[m]\nmodel = /m.gguf\nmax-tokens = 0\n",
                3,
                "max-tokens = '0' is not a number of tokens, at least 1",
            ),
            (
                "[*]\nmax-tokens-limit = lots\n[m]\nmodel = /m.gguf\n",
                2,
                "max-tokens-limit = 'lots' is not a number of tokens, at least 1",
            ),
            (
                "[m]\nmodel = /m.gguf\ntemperature = 0.5\ntemperature = 0.7\n",
                4,
                "'temperature' stands twice in [m] (first at line 3)",
            ),
            // And the two that cannot hold together, wherever each stands.
            (
                "[m]\nmodel = /m.gguf\nmax-tokens = 8192\nmax-tokens-limit = 4096\n",
                4,
                "[m]: max-tokens = 8192 is more than max-tokens-limit = 4096",
            ),
            (
                "[*]\nmax-tokens = 8192\n[m]\nmodel = /m.gguf\n[n]\nmodel = /n.gguf\n\
                 max-tokens-limit = 4096\n",
                7,
                "[n]: max-tokens = 8192 is more than max-tokens-limit = 4096",
            ),
            (
                "[*]\nmax-tokens-limit = 10\nmax-tokens = 20\n[m]\nmodel = /m.gguf\n",
                3,
                "[m]: max-tokens = 20 is more than max-tokens-limit = 10",
            ),
        ] {
            let error = parse(text, &[]).expect_err(text);
            assert_eq!(error.file, Path::new("m.ini"), "{text}");
            assert_eq!(error.line, line, "{text}: {error}");
            assert!(error.reason.contains(reason), "{text}: {error}");
            let said = error.to_string();
            let head = if line == 0 { "m.ini: ".to_owned() } else { format!("m.ini:{line}: ") };
            assert!(said.starts_with(&head), "{said}");
        }
        assert!(parse(model, &[]).is_ok());
        // A file that is not there is the file's error.
        let error = Presets::read(Path::new("/nowhere/models.ini"), &[]).unwrap_err();
        assert_eq!(error.line, 0);
        assert!(error.to_string().starts_with("/nowhere/models.ini: cannot read the preset file"));
    }

    #[test]
    fn the_prompt_cache_on_ssd_is_set_for_all_models_or_for_one() {
        let text = "[*]\ncache = /var/cache/ares\ncache-size = 64G\n\n\
                    [a]\nmodel = /a.gguf\n\n[b]\nmodel = /b.gguf\ncache-size = 8G\n";
        let presets = parse(text, &[]).unwrap();
        let [a, b] = presets.models.as_slice() else { panic!("{presets:?}") };
        assert_eq!(a.options.load.cache, Some(PathBuf::from("/var/cache/ares")));
        assert_eq!(a.options.load.cache_size, Some(64 << 30));
        assert_eq!(b.options.load.cache, Some(PathBuf::from("/var/cache/ares")));
        assert_eq!(b.options.load.cache_size, Some(8 << 30));
        // Without the key nothing is written anywhere.
        let presets = parse("[a]\nmodel = /a.gguf\n", &[]).unwrap();
        assert_eq!(presets.models[0].options.load.cache, None);
    }

    #[test]
    fn an_option_of_llama_server_is_answered_with_the_one_of_ares() {
        for (key, says) in [
            ("ctx-size = 262144", "the context is 'context'"),
            ("c = 4096", "the context is 'context'"),
            ("tensor-split = 0.625,0.375", "'shares'"),
            ("split-mode = tensor", "'split'"),
            ("device = CUDA0,CUDA1", "'devices'"),
            ("ctk = q5_1", "'kv'"),
            ("chat-template-kwargs = {\"reasoning_effort\": \"medium\"}", "'chat-var'"),
            ("spec-type = ngram-simple,draft-mtp", "'drafters'"),
            ("spec-draft-vocab = /v.txt", "'draft-vocab'"),
            ("cache-ram = 12288", "'ram'"),
            ("cache-disk = /var/cache/llama", "'cache'"),
            ("cache-disk-size = 65536", "'cache-size'"),
            ("ctx-checkpoints = 32", "the engine places its checkpoints"),
            ("cache-save-min = 1024", "the engine places its checkpoints"),
            ("sleep-idle-seconds = 600", "'idle-unload'"),
            ("alias = x", "'aliases'"),
            ("stop-timeout = 180", "an unload writes nothing"),
            ("ngl = 99", "all of a model is on the devices"),
            ("jinja = true", "that is always on"),
            // The sampler's defaults and the length of a reply.
            ("temp = 0.7", "'temperature'"),
            ("top-nsigma = 2", "'top-n-sigma'"),
            ("typical = 0.9", "'typical-p'"),
            ("dynatemp-exp = 1.5", "'dynatemp-exponent'"),
            ("mirostat-lr = 0.1", "'mirostat-eta'"),
            ("mirostat-ent = 5", "'mirostat-tau'"),
            ("dry-sequence-breaker = none", "'dry-sequence-breakers'"),
            ("sampler-seq = edskypmxt", "'samplers'"),
            ("sampling-seq = edskypmxt", "'samplers'"),
            ("s = 42", "'seed'"),
            ("n-predict = 96000", "'max-tokens'"),
            ("predict = 96000", "'max-tokens-limit' is the most any request gets"),
            ("n = 96000", "'max-tokens'"),
            ("ignore-eos = true", "ignore_eos"),
            ("logit-bias = 15043+1", "logit_bias"),
            ("adaptive-target = 0.5", "adaptive-p is not built"),
        ] {
            let said = refused(&format!("[m]\nmodel = /m.gguf\n{key}\n"));
            let name = key.split(' ').next().unwrap();
            assert!(said.starts_with("m.ini:3: Ares has no option '"), "{said}");
            assert!(said.contains(name) && said.contains(says), "{said}");
        }
        // The field of a request is a key under its name with dashes.
        let said = refused("[m]\nmodel = /m.gguf\ntop_p = 0.95\n");
        assert!(said.ends_with("Ares has no option 'top_p': the key is 'top-p', with dashes"));
        let said = refused("[m]\nmodel = /m.gguf\nmax_tokens = 100\n");
        assert!(said.contains("the key is 'max-tokens'"), "{said}");
        // One that Ares has nothing for is named and not taken.
        let said = refused("[m]\nmodel = /m.gguf\nmmproj-offload = false\n");
        assert!(said.contains("'mmproj-weights'"), "{said}");
    }

    #[test]
    fn the_defaults_of_requests_are_keys_of_a_model_or_of_all_models() {
        // The sampling lines of the production preset of llama-server stand as
        // they are; `n-predict` is `max-tokens`.
        let text = "\
[*]
max-tokens = 96000
reasoning = on

[Qwen3.8-27b]
model = /models/Qwen3.8-27B-UD-Q4_K_S.gguf
temperature = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0

[small]
model = /models/small.gguf
temperature = 0.7
max-tokens = 4096
max-tokens-limit = 8192
seed = 7
samplers = penalties, dry, top_k, top_p, min_p, temperature
dry-sequence-breakers = [\"\\n\", \"a = b\"]
";
        let presets = parse(text, &[]).unwrap();
        let [big, small] = presets.models.as_slice() else { panic!("{presets:?}") };
        let defaults = &big.options.defaults;
        let sampling = ares_api::Sampling {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            presence_penalty: Some(0.0),
            repeat_penalty: Some(1.0),
            ..ares_api::Sampling::default()
        };
        assert_eq!(defaults.sampling, sampling);
        assert_eq!((defaults.max_tokens, defaults.max_tokens_limit), (Some(96000), None));

        let defaults = &small.options.defaults;
        assert_eq!(defaults.sampling.temperature, Some(0.7));
        assert_eq!(defaults.sampling.top_p, None, "another model's key is not this one's");
        assert_eq!((defaults.max_tokens, defaults.max_tokens_limit), (Some(4096), Some(8192)));
        assert_eq!(defaults.sampling.seed, Some(7));
        assert_eq!(defaults.sampling.samplers.as_ref().map(Vec::len), Some(6));
        // A value may hold an equals sign and a backslash.
        let breakers = vec!["\n".to_owned(), "a = b".to_owned()];
        assert_eq!(defaults.sampling.dry_sequence_breakers, Some(breakers));
    }

    #[test]
    fn the_command_line_takes_part_in_what_has_to_hold_together() {
        let text = "[*]\nmax-tokens = 8192\n[m]\nmodel = /m.gguf\n[n]\nmodel = /n.gguf\n\
                    max-tokens = 2048\n";
        // A limit on the command line holds for every model: the default of
        // [*], which [m] takes, is above it, and the line is that key's.
        let error = parse(text, &[("--max-tokens-limit", "4096")]).unwrap_err();
        assert_eq!(error.line, 2, "{error}");
        assert!(
            error.reason.contains("[m]: max-tokens = 8192 is more than max-tokens-limit = 4096"),
            "{error}"
        );
        // The command line's value takes the place of the file's, and then
        // they hold together.
        let given = [("--max-tokens-limit", "4096"), ("--max-tokens", "100")];
        let presets = parse(text, &given).unwrap();
        for model in &presets.models {
            let defaults = &model.options.defaults;
            assert_eq!((defaults.max_tokens, defaults.max_tokens_limit), (Some(100), Some(4096)));
        }
        // Two options of the command line that do not: no line of the file.
        let given = [("--max-tokens-limit", "10"), ("--max-tokens", "100")];
        let error = parse(text, &given).unwrap_err();
        assert_eq!(error.line, 0, "{error}");
        assert!(error.to_string().starts_with("m.ini: [m]: max-tokens = 100 is more"), "{error}");
    }

    #[test]
    fn idle_times_are_seconds_or_a_number_with_a_unit() {
        let time = |value: &str| {
            let text = format!("[m]\nmodel = /m.gguf\nidle-unload = {value}\n");
            parse(&text, &[]).map(|presets| presets.models[0].idle_unload)
        };
        assert_eq!(time("600"), Ok(Some(Duration::from_secs(600))));
        assert_eq!(time("90s"), Ok(Some(Duration::from_secs(90))));
        assert_eq!(time("30m"), Ok(Some(Duration::from_secs(1800))));
        assert_eq!(time("2h"), Ok(Some(Duration::from_secs(7200))));
        assert_eq!(time("never"), Ok(None));
        for bad in ["", "m", "1d", "1.5h", "-1", "99999999999999999999h"] {
            assert!(time(bad).is_err(), "{bad:?}");
        }
    }
}
