//! The options: [`OPTIONS`], one table of every option of the server and of a
//! model with its names, where it may stand and its help, and the options of
//! one model, [`ModelOptions`]. What `ares --model` takes on its command line
//! (and in the environment and a configuration file) for the model it
//! serves, and what a section of a preset file holds under the same names,
//! go through [`lookup`] and [`set`], so there is one vocabulary and one
//! check of every value.

use std::path::PathBuf;

use ares_cuda::Precision;
use ares_tokenizer::Json;

use crate::ReasoningFormat;
use crate::defaults::{self, Conflict, Defaults, Setting, field};
use crate::plan::{self, Drafter, ExpertPrune, LoadOptions, SplitKind};

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

/// Who an option is for, which says where it may stand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// The command line alone: `--config` names where the others come from
    /// (or `ARES_CONFIG`), `--help` prints them.
    Command,
    /// The server, which no model has: the command line, its variable of the
    /// environment and the configuration file, not a section of a preset file.
    Server,
    /// A model: every place, a section of a preset file among them.
    Model,
}

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

/// One option: its names, where it may stand, how it is written, and its
/// lines in `ares --help`. Its key in a configuration or preset file is the
/// name without the dashes (`context`), its variable of the environment
/// `ARES_` and that key in capitals with underscores (`ARES_CONTEXT`).
#[derive(Debug, Clone, Copy)]
pub struct Opt {
    /// The one name (`--context`), under which a setter takes it.
    pub flag: &'static str,
    /// A shorter spelling of the command line, as llama-server has it (`-rea`).
    pub short: Option<&'static str>,
    /// What the value is, in the help (`<TOKENS>`); `None` for a switch.
    pub value: Option<&'static str>,
    /// The spelling of the command line that switches a switch off.
    pub off: Option<&'static str>,
    pub scope: Scope,
    /// Whether it may be given several times, each adding to the others.
    pub repeats: bool,
    /// The heading of the group of options it opens in the help.
    pub heading: Option<&'static str>,
    /// What it does, and its default, as the help says it: a paragraph, or
    /// lines apart where it has `\n`.
    pub help: &'static str,
    /// For a default of a field of a request: how the field is read, shown
    /// and filled.
    pub(crate) setting: Option<Setting>,
}

impl Opt {
    const fn new(
        scope: Scope,
        flag: &'static str,
        value: &'static str,
        help: &'static str,
    ) -> Self {
        Self {
            flag,
            short: None,
            value: Some(value),
            off: None,
            scope,
            repeats: false,
            heading: None,
            help,
            setting: None,
        }
    }

    const fn command(flag: &'static str, value: &'static str, help: &'static str) -> Self {
        Self::new(Scope::Command, flag, value, help)
    }

    const fn server(flag: &'static str, value: &'static str, help: &'static str) -> Self {
        Self::new(Scope::Server, flag, value, help)
    }

    const fn model(flag: &'static str, value: &'static str, help: &'static str) -> Self {
        Self::new(Scope::Model, flag, value, help)
    }

    /// The default of the field of a request that `setting` sets.
    pub(crate) const fn field(
        flag: &'static str,
        value: &'static str,
        help: &'static str,
        setting: Setting,
    ) -> Self {
        let mut opt = Self::model(flag, value, help);
        opt.setting = Some(setting);
        opt
    }

    /// The option is a switch; `off` is the spelling that switches it off,
    /// if it has one.
    const fn switch(mut self, off: Option<&'static str>) -> Self {
        self.value = None;
        self.off = off;
        self
    }

    const fn short(mut self, short: &'static str) -> Self {
        self.short = Some(short);
        self
    }

    const fn repeats(mut self) -> Self {
        self.repeats = true;
        self
    }

    const fn heading(mut self, heading: &'static str) -> Self {
        self.heading = Some(heading);
        self
    }

    /// The key of the option in a file.
    pub fn key(&self) -> &'static str {
        self.flag.trim_start_matches('-')
    }

    /// The value a key of a file gives the option's setter: a switch takes
    /// `true` or `false`.
    pub fn value_of_key(&self, value: &str) -> Result<String, String> {
        match (self.value, value) {
            (Some(_), _) | (None, "true" | "false") => Ok(value.to_owned()),
            (None, other) => Err(format!("{} = '{other}': write true or false", self.key())),
        }
    }
}

/// Every option, in the order of the help.
pub const OPTIONS: &[Opt] = &[
    Opt::command(
        "--config",
        "<FILE>",
        "A file of options, one on each line: name = value; lines that begin with # or ; are \
         comments [default: ARES_CONFIG, else none]",
    ),
    Opt::command("--help", "", "Print this help").short("-h").switch(None),
    Opt::model(
        "--model",
        "<PATH>",
        "GGUF model file to load and serve (any part of a split model)",
    ),
    Opt::server(
        "--models-preset",
        "<FILE>",
        "Serve several models from a preset file: an INI file with one section for each \
         model, named by the id requests use, and a section [*] for all of them. A section \
         holds the options below under their names (context = 32768), and aliases, \
         load-on-startup and idle-unload. A model is loaded when a request names it and \
         unloaded when another needs its room or when it was idle for its idle-unload. An \
         option given here on the command line holds for every model",
    ),
    Opt::server("--host", "<ADDR>", "Address to listen on [default: 127.0.0.1]"),
    Opt::server("--port", "<PORT>", "Port to listen on, 0 picks a free one [default: 8000]"),
    Opt::server(
        "--timeout",
        "<SECONDS>",
        "How long a client may take to send a request, and to take what one write of an \
         answer sent; then the connection is closed and its request ended. A reply the \
         engine takes long for is never cut [default: 600]",
    ),
    Opt::server(
        "--api-key",
        "<KEYS>",
        "Keys a request must carry one of, separated by commas, as 'Authorization: Bearer \
         <key>' or 'x-api-key: <key>'; /health stays open [default: none, no key is asked for]",
    )
    .heading("Who may use the server."),
    Opt::server(
        "--api-key-file",
        "<FILE>",
        "A file of more keys, one on each line; lines that begin with # are comments",
    ),
    Opt::server(
        "--allowed-hosts",
        "<LIST>",
        "Names in a request's Host that the server answers to besides its local ones \
         (localhost, addresses), separated by commas, or * for any. A server on a local \
         address without keys answers only to its local names, against web pages that \
         rebind a name of theirs to it; with these names the check holds for every address \
         and with keys too [default: the local names on a local address without keys, else \
         any]",
    ),
    Opt::server(
        "--cors-origins",
        "<LIST>",
        "Origins whose web pages may call the server from a browser and read its answers \
         (CORS), such as https://chat.example.org, separated by commas, or * for any. Without \
         keys a request from a page of any other site is refused, unless the page is local \
         (localhost, an address) or of the server's own name or an allowed host [default: \
         none]",
    ),
    Opt::model(
        "--devices",
        "<LIST>",
        "CUDA devices to use, by the driver's index: 0,1; a GPU of a class this build does \
         not run is skipped [default: all]",
    )
    .heading(
        "Where the model goes. What is left out is decided from the devices and the model, \
         and the log says how.",
    ),
    Opt::model(
        "--split",
        "<KIND>",
        "single: one device holds the model\n\
         tensor: every device holds a slice of every block (Qwen3.8-Flash-Next: two devices)\n\
         layer: every device holds a range of blocks\n\
         [default: single on one device. For the 27B single if one device holds the model \
         with the context named (8192 tokens without --context), else tensor; for \
         Qwen3.8-Flash-Next tensor on two devices that can write each other's memory, else \
         layer]",
    ),
    Opt::model(
        "--shares",
        "<LIST>",
        "Each device's share of the weights under a split: 0.625,0.375 [default: by the \
         memory bandwidth the devices report, as far as their memory allows]",
    ),
    Opt::model(
        "--context",
        "<TOKENS>",
        "Tokens of sequence state held for all sequences together; one request holds at \
         most this or the model's own context [default: what the devices' free memory holds]",
    )
    .heading("Sequence state."),
    Opt::model(
        "--parallel",
        "<N>",
        "Sequences that run at once; each takes memory for its recurrent state, further \
         requests wait [default: 4]",
    ),
    Opt::model(
        "--kv",
        "<FORMAT>",
        "Storage format of K and V: f16, q8, q7, q6, q5 or q4, or two names (K,V); both \
         models take every format. The formats from q7 on store a tile of 128 cells relative \
         to its mean and keep the first tile and the newest cells exact [default: q8]",
    ),
    Opt::model(
        "--ram",
        "<SIZE>",
        "The most RAM the whole engine may use, such as 16G or 512M, for all loaded models \
         together; ram in a model's section of a preset file is that model's part of it, \
         and a model without one takes what is left. What the engine needs to run is taken \
         from it first (with --cache that includes one checkpoint and the pages and \
         checkpoints state passes through on its way to the drive and back); the rest is \
         for the experts of a mixture of experts that have no slot in VRAM, and then for \
         the prompt cache in RAM: the K and V that left the devices and the checkpoints of \
         recurrent state that let a conversation continue from what was computed. Below the \
         limit the engine holds what the machine has available less a small headroom, and \
         gives memory back when other programs need it, the prompt cache first [default: \
         all the machine has]",
    )
    .heading("Host memory."),
    Opt::model(
        "--cache",
        "<DIR>",
        "A directory in which the engine keeps the state of what it has computed (K and V, \
         and checkpoints of recurrent state), so that a prompt that starts with it resumes \
         from there after the state has left RAM, after the model was unloaded and after a \
         restart. Texts of 2048 tokens and more are written when they leave RAM, and what \
         RAM and the devices hold when the model is unloaded or the server stops; a store \
         fits one model file with its formats and split, and the directory may be shared by \
         all models. A mixture of experts keeps the usage profile of its experts there, \
         which fills its tiers at the next start [default: none, and nothing is written; \
         the profiles are then kept in $XDG_CACHE_HOME/ares, else in ~/.cache/ares]",
    )
    .heading("Prompt cache on SSD, below the one in RAM."),
    Opt::model(
        "--cache-size",
        "<SIZE>",
        "The most that directory may take, such as 64G; what was used least recently goes \
         first [default: 32G]",
    ),
    Opt::model(
        "--precision",
        "<KIND>",
        "fast: 8-bit activations where they are faster\n\
         exact: f16 products everywhere, the same text whatever else runs in a step\n\
         [default: fast]",
    )
    .heading("Speed against exactness."),
    Opt::model(
        "--drafters",
        "<LIST>",
        "The model drafters that take part, such as mtp (the MTP block in the model file) \
         or none; a file name is a drafter from a file of its own, an MTP block or a DFlash2 \
         drafter, by the architecture the file names. The n-gram drafter takes part with \
         them and never alone; with none the server does not speculate. The length of a \
         draft and whose it is are chosen for every step from the price of a step and the \
         acceptance measured [default: for the 27B all it has, mtp if its file has an MTP \
         block; for Qwen3.8-Flash-Next none, it speculates only with a drafter named here]",
    ),
    Opt::model(
        "--batch-size",
        "<ROWS>",
        "Most rows of prompt in one step while a prompt is read alone. Above --ubatch-size \
         only for a mixture of experts (Qwen3.8-Flash-Next): its expert slots in VRAM lend \
         the room for the step's rows for its time, and every expert is read once a step, not \
         once per --ubatch-size rows. With K and V in f16 or q8 and no DFlash drafter such a \
         step is read layer by layer, and a request that arrives waits for a piece of it; \
         otherwise it waits for the step in flight [default: for a mixture of experts 32768 \
         layer by layer, else 8192, as far as the slots can lend the room; for the 27B \
         --ubatch-size]",
    )
    .short("-b"),
    Opt::model(
        "--ubatch-size",
        "<ROWS>",
        "Most rows of a step in the memory the model keeps for good, and of a piece of \
         prompt while another request waits [default: what a thirty-second of the smaller \
         device's memory holds for a mixture of experts, 512 for the 27B]",
    )
    .short("-ub"),
    Opt::model(
        "--window",
        "<ROWS>",
        "Most rows of one sequence's window, its newest token and the drafts verified with \
         it, at least 2; Qwen3.8-Flash-Next takes at most 8. The room for them is VRAM \
         (about 5 MiB a row for the 27B, for every sequence) [default: 8 for the 27B, the \
         rows one launch of the few-row weight kernels takes; 4 for Qwen3.8-Flash-Next, \
         where a rejected draft costs what a token costs]",
    ),
    Opt::model(
        "--draft-vocab",
        "<FILE>",
        "Token ids (one per line) the model drafters (mtp and dflash) may propose: a shorter \
         list makes a draft cheaper [default: every token whose head row is on the drafters' \
         device]",
    ),
    Opt::model(
        "--expert-prune",
        "<RULE:RANK>",
        "For a mixture of experts: what a step does with a pick whose expert only the SSD \
         holds, from the RANK-th pick of a row on (1 the strongest), instead of waiting for \
         the drive. skip leaves it out, router takes the router's next best expert that VRAM \
         or RAM holds, similar the one most like it there (from a table of the model file \
         kept beside the usage profile). Changes the output; similar:8 touches the 8th to \
         10th of ten picks [default: off]",
    ),
    Opt::model(
        "--mmproj",
        "<FILE>",
        "The projector the model reads images through: the mmproj GGUF of its image tower \
         beside the model file. Images come in requests as data URLs (image_url, Anthropic \
         image blocks, input_image); remote URLs are not fetched [default: none, images are \
         refused]",
    )
    .heading("Images."),
    Opt::model(
        "--mmproj-weights",
        "<ram|vram>",
        "Where the projector's weights stay between images: ram, page-locked inside --ram \
         and uploaded block by block for every new image, or vram, which the context gives \
         up (about 0.9 GiB) [default: ram]",
    ),
    Opt::model(
        "--image-max-tokens",
        "<ROWS>",
        "The most rows (tokens) one image becomes, 8 to 4096; a larger image is shrunk to \
         fit. The VRAM of the tower's scratch follows it (about 340 MiB at 4096) [default: \
         4096, as llama.cpp]",
    ),
    Opt::model(
        "--chat-var",
        "<NAME=VALUE>",
        "A variable of the model's chat template, such as add_vision_id=true; may be given \
         several times",
    )
    .repeats()
    .heading("Prompts."),
    Opt::model(
        "--chat-template-file",
        "<FILE>",
        "A chat template (Jinja) that takes the place of the model file's, for every \
         conversation [default: the model file's]",
    ),
    Opt::model(
        "--queue",
        "<N>",
        "The most requests that may wait for a sequence while --parallel run; one more is \
         refused with 503 and Retry-After [default: no limit, a request waits until it runs]",
    )
    .heading("The line of requests."),
    Opt::model(
        "--generation-floor",
        "<TOKENS/S>",
        "The rate every generating request keeps while a prompt is read beside it; the \
         prompt gets the rest of the devices' time, and at least half. 0 reads prompts at \
         once (a request that generates then gets a token per step of prompt) [default: 20]",
    ),
    Opt::model(
        "--reasoning",
        "<on|off|auto>",
        "Whether the model thinks before it answers: auto leaves it to the chat template \
         [default: auto]",
    )
    .short("-rea")
    .heading(
        "Thinking, under the names llama-server gives these. A request can change each of \
         them for itself: enable_thinking and the other variables in chat_template_kwargs, \
         reasoning_effort, reasoning_format; its budget is among the defaults of the fields \
         below.",
    ),
    Opt::model(
        "--reasoning-format",
        "<FORMAT>",
        "none: the thinking stays in content, tags and all\n\
         auto: the thinking goes to reasoning_content, in streamed deltas too\n\
         [default: auto]",
    ),
    Opt::model(
        "--reasoning-effort",
        "<LEVEL>",
        "A level the chat template knows, such as low or medium, or default to leave it to \
         the template [default: default]",
    ),
    Opt::model(
        "--reasoning-preserve",
        "",
        "Whether the thinking of earlier turns stays in the prompt, as a client sends it back \
         in reasoning_content: it must for a conversation to continue from what was computed \
         [default: on]",
    )
    .switch(Some("--no-reasoning-preserve")),
    field!(
        "--temperature",
        temperature,
        number,
        "<T>",
        "a number from 0 on (0 picks the best token)",
        "0 picks the most probable token [default: 0]"
    )
    .heading(
        "What a request gets for a field it leaves out. An option has the name of the field \
         with dashes (--top-p for top_p), and a request's own value always holds. The \
         defaults named here are the engine's own.",
    ),
    field!(
        "--top-k",
        top_k,
        count,
        "<N>",
        "a number of tokens, 0 for no cut",
        "Keep the N most probable tokens, 0 for all [default: 0]"
    ),
    field!(
        "--top-p",
        top_p,
        number,
        "<P>",
        "a number above 0 and at most 1 (1 cuts nothing)",
        "Keep the most probable tokens that hold P of the probability, 1 for all [default: 1]"
    ),
    field!(
        "--min-p",
        min_p,
        number,
        "<P>",
        "a number from 0 to 1 (0 cuts nothing)",
        "Drop tokens less than P times as probable as the best, 0 for none [default: 0]"
    ),
    field!(
        "--typical-p",
        typical_p,
        number,
        "<P>",
        "a number above 0 and at most 1 (1 cuts nothing)",
        "Locally typical sampling, 1 is off [default: 1]"
    ),
    field!(
        "--top-n-sigma",
        top_n_sigma,
        number,
        "<N>",
        "a number (0 or less cuts nothing)",
        "Keep the tokens within N standard deviations of the best logit, 0 is off \
         [default: 0]"
    ),
    field!(
        "--xtc-probability",
        xtc_probability,
        number,
        "<P>",
        "a number from 0 to 1 (0 is off)",
        "XTC: the chance that the most probable tokens are removed [default: 0, off]"
    ),
    field!(
        "--xtc-threshold",
        xtc_threshold,
        number,
        "<P>",
        "a number from 0 to 1",
        "XTC: the probability from which a token is one of them [default: 0.1]"
    ),
    field!(
        "--dynatemp-range",
        dynatemp_range,
        number,
        "<R>",
        "a number from 0 on (0 is off)",
        "A temperature that moves by R around --temperature with the entropy [default: 0, \
         off]"
    ),
    field!(
        "--dynatemp-exponent",
        dynatemp_exponent,
        number,
        "<E>",
        "a number from 0 on",
        "The exponent of that move [default: 1]"
    ),
    field!(
        "--repeat-penalty",
        repeat_penalty,
        number,
        "<N>",
        "a number above 0 (1 changes nothing)",
        "Lowers the logits of the tokens that stand in the last --repeat-last-n tokens by \
         this factor, 1 is off [default: 1]"
    ),
    field!(
        "--presence-penalty",
        presence_penalty,
        number,
        "<N>",
        "a number from -2 to 2",
        "Subtracted from the logits of those tokens once [default: 0]"
    ),
    field!(
        "--frequency-penalty",
        frequency_penalty,
        number,
        "<N>",
        "a number from -2 to 2",
        "Subtracted from the logits of those tokens for every time they stand there \
         [default: 0]"
    ),
    field!(
        "--repeat-last-n",
        repeat_last_n,
        count,
        "<N>",
        "a number of tokens from 0 to 1024 (0 switches the three penalties off)",
        "The tokens the three penalties look at, at most 1024, 0 switches them off \
         [default: 64]"
    ),
    field!(
        "--dry-multiplier",
        dry_multiplier,
        number,
        "<N>",
        "a number from 0 on (0 is off)",
        "DRY, the penalty for a token that would repeat a sequence: its strength \
         [default: 0, off]"
    ),
    field!(
        "--dry-base",
        dry_base,
        number,
        "<N>",
        "a number from 0 on",
        "DRY's base, which the penalty grows by with the length of the repeat [default: 1.75]"
    ),
    field!(
        "--dry-allowed-length",
        dry_allowed_length,
        count,
        "<N>",
        "a number of tokens up to 1024",
        "The longest repeat DRY lets pass [default: 2]"
    ),
    field!(
        "--dry-penalty-last-n",
        dry_penalty_last_n,
        count,
        "<N>",
        "a number of tokens up to 1024",
        "The tokens DRY looks back on [default: 64]"
    ),
    field!(
        "--dry-sequence-breakers",
        dry_sequence_breakers,
        strings,
        "<JSON>",
        "a JSON array of strings, such as [\"\\n\", \":\"]",
        "The strings a repeated sequence does not reach across, as a JSON array \
         [default: [\"\\n\", \":\", \"\\\"\", \"*\"]]"
    ),
    field!(
        "--mirostat",
        mirostat,
        version,
        "<0|1|2>",
        "0 (off), 1 or 2",
        "Mirostat in place of the other samplers [default: 0, off]"
    ),
    field!(
        "--mirostat-tau",
        mirostat_tau,
        number,
        "<N>",
        "a number above 0",
        "Mirostat's target surprise [default: 5]"
    ),
    field!(
        "--mirostat-eta",
        mirostat_eta,
        number,
        "<N>",
        "a number from 0 on",
        "Mirostat's rate [default: 0.1]"
    ),
    field!(
        "--samplers",
        samplers,
        order,
        "<LIST>",
        "names of samplers separated by commas, penalties and dry before the others and once \
         each, at most 16 others (penalties,dry,top_n_sigma,top_k,typ_p,top_p,min_p,xtc,\
         temperature)",
        "The samplers that apply and their order, by name [default: \
         penalties,dry,top_n_sigma,top_k,typ_p,top_p,min_p,xtc,temperature]"
    ),
    field!(
        "--seed",
        seed,
        seed,
        "<N>",
        "a whole number from 0 on (without the option every request draws its own)",
        "The seed of a request that names none, which then draws the same for the same \
         prompt; a request asks for a draw of its own with seed -1 [default: a new one for \
         every request]"
    ),
    field!(
        "--reasoning-budget-tokens",
        reasoning_budget_tokens,
        count,
        "<N>",
        "a number of tokens of thinking, after which the thinking is ended",
        "The most tokens the thinking of a reply may take: then the budget's message and the \
         end of the thinking are written, and the answer follows [default: none, no budget]"
    ),
    field!(
        "--reasoning-budget-message",
        reasoning_budget_message,
        text,
        "<TEXT>",
        "a text written before the end of the thinking when its budget is spent",
        "What is written before the end of the thinking when its budget is spent \
         [default: none]"
    ),
    Opt::model(
        defaults::MAX_TOKENS,
        "<N>",
        "The limit of tokens for a request that names none (max_tokens, \
         max_completion_tokens, max_output_tokens, n_predict) [default: none, a reply ends \
         when the model ends it or the context is full]",
    )
    .heading("The length of a reply."),
    Opt::model(
        defaults::MAX_TOKENS_LIMIT,
        "<N>",
        "The most tokens of any reply: a request that asks for more gets this many, and its \
         response says so in the header x-ares-max-tokens-limit; a request that names no \
         limit gets --max-tokens or, without it, this [default: none]",
    ),
    Opt::server(
        "--log-format",
        "<text|json>",
        "text: a line for people\n\
         json: one JSON object for every line, for collectors\n\
         [default: text]",
    )
    .heading("The log."),
    Opt::server("--log-file", "<FILE>", "Append the log to this file [default: standard error]"),
    Opt::server(
        "--log-text",
        "",
        "The prompt and the reply of every request go into the log as well; without it the \
         log holds neither",
    )
    .switch(None),
];

/// The option that is written `name` on the command line (`--context`,
/// `-rea`, `--no-reasoning-preserve`), and how this spelling is written.
pub fn find(name: &str) -> Option<(&'static Opt, Kind)> {
    OPTIONS.iter().find_map(|opt| {
        let kind = match opt.value {
            Some(_) => Kind::Value,
            None => Kind::Switch(true),
        };
        if opt.flag == name || opt.short == Some(name) {
            Some((opt, kind))
        } else if opt.off == Some(name) {
            Some((opt, Kind::Switch(false)))
        } else {
            None
        }
    })
}

/// The option of a model that is written `name` on the command line: its one
/// name, under which [`set`] takes it, and how this spelling is written.
/// `None` for anything else, an option of the server among it.
pub fn lookup(name: &str) -> Option<(&'static str, Kind)> {
    find(name).filter(|(opt, _)| opt.scope == Scope::Model).map(|(opt, kind)| (opt.flag, kind))
}

/// The option a key of a file names (`context`). `unknown` says, for a key
/// that is no option and no name of llama-server's, where the keys are
/// listed.
pub fn of_key(name: &str, unknown: &str) -> Result<&'static Opt, String> {
    match find(&format!("--{name}")) {
        Some((opt, Kind::Switch(false))) => Err(format!(
            "'{name}' is the command line's way to switch {0} off: write {0} = false",
            opt.key()
        )),
        Some((opt, _)) => Ok(opt),
        None => {
            // The field of a request, written as the request writes it.
            let dashed = name.replace('_', "-");
            if dashed != name && find(&format!("--{dashed}")).is_some() {
                return Err(format!(
                    "Ares has no option '{name}': the key is '{dashed}', with dashes"
                ));
            }
            Err(match other_name(name) {
                Some(hint) => format!("Ares has no option '{name}': {hint}"),
                None => format!("Ares has no option '{name}' ({unknown})"),
            })
        }
    }
}

/// What to write instead of an option of llama-server that Ares has under
/// another name, or does not have because it has nothing to choose there.
/// `name` is the key, the name without dashes.
pub fn other_name(name: &str) -> Option<&'static str> {
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
/// the model's MTP block, `none` (alone) is no drafter, anything else is
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
            file => Drafter::File(PathBuf::from(file)),
        };
        if list.contains(&drafter) {
            return Err(invalid("--drafters", value, expected));
        }
        list.push(drafter);
    }
    Ok(list)
}

/// `--expert-prune`: `off`, or a rule and the rank from which it may touch a
/// pick (`skip:8`, `router:8`, `similar:1`).
fn expert_prune(value: &str) -> Result<Option<ExpertPrune>, Invalid> {
    let expected = "off, or skip, router or similar with the rank from which a pick may be \
                    touched (similar:8)";
    if value.trim() == "off" {
        return Ok(None);
    }
    let (rule, rank) =
        value.split_once(':').ok_or_else(|| invalid("--expert-prune", value, expected))?;
    let rule = match rule.trim() {
        "skip" => ares_cuda::PruneRule::Skip,
        "router" => ares_cuda::PruneRule::Router,
        "similar" => ares_cuda::PruneRule::Similar,
        _ => return Err(invalid("--expert-prune", value, expected)),
    };
    let from_rank: u32 = number("--expert-prune", rank, expected)?;
    if from_rank == 0 {
        return Err(invalid("--expert-prune", value, expected));
    }
    Ok(Some(ExpertPrune { rule, from_rank }))
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
        "--batch-size" | "--ubatch-size" => {
            let expected = "a number of rows, at least 8";
            let rows: u32 = number(flag, value, expected)?;
            if rows < 8 {
                return Err(invalid(flag, value, expected));
            }
            if flag == "--batch-size" {
                load.batch = Some(rows);
            } else {
                load.ubatch = Some(rows);
            }
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
        "--mmproj" => {
            if value.trim().is_empty() {
                return Err(invalid("--mmproj", value, "a projector file"));
            }
            load.mmproj = Some(PathBuf::from(value));
        }
        "--mmproj-weights" => {
            load.mmproj_weights = match value {
                "ram" => plan::ProjectorWeights::Ram,
                "vram" => plan::ProjectorWeights::Vram,
                _ => return Err(invalid("--mmproj-weights", value, "ram or vram")),
            };
        }
        "--image-max-tokens" => {
            let rows: u32 = number("--image-max-tokens", value, "a number of rows, 8 to 4096")?;
            if !(plan::IMAGE_MIN_TOKENS..=plan::IMAGE_MAX_TOKENS).contains(&rows) {
                return Err(invalid("--image-max-tokens", value, "a number of rows, 8 to 4096"));
            }
            load.image_max_tokens = rows;
        }
        "--chat-template-file" => {
            if value.trim().is_empty() {
                return Err(invalid("--chat-template-file", value, "a file"));
            }
            load.chat_template = Some(PathBuf::from(value));
        }
        "--queue" => {
            load.queue = Some(number("--queue", value, "a number of requests, 0 or more")?);
        }
        "--generation-floor" => {
            let expected = "a rate in tokens per second, 0 or more";
            let floor: f64 = number("--generation-floor", value, expected)?;
            if !(floor >= 0.0 && floor.is_finite()) {
                return Err(invalid("--generation-floor", value, expected));
            }
            load.generation_floor = Some(floor);
        }
        "--expert-prune" => load.expert_prune = expert_prune(value)?,
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
            options.reasoning_format = ReasoningFormat::from_name(value)
                .ok_or_else(|| invalid("--reasoning-format", value, "none or auto"))?;
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
