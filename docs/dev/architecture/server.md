# Server

`ares --model <file.gguf>` loads a model on the devices and serves it;
`ares --models-preset <file.ini>` serves several, each loaded when a request
names it (see "Several models" below).
Without options it plans for itself (the 27B here; Flash-Next's plan is at the end of
[API](api.md)): the first device that holds the weights
and a context, or else a tensor split over all devices with shares from the
memory bandwidth each device reports, and the model's own context (its
file's context length) as far as the VRAM that is left holds it (a tensor
split moves attention heads with their K and V to the devices with room as
far as that context needs: see "What the second card is for"); the start-up
report says what was chosen, why, what context was wanted and how much of it
is in VRAM, and where the VRAM went. Devices, split, shares, context, K/V format, the model
drafters that take part (`--drafters`) and the room for their windows
(`--window`), and the RAM the engine may use (`--ram`) can be given instead. The RAM limit defaults
to all the machine has, and below any limit the engine holds what the machine
has available less a headroom (see "Host memory"); the
engine's step buffers are taken from it, and the rest is the budget of the
checkpoints that let a conversation continue from what was computed (see
"Host memory"). `--cache <dir>` gives the prompt cache its level on SSD and
`--cache-size` the most the directory may take (32G when it is not given);
without a directory nothing is written to any drive (see "On SSD" under
"Sequence state"). In a preset file the keys are `cache` and `cache-size`,
for all models in `[*]` or for one. What a request gets for a sampling field
it leaves out, and how long a reply may be, are options of the model too
(`--temperature`, `--top-p`, `--max-tokens` and the others: see "The defaults
of a model's requests" below).

It answers `POST /v1/chat/completions` and `/v1/completions`, streamed and
not, Anthropic's `POST /v1/messages` and `/v1/messages/count_tokens` (see
"The Messages API" below), OpenAI's `POST /v1/responses` without state, with
`/v1/responses/input_tokens` and `/v1/chat/completions/input_tokens`, which
count a request and generate nothing (see "The Responses API" below),
`GET /v1/models`, `/health` and `/metrics`, `GET /props`, `POST /tokenize` and
`/detokenize` (see "Tokens and properties" below), with
timings in every answer of the OpenAI routes
(prompt tokens and time, generated tokens and time, drafts proposed and
accepted, and `token_hash`, a hash of the reply's token ids in the final message
and in the last event of a stream). A request can switch drafting off for itself
with `"draft": false`; the two replies of one request, with and without drafts,
are then compared by their hashes over HTTP (they are equal: see
"The text with drafts is the text without" under [Speculation](speculation.md)).
The prefix cache is on for both models without a flag: a prompt that
starts with text the engine has computed reads the cached part from the pages
and the checkpoint instead of computing it, the answer's `usage` says how much
(`prompt_tokens_details.cached_tokens`, as OpenAI does), `timings.prompt_n` counts
the tokens that were computed, and `/metrics` has the counters (starts from the
cache, cells taken from it, rows replayed, checkpoints saved, restored and
dropped, pages copied, pages cached, RAM locked for checkpoints). The sampling
fields of the OpenAI API and of llama.cpp's server are read under their names
(see [Sampling](sampling.md)), `n` completions of a prompt are `n` requests that
read the prompt once, and the probabilities of the tokens come back in the
answer, streamed or not, and `grammar`, `json_schema` and `response_format` make the
reply follow a GBNF grammar or a JSON schema (see [Sampling](sampling.md), "Structured
output"; the same fields on chat and on completions, and a model that thinks is free
while it thinks). A field of a request that the server does not hand to
the engine is refused by name, not ignored: images without `--mmproj`, and
the sampler's few limits (a history of more than 1024 tokens for the
penalties and DRY, a DRY length allowed above 1024, more than 20
alternatives, `min_keep` above 1, adaptive-p).
A client that disconnects frees its sequence. The
first signal lets running requests finish, the second ends them, and the
devices are released before the process exits.

**Where the server's code is** (`crates/server/src`). Each API has a module
of its own: `openai.rs` (the chat and completion routes, with
`openai/request.rs`, the reader of a chat or completion body, which every
other API writes its body over into, and `openai/reply.rs`, the answers and
the model list), `anthropic.rs` and `anthropic/` (Messages), `responses.rs`
and `responses/` (Responses), `systemone.rs` and `props.rs` (`/props`,
`/tokenize`, `/detokenize`). They share `chat_body.rs` (the toolkit of the
readers that write a chat body), `generate.rs` (preparing a request for its
model, its run in the engine, the count of a prompt), `sorter.rs` with
`reasoning.rs` and `tools.rs` (the reply's text sorted into reasoning,
content and calls), `answer.rs` (the answer of Messages and Responses, whole
or streamed), and `bridge.rs` (the hand-over between the engine's threads and
the handlers). `routes.rs` has the routes that are no API's own (health, the
model list, the loads and events of router mode) and what every handler
uses; `metrics.rs` is `/metrics`, with `metrics/latency.rs`. The options are
`options.rs` (the table), `defaults.rs`, `ini.rs`, `preset.rs` and the
binary's `cli.rs`; `models.rs` keeps the models of router mode, and `load/`
with `plan.rs` loads one onto the devices: `load/qwen35.rs` the dense model and
`load/qwen4exp.rs` Flash-Next, each in its phases (what was asked and can be
refused before a device is touched, the devices and the split, the shares or
the role of the second device, the context, the rows of a step, the plan in
the log, the model built, the engine started, what it took), sharing
`load/files.rs` (the drafters, the draft vocabulary, the projector, the chat
template), `load/devices.rs` (the devices and their links), `load/ram.rs` (one
budget of host RAM for both: the limit, what the engine needs to run, the
plan's line of the RAM, the machine's memory the engines follow) and
`load/start.rs` (the runtime's settings, the start, what the engine found).
What a device takes is the model's to say, not the server's: the 27B's
estimate is `ares_model::qwen35::device::Estimate` (`plan.rs`, the placement
`Qwen35::new` makes and each device's `Footprint`), Flash-Next's placement,
pages, slot floors and lent rows are `ares_model::qwen4exp::device::runtime::plan`.

**Who may use the server** (`crates/server/src/access.rs`). One layer over every
route looks at a request in this order:

1. The name it was sent to (`Host`). A server on a local address without keys
   answers only to its local names: `localhost` and the names under it, and
   addresses written as numbers. A web page that rebinds a name of its own to
   127.0.0.1 sends that name and gets 403 (`forbidden_host`).
   `--allowed-hosts` adds names (`*` lifts the check); names given that way hold
   for every address and with keys too. A request without `Host` (HTTP/1.0)
   passes: no browser sends one.
2. A browser's preflight (`OPTIONS` with `Access-Control-Request-Method`) from
   an origin of `--cors-origins` is answered at once, without a key, which a
   browser does not send with it: `GET, POST` and the headers it asks for.
3. The page a request comes from (`Origin`). Without keys, a request a browser
   sent from a page of another site gets 403 (`forbidden_origin`): pages of
   this machine (`localhost`, a loopback address), of the server's own name, of
   an allowed host and of `--cors-origins` pass. Programs send no `Origin`.
4. The key. With keys (`--api-key`, several separated by commas, and
   `--api-key-file`, one on each line), every route but `/health` wants one,
   as `Authorization: Bearer <key>` or `x-api-key: <key>` (Anthropic's header,
   which its SDKs send). The key of a request is compared with every key in
   full, so the time says nothing about any of them but its length. Without
   one: 401 with `WWW-Authenticate: Bearer`, in Anthropic's shape on
   `/v1/messages`.

With keys the key decides, as at strata: a rebound name or a foreign page has
no key, so the name and the page are not checked unless `--allowed-hosts` says
so. An answer to an origin of `--cors-origins`, errors included, carries
`Access-Control-Allow-Origin` (and exposes `Retry-After` and the header of the
limit of tokens). The methods and headers of CORS are not options: the routes
take `GET` and `POST`, and a header a preflight asks for is allowed. A server
that listens beyond this machine without keys says so in the log at start.

**Where options come from** (`crates/server/src/cli.rs`). One table,
`options::OPTIONS`, has every option of the server and of a model: its name,
a shorter spelling of llama-server's (`-rea`, `-b`), whether it takes a value
or is a switch (and its spelling for off, `--no-reasoning-preserve`), whether
it is the server's or a model's, whether it may repeat, and its help; a
default of a request's field also has there how the field is read, shown and
filled (`defaults::Setting`). `ares --help` is printed from the table, in its
groups, so it names every option and no other. Every option is also a
variable of the environment, `ARES_` and its name in capitals with
underscores (`ARES_TOP_P` for `--top-p`), and a line of the file `--config`
(or `ARES_CONFIG`) names: `name = value` with the name without the dashes, as
a key of a preset file is written (a switch takes `true` or `false`, `#` and
`;` begin a comment, a name stands once unless its option may repeat); the
file has no sections, the models of several stand in a preset file
(`models-preset = <file>`). The two files are read by one reader
(`crates/server/src/ini.rs`), and a key is resolved to its option in one place
(`options::of_key`), which also answers a name of llama-server's with the
option of Ares that does its job, on the command line too. The command line
holds over the environment and the environment over the file; with a preset
file the three hold over its sections as the command line alone did. Only the
variables of Ares' options are read (`ARES_LIBCUDA` and the variables of tests
are other things), never a name without the prefix. An error names the
variable or the file and its line.

**What the binary is** (`crates/server/build.rs`). `ares --version` prints
`ares_server::BUILD_INFO`, `ares <version> (<commit>, <date>)`; the log's first
line and `build_info` of `/props` (llama-server's name for its build) say the
same. The build script takes the commit from `git rev-parse --short HEAD` when
the workspace is the top of a git checkout, else from `ARES_BUILD_COMMIT` (a
`git archive` snapshot, a copy without `.git`), else `unknown`; the date is
the UTC day the script ran, or that of `SOURCE_DATE_EPOCH`. It runs again when
the checkout's HEAD moves, not on every edit.

**Tokens and properties** (`crates/server/src/props.rs`). `POST /tokenize` takes
`content`, a text, and gives the tokens the engine reads it as when it is the
prompt of `/v1/completions`; or a chat request (`messages` and the fields of
`/v1/chat/completions`), and gives the tokens of the prompt it comes to through
the chat template, with the model's settings and the request's thinking and
tools, as `/v1/chat/completions/input_tokens` counts them. `with_pieces` adds
the text of every token, a string where its bytes are whole characters, else
the bytes (llama.cpp's shape). `POST /detokenize` takes `tokens` and gives the
text they write in a reply. A control token (`<|im_end|>`) writes nothing in
both, as in a reply; llama-server writes its text. A text is read the one way
a prompt of `/v1/completions` is: the text of a control token (`<|im_start|>`)
is that token, as llama-server reads the prompt of a completion and, by
default (`parse_special: true`, accepted), of `/tokenize` (until 2026-11-22
the text of a control token was text here); `parse_special: false` and
`add_special` are refused by name. The same reading applies to the text keys
of a logit bias and to DRY's breakers, which llama.cpp reads without control
tokens: a key `<|im_end|>` is the control token here. The engine
answers through `Engine::tokenize` and `Engine::pieces` (`Model::encode` and
`Model::piece` in the runtime). `model` may be left out when the server has one
model.

`GET /props` (`?model=<id>` with several models) says what a model is, in
llama.cpp's shape where it fits, with `"loaded"`; a model that is not loaded is
not loaded for it (as `/metrics` does not), and is described from its preset
and its file (`load::read_described`: the metadata and the tokenizer, no
tensor): its defaults, its chat template and what it declares, and as
`context_length` the context its preset names or else the longest its file
declares, since only a load decides what fits the devices. For a loaded model:
`context_length` (and `default_generation_settings.n_ctx`), what a request
gets for the fields it leaves out (`default_generation_settings.params`, the
`defaults` of `/v1/models`, with `max_tokens_limit`), `chat_template` (the
text of the template in effect, from `Engine::chat_template`),
`chat_template_caps` (tools and calls, thinking, whether `reasoning_effort` and
`enable_thinking` change the prompt, whether it thinks by default),
`reasoning` (its tags) and `modalities`: `vision` when the model reads images
(`Engine::reads_images`, a model loaded with a projector; for a model that is not
loaded, a preset that names `mmproj`), as llama-server says it and Open WebUI reads
it to offer an upload; `audio` is always false.

**The stop string and the thinking of a reply** (`crates/runtime/src/watch.rs`).
The conductor matches stop strings where it turns tokens into text. It carries
a `Watch` for every request, built where the request is taken
(`Runtime::start_request`), and tells it every token it accepts with its bytes
and every character of their text; the watch is the stop strings' matcher and says which
of them matched (`FinishReason::StopString`, its place in the request's list),
counts the tokens of the thinking, and notes when the request left the line.
The server names the string: `stop_reason` and `stop_sequence` on
`/v1/messages`, and `stop_reason` next to `finish_reason: "stop"` on the OpenAI
routes, as vLLM writes it (only when a stop string ended the reply).

Where the thinking is, the engine reads off the prompt: it reads the prompt's
tokens backwards to the last tag of the thinking the template declares. When
that is the opening tag (the template opened the thinking, or a continued
message stands inside it), the reply begins inside it; otherwise the reply may
open it with its first text, and an answer that begins with anything else has
none. The thinking ends at the closing tag, whose tokens are the thinking's,
or at the marker of a call that the model opens inside its thinking (the
request's rule says it), whose tokens are not. One `watch::Tracker` follows
that over the bytes of the tokens, for this count and for the guard's budget
(which does not count the tags). Tokens are counted as the conductor accepts
them, drafts or not, so the count is exact: every byte carries the number of
the token it came with, also where a token cuts a tag. The usage has it as
`completion_tokens_details.reasoning_tokens` (and `output_tokens_details` on
`/v1/responses`) for a model that thinks; Anthropic's usage has no such field.

**Continuing a message.** A chat request whose last message is the
assistant's continues it, as llama-server prefills it, unless it says
`continue_final_message: false` (vLLM's field; `add_generation_prompt: false`,
which vLLM asks to go with a continuation, says the same). A last message that
calls tools begins a turn of its own unless the request asks to continue it,
which is refused: what follows calls are their results. On `/v1/messages` a
last message of the assistant is Anthropic's prefill and is continued. The
prompt (`Prompt::Chat::continue_final`, `chat::continued` in `ares-model`) is
the conversation as the template writes it, up to the end of the last
message's text: a mark of private use characters is put at the end of that
text, the conversation rendered, and the rendering cut at the mark, so that the
template lays the turn out as it does every turn (an empty block of thinking
before an answer without reasoning, in Qwen3.8's template) and nothing of the
end of the turn follows. The text is the content, or the reasoning when the
content is empty and the reasoning is not: the reply then goes on thinking,
and the server sorts it as reasoning until the closing tag. A template that
does not write the text as it is given is refused. The reply is what comes
after the given text; the given text is not repeated.

**The template of a file.** `--chat-template-file <FILE>` (a key of a preset,
too) gives a chat template in Jinja that takes the place of the model file's
for every conversation, with tools or without. It is compiled with the model's
tokenizer at load (a template that does not compile ends the load with the
file's name and the reason), and what the server reads off a template (the
tags of the thinking, the form of the calls) is read off it. It travels in
`ChatOptions::template` of `ares-tokenizer`, so the models' own code is not
changed for it.

**The line and the timeouts.** A request waits for a sequence as long as it
takes (decision 16 of the feature checklist). `--queue <N>` bounds the line of
a model: the engine counts the requests it holds (one for every event sink it
has not let go of) and refuses one more than `--parallel` and N with
`EngineError::Overloaded`, which is 503 `overloaded` with `Retry-After: 1`.
`--timeout <SECONDS>` (600, llama-server's name) bounds what a client may take:
the head of a request (hyper's timer, which also closes a connection that waits
longer for its next request), its body (408 `request_timeout`), and one write of
an answer (`listen::Timed`: a write that waits longer for the client to take
what was sent fails, the connection closes and the request is cancelled). A
connection that waits for the engine writes nothing and is never cut. The
server accepts its connections itself (`listen.rs`, hyper's HTTP/1.1 with the
graceful shutdown of hyper-util) where it used `axum::serve`, which has no
timeouts.

**Priorities of requests.** A request has a priority, a whole number of 32
bits, a lower number first, 0 when it says nothing: the member `priority` on
`/v1/chat/completions`, `/v1/completions` and `/v1/responses` (vLLM's member,
which SGLang has too), and the header `x-ares-priority` on every route that
generates, for a client that cannot add a member to a body (Claude Code's
`ANTHROPIC_CUSTOM_HEADERS`, the `http_headers` of a provider of Codex); the
Messages API has no such member, so there it is the header alone. Body and
header that name two different ones are refused. It is always on: a server
whose requests all say nothing behaves as before. The randomized test of
admission (`crates/sched/tests/admission.rs`) plays one priority in some runs
and two or three in others, where it checks that a shortage spills the worst
priority first, that a running request gives way only to a better one, and that
a spilled request still in line holds back its own and every worse priority.

What the others do. llama.cpp's server has no priority of requests (it puts
only its own tasks, cancels and the like, at the front of its queue). vLLM
(read in its source, `vllm/v1/core/sched`) has `priority`, lower first, only
with `--scheduling-policy priority`: its line is a heap by priority and
arrival, and when a running request finds no blocks the victim is the worst
priority, latest arrival; a waiting request does not make a running one give
way. SGLang (`--enable-priority-scheduling`, read in its source through a
fetch) has `priority`, higher first unless `--schedule-low-priority-values-first`;
its line is sorted by priority and arrival, a waiting request preempts running
ones whose priority is worse by more than a threshold (10 by default) when
they free enough tokens, and a full line (`--max-queued-requests`) refuses
its worst waiting request (503) for a better one, or else the new one.

Here every place where the engine chooses between requests honours it:

- *The line* (`ares_sched::Admission`) is taken by priority first. Among the
  requests of one priority the old rules hold: a request that gave way comes
  back first, a starved one goes first and holds back the others of its
  priority and of worse ones (never a better one), then the fewest uncached
  rows.
- *Giving way.* A request that does not fit, for want of a place on the
  devices (`--parallel`) or of pages, makes running requests of a worse
  priority give way: the worst priority first, among those the one a
  shortage would pick (the fewest generated tokens), as many as it takes, or
  none when all of them would not make it fit (SGLang's rule, its check that
  the victims free enough included). Any gap of priority is enough; there is
  no threshold. The victims are spilled as for a shortage: their pages go to
  the prefix cache and their state becomes a checkpoint, so they come back as
  a prefix hit, and they do not come back in the pass that made them give
  way. What they let go of is room for the newcomer except the pages of its
  cached prefix that running sequences hold (it takes those again, and its
  charge left them out; until 2026-11-25 they were counted twice). A shortage
  of pages spills the worst priority first (vLLM's victim), then by the old
  rule. The sequence of a step the model paused (a prompt read
  layer by layer) gives way to neither (`Claims::keep`): its state is neither
  before nor after its rows.
- *The rows of a step* (`ares_sched::plan`): the newest tokens, the rows of
  decisions, the drafts and the prompts go to the better priority first; a
  starved prompt goes before a shorter one only of its own priority.
- *Prompts and generation* (`ares_sched::Share`): a started prompt that
  outranks every sequence that generates owes them nothing and is read in
  pieces at once (the generating sequences ride along with a token each); its
  pieces are owed all the same, by the prompts after it. A prompt that does
  not outrank them takes turns with them as before, whatever its priority:
  the floor of generation, not the priority, bounds what a background prompt
  costs a foreground reply (open).
- *A full line* (`--queue`): a request that comes when the line is full is
  decided by the conductor, which knows the line: it takes the place of the
  worst waiting request of a worse priority that has not started (the latest
  of them), which is refused as overloaded (503 with `Retry-After`), or is
  refused itself. A request that ran and gave way has sent its client text
  and is not refused. The refusal is an event now and no longer the answer of
  `submit`; the routes give it its 503 all the same.

`/metrics` counts the requests that gave way to a better priority
(`ares_requests_outranked_total`, counted in the requests that gave way too)
and the ones a better priority took the place of in a full line
(`ares_requests_displaced_total`). The log line of a request names its
priority.

Tests: `ares-sched` (the line, giving way for a place and for pages, nobody
spilled when it would not do, the shortage's victim, the rows of a step, the
prompt that does not wait for the debt), `crates/runtime/tests/priority.rs`
(on the synthetic model through the conductor: a request of priority 0 takes
the one place of a long request of priority 5 and ends first, while one of
priority 5 waits; the long one comes back and says what it says alone; a full
line takes a better priority and refuses the worst waiting request) and
`crates/server/tests/priority.rs` (the member and the header on all four
routes).

**The watchdog.** A device that never finishes its step leaves every request
of its engine waiting for ever. The conductor notes when it hands a step to
the devices and when the step is back (`Heartbeat`), and a thread of the
runtime looks every second: a step that has run longer than
`RuntimeConfig::watchdog` (600 s; SGLang's `--watchdog-timeout` is 300 s and
kills its server) fails the engine as a device error does (`Engine::failure`,
whose reason names the watchdog). A request that waits for such an engine
looks at the failure every second (`bridge::Run`) and ends with it, since the
stuck conductor sends nothing more; the router then unloads the model as a
broken one, and the runtime, which would wait for ever for the stuck
conductor, leaves its threads behind, with the memory they hold on the
devices, until the process ends. The limit is above any step that works: the
widest step of a prompt of Flash-Next with experts from the SSD takes tens of
seconds. A step is watched from its dispatch to its collect; the steps of the
warm-up are not.

**What the server logs and measures.** Every generating request has a line
with its parameters when it is taken (the model, the limit of tokens, the
sampling fields in effect, the stop strings and tools it has, whether it
continues a message) and one when it ends (the reason, prompt and cached
tokens, generated and reasoning tokens, the wait in the line, the times, the
drafts, the seed it asked for). `--log-text` adds the prompt and the reply,
which a log otherwise never holds. `--log-format json` writes every line as a
JSON object for collectors, `--log-file <FILE>` appends to a file instead of
standard error; the level is `RUST_LOG`. `/metrics` has, for every model, the
histograms of the requests that ended with a reply (`crates/server/src/metrics/latency.rs`):
the wait in the engine's line (`ares_request_queue_seconds`, from the watch),
the prompt after it, the time to the first token, the time per token after the
first, and the whole request, all from the hand-over to the engine.

**Several models (router mode).** `ares --models-preset <file.ini>` serves the
models of a preset file. A model is loaded when a request names it, stays
loaded while the devices and the RAM hold it next to the others, and is
unloaded when another model needs its room or when it was idle for its time.
`ares --model <file.gguf>` is the same server with one preset that is loaded
before the server listens and never unloaded; a load that fails there ends
the start, as before.

*The preset file* is an INI file (`crates/server/src/preset.rs`). The section
`[*]` holds what is the same for every model; every other section is one
model, and its name is the id requests use and `/v1/models` lists.

```ini
[*]
ram = 24G
idle-unload = 30m

[qwen-27b]
model = /models/Qwen3.8-27B-UD-Q4_K_S.gguf
aliases = default, gpt-4o
reasoning-effort = medium
load-on-startup = true

[small]
model = /models/Qwen3.5-0.8B-Q4_K_M.gguf
devices = 0
context = 16384
ram = 2G
temperature = 0.7
top-p = 0.8
max-tokens = 4096
```

A key is an option of the command line for one model under its name without
the dashes, with the value the option takes (`context`, `kv`, `devices`,
`chat-var`, which may stand several times; a switch such as
`reasoning-preserve` takes `true` or `false`; `temperature`, `top-p`,
`max-tokens` and the other defaults of a model's requests). The command line
and the file
are read by the same code (`options.rs`), so there is one vocabulary and one
check of a value. Three keys exist only in the file:

| Key | Meaning | Default |
| --- | --- | --- |
| `aliases` | more ids the model answers to, separated by commas | none |
| `load-on-startup` | `true`: loaded when the server starts, in the background, not at its first request | `false` |
| `idle-unload` | how long the model stays loaded without a request: seconds, or a number with `s`, `m` or `h`; `never` | `never` |

What holds for a model is, from the strongest: the command line (an option of
a model given next to `--models-preset` holds for every model), the model's
section, the section `[*]`, the built-in defaults. `ram` is the one key that
means something else by where it stands: on the command line and in `[*]` it
is the most RAM of the server, for all loaded models together; in a model's
section it is that model's part of it, and a model without one takes what the
limit has left when it is loaded. Without a limit of the server there are no
parts to fit: every model may hold up to its own `ram`, or all the machine
has, and the loaded models share what the machine has available through the
one watch over it (see "Host memory": the tiers of experts first and evenly,
the checkpoints with the rest, and all of them give memory back when the
machine needs it). With a limit, the parts cap the models as before and each
follows the machine below its part. `host` and `port` are the server's and stand
on the command line only; `--model` and `--models-preset` exclude each other.

Nothing in the file is ignored. The server does not start, and says the file,
the line and the reason, for: a key it does not know, a value an option does
not take, a key before the first section, a key or a section that stands
twice, a section without `model`, a model file that does not exist, an id or
alias that two sections answer to, `model` or `aliases` in `[*]`, a
model's `ram` above the server's, and a `max-tokens` above the
`max-tokens-limit` that holds for the same model. An option of llama-server is not accepted
under its name: the error names the option of Ares that does the job where
there is one (`ctx-size`: `context`; `tensor-split`: `shares`; `ctk` and
`ctv`: `kv`; `chat-template-kwargs`: `chat-var`; `sleep-idle-seconds`:
`idle-unload`; `alias`: `aliases`, and the others of `options::other_name`). What only the devices or the model file can say (a split the
devices cannot do, a context that fits nowhere) is found when the model is
loaded, and is then the reason of a failed load.

*A request and its model* (`crates/server/src/models.rs`). The `model` of a
request is an id or an alias; anything else is a 404 `model_not_found` that
lists the ids. The request takes a hold on the model for as long as it runs.
If the model is not loaded the request waits until it is, behind a load that
already runs, and the answer names the model by its id whatever alias the
request used. A client that goes away while it waits gives its place up: a
load nobody waits for is not started, and one that has begun is finished (the
way of the weights to the devices is not interrupted) and leaves the model
loaded and idle. One task does every load and every unload, one at a time, so
that what the devices report as free is what the next load is planned
against.

*Making room.* Nothing counts models. The loader plans a model as on the
command line, against what the devices report as free and what the server's
RAM limit has left, and says "no room" before it reads the weights when the
plan does not fit: the estimate of the model with the context its preset
names is more than a device has free; or, for a preset that names no context,
less than the least context a device is chosen for (8192 tokens, or the
model's own) fits; or, when the server has a RAM limit, the RAM left is less
than what the engine needs to run.
Then the model that has been idle the longest is unloaded and the load is
planned again, until it fits or no model is left, in which case the load
fails with the plan's words. A preset that names no context gets what the
devices have free when it is loaded, as on the command line, which next to
other models is less than alone: presets that are to share a device name
their `context`, and their `ram`.

A model that is serving requests is never unloaded under them. When a load
finds no room and only such models are loaded, the one used least recently
gives way: it gets no new requests (they wait in line, behind the load, and
load it again afterwards), and it is unloaded when its running requests have
ended. The wait of the new request is therefore bounded by the requests that
run when it arrives, so it waits, and no request is answered 503 because
another model is busy (open decision 16 of the feature checklist: Ares waits
here as it does for a full queue). `/v1/models` shows such a model as
`loading` with the reason `waiting for room: ...`. What this costs: with
requests for two models that cannot be loaded together arriving all the
time, the server changes between them at the pace of its loads.

*Idle unload.* A model whose preset has an `idle-unload` is unloaded when no
request has held it for that long; the time counts from the end of its last
request. The next request loads it again. The time is one setting of a
preset, with `[*]` as the default for all (open decision 19).

*What an unload releases.* The engine of the model is shut down (its
requests ended, its threads joined) and dropped, with the pages of sequence
state, the weights and the page-locked RAM it holds; a device that no other
model is on is closed with its driver context, so that a server with nothing
loaded holds no device memory. The unload compares what every device reports
as free before and after with what the model had taken at its load, logs
both, and warns when less came back than was taken.

Without `--cache` the prefix cache does not outlive its model. Its cells are
pages of K and V in VRAM and its checkpoints are page-locked RAM, both the
engine's, and a checkpoint without the pages it belongs to resumes nothing;
keeping either across an unload would keep the memory the unload is for. A model
that is loaded again starts with an empty cache, and the first request of a
conversation computes its prompt again. With `--cache` the unload writes what is
worth keeping of RAM and the devices to the store on the SSD before the memory
goes, and a load of the same model file with the same formats and split resumes
a conversation from there (see "On SSD" in [Prompt cache](prompt-cache.md)).

*Endpoints.*

| Endpoint | What it does |
| --- | --- |
| `GET /v1/models` | every preset in the order of the file: `id`, `aliases`, `status` (`{"value": "unloaded" \| "loading" \| "loaded" \| "unloading" \| "failed", "reason"}`), `defaults` (what a field means that a request to the model leaves out) and `max_tokens_limit` if it has one, and `created` and `context_length` once the model was loaded |
| `POST /models/load` `{"model": id}` | puts the model in the line of loads and answers `{"success": true}` at once |
| `POST /models/unload` `{"model": id}` | the model gets no new requests and is unloaded when those that run have ended; answers `{"success": true}` at once |
| `GET /models/sse` | Server-Sent Events: the state of every model as it is, then every change, as `{"model", "event": "model_status", "data": {"status", "reason"}}` |
| `GET /metrics?model=id` | the statistics of one loaded model; without `model` when the server has one |

The bodies and the answers of the two `POST` routes and the events are
llama-server's. Where they differ: a failure is a state of its own with its
reason (llama-server writes `unloaded` with `failed: true` and an exit code,
since its models are processes); a member of a body other than `model` is
refused; the event stream begins with the state of every model, so that a
client that connects late knows where it is; and there are no `args`,
`progress` or download events. A load asked through the endpoint is not
waited for by the answer, as in llama-server: whether it succeeded is in the
list and in the events. An unload does not end running requests; a way to
end them by force is not built.

*Failures and the end.* A load that fails leaves its model `failed` with the
reason, which every request that waited for it gets as a 503
`model_load_failed`; the other models are not touched, and the next request
for the model tries the load again. An engine whose device fails says so
(`Engine::failure`; the runtime's conductor sets it when a step fails on a
device): its model gets no new requests, is unloaded when the requests the
failure ended have let go, and is listed as `failed` with the engine's reason
until a request loads it again; models on other devices serve on. On the
first signal the server stops as before, and then unloads every model before
the process exits; a load that runs then is finished first.

*What is not there.* The plan of a load cannot be asked without the load
(the mixture-of-experts model's placement reads the devices itself), so the
router finds the models that have to go by trying: a model that fits
nowhere has unloaded the idle models, and waited for the busy ones, before
it fails. A model is planned against free memory only, so a preset without a
`context` that is loaded next to others gets less than alone. With a RAM
limit of the server, the limits of loaded models are parts of it that are
booked at load. What the models really hold they share through the one watch
over the machine's memory (see "Host memory"): the tiers of experts of all
loaded models get the same as far as each can use it, the checkpoints what is
left, and all give memory back when the machine needs it. So a model that is
loaded next to a tier that had taken what was free gets its share at the next
look, out of that tier; that is the only way RAM passes from one loaded model
to another, and nothing moves it by which model is in use. Without a limit of
the server no model is unloaded for another's RAM: what a second model needs
to run comes out of the elastic memory of the first. A model option
given on the command line needs text that is UTF-8, paths included.

The set of flags is the engine's own and starts from nothing. A flag exists
only for something that is worth setting: a real choice (KV formats, context
budget, the prompt share next to generation, expert pruning) or a fact about
the machine. There are no flags without effect and none that switch on what
is always on; an unknown flag or preset key is an error that names it.
