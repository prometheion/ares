# The APIs: request defaults, thinking, tools, Messages, Responses

**The defaults of a model's requests.** A model has its own value for every
sampling field a request can leave out, and its own limits for the length of
a reply (`crates/server/src/defaults.rs`). They are options of the model, on
the command line and as keys of a preset file, for all models in `[*]` or for
one, with the precedence of every option (the command line, the model's
section, `[*]`, the engine's own value).

An option that gives a field of a request its default has the name of the
field, with dashes, and takes what the field takes, in the range a request is
held to:

| Options | Field of a request |
| --- | --- |
| `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--typical-p`, `--top-n-sigma` | `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `top_n_sigma` |
| `--xtc-probability`, `--xtc-threshold`, `--dynatemp-range`, `--dynatemp-exponent` | the same with underscores |
| `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`, `--repeat-last-n` | the three penalties and their window (at most 1024) |
| `--dry-multiplier`, `--dry-base`, `--dry-allowed-length`, `--dry-penalty-last-n`, `--dry-sequence-breakers` | DRY; the breakers are a JSON array of strings, as in a request |
| `--mirostat`, `--mirostat-tau`, `--mirostat-eta` | mirostat 0, 1 or 2, its target and its rate |
| `--samplers` | the samplers that apply and their order, as names separated by commas; penalties and dry stand first |
| `--seed` | the seed of a request that names none |
| `--max-tokens` | the limit of tokens: `max_tokens`, `max_completion_tokens`, `max_output_tokens`, `n_predict` |

A request's own value always holds, also one that switches a sampler off
(`top_k: 0`). A field it leaves out or writes as `null` takes the model's
default, and without one the engine's own value (see [Sampling](sampling.md)).
The defaults are filled in where every generating route takes its model
(`generate::prepare`), so they are the same on `/v1/chat/completions`,
`/v1/completions`, `/v1/messages` and `/v1/responses`, also for a field the API
of a route has no member for: a Messages request has no `min_p`, and gets the
model's. A logit bias, `ignore_eos`, the probabilities to report, `n` and the
stop strings are what a request asks for and have no default.

`--seed` makes a request that names no seed draw the same for the same prompt,
which is the one way to a reproducible reply on the two routes whose API has
no seed. The `n` completions of one prompt draw from the seed and the ones
after it. A request asks for a draw of its own in llama.cpp's words, `seed:
-1`.

*One set.* Strata and TensorFold keep two sets of defaults for a model, one for
replies that think and one for those that do not, after the two sets Qwen's
model cards give. Ares keeps one. The production setup it replaces has one
(llama-server's preset has no second), and the one value Qwen's card gives
differently for the two, a presence penalty of 1.5 without thinking, was taken
back there on 2026-11-03: scripts without reasoning broke twice as often with
it. A second set would be more keys of the same kind, filled in at the same
place, where the server also reads what a request says about thinking;
nothing in the first set changes when it comes.

*The length of a reply.* Two options, because they answer two questions.
`--max-tokens` is the limit of a reply whose request names none (no
`max_tokens`, `max_completion_tokens`, `max_output_tokens` or `n_predict`, or
llama.cpp's `-1`): what llama-server's `--n-predict` is, which bounds no
request that names its own limit. `--max-tokens-limit` is the most tokens of
any reply, the kind of limit KoboldCpp's `--genlimit` is: a request that asks
for more gets this many, and a request that names no limit gets `--max-tokens`
or, without it, this. A default above the limit is refused at start.

A request above the limit is served with the limit and not refused. A limit
of tokens is an upper bound, not an order for that many, so the shorter reply
is one the request allows, as a reply that the end of the context cut is; and
a refusal would lock out every client that sends the same large number with
each request, which the Messages API makes all of its clients do. What
happened can be seen in the response: it has the header
`x-ares-max-tokens-limit` with the limit that took the place of the request's,
streamed or not; a reply the limit cut ends with `length` (`max_tokens` on
`/v1/messages`, `incomplete` with `max_output_tokens` on `/v1/responses`) and
its usage has the count; and the response object of `/v1/responses`, which
says back what was applied, has the limit in effect in `max_output_tokens`
and the model's defaults in `temperature` and `top_p` where the request left
them out.

*Where they show.* `GET /v1/models` has for every model `defaults`, an object
under the names of the request's fields with the values as a request writes
them (`{"temperature": 1.0, "top_p": 0.95, "max_tokens": 96000}`; `max_tokens`
is what a request without a limit gets, the default or else the limit), and
`max_tokens_limit` if the model has one. A field that is not listed is the
engine's own. The log of a start names them once for every model. The
start refuses, with the file and the line in a preset: a value outside the
range of its field, and a default length above the limit, wherever the two
stand (a limit on the command line against a default in a section is the
section's line).

llama-server's names for these that Ares writes differently are answered in a
preset with the name of Ares, like its other options: `temp`, `typical`,
`top-nsigma`, `dynatemp-exp`, `mirostat-lr`, `mirostat-ent`,
`dry-sequence-breaker`, `sampler-seq`, `s`, and `n-predict`, which is
`max-tokens`. Sampling settings in the metadata of a model file are not read.

**Thinking.** Qwen3.8 thinks before it answers: the template ends the prompt
with `<think>\n`, and the model writes its reasoning, `\n</think>\n\n` and the
answer. The server returns the reasoning in `reasoning_content` of the message
(and of the deltas of a stream, as their own chunks) and the answer in `content`;
a reply that `max_tokens` cuts inside the thinking has reasoning and an empty
content. An assistant message of a request may carry `reasoning_content` back,
and the chat template writes it into the prompt. What marks the reasoning is
read off the model's own template, not looked up by the name of a model: when a
model is set up, `ChatTemplate::thinking_format` (`ares-tokenizer`) renders small
conversations and compares the texts, as llama.cpp's autoparser does. An
assistant message with and without `reasoning_content` gives the pair of tags
and the white space the template writes around them; the end of the prompt with
`enable_thinking` on and off says whether the model continues inside the block
or the template closes an empty one; a few values of `reasoning_effort` say
whether the template reads it and which values it refuses. The tokenizer gives
the API's own types (`ares_api::ThinkingFormat`, and `ares_api::ToolFormat` for
the calls of tools, `ChatTemplate::tool_format`), and whether the model thinks
when a request does not say comes from the options the model is rendered with.
Whether the template takes a system message only first (Qwen's raise an
exception for a later one) is read the same way (`ChatTemplate::system_first`:
a system message rendered before a user's and after one). The result travels as `ares_api::ChatFormat` (`Model::chat_format`,
`Engine::chat_format`) and the server's `Splitter` sorts the text with it as it
streams: it holds back what may still be a tag and the white space the template
writes around the reasoning, and gives the same text however the pieces are cut.
A template with no pair of tags around the reasoning has no thinking, and its
replies stay whole. The newline the template writes before the closing tag is
not part of `reasoning_content` (llama-server leaves it in, which a template
that does not trim would write twice).

The controls are llama-server's, under its names and defaults:

| Control | Start | Request | Here |
| --- | --- | --- | --- |
| Thinking on, off or left to the template | `--reasoning on\|off\|auto` (`-rea`) | `chat_template_kwargs.enable_thinking`; `reasoning_effort: "none"` is off | the variable `enable_thinking`; `auto` sets nothing |
| Format of the reply | `--reasoning-format` (default `auto`) | `reasoning_format` | `none` leaves the text whole in `content`, the end of the prompt from the opening tag on included, as llama-server does; `auto` extracts; llama-server's `deepseek` and `deepseek-legacy`, which it parses as `auto`, are refused (no second name for one behaviour) |
| Effort | `--reasoning-effort` | `reasoning_effort` | the variable `reasoning_effort`; refused if the template does not read it; of the usual values (`minimal` to `xhigh`) the ones the template raises for are found when the model is set up (`refused_efforts`: Flash-Next `minimal` and `high`, the 27B `minimal`): `high` is then the template's default, as on `/v1/messages`, and another one is refused |
| Thinking of earlier turns stays in the prompt | `--reasoning-preserve` (default on), `--no-reasoning-preserve` | `chat_template_kwargs.preserve_reasoning` | the variable `preserve_reasoning` |
| Any other variable of the template | `--chat-var` | `chat_template_kwargs` | the variable; `messages`, `tools`, `add_generation_prompt`, `bos_token` and `eos_token` are the server's |
| A budget of thinking tokens and the message injected when it is spent | `--reasoning-budget-tokens` (llama-server's `--reasoning-budget`), `--reasoning-budget-message` | `reasoning_budget_tokens`, `thinking_budget_tokens`, vLLM's `thinking_token_budget` (also on `/v1/responses`), `reasoning_budget_message`; Anthropic's `thinking.budget_tokens` | the guard of the reply (below); `-1` is no budget of the request's own (the model's default holds, as in llama-server); `reasoning_control` (llama-server's arming of its route that ends the thinking while it runs) is accepted: `/v1/chat/completions/control` with `reasoning_end` ends any running reply's thinking (`Engine::end_thinking`, the guard forced as a spent budget forces it) |

*The budget and the guard against loops* (`crates/runtime/src/guard.rs`).
llama-server keeps the budget in its sampler, which forces the tokens of the
message and of the closing tag into the sequence once the budget is spent. The
engine does the same with a rule of the request (`TokenFilter`, around the
request's own grammar if it has one): every reply gets a `Guard`, which reads
the reply token by token as text, finds where its thinking opens and closes
(the tags the template declares; a call the model opens inside its thinking
ends it too) with the same tracker as the usage's count (`watch::Tracker`,
below), and counts the thinking's tokens. Once the thinking has taken
the budget's tokens, after the character a token cut in two is whole, the
next rows carry a mask of one token each: the tokens of the message and of the
end of the thinking as the template writes it (`\n</think>\n\n`), which
the sampler therefore picks whatever it draws; a draft that is not that token
is cut from the window, and the rows before are free, so the model's drafts
go on there. The closing tag's tokens are not counted, as in llama.cpp, and
the usage counts the forced ones with the thinking. A model whose template
has no thinking has nothing to bound: a request's own budget is refused there
(`reasoning_budget_tokens`), a model's default does nothing.

The same guard ends a loop, which nothing asks for: the last 512 tokens of
the thinking going round one cycle of at most 64 tokens end the thinking the
same way (the answer follows), and in the answer one token 256 times in a row
ends the reply as a limit would (`length`; Strata's guard against a model that
writes one token for ever, issue 606 there). A cycle of several tokens is not
cut in an answer: a matrix of zeros is one. The real-model test is
`real_messages`, `a_budget_ends_the_thinking_and_the_answer_follows` (the chat
and the Messages route, the budget and `display: "omitted"`): with the 0.8B on
the test machine (2026-11-22) a budget of 24 tokens gave 26 tokens of thinking
(the budget, a newline and the closing tag) and then the answer (17 primes
below 60, counted step by step), and the omitted thinking sent back counted the
same prompt as the thinking shown.

A request is refused, too, for an effort the template does not read, for
`reasoning_effort: "none"` when `enable_thinking` does not change the prompt, and
for `reasoning_format`, `reasoning_effort` and `chat_template_kwargs` on
`/v1/completions`, which has no template.

**A conversation that continues from the cache.** The prefix cache continues a
conversation at the end of the last reply only if the next prompt, rendered from
what the client sends back, begins with the prompt and the tokens the model
generated. That is why the reasoning has to come back: the template of
Qwen3.8 writes `<think>\n`, the reasoning, `\n</think>\n\n` and the content
into the prompt for an assistant message, and without `reasoning_content` it
writes an empty block where the model had begun to think, so that the new
prompt parts from the cached sequence at the last token of the old prompt
(the newline of `<think>\n` is one token with the next, `<think>\n\n`). The
server does not cut or reshape what it returns for this: the reasoning is the
text between the tags without the layout the template writes itself, and the
content is the rest, both exactly as generated, so that the template writes the
turn back as the model generated it. `reasoning/templates.rs` in `ares-server`
plays that through the real templates and vocabularies of Qwen3.8-27B and of
Qwen3.8-Flash-Next (their templates declare the same and give the same
numbers) and counts the tokens of the sequence that the next prompt does not
repeat:

- None, for a reply with thinking (German with umlauts and emoji, code blocks,
  blank lines, tags that are only text), for a reply without thinking (the
  template closes an empty block in the prompt), for a reply the limit of
  tokens cut inside the answer or inside the thinking (also right after a full
  stop), and over four turns in a row, one with thinking off.
- Where the template trims or lays out the text other than the model did:
  a model that thinks of nothing (`\n</think>` right after `<think>\n`) parts at
  the last token of the prompt and the reply is computed again (the two
  newlines are two tokens for the model and one for the tokenizer); a newline
  after the answer costs it and `<|im_end|>` (2 tokens), a newline before the
  reasoning the whole reply, an empty line at the end of the reasoning 6
  tokens, an extra empty line or a single newline after the closing tag 4, a
  closing tag without the newline before it 5 (the replies of the test, 66 to
  68 tokens in all).
- Tokens the model sampled that are not the tokenizer's tokens for their own
  text: a sampled reply of 1500 tokens in the test through the server wrote
  `...` and `"?` where the tokenizer makes `..."` and `?`, and the next prompt
  parted there (at token 1128 of 1867, so the engine cached the 1024 of the
  whole pages before it). Greedy replies did not do it in the tests; a
  test with the real vocabularies (`tokens_that_are_not_the_tokenizers_own...`)
  holds the arithmetic of such a parting.
- Qwen3.5-0.8B's own template is another one: it writes the thinking of a turn
  into the prompt only after the last user message, `preserve_reasoning` or
  not, so no reply is ever repeated: the new prompt parts 2 tokens before the
  end of the old one (4 with thinking off), whatever the client sends back.

Through the server, with the 0.8B on a GPU and served with the template of
Qwen3.8-27B (read from its file; `tests/real_reasoning_cache.rs`, asked greedily), a client that sends
`reasoning_content` back gets this from the engine's report of each request
(`Snapshot::requests`) and `/metrics`:

| Conversation | Turn | Prompt tokens | Cached | Computed | Replayed |
| --- | --- | --- | --- | --- | --- |
| Short questions, 314, 259 and 158 tokens generated | 1 | 66 | 0 | 66 | 0 |
| | 2 | 406 | 379 | 27 | 0 |
| | 3 | 691 | 664 | 27 | 0 |
| A system message first; the first two replies cut by the limit of 600 tokens | 1 | 405 | 0 | 405 | 0 |
| | 2 | 1035 | 1004 | 31 | 0 |
| | 3 | 1665 | 1634 | 31 | 0 |

The cached tokens of a turn are the prompt and the tokens generated in the turn
before but the last, which is generated and never computed (the end token, or
the token that the limit cut off); what is computed is that token, the line
break after the reply, the new message and the start of the next reply. The same
long conversation with a client that sends back only `content` (turn 2: 434
prompt tokens, 384 cached, 50 computed; turn 3: 463, 384, 79 computed; a
sampled run had 128 rows replayed in turn 3) reads the cached whole pages
before the last token of the old prompt and computes the rest again, up to a
page, with a replay of the rows between the last checkpoint and the cached
cells when the two do not meet. Measured
through the server before the reasoning came back, Qwen3.8-27B with a history
of 28 000 tokens computed 80 to 165 rows again in every turn, the first token
after 0.34 to 0.55 s where the continued sequence gives about 0.15 s. A resume
point at the end of the prompt, which would shrink that cost for a client that
drops the reasoning, is another piece of work and is not built here.

Measured 2026-11-11 with Qwen3.8-27B on both cards: 79.2, 75.3 and 78.8
tokens per second for German, English and code through HTTP, which is what
the same model gives without HTTP; the first token after 76 to 121 ms; two
requests at once at 63 tokens per second each.

**Tool calls.** A chat request can bring `tools`. The chat template writes
them into the prompt, and the model calls one by writing the call into its
reply in the form the template writes the calls of earlier turns. That form is
read off the template, not looked up by the name of a model: when a model is
set up, `ToolFormat::detect` (`ares-tokenizer`) renders an assistant message
that calls made-up tools with made-up arguments, takes the texts between the
names, the keys and the values as the pieces of the format, and accepts them
only if they write the rendered calls again exactly. One form is understood,
the one the templates of Qwen3.8-27B, Qwen3.8-Flash-Next and Qwen3.5-0.8B
write (llama.cpp knows it as Qwen3-Coder's):

```text
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

A value that is a string stands as it is, every other value is JSON. After the
text of a message the template writes an empty line before the first call, and
a line break between two calls. The format travels in `ares_api::ChatFormat`
next to the thinking. A template that writes calls in another form (one JSON
object between the tags, as Hermes' does) has no format here, and a request that
brings tools for it is refused with a 400 that says so; a format for such
templates is not built.

The reply is held to the calls by a grammar and not trusted to be one.
`ares_grammar::calls` writes it in the engine's Lark syntax, after the grammar
llama-server builds for this format (`common/parsers/qwen3-coder.cpp`): the name
is one of the tools of the request; the arguments are the properties of the
tool's schema, the required ones first, each once, in any order (in the order of
the schema when there are more than six, llama.cpp's limit), then the others in
any order; a value whose schema allows no string is JSON that the schema accepts
(through the engine's JSON compiler, with the definitions of the tool's schema
for its references); a value that can only be a string is text up to the closing
tag, or one of the strings of its `enum` or `const`; a value that may be either
is text up to the closing tag. A call may be followed by white space and, when
several calls are allowed, by another call; then the reply ends. That holds
whether a tool says `strict` or not. A string is written raw, so the
`pattern`, `minLength`, `maxLength` and `format` of a string are not enforced.

| `tool_choice` | The reply |
| --- | --- |
| `auto` (the default) | free until the model writes `<tool_call>` (the rule's trigger), in its answer or inside its thinking; from there calls, and nothing after them |
| `required` | calls from the first token of the answer, after the empty line the template writes there, or from `<tool_call>` inside the thinking: at least one |
| a named function | the same, and every call is of that function; the prompt still lists all tools |
| `none` | text: the tools are in the prompt, no rule binds the reply, and a call the model writes stays in `content`, as with llama-server |

`parallel_tool_calls: false` allows one call. The reasoning of a model that
thinks is free in every case, and the rule (or the wait for the trigger) starts
after its closing tag, or at a call the model opens before it.

*A call inside the thinking.* These models sometimes write `<tool_call>` while
they still think, without `</think>` before it. When the tools of a request may
be called (`auto`, `required`, a named function), the marker ends the reasoning
where it stands, as llama-server has it for these templates
(`thinking_end_tags` and the parser of `common/parsers/qwen3-coder.cpp`). The
marker is the trigger of the detected `ToolFormat` and the tags are those of
the detected thinking; nothing is known by the name of a model.

- The engine: the structure of the request names the marker as an interrupt
  (see [Sampling](sampling.md)), with the rule of a call whose marker is written.
  From the token after the marker the reply is held as a call after `</think>`
  is: a name among the tools, arguments by the schema, the closing tags, then
  white space, another call if several are allowed, or the end. With `required`
  or a named function the end tokens are not allowed while the reply thinks, so
  it cannot end as reasoning alone: it ends with a call or at the limit of
  tokens. llama-server reaches the same by a grammar over the whole reply.
- The reply: the reasoning's splitter ends the reasoning at the marker
  (`Splitter::ends_at`, set by `Sorter::new` when the request has a splitter of
  calls) and passes the marker on as the first text of the content, where
  `CallSplitter` reads the call. `reasoning_content` is the text before the
  marker without the white space between the two (llama-server leaves it in),
  `content` is empty, and no part of the marker reaches either, however a
  stream cuts the text. White space at the end of the reasoning is held back
  for one piece of a stream, until the next shows whether the marker follows. A
  raw reply (`reasoning_format: none`) keeps its thinking in `content` and has
  the call too: `CallSplitter` reads the first marker of a reply wherever it
  stands.
- `</think>` after such a call cannot be written: after a call the rule allows
  white space, another call and the end, and no text. A reply that never writes
  it lacks nothing. When the turn comes back, the template writes `<think>`, the
  reasoning, `</think>`, an empty line and the call, the same prompt as for a
  reply that closed its thinking; checked with the templates and vocabularies of
  Qwen3.5-0.8B, Qwen3.8-27B and Qwen3.8-Flash-Next (`tools/templates.rs` in
  `ares-server`). The cached sequence is repeated up to the end of the
  reasoning and the call is computed again (41 of the 56 generated tokens in
  the test), where a reply that closed its thinking before the call is repeated
  whole.
- With `tool_choice: "none"`, or without tools, nothing binds the reply and
  nothing reads calls: the marker inside the thinking is reasoning like any
  other text. llama-server ends the reasoning at the marker there too and
  returns the rest of the thinking as `content`.

The server reads the calls out of the answer as it comes (`CallSplitter`, on
what the reasoning's splitter calls content): it holds back what may still be
markup, so that the pieces of a stream are the same however the tokens cut the
text and no markup reaches `content`; the white space between the content and a
call, and after the last call, is the template's layout and belongs to neither.
A call comes back in `tool_calls` with an id (`call_`, the request's id and the
place of the call), the name and the arguments as the text of one JSON object,
written as llama-server writes it (`{"city":"Köln","days":3}`): a string of the
format as a JSON string, any other value as the JSON the model wrote, and for an
argument that may be either, JSON if it is JSON of a kind the schema allows and
a string otherwise. A stream sends OpenAI's deltas: one with the place, the id,
the type and the name of a call, then fragments of its arguments. `content` of a
reply that is only calls is the empty text, as llama-server writes it.
`finish_reason` is `tool_calls` when the reply ended by itself (or at a stop
string) and has a call. Assistant messages with `tool_calls` and messages of
the role `tool` are checked (a name, arguments that are a JSON object) and
handed to the template, which writes them into the prompt.

A reply that the limit of tokens cuts inside a call ends with `length` and
returns the call as far as it came: the name if it was whole, and arguments that
are not whole JSON (`{"city":"Köl`); a call whose name was cut is not returned.
llama-server and OpenAI do the same. A stop string that cuts a call gives the
same call with `tool_calls` as the reason, as llama-server does.

The text with drafts is the text without, as for every rule (host tests on the
synthetic model: `crates/runtime/tests/tools.rs` under the rule, and
`crates/server/tests/tools_engine.rs` from the request to the answer, where the
reply is also the same whole and streamed; both with a call after the thinking
and with one inside it). A request with tools that may be
called has a rule. With `auto` the rule is free for the thinking and for the
text before a call: there the request takes the model's drafts like a
request without tools, and the conductor stops taking what a window gives at
the token that makes the rule bind (the trigger, also when it stands inside the
thinking; see "Drafts" under [Sampling](sampling.md)). Inside a call it takes the
n-gram drafter's drafts alone. A request that has to call (`required`, a named
function) may not end while it thinks, so its thinking rows carry a mask and
are not free: it takes the n-gram drafter's drafts alone for its whole reply.

On the vocabulary of the 0.8B (248,320 tokens) the grammar of a tool compiles in
about a millisecond, and the mask of a step along random walks through it takes a
median of 2 microseconds inside the text of a string argument (3 at the 90th
percentile, 1.8 milliseconds at the longest) and 3 inside JSON arguments (5, and
53 microseconds at the longest); host time on one core, with the machine under
load (`mask_cost` of `ares-grammar`, which has the two grammars).

Where this is not llama-server: `required` there lets text come before the first
call, here the answer is a call from its first token; a named function is read
as `auto` there (it reads `tool_choice` as a string), here it is enforced; a
`response_format` or `json_schema` next to tools wins over them without a word
there and is refused here, as `grammar` is by both (with `tool_choice: none`
the other rule applies and the tools are only in the prompt); llama-server takes
`<tool_call>` inside the thinking as the end of the thinking with and without
tools, here only when the tools may be called; the strings of an `enum` are
enforced here and any text is accepted there; the white space between the
content or the reasoning and a call is part of neither here. The legacy
`functions` and `function_call` are refused.
The tests with real models (`crates/server/tests/real_tools.rs`: the 0.8B on
one card, the 27B on two) pass. Whether a model opens a call inside its
thinking is the model's business, so the test of that does not wait for it: it
bans the token of `</think>` with a `logit_bias` and puts a bias on the token
of the marker, the least of a ladder with which the reply has a call, and
offers tools without an argument of free text (a string may hold the marker,
and the bias would write it there). Under the ban alone the 0.8B thinks on to
the limit and never writes the marker. With a bias of 6 on the marker (8 for a
sampled reply) it thinks, opens the call inside its thinking, and the call
comes back with the reasoning before it, the same raw, streamed and without
drafts. The last step of the ladder is more than two logits are apart, so a
model that follows no smaller one has the marker as its first token.

**The Messages API.** `POST /v1/messages` is Anthropic's Messages API. A
request is served as the chat request that asks the same: the `anthropic`
module of `ares-server` writes its body over into a chat body, and from there
it is the chat route's path (the same reader, the same engine request, the same
prefix cache, the same rule for calls, the same end when the client leaves).
The reply comes back as a message with content blocks, or as Anthropic's
events.

| Field | Here |
| --- | --- |
| `model`, `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stream` | as on the chat route |
| `system` | a string or text blocks: one system message |
| `messages` | turns of `user`, `assistant` and `system`, a string or blocks. User: `text`, `image` (a `base64` source with an `image/` media type, an `image_url` part of the chat body; a model loaded with `--mmproj` reads it), and `tool_result` (a string, or text and image blocks), which becomes a message of the role `tool` where it stands. Assistant: `text`, `thinking`, `redacted_thinking`, `tool_use`. System: text; before every other turn it joins the system prompt, after one it is a user turn with its text (below) |
| `tools` | `name`, `description`, `input_schema` (and `type: custom`, `strict`, `eager_input_streaming`, which ask for what is always done) |
| `tool_choice` | `auto`, `any` (the chat route's `required`), `tool` with a name, `none`; `disable_parallel_tool_use` allows one call |
| `thinking` | `enabled` and `adaptive` switch thinking on, `disabled` off; a request without it leaves it to `--reasoning`; `display: "omitted"` gives thinking blocks an empty text and a signature that carries the thinking (below), `summarized` and `updates` the thinking whole |
| `stop_sequences` | the stop strings |
| `output_config` | `format` with a JSON schema is the chat route's `response_format`; `effort` is its `reasoning_effort`, except `high`, Anthropic's default, which is the template's default |
| `context_management` | the edits `clear_thinking_20251015` and `clear_tool_uses_20250919`, as Anthropic's service makes them, and `applied_edits` in the response (below) |
| `metadata`, `service_tier` | bookkeeping: accepted, without effect |
| `cache_control` | accepted on the body, on blocks and on tools; without effect (below) |

The text blocks of one message are joined with a line break. A result that says
`is_error` has `Error: ` before its text, because the chat template has no
other place for it. Everything else is refused with a 400 that names it, in
Anthropic's shape of an error (`{"type": "error", "error": {"type",
"message"}}`, with the status the chat route gives): `document` blocks (input
other than text and images is not built), an image anywhere but in a turn of
the user or the result of a tool, an image at a URL (images are not fetched),
tools of a `type` that Anthropic's
service defines or runs (`web_search_...`, `bash_...`) and the blocks they
leave in a history, `defer_loading`, `mcp_servers`, `container`, other edits of
`context_management` (compaction), citations, and any member the module does not know. A
last message of the assistant is a prefill, which the reply continues (see
"Continuing a message" below). `x-api-key` is one of the two headers a key
may come in (see "Who may use the server"); `anthropic-version` and
`anthropic-beta` are not looked at: the server has one version of the API, and
a field of a beta is served or refused by its name in the body.

- **A budget for thinking** (`thinking.budget_tokens`) is the chat route's
  `reasoning_budget_tokens`, as vLLM maps it: the thinking ends there and the
  answer follows (see the budget under "Thinking"). At Anthropic the budget is
  a target the model may overshoot; here it is a bound. `max_tokens` bounds
  thinking and answer together, as at Anthropic. Claude Code 2.1 sends
  `adaptive` thinking, which has no budget.
- **Signatures.** A thinking block of a reply carries the fixed signature
  `unsigned`: the server has no secret to sign with, and an empty signature
  may make a client drop the block. A thinking block that comes back is the
  reasoning of its message whatever its signature says. A reply's blocks sent
  back unchanged are the turn the chat route gets back with
  `reasoning_content`, so the template writes what the model generated and the
  prefix cache continues (see "A conversation that continues from the cache").
  `redacted_thinking` is accepted and nothing of it enters the prompt: only
  Anthropic's service can read it.
- **Cache markers.** The server caches every prefix by itself, so a
  `cache_control` marker changes nothing and its `ttl` is no promise. The
  usage has `cache_read_input_tokens` (the part of the prompt the cache gave),
  `input_tokens` (the rest: the two add up to the prompt, as at Anthropic) and
  `cache_creation_input_tokens`, which is 0.
- **Why a reply ended.** `end_turn`, `max_tokens`, `tool_use`, or
  `stop_sequence` with the string in `stop_sequence` (in the message and in the
  `message_delta` of a stream). The engine names the stop string that matched
  (`FinishReason::StopString` with its place in the request's list; see "The
  stop string and the thinking of a reply" below). A reply with calls ended to
  call them, also when a stop string cut it.
- **What Claude Code 2.1 sends.** Every request of Claude Code 2.1.301 has
  `context_management` with `clear_thinking_20251015` and `keep: "all"`,
  `thinking: {"type": "adaptive", "display": "updates"}` and `output_config:
  {"effort": "high"}`, and its messages have turns of the role `system`: its
  environment after the first question, and `<total_tokens>N tokens
  left</total_tokens>` after the results of tools (these stay in the history of
  later requests). Each of them was refused until 2026-11-22. The templates of
  Qwen3.8 refuse a system message anywhere but first, so a system turn after
  another turn is written as a turn of the user with the same text, which is
  where a model reads what came in between; Flash-Next's template knows the
  efforts `xhigh` (its default), `medium` and `low`, so `high`, which is
  Anthropic's default, is the template's default. `display: "updates"` asks
  for notes of progress between calls, which the server does not have; the
  thinking comes back whole, which is also what writes it into the next prompt
  as the model generated it. A system turn after another turn is a turn of the
  user for a template that takes a system message only first, by the rule of
  every route (`generate::place_system_messages`, see "The Responses API").
- **`thinking.display: "omitted"`.** Anthropic's API answers it with thinking
  blocks whose `thinking` is the empty text and whose signature carries the
  thinking it does not show, so that a client that sends the block back
  continues the conversation; the server does the same: the signature is
  `ares-reasoning:` and the reasoning in base64, and a block that comes back
  with an empty text and such a signature is the reasoning of its turn (a
  text that is there wins; a broken carried signature is refused). The next
  prompt is then the one a client that shows the thinking sends, and the
  prefix cache continues it. A stream has no `thinking_delta`: the block opens
  empty and its `signature_delta` carries the whole of it before it closes.
  `summarized` gets the thinking whole: there is no summary, and the thinking
  shows more than one, never less.
- **Context edits.** `context_management.edits` are read with the body and
  carried out on the chat request once its model is known (`anthropic::edits`),
  because a trigger counts the tokens of the prompt, which only the engine
  does. `clear_thinking_20251015` takes the thinking out of the turns of the
  assistant before the last `keep.value` turns; a turn begins with a message
  of the user that is not only results of tools, the turn in progress and
  turns without thinking count, a request without `keep` keeps one, `"all"`
  keeps all, and the edit stands first among several, as at Anthropic.
  `clear_tool_uses_20250919` replaces the results of the oldest calls with
  `[tool result cleared to save context]` once the prompt is past `trigger`
  (`input_tokens`, 100 000 by default, or `tool_uses`), all but the last
  `keep.value` calls (3) and the calls of the tools in `exclude_tools`; with
  `clear_tool_inputs` their arguments become `{}`; it changes nothing when it
  would take fewer than `clear_at_least` tokens away. The response lists what
  changed the prompt in `context_management.applied_edits`
  (`cleared_thinking_turns` or `cleared_tool_uses`, and `cleared_input_tokens`),
  in the message and in the last `message_delta` of a stream, an empty list
  when nothing did, and no member for a request without edits;
  `/v1/messages/count_tokens` adds `context_management.original_input_tokens`.
  The tokens come from counting the prompt before and after an edit that
  changes it (a rendering and a tokenization each, no device); an edit that
  changes nothing costs at most the count its trigger needs. A cleared result
  in the middle of a conversation parts the prompt from the cached one there,
  as at Anthropic: the checkpoints at message boundaries keep that to the rows
  after it. Anthropic does not publish its placeholder; this one is Ares' own.
- **The billing line of Claude Code.** Claude Code begins its system prompt
  with `x-anthropic-billing-header: ...; cch=a5145;`, whose `cch` is another
  one in every request. Left alone it would part every prompt from the cached
  one at its first line, so the five bytes after `cch=` become `fffff` when a
  `;` follows them, in a system prompt that is a string and in every text
  block of one, exactly as upstream llama-server does it (PR 21793). Nothing
  else of a prompt is changed.

A stream is `message_start`, then for every block `content_block_start`, its
deltas (`thinking_delta`, `text_delta`, `input_json_delta`; a thinking block
gets a `signature_delta` before it closes) and `content_block_stop`, then
`message_delta` with the reason and the usage, and `message_stop`. The blocks
are in the order the model wrote them and numbered from 0; a whole message is
the same events put together, so the two cannot differ. The engine counts the
prompt when the reply ends, so the usage of `message_start` is zeros and
`message_delta` has all of it. While the engine has nothing to say the stream
sends `ping`. The response starts when the engine has produced its first step,
so that a request the engine refuses gets an HTTP status, or after
`Config::stream_wait` (10 s) without it (`Generation::first`, on all three routes
that stream): Flash-Next computed the 16 thousand tokens of Claude Code's first
prompt for three and a half minutes, and Claude Code gave up on a response that
had not started after six and sent the request again. A failure after the
stream began is an `error` event, and the stream ends without `message_stop`. The input of a `tool_use` block is the
JSON text the model wrote, byte for byte; a call that `max_tokens` cut has the
fragments in a stream and `{}` in a whole message. Call ids are `toolu_`, the
request's id and the place of the call.

`POST /v1/messages/count_tokens` reads a body as `/v1/messages` does (without
`max_tokens` if the client leaves it out) and returns `{"input_tokens": N}`,
the tokens of the prompt the engine would have been given; nothing is
generated. The count is `Engine::count_tokens` of `ares-api`, which renders and
tokenizes a prompt; the runtime answers it with the model's own encoding of the
prompt, so the count is the prompt a run of the same request reports (computed
plus cached), which a host test checks through the real runtime. An engine that
cannot count answers 501.

Both routes run on the core of the chat route in `generate.rs`: `prepare`
applies what a parsed chat request says about thinking and tools and returns
the sorter of its reply, `Generation::start` hands the request to the engine, and
`Generation::next` gives the reply as pieces (reasoning, content, the start of a
call, fragments of its arguments) and its end (reason, usage, timings). A
route of another API writes its body over into the chat body, calls these, and
writes the pieces in its own format.

Where this is not llama-server: it concatenates system blocks with nothing
between them, moves the text of a user turn before its tool results, reads
`tool_choice` `tool` as `any` and does not read `none`, drops
`thinking: disabled`, `is_error`, `disable_parallel_tool_use` and every member
it does not know, passes images on, and enforces `budget_tokens` in its
sampler. Its `stop_reason` is `end_turn` for a stop string as well (it names the
string in `stop_sequence`); here it is `stop_sequence`, as at Anthropic. The tests with a real model
(`crates/server/tests/real_messages.rs`, the 0.8B on one card) pass.

**The Responses API.** `POST /v1/responses` is OpenAI's Responses API,
without state. A request is served as the chat request that asks the same, as
a Messages request is: the `responses` module of `ares-server` writes its body
over into a chat body, and from there it is the chat route's path (the same
reader, the same engine request, the same prefix cache, the same rule for
calls, the same end when the client leaves). The reply comes back as a
response object with its output items, or as the typed events of the API.

| Field | Here |
| --- | --- |
| `model`, `temperature`, `top_p`, `stream` | as on the chat route |
| `instructions` | a system message |
| `input` | a string (one message of the user), or items: messages (`user`, `assistant`, `system`, `developer`; a string or `input_text` and `output_text` parts, and in a message of the user `input_image` parts with a data URL in `image_url`, `detail` accepted), `function_call`, `function_call_output` (a string or `input_text` parts), `reasoning` |
| `tools` | functions in the shape of this API: `type: "function"`, `name`, `description`, `parameters`, `strict` (the arguments are held to the schema either way) |
| `tool_choice` | `auto`, `none`, `required`, `{"type": "function", "name"}` |
| `parallel_tool_calls` | as on the chat route |
| `max_output_tokens` | the limit of tokens; a reply it cuts has the status `incomplete` and `incomplete_details.reason: "max_output_tokens"` |
| `reasoning` | `effort` is the chat route's `reasoning_effort` (`none` switches thinking off, any other value is the template's variable); `summary: "auto"` is accepted and no summary is written |
| `text.format` | `text`, `json_object`, `json_schema`: the chat route's `response_format` |
| `include` | `reasoning.encrypted_content` is accepted and adds nothing (below) |
| `truncation` | `disabled` (the default): an input that does not fit is an error; `auto`: the oldest turns leave the conversation until it fits (below) |
| `store`, `background` | `false` (below) |
| `metadata`, `user`, `safety_identifier`, `service_tier`, `client_metadata` | bookkeeping: accepted, without effect; `metadata` and `user` are said back |
| `prompt_cache_key`, `prompt_cache_retention`, `prompt_cache_options` | accepted, without effect, as `cache_control` is on `/v1/messages`: the server caches every prefix by itself and drops what it kept by its own budget, so a retention is no promise |

The parts of one message are concatenated, as the chat route does it. The
items one reply returned for a turn (a reasoning item, a message, calls) are
one message of the assistant again; an item that does not continue a turn in
that order begins another message. `instructions` and the messages of the
roles `system` and `developer` that stand before the conversation are one
system message, joined by an empty line: Codex puts its instructions at the
head of `input` and sends no `instructions` (read in its source, not tried),
and the chat template of Qwen3.5-0.8B refuses a system message that is not
the first one (`System message must be at the beginning.`, read in the model
file). Such a message inside a conversation is a message of the user with its
text, by the rule of every route for a template that declares it
(`ChatFormat::system_first`, read off the template, which the three models'
templates do): where a request takes its model (`generate::prepare`,
`place_system_messages`), the system messages before every other message
become one system message (joined by an empty line) and a later one a turn of
the user, which is where the Messages route always put a system turn between
the others. A template that takes a system message anywhere gets it where the
client put it.

Everything else is refused with a 400 that names it, in OpenAI's shape of an
error, as on the chat route: `input_file` and `input_audio` parts (input
other than text and images is not built), an `input_image` outside a message of
the user or one given by `file_id` (the server keeps no files), tools that are not functions
(`web_search`, `file_search`, `code_interpreter`, `computer_use`, `mcp`,
`local_shell`, `custom` and the like), the items such tools leave in a
history and a `tool_choice` that names one or a subset (`allowed_tools`),
`prompt` templates, `top_logprobs` and `include:
["message.output_text.logprobs"]` (the pieces of a reply are not told apart
into the tokens of the answer, of the reasoning and of calls),
`text.verbosity`, `reasoning.summary` `concise` and `detailed`,
`max_tool_calls`, `stream_options.include_obfuscation:
true`, and any member the module does not know. Only members of this API are
read: the sampler's fields of the chat route (`top_k`, `min_p`, `seed`, `stop`
and the others) are refused here, as on `/v1/messages`.

- **No state** (open decision 20 of the feature checklist). The route keeps
  no responses, as llama-server, ollama and Strata. A response says
  `"store": false`, also when the request left `store` out, where OpenAI's
  default is `true`. A request that sets `store: true` is refused: it asks
  for something the server does not do, and the clients that work without
  stored responses say so themselves. Codex sends `store: false` with every
  request (read in its source, `codex-rs/core/src/client.rs`, on 2026-11-12);
  the OpenAI SDKs send the member only when their caller sets it.
  `previous_response_id`, `conversation`, `background: true`, an
  `item_reference` in `input` and a `prompt` template are refused by name,
  and `GET` and `DELETE /v1/responses/{id}`, `cancel`, `input_items` and
  `compact` answer 404 and say that no response is kept. Stored responses are
  row A05 of the checklist and are not built.
- **Reasoning items.** The reasoning of a reply is one item of the type
  `reasoning`, before the message. Its `content` is the reasoning as the
  model wrote it, in a part of the type `reasoning_text`, which is what the
  API has for models whose reasoning is not hidden; its `summary` is empty,
  because the server writes none; and it has no `encrypted_content`, because
  the server has nothing to hide and no secret to encrypt with. A reasoning
  item that comes back is the reasoning of its turn: the text of its
  `content`, whatever its id says. The item of a reply sent back unchanged is
  therefore the turn the chat route gets back with `reasoning_content`, so the
  template writes what the model generated, on every turn the same, and the
  prefix cache continues (see "A conversation that continues from the
  cache"). A `summary` and an `encrypted_content` of an item are accepted and
  nothing of them enters the prompt: a summary is not the reasoning, and
  encrypted content is text only the service that wrote it can read. A client
  that sends back only those two (and not `content`) gets the same answers and
  pays what a chat client pays that drops `reasoning_content`: the last turn
  is computed again.
- **`include`.** `reasoning.encrypted_content` asks for reasoning items that
  a client without stored responses can send back, and Codex asks for it with
  every request. Every reasoning item of this server is such an item, so the
  value is accepted and adds nothing. The other values are refused by name.
- **`truncation`.** With `disabled` an input that does not fit the context
  is the error `context_length_exceeded`. With `auto` (OpenAI: "drop input
  items in the middle of the conversation") the oldest turns leave it, whole,
  until the prompt and a token of the reply fit the model's context: a turn is
  a message of the user and what follows it up to the next (the answer, its
  calls and their results, so no result is left without its call); the system
  messages before the conversation and the last turn stay, and a conversation
  that does not fit with its last turn alone gets the error. The number of
  turns is found by halving, with a count of the prompt for each try
  (`responses::truncate`). The cut parts the prompt from a cached one at its
  first turn.

  *Truncation and context shift elsewhere* (F18 of the feature checklist).
  llama.cpp's server no longer cuts a prompt that is too long (it answers
  "exceeds the available context size"), and its context shift
  (`--context-shift`, off by default, with `n_keep` and `n_discard`) drops
  tokens after the first `n_keep` from the K/V of a running sequence and moves
  the rest down. It switches that off for a model whose memory cannot shift
  (`llama_memory_can_shift`), which is every model with recurrent state: the
  state of a delta-rule layer holds every token it has read, and nothing can
  take a token out of it again. Both models here are such models, so context
  shift is not possible for them without reading the kept tokens again from
  the start, a prompt as long as the context; a generation that reaches the
  end of the context ends with `length`, as in llama.cpp. vLLM's
  `truncate_prompt_tokens` and TGI's `truncate` cut a prompt to its last N
  tokens, SGLang's `--allow-auto-truncate` cuts one that does not fit; such a
  cut works on any model, but in a rendered conversation it lands anywhere,
  the system prompt included, and is not built: a conversation is cut by
  whole turns (`truncation: "auto"` here, and a client's own choice
  elsewhere).
- **The usage.** `input_tokens` is the whole prompt,
  `input_tokens_details.cached_tokens` the part of it the prefix cache gave,
  `output_tokens` what was generated, and for a model that thinks
  `output_tokens_details.reasoning_tokens` the tokens of its thinking, which
  the engine counts. The response has the `timings` of the chat route next to
  the usage.
- **`output_text`.** The response has no member of that name. The official
  SDKs put it together from the items; llama-server does not send it either,
  and its own test reads it from the Python SDK.
- **What a response says back.** `instructions`, `max_output_tokens`,
  `temperature`, `top_p`, `tools`, `tool_choice`, `parallel_tool_calls`,
  `text`, `reasoning`, `metadata` and `user` as the request gave them (`null`
  where it gave nothing: the engine's own defaults are not known to the
  route), the `truncation` asked for (`disabled` without one), and
  `store: false`, `background: false`,
  `previous_response_id: null`.

The output is in the order the model wrote it: the reasoning item, the
message (one `output_text` part), then one `function_call` item for every
call, with `call_id` (`call_`, the request's id and the place of the call),
`name` and `arguments` as the JSON text the model wrote. The ids of items are
`rs_`, `msg_` or `fc_`, the request's id and the place in the output. An item
the limit of tokens cut has the status `incomplete`, and a call it cut has its
arguments as far as they came.

A stream is `response.created` and `response.in_progress`, then for every
item `response.output_item.added`, its events and `response.output_item.done`,
then `response.completed` or `response.incomplete` with the response as it is
returned whole; every event has its `sequence_number`, counted from 0. A
reasoning item and a message have `response.content_part.added`, their deltas
(`response.reasoning_text.delta`, `response.output_text.delta`), the text once
more (`response.reasoning_text.done`, `response.output_text.done`) and
`response.content_part.done`; a call has
`response.function_call_arguments.delta` and
`response.function_call_arguments.done`. The stream and the whole response are
made from the same thing (`Output` in `responses/reply.rs`, which takes the
pieces of a reply and keeps the items), so the two cannot differ. The usage is
in the last event only, since the engine counts the prompt when the reply
ends. A failure after the stream began is an `error` event (`code`, `message`,
`param`), and the stream ends with it. There is no `[DONE]`.

`POST /v1/responses/input_tokens` reads a body as `/v1/responses` does, and
`POST /v1/chat/completions/input_tokens` one as `/v1/chat/completions` does;
both return `{"object": "response.input_tokens", "input_tokens": N}`, the
shape OpenAI and llama-server have, and generate nothing. The count is
`Engine::count_tokens`, as for `/v1/messages/count_tokens`, so it is the
prompt a run of the same request reports (host tests on the scripted engine
and through the real runtime).

Where this is not llama-server: it drops tools that are not functions,
`store`, `background`, `truncation`, `include`, `text` and every member it
does not know without a word, passes the chat route's fields through, and
reads a `tool_choice` that names a function as `auto`; here each is served or
refused. It requires a reasoning item to have a `summary` and a `content`
that is not empty, and writes `encrypted_content: ""`. Its stream has no
`sequence_number`, no `response.reasoning_text.done`, no
`response.function_call_arguments.done` and no events for the part of a
reasoning item, it numbers the items by what kinds it has seen, and it calls
every response `completed`, also one the limit of tokens cut.

What is not there: a
switch that turns thinking on for a template whose default is off and that
does not read `reasoning_effort` (the route has `effort` alone, and
`--reasoning` decides for the server), the probabilities of tokens, custom
tools (a tool that takes free text), and a `text.format` next to tools that
may be called (the reply follows one rule, as on the chat route). What Codex
sends was read in its source and not tried against a running Codex. The
tests with a real model (`crates/server/tests/real_responses.rs`, the 0.8B on
one card) pass.

For a mixture of experts (`qwen4exp`) the plan is another one
(`crates/server/src/load/qwen4exp.rs`): where the layers go (`--split tensor`:
both devices on every layer, `layer`: a range each, by default the first when
there are two devices and each can write the other's memory), the context
(without `--context`, the model's own: the indexer's keys of every page in VRAM,
the K and V of the first pages up to a share of the room for expert slots, those
of the pages beyond in page-locked RAM as the sequences grow into them; a
context named with `--context`, for all sequences together, is placed the same
way), and the RAM (`--ram`, whose limit less
what the engine needs is the tier of experts, and what the tier leaves the
prompt cache's). Before it listens the server fills the tiers of experts from
the usage profile kept for the model file (in `--cache`, else the user's cache
directory), or without one reads a short text of German, English and code, so
that the first request does not find every slot empty. It has the prefix cache
and the prompt cache on SSD as the 27B has them; it speculates only with a model
drafter that `--drafters` names (its MTP block, from the model file or a file of
its own, and a DFlash drafter), whose window is 4 rows by default and at most 8,
and takes `--draft-vocab` with one. It has no `--shares` and no `--precision
exact`; asking for either is an error that names it. Thinking is handled as for the
27B: the model declares what its chat template says (`Model::chat_format`), so
the reasoning of a reply comes apart from the content, in streams too, and
`--reasoning` and the controls of a request work the same way; a reply that
`max_tokens` cuts inside the thinking has reasoning and no content.

Images are read by both models through a projector (`--mmproj`, see
[Images](images.md)): an `image_url` part with a data URL on the chat route, an
`image` block on Messages and an `input_image` part on Responses.
