# Structured output

**The hook.** A request can carry a `TokenFilter`
(`Runtime::submit_filtered`): before each step the conductor asks it, for every
wanted row, which tokens may follow the reply and the drafts before the row, and
writes the answer as a mask into the record; it may also say how many of the
drafts it allows, and that it has failed. Grammars are one such filter.

**Structured output.** A request can carry a `Structure` (`ares-api`): a rule
for the text of the reply, a grammar in GBNF or a JSON schema, and where in the
reply the rule starts to bind. The engine turns it into the set of tokens that
may come next at every row, writes that set as a mask into the step record, and
the sampler applies it before the draw like any other mask (above), so a reply
follows the rule from the first token to the last or the request ends in an
error. `Runtime::submit_filtered` still takes any `TokenFilter`; a request with
a `Structure` gets one made by the runtime.

*The engine.* `ares-grammar` wraps llguidance (MIT, crates.io, version 1.9;
Cargo.toml names it with the thread pool of its C interface left out; it builds
offline once fetched): an Earley parser over lexemes that a lexer built on
derivatives of regular expressions finds, with the token mask computed by
walking the vocabulary's trie. It is suitable because it takes the vocabulary
as bytes per token, which is what `Model::piece` gives, and needs nothing but
that: the tokens a control token stands for are empty and no grammar allows
them, and the end tokens are allowed exactly where the grammar can end. Reading
the vocabulary of 248,320 tokens takes about a second; the runtime does it on a
thread of its own when it starts, and a request with a structure waits for it
if it comes sooner. Compiling a grammar takes one to two milliseconds.

*Kinds.* `grammar` is GBNF as llama.cpp reads it (`root`, literals and
classes, groups, `*`, `+`, `?`, `{m,n}`, `.`, `<[id]>` and `!<[id]>` tokens,
comments, the same rules for where a line ends a rule); it is translated to
the engine's Lark syntax, with the position of a syntax error (line and column)
in the message. Left recursion, which llama.cpp refuses, works. `json_schema`
and the schemas of `response_format` go to the engine's JSON compiler, in
llama.cpp's way: keywords it does not know are left out (the engine's default
refuses them), white space between the parts is llama.cpp's own (one space, or
line breaks and an indent of up to 20, so a model cannot fill its reply with
blanks), `$ref` is followed inside the schema and a remote one is refused (the
server fetches nothing). An empty schema is any object, as llama-server reads
it.

*Where the rule starts.* A model that thinks is free while it thinks. The
server knows the format from the chat template (`ThinkingFormat`) and sets
`Start` for the request: from the first token when the template has no
thinking or the prompt has closed the block (thinking off); after the closing
tag when the prompt has opened the block (`<think>\n`); and for a template that
writes nothing into the prompt, either the opening tag followed by reasoning
or the rule from the first token. llama-server does this for `response_format`
and `json_schema` (it builds the grammar around the reasoning parser) and binds
a plain `grammar` to the first token, which keeps a thinking model from
thinking; this server binds all three after the reasoning, and
`enable_thinking: false` gives the other behaviour. The answer starts at the
token after the closing tag: the blank line the template writes there is not
allowed to the model, and a closing tag that shares a token with the start of
the answer is not checked against the rule for the rest of that token. With the
0.8B (`crates/server/tests/real_structured.rs`) the thinking under a
`response_format` is the thinking of the same request without one, character
for character, and the answer after it obeys the schema. When the thinking
ends is the model's business: taking the most probable token, the 0.8B thinks
about the question of that test in a loop to any limit, with the rule and
without, so the test puts a bias on the token of the closing tag, the least
of a ladder with which the reply ends by itself (1 for the 0.8B).

*A rule on a trigger.* A `Structure` may name a trigger: the answer is then
free from where the rule would start until its text contains the trigger, and
the rule binds what follows it; a reply that never writes the trigger stays
free to its end, and the rows before the trigger carry no mask. The rule is
checked from the token after the one that completes the trigger, as it is after
the closing tag of the reasoning: what the same token holds after the trigger is
not checked, and the tags in question are tokens of their own in the
vocabularies of the supported models. This is what llama.cpp calls a lazy
grammar and what the calls of tools use; the third kind of rule, a grammar in
the engine's own Lark syntax (`Shape::Lark`), is the one the server writes for
them. No field of a request sets a trigger.

*A text that ends the reasoning.* A model that thinks may write what belongs
after its reasoning without closing the reasoning first: the marker that opens
the call of a tool. A `Structure` may name an interrupt (`Interrupt`): a text,
and the rule for what follows it. While a reply thinks the guide looks for the
closing tag and for that text; at whichever comes first the reasoning ends, and
after the text the interrupt's rule binds, from the token after the one that
completes it, as after a trigger. After the closing tag the structure's own
rule and trigger apply as without an interrupt, and a trigger that begins in
the token of the closing tag is seen. A structure with an interrupt and without
a trigger binds every reply, through one rule or the other, so such a reply may
not end while it thinks: the rows of its reasoning carry a mask of every token
but the ones that end a reply, and a draft that would end it there is cut. With
a trigger the reasoning carries no mask, and a reply may end in it. The guide
compiles the interrupt's rule next to the structure's (one grammar when the two
are the same), and refuses an interrupt whose text is empty. No field of a
request sets an interrupt; the server writes one for the tools of a request
(see [Server](server.md), "Tool calls").

*Drafts.* The text with drafts is the text without. Every row of a window gets
the mask of its own path, so a draft the rule forbids is never the pick of the
row before it and is rejected whatever happens. On top of that the conductor
cuts a forbidden draft, and the ones after it, from the n-gram drafter's proposal
before the window is chosen (`TokenFilter::admitted`), so the rows are not paid
for: a test with runs of one letter has every draft verified accepted, and 74 of
119 without the cut. Tokens the rule forces (one token allowed) are not offered
as drafts: the n-gram drafter is the drafter that joins next to an MTP block, and a
forced token would be a third source with a chance of one; it is not built.

*The model's drafts.* The tokens of the MTP block's drafts are not known
when the record is written, so the rows of a window that takes them carry no
mask. For the rows a rule leaves free that is exact: no mask applies there with
drafts or without. Most rules are free for most of a reply (the reasoning before
a `response_format`, a schema or a grammar; with tools that may be called, the
reasoning and the text before the trigger), so a request with a rule takes the
model's drafts whenever the next token is free, as a request without a rule
does, and takes the n-gram drafter's alone while the rule is bound. Where the rule
starts to bind inside such a window is found when the picks are back
(`TokenFilter::free_rows`, which a `Structure` answers from the phase of the
reply, `Guide::free_rows`): the token that makes it bind (the last token of the
closing tag, of the trigger) was picked by a free row and is taken; the rows
after it needed a mask, so their picks are not the request's and their cells are
cut, as those of rejected drafts are. That token is then the newest token of the
next window, whose first row has its mask; its row is computed a second time,
which is the cost of a binding token inside a window, once per phase. The
devices are not involved: they get the count of accepted rows from the host as
for every window, and a window of this kind looks like any window of the model's
drafts. A filter that does not say which rows it leaves free is taken to bind
every row.

On the host (`crates/runtime/tests/free_drafts.rs`, the synthetic model with
text that never repeats and an MTP block that is right nine times in ten, a
window of 8 rows, a step priced at 3 ms and 0.03 ms a row and 0.3 ms a link of
the chain): a reply of 310 free tokens and 6 bound ones had 255 of the block's
drafts accepted and took 60 steps, and one of 310 free tokens and a call of 42
took 89; the same 310 tokens without a rule had 255 accepted in 55 steps, and
the two replies without drafts took 316 and 352 steps. The same file puts the
binding token at every pick of a window in turn, for both kinds of rule, and has
a filter that is free and bound in turn, after which the drafts come back.

On the devices the synthetic model shows the same on every placement
(`a_rule_that_is_free_in_stretches_takes_the_models_drafts_on_every_placement`
in `crates/runtime/tests/gpu.rs`: a rule that binds the last three of every 7,
10 or 13 tokens, on a second runtime of a model whose MTP block has seen the
reply): of a reply of 150 tokens, 22 to 30 windows took the block's drafts and
56 to 92 of its drafts were accepted, and the reply is the one without drafts.
On the 27B over both cards (`the_27b_calls_the_same_with_drafts_and_without`
in `crates/server/tests/real_tools.rs`) a request with tools whose reply thinks
and then calls (100 tokens) had 57 of the block's drafts verified and 40
accepted, next to 32 and 29 of the n-gram drafter's, and was the reply without drafts
token for token; the server reported 84 tokens a second for it and 31 for the
same request without drafts. That is one run of each, with an n-gram drafter that had
the text of two calls before it, and no benchmark. A reply that is a call from
its first token (the same test, thinking off) takes none of the block's
drafts, as the rule binds all of it: 20 of 27 and 37 of 37 of the n-gram drafter's
were accepted there.

Under a bound rule the model's drafts are not taken. A window verified without
masks could be kept as far as each pick is a token the mask of its row allows,
which the host can work out afterwards; that is exact for a row that takes the
most probable token and not for a sampled one, whose cuts (top-k, top-p and the
others) are made within the tokens the mask leaves. It is not built.

*Cost.* Host time, on the vocabulary of the 0.8B (248,320 tokens), one core of
the machine (`cargo run --release -p ares-grammar --example mask_cost`): reading
the vocabulary 0.5 s, once per runtime; compiling a JSON schema of moderate size
(eight properties, a pattern, enums, a nested object, an array) 0.7 ms; the mask
of a step along random walks through it a median of 1 microsecond, 61 at the 90th
percentile and 1.9 milliseconds at the longest (a step that opens something the
engine has not seen); `json_object` a median of 2 microseconds. A GBNF grammar
is read one character at a time, as in llama.cpp, and costs more: llama.cpp's
`json.gbnf` takes 6.8 milliseconds a mask (compiling 1.2 ms), a grammar of a few
literals 3 microseconds. The mask is worked out on the conductor's thread before
the step is launched, one row after the other, so it adds to the step; working it
out during the device's step (vLLM) is not built, and would hide the 6.8
milliseconds of a character grammar behind a step of a large model. Merging the
rules that are one lexeme (a string in quotes) into regular expressions was
tried and gave nothing for `json.gbnf`, whose string rule ends in optional white
space. The device applies the mask in the sampler, which costs what the section
on masks above says (6 to 16 percent of the draw of the most probable token).
Through the server, the 0.8B on the RTX 4070 wrote 400 tokens under a JSON
schema (an array of integers) at 412 tokens a second and 400 tokens of free
text at 415 (`the_grammar_costs_what_the_mask_costs_per_step` in
`crates/server/tests/real_structured.rs`, the best of three runs each).

*Errors.* A grammar or schema the engine refuses is a 400 that names the field
and the problem; both `grammar` and `json_schema` (or `response_format`) in one
request is a 400, where llama-server takes one of them silently. If the engine
fails in the middle of a reply (it ran out of the fuel it gives a grammar, or
the grammar allows no token and the model has no end token) the request ends
with an error and the text is not sent.
