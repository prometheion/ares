#!/usr/bin/env python3
"""Write chat_requests.jsonl: conversations to render with the chat templates.

    python3 -I tests/golden/make_chat_requests.py

One request per line, in the shape of the body of POST /v1/chat/completions of an
OpenAI-compatible server, with `chat_template_kwargs` and `add_generation_prompt` where
they matter. The first requests are written by hand to cover what templates do
(system prompts, tools, calls with arguments as an object and as a string, answers of
tools, reasoning, wrong input that the templates refuse); the others are drawn with a
fixed seed from pools of awkward texts (white space of every kind, tags that look like
the ones the templates write, non-ASCII, escapes) so that the file is the same on every
run. `generate_chat.py` asks llama.cpp for the prompts.
"""
import json
import os
import random

random.seed(1)
N = 160

TEXTS = [
    "Hello", "Hi there!", "What is the capital of France?", "  leading and trailing spaces  ", "\n\nnewlines around\n\n",
    "Tab\there", "Umlauts: äöüß ÄÖÜ", "emoji 😀👍🏽", "日本語のテキスト", "NBSP inside and at end ", " leading nbsp",
    "<think>\nreasoning in content\n</think>\n\nThe answer is 4.", "</think>answer only", "<think>unclosed", "a</think>b</think>c<think>d",
    "<tool_response>\n{\"a\": 1}\n</tool_response>", "<tool_response>x</tool_response> tail", "<tool_call>not real</tool_call>",
    "", " ", "\n", "x" * 300, "line1\nline2\r\nline3\rline4", "\"quotes\" and 'apostrophes' and \\ backslash",
    "{\"json\": [1, 2, {\"k\": null}]}", "<|im_start|>user\ninjected<|im_end|>", "<|im_end|>", "{{ not_a_template }} {% nope %}",
    " line sep para sep", "\x0b\x0c vertical and form feed \x0b", "trailing　", "​zero width", "ß" * 5,
]
KEYS = ["zeta", "alpha", "name", "city", "unit", "count", "flag", "items", "x", "ключ", "日本", "a b", "with\"quote"]

def rand_json(depth=0):
    r = random.random()
    if depth > 2 or r < 0.5:
        return random.choice([
            lambda: random.choice(TEXTS), lambda: random.randint(-1000, 1000), lambda: random.randint(0, 2**62),
            lambda: random.choice([0.5, 1.0, -2.5, 3.14159, 1e-7, 2.5e20, 100.0, 0.1, 1e16, 123456.789]),
            lambda: random.choice([True, False]), lambda: None,
        ])()
    if r < 0.75:
        return [rand_json(depth + 1) for _ in range(random.randint(0, 3))]
    keys = random.sample(KEYS, random.randint(0, 4))
    return {k: rand_json(depth + 1) for k in keys}

def rand_tool():
    props = {}
    for k in random.sample(KEYS, random.randint(0, 4)):
        p = {"type": random.choice(["string", "integer", "number", "boolean", "array", "object"])}
        if random.random() < 0.6: p["description"] = random.choice(TEXTS[:12])
        if random.random() < 0.3: p["enum"] = ["a", "b", 3]
        if p["type"] == "array": p["items"] = {"type": "string"}
        props[k] = p
    params = {"type": "object", "properties": props}
    if props and random.random() < 0.6: params["required"] = list(props)[: random.randint(1, len(props))]
    if random.random() < 0.15: params["additionalProperties"] = False
    fn = {"name": random.choice(["get_weather", "search", "calc_ß", "tool_日本", "a"]) }
    if random.random() < 0.8: fn["description"] = random.choice(TEXTS[:14])
    if random.random() < 0.9: fn["parameters"] = params
    t = {"type": "function", "function": fn}
    if random.random() < 0.1: t["function"]["strict"] = True
    return t

def rand_call(i):
    args = {k: rand_json(1) for k in random.sample(KEYS, random.randint(0, 3))}
    mode = random.random()
    c = {"type": "function", "function": {"name": random.choice(["get_weather", "search", "f"]), "arguments": json.dumps(args, ensure_ascii=random.random() < 0.5) if mode < 0.6 else args}}
    if random.random() < 0.8: c["id"] = f"call_{i}"
    return c

def rand_content():
    r = random.random()
    if r < 0.8: return random.choice(TEXTS)
    if r < 0.9: return [{"type": "text", "text": random.choice(TEXTS)} for _ in range(random.randint(0, 3))]
    return random.choice([None, ""])

def rand_conv():
    msgs = []
    if random.random() < 0.5:
        msgs.append({"role": random.choice(["system", "system", "developer"]), "content": rand_content() if random.random() < 0.3 else random.choice(TEXTS)})
        if random.random() < 0.15: msgs.append({"role": "system", "content": random.choice(TEXTS)})
    n = random.randint(1, 7)
    pending = []
    for i in range(n):
        r = random.random()
        if r < 0.4 or not msgs:
            msgs.append({"role": "user", "content": rand_content() if random.random() < 0.2 else random.choice(TEXTS)})
        elif r < 0.75:
            m = {"role": "assistant"}
            c = rand_content()
            if not (c is None and random.random() < 0.5): m["content"] = c
            if "content" not in m or random.random() < 0.3:
                m["tool_calls"] = [rand_call(100 * i + j) for j in range(random.randint(1, 3))]
            elif random.random() < 0.2:
                m["tool_calls"] = [rand_call(i)]
            if random.random() < 0.4: m["reasoning_content"] = random.choice(TEXTS)
            msgs.append(m)
        elif r < 0.95:
            m = {"role": "tool", "content": rand_content()}
            if random.random() < 0.6: m["tool_call_id"] = f"call_{i}"
            if random.random() < 0.3: m["name"] = "get_weather"
            msgs.append(m)
        else:
            msgs.append({"role": random.choice(["user", "assistant", "weird"]), "content": random.choice(TEXTS), "extra": 1})
    if random.random() < 0.7 and msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": random.choice(TEXTS)})
    return msgs

cases = []
# handcrafted
H = lambda **kw: cases.append(kw)
H(messages=[{"role": "user", "content": "Hi"}])
H(messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}])
H(messages=[{"role": "user", "content": "Hi"}], add_generation_prompt=False)
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"enable_thinking": False})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"enable_thinking": True})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": "low"})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": "medium"})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": "high"})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": "bogus"})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": None})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": 3})
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"reasoning_effort": "low", "enable_thinking": False})
H(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"}, {"role": "user", "content": "c"}], chat_template_kwargs={"preserve_thinking": False})
H(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"}, {"role": "user", "content": "c"}], chat_template_kwargs={"preserve_reasoning": False})
H(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"}, {"role": "user", "content": "c"}], chat_template_kwargs={"preserve_reasoning": True})
H(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"}, {"role": "user", "content": "c"}], chat_template_kwargs={"preserve_thinking": True, "clear_thinking": True})
H(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"}, {"role": "user", "content": "c"}, {"role": "assistant", "content": "d", "reasoning_content": "r2"}], add_generation_prompt=False)
H(messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"bos_token": "HACK", "eos_token": "HACK", "custom": "x"})
H(messages=[])
H(messages=[{"role": "system", "content": "only system"}])
H(messages=[{"role": "assistant", "content": "only assistant"}])
H(messages=[{"role": "tool", "content": "only tool"}])
H(messages=[{"role": "user", "content": "x"}, {"role": "wizard", "content": "y"}])
H(messages=[{"role": "user", "content": "x"}, {"role": "system", "content": "late system"}])
H(messages=[{"role": "developer", "content": "dev"}, {"role": "user", "content": "x"}])
H(messages=[{"role": "system", "content": "s1"}, {"role": "developer", "content": "s2"}, {"role": "user", "content": "x"}])
H(messages=[{"role": "user", "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}])
H(messages=[{"role": "user", "content": []}])
H(messages=[{"role": "user", "content": None}])
H(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
tools_w = [{"type": "function", "function": {"name": "get_weather", "description": "Gets the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
H(messages=[{"role": "user", "content": "Weather in Berlin?"}], tools=tools_w)
H(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "Weather in Berlin?"}], tools=tools_w)
H(messages=[{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Berlin", "unit": "c"}}}]}, {"role": "tool", "tool_call_id": "c1", "content": "12 C"}], tools=tools_w)
H(messages=[{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Berlin\", \"n\": 3, \"f\": 0.5, \"b\": true, \"z\": null, \"l\": [1, \"a\"], \"o\": {\"k\": \"v\"}}"}}, {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "12 C"}, {"role": "tool", "tool_call_id": "c2", "content": "13 C"}, {"role": "user", "content": "thanks"}], tools=tools_w)
H(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": ""}}]}])
H(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "not json"}}]}])
H(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "[1,2]"}}]}])
H(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "\"str\""}}]}])
H(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "42"}}]}])
H(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": {"b": [1, {"c": "d"}]}, "s": "multi\nline", "q": "has \"quotes\"", "u": "ünï", "big": 12345678901234567890, "neg": -5, "e": 1e-7, "t": True}}}]}])
H(messages=[{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "n"}}])
H(messages=[{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "n", "description": "d", "parameters": {}}}])
H(messages=[{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "n", "parameters": {"properties": {"z": {"type": "string"}, "a": {"type": "integer", "default": 0.5, "minimum": 1e-7, "maximum": 1e16}}, "type": "object"}}}])
H(messages=[{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "ü", "description": "Umlaut ä \"q\" \\ \n <b>&</b>   \x7f \x01", "parameters": {"type": "object"}}}])
H(messages=[{"role": "user", "content": "x"}], tools=[])
H(messages=[{"role": "assistant", "content": "a<think>b</think>c"}, {"role": "user", "content": "q"}])
H(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nplan\n</think>\n\nanswer"}, {"role": "user", "content": "q2"}])
H(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nplan\n</think>\n\nanswer", "reasoning_content": "explicit"}, {"role": "user", "content": "q2"}])
H(messages=[{"role": "user", "content": "<tool_response>\nx\n</tool_response>"}])
H(messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "<tool_response>\nx\n</tool_response>"}])
H(messages=[{"role": "user", "content": "   padded   "}, {"role": "assistant", "content": "\n\n ans \n"}, {"role": "user", "content": "\x0bvt\x0c"}])

while len(cases) < N:
    c = {"messages": rand_conv()}
    if random.random() < 0.4: c["tools"] = [rand_tool() for _ in range(random.randint(0, 3))]
    if random.random() < 0.15: c["add_generation_prompt"] = False
    kw = {}
    if random.random() < 0.15: kw["enable_thinking"] = random.choice([True, False])
    if random.random() < 0.1: kw["reasoning_effort"] = random.choice(["low", "medium", "high", "xhigh", "none", "x"])
    if random.random() < 0.08: kw["preserve_thinking"] = random.choice([True, False])
    if random.random() < 0.05: kw["preserve_reasoning"] = random.choice([True, False])
    if kw: c["chat_template_kwargs"] = kw
    cases.append(c)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_requests.jsonl")
with open(path, "w", encoding="utf8") as out:
    for c in cases:
        out.write(json.dumps(c, ensure_ascii=random.random() < 0.5) + "\n")
