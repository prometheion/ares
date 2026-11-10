#!/usr/bin/env python3
"""Write probes.jsonl: small templates, each run with many values, for examples/probes.rs.

    python3 -I tools/make_probes.py > probes.jsonl

A probe is one template snippet (a filter, a test, a method, an operator, a statement) and
one value `v` for it, handed to the template as a variable. The snippets cover what the
dialect of this crate supports and a good deal that it does not; the values are the awkward
ones (empty and padded strings, white space of every kind, floats, numbers beyond 64 bits,
nesting, none, and no value at all). examples/probes.rs runs every probe through llama.cpp
and through this crate and sorts the results by whether they agree.
"""
import json
vals = [
    "", " ", "abc", "a,b,c", " a b ", "\n x \n", " nbsp ", " em ", "　", "ä", "日本語", "aaa", ",,,", "ab", "\t\x0b\x0c\r\n", "x\x1cy\x1c", "\x1f", "\u0085nel\u0085", "a b", "ab,cd,ab", "<b>&amp;</b>",
    0, 1, -1, 2, 7, 1.5, -0.0, 0.1, 1e-7, 1e16, 1e22, 100.0, 2.5e20, 12345678901234567890, 123456789012, 
    True, False, None,
    [], [1, 2, 3], ["a", "b"], [[1, 2], [3, 4]], [{"x": 1}, {"x": 2}], ["a", None, 1.5, True], [0, ""],
    {}, {"a": 1, "b": 2}, {"a": {"x": 1}, "b": [1]}, {"items": 5, "keys": 6, "length": 7}, {"z": "ä", "a": 0.5, "m": None},
    "__undefined__",
]
snippets = [
    "{{ v|trim }}", "{{ v|string }}", "{{ v|safe }}", "{{ v|tojson }}", "{{ v|length }}", "{{ v|count }}",
    "{{ v|default('D') }}", "{{ v|default('D', true) }}", "{{ v|default('D', boolean=true) }}", "{{ v|d('D') }}", "{{ v|default }}",
    "{% for k, x in v|items %}{{ k }}={{ x }};{% endfor %}", "{{ v|items }}",
    "{{ v }}", "{{ v|string|length }}", "{{ (v|string)|trim }}",
    "{{ v is string }}|{{ v is mapping }}|{{ v is iterable }}|{{ v is sequence }}|{{ v is none }}|{{ v is defined }}|{{ v is undefined }}|{{ v is true }}|{{ v is false }}",
    "{{ v is not string }}|{{ v is not defined }}",
    "{{ 'x' ~ v }}", "{{ v ~ 'x' }}", "{{ 'x' + v }}", "{{ v + 'x' }}", "{{ v ~ v }}", "{{ v + v }}",
    "{{ v.startswith('a') }}", "{{ v.endswith('c') }}", "{{ v.startswith('') }}", "{{ v.split() }}", "{{ v.split(',') }}", "{{ v.split(',', 1) }}", "{{ v.rsplit(',', 1) }}", "{{ v.rsplit(',') }}", "{{ v.split(',')|length }}",
    "{{ v.split('b')[0] }}", "{{ v.split('b')[-1] }}", "{{ v.strip() }}", "{{ v.strip('ab') }}", "{{ v.lstrip() }}", "{{ v.rstrip() }}", "{{ v.rstrip('\\n') }}", "{{ v.lstrip('\\n') }}", "{{ v.strip(' ,') }}",
    "{{ v.items() }}", "{% for k, x in v.items() %}{{ k }}:{{ x }};{% endfor %}", "{{ v.keys() }}", "{{ v.values() }}", "{{ v.get('a') }}", "{{ v.get('zz', 'dflt') }}", "{{ v.get('zz') }}", "{{ v.get('a', 'dflt') }}",
    "{{ v[0] }}", "{{ v[1:3] }}", "{{ v[::-1] }}", "{{ v[-1] }}", "{{ v[1] }}", "{{ v[:2] }}", "{{ v[1:] }}", "{{ v[::2] }}", "{{ v[5] }}",
    "{{ v.a }}", "{{ v['a'] }}", "{{ v.zz }}", "{{ v.a.x }}", "{{ v[0].x }}", "{{ v.zz.yy }}", "{{ v['zz']['yy'] }}",
    "{{ v == 1 }}", "{{ v != 1 }}", "{{ v < 2 }}", "{{ v >= 2 }}", "{{ v == 'a' }}", "{{ v == 'abc' }}", "{{ v == [1, 2, 3] }}", "{{ v == {} }}", "{{ v == none }}", "{{ v == true }}", "{{ v == false }}", "{{ v == 0 }}", "{{ v == '' }}", "{{ v == 1.5 }}", "{{ v < 'b' }}", "{{ v > 'a' }}",
    "{{ v in [1, 2, 'a', 'abc'] }}", "{{ 'a' in v }}", "{{ 1 in v }}", "{{ 'a' not in v }}", "{{ v not in 'abc' }}", "{{ v in 'abc' }}", "{{ v in (1, 2) }}", "{{ 'a' in v and 'b' in v }}",
    "{{ v + 1 }}", "{{ v - 1 }}", "{{ v * 2 }}", "{{ -v }}", "{{ 1 + v }}", "{{ 2 * v }}", "{{ v * 1.5 }}", "{{ (v|length) - 1 }}", "{{ v * 'ab' }}", "{{ 'ab' * v }}",
    "{% if v %}T{% else %}F{% endif %}", "{% if not v %}T{% else %}F{% endif %}", "{{ 'a' if v else 'b' }}", "{{ v and 'yes' }}", "{{ v or 'no' }}", "{{ (v and 1) or 2 }}", "{{ not v }}",
    "{% for x in v %}{{ loop.index }}/{{ loop.length }}/{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }}:{{ x }}|{{ loop.previtem }}|{{ loop.nextitem }};{% endfor %}",
    "{% for x in v %}{{ x }}{% else %}EMPTY{% endfor %}", "{% for k in v %}{{ k }},{% endfor %}", "{% for x in v if x %}{{ x }},{% endfor %}",
    "{% set ns = namespace(a=1, b='x') %}{% for x in v %}{% set ns.a = ns.a + 1 %}{% set ns.b = ns.b ~ x %}{% endfor %}{{ ns.a }}{{ ns.b }}",
    "{% set a, b = v %}{{ a }}-{{ b }}", "{% set x = v %}{{ x }}", "{% for a, b in v %}{{ a }}-{{ b }};{% endfor %}",
    "{% macro m(a, b='dflt', c=v) %}[{{ a }}|{{ b }}|{{ c }}]{% endmacro %}{{ m(1) }}{{ m(1, 2) }}{{ m(1, c=3) }}{{ m(a=v) }}{{ m(v, b=v) }}",
    "{% macro m(x) %}{{ x|trim }}{% endmacro %}{{ m(v) }}|{{ m(v)|length }}",
    "{% macro m(x) %}{% if x is string %}S{% elif x is iterable %}I{% else %}O{% endif %}{% endmacro %}{{ m(v) }}",
    "{% set t = ('xhigh', 'medium') %}{{ v in t }}|{{ v not in t }}",
    "{{ [1, 2, v] }}", "{{ [v, v]|length }}", "{{ (1, v) }}",
    "{{ v|string|trim|length }}", "{{ v|tojson|safe }}", "{{ v|tojson|length }}", "{{ v|items|length }}",
    "{{ true }}|{{ false }}|{{ none }}|{{ 1.0 }}|{{ 0.5 }}|{{ 1e3 }}|{{ -3 }}|{{ 2 * 3 }}|{{ 10 - 4 }}|{{ 'a' ~ 1 ~ 2.5 ~ true }}",
    "{{ raise_exception('boom ' ~ v) }}", "{{ raise_exception(v) }}",
    "{{ v.foo() }}", "{{ v|nosuchfilter }}", "{{ v is nosuchtest }}", "{{ nosuchfunc(v) }}",
    "{{ v|first }}", "{{ v|last }}", "{{ v|join(',') }}", "{{ v|list }}", "{{ v|upper }}", "{{ v|lower }}", "{{ v|replace('a', 'b') }}", "{{ v|int }}", "{{ v|float }}", "{{ v|abs }}", "{{ v|reverse }}", "{{ v|sort }}", "{{ v|unique }}", "{{ v|map(attribute='x') }}", "{{ v|selectattr('x') }}", "{{ v|indent(2) }}", "{{ v|title }}", "{{ v|capitalize }}", "{{ v|escape }}", "{{ v|e }}", "{{ v|tojson(indent=2) }}", "{{ v|dictsort }}", "{{ v|batch(2) }}", "{{ v|sum }}", "{{ v|min }}", "{{ v|max }}", "{{ v|round }}", "{{ v|format('x') }}", "{{ v|wordcount }}", "{{ v|truncate(3) }}",
    "{{ v.upper() }}", "{{ v.lower() }}", "{{ v.replace('a', 'b') }}", "{{ v.format() }}", "{{ v.title() }}", "{{ v.capitalize() }}", "{{ v.join(',') }}", "{{ v.index(1) }}", "{{ v.count('a') }}", "{{ v.find('a') }}", "{{ v.isdigit() }}", "{{ v.append(1) }}", "{{ v.pop() }}", "{{ v.update() }}",
    "{{ v // 2 }}", "{{ v / 2 }}", "{{ v % 2 }}", "{{ v ** 2 }}", "{{ 1 < v < 3 }}",
    "{{ range(3) }}", "{{ dict(a=1) }}", "{{ lipsum() }}", "{{ cycler(1,2) }}", "{{ strftime_now('%Y') }}", "{{ date_string }}", "{{ datetime }}",
    "{% set x %}captured{% endset %}{{ x }}", "{% with a = 1 %}{{ a }}{% endwith %}", "{% filter upper %}x{% endfilter %}", "{% do v %}", "{% for i in range(2) %}{% break %}{% endfor %}", "{% include 'x' %}", "{% call m() %}x{% endcall %}", "{% raw %}{{ v }}{% endraw %}", "{# comment #}[{{ v }}]", "{{ {'a': 1} }}", "{{ {'a': 1}.a }}", "{% for x in v recursive %}{{ x }}{% endfor %}",
]
cases = []
for sn in snippets:
    for v in vals:
        c = {"label": sn, "template": "<<" + sn + ">>", "messages": [{"role": "user", "content": "x"}], "add_generation_prompt": False}
        if v != "__undefined__":
            c["kwargs"] = {"v": v}
        else:
            c["kwargs"] = {}
        cases.append(c)
for c in cases:
    print(json.dumps(c, ensure_ascii=False))
