//! Chat templates, with templates written for the tests.

mod common;

use ares_tokenizer::{
    ChatError, ChatOptions, ChatTemplate, Content, Json, Message, Tool, ToolCall,
};
use common::*;

/// A template in the shape of ChatML: turns introduced by `<|im_start|>role` and ended by
/// `<|im_end|>`, tools in the first turn, calls written as `<call>`.
const CHATML: &str = r#"
{%- macro text(content) -%}
    {%- if content is string -%}{{ content }}
    {%- else -%}{%- for part in content -%}{{ part.text }}{%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- if tools -%}
<|im_start|>system
{% for tool in tools %}{{ tool | tojson }}
{% endfor %}<|im_end|>
{% endif -%}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ text(message.content) | trim }}
{%- for call in message.tool_calls or [] %}
<call name="{{ call.function.name }}">{{ call.function.arguments | tojson }}</call>
{%- endfor %}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{% if enable_thinking is false %}<think></think>
{% endif -%}
{%- endif -%}
"#;

fn chatml() -> ChatTemplate {
    ChatTemplate::new(CHATML, "", "").unwrap()
}

fn render(
    template: &str,
    messages: &[Message],
    options: &ChatOptions,
) -> Result<String, ChatError> {
    ChatTemplate::new(template, "<bos>", "<eos>")?.render(messages, &[], options)
}

fn user(text: &str) -> Message {
    Message::new("user", text)
}

fn no_prompt() -> ChatOptions {
    ChatOptions { add_generation_prompt: false, ..ChatOptions::default() }
}

fn json(text: &str) -> Json {
    Json::parse(text).unwrap()
}

#[test]
fn renders_a_conversation_with_the_start_of_the_answer() {
    let messages = [
        Message::new("system", "Be brief."),
        user("Hi"),
        Message::new("assistant", "Hello!"),
        user("Bye"),
    ];
    assert_eq!(
        chatml().render(&messages, &[], &ChatOptions::default()).unwrap(),
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
         <|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        chatml().render(&messages[1..2], &[], &no_prompt()).unwrap(),
        "<|im_start|>user\nHi<|im_end|>\n"
    );
}

#[test]
fn passes_the_thinking_switch_to_the_template() {
    let off = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
    assert_eq!(
        chatml().render(&[user("Hi")], &[], &off).unwrap(),
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>\n"
    );
    // On is the default, and a template that does not look at it does not care.
    assert!(
        !chatml().render(&[user("Hi")], &[], &ChatOptions::default()).unwrap().contains("<think>")
    );
}

#[test]
fn treats_the_role_developer_as_system() {
    let template = "{% for m in messages %}[{{ m.role }}]{% endfor %}";
    let messages = [Message::new("developer", "x"), user("y")];
    assert_eq!(render(template, &messages, &no_prompt()).unwrap(), "[system][user]");
    // Unless the template deals with the channels of gpt-oss itself.
    let channels = "{# <|channel|> #}{% for m in messages %}[{{ m.role }}]{% endfor %}";
    assert_eq!(render(channels, &messages, &no_prompt()).unwrap(), "[developer][user]");
}

#[test]
fn gives_content_as_a_string_or_as_a_list_of_text_parts() {
    let template = "{% for m in messages %}{{ m.content is string }}:{{ m.content | length }}:{% if m.content is not string %}{{ m.content[0].type }}/{{ m.content[1].text }}{% endif %};{% endfor %}";
    let parts = Message {
        role: "user".into(),
        content: Content::Parts(vec!["ab".into(), "cd".into()]),
        ..Message::default()
    };
    let empty_parts =
        Message { role: "user".into(), content: Content::Parts(vec![]), ..Message::default() };
    assert_eq!(
        render(
            template,
            &[user("abc"), parts, empty_parts, Message::new("user", "")],
            &no_prompt()
        )
        .unwrap(),
        "True:3:;False:2:text/cd;True:0:;True:0:;"
    );
    // The text of the parts reaches the prompt through a template that knows both.
    let parts = Message {
        role: "user".into(),
        content: Content::Parts(vec!["ab".into(), "cd".into()]),
        ..Message::default()
    };
    assert_eq!(
        chatml().render(&[parts], &[], &no_prompt()).unwrap(),
        "<|im_start|>user\nabcd<|im_end|>\n"
    );
}

#[test]
fn gives_the_optional_fields_only_when_they_are_there() {
    let template = "{% for m in messages %}{{ m.reasoning_content is defined }}{{ m.name is defined }}{{ m.tool_call_id is defined }}{{ m.tool_calls is defined }};{% endfor %}";
    let full = Message {
        role: "tool".into(),
        content: Content::Text("r".into()),
        reasoning_content: "why".into(),
        name: "get".into(),
        tool_call_id: "c1".into(),
        tool_calls: vec![ToolCall { id: None, name: "f".into(), arguments: "{}".into() }],
    };
    assert_eq!(
        render(template, &[user("x"), full], &no_prompt()).unwrap(),
        "FalseFalseFalseFalse;TrueTrueTrueTrue;"
    );
}

#[test]
fn turns_the_arguments_of_a_call_into_an_object_and_keeps_the_order_of_keys() {
    let call = ToolCall {
        id: Some("c1".into()),
        name: "search".into(),
        arguments: r#"{"zeta": 1, "alpha": {"y": [1, 2.5, "x", null, true], "b": "ä\n"}, "n": 12345678901234567890}"#.into(),
    };
    let message =
        Message { role: "assistant".into(), tool_calls: vec![call], ..Message::default() };
    assert_eq!(
        chatml().render(&[message], &[], &no_prompt()).unwrap(),
        "<|im_start|>assistant\n<call name=\"search\">{\"zeta\": 1, \"alpha\": {\"y\": [1, 2.5, \"x\", null, true], \
         \"b\": \"ä\\n\"}, \"n\": -6101065172474983726}</call><|im_end|>\n"
    );
    let ids =
        "{% for m in messages %}{{ m.tool_calls[0].id }}|{{ m.tool_calls[0].type }};{% endfor %}";
    let with_id = Message {
        role: "assistant".into(),
        tool_calls: vec![ToolCall {
            id: Some("c9".into()),
            name: "f".into(),
            arguments: "{}".into(),
        }],
        ..Message::default()
    };
    let without_id = Message {
        role: "assistant".into(),
        tool_calls: vec![ToolCall { id: None, name: "f".into(), arguments: "{}".into() }],
        ..Message::default()
    };
    assert_eq!(
        render(ids, &[with_id, without_id], &no_prompt()).unwrap(),
        "c9|function;|function;"
    );
}

#[test]
fn refuses_tool_arguments_that_are_not_json() {
    let call = ToolCall { id: None, name: "search".into(), arguments: "{not json".into() };
    let message =
        Message { role: "assistant".into(), tool_calls: vec![call], ..Message::default() };
    let error = chatml().render(&[message], &[], &no_prompt()).unwrap_err();
    assert!(matches!(&error, ChatError::Invalid(what) if what.contains("search")), "{error}");
}

#[test]
fn shows_the_tools_as_the_api_describes_them() {
    let tools = [
        Tool {
            name: "weather".into(),
            description: "Gets the \"weather\"".into(),
            parameters: json(
                r#"{"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "default": 0.5}}, "required": ["city"]}"#,
            ),
        },
        Tool { name: "ping".into(), description: String::new(), parameters: json("{}") },
    ];
    assert_eq!(
        chatml().render(&[user("x")], &tools, &no_prompt()).unwrap(),
        "<|im_start|>system\n\
         {\"type\": \"function\", \"function\": {\"name\": \"weather\", \"description\": \"Gets the \\\"weather\\\"\", \
         \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \
         \"days\": {\"type\": \"integer\", \"default\": 0.5}}, \"required\": [\"city\"]}}}\n\
         {\"type\": \"function\", \"function\": {\"name\": \"ping\", \"description\": \"\", \"parameters\": {}}}\n\
         <|im_end|>\n<|im_start|>user\nx<|im_end|>\n"
    );
    // No tools, no `tools` for the template.
    assert_eq!(render("{{ tools is defined }}", &[user("x")], &no_prompt()).unwrap(), "False");
}

#[test]
fn writes_values_into_the_prompt_as_llama_cpp_does() {
    let r = |t: &str| render(t, &[user("x")], &no_prompt()).unwrap();
    assert_eq!(
        r("{{ true }}|{{ false }}|{{ none }}|{{ undefined_name }}|{{ 7 }}|{{ -3 }}"),
        "True|False|||7|-3"
    );
    assert_eq!(r("{{ 1.5 }}|{{ 2.0 }}|{{ 0.25 }}"), "1.5|2.0|0.25");
    // A list writes its items one after the other, a mapping nothing.
    assert_eq!(r("{{ [1, 'a', true] }}|{{ x }}"), "1aTrue|");
    // `string` writes Python's str().
    assert_eq!(
        r(
            "{{ true|string }}|{{ none|string }}|{{ 1.0|string }}|{{ 1e-5|string }}|{{ 'x'|string }}"
        ),
        "True|None|1.0|1e-05|x"
    );
    assert_eq!(r("{{ 'a' ~ 1 ~ 'b' }}|{{ 'a' + 'b' }}"), "a1b|ab");
}

#[test]
fn trims_ascii_white_space_only_and_counts_bytes() {
    // C's isspace: space, \t \n \v \f \r. Not U+00A0, U+2003, U+0085.
    let padded = Message::new("user", " \t\n\u{b}\u{c}\r\u{a0}é\u{2003}\u{85} \n");
    let template = "{% for m in messages %}[{{ m.content | trim }}]{{ m.content | trim | length }};{% endfor %}";
    assert_eq!(render(template, &[padded], &no_prompt()).unwrap(), "[\u{a0}é\u{2003}\u{85}]9;");
    let strips = "{{ ' ab '.strip() }}|{{ 'xxabxx'.strip('x') }}|{{ 'xxabxx'.lstrip('x') }}|{{ 'xxabxx'.rstrip('x') }}|{{ '\n a \n'.rstrip('\n') }}";
    assert_eq!(render(strips, &[user("x")], &no_prompt()).unwrap(), "ab|ab|abxx|xxab|\n a ");
}

#[test]
fn writes_json_the_way_python_does() {
    let template = "{{ v | tojson }}";
    let with = |v: &str| {
        let options = ChatOptions {
            add_generation_prompt: false,
            variables: vec![("v".into(), json(v))],
            ..ChatOptions::default()
        };
        render(template, &[user("x")], &options).unwrap()
    };
    assert_eq!(with(r#"{"b": [1, 2], "a": {}, "c": []}"#), r#"{"b": [1, 2], "a": {}, "c": []}"#);
    assert_eq!(
        with(r#""ä€😀 \" \\ \n \t \u0001 </script> & '""#),
        "\"ä€😀 \\\" \\\\ \\n \\t \\u0001 </script> & '\""
    );
    assert_eq!(
        with("[0.5, 1.0, 100.0, 1e-5, 1e16, 1e22, -0.0, 3.14159, 123456789.125]"),
        "[0.5, 1.0, 100.0, 1e-05, 1e+16, 1e+22, -0.0, 3.14159, 123456789.125]"
    );
    assert_eq!(
        with("[0, -7, 9223372036854775807, 9223372036854775808]"),
        "[0, -7, 9223372036854775807, -9223372036854775808]"
    );
    assert_eq!(with("[true, false, null]"), "[true, false, null]");
}

#[test]
fn turns_raise_exception_into_an_error_with_its_message() {
    let template = "{% if messages|length > 1 %}{{ raise_exception('too many: ' ~ messages|length) }}{% endif %}ok";
    assert_eq!(render(template, &[user("x")], &no_prompt()).unwrap(), "ok");
    let error = render(template, &[user("x"), user("y")], &no_prompt()).unwrap_err();
    assert!(
        matches!(&error, ChatError::Render(what) if what.contains("too many: 2") && what.contains("line 1")),
        "{error}"
    );
}

#[test]
fn refuses_templates_that_use_what_is_not_supported_and_says_what_and_where() {
    let cases = [
        ("{{ x | upper }}", "the filter 'upper'"),
        ("{{ x | join(',') }}", "the filter 'join'"),
        ("{{ x is divisibleby 2 }}", "the test 'divisibleby'"),
        ("a\n{{ x.upper() }}", "the method '.upper()'"),
        ("{{ range(3) }}", "the function 'range'"),
        ("{{ strftime_now('%Y') }}", "'strftime_now'"),
        ("{{ date_string }}", "'date_string'"),
        ("{{ a // b }}", "an arithmetic operator"),
        ("{{ a / b }}", "an arithmetic operator"),
        ("{{ a % b }}", "an arithmetic operator"),
        ("{{ a ** b }}", "an arithmetic operator"),
        ("{{ 1 < a < 3 }}", "a chained comparison"),
        ("{{ a[1:2] }}", "a slice other than"),
        ("{{ a[::2] }}", "a slice other than"),
        ("{{ {'a': 1} }}", "a dictionary literal"),
        ("{{ x.items }}", "'.items' without calling it"),
        ("{{ schema.keys }}", "'.keys' without calling it"),
        ("{% set x %}y{% endset %}", "a set block"),
        ("{% with a = 1 %}{{ a }}{% endwith %}", "a with block"),
        ("{% filter upper %}x{% endfilter %}", "a filter block"),
        (
            "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(a=1) }}",
            "keyword arguments in a call of the macro 'm'",
        ),
        ("{{ 'a' if x }}", "a conditional expression without else"),
        ("{{ f(*args) }}", "the function 'f'"),
    ];
    for (template, what) in cases {
        match ChatTemplate::new(template, "", "") {
            Err(ChatError::Unsupported { what: found, line }) => {
                assert!(found.contains(what), "{template:?}: {found}");
                assert_eq!(line, if template.starts_with("a\n") { 2 } else { 1 }, "{template:?}");
            }
            other => panic!("{template:?}: {other:?}"),
        }
    }
    // Statements MiniJinja has no switch for here are syntax errors.
    for template in [
        "{% include 'x' %}",
        "{% extends 'x' %}",
        "{% import 'x' as y %}",
        "{% for i in x %}{% break %}{% endfor %}",
        "{% call m() %}x{% endcall %}",
        "{% do x %}",
    ] {
        let error = ChatTemplate::new(template, "", "").unwrap_err();
        assert!(
            matches!(error, ChatError::Syntax(_) | ChatError::Unsupported { .. }),
            "{template:?}: {error}"
        );
    }
}

#[test]
fn reports_a_template_that_is_not_jinja() {
    for template in [
        "{% if %}",
        "{{ ",
        "{% for x in %}{% endfor %}",
        "{% endif %}",
        "{{ 'a }}",
        "{% for a.b in x %}{% endfor %}",
    ] {
        let error = ChatTemplate::new(template, "", "").unwrap_err();
        assert!(matches!(error, ChatError::Syntax(_)), "{template:?}: {error}");
    }
}

#[test]
fn reads_a_list_backwards_without_failing_on_an_empty_one() {
    let template = "[{% for m in messages[::-1] %}{{ m.role }},{% endfor %}][{% for t in tools[::-1] %}{{ t }}{% endfor %}][{{ messages[::-1] | length }}]";
    assert_eq!(render(template, &[], &no_prompt()).unwrap(), "[][][0]");
    assert_eq!(
        render(
            template,
            &[user("a"), Message::new("assistant", "b"), Message::new("tool", "c")],
            &no_prompt()
        )
        .unwrap(),
        "[tool,assistant,user,][][3]"
    );
    // Only that slice is supported, and only on a list: a string is refused when it is read.
    let on_string = "{{ x[::-1] }}";
    let options = ChatOptions {
        variables: vec![("x".into(), Json::String("abc".into()))],
        ..ChatOptions::default()
    };
    assert!(matches!(render(on_string, &[user("x")], &options), Err(ChatError::Render(_))));
}

#[test]
fn lets_the_caller_set_variables_but_not_the_tokens_of_the_model() {
    // enable_thinking, preserve_thinking, clear_thinking, drop_thinking, reasoning_effort,
    // reasoning_strength, bos_token, eos_token, extra
    let template = "{{ enable_thinking }}|{{ preserve_thinking }}|{{ clear_thinking }}|{{ drop_thinking }}|{{ reasoning_effort }}|{{ reasoning_strength }}|{{ bos_token }}|{{ eos_token }}|{{ extra }}";
    let r = |variables: Vec<(&str, Json)>, thinking: bool| {
        let options = ChatOptions {
            add_generation_prompt: false,
            enable_thinking: thinking,
            variables: variables.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
            template: None,
        };
        render(template, &[user("x")], &options)
            .unwrap()
            .split('|')
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let fields = |f: [&str; 9]| f.map(str::to_owned).to_vec();
    assert_eq!(r(vec![], true), fields(["True", "", "", "", "", "", "<bos>", "<eos>", ""]));
    assert_eq!(r(vec![], false), fields(["False", "", "", "", "", "", "<bos>", "<eos>", ""]));
    // The key of the API sets four names that templates use for it.
    assert_eq!(
        r(vec![("preserve_reasoning", Json::Bool(true))], true),
        fields(["True", "True", "False", "False", "", "", "<bos>", "<eos>", ""])
    );
    // A variable of the caller wins over what is derived.
    assert_eq!(
        r(
            vec![("preserve_reasoning", Json::Bool(true)), ("clear_thinking", Json::Bool(true))],
            true
        ),
        fields(["True", "True", "True", "False", "", "", "<bos>", "<eos>", ""])
    );
    // The effort is also given under a second name, unless it is empty.
    assert_eq!(
        r(vec![("reasoning_effort", Json::String("low".into()))], true),
        fields(["True", "", "", "", "low", "low", "<bos>", "<eos>", ""])
    );
    assert_eq!(
        r(vec![("reasoning_effort", Json::String(String::new()))], true),
        fields(["True", "", "", "", "", "", "<bos>", "<eos>", ""])
    );
    assert_eq!(
        r(vec![("enable_thinking", Json::Bool(false))], true),
        fields(["False", "", "", "", "", "", "<bos>", "<eos>", ""])
    );
    // The tokens of the model cannot be replaced by the caller.
    assert_eq!(
        r(vec![("extra", Json::Int(5)), ("bos_token", Json::String("X".into()))], true),
        fields(["True", "", "", "", "", "", "<bos>", "<eos>", "5"])
    );
}

#[test]
fn offers_loop_variables_and_namespaces() {
    let template = "{% set ns = namespace(sum=0, last='') %}{% for m in messages %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.length }}/{{ loop.revindex }}/{{ loop.first }}/{{ loop.last }}/{{ loop.previtem.role }}/{{ loop.nextitem.role }};{% set ns.sum = ns.sum + loop.index %}{% set ns.last = m.role %}{% endfor %}{{ ns.sum }}{{ ns.last }}";
    assert_eq!(
        render(template, &[user("a"), Message::new("assistant", "b")], &no_prompt()).unwrap(),
        "1/0/2/2/True/False//assistant;2/1/2/1/False/True/user/;3assistant"
    );
}

#[test]
fn defines_macros_with_default_arguments() {
    // A default is read from the template, not from the arguments before it.
    let template = "{% set d = 'D' %}{% macro m(a, b='B', c=d) %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ m('x') }}{{ m('x', 'y') }}{{ m('x', 'y', 'z') }}";
    assert_eq!(render(template, &[user("x")], &no_prompt()).unwrap(), "xBDxyDxyz");
}

#[test]
fn reads_the_conditions_of_a_template_like_llama_cpp() {
    let r = |t: &str| render(t, &[user("x")], &no_prompt()).unwrap();
    // `iterable` and `sequence` are true for strings, lists, mappings and undefined values.
    assert_eq!(
        r(
            "{{ 'a' is iterable }}{{ [] is iterable }}{{ messages[0] is iterable }}{{ nope is iterable }}{{ 1 is iterable }}{{ none is iterable }}"
        ),
        "TrueTrueTrueTrueFalseFalse"
    );
    assert_eq!(
        r(
            "{{ 'a' is sequence }}{{ nope is sequence }}{{ messages[0] is mapping }}{{ [] is mapping }}"
        ),
        "TrueTrueTrueFalse"
    );
    assert_eq!(
        r(
            "{{ true is true }}{{ 1 is true }}{{ false is false }}{{ none is false }}{{ none is none }}{{ nope is undefined }}{{ nope is defined }}"
        ),
        "TrueFalseTrueFalseTrueTrueFalse"
    );
    // `default` replaces undefined and none, and with `boolean` anything false.
    assert_eq!(
        r(
            "{{ nope | default('d') }}{{ none | default('d') }}{{ '' | default('d') }}{{ '' | default('d', true) }}{{ 0 | default('d', boolean=true) }}"
        ),
        "dddd"
    );
    // Out of range and unknown attributes are undefined, and undefined has attributes.
    assert_eq!(r("[{{ messages[5] }}][{{ messages[0].nope.deeper }}][{{ nope.a.b }}]"), "[][][]");
    assert_eq!(
        r(
            "{{ 'a,b,,c'.split(',') | length }}{{ 'a b'.split()[1] }}{{ 'a,b,c'.split(',', 1)[1] }}{{ 'a,b,c'.rsplit(',', 1)[0] }}"
        ),
        "4bb,ca,b"
    );
    assert_eq!(
        r("{{ 'abc'.startswith('ab') }}{{ 'abc'.endswith('bc') }}{{ 'abc'.endswith('x') }}"),
        "TrueTrueFalse"
    );
    assert_eq!(
        r(
            "{% for k, v in messages[0].items() %}{{ k }}={{ v }};{% endfor %}{{ messages[0].keys() }}{{ messages[0].get('role') }}{{ messages[0].get('x', 'd') }}"
        ),
        "role=user;content=x;rolecontentuserd"
    );
    assert_eq!(r("{% for k, v in messages[0] | items %}{{ k }};{% endfor %}"), "role;content;");
}

#[test]
fn takes_the_templates_of_the_model_file() {
    let template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<a>{% endif %}";
    let t = tiny().str("tokenizer.chat_template", template).load("chat").unwrap();
    assert!(t.has_chat_template());
    assert_eq!(t.render_chat(&[user("hi")], &[], &ChatOptions::default()).unwrap(), "<user>hi<a>");
    // The text still has the control tokens as text; encoding it with parse_special makes them tokens.
    let prompt = t.render_chat(&[user("hi")], &[], &no_prompt()).unwrap();
    assert_eq!(prompt, "<user>hi");

    let none = tiny_tokenizer("nochat");
    assert!(!none.has_chat_template());
    assert!(matches!(
        none.render_chat(&[user("hi")], &[], &ChatOptions::default()),
        Err(ChatError::NoTemplate)
    ));
}

#[test]
fn uses_the_template_for_tools_when_there_are_tools() {
    let t = tiny()
        .str("tokenizer.chat_template", "default")
        .str("tokenizer.chat_template.tool_use", "tools")
        .load("tooluse")
        .unwrap();
    let tool = Tool { name: "f".into(), description: String::new(), parameters: json("{}") };
    assert_eq!(t.render_chat(&[user("x")], &[], &ChatOptions::default()).unwrap(), "default");
    assert_eq!(
        t.render_chat(&[user("x")], std::slice::from_ref(&tool), &ChatOptions::default()).unwrap(),
        "tools"
    );
    let only_tools =
        tiny().str("tokenizer.chat_template.tool_use", "tools").load("onlytools").unwrap();
    assert_eq!(
        only_tools.render_chat(&[user("x")], &[], &ChatOptions::default()).unwrap(),
        "tools"
    );
}

#[test]
fn loads_a_model_whose_template_is_not_supported_and_fails_when_it_is_used() {
    let t = tiny().str("tokenizer.chat_template", "{{ x | upper }}").load("badchat").unwrap();
    assert!(t.has_chat_template());
    let error = t.render_chat(&[user("x")], &[], &ChatOptions::default()).unwrap_err();
    assert!(
        matches!(&error, ChatError::Unsupported { what, .. } if what.contains("upper")),
        "{error}"
    );
    // Text still works.
    assert_eq!(t.encode("hello", Default::default()), [HELLO]);
}

#[test]
fn gives_the_template_the_tokens_of_the_model_and_removes_them_where_the_tokenizer_adds_them() {
    let template =
        "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}";
    let t = tiny()
        .u32("tokenizer.ggml.bos_token_id", IM_START)
        .str("tokenizer.chat_template", template)
        .load("tokens")
        .unwrap();
    // The tokenizer adds neither, so the text keeps both.
    assert_eq!(t.render_chat(&[user("x")], &[], &no_prompt()).unwrap(), "<|im_start|>x<|im_end|>");
    let t = tiny()
        .u32("tokenizer.ggml.bos_token_id", IM_START)
        .bool("tokenizer.ggml.add_bos_token", true)
        .bool("tokenizer.ggml.add_eos_token", true)
        .str("tokenizer.chat_template", template)
        .load("tokens2")
        .unwrap();
    // Encoding adds them, so the text must not have them as well.
    assert_eq!(t.render_chat(&[user("x")], &[], &no_prompt()).unwrap(), "x");
    // A model without a bos token gives the template an empty one.
    let t = tiny().str("tokenizer.chat_template", "[{{ bos_token }}]").load("nobos").unwrap();
    assert_eq!(t.render_chat(&[user("x")], &[], &no_prompt()).unwrap(), "[]");
}
