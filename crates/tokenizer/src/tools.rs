//! How a chat template writes the calls of tools.
//!
//! A model calls a tool by writing the call into its reply, in the form its
//! chat template writes the calls of earlier turns into the prompt.
//! [`ChatTemplate::tool_format`] reads that form off the template into a
//! [`ToolFormat`]: it renders an
//! assistant message that calls made-up tools with made-up arguments, finds the
//! names, keys and values in the text, and takes what stands between them as
//! the pieces of the format. Nothing is known by the name of a model.
//!
//! One form is understood, the one of Qwen3.8's template: the name of the tool
//! in a tag, and every argument between a tag that names it and a tag that
//! closes it, a string as it is and every other value as JSON.
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! The pieces are checked: the calls of the rendered message are written again
//! from the pieces, and a template whose text is not exactly that has no format
//! here. Templates that write a call as one JSON object (`{"name": ...,
//! "arguments": ...}` between tags) are among those.

use ares_api::ToolFormat;

use crate::chat::{ChatOptions, ChatTemplate, Message, Tool, ToolCall};
use crate::json::Json;

const ANSWER: &str = "ANSWER_PROBE_c91d";
const FIRST: &str = "tool_probe_a41c";
const SECOND: &str = "tool_probe_b72e";
const THIRD: &str = "tool_probe_c93f";
const TEXT_KEY: &str = "key_probe_d15a";
const TEXT_VALUE: &str = "VALUE_PROBE_e26b";
const NUMBER_KEY: &str = "key_probe_f37c";
const NUMBER_VALUE: &str = "7204";
const LAST_KEY: &str = "key_probe_a48d";
const LAST_VALUE: &str = "VALUE_PROBE_b59e";

/// Reads the format off `template`, or `None` if the template writes calls in
/// another form, or not at all ([`ChatTemplate::tool_format`]).
pub(crate) fn detect(template: &ChatTemplate) -> Option<ToolFormat> {
    let text = rendered(template)?;
    // The names stand in the list of tools too, which comes first.
    let calls = &text[text.find(ANSWER)? + ANSWER.len()..];
    let mut rest = calls;
    let mut gaps = Vec::new();
    for probe in
        [FIRST, TEXT_KEY, TEXT_VALUE, NUMBER_KEY, NUMBER_VALUE, SECOND, LAST_KEY, LAST_VALUE, THIRD]
    {
        let at = rest.find(probe)?;
        gaps.push(&rest[..at]);
        rest = &rest[at + probe.len()..];
    }
    let [lead, name_to_key, key_close, value_to_key, _, value_to_call, _, _, _] = gaps[..] else {
        return None;
    };

    // A piece that opens something starts at its tag.
    let param_open = &name_to_key[name_to_key.rfind('<')?..];
    let name_close = name_to_key.strip_suffix(param_open)?;
    let param_close = value_to_key.strip_suffix(param_open)?;
    // From the end of one call to the name of the next, and from the text
    // of the message to the name of the first: what both end with opens a
    // call, less the white space that separates.
    let call_to_call = value_to_call.strip_prefix(param_close)?;
    let shared = common_suffix(call_to_call, lead);
    let call_open = shared.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let before_calls = lead.strip_suffix(call_open)?;
    let closed = call_to_call.strip_suffix(call_open)?;
    let call_close = closed.trim_end_matches(|c: char| c.is_ascii_whitespace());
    let between_calls = &closed[call_close.len()..];

    let format = ToolFormat {
        call_open: call_open.to_owned(),
        name_close: name_close.to_owned(),
        param_open: param_open.to_owned(),
        key_close: key_close.to_owned(),
        param_close: param_close.to_owned(),
        call_close: call_close.to_owned(),
        before_calls: before_calls.to_owned(),
        between_calls: between_calls.to_owned(),
    };
    let pieces = [
        &format.call_open,
        &format.name_close,
        &format.param_open,
        &format.key_close,
        &format.param_close,
        &format.call_close,
    ];
    if pieces.iter().any(|piece| piece.is_empty()) || format.trigger().is_empty() {
        return None;
    }
    // The pieces must write the three calls exactly as the template did.
    calls.starts_with(&written(&format)).then_some(format)
}

/// The calls of the probe message as the pieces of `format` write them.
fn written(format: &ToolFormat) -> String {
    let mut text = format.before_calls.clone();
    let calls: [(&str, &[(&str, &str)]); 3] = [
        (FIRST, &[(TEXT_KEY, TEXT_VALUE), (NUMBER_KEY, NUMBER_VALUE)]),
        (SECOND, &[(LAST_KEY, LAST_VALUE)]),
        (THIRD, &[]),
    ];
    for (index, (name, arguments)) in calls.iter().enumerate() {
        if index > 0 {
            text.push_str(&format.between_calls);
        }
        text.push_str(&format.call_open);
        text.push_str(name);
        text.push_str(&format.name_close);
        for (key, value) in *arguments {
            text.push_str(&format.param_open);
            text.push_str(key);
            text.push_str(&format.key_close);
            text.push_str(value);
            text.push_str(&format.param_close);
        }
        text.push_str(&format.call_close);
    }
    text
}

/// A conversation in which the assistant calls three tools, rendered.
fn rendered(template: &ChatTemplate) -> Option<String> {
    let call = |name: &str, arguments: Json| ToolCall {
        id: Some(format!("call_{name}")),
        name: name.to_owned(),
        arguments: arguments.dump(),
    };
    let text = |value: &str| Json::String(value.to_owned());
    let mut assistant = Message::new("assistant", ANSWER);
    assistant.tool_calls = vec![
        call(
            FIRST,
            Json::Object(vec![
                (TEXT_KEY.to_owned(), text(TEXT_VALUE)),
                (NUMBER_KEY.to_owned(), Json::Int(7204)),
            ]),
        ),
        call(SECOND, Json::Object(vec![(LAST_KEY.to_owned(), text(LAST_VALUE))])),
        call(THIRD, Json::Object(Vec::new())),
    ];
    let mut messages = vec![Message::new("user", "Hi"), assistant];
    for name in [FIRST, SECOND, THIRD] {
        let mut result = Message::new("tool", "done");
        result.name = name.to_owned();
        result.tool_call_id = format!("call_{name}");
        messages.push(result);
    }
    let tools: Vec<Tool> = [FIRST, SECOND, THIRD]
        .iter()
        .map(|name| Tool {
            name: (*name).to_owned(),
            description: "A tool.".to_owned(),
            parameters: Json::Object(vec![("type".to_owned(), text("object"))]),
        })
        .collect();
    template
        .render(
            &messages,
            &tools,
            &ChatOptions { add_generation_prompt: false, ..ChatOptions::default() },
        )
        .ok()
}

/// The longest text both `a` and `b` end with.
fn common_suffix<'a>(a: &'a str, b: &str) -> &'a str {
    let same = a.bytes().rev().zip(b.bytes().rev()).take_while(|(x, y)| x == y).count();
    let mut at = a.len() - same;
    while !a.is_char_boundary(at) {
        at += 1;
    }
    &a[at..]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The part of Qwen3.8's template that writes an assistant message and the
    /// results of its calls.
    const TAGGED: &str = r#"{%- if tools %}{{- '<|im_start|>system\n<tools>' }}{%- for tool in tools %}{{- '\n' }}{{- tool | tojson }}{%- endfor %}{{- '\n</tools><|im_end|>\n' }}{%- endif %}
{%- for message in messages %}
{%- if message.role == 'assistant' %}
{{- '<|im_start|>assistant\n' + message.content }}
{%- for tool_call in message.tool_calls %}
{%- set tool_call = tool_call.function %}
{%- if loop.first %}{{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}{%- else %}{{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}{%- endif %}
{%- for args_name, args_value in tool_call.arguments|items %}
{{- '<parameter=' + args_name + '>\n' }}
{%- set args_value = args_value | string if args_value is string else args_value | tojson %}
{{- args_value }}
{{- '\n</parameter>\n' }}
{%- endfor %}
{{- '</function>\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- elif message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}"#;

    /// Hermes' form: a call is one JSON object between the tags.
    const AS_JSON: &str = r#"{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content }}
{%- for tool_call in message.tool_calls %}
{{- '\n<tool_call>\n{"name": "' + tool_call.function.name + '", "arguments": ' + (tool_call.function.arguments | tojson) + '}\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- endfor %}"#;

    fn template(source: &str) -> ChatTemplate {
        ChatTemplate::new(source, "", "").expect("the template compiles")
    }

    #[test]
    fn the_tags_of_a_call_are_read_off_the_template() {
        let format = detect(&template(TAGGED)).expect("a format");
        assert_eq!(
            format,
            ToolFormat {
                call_open: "<tool_call>\n<function=".into(),
                name_close: ">\n".into(),
                param_open: "<parameter=".into(),
                key_close: ">\n".into(),
                param_close: "\n</parameter>\n".into(),
                call_close: "</function>\n</tool_call>".into(),
                before_calls: "\n\n".into(),
                between_calls: "\n".into(),
            }
        );
        assert_eq!(format.trigger(), "<tool_call>");
    }

    #[test]
    fn a_template_that_writes_calls_as_json_or_not_at_all_has_no_format() {
        assert_eq!(detect(&template(AS_JSON)), None);
        let plain = "{%- for m in messages %}{{- m.role + ': ' + m.content + '\n' }}{%- endfor %}";
        assert_eq!(detect(&template(plain)), None);
    }

    #[test]
    fn other_tags_in_the_same_form_are_a_format_too() {
        let other = TAGGED
            .replace("tool_call>", "call>")
            .replace("<function=", "<fn=")
            .replace("</function>", "</fn>")
            .replace("parameter", "arg");
        let format = detect(&template(&other)).expect("a format");
        assert_eq!(format.call_open, "<call>\n<fn=");
        assert_eq!(format.param_close, "\n</arg>\n");
        assert_eq!(format.call_close, "</fn>\n</call>");
        assert_eq!(format.trigger(), "<call>");
    }
}
