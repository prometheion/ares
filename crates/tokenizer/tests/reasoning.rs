//! What the chat templates say about thinking: templates written for the tests,
//! with the structure of the templates of the Qwen3.5 family, and the real
//! templates of three model files.
//!
//! The real templates are read from the model files named by
//! `ARES_TEST_QWEN35_0_8B` (Qwen3.5-0.8B), `ARES_TEST_QWEN38_27B` (Qwen3.8-27B) and
//! `ARES_TEST_FLASH_NEXT` (Qwen3.8-Flash-Next, first part); a test says so and
//! passes when a file is not named. Reading the metadata of a file needs no GPU.

use ares_gguf::Gguf;
use ares_tokenizer::{ChatTemplate, ReasoningFormat, Tokenizer};

/// The structure of the Qwen3.5 template: the reasoning of an assistant turn
/// between think tags, trimmed, kept for earlier turns unless
/// `preserve_thinking` is off, a prompt that opens the block or closes an empty
/// one, and an instruction when the effort is low.
const QWEN_LIKE: &str = r#"
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- if reasoning_effort|default('xhigh') == 'low' %}
        {{- '<|im_start|>system\nThink briefly.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(last_query_index=0) %}
{%- for message in messages %}
    {%- if message.role == "user" %}{%- set ns.last_query_index = loop.index0 %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = message.content|trim %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning = (message.reasoning_content if message.reasoning_content is string else '')|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\n<think>\n' + reasoning + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>assistant\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"#;

fn template(source: &str) -> ChatTemplate {
    ChatTemplate::new(source, "", "").unwrap()
}

fn detect(source: &str) -> Option<ReasoningFormat> {
    ReasoningFormat::detect(&template(source))
}

#[test]
fn reads_the_tags_and_the_white_space_the_template_writes_around_the_reasoning() {
    let format = detect(QWEN_LIKE).unwrap();
    assert_eq!(format.start, "<think>\n");
    assert_eq!(format.end, "\n</think>\n\n");
    assert_eq!((format.start_tag(), format.end_tag()), ("<think>", "</think>"));
}

#[test]
fn sees_how_the_prompt_ends_with_thinking_on_and_off() {
    let format = detect(QWEN_LIKE).unwrap();
    assert_eq!(format.prompt_on, "<think>\n");
    assert_eq!(format.prompt_off, "<think>\n\n</think>\n\n");
    assert!(format.opens(), "with thinking on the model continues inside the block");
    assert!(format.switchable());
}

#[test]
fn a_prompt_that_does_not_mention_the_tags_leaves_the_opening_to_the_model() {
    // The tags are in the history only; the model writes the opening itself.
    let source = QWEN_LIKE
        .replace("{{- '<think>\\n' }}", "")
        .replace("{{- '<think>\\n\\n</think>\\n\\n' }}", "");
    let format = detect(&source).unwrap();
    assert_eq!((format.prompt_on.as_str(), format.prompt_off.as_str()), ("", ""));
    assert!(!format.opens());
    assert!(!format.switchable());
}

#[test]
fn a_switch_that_changes_nothing_in_the_prompt_is_no_switch() {
    let source =
        QWEN_LIKE.replace("enable_thinking is defined and enable_thinking is false", "false");
    let format = detect(&source).unwrap();
    assert_eq!(format.prompt_on, "<think>\n");
    assert_eq!(format.prompt_off, "<think>\n");
    assert!(!format.switchable());
}

#[test]
fn finds_out_whether_the_template_keeps_the_reasoning_of_earlier_turns() {
    assert!(detect(QWEN_LIKE).unwrap().keeps_history);
    // The reasoning of the turn that follows the last question only.
    let source = QWEN_LIKE.replace(
        "preserve_thinking is undefined or preserve_thinking is true or loop.index0",
        "loop.index0",
    );
    assert!(!detect(&source).unwrap().keeps_history);
}

#[test]
fn finds_out_whether_the_template_reads_the_effort() {
    assert!(detect(QWEN_LIKE).unwrap().effort);
    let source = QWEN_LIKE.replace("reasoning_effort|default('xhigh') == 'low'", "false");
    assert!(!detect(&source).unwrap().effort);
}

#[test]
fn an_effort_the_template_refuses_does_not_hide_the_ones_it_takes() {
    let source = QWEN_LIKE.replace(
        "{%- if reasoning_effort|default('xhigh') == 'low' %}",
        "{%- if reasoning_effort is defined and reasoning_effort not in ('low', 'medium') %}\
         {{- raise_exception('no such effort') }}{%- endif %}\
         {%- if reasoning_effort|default('xhigh') == 'low' %}",
    );
    let format = detect(&source).unwrap();
    assert!(format.effort);
    assert_eq!(format.refused_efforts, ["minimal", "high", "xhigh"]);
    // A template that does not read the effort refuses none.
    let deaf = QWEN_LIKE.replace("reasoning_effort|default('xhigh') == 'low'", "false");
    assert_eq!(detect(&deaf).unwrap().refused_efforts, Vec::<String>::new());
}

#[test]
fn square_brackets_make_a_pair_of_tags_too() {
    let source = QWEN_LIKE.replace("<think>", "[THINK]").replace("</think>", "[/THINK]");
    let format = detect(&source).unwrap();
    assert_eq!(format.start, "[THINK]\n");
    assert_eq!(format.end, "\n[/THINK]\n\n");
}

#[test]
fn tags_that_sit_flush_against_the_reasoning_have_no_white_space() {
    let source = QWEN_LIKE.replace(
        "<think>\\n' + reasoning + '\\n</think>\\n\\n'",
        "<think>' + reasoning + '</think>'",
    );
    let format = detect(&source).unwrap();
    assert_eq!((format.start.as_str(), format.end.as_str()), ("<think>", "</think>"));
}

#[test]
fn a_template_without_a_pair_of_tags_around_the_reasoning_has_no_thinking() {
    // ChatML: the reasoning is not written at all.
    let chatml = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
                  {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    assert_eq!(detect(chatml), None);
    // The reasoning is written, with an opening tag and no closing one.
    let open_only = chatml.replace(
        "{{ m.content }}",
        "{% if m.reasoning_content %}<think>{{ m.reasoning_content }} {% endif %}{{ m.content }}",
    );
    assert_eq!(detect(&open_only), None);
    // The reasoning is written without any tag.
    let bare = chatml.replace(
        "{{ m.content }}",
        "{% if m.reasoning_content %}{{ m.reasoning_content }}\n{% endif %}{{ m.content }}",
    );
    assert_eq!(detect(&bare), None);
}

#[test]
fn a_template_that_fails_on_a_probe_has_no_thinking_instead_of_an_error() {
    assert_eq!(detect("{{ raise_exception('no') }}"), None);
}

// The real templates.

fn tokenizer_of(variable: &str, what: &str) -> Option<Tokenizer> {
    let Some(path) = std::env::var_os(variable) else {
        eprintln!("skipped: set {variable} to the model file of {what}");
        return None;
    };
    Some(Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap())
}

/// What the templates of Qwen3.8-27B and Qwen3.8-Flash-Next declare: the
/// structure of `QWEN_LIKE`.
fn qwen_format() -> ReasoningFormat {
    ReasoningFormat {
        start: "<think>\n".into(),
        end: "\n</think>\n\n".into(),
        prompt_on: "<think>\n".into(),
        prompt_off: "<think>\n\n</think>\n\n".into(),
        keeps_history: true,
        effort: true,
        refused_efforts: Vec::new(),
    }
}

#[test]
fn the_template_of_qwen35_0_8b_has_the_same_tags_but_drops_the_thinking_of_earlier_turns() {
    let Some(tokenizer) = tokenizer_of("ARES_TEST_QWEN35_0_8B", "Qwen3.5-0.8B") else { return };
    // It is the first template of the family: the reasoning of a turn stays in the prompt
    // only after the last user message, whatever `preserve_reasoning` says, and the
    // template does not read an effort.
    let format = ReasoningFormat { keeps_history: false, effort: false, ..qwen_format() };
    assert_eq!(tokenizer.reasoning_format(), Some(format));
}

#[test]
fn the_template_of_qwen38_27b_declares_the_qwen_format() {
    let Some(tokenizer) = tokenizer_of("ARES_TEST_QWEN38_27B", "Qwen3.8-27B") else { return };
    // It takes low, medium, high and xhigh.
    let format = ReasoningFormat { refused_efforts: vec!["minimal".into()], ..qwen_format() };
    assert_eq!(tokenizer.reasoning_format(), Some(format));
}

#[test]
fn the_template_of_qwen38_flash_next_declares_the_qwen_format() {
    let Some(tokenizer) = tokenizer_of("ARES_TEST_FLASH_NEXT", "Qwen3.8-Flash-Next") else {
        return;
    };
    // It takes low, medium and xhigh, its default.
    let refused_efforts = vec!["minimal".into(), "high".into()];
    let format = ReasoningFormat { refused_efforts, ..qwen_format() };
    assert_eq!(tokenizer.reasoning_format(), Some(format));
}
