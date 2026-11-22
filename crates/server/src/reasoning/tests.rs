use super::*;

/// What the Qwen3.5 family declares.
fn qwen() -> ThinkingFormat {
    ThinkingFormat {
        start: "<think>\n".into(),
        end: "\n</think>\n\n".into(),
        prompt_on: "<think>\n".into(),
        prompt_off: "<think>\n\n</think>\n\n".into(),
        on_by_default: true,
        effort: true,
        refused_efforts: Vec::new(),
    }
}

/// A reply to a prompt that opened the block.
fn thinking() -> Splitter {
    let format = qwen();
    Splitter::extracting(&format, &format.prompt_on)
}

/// The whole text through a splitter in one piece.
fn whole(mut splitter: Splitter, text: &str) -> (String, String) {
    let mut sorted = splitter.push(text);
    let rest = splitter.finish();
    sorted.reasoning.push_str(&rest.reasoning);
    sorted.content.push_str(&rest.content);
    (sorted.reasoning, sorted.content)
}

/// The text through a splitter in the given pieces, and what each piece gave.
fn pieces(mut splitter: Splitter, parts: &[&str]) -> (Vec<Split>, Split) {
    let given = parts.iter().map(|part| splitter.push(part)).collect();
    (given, splitter.finish())
}

fn pair(reasoning: &str, content: &str) -> (String, String) {
    (reasoning.to_owned(), content.to_owned())
}

#[test]
fn a_reply_is_split_into_the_reasoning_and_the_content() {
    assert_eq!(
        whole(thinking(), "I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?"),
        pair("I'm\nthinking", "Hello, world!\nWhat's up?")
    );
}

#[test]
fn the_text_is_the_same_however_it_is_cut() {
    let texts = [
        "I'm\nthinking\n</think>\n\nHello, world!",
        "Zwei \u{e4}pfel, \u{20ac}3, \u{1f600}\n</think>\n\n\u{fc}ber alles",
        "no end of the thinking",
        "short\n</think>",
        "short\n</think>\n",
        "x</think>y",
        "\n\n  \n</think>\n\nanswer",
        "a\n\n</think>\n\n\nb",
        "a<think>b</think>c\n</think>\n\nd",
        "",
    ];
    for text in texts {
        let expected = whole(thinking(), text);
        let boundaries: Vec<usize> =
            text.char_indices().map(|(at, _)| at).chain([text.len()]).collect();
        for &first in &boundaries {
            for &second in &boundaries {
                if second < first {
                    continue;
                }
                let (given, last) =
                    pieces(thinking(), &[&text[..first], &text[first..second], &text[second..]]);
                let reasoning: String =
                    given.iter().chain([&last]).map(|s| s.reasoning.as_str()).collect();
                let content: String =
                    given.iter().chain([&last]).map(|s| s.content.as_str()).collect();
                assert_eq!((reasoning, content), expected, "{text:?} cut at {first} and {second}");
            }
        }
        // One character at a time.
        let characters: Vec<String> = text.chars().map(String::from).collect();
        let parts: Vec<&str> = characters.iter().map(String::as_str).collect();
        let (given, last) = pieces(thinking(), &parts);
        let reasoning: String = given.iter().chain([&last]).map(|s| s.reasoning.as_str()).collect();
        let content: String = given.iter().chain([&last]).map(|s| s.content.as_str()).collect();
        assert_eq!((reasoning, content), expected, "{text:?} one character at a time");
    }
}

#[test]
fn the_reasoning_streams_as_it_comes() {
    let (given, last) =
        pieces(thinking(), &["Let me ", "think", " about it", "\n</think>\n\n", "Done."]);
    let reasoning: Vec<&str> = given.iter().map(|s| s.reasoning.as_str()).collect();
    assert_eq!(reasoning, ["Let me ", "think", " about it", "", ""]);
    assert_eq!(given[4].content, "Done.");
    assert!(last.is_empty());
}

#[test]
fn what_may_be_the_closing_tag_is_held_back_until_it_is_known() {
    let (given, last) = pieces(thinking(), &["abc\n</th", "ink>", "\n\nrest"]);
    assert_eq!(given[0], Split { reasoning: "abc".into(), content: String::new() });
    assert_eq!(
        given[1],
        Split::default(),
        "the tag is closed, the white space after it is not known"
    );
    assert_eq!(given[2], Split { reasoning: String::new(), content: "rest".into() });
    assert!(last.is_empty());

    // It was not the tag after all: the held text is reasoning.
    let (given, last) = pieces(thinking(), &["abc\n</th", "ing is a word"]);
    assert_eq!(given[0].reasoning, "abc");
    assert_eq!(given[1].reasoning, "\n</thing is a word");
    assert!(last.is_empty());
}

#[test]
fn the_newline_before_the_closing_tag_is_the_templates_and_not_the_reasoning() {
    assert_eq!(whole(thinking(), "abc\n</think>\n\nx"), pair("abc", "x"));
    // Only the one the template writes: another is the model's.
    assert_eq!(whole(thinking(), "abc\n\n</think>\n\nx"), pair("abc\n", "x"));
    // A model that writes the tag without it.
    assert_eq!(whole(thinking(), "abc</think>\n\nx"), pair("abc", "x"));
}

#[test]
fn only_the_white_space_the_template_writes_after_the_tag_is_taken_off_the_content() {
    assert_eq!(whole(thinking(), "r\n</think>\n\nx"), pair("r", "x"));
    assert_eq!(whole(thinking(), "r\n</think>\nx"), pair("r", "x"));
    assert_eq!(whole(thinking(), "r\n</think>x"), pair("r", "x"));
    assert_eq!(whole(thinking(), "r\n</think>\n\n\nx"), pair("r", "\nx"));
    assert_eq!(whole(thinking(), "r\n</think>\n\n  x"), pair("r", "  x"));
}

#[test]
fn a_reply_that_ends_inside_the_reasoning_has_the_reasoning_and_no_content() {
    assert_eq!(whole(thinking(), "I was still thinking"), pair("I was still thinking", ""));
    // What was held back as a possible tag is text now.
    assert_eq!(whole(thinking(), "still thinking\n"), pair("still thinking\n", ""));
    assert_eq!(whole(thinking(), "still thinking</thi"), pair("still thinking</thi", ""));
    assert_eq!(whole(thinking(), "still thinking\n</think"), pair("still thinking\n</think", ""));
}

#[test]
fn a_reply_that_ends_right_after_the_closing_tag_has_no_content() {
    assert_eq!(whole(thinking(), "done\n</think>"), pair("done", ""));
    assert_eq!(whole(thinking(), "done\n</think>\n\n"), pair("done", ""));
}

#[test]
fn a_model_that_does_not_think_has_empty_reasoning() {
    assert_eq!(whole(thinking(), "\n</think>\n\nanswer"), pair("", "answer"));
    assert_eq!(whole(thinking(), "</think>\n\nanswer"), pair("", "answer"));
    // White space alone is not reasoning.
    assert_eq!(whole(thinking(), "  \n\n</think>\n\nanswer"), pair("", "answer"));
    assert_eq!(whole(thinking(), " \n"), pair("", ""));
    // White space before the reasoning is kept, and comes with it.
    assert_eq!(whole(thinking(), "\n\nword\n</think>\n\nanswer"), pair("\n\nword", "answer"));
}

#[test]
fn a_prompt_that_closed_the_block_leaves_the_whole_reply_to_the_content() {
    let format = qwen();
    let off = || Splitter::extracting(&format, &format.prompt_off);
    assert_eq!(whole(off(), "Hello there"), pair("", "Hello there"));
    assert_eq!(whole(off(), "<th is not a tag"), pair("", "<th is not a tag"));
    assert_eq!(whole(off(), ""), pair("", ""));
    // A model that starts a block of its own anyway has one.
    assert_eq!(whole(off(), "<think>\nhm\n</think>\n\nyes"), pair("hm", "yes"));
    // The start of the tag waits for the rest of it.
    let (given, last) = pieces(off(), &["<thi", "nk>\nhm", "\n</think>\n\nyes"]);
    assert_eq!(given[0], Split::default());
    assert_eq!(given[1].reasoning, "hm");
    assert_eq!(given[2].content, "yes");
    assert!(last.is_empty());
}

#[test]
fn a_template_that_leaves_the_opening_to_the_model_gives_the_same_split() {
    let format = ThinkingFormat { prompt_on: String::new(), prompt_off: String::new(), ..qwen() };
    let leads = || Splitter::extracting(&format, "");
    assert!(format.prompt_on.is_empty());
    assert_eq!(whole(leads(), "<think>\nabc\n</think>\n\ndef"), pair("abc", "def"));
    assert_eq!(whole(leads(), "plain answer"), pair("", "plain answer"));
    // The tag is told from other text only when it is complete.
    let (given, _) = pieces(leads(), &["<th", "ello"]);
    assert_eq!(given[0], Split::default());
    assert_eq!(given[1].content, "<thello");
}

#[test]
fn square_brackets_and_other_layouts_are_read_from_the_declaration() {
    let format = ThinkingFormat {
        start: "[THINK]".into(),
        end: "[/THINK]".into(),
        prompt_on: String::new(),
        prompt_off: String::new(),
        ..qwen()
    };
    let splitter = || Splitter::extracting(&format, "");
    assert_eq!(whole(splitter(), "[THINK]abc\n[/THINK]\n\ndef"), pair("abc\n", "\n\ndef"));
    // The tags without white space: what the model writes around them is its own.
    assert_eq!(whole(splitter(), "[THINK] abc [/THINK]def"), pair(" abc ", "def"));
}

#[test]
fn the_raw_text_comes_with_the_end_of_the_prompt() {
    let format = qwen();
    assert_eq!(
        whole(Splitter::raw(&format.prompt_on), "I'm\nthinking\n</think>\n\nHello"),
        pair("", "<think>\nI'm\nthinking\n</think>\n\nHello")
    );
    assert_eq!(
        whole(Splitter::raw(&format.prompt_off), "Hello"),
        pair("", "<think>\n\n</think>\n\nHello")
    );
    // The prompt comes once, and also with an empty reply.
    let (given, last) = pieces(Splitter::raw("<think>\n"), &["a", "b"]);
    assert_eq!((given[0].content.as_str(), given[1].content.as_str()), ("<think>\na", "b"));
    assert!(last.is_empty());
    assert_eq!(whole(Splitter::raw("<think>\n"), ""), pair("", "<think>\n"));
}

#[test]
fn a_plain_splitter_passes_everything_as_content() {
    assert_eq!(whole(Splitter::plain(), "a\n</think>\n\nb"), pair("", "a\n</think>\n\nb"));
}

// What a request asks.

// A reply that may call tools: the marker of a call ends the reasoning too.

const MARKER: &str = "<tool_call>";

/// A reply to a prompt that opened the block, for a request whose tools may
/// be called.
fn calling() -> Splitter {
    thinking().ends_at(MARKER)
}

#[test]
fn the_marker_of_a_call_ends_the_reasoning_and_is_the_first_of_the_content() {
    let call = "<tool_call>\n<function=now>\n</function>\n</tool_call>";
    // The white space between the reasoning and the marker is neither's.
    for space in ["", " ", "\n", "\n\n", " \n\t\n "] {
        assert_eq!(
            whole(calling(), &format!("I look it up.{space}{call}")),
            pair("I look it up.", call),
            "{space:?}"
        );
    }
    // A reply that opens the call at once has no reasoning.
    assert_eq!(whole(calling(), call), pair("", call));
    assert_eq!(whole(calling(), &format!("\n \n{call}")), pair("", call));
    // What follows the marker is content, the closing tag too: the
    // reasoning is over.
    assert_eq!(
        whole(calling(), "one<tool_call> two\n</think>\n\nthree"),
        pair("one", "<tool_call> two\n</think>\n\nthree")
    );
    assert_eq!(
        whole(calling(), "one<tool_call><tool_call>"),
        pair("one", "<tool_call><tool_call>")
    );
}

#[test]
fn a_marker_after_the_closing_tag_is_content_as_before() {
    for text in [
        "I look.\n</think>\n\n<tool_call>\nx",
        "I look.\n</think><tool_call>\nx",
        "I look.\n</think>\n\nSure.\n\n<tool_call>\nx",
        "I look. <tool_cal\n</think>\n\nSure. <tool_call>",
        "no marker at all\n</think>\n\nanswer",
        "cut inside the thinking \n",
        "",
    ] {
        assert_eq!(whole(calling(), text), whole(thinking(), text), "{text:?}");
    }
    // What only begins like the marker is reasoning.
    assert_eq!(
        whole(calling(), "a <tool_cal b <tool_call\n</think>\n\nc"),
        pair("a <tool_cal b <tool_call", "c")
    );
    // At the end of a reply too, with the white space before it.
    assert_eq!(whole(calling(), "cut \n<tool_ca"), pair("cut \n<tool_ca", ""));
    assert_eq!(whole(calling(), "cut \n\n"), pair("cut \n\n", ""));
}

#[test]
fn the_marker_ends_the_reasoning_however_the_text_is_cut() {
    let texts = [
        "I look it up.\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>",
        "I look.<tool_call>\nrest\n</think>\n\nmore",
        "<tool_call>\nx",
        " \n <tool_call>x",
        "Zwei \u{e4}pfel \u{1f600} \n<tool_call>\n\u{fc}",
        "I look.\n</think>\n\n<tool_call>\nx",
        "I look.\n</think><tool_call>x",
        "I look.\n</think>\n\nSure.\n\n<tool_call>x",
        "I look. <tool_cal\n</think>\n\nSure.<tool_call>y",
        "a < b <t <tool <tool_call\n<tool_call>z",
        "ends in white space \n\n",
        "ends in <tool_c",
    ];
    for text in texts {
        let expected = whole(calling(), text);
        let boundaries: Vec<usize> =
            text.char_indices().map(|(at, _)| at).chain([text.len()]).collect();
        let check = |parts: &[&str]| {
            let (given, last) = pieces(calling(), parts);
            let mut marked = false;
            let (mut reasoning, mut content) = (String::new(), String::new());
            for split in given.iter().chain([&last]) {
                // Once content has come, no more reasoning does.
                assert!(content.is_empty() || split.reasoning.is_empty(), "{text:?} {parts:?}");
                reasoning.push_str(&split.reasoning);
                content.push_str(&split.content);
                marked |= split.reasoning.contains(MARKER);
            }
            assert!(!marked, "the marker in the reasoning of {text:?} as {parts:?}");
            assert_eq!((reasoning, content), expected, "{text:?} as {parts:?}");
        };
        // Three pieces, cut at every pair of positions: the marker is split
        // at every byte, and so is the white space before it.
        for &first in &boundaries {
            for &second in boundaries.iter().filter(|second| **second >= first) {
                check(&[&text[..first], &text[first..second], &text[second..]]);
            }
        }
        // One character at a time.
        let characters: Vec<&str> =
            boundaries.windows(2).map(|pair| &text[pair[0]..pair[1]]).collect();
        check(&characters);
    }
}

#[test]
fn the_reasoning_before_a_marker_streams_but_for_the_white_space_at_its_end() {
    let (given, last) = pieces(calling(), &["First ", "second.\n", "\n", "<tool", "_call>", "\nx"]);
    let reasoning: Vec<&str> = given.iter().map(|s| s.reasoning.as_str()).collect();
    assert_eq!(reasoning, ["First", " second.", "", "", "", ""]);
    let content: Vec<&str> = given.iter().map(|s| s.content.as_str()).collect();
    assert_eq!(content, ["", "", "", "", "<tool_call>", "\nx"]);
    assert!(last.is_empty());
    // White space that no marker follows is the reasoning's, a piece later.
    let (given, _) = pieces(calling(), &["One. \n", "Two.<to", "ol ", "x"]);
    let reasoning: Vec<&str> = given.iter().map(|s| s.reasoning.as_str()).collect();
    assert_eq!(reasoning, ["One.", " \nTwo.", "<tool", " x"]);
}

#[test]
fn without_a_marker_the_text_of_one_is_reasoning() {
    // A request without tools, or with `tool_choice: none`.
    let text = "I could write\n\n<tool_call>\n<function=now>\n</function>\n</tool_call> here.\n</think>\n\nSure.";
    assert_eq!(
        whole(thinking(), text),
        pair(
            "I could write\n\n<tool_call>\n<function=now>\n</function>\n</tool_call> here.",
            "Sure."
        )
    );
    // And white space streams as it comes.
    let (given, _) = pieces(thinking(), &["One. ", "Two.\n\n", "x"]);
    let reasoning: Vec<&str> = given.iter().map(|s| s.reasoning.as_str()).collect();
    assert_eq!(reasoning, ["One. ", "Two.\n", "\nx"]);
    // An empty marker ends nothing.
    assert_eq!(whole(thinking().ends_at(""), text), whole(thinking(), text));
}

#[test]
fn the_marker_ends_the_reasoning_a_reply_opened_itself_and_changes_no_other_reply() {
    let lead = ThinkingFormat { prompt_on: String::new(), ..qwen() };
    let opened = || Splitter::extracting(&lead, "").ends_at(MARKER);
    assert_eq!(whole(opened(), "<think>\nso\n<tool_call>x"), pair("so", "<tool_call>x"));
    assert_eq!(whole(opened(), "<tool_call>x"), pair("", "<tool_call>x"));
    assert_eq!(whole(opened(), "Sure. <tool_call>x"), pair("", "Sure. <tool_call>x"));
    // Thinking off, a raw reply, a model that does not think: all content.
    let format = qwen();
    let off = Splitter::extracting(&format, &format.prompt_off).ends_at(MARKER);
    assert_eq!(whole(off, "Sure. \n<tool_call>x"), pair("", "Sure. \n<tool_call>x"));
    let raw = Splitter::raw("<think>\n").ends_at(MARKER);
    assert_eq!(whole(raw, "so \n<tool_call>x"), pair("", "<think>\nso \n<tool_call>x"));
    assert_eq!(
        whole(Splitter::plain().ends_at(MARKER), "a \n<tool_call>"),
        pair("", "a \n<tool_call>")
    );
}

fn object(members: &[(&str, Json)]) -> Vec<(String, Json)> {
    members.iter().map(|(name, value)| ((*name).to_owned(), value.clone())).collect()
}

fn asked_from(members: &[(&str, Json)]) -> Result<Asked, ApiError> {
    asked(&Fields(&object(members)))
}

fn text(value: &str) -> Json {
    Json::String(value.to_owned())
}

fn refused(members: &[(&str, Json)]) -> String {
    let error = asked_from(members).unwrap_err();
    error.message
}

#[test]
fn a_request_without_the_fields_asks_nothing() {
    assert_eq!(asked_from(&[]).unwrap(), Asked::default());
    // `null` is not set.
    assert_eq!(
        asked_from(&[("reasoning_effort", Json::Null), ("chat_template_kwargs", Json::Null)])
            .unwrap(),
        Asked::default()
    );
}

#[test]
fn the_fields_of_llama_server_are_read() {
    let asked = asked_from(&[
        ("reasoning_format", text("none")),
        ("reasoning_effort", text("low")),
        (
            "chat_template_kwargs",
            Json::Object(object(&[
                ("enable_thinking", Json::Bool(false)),
                ("preserve_reasoning", Json::Bool(true)),
            ])),
        ),
    ])
    .unwrap();
    assert_eq!(asked.format, Some(ReasoningFormat::None));
    assert_eq!(asked.effort.as_deref(), Some("low"));
    assert_eq!(asked.variables.len(), 2);
    for (name, format) in [
        ("auto", ReasoningFormat::Auto),
        ("deepseek", ReasoningFormat::Deepseek),
        ("deepseek-legacy", ReasoningFormat::DeepseekLegacy),
    ] {
        assert_eq!(asked_from(&[("reasoning_format", text(name))]).unwrap().format, Some(format));
    }
}

#[test]
fn what_is_wrong_in_the_fields_is_told_by_name() {
    assert!(refused(&[("reasoning_format", text("think"))]).contains("none, auto, deepseek"));
    assert!(refused(&[("reasoning_format", Json::Int(1))]).contains("reasoning_format must be"));
    assert!(refused(&[("chat_template_kwargs", text("x"))]).contains("chat_template_kwargs"));
    let kwargs = |name: &str, value: Json| {
        refused(&[("chat_template_kwargs", Json::Object(object(&[(name, value)])))])
    };
    assert!(kwargs("enable_thinking", text("false")).contains("must be a boolean"));
    for name in ["messages", "tools", "add_generation_prompt", "bos_token", "eos_token"] {
        assert!(kwargs(name, Json::Bool(true)).contains("set by the server"), "{name}");
    }
}

#[test]
fn a_budget_for_the_thinking_is_refused_by_name() {
    for name in ["reasoning_budget_tokens", "thinking_budget_tokens"] {
        assert!(refused(&[(name, Json::Int(0))]).contains(name));
        assert!(refused(&[(name, Json::Int(1024))]).contains("budget"));
        assert!(refused(&[(name, text("many"))]).contains("must be a number"));
        // No budget is no refusal.
        assert!(asked_from(&[(name, Json::Int(-1))]).is_ok());
        assert!(asked_from(&[(name, Json::Float(-1.0))]).is_ok());
    }
    assert!(refused(&[("reasoning_budget_message", text("Enough."))]).contains("not supported"));
    assert!(asked_from(&[("reasoning_budget_message", text(""))]).is_ok());
    assert!(refused(&[("reasoning_control", Json::Bool(true))]).contains("reasoning_control"));
    assert!(asked_from(&[("reasoning_control", Json::Bool(false))]).is_ok());
}

// What it comes to.

fn planned(members: &[(&str, Json)], thinking: Option<&ThinkingFormat>) -> Result<Plan, ApiError> {
    plan(&asked_from(members)?, thinking, ReasoningFormat::Auto)
}

fn variable<'a>(plan: &'a Plan, name: &str) -> Option<&'a Json> {
    plan.variables.iter().find(|(known, _)| known == name).map(|(_, value)| value)
}

fn sorted(mut plan: Plan, text: &str) -> (String, String) {
    let mut split = plan.splitter.push(text);
    let rest = plan.splitter.finish();
    split.reasoning.push_str(&rest.reasoning);
    split.content.push_str(&rest.content);
    (split.reasoning, split.content)
}

#[test]
fn a_request_that_says_nothing_gets_the_templates_own_way() {
    let format = qwen();
    let plan = planned(&[], Some(&format)).unwrap();
    assert!(plan.variables.is_empty());
    assert_eq!(sorted(plan, "r\n</think>\n\nanswer"), pair("r", "answer"));
}

#[test]
fn thinking_off_in_the_request_closes_the_block_and_goes_to_the_template() {
    let format = qwen();
    let kwargs = Json::Object(object(&[("enable_thinking", Json::Bool(false))]));
    let plan = planned(&[("chat_template_kwargs", kwargs)], Some(&format)).unwrap();
    assert_eq!(variable(&plan, "enable_thinking"), Some(&Json::Bool(false)));
    // The prompt closed the empty block: the reply is the content.
    assert_eq!(sorted(plan, "Hello"), pair("", "Hello"));

    // `reasoning_effort: none` is the same.
    let plan = planned(&[("reasoning_effort", text("none"))], Some(&format)).unwrap();
    assert_eq!(variable(&plan, "enable_thinking"), Some(&Json::Bool(false)));
    assert_eq!(variable(&plan, "reasoning_effort"), None);
    assert_eq!(sorted(plan, "Hello"), pair("", "Hello"));
}

#[test]
fn a_model_that_does_not_think_by_default_is_not_taken_to_think() {
    let format = ThinkingFormat { on_by_default: false, ..qwen() };
    let plan = planned(&[], Some(&format)).unwrap();
    assert_eq!(sorted(plan, "Hello"), pair("", "Hello"));
    // A request can turn it on.
    let kwargs = Json::Object(object(&[("enable_thinking", Json::Bool(true))]));
    let plan = planned(&[("chat_template_kwargs", kwargs)], Some(&format)).unwrap();
    assert_eq!(sorted(plan, "r\n</think>\n\nanswer"), pair("r", "answer"));
}

#[test]
fn the_effort_goes_to_the_template_when_it_reads_it() {
    let format = qwen();
    let plan = planned(&[("reasoning_effort", text("low"))], Some(&format)).unwrap();
    assert_eq!(variable(&plan, "reasoning_effort"), Some(&text("low")));
    assert_eq!(variable(&plan, "enable_thinking"), None);

    let deaf = ThinkingFormat { effort: false, ..qwen() };
    let error = planned(&[("reasoning_effort", text("low"))], Some(&deaf)).unwrap_err();
    assert_eq!(error.param.as_deref(), Some("reasoning_effort"));
    let error = planned(&[("reasoning_effort", text("low"))], None).unwrap_err();
    assert!(error.message.contains("does not use it"), "{}", error.message);
}

#[test]
fn a_switch_off_that_the_template_ignores_is_refused() {
    let stuck = ThinkingFormat { prompt_off: "<think>\n".into(), ..qwen() };
    let error = planned(&[("reasoning_effort", text("none"))], Some(&stuck)).unwrap_err();
    assert!(error.message.contains("cannot switch"), "{}", error.message);
    // A model that does not think is off already.
    assert!(planned(&[("reasoning_effort", text("none"))], None).is_ok());
}

#[test]
fn the_format_of_the_request_wins_over_the_one_the_server_started_with() {
    let format = qwen();
    let default = ReasoningFormat::Auto;
    let request = asked_from(&[("reasoning_format", text("none"))]).unwrap();
    let raw = plan(&request, Some(&format), default).unwrap();
    assert_eq!(sorted(raw, "r\n</think>\n\na"), pair("", "<think>\nr\n</think>\n\na"));
    // And the other way round.
    let split = plan(&Asked::default(), Some(&format), ReasoningFormat::None).unwrap();
    assert_eq!(sorted(split, "r\n</think>\n\na"), pair("", "<think>\nr\n</think>\n\na"));
    let request = asked_from(&[("reasoning_format", text("deepseek"))]).unwrap();
    let split = plan(&request, Some(&format), ReasoningFormat::None).unwrap();
    assert_eq!(sorted(split, "r\n</think>\n\na"), pair("r", "a"));
}

#[test]
fn a_model_without_thinking_returns_its_text_whole() {
    let plan = planned(&[("reasoning_format", text("deepseek"))], None).unwrap();
    assert_eq!(sorted(plan, "a\n</think>\n\nb"), pair("", "a\n</think>\n\nb"));
}
