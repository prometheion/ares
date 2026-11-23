//! The round trip of a reply, through the real chat templates.
//!
//! A model generates tokens after the prompt. The server hands the client the
//! text of them in two parts, `reasoning_content` and `content`; the client
//! sends the two back in the next request; the chat template writes them into
//! the prompt and the prompt is tokenized again. The engine's prefix cache
//! continues at the end of the reply only if the tokens of that prompt start
//! with the prompt and the tokens the model generated.
//!
//! Each test here plays that through for a reply, with the template and the
//! vocabulary of a model file, and counts the tokens of the previous sequence
//! (prompt and reply) that the next prompt does not repeat: the tokens the
//! cache cannot give. The models say what they say in the tests; the engine is
//! not run. The model files are named by `ARES_TEST_GGUF` (Qwen3.5-0.8B),
//! `ARES_TEST_QWEN38_27B` (Qwen3.8-27B) and `ARES_TEST_FLASH_NEXT`
//! (Qwen3.8-Flash-Next, first part); a test says so and passes when the file
//! it needs is not named.

use ares_gguf::Gguf;
use ares_tokenizer::{ChatOptions, EncodeOptions, Message, Tokenizer};

use super::*;

/// A model file's tokenizer and template.
struct Rig {
    name: &'static str,
    tokenizer: Tokenizer,
}

/// The text of the tokens `ids` in the pieces the engine hands it on in: the
/// text of each token, with the start of a character held back until its
/// other bytes arrive. (What the tests generate is valid UTF-8.)
pub(crate) fn streamed(tokenizer: &Tokenizer, ids: &[u32]) -> Vec<String> {
    let mut held = Vec::new();
    ids.iter()
        .map(|&id| {
            held.extend_from_slice(tokenizer.piece(id, false).unwrap());
            let whole = match std::str::from_utf8(&held) {
                Ok(text) => text.len(),
                Err(error) => error.valid_up_to(),
            };
            String::from_utf8(held.drain(..whole).collect()).unwrap()
        })
        .collect()
}

fn rig(variable: &str, name: &'static str) -> Option<Rig> {
    let Some(path) = std::env::var_os(variable) else {
        eprintln!("skipped: set {variable} to the model file of {name}");
        return None;
    };
    let tokenizer = Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap();
    Some(Rig { name, tokenizer })
}

/// What a model says after the prompt, and how it ends.
struct Say<'a> {
    text: &'a str,
    /// Whether the model ends its turn (`<|im_end|>`) or is cut off by the
    /// limit of tokens.
    stops: bool,
}

fn stops(text: &str) -> Say<'_> {
    Say { text, stops: true }
}

fn cut(text: &str) -> Say<'_> {
    Say { text, stops: false }
}

/// What came of one reply.
struct Trip {
    /// The prompt and the tokens the model generated.
    sequence: Vec<u32>,
    /// How many of them are the prompt.
    prompt: usize,
    /// The next prompt.
    next: Vec<u32>,
    /// The reply as the client sends it back.
    reply: Message,
}

impl Trip {
    /// The leading tokens of the sequence that the next prompt repeats.
    fn kept(&self) -> usize {
        self.sequence.iter().zip(&self.next).take_while(|(a, b)| a == b).count()
    }

    /// The tokens of the sequence that the next prompt does not repeat.
    fn lost(&self) -> usize {
        self.sequence.len() - self.kept()
    }
}

impl Rig {
    fn options(&self, thinking: bool) -> ChatOptions {
        // The server's defaults: the thinking of earlier turns stays in the prompt.
        let mut variables =
            vec![("preserve_reasoning".to_owned(), ares_tokenizer::Json::Bool(true))];
        if !thinking {
            variables.push(("enable_thinking".to_owned(), ares_tokenizer::Json::Bool(false)));
        }
        ChatOptions { variables, ..ChatOptions::default() }
    }

    fn prompt(&self, messages: &[Message], thinking: bool) -> Vec<u32> {
        let text = self.tokenizer.render_chat(messages, &[], &self.options(thinking)).unwrap();
        self.tokenizer.encode(&text, EncodeOptions { add_special: true, parse_special: true })
    }

    /// What the server makes of the model's tokens: the splitter it builds for
    /// a request, fed the text of the tokens one after the other.
    fn sorted(&self, ids: &[u32], thinking: bool) -> Split {
        let format = ares_model::chat::chat_format(&self.tokenizer, &self.options(true))
            .thinking
            .expect("the template thinks");
        let kwargs = if thinking {
            Vec::new()
        } else {
            vec![(
                "chat_template_kwargs".to_owned(),
                Json::Object(vec![("enable_thinking".to_owned(), Json::Bool(false))]),
            )]
        };
        let asked = asked(&Fields(&kwargs)).unwrap();
        let mut splitter = plan(&asked, Some(&format), ReasoningFormat::Auto).unwrap().splitter;
        let mut sorted = Split::default();
        for text in streamed(&self.tokenizer, ids) {
            let piece = splitter.push(&text);
            sorted.reasoning.push_str(&piece.reasoning);
            sorted.content.push_str(&piece.content);
        }
        let last = splitter.finish();
        sorted.reasoning.push_str(&last.reasoning);
        sorted.content.push_str(&last.content);
        sorted
    }

    /// The model answers `history` (which ends with a user message) with `say`,
    /// the client sends the reply back with another question.
    fn trip(&self, history: &[Message], say: &Say<'_>, thinking: bool) -> Trip {
        let mut generated = self.tokens(say.text);
        if say.stops {
            generated.push(self.tokenizer.special_token_id("<|im_end|>").expect("a token"));
        }
        self.trip_of(history, generated, thinking)
    }

    /// The tokens of `text` as the tokenizer makes them.
    fn tokens(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, EncodeOptions { add_special: false, parse_special: true })
    }

    /// Like [`Rig::trip`], for a model that generated these tokens.
    fn trip_of(&self, history: &[Message], generated: Vec<u32>, thinking: bool) -> Trip {
        let mut sequence = self.prompt(history, thinking);
        let prompt = sequence.len();
        let sorted = self.sorted(&generated, thinking);
        sequence.extend(&generated);

        let mut reply = Message::new("assistant", sorted.content);
        reply.reasoning_content = sorted.reasoning;
        let mut messages = history.to_vec();
        messages.push(reply.clone());
        messages.push(Message::new("user", "And 3 + 3?"));
        Trip { next: self.prompt(&messages, thinking), sequence, prompt, reply }
    }

    /// The first question.
    fn question(&self) -> Vec<Message> {
        vec![Message::new("user", "What is 2 + 2?")]
    }

    /// Where the sequence and the next prompt part, for a failure to show.
    fn parting(&self, trip: &Trip) -> String {
        let at = trip.kept();
        let text = |ids: &[u32]| {
            let from = at.saturating_sub(3);
            let ids = &ids[from..ids.len().min(at + 4)];
            ids.iter()
                .map(|id| {
                    String::from_utf8_lossy(self.tokenizer.piece(*id, true).unwrap()).into_owned()
                })
                .collect::<Vec<_>>()
        };
        format!(
            "{}: parts at token {at} of {}: the sequence goes on {:?}, the prompt {:?}",
            self.name,
            trip.sequence.len(),
            text(&trip.sequence),
            text(&trip.next)
        )
    }

    /// The reply is repeated by the next prompt, all of it.
    fn assert_exact(&self, trip: &Trip) {
        assert_eq!(trip.lost(), 0, "{}", self.parting(trip));
    }
}

/// The rigs of the models whose template keeps the thinking of earlier turns.
fn preserving() -> Vec<Rig> {
    [rig("ARES_TEST_QWEN38_27B", "Qwen3.8-27B"), rig("ARES_TEST_FLASH_NEXT", "Qwen3.8-Flash-Next")]
        .into_iter()
        .flatten()
        .collect()
}

#[test]
fn the_models_template_declares_the_thinking_the_splitter_needs() {
    for rig in preserving().iter().chain(&rig("ARES_TEST_GGUF", "Qwen3.5-0.8B")) {
        let chat = ares_model::chat::chat_format(&rig.tokenizer, &rig.options(true));
        let format: ThinkingFormat = chat.thinking.expect(rig.name);
        assert_eq!(format.prompt_on, "<think>\n", "{}", rig.name);
        assert_eq!(format.prompt_off, "<think>\n\n</think>\n\n", "{}", rig.name);
        assert!(format.on_by_default && format.switchable(), "{}", rig.name);
    }
}

#[test]
fn a_reply_that_thinks_is_repeated_by_the_next_prompt() {
    for rig in preserving() {
        let say = stops("The user asks for a sum.\n2 and 2 are 4.\n</think>\n\nThe answer is 4.");
        let trip = rig.trip(&rig.question(), &say, true);
        assert_eq!(trip.reply.reasoning_content, "The user asks for a sum.\n2 and 2 are 4.");
        assert_eq!(trip.reply.content, ares_tokenizer::Content::Text("The answer is 4.".into()));
        rig.assert_exact(&trip);
        // The prompt that follows the reply starts the next turn.
        assert!(trip.next.len() > trip.sequence.len() + 5);
    }
}

#[test]
fn text_of_every_kind_is_repeated() {
    for rig in preserving() {
        for say in [
            // German with umlauts, a symbol outside the basic plane.
            "Zwei und zwei ist vier: \u{e4}\u{f6}\u{fc}\u{df} \u{1f600}\n</think>\n\nDie Antwort ist 4.",
            // Blank lines inside the thinking, lists, code.
            "First:\n\n1. add\n2. check\n\n```py\nprint(2 + 2)\n```\n</think>\n\nIt is 4.\n\nDone.",
            // Tags that are only text.
            "The tag <think> would open it, and </thinking> is not the end.\n</think>\n\nOK.",
            // A single word, and punctuation at the end.
            "Hmm.\n</think>\n\nFour!",
        ] {
            let trip = rig.trip(&rig.question(), &stops(say), true);
            rig.assert_exact(&trip);
        }
    }
}

#[test]
fn a_reply_without_thinking_is_repeated_when_the_template_closed_the_block() {
    for rig in preserving() {
        let trip = rig.trip(&rig.question(), &stops("The answer is 4."), false);
        assert_eq!(trip.reply.reasoning_content, "");
        rig.assert_exact(&trip);
    }
}

#[test]
fn a_reply_cut_by_the_limit_inside_the_answer_is_repeated() {
    for rig in preserving() {
        let trip = rig.trip(&rig.question(), &cut("Add.\n</think>\n\nThe answer is"), true);
        assert_eq!(trip.reply.content, ares_tokenizer::Content::Text("The answer is".into()));
        rig.assert_exact(&trip);
    }
}

#[test]
fn a_reply_cut_inside_the_thinking_is_repeated_up_to_what_the_template_adds() {
    for rig in preserving() {
        // After a word, where nothing merges with the newline the template writes.
        let trip = rig.trip(&rig.question(), &cut("The user asks for a sum and"), true);
        assert_eq!(trip.reply.reasoning_content, "The user asks for a sum and");
        assert_eq!(trip.reply.content, ares_tokenizer::Content::Text(String::new()));
        rig.assert_exact(&trip);
        // Also when the thinking is cut in the middle of a line.
        let trip = rig.trip(&rig.question(), &cut("Line one\nLine two is"), true);
        rig.assert_exact(&trip);
    }
}

#[test]
fn a_reply_cut_after_punctuation_is_repeated_too() {
    for rig in preserving() {
        // The tokenizer has no token for "." and a newline together, so the
        // newline the template writes after the reasoning is a token of its own.
        let trip = rig.trip(&rig.question(), &cut("The user asks for a sum."), true);
        rig.assert_exact(&trip);
    }
}

#[test]
fn tokens_that_are_not_the_tokenizers_own_for_their_text_part_where_they_differ() {
    for rig in preserving() {
        // A model that samples tokens may write a text with other tokens than the
        // tokenizer would make of it. Here the quotation mark goes with the question
        // mark and not with the dots; the text is the same, the next prompt has the
        // tokenizer's. A sampled reply of 1500 tokens from the 0.8B had exactly this
        // when the engine was tested through the server.
        let before = rig.tokens("Is it a whole number...");
        let after = rig.tokens("\"? Yes.\n</think>\n\nFour.");
        let whole = rig.tokens("Is it a whole number...\"? Yes.\n</think>\n\nFour.");
        let sampled: Vec<u32> = [&before[..], &after[..]].concat();
        assert_ne!(sampled, whole, "{}: the two tokenizations of one text", rig.name);
        let mut generated = sampled.clone();
        generated.push(rig.tokenizer.special_token_id("<|im_end|>").expect("a token"));
        let trip = rig.trip_of(&rig.question(), generated, true);
        assert_eq!(trip.reply.reasoning_content, "Is it a whole number...\"? Yes.");
        // The next prompt repeats the tokens up to the first one that differs, and
        // the rest of the reply is computed again.
        let same = sampled.iter().zip(&whole).take_while(|(a, b)| a == b).count();
        assert_eq!(trip.kept(), trip.prompt + same, "{}", rig.parting(&trip));
        assert_eq!(trip.lost(), sampled.len() + 1 - same);
    }
}

#[test]
fn a_model_that_thinks_of_nothing_parts_from_the_next_prompt_at_the_last_token_of_the_prompt() {
    for rig in preserving() {
        // The prompt ends with "<think>\n" and the model writes "\n</think>": for
        // the model that is two newline tokens, for the tokenizer, which gets the
        // text of the next prompt, "<think>\n\n</think>" has one token for both.
        // The template cannot help it: the reasoning is empty, and the layout
        // around it is the same.
        let trip = rig.trip(&rig.question(), &stops("\n</think>\n\nThe answer is 4."), true);
        assert_eq!(trip.reply.reasoning_content, "");
        assert_eq!(trip.kept(), trip.prompt - 1, "{}", rig.parting(&trip));
        assert_eq!(trip.lost(), 11, "the last token of the prompt and the whole reply");
    }
}

#[test]
fn white_space_beyond_the_layout_of_the_template_cannot_be_repeated() {
    // The template trims the reasoning and the content, and writes its own
    // layout around them. A model that writes other white space is not repeated
    // from the first token that differs. The numbers are the tokens of the
    // sequence from there on, and are the same for both templates.
    let cases = [
        // A newline after the answer: it and <|im_end|>.
        ("Line\n</think>\n\nAnswer.\n", 2),
        ("Line\n</think>\n\nAnswer.\n\n", 2),
        // A newline before the reasoning: all the reply (the prompt is repeated).
        ("\nLine\n</think>\n\nAnswer.", 8),
        // An empty line at the end of the reasoning: from there on.
        ("Line\n\n</think>\n\nAnswer.", 6),
        // An empty line before the answer, or only one newline.
        ("Line\n</think>\n\n\nAnswer.", 4),
        ("Line\n</think>\nAnswer.", 4),
        // No newline before the closing tag.
        ("Line</think>\n\nAnswer.", 5),
    ];
    for rig in preserving() {
        for (say, lost) in cases {
            let trip = rig.trip(&rig.question(), &stops(say), true);
            assert_eq!(trip.lost(), lost, "{say:?}: {}", rig.parting(&trip));
        }
    }
}

#[test]
fn several_turns_in_a_row_each_continue_at_the_end_of_the_one_before() {
    for rig in preserving() {
        let says = [
            (stops("Add the numbers.\n</think>\n\n2 + 2 is 4."), true),
            (stops("The answer is 6.\nCheck: 3 + 3.\n</think>\n\nIt is 6."), true),
            (stops("A plain answer, without thinking."), false),
            (cut("Hmm\n</think>\n\nThe last answer is"), true),
        ];
        let mut history = rig.question();
        let mut sequences: Vec<Vec<u32>> = Vec::new();
        for (turn, (say, thinking)) in says.iter().enumerate() {
            // A turn that switches thinking off is a prompt of its own; the others
            // continue the one before.
            let trip = rig.trip(&history, say, *thinking);
            rig.assert_exact(&trip);
            if let Some(before) = sequences.last()
                && *thinking == says[turn - 1].1
            {
                assert_eq!(
                    &trip.sequence[..before.len()],
                    &before[..],
                    "{}: turn {turn}",
                    rig.name
                );
            }
            history.push(trip.reply.clone());
            history.push(Message::new("user", "And 3 + 3?"));
            sequences.push(trip.sequence);
        }
    }
}

#[test]
fn a_client_that_sends_only_the_content_back_parts_where_the_reasoning_began() {
    for rig in preserving() {
        let say = stops("The user asks for a sum.\n2 and 2 are 4.\n</think>\n\nThe answer is 4.");
        let mut trip = rig.trip(&rig.question(), &say, true);
        // The same reply, with the reasoning left out.
        let mut messages = rig.question();
        messages.push(Message::new("assistant", "The answer is 4."));
        messages.push(Message::new("user", "And 3 + 3?"));
        trip.next = rig.prompt(&messages, true);
        // The template writes an empty block, "<think>\n\n</think>\n\n", where the
        // prompt had "<think>\n": the newline of the prompt is one token with the
        // template's, and everything the model generated is computed again.
        assert_eq!(trip.kept(), trip.prompt - 1, "{}", rig.parting(&trip));
        assert_eq!(trip.lost(), trip.sequence.len() - trip.prompt + 1);
    }
}

/// What Qwen3.5-0.8B's own template does, which is another one: it writes the
/// thinking of a turn into the prompt only after the last user message. Earlier
/// turns lose it, `preserve_reasoning` or not, so no reply is repeated, whatever
/// it is and whatever the client sends back.
#[test]
fn the_template_of_qwen35_0_8b_never_repeats_the_thinking_of_an_earlier_turn() {
    let Some(rig) = rig("ARES_TEST_GGUF", "Qwen3.5-0.8B") else { return };
    for say in [
        stops("The user asks for a sum.\n2 and 2 are 4.\n</think>\n\nThe answer is 4."),
        stops("\n</think>\n\nThe answer is 4."),
        cut("The user asks for a sum and"),
        cut("Add.\n</think>\n\nThe answer is"),
    ] {
        let trip = rig.trip(&rig.question(), &say, true);
        // The prompt ended with "assistant\n<think>\n"; the next one has "assistant\n"
        // and the answer: it parts after the newline that follows "assistant", and
        // the whole reply is computed again.
        assert_eq!(trip.kept(), trip.prompt - 2, "{}", rig.parting(&trip));
    }
    // With thinking off the prompt has the empty block, which the next one does not.
    let trip = rig.trip(&rig.question(), &stops("The answer is 4."), false);
    assert_eq!(trip.kept(), trip.prompt - 4, "{}", rig.parting(&trip));

    // The same in every turn of a conversation.
    let mut history = rig.question();
    for _ in 0..3 {
        let trip = rig.trip(&history, &stops("Add.\n</think>\n\n4."), true);
        assert_eq!(trip.kept(), trip.prompt - 2, "{}", rig.parting(&trip));
        history.push(trip.reply);
        history.push(Message::new("user", "And 3 + 3?"));
    }
}
