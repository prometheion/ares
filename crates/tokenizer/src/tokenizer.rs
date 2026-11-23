use std::cell::RefCell;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use ares_gguf::Gguf;

use crate::bpe::encode_word;
use crate::cache::{MAX_WORD, WordCache};
use crate::chat::{ChatError, ChatOptions, ChatTemplate, Message, Tool};
use crate::error::{Error, Result};
use crate::pretok;
use crate::reasoning::ReasoningFormat;
use crate::tools::ToolFormat;
use crate::unicode::{classes, sanitize};
use crate::vocab::{Pre, SpecialTokens, Vocab, attr};

/// How [`Tokenizer::encode`] treats the text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Add the beginning-of-text and end-of-text tokens that the model asks for.
    /// Qwen models ask for neither.
    pub add_special: bool,
    /// Turn the text of control tokens (`<|im_start|>`, ...) into the tokens
    /// themselves. A chat template needs this; the text of a user never should.
    ///
    /// Tokens the file calls user defined (`<think>`, `<tool_call>`, ...) are
    /// turned into their tokens in either case, as llama.cpp does.
    pub parse_special: bool,
}

/// What a token is.
///
/// A token can be listed as normal in the file and still be an end-of-text
/// marker by name; such a token is reported as [`Control`](Self::Control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    /// Text pieces: what BPE builds.
    Normal,
    /// Structure of the conversation (`<|im_start|>`, `<|endoftext|>`), which
    /// stays out of the text unless it is asked for.
    Control,
    /// Tokens the text can spell out, like `<think>`; they are written into the
    /// text as they are.
    UserDefined,
    /// The token for what the vocabulary cannot write.
    Unknown,
    /// A slot of the vocabulary that no text uses.
    Unused,
    /// A token the file gave a type that the format does not define.
    Undefined,
}

/// Texts shorter than this are encoded without the cache of words.
const CACHE_FROM: usize = 2048;

thread_local! {
    /// The words this thread has seen. A thread that encodes long text gets one,
    /// 256 KiB, and keeps it.
    static CACHE: RefCell<Option<WordCache>> = const { RefCell::new(None) };
}

/// A token whose text the encoder finds in the input.
struct Marker {
    id: u32,
    text: Box<[u8]>,
}

/// A piece of the input: text for BPE, or a token that was spelled out.
enum Piece {
    Text(Range<usize>),
    Token(u32),
}

/// Turns text into token ids and back, as llama.cpp does for the model it was
/// built from.
///
/// It is immutable once built and can be shared between threads.
#[derive(Debug)]
pub struct Tokenizer {
    /// Tells caches of words apart; one per tokenizer built.
    id: u64,
    vocab: Vocab,
    pre: Pre,
    /// The tokens a text can spell out, longest text first, in the order in which
    /// they cut the text. With `parse_special` all of them; without, only the user
    /// defined ones.
    markers_all: Vec<Marker>,
    markers_user: Vec<Marker>,
    /// The chat templates of the model file: the default one, and the one for
    /// conversations with tools if the file has it. A template that cannot be
    /// used keeps its error until someone renders a chat.
    chat: Option<std::result::Result<ChatTemplate, ChatError>>,
    chat_tool_use: Option<std::result::Result<ChatTemplate, ChatError>>,
}

impl std::fmt::Debug for Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:?}", self.id, String::from_utf8_lossy(&self.text))
    }
}

impl Tokenizer {
    /// Builds the tokenizer of a model from its `tokenizer.*` metadata.
    ///
    /// # Errors
    ///
    /// The file has no vocabulary, or one this engine does not implement: a
    /// `tokenizer.ggml.model` other than `gpt2`, a `tokenizer.ggml.pre` other
    /// than `qwen35`, a vocabulary with byte tokens, or merges that do not
    /// produce tokens. The error names what it found.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let (vocab, pre) = Vocab::load(gguf)?;
        // The order in which llama.cpp cuts: the longest text first. Among texts
        // of equal length it is not defined; the lower id goes first here.
        let mut ids: Vec<u32> = (0..vocab.len() as u32)
            .filter(|id| vocab.attrs[*id as usize] & attr::SPECIAL != 0)
            .collect();
        ids.sort_by_key(|id| std::cmp::Reverse(vocab.texts.get(*id as usize).len()));
        let marker = |id: &u32| Marker { id: *id, text: vocab.texts.get(*id as usize).into() };
        let markers_all = ids.iter().map(marker).collect();
        let markers_user = ids
            .iter()
            .filter(|id| vocab.attrs[**id as usize] & (attr::CONTROL | attr::UNKNOWN) == 0)
            .map(marker)
            .collect();
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let mut tokenizer = Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            vocab,
            pre,
            markers_all,
            markers_user,
            chat: None,
            chat_tool_use: None,
        };
        let text_of = |id: Option<u32>| {
            id.and_then(|id| tokenizer.piece(id, true).ok())
                .map(|piece| String::from_utf8_lossy(piece).into_owned())
                .unwrap_or_default()
        };
        let (bos, eos) =
            (text_of(tokenizer.vocab.special.bos), text_of(tokenizer.vocab.special.eos));
        let compile = |key: &str| gguf.str(key).map(|source| ChatTemplate::new(source, &bos, &eos));
        let (chat, chat_tool_use) =
            (compile("tokenizer.chat_template"), compile("tokenizer.chat_template.tool_use"));
        tokenizer.chat = chat;
        tokenizer.chat_tool_use = chat_tool_use;
        Ok(tokenizer)
    }

    /// The number of tokens.
    pub fn n_tokens(&self) -> usize {
        self.vocab.len()
    }

    /// The tokens the model names for its structure.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.vocab.special
    }

    /// Whether [`encode`](Self::encode) with `add_special` puts the
    /// beginning-of-text token first.
    pub fn adds_bos(&self) -> bool {
        self.vocab.add_bos
    }

    /// Whether [`encode`](Self::encode) with `add_special` puts the
    /// end-of-text token last.
    pub fn adds_eos(&self) -> bool {
        self.vocab.add_eos
    }

    /// Whether the model's generation stops at this token: the end-of-text and
    /// end-of-turn tokens, and the markers of fill-in-the-middle that follow the
    /// generated text.
    pub fn is_end_of_generation(&self, id: u32) -> bool {
        self.vocab.end_of_generation.binary_search(&id).is_ok()
    }

    /// The text of a token as the file stores it. For normal tokens this is in
    /// the byte-level alphabet (`Ġ` for a space); [`piece`](Self::piece) has
    /// the bytes it stands for.
    pub fn token_text(&self, id: u32) -> Option<&[u8]> {
        ((id as usize) < self.n_tokens()).then(|| self.vocab.texts.get(id as usize))
    }

    /// What a token is, or `None` for an id outside the vocabulary.
    pub fn kind(&self, id: u32) -> Option<TokenKind> {
        let attrs = *self.vocab.attrs.get(id as usize)?;
        Some(if attrs & attr::CONTROL != 0 {
            TokenKind::Control
        } else if attrs & attr::USER_DEFINED != 0 {
            TokenKind::UserDefined
        } else if attrs & attr::UNKNOWN != 0 {
            TokenKind::Unknown
        } else if attrs & attr::NORMAL != 0 {
            TokenKind::Normal
        } else if attrs & attr::UNUSED != 0 {
            TokenKind::Unused
        } else {
            TokenKind::Undefined
        })
    }

    /// The id of the control, user defined or unknown token with exactly this
    /// text, for example `"<|im_end|>"`.
    pub fn special_token_id(&self, text: &str) -> Option<u32> {
        self.markers_all.iter().find(|m| *m.text == *text.as_bytes()).map(|m| m.id)
    }

    /// The tokens of `text`.
    ///
    /// The text is bytes, not `&str`: llama.cpp accepts anything, and so does
    /// this. Every byte that does not start a valid UTF-8 sequence is read as
    /// U+FFFD, one replacement per byte, so a truncated sequence gives as many
    /// replacements as it has bytes. (The surrogates U+D800 to U+DFFF, which
    /// strict UTF-8 refuses, are accepted.)
    pub fn encode(&self, text: impl AsRef<[u8]>, options: EncodeOptions) -> Vec<u32> {
        let text = text.as_ref();
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        self.encode_into(text, options, &mut out);
        out
    }

    /// Like [`encode`](Self::encode), appending to `out`.
    pub fn encode_into(&self, text: &[u8], options: EncodeOptions, out: &mut Vec<u32>) {
        if options.add_special && self.vocab.add_bos {
            out.extend(self.vocab.special.bos);
        }
        match self.pieces(text, options.parse_special) {
            None => self.encode_text(text, out),
            Some(pieces) => {
                for piece in pieces {
                    match piece {
                        Piece::Text(range) => self.encode_text(&text[range], out),
                        Piece::Token(id) => out.push(id),
                    }
                }
            }
        }
        if options.add_special && self.vocab.add_eos {
            out.extend(self.vocab.special.eos);
        }
    }

    /// The words the pre-tokenizer cuts `text` into, as BPE sees them (after
    /// invalid bytes have become U+FFFD). Special tokens are not looked for.
    pub fn pre_tokenize(&self, text: impl AsRef<[u8]>) -> Vec<Vec<u8>> {
        let text = sanitize(text.as_ref());
        let mut words = Vec::new();
        match self.pre {
            Pre::Qwen35 => pretok::split(&text, classes(), |w| words.push(w.to_vec())),
        }
        words
    }

    /// BPE of text without special tokens.
    fn encode_text(&self, text: &[u8], out: &mut Vec<u32>) {
        let text = sanitize(text);
        let (merges, byte_token) = (&self.vocab.merges, &self.vocab.byte_token);
        // Short texts do not repeat enough to pay for filling a cache.
        if text.len() < CACHE_FROM {
            match self.pre {
                Pre::Qwen35 => {
                    pretok::split(&text, classes(), |word| {
                        encode_word(merges, byte_token, word, out)
                    });
                }
            }
            return;
        }
        CACHE.with_borrow_mut(|cache| {
            let cache = cache.get_or_insert_with(|| WordCache::new(self.id));
            cache.belong_to(self.id);
            match self.pre {
                Pre::Qwen35 => pretok::split(&text, classes(), |word| {
                    if word.len() > MAX_WORD {
                        return encode_word(merges, byte_token, word, out);
                    }
                    if let Some(tokens) = cache.get(word) {
                        return out.extend_from_slice(tokens);
                    }
                    let from = out.len();
                    encode_word(merges, byte_token, word, out);
                    cache.put(word, &out[from..]);
                }),
            }
        });
    }

    /// Cuts the special tokens out of `text`, or `None` when it has none.
    ///
    /// This follows llama.cpp token by token: each marker, the longest first,
    /// is looked for in the pieces of text that the markers before it left,
    /// and every occurrence is cut out. A marker never reaches into the text
    /// of one cut before it, even when that makes the markers overlap.
    fn pieces(&self, text: &[u8], parse_special: bool) -> Option<Vec<Piece>> {
        let markers = if parse_special { &self.markers_all } else { &self.markers_user };
        // The places where some marker could start, by first byte.
        let mut leads = [false; 256];
        for marker in markers {
            leads[usize::from(marker.text[0])] = true;
        }
        let mut starts: Vec<Vec<usize>> = vec![Vec::new(); 256];
        let mut any = false;
        for (i, byte) in text.iter().enumerate() {
            if leads[usize::from(*byte)] {
                starts[usize::from(*byte)].push(i);
                any = true;
            }
        }
        if !any {
            return None;
        }

        let mut pieces = vec![Piece::Text(0..text.len())];
        let mut cut_any = false;
        for marker in markers {
            let candidates = &starts[usize::from(marker.text[0])];
            if candidates.is_empty() {
                continue;
            }
            let mut next = Vec::with_capacity(pieces.len() + 2);
            for piece in pieces.drain(..) {
                let Piece::Text(range) = piece else {
                    next.push(piece);
                    continue;
                };
                let mut from = range.start;
                let mut at = candidates.partition_point(|c| *c < range.start);
                while let Some(&candidate) = candidates.get(at) {
                    if candidate + marker.text.len() > range.end {
                        break;
                    }
                    at += 1;
                    if candidate < from
                        || text[candidate..candidate + marker.text.len()] != *marker.text
                    {
                        continue;
                    }
                    if candidate > from {
                        next.push(Piece::Text(from..candidate));
                    }
                    next.push(Piece::Token(marker.id));
                    cut_any = true;
                    from = candidate + marker.text.len();
                }
                if from < range.end {
                    next.push(Piece::Text(from..range.end));
                }
            }
            pieces = next;
        }
        cut_any.then_some(pieces)
    }
}

impl Tokenizer {
    /// The bytes that token `id` stands for in text.
    ///
    /// Control and unknown tokens stand for nothing unless `special` is set; then
    /// they are their own text. User defined tokens are always their text.
    /// Unused tokens never stand for anything.
    ///
    /// # Errors
    ///
    /// [`Error::TokenOutOfRange`] when `id` is not in the vocabulary.
    pub fn piece(&self, id: u32, special: bool) -> Result<&[u8]> {
        let attrs = *self
            .vocab
            .attrs
            .get(id as usize)
            .ok_or(Error::TokenOutOfRange { id, n_tokens: self.n_tokens() })?;
        if !special && attrs & (attr::CONTROL | attr::UNKNOWN) != 0 {
            return Ok(b"");
        }
        Ok(self.vocab.pieces.get(id as usize))
    }

    /// The bytes of a sequence of tokens. With `special`, control tokens are
    /// written as their text; without, they are left out.
    ///
    /// Decoding with `special` the tokens of a text gives the text back byte for
    /// byte, when the text is valid UTF-8 (see [`encode`](Self::encode) for what
    /// becomes of the rest).
    ///
    /// # Errors
    ///
    /// [`Error::TokenOutOfRange`] for an id that is not in the vocabulary.
    pub fn decode(&self, ids: &[u32], special: bool) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            out.extend_from_slice(self.piece(id, special)?);
        }
        Ok(out)
    }
}

impl Tokenizer {
    /// Whether the model file has a chat template.
    pub fn has_chat_template(&self) -> bool {
        self.chat.is_some() || self.chat_tool_use.is_some()
    }

    /// The text of the model file's chat template: the default one, else the
    /// one for conversations with tools.
    pub fn chat_template_source(&self) -> Option<&str> {
        let template = self.chat.as_ref().or(self.chat_tool_use.as_ref())?.as_ref().ok()?;
        Some(template.source())
    }

    /// Compiles `source` as a chat template of this model, with the texts of
    /// its beginning and end tokens: a template that takes the place of the
    /// file's ([`ChatOptions::template`]).
    ///
    /// # Errors
    ///
    /// Those of [`ChatTemplate::new`].
    pub fn compile_template(&self, source: &str) -> std::result::Result<ChatTemplate, ChatError> {
        let text_of = |id: Option<u32>| {
            id.and_then(|id| self.piece(id, true).ok())
                .map(|piece| String::from_utf8_lossy(piece).into_owned())
                .unwrap_or_default()
        };
        let special = &self.vocab.special;
        ChatTemplate::new(source, &text_of(special.bos), &text_of(special.eos))
    }

    /// How the model's chat template handles thinking, or `None` if it has
    /// none (or no chat template, or one that cannot be used).
    pub fn reasoning_format(&self) -> Option<ReasoningFormat> {
        let template = self.chat.as_ref().or(self.chat_tool_use.as_ref())?.as_ref().ok()?;
        ReasoningFormat::detect(template)
    }

    /// How the model's chat template writes the calls of tools, or `None` if
    /// it writes them in a form that is not understood (or has no template for
    /// them). The template asked is the one a conversation with tools is
    /// rendered by.
    pub fn tool_format(&self) -> Option<ToolFormat> {
        let template = self.chat_tool_use.as_ref().or(self.chat.as_ref())?.as_ref().ok()?;
        ToolFormat::detect(template)
    }

    /// The prompt for a conversation, made by the model's chat template.
    ///
    /// A conversation with tools is rendered by the model's template for tools
    /// if it has one, and by its default template otherwise. The text still has
    /// to be encoded, with [`EncodeOptions::parse_special`] set: the template
    /// writes the control tokens (`<|im_start|>`) as text.
    ///
    /// If the model adds a beginning-of-text token when it encodes, a prompt
    /// that begins with that token's text loses it here, so that it is not there
    /// twice; the same for the end-of-text token at the end.
    ///
    /// # Errors
    ///
    /// [`ChatError::NoTemplate`] if the file has none, the error of the template
    /// if it uses something that is not supported, and the errors of
    /// [`ChatTemplate::render`].
    pub fn render_chat(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &ChatOptions,
    ) -> std::result::Result<String, ChatError> {
        let template = match (&options.template, &self.chat_tool_use, &self.chat) {
            (Some(given), _, _) => given.as_ref(),
            (None, Some(tool_use), _) if !tools.is_empty() => {
                tool_use.as_ref().map_err(Clone::clone)?
            }
            (None, _, Some(default)) => default.as_ref().map_err(Clone::clone)?,
            (None, Some(tool_use), None) => tool_use.as_ref().map_err(Clone::clone)?,
            (None, None, None) => return Err(ChatError::NoTemplate),
        };
        let mut prompt = template.render(messages, tools, options)?;
        let special = &self.vocab.special;
        let text_of = |id: Option<u32>| {
            id.and_then(|id| self.piece(id, true).ok())
                .filter(|p| !p.is_empty())
                .map(<[u8]>::to_vec)
        };
        if let (true, Some(bos)) = (self.vocab.add_bos, text_of(special.bos))
            && prompt.as_bytes().starts_with(&bos)
        {
            prompt.drain(..bos.len());
        }
        if let (true, Some(eos)) = (self.vocab.add_eos, text_of(special.eos))
            && prompt.as_bytes().ends_with(&eos)
        {
            prompt.truncate(prompt.len() - eos.len());
        }
        Ok(prompt)
    }
}
