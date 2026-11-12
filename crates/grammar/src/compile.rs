//! Turning a grammar of any kind into something that can be stepped through.

use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use toktrie::InferenceCapabilities;

use crate::cursor::Cursor;
use crate::{Error, Vocabulary, gbnf, schema};

/// What a request asks the reply to look like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source<'a> {
    /// A grammar in GBNF, as llama.cpp defines it; `root` is where it starts.
    Gbnf(&'a str),
    /// A JSON schema, as JSON text.
    JsonSchema(&'a str),
    /// A grammar in Lark syntax, as llguidance defines it (see its
    /// documentation); `start` is where it starts.
    Lark(&'a str),
    /// A regular expression the whole reply must match.
    Regex(&'a str),
}

/// Compiles grammars for one vocabulary. Make one for a model and keep it:
/// building it reads the whole vocabulary.
pub struct Compiler {
    factory: ParserFactory,
    vocab: Vocabulary,
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler").field("vocab", &self.vocab).finish_non_exhaustive()
    }
}

impl Compiler {
    /// The compiler for `vocab`.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] when the engine cannot be set up for the vocabulary.
    pub fn new(vocab: Vocabulary) -> Result<Self, Error> {
        let mut factory = ParserFactory::new(
            vocab.env(),
            InferenceCapabilities::default(),
            &llguidance::earley::SlicedBiasComputer::general_slices(),
        )
        .map_err(|e| {
            Error::Compile(format!("the grammar engine cannot read the vocabulary: {e}"))
        })?;
        // Messages go to the caller, not to the standard error stream.
        factory.quiet();
        Ok(Self { factory, vocab })
    }

    /// The vocabulary grammars are compiled against.
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Compiles `source`.
    ///
    /// # Errors
    ///
    /// [`Error::Gbnf`] for a grammar that does not parse, [`Error::Schema`]
    /// for a schema that is not JSON, and [`Error::Compile`] when the engine
    /// refuses what is left (a schema it cannot turn into a grammar, a
    /// regular expression that is too big).
    pub fn compile(&self, source: Source<'_>) -> Result<Grammar, Error> {
        let top = match source {
            Source::Gbnf(text) => TopLevelGrammar::from_lark(gbnf::to_lark(text, &self.vocab)?),
            Source::JsonSchema(text) => TopLevelGrammar::from_json_schema(schema::prepare(text)?),
            Source::Lark(text) => TopLevelGrammar::from_lark(text.to_owned()),
            Source::Regex(text) => TopLevelGrammar::from_regex(text),
        };
        let matcher = Matcher::new(self.factory.create_parser(top));
        let mut grammar = Grammar { matcher, vocab: self.vocab.clone() };
        // The engine builds parts of a grammar on first use; a grammar it
        // cannot build is found here and not in the middle of a reply.
        grammar.start().mask().map_err(|e| {
            Error::Compile(match source {
                Source::JsonSchema(_) => format!("the JSON schema cannot be used: {e}"),
                _ => format!("the grammar cannot be used: {e}"),
            })
        })?;
        Ok(grammar)
    }
}

/// A compiled grammar. It holds no position: [`Grammar::start`] makes the
/// [`Cursor`] one reply walks through it with.
#[derive(Clone)]
pub struct Grammar {
    matcher: Matcher,
    vocab: Vocabulary,
}

impl std::fmt::Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grammar").finish_non_exhaustive()
    }
}

impl Grammar {
    /// A cursor at the start of the grammar.
    pub fn start(&mut self) -> Cursor {
        Cursor::new(self.matcher.clone(), self.vocab.clone())
    }
}
