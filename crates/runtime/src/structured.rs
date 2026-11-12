//! Structured output: the grammar engine behind the sampler's token filter.
//!
//! A request that carries a [`Structure`] gets a [`TokenFilter`] made of a
//! [`Guide`]: before each row the guide says which tokens the grammar allows
//! after the reply and the drafts before the row, and the conductor cuts a draft
//! the grammar forbids from the window (see [`TokenFilter::admitted`]).
//!
//! Reading a model's vocabulary into the engine takes about a second for a
//! vocabulary of a quarter million tokens, so a runtime does it once, on a thread
//! of its own, when it starts. A request with a structure waits for it if it
//! comes sooner.

use std::sync::{Arc, OnceLock};

use ares_api::{EngineError, Structure};
use ares_grammar::{Compiler, Guide, Vocabulary};
use ares_sampler::TokenMask;

use crate::model::Model;
use crate::sampling::TokenFilter;

type Slot = Arc<OnceLock<Result<Compiler, String>>>;

/// The grammar engine of one model.
pub(crate) struct Grammars {
    compiler: Slot,
}

impl Grammars {
    /// Starts reading the vocabulary of `model`. The tokens' text is read
    /// here, and the engine is built on a thread of its own.
    pub fn start(model: &dyn Model) -> Self {
        let tokens = model.vocab();
        let mut text = Vec::new();
        let pieces: Vec<Vec<u8>> = (0..tokens)
            .map(|token| {
                text.clear();
                model.piece(token, &mut text);
                text.clone()
            })
            .collect();
        let ends: Vec<u32> = (0..tokens).filter(|&token| model.ends(token)).collect();
        let compiler: Slot = Arc::new(OnceLock::new());
        let slot = compiler.clone();
        let spawned =
            std::thread::Builder::new().name("ares-grammar".to_owned()).spawn(move || {
                let built = Vocabulary::new(&pieces, &ends)
                    .and_then(Compiler::new)
                    .map_err(|e| e.to_string());
                let _ = slot.set(built);
            });
        if let Err(e) = spawned {
            let _ = compiler.set(Err(format!("cannot start the grammar engine: {e}")));
        }
        Self { compiler }
    }

    /// The filter that makes a reply follow `structure`.
    ///
    /// # Errors
    ///
    /// [`EngineError::InvalidRequest`] naming what is wrong with the grammar or
    /// the schema; [`EngineError::Internal`] when the engine could not be built.
    pub fn filter(&self, structure: &Structure) -> Result<Box<dyn TokenFilter>, EngineError> {
        let compiler = match self.compiler.wait() {
            Ok(compiler) => compiler,
            Err(message) => {
                return Err(EngineError::Internal(format!(
                    "the grammar engine is not available: {message}"
                )));
            }
        };
        let guide = compiler
            .guide(structure)
            .map_err(|e| EngineError::InvalidRequest(describe(structure, &e.to_string())))?;
        Ok(Box::new(GrammarFilter { guide, failure: None }))
    }
}

/// The message of a refusal, with the field it is about.
fn describe(structure: &Structure, message: &str) -> String {
    match structure.shape {
        ares_api::Shape::Gbnf(_) => format!("grammar: {message}"),
        ares_api::Shape::JsonSchema(_) => format!("json_schema: {message}"),
        ares_api::Shape::Lark(_) => format!("tools: {message}"),
    }
}

struct GrammarFilter {
    guide: Guide,
    failure: Option<String>,
}

impl GrammarFilter {
    fn failed(&mut self, message: String) {
        self.failure.get_or_insert(message);
    }
}

impl TokenFilter for GrammarFilter {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        match self.guide.mask(reply, drafts) {
            Ok(Some(words)) => {
                if words.iter().all(|&w| w == 0) {
                    self.failed("the grammar allows no token here".into());
                    return false;
                }
                mask.copy_words(&words);
                true
            }
            Ok(None) => false,
            Err(e) => {
                self.failed(e.to_string());
                false
            }
        }
    }

    fn admitted(&mut self, reply: &[u32], drafts: &[u32]) -> usize {
        match self.guide.admitted(reply, drafts) {
            Ok(n) => n,
            Err(e) => {
                self.failed(e.to_string());
                0
            }
        }
    }

    fn free_rows(&mut self, reply: &[u32], tokens: &[u32]) -> usize {
        match self.guide.free_rows(reply, tokens) {
            Ok(rows) => rows,
            Err(e) => {
                self.failed(e.to_string());
                0
            }
        }
    }

    fn failure(&mut self) -> Option<String> {
        self.failure.clone()
    }
}
