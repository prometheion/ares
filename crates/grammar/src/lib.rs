//! The grammar engine on the host.

mod compile;
mod cursor;
mod error;
mod gbnf;
mod guide;
mod schema;
mod vocab;

pub use compile::{Compiler, Grammar, Source};
pub use cursor::Cursor;
pub use error::Error;
pub use guide::Guide;
pub use vocab::Vocabulary;

/// Whether the mask `words` allows `token`.
pub fn mask_allows(words: &[u32], token: u32) -> bool {
    words.get((token / 32) as usize).is_some_and(|w| w >> (token % 32) & 1 != 0)
}
