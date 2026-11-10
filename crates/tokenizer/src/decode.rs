use crate::error::Result;
use crate::tokenizer::Tokenizer;

/// Turns tokens that arrive one at a time into text that can be sent on.
///
/// A character can be spread over several tokens, so the bytes of a token are
/// not always text that stands alone. The detokenizer releases the bytes of a
/// token at once when they end on a character boundary and holds back the bytes
/// of an unfinished character until the rest arrives. Bytes that cannot be part
/// of any valid character are released right away, so nothing is delayed for
/// no reason and nothing is lost: the released bytes, followed by what
/// [`finish`](Self::finish) releases, are exactly what
/// [`Tokenizer::decode`] gives for all the tokens.
///
/// What is released is valid UTF-8 as long as the model wrote valid UTF-8.
///
/// ```no_run
/// # fn demo(tokenizer: &ares_tokenizer::Tokenizer, ids: &[u32]) -> ares_tokenizer::Result<()> {
/// let mut detokenizer = tokenizer.detokenizer(false);
/// let mut text = Vec::new();
/// for &id in ids {
///     text.clear();
///     detokenizer.push(id, &mut text)?;
///     print!("{}", String::from_utf8_lossy(&text));
/// }
/// text.clear();
/// detokenizer.finish(&mut text);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Detokenizer<'a> {
    tokenizer: &'a Tokenizer,
    special: bool,
    /// The start of an unfinished character: one to three bytes.
    held: Vec<u8>,
}

impl<'a> Detokenizer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, special: bool) -> Self {
        Self { tokenizer, special, held: Vec::with_capacity(4) }
    }

    /// Takes the next token and appends the bytes that can be released to
    /// `out`: possibly none.
    ///
    /// # Errors
    ///
    /// [`Error::TokenOutOfRange`](crate::Error::TokenOutOfRange) for an id that is
    /// not in the vocabulary. Nothing is changed then.
    pub fn push(&mut self, id: u32, out: &mut Vec<u8>) -> Result<()> {
        let piece = self.tokenizer.piece(id, self.special)?;
        if piece.is_empty() {
            return Ok(());
        }
        if self.held.is_empty() {
            self.release(piece, out);
        } else {
            let mut bytes = std::mem::take(&mut self.held);
            bytes.extend_from_slice(piece);
            self.release(&bytes, out);
        }
        Ok(())
    }

    /// Releases what is held back. Call it when the generation has ended: the
    /// model stopped in the middle of a character, and these bytes are what it
    /// wrote of it.
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        out.append(&mut self.held);
    }

    /// The bytes that are held back for now.
    pub fn held(&self) -> &[u8] {
        &self.held
    }

    /// Appends `bytes` to `out` except for an unfinished character at the end,
    /// which becomes the new `held`.
    fn release(&mut self, bytes: &[u8], out: &mut Vec<u8>) {
        let mut rest = bytes;
        loop {
            match std::str::from_utf8(rest) {
                Ok(_) => {
                    out.extend_from_slice(rest);
                    self.held.clear();
                    return;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    match error.error_len() {
                        // The bytes cannot be part of a character, whatever
                        // comes next: they are released as they are.
                        Some(bad) => {
                            out.extend_from_slice(&rest[..valid + bad]);
                            rest = &rest[valid + bad..];
                        }
                        // The end of the bytes is the start of a character.
                        None => {
                            out.extend_from_slice(&rest[..valid]);
                            self.held.clear();
                            self.held.extend_from_slice(&rest[valid..]);
                            return;
                        }
                    }
                }
            }
        }
    }
}
