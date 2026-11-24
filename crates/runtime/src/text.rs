//! Between tokens and text: the byte vocabulary and stop strings.

/// The text of byte token `byte` as a printable character.
///
/// Printable ASCII maps to itself and every other byte to its own character
/// in U+0100..U+01FF (Latin Extended), so the mapping is one to one: two
/// different outputs never print the same, which is what lets a test compare
/// text instead of tokens. The characters are all printable, so a stream of
/// picked bytes is readable in a log and valid UTF-8 by construction.
pub fn printable(byte: u8) -> char {
    if (0x20..0x7f).contains(&byte) {
        char::from(byte)
    } else {
        char::from_u32(0x100 + u32::from(byte)).expect("a BMP scalar")
    }
}

/// The byte a character from [`printable`] stands for, or `None` for any
/// other character. Prompts are given as text, and one token is one byte of
/// it, so a prompt that holds a character outside the byte vocabulary is
/// rejected by the model's `encode`.
pub fn unprintable(ch: char) -> Option<u8> {
    match u32::from(ch) {
        c @ 0x20..0x7f => Some(c as u8),
        c @ 0x100..0x200 => Some((c - 0x100) as u8),
        _ => None,
    }
}

/// Cuts generated text at the first stop string and holds back text that
/// might still turn into one.
///
/// A client's stop strings end generation "before any of these strings would
/// be emitted", and a token can complete a stop string that earlier tokens
/// began. So text is released only when it can no longer be the start of a
/// stop string; the rest waits for the next token.
#[derive(Debug, Default)]
pub(crate) struct StopMatcher {
    /// The stop strings with their places in the list the matcher was given.
    stops: Vec<(usize, String)>,
    /// Text that has been generated and not released: it is a prefix of at
    /// least one stop string (or empty).
    held: String,
    /// The place of the stop string that matched.
    matched: Option<usize>,
}

impl StopMatcher {
    /// A matcher for `stops`. Empty strings are ignored: they would match
    /// before any text.
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().cloned().enumerate().filter(|(_, s)| !s.is_empty()).collect(),
            held: String::new(),
            matched: None,
        }
    }

    /// The place in the list it was given of the stop string that matched,
    /// once one has.
    pub fn matched(&self) -> Option<usize> {
        self.matched
    }

    /// Feeds one character of generated text. Text that is safe to send is
    /// appended to `out`. Returns `true` when a stop string was completed: the
    /// text before it has been appended to `out`, the stop string itself and
    /// anything after it dropped, and the caller ends the request.
    pub fn push(&mut self, ch: char, out: &mut String) -> bool {
        if self.stops.is_empty() {
            out.push(ch);
            return false;
        }
        self.held.push(ch);
        // A stop string that now ends the held text. Held text is a prefix
        // of some stop string, so a completed one can only be at its end,
        // but another stop string may be a suffix of a longer prefix; check
        // all by position, earliest first.
        let earliest = self
            .stops
            .iter()
            .filter_map(|(index, stop)| self.held.find(stop.as_str()).map(|at| (at, *index)))
            .min();
        if let Some((at, index)) = earliest {
            out.push_str(&self.held[..at]);
            self.held.clear();
            self.matched = Some(index);
            return true;
        }
        // Release everything before the longest suffix that is a prefix of
        // some stop string.
        let keep_from = self
            .held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| self.stops.iter().any(|(_, stop)| stop.starts_with(&self.held[i..])))
            .unwrap_or(self.held.len());
        out.push_str(&self.held[..keep_from]);
        self.held.drain(..keep_from);
        false
    }

    /// Releases the held text, for a request that ended some other way.
    pub fn flush(&mut self, out: &mut String) {
        out.push_str(&self.held);
        self.held.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], text: &str) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let mut matcher = StopMatcher::new(&stops);
        let mut out = String::new();
        for ch in text.chars() {
            if matcher.push(ch, &mut out) {
                return (out, true);
            }
        }
        matcher.flush(&mut out);
        (out, false)
    }

    #[test]
    fn every_byte_has_its_own_printable_character_and_back() {
        let mut seen = std::collections::HashSet::new();
        for byte in 0..=255u8 {
            let ch = printable(byte);
            assert!(!ch.is_control() && !ch.is_whitespace() || ch == ' ', "{byte}: {ch:?}");
            assert!(seen.insert(ch), "byte {byte} prints like another byte");
            assert_eq!(unprintable(ch), Some(byte));
        }
        assert_eq!(printable(b'a'), 'a');
        assert_eq!(unprintable('\u{4e2d}'), None);
        assert_eq!(unprintable('\n'), None);
    }

    #[test]
    fn without_stop_strings_text_passes_through() {
        assert_eq!(run(&[], "hello"), ("hello".to_owned(), false));
    }

    #[test]
    fn text_before_a_stop_string_is_kept_and_the_string_is_not_emitted() {
        assert_eq!(run(&["END"], "abcENDdef"), ("abc".to_owned(), true));
    }

    #[test]
    fn a_stop_string_spanning_several_tokens_is_still_found() {
        assert_eq!(run(&["\n\n"], "a\n\nb"), ("a".to_owned(), true));
    }

    #[test]
    fn a_false_start_is_released_when_it_turns_out_not_to_match() {
        // "EN" is held while it could become "END", and comes out with the next character.
        assert_eq!(run(&["END"], "xENy"), ("xENy".to_owned(), false));
        assert_eq!(run(&["END"], "EENDz"), ("E".to_owned(), true));
    }

    #[test]
    fn held_text_is_released_when_the_request_ends_otherwise() {
        assert_eq!(run(&["END"], "abEN"), ("abEN".to_owned(), false));
    }

    #[test]
    fn the_earliest_of_several_stop_strings_wins() {
        assert_eq!(run(&["xyz", "bc"], "abcxyz"), ("a".to_owned(), true));
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        assert_eq!(run(&[""], "abc"), ("abc".to_owned(), false));
    }
}
