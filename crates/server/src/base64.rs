//! Base64 (RFC 4648): the signature that carries an omitted thinking block's
//! text, written and read with the standard alphabet and padding, and images
//! in data URLs, read as leniently as clients write them.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Appends `bytes` in base64 to `out`.
pub(crate) fn encode(bytes: &[u8], out: &mut String) {
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> (18 - 6 * i) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
}

/// The bytes `text` stands for, or `None` for text that is not base64 with
/// the standard alphabet and padding, as [`encode`] writes it.
pub(crate) fn decode(text: &str) -> Option<Vec<u8>> {
    let padding = text.bytes().rev().take_while(|&c| c == b'=').count();
    let standard = text.bytes().all(|c| ALPHABET.contains(&c) || c == b'=');
    if !text.len().is_multiple_of(4) || padding > 2 || !standard {
        return None;
    }
    decode_lenient(text)
}

/// The bytes `text` stands for, read as data URLs are written: the standard
/// or the URL alphabet, with or without padding; white space is skipped (a
/// data URL broken into lines).
pub(crate) fn decode_lenient(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut ended = false;
    for &c in text.as_bytes() {
        if c.is_ascii_whitespace() {
            continue;
        }
        if c == b'=' {
            ended = true;
            continue;
        }
        if ended {
            return None;
        }
        acc = (acc << 6) | value(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    // Leftover bits are padding and must be zero; one character alone is not
    // a byte.
    (bits < 6 && acc == 0).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_examples_of_the_rfc_and_back() {
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            let mut out = String::new();
            encode(plain.as_bytes(), &mut out);
            assert_eq!(out, coded);
            assert_eq!(decode(coded).unwrap(), plain.as_bytes());
        }
        let text = "Grüße, 世界 🙂\n";
        let mut out = String::new();
        encode(text.as_bytes(), &mut out);
        assert_eq!(decode(&out).unwrap(), text.as_bytes());
        for bad in ["Zg=", "Z===", "Zg==Zg==", "Zm9*", "Zm9v\n", "Zm9v====", "Zm-v"] {
            assert_eq!(decode(bad), None, "{bad}");
        }
        // Data URLs: either alphabet, padding or none, lines broken.
        assert_eq!(decode_lenient("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_lenient("aGVsbG8").unwrap(), b"hello");
        assert_eq!(decode_lenient("aGVs\nbG8=").unwrap(), b"hello");
        assert_eq!(decode_lenient("-_-_").unwrap(), [0xFB, 0xFF, 0xBF]);
        assert_eq!(decode_lenient("+/+/").unwrap(), [0xFB, 0xFF, 0xBF]);
        for bad in ["Z", "Zh", "Zg=A", "a*==", "aGVsbG8=x"] {
            assert_eq!(decode_lenient(bad), None, "{bad}");
        }
    }
}
