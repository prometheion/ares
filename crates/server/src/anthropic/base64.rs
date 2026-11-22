//! Base64 with the standard alphabet and padding (RFC 4648), for the
//! signature that carries an omitted thinking block's text.

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
/// padding.
pub(crate) fn decode(text: &str) -> Option<Vec<u8>> {
    let text = text.as_bytes();
    if !text.len().is_multiple_of(4) {
        return None;
    }
    let value = |c: u8| ALPHABET.iter().position(|&a| a == c).map(|v| v as u32);
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    for (k, chunk) in text.chunks(4).enumerate() {
        let last = k + 1 == text.len() / 4;
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 || (pad > 0 && !last) {
            return None;
        }
        let mut n = 0u32;
        for &c in &chunk[..4 - pad] {
            n = n << 6 | value(c)?;
        }
        n <<= 6 * pad as u32;
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    Some(out)
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
        for bad in ["Zg=", "Z===", "Zg==Zg==", "Zm9*", "Zm9v\n"] {
            assert_eq!(decode(bad), None, "{bad}");
        }
    }
}
