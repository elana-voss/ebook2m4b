//! Bytes to text: UTF-8 (a BOM dropped), UTF-16 with a BOM, else cp1252.

/// A file's bytes as text.
pub fn decode(b: &[u8]) -> String {
    if let Some(b) = b.strip_prefix(b"\xEF\xBB\xBF") {
        return String::from_utf8_lossy(b).into_owned();
    }
    let utf16 = |b: &[u8], le: bool| {
        let units: Vec<u16> = b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| {
                if le {
                    u16::from_le_bytes([c[0], c[1]])
                } else {
                    u16::from_be_bytes([c[0], c[1]])
                }
            })
            .collect();
        String::from_utf16_lossy(&units)
    };
    if let Some(b) = b.strip_prefix(b"\xFF\xFE") {
        return utf16(b, true);
    }
    if let Some(b) = b.strip_prefix(b"\xFE\xFF") {
        return utf16(b, false);
    }
    match std::str::from_utf8(b) {
        Ok(s) => s.to_string(),
        Err(_) => decode_as(b, false),
    }
}

/// Bytes as text: UTF-8, else cp1252.
pub fn decode_as(b: &[u8], utf8: bool) -> String {
    if utf8 {
        return String::from_utf8_lossy(b).into_owned();
    }
    const HIGH: &str = "€\u{81}‚ƒ„…†‡ˆ‰Š‹Œ\u{8d}Ž\u{8f}\u{90}‘’“”•–—˜™š›œ\u{9d}žŸ";
    b.iter()
        .map(|&c| match c {
            0x80..=0x9F => HIGH.chars().nth(usize::from(c - 0x80)).unwrap_or('?'),
            c => char::from(c),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_bytes_decode() {
        assert_eq!(decode(b"\xEF\xBB\xBFcaf\xC3\xA9"), "café");
        assert_eq!(decode(b"\x93caf\xe9\x94"), "\u{201c}café\u{201d}");
        assert_eq!(decode(b"\xFF\xFEa\x00b\x00"), "ab");
    }
}
