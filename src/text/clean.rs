//! A paragraph's text made fit to be read aloud, numbers in English words.
//!
//! Mends UTF-8 read as Windows-1252 ("a€”", "Ã©"), drops note marks ("[12]", "*"), joins a scan's broken
//! hyphen ("threw- back"), turns stray marks to spaces, reads Roman numerals after chapter/book/part/
//! volume/act/scene and every number as words, and collapses whitespace.
//! Traps: a year 1100-2099 without a comma is read as a year; what a lookaround would say is checked by
//! hand (Rust's regex has none).
use crate::book::re;
use regex::Regex;
use std::sync::LazyLock;

static CP1252_UTF8: LazyLock<Regex> = LazyLock::new(|| re(r"[aâ]€([\x{80}-\x{FFFF}])"));
static LATIN1_UTF8: LazyLock<Regex> = LazyLock::new(|| re(r"[AÃ]([¡-¿])"));
static NOTE_MARKS: LazyLock<Regex> = LazyLock::new(|| re(r"\[\d+\]|\*+"));
/// That an end or a mark follows the numeral is checked by hand in `romans`.
static ROMAN: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b((?i:chapter|book|part|volume|act|scene))\s+([IVXLC]+)\b"));
/// Comma groups are one number only if no digit follows; group 2 is what follows, given back.
static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| re(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(\D|$)|\d+(?:\.\d+)?"));
static SPACES: LazyLock<Regex> = LazyLock::new(|| re(r"\s+"));

pub fn clean(text: &str) -> String {
    let text = CP1252_UTF8.replace_all(text, |c: &regex::Captures| {
        utf8_dash(&c[1]).unwrap_or_else(|| c[0].to_string())
    });
    let text = LATIN1_UTF8.replace_all(&text, |c: &regex::Captures| {
        let low = c[1].chars().next().map_or(0, |ch| ch as u32);
        char::from_u32(0xC0 + low - 0x80).map_or_else(|| c[0].to_string(), String::from)
    });
    let text = NOTE_MARKS.replace_all(&text, "");
    let text = broken_hyphens(&text);
    let text: String = text
        .chars()
        .map(|c| if "~`|/[]{}<>^_".contains(c) { ' ' } else { c })
        .collect();
    let text = romans(&text);
    let text = NUMBER.replace_all(&text, |c: &regex::Captures| match c.get(1) {
        Some(n) => number(n.as_str()) + &c[2],
        None => number(&c[0]),
    });
    SPACES.replace_all(&text, " ").trim().to_string()
}

/// Windows-1252's 0x80-0x9F; '\0' where it has no character.
const CP1252_HIGH: &str = "€\0‚ƒ„…†‡ˆ‰Š‹Œ\0Ž\0\0‘’“”•–—˜™š›œ\0žŸ";

/// "a€”": the third byte of an E2 80 xx character (a dash, a quote) read as Windows-1252.
fn utf8_dash(s: &str) -> Option<String> {
    let c = s.chars().next()?;
    let byte = match c as u32 {
        0xA0..=0xFF => c as u32,
        _ => 0x80 + CP1252_HIGH.chars().position(|h| h == c)? as u32,
    };
    // E2 80 xx is valid UTF-8 only for a continuation byte; it is then U+2000 + (xx - 0x80)
    if !(0x80..=0xBF).contains(&byte) {
        return None;
    }
    char::from_u32(0x2000 + byte - 0x80).map(String::from)
}

/// "threw- back" -> "threw-back": a hyphen and a space between two lower-case letters.
fn broken_hyphens(text: &str) -> String {
    let cs: Vec<char> = text.chars().collect();
    let lower = |i: usize| cs.get(i).is_some_and(|c| c.is_ascii_lowercase());
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < cs.len() {
        out.push(cs[i]);
        if cs[i] == '-' && i > 0 && lower(i - 1) && cs.get(i + 1) == Some(&' ') && lower(i + 2) {
            i += 1; // the space
        }
        i += 1;
    }
    out
}

/// "Chapter XIV." -> "Chapter fourteen.": only where the line ends or a mark follows ("a scene I saw" stays).
fn romans(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let (mut last, mut pos) = (0, 0);
    while let Some(c) = ROMAN.captures_at(text, pos) {
        let m = c.get(0).expect("a whole match");
        let rest = text[m.end()..].trim_start();
        if rest.is_empty() || rest.starts_with(['.', ':', ',', ';', '—', '-']) {
            out.push_str(&text[last..m.start()]);
            out.push_str(&format!("{} {}", &c[1], cardinal(roman(&c[2]))));
            last = m.end();
        }
        pos = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// By the subtracting rule, whatever the order: what is subtracted is always less than what follows it.
fn roman(s: &str) -> u128 {
    let n: Vec<i64> = s
        .chars()
        .map(|c| match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            _ => 100, // only C is left: the pattern allows IVXLC
        })
        .collect();
    let next = |i: usize| n.get(i + 1).copied().unwrap_or(0);
    (0..n.len())
        .map(|i| if n[i] < next(i) { -n[i] } else { n[i] })
        .sum::<i64>()
        .unsigned_abs() as u128
}

/// A decimal by its digits, a year 1100-2099 written without a comma as a year, else a cardinal.
fn number(m: &str) -> String {
    let s = m.replace(',', "");
    if s.contains('.') {
        return decimal(&s).unwrap_or_else(|| m.to_string());
    }
    match s.parse::<u128>() {
        Ok(n) if (1100..=2099).contains(&n) && !m.contains(',') => year(n),
        Ok(n) => cardinal(n),
        // over 38 digits, or digits other than 0-9: left as written
        Err(_) => m.to_string(),
    }
}

/// "3.14" -> "three point one four", "2.50" -> "two point five", "7.0" -> "seven point zero".
fn decimal(s: &str) -> Option<String> {
    let (whole, places) = s.split_once('.')?;
    let places = match places.trim_end_matches('0') {
        "" => "0",
        kept => kept,
    };
    let mut out = vec![cardinal(whole.parse().ok()?), "point".to_string()];
    for d in places.chars() {
        out.push(LOW[d.to_digit(10)? as usize].to_string());
    }
    Some(out.join(" "))
}

/// "nineteen eighty-four", "twelve oh-five", "nineteen hundred", but "two thousand and five".
fn year(n: u128) -> String {
    let (high, low) = (n / 100, n % 100);
    if (2000..=2009).contains(&n) {
        return cardinal(n);
    }
    let low = match low {
        0 => "hundred".to_string(),
        1..=9 => format!("oh-{}", cardinal(low)),
        _ => cardinal(low),
    };
    format!("{} {low}", cardinal(high))
}

const LOW: [&str; 21] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
];

const TENS: [&str; 8] = [
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
/// The name of each group of three digits; 10^36 is the last that fits in a u128.
const GROUPS: [&str; 13] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
    "undecillion",
];

/// 1-999: "twenty-one", "one hundred", "one hundred and one".
fn group(n: usize) -> String {
    let (hundreds, rest) = (n / 100, n % 100);
    let below = match rest {
        0..=20 => LOW[rest].to_string(),
        _ if rest % 10 == 0 => TENS[rest / 10 - 2].to_string(),
        _ => format!("{}-{}", TENS[rest / 10 - 2], LOW[rest % 10]),
    };
    match (hundreds, rest) {
        (0, _) => below,
        (_, 0) => format!("{} hundred", LOW[hundreds]),
        _ => format!("{} hundred and {below}", LOW[hundreds]),
    }
}

/// "one million, one thousand", "one thousand, one hundred", "one thousand and one".
fn cardinal(n: u128) -> String {
    if n == 0 {
        return LOW[0].to_string();
    }
    let mut groups = Vec::new();
    let (mut rest, mut place) = (n, 0);
    while rest > 0 {
        groups.push(((rest % 1000) as usize, place));
        rest /= 1000;
        place += 1;
    }
    let mut out = String::new();
    for &(g, place) in groups.iter().rev().filter(|(g, _)| *g > 0) {
        if !out.is_empty() {
            out.push_str(if place == 0 && g < 100 { " and " } else { ", " });
        }
        out.push_str(&group(g));
        if place > 0 {
            out.push(' ');
            out.push_str(GROUPS[place]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_in_words() {
        for (n, words) in [
            (0, "zero"),
            (1, "one"),
            (13, "thirteen"),
            (21, "twenty-one"),
            (101, "one hundred and one"),
            (110, "one hundred and ten"),
            (999, "nine hundred and ninety-nine"),
            (1001, "one thousand and one"),
            (1100, "one thousand, one hundred"),
            (100000, "one hundred thousand"),
            (1001000, "one million, one thousand"),
            (
                123456789,
                "one hundred and twenty-three million, four hundred and fifty-six thousand, seven hundred and eighty-nine",
            ),
            (1_000_000_000_000_007, "one quadrillion and seven"),
            (5 * 10u128.pow(36) + 3, "five undecillion and three"),
        ] {
            assert_eq!(cardinal(n), words, "{n}");
        }
    }

    #[test]
    fn cleaning_makes_text_to_read_aloud() {
        for (text, want) in [
            (
                "CHAPTER XIV. He paid 1,250 coins in 1984, 2005 and 1900; 1205 or 0.5 then 3.14 and 2.50 or 7.0 and 1,234.50",
                "CHAPTER fourteen. He paid one thousand, two hundred and fifty coins in nineteen eighty-four, two thousand and five and nineteen hundred; twelve oh-five or zero point five then three point one four and two point five or seven point zero and one thousand, two hundred and thirty-four point five",
            ),
            (
                "It was a€”so Ã© said [12] *x* threw- back ~a|b_c Part III: scene ii. act IV x Book IX",
                "It was —so é said x threw-back a b c Part three: scene ii. act IV x Book nine",
            ),
            (
                "  1,000,000 and 12,3456 or 101 and 115 and 1000001 and 2010 and 3002000 0 1.05  ",
                "one million and twelve,three thousand, four hundred and fifty-six or one hundred and one and one hundred and fifteen and one million and one and twenty ten and three million, two thousand zero one point zero five",
            ),
        ] {
            assert_eq!(clean(text), want);
        }
    }

    #[test]
    fn comma_groups_are_one_number_before_a_letter_but_not_before_a_digit() {
        assert_eq!(clean("the 10,000th and 1,000s"), "the ten thousandth and one thousands");
        assert_eq!(clean("2,500.5ft, 1,000"), "two thousand, five hundred point fiveft, one thousand");
        assert_eq!(clean("12,3456"), "twelve,three thousand, four hundred and fifty-six");
    }
}
