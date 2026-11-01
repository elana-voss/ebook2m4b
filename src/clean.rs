//! A paragraph's text made fit to be read aloud (extract.py's clean() and the English num2words it calls).
//!
//! Mends UTF-8 read as Windows-1252 ("a€”", "Ã©"), drops note marks ("[12]", "*"), joins a scan's broken
//! hyphen ("threw- back"), turns stray marks to spaces, reads Roman numerals after chapter/book/part/
//! volume/act/scene and every number as words, and collapses whitespace.
//! Traps: num2words's quirks are kept on purpose so the text equals Python's ("12,3456" -> "...forty-
//! fivesix", a year 1100-2099 without a comma read as a year, "7.0" read as "seven"); Python's regex
//! lookarounds are done by hand (Rust's regex has none).
use regex::Regex;
use std::sync::LazyLock;

fn re(s: &str) -> Regex {
    Regex::new(s).expect("a valid pattern")
}

static CP1252_UTF8: LazyLock<Regex> = LazyLock::new(|| re(r"[aâ]€([\x{80}-\x{FFFF}])"));
static LATIN1_UTF8: LazyLock<Regex> = LazyLock::new(|| re(r"[AÃ]([¡-¿])"));
static NOTE_MARKS: LazyLock<Regex> = LazyLock::new(|| re(r"\[\d+\]|\*+"));
/// Python's lookahead (an end or a mark after the numeral) is checked by hand in `romans`.
static ROMAN: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b((?i:chapter|book|part|volume|act|scene))\s+([IVXLC]+)\b"));
static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| re(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"));
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
    let text = NUMBER.replace_all(&text, |c: &regex::Captures| number(&c[0]));
    SPACES.replace_all(&text, " ").trim().to_string()
}

/// Windows-1252's 0x80-0x9F; '\0' where it has no character (Python's codec refuses those).
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
            out.push_str(&format!("{} {}", &c[1], signed(roman(&c[2]))));
            last = m.end();
        }
        pos = m.end();
    }
    out.push_str(&text[last..]);
    out
}

fn roman(s: &str) -> i64 {
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
        .sum()
}

/// "IIIIIIIIIIIX" is -1 by the subtracting rule; num2words reads it "minus one".
fn signed(n: i64) -> String {
    let words = cardinal(n.unsigned_abs() as u128);
    if n < 0 {
        format!("minus {words}")
    } else {
        words
    }
}

/// A number as Python's `_number` reads it: a decimal by digits, a year 1100-2099 written without a
/// comma as a year, else a cardinal.
fn number(m: &str) -> String {
    let s = m.replace(',', "");
    if s.contains('.') {
        return decimal(&s).unwrap_or_else(|| m.to_string());
    }
    match s.parse::<u128>() {
        Ok(n) if (1100..=2099).contains(&n) && !m.contains(',') => year(n),
        Ok(n) => cardinal(n),
        // over 38 digits, or digits other than 0-9: left as written (Python reads them; no book has them)
        Err(_) => m.to_string(),
    }
}

/// num2words's float reading: "3.14" -> "three point one four", "7.0" -> "seven", "2.50" -> "two point five".
fn decimal(s: &str) -> Option<String> {
    let v: f64 = s.parse().ok()?;
    if !v.is_finite() || v >= 1e38 {
        return None;
    }
    let pre = v.trunc();
    if v == pre {
        return Some(cardinal(pre as u128));
    }
    // the places of the shortest repr (Python's str(float)), the digits after them floored, but
    // rounded where within 0.01 (1.239999 is 1.24)
    let shown = format!("{v}");
    let places = shown.split_once('.').map_or(0, |(_, f)| f.len());
    let post = (v - pre).abs() * format!("1e{places}").parse::<f64>().ok()?;
    let near = post.round_ties_even();
    let post = if (near - post).abs() < 0.01 {
        near
    } else {
        post.floor()
    } as u128;
    let digits = format!("{post:0>places$}");
    let mut out = vec![cardinal(pre as u128), "point".to_string()];
    for d in digits.chars().take(places) {
        out.push(LOW[d.to_digit(10)? as usize].to_string());
    }
    Some(out.join(" "))
}

/// num2words's to_year: "nineteen eighty-four", "twelve oh-five", "nineteen hundred", "two thousand and five".
fn year(n: u128) -> String {
    let (high, low) = (n / 100, n % 100);
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
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

/// num2words's cards, largest first: the -illions that fit in a u128 (10^36 down to 10^6), then the
/// mid words, then twenty down to zero.
static CARDS: LazyLock<Vec<(u128, String)>> = LazyLock::new(|| {
    let high = [
        "undec", "dec", "non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m",
    ];
    let mut cards: Vec<(u128, String)> = high
        .iter()
        .zip((2..=12).rev())
        .map(|(w, k)| (10u128.pow(3 * k), format!("{w}illion")))
        .collect();
    for (n, w) in [
        (1000, "thousand"),
        (100, "hundred"),
        (90, "ninety"),
        (80, "eighty"),
        (70, "seventy"),
        (60, "sixty"),
        (50, "fifty"),
        (40, "forty"),
        (30, "thirty"),
    ] {
        cards.push((n, w.to_string()));
    }
    cards.extend((0..=20).rev().map(|n| (n as u128, LOW[n].to_string())));
    cards
});

/// num2words's split of a number: (words, value) pairs and nested lists, merged by `join`.
enum Part {
    Pair(String, u128),
    List(Vec<Part>),
}

/// base.splitnum: the largest card not above the value, with what it is times and what is left.
/// Its tally case (div == value) needs a card of 1 below a value of 2 or more, which English never has.
fn splitnum(value: u128) -> Vec<Part> {
    let (elem, word) = CARDS
        .iter()
        .find(|(n, _)| *n <= value)
        .expect("zero is a card");
    let (div, rem) = if value == 0 {
        (1, 0)
    } else {
        (value / elem, value % elem)
    };
    let mut out = vec![if div == 1 {
        Part::Pair(LOW[1].to_string(), 1)
    } else {
        Part::List(splitnum(div))
    }];
    out.push(Part::Pair(word.clone(), *elem));
    if rem != 0 {
        out.push(Part::List(splitnum(rem)));
    }
    out
}

/// base.clean: merge the first two pairs, the rest as one list; lists are cleaned first.
fn join(mut val: Vec<Part>) -> Part {
    loop {
        if val.len() == 1 {
            return val.pop().expect("one part");
        }
        let mut out = Vec::new();
        if matches!((&val[0], &val[1]), (Part::Pair(..), Part::Pair(..))) {
            let rest = val.split_off(2);
            let right = pair(val.pop().expect("two parts"));
            let left = pair(val.pop().expect("two parts"));
            out.push(merge(left, right));
            if !rest.is_empty() {
                out.push(Part::List(rest));
            }
        } else {
            for elem in val {
                out.push(match elem {
                    Part::List(mut l) if l.len() == 1 => l.pop().expect("one part"),
                    Part::List(l) => join(l),
                    pair => pair,
                });
            }
        }
        val = out;
    }
}

fn pair(p: Part) -> (String, u128) {
    match p {
        Part::Pair(t, n) => (t, n),
        Part::List(l) => pair(join(l)),
    }
}

/// lang_EN.merge: "twenty-one", "one hundred and one", "five thousand", "one million, one thousand".
fn merge((ltext, lnum): (String, u128), (rtext, rnum): (String, u128)) -> Part {
    if lnum == 1 && rnum < 100 {
        Part::Pair(rtext, rnum)
    } else if 100 > lnum && lnum > rnum {
        Part::Pair(format!("{ltext}-{rtext}"), lnum + rnum)
    } else if lnum >= 100 && 100 > rnum {
        Part::Pair(format!("{ltext} and {rtext}"), lnum + rnum)
    } else if rnum > lnum {
        Part::Pair(format!("{ltext} {rtext}"), lnum * rnum)
    } else {
        Part::Pair(format!("{ltext}, {rtext}"), lnum + rnum)
    }
}

/// An English cardinal exactly as Python's num2words writes it.
fn cardinal(n: u128) -> String {
    pair(join(splitnum(n))).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_as_python_num2words_reads_them() {
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
    fn cleaning_matches_python() {
        for (text, want) in [
            (
                "CHAPTER XIV. He paid 1,250 coins in 1984, 2005 and 1900; 1205 or 0.5 then 3.14 and 2.50 or 7.0",
                "CHAPTER fourteen. He paid one thousand, two hundred and fifty coins in nineteen eighty-four, two thousand and five and nineteen hundred; twelve oh-five or zero point five then three point one four and two point five or seven",
            ),
            (
                "It was a€”so Ã© said [12] *x* threw- back ~a|b_c Part III: scene ii. act IV x Book IX",
                "It was —so é said x threw-back a b c Part three: scene ii. act IV x Book nine",
            ),
            (
                "  1,000,000 and 12,3456 or 101 and 115 and 1000001 and 2010 and 3002000 0 1.05  ",
                "one million and twelve thousand, three hundred and forty-fivesix or one hundred and one and one hundred and fifteen and one million and one and twenty ten and three million, two thousand zero one point zero five",
            ),
        ] {
            assert_eq!(clean(text), want);
        }
    }
}
