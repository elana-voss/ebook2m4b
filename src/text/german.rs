//! German text made fit to be read aloud: abbreviations, signs and numbers in German words.
//!
//! Numbers: "1.250" and "3,14" as German writes them; a year 1100-1999 as "neunzehnhundert..", a time
//! before "Uhr", "20er", a date "12.3.1945", "1" before a capital as "ein". "3." becomes an ordinal only
//! before a lower-case word, a month or a counted noun, or after an article (whose word gives the ending);
//! else the dot ends a sentence ("Es waren 30. Dann"). Traps: the endings are guessed from the word before
//! or the noun after, not from grammar ("der 2. Frau" reads "zweite"); a Roman numeral after a name
//! ("Friedrich II.") stays as it is: its case can't be told.
use crate::book::re;
use crate::text::clean::roman_after;
use regex::Regex;
use std::sync::LazyLock;

/// When an abbreviation is read out.
#[derive(Clone, Copy, PartialEq)]
enum When {
    Always,
    /// It can end a sentence: its dot stays before a capital or the text's end.
    Ends,
    /// Only before a digit ("Kap. 5"; "das Kap. Dann" is a cape).
    Digit,
    /// Only before a capital ("St. Gallen").
    Capital,
}

const ABBREVIATIONS: &[(&str, &str, When)] = &[
    ("z. B.", "zum Beispiel", When::Always),
    ("z. T.", "zum Teil", When::Always),
    ("d. h.", "das heißt", When::Always),
    ("u. a.", "unter anderem", When::Ends),
    ("u. ä.", "und ähnliches", When::Ends),
    ("o. ä.", "oder ähnliches", When::Ends),
    ("u. U.", "unter Umständen", When::Always),
    ("i. d. R.", "in der Regel", When::Always),
    ("a. D.", "außer Dienst", When::Ends),
    ("v. Chr.", "vor Christus", When::Ends),
    ("n. Chr.", "nach Christus", When::Ends),
    ("u. s. w.", "und so weiter", When::Ends),
    ("usw.", "und so weiter", When::Ends),
    ("etc.", "et cetera", When::Ends),
    ("bzw.", "beziehungsweise", When::Always),
    ("ca.", "circa", When::Always),
    ("evtl.", "eventuell", When::Always),
    ("ggf.", "gegebenenfalls", When::Always),
    ("vgl.", "vergleiche", When::Always),
    ("Dr.", "Doktor", When::Always),
    ("Prof.", "Professor", When::Always),
    ("Hr.", "Herr", When::Capital),
    ("Hrn.", "Herrn", When::Capital),
    ("Fr.", "Frau", When::Capital),
    ("Frl.", "Fräulein", When::Capital),
    ("St.", "Sankt", When::Capital),
    ("Mr.", "Mister", When::Capital),
    ("Mrs.", "Misses", When::Capital),
    ("Nr.", "Nummer", When::Digit),
    ("S.", "Seite", When::Digit),
    ("Bd.", "Band", When::Digit),
    ("Kap.", "Kapitel", When::Digit),
    ("Jh.", "Jahrhundert", When::Ends),
    ("Jhd.", "Jahrhundert", When::Ends),
    ("Mio.", "Millionen", When::Ends),
    ("Mrd.", "Milliarden", When::Ends),
    ("Tsd.", "tausend", When::Ends),
    ("Std.", "Stunden", When::Ends),
    ("Min.", "Minuten", When::Ends),
    ("Sek.", "Sekunden", When::Ends),
];

/// The table's forms with any space between their pieces; the longest first.
static ABBREVIATION: LazyLock<Regex> = LazyLock::new(|| {
    let mut forms: Vec<&str> = ABBREVIATIONS.iter().map(|a| a.0).collect();
    forms.sort_by_key(|f| std::cmp::Reverse(f.len()));
    let forms: Vec<String> = forms
        .iter()
        .map(|f| f.split(' ').map(regex::escape).collect::<Vec<_>>().join(r"\s*"))
        .collect();
    re(&format!(r"\b(?:{})", forms.join("|")))
});

const SIGNS: &[(&str, &str)] = &[
    ("km/h", "Kilometer pro Stunde"),
    ("km", "Kilometer"),
    ("cm", "Zentimeter"),
    ("mm", "Millimeter"),
    ("qm", "Quadratmeter"),
    ("kg", "Kilogramm"),
    ("m", "Meter"),
    ("g", "Gramm"),
    ("t", "Tonnen"),
    ("l", "Liter"),
    ("DM", "Mark"),
    ("%", "Prozent"),
    ("€", "Euro"),
    ("°C", "Grad Celsius"),
    ("°", "Grad"),
];

/// A sign after a digit; that no letter or digit follows is checked by hand.
static SIGN: LazyLock<Regex> = LazyLock::new(|| {
    let signs: Vec<String> = SIGNS.iter().map(|s| regex::escape(s.0)).collect();
    re(&format!(r"(\d)\s*({})", signs.join("|")))
});

/// A time, a date, a number in groups, a decimal, "20er", a plain number: first match wins.
static NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    re(concat!(
        r"(?P<hour>\d{1,2})(?:[.:](?P<minute>\d{2}))?\s+Uhr\b",
        r"|(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4}|\d{2}\b)?",
        r"|(?P<grouped>\d{1,3}(?:\.\d{3})+)(?:,(?P<fraction>\d+))?",
        r"|(?P<whole>\d+),(?P<decimals>\d+)",
        r"|(?P<decade>\d+)(?P<er>ern?)\b",
        r"|(?P<plain>\d+)",
    ))
});

static COUNTED_ROMAN: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b((?i:kapitel|buch|teil|band|akt|szene|abschnitt))\s+([IVXLC]+)\b"));

const MONTHS: &[&str] = &[
    "Januar", "Jänner", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
    "Oktober", "November", "Dezember",
];
/// A noun that is counted, with its ordinal's ending when no article comes before.
const COUNTED: &[(&str, &str)] = &[
    ("Kapitel", "es"),
    ("Buch", "es"),
    ("Mal", "es"),
    ("Jahr", "es"),
    ("Jahrhundert", "es"),
    ("Teil", "er"),
    ("Band", "er"),
    ("Akt", "er"),
    ("Abschnitt", "er"),
    ("Tag", "er"),
    ("Stock", "er"),
    ("Szene", "e"),
    ("Auflage", "e"),
    ("Strophe", "e"),
];
/// The word before an ordinal and the ending it gives.
const ARTICLES: &[(&str, &str)] = &[
    ("der", "e"),
    ("die", "e"),
    ("das", "e"),
    ("des", "en"),
    ("dem", "en"),
    ("den", "en"),
    ("am", "en"),
    ("im", "en"),
    ("zum", "en"),
    ("zur", "en"),
    ("vom", "en"),
    ("beim", "en"),
];

/// The text with its abbreviations, signs and numbers in words.
pub fn read_out(text: &str) -> String {
    let text = abbreviations(text);
    let text = signs(&text);
    let text = text.replace('&', "und").replace('€', "Euro").replace('§', "Paragraph");
    let text = roman_after(&text, &COUNTED_ROMAN, cardinal);
    numbers(&text)
}

fn abbreviations(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for m in ABBREVIATION.find_iter(text) {
        let form: String = m.as_str().split_whitespace().collect();
        let Some(&(_, words, when)) =
            ABBREVIATIONS.iter().find(|a| a.0.replace(' ', "") == form)
        else {
            continue; // the pattern is made of the table's forms
        };
        let next = text[m.end()..].trim_start().chars().next();
        let capital = next.is_some_and(char::is_uppercase);
        let dot = when == When::Ends && (capital || next.is_none());
        let read = match when {
            When::Always | When::Ends => true,
            When::Digit => next.is_some_and(|c| c.is_ascii_digit()),
            When::Capital => capital,
        };
        if read {
            out.push_str(&text[last..m.start()]);
            out.push_str(words);
            if dot {
                out.push('.');
            }
            last = m.end();
        }
    }
    out.push_str(&text[last..]);
    out
}

/// "90 km/h" -> "90 Kilometer pro Stunde".
fn signs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for c in SIGN.captures_iter(text) {
        let whole = c.get(0).expect("a whole match");
        if text[whole.end()..].chars().next().is_some_and(char::is_alphanumeric) {
            continue;
        }
        let Some(&(_, words)) = SIGNS.iter().find(|s| s.0 == &c[2]) else {
            continue; // the pattern is made of the table's signs
        };
        out.push_str(&text[last..whole.start()]);
        out.push_str(&c[1]);
        out.push(' ');
        out.push_str(words);
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

fn numbers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let (mut last, mut pos) = (0, 0);
    while let Some(c) = NUMBER.captures_at(text, pos) {
        let whole = c.get(0).expect("a whole match");
        let (before, after) = (&text[last..whole.start()], &text[whole.end()..]);
        out.push_str(before);
        let (read, skip) = number(&c, &text[..whole.start()], after);
        out.push_str(&read);
        last = whole.end() + skip;
        pos = last;
    }
    out.push_str(&text[last..]);
    out
}

/// One match in words, and how many bytes after it that used (an ordinal's dot).
fn number(c: &regex::Captures, before: &str, after: &str) -> (String, usize) {
    let value = |name: &str| c.name(name).map_or(0, |m| m.as_str().parse::<u128>().unwrap_or(0));
    if c.name("hour").is_some() {
        return (time(value("hour"), value("minute")), 0);
    }
    if c.name("day").is_some() {
        return (date(c, before), 0);
    }
    if let Some(g) = c.name("grouped") {
        let n: u128 = g.as_str().replace('.', "").parse().unwrap_or(0);
        return (fraction(cardinal(n), c.name("fraction").map(|m| m.as_str())), 0);
    }
    if let Some(w) = c.name("whole") {
        let whole = digits_or(w.as_str(), cardinal);
        return (fraction(whole, c.name("decimals").map(|m| m.as_str())), 0);
    }
    if let Some(d) = c.name("decade") {
        return (digits_or(d.as_str(), year) + &c["er"], 0);
    }
    plain(&c["plain"], before, after)
}

/// A plain number: an ordinal, "ein" before a capital, a year, a cardinal.
fn plain(digits: &str, before: &str, after: &str) -> (String, usize) {
    let n = digits.parse::<u128>().unwrap_or(0);
    if !is_digit_string(digits)
        && let Some(rest) = after.strip_prefix('.')
        && let Some(ending) = ordinal_ending(before, rest.trim_start())
    {
        let word = ordinal(n) + ending;
        let opens = before.trim_end().chars().last().is_none_or(|c| ".!?:".contains(c));
        return (if opens { capitalized(&word) } else { word }, 1);
    }
    let capital = after.trim_start().chars().next().is_some_and(char::is_uppercase);
    if n == 1 && capital && after.starts_with(char::is_whitespace) {
        return ("ein".to_string(), 0);
    }
    (digits_or(digits, year), 0)
}

/// The ending of the ordinal "3." given the text before and what follows its dot, or None: the dot ends a sentence.
fn ordinal_ending(before: &str, rest: &str) -> Option<&'static str> {
    let next: String = rest.chars().take_while(|c| c.is_alphabetic()).collect();
    if next.is_empty() {
        return None;
    }
    let word = before
        .trim_end()
        .rsplit(|c: char| !c.is_alphabetic())
        .next()
        .unwrap_or("")
        .to_lowercase();
    if let Some(a) = ARTICLES.iter().find(|a| a.0 == word) {
        return Some(a.1);
    }
    if let Some(n) = COUNTED.iter().find(|n| n.0 == next) {
        return Some(n.1);
    }
    let lower = next.chars().next().is_some_and(char::is_lowercase);
    (lower || MONTHS.contains(&next.as_str())).then_some("er")
}

/// A number with a leading zero or too long to say is read digit by digit.
fn is_digit_string(digits: &str) -> bool {
    (digits.len() > 1 && digits.starts_with('0')) || digits.len() > 15
}

fn digits_or(digits: &str, read: fn(u128) -> String) -> String {
    if is_digit_string(digits) {
        one_by_one(digits)
    } else {
        read(digits.parse().unwrap_or(0))
    }
}

fn one_by_one(digits: &str) -> String {
    digits
        .chars()
        .map(|d| cardinal(d.to_digit(10).unwrap_or(0) as u128))
        .collect::<Vec<_>>()
        .join(" ")
}

fn fraction(whole: String, decimals: Option<&str>) -> String {
    match decimals {
        Some(d) => format!("{whole} Komma {}", one_by_one(d)),
        None => whole,
    }
}

fn time(hour: u128, minute: u128) -> String {
    let h = if hour == 1 { "ein".to_string() } else { cardinal(hour) };
    match minute {
        0 => format!("{h} Uhr"),
        _ => format!("{h} Uhr {}", cardinal(minute)),
    }
}

/// "12.3.1945": "zwölften dritten neunzehnhundert.." after "am" and the like, else "zwölfter dritter ..".
fn date(c: &regex::Captures, before: &str) -> String {
    let (day, month) = (c["day"].parse().unwrap_or(0), c["month"].parse().unwrap_or(0));
    if !(1..=31).contains(&day) || !(1..=12).contains(&month) {
        // not a date ("1.50."): the numbers as they stand
        let year = c.name("year").map_or(String::new(), |y| year_words(y.as_str()));
        return format!("{}.{}.{year}", cardinal(day), cardinal(month));
    }
    let word = before.trim_end().rsplit(|ch: char| !ch.is_alphabetic()).next().unwrap_or("");
    let ending = if ARTICLES.iter().any(|a| a.0 == word.to_lowercase()) { "en" } else { "er" };
    let mut out = format!("{}{ending} {}{ending}", ordinal(day), ordinal(month));
    if let Some(y) = c.name("year") {
        out.push(' ');
        out.push_str(&year_words(y.as_str()));
    }
    out
}

fn year_words(digits: &str) -> String {
    digits_or(digits, year)
}

fn capitalized(word: &str) -> String {
    let mut cs = word.chars();
    cs.next().map_or(String::new(), |c| c.to_uppercase().chain(cs).collect())
}

/// A year 1100-1999 by its hundreds; everything else, 2005 too, a cardinal.
fn year(n: u128) -> String {
    if !(1100..=1999).contains(&n) {
        return cardinal(n);
    }
    let rest = if !n.is_multiple_of(100) { below_million(n % 100) } else { String::new() };
    format!("{}hundert{rest}", below_million(n / 100))
}

/// The ordinal's stem, ready for its ending: "dritt", "zwanzigst".
fn ordinal(n: u128) -> String {
    let last = n % 100;
    if last == 0 || last >= 20 {
        return cardinal(n) + "st";
    }
    let head = if n >= 100 { cardinal(n - last) } else { String::new() };
    let stem = match last {
        1 => "erst".to_string(),
        3 => "dritt".to_string(),
        7 => "siebt".to_string(),
        8 => "acht".to_string(),
        _ => cardinal(last) + "t",
    };
    head + &stem
}

const ONES: [&str; 20] = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf",
    "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
];
const TENS: [&str; 10] = [
    "", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig",
];
const LARGE: [(u128, &str, &str); 3] = [
    (1_000_000_000_000, "Billion", "Billionen"),
    (1_000_000_000, "Milliarde", "Milliarden"),
    (1_000_000, "Million", "Millionen"),
];

/// Below a million, written as one word; "ein" and not "eins" before "und", "hundert", "tausend".
fn below_million(n: u128) -> String {
    let head = |h: u128| {
        let word = below_million(h);
        word.strip_suffix("eins").map_or_else(|| word.clone(), |w| format!("{w}ein"))
    };
    match n {
        0..=19 => ONES[n as usize].to_string(),
        20..=99 => match n % 10 {
            0 => TENS[(n / 10) as usize].to_string(),
            o => format!("{}und{}", head(o), TENS[(n / 10) as usize]),
        },
        100..=999 => {
            let rest = if !n.is_multiple_of(100) { below_million(n % 100) } else { String::new() };
            format!("{}hundert{rest}", head(n / 100))
        }
        _ => {
            let rest = if !n.is_multiple_of(1000) { below_million(n % 1000) } else { String::new() };
            format!("{}tausend{rest}", head(n / 1000))
        }
    }
}

pub fn cardinal(n: u128) -> String {
    let mut parts = Vec::new();
    let mut rest = n;
    for (size, one, many) in LARGE {
        if rest >= size {
            let count = rest / size;
            parts.push(match count {
                1 => format!("eine {one}"),
                _ => format!("{} {many}", cardinal(count)),
            });
            rest %= size;
        }
    }
    if rest > 0 || parts.is_empty() {
        parts.push(below_million(rest));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals_read_in_german_words() {
        for (n, words) in [
            (0, "null"),
            (1, "eins"),
            (13, "dreizehn"),
            (16, "sechzehn"),
            (17, "siebzehn"),
            (21, "einundzwanzig"),
            (30, "dreißig"),
            (67, "siebenundsechzig"),
            (100, "einhundert"),
            (101, "einhunderteins"),
            (1000, "eintausend"),
            (2345, "zweitausenddreihundertfünfundvierzig"),
            (21000, "einundzwanzigtausend"),
            (1_000_000, "eine Million"),
            (2_000_001, "zwei Millionen eins"),
            (3_000_000_000, "drei Milliarden"),
        ] {
            assert_eq!(cardinal(n), words, "{n}");
        }
    }

    #[test]
    fn numbers_in_text() {
        for (text, want) in [
            ("Er zahlte 1.250 Mark im Jahr 1984 und 2005.", "Er zahlte eintausendzweihundertfünfzig Mark im Jahr neunzehnhundertvierundachtzig und zweitausendfünf."),
            ("Es waren 3,14 und 2,50 Meter.", "Es waren drei Komma eins vier und zwei Komma fünf null Meter."),
            ("Er kaufte 1 Haus und nahm 1 davon.", "Er kaufte ein Haus und nahm eins davon."),
            ("Am 3. Mai 1945 kam der 2. Mann zum 7. Mal.", "Am dritten Mai neunzehnhundertfünfundvierzig kam der zweite Mann zum siebten Mal."),
            ("Um 12.30 Uhr und um 7:05 Uhr, nicht um 9 Uhr.", "Um zwölf Uhr dreißig und um sieben Uhr fünf, nicht um neun Uhr."),
            ("Die 20er Jahre, die 1920er Jahre.", "Die zwanziger Jahre, die neunzehnhundertzwanziger Jahre."),
            ("Es waren 30. Dann ging er.", "Es waren dreißig. Dann ging er."),
            ("1. Kapitel", "Erstes Kapitel"),
            ("3. Teil", "Dritter Teil"),
            ("Kapitel XIV.", "Kapitel vierzehn."),
            ("Eine Szene I sah er.", "Eine Szene I sah er."),
            ("Am 12.3.1945 kam er.", "Am zwölften dritten neunzehnhundertfünfundvierzig kam er."),
            ("Agent 007.", "Agent null null sieben."),
        ] {
            assert_eq!(read_out(text), want);
        }
    }

    #[test]
    fn abbreviations_and_signs_in_words() {
        for (text, want) in [
            ("Das sind z. B. Hunde, d.h. Tiere, usw. Dann kam er.", "Das sind zum Beispiel Hunde, das heißt Tiere, und so weiter. Dann kam er."),
            ("Dr. Weber und Prof. Lang trafen Hr. Kuhn in St. Gallen.", "Doktor Weber und Professor Lang trafen Herr Kuhn in Sankt Gallen."),
            ("Im Jahr 44 v. Chr. fuhr er 90 km/h, ca. 5 km, bzw. 10 %.", "Im Jahr vierundvierzig vor Christus fuhr er neunzig Kilometer pro Stunde, circa fünf Kilometer, beziehungsweise zehn Prozent."),
            ("Er nahm Nr. 5 auf S. 12 für 3 € & mehr.", "Er nahm Nummer fünf auf Seite zwölf für drei Euro und mehr."),
            ("Der Abend usw.", "Der Abend und so weiter."),
            ("Er umsegelte das Kap. Dann ging er, mit 5 mal.", "Er umsegelte das Kap. Dann ging er, mit fünf mal."),
        ] {
            assert_eq!(read_out(text), want);
        }
    }
}
