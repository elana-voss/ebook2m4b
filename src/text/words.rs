//! English words as Kokoro's text-to-sound step (misaki) knows them.
//!
//! known() tells a word of the language from a misprint or an invented name. data/known-words.txt is
//! misaki's US gold and silver dictionaries in lower case: the lower-case keys of both, and the
//! capitalised keys of gold lowered. known_in() does the same for a book's language; German's list is
//! LibreOffice's (data/german-words-source.txt), kept gzipped in the program.
use crate::setup::archive::gunzip;
use crate::text::language::Language;
use std::collections::HashSet;
use std::sync::LazyLock;

static KNOWN: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| include_str!("../../data/known-words.txt").lines().collect());

/// data/german-words.txt.gz (its source and licence: data/german-words-source.txt), read at the first German word.
static GERMAN: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let list = gunzip(include_bytes!("../../data/german-words.txt.gz")).unwrap_or_default();
    list.leak().lines().collect()
});
const GERMAN_PART: usize = 4;
const GERMAN_LINKS: [&str; 5] = ["s", "es", "n", "en", "e"];
/// The words another may have run into in German ("derkelmar", "nichtmehr"), so no part of a compound:
/// no preposition, which is also a verb's prefix.
pub const GERMAN_LITTLE: [&str; 37] = [
    "der", "die", "das", "dem", "den", "des", "eine", "einen", "einem", "einer", "eines", "und", "oder", "aber",
    "auch", "nicht", "sich", "sie", "er", "es", "ich", "du", "wir", "ihr", "als", "noch", "doch", "immer", "ganz",
    "sehr", "schon", "dann", "denn", "wenn", "weil", "dass", "daß",
];
/// Words that end a run-together pair ("immerwieder") but also start true compounds ("wiederkehr").
const GERMAN_NEVER_LAST: [&str; 3] = ["mehr", "wieder", "genau"];

const PREFIXES: [&str; 5] =["over", "under", "out", "fore", "counter"];

const ENDINGS: [(&str, &[&str]); 12] = [
    ("s", &[""]),
    ("es", &[""]),
    ("ies", &["y"]),
    ("ed", &["", "e"]),
    ("ied", &["y"]),
    ("ing", &["", "e"]),
    ("er", &["", "e"]),
    ("ers", &["", "e"]),
    ("est", &["", "e"]),
    ("ly", &[""]),
    ("ness", &[""]),
    ("'s", &[""]),
];

/// A word of the dictionary, or one with an ending it lacks: "carriers", "maddened", "clutching";
/// or one made with -like: "lionlike"; or a variant spelling of one (variant()). A word the dictionary
/// lacks in -er or -ers (the loosest ending) whose two neighbouring letters swapped make one it holds is
/// a misprint: "soliders" ("soldiers", not "solid" + "ers").
pub fn known(word: &str) -> bool {
    let w = word.to_lowercase();
    let er = w.ends_with("er") || w.ends_with("ers");
    KNOWN.contains(w.as_str()) || derived(&w) && !(er && swapped(&w))
}

/// A word of the book's language. German: the list, the old spelling ("daß"), or a compound.
pub fn known_in(word: &str, language: Language) -> bool {
    if language == Language::English {
        return known(word);
    }
    let w = word.to_lowercase();
    let listed = |s: &str| GERMAN.contains(s);
    let modern = w.replace('ß', "ss");
    listed(&w) || listed(&modern) || german_compound(&w.chars().collect::<Vec<_>>(), 3)
        || german_compound(&modern.chars().collect::<Vec<_>>(), 3)
}

/// Two or three listed parts of at least four letters, a linking s, es, n, en or e allowed between them:
/// "leuchtturmwärter", "liebesbrief"; a little word is no part ("nichtmehr"). What is listed with a shorter part ("hauptbahnhof") is in the list.
fn german_compound(w: &[char], parts: usize) -> bool {
    if parts < 2 {
        return false;
    }
    let listed = |cs: &[char]| {
        let s: String = cs.iter().collect();
        let little = GERMAN_LITTLE.contains(&s.as_str()) || GERMAN_NEVER_LAST.contains(&s.as_str());
        GERMAN.contains(s.as_str()) && !little
    };
    // A first part may have lost its ending: "leucht-turm" (leuchten), "schul-buch" (Schule).
    let stem = |cs: &[char]| {
        let s: String = cs.iter().collect();
        !GERMAN_LITTLE.contains(&s.as_str())
            && (GERMAN.contains(s.as_str()) || ["e", "en"].iter().any(|e| GERMAN.contains(format!("{s}{e}").as_str())))
    };
    (GERMAN_PART..=w.len().saturating_sub(GERMAN_PART)).any(|i| {
        stem(&w[..i])
            && [0, 1, 2].iter().any(|&link| {
                let linked: String = w[i..w.len().min(i + link)].iter().collect();
                let rest = &w[w.len().min(i + link)..];
                (link == 0 || GERMAN_LINKS.contains(&linked.as_str()))
                    && rest.len() >= GERMAN_PART
                    && (listed(rest) || german_compound(rest, parts - 1))
            })
    })
}

/// A dictionary word with two neighbouring letters swapped: "soliders" ("soldiers").
fn swapped(w: &str) -> bool {
    let c: Vec<char> = w.chars().collect();
    (1..c.len()).any(|i| {
        let mut s = c.clone();
        s.swap(i - 1, i);
        s != c && KNOWN.contains(s.iter().collect::<String>().as_str())
    })
}

/// known() without the dictionary's own entry and the swap check.
fn derived(w: &str) -> bool {
    let n = w.chars().count();
    let mut stems = vec![w.to_string()];
    let mut doubled = Vec::new();
    for (e, rs) in ENDINGS {
        let Some(stem) = w.strip_suffix(e) else { continue };
        if e == "s" && stem.ends_with('s') {
            continue; // "wristss": a plural of a word in -s takes -es
        }
        // An -er stem with its e back has four letters: "baker" -> "bake", never "ther" -> "the".
        let least = if e == "er" || e == "ers" { 4 } else { 3 };
        for r in rs {
            if n - e.chars().count() + r.chars().count() >= least {
                stems.push(format!("{stem}{r}")); // "tries" -> "try"
            }
        }
        // A consonant doubles only before an ending: "stopped" -> "stopp" -> "stop"; never "furr" -> "fur".
        let c: Vec<char> = stem.chars().collect();
        if ["ed", "ing", "er", "ers", "est"].contains(&e) && c.len() > 3 && c[c.len() - 1] == c[c.len() - 2] {
            doubled.push(c[..c.len() - 1].iter().collect::<String>());
        }
    }
    if stems
        .iter()
        .chain(&doubled)
        .any(|s| KNOWN.contains(s.as_str()))
    {
        return true;
    }
    n > 6 && w.ends_with("like") && known(&w[..w.len() - 4]) || variant(w).is_some_and(|v| known(&v))
}

/// The spelling of a known word a variant stands for, as the fork's lexicon reads it (stem_more):
/// "blonde" -> "blond", "geometrical" -> "geometric", "stylised" -> "stylized". Shorter, or without
/// "is", so known() ends.
fn variant(w: &str) -> Option<String> {
    const ISE: [&str; 8] = ["ise", "ised", "ises", "ising", "iser", "isers", "isation", "isations"];
    if let Some(e) = ISE.iter().find(|e| w.ends_with(*e)) {
        return Some(format!("{}iz{}", &w[..w.len() - e.len()], &e[2..]));
    }
    if let Some(s) = w.strip_suffix("ical") {
        return Some(format!("{s}ic"));
    }
    w.strip_suffix('e')
        .filter(|s| s.chars().count() >= 5 || s.chars().count() == 4 && s.ends_with("ss")) // "blonde", "masse", never "frome" -> "from"
        .map(str::to_string)
}

/// over-, under-, out-, fore-, counter- before a word: "overtook", or a misprint ("overhelmed") to be
/// mended, never parted.
pub fn compound(word: &str) -> bool {
    PREFIXES.iter().any(|p| {
        word.strip_prefix(p)
            .is_some_and(|rest| rest.chars().count() >= 4 && known(rest))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endings_doubled_letters_like_and_compounds() {
        for w in [
            "carriers",
            "maddened",
            "clutching",
            "stopped",
            "tries",
            "lionlike",
            "Their",
        ] {
            assert!(known(w), "{w}");
        }
        for w in ["thier", "chldren", "vornsman", "barl"] {
            assert!(!known(w), "{w}");
        }
        assert!(compound("overtook") && compound("overhelmed") && !compound("overt"));
    }

    #[test]
    fn variants_of_a_known_word_are_known() {
        for w in ["blonde", "geometrical", "stylised", "civilisations"] {
            assert!(known(w), "{w}");
        }
        assert!(!known("barle") && !known("barlical") && !known("frome") && !known("thate"));
        assert!(known("masse"));
    }

    #[test]
    fn a_doubled_letter_only_before_an_ending() {
        for w in ["furr", "dinn", "wristss", "carefullly", "chatte"] {
            assert!(!known(w), "{w}");
        }
        for w in ["running", "bigger", "kisses", "dresses", "planned"] {
            assert!(known(w), "{w}");
        }
    }

    #[test]
    fn er_needs_a_stem_that_takes_it() {
        for w in ["ther", "soliders"] {
            assert!(!known(w), "{w}");
        }
        for w in ["baker", "bakers", "users", "dyers", "hunters", "taller"] {
            assert!(known(w), "{w}");
        }
    }

    #[test]
    fn german_words_their_old_spelling_and_compounds() {
        let g = Language::German;
        for w in ["Haus", "häuser", "daß", "Leuchtturmwärter", "Hundehütte", "Liebesbrief"] {
            assert!(known_in(w, g), "{w}");
        }
        // little words run together are no compound
        for w in ["Hauus", "Xarbon", "Müler", "nichtmehr", "nochnicht", "immerwieder", "dochnicht", "ganzgenau", "einenmann"] {
            assert!(!known_in(w, g), "{w}");
        }
    }

    /// A measure, run by hand: known() of each word of EBOOK2M4B_WORDS (one a line) into EBOOK2M4B_KNOWN ("word TAB 0|1").
    #[test]
    #[ignore]
    fn known_of_a_word_list() {
        let (Ok(i), Ok(o)) = (std::env::var("EBOOK2M4B_WORDS"), std::env::var("EBOOK2M4B_KNOWN")) else { return };
        let words = std::fs::read_to_string(i).unwrap();
        let out: String = words.lines().map(|w| format!("{w}\t{}\n", known(w) as u8)).collect();
        std::fs::write(o, out).unwrap();
    }
}
