//! English words as Kokoro's text-to-sound step (misaki) knows them.
//!
//! known() tells a word of the language from a misprint or an invented name. data/known-words.txt is
//! misaki's US gold and silver dictionaries in lower case: the lower-case keys of both, and the
//! capitalised keys of gold lowered.
use std::collections::HashSet;
use std::sync::LazyLock;

static KNOWN: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| include_str!("../../data/known-words.txt").lines().collect());

const PREFIXES: [&str; 5] = ["over", "under", "out", "fore", "counter"];

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

    /// tools/known_flip.py: known() of each word of EBOOK2M4B_WORDS (one a line) into EBOOK2M4B_KNOWN ("word TAB 0|1").
    #[test]
    #[ignore]
    fn known_of_a_word_list() {
        let (Ok(i), Ok(o)) = (std::env::var("EBOOK2M4B_WORDS"), std::env::var("EBOOK2M4B_KNOWN")) else { return };
        let words = std::fs::read_to_string(i).unwrap();
        let out: String = words.lines().map(|w| format!("{w}\t{}\n", known(w) as u8)).collect();
        std::fs::write(o, out).unwrap();
    }
}
