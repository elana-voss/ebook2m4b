//! The words eSpeak speaks, and how an invented word is said.
//! found(): of a book's chunks, the words misaki's dictionaries do not answer, English (words::known)
//! or invented. settle(): each invented word not in work/sounds.json yet goes the round trip once, in its
//! shortest passage: spoken by Kokoro, heard by Whisper with the word left out of the hint, scored by the
//! letter edit distance of the heard words to the written word (distance). eSpeak's guess first; a word
//! it fails is respelt in English by the language model ("CHOO-eer"), which eSpeak sounds and the trip
//! tests; the smaller distance is kept. apply(): the book's words whose respelling won are said with it.
use crate::speech::chunks::Chunk;
use crate::speech::pronounce::{Pronounce, holds};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// A trip passes at a distance of a third of the word's letters or less.
pub const PASS: f64 = 1.0 / 3.0;

/// The words settled so far, every book's: {"word": {"say": respelling or null, "espeak": distance,
/// "respelling": distance or null}}.
pub const FILE: &str = "work/sounds.json";

/// One word's trips: the respelling kept (none: eSpeak's guess), and both distances.
#[derive(Clone, Debug, PartialEq)]
pub struct Sound {
    pub say: Option<String>,
    pub espeak: f64,
    pub respelling: Option<f64>,
}

pub type Store = BTreeMap<String, Sound>;

/// Pieces of a contraction misaki's tokens leave ("ca" "n't"), not words.
const PIECES: [&str; 8] = ["ca", "wo", "sha", "ll", "ve", "re", "n't", "n’t"];

/// A word: letters, apostrophes or hyphens inside, two letters or more; not a contraction's piece nor
/// words run together by a dash ("is—Be"), which misaki sends to eSpeak whole.
fn a_word(w: &str) -> bool {
    let inner = |c: char| c.is_alphabetic() || "-'’".contains(c);
    w.chars().all(inner)
        && w.chars().next().is_some_and(char::is_alphabetic)
        && w.chars().next_back().is_some_and(char::is_alphabetic)
        && w.chars().filter(|c| c.is_alphabetic()).count() >= 2
        && !PIECES.contains(&w.to_lowercase().as_str())
}

/// The book's words eSpeak speaks: (English ones in lower case, invented ones as written).
pub fn found(cs: &[Chunk], p: &Pronounce) -> Result<(Vec<String>, Vec<String>)> {
    let (mut english, mut invented) = (BTreeSet::new(), BTreeSet::new());
    for c in cs {
        for w in p.espeak_words(&c.text)?.into_iter().filter(|w| a_word(w)) {
            if crate::text::words::known(&w) {
                english.insert(w.to_lowercase());
            } else {
                invented.insert(w);
            }
        }
    }
    Ok((english.into_iter().collect(), invented.into_iter().collect()))
}

fn letters(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphabetic())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The least letter edit distance of `word` to one to three heard words joined, over its letters.
pub fn distance(word: &str, heard: &str) -> f64 {
    let w = letters(word);
    let n = w.chars().count();
    if n == 0 {
        return 0.0;
    }
    let hs: Vec<String> = heard
        .split_whitespace()
        .map(letters)
        .filter(|h| !h.is_empty())
        .collect();
    let mut best = n;
    for i in 0..hs.len() {
        let mut joined = String::new();
        for h in &hs[i..(i + 3).min(hs.len())] {
            joined += h;
            best = best.min(crate::text::edit::distance(&w, &joined));
        }
    }
    best as f64 / n as f64
}

/// a line of fewer sounds counts as this many: Whisper hears a short line without context, so one or two
/// sounds off there ("Be sad" for "be said") are the ear's, and a short line is unsure only from three
const SHORT_LINE: usize = 20;

/// Edit distance of the sounds of what was heard against the text's, over the text's sounds' length
/// (numbers as words, as the voice was given them), at least SHORT_LINE. A word
/// Whisper writes in capitals ("MIP") is a word it heard, not letters, so it is lowered before its sounds
/// are asked. What Whisper hears beyond the text's end is free: it goes on after the voice stopped
/// (repeats the line, finishes the sentence), the voice reads its phonemes and adds none.
pub fn heard_error(p: &Pronounce, expected: &str, heard: &str) -> Result<f32> {
    let sounds = |t: &str| -> Result<String> {
        let t = crate::text::clean::clean(t);
        if t.trim().is_empty() {
            return Ok(String::new()); // nothing heard: no phonemes to ask for
        }
        p.sounds(&t)
    };
    let heard: String = heard
        .split(' ')
        .map(|w| {
            if w.chars().filter(|c| c.is_alphabetic()).count() > 1
                && !w.chars().any(|c| c.is_lowercase())
            {
                w.to_lowercase()
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let heard = names_as_heard(p, expected, &heard)?;
    let (a, b) = (sounds(expected)?, sounds(&heard)?);
    let wrong = crate::text::edit::distance_to_prefix(&a, &b);
    Ok(wrong as f32 / a.chars().count().max(SHORT_LINE) as f32)
}

/// `heard` with Whisper's spelling of the text's invented names put right: a word of `expected` outside the
/// dictionary (words::known), of four letters or more, replaces the one to three heard words in a row whose
/// sounds differ from its own in at most half ("hair of Ollie" for "Haravoli", "B-Day Toshik" for
/// "Bidaitoshik"), or the one heard word whose letters differ in at most a third ("Choir" for "Chooir",
/// "Malukta" for "Mel-Ok-Ta"); of those, one spelt near, else the nearest in sounds. A name heard as spelt stays. Whisper writes words it knows for a name it does not, and
/// eSpeak's sounds for its spelling are far from the name's, though what was heard was the name.
fn names_as_heard(p: &Pronounce, expected: &str, heard: &str) -> Result<String> {
    let letters = |w: &str| -> String {
        w.chars()
            .filter(char::is_ascii_alphabetic)
            .flat_map(char::to_lowercase)
            .collect()
    };
    let mut out: Vec<String> = heard.split(' ').map(String::from).collect();
    let said: Vec<String> = out.iter().map(|w| p.sounds(w)).collect::<Result<_>>()?;
    let names: Vec<&str> = expected
        .split(|c: char| c.is_whitespace())
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.chars().filter(char::is_ascii_alphabetic).count() >= 4 && !crate::text::words::known(w))
        .collect();
    // a name heard as spelt is that name: not another's nearest sound
    let heard: Vec<String> = out.iter().map(|w| letters(w)).collect();
    let mut taken: Vec<bool> = heard.iter().map(|w| names.iter().any(|n| letters(n) == *w)).collect();
    for name in names.into_iter().filter(|n| !heard.contains(&letters(n))) {
        let want = p.sounds(name)?;
        let spelt = letters(name);
        let mut near = None;
        for k in 0..out.len() {
            for len in 1..=3.min(out.len() - k) {
                if taken[k..k + len].iter().any(|&t| t) {
                    break;
                }
                let got: String = said[k..k + len].concat();
                let d = crate::text::edit::distance(&want, &got);
                let by_letters = len == 1 && crate::text::edit::distance(&spelt, &heard[k]) * 3 <= spelt.len();
                if (by_letters || d * 2 <= want.chars().count())
                    && near.is_none_or(|n| (!by_letters, d, len, k) < n)
                {
                    near = Some((!by_letters, d, len, k));
                }
            }
        }
        if let Some((_, _, len, k)) = near {
            let tail: String = out[k + len - 1].chars().rev().take_while(|c| !c.is_alphanumeric()).collect();
            out[k] = format!("{name}{}", tail.chars().rev().collect::<String>());
            out[k + 1..k + len].iter_mut().for_each(String::clear);
            taken[k..k + len].iter_mut().for_each(|t| *t = true);
        }
    }
    Ok(out.into_iter().filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" "))
}

/// The shortest chunk holding `word`; else the shortest holding it once apostrophes are dropped
/// (misaki reads "Ul-’Mesa" as the word "Ul-Mesa").
fn passage<'a>(cs: &'a [Chunk], word: &str) -> Option<&'a str> {
    let shortest = |held: &dyn Fn(&str) -> bool| {
        cs.iter().map(|c| c.text.as_str()).filter(|t| held(t)).min_by_key(|t| t.len())
    };
    shortest(&|t| holds(t, word)).or_else(|| shortest(&|t| holds(&t.replace(['’', '\''], ""), word)))
}

/// Whisper's hint for a passage, the book's names (chunks::book_words) less those holding `word`.
fn hint_without(names: &str, word: &str, text: &str, limit: usize) -> String {
    let lower = word.to_lowercase();
    let kept: Vec<&str> = names
        .split(", ")
        .filter(|n| !n.to_lowercase().contains(&lower))
        .collect();
    crate::speech::chunks::hint(&kept.join(", "), text, limit)
}

/// (phonemes, hint) or (word, passage).
type Pair = (String, String);
/// Word -> its English respelling.
type Respelt = HashMap<String, String>;

/// The book's invented words not settled yet, each through the round trip: `trip` speaks and hears
/// (phonemes, hint) pairs, `respell` gives (word, passage) pairs an English respelling. Returns the
/// book's invented words.
pub fn settle(
    cs: &[Chunk],
    names: &str,
    hint_chars: usize,
    p: &Pronounce,
    store: &mut Store,
    trip: &mut dyn FnMut(&[Pair]) -> Result<Vec<String>>,
    respell: &mut dyn FnMut(&[Pair]) -> Result<Respelt>,
) -> Result<Vec<String>> {
    let (_, invented) = found(cs, p)?;
    let todo: Vec<(&str, &str)> = invented
        .iter()
        .filter(|w| !store.contains_key(*w))
        .filter_map(|w| Some((w.as_str(), passage(cs, w)?)))
        .collect();
    if todo.is_empty() {
        return Ok(invented);
    }
    let pairs = |said: &dyn Fn(&str) -> Option<String>, words: &[(&str, &str)]| {
        words
            .iter()
            .map(|(w, text)| {
                p.say_as(w, said(w));
                let ps = p.phonemes(text);
                p.say_as(w, None);
                Ok((ps?, hint_without(names, w, text, hint_chars)))
            })
            .collect::<Result<Vec<_>>>()
    };
    let heard = trip(&pairs(&|_| None, &todo)?)?;
    let first: Vec<f64> = todo.iter().zip(&heard).map(|((w, _), h)| distance(w, h)).collect();
    let failed: Vec<(&str, &str)> = todo
        .iter()
        .zip(&first)
        .filter(|(_, d)| **d > PASS)
        .map(|(t, _)| *t)
        .collect();
    let asked: Vec<(String, String)> = failed
        .iter()
        .map(|(w, t)| (w.to_string(), t.to_string()))
        .collect();
    // keyed in lower case: the model may answer "chooir" for "Chooir"
    let answers: HashMap<String, String> = if asked.is_empty() {
        HashMap::new()
    } else {
        respell(&asked)?.into_iter().map(|(w, r)| (w.to_lowercase(), r)).collect()
    };
    let respelt: Vec<(&str, &str)> = failed
        .into_iter()
        .filter(|(w, _)| answers.get(&w.to_lowercase()).is_some_and(|r| !letters(r).is_empty()))
        .collect();
    let sounds: HashMap<&str, String> = respelt
        .iter()
        .map(|(w, _)| Ok((*w, p.respelt(&answers[&w.to_lowercase()])?)))
        .collect::<Result<_>>()?;
    let second = trip(&pairs(&|w| sounds.get(w).cloned(), &respelt)?)?;
    let again: HashMap<&str, f64> = respelt
        .iter()
        .zip(&second)
        .map(|((w, _), h)| (*w, distance(w, h)))
        .collect();
    for ((w, _), espeak) in todo.iter().zip(first) {
        let respelling = again.get(w).copied();
        let say = respelling
            .filter(|r| *r < espeak)
            .map(|_| answers[&w.to_lowercase()].clone());
        store.insert(w.to_string(), Sound { say, espeak, respelling });
    }
    Ok(invented)
}

/// A word's won respelling, its own or a case twin's ("Wqxor" at a sentence start says "wqxor"'s).
fn said<'a>(store: &'a Store, w: &str) -> Option<&'a String> {
    let lower = w.to_lowercase();
    store.get(w).and_then(|s| s.say.as_ref()).or_else(|| {
        store
            .iter()
            .filter(|(k, _)| k.to_lowercase() == lower)
            .find_map(|(_, s)| s.say.as_ref())
    })
}

/// The book's `words` whose respelling won (said()) said with it from now on (a word's sound is the
/// same in every book, so another book's stay).
pub fn apply(words: &[String], p: &Pronounce, store: &Store) -> Result<()> {
    for w in words {
        if let Some(r) = said(store, w) {
            p.say_as(w, Some(p.respelt(r)?));
        }
    }
    Ok(())
}

/// The pronunciation a book is spoken with, for its work_id: the fork's rules' version and
/// sounds.json's entries for the book's words.
pub fn version(words: &[String], store: &Store) -> Value {
    let said: serde_json::Map<String, Value> = words
        .iter()
        .filter_map(|w| {
            let s = said(store, w);
            (s.is_some() || store.contains_key(w)).then(|| (w.clone(), json!(s)))
        })
        .collect();
    json!({"rules": crate::speech::pronounce::RULES, "sounds": said})
}

pub fn load(path: &Path) -> Result<Store> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(Store::new()); // none yet
    };
    let v: Value = serde_json::from_str(&text).with_context(|| path.display().to_string())?;
    let mut out = Store::new();
    for (w, s) in v.as_object().context("sounds.json is no object")? {
        out.insert(
            w.clone(),
            Sound {
                say: s["say"].as_str().map(str::to_string),
                espeak: s["espeak"].as_f64().context("a sound without its distance")?,
                respelling: s["respelling"].as_f64(),
            },
        );
    }
    Ok(out)
}

pub fn save(path: &Path, store: &Store) -> Result<()> {
    let v: serde_json::Map<String, Value> = store
        .iter()
        .map(|(w, s)| {
            let entry = json!({"say": s.say, "espeak": s.espeak, "respelling": s.respelling});
            (w.clone(), entry)
        })
        .collect();
    let text = serde_json::to_string_pretty(&v)? + "\n";
    std::fs::write(path, text).with_context(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::pronounce::tests::P;
    use crate::speech::speak::MAX_ERROR;

    fn chunk(text: &str) -> Chunk {
        Chunk {
            chapter: 0,
            text: text.into(),
            pause: 0.0,
            cut: false,
        }
    }

    #[test]
    fn english_and_invented_words_are_told_apart() {
        let cs = [
            chunk("Vasena smiled."),
            chunk("“Zarl!” she said. “Chooir,” I reminded her. “I can't.”"),
        ];
        let (english, invented) = found(&cs, &P).unwrap();
        assert!(english.is_empty(), "{english:?}");
        assert_eq!(invented, ["Chooir", "Vasena", "Zarl"]);
    }

    #[test]
    fn the_distance_is_to_the_closest_heard_words_over_the_letters() {
        assert_eq!(distance("Zo-ri-ba", "the city of Zo ri ba fell"), 0.0);
        assert_eq!(distance("Chooir", "I reminded Choor then"), 1.0 / 6.0);
        assert_eq!(distance("Zarl", "nothing like it"), 1.0);
    }

    #[test]
    fn the_hint_leaves_the_word_out() {
        let h = hint_without("Zarl Baker, Vasena, Zo-ri-ba", "Zarl", "Zarl and Vasena.", 200);
        assert_eq!(h, "Vasena, Zo-ri-ba");
    }

    #[test]
    fn a_respelling_is_kept_only_where_it_is_heard_better() {
        let cs = [
            chunk("Wqxor stood. Wqxor is a long name here."),
            chunk("Wqxor smiled."),
            chunk("Vvoplam waited."),
        ];
        let asked = std::cell::RefCell::new(Vec::new());
        let mut trip = |pairs: &[(String, String)]| -> Result<Vec<String>> {
            // eSpeak's guesses are heard wrong; a respelt Wqxor is heard right
            Ok(pairs
                .iter()
                .map(|(ps, hint)| {
                    assert!(!(hint.contains("Wqxor") && hint.contains("Vvoplam")), "{hint}");
                    if ps.contains(&P.respelt("zeye-kat").unwrap()) {
                        "wqxor smiled".to_string()
                    } else {
                        "something else".to_string()
                    }
                })
                .collect())
        };
        let mut respell = |items: &[(String, String)]| -> Result<HashMap<String, String>> {
            asked.borrow_mut().extend(items.iter().cloned());
            Ok(HashMap::from([
                ("wqxor".to_string(), "ZEYE-kat".to_string()), // the model's case, not the word's
                ("Vvoplam".to_string(), "VOP-lam".to_string()),
            ]))
        };
        let mut store = Store::new();
        let names = "Wqxor, Vvoplam";
        let words = settle(&cs, names, 200, &P, &mut store, &mut trip, &mut respell).unwrap();
        assert_eq!(words, ["Vvoplam", "Wqxor"]);
        assert_eq!(asked.borrow()[1], ("Wqxor".to_string(), "Wqxor smiled.".to_string()));
        assert_eq!(store["Wqxor"].say.as_deref(), Some("ZEYE-kat"));
        assert_eq!(store["Wqxor"].respelling, Some(0.0));
        assert_eq!(store["Vvoplam"].say, None);
        assert!(store["Vvoplam"].espeak > PASS);
        // once per word: settled words go no trip again
        let mut no_trip = |_: &[(String, String)]| -> Result<Vec<String>> { anyhow::bail!("again") };
        settle(&cs, names, 200, &P, &mut store, &mut no_trip, &mut respell).unwrap();
        let file = std::env::temp_dir().join(format!("ebook2m4b-sounds-{}.json", std::process::id()));
        save(&file, &store).unwrap();
        assert_eq!(load(&file).unwrap(), store);
        let _ = std::fs::remove_file(&file); // a test's leftover
    }

    #[test]
    fn a_won_respelling_is_said() {
        let store = Store::from([(
            "Qxvat".to_string(),
            Sound { say: Some("CHOO-eer".into()), espeak: 1.0, respelling: Some(0.0) },
        )]);
        apply(&["Qxvat".to_string()], &P, &store).unwrap();
        let got = P.phonemes("Qxvat ran.").unwrap();
        P.say_as("Qxvat", None);
        assert!(got.starts_with(&P.respelt("CHOO-eer").unwrap()), "{got}");
    }

    #[test]
    fn a_case_twins_respelling_is_said() {
        let store = Store::from([(
            "qxvut".to_string(),
            Sound { say: Some("CHOO-eer".into()), espeak: 1.0, respelling: Some(0.0) },
        )]);
        apply(&["Qxvut".to_string()], &P, &store).unwrap();
        let got = P.phonemes("Qxvut ran.").unwrap();
        P.say_as("Qxvut", None);
        assert!(got.starts_with(&P.respelt("CHOO-eer").unwrap()), "{got}");
    }

    #[test]
    fn a_word_inside_a_hyphened_one_is_not_held() {
        assert!(!holds("Ro-Tan ran.", "Ro") && !holds("Ta-Ro ran.", "Ro") && holds("Ro ran.", "Ro"));
    }

    #[test]
    fn a_word_misaki_read_without_its_apostrophe_has_a_passage() {
        let cs = [chunk("A long line of Ul-’Mesa, or Lo-’Mar, and more."), chunk("Mesa’s at Ul-’Mesa.")];
        assert_eq!(passage(&cs, "Ul-Mesa"), Some("Mesa’s at Ul-’Mesa."));
        assert_eq!(passage(&cs, "Mesa"), Some("Mesa’s at Ul-’Mesa."));
        assert_eq!(passage(&cs, "Lo-Mar"), Some("A long line of Ul-’Mesa, or Lo-’Mar, and more."));
    }

    #[test]
    fn heard_error_compares_sounds() {
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert_eq!(
            err(
                "The Ivok's daughter, 3 barls.",
                "the ivok's daughter three barls"
            ),
            0.0
        );
        assert!(err("The Ivok's daughter.", "") > 0.5); // nothing heard
        let one = err("The cat sat.", "The bat sat.");
        assert!(one > 0.0 && one < 0.2, "{one}");
        // Whisper writes a name in capitals ("MIP"); the voice would have spelt it, Whisper heard a word
        assert_eq!(err("Chapter thirteen Mip", "Chapter 13 MIP"), 0.0);
    }

    #[test]
    fn a_name_whisper_nearly_spells_is_the_name() {
        let names = |e, h| names_as_heard(&P, e, h).unwrap();
        // Kokoro's sounds of "Choir" are far from those of "Chooir"; the letters are not
        assert_eq!(
            names("“I shall go,” said Chooir. Mel-Ok-Ta left.", "I shall go, said Choir. Malukta left."),
            "I shall go, said Chooir. Mel-Ok-Ta left."
        );
        // one spelt far from it stays; a letter of four may differ ("Carl" for "Zarl")
        assert_eq!(names("said Chooir.", "said table."), "said table.");
        assert_eq!(names("said Zarl.", "said Carl."), "said Zarl.");
    }

    #[test]
    fn a_name_whisper_spells_by_its_sound_is_the_name() {
        // Whisper writes words it knows that sound like the name, one or several
        let names = |e, h| names_as_heard(&P, e, h).unwrap();
        assert_eq!(names("said the Haravoli.", "said the hair of Ollie."), "said the Haravoli.");
        assert!(names("Amblis jumped;", "Am bliss jumped.").ends_with("Amblis jumped."));
        assert_eq!(names("asked Bidaitoshik.", "asked B-Day Toshik."), "asked Bidaitoshik.");
        // a name heard as spelt is that name, not the nearest sound of another one
        assert_eq!(
            names("Outside I encountered Mel-Ok-Ta and Mel-Da-Sa.", "Outside I encountered Malukta and Mel-Da-Sa."),
            "Outside I encountered Mel-Ok-Ta and Mel-Da-Sa."
        );
        assert_eq!(names("“I hunt,” said Chooir. “Of course,” said Selmus.", "I hunt, said Choir. Of course, said Selmus."), "I hunt, said Chooir. Of course, said Selmus.");
        assert_eq!(names("the dreen ran", "the drain ran"), "the dreen ran");
    }

    #[test]
    fn a_tail_heard_beyond_the_text_is_whispers() {
        // Whisper goes on after the voice stopped (repeats, finishes the sentence); the voice adds no word
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert_eq!(
            err(
                "‘Go in peace,’ said the Initiate, stepping aside.",
                "Go in peace, said the initiate, stepping aside Go in peace, said the initiate, stepping aside"
            ),
            0.0
        );
        // a tail the voice dropped still counts
        assert!(err("The cat sat on the mat.", "The cat sat") > 0.2);
    }

    #[test]
    fn a_short_line_is_unsure_only_from_three_sounds_off() {
        // one or two sounds off in a line of a few words is the ear without context ("Be sad" for "be said")
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert!(err("be said.", "Be sad.") <= MAX_ERROR);
        assert!(err("Zarl!", "Mary!") > MAX_ERROR);
    }
}
