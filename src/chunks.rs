//! A book's chapters as chunks to speak: whole sentences, at most LIMIT (Kokoro's
//! 510 phonemes, as the caller's length counts them); a sentence longer than that is cut at a `;` or `:`
//! near its middle, else a comma, else a space, into pieces of about even length, each but the last
//! marked `cut` (assemble.rs joins it closer). Each chunk carries the pause after it: a sentence's, a paragraph's, a
//! scene break's, a chapter's, and after the book's title alone the opening's. A paragraph under SHORT
//! characters is spoken with the next (a lone word or two is spoken badly).
//! Sentences are told apart by rule, not by pysbd: an end mark (. ! ? …, closing quotes after it) before
//! a space and a capital, digit or quote, unless the word before is a title or an initial ("Mr.", "J.").
use crate::text::extract::{Chapter, SCENE};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

pub const LIMIT: usize = crate::kokoro::MAX_PHONEMES;
const SHORT: usize = 30;
const SENTENCE: f32 = 0.3;
const PARAGRAPH: f32 = 0.7;
const SCENE_PAUSE: f32 = 1.5;
const TITLE: f32 = 1.2;
const CHAPTER: f32 = 3.0; // before the next title
const OPENING: f32 = 2.0; // after the book's title; LEAD (assemble.rs) goes before it

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub chapter: usize,
    pub text: String,
    /// seconds of silence after it
    pub pause: f32,
    /// it ends inside a sentence (a long sentence's piece but its last)
    pub cut: bool,
}

const TITLES: [&str; 14] = [
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "capt", "col", "gen", "lt", "rev", "vs",
];

/// The sentences of a paragraph.
pub fn sentences(para: &str) -> Vec<String> {
    static END: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"[.!?…]+["'”’)\]]*\s+["'“‘(\[]?[A-Z0-9]"#).unwrap());
    let mut out = Vec::new();
    let mut start = 0;
    for m in END.find_iter(para) {
        let mark = m.start();
        let word = para[start..mark]
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .unwrap_or_default();
        let w = word.trim_start_matches(|c: char| !c.is_alphanumeric());
        let initial = w.chars().count() == 1 && w.chars().all(char::is_uppercase);
        if para[mark..].starts_with('.') && (initial || TITLES.contains(&w.to_lowercase().as_str()))
        {
            continue;
        }
        let space = m.as_str().find(char::is_whitespace).unwrap_or(0);
        out.push(para[start..mark + space].trim().to_string());
        start = mark + space;
    }
    out.push(para[start..].trim().to_string());
    out.retain(|s| !s.is_empty());
    out
}

/// A sentence longer than `limit` (by `len`) cut into pieces of about even length, none over it: at
/// the `;` or `:` nearest the even point within a quarter of the even length either side, else the `,`
/// nearest it there, else the space nearest it (Kokoro's pipeline cuts at these marks too). The even
/// point is counted in characters (phonemes run about one per character).
fn pieces(sentence: &str, limit: usize, len: &dyn Fn(&str) -> usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = sentence.trim();
    while len(rest) > limit {
        let cs: Vec<(usize, char)> = rest.char_indices().collect();
        let n = cs.len();
        let even = n / len(rest).div_ceil(limit);
        let reach = even / 4;
        let space = |c: usize| cs.get(c).is_some_and(|x| x.1.is_whitespace());
        // a cut c: the piece is the characters before it, a space at c (after the mark at c - 1)
        let near = |marks: &str| {
            (even.saturating_sub(reach)..=(even + reach).min(n - 1))
                .filter(|&c| c > 0 && space(c) && (marks.is_empty() || marks.contains(cs[c - 1].1)))
                .min_by_key(|&c| c.abs_diff(even))
        };
        let mut c = near(";:")
            .or_else(|| near(","))
            .or_else(|| near(""))
            .or_else(|| (1..n).filter(|&c| space(c)).min_by_key(|&c| c.abs_diff(even)))
            .unwrap_or(even.max(1));
        while c > 1 && len(rest[..cs[c].0].trim_end()) > limit {
            c = (1..c).rev().find(|&i| space(i)).unwrap_or(c - 1);
        }
        out.push(rest[..cs[c].0].trim_end().to_string());
        rest = rest[cs[c].0..].trim_start();
    }
    out.push(rest.to_string());
    out
}

fn join(a: &str, b: &str) -> String {
    format!("{a} {b}").trim().to_string()
}

pub fn chunks(chapters: &[Chapter], len: &dyn Fn(&str) -> usize) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::new();
    for (n, ch) in chapters.iter().enumerate() {
        let chunk = |text: String, pause, cut| Chunk {
            chapter: n,
            text,
            pause,
            cut,
        };
        out.push(chunk(ch.title.clone(), TITLE, false));
        let mut carry = String::new();
        for (k, para) in ch.paragraphs.iter().enumerate() {
            if para == SCENE {
                if !carry.is_empty() {
                    out.push(chunk(std::mem::take(&mut carry), PARAGRAPH, false));
                }
                if let Some(last) = out.last_mut() {
                    last.pause = SCENE_PAUSE;
                }
                continue;
            }
            let (mut cur, mut cut, mut inside) = (std::mem::take(&mut carry), false, false);
            for s in sentences(para) {
                let ps = pieces(&s, LIMIT, len);
                let last = ps.len() - 1;
                for (i, p) in ps.into_iter().enumerate() {
                    if !cur.is_empty() && len(&join(&cur, &p)) > LIMIT {
                        out.push(chunk(std::mem::replace(&mut cur, p), SENTENCE, inside));
                        cut = true;
                    } else {
                        cur = join(&cur, &p);
                    }
                    inside = i < last; // the chunk so far ends inside this sentence
                }
            }
            if !cur.is_empty() && !cut && cur.chars().count() < SHORT && k + 1 < ch.paragraphs.len()
            {
                carry = cur;
            } else if !cur.is_empty() {
                out.push(chunk(cur, PARAGRAPH, false));
            }
        }
        if let Some(last) = out.last_mut() {
            last.pause = if ch.paragraphs.is_empty() {
                OPENING
            } else {
                CHAPTER
            }; // the title alone
        }
    }
    out
}

/// The book's names, most used first, for Whisper: capitalised words inside a sentence ("Mad Hatter",
/// "Zo-ri-ba"); "And" after a comma or "Time" are words of the language, as the book uses them in lower case.
pub fn book_words(chapters: &[Chapter], n: usize) -> String {
    static NAME: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"[a-z,;] ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)").unwrap());
    static PLAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[a-z]+\b").unwrap());
    let paras = || chapters.iter().flat_map(|c| &c.paragraphs);
    let mut count: HashMap<&str, (usize, usize)> = HashMap::new(); // name: (uses, first seen)
    let mut plain = HashSet::new();
    for p in paras() {
        let mut at = 0;
        while let Some(c) = NAME.captures_at(p, at) {
            let m = c.get(1).unwrap();
            let next = count.len();
            count.entry(m.as_str()).or_insert((0, next)).0 += 1;
            at = m.end();
        }
        plain.extend(PLAIN.find_iter(p).map(|m| m.as_str()));
    }
    let mut names: Vec<_> = count.into_iter().collect();
    names.sort_by_key(|(_, (uses, first))| (std::cmp::Reverse(*uses), *first));
    let keep = names.into_iter().map(|(w, _)| w).filter(|w| {
        w.contains(' ') || w.chars().count() > 1 && !plain.contains(w.to_lowercase().as_str())
    });
    keep.take(n).collect::<Vec<_>>().join(", ")
}

/// Whisper's prompt for a chunk: of the book's names (book_words), those in `text` first, then the most
/// used, within `limit` characters (a long list is cut at its front by Whisper and dilutes the rest).
pub fn hint(names: &str, text: &str, limit: usize) -> String {
    let (own, rest): (Vec<&str>, Vec<&str>) = names
        .split(", ")
        .filter(|n| !n.is_empty())
        .partition(|n| text.contains(n));
    let mut out = String::new();
    for n in own.into_iter().chain(rest) {
        if !out.is_empty() && out.len() + 2 + n.len() > limit {
            break;
        }
        if !out.is_empty() {
            out += ", ";
        }
        out += n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> usize {
        s.chars().count()
    }

    #[test]
    fn a_long_sentence_is_cut_after_the_comma_near_its_middle() {
        let long = format!("{} stop, {}", "word ".repeat(9).trim(), "word ".repeat(15).trim()); // 125
        let p = pieces(&long, 100, &chars);
        assert_eq!(p[0], format!("{} stop,", "word ".repeat(9).trim()), "{p:?}");
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn a_long_sentences_pieces_but_the_last_end_inside_it() {
        let long = format!("{}, {}.", "a".repeat(300), "b ".repeat(200).trim()); // over LIMIT
        let c = chunks(&[ch("1", &[&long, "Next."])], &chars);
        let cut: Vec<bool> = c.iter().map(|c| c.cut).collect();
        assert_eq!(cut, [false, true, false, false], "{c:?}");
    }

    fn ch(title: &str, paras: &[&str]) -> Chapter {
        Chapter {
            title: title.into(),
            paragraphs: paras.iter().map(|p| p.to_string()).collect(),
        }
    }

    #[test]
    fn sentences_end_at_marks_but_not_after_titles_or_initials() {
        let s = sentences(
            "Mr. Baker met J. Smith. \"Go!\" he said. It was 4 p.m. and dark… Then rain.",
        );
        assert_eq!(
            s,
            [
                "Mr. Baker met J. Smith.",
                "\"Go!\" he said.",
                "It was 4 p.m. and dark…",
                "Then rain."
            ]
        );
    }

    #[test]
    fn a_long_sentence_is_cut_after_commas_then_between_words() {
        let long = format!("{}, {}", "a".repeat(200), "b ".repeat(60).trim());
        let p = pieces(&long, 280, &chars);
        assert!(p[0] == format!("{},", "a".repeat(200)) && p[1].starts_with("b b"));
        assert_eq!(p.concat().len() + 1, long.len());
        assert!(p.iter().all(|x| x.chars().count() <= 280) && p.len() == 2);
    }

    #[test]
    fn a_long_sentence_is_cut_into_even_pieces() {
        // not a full piece and a tail of two words ("of Wonderland are known."): about equal pieces
        let long = "word ".repeat(70).trim().to_string(); // 349 characters
        let p = pieces(&long, 280, &chars);
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| (150..=200).contains(&x.len())), "{p:?}");
        // the even length not on a word's end: still two pieces, no lone word (review)
        let p = pieces("word ".repeat(69).trim(), 280, &chars);
        assert_eq!(p.len(), 2, "{p:?}");
    }

    #[test]
    fn chunks_carry_pauses_scenes_and_short_paragraphs() {
        let book = [
            ch("Alice, by Lewis Carroll.", &[]),
            ch(
                "Chapter 1",
                &[
                    "Yes.",
                    "He came. She went.",
                    SCENE,
                    &"Long one. ".repeat(60),
                ],
            ),
        ];
        let c = chunks(&book, &chars);
        let got: Vec<(usize, &str, f32)> = c
            .iter()
            .map(|c| (c.chapter, &c.text[..c.text.len().min(12)], c.pause))
            .collect();
        assert_eq!(
            got[..4],
            [
                (0, "Alice, by Le", 2.0),
                (1, "Chapter 1", 1.2),
                (1, "Yes. He came", 1.5),
                (1, "Long one. Lo", 0.3)
            ]
        );
        assert_eq!(c.last().unwrap().pause, 3.0); // a chapter's end
        assert!(c.iter().all(|c| c.text.chars().count() <= LIMIT));
    }

    #[test]
    fn book_words_are_names_inside_sentences() {
        let book = [ch(
            "1",
            &[
                "He saw Zarl Baker, and Vasena. And then zarl. Then Zo-ri-ba and Zarl Baker; the Time, a time.",
            ],
        )];
        assert_eq!(book_words(&book, 60), "Zarl Baker, Vasena, Zo-ri-ba");
    }

    #[test]
    fn a_chunks_hint_is_its_own_names_first_within_the_limit() {
        let names = "Zarl Baker, Vasena, Zo-ri-ba, Ux";
        assert_eq!(hint(names, "He left Ux for Zo-ri-ba.", 100), "Zo-ri-ba, Ux, Zarl Baker, Vasena");
        assert_eq!(hint(names, "He left Ux for Zo-ri-ba.", 14), "Zo-ri-ba, Ux");
        assert_eq!(hint(names, "Nobody.", 10), "Zarl Baker");
        assert_eq!(hint("", "Nobody.", 10), "");
    }
}
