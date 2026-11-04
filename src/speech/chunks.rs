//! A book's chapters as chunks to speak: whole sentences, at most LIMIT (Kokoro's
//! 510 phonemes, as the caller's length counts them); a sentence longer than that is cut at a `;` or `:`
//! near its middle, else a comma, else a space, into pieces of about even length, each but the last
//! marked `cut` (assemble.rs joins it closer). Each chunk carries the pause after it: a sentence's, a paragraph's, a
//! scene break's, a chapter's, and after the book's title alone the opening's. A paragraph under SHORT
//! characters is spoken with the next (a lone word or two is spoken badly).
//! Sentences are told apart by rule, not by pysbd: an end mark (. ! ? …, closing quotes after it) before
//! a space and a capital, digit or quote, unless the word before is a title or an initial ("Mr.", "J.").
//! German: its quotes („“»«›‹) and capitals too. book_words() is for English, where a capital inside a
//! sentence tells a name; in German every noun has one (misprints::names).
//! Length is costly (misaki over the text), so a grown chunk's is the sum of its parts', and only one
//! within MARGIN of LIMIT is counted whole; chapters are cut on up to machine::CORES threads.
use crate::text::extract::{Chapter, SCENE};
use crate::text::language::Language;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

pub const LIMIT: usize = crate::speech::kokoro::MAX_PHONEMES;
/// af_bella at speed 1: a novel of 374,000 characters in 7:04 with its pauses
const CHARS_PER_SECOND: f32 = 16.0;
const SHORT: usize = 30;
const SENTENCE: f32 = 0.3;
const PARAGRAPH: f32 = 0.7;
const SCENE_PAUSE: f32 = 1.5;
const TITLE: f32 = 1.2;
const CHAPTER: f32 = 3.0; // before the next title
const OPENING: f32 = 2.0; // after the book's title; LEAD (assemble.rs) goes before it
/// A joined text's length differs from the sum of its parts' by less than this: measured on 4 books,
/// largest difference 1, the chunk lists identical to counting every text whole.
const MARGIN: usize = 100;
#[cfg(test)]
pub static DRIFT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The seconds --demo speaks, at each of its places.
pub const DEMO_SECONDS: f32 = 60.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub chapter: usize,
    pub text: String,
    /// seconds of silence after it
    pub pause: f32,
    /// it ends inside a sentence (a long sentence's piece but its last)
    pub cut: bool,
}

/// A chunk of `text` in chapter 0, for tests.
#[cfg(test)]
pub fn plain(text: &str) -> Chunk {
    Chunk {
        chapter: 0,
        text: text.into(),
        pause: 0.0,
        cut: false,
    }
}

const TITLES: [&str; 14] = [
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "capt", "col", "gen", "lt", "rev", "vs",
];

/// The sentences of a paragraph.
pub fn sentences(para: &str, language: Language) -> Vec<String> {
    static END: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"[.!?…]+["'”’)\]]*\s+["'“‘(\[]?[A-Z0-9]"#).unwrap());
    static END_DE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"[.!?…]+["'”’“«»›‹)\]]*\s+["'“‘„»«›‹(\[]?[A-ZÄÖÜ0-9]"#).unwrap()
    });
    let end: &Regex = if language == Language::German { &END_DE } else { &END };
    let mut out = Vec::new();
    let mut start = 0;
    for m in end.find_iter(para) {
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
/// point is counted in characters (phonemes run about one per character). Each piece with its length.
fn pieces(sentence: &str, limit: usize, len: &dyn Fn(&str) -> usize) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut rest = sentence.trim();
    loop {
        let whole = len(rest);
        if whole <= limit {
            out.push((rest.to_string(), whole));
            return out;
        }
        let cs: Vec<(usize, char)> = rest.char_indices().collect();
        let n = cs.len();
        let even = n / whole.div_ceil(limit);
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
        let mut head = len(rest[..cs[c].0].trim_end());
        while c > 1 && head > limit {
            c = (1..c).rev().find(|&i| space(i)).unwrap_or(c - 1);
            head = len(rest[..cs[c].0].trim_end());
        }
        out.push((rest[..cs[c].0].trim_end().to_string(), head));
        rest = rest[cs[c].0..].trim_start();
    }
}

fn join(a: &str, b: &str) -> String {
    format!("{a} {b}").trim().to_string()
}

pub fn chunks(
    chapters: &[Chapter],
    len: &(dyn Fn(&str) -> usize + Sync),
    language: Language,
) -> Vec<Chunk> {
    let cut = |n, ch: &Chapter| chapter(n, ch, len, language, MARGIN);
    crate::machine::in_order(chapters, cut).concat()
}

/// Would `cur` (of length about `cur_len`, the sum) and `p` joined pass LIMIT? Counted whole only within
/// `margin` of LIMIT; the sum decides outside it.
fn over(cur: &str, cur_len: usize, p: &str, p_len: usize, len: &dyn Fn(&str) -> usize, margin: usize) -> bool {
    let sum = cur_len + 1 + p_len;
    if sum.abs_diff(LIMIT) > margin {
        return sum > LIMIT;
    }
    let whole = len(&join(cur, p));
    #[cfg(test)]
    DRIFT.fetch_max(whole.abs_diff(sum), std::sync::atomic::Ordering::Relaxed);
    whole > LIMIT
}

/// Chapter `n`'s chunks.
fn chapter(
    n: usize,
    ch: &Chapter,
    len: &dyn Fn(&str) -> usize,
    language: Language,
    margin: usize,
) -> Vec<Chunk> {
    let chunk = |text: String, pause, cut| Chunk {
        chapter: n,
        text,
        pause,
        cut,
    };
    let mut out = vec![chunk(ch.title.clone(), TITLE, false)];
    let mut carry = (String::new(), 0);
    for (k, para) in ch.paragraphs.iter().enumerate() {
        if para == SCENE {
            if !carry.0.is_empty() {
                out.push(chunk(std::mem::take(&mut carry).0, PARAGRAPH, false));
            }
            if let Some(last) = out.last_mut() {
                last.pause = SCENE_PAUSE;
            }
            continue;
        }
        let ((mut cur, mut cur_len), mut cut, mut inside) = (std::mem::take(&mut carry), false, false);
        for s in sentences(para, language) {
            let ps = pieces(&s, LIMIT, len);
            let last = ps.len() - 1;
            for (i, (p, p_len)) in ps.into_iter().enumerate() {
                if cur.is_empty() {
                    (cur, cur_len) = (p, p_len);
                } else if over(&cur, cur_len, &p, p_len, len, margin) {
                    out.push(chunk(std::mem::replace(&mut cur, p), SENTENCE, inside));
                    cur_len = p_len;
                    cut = true;
                } else {
                    cur = join(&cur, &p);
                    cur_len += 1 + p_len;
                }
                inside = i < last; // the chunk so far ends inside this sentence
            }
        }
        if !cur.is_empty() && !cut && cur.chars().count() < SHORT && k + 1 < ch.paragraphs.len() {
            carry = (cur, cur_len);
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
    out
}

/// Every chunk by counting each joined text whole: the reference the sum is compared with.
#[cfg(test)]
pub fn chunks_counted(chapters: &[Chapter], len: &dyn Fn(&str) -> usize, language: Language) -> Vec<Chunk> {
    chapters.iter().enumerate().flat_map(|(n, ch)| chapter(n, ch, len, language, usize::MAX)).collect()
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
        .partition(|n| crate::speech::pronounce::holds(text, n));
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

/// About `seconds` of chunks at the start, the middle and the end, never overlapping: a book too short
/// for three parts gets its start and end, or its start.
pub fn demo_picks(cs: &[Chunk], seconds: f32) -> Vec<Vec<usize>> {
    let (mut k, mut total) = (0, 0.0);
    while k < cs.len() && total < seconds {
        total += cs[k].text.chars().count() as f32 / CHARS_PER_SECOND + cs[k].pause;
        k += 1;
    }
    let n = cs.len();
    let mut starts = vec![0];
    if n >= 3 * k {
        starts.push(n / 2 - k / 2);
    }
    if n >= 2 * k {
        starts.push(n - k);
    }
    starts.dedup(); // an empty book
    starts
        .into_iter()
        .map(|s| (s..(s + k).min(n)).collect())
        .collect()
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
        let p = piece_texts(&long, 100, &chars);
        assert_eq!(p[0], format!("{} stop,", "word ".repeat(9).trim()), "{p:?}");
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn a_long_sentences_pieces_but_the_last_end_inside_it() {
        let long = format!("{}, {}.", "a".repeat(300), "b ".repeat(200).trim()); // over LIMIT
        let c = chunks(&[ch("1", &[&long, "Next."])], &chars, Language::English);
        let cut: Vec<bool> = c.iter().map(|c| c.cut).collect();
        assert_eq!(cut, [false, true, false, false], "{c:?}");
    }

    fn piece_texts(s: &str, limit: usize, len: &dyn Fn(&str) -> usize) -> Vec<String> {
        let got = pieces(s, limit, len);
        assert!(got.iter().all(|(p, l)| *l == len(p)), "a piece's length is its own");
        got.into_iter().map(|(p, _)| p).collect()
    }

    #[test]
    fn the_sum_cuts_as_counting_each_joined_text_does(){
        // a length unlike a sum: each text counted costs 3 more, as a text's ends do in phonemes
        let len = |t: &str| t.chars().count() + 3;
        let para = "Word word word word word word word. ".repeat(60);
        let book = [ch("1", &[para.trim(), "Hi.", "Next one.", "Another."]), ch("2", &[])];
        assert_eq!(chunks(&book, &len, Language::English), chunks_counted(&book, &len, Language::English));
    }

    #[test]
    fn a_name_inside_a_word_is_not_the_chunks_own(){
        assert_eq!(hint("Tal, Ux", "Luxury and Ux.", 200), "Ux, Tal");
        assert_eq!(hint("Tal, Ux", "Luxury.", 200), "Tal, Ux");
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
            Language::English,
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
    fn german_sentences_end_before_their_quotes_and_umlauts() {
        let s = sentences("Er sah es. „Ja!“ Über ihm. »Nein.« Äpfel fielen.", Language::German);
        assert_eq!(s, ["Er sah es.", "„Ja!“", "Über ihm.", "»Nein.«", "Äpfel fielen."]);
    }

    #[test]
    fn a_long_sentence_is_cut_after_commas_then_between_words() {
        let long = format!("{}, {}", "a".repeat(200), "b ".repeat(60).trim());
        let p = piece_texts(&long, 280, &chars);
        assert!(p[0] == format!("{},", "a".repeat(200)) && p[1].starts_with("b b"));
        assert_eq!(p.concat().len() + 1, long.len());
        assert!(p.iter().all(|x| x.chars().count() <= 280) && p.len() == 2);
    }

    #[test]
    fn a_long_sentence_is_cut_into_even_piece_texts() {
        // not a full piece and a tail of two words ("of Wonderland are known."): about equal pieces
        let long = "word ".repeat(70).trim().to_string(); // 349 characters
        let p = piece_texts(&long, 280, &chars);
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| (150..=200).contains(&x.len())), "{p:?}");
        // the even length not on a word's end: still two pieces, no lone word (review)
        let p = piece_texts("word ".repeat(69).trim(), 280, &chars);
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
        let c = chunks(&book, &chars, Language::English);
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

    /// A chunk of this text, no pause.

    #[test]
    fn demo_picks_take_a_minute_at_start_middle_and_end() {
        let cs: Vec<Chunk> = (0..100).map(|_| plain(&"x".repeat(160))).collect(); // 10 s each
        let p = demo_picks(&cs, 60.0);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], (0..6).collect::<Vec<_>>());
        assert_eq!(p[1][0], 47);
        assert_eq!(p[2], (94..100).collect::<Vec<_>>());
        let few = demo_picks(&cs[..4], 60.0);
        assert_eq!(few, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn demo_parts_never_overlap() {
        let cs: Vec<Chunk> = (0..18).map(|_| plain(&"x".repeat(160))).collect(); // 10 s each, 6 to a part
        let range = |a: usize, b: usize| (a..b).collect::<Vec<_>>();
        assert_eq!(demo_picks(&cs[..10], 60.0), [range(0, 6)]);
        assert_eq!(demo_picks(&cs[..14], 60.0), [range(0, 6), range(8, 14)]);
        assert_eq!(demo_picks(&cs, 60.0), [range(0, 6), range(6, 12), range(12, 18)]);
    }
}
