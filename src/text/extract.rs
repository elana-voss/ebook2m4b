//! From an ebook to clean chapters of text: what is read aloud and nothing else.
//!
//! A book comes as a Book (book/mod.rs, from any of its readers). Chapters are cut where the book's table of
//! contents points (a file, or an anchor in one), else its own contents page, else lines like "Chapter
//! One", else each page; contents, copyright and similar pieces and short front matter are left out;
//! a scene break (<hr>, "* * *") becomes SCENE. Paragraphs cut by a page are joined, words in capitals
//! read as words. What the rules cannot settle a Judge is asked (judge.rs: a language model).
//! Each block's text goes through clean.rs, the chapters through misprints.rs. A German book has its own
//! patterns (the _DE statics; each also knows "Cover" and "Copyright") and its own words in a made title.
use crate::model::judge::Judge;
use crate::text::clean::{clean_in, roman};
use crate::text::language::Language;
use crate::book::{self, Book, Elem, Node, re};
use anyhow::Result;
use regex::Regex;
use std::sync::LazyLock;

/// A paragraph that stands for a scene break (<hr> or "* * *").
pub const SCENE: &str = "⁂";
const BLOCKS: [&str; 14] = [
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "div",
    "pre",
    "dd",
    "dt",
    "td",
];
/// characters: a chapter shorter than this before the first longer one is front matter
const FRONT: usize = 1500;

static SKIP: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)^(table of )?contents$|^copyright|^index$|^cover$|^title page$|^also by\b|^(list of )?illustrations$|project gutenberg",
    )
});
static HEADING: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)^(chapter|prologue|epilogue|prolegomena|book|part)\b.{0,80}$"));
/// the same, of a paragraph in a chapter (SKIP is a chapter's title): a line that only names such a page or notice, not a sentence
/// that starts with its word ("Copyright is theft, he said."). Numbers are words by then; a year's start
/// tells it, or the sign, or no sentence's end.
static NOTICE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)^(table of )?contents$|^index$|^cover$|^title page$|^(list of )?illustrations$|^copyright\w*\b.*(©|\(c\)|\b\d{4}\b|\b(nineteen|twenty|two thousand)\b)|^copyright\w*\b[^.!?]*$|^also by\b[^.!?]*$|^[^.!?]*project gutenberg[^.!?]*$",
    )
});
/// a scan's page mark and nothing else, numbers already words: "Page", "page iv", "End of page three of two hundred"
static PAGE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)^(end of )?pa+g+e(\s+([ivxlc]+|\d+|((one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(thir|four|fif|six|seven|eigh|nine)teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and|of)[\s-]*)+))?\W*$",
    )
});
static SKIP_DE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)^(inhalt|inhaltsverzeichnis|inhaltsübersicht|impressum|cover|titelseite|titelblatt|titel|index)$|^copyright|^auch von\b|^von (demselben|derselben)\b|proje[ck]t gutenberg",
    )
});
/// The ordinal words of "Erstes Kapitel", "Zwanzigster Teil".
const ORDINAL_DE: &str = r"(erst|zweit|dritt|viert|fünft|sechst|siebt|siebent|acht|neunt|zehnt|elft|zwölft|\p{L}+zehnt|\p{L}*zigst|\p{L}*ßigst|\p{L}*hundertst)(es|er|e)";
static HEADING_DE: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r"(?i)^((kapitel|prolog|epilog|vorwort|nachwort|einleitung|buch|teil)\b|{ORDINAL_DE}\s+(kapitel|teil|buch)\b).{{0,80}}$"
    ))
});
static NOTICE_DE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)^(inhalt|inhaltsverzeichnis|impressum|cover|titelseite|titelblatt|index)$|^copyright\w*\b.*(©|\(c\)|\b\d{4}\b|\b(neunzehn|zwanzig|zweitausend)\w*)|^copyright\w*\b[^.!?]*$|^auch von\b[^.!?]*$|^alle rechte vorbehalten\W*$|^[^.!?]*proje[ck]t gutenberg[^.!?]*$",
    )
});
static PAGE_DE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"(?i)^(ende von )?seite(\s+([ivxlc]+|\d+|((ein|zwei|drei|vier|fünf|sechs|sech|sieben|sieb|acht|neun|zehn|elf|zwölf|zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig|hundert|tausend|und|s|e)+(\s+von\s+)?)+))?\W*$",
    )
});
static GUTENBERG_END: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)^\W*end of (the )?project gutenberg"));
static BREAK: LazyLock<Regex> =
    LazyLock::new(|| re(r"^(?:\s*[*#~⁂](\s*[*#~⁂])*\s*|\s*[-_=]{3,}\s*)$"));
static CAPS_RUN: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b[A-ZÄÖÜ][A-ZÄÖÜß'’]*(?:\s+[A-ZÄÖÜ][A-ZÄÖÜß'’]*)+\b"));
static CAPS_WORD: LazyLock<Regex> = LazyLock::new(|| re(r"\b[A-ZÄÖÜ][A-ZÄÖÜß'’]+\b"));

/// The pattern of the book's language.
fn of(language: Language, english: &'static LazyLock<Regex>, german: &'static LazyLock<Regex>) -> &'static Regex {
    match language {
        Language::English => english,
        Language::German => german,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    pub title: String,
    pub paragraphs: Vec<String>,
}

fn len(s: &str) -> usize {
    s.chars().count()
}

fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The book's author(s) as its EPUB names them, or "".
pub fn author(book: &Book) -> String {
    book.creators.join(" & ")
}

/// The first thing heard, "Alice's Adventures in Wonderland, by Lewis Carroll.": the EPUB's title, else fallback.
pub fn opening(book: &Book, fallback: &str, language: Language) -> Chapter {
    let title = clean_in(book.titles.first().map_or(fallback, |t| t.as_str()), language);
    let by = author(book);
    let word = if language == Language::German { "von" } else { "by" };
    let title = if by.is_empty() {
        format!("{title}.")
    } else {
        format!("{title}, {word} {by}.")
    };
    Chapter {
        title,
        paragraphs: Vec::new(),
    }
}

/// A scene break line: "* * *", "#", "-----".
fn is_break(text: &str) -> bool {
    BREAK.is_match(text)
}

/// Sup, note references, asides, scripts and styles dropped; a line break a space.
fn prune(e: &mut Elem) {
    e.children.retain(|c| match c {
        Node::Elem(k) => {
            !matches!(k.name.as_str(), "sup" | "aside" | "script" | "style")
                && !(k.name == "a" && k.attr("epub:type") == Some("noteref"))
        }
        Node::Text(_) => true,
    });
    for c in &mut e.children {
        if let Node::Elem(k) = c {
            if k.name == "br" {
                *c = Node::Text(" ".into());
            } else {
                prune(k);
            }
        }
    }
}

fn has_block(e: &Elem) -> bool {
    e.children
        .iter()
        .any(|c| matches!(c, Node::Elem(k) if BLOCKS.contains(&k.name.as_str()) || has_block(k)))
}

/// The first string inside and whether it lies directly in `e`.
fn first_string(e: &Elem) -> Option<(&str, bool)> {
    for c in &e.children {
        match c {
            Node::Text(t) => return Some((t, true)),
            Node::Elem(k) => {
                if let Some((t, _)) = first_string(k) {
                    return Some((t, false));
                }
            }
        }
    }
    None
}

/// (ids seen since the last block, text) for each innermost block of the page.
fn blocks(body: &Elem, language: Language) -> Vec<(Vec<String>, String)> {
    let mut body = body.clone();
    prune(&mut body);
    let mut all = Vec::new();
    body.descendants(&mut all);
    let mut out = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    for el in all {
        if let Some(id) = el.attr("id").filter(|i| !i.is_empty()) {
            ids.push(id.into());
        }
        let block = BLOCKS.contains(&el.name.as_str());
        if el.name == "hr" || block && is_break(&el.text("")) {
            out.push((std::mem::take(&mut ids), SCENE.to_string()));
        } else if block && !has_block(el) {
            let mut v = Vec::new();
            el.strings(&mut v);
            let mut text: String = v.concat();
            if let Some((first, direct)) = first_string(el) {
                let t = first.trim();
                if !direct && len(t) == 1 {
                    text = format!("{t}{}", &text[first.len()..]); // a drop cap and its spacer
                }
            }
            let text = clean_in(&text, language);
            if !text.is_empty() {
                let mut inner = Vec::new();
                el.descendants(&mut inner);
                ids.extend(inner.iter().filter_map(|d| d.attr("id")).map(String::from)); // a converted book puts a heading's anchor inside it
                out.push((std::mem::take(&mut ids), text));
            }
        }
    }
    out
}

/// A title as compared: "CHAPTER I." and "Chapter I" are the same.
fn key(title: &str, language: Language) -> String {
    clean_in(title, language)
        .to_lowercase()
        .trim_matches([' ', '.', ':'])
        .to_string()
}

/// The links of the book's own contents page: the page linking to the most other pages.
fn own_contents(book: &Book) -> Vec<(String, String)> {
    let mut best = Vec::new();
    for page in book.spine.iter().filter(|p| p.is_doc) {
        let Some(body) = &page.body else { continue };
        let mut all = Vec::new();
        body.descendants(&mut all);
        let links: Vec<(String, String)> = all
            .iter()
            .filter(|a| a.name == "a")
            .filter_map(|a| Some((a.text(" ").trim().to_string(), a.attr("href")?.to_string())))
            .filter(|(_, h)| {
                let n = book::base_name(h.split('#').next().unwrap_or(""));
                n != page.name && book.spine.iter().any(|p| p.name == n)
            })
            .collect();
        if links.len() > best.len() {
            best = links;
        }
    }
    best
}

type Block = (String, Vec<String>, String); // (file, ids, text)

/// (block index, title) of each entry found, in reading order.
fn resolve(entries: &[(String, String)], blocks: &[Block]) -> Vec<(usize, String)> {
    let mut starts: Vec<(usize, String)> = Vec::new();
    for (title, href) in entries {
        let (name, anchor) = href.split_once('#').unwrap_or((href, ""));
        let name = book::base_name(name);
        let at = blocks.iter().position(|(f, ids, _)| {
            f == name && (anchor.is_empty() || ids.iter().any(|i| i == anchor))
        });
        if let Some(at) = at
            && starts.last().is_none_or(|s| at > s.0)
        {
            starts.push((at, title.clone()));
        }
    }
    starts
}

/// Where chapters start: the TOC, else the book's own contents page, else lines like "Chapter One"
/// (each confirmed by the judge, if any: "Part of me knew" is text), else each page (titled by its
/// first line if short).
fn starts(
    book: &Book,
    contents: &[(String, String)],
    blocks: &[Block],
    judge: Option<&dyn Judge>,
    notes: &mut Vec<String>,
    language: Language,
) -> Result<Vec<(usize, String)>> {
    let mut tries = vec![
        resolve(&book.toc, blocks),
        resolve(contents, blocks),
    ];
    if tries.iter().all(|t| t.len() < 3) {
        let mut heads: Vec<(usize, String)> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| of(language, &HEADING, &HEADING_DE).is_match(&b.2))
            .map(|(i, b)| (i, b.2.clone()))
            .collect();
        if let Some(j) = judge
            && !heads.is_empty()
        {
            let asked: Vec<(String, String)> = heads
                .iter()
                .map(|(i, t)| {
                    (
                        t.clone(),
                        blocks.get(i + 1).map_or(String::new(), |b| head(&b.2, 200)),
                    )
                })
                .collect();
            let oks = j.headings(&asked)?;
            let mut kept = Vec::new();
            for (h, ok) in heads
                .into_iter()
                .zip(oks.iter().chain(std::iter::repeat(&true)))
            {
                if *ok {
                    kept.push(h);
                } else {
                    notes.push(format!("Not a heading: {} (the language model)", h.1));
                }
            }
            heads = kept;
        }
        tries.push(heads);
    }
    let found = match tries.iter().position(|t| t.len() >= 3) {
        Some(i) => tries.swap_remove(i),
        None => {
            let mut best = 0; // the first of the longest
            for (i, t) in tries.iter().enumerate() {
                if t.len() > tries[best].len() {
                    best = i;
                }
            }
            tries.swap_remove(best)
        }
    };
    if !found.is_empty() {
        return Ok(found);
    }
    let part = if language == Language::German { "Teil" } else { "Part" };
    let pages = (0..blocks.len()).filter(|&i| i == 0 || blocks[i].0 != blocks[i - 1].0);
    Ok(pages
        .enumerate()
        .map(|(n, i)| {
            (
                i,
                if len(&blocks[i].2) < 80 {
                    blocks[i].2.clone()
                } else {
                    format!("{part} {}", n + 1)
                },
            )
        })
        .collect())
}

/// "THE LAKE" -> "The Lake".
fn capitalize(w: &str) -> String {
    let mut c = w.chars();
    c.next().map_or(String::new(), |f| {
        f.to_uppercase()
            .chain(c.flat_map(char::to_lowercase))
            .collect()
    })
}

/// Words in capitals, two or more in a row, read as words: "THE TALE BEGINS" -> "The Tale Begins"
/// (Kokoro spells a word in capitals letter by letter); one alone, "FBI", stays.
fn capitals(text: &str) -> String {
    CAPS_RUN
        .replace_all(text, |m: &regex::Captures| {
            CAPS_WORD
                .replace_all(&m[0], |w: &regex::Captures| capitalize(&w[0]))
                .into_owned()
        })
        .into_owned()
}

/// Joins paragraphs cut by a page, drops contents lines, links and empty ones, keeps one SCENE
/// between text, reads lines in capitals as words. A line equal to a title goes, unless it ends a sentence
/// with no title beside it: "Silence." in a chapter titled Silence is text.
fn tidy(paras: &[String], titles: &std::collections::HashSet<String>, language: Language) -> Vec<String> {
    static LINK: LazyLock<Regex> = LazyLock::new(|| re(r"www\.|https?://"));
    static CUT: LazyLock<Regex> = LazyLock::new(|| re(r"[a-z]-$"));
    static ENDS: LazyLock<Regex> = LazyLock::new(|| re(r#"[.!?:;"'’”)\]—-]$"#));
    static CUT_DE: LazyLock<Regex> = LazyLock::new(|| re(r"[a-zäöüß]-$"));
    static ENDS_DE: LazyLock<Regex> = LazyLock::new(|| re(r#"[.!?:;"'’”“«‹)\]—–-]$"#)); // „ » › open a quote
    let (cut, ends) = (of(language, &CUT, &CUT_DE), of(language, &ENDS, &ENDS_DE));
    let is_title = |p: &String| titles.contains(&key(p.trim(), language));
    let mut joined: Vec<String> = Vec::new();
    for (i, t) in paras.iter().enumerate() {
        let t = t.trim().to_string();
        let n = len(&t);
        if t.is_empty()
            || LINK.is_match(&t)
            || n < 60 && of(language, &PAGE, &PAGE_DE).is_match(&t)
            || is_title(&t)
                && !(t.ends_with(['.', '!', '?'])
                    && !paras[..i].last().is_some_and(is_title)
                    && !paras.get(i + 1).is_some_and(is_title))
            || n < 40 && of(language, &NOTICE, &NOTICE_DE).is_match(&t)
        {
            continue;
        }
        let lower = t.chars().next().is_some_and(char::is_lowercase);
        let after_text = joined.last().is_some_and(|l| l != SCENE);
        match joined.last_mut() {
            Some(last) if lower && cut.is_match(last) => last.push_str(&t), // a word cut at a line end
            Some(last) if lower && !ends.is_match(last) => {
                last.push(' ');
                last.push_str(&t);
            }
            _ if t != SCENE || after_text => joined.push(capitals(&t)),
            _ => {}
        }
    }
    while joined.last().is_some_and(|l| l == SCENE) {
        joined.pop();
    }
    joined
}

/// A chapter title as read: "12 THE LAKE" -> "Chapter twelve The Lake"; a word in capitals is a word of
/// the title, also alone ("ANNA"; Kokoro would spell it).
fn title(t: &str, language: Language) -> String {
    static BARE: LazyLock<Regex> = LazyLock::new(|| re(r"^(?:\d+\b|([IVXLC]+)(?:[.:]|$))"));
    static ROMAN: LazyLock<Regex> =
        LazyLock::new(|| re(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"));
    static NAMES: LazyLock<Regex> =
        LazyLock::new(|| re(r"\b(CHAPTER|BOOK|PART|VOLUME|PROLOGUE|EPILOGUE)\b"));
    let mut t = t.trim().to_string();
    let bare = BARE.captures(&t).is_some_and(|c| c.get(1).is_none_or(|r| ROMAN.is_match(r.as_str())));
    static COUNTED: LazyLock<Regex> =
        LazyLock::new(|| re(r"(?i)^(?:\d+|([IVXLC]+))\.?\s+((?:kapitel|teil|buch)\b.*)"));
    if bare && language == Language::English {
        t = format!("Chapter {t}"); // a bare number starts it
    } else if bare {
        t = match COUNTED.captures(&t) {
            None => format!("Kapitel {t}"), // not "1. Kapitel"
            Some(c) => match c.get(1) {
                Some(numeral) => format!("{}. {}", roman(numeral.as_str()), &c[2]), // "II. Kapitel" as "2. Kapitel"
                None => t.clone(),
            },
        };
    }
    let t = clean_in(&t, language); // Roman numerals to words
    let t = CAPS_WORD
        .replace_all(&t, |m: &regex::Captures| {
            if ROMAN.is_match(&m[0]) {
                m[0].to_string() // a Roman numeral clean left ("Chapter II The Lake")
            } else {
                capitalize(&m[0])
            }
        })
        .into_owned();
    NAMES
        .replace_all(&t, |m: &regex::Captures| capitalize(&m[0]))
        .into_owned()
}

/// The text of the book's pages, in order, without Project Gutenberg's licence after the book.
fn all_blocks(book: &Book, language: Language) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    for page in book.spine.iter().filter(|p| p.is_doc) {
        if let Some(body) = &page.body {
            blocks.extend(
                self::blocks(body, language)
                    .into_iter()
                    .map(|(ids, t)| (page.name.clone(), ids, t)),
            );
        }
    }
    if let Some(end) = blocks.iter().position(|b| GUTENBERG_END.is_match(&b.2)) {
        blocks.truncate(end);
    }
    blocks
}

/// The language of the book's text; None when it is in neither. Very common words tell it, so the text
/// cleaned as English does.
pub fn language(book: &Book) -> Option<Language> {
    crate::text::language::detect(all_blocks(book, Language::English).iter().map(|b| b.2.as_str()))
}

/// The book's chapters as read. judge: asked what the rules cannot tell with confidence; without one
/// the rules decide alone. series: the invented words of the books read before, left as they are.
/// notes: a line per piece left out, heading refused and word mended, for the reader.
pub fn chapters(
    book: &Book,
    judge: Option<&dyn Judge>,
    series: &std::collections::HashSet<String>,
    notes: &mut Vec<String>,
    language: Language,
) -> Result<Vec<Chapter>> {
    let out = unmended(book, judge, notes, language)?;
    crate::text::misprints::mend(out, judge, series, notes, language)
}

fn unmended(
    book: &Book,
    judge: Option<&dyn Judge>,
    notes: &mut Vec<String>,
    language: Language,
) -> Result<Vec<Chapter>> {
    let key = |t: &str| key(t, language);
    let blocks = all_blocks(book, language);
    let contents = own_contents(book);
    let starts = starts(book, &contents, &blocks, judge, notes, language)?;
    let titles: std::collections::HashSet<String> = book
        .toc
        .iter()
        .chain(contents.iter())
        .map(|(t, _)| key(t))
        .chain(starts.iter().map(|(_, t)| key(t)))
        .collect();
    let mut out = Vec::new();
    for (n, (at, t)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map_or(blocks.len(), |s| s.0);
        if of(language, &SKIP, &SKIP_DE).is_match(t) {
            notes.push(format!(
                "Left out: {t} (a contents, copyright or similar page)"
            ));
            continue;
        }
        let mut paras: Vec<String> = blocks[*at..end].iter().map(|b| b.2.clone()).collect();
        // the title as its first blocks hold it, whole ("CHAPTER I", "The Lake"), is not read again
        let mut rest = key(t);
        let mut n = 0;
        while let Some(p) = paras.get(n).map(|p| key(p)).filter(|p| !p.is_empty()) {
            let Some(r) = rest.strip_prefix(p.as_str()) else {
                break;
            };
            rest = r.trim_start_matches([' ', '.', ':']).to_string();
            n += 1;
            if rest.is_empty() {
                paras.drain(..n);
                break;
            }
        }
        let paras = tidy(&paras, &titles, language);
        if !paras.is_empty() {
            out.push(Chapter {
                title: title(t, language),
                paragraphs: paras,
            });
        }
    }
    let size = |c: &Chapter| c.paragraphs.iter().map(|p| len(p)).sum::<usize>();
    // Short pieces before the first long chapter are front matter (title page, blurb, a list of books).
    let first = out.iter().position(|c| size(c) >= FRONT).unwrap_or(0);
    notes.extend(
        out.drain(..first)
            .map(|c| format!("Left out: {} (short, before the first chapter)", c.title)),
    );
    if let Some(j) = judge {
        // the first and last pieces left: front and back matter (adverts), or the book?
        let mut ends: Vec<usize> = (0..out.len().min(6))
            .chain(out.len().saturating_sub(4)..out.len())
            .collect();
        ends.sort_unstable();
        ends.dedup();
        let asked: Vec<(String, String)> = ends
            .iter()
            .map(|&i| {
                (
                    out[i].title.clone(),
                    head(&out[i].paragraphs.join(" "), 400),
                )
            })
            .collect();
        let read = j.read(&asked)?;
        static CHAPTER: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)^(chapter|book|part)\b"));
        static CHAPTER_DE: LazyLock<Regex> =
            LazyLock::new(|| re(&format!(r"(?i)^(kapitel|buch|teil|{ORDINAL_DE}\s+(kapitel|teil|buch))\b")));
        // The model may drop only a short piece not titled as a chapter: E2B called whole chapters front matter.
        let mut kept = Vec::new();
        for (i, c) in out.into_iter().enumerate() {
            let asked_no = ends
                .iter()
                .position(|&e| e == i)
                .is_some_and(|k| !read.get(k).copied().unwrap_or(true));
            if asked_no && size(&c) < FRONT && !of(language, &CHAPTER, &CHAPTER_DE).is_match(&c.title) {
                notes.push(format!(
                    "Left out: {} (the language model: not read aloud)",
                    c.title
                ));
            } else {
                kept.push(c);
            }
        }
        out = kept;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A judge that changes nothing.
    struct Unchanged;

    impl Judge for Unchanged {
        fn headings(&self, items: &[(String, String)]) -> Result<Vec<bool>> {
            Ok(vec![true; items.len()])
        }
        fn read(&self, items: &[(String, String)]) -> Result<Vec<bool>> {
            Ok(vec![true; items.len()])
        }
        fn misprints(
            &self,
            cases: &[(String, String, Option<String>)],
            _: &[String],
        ) -> Result<std::collections::HashMap<String, String>> {
            Ok(cases.iter().map(|c| (c.0.clone(), c.0.clone())).collect())
        }
    }

    fn page(name: &str, inner: &str) -> book::Page {
        let x = format!(
            r#"<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>{inner}</body></html>"#
        );
        book::Page {
            name: name.into(),
            body: book::body(&x).unwrap(),
            is_doc: true,
        }
    }

    fn texts(inner: &str) -> Vec<String> {
        blocks(page("a", inner).body.as_ref().unwrap(), Language::English)
            .into_iter()
            .map(|b| b.1)
            .collect()
    }

    #[test]
    fn blocks_drop_notes_mend_drop_caps_and_mark_breaks() {
        let t = texts(
            "\n<p>One<sup>1</sup> two<br/>three</p>\n<hr/>\n<p><span>W </span>hen</p>\n<p>* * *</p>\n<div><p>in</p></div>\n",
        );
        assert_eq!(t, ["One two three", SCENE, "When", SCENE, "in"]);
        assert_eq!(texts("<p><span>T</span><span>he</span></p>"), ["The"]);
    }

    #[test]
    fn only_a_page_mark_is_dropped_as_one() {
        for mark in ["Page", "page twelve", "PAGE iv", "End of page three of two hundred."] {
            assert!(tidy(&[mark.to_string()], &Default::default(), Language::English).is_empty(), "{mark}");
        }
        let line = "Page by page, she read it.".to_string();
        assert_eq!(tidy(std::slice::from_ref(&line), &Default::default(), Language::English), [line]);
    }

    #[test]
    fn only_a_copyright_notice_is_dropped_as_one() {
        for notice in [
            "Copyright",
            "Copyright page",
            "Copyright nineteen twenty-three.",
            "Copyright © the author.",
            "Copyright (c) the author.",
            "Also by the author",
            "The Project Gutenberg eBook",
        ] {
            assert!(tidy(&[notice.to_string()], &Default::default(), Language::English).is_empty(), "{notice}");
        }
        for line in [
            "Copyright is theft, he said.",
            "Also by the lake, she sat.",
            "He read Project Gutenberg.",
        ] {
            assert_eq!(tidy(&[line.to_string()], &Default::default(), Language::English), [line]);
        }
    }

    #[test]
    fn ids_go_with_the_next_block() {
        let b = blocks(
            page(
                "a",
                "\n<a id=\"x\"/>\n<h2><a id=\"c1\"/>One</h2>\n<p>t</p>\n",
            )
            .body
            .as_ref()
            .unwrap(),
            Language::English,
        );
        assert_eq!(b[0].0, ["x", "c1"]);
        assert_eq!(b[1].0, ["c1"]); // the anchor seen again after the heading
    }

    #[test]
    fn a_title_said_in_the_text_stays_and_a_contents_line_goes() {
        let p = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let titles: std::collections::HashSet<String> =
            ["silence", "the road", "caught!", "the chain"].map(String::from).into();
        let t = tidy(&p(&["He listened.", "Silence.", "It went on."]), &titles, Language::English);
        assert_eq!(t, ["He listened.", "Silence.", "It went on."]);
        // no end mark: a title, wherever it stands
        assert_eq!(tidy(&p(&["He listened.", "Silence", "It went on."]), &titles, Language::English), ["He listened.", "It went on."]);
        // beside other titles: a contents page, end mark or not
        assert!(tidy(&p(&["The Road", "Caught!", "The Chain"]), &titles, Language::English).is_empty());
    }

    #[test]
    fn tidy_joins_cut_paragraphs_and_keeps_one_scene() {
        let p = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let t = tidy(
            &p(&[
                "THE TALE BEGINS here",
                "he went",
                "on- ",
                "ward",
                SCENE,
                SCENE,
                "Page 12",
                "X",
                SCENE,
            ]),
            &Default::default(),
            Language::English,
        );
        assert_eq!(t, ["The Tale Begins here he went on-ward", SCENE, "X"]);
    }

    #[test]
    fn titles_read_as_words() {
        assert_eq!(title("12 THE LAKE", Language::English), "Chapter twelve The Lake");
        assert_eq!(title("CHAPTER One", Language::English), "Chapter One");
        assert_eq!(title("IV.", Language::English), "Chapter four.");
        assert_eq!(title("CIVIL.", Language::English), "Civil."); // only IVXLC, yet no numeral
        assert_eq!(title("XL:", Language::English), "Chapter forty:");
        // a lone word in capitals is a title's word, not an acronym (Kokoro would spell "ANNA")
        assert_eq!(title("Chapter Six: ANNA", Language::English), "Chapter Six: Anna");
        assert_eq!(title("30 HOLT’S PLAN", Language::English), "Chapter thirty Holt’s Plan");
        // a Roman numeral clean leaves is no word (review)
        assert_eq!(title("Chapter II The Lake", Language::English), "Chapter II The Lake");
    }

    #[test]
    fn german_titles_read_in_german() {
        assert_eq!(title("12 DER SEE", Language::German), "Kapitel zwölf Der See");
        let once = title("1. Kapitel", Language::German);
        assert_eq!(once.to_lowercase().matches("kapitel").count(), 1, "{once}");
        assert_eq!(title("II. Kapitel", Language::German), "Zweites Kapitel");
        assert_eq!(title("III. Teil", Language::German), "Dritter Teil");
        assert_eq!(title("IV.", Language::German), "Kapitel vier.");
        assert_eq!(title("1 DER SEE", Language::German), "Kapitel eins Der See");
    }

    #[test]
    fn german_chapters_are_found_by_their_headings() {
        let paragraph = "Es war ein langer Tag, und sie gingen nach Hause. ".repeat(4);
        let inner = format!(
            "<p>Erstes Kapitel</p><p>{paragraph}</p><p>12. Kapitel</p><p>{paragraph}</p><p>Dreizehntes Kapitel</p><p>{paragraph}</p>"
        );
        let chs = chapters(
            &book(vec![page("1", &inner)], &[]),
            None,
            &Default::default(),
            &mut Vec::new(),
            Language::German,
        )
        .unwrap();
        assert_eq!(chs.len(), 3, "{:?}", chs.iter().map(|c| &c.title).collect::<Vec<_>>());
        assert_eq!(chs[0].title, "Erstes Kapitel");
    }

    #[test]
    fn german_contents_and_page_numbers_are_left_out() {
        let spine = vec![
            page("1", "<p>Impressum</p><p>Verlag und Druck in Berlin.</p>"),
            page("2", "<p>Der See</p><p>Sie ging.</p><p>Seite 12</p><p>Er kam.</p>"),
        ];
        let toc = [("Impressum", "1.xhtml"), ("Der See", "2.xhtml")];
        let chs = chapters(
            &book(spine, &toc),
            None,
            &Default::default(),
            &mut Vec::new(),
            Language::German,
        )
        .unwrap();
        assert_eq!(chs.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(), ["Der See"]);
        assert_eq!(chs[0].paragraphs, ["Sie ging.", "Er kam."]);
    }

    #[test]
    fn german_paragraphs_cut_by_a_page_are_joined_also_at_an_opening_quote() {
        let p = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for open in ["„", "»", "›"] {
            let t = tidy(&p(&[&format!("Sie sagte {open}"), "ja und ging"]), &Default::default(), Language::German);
            assert_eq!(t.len(), 1, "{t:?}");
        }
        for close in ["“", "«", "‹"] {
            let t = tidy(&p(&[&format!("Sie sagte {close}"), "und ging"]), &Default::default(), Language::German);
            assert_eq!(t.len(), 2, "{t:?}");
        }
        let t = tidy(&p(&["er ging", "nach Hause."]), &Default::default(), Language::German);
        assert_eq!(t, ["er ging nach Hause."]);
    }

    #[test]
    fn capitals_keep_their_apostrophes() {
        assert_eq!(capitals("HOLT’S PLAN and FBI"), "Holt’s Plan and FBI");
    }

    fn book(spine: Vec<book::Page>, toc: &[(&str, &str)]) -> Book {
        let toc = toc
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        Book {
            spine,
            toc,
            creators: vec!["A".into(), "B".into()],
            titles: vec!["T".into()],
        }
    }

    #[test]
    fn a_title_made_of_the_first_blocks_is_not_read_twice() {
        let long = "word ".repeat(400);
        let spine = vec![page(
            "1.xhtml",
            &format!("<h2>ONE</h2><h3>The Lake</h3><p>{long}</p><p>ONE</p>"),
        )];
        let toc = [("ONE. The Lake", "1.xhtml")];
        let chs = chapters(
            &book(spine, &toc),
            None,
            &Default::default(),
            &mut Vec::new(),
            Language::English,
        )
        .unwrap();
        assert_eq!(chs[0].paragraphs.len(), 2);
        assert_eq!(chs[0].paragraphs[1], "ONE");
    }

    #[test]
    fn chapters_follow_the_contents_and_drop_front_matter() {
        let long = "word ".repeat(400);
        let spine = vec![
            page("t.xhtml", "<p>By the author</p>"),
            page("c.xhtml", "<h1>Contents</h1><p>x</p>"),
            page("1.xhtml", &format!("<h1>Chapter 1</h1><p>{long}</p>")),
            page("2.xhtml", "<h1 id=\"b\">Two</h1><p>short</p>"),
        ];
        let toc = [
            ("Front", "t.xhtml"),
            ("Contents", "c.xhtml"),
            ("Chapter 1", "1.xhtml"),
            ("Two", "2.xhtml#b"),
        ];
        let mut notes = Vec::new();
        let chs = chapters(
            &book(spine, &toc),
            Some(&Unchanged),
            &Default::default(),
            &mut notes,
            Language::English,
        )
        .unwrap();
        assert_eq!(
            chs.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(),
            ["Chapter one", "Two"]
        );
        assert_eq!(chs[1].paragraphs, ["short"]);
        assert_eq!(
            notes,
            [
                "Left out: Contents (a contents, copyright or similar page)",
                "Left out: Front (short, before the first chapter)"
            ]
        );
        assert_eq!(opening(&book(vec![], &[]), "f", Language::English).title, "T, by A & B.");
        assert_eq!(opening(&book(vec![], &[]), "f", Language::German).title, "T, von A & B.");
    }

    #[test]
    fn without_contents_headings_then_pages() {
        let spine = vec![page(
            "1",
            "<p>Chapter One</p><p>a</p><p>Chapter Two</p><p>b</p><p>Chapter Three</p><p>c</p>",
        )];
        let chs = chapters(
            &book(spine, &[]),
            None,
            &Default::default(),
            &mut Vec::new(),
            Language::English,
        )
        .unwrap();
        assert_eq!(chs.len(), 3);
        let spine = vec![
            page("1", "<p>First</p><p>a</p>"),
            page("2", "<p>Second</p><p>b</p>"),
        ];
        let chs = chapters(
            &book(spine, &[]),
            None,
            &Default::default(),
            &mut Vec::new(),
            Language::English,
        )
        .unwrap();
        assert_eq!(
            chs.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(),
            ["First", "Second"]
        );
    }
}
