//! TXT and HTML read as a Book, and book(): one HTML document (theirs, or the one fb2.rs or
//! docx.rs makes) cut into pages through html.rs, as extract.rs reads them.
//!
//! book(): a page ends at each <mbp:pagebreak/> (or <pagebreak/>) and before each element a link of
//! the document points at ("#id", or <a name>), so that the book's own contents page links other
//! pages; not inside a paragraph, heading or list item, nor while the page has no text. Each h1-h3
//! is an entry of the contents (an id given where it has none), one right after another (no text
//! between) joins the entry's title ("CHAPTER I. The Lake"); links "#id" name the page.
//! Text bytes: UTF-8 (a BOM dropped), UTF-16 with a BOM, else cp1252.
//! TXT: paragraphs are the lines between blank lines, joined; a file without blank lines has one per
//! line, or, where lines are indented, one from each indented line on. No contents: extract.rs finds
//! lines like "Chapter One". HTML: its <title> and <meta name="author">.
use crate::book::{Book, Page, body};
use crate::book::soup::{Tok, Xhtml, tokens, unescape};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A title's lines as one: ". " between them where a line ends in a letter or digit (FB2's title
/// lines, a heading right after another).
pub fn join_title(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ if a.ends_with(|c: char| !c.is_alphanumeric()) => format!("{a} {b}"),
        _ => format!("{a}. {b}"),
    }
}

/// Text that holds no markup made HTML.
pub fn esc(t: &str) -> String {
    t.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

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
        Err(_) => crate::book::mobi::decode(b, false),
    }
}

fn page_name(i: usize) -> String {
    format!("part{i:04}.html")
}

fn attr<'a>(a: &'a [(String, String)], key: &str) -> Option<&'a str> {
    a.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// The book of one HTML document (the head of this file).
pub fn book(html: &str, titles: Vec<String>, creators: Vec<String>) -> Result<Book> {
    let toks = tokens(html.as_bytes());
    let targets: HashSet<&str> = toks
        .iter()
        .filter_map(|(_, t)| match t {
            Tok::Start(_, a, _) => attr(a, "href")?.strip_prefix('#'),
            _ => None,
        })
        .collect();
    let mut w = Xhtml::default();
    let mut pages = Vec::new();
    let mut at: HashMap<String, usize> = HashMap::new();
    let mut toc: Vec<(String, String)> = Vec::new();
    // the open heading: its name, id and text (a line break a space, a <sup> left out, as extract.rs reads it)
    let mut head: Option<(String, String, String)> = None;
    let mut sup = 0;
    let mut heads = 0;
    // whether text came after the last heading (else the next one joins its title)
    let mut after_text = true;
    for (_, t) in &toks {
        match t {
            Tok::Text(b) => {
                let s = String::from_utf8_lossy(b);
                if head.is_none() && !s.trim().is_empty() {
                    after_text = true;
                }
                if let Some(h) = &mut head
                    && sup == 0
                {
                    h.2.push_str(&s);
                }
                w.text(&s);
            }
            Tok::Start(n, _, _) if n == "pagebreak" => {
                if w.has_text() && !w.in_block() {
                    pages.push(w.cut());
                }
            }
            Tok::Start(n, a, closed) => {
                if let Some(h) = &mut head {
                    match n.as_str() {
                        "br" => h.2.push(' '),
                        "sup" if !closed => sup += 1,
                        _ => {}
                    }
                }
                let mut a = a.clone();
                if attr(&a, "id").is_none()
                    && n == "a"
                    && let Some(name) = attr(&a, "name")
                {
                    a.push(("id".into(), name.to_string()));
                }
                let heading = head.is_none() && !closed && matches!(n.as_str(), "h1" | "h2" | "h3");
                if heading && attr(&a, "id").is_none() {
                    a.push(("id".into(), format!("ebook2m4b-{heads}")));
                    heads += 1;
                }
                if let Some(id) = attr(&a, "id") {
                    if targets.contains(id) && w.has_text() && !w.in_block() {
                        pages.push(w.cut());
                    }
                    at.entry(id.to_string()).or_insert(pages.len());
                    if heading {
                        head = Some((n.clone(), id.to_string(), String::new()));
                    }
                }
                w.start(n, &a, *closed);
            }
            Tok::End(n) => {
                if n == "sup" {
                    sup = usize::saturating_sub(sup, 1);
                }
                if head.as_ref().is_some_and(|h| h.0 == *n)
                    && let Some((_, id, text)) = head.take()
                {
                    let title = unescape(&text)
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ");
                    match toc.last_mut() {
                        Some((last, _)) if !after_text && !title.is_empty() => {
                            *last = join_title(last, &title);
                        }
                        _ if !title.is_empty() => {
                            toc.push((title, id));
                            after_text = false;
                        }
                        _ => {}
                    }
                }
                w.end(n);
            }
        }
    }
    pages.push(w.finish());
    let link = regex::Regex::new(r##"href="#([^"]*)""##).unwrap();
    let spine = pages
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let p = link.replace_all(p, |c: &regex::Captures| match at.get(&c[1]) {
                Some(&n) => format!("href=\"{}#{}\"", page_name(n), &c[1]),
                None => c[0].to_string(),
            });
            Ok(Page {
                name: page_name(i),
                body: body(&p)?,
                is_doc: true,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let toc = toc
        .into_iter()
        .map(|(t, id)| (t, format!("{}#{id}", page_name(at[&id]))))
        .collect();
    Ok(Book {
        spine,
        toc,
        creators,
        titles,
    })
}

/// A text file's paragraphs (the head of this file).
fn paragraphs(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let blank = |l: &str| l.trim().is_empty();
    let has_blank = lines.iter().any(|l| blank(l));
    let indented = |l: &str| l.starts_with([' ', '\t', '\u{3000}']);
    let has_indent = lines.iter().any(|l| indented(l));
    let mut out = Vec::new();
    let mut para: Vec<&str> = Vec::new();
    for l in lines {
        let starts = if has_blank {
            blank(l)
        } else {
            !has_indent || indented(l)
        };
        if starts && !para.is_empty() {
            out.push(std::mem::take(&mut para).join(" "));
        }
        if !blank(l) {
            para.push(l.trim());
        }
    }
    if !para.is_empty() {
        out.push(para.join(" "));
    }
    out
}

pub fn txt(path: &Path) -> Result<Book> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    let html: String = paragraphs(&decode(&b))
        .iter()
        .map(|p| format!("<p>{}</p>\n", esc(p)))
        .collect();
    book(&html, Vec::new(), Vec::new())
}

pub fn html(path: &Path) -> Result<Book> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    let html = decode(&b);
    let toks = tokens(html.as_bytes());
    let mut titles = Vec::new();
    let mut creators = Vec::new();
    for (k, (_, t)) in toks.iter().enumerate() {
        match t {
            Tok::Start(n, _, false) if n == "title" && titles.is_empty() => {
                if let Some((_, Tok::Text(b))) = toks.get(k + 1) {
                    let t = unescape(&String::from_utf8_lossy(b)).trim().to_string();
                    if !t.is_empty() {
                        titles.push(t);
                    }
                }
            }
            Tok::Start(n, a, _) if n == "meta" => {
                let name = attr(a, "name").unwrap_or("").to_ascii_lowercase();
                if (name == "author" || name == "dc.creator")
                    && let Some(c) = attr(a, "content").filter(|c| !c.trim().is_empty())
                {
                    creators.push(unescape(c).trim().to_string());
                }
            }
            _ => {}
        }
    }
    book(&html, titles, creators)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(b: &Book) -> Vec<Vec<String>> {
        b.spine
            .iter()
            .map(|p| {
                let mut all = Vec::new();
                p.body.as_ref().unwrap().descendants(&mut all);
                all.iter()
                    .filter(|e| matches!(e.name.as_str(), "p" | "h1" | "h2" | "h3" | "h4"))
                    .map(|e| e.text("").trim().to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pages_cut_at_breaks_and_link_targets_and_headings_are_contents() {
        let b = book(
            r##"<p><a href="#c1">One</a></p><p><a href="#c2">Two</a></p><div id="c1"><h2>Chapter One</h2><p>Text <a href="#c2">see</a>.</p></div><mbp:pagebreak/><p>More &amp; more</p><h2 id="c2">Chapter  Two</h2><p>End</p>"##,
            vec!["T".into()],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            texts(&b),
            [
                vec!["One", "Two"],
                vec!["Chapter One", "Text see."],
                vec!["More & more"],
                vec!["Chapter  Two", "End"]
            ]
        );
        assert_eq!(
            b.toc,
            [
                ("Chapter One".to_string(), "part0001.html#ebook2m4b-0".to_string()),
                ("Chapter Two".to_string(), "part0003.html#c2".to_string())
            ]
        );
        let mut all = Vec::new();
        b.spine[0].body.as_ref().unwrap().descendants(&mut all);
        let hrefs: Vec<_> = all.iter().filter_map(|e| e.attr("href")).collect();
        assert_eq!(hrefs, ["part0001.html#c1", "part0003.html#c2"]);
    }

    #[test]
    fn heading_titles_read_as_extract_reads_them() {
        let b = book(
            "<h1><img src=\"t.png\"/></h1><p>Front</p><h2>Chapter I<br/>The Lake<sup>1</sup></h2><p>x</p>",
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            b.toc,
            [(
                "Chapter I The Lake".to_string(),
                "part0000.html#ebook2m4b-1".to_string()
            )]
        );
    }

    #[test]
    fn a_heading_right_after_another_joins_its_title() {
        let b = book(
            "<h2>CHAPTER I</h2>\n<h3>The Lake</h3><p>x</p><h2>CHAPTER II</h2><p>y</p>",
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            b.toc,
            [
                (
                    "CHAPTER I. The Lake".to_string(),
                    "part0000.html#ebook2m4b-0".to_string()
                ),
                ("CHAPTER II".to_string(), "part0000.html#ebook2m4b-2".to_string())
            ]
        );
    }

    #[test]
    fn a_target_inside_a_paragraph_does_not_cut() {
        let b = book(
            r##"<p>a <a href="#n">1</a> b <a name="n"></a>c</p>"##,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(texts(&b), [vec!["a 1 b c"]]);
    }

    #[test]
    fn text_bytes_decode() {
        assert_eq!(decode(b"\xEF\xBB\xBFcaf\xC3\xA9"), "café");
        assert_eq!(decode(b"\x93caf\xe9\x94"), "\u{201c}café\u{201d}");
        assert_eq!(decode(b"\xFF\xFEa\x00b\x00"), "ab");
    }

    #[test]
    fn text_paragraphs() {
        assert_eq!(
            paragraphs("CHAPTER I.\nDown\n\nIt was\na day.\n\n\n* * *\n"),
            ["CHAPTER I. Down", "It was a day.", "* * *"]
        );
        assert_eq!(paragraphs("One.\nTwo.\n"), ["One.", "Two."]);
        assert_eq!(
            paragraphs("  It was\na day.\n  Then\nnot.\n"),
            ["It was a day.", "Then not."]
        );
    }

    #[test]
    fn html_title_and_author() {
        let dir = std::env::temp_dir().join("ebook2m4b-markup-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("b.html");
        std::fs::write(
            &p,
            r#"<html><head><title>A &amp; B</title><meta name="author" content="Ann Lee"></head><body><h1>X</h1><p>y</p></body></html>"#,
        )
        .unwrap();
        let b = html(&p).unwrap();
        assert_eq!(
            (b.titles, b.creators),
            (vec!["A & B".to_string()], vec!["Ann Lee".to_string()])
        );
        assert_eq!(
            b.toc,
            [("X".to_string(), "part0000.html#ebook2m4b-0".to_string())]
        );
    }
}
