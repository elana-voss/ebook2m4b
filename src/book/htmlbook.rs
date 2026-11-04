//! book(): one HTML document (theirs, or the one fb2.rs or docx.rs makes) cut into pages through
//! soup.rs, as extract.rs reads them.
//!
//! A page ends at each <mbp:pagebreak/> (or <pagebreak/>) and before each element a link of
//! the document points at ("#id", or <a name>), so that the book's own contents page links other
//! pages; not inside a paragraph, heading or list item, nor while the page has no text. Each h1-h3
//! is an entry of the contents (an id given where it has none), one right after another (no text
//! between) joins the entry's title ("CHAPTER I. The Lake"); links "#id" name the page.
use crate::book::soup::{Tok, Xhtml, tokens, unescape};
use crate::book::{Book, Page, attr, body};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

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

fn page_name(i: usize) -> String {
    format!("part{i:04}.html")
}

/// The ids links in the document point to.
fn link_targets(toks: &[(usize, Tok)]) -> HashSet<&str> {
    toks.iter()
        .filter_map(|(_, t)| match t {
            Tok::Start(_, a, _) => attr(a, "href")?.strip_prefix('#'),
            _ => None,
        })
        .collect()
}

/// A heading closed: into the contents, joined to the last title when no text came between.
fn add_heading(toc: &mut Vec<(String, String)>, after_text: &mut bool, id: String, text: &str) {
    let title = unescape(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    match toc.last_mut() {
        Some((last, _)) if !*after_text && !title.is_empty() => {
            *last = join_title(last, &title);
        }
        _ if !title.is_empty() => {
            toc.push((title, id));
            *after_text = false;
        }
        _ => {}
    }
}

/// The pages as the spine, each link naming the page its target landed on.
fn spine_of(pages: &[String], at: &HashMap<String, usize>) -> Result<Vec<Page>> {
    let link = crate::book::re(r##"href="#([^"]*)""##);
    pages
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
        .collect()
}

/// The book of one HTML document (the head of this file).
pub fn book(html: &str, titles: Vec<String>, creators: Vec<String>) -> Result<Book> {
    let toks = tokens(html.as_bytes());
    let targets = link_targets(&toks);
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
                    add_heading(&mut toc, &mut after_text, id, &text);
                }
                w.end(n);
            }
        }
    }
    pages.push(w.finish());
    let spine = spine_of(&pages, &at)?;
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
}
