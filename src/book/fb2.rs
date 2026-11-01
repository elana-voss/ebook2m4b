//! An FB2 (FictionBook 2) read as an epub::Book: its XML made one HTML document for markup::book().
//!
//! title-info gives the title (book-title) and authors (first, middle, last name, else nickname).
//! Each body is read but one with a name (notes, comments); a note's link (type="note") is left out
//! with its number, as are images and binaries. Each section starts a page; its title (its lines
//! joined, ". " where a line ends in a letter or digit) is an h2, an entry of the contents (note
//! links left out); another title (a poem's, the body's) and a subtitle an h4; p, v (a verse line) and text-author paragraphs; poem, stanza, cite, epigraph a div.
//! empty-line is nothing: converters write it for a scene break and for a blank line alike.
use crate::book::epub::{Book, Elem, Node};
use crate::book::markup::{decode, esc};
use anyhow::{Context, Result};
use std::path::Path;

fn kids<'a>(e: &'a Elem, name: &'a str) -> impl Iterator<Item = &'a Elem> {
    e.children.iter().filter_map(move |c| match c {
        Node::Elem(k) if k.name == name => Some(k),
        _ => None,
    })
}

fn text_of(e: &Elem, name: &str) -> String {
    kids(e, name)
        .next()
        .map_or(String::new(), |k| k.text("").trim().to_string())
}

/// The title's lines as one.
fn title(e: &Elem) -> String {
    fn plain(e: &Elem, out: &mut String) {
        for c in &e.children {
            match c {
                Node::Text(t) => out.push_str(t),
                Node::Elem(k) if k.name == "a" && k.attr("type") == Some("note") => {}
                Node::Elem(k) => plain(k, out),
            }
        }
    }
    let mut out = String::new();
    for p in kids(e, "p") {
        let mut t = String::new();
        plain(p, &mut t);
        let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
        if t.is_empty() {
            continue;
        }
        out = crate::book::markup::join_title(&out, &t);
    }
    out
}

fn wrap(e: &Elem, tag: &str, out: &mut String) {
    out.push_str(&format!("<{tag}>"));
    walk(e, out);
    out.push_str(&format!("</{tag}>"));
}

fn walk(e: &Elem, out: &mut String) {
    for c in &e.children {
        let k = match c {
            Node::Text(t) => {
                out.push_str(&esc(t));
                continue;
            }
            Node::Elem(k) => k,
        };
        match k.name.as_str() {
            "section" => {
                out.push_str("<mbp:pagebreak/>");
                wrap(k, "div", out);
            }
            // a section's title is a chapter's; a poem's or the body's is not
            "title" => {
                let h = if e.name == "section" { "h2" } else { "h4" };
                out.push_str(&format!("<{h}>{}</{h}>\n", esc(&title(k))));
            }
            "subtitle" => wrap(k, "h4", out),
            "p" | "v" | "text-author" => wrap(k, "p", out),
            "poem" | "stanza" | "cite" | "epigraph" | "annotation" => wrap(k, "div", out),
            "emphasis" => wrap(k, "i", out),
            "strong" => wrap(k, "b", out),
            "sup" => wrap(k, "sup", out),
            "a" if k.attr("type") == Some("note") => {}
            "image" | "binary" | "empty-line" => {}
            _ => walk(k, out),
        }
    }
}

pub fn read(path: &Path) -> Result<Book> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    let root =
        crate::book::epub::tree(&decode(&b)).with_context(|| format!("{} is no FB2", path.display()))?;
    let info = kids(&root, "description")
        .flat_map(|d| kids(d, "title-info"))
        .next();
    let titles: Vec<String> = info
        .map(|i| text_of(i, "book-title"))
        .filter(|t| !t.is_empty())
        .into_iter()
        .collect();
    let creators = info
        .into_iter()
        .flat_map(|i| kids(i, "author"))
        .map(|a| {
            let names: Vec<String> = ["first-name", "middle-name", "last-name"]
                .iter()
                .map(|n| text_of(a, n))
                .filter(|n| !n.is_empty())
                .collect();
            if names.is_empty() {
                text_of(a, "nickname")
            } else {
                names.join(" ")
            }
        })
        .filter(|a| !a.is_empty())
        .collect();
    let mut html = String::new();
    for body in kids(&root, "body").filter(|b| b.attr("name").is_none()) {
        walk(body, &mut html);
    }
    crate::book::markup::book(&html, titles, creators)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_sections_title_is_a_heading() {
        let body = crate::book::epub::tree(
            r##"<body><title><p>Book</p></title><section><title><p>Chapter One,</p><p>The Lake<a type="note">1</a></p></title><poem><title><p>Song</p></title></poem></section></body>"##,
        )
        .unwrap();
        let mut out = String::new();
        walk(&body, &mut out);
        assert_eq!(
            out,
            "<h4>Book</h4>\n<mbp:pagebreak/><div><h2>Chapter One, The Lake</h2>\n<div><h4>Song</h4>\n</div></div>"
        );
    }

    #[test]
    fn sections_titles_notes_and_metadata() {
        let dir = std::env::temp_dir().join("ebook2m4b-fb2-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("b.fb2");
        std::fs::write(
            &p,
            r##"<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" xmlns:l="http://www.w3.org/1999/xlink">
<description><title-info><author><first-name>Lewis</first-name><last-name>Carroll</last-name></author>
<book-title>Sylvie &amp; Bruno</book-title></title-info></description>
<body><section><title><p>Chapter One</p><p>The Lake</p></title><p>It <emphasis>was</emphasis> a day<a l:href="#n1" type="note">1</a>.</p>
<empty-line/><subtitle>* * *</subtitle><poem><stanza><v>A line</v></stanza></poem></section>
<section><p>Untitled</p></section></body>
<body name="notes"><section id="n1"><p>A note</p></section></body>
<binary id="img">AAAA</binary></FictionBook>"##,
        )
        .unwrap();
        let b = read(&p).unwrap();
        assert_eq!(b.titles, ["Sylvie & Bruno"]);
        assert_eq!(b.creators, ["Lewis Carroll"]);
        assert_eq!(
            b.toc,
            [(
                "Chapter One. The Lake".to_string(),
                "part0000.html#ebook2m4b-0".to_string()
            )]
        );
        let pages: Vec<String> = b
            .spine
            .iter()
            .map(|p| {
                p.body
                    .as_ref()
                    .unwrap()
                    .text("")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        assert_eq!(
            pages,
            [
                "Chapter One. The Lake It was a day. * * *A line",
                "Untitled"
            ]
        );
    }
}
