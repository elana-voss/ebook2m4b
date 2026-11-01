//! An HTML file read as a Book: its <title> and <meta name="author">, its document by htmlbook.rs.
use crate::book::htmlbook::book;
use crate::book::soup::{Tok, tokens, unescape};
use crate::book::text::decode;
use crate::book::{Book, attr};
use anyhow::{Context, Result};
use std::path::Path;

pub fn read(path: &Path) -> Result<Book> {
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
        let b = read(&p).unwrap();
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
