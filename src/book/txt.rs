//! A TXT read as a Book: paragraphs are the lines between blank lines, joined; a file without blank
//! lines has one per line, or, where lines are indented, one from each indented line on. No contents:
//! extract.rs finds lines like "Chapter One".
use crate::book::Book;
use crate::book::htmlbook::{book, esc};
use crate::book::text::decode;
use anyhow::{Context, Result};
use std::path::Path;

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

pub fn read(path: &Path) -> Result<Book> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    let html: String = paragraphs(&decode(&b))
        .iter()
        .map(|p| format!("<p>{}</p>\n", esc(p)))
        .collect();
    book(&html, Vec::new(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
