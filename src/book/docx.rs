//! A DOCX (Word) read as an epub::Book: its main part (as _rels/.rels names it, else word/document.xml)
//! made one HTML document for markup::book().
//!
//! docProps/core.xml gives the title and author (dc:title, dc:creator). Each paragraph (w:p, also in
//! tables) is a <p>, or h1-h4 where its style (word/styles.xml: named "heading N", or an outline
//! level) is a heading; only h1-h3 are entries of the contents, a "Title" paragraph is text. Text is
//! w:t's; a tab or line break a space, a non-breaking hyphen "-"; a bookmark between paragraphs an anchor too; footnote marks, deleted text, field codes and drawings are
//! left out. An empty paragraph with a border above or below (its own or its style's)
//! is a scene break. A page break (w:pageBreakBefore, w:br type="page") starts a page; a bookmark is an
//! anchor before its paragraph and an internal hyperlink links it, so that a contents page works;
//! in a link, what follows a tab (a contents' page number) is left out.
use crate::setup::archive::Archive;
use crate::book::epub::{Book, Elem, Node};
use crate::book::markup::esc;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

fn kid<'a>(e: &'a Elem, name: &str) -> Option<&'a Elem> {
    e.children.iter().find_map(|c| match c {
        Node::Elem(k) if k.name == name => Some(k),
        _ => None,
    })
}

/// A paragraph style: its heading level (1-9; 0 for "Title"), whether it draws a rule; each, where
/// the style does not say, its w:basedOn style's.
#[derive(Clone, Copy, Default)]
struct Style {
    level: Option<usize>,
    rule: bool,
}

/// Whether paragraph properties draw a border above or below (a rule).
fn rule(ppr: Option<&Elem>) -> bool {
    ppr.and_then(|p| kid(p, "pbdr")).is_some_and(|b| {
        ["top", "bottom"].iter().any(|s| {
            kid(b, s).is_some_and(|e| !matches!(e.attr("val"), Some("none" | "nil") | None))
        })
    })
}

fn styles(xml: &str) -> Result<HashMap<String, Style>> {
    let root = crate::book::epub::tree(xml)?;
    let heading = regex::Regex::new(r"(?i)^heading ([1-9])$").unwrap();
    // a style's own level and rule (None: not said) and the style it is based on
    type Own = (Option<usize>, Option<bool>, Option<String>);
    let mut own: HashMap<String, Own> = HashMap::new();
    for c in &root.children {
        let Node::Elem(s) = c else { continue };
        let Some(id) = s.attr("styleId").filter(|_| s.name == "style") else {
            continue;
        };
        let name = kid(s, "name").and_then(|n| n.attr("val")).unwrap_or("");
        let outline = kid(s, "ppr")
            .and_then(|p| kid(p, "outlinelvl"))
            .and_then(|o| o.attr("val")?.parse::<usize>().ok());
        let level = if name.eq_ignore_ascii_case("title") {
            Some(0)
        } else if let Some(c) = heading.captures(name) {
            c[1].parse().ok()
        } else {
            outline.filter(|&o| o < 9).map(|o| o + 1)
        };
        let ppr = kid(s, "ppr");
        let border = ppr.and_then(|p| kid(p, "pbdr")).map(|_| rule(ppr));
        let based = kid(s, "basedon")
            .and_then(|b| b.attr("val"))
            .map(String::from);
        own.insert(id.to_string(), (level, border, based));
    }
    let mut out = HashMap::new();
    for id in own.keys() {
        let (mut level, mut border) = (None, None);
        let mut at = Some(id);
        for _ in 0..10 {
            let Some((l, b, based)) = at.and_then(|a| own.get(a)) else {
                break;
            };
            level = level.or(*l);
            border = border.or(*b);
            at = based.as_ref();
        }
        out.insert(
            id.clone(),
            Style {
                level,
                rule: border.unwrap_or(false),
            },
        );
    }
    Ok(out)
}

/// A paragraph as its runs give it: its text (HTML), its bookmarks, the text before each page break in it.
struct Para<'a> {
    text: String,
    anchors: Vec<&'a str>,
    /// text written before a page break inside the paragraph
    before: Vec<String>,
}

fn runs<'a>(e: &'a Elem, p: &mut Para<'a>, link: &mut Option<bool>) {
    for c in &e.children {
        let Node::Elem(k) = c else { continue };
        match k.name.as_str() {
            "t" => {
                if *link != Some(true) {
                    p.text.push_str(&esc(&k.text("")));
                }
            }
            "tab" => match link {
                Some(after_tab) => *after_tab = true,
                None => p.text.push(' '),
            },
            "br" | "cr" if k.attr("type") == Some("page") => {
                p.before.push(std::mem::take(&mut p.text));
            }
            "br" | "cr" | "ptab" => p.text.push(' '),
            "nobreakhyphen" => p.text.push('-'),
            "bookmarkstart" => {
                if let Some(n) = k.attr("name").filter(|n| *n != "_GoBack") {
                    p.anchors.push(n);
                }
            }
            "hyperlink" => match k.attr("anchor") {
                Some(a) if link.is_none() => {
                    p.text.push_str(&format!("<a href=\"#{}\">", esc(a)));
                    let mut inner = Some(false);
                    runs(k, p, &mut inner);
                    p.text.push_str("</a>");
                }
                _ => runs(k, p, link),
            },
            "ppr" | "rpr" | "deltext" | "instrtext" | "drawing" | "pict" | "footnotereference"
            | "endnotereference" | "object" => {}
            _ => runs(k, p, link),
        }
    }
}

fn para(e: &Elem, styles: &HashMap<String, Style>, out: &mut String) {
    let ppr = kid(e, "ppr");
    let style = ppr
        .and_then(|p| kid(p, "pstyle"))
        .and_then(|s| s.attr("val"));
    let outline = ppr
        .and_then(|p| kid(p, "outlinelvl"))
        .and_then(|o| o.attr("val")?.parse::<usize>().ok())
        .filter(|&o| o < 9)
        .map(|o| o + 1);
    let st = style
        .and_then(|s| styles.get(s))
        .copied()
        .unwrap_or_default();
    let level = st.level.or(outline).unwrap_or(0);
    let tag = match level {
        0 => "p".to_string(),
        l => format!("h{}", l.min(4)),
    };
    if ppr
        .and_then(|p| kid(p, "pagebreakbefore"))
        .is_some_and(|b| !matches!(b.attr("val"), Some("0" | "false" | "off")))
    {
        out.push_str("<mbp:pagebreak/>");
    }
    let mut p = Para {
        text: String::new(),
        anchors: Vec::new(),
        before: Vec::new(),
    };
    runs(e, &mut p, &mut None);
    for t in &p.before {
        out.push_str(&format!("<p>{t}</p>\n<mbp:pagebreak/>"));
    }
    for a in &p.anchors {
        out.push_str(&format!("<a id=\"{}\"></a>", esc(a)));
    }
    if p.text.trim().is_empty() && p.before.is_empty() && (st.rule || rule(ppr)) {
        out.push_str("<hr/>\n"); // an empty paragraph with a rule: a scene break
    } else {
        out.push_str(&format!("<{tag}>{}</{tag}>\n", p.text));
    }
}

fn body(e: &Elem, styles: &HashMap<String, Style>, out: &mut String) {
    for c in &e.children {
        if let Node::Elem(k) = c {
            match k.name.as_str() {
                "p" => para(k, styles, out),
                "bookmarkstart" => {
                    if let Some(n) = k.attr("name") {
                        out.push_str(&format!("<a id=\"{}\"></a>", esc(n)));
                    }
                }
                "sectpr" => {}
                _ => body(k, styles, out),
            }
        }
    }
}

pub fn read(path: &Path) -> Result<Book> {
    let mut z = Archive::open(path).with_context(|| format!("{} is no DOCX", path.display()))?;
    // the main part, as _rels/.rels names it
    let main = z
        .text("_rels/.rels")
        .ok()
        .and_then(|r| {
            let rels = crate::book::epub::tree(&r).ok()?;
            rels.children.iter().find_map(|c| match c {
                Node::Elem(k)
                    if k.attr("Type")
                        .is_some_and(|t| t.ends_with("/officeDocument")) =>
                {
                    Some(k.attr("Target")?.trim_start_matches('/').to_string())
                }
                _ => None,
            })
        })
        .unwrap_or_else(|| "word/document.xml".into());
    let doc = crate::book::epub::tree(&z.text(&main)?)?;
    let styles = match z.text("word/styles.xml") {
        Ok(s) => styles(&s)?,
        Err(_) => HashMap::new(), // a document without styles: plain paragraphs
    };
    let (mut titles, mut creators) = (Vec::new(), Vec::new());
    if let Ok(core) = z.text("docProps/core.xml") {
        for c in &crate::book::epub::tree(&core)?.children {
            let Node::Elem(k) = c else { continue };
            let t = k.text("").trim().to_string();
            match k.name.as_str() {
                "title" if !t.is_empty() => titles.push(t),
                "creator" if !t.is_empty() => creators.push(t),
                _ => {}
            }
        }
    } // else no metadata: the file's name is the title
    let mut html = String::new();
    if let Some(b) = kid(&doc, "body") {
        body(b, &styles, &mut html);
    }
    crate::book::markup::book(&html, titles, creators)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main""#;

    #[test]
    fn headings_breaks_bookmarks_links_and_metadata() {
        let doc = format!(
            r#"<w:document {W}><w:body>
<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>The Book</w:t></w:r></w:p>
<w:p><w:hyperlink w:anchor="_Toc1"><w:r><w:t>Chapter One</w:t></w:r><w:r><w:tab/><w:t>5</w:t></w:r></w:hyperlink></w:p>
<w:p><w:hyperlink w:anchor="_Toc2"><w:r><w:t>Chapter Two</w:t></w:r></w:hyperlink></w:p>
<w:p><w:pPr><w:pStyle w:val="H1"/><w:pageBreakBefore/></w:pPr><w:bookmarkStart w:id="0" w:name="_Toc1"/><w:r><w:t>Chapter One</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">It was </w:t></w:r><w:r><w:t>a</w:t><w:tab/><w:t>day</w:t><w:footnoteReference w:id="1"/></w:r><w:del><w:r><w:delText>gone</w:delText></w:r></w:del></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:pPr><w:pStyle w:val="Chap"/></w:pPr><w:r><w:br w:type="page"/></w:r><w:bookmarkStart w:id="1" w:name="_Toc2"/><w:r><w:t>Chapter Two</w:t></w:r></w:p>
<w:p><w:pPr><w:pBdr><w:top w:val="single"/></w:pBdr></w:pPr></w:p><w:p><w:pPr><w:pStyle w:val="R"/></w:pPr></w:p><w:p/>
<w:sectPr/></w:body></w:document>"#
        );
        let styles = format!(
            r#"<w:styles {W}><w:style w:type="paragraph" w:styleId="H1"><w:name w:val="heading 1"/></w:style><w:style w:type="paragraph" w:styleId="Chap"><w:name w:val="Chapter"/><w:basedOn w:val="H1"/></w:style><w:style w:type="paragraph" w:styleId="R"><w:name w:val="R"/><w:pPr><w:pBdr><w:bottom w:val="inset"/></w:pBdr></w:pPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style></w:styles>"#
        );
        let core = r#"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>The Book</dc:title><dc:creator>Ann Lee</dc:creator></cp:coreProperties>"#;
        let dir = std::env::temp_dir().join("ebook2m4b-docx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("b.docx");
        crate::setup::archive::write(
            &p,
            &[
                ("word/document.xml", doc.as_bytes()),
                ("word/styles.xml", styles.as_bytes()),
                ("docProps/core.xml", core.as_bytes()),
            ],
        );
        let b = read(&p).unwrap();
        assert_eq!(
            (b.titles.clone(), b.creators.clone()),
            (vec!["The Book".to_string()], vec!["Ann Lee".to_string()])
        );
        assert_eq!(
            b.toc,
            [
                ("Chapter One".to_string(), "part0001.html#ebook2m4b-0".to_string()),
                ("Chapter Two".to_string(), "part0002.html#ebook2m4b-1".to_string())
            ]
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
                "The Book Chapter One Chapter Two",
                "Chapter One It was a day Cell",
                "Chapter Two"
            ]
        );
        let mut all = Vec::new();
        b.spine[0].body.as_ref().unwrap().descendants(&mut all);
        let hrefs: Vec<_> = all.iter().filter_map(|e| e.attr("href")).collect();
        assert_eq!(hrefs, ["part0001.html#_Toc1", "part0002.html#_Toc2"]);
        let mut last = Vec::new();
        b.spine[2].body.as_ref().unwrap().descendants(&mut last);
        assert_eq!(last.iter().filter(|e| e.name == "hr").count(), 2);
    }

    #[test]
    fn the_part_rels_name_loose_bookmarks_and_hyphens() {
        let doc = format!(
            r#"<w:document {W}><w:body><w:p><w:hyperlink w:anchor="b"><w:r><w:t>Go</w:t></w:r></w:hyperlink></w:p><w:p><w:r><w:t>Text</w:t></w:r></w:p><w:bookmarkStart w:id="0" w:name="b"/><w:p><w:r><w:t>self</w:t><w:noBreakHyphen/><w:t>aware</w:t><w:ptab/><w:t>x</w:t></w:r></w:p></w:body></w:document>"#
        );
        let rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="/word/document2.xml"/></Relationships>"#;
        let dir = std::env::temp_dir().join("ebook2m4b-docx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.docx");
        crate::setup::archive::write(
            &p,
            &[
                ("_rels/.rels", rels.as_bytes()),
                ("word/document2.xml", doc.as_bytes()),
            ],
        );
        let b = read(&p).unwrap();
        let pages: Vec<String> = b
            .spine
            .iter()
            .map(|p| p.body.as_ref().unwrap().text("").trim().to_string())
            .collect();
        assert_eq!(pages, ["Go\nText", "self-aware x"]);
        let mut all = Vec::new();
        b.spine[0].body.as_ref().unwrap().descendants(&mut all);
        assert_eq!(
            all.iter().find_map(|e| e.attr("href")),
            Some("part0001.html#b")
        );
    }
}
