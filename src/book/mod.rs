//! A book as its readers give it and extract.rs takes it: the spine's pages (documents parsed into a
//! small tree), the contents flattened, the creators and titles; read(path) picks the reader by extension.
//! XML is read with DOCTYPEs allowed; a page's HTML entities are decoded and an undeclared `epub:` declared first.
//! Traps:
//! - only the body's children are kept (text before the first is lost);
//! - a string of ASCII spaces only becomes "\n" if it holds one, else " ", not in <pre>;
//! - comments hold no text.
pub mod docx;
pub mod epub;
pub mod fb2;
pub mod html;
pub mod htmlbook;
pub mod kf8;
pub mod mobi;
pub mod soup;
pub mod text;
pub mod txt;

use anyhow::{Result, anyhow};
use std::path::Path;

/// A piece of a page: text, or an element with its children.
#[derive(Clone, Debug)]
pub enum Node {
    Text(String),
    Elem(Elem),
}

#[derive(Clone, Debug)]
pub struct Elem {
    /// lower-case local name
    pub name: String,
    /// local names, but "epub:type" for the OPS namespace's type
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

impl Elem {
    pub fn attr(&self, key: &str) -> Option<&str> {
        attr(&self.attrs, key)
    }

    /// Every string inside, in reading order.
    pub fn strings<'a>(&'a self, out: &mut Vec<&'a str>) {
        for c in &self.children {
            match c {
                Node::Text(t) => out.push(t),
                Node::Elem(e) => e.strings(out),
            }
        }
    }

    /// The text inside, strings joined by `sep`.
    pub fn text(&self, sep: &str) -> String {
        let mut v = Vec::new();
        self.strings(&mut v);
        v.join(sep)
    }

    /// The elements inside, in document order (not this one).
    pub fn descendants<'a>(&'a self, out: &mut Vec<&'a Elem>) {
        for c in &self.children {
            if let Node::Elem(e) = c {
                out.push(e);
                e.descendants(out);
            }
        }
    }
}

/// A key's value among a tag's attributes.
pub fn attr<'a>(a: &'a [(String, String)], key: &str) -> Option<&'a str> {
    a.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// One item of the spine.
pub struct Page {
    /// the file's name without folders, as the contents' hrefs are matched
    pub name: String,
    /// an XHTML document: its body (None for a cover page or a body-less page)
    pub body: Option<Elem>,
    pub is_doc: bool,
}

pub struct Book {
    pub spine: Vec<Page>,
    /// (title, href) of each entry with an href, in reading order of the contents (parents first)
    pub toc: Vec<(String, String)>,
    pub creators: Vec<String>,
    pub titles: Vec<String>,
}

/// The last part of a path (either slash).
pub fn base_name(p: &str) -> &str {
    p.trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
}

const OPS: &str = "http://www.idpf.org/2007/ops";

/// A book by its file's extension (a mobi, azw or azw3 by default).
pub fn read(path: &Path) -> Result<Book> {
    let ext = path
        .extension()
        .map_or(String::new(), |e| e.to_string_lossy().to_lowercase());
    match ext.as_str() {
        "epub" => epub::read(path),
        "fb2" => fb2::read(path),
        "docx" => docx::read(path),
        "txt" => txt::read(path),
        "html" | "htm" | "xhtml" => html::read(path),
        _ => mobi::read(path),
    }
}

/// An XML file parsed; a DOCTYPE is allowed (EPUBs often carry one).
pub(crate) fn xml(text: &str) -> Result<roxmltree::Document<'_>, roxmltree::Error> {
    roxmltree::Document::parse_with_options(
        text,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
}

/// A page made XML: HTML's named entities as characters (XML's five are left), and the `epub:`
/// prefix declared on <html> when the page uses it undeclared.
pub(crate) fn as_xml(text: &str) -> String {
    static ENTITY: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"&([A-Za-z][A-Za-z0-9]*);").unwrap());
    let mut t = ENTITY
        .replace_all(text, |c: &regex::Captures| match &c[1] {
            "amp" | "lt" | "gt" | "quot" | "apos" => c[0].to_string(),
            _ => soup::unescape(&c[0]),
        })
        .into_owned();
    if t.contains("epub:")
        && !t.contains("xmlns:epub")
        && let Some(i) = t.find("<html")
    {
        t.insert_str(i + 5, &format!(r#" xmlns:epub="{OPS}""#));
    }
    t
}

/// The page's body as the head of this file says, or None without a body.
pub fn body(xhtml: &str) -> Result<Option<Elem>> {
    let text = as_xml(xhtml);
    let doc = xml(&text).map_err(|e| anyhow!("not XHTML: {e}"))?;
    let Some(b) = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "body")
    else {
        return Ok(None);
    };
    // body.text (before its first element or comment) is dropped
    let start = b
        .children()
        .position(|c| !c.is_text())
        .unwrap_or(usize::MAX);
    let nodes: Vec<_> = b.children().skip(start).collect();
    Ok(Some(elem("body".into(), Vec::new(), &nodes, false)))
}

/// An XML document's root element as an Elem (FB2, DOCX), by local names.
pub fn tree(text: &str) -> Result<Elem> {
    let doc = xml(text).map_err(|e| anyhow!("not XML: {e}"))?;
    let root = doc.root_element();
    let nodes: Vec<_> = root.children().collect();
    let name = root.tag_name().name().to_lowercase();
    Ok(elem(name, Vec::new(), &nodes, false))
}

fn elem(
    name: String,
    attrs: Vec<(String, String)>,
    nodes: &[roxmltree::Node],
    pre: bool,
) -> Elem {
    let nodes: Vec<_> = nodes
        .iter()
        .filter(|n| n.is_text() || n.is_element() || n.is_comment())
        .collect();
    let pre = pre || name == "pre" || name == "textarea";
    let mut children = Vec::new();
    for n in &nodes {
        if n.is_text() {
            let t = n.text().unwrap_or("");
            let blank = t
                .chars()
                .all(|c| matches!(c, ' ' | '\n' | '\t' | '\x0c' | '\r'));
            children.push(Node::Text(if blank && !pre {
                (if t.contains('\n') { "\n" } else { " " }).into()
            } else {
                t.into()
            }));
        } else if n.is_element() {
            let name = n.tag_name().name().to_lowercase();
            let attrs = n
                .attributes()
                .map(|a| {
                    let k = if a.namespace() == Some(OPS) {
                        format!("epub:{}", a.name())
                    } else {
                        a.name().to_string()
                    };
                    (k, a.value().to_string())
                })
                .collect();
            let sub: Vec<_> = n.children().collect();
            children.push(Node::Elem(elem(name, attrs, &sub, pre)));
        }
    }
    Elem {
        name,
        attrs,
        children,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(inner: &str) -> Elem {
        body(&format!(r#"<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="{OPS}"><body>{inner}</body></html>"#)).unwrap().unwrap()
    }

    #[test]
    fn spans_join_and_text_stays() {
        assert_eq!(page("<p><span>T</span><span>he</span></p>").text(""), "The");
        assert_eq!(page("<p>x <b>y</b></p>\n<p>z</p>").text(""), "x y\nz");
        assert_eq!(page("<p>a&nbsp;b</p>").text(""), "a\u{a0}b");
        assert_eq!(
            page(r#"<a epub:type="noteref">1</a>"#).children[0]
                .clone()
                .into_elem()
                .attr("epub:type"),
            Some("noteref")
        );
    }

    impl Node {
        fn into_elem(self) -> Elem {
            match self {
                Node::Elem(e) => e,
                Node::Text(_) => panic!("text"),
            }
        }
    }

    #[test]
    fn any_html_entity_and_an_undeclared_epub_prefix_read() {
        let b = body(r#"<html xmlns="http://www.w3.org/1999/xhtml"><body><p epub:type="x">&Agrave;&frac12;&amp;&rsquo;</p></body></html>"#)
            .unwrap()
            .unwrap();
        assert_eq!(b.text(""), "À½&’");
    }
}
