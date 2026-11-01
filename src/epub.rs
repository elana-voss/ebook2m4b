//! An EPUB as extract.rs needs it, read the way the Python version's ebooklib and BeautifulSoup see it.
//!
//! Book: the spine's items (documents parsed into a small tree), the table of contents flattened
//! (the NCX the spine names, the EPUB 3 nav when that gives none), the DC creators and titles.
//! XML is read with DOCTYPEs allowed; a page's HTML entities are decoded and an undeclared `epub:` declared first.
//! Traps, each copied on purpose so the text matches extract.py's:
//! - manifest hrefs are percent-decoded, hrefs in the contents and in links are not;
//! - only the body's children are kept (text before the first is lost), and the page is "pretty
//!   printed" as ebooklib does: an element whose parent is so formatted and that has no text child
//!   gets a newline before each child and after the last, so <p><span>T</span><span>he</span></p> reads "T he";
//! - a string of ASCII spaces only becomes "\n" if it holds one, else " " (BeautifulSoup), not in <pre>;
//! - comments hold no text; a "cover"-properties page has none either.
use anyhow::{Context, Result, anyhow};
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
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
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

    /// The text inside, strings joined by `sep` (BeautifulSoup's get_text).
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

/// The last part of a path, as Python's Path(p).name (either slash).
pub fn base_name(p: &str) -> &str {
    p.trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
}

const OPF: &str = "http://www.idpf.org/2007/opf";
const DC: &str = "http://purl.org/dc/elements/1.1/";
const OPS: &str = "http://www.idpf.org/2007/ops";

/// `dir/href` with "." and ".." resolved, as zip names are.
fn join(dir: &str, href: &str) -> String {
    let mut parts: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    for p in href.split('/') {
        match p {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(p),
        }
    }
    parts.join("/")
}

fn kids<'a, 'i>(
    n: roxmltree::Node<'a, 'i>,
    ns: &'a str,
    name: &'a str,
) -> impl Iterator<Item = roxmltree::Node<'a, 'i>> {
    n.children().filter(move |c| {
        c.is_element() && c.tag_name().name() == name && c.tag_name().namespace() == Some(ns)
    })
}

/// An XML file parsed; a DOCTYPE is allowed (EPUBs often carry one).
fn xml(text: &str) -> Result<roxmltree::Document<'_>, roxmltree::Error> {
    roxmltree::Document::parse_with_options(
        text,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
}

pub fn read(path: &Path) -> Result<Book> {
    let mut z = crate::archive::Archive::open(path)
        .with_context(|| format!("{} is no EPUB", path.display()))?;
    let container = z.text("META-INF/container.xml")?;
    let c = xml(&container)?;
    let opf_path = c
        .descendants()
        .find(|n| n.tag_name().name() == "rootfile")
        .and_then(|n| n.attribute("full-path"))
        .context("container.xml names no package")?
        .to_string();
    let opf_dir = opf_path.rsplit_once('/').map_or("", |(d, _)| d).to_string();
    let opf_text = z.text(&opf_path)?;
    let opf = xml(&opf_text)?;
    let root = opf.root_element();
    let meta = kids(root, OPF, "metadata").next();
    let dc = |name: &str| -> Vec<String> {
        meta.map(|m| {
            kids(m, DC, name)
                .map(|n| n.text().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default()
    };
    // (id, decoded href, media type, properties)
    let manifest: Vec<(String, String, String, String)> = kids(root, OPF, "manifest")
        .flat_map(|m| kids(m, OPF, "item"))
        .map(|i| {
            let href = percent_encoding::percent_decode_str(i.attribute("href").unwrap_or(""))
                .decode_utf8_lossy()
                .into_owned();
            let a = |k| i.attribute(k).unwrap_or("").to_string();
            (a("id"), href, a("media-type"), a("properties"))
        })
        .collect();
    let item = |id: &str| manifest.iter().find(|m| m.0 == id);
    let spine_el = kids(root, OPF, "spine")
        .next()
        .context("the EPUB has no spine")?;
    let mut spine = Vec::new();
    for r in kids(spine_el, OPF, "itemref") {
        let Some((_, href, media, props)) = item(r.attribute("idref").unwrap_or("")) else {
            continue;
        };
        let is_doc = media == "application/xhtml+xml";
        let cover = props.split_whitespace().any(|p| p == "cover");
        let body = if is_doc && !cover {
            body(&z.text(&join(&opf_dir, href))?).with_context(|| href.clone())?
        } else {
            None
        };
        spine.push(Page {
            name: base_name(href).to_string(),
            body,
            is_doc,
        });
    }
    let mut toc = Vec::new();
    if let Some((_, href, ..)) = spine_el.attribute("toc").and_then(item) {
        ncx(&z.text(&join(&opf_dir, href))?, &mut toc)?;
    }
    // ebooklib reads the nav when the NCX gave nothing
    if toc.is_empty()
        && let Some((_, href, ..)) = manifest
            .iter()
            .find(|m| m.3.split_whitespace().any(|p| p == "nav"))
    {
        nav(&z.text(&join(&opf_dir, href))?, &mut toc)?;
    }
    Ok(Book {
        spine,
        toc,
        creators: dc("creator"),
        titles: dc("title"),
    })
}

/// The NCX's navPoints, parents first: label = navLabel's first child's text, href = content's src.
fn ncx(text: &str, out: &mut Vec<(String, String)>) -> Result<()> {
    const D: &str = "http://www.daisy.org/z3986/2005/ncx/";
    let doc = xml(text)?;
    fn walk(p: roxmltree::Node, out: &mut Vec<(String, String)>) {
        let mut label = String::new();
        let mut src = String::new();
        for a in p
            .children()
            .filter(|c| c.is_element() && c.tag_name().namespace() == Some(D))
        {
            match a.tag_name().name() {
                "navLabel" => {
                    label = a
                        .children()
                        .find(|c| c.is_element())
                        .and_then(|t| t.text())
                        .unwrap_or("")
                        .to_string()
                }
                "content" => src = a.attribute("src").unwrap_or("").to_string(),
                _ => {}
            }
        }
        if !src.is_empty() {
            out.push((label.trim().to_string(), src));
        }
        for c in kids(p, D, "navPoint") {
            walk(c, out);
        }
    }
    if let Some(map) = doc.descendants().find(|n| n.tag_name().name() == "navMap") {
        for p in kids(map, D, "navPoint") {
            walk(p, out);
        }
    }
    Ok(())
}

/// The EPUB 3 nav's toc list: each li's a (its text before any child) and nested ol.
fn nav(text: &str, out: &mut Vec<(String, String)>) -> Result<()> {
    let text = as_xml(text);
    let doc = xml(&text)?;
    fn list(ol: roxmltree::Node, out: &mut Vec<(String, String)>) {
        for li in ol.children().filter(|c| c.tag_name().name() == "li") {
            let first = |name: &str| {
                li.children()
                    .find(|c| c.is_element() && c.tag_name().name() == name)
            };
            if let Some(a) = first("a") {
                let title = if first("ol").is_some() {
                    li.first_element_child().and_then(|f| f.text())
                } else {
                    a.text()
                };
                out.push((
                    title.unwrap_or("").trim().to_string(),
                    a.attribute("href").unwrap_or("").to_string(),
                ));
            }
            if let Some(sub) = first("ol") {
                list(sub, out);
            }
        }
    }
    let toc = doc
        .descendants()
        .find(|n| n.tag_name().name() == "nav" && n.attributes().any(|a| a.value() == "toc"))
        .and_then(|n| {
            n.children()
                .find(|c| c.is_element() && c.tag_name().name() == "ol")
        });
    if let Some(ol) = toc {
        list(ol, out);
    }
    Ok(())
}

/// A page made XML: HTML's named entities as characters (XML's five are left), and the `epub:`
/// prefix declared on <html> when the page uses it undeclared.
fn as_xml(text: &str) -> String {
    static ENTITY: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"&([A-Za-z][A-Za-z0-9]*);").unwrap());
    let mut t = ENTITY
        .replace_all(text, |c: &regex::Captures| match &c[1] {
            "amp" | "lt" | "gt" | "quot" | "apos" => c[0].to_string(),
            _ => html_escape::decode_html_entities(&c[0]).into_owned(),
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

/// The page's body as the Python version sees it (the head of this file), or None without a body.
pub fn body(xhtml: &str) -> Result<Option<Elem>> {
    let text = as_xml(xhtml);
    let doc = xml(&text).map_err(|e| anyhow!("not XHTML: {e}"))?;
    let Some(b) = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "body")
    else {
        return Ok(None);
    };
    // body.text (before its first element or comment) is dropped by ebooklib
    let start = b
        .children()
        .position(|c| !c.is_text())
        .unwrap_or(usize::MAX);
    let nodes: Vec<_> = b.children().skip(start).collect();
    Ok(Some(elem("body".into(), Vec::new(), &nodes, true, false)))
}

/// An XML document's root element as an Elem (FB2, DOCX): local names, no pretty printing.
pub fn tree(text: &str) -> Result<Elem> {
    let doc = xml(text).map_err(|e| anyhow!("not XML: {e}"))?;
    let root = doc.root_element();
    let nodes: Vec<_> = root.children().collect();
    let name = root.tag_name().name().to_lowercase();
    Ok(elem(name, Vec::new(), &nodes, false, false))
}

fn elem(
    name: String,
    attrs: Vec<(String, String)>,
    nodes: &[roxmltree::Node],
    parent_formatted: bool,
    pre: bool,
) -> Elem {
    let nodes: Vec<_> = nodes
        .iter()
        .filter(|n| n.is_text() || n.is_element() || n.is_comment())
        .collect();
    let formatted = parent_formatted && !nodes.iter().any(|n| n.is_text());
    let pre = pre || name == "pre" || name == "textarea";
    let mut children = Vec::new();
    for n in &nodes {
        if formatted {
            children.push(Node::Text("\n".into()));
        }
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
            children.push(Node::Elem(elem(name, attrs, &sub, formatted, pre)));
        }
    }
    if formatted && !nodes.is_empty() {
        children.push(Node::Text("\n".into()));
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
    fn pretty_printing_parts_spans_and_keeps_text() {
        assert_eq!(
            page("<p><span>T</span><span>he</span></p>").text(""),
            "\n\nT\nhe\n\n"
        );
        assert_eq!(page("<p>x <b>y</b></p>\n<p>z</p>").text(""), "x y\nz");
        assert_eq!(page("<p>a&nbsp;b</p>").text(""), "\na\u{a0}b\n");
        assert_eq!(
            page(r#"<a epub:type="noteref">1</a>"#).children[1]
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
    fn paths_join_and_name() {
        assert_eq!(join("OEBPS", "../Text/a.xhtml"), "Text/a.xhtml");
        assert_eq!(join("", "a b.xhtml"), "a b.xhtml");
        assert_eq!(base_name("../Text/c%201.xhtml"), "c%201.xhtml");
    }
}

#[cfg(test)]
mod read_tests {
    use super::*;

    const NAV: &str = r#"<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol><li><a href="c1.xhtml">One</a></li></ol></nav></body></html>"#;

    /// A small EPUB: the package names `ncx` as its toc file and `nav.xhtml` as its nav.
    fn book(name: &str, ncx: &str) -> Book {
        let path =
            std::env::temp_dir().join(format!("t2a-epub-{}-{name}.epub", std::process::id()));
        let opf = r#"<?xml version="1.0"?><!DOCTYPE package><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title></metadata><manifest><item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest><spine toc="ncx"><itemref idref="c1"/></spine></package>"#;
        let files = [
            (
                "META-INF/container.xml",
                r#"<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>"#,
            ),
            ("content.opf", opf),
            ("toc.ncx", ncx),
            ("nav.xhtml", NAV),
            (
                "c1.xhtml",
                r#"<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Hi</p></body></html>"#,
            ),
        ];
        let files: Vec<_> = files.iter().map(|(n, t)| (*n, t.as_bytes())).collect();
        crate::archive::write(&path, &files);
        let b = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap(); // the test's own file
        b
    }

    #[test]
    fn an_ncx_with_a_doctype_reads() {
        let ncx = r#"<?xml version="1.0"?><!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd"><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap><navPoint><navLabel><text>First</text></navLabel><content src="c1.xhtml"/></navPoint></navMap></ncx>"#;
        assert_eq!(
            book("doctype", ncx).toc,
            [("First".into(), "c1.xhtml".into())]
        );
    }

    #[test]
    fn an_empty_ncx_gives_way_to_the_nav() {
        let ncx = r#"<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap/></ncx>"#;
        assert_eq!(book("empty", ncx).toc, [("One".into(), "c1.xhtml".into())]);
    }

    #[test]
    fn any_html_entity_and_an_undeclared_epub_prefix_read() {
        let b = body(r#"<html xmlns="http://www.w3.org/1999/xhtml"><body><p epub:type="x">&Agrave;&frac12;&amp;&rsquo;</p></body></html>"#)
            .unwrap()
            .unwrap();
        assert_eq!(
            b.text(""),
            "
À½&’
"
        );
    }
}
