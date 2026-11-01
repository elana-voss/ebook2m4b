//! An EPUB read as a Book (book/mod.rs): the spine's items, the table of contents flattened
//! (the NCX the spine names, the EPUB 3 nav when that gives none), the DC creators and titles.
//! Traps:
//! - manifest hrefs are percent-decoded, hrefs in the contents and in links are not;
//! - a "cover"-properties page has no text.
use crate::book::{Book, Page, as_xml, base_name, body, xml};
use anyhow::{Context, Result};
use std::path::Path;

const OPF: &str = "http://www.idpf.org/2007/opf";
const DC: &str = "http://purl.org/dc/elements/1.1/";

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

pub fn read(path: &Path) -> Result<Book> {
    let mut z = crate::setup::archive::Archive::open(path)
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
    // the nav when the NCX gave nothing
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

#[cfg(test)]
mod tests {
    use super::*;

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
            std::env::temp_dir().join(format!("ebook2m4b-epub-{}-{name}.epub", std::process::id()));
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
        crate::setup::archive::write(&path, &files);
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
}
