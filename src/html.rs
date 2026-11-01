//! HTML tag soup (a mobi's markup) made XHTML that epub.rs reads, page by page.
//!
//! tokens() cuts the bytes into text, start and end tags, each with its byte offset (a mobi's links
//! point at byte offsets); attributes may be unquoted, names are lower-cased, `mbp:name` loses its
//! prefix and other prefixed names become span; a name XML cannot hold makes "<" text, such a key
//! is dropped; comments and <!...>/<?...?> are dropped. Xhtml writes them well formed: a start of a
//! block closes an open <p> (as libxml2 does), an end tag without its start is dropped, void
//! elements are closed, entities decoded, characters XML cannot hold left out. The head is left
//! out (it ends at its end or the first element not of a head), and title/style/script anywhere.
//! cut() ends the page there: the open elements are closed and opened again on the next page.
//! The page sits in a <div>: epub.rs drops text standing directly in the body before its first
//! element. It ends with a newline, as in Calibre's pages.
use regex::Regex;
use std::sync::LazyLock;

static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]*>").expect("a valid pattern"));
static ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#" id="[^"]*""#).expect("a valid pattern"));

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Text(Vec<u8>),
    /// name, attributes (raw values), written as `<x/>`
    Start(String, Vec<(String, String)>, bool),
    End(String),
}

const VOID: [&str; 12] = [
    "br",
    "hr",
    "img",
    "meta",
    "link",
    "input",
    "col",
    "area",
    "base",
    "wbr",
    "param",
    "pagebreak",
];
/// Elements whose start closes an open <p>.
const CLOSES_P: [&str; 19] = [
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "dl",
    "table",
    "blockquote",
    "pre",
    "hr",
    "address",
    "center",
    "form",
];

/// Head elements: one of these in an open head leaves it open.
const HEAD: [&str; 8] = [
    "title",
    "meta",
    "link",
    "style",
    "script",
    "base",
    "guide",
    "reference",
];
/// Elements whose text is not read, wherever they stand.
const UNREAD: [&str; 3] = ["title", "style", "script"];

/// The name lower-cased; `mbp:x` is `x`, any other prefixed name a span (`o:p` must not close a <p>).
fn local(name: &str) -> String {
    let n = name.to_ascii_lowercase();
    match n.split_once(':') {
        Some(("mbp", l)) => l.to_string(),
        Some(_) => "span".into(),
        None => n,
    }
}

/// Whether XML can hold `b` as a name: a letter or `_` first, then letters, digits, `_.:-`.
fn xml_name(b: &[u8]) -> bool {
    b.first()
        .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || b"_.:-".contains(c))
}

/// The tokens of `raw`, each with the offset of its first byte.
pub fn tokens(raw: &[u8]) -> Vec<(usize, Tok)> {
    let mut out = Vec::new();
    let mut i = 0;
    let n = raw.len();
    while i < n {
        if raw[i] != b'<' {
            let s = i;
            while i < n && raw[i] != b'<' {
                i += 1;
            }
            out.push((s, Tok::Text(raw[s..i].to_vec())));
            continue;
        }
        let s = i;
        if raw[i..].starts_with(b"<!--") {
            i = find(raw, i + 4, b"-->").map_or(n, |e| e + 3);
            continue;
        }
        let next = raw.get(i + 1).copied().unwrap_or(0);
        if next == b'!' || next == b'?' {
            i = find(raw, i, b">").map_or(n, |e| e + 1);
            continue;
        }
        let end = next == b'/';
        let mut j = i + 1 + usize::from(end);
        let ns = j;
        while j < n && !raw[j].is_ascii_whitespace() && raw[j] != b'>' && raw[j] != b'/' {
            j += 1;
        }
        if !raw[ns..j].first().is_some_and(u8::is_ascii_alphabetic) || !xml_name(&raw[ns..j]) {
            // a lone "<" is text
            out.push((s, Tok::Text(vec![b'<'])));
            i = s + 1;
            continue;
        }
        let name = local(&String::from_utf8_lossy(&raw[ns..j]));
        let mut attrs = Vec::new();
        let mut closed = false;
        loop {
            while j < n && raw[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= n {
                break;
            }
            if raw[j] == b'>' {
                j += 1;
                break;
            }
            if raw[j] == b'/' {
                closed = true;
                j += 1;
                continue;
            }
            let ks = j;
            while j < n && !raw[j].is_ascii_whitespace() && !b"=>/".contains(&raw[j]) {
                j += 1;
            }
            let key = String::from_utf8_lossy(&raw[ks..j]).to_ascii_lowercase();
            while j < n && raw[j].is_ascii_whitespace() {
                j += 1;
            }
            let mut val = String::new();
            if j < n && raw[j] == b'=' {
                j += 1;
                while j < n && raw[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < n && (raw[j] == b'"' || raw[j] == b'\'') {
                    let q = raw[j];
                    let vs = j + 1;
                    j = vs;
                    while j < n && raw[j] != q {
                        j += 1;
                    }
                    val = String::from_utf8_lossy(&raw[vs..j]).into_owned();
                    j = (j + 1).min(n);
                } else {
                    let vs = j;
                    while j < n && !raw[j].is_ascii_whitespace() && raw[j] != b'>' {
                        j += 1;
                    }
                    val = String::from_utf8_lossy(&raw[vs..j]).into_owned();
                }
            }
            if xml_name(key.as_bytes())
                && !key.contains(':')
                && !attrs.iter().any(|(k, _)| *k == key)
            {
                attrs.push((key, val));
            }
        }
        out.push((
            s,
            if end {
                Tok::End(name)
            } else {
                Tok::Start(name, attrs, closed)
            },
        ));
        i = j;
    }
    out
}

fn find(hay: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    hay.get(from..)?
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// Text made XML: entities decoded, then & < > " escaped, characters XML forbids left out.
pub fn escape(text: &str) -> String {
    let t = html_escape::decode_html_entities(text);
    let mut s = String::with_capacity(t.len());
    for c in t.chars() {
        match c {
            '&' => s.push_str("&amp;"),
            '<' => s.push_str("&lt;"),
            '>' => s.push_str("&gt;"),
            '"' => s.push_str("&quot;"),
            '\t' | '\n' | '\r' => s.push(c),
            c if (c as u32) < 0x20 || c == '\u{fffe}' || c == '\u{ffff}' => {}
            c => s.push(c),
        }
    }
    s
}

/// Text with its entities decoded.
pub fn unescape(text: &str) -> String {
    html_escape::decode_html_entities(text).into_owned()
}

/// Writes tokens as XHTML pages.
#[derive(Default)]
pub struct Xhtml {
    /// open elements: name and the start tag as written
    open: Vec<(String, String)>,
    page: String,
    in_head: bool,
    /// open title, style and script elements
    unread: usize,
}

impl Xhtml {
    pub fn text(&mut self, t: &str) {
        if !self.in_head && self.unread == 0 {
            self.page.push_str(&escape(t));
        }
    }

    pub fn start(&mut self, name: &str, attrs: &[(String, String)], closed: bool) {
        if UNREAD.contains(&name) {
            self.unread += usize::from(!closed);
            return;
        }
        match name {
            "html" => return,
            "head" => {
                self.in_head = true;
                return;
            }
            "body" => {
                self.in_head = false;
                return;
            }
            _ if self.in_head && HEAD.contains(&name) => return,
            _ => self.in_head = false,
        }
        if CLOSES_P.contains(&name)
            && let Some(p) = self.open.iter().rposition(|(n, _)| n == "p")
        {
            self.close_to(p);
        }
        if name == "li"
            && let Some(p) = self.open.iter().rposition(|(n, _)| n == "li")
            && !self.open[p..].iter().any(|(n, _)| n == "ul" || n == "ol")
        {
            self.close_to(p);
        }
        let mut tag = format!("<{name}");
        for (k, v) in attrs {
            tag.push_str(&format!(" {k}=\"{}\"", escape(v)));
        }
        if closed || VOID.contains(&name) {
            self.page.push_str(&tag);
            self.page.push_str("/>");
        } else {
            tag.push('>');
            self.page.push_str(&tag);
            self.open.push((name.to_string(), tag));
        }
    }

    pub fn end(&mut self, name: &str) {
        if name == "head" {
            self.in_head = false;
        }
        if UNREAD.contains(&name) {
            self.unread = self.unread.saturating_sub(1);
            return;
        }
        if let Some(p) = self.open.iter().rposition(|(n, _)| n == name) {
            self.close_to(p);
        }
    }

    /// Closes the open elements from index `p` up.
    fn close_to(&mut self, p: usize) {
        while self.open.len() > p {
            let (n, _) = self.open.pop().unwrap();
            self.page.push_str(&format!("</{n}>"));
        }
    }

    /// Whether the page so far holds any text beyond whitespace.
    pub fn has_text(&self) -> bool {
        !TAG.replace_all(&self.page, "").trim().is_empty()
    }

    /// Whether a paragraph, heading or list item is open (a page is not cut inside one).
    pub fn in_block(&self) -> bool {
        self.open.iter().any(|(n, _)| {
            matches!(
                n.as_str(),
                "p" | "li" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
            )
        })
    }

    /// The page so far as a document; the open elements are opened again (without ids) for the next.
    pub fn cut(&mut self) -> String {
        let open = self.open.clone();
        self.close_to(0);
        let page = std::mem::take(&mut self.page);
        for (n, tag) in open {
            let tag = ID.replace(&tag, "").into_owned();
            self.page.push_str(&tag);
            self.open.push((n, tag));
        }
        format!(
            r#"<html xmlns="http://www.w3.org/1999/xhtml"><body><div>{page}
</div></body></html>"#
        )
    }

    /// The last page.
    pub fn finish(mut self) -> String {
        self.cut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xhtml(raw: &str) -> Vec<String> {
        let mut w = Xhtml::default();
        let mut pages = Vec::new();
        for (_, t) in tokens(raw.as_bytes()) {
            match t {
                Tok::Text(b) => w.text(&String::from_utf8_lossy(&b)),
                Tok::Start(n, _, _) if n == "pagebreak" => pages.push(w.cut()),
                Tok::Start(n, a, c) => w.start(&n, &a, c),
                Tok::End(n) => w.end(&n),
            }
        }
        pages.push(w.finish());
        pages
            .iter()
            .map(|p| p[54..p.len() - 21].to_string())
            .collect()
    }

    #[test]
    fn soup_becomes_xhtml_and_pages() {
        assert_eq!(
            xhtml(
                r#"<html><head><guide><reference filepos=01 /></guide></head><body><p align=left>a &amp; b<br><b>c</p><hr width="4%"></hr><p>x < y"#
            ),
            [r#"<p align="left">a &amp; b<br/><b>c</b></p><hr width="4%"/><p>x &lt; y</p>"#]
        );
        assert_eq!(
            xhtml(r#"<body><div id="d"><p>one</p><mbp:pagebreak/><p>two</p></div></body></x>"#),
            [r#"<div id="d"><p>one</p></div>"#, "<div><p>two</p></div>"]
        );
        assert_eq!(
            xhtml("<body><p>a\u{1}&#8212;&nbsp;<!-- c --></i>b</p></body>"),
            ["<p>a\u{2014}\u{a0}b</p>"]
        );
    }

    #[test]
    fn inline_elements_read_without_spaces_added() {
        let mut w = Xhtml::default();
        for (_, t) in tokens("<p><tt>(</tt><tt><i>with love</i></tt><tt>).</tt></p>".as_bytes()) {
            match t {
                Tok::Text(b) => w.text(&String::from_utf8_lossy(&b)),
                Tok::Start(n, a, c) => w.start(&n, &a, c),
                Tok::End(n) => w.end(&n),
            }
        }
        let body = crate::epub::body(&w.finish()).unwrap().unwrap();
        assert_eq!(body.text("").trim(), "(with love).");
    }

    #[test]
    fn text_at_a_page_start_is_kept() {
        let mut w = Xhtml::default();
        for (_, t) in tokens(b"<body>a<p>b</p>") {
            match t {
                Tok::Text(b) => w.text(&String::from_utf8_lossy(&b)),
                Tok::Start(n, a, c) => w.start(&n, &a, c),
                Tok::End(n) => w.end(&n),
            }
        }
        let body = crate::epub::body(&w.finish()).unwrap().unwrap();
        assert!(body.text("").contains('a'));
    }

    #[test]
    fn a_head_left_open_ends_at_the_first_body_element() {
        assert_eq!(
            xhtml("<html><head><title>T</title><meta a=b><p>x</p>"),
            ["<p>x</p>"]
        );
    }

    #[test]
    fn text_before_any_tag_is_kept() {
        assert_eq!(xhtml("a<p>b</p>"), ["a<p>b</p>"]);
    }

    #[test]
    fn title_style_script_text_is_left_out_anywhere() {
        assert_eq!(
            xhtml("<body><p>a<script>s</script><style>p{}</style>b</p><title>t</title>"),
            ["<p>ab</p>"]
        );
    }

    #[test]
    fn names_xml_cannot_hold_are_text_or_dropped() {
        assert_eq!(xhtml("<p>a<a=b>c</p>"), ["<p>a&lt;a=b&gt;c</p>"]);
        assert_eq!(xhtml("<p a@b=1 c=2>x</p>"), [r#"<p c="2">x</p>"#]);
    }

    #[test]
    fn a_prefixed_attribute_leaves_the_page_readable() {
        let mut w = Xhtml::default();
        for (_, t) in tokens(br#"<section epub:type="chapter" xml:lang="en"><p>x</p></section>"#) {
            match t {
                Tok::Text(b) => w.text(&String::from_utf8_lossy(&b)),
                Tok::Start(n, a, c) => w.start(&n, &a, c),
                Tok::End(n) => w.end(&n),
            }
        }
        let body = crate::epub::body(&w.finish()).unwrap().unwrap();
        assert_eq!(body.text("").trim(), "x");
    }

    #[test]
    fn a_prefixed_name_other_than_mbp_becomes_span() {
        assert_eq!(xhtml("<p>a<o:p>b</o:p>c</p>"), ["<p>a<span>b</span>c</p>"]);
    }
}
