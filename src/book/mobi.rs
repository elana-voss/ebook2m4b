//! A mobi, azw or azw3 read as an epub::Book, without Calibre, the way Calibre's conversion gives it.
//!
//! A Palm database: record 0 holds the PalmDOC and MOBI headers and EXTH (title 503, authors 100,
//! "Last, First" turned); the text records follow (as many as there are), each with
//! trailing entries (extra data flags, MOBI 5 on) cut off, then PalmDOC
//! (LZ77), HUFF/CDIC (kf8.rs) or not compressed; DRM is refused. MOBI 6: the markup is tag soup;
//! each `filepos=` link target gets an anchor `fileposN` there, the pages are cut at <mbp:pagebreak>
//! (index_split_NNN.html), the contents come from the NCX index (INDX records,
//! labels in CNCX). A file holding MOBI 6 and KF8 is read by
//! its MOBI 6 part; a KF8-only one (azw3, version 8) by kf8.rs, whose kindle:pos links and NCX tag 6
//! places become anchors as filepos ones do (book()). Text is UTF-8 or cp1252 as the header says.
use crate::book::epub::{self, Book, Page};
use crate::book::soup::{Tok, Xhtml, tokens};
use crate::book::kf8::{Huff, kf8};
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(crate) fn u16_at(d: &[u8], at: usize) -> u32 {
    d.get(at..at + 2)
        .map_or(0, |b| u32::from(u16::from_be_bytes([b[0], b[1]])))
}

pub(crate) fn u32_at(d: &[u8], at: usize) -> u32 {
    d.get(at..at + 4)
        .map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// The database's records.
fn records(data: &[u8]) -> Result<Vec<&[u8]>> {
    let n = u16_at(data, 76) as usize;
    let offs: Vec<usize> = (0..n).map(|i| u32_at(data, 78 + 8 * i) as usize).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let end = offs.get(i + 1).copied().unwrap_or(data.len());
        out.push(
            data.get(offs[i]..end)
                .context("a record lies outside the file")?,
        );
    }
    Ok(out)
}

/// What record 0 says.
#[derive(Default)]
pub(crate) struct Header {
    compression: u32,
    text_len: usize,
    text_records: usize,
    drm: u32,
    pub(crate) utf8: bool,
    version: u32,
    extra_flags: u32,
    pub(crate) ncx: u32,
    /// KF8's records (relative to its record 0): the flows' table, fragments, skeletons
    pub(crate) fdst: u32,
    pub(crate) frag: u32,
    pub(crate) skel: u32,
    /// HUFF/CDIC: the HUFF record and how many records with the CDICs after it
    huff: u32,
    huff_count: u32,
    full_name: String,
    exth: Vec<(u32, Vec<u8>)>,
}

fn header(r0: &[u8]) -> Result<Header> {
    if r0.get(16..20) != Some(b"MOBI") {
        bail!("no MOBI header");
    }
    let hlen = u32_at(r0, 20) as usize;
    let utf8 = u32_at(r0, 28) == 65001;
    let mut exth = Vec::new();
    if u32_at(r0, 0x80) & 0x40 != 0 && r0.get(16 + hlen..20 + hlen) == Some(b"EXTH") {
        let mut p = 16 + hlen + 12;
        for _ in 0..u32_at(r0, 16 + hlen + 8) {
            let (t, len) = (u32_at(r0, p), u32_at(r0, p + 4) as usize);
            if len < 8 || p + len > r0.len() {
                break;
            }
            exth.push((t, r0[p + 8..p + len].to_vec()));
            p += len;
        }
    }
    let (at, len) = (u32_at(r0, 0x54) as usize, u32_at(r0, 0x58) as usize);
    let full_name = decode(r0.get(at..at + len).unwrap_or_default(), utf8);
    // a field the header is too short to hold is absent
    let field = |at: usize| {
        if hlen + 16 >= at + 4 {
            u32_at(r0, at)
        } else {
            u32::MAX
        }
    };
    Ok(Header {
        compression: u16_at(r0, 0),
        text_len: u32_at(r0, 4) as usize,
        text_records: u16_at(r0, 8) as usize,
        drm: u16_at(r0, 12),
        utf8,
        version: u32_at(r0, 0x24),
        extra_flags: if hlen >= 0xE4 && u32_at(r0, 0x24) >= 5 {
            u16_at(r0, 0xF2)
        } else {
            0
        },
        ncx: if hlen >= 0xE8 {
            u32_at(r0, 0xF4)
        } else {
            u32::MAX
        },
        fdst: field(0xC0),
        frag: field(0xF8),
        skel: field(0xFC),
        huff: field(0x70),
        huff_count: field(0x74),
        full_name,
        exth,
    })
}

/// Bytes as text: UTF-8, else cp1252.
pub(crate) fn decode(b: &[u8], utf8: bool) -> String {
    if utf8 {
        return String::from_utf8_lossy(b).into_owned();
    }
    const HIGH: &str = "€\u{81}‚ƒ„…†‡ˆ‰Š‹Œ\u{8d}Ž\u{8f}\u{90}‘’“”•–—˜™š›œ\u{9d}žŸ";
    b.iter()
        .map(|&c| match c {
            0x80..=0x9F => HIGH.chars().nth(usize::from(c - 0x80)).unwrap_or('?'),
            c => char::from(c),
        })
        .collect()
}

/// A text record without its trailing entries.
fn trailing(rec: &[u8], flags: u32) -> &[u8] {
    let mut r = rec;
    // bit 1's entry lies last, the multibyte bytes (bit 0) before the entries
    for bit in 1..16 {
        if flags & (1 << bit) != 0 {
            // a size written backwards at the end, 7 bits a byte, the last read has the high bit
            let (mut size, mut shift) = (0usize, 0);
            for k in 1..=4.min(r.len()) {
                let b = r[r.len() - k];
                size |= usize::from(b & 0x7F) << shift;
                shift += 7;
                if b & 0x80 != 0 {
                    break;
                }
            }
            r = &r[..r.len().saturating_sub(size)];
        }
    }
    if flags & 1 != 0
        && let Some(&last) = r.last()
    {
        r = &r[..r.len().saturating_sub(usize::from(last & 3) + 1)];
    }
    r
}

/// PalmDOC's LZ77.
fn palmdoc(d: &[u8]) -> Vec<u8> {
    let mut o: Vec<u8> = Vec::with_capacity(d.len() * 2);
    let mut i = 0;
    while i < d.len() {
        let c = d[i];
        i += 1;
        match c {
            0 | 9..=0x7F => o.push(c),
            1..=8 => {
                let end = (i + usize::from(c)).min(d.len());
                o.extend_from_slice(&d[i..end]);
                i = end;
            }
            0xC0.. => {
                o.push(b' ');
                o.push(c ^ 0x80);
            }
            _ => {
                let Some(&lo) = d.get(i) else { break };
                i += 1;
                let pair = (usize::from(c) << 8) | usize::from(lo);
                let (dist, n) = ((pair >> 3) & 0x7FF, (pair & 7) + 3);
                if dist == 0 || dist > o.len() {
                    continue;
                }
                for _ in 0..n {
                    o.push(o[o.len() - dist]);
                }
            }
        }
    }
    o
}

/// The book's text.
pub(crate) fn text(recs: &[&[u8]], h: &Header) -> Result<Vec<u8>> {
    if h.drm != 0 {
        bail!("the book has DRM");
    }
    let mut huff = match h.compression {
        17480 => {
            let first = h.huff as usize;
            let cdics = recs
                .get(first + 1..first + h.huff_count as usize)
                .context("the CDIC records lie outside the file")?;
            Some(Huff::new(
                recs.get(first).context("no HUFF record")?,
                cdics,
            )?)
        }
        _ => None,
    };
    let mut out = Vec::new();
    // a book with fewer text records than its header says is read as far as they go
    for rec in recs.iter().skip(1).take(h.text_records) {
        let r = trailing(rec, h.extra_flags);
        match (h.compression, huff.as_mut()) {
            (1, _) => out.extend_from_slice(r),
            (2, _) => out.extend(palmdoc(r)),
            (_, Some(huff)) => out.extend(huff.unpack(r)?),
            (c, None) => bail!("unsupported compression {c}"),
        }
    }
    out.truncate(h.text_len);
    Ok(out)
}

/// A number as indexes write it: 7 bits a byte, the last with the high bit. (value, bytes read)
fn varint(d: &[u8]) -> (u32, usize) {
    let mut v = 0u32;
    for (k, &b) in d.iter().enumerate().take(5) {
        v = (v << 7) | u32::from(b & 0x7F);
        if b & 0x80 != 0 {
            return (v, k + 1);
        }
    }
    (v, d.len().min(5))
}

/// One entry of an index: its name and tag values.
pub(crate) struct Entry {
    pub(crate) name: Vec<u8>,
    pub(crate) tags: BTreeMap<u32, Vec<u32>>,
}

/// The entries of the index whose header is record `at`, and its CNCX strings by offset.
pub(crate) fn index(
    recs: &[&[u8]],
    at: usize,
    utf8: bool,
) -> Result<(Vec<Entry>, BTreeMap<u32, String>)> {
    let h = recs.get(at).context("no index record")?;
    if h.get(..4) != Some(b"INDX") {
        bail!("no INDX header");
    }
    let hlen = u32_at(h, 4) as usize;
    let (count, ncncx) = (u32_at(h, 24) as usize, u32_at(h, 52) as usize);
    let tagx = h.get(hlen..).context("no TAGX")?;
    if tagx.get(..4) != Some(b"TAGX") {
        bail!("no TAGX");
    }
    let (tlen, cbytes) = (u32_at(tagx, 4) as usize, u32_at(tagx, 8) as usize);
    let tags: Vec<[u8; 4]> = tagx
        .get(12..tlen)
        .unwrap_or_default()
        .as_chunks::<4>()
        .0
        .to_vec();
    let mut entries = Vec::new();
    for r in recs.iter().skip(at + 1).take(count) {
        let (start, n) = (u32_at(r, 20) as usize, u32_at(r, 24) as usize);
        let offs: Vec<usize> = (0..n)
            .map(|k| u16_at(r, start + 4 + 2 * k) as usize)
            .collect();
        for (k, &off) in offs.iter().enumerate() {
            let end = offs.get(k + 1).copied().unwrap_or(start).min(r.len());
            let e = r.get(off..end).unwrap_or_default();
            let Some(&id_len) = e.first() else { continue };
            let mut p = 1 + usize::from(id_len);
            let control = e.get(p..p + cbytes).unwrap_or_default().to_vec();
            p += cbytes;
            entries.push(Entry {
                name: e.get(1..p - cbytes).unwrap_or_default().to_vec(),
                tags: tag_values(&tags, &control, e.get(p..).unwrap_or_default()),
            });
        }
    }
    let mut strings = BTreeMap::new();
    for (k, r) in recs.iter().skip(at + 1 + count).take(ncncx).enumerate() {
        cncx(r, k, utf8, &mut strings);
    }
    Ok((entries, strings))
}

/// The strings of the `k`th CNCX record, each a varint length and its bytes, by offset.
fn cncx(r: &[u8], k: usize, utf8: bool, out: &mut BTreeMap<u32, String>) {
    let mut p = 0;
    while p < r.len() {
        let (len, used) = varint(&r[p..]);
        let s = r.get(p + used..p + used + len as usize).unwrap_or_default();
        out.insert((k as u32) * 0x10000 + p as u32, decode(s, utf8));
        p += used + len as usize;
    }
}

/// An entry's tag values by the TAGX table.
fn tag_values(tags: &[[u8; 4]], control: &[u8], data: &[u8]) -> BTreeMap<u32, Vec<u32>> {
    let mut cb = 0;
    // (tag, value count, value bytes, values per entry)
    let mut found = Vec::new();
    for &[tag, per, mask, end] in tags {
        if end == 1 {
            cb += 1;
            continue;
        }
        let Some(&c) = control.get(cb) else { break };
        let value = c & mask;
        if value == 0 {
            continue;
        }
        if value == mask {
            found.push((tag, if mask.count_ones() > 1 { None } else { Some(1) }, per));
        } else {
            found.push((tag, Some(u32::from(value >> mask.trailing_zeros())), per));
        }
    }
    let mut p = 0;
    let mut sizes = Vec::new();
    for f in &mut found {
        if f.1.is_none() {
            let (v, used) = varint(data.get(p..).unwrap_or_default());
            p += used;
            sizes.push(Some(v as usize));
        } else {
            sizes.push(None);
        }
    }
    let mut out = BTreeMap::new();
    for ((tag, count, per), size) in found.into_iter().zip(sizes) {
        let mut values = Vec::new();
        if let Some(c) = count {
            for _ in 0..c * u32::from(per) {
                let (v, used) = varint(data.get(p..).unwrap_or_default());
                p += used;
                values.push(v);
            }
        } else {
            let mut used_all = 0;
            while used_all < size.unwrap_or(0) && p < data.len() {
                let (v, used) = varint(&data[p..]);
                p += used;
                used_all += used;
                values.push(v);
            }
        }
        out.insert(u32::from(tag), values);
    }
    out
}

/// An NCX entry: its label, text offset (tag 1) and KF8 position (tag 6: fragment, offset).
pub(crate) type NcxEntry = (String, u32, Option<(usize, usize)>);

/// The NCX index, parents before their children.
pub(crate) fn ncx(recs: &[&[u8]], at: usize, utf8: bool) -> Result<Vec<NcxEntry>> {
    let (entries, cncx) = index(recs, at, utf8)?;
    let parent = |e: &Entry| e.tags.get(&21).and_then(|v| v.first()).copied();
    let mut kids: BTreeMap<Option<u32>, Vec<usize>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() {
        kids.entry(parent(e)).or_default().push(i);
    }
    fn walk(
        i: usize,
        entries: &[Entry],
        cncx: &BTreeMap<u32, String>,
        kids: &BTreeMap<Option<u32>, Vec<usize>>,
        out: &mut Vec<NcxEntry>,
    ) {
        let e = &entries[i];
        let first = |t| e.tags.get(&t).and_then(|v: &Vec<u32>| v.first()).copied();
        let label = first(3)
            .and_then(|o| cncx.get(&o))
            .cloned()
            .unwrap_or_default();
        let pos = e.tags.get(&6).filter(|v| v.len() >= 2);
        let pos = pos.map(|v| (v[0] as usize, v[1] as usize));
        out.push((label, first(1).unwrap_or(0), pos));
        for &k in kids.get(&Some(i as u32)).map_or(&[][..], |v| v) {
            walk(k, entries, cncx, kids, out);
        }
    }
    let mut out = Vec::new();
    for &i in kids.get(&None).map_or(&[][..], |v| v) {
        walk(i, &entries, &cncx, &kids, &mut out);
    }
    Ok(out)
}

/// `filepos=` values in the markup.
fn link_targets(raw: &[u8]) -> BTreeSet<usize> {
    static RE: std::sync::LazyLock<regex::bytes::Regex> = std::sync::LazyLock::new(|| {
        regex::bytes::Regex::new(r#"(?i)filepos=['"]?0*(\d+)"#).unwrap()
    });
    RE.captures_iter(raw)
        .filter_map(|c| std::str::from_utf8(&c[1]).ok()?.parse().ok())
        .collect()
}

fn page_name(i: usize) -> String {
    format!("index_split_{i:03}.html")
}

/// What a tag's attribute links to (a filepos, a kindle:pos), as an offset in the markup.
pub(crate) type Link<'a> = &'a dyn Fn(&str, &str) -> Option<usize>;

/// The markup as pages with anchors at the targets, a link made `href="#fileposN"`; (pages, the
/// page of each target).
fn pages(
    raw: &[u8],
    targets: &BTreeSet<usize>,
    link: Link,
    utf8: bool,
) -> (Vec<String>, BTreeMap<usize, usize>) {
    let toks = tokens(raw);
    let mut w = Xhtml::default();
    let mut out = Vec::new();
    let mut at_page = BTreeMap::new();
    let mut next = targets.iter().peekable();
    let mut anchor =
        |w: &mut Xhtml, upto: usize, page: usize, at_page: &mut BTreeMap<usize, usize>| {
            while let Some(&&t) = next.peek().filter(|&&&t| t <= upto) {
                next.next();
                w.start("a", &[("id".into(), format!("filepos{t}"))], false);
                w.end("a");
                at_page.insert(t, page);
            }
        };
    for (k, (at, tok)) in toks.iter().enumerate() {
        let end = toks.get(k + 1).map_or(raw.len(), |t| t.0);
        match tok {
            Tok::Text(b) => {
                let mut s = 0;
                for &t in targets.range(at..&end) {
                    let cut = t - at;
                    w.text(&decode(&b[s..cut], utf8));
                    anchor(&mut w, t, out.len(), &mut at_page);
                    s = cut;
                }
                w.text(&decode(&b[s..], utf8));
            }
            _ => {
                anchor(
                    &mut w,
                    end.saturating_sub(1).max(*at),
                    out.len(),
                    &mut at_page,
                );
                match tok {
                    // a page without text is not cut off: its anchors open the next
                    Tok::Start(n, _, _) if n == "pagebreak" => {
                        if w.has_text() {
                            out.push(w.cut())
                        }
                    }
                    Tok::Start(n, a, c) => {
                        // a link replaces the href it may also have
                        match a.iter().find_map(|(k, v)| link(k, v)) {
                            Some(t) => {
                                let mut a: Vec<_> = a
                                    .iter()
                                    .filter(|(k, v)| k != "href" && link(k, v).is_none())
                                    .cloned()
                                    .collect();
                                a.push(("href".into(), format!("#filepos{t}")));
                                w.start(n, &a, *c)
                            }
                            None => w.start(n, a, *c),
                        }
                    }
                    Tok::End(n) => w.end(n),
                    Tok::Text(_) => {}
                }
            }
        }
    }
    out.push(w.finish());
    (out, at_page)
}

/// The book's metadata: titles and authors.
fn metadata(h: &Header) -> (Vec<String>, Vec<String>) {
    let exth = |t: u32| -> Vec<String> {
        h.exth
            .iter()
            .filter(|(k, _)| *k == t)
            .map(|(_, v)| decode(v, h.utf8).trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let titles = exth(503);
    let titles = if titles.is_empty() {
        vec![h.full_name.trim().to_string()]
    } else {
        titles
    };
    // "Carroll, Lewis" is "Lewis Carroll"
    let turn = regex::Regex::new(r"^([^,]+?)\s*,\s+([^,]+)$").unwrap();
    let authors = exth(100)
        .iter()
        .map(|a| turn.replace(a, "$2 $1").into_owned())
        .collect();
    (titles, authors)
}

/// The book.
pub fn read(path: &Path) -> Result<Book> {
    let data = std::fs::read(path).with_context(|| path.display().to_string())?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let recs = records(&data)?;
    let h = header(recs.first().context("no records")?)
        .with_context(|| format!("{name} is no mobi"))?;
    if h.version == 8 {
        return kf8(&recs, &h).with_context(|| format!("{name} cannot be read"));
    }
    let raw = text(&recs, &h).with_context(|| format!("{name} cannot be read"))?;
    let toc = if h.ncx != u32::MAX && (h.ncx as usize) < recs.len() {
        ncx(&recs, h.ncx as usize, h.utf8)?
            .into_iter()
            .map(|(label, at, _)| (label, at as usize))
            .collect()
    } else {
        Vec::new()
    };
    let filepos = |k: &str, v: &str| {
        let t = (k == "filepos").then(|| link_targets(format!("{k}={v}").as_bytes()))?;
        t.first().copied()
    };
    book(&raw, toc, link_targets(&raw), &filepos, &h)
}

/// The book of the markup, its contents (label, offset) and link targets.
pub(crate) fn book(
    raw: &[u8],
    toc: Vec<(String, usize)>,
    mut targets: BTreeSet<usize>,
    link: Link,
    h: &Header,
) -> Result<Book> {
    targets.extend(toc.iter().map(|e| e.1));
    targets.retain(|&t| t < raw.len());
    let (texts, at_page) = pages(raw, &targets, link, h.utf8);
    let spine = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let html = t.replace("href=\"#filepos", "href=\"\u{0}");
            let html = regex::Regex::new("href=\"\u{0}(\\d+)\"")
                .unwrap()
                .replace_all(&html, |c: &regex::Captures| {
                    let t: usize = c[1].parse().unwrap_or(0);
                    format!(
                        "href=\"{}#filepos{t}\"",
                        page_name(at_page.get(&t).copied().unwrap_or(0))
                    )
                })
                .into_owned();
            Ok(Page {
                name: page_name(i),
                body: epub::body(&html)?,
                is_doc: true,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let toc = toc
        .into_iter()
        .filter_map(|(label, at)| {
            let p = at_page.get(&at)?;
            Some((
                label.trim().to_string(),
                format!("{}#filepos{at}", page_name(*p)),
            ))
        })
        .collect();
    let (titles, creators) = metadata(h);
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

    #[test]
    fn palmdoc_and_trailing_entries() {
        // literal, a run of 2 copied bytes, a space pair, then distance 3 length 3
        assert_eq!(palmdoc(b"ab\x02cd\xC5\x80\x18"), b"abcd Ed E".to_vec());
        // a trailing entry of size 3 last, before it the multibyte bytes (1 + 1)
        assert_eq!(trailing(b"text\xAA\x01\x00\x00\x83", 3), b"text");
        assert_eq!(decode(b"\x93a\xe9\x94", false), "\u{201c}a\u{e9}\u{201d}");
    }

    #[test]
    fn a_page_without_text_is_not_cut_off() {
        let raw = b"<p>a</p><mbp:pagebreak/><br/> <mbp:pagebreak/><p>b</p>";
        let (pages, at_page) = pages(raw, &BTreeSet::from([25]), &|_, _| None, true);
        assert_eq!((pages.len(), at_page[&25]), (2, 1));
    }

    #[test]
    fn a_link_with_href_and_filepos_has_one_href() {
        let (pages, _) = pages(
            br#"<p><a href="x" filepos=3>a</a></p>"#,
            &BTreeSet::from([3]),
            &|k, v| (k == "filepos").then(|| v.parse().ok())?,
            true,
        );
        assert_eq!(pages[0].matches("href=").count(), 1);
    }

    #[test]
    fn authors_written_last_first_are_turned() {
        let h = Header {
            exth: vec![
                (100, b"Carroll, Lewis".to_vec()),
                (100, b"Kim Stanley".to_vec()),
            ],
            ..Default::default()
        };
        assert_eq!(metadata(&h).1, ["Lewis Carroll", "Kim Stanley"]);
    }

    #[test]
    fn an_empty_cncx_string_takes_one_byte() {
        let mut m = BTreeMap::new();
        cncx(b"\x80\x82ab", 0, true, &mut m);
        assert_eq!(m.get(&1).map(String::as_str), Some("ab"));
    }

    #[test]
    fn extra_flags_only_from_version_5() {
        let r0 = |version: u8| {
            let mut r = vec![0u8; 16 + 0xE8];
            r[16..20].copy_from_slice(b"MOBI");
            r[23] = 0xE8;
            r[0x27] = version;
            r[0xF3] = 3;
            r
        };
        assert_eq!(header(&r0(4)).unwrap().extra_flags, 0);
        assert_eq!(header(&r0(6)).unwrap().extra_flags, 3);
    }

    #[test]
    fn fewer_text_records_than_the_header_says_are_read() {
        let h = Header {
            compression: 1,
            text_len: 10,
            text_records: 2,
            ..Default::default()
        };
        assert_eq!(text(&[b"r0", b"ab"], &h).unwrap(), b"ab");
    }
}
