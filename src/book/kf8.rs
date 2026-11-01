//! KF8 (azw3) and HUFF/CDIC, for mobi.rs.
//!
//! KF8: flow 0 of the FDST holds the skeletons, each followed by its fragments; parts() puts each
//! fragment in at its insert position (the fragment index's entry name, decimal), a page break
//! between two parts. kindle:pos:fid:XXXX:off:YYYYYYYYYY (base 32) counts from a fragment's place
//! in the flow; the NCX's tag 6 is such a place. Skeleton tag 6 is (start, length), tag 1 the
//! fragment count; fragment tag 6 [1] the length. Without those indexes the flow is one part; an NCX
//! entry whose tag 6 lies nowhere takes its tag 1 offset. No KF8 guide: contents only from the NCX.
//! HUFF/CDIC: Huff unpacks a text record; a phrase may itself be coded and is unpacked once.
use crate::book::Book;
use crate::book::mobi::{Entry, Header, book, index, ncx, text, u16_at, u32_at};
use anyhow::{Context, Result, bail};

/// HUFF/CDIC: Huffman codes for the phrases of a dictionary (the CDIC records), a phrase perhaps
/// coded itself. Codes are read 32 bits at a time: the first 8 bits find the length in the cache
/// table, or the shortest length to try; the base table holds each length's lowest and highest code.
pub(crate) struct Huff {
    /// by a code's first 8 bits: (length, final, highest code)
    cache: Vec<(u32, bool, u64)>,
    /// by length 1..=32: (lowest code, highest code), both as 32-bit numbers
    base: Vec<(u64, u64)>,
    /// the phrases and whether each is plain; None while one is unpacked
    dict: Vec<Option<(Vec<u8>, bool)>>,
}

impl Huff {
    pub(crate) fn new(huff: &[u8], cdics: &[&[u8]]) -> Result<Huff> {
        if huff.get(..4) != Some(b"HUFF") {
            bail!("no HUFF record");
        }
        let (c, b) = (u32_at(huff, 8) as usize, u32_at(huff, 12) as usize);
        if huff.len() < c + 4 * 256 || huff.len() < b + 8 * 32 {
            bail!("the HUFF record is short");
        }
        let cache = (0..256)
            .map(|i| {
                let v = u32_at(huff, c + 4 * i);
                let len = (v & 0x1F).max(1);
                (
                    len,
                    v & 0x80 != 0,
                    ((u64::from(v >> 8) + 1) << (32 - len)) - 1,
                )
            })
            .collect();
        let base = (1..=32u32)
            .map(|len| {
                let at = b + 8 * (len as usize - 1);
                let (lo, hi) = (u64::from(u32_at(huff, at)), u64::from(u32_at(huff, at + 4)));
                (lo << (32 - len), ((hi + 1) << (32 - len)) - 1)
            })
            .collect();
        let mut dict = Vec::new();
        for r in cdics {
            if r.get(..4) != Some(b"CDIC") {
                bail!("no CDIC record");
            }
            let (total, bits) = (u32_at(r, 8) as usize, u32_at(r, 12));
            // a record holds 2^bits phrases, the last the rest
            let n = (1usize << bits.min(16)).min(total.saturating_sub(dict.len()));
            for i in 0..n {
                let at = 16 + u16_at(r, 16 + 2 * i) as usize;
                let len = u16_at(r, at) as usize;
                let p = r
                    .get(at + 2..at + 2 + (len & 0x7FFF))
                    .context("a CDIC phrase lies outside its record")?;
                dict.push(Some((p.to_vec(), len & 0x8000 != 0)));
            }
        }
        Ok(Huff { cache, base, dict })
    }

    pub(crate) fn unpack(&mut self, d: &[u8]) -> Result<Vec<u8>> {
        let word = |p: usize| {
            let mut w = [0u8; 8];
            for (k, b) in w.iter_mut().enumerate() {
                *b = d.get(p + k).copied().unwrap_or(0);
            }
            u64::from_be_bytes(w)
        };
        let mut left = d.len() as i64 * 8;
        let (mut pos, mut n) = (0, 32i64);
        let mut x = word(0);
        let mut out = Vec::new();
        loop {
            if n <= 0 {
                pos += 4;
                x = word(pos);
                n += 32;
            }
            let code = (x >> n) & 0xFFFF_FFFF;
            let (mut len, last, mut max) = self.cache[(code >> 24) as usize];
            if !last {
                while len < 32 && code < self.base[len as usize - 1].0 {
                    len += 1;
                }
                max = self.base[len as usize - 1].1;
            }
            n -= i64::from(len);
            left -= i64::from(len);
            if left < 0 {
                break;
            }
            let r =
                (max.checked_sub(code).context("a HUFF code out of range")? >> (32 - len)) as usize;
            let Some(entry) = self.dict.get_mut(r) else {
                bail!("a HUFF code past the dictionary");
            };
            let Some((p, plain)) = entry.take() else {
                bail!("a HUFF phrase holds itself");
            };
            let p = if plain { p } else { self.unpack(&p)? };
            out.extend_from_slice(&p);
            self.dict[r] = Some((p, true));
        }
        Ok(out)
    }
}

/// A KF8 book's XHTML: its parts (skeletons with their fragments put in) one after another,
/// a page break between two.
struct Parts {
    text: Vec<u8>,
    /// each part: (start in the flow, start in text, length)
    at: Vec<(usize, usize, usize)>,
    /// each fragment's place in the flow (where kindle:pos:fid counts from)
    frags: Vec<usize>,
}

impl Parts {
    /// A place in the flow as an offset in text.
    fn at(&self, pos: usize) -> Option<usize> {
        let &(s, t, _) = self.at.iter().find(|&&(s, _, n)| pos >= s && pos < s + n)?;
        Some(t + pos - s)
    }

    /// kindle:pos:fid:off as an offset in text.
    fn pos(&self, fid: usize, off: usize) -> Option<usize> {
        self.at(self.frags.get(fid)? + off)
    }

    /// An NCX entry's place: its kindle:pos (tag 6), else its offset in the flow (tag 1).
    fn place(&self, at: u32, pos: Option<(usize, usize)>) -> Option<usize> {
        pos.and_then(|(f, o)| self.pos(f, o))
            .or_else(|| self.at(at as usize))
    }
}

/// The parts of flow 0: skeletons (start, length, fragment count) each followed by its fragments
/// (insert position, length), a fragment inserted at its position in the part as put together.
/// Without skeletons the flow is one part.
fn parts(flow: &[u8], skels: &[(usize, usize, usize)], frags: &[(usize, usize)]) -> Parts {
    let whole = [(0, flow.len(), 0)];
    let skels = if skels.is_empty() { &whole[..] } else { skels };
    let (mut text, mut at, mut next) = (Vec::new(), Vec::new(), 0);
    for &(start, len, count) in skels {
        let mut part = flow.get(start..start + len).unwrap_or_default().to_vec();
        let mut p = start + len;
        for &(insert, n) in frags.iter().skip(next).take(count) {
            let piece = flow.get(p..p + n).unwrap_or_default();
            let i = insert.saturating_sub(start).min(part.len());
            part.splice(i..i, piece.iter().copied());
            p += n;
        }
        next += count;
        if !at.is_empty() {
            text.extend_from_slice(b"<mbp:pagebreak/>");
        }
        at.push((start, text.len(), part.len()));
        text.extend(part);
    }
    let frags = frags.iter().map(|f| f.0).collect();
    Parts { text, at, frags }
}

/// A link `kindle:pos:fid:XXXX:off:YYYYYYYYYY` (base 32) as (fragment, offset).
fn kindle_pos(v: &str) -> Option<(usize, usize)> {
    let (fid, off) = v.strip_prefix("kindle:pos:fid:")?.split_once(":off:")?;
    let off = off.split(|c: char| !c.is_ascii_alphanumeric()).next()?;
    Some((
        usize::from_str_radix(fid, 32).ok()?,
        usize::from_str_radix(off, 32).ok()?,
    ))
}

/// A KF8 book: the flows' table, skeleton and fragment indexes, the NCX
/// pointing at kindle:pos places.
pub(crate) fn kf8(recs: &[&[u8]], h: &Header) -> Result<Book> {
    let raw = text(recs, h)?;
    let fdst = recs
        .get(h.fdst as usize)
        .filter(|r| r.get(..4) == Some(b"FDST"));
    let flow = match fdst {
        Some(r) => {
            let (s, e) = (u32_at(r, 12) as usize, u32_at(r, 16) as usize);
            raw.get(s..e.min(raw.len())).unwrap_or_default()
        }
        None => &raw[..],
    };
    let two = |e: &Entry, t: u32| {
        let v = e.tags.get(&t).map_or(&[][..], |v| v);
        (
            v.first().copied().unwrap_or(0) as usize,
            v.get(1).copied().unwrap_or(0) as usize,
        )
    };
    // a book without its skeleton or fragment index is read as one part (parts())
    let skel = index(recs, h.skel as usize, h.utf8).map_or_else(|_| Vec::new(), |r| r.0);
    let skels: Vec<_> = skel
        .iter()
        .map(|e| {
            let (start, len) = two(e, 6);
            (start, len, two(e, 1).0)
        })
        .collect();
    let frag = index(recs, h.frag as usize, h.utf8).map_or_else(|_| Vec::new(), |r| r.0);
    let frags: Vec<_> = frag
        .iter()
        .map(|e| {
            let insert = std::str::from_utf8(&e.name)
                .ok()
                .and_then(|n| n.parse().ok());
            (insert.unwrap_or(0), two(e, 6).1)
        })
        .collect();
    let p = parts(flow, &skels, &frags);
    let toc = if h.ncx != u32::MAX && (h.ncx as usize) < recs.len() {
        ncx(recs, h.ncx as usize, h.utf8)?
    } else {
        Vec::new()
    };
    let toc = toc
        .into_iter()
        .filter_map(|(label, at, pos)| Some((label, p.place(at, pos)?)))
        .collect();
    static RE: std::sync::LazyLock<regex::bytes::Regex> = std::sync::LazyLock::new(|| {
        regex::bytes::Regex::new(r"kindle:pos:fid:[0-9A-Va-v]+:off:[0-9A-Va-v]+").unwrap()
    });
    let targets = RE
        .find_iter(&p.text)
        .filter_map(|m| kindle_pos(std::str::from_utf8(m.as_bytes()).ok()?))
        .filter_map(|(f, o)| p.pos(f, o))
        .collect();
    let link = |k: &str, v: &str| {
        let (f, o) = kindle_pos(v).filter(|_| k == "href")?;
        p.pos(f, o)
    };
    book(&p.text, toc, targets, &link, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A HUFF record and one CDIC record for phrases 0 "ab", 1 "c", 2 compressed ("cabcabc"), coded
    /// "1" (length 1, found in the cache table) and "01", "00" (length 2, by the base table).
    fn huff_tables() -> (Vec<u8>, Vec<u8>) {
        let mut huff = b"HUFF\0\0\0\x18\0\0\0\x18\0\0\x04\x18".to_vec();
        huff.extend([0u8; 8]);
        for i in 0..256u32 {
            let v: u32 = if i >= 128 { (1 << 8) | 0x80 | 1 } else { 1 };
            huff.extend(v.to_be_bytes());
        }
        let mut base = [0u32; 64];
        base[0] = 1; // no code of length 1 below "1"
        base[3] = 2; // length 2: "01" is phrase 1, "00" phrase 2
        for v in base {
            huff.extend(v.to_be_bytes());
        }
        let phrases: [(&[u8], bool); 3] = [(b"ab", true), (b"c", true), (&[0x6D], false)];
        let mut cdic = b"CDIC\0\0\0\x10\0\0\0\x03\0\0\0\x02".to_vec();
        let mut body = Vec::new();
        let mut offs = Vec::new();
        for (p, plain) in phrases {
            offs.push(6 + body.len() as u16);
            let len = p.len() as u16 | if plain { 0x8000 } else { 0 };
            body.extend(len.to_be_bytes());
            body.extend(p);
        }
        for o in offs {
            cdic.extend(o.to_be_bytes());
        }
        cdic.extend(body);
        (huff, cdic)
    }

    #[test]
    fn huff_cdic_text_is_unpacked() {
        let (huff, cdic) = huff_tables();
        let mut h = Huff::new(&huff, &[&cdic]).unwrap();
        // 1 01 00 1 01: ab c (c ab c ab c) ab c
        assert_eq!(h.unpack(&[0xA5]).unwrap(), b"abccabcabcabc");
    }

    #[test]
    fn kf8_fragments_go_into_their_skeletons() {
        let flow = b"<body></body><p>a</p><body>x</body>";
        let p = parts(flow, &[(0, 13, 1), (21, 14, 0)], &[(6, 8)]);
        assert_eq!(
            String::from_utf8_lossy(&p.text),
            "<body><p>a</p></body><mbp:pagebreak/><body>x</body>"
        );
        let link = kindle_pos("kindle:pos:fid:0000:off:0000000003").unwrap();
        assert_eq!(p.pos(link.0, link.1), Some(9));
        assert_eq!(
            kindle_pos("kindle:pos:fid:000A:off:0000000010"),
            Some((10, 32))
        );
    }

    #[test]
    fn a_contents_place_past_the_fragments_falls_back_to_its_offset() {
        let flow = b"<body></body><p>a</p><body>x</body>";
        let p = parts(flow, &[(0, 13, 1), (21, 14, 0)], &[(6, 8)]);
        assert_eq!(p.place(27, Some((5, 0))), Some(43));
        assert_eq!(p.place(27, Some((0, 3))), Some(9));
    }

    #[test]
    fn a_flow_without_skeletons_is_one_part() {
        let p = parts(b"<p>a</p>", &[], &[]);
        assert_eq!(p.text, b"<p>a</p>");
        assert_eq!(p.at(3), Some(3));
    }
}
