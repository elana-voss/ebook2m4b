//! A scan's misprints mended, and the book's own words told (extract.py's _misprints and invented).
//!
//! A word is a run of a-z with no letter, digit, apostrophe or hyphen on either side (Python's WORD):
//! "I'mtired" and "S-four-otwelve" are too mangled to guess. Words are counted in order of first use,
//! as Python's Counter, since ties and the judge's list follow that order.
use crate::extract::{Chapter, Judge};
use crate::words::{compound, known};
use std::collections::{HashMap, HashSet};

/// The words of a text and their counts, in order of first use.
#[derive(Default)]
struct Count {
    order: Vec<String>,
    n: HashMap<String, usize>,
}

impl Count {
    fn of(chs: &[Chapter]) -> Count {
        let mut c = Count::default();
        for p in chs.iter().flat_map(|ch| &ch.paragraphs) {
            for (a, b) in words(p) {
                let w = &p[a..b];
                match c.n.get_mut(w) {
                    Some(k) => *k += 1,
                    None => {
                        c.order.push(w.to_string());
                        c.n.insert(w.to_string(), 1);
                    }
                }
            }
        }
        c
    }
    fn get(&self, w: &str) -> usize {
        self.n.get(w).copied().unwrap_or(0)
    }
    fn has(&self, w: &str) -> bool {
        self.n.contains_key(w)
    }
}

fn joins(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '\'' | '’' | '-'))
}

/// The byte spans of the words in p.
fn words(p: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut prev: Option<char> = None;
    let mut it = p.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if !c.is_ascii_lowercase() {
            prev = Some(c);
            continue;
        }
        let mut end = i + 1;
        while let Some(&(j, d)) = it.peek() {
            if !d.is_ascii_lowercase() {
                break;
            }
            end = j + 1;
            it.next();
        }
        if !joins(prev) && !joins(p[end..].chars().next()) {
            out.push((i, end));
        }
        prev = p[..end].chars().next_back();
    }
    out
}

/// Levenshtein distance in characters (speak.rs's heard_error too).
pub fn distance(a: &str, b: &str) -> usize {
    *distances(a, b).last().expect("a row has b's length + 1")
}

/// The least distance of `a` to a prefix of `b`: what `b` has beyond `a`'s end costs nothing.
pub fn distance_to_prefix(a: &str, b: &str) -> usize {
    distances(a, b).into_iter().min().expect("a row has b's length + 1")
}

/// Levenshtein's last row: the distance of `a` to each prefix of `b`.
fn distances(a: &str, b: &str) -> Vec<usize> {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for j in 1..=b.len() {
            let here = (row[j] + 1)
                .min(row[j - 1] + 1)
                .min(prev + usize::from(ca != b[j - 1]));
            prev = row[j];
            row[j] = here;
        }
    }
    row
}

/// The words one letter off w: one dropped, swapped, changed or added.
fn near(w: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let c: Vec<char> = w.chars().collect();
    let s = |v: &[char]| v.iter().collect::<String>();
    for i in 0..=c.len() {
        let (a, b) = (&c[..i], &c[i..]);
        if !b.is_empty() {
            out.insert(s(a) + &s(&b[1..]));
        }
        if b.len() > 1 {
            out.insert(s(a) + &b[1].to_string() + &b[0].to_string() + &s(&b[2..]));
        }
        for l in 'a'..='z' {
            if !b.is_empty() {
                out.insert(s(a) + &l.to_string() + &s(&b[1..]));
            }
            out.insert(s(a) + &l.to_string() + &s(b));
        }
    }
    out.remove(w);
    out
}

/// One letter off an English word the book uses five times as often: "thier", "oters"; not "rence"
/// beside a rare "fence".
fn misspelt(w: &str, count: &Count) -> bool {
    let at_least = 5 * count.get(w).max(1);
    near(w)
        .iter()
        .any(|v| count.has(v) && count.get(v) >= at_least && known(v))
}

/// The book's own words: unknown to English, used three times or more ("barl", "vornsman"), and not a
/// recurring misprint.
pub fn invented(chs: &[Chapter]) -> HashSet<String> {
    let count = Count::of(chs);
    count
        .order
        .iter()
        .filter(|w| count.get(w) >= 3 && !known(w) && !misspelt(w, &count))
        .cloned()
        .collect()
}

/// Mends a scan's misprints in words unknown to English and used once, or recurring beside the English
/// word they miss (misspelt): two words run together ("andher", "abasket") are parted where both parts
/// are used elsewhere. The rest, unless a series word (invented() of the books before) or a form of an
/// unknown word also used ("barl", "barls"), unless again misspelt; with a judge: its answer, if made of
/// English or book words and at most two letters off; without: a word of six letters or more becomes the
/// one word one letter off that is English or used three times in the book ("chldren"), where there is
/// exactly one. notes: a line per mended word.
pub fn mend(
    mut chs: Vec<Chapter>,
    judge: Option<&dyn Judge>,
    series: &HashSet<String>,
    notes: &mut Vec<String>,
) -> anyhow::Result<Vec<Chapter>> {
    let count = Count::of(&chs);
    let len = |s: &str| s.chars().count();
    // a short part must be a frequent word: "al read", "a sid" are not
    let common = |part: &str| {
        count.get(part) >= if len(part) <= 3 { 20 } else { 2 }
            && (len(part) > 1 || part == "a" || part == "i")
    };
    let mut fix: Vec<(String, String, &str)> = Vec::new(); // (word, mended, why)
    let mut ask: Vec<(String, Option<String>)> = Vec::new();
    for w in &count.order {
        let n = count.get(w);
        if len(w) < 4 || known(w) || n > 1 && !misspelt(w, &count) {
            continue; // "thier" 28 times beside "their" is mended
        }
        // Both parts used elsewhere in the book: the dictionary alone has "dat", "ings", "snot".
        let mut parts: Vec<(&str, &str)> = (1..w.len())
            .map(|i| (&w[..i], &w[i..]))
            .filter(|(a, b)| common(a) && b.len() > 1 && common(b))
            .collect();
        if parts.is_empty() && w.starts_with('a') && known(&w[1..]) {
            parts = vec![("a", &w[1..])]; // "adiplomat": the article run into a word used once
        }
        if compound(w) {
            // "overtook": not parted; a judge tells it from a misprint ("overhelmed")
            if judge.is_none() {
                continue;
            }
            parts.clear();
        }
        if !parts.is_empty() {
            // "does not", not "doe snot": the first pair whose rarer part is the most used
            let score = |(a, b): &(&str, &str)| count.get(a).min(count.get(b));
            let best = parts.iter().map(score).max().unwrap_or(0);
            let (a, b) = parts.iter().find(|p| score(p) == best).expect("a part");
            fix.push((w.clone(), format!("{a} {b}"), "run together"));
            continue;
        }
        let less = w.strip_suffix('s').unwrap_or("");
        let other = [format!("{w}s"), less.to_string()]
            .iter()
            .any(|v| count.get(v) > 0 && !known(v)); // not "acces"
        if (series.contains(w) || other) && !misspelt(w, &count) {
            continue;
        }
        if len(w) < 6 && judge.is_none() {
            continue;
        }
        let found: Vec<String> = near(w)
            .into_iter()
            .filter(|v| count.has(v) && (known(v) || count.get(v) >= 3))
            .collect();
        let guess = (found.len() == 1).then(|| found[0].clone());
        if judge.is_some() {
            ask.push((w.clone(), guess));
        } else if let Some(g) = guess {
            fix.push((w.clone(), g, "the rules"));
        }
    }
    if let Some(j) = judge.filter(|_| !ask.is_empty()) {
        ask_judge(j, &chs, &count, &ask, &mut fix)?;
    }
    fix.retain(|(w, f, _)| f != w);
    fix.sort();
    notes.extend(fix.iter().map(|(w, f, why)| format!("{w} -> {f} ({why})")));
    if !fix.is_empty() {
        let map: HashMap<&str, &str> = fix
            .iter()
            .map(|(w, f, _)| (w.as_str(), f.as_str()))
            .collect();
        for p in chs.iter_mut().flat_map(|ch| ch.paragraphs.iter_mut()) {
            let mut out = String::with_capacity(p.len());
            let mut at = 0;
            for (a, b) in words(p) {
                if let Some(f) = map.get(&p[a..b]) {
                    out.push_str(&p[at..a]);
                    out.push_str(f);
                    at = b;
                }
            }
            out.push_str(&p[at..]);
            *p = out;
        }
    }
    Ok(chs)
}

/// The 150 characters on each side of the word `w`'s first use in the book.
fn around(chs: &[Chapter], w: &str) -> String {
    let (p, at) = chs
        .iter()
        .flat_map(|ch| &ch.paragraphs)
        .find_map(|p| {
            let at = words(p).into_iter().find(|&(a, b)| &p[a..b] == w)?.0;
            Some((p, at))
        })
        .expect("an asked word is in the book");
    let c: Vec<char> = p.chars().collect();
    let i = p[..at].chars().count();
    let end = (i + w.chars().count() + 150).min(c.len());
    c[i.saturating_sub(150)..end].iter().collect()
}

/// The judge asked about the words the rules cannot settle; its answers checked, else the rules' guess.
fn ask_judge(
    judge: &dyn Judge,
    chs: &[Chapter],
    count: &Count,
    ask: &[(String, Option<String>)],
    fix: &mut Vec<(String, String, &str)>,
) -> anyhow::Result<()> {
    let cases: Vec<(String, String, Option<String>)> = ask
        .iter()
        .map(|(w, g)| (w.clone(), around(chs, w), g.clone()))
        .collect();
    // the book's own words, most used first (ties in order of first use)
    let mut keep: Vec<&String> = count
        .order
        .iter()
        .filter(|w| count.get(w) >= 3 && !known(w))
        .collect();
    keep.sort_by_key(|w| std::cmp::Reverse(count.get(w)));
    let keep: Vec<String> = keep.into_iter().take(80).cloned().collect();
    let answers = judge.misprints(&cases, &keep)?;
    for (w, _) in ask {
        let Some(answer) = answers.get(w) else {
            continue;
        };
        let answer = answer.trim().to_lowercase();
        let mut parts: Vec<String> = answer.split_whitespace().map(str::to_string).collect();
        if parts.len() == 1 && &parts[0] != w {
            let p = parts[0].clone();
            let rest = w
                .strip_prefix(p.as_str())
                .map(|r| (r, true))
                .or_else(|| w.strip_suffix(p.as_str()).map(|r| (r, false)));
            if let Some((rest, first)) = rest
                && (rest == "a" || rest == "i" || rest.chars().count() > 1 && known(rest))
            {
                // "gavea" -> "gave": a word dropped
                parts = if first {
                    vec![p, rest.to_string()]
                } else {
                    vec![rest.to_string(), p]
                };
            }
        }
        let fits = !parts.is_empty()
            && parts.iter().all(|p| {
                !p.is_empty()
                    && p.chars().all(|c| c.is_ascii_lowercase())
                    && (known(p) || count.has(p))
            });
        if fits && distance(w, &parts.concat()) <= 2 {
            fix.push((w.clone(), parts.join(" "), "the language model"));
        }
    }
    // The model silent, or its answer failing the check: the rules' guess.
    let taken: HashSet<String> = fix.iter().map(|(w, _, _)| w.clone()).collect();
    for (w, g) in ask {
        if let Some(g) = g
            && w.chars().count() >= 6
            && !taken.contains(w)
            && answers.get(w).map_or(String::new(), |a| a.to_lowercase()) != *w
        {
            fix.push((w.clone(), g.clone(), "the rules"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(paras: &[&str]) -> Vec<Chapter> {
        vec![Chapter {
            title: "One".into(),
            paragraphs: paras.iter().map(|p| p.to_string()).collect(),
        }]
    }

    #[test]
    fn the_context_centres_on_the_whole_word() {
        let long = "a ".repeat(100);
        let got = around(&ch(&[&format!("Another {long}ther was")]), "ther");
        assert!(got.ends_with("ther was"), "{got}");
    }

    struct Answers(HashMap<String, String>);
    impl Judge for Answers {
        fn headings(&self, items: &[(String, String)]) -> anyhow::Result<Vec<bool>> {
            Ok(vec![true; items.len()])
        }
        fn read(&self, items: &[(String, String)]) -> anyhow::Result<Vec<bool>> {
            Ok(vec![true; items.len()])
        }
        fn misprints(
            &self,
            _: &[(String, String, Option<String>)],
            _: &[String],
        ) -> anyhow::Result<HashMap<String, String>> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn words_stand_alone() {
        let p = "I'mtired the S-three, and \"andher\" Xyz abc_d";
        let w: Vec<&str> = words(p).iter().map(|&(a, b)| &p[a..b]).collect();
        assert_eq!(w, ["the", "and", "andher"]);
    }

    #[test]
    fn rules_part_run_together_words_and_mend_long_misprints() {
        let mut text = vec!["and her children".to_string(); 20];
        text.push("andher chldren barl barl barl overtook".into());
        let refs: Vec<&str> = text.iter().map(String::as_str).collect();
        let mut notes = Vec::new();
        let out = mend(ch(&refs), None, &HashSet::new(), &mut notes).unwrap();
        assert_eq!(
            out[0].paragraphs[20],
            "and her children barl barl barl overtook"
        );
        assert_eq!(
            notes,
            [
                "andher -> and her (run together)",
                "chldren -> children (the rules)"
            ]
        );
        assert!(invented(&out).contains("barl"));
    }

    #[test]
    fn the_judges_answer_is_checked() {
        let answers = [
            ("gavea", "gave"),
            ("overhelmed", "overwhelmed"),
            ("horsmn", "a horse"),
        ];
        let judge = Answers(
            answers
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        );
        let mut notes = Vec::new();
        mend(
            ch(&["gavea overhelmed horsmn"]),
            Some(&judge),
            &HashSet::new(),
            &mut notes,
        )
        .unwrap();
        assert_eq!(
            notes,
            [
                "gavea -> gave a (the language model)",
                "overhelmed -> overwhelmed (the language model)"
            ]
        );
    }
}
