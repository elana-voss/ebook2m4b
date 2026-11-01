//! One book's run, and the commands beside it: read, settle its sounds, speak, assemble (make); the chapters
//! as text (write_texts), the phonemes of a text file (write_phonemes), a book's words (words), a book's
//! unsure passages again (recheck). What each prints is here. main.rs's head names the output.
use crate::book;
use crate::machine;
use crate::model::judge::{Judge, ModelJudge};
use crate::model::llm;
use crate::setup::Tools;
use crate::speech::kokoro::Kokoro;
use crate::speech::pronounce::Pronounce;
use crate::speech::whisper::Ear;
use crate::speech::{assemble, chunks, sounds, speak};
use crate::text::{extract, misprints};
use anyhow::{Context, Result, bail};
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const BOOKS: [&str; 10] = [
    "mobi", "azw", "azw3", "epub", "fb2", "docx", "txt", "html", "htm", "xhtml",
];

/// The books at `src`, a book or a folder of them; none is an error.
pub fn books(src: &Path) -> Result<Vec<PathBuf>> {
    let books = targets(src)?;
    if books.is_empty() || !books.iter().all(|p| p.exists()) {
        bail!("No book found at {}", src.display());
    }
    Ok(books)
}

fn targets(src: &Path) -> Result<Vec<PathBuf>> {
    let book = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| BOOKS.contains(&e.to_lowercase().as_str()))
    };
    let mut out = if src.is_dir() {
        std::fs::read_dir(src)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| book(p))
            .collect()
    } else {
        vec![src.to_path_buf()]
    };
    out.sort();
    Ok(out)
}

/// The book's name: its file's stem without the characters a folder cannot hold.
fn book_name(src: &Path) -> String {
    let stem = src.file_stem().unwrap_or_default().to_string_lossy();
    stem.chars()
        .filter(|c| !r#"<>:"/\|?*"#.contains(*c))
        .collect::<String>()
        .trim()
        .to_string()
}

/// Each book's chapters, as read, to <dir>/<book>/text.txt and what was changed to changes.txt; the
/// book's invented words added to series. Prints the chapters' count and words.
fn read_book(
    src: &Path,
    dir: Option<&Path>,
    series: &mut HashSet<String>,
    judge: &dyn Judge,
) -> Result<(book::Book, Vec<extract::Chapter>)> {
    let name = book_name(src);
    let work = Path::new("work").join(&name);
    let book = book::read(src)?;
    let mut notes = Vec::new();
    let chs = extract::chapters(&book, Some(judge), series, &mut notes)?;
    judge.close(); // the GPU's memory to the voice; answers stay in the cache
    // the opening (title and authors) first, so that two routes' metadata compare too
    let text: Vec<String> = std::iter::once(extract::opening(&book, &name).title)
        .chain(
            chs.iter()
                .map(|c| format!("# {}\n\n", c.title) + &c.paragraphs.join("\n\n")),
        )
        .collect();
    let out = dir.map_or(work, |d| d.join(&name));
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("text.txt"), text.join("\n\n"))?;
    std::fs::write(out.join("changes.txt"), notes.join("\n") + "\n")?;
    series.extend(misprints::invented(&chs));
    let words: usize = chs
        .iter()
        .flat_map(|c| &c.paragraphs)
        .map(|p| p.split_whitespace().count())
        .sum();
    println!(
        "{name}: {} chapters, {words} words; the text as read: {}",
        chs.len(),
        out.join("text.txt").display()
    );
    Ok((book, chs))
}

/// work/invented-words.json: the invented words of every book read so far, sorted.
const SERIES: &str = "work/invented-words.json";
/// Whisper's prompt of names is at most this long (chunks::hint)
const HINT_CHARS: usize = 200;

fn read_series() -> Result<HashSet<String>> {
    match std::fs::read_to_string(SERIES) {
        Ok(s) => Ok(serde_json::from_str(&s).with_context(|| format!("{SERIES} is no word list"))?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(HashSet::new()),
        Err(e) => Err(e).context(SERIES),
    }
}

fn write_series(series: &HashSet<String>) -> Result<()> {
    let mut words: Vec<&String> = series.iter().collect();
    words.sort();
    let quoted: Vec<String> = words
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<_, _>>()?;
    std::fs::write(SERIES, format!("[{}]", quoted.join(", "))).context(SERIES)
}

/// The phonemes of each line of the text file `src` to `out`.
pub fn write_phonemes(tools: &Tools, src: &Path, out: &Path) -> Result<()> {
    let p = Pronounce::new(tools.espeak()?)?;
    let text = std::fs::read_to_string(src).with_context(|| src.display().to_string())?;
    let lines: Vec<String> = text.lines().map(|l| p.phonemes(l)).collect::<Result<_>>()?;
    std::fs::write(out, lines.join("\n") + "\n").with_context(|| out.display().to_string())
}

/// The language model, its answers kept in work/llm-cache.json.
pub fn judge(tools: &Tools) -> Result<ModelJudge> {
    let place = llm::place(&tools.home, &tools.found)?;
    Ok(ModelJudge(RefCell::new(llm::Llm::new(
        place,
        "work/llm-cache.json".into(),
    )?)))
}

/// Each book's chapters to <dir>/<book>/; `no_judge`: no model asked.
pub fn write_texts(books: &[PathBuf], dir: &Path, no_judge: bool, judge: &ModelJudge) -> Result<()> {
    let judge: &dyn Judge = if no_judge { &extract::Unchanged } else { judge };
    for src in books {
        read_book(src, Some(dir), &mut HashSet::new(), judge)?;
    }
    Ok(())
}

/// What the commands that speak share.
pub struct Run<'a> {
    tools: &'a Tools,
    judge: &'a ModelJudge,
    voice: &'a str,
    speed: f32,
    pronounce: Pronounce,
    ffmpeg: &'a Path,
}

impl<'a> Run<'a> {
    pub fn new(tools: &'a Tools, judge: &'a ModelJudge, voice: &'a str, speed: f32) -> Result<Self> {
        println!(
            "Ready on the {} route: {} pieces in {}.",
            tools.route(),
            tools.found.len(),
            tools.home.display()
        );
        let pronounce = Pronounce::new(tools.espeak()?)?;
        let ffmpeg = tools.ffmpeg()?;
        Ok(Run { tools, judge, voice, speed, pronounce, ffmpeg })
    }

    fn kokoro(&self) -> Result<Kokoro> {
        Kokoro::new(self.tools.kokoro(self.voice)?, self.speed)
    }

    fn ear(&self) -> Result<Ear> {
        Ear::new(self.tools.card, &self.tools.found)
    }

    /// The book as read, its opening first.
    fn chapters(&self, src: &Path, series: &mut HashSet<String>) -> Result<(book::Book, Vec<extract::Chapter>)> {
        let (book, chs) = read_book(src, None, series, self.judge)?;
        let mut all = vec![extract::opening(&book, &book_name(src))];
        all.extend(chs);
        Ok((book, all))
    }

    /// Chunks are cut before the book's respellings are known, to find its words, and again after (one), so
    /// none passes Kokoro's limit as it is spoken.
    fn unsettled(&self, all: &[extract::Chapter]) -> Vec<chunks::Chunk> {
        chunks::chunks(all, &|t| self.pronounce.unsettled_length(t))
    }

    /// The book's invented words through the round trip (sounds::settle; Kokoro and Whisper leave the GPU
    /// before the language model is asked, and it before they come back), work/sounds.json saved, the won
    /// respellings said; returns the book's pronunciation version for its work_id.
    fn settle(&self, cs: &[chunks::Chunk], names: &str) -> Result<serde_json::Value> {
        let file = Path::new(sounds::FILE);
        let mut store = sounds::load(file)?;
        let mut said = false;
        let mut trip = |pairs: &[(String, String)]| -> Result<Vec<String>> {
            if pairs.is_empty() {
                return Ok(Vec::new());
            }
            if !said {
                println!("  {} invented words: the round trip", pairs.len()); // the first trip holds them all
                said = true;
            }
            let (mut k, mut e) = (self.kokoro()?, self.ear()?);
            pairs
                .iter()
                .map(|(ps, hint)| {
                    machine::wait_while_hot(); // as before each chunk spoken
                    e.hear(&k.say(ps)?, hint)
                })
                .collect()
        };
        let mut respell = |items: &[(String, String)]| {
            let got = self.judge.respell(items);
            self.judge.close();
            got
        };
        let p = &self.pronounce;
        let words = sounds::settle(cs, names, HINT_CHARS, p, &mut store, &mut trip, &mut respell)?;
        sounds::save(file, &store)?;
        sounds::apply(&words, p, &store)?;
        Ok(sounds::version(&words, &store))
    }

    /// Prints each book's words eSpeak speaks, English and invented.
    pub fn words(&self, books: &[PathBuf]) -> Result<()> {
        let series = read_series()?;
        for src in books {
            let (_, all) = self.chapters(src, &mut series.clone())?;
            let (english, invented) = sounds::found(&self.unsettled(&all), &self.pronounce)?;
            println!("{}\tenglish\t{}\t{}", book_name(src), english.len(), english.join(" "));
            println!("{}\tinvented\t{}\t{}", book_name(src), invented.len(), invented.join(" "));
        }
        Ok(())
    }

    /// The passages `listed` (a book's unsure.txt) spoken and heard again, and reported.
    pub fn recheck(&self, src: &Path, listed: &Path) -> Result<()> {
        let (_, all) = self.chapters(src, &mut read_series()?)?;
        let names = chunks::book_words(&all, 60);
        self.settle(&self.unsettled(&all), &names)?;
        let listed = std::fs::read_to_string(listed).with_context(|| listed.display().to_string())?;
        let (mut kokoro, mut ear) = (self.kokoro()?, self.ear()?);
        let text = RefCell::new(String::new()); // the chunk being spoken, for its hint
        let got = speak::recheck(
            &listed,
            &mut |t| {
                *text.borrow_mut() = t.to_string();
                kokoro.say(&self.pronounce.phonemes(t)?)
            },
            &mut |wav| ear.hear(wav, &chunks::hint(&names, &text.borrow(), HINT_CHARS)),
            &self.pronounce,
        )?;
        for (i, old, err, heard, text) in &got {
            let heard = heard.split_whitespace().collect::<Vec<_>>().join(" "); // Whisper's newlines
            println!("{i}\t{old:.2}\t{err:.2}\t{heard}\t{text}");
        }
        let still = got.iter().filter(|g| g.2 > speak::MAX_ERROR).count();
        println!("unsure: {still} of {}", got.len());
        Ok(())
    }

    /// One book, read to audiobook at `out`.
    fn one(&self, src: &Path, out: &Path, demo: bool, overwrite: bool, series: &mut HashSet<String>) -> Result<()> {
        let name = book_name(src);
        let (book, all) = self.chapters(src, series)?;
        write_series(series)?;
        let work = Path::new("work").join(&name).join(self.voice);
        let names = chunks::book_words(&all, 60);
        let version = self.settle(&self.unsettled(&all), &names)?;
        let cs = chunks::chunks(&all, &|t| self.pronounce.length(t));
        speak::speak_book(
            &work,
            &cs,
            self.speed,
            demo,
            overwrite,
            &mut || self.kokoro(),
            &mut || self.ear(),
            &|text| chunks::hint(&names, text, HINT_CHARS),
            &self.pronounce,
            &version,
        )?;
        let titles: Vec<String> = all.iter().map(|c| c.title.clone()).collect();
        let tags = assemble::tags(&name, &extract::author(&book), self.voice);
        assemble::audiobook(&work, &cs, &titles, demo, out, &tags, self.ffmpeg)?;
        println!("  wrote {}", out.display());
        Ok(())
    }

    /// Each book not finished yet (or every one, `overwrite`) to its audiobook in `out`; a book that fails is
    /// reported and the others go on.
    pub fn make(&self, books: &[PathBuf], out: &Path, demo: bool, overwrite: bool) -> Result<()> {
        let mut series = read_series()?;
        let mut failed = Vec::new();
        for src in books {
            let name = book_name(src);
            let dest = assemble::output(out, &name, demo);
            if dest.exists() && !overwrite {
                println!("{name}: done already, skipped ({})", dest.display());
                continue;
            }
            if let Err(e) = self.one(src, &dest, demo, overwrite, &mut series) {
                // a book that cannot be read or made; the others go on
                println!("  {e:#}");
                failed.push(name);
            }
        }
        if !failed.is_empty() {
            bail!("Not done: {}", failed.join(", "));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_holds_every_format() {
        let dir = std::env::temp_dir().join("ebook2m4b-targets-test");
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["a.fb2", "b.DOCX", "c.txt", "d.htm", "e.epub", "f.jpg"] {
            std::fs::write(dir.join(f), "").unwrap();
        }
        let names: Vec<_> = targets(&dir)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a.fb2", "b.DOCX", "c.txt", "d.htm", "e.epub"]);
    }
}
