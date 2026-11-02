//! One book's run: read, tell its language, settle its sounds, speak, assemble (make). What it prints is
//! here. main.rs's head names the output. A German book is read by Kerstin (--voice is for English books),
//! by eSpeak's sounds, with no language model, no mended words and no round trip; a book of neither language
//! is reported not done.
use crate::book;
use crate::machine;
use crate::model::judge::{Judge, ModelJudge};
use crate::model::llm;
use crate::setup::Tools;
use crate::speech::kokoro::Kokoro;
use crate::speech::pronounce::Pronounce;
use crate::speech::whisper::Ear;
use crate::speech::{assemble, chunks, sounds, speak};
use crate::text::language::Language;
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

/// The German voice, its name in work folders and tags.
const KERSTIN: &str = "df_kerstin";

/// The book's language and chapters; a book in neither language is an error. Only an English book asks
/// the judge and adds its invented words to series.
fn read_text(
    book: &book::Book,
    series: &mut HashSet<String>,
    judge: &dyn Judge,
    notes: &mut Vec<String>,
) -> Result<(Language, Vec<extract::Chapter>)> {
    let Some(language) = extract::language(book) else {
        bail!("The book's text is in neither English nor German: not read");
    };
    let chs = if language == Language::German {
        extract::german_chapters(book, notes)?
    } else {
        let chs = extract::chapters(book, Some(judge), series, notes)?;
        series.extend(misprints::invented(&chs, language));
        chs
    };
    Ok((language, chs))
}

/// Each book's chapters, as read, to work/<book>/text.txt and what was changed to changes.txt; the
/// book's invented words added to series. Prints the chapters' count and words.
fn read_book(
    src: &Path,
    series: &mut HashSet<String>,
    judge: &dyn Judge,
) -> Result<(book::Book, Language, Vec<extract::Chapter>)> {
    let name = book_name(src);
    let out = Path::new("work").join(&name);
    let book = book::read(src)?;
    let mut notes = Vec::new();
    let (language, chs) = read_text(&book, series, judge, &mut notes)?;
    judge.close(); // the GPU's memory to the voice; answers stay in the cache
    // the opening (title and authors) first
    let text: Vec<String> = std::iter::once(extract::opening(&book, &name, language).title)
        .chain(
            chs.iter()
                .map(|c| format!("# {}\n\n", c.title) + &c.paragraphs.join("\n\n")),
        )
        .collect();
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("text.txt"), text.join("\n\n"))?;
    std::fs::write(out.join("changes.txt"), notes.join("\n") + "\n")?;
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
    Ok((book, language, chs))
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

/// The language model, its answers kept in work/llm-cache.json.
pub fn judge(tools: &Tools) -> Result<ModelJudge> {
    let place = llm::place(&tools.home, &tools.found)?;
    Ok(ModelJudge(RefCell::new(llm::Llm::new(
        place,
        "work/llm-cache.json".into(),
    )?)))
}

/// What making a book needs.
pub struct Run<'a> {
    tools: &'a Tools,
    judge: &'a ModelJudge,
    voice: &'a str,
    speed: f32,
    pronounce: Pronounce,
    german: Pronounce,
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
        let german = pronounce.german();
        Ok(Run { tools, judge, voice, speed, pronounce, german, ffmpeg })
    }

    fn kokoro(&self, language: Language) -> Result<Kokoro> {
        let files = match language {
            Language::English => self.tools.kokoro(self.voice)?,
            Language::German => self.tools.kerstin()?,
        };
        Kokoro::new(files, self.speed)
    }

    fn ear(&self, language: Language) -> Result<Ear> {
        Ear::new(self.tools.card, &self.tools.found, language.code())
    }

    /// The book as read, its opening first.
    fn chapters(
        &self,
        src: &Path,
        series: &mut HashSet<String>,
    ) -> Result<(book::Book, Language, Vec<extract::Chapter>)> {
        let (book, language, chs) = read_book(src, series, self.judge)?;
        let mut all = vec![extract::opening(&book, &book_name(src), language)];
        all.extend(chs);
        Ok((book, language, all))
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
            let (mut k, mut e) = (self.kokoro(Language::English)?, self.ear(Language::English)?);
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

    /// One book, read to audiobook at `out`.
    fn one(&self, src: &Path, out: &Path, demo: bool, overwrite: bool, series: &mut HashSet<String>) -> Result<()> {
        let name = book_name(src);
        let (book, language, all) = self.chapters(src, series)?;
        write_series(series)?;
        let names = chunks::book_words(&all, 60);
        let (voice, pronounce, version) = match language {
            Language::English => (self.voice, &self.pronounce, self.settle(&self.unsettled(&all), &names)?),
            Language::German => {
                println!("  German: read by Kerstin");
                (KERSTIN, &self.german, serde_json::json!({"german": crate::speech::pronounce::RULES}))
            }
        };
        let work = Path::new("work").join(&name).join(voice);
        let cs = chunks::chunks(&all, &|t| pronounce.length(t));
        speak::speak_book(
            &work,
            &cs,
            self.speed,
            demo,
            overwrite,
            &mut || self.kokoro(language),
            &mut || self.ear(language),
            &|text| chunks::hint(&names, text, HINT_CHARS),
            pronounce,
            &version,
        )?;
        let titles: Vec<String> = all.iter().map(|c| c.title.clone()).collect();
        let tags = assemble::tags(&name, &extract::author(&book), voice);
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

    /// A judge that must not be asked: a German book's text goes without it.
    struct NotAsked;

    impl Judge for NotAsked {
        fn headings(&self, _: &[(String, String)]) -> Result<Vec<bool>> {
            panic!("asked about headings")
        }
        fn read(&self, _: &[(String, String)]) -> Result<Vec<bool>> {
            panic!("asked what to read")
        }
        fn misprints(
            &self,
            _: &[(String, String, Option<String>)],
            _: &[String],
        ) -> Result<std::collections::HashMap<String, String>> {
            panic!("asked about misprints")
        }
    }

    fn read_txt(file: &str, text: &str) -> Result<(Language, Vec<extract::Chapter>)> {
        let dir = std::env::temp_dir().join("ebook2m4b-language-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(file);
        std::fs::write(&path, text).unwrap();
        let mut series = HashSet::new();
        let got = read_text(&book::read(&path).unwrap(), &mut series, &NotAsked, &mut Vec::new());
        assert!(series.is_empty());
        got
    }

    #[test]
    fn a_german_book_is_read_without_the_judge_or_mending() {
        let text = "Er sagte, dass die Frau nicht zu Hause ist und der Hund mit dem Mann in den Wald lief.\n\n".repeat(40);
        let (language, chs) = read_txt("de.txt", &text).unwrap();
        assert_eq!(language, Language::German);
        assert!(chs[0].paragraphs[0].contains("die Frau"));
    }

    #[test]
    fn a_book_in_neither_language_is_refused() {
        let text = "Il était une fois une vieille femme qui vivait dans une petite maison.\n\n".repeat(40);
        let e = read_txt("fr.txt", &text).unwrap_err();
        assert!(e.to_string().contains("neither English nor German"), "{e}");
    }

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
