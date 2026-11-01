//! Ebooks (mobi, azw3, epub, fb2, docx, txt, html) to audiobooks: `ebook2m4b <book or folder> [options]`
//!
//! Kokoro reads with one of its voices (--voice af_bella, the default; English voices start with a_
//! or b_). One output per book, out/<book>.m4b; a finished book is skipped unless --overwrite;
//! --demo speaks about a minute at the start, middle and end; --speed 0.9 is a little slower.
//! The first start fetches what the card needs (setup.rs); the models run on the graphics card only.
//! Each book's chapters as read go to work/<book>/text.txt, what was changed to changes.txt, its
//! invented words into work/invented-words.json (a series shares them).
//! What the rules cannot settle a language model is asked (judge.rs, llm.rs; local or remote), its
//! answers kept in work/llm-cache.json.
//! Hidden: --text <dir> writes each book's chapters to <dir>/<book>/ instead, without the series'
//! words, the opening line first, and stops (tools/mobi_compare.py; --no-judge: no model asked). A mobi, azw or azw3 is read by mobi.rs,
//! an EPUB by epub.rs, an FB2 by fb2.rs, a DOCX by docx.rs, a TXT or HTML by markup.rs.
//! Each book's chunks are spoken into work/<book>/<voice>/ (speak.rs) and heard back by Whisper
//! (whisper.rs), the doubtful ones in its unsure.txt; then joined into the audiobook (assemble.rs).
//! A book that fails is reported and the others go on; the run ends with "Not done: ...".
//! --card other takes the route of a non-NVIDIA card (DirectML, Vulkan) on any Windows machine (its test); Linux refuses it.
//! --phonemes <file> writes Kokoro's phonemes of each line of <src> (pronounce.rs) and stops.
//! --words prints each book's words eSpeak speaks, English and invented (sounds::found; tools/pron_survey.py).
//! --recheck <unsure.txt> speaks and hears the book's listed passages again (speak::recheck) and prints
//! each as "chunk TAB error listed TAB error now TAB heard TAB text", then "unsure: k of n" (tools/unsure_recheck.py).
//! Before --recheck and before speaking, the book's invented words are settled (sounds::settle, work/sounds.json).
mod archive;
mod assemble;
mod chunks;
mod clean;
mod docx;
mod downloads;
#[cfg(windows)]
mod ear;
mod epub;
mod extract;
mod fb2;
mod gpu;
mod hash;
mod heat;
mod html;
mod http;
mod judge;
mod kf8;
mod kokoro;
mod llm;
mod markup;
mod misprints;
mod mobi;
mod onnx;
mod os;
mod pronounce;
mod setup;
mod sounds;
mod speak;
mod whisper;
mod words;

use anyhow::{Context, Result, bail};
use clap::Parser;
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const BOOKS: [&str; 10] = [
    "mobi", "azw", "azw3", "epub", "fb2", "docx", "txt", "html", "htm", "xhtml",
];

/// Ebooks to audiobooks.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// a book (mobi, azw3, epub, fb2, docx, txt, html) or a folder of them
    src: PathBuf,
    /// a Kokoro voice
    #[arg(long, default_value = "af_bella")]
    voice: String,
    /// only about a minute at start, middle and end
    #[arg(long)]
    demo: bool,
    /// speaking speed; 0.9 is a little slower
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    /// redo this book even if finished
    #[arg(long)]
    overwrite: bool,
    /// folder for the audiobooks
    #[arg(long, default_value = "out")]
    out: PathBuf,
    /// writes each book's chapters as text into this folder and stops (the comparison tools)
    #[arg(long, hide = true)]
    text: Option<PathBuf>,
    /// with --text: the judge changes nothing (extract::Unchanged), so that routes compare in seconds
    #[arg(long, hide = true, requires = "text")]
    no_judge: bool,
    /// writes the phonemes of each line of <src> (a text file) to this file and stops (the comparison
    /// with KPipeline)
    #[arg(long, hide = true)]
    phonemes: Option<PathBuf>,
    /// the route of a card other than NVIDIA's (DirectML, Vulkan) even where an NVIDIA card is found
    #[arg(long, hide = true, value_parser = ["other"])]
    card: Option<String>,
    /// a book's unsure.txt: its passages checked again and reported, the book left as it is
    #[arg(long, hide = true)]
    recheck: Option<PathBuf>,
    /// prints each book's words eSpeak speaks, English and invented, and stops (tools/pron_survey.py)
    #[arg(long, hide = true)]
    words: bool,
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
    judge: &dyn extract::Judge,
) -> Result<(epub::Book, Vec<extract::Chapter>)> {
    let name = book_name(src);
    let work = Path::new("work").join(&name);
    let ext = src
        .extension()
        .map_or(String::new(), |e| e.to_string_lossy().to_lowercase());
    let book = match ext.as_str() {
        "epub" => epub::read(src)?,
        "fb2" => fb2::read(src)?,
        "docx" => docx::read(src)?,
        "txt" => markup::txt(src)?,
        "html" | "htm" | "xhtml" => markup::html(src)?,
        _ => mobi::read(src)?,
    };
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

/// af_bella, bm_george: an English voice's name.
fn english_voice(v: &str) -> bool {
    let b = v.as_bytes();
    b.len() > 3
        && matches!(b[0], b'a' | b'b')
        && matches!(b[1], b'f' | b'm')
        && b[2] == b'_'
        && b[3..].iter().all(u8::is_ascii_lowercase)
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

/// The book's invented words through the round trip (sounds::settle; Kokoro and Whisper leave the GPU
/// before the language model is asked, and it before they come back), work/sounds.json saved, the won
/// respellings said; returns the book's pronunciation version for its work_id.
fn settle_book(
    cs: &[chunks::Chunk],
    names: &str,
    p: &pronounce::Pronounce,
    voice: &mut dyn FnMut() -> Result<kokoro::Kokoro>,
    ear: &mut dyn FnMut() -> Result<whisper::Ear>,
    judge: &judge::ModelJudge,
) -> Result<serde_json::Value> {
    let file = Path::new(sounds::FILE);
    let mut store = sounds::load(file)?;
    let mut trip = |pairs: &[(String, String)]| -> Result<Vec<String>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let (mut k, mut e) = (voice()?, ear()?);
        pairs
            .iter()
            .map(|(ps, hint)| {
                heat::wait(); // as before each chunk spoken
                e.hear(&k.say(ps)?, hint)
            })
            .collect()
    };
    let mut respell = |items: &[(String, String)]| {
        let got = judge.respell(items);
        extract::Judge::close(judge);
        got
    };
    let words = sounds::settle(cs, names, HINT_CHARS, p, &mut store, &mut trip, &mut respell)?;
    sounds::save(file, &store)?;
    sounds::apply(&words, p, &store)?;
    Ok(sounds::version(&words, &store))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !english_voice(&args.voice) {
        bail!(
            "{} is no English Kokoro voice (af_bella, am_michael, bf_emma, bm_george, ...)",
            args.voice
        );
    }
    let books = if args.phonemes.is_some() {
        Vec::new()
    } else {
        let books = targets(&args.src)?;
        if books.is_empty() || !books.iter().all(|p| p.exists()) {
            bail!("No book found at {}", args.src.display());
        }
        books
    };
    let tools = setup::ensure(gpu::chosen(args.card.as_deref()), &args.voice)?;
    if let Some(out) = &args.phonemes {
        let data = tools
            .found
            .get("espeak-ng-data")
            .context("eSpeak's data is not set up")?;
        let p = pronounce::Pronounce::new(data)?;
        let text =
            std::fs::read_to_string(&args.src).with_context(|| args.src.display().to_string())?;
        let lines: Vec<String> = text.lines().map(|l| p.phonemes(l)).collect::<Result<_>>()?;
        std::fs::write(out, lines.join("\n") + "\n").with_context(|| out.display().to_string())?;
        return Ok(());
    }
    let place = llm::place(&tools.home, &tools.found)?;
    let judge = judge::ModelJudge(RefCell::new(llm::Llm::new(
        place,
        "work/llm-cache.json".into(),
    )?));
    if let Some(dir) = &args.text {
        let judge: &dyn extract::Judge = if args.no_judge {
            &extract::Unchanged
        } else {
            &judge
        };
        for src in &books {
            read_book(src, Some(dir), &mut HashSet::new(), judge)?;
        }
        return Ok(());
    }
    let route = if tools.card == gpu::Card::Nvidia {
        "CUDA"
    } else {
        "DirectML"
    };
    println!(
        "Ready on the {route} route: {} pieces in {}.",
        tools.found.len(),
        tools.home.display()
    );
    let piece = |key: &str| {
        tools
            .found
            .get(key)
            .with_context(|| format!("{key} is not set up"))
    };
    let pronounce = pronounce::Pronounce::new(piece("espeak-ng-data")?)?;
    // Chunks are cut before the book's respellings are known, to find its words, and again after, so
    // none passes Kokoro's limit as it is spoken.
    let measure = |t: &str| pronounce.unsettled_length(t);
    let mut voice = || {
        kokoro::Kokoro::new(
            piece("onnxruntime")?,
            piece(setup::gpu_model(tools.card).0)?,
            piece("kokoro-tokenizer")?,
            piece(&format!("voice-{}", args.voice))?,
            tools.card == gpu::Card::Nvidia,
            args.speed,
        )
    };
    let mut ear = || whisper::Ear::new(tools.card, &tools.found);
    let ffmpeg = piece("ffmpeg")?;
    let mut series = read_series()?;
    if args.words {
        for src in &books {
            let (book, chs) = read_book(src, None, &mut series.clone(), &judge)?;
            let mut all = vec![extract::opening(&book, &book_name(src))];
            all.extend(chs);
            let (english, invented) = sounds::found(&chunks::chunks(&all, &measure), &pronounce)?;
            println!("{}\tenglish\t{}\t{}", book_name(src), english.len(), english.join(" "));
            println!("{}\tinvented\t{}\t{}", book_name(src), invented.len(), invented.join(" "));
        }
        return Ok(());
    }
    if let Some(listed) = &args.recheck {
        let (book, chs) = read_book(&books[0], None, &mut series.clone(), &judge)?;
        let mut all = vec![extract::opening(&book, &book_name(&books[0]))];
        all.extend(chs);
        let names = chunks::book_words(&all, 60);
        settle_book(&chunks::chunks(&all, &measure), &names, &pronounce, &mut voice, &mut ear, &judge)?;
        let listed = std::fs::read_to_string(listed).with_context(|| listed.display().to_string())?;
        let (mut kokoro, mut ear) = (voice()?, ear()?);
        let text = std::cell::RefCell::new(String::new()); // the chunk being spoken, for its hint
        let got = speak::recheck(
            &listed,
            &mut |t| {
                *text.borrow_mut() = t.to_string();
                kokoro.say(&pronounce.phonemes(t)?)
            },
            &mut |wav| ear.hear(wav, &chunks::hint(&names, &text.borrow(), HINT_CHARS)),
            &pronounce,
        )?;
        for (i, old, err, heard, text) in &got {
            let heard = heard.split_whitespace().collect::<Vec<_>>().join(" "); // Whisper's newlines
            println!("{i}\t{old:.2}\t{err:.2}\t{heard}\t{text}");
        }
        let still = got.iter().filter(|g| g.2 > speak::MAX_ERROR).count();
        println!("unsure: {still} of {}", got.len());
        return Ok(());
    }
    let mut failed = Vec::new();
    for src in &books {
        let name = book_name(src);
        let out = assemble::output(&args.out, &name, args.demo);
        if out.exists() && !args.overwrite {
            println!("{name}: done already, skipped ({})", out.display());
            continue;
        }
        let mut one = || -> Result<()> {
            let (book, chs) = read_book(src, None, &mut series, &judge)?;
            write_series(&series)?;
            let mut all = vec![extract::opening(&book, &name)];
            all.extend(chs);
            let work = Path::new("work").join(&name).join(&args.voice);
            let cs = chunks::chunks(&all, &measure);
            let names = chunks::book_words(&all, 60);
            let version = settle_book(&cs, &names, &pronounce, &mut voice, &mut ear, &judge)?;
            let cs = chunks::chunks(&all, &|t| pronounce.length(t));
            speak::speak_book(
                &work,
                &cs,
                args.speed,
                args.demo,
                args.overwrite,
                &mut voice,
                &mut ear,
                &|text| chunks::hint(&names, text, HINT_CHARS),
                &pronounce,
                &version,
            )?;
            let titles: Vec<String> = all.iter().map(|c| c.title.clone()).collect();
            let tags = assemble::tags(&name, &extract::author(&book), &args.voice);
            assemble::audiobook(&work, &cs, &titles, args.demo, &out, &tags, ffmpeg)?;
            println!("  wrote {}", out.display());
            Ok(())
        };
        if let Err(e) = one() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_voices_are_told_apart() {
        assert!(english_voice("af_bella") && english_voice("bm_george"));
        assert!(!english_voice("jf_alpha") && !english_voice("af_") && !english_voice("af_Bella"));
    }

    #[test]
    fn a_comparison_reads_without_the_judge() {
        let a = Args::try_parse_from(["t", "books", "--text", "d", "--no-judge"]).unwrap();
        assert!(a.no_judge);
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
