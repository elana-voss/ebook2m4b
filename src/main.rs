//! Ebooks (mobi, azw3, epub, fb2, docx, txt, html) to audiobooks: `ebook2m4b <book or folder> [options]`
//!
//! Kokoro reads with one of its voices (--voice af_bella, the default; English voices start with a_
//! or b_). One output per book, out/<book>.m4b; a finished book is skipped unless --overwrite;
//! --demo speaks about a minute at the start, middle and end; --speed 0.9 is a little slower.
//! The first start fetches what the card needs (setup/); the models run on the graphics card only.
//! The work is run.rs's; here are the options and which of its commands they ask for.
//! Each book's chapters as read go to work/<book>/text.txt, what was changed to changes.txt, its
//! invented words into work/invented-words.json (a series shares them).
//! What the rules cannot settle a language model is asked (judge.rs, llm.rs; local or remote), its
//! answers kept in work/llm-cache.json.
//! Hidden: --text <dir> writes each book's chapters to <dir>/<book>/ instead, without the series'
//! words, the opening line first, and stops (tools/mobi_compare.py; --no-judge: no model asked). Each format has its reader in book/.
//! Each book's chunks are spoken into work/<book>/<voice>/ (speak.rs) and heard back by Whisper
//! (whisper.rs), the doubtful ones in its unsure.txt; then joined into the audiobook (assemble.rs).
//! A book that fails is reported and the others go on; the run ends with "Not done: ...".
//! --card other takes the route of a non-NVIDIA card (DirectML, Vulkan) on any Windows machine (its test); Linux refuses it.
//! --phonemes <file> writes Kokoro's phonemes of each line of <src> (pronounce.rs) and stops.
//! --words prints each book's words eSpeak speaks, English and invented (sounds::found; tools/pron_survey.py).
//! --recheck <unsure.txt> speaks and hears the book's listed passages again (speak::recheck) and prints
//! each as "chunk TAB error listed TAB error now TAB heard TAB text", then "unsure: k of n" (tools/unsure_recheck.py).
//! Before --recheck and before speaking, the book's invented words are settled (sounds::settle, work/sounds.json).
mod book;
mod machine;
mod model;
mod run;
mod setup;
mod speech;
mod text;

use anyhow::{Result, bail};
use clap::Parser;
use std::path::PathBuf;

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

/// af_bella, bm_george: an English voice's name.
fn english_voice(v: &str) -> bool {
    let b = v.as_bytes();
    b.len() > 3
        && matches!(b[0], b'a' | b'b')
        && matches!(b[1], b'f' | b'm')
        && b[2] == b'_'
        && b[3..].iter().all(u8::is_ascii_lowercase)
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
        run::books(&args.src)?
    };
    let tools = setup::ensure(machine::chosen(args.card.as_deref()), &args.voice)?;
    if let Some(out) = &args.phonemes {
        return run::write_phonemes(&tools, &args.src, out);
    }
    let judge = run::judge(&tools)?;
    if let Some(dir) = &args.text {
        return run::write_texts(&books, dir, args.no_judge, &judge);
    }
    let run = run::Run::new(&tools, &judge, &args.voice, args.speed)?;
    if args.words {
        return run.words(&books);
    }
    if let Some(listed) = &args.recheck {
        return run.recheck(&books[0], listed);
    }
    run.make(&books, &args.out, args.demo, args.overwrite)
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
}
