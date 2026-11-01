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
//! Each format has its reader in book/.
//! Each book's chunks are spoken into work/<book>/<voice>/ (speak.rs) and heard back by Whisper
//! (whisper.rs), the doubtful ones in its unsure.txt; then joined into the audiobook (assemble.rs).
//! A book that fails is reported and the others go on; the run ends with "Not done: ...".
//! --card other takes the route of a non-NVIDIA card (DirectML, Vulkan) on any Windows machine (its test); Linux refuses it.
//! Before speaking, the book's invented words are settled (sounds::settle, work/sounds.json).
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
    /// the route of a card other than NVIDIA's (DirectML, Vulkan) even where an NVIDIA card is found
    #[arg(long, hide = true, value_parser = ["other"])]
    card: Option<String>,
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
    let books = run::books(&args.src)?;
    let tools = setup::ensure(machine::chosen(args.card.as_deref()), &args.voice)?;
    let judge = run::judge(&tools)?;
    let run = run::Run::new(&tools, &judge, &args.voice, args.speed)?;
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
}
