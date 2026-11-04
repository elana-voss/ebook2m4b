//! A book's chunks spoken by Kokoro into work/<book>/<voice>/<n:06>.wav (float WAV, 24 kHz):
//! a kept chunk is not spoken again; work/<book>/<voice>/text-id names the texts, speed and pronunciation
//! (sounds::version) they were spoken from, another id (or --overwrite) clears the folder; one
//! not cleared stops the book. The heat wait before each chunk. --demo speaks about a minute at the
//! start, the middle and the end.
//! Each chunk goes to part.wav first, then is renamed: a stopped run leaves no half chunk.
//! Each new chunk is heard back by Whisper (whisper.rs, told the book's names: `hint` of the chunk's text) and compared by sound
//! (heard::heard_error); over MAX_ERROR it is spoken again, the best of TRIES kept, and one still over goes into
//! unsure.txt as "<n> TAB <error> TAB <text>". Voice and Whisper are made at the first chunk to speak
//! and leave the GPU when speak_all ends (the language model gets its memory for the next book).
use crate::speech::chunks::Chunk;
use crate::speech::kokoro::Kokoro;
use crate::speech::wav;
use crate::speech::pronounce::Pronounce;
use crate::speech::whisper::Ear;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;

/// a chunk is spoken at most this often
const TRIES: usize = 3;
/// a chunk heard with more of its sounds wrong is spoken again
pub const MAX_ERROR: f32 = 0.1;
/// The chunk spoken until heard right, else the best of TRIES; with its error and what was heard.
fn checked(
    text: &str,
    say: &mut dyn FnMut() -> Result<Vec<f32>>,
    hear: &mut dyn FnMut(&[f32]) -> Result<String>,
    p: &Pronounce,
) -> Result<(Vec<f32>, f32, String)> {
    let mut best: Option<(Vec<f32>, f32, String)> = None;
    for _ in 0..TRIES {
        let wav = say()?;
        let heard = hear(&wav)?;
        let err = crate::speech::heard::heard_error(p, text, &heard)?;
        if best.as_ref().is_none_or(|b| err < b.1) {
            best = Some((wav, err, heard));
        }
        if err <= MAX_ERROR {
            break;
        }
    }
    Ok(best.expect("TRIES is above 0"))
}

/// unsure.txt's line for chunk `i` heard with `err` of its sounds wrong.
fn unsure_line(i: usize, err: f32, text: &str) -> String {
    format!("{i}\t{err:.2}\t{text}\n")
}

/// What a chunk folder was spoken from: the texts, the speed and the pronunciation (sounds::version).
pub fn work_id(cs: &[Chunk], speed: f32, version: &Value) -> String {
    let texts: Vec<&str> = cs.iter().map(|c| c.text.as_str()).collect();
    let v = json!([texts, {"speed": speed.to_string(), "pronounce": version}]);
    crate::setup::hash::json_key(&v)
}

/// Speaks the chunks at `picks` not kept in `folder` yet; prints how far it is and, at the end, the
/// seconds taken per minute of audio spoken.
pub fn speak_all(
    cs: &[Chunk],
    picks: &[usize],
    folder: &Path,
    voice: &mut dyn FnMut() -> Result<Kokoro>,
    ear: &mut dyn FnMut() -> Result<Ear>,
    hint: &dyn Fn(&str) -> String,
    pronounce: &Pronounce,
) -> Result<()> {
    let (mut kokoro, mut whisper): (Option<Kokoro>, Option<Ear>) = (None, None);
    let (start, mut samples) = (Instant::now(), 0usize);
    for (done, &i) in picks.iter().enumerate() {
        let f = folder.join(format!("{i:06}.wav"));
        if !f.exists() {
            crate::machine::wait_while_hot();
            if kokoro.is_none() {
                kokoro = Some(voice()?);
            }
            if whisper.is_none() {
                whisper = Some(ear()?);
            }
            let k = kokoro.as_mut().expect("set just above");
            let w = whisper.as_mut().expect("set just above");
            let phonemes = pronounce.phonemes(&cs[i].text)?;
            let (wav, err, _) = checked(
                &cs[i].text,
                &mut || k.say(&phonemes),
                &mut |wav| w.hear(wav, &hint(&cs[i].text)),
                pronounce,
            )?;
            samples += wav.len();
            let part = folder.join("part.wav");
            std::fs::write(&part, wav::wav(&wav, wav::RATE))?;
            std::fs::rename(&part, &f)?;
            if err > MAX_ERROR {
                let mut log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(folder.join("unsure.txt"))?;
                std::io::Write::write_all(&mut log, unsure_line(i, err, &cs[i].text).as_bytes())?;
            }
        }
        print!("\r  {}/{} chunks", done + 1, picks.len());
        // a progress line that does not show is no error
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();
    if samples > 0 {
        let minutes = samples as f64 / f64::from(wav::RATE) / 60.0;
        println!(
            "  {:.1} minutes spoken in {:.0} s: {:.1} s per minute of audio",
            minutes,
            start.elapsed().as_secs_f64(),
            start.elapsed().as_secs_f64() / minutes
        );
    }
    Ok(())
}

/// The book's chunks spoken into work/<name>/<voice>; the kept ones are reused when the text-id matches.
#[allow(clippy::too_many_arguments)] // the book's settings and its three tools, each used once here
pub fn speak_book(
    work: &Path,
    cs: &[Chunk],
    speed: f32,
    demo: bool,
    overwrite: bool,
    voice: &mut dyn FnMut() -> Result<Kokoro>,
    ear: &mut dyn FnMut() -> Result<Ear>,
    hint: &dyn Fn(&str) -> String,
    pronounce: &Pronounce,
    version: &Value,
) -> Result<()> {
    let id = work_id(cs, speed, version);
    let id_file = work.join("text-id");
    if overwrite || std::fs::read_to_string(&id_file).is_ok_and(|old| old != id) {
        // a folder not there yet is nothing to clear; one not cleared would mix old chunks in
        match std::fs::remove_dir_all(work) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).with_context(|| format!("clearing {}", work.display()));
            }
            _ => {}
        }
    }
    std::fs::create_dir_all(work)?;
    std::fs::write(&id_file, &id)?;
    let picks: Vec<usize> = if demo {
        crate::speech::chunks::demo_picks(cs, crate::speech::chunks::DEMO_SECONDS).concat()
    } else {
        (0..cs.len()).collect()
    };
    speak_all(cs, &picks, work, voice, ear, hint, pronounce)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_hint(_: &str) -> String {
        String::new()
    }

    use crate::speech::chunks::plain as chunk;

    use crate::speech::pronounce::tests::P;

    #[test]
    fn a_chunk_heard_wrong_is_spoken_again_and_the_best_kept() {
        let heard = ["the bat ran", "dog", "the cat sat"];
        let (mut said, mut n) = (0.0, 0);
        let (wav, err, got) = checked(
            "The cat sat.",
            &mut || {
                said += 1.0;
                Ok(vec![said])
            },
            &mut |_| {
                n += 1;
                Ok(heard[n - 1].into())
            },
            &P,
        )
        .unwrap();
        assert_eq!((wav, err, n, got.as_str()), (vec![3.0], 0.0, 3, "the cat sat"));
        // never heard right: the best of three
        let n = std::cell::Cell::new(0);
        let (wav, err, _) = checked(
            "The cat sat.",
            &mut || Ok(vec![n.get() as f32]),
            &mut |_| {
                n.set(n.get() + 1);
                Ok(["dog", "the bat ran", "dog"][n.get() - 1].into())
            },
            &P,
        )
        .unwrap();
        assert!(
            err > MAX_ERROR && wav == vec![1.0] && n.get() == 3,
            "{err} {wav:?}"
        );
    }

    #[test]
    fn an_unsure_line_holds_chunk_error_and_text() {
        assert_eq!(unsure_line(12, 0.254, "Zo-ri-ba."), "12\t0.25\tZo-ri-ba.\n");
    }

    #[test]
    fn the_work_id_holds_the_texts_speed_and_pronunciation() {
        // SHA-256 of [["Héllo.","Two"],{"speed":"0.9","pronounce":1}]
        let cs = [chunk("Héllo."), chunk("Two")];
        assert_eq!(
            work_id(&cs, 0.9, &json!(1)),
            "66a4a833b8080294baeb645be9aca8718e13704b22a618efdc997cf8c493cd5a"
        );
        assert_ne!(work_id(&cs, 0.9, &json!(1)), work_id(&cs, 0.9, &json!(2)));
    }

    #[test]
    fn kept_chunks_are_reused_unless_overwrite() {
        let work = std::env::temp_dir().join(format!("ebook2m4b-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work); // none there is fine
        std::fs::create_dir_all(&work).unwrap();
        let cs = [chunk("One."), chunk("Two.")];
        std::fs::write(work.join("text-id"), work_id(&cs, 1.0, &json!(1))).unwrap();
        for i in 0..2 {
            std::fs::write(work.join(format!("{i:06}.wav")), b"kept").unwrap();
        }
        let mut voice = || -> Result<Kokoro> { anyhow::bail!("spoken again") };
        let mut ear = || -> Result<Ear> { anyhow::bail!("heard again") };
        let mut run = |overwrite| {
            speak_book(
                &work, &cs, 1.0, false, overwrite, &mut voice, &mut ear, &no_hint, &P, &json!(1),
            )
        };
        assert!(run(false).is_ok());
        assert!(run(true).is_err() && !work.join("000001.wav").exists());
    }

    #[cfg(windows)]
    #[test]
    fn a_work_folder_not_cleared_stops_the_book() {
        use std::os::windows::fs::OpenOptionsExt;
        let work = std::env::temp_dir().join(format!("ebook2m4b-held-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work); // none there is fine
        std::fs::create_dir_all(&work).unwrap();
        let cs = [chunk("One.")];
        std::fs::write(work.join("000000.wav"), b"old").unwrap();
        // held open by another program, so the folder cannot be cleared
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(work.join("000000.wav"))
            .unwrap();
        let mut voice = || -> Result<Kokoro> { anyhow::bail!("spoken") };
        let mut ear = || -> Result<Ear> { anyhow::bail!("heard") };
        let r = speak_book(&work, &cs, 1.0, false, true, &mut voice, &mut ear, &no_hint, &P, &json!(1));
        drop(held);
        let id = std::fs::read_to_string(work.join("text-id")).ok();
        let _ = std::fs::remove_dir_all(&work); // the test's own folder
        assert!(r.is_err() && id.is_none());
    }
}
