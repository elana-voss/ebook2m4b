//! A book's chunks spoken by Kokoro into work/<book>/<voice>/<n:06>.wav (float WAV, 24 kHz):
//! a kept chunk is not spoken again; work/<book>/<voice>/text-id names the texts, speed and pronunciation
//! (sounds::version) they were spoken from, another id (or --overwrite) clears the folder; one
//! not cleared stops the book. The heat wait before each chunk. --demo speaks about a minute at the
//! start, the middle and the end.
//! Each chunk goes to part.wav first, then is renamed: a stopped run leaves no half chunk.
//! Each new chunk is heard back by Whisper (whisper.rs, told the book's names: `hint` of the chunk's text) and compared by sound
//! (sounds::heard_error); over MAX_ERROR it is spoken again, the best of TRIES kept, and one still over goes into
//! unsure.txt as "<n> TAB <error> TAB <text>". Voice and Whisper are made at the first chunk to speak
//! and leave the GPU when speak_all ends (the language model gets its memory for the next book).
use crate::speech::chunks::Chunk;
use crate::speech::kokoro::{self, Kokoro};
use crate::speech::pronounce::Pronounce;
use crate::speech::whisper::Ear;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;

/// af_bella at speed 1: a novel of 374,000 characters in 7:04 with its pauses
const CHARS_PER_SECOND: f32 = 16.0;
/// a chunk is spoken at most this often
const TRIES: usize = 3;
/// a chunk heard with more of its sounds wrong is spoken again
pub const MAX_ERROR: f32 = 0.1;
/// The chunk spoken until heard right, else the best of TRIES; with its error and what was heard.
pub fn checked(
    text: &str,
    say: &mut dyn FnMut() -> Result<Vec<f32>>,
    hear: &mut dyn FnMut(&[f32]) -> Result<String>,
    p: &Pronounce,
) -> Result<(Vec<f32>, f32, String)> {
    let mut best: Option<(Vec<f32>, f32, String)> = None;
    for _ in 0..TRIES {
        let wav = say()?;
        let heard = hear(&wav)?;
        let err = crate::speech::sounds::heard_error(p, text, &heard)?;
        if best.as_ref().is_none_or(|b| err < b.1) {
            best = Some((wav, err, heard));
        }
        if err <= MAX_ERROR {
            break;
        }
    }
    Ok(best.expect("TRIES is above 0"))
}

/// The passages of an unsure.txt (`listed`, its lines) checked again, each spoken from its text by `say`:
/// (chunk, the error listed, the error now, what was heard, the text). tools/unsure_recheck.py's measure.
pub type Rechecked = (usize, f32, f32, String, String);

pub fn recheck(
    listed: &str,
    say: &mut dyn FnMut(&str) -> Result<Vec<f32>>,
    hear: &mut dyn FnMut(&[f32]) -> Result<String>,
    p: &Pronounce,
) -> Result<Vec<Rechecked>> {
    let mut out = Vec::new();
    for line in listed.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.splitn(3, '\t');
        let (i, old, text) = (|| Some((parts.next()?, parts.next()?, parts.next()?)))()
            .with_context(|| format!("no unsure line: {line:?}"))?;
        let (i, old): (usize, f32) = (i.parse()?, old.parse()?);
        let (_, err, heard) = checked(text, &mut || say(text), hear, p)?;
        out.push((i, old, err, heard, text.to_string()));
    }
    Ok(out)
}

/// unsure.txt's line for chunk `i` heard with `err` of its sounds wrong.
fn unsure_line(i: usize, err: f32, text: &str) -> String {
    format!("{i}\t{err:.2}\t{text}\n")
}

/// About `seconds` of chunks at the start, the middle and the end, never overlapping: a book too short
/// for three parts gets its start and end, or its start.
pub fn demo_picks(cs: &[Chunk], seconds: f32) -> Vec<Vec<usize>> {
    let (mut k, mut total) = (0, 0.0);
    while k < cs.len() && total < seconds {
        total += cs[k].text.chars().count() as f32 / CHARS_PER_SECOND + cs[k].pause;
        k += 1;
    }
    let n = cs.len();
    let mut starts = vec![0];
    if n >= 3 * k {
        starts.push(n / 2 - k / 2);
    }
    if n >= 2 * k {
        starts.push(n - k);
    }
    starts.dedup(); // an empty book
    starts
        .into_iter()
        .map(|s| (s..(s + k).min(n)).collect())
        .collect()
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
            crate::heat::wait();
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
            std::fs::write(&part, kokoro::wav(&wav, kokoro::RATE))?;
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
        let minutes = samples as f64 / f64::from(kokoro::RATE) / 60.0;
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
        demo_picks(cs, 60.0).concat()
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

    fn chunk(text: &str) -> Chunk {
        Chunk {
            chapter: 0,
            text: text.into(),
            pause: 0.0,
            cut: false,
        }
    }

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
    fn listed_passages_are_rechecked_and_reported() {
        let listed = "3\t0.21\tThe cat sat.\n9\t0.50\tdog\n";
        let mut heard = ["the cat sat", "dog", "dog", "dog"].into_iter();
        let report = recheck(
            listed,
            &mut |_| Ok(vec![0.0]),
            &mut |_| Ok(heard.next().unwrap().into()),
            &P,
        )
        .unwrap();
        let want = [(3, 0.21, 0.0, "the cat sat", "The cat sat."), (9, 0.5, 0.0, "dog", "dog")];
        let want: Vec<_> = want
            .into_iter()
            .map(|(i, o, e, h, t)| (i, o, e, h.to_string(), t.to_string()))
            .collect();
        assert_eq!(report, want);
        assert!(recheck("no tabs", &mut |_| Ok(vec![]), &mut |_| Ok(String::new()), &P).is_err());
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

    #[test]
    fn demo_picks_take_a_minute_at_start_middle_and_end() {
        let cs: Vec<Chunk> = (0..100).map(|_| chunk(&"x".repeat(160))).collect(); // 10 s each
        let p = demo_picks(&cs, 60.0);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], (0..6).collect::<Vec<_>>());
        assert_eq!(p[1][0], 47);
        assert_eq!(p[2], (94..100).collect::<Vec<_>>());
        let few = demo_picks(&cs[..4], 60.0);
        assert_eq!(few, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn demo_parts_never_overlap() {
        let cs: Vec<Chunk> = (0..18).map(|_| chunk(&"x".repeat(160))).collect(); // 10 s each, 6 to a part
        let range = |a: usize, b: usize| (a..b).collect::<Vec<_>>();
        assert_eq!(demo_picks(&cs[..10], 60.0), [range(0, 6)]);
        assert_eq!(demo_picks(&cs[..14], 60.0), [range(0, 6), range(8, 14)]);
        assert_eq!(demo_picks(&cs, 60.0), [range(0, 6), range(6, 12), range(12, 18)]);
    }
}
