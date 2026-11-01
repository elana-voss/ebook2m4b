//! A book's chunks spoken by Kokoro into work/<book>/<voice>/<n:06>.wav (float WAV, 24 kHz), as t2a.py:
//! a kept chunk is not spoken again; work/<book>/<voice>/text-id names the texts, speed and pronunciation
//! (sounds::version) they were spoken from, another id (or --overwrite) clears the folder; one
//! not cleared stops the book. The heat wait before each chunk. --demo speaks about a minute at the
//! start, the middle and the end (t2a.py's demo_picks).
//! Each chunk goes to part.wav first, then is renamed: a stopped run leaves no half chunk.
//! Each new chunk is heard back by Whisper (whisper.rs, told the book's names: `hint` of the chunk's text) and compared by sound
//! (heard_error); over MAX_ERROR it is spoken again, the best of TRIES kept, and one still over goes into
//! unsure.txt as "<n> TAB <error> TAB <text>" (t2a.py's). Voice and Whisper are made at the first chunk to speak
//! and leave the GPU when speak_all ends (the language model gets its memory for the next book).
use crate::chunks::Chunk;
use crate::kokoro::{self, Kokoro};
use crate::pronounce::Pronounce;
use crate::whisper::Ear;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;

/// af_bella at speed 1: a novel of 374,000 characters in 7:04 with its pauses (t2a.py)
const CHARS_PER_SECOND: f32 = 16.0;
/// a chunk is spoken at most this often
const TRIES: usize = 3;
/// a chunk heard with more of its sounds wrong is spoken again
pub const MAX_ERROR: f32 = 0.1;
/// a line of fewer sounds counts as this many: Whisper hears a short line without context, so one or two
/// sounds off there ("Be sad" for "be said") are the ear's, and a short line is unsure only from three
const SHORT_LINE: usize = 20;

/// Edit distance of the sounds of what was heard against the text's, over the text's sounds' length
/// (speak.py's heard_error; numbers as words, as the voice was given them), at least SHORT_LINE. A word
/// Whisper writes in capitals ("MIP") is a word it heard, not letters, so it is lowered before its sounds
/// are asked. What Whisper hears beyond the text's end is free: it goes on after the voice stopped
/// (repeats the line, finishes the sentence), the voice reads its phonemes and adds none.
pub fn heard_error(p: &Pronounce, expected: &str, heard: &str) -> Result<f32> {
    let sounds = |t: &str| -> Result<String> {
        let t = crate::clean::clean(t);
        if t.trim().is_empty() {
            return Ok(String::new()); // nothing heard: no phonemes to ask for
        }
        p.sounds(&t)
    };
    let heard: String = heard
        .split(' ')
        .map(|w| {
            if w.chars().filter(|c| c.is_alphabetic()).count() > 1
                && !w.chars().any(|c| c.is_lowercase())
            {
                w.to_lowercase()
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let heard = names_as_heard(p, expected, &heard)?;
    let (a, b) = (sounds(expected)?, sounds(&heard)?);
    let wrong = crate::misprints::distance_to_prefix(&a, &b);
    Ok(wrong as f32 / a.chars().count().max(SHORT_LINE) as f32)
}

/// `heard` with Whisper's spelling of the text's invented names put right: a word of `expected` outside the
/// dictionary (words::known), of four letters or more, replaces the one to three heard words in a row whose
/// sounds differ from its own in at most half ("hair of Ollie" for "Haravoli", "B-Day Toshik" for
/// "Bidaitoshik"), or the one heard word whose letters differ in at most a third ("Choir" for "Chooir",
/// "Malukta" for "Mel-Ok-Ta"); of those, one spelt near, else the nearest in sounds. A name heard as spelt stays. Whisper writes words it knows for a name it does not, and
/// eSpeak's sounds for its spelling are far from the name's, though what was heard was the name.
fn names_as_heard(p: &Pronounce, expected: &str, heard: &str) -> Result<String> {
    let letters = |w: &str| -> String {
        w.chars()
            .filter(char::is_ascii_alphabetic)
            .flat_map(char::to_lowercase)
            .collect()
    };
    let mut out: Vec<String> = heard.split(' ').map(String::from).collect();
    let said: Vec<String> = out.iter().map(|w| p.sounds(w)).collect::<Result<_>>()?;
    let names: Vec<&str> = expected
        .split(|c: char| c.is_whitespace())
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.chars().filter(char::is_ascii_alphabetic).count() >= 4 && !crate::words::known(w))
        .collect();
    // a name heard as spelt is that name: not another's nearest sound
    let heard: Vec<String> = out.iter().map(|w| letters(w)).collect();
    let mut taken: Vec<bool> = heard.iter().map(|w| names.iter().any(|n| letters(n) == *w)).collect();
    for name in names.into_iter().filter(|n| !heard.contains(&letters(n))) {
        let want = p.sounds(name)?;
        let spelt = letters(name);
        let mut near = None;
        for k in 0..out.len() {
            for len in 1..=3.min(out.len() - k) {
                if taken[k..k + len].iter().any(|&t| t) {
                    break;
                }
                let got: String = said[k..k + len].concat();
                let d = crate::misprints::distance(&want, &got);
                let by_letters = len == 1 && crate::misprints::distance(&spelt, &heard[k]) * 3 <= spelt.len();
                if (by_letters || d * 2 <= want.chars().count())
                    && near.is_none_or(|n| (!by_letters, d, len, k) < n)
                {
                    near = Some((!by_letters, d, len, k));
                }
            }
        }
        if let Some((_, _, len, k)) = near {
            let tail: String = out[k + len - 1].chars().rev().take_while(|c| !c.is_alphanumeric()).collect();
            out[k] = format!("{name}{}", tail.chars().rev().collect::<String>());
            out[k + 1..k + len].iter_mut().for_each(String::clear);
            taken[k..k + len].iter_mut().for_each(|t| *t = true);
        }
    }
    Ok(out.into_iter().filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" "))
}

/// The chunk spoken until heard right, else the best of TRIES; with its error and what was heard
/// (speak.py's checked).
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
        let err = heard_error(p, text, &heard)?;
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

/// t2a.py's work_id with the pronunciation (sounds::version): sha1 of
/// json.dumps([texts, {"speed": speed, "pronounce": version}]).
pub fn work_id(cs: &[Chunk], speed: f32, version: &Value) -> String {
    let texts: Vec<&str> = cs.iter().map(|c| c.text.as_str()).collect();
    // as Python prints the float the user typed (0.9, not 0.8999999761581421)
    let speed: f64 = speed.to_string().parse().unwrap_or(1.0);
    let v = json!([texts, {"speed": speed, "pronounce": version}]);
    crate::hash::sha1(crate::llm::py_json(&v, true))
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

    use crate::pronounce::tests::P;

    #[test]
    fn heard_error_compares_sounds() {
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert_eq!(
            err(
                "The Ivok's daughter, 3 barls.",
                "the ivok's daughter three barls"
            ),
            0.0
        );
        assert!(err("The Ivok's daughter.", "") > 0.5); // nothing heard
        let one = err("The cat sat.", "The bat sat.");
        assert!(one > 0.0 && one < 0.2, "{one}");
        // Whisper writes a name in capitals ("MIP"); the voice would have spelt it, Whisper heard a word
        assert_eq!(err("Chapter thirteen Mip", "Chapter 13 MIP"), 0.0);
    }

    #[test]
    fn a_name_whisper_nearly_spells_is_the_name() {
        let names = |e, h| names_as_heard(&P, e, h).unwrap();
        // Kokoro's sounds of "Choir" are far from those of "Chooir"; the letters are not
        assert_eq!(
            names("“I shall go,” said Chooir. Mel-Ok-Ta left.", "I shall go, said Choir. Malukta left."),
            "I shall go, said Chooir. Mel-Ok-Ta left."
        );
        // one spelt far from it stays; a letter of four may differ ("Carl" for "Zarl")
        assert_eq!(names("said Chooir.", "said table."), "said table.");
        assert_eq!(names("said Zarl.", "said Carl."), "said Zarl.");
    }

    #[test]
    fn a_name_whisper_spells_by_its_sound_is_the_name() {
        // Whisper writes words it knows that sound like the name, one or several
        let names = |e, h| names_as_heard(&P, e, h).unwrap();
        assert_eq!(names("said the Haravoli.", "said the hair of Ollie."), "said the Haravoli.");
        assert!(names("Amblis jumped;", "Am bliss jumped.").ends_with("Amblis jumped."));
        assert_eq!(names("asked Bidaitoshik.", "asked B-Day Toshik."), "asked Bidaitoshik.");
        // a name heard as spelt is that name, not the nearest sound of another one
        assert_eq!(
            names("Outside I encountered Mel-Ok-Ta and Mel-Da-Sa.", "Outside I encountered Malukta and Mel-Da-Sa."),
            "Outside I encountered Mel-Ok-Ta and Mel-Da-Sa."
        );
        assert_eq!(names("“I hunt,” said Chooir. “Of course,” said Selmus.", "I hunt, said Choir. Of course, said Selmus."), "I hunt, said Chooir. Of course, said Selmus.");
        assert_eq!(names("the dreen ran", "the drain ran"), "the dreen ran");
    }

    #[test]
    fn a_tail_heard_beyond_the_text_is_whispers() {
        // Whisper goes on after the voice stopped (repeats, finishes the sentence); the voice adds no word
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert_eq!(
            err(
                "‘Go in peace,’ said the Initiate, stepping aside.",
                "Go in peace, said the initiate, stepping aside Go in peace, said the initiate, stepping aside"
            ),
            0.0
        );
        // a tail the voice dropped still counts
        assert!(err("The cat sat on the mat.", "The cat sat") > 0.2);
    }

    #[test]
    fn a_short_line_is_unsure_only_from_three_sounds_off() {
        // one or two sounds off in a line of a few words is the ear without context ("Be sad" for "be said")
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert!(err("be said.", "Be sad.") <= MAX_ERROR);
        assert!(err("Zarl!", "Mary!") > MAX_ERROR);
    }

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
    fn unsure_lines_are_t2a_pys() {
        assert_eq!(unsure_line(12, 0.254, "Zo-ri-ba."), "12\t0.25\tZo-ri-ba.\n");
    }

    #[test]
    fn the_work_id_is_t2a_pys_with_the_pronunciation() {
        // python: hashlib.sha1(json.dumps([["Héllo.", "Two"], {"speed": 0.9, "pronounce": 1}]).encode()).hexdigest()
        let cs = [chunk("Héllo."), chunk("Two")];
        assert_eq!(
            work_id(&cs, 0.9, &json!(1)),
            "37af121ba6cfde4e293f18408d5bd3d8215e902f"
        );
        assert_ne!(work_id(&cs, 0.9, &json!(1)), work_id(&cs, 0.9, &json!(2)));
    }

    #[test]
    fn kept_chunks_are_reused_unless_overwrite() {
        let work = std::env::temp_dir().join(format!("t2a-resume-{}", std::process::id()));
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
        let work = std::env::temp_dir().join(format!("t2a-held-{}", std::process::id()));
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
