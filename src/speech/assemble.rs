//! A book's kept chunks into one audiobook: each chapter's
//! chunks joined with their pauses into a float WAV in work/<book>/<voice>/, then ffmpeg makes AAC with
//! chapter marks (an FFMETADATA file; times from the WAVs' sample counts) and the tags; M4B brand for the
//! book, plain MP4 for the demo's m4a. Output: out/<book>.m4b, out/<book>.demo.m4a (chapters Start, Middle,
//! End, of a short book Start, End; 1.5 s after each), out/<book>.unsure.txt.
//! LEAD seconds of silence go before the book's title (chunk 0), in the book and in the demo's start.
//! A chunk ending inside a sentence (Chunk::cut) meets the next with its own silence trimmed and 0.4 s.
//! A finished book's work folder is removed; a demo's stays, so the full run reuses its chunks.
use crate::speech::chunks::Chunk;
use crate::speech::wav::{RATE, read_wav};
use anyhow::{Context, Result, bail};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// seconds of silence before the book's title
pub const LEAD: f32 = 1.0;
/// seconds of silence after each piece of the demo
const DEMO_GAP: f32 = 1.5;
/// seconds between two pieces of one sentence, their own silence trimmed
const CUT_GAP: f32 = 0.4;
/// -45 dB: quieter samples at a cut piece's end and the next one's start are its silence
const QUIET: f32 = 0.005_623;
/// a chunk's WAV, the seconds of silence after it, whether it ends inside a sentence
pub type Piece = (PathBuf, f32, bool);

/// out/<book>.m4b, or out/<book>.demo.m4a for --demo.
pub fn output(out: &Path, book: &str, demo: bool) -> PathBuf {
    out.join(if demo {
        format!("{book}.demo.m4a")
    } else {
        format!("{book}.m4b")
    })
}

/// No artist: players show "artist - title"; the author goes in album artist, the voice in composer.
pub fn tags(book: &str, author: &str, voice: &str) -> Vec<(&'static str, String)> {
    vec![
        ("title", book.into()),
        ("album", book.into()),
        ("album_artist", author.into()),
        ("composer", format!("read by Kokoro {voice}")),
        ("genre", "Audiobook".into()),
    ]
}

/// The WAVs one after another into a float WAV at dest, `lead` seconds of silence first and each WAV
/// followed by its pause; a WAV marked as ending inside a sentence (Chunk::cut) has the silence at its
/// end and at the next one's start (under QUIET) trimmed and CUT_GAP seconds between them instead.
/// Returns the samples written.
pub fn join(pieces: &[Piece], lead: f32, dest: &Path) -> Result<u64> {
    let silence = |s: f32| vec![0f32; (s * RATE as f32) as usize];
    let mut f = BufWriter::new(std::fs::File::create(dest)?);
    f.write_all(&crate::speech::wav::wav(&[], RATE))?; // the header; its sizes are written at the end
    let mut n = 0u64;
    let mut put = |f: &mut BufWriter<std::fs::File>, s: &[f32]| -> Result<()> {
        crate::speech::wav::put_samples(f, s)?;
        n += s.len() as u64;
        Ok(())
    };
    put(&mut f, &silence(lead))?;
    let mut trim_start = false;
    for (p, pause, cut) in pieces {
        let w = read_wav(p)?;
        let loud = |x: &f32| x.abs() >= QUIET;
        let from = if trim_start { w.iter().position(loud).unwrap_or(w.len()) } else { 0 };
        let to = if *cut { w.iter().rposition(loud).map_or(from, |i| i + 1).max(from) } else { w.len() };
        put(&mut f, &w[from..to])?;
        put(&mut f, &silence(if *cut { CUT_GAP } else { *pause }))?;
        trim_start = *cut;
    }
    let data = u32::try_from(n * 4).context("a chapter over 4 GB")?;
    f.seek(SeekFrom::Start(4))?;
    f.write_all(&(36 + data).to_le_bytes())?;
    f.seek(SeekFrom::Start(40))?;
    f.write_all(&data.to_le_bytes())?;
    f.flush()?;
    Ok(n)
}

/// ffmpeg's escape of = ; # \ and line ends in FFMETADATA.
fn esc(s: &str) -> String {
    s.chars()
        .flat_map(|c| {
            let e = "=;#\\\n".contains(c).then_some('\\');
            e.into_iter().chain(std::iter::once(c))
        })
        .collect()
}

/// The FFMETADATA file: the tags, then each (title, milliseconds) chapter one after another.
fn metadata(tags: &[(&str, String)], chapters: &[(String, u64)]) -> String {
    let mut lines = vec![";FFMETADATA1".to_string()];
    lines.extend(tags.iter().map(|(k, v)| format!("{k}={}", esc(v))));
    let mut t = 0;
    for (title, ms) in chapters {
        lines.push("[CHAPTER]\nTIMEBASE=1/1000".into());
        lines.push(format!("START={t}\nEND={}\ntitle={}", t + ms, esc(title)));
        t += ms;
    }
    lines.join("\n") + "\n"
}

/// The chapters (title, WAV, its samples) into `out` as AAC with chapter marks and the tags; M4B for a
/// .m4b, plain MP4 else. Written to a temporary name and renamed: a stopped run leaves no half book.
pub fn build(
    chapters: &[(String, PathBuf, u64)],
    tags: &[(&str, String)],
    out: &Path,
    ffmpeg: &Path,
) -> Result<()> {
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    let ext = out.extension().unwrap_or_default().to_string_lossy();
    let dir = out.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let list = dir.join(format!("{stem}.list.txt"));
    let meta = dir.join(format!("{stem}.meta.txt"));
    let tmp = dir.join(format!("{stem}.encoding.{ext}"));
    let quoted: String = chapters
        .iter()
        .map(|(_, p, _)| {
            let p = std::path::absolute(p)?.to_string_lossy().replace('\\', "/");
            Ok(format!("file '{}'\n", p.replace('\'', "'\\''"))) // ffmpeg's quoting of '
        })
        .collect::<Result<_>>()?;
    std::fs::write(&list, quoted)?;
    let ms: Vec<(String, u64)> = chapters
        .iter()
        .map(|(t, _, n)| {
            (
                t.clone(),
                (n * 1000 + u64::from(RATE) / 2) / u64::from(RATE),
            )
        })
        .collect();
    std::fs::write(&meta, metadata(tags, &ms))?;
    let form: &[&str] = if ext == "m4b" {
        &["-brand", "M4B ", "-f", "ipod"]
    } else {
        &["-f", "mp4"]
    };
    let r = Command::new(ffmpeg)
        .args(["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i"])
        .arg(&list)
        .arg("-i")
        .arg(&meta)
        .args([
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
        ])
        .args(form)
        .arg(&tmp)
        .output()
        .with_context(|| ffmpeg.display().to_string())?;
    // the lists are of no use either way; one not removed is left in out/
    let _ = (std::fs::remove_file(&list), std::fs::remove_file(&meta));
    if !r.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&r.stderr).trim()
        );
    }
    std::fs::rename(&tmp, out)?;
    Ok(())
}

/// A part of the audiobook: its title, its chunks with their pauses, the silence before them.
type Part = (String, Vec<Piece>, f32);

/// The book from its chunks kept in `work` (all spoken): the chapters (or, for the demo, its start,
/// middle and end) joined, built into `out`, the joined WAVs removed; after a full run the unsure list
/// goes beside the book and `work` is removed.
pub fn audiobook(
    work: &Path,
    cs: &[Chunk],
    titles: &[String],
    demo: bool,
    out: &Path,
    tags: &[(&str, String)],
    ffmpeg: &Path,
) -> Result<()> {
    let file = |i: usize| work.join(format!("{i:06}.wav"));
    let groups: Vec<Part> = if demo {
        let picks = crate::speech::chunks::demo_picks(cs, crate::speech::chunks::DEMO_SECONDS);
        let names: &[&str] = if picks.len() == 2 {
            &["Start", "End"]
        } else {
            &["Start", "Middle", "End"]
        };
        picks
            .into_iter()
            .zip(names.iter().copied())
            .map(|(g, name)| {
                let lead = if g.first() == Some(&0) { LEAD } else { 0.0 };
                let mut p: Vec<Piece> =
                    g.iter().map(|&i| (file(i), cs[i].pause, cs[i].cut)).collect();
                if let Some(last) = p.last_mut() {
                    last.1 += DEMO_GAP;
                    last.2 = false; // the demo's gap, untrimmed
                }
                (name.to_string(), p, lead)
            })
            .collect()
    } else {
        titles
            .iter()
            .enumerate()
            .map(|(n, t)| {
                let p = (0..cs.len())
                    .filter(|&i| cs[i].chapter == n)
                    .map(|i| (file(i), cs[i].pause, cs[i].cut))
                    .collect();
                (t.clone(), p, if n == 0 { LEAD } else { 0.0 })
            })
            .collect()
    };
    let mut parts = Vec::new();
    for (n, (title, pieces, lead)) in groups.iter().enumerate() {
        let p = work.join(format!("part-{n:04}.wav"));
        let samples = join(pieces, *lead, &p)?;
        parts.push((title.clone(), p, samples));
    }
    let built = build(&parts, tags, out, ffmpeg);
    for (_, p, _) in &parts {
        // a part left behind is joined again next time
        let _ = std::fs::remove_file(p);
    }
    built?;
    if !demo {
        let name = out.file_stem().unwrap_or_default().to_string_lossy();
        let unsure = out.with_file_name(format!("{name}.unsure.txt"));
        match std::fs::rename(work.join("unsure.txt"), &unsure) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // an earlier run's list would no longer be true; none there is fine
                let _ = std::fs::remove_file(&unsure);
            }
            Err(e) => return Err(e).context("unsure.txt"),
        }
        std::fs::remove_dir_all(work)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(chapter: usize, pause: f32) -> Chunk {
        Chunk {
            chapter,
            text: "x".repeat(160), // 10 s for demo_picks
            pause,
            cut: false,
        }
    }

    fn probe(out: &Path) -> serde_json::Value {
        let r = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_chapters",
                "-show_format",
                "-of",
                "json",
            ])
            .arg(out)
            .output()
            .unwrap();
        serde_json::from_slice(&r.stdout).unwrap()
    }

    fn titles(v: &serde_json::Value) -> Vec<String> {
        v["chapters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["tags"]["title"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn outputs_are_named_like_the_book() {
        let out = Path::new("out");
        assert_eq!(output(out, "My Book", false), out.join("My Book.m4b"));
        assert_eq!(output(out, "My Book", true), out.join("My Book.demo.m4a"));
        assert_eq!(tags("B", "A", "af_bella")[3].1, "read by Kokoro af_bella");
    }

    #[test]
    fn chunks_are_joined_with_the_lead_and_their_pauses() {
        let d = tempdir("join");
        std::fs::write(d.join("a.wav"), crate::speech::wav::wav(&[0.5, -0.5], RATE)).unwrap();
        std::fs::write(d.join("b.wav"), crate::speech::wav::wav(&[0.25], RATE)).unwrap();
        let pieces = [(d.join("a.wav"), 0.5, false), (d.join("b.wav"), 0.0, false)];
        let n = join(&pieces, 1.0, &d.join("j.wav")).unwrap();
        let w = read_wav(&d.join("j.wav")).unwrap();
        let (lead, half) = (RATE as usize, RATE as usize / 2);
        assert_eq!(
            (n as usize, w.len()),
            (lead + 2 + half + 1, lead + 2 + half + 1)
        );
        assert!(w[..lead].iter().all(|&x| x == 0.0));
        assert_eq!(&w[lead..lead + 2], &[0.5, -0.5]);
        assert_eq!(w[lead + 2 + half], 0.25);
    }

    #[test]
    fn a_piece_ending_inside_a_sentence_joins_trimmed_with_its_gap() {
        let d = tempdir("cut");
        let (a, b) = ([0.5, 0.3, 0.001, 0.0, 0.0], [0.0, -0.002, 0.0, -0.4, 0.2, 0.0]);
        std::fs::write(d.join("a.wav"), crate::speech::wav::wav(&a, RATE)).unwrap();
        std::fs::write(d.join("b.wav"), crate::speech::wav::wav(&b, RATE)).unwrap();
        let pieces = [(d.join("a.wav"), 0.3, true), (d.join("b.wav"), 0.0, false)];
        join(&pieces, 0.0, &d.join("j.wav")).unwrap();
        let w = read_wav(&d.join("j.wav")).unwrap();
        let gap = (CUT_GAP * RATE as f32) as usize;
        assert_eq!(w.len(), 2 + gap + 3);
        assert_eq!((&w[..2], &w[2 + gap..]), (&[0.5, 0.3][..], &[-0.4, 0.2, 0.0][..]));
    }

    #[test]
    fn the_book_has_its_chapters_tags_and_brand_and_its_work_is_cleared() {
        let d = tempdir("book");
        let work = d.join("work");
        std::fs::create_dir_all(&work).unwrap();
        let cs = [chunk(0, 2.0), chunk(1, 0.3), chunk(1, 2.0), chunk(2, 2.0)];
        for i in 0..cs.len() {
            std::fs::write(
                work.join(format!("{i:06}.wav")),
                crate::speech::wav::wav(&[0.1; 2400], RATE),
            )
            .unwrap();
        }
        std::fs::write(work.join("unsure.txt"), "1\t0.20\tx\n").unwrap();
        let t: Vec<String> = ["Alice, by L.", "One", "Two"].map(String::from).into();
        let out = output(&d.join("out"), "Alice's book", false);
        let tg = tags("Alice's book", "L. C.", "af_bella");
        audiobook(&work, &cs, &t, false, &out, &tg, Path::new("ffmpeg")).unwrap();
        let v = probe(&out);
        assert_eq!(titles(&v), t);
        let ch = &v["chapters"][0];
        assert_eq!(ch["end_time"].as_str().unwrap(), "3.100000"); // 1 s lead, 0.1 s, 2 s
        let f = &v["format"]["tags"];
        assert_eq!(
            (f["title"].as_str(), f["album_artist"].as_str()),
            (Some("Alice's book"), Some("L. C."))
        );
        assert_eq!(f["major_brand"].as_str().unwrap().trim(), "M4B");
        assert!(d.join("out/Alice's book.unsure.txt").exists() && !work.exists());
    }

    #[test]
    fn the_demo_has_start_middle_and_end_and_keeps_its_chunks() {
        let d = tempdir("demo");
        let work = d.join("work");
        std::fs::create_dir_all(&work).unwrap();
        let cs: Vec<Chunk> = (0..20).map(|_| chunk(0, 0.0)).collect();
        for i in 0..cs.len() {
            std::fs::write(
                work.join(format!("{i:06}.wav")),
                crate::speech::wav::wav(&[0.1; 240], RATE),
            )
            .unwrap();
        }
        let out = output(&d.join("out"), "B", true);
        let tg = tags("B", "", "af_bella");
        audiobook(
            &work,
            &cs,
            &["B.".into()],
            true,
            &out,
            &tg,
            Path::new("ffmpeg"),
        )
        .unwrap();
        let v = probe(&out);
        assert_eq!(titles(&v), ["Start", "Middle", "End"]);
        // 6 chunks of 0.01 s, the lead before the start and 1.5 s after it
        assert_eq!(v["chapters"][0]["end_time"].as_str().unwrap(), "2.560000");
        assert!(work.join("000000.wav").exists());
    }

    fn tempdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ebook2m4b-assemble-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d); // none there is fine
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
