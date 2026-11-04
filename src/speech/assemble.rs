//! A book's kept chunks into one audiobook: each chapter's chunks joined with their pauses straight into an
//! ffmpeg of its own, up to machine::CORES at a time, each making ADTS AAC in work/<book>/<voice>/; their bytes
//! joined, then one ffmpeg copies the sound into the book with chapter marks (an FFMETADATA file; times from
//! the frames counted, so AAC's rounding cannot add up) and the tags; M4B brand for the book, plain MP4 for
//! the demo's m4a. A chapter ends on a whole frame, its last pause cut or grown by under half a frame.
//! Output: out/<book>.m4b, out/<book>.demo.m4a (chapters Start, Middle, End, of a short book Start, End;
//! 1.5 s after each), out/<book>.unsure.txt.
//! LEAD seconds of silence go before the book's title (chunk 0), in the book and in the demo's start.
//! A chunk ending inside a sentence (Chunk::cut) meets the next with its own silence trimmed and 0.4 s.
//! A finished book's work folder is removed; a demo's stays, so the full run reuses its chunks.
use crate::speech::chunks::Chunk;
use crate::speech::wav::{RATE, read_wav};
use anyhow::{Context, Result, bail};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// seconds of silence before the book's title
pub const LEAD: f32 = 1.0;
/// seconds of silence after each piece of the demo
const DEMO_GAP: f32 = 1.5;
/// seconds between two pieces of one sentence, their own silence trimmed
const CUT_GAP: f32 = 0.4;
/// -45 dB: quieter samples at a cut piece's end and the next one's start are its silence
const QUIET: f32 = 0.005_623;
/// samples in an AAC frame
const FRAME: u64 = 1024;
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

/// The WAVs one after another as raw samples into `to`, `lead` seconds of silence first and each WAV
/// followed by its pause; a WAV marked as ending inside a sentence (Chunk::cut) has the silence at its
/// end and at the next one's start (under QUIET) trimmed and CUT_GAP seconds between them instead.
/// The last pause is cut or grown to end on a whole AAC frame (encode's frames then come to the nearest
/// whole frame of the true length), never into the speech. Returns the samples written.
fn join(pieces: &[Piece], lead: f32, to: &mut impl Write) -> Result<u64> {
    fn put(to: &mut impl Write, n: &mut u64, s: &[f32]) -> Result<()> {
        crate::speech::wav::put_samples(to, s)?;
        *n += s.len() as u64;
        Ok(())
    }
    let samples = |s: f32| (s * RATE as f32) as u64;
    let silence = |n: u64| vec![0f32; n as usize];
    let mut n = 0u64;
    let mut pending = samples(lead); // the silence owed: written before the next piece, the last one's measured
    let mut trim_start = false;
    for (p, pause, cut) in pieces {
        put(to, &mut n, &silence(pending))?;
        let w = read_wav(p)?;
        let loud = |x: &f32| x.abs() >= QUIET;
        let from = if trim_start { w.iter().position(loud).unwrap_or(w.len()) } else { 0 };
        let end = if *cut { w.iter().rposition(loud).map_or(from, |i| i + 1).max(from) } else { w.len() };
        put(to, &mut n, &w[from..end])?;
        pending = samples(if *cut { CUT_GAP } else { *pause });
        trim_start = *cut;
    }
    // AAC makes of n samples ceil(n / FRAME) + 1 frames, so one frame fewer is fed
    let whole = (n + pending + FRAME / 2) / FRAME * FRAME;
    let last = silence(whole.saturating_sub(FRAME).saturating_sub(n));
    put(to, &mut n, &last)?;
    Ok(n)
}

/// How many frames an ADTS stream holds, from the lengths in their headers.
fn adts_frames(b: &[u8]) -> Result<u64> {
    let (mut at, mut frames) = (0, 0);
    while at < b.len() {
        let Some(h) = b.get(at..at + 6).filter(|h| h[0] == 0xff && h[1] & 0xf0 == 0xf0) else {
            bail!("no AAC frame at byte {at}");
        };
        let len = usize::from(h[3] & 3) << 11 | usize::from(h[4]) << 3 | usize::from(h[5]) >> 5;
        if len == 0 {
            bail!("an empty AAC frame at byte {at}");
        }
        at += len;
        frames += 1;
    }
    Ok(frames)
}

/// One part's chunks encoded into the ADTS file `dest`, joined straight into ffmpeg; returns its frames.
fn encode(pieces: &[Piece], lead: f32, dest: &Path, ffmpeg: &Path) -> Result<u64> {
    let log = dest.with_extension("log");
    let mut child = Command::new(ffmpeg)
        .args(["-v", "error", "-y", "-f", "f32le", "-ar"])
        .arg(RATE.to_string())
        .args(["-ac", "1", "-i", "-", "-c:a", "aac", "-b:a", "64k", "-f", "adts"])
        .arg(dest)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(&log)?) // a file: a full pipe would stop ffmpeg while it is being fed
        .spawn()
        .with_context(|| ffmpeg.display().to_string())?;
    let stdin = child.stdin.take().context("ffmpeg's stdin")?;
    let mut to = BufWriter::with_capacity(1 << 20, stdin);
    let joined = join(pieces, lead, &mut to).and_then(|n| to.flush().map(|()| n).map_err(Into::into));
    drop(to); // ffmpeg's end of input
    let status = child.wait()?;
    let said = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log); // one not removed stays in work, which goes with the book
    if !status.success() {
        // what it said explains a closed pipe, too
        bail!("ffmpeg failed: {}", said.trim());
    }
    if joined? == 0 {
        return Ok(0); // an empty part: ffmpeg wrote no frame
    }
    adts_frames(&std::fs::read(dest)?)
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

/// The FFMETADATA file: the tags, then each chapter with its (title, frames). A chapter's mark is where its
/// sound starts: after the frames before it and the one frame each encoding starts with (AAC's priming),
/// from samples to milliseconds, not milliseconds summed; the first one starts at 0.
fn metadata(tags: &[(&str, String)], chapters: &[(String, u64)]) -> String {
    let ms = |frames: u64| (frames * FRAME * 1000 + u64::from(RATE) / 2) / u64::from(RATE);
    let total: u64 = chapters.iter().map(|c| c.1).sum();
    let mut lines = vec![";FFMETADATA1".to_string()];
    lines.extend(tags.iter().map(|(k, v)| format!("{k}={}", esc(v))));
    let mark = |before: u64| if before == 0 { 0 } else { (before + 1).min(total) };
    let mut before = 0;
    for (k, (title, frames)) in chapters.iter().enumerate() {
        let after = before + frames;
        let end = if k + 1 == chapters.len() { total } else { mark(after) };
        lines.push("[CHAPTER]\nTIMEBASE=1/1000".into());
        lines.push(format!("START={}\nEND={}\ntitle={}", ms(mark(before)), ms(end), esc(title)));
        before = after;
    }
    lines.join("\n") + "\n"
}

/// The chapters' (title, frames) of the ADTS stream `aac` into `out` with chapter marks and the tags, the
/// sound copied; M4B for a .m4b, plain MP4 else. Written to a temporary name and renamed: a stopped run
/// leaves no half book. Byte-joined ADTS is what keeps the timestamps even (concat of MP4s does not); the
/// time base is needed: the aac reader's own wrecks the chapter track.
fn build(aac: &Path, chapters: &[(String, u64)], tags: &[(&str, String)], out: &Path, ffmpeg: &Path) -> Result<()> {
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    let ext = out.extension().unwrap_or_default().to_string_lossy();
    let dir = out.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let meta = dir.join(format!("{stem}.meta.txt"));
    let tmp = dir.join(format!("{stem}.encoding.{ext}"));
    std::fs::write(&meta, metadata(tags, chapters))?;
    let form: &[&str] = if ext == "m4b" {
        &["-brand", "M4B ", "-f", "ipod"]
    } else {
        &["-f", "mp4"]
    };
    let r = Command::new(ffmpeg)
        .args(["-v", "error", "-y", "-f", "aac", "-i"])
        .arg(aac)
        .arg("-i")
        .arg(&meta)
        .args(["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1", "-time_base"])
        .arg(format!("1/{RATE}"))
        .args(["-c", "copy"])
        .args(form)
        .arg(&tmp)
        .output()
        .with_context(|| ffmpeg.display().to_string())?;
    // the list is of no use either way; one not removed is left in out/
    let _ = std::fs::remove_file(&meta);
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
/// middle and end) encoded and built into `out`, the encoded parts removed; after a full run the unsure list
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
    let part = |n: usize| work.join(format!("part-{n:04}.aac"));
    let whole = work.join("whole.aac");
    let frames = crate::machine::in_order(&groups, |n, (_, pieces, lead)| encode(pieces, *lead, &part(n), ffmpeg));
    let built = frames.into_iter().collect::<Result<Vec<u64>>>().and_then(|frames| {
        let mut to = std::fs::File::create(&whole)?;
        for (n, f) in frames.iter().enumerate() {
            if *f > 0 {
                std::io::copy(&mut std::fs::File::open(part(n))?, &mut to)?;
            }
        }
        let chapters: Vec<(String, u64)> = groups.iter().map(|g| g.0.clone()).zip(frames).collect();
        build(&whole, &chapters, tags, out, ffmpeg)
    });
    for p in (0..groups.len()).map(part).chain([whole]) {
        // a part left behind is encoded again next time
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

    /// The samples of a part as join gives them.
    fn joined(pieces: &[Piece], lead: f32) -> Vec<f32> {
        let mut raw = Vec::new();
        let n = join(pieces, lead, &mut raw).unwrap();
        assert_eq!(n as usize * 4, raw.len());
        raw.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    #[test]
    fn chunks_are_joined_with_the_lead_and_their_pauses() {
        let d = tempdir("join");
        std::fs::write(d.join("a.wav"), crate::speech::wav::wav(&[0.5, -0.5], RATE)).unwrap();
        std::fs::write(d.join("b.wav"), crate::speech::wav::wav(&[0.25], RATE)).unwrap();
        let pieces = [(d.join("a.wav"), 0.5, false), (d.join("b.wav"), 0.0, false)];
        let w = joined(&pieces, 1.0);
        let (lead, half) = (RATE as usize, RATE as usize / 2);
        assert_eq!(w.len(), lead + 2 + half + 1);
        assert!(w[..lead].iter().all(|&x| x == 0.0));
        assert_eq!(&w[lead..lead + 2], &[0.5, -0.5]);
        assert_eq!(w[lead + 2 + half], 0.25);
    }

    #[test]
    fn a_part_ends_a_frame_short_of_the_nearest_whole_frame() {
        let d = tempdir("frame");
        std::fs::write(d.join("a.wav"), crate::speech::wav::wav(&[0.5; 5000], RATE)).unwrap();
        // 5000 + 24000 samples, 28.3 frames: 28 frames to come, so 27 fed; never into the speech
        assert_eq!(joined(&[(d.join("a.wav"), 1.0, false)], 0.0).len(), 27 * 1024);
        assert_eq!(joined(&[(d.join("a.wav"), 0.0, false)], 0.0).len(), 5000);
    }

    #[test]
    fn a_piece_ending_inside_a_sentence_joins_trimmed_with_its_gap() {
        let d = tempdir("cut");
        let (a, b) = ([0.5, 0.3, 0.001, 0.0, 0.0], [0.0, -0.002, 0.0, -0.4, 0.2, 0.0]);
        std::fs::write(d.join("a.wav"), crate::speech::wav::wav(&a, RATE)).unwrap();
        std::fs::write(d.join("b.wav"), crate::speech::wav::wav(&b, RATE)).unwrap();
        let pieces = [(d.join("a.wav"), 0.3, true), (d.join("b.wav"), 0.0, false)];
        let w = joined(&pieces, 0.0);
        let gap = (CUT_GAP * RATE as f32) as usize;
        assert_eq!(w.len(), 2 + gap + 3);
        assert_eq!((&w[..2], &w[2 + gap..]), (&[0.5, 0.3][..], &[-0.4, 0.2, 0.0][..]));
    }

    #[test]
    fn adts_frames_are_counted_from_their_headers() {
        let frame = |len: usize| {
            let mut f = vec![0u8; len];
            f[0] = 0xff;
            f[1] = 0xf1;
            f[3] = (len >> 11) as u8;
            f[4] = (len >> 3) as u8;
            f[5] = (len << 5) as u8;
            f
        };
        assert_eq!(adts_frames(&[frame(7), frame(300)].concat()).unwrap(), 2);
        assert!(adts_frames(&[frame(7), vec![0; 7]].concat()).is_err());
    }

    #[test]
    fn the_marks_fall_on_the_chapters_sound_and_the_seams_are_silent() {
        let d = tempdir("marks");
        let work = d.join("work");
        std::fs::create_dir_all(&work).unwrap();
        // chapters of different lengths, none a whole number of frames: rounding must not add up
        let lens = [24_000 * 3 + 137, 24_000 * 2 + 611, 24_000 * 5 + 77, 24_000 + 901];
        let tone = |n: usize| -> Vec<f32> {
            (0..n).map(|i| 0.3 * (i as f32 * 220.0 * std::f32::consts::TAU / RATE as f32).sin()).collect()
        };
        let cs: Vec<Chunk> = (0..lens.len()).map(|n| chunk(n, 3.0)).collect();
        for (i, n) in lens.iter().enumerate() {
            std::fs::write(work.join(format!("{i:06}.wav")), crate::speech::wav::wav(&tone(*n), RATE)).unwrap();
        }
        let t: Vec<String> = ["a", "b", "c", "d"].map(String::from).into();
        let out = output(&d.join("out"), "b", false);
        audiobook(&work, &cs, &t, false, &out, &tags("b", "", "v"), Path::new("ffmpeg")).unwrap();
        let v = probe(&out);
        let raw = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&out).args(["-f", "f32le", "-"]).output().unwrap().stdout;
        let w: Vec<f32> = raw.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let rate = RATE as f32;
        let true_length = 24_000.0 + lens.iter().map(|n| *n as f32 + 3.0 * rate).sum::<f32>();
        assert!((w.len() as f32 - true_length).abs() <= 1024.0 * lens.len() as f32, "{}", w.len());
        for k in 0..lens.len() {
            let mark = v["chapters"][k]["start_time"].as_str().unwrap().parse::<f32>().unwrap() * rate;
            let from = (mark as usize).saturating_sub(2400);
            let sound = (from + w[from..].iter().position(|x| x.abs() > 0.1).unwrap()) as f32;
            let lead = if k == 0 { rate } else { 0.0 }; // the first mark is at 0, before the lead
            assert!((sound - mark - lead).abs() < 0.05 * rate, "chapter {k}: {} ms", (sound - mark - lead) / rate * 1000.0);
            if k > 0 {
                // the seam, a frame before the mark, lies in the pause: nothing above its own quiet
                let (from, to) = ((mark - 2.0 * rate) as usize, (mark - 2048.0) as usize);
                let peak = w[from..to].iter().fold(0f32, |m, x| m.max(x.abs()));
                assert!(peak < 0.01, "chapter {k}: {peak}");
            }
        }
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
        assert_eq!(ch["end_time"].as_str().unwrap(), "3.157000"); // 1 s lead, 0.1 s, 2 s to a whole frame, the priming
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
        // 6 chunks of 0.01 s, the lead before the start and 1.5 s after it, then the next part's priming
        assert_eq!(v["chapters"][0]["end_time"].as_str().unwrap(), "2.603000");
        assert!(work.join("000000.wav").exists());
    }

    fn tempdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ebook2m4b-assemble-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d); // none there is fine
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
