//! What ebook2m4b and ebook2m4b-whisper (Linux) say to each other through the helper's stdin and stdout:
//! a question is the hint's length (u32) and UTF-8 bytes, then the sample count (u32) and the samples (f32);
//! the answer a mark, the text's length (u32) and UTF-8 bytes; all little-endian. The asker passes over
//! whatever else comes before the mark and fails a question not answered in time. Only std is named here,
//! so that both programs can include it.
#![allow(dead_code)] // each program uses one side: ebook2m4b asks and hears, the helper answers
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::Duration;

/// What starts each answer.
const MARK: &[u8; 4] = b"\0e2m";

fn put(w: &mut dyn Write, bytes: &[u8]) -> Result<()> {
    let n = u32::try_from(bytes.len()).map_err(|_| Error::from(ErrorKind::InvalidInput))?;
    w.write_all(&n.to_le_bytes())?;
    w.write_all(bytes)
}

fn take(r: &mut dyn Read, size: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; size];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// A u32, None at the end of the stream before its first byte.
fn count(r: &mut dyn Read) -> Result<Option<usize>> {
    let mut b = [0u8; 4];
    match r.read(&mut b[..1])? {
        0 => return Ok(None),
        _ => r.read_exact(&mut b[1..])?,
    }
    Ok(Some(u32::from_le_bytes(b) as usize))
}

fn text(r: &mut dyn Read, n: usize) -> Result<String> {
    String::from_utf8(take(r, n)?).map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

pub fn ask(w: &mut dyn Write, hint: &str, wav: &[f32]) -> Result<()> {
    put(w, hint.as_bytes())?;
    let bytes: Vec<u8> = wav.iter().flat_map(|s| s.to_le_bytes()).collect();
    let n = u32::try_from(wav.len()).map_err(|_| Error::from(ErrorKind::InvalidInput))?;
    w.write_all(&n.to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()
}

/// The next question (hint, samples); None when the asker closed the stream.
pub fn question(r: &mut dyn Read) -> Result<Option<(String, Vec<f32>)>> {
    let Some(n) = count(r)? else {
        return Ok(None);
    };
    let hint = text(r, n)?;
    let n = count(r)?.ok_or(Error::from(ErrorKind::UnexpectedEof))?;
    let bytes = take(r, n * 4)?;
    let wav = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    Ok(Some((hint, wav)))
}

pub fn answer(w: &mut dyn Write, text: &str) -> Result<()> {
    w.write_all(MARK)?;
    put(w, text.as_bytes())?;
    w.flush()
}

/// The answer after its mark: what else the helper's libraries printed on stdout before it is passed over.
pub fn heard(r: &mut dyn Read) -> Result<String> {
    let (mut last, mut byte) = ([0u8; 4], [0u8; 1]);
    while last != *MARK {
        r.read_exact(&mut byte)?;
        last.rotate_left(1);
        last[3] = byte[0];
    }
    let n = count(r)?.ok_or(Error::from(ErrorKind::UnexpectedEof))?;
    text(r, n)
}

/// The asker's side with a time limit: a thread writes each question and reads its answer, so that a
/// helper that hangs (reading nothing, answering nothing) fails the question after the wait, and every
/// question after it.
pub struct Asker {
    questions: Sender<(String, Vec<f32>)>,
    answers: Receiver<Result<String>>,
    failed: bool,
}

impl Asker {
    pub fn new(mut w: impl Write + Send + 'static, mut r: impl Read + Send + 'static) -> Asker {
        let (questions, asked) = channel::<(String, Vec<f32>)>();
        let (answered, answers) = channel();
        std::thread::spawn(move || {
            for (hint, wav) in asked {
                let answer = ask(&mut w, &hint, &wav).and_then(|()| heard(&mut r));
                if answered.send(answer).is_err() {
                    break; // the asker is gone
                }
            }
        });
        Asker { questions, answers, failed: false }
    }

    pub fn ask(&mut self, hint: &str, wav: &[f32], wait: Duration) -> Result<String> {
        let gone = || Error::from(ErrorKind::BrokenPipe);
        if self.failed {
            return Err(gone());
        }
        self.questions.send((hint.into(), wav.to_vec())).map_err(|_| gone())?;
        let answer = match self.answers.recv_timeout(wait) {
            Ok(answer) => answer,
            Err(RecvTimeoutError::Timeout) => Err(Error::new(ErrorKind::TimedOut, "no answer in time")),
            Err(RecvTimeoutError::Disconnected) => Err(gone()),
        };
        self.failed = answer.is_err();
        answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_question_and_its_answer_pass_through_the_pipe() {
        let mut pipe = Vec::new();
        ask(&mut pipe, "Alice Liddell", &[0.5, -1.0]).unwrap();
        ask(&mut pipe, "", &[]).unwrap();
        let mut r = &pipe[..];
        assert_eq!(
            question(&mut r).unwrap(),
            Some(("Alice Liddell".into(), vec![0.5, -1.0]))
        );
        assert_eq!(question(&mut r).unwrap(), Some((String::new(), vec![])));
        assert_eq!(question(&mut r).unwrap(), None);
        let mut back = Vec::new();
        answer(&mut back, " I am Alice Liddell.").unwrap();
        assert_eq!(heard(&mut &back[..]).unwrap(), " I am Alice Liddell.");
        assert!(heard(&mut &b""[..]).is_err());
    }

    #[test]
    fn stray_output_before_an_answer_is_passed_over() {
        let mut back = b"ggml_cuda_init: found 1 CUDA devices\n".to_vec();
        answer(&mut back, " Alice").unwrap();
        assert_eq!(heard(&mut &back[..]).unwrap(), " Alice");
    }

    /// A helper that never answers.
    struct Hung;
    impl Read for Hung {
        fn read(&mut self, _: &mut [u8]) -> Result<usize> {
            std::thread::sleep(std::time::Duration::from_secs(3600));
            Ok(0)
        }
    }

    #[test]
    fn a_question_is_answered_or_fails_after_the_wait() {
        let wait = std::time::Duration::from_millis(200);
        let mut back = Vec::new();
        answer(&mut back, " Alice").unwrap();
        let mut asker = Asker::new(Vec::new(), std::io::Cursor::new(back));
        assert_eq!(asker.ask("", &[0.5], wait).unwrap(), " Alice");
        let mut hung = Asker::new(Vec::new(), Hung);
        let e = hung.ask("", &[0.5], wait).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::TimedOut);
        assert!(hung.ask("", &[0.5], wait).is_err(), "a helper past its wait stays failed");
    }
}
