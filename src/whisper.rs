//! Whisper hears each spoken chunk back (Whisper large-v3-turbo q8_0, setup's "whisper-model"), in English,
//! told the book's names as its prompt. Windows, an NVIDIA card: whisper.cpp's CUDA server
//! (setup's "whisper-server" folder), started for the book on PORT and killed on drop; another card:
//! whisper.cpp built in on Vulkan (ear.rs). Linux (NVIDIA only): the program ebook2m4b-whisper beside this one
//! (src/bin, whisper.cpp built in with CUDA), asked through its stdin and stdout (frames.rs, an answer waited
//! for at most 180 s) and killed on drop, so that this program needs no CUDA library to start, before
//! setup has fetched them.
//! Traps: a server already answering on PORT is refused (one left running would hear for us with another
//! model); building needs VULKAN_SDK and LIBCLANG_PATH on Windows, CUDA's compiler on Linux (CLAUDE.md, Rust).
use crate::gpu::Card;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;
#[cfg(windows)]
use std::path::Path;

#[cfg(windows)]
const PORT: u16 = 8931;

pub enum Ear {
    /// held only to stop the server when the Ear is dropped
    #[cfg(windows)]
    Server(#[allow(dead_code)] crate::http::Served),
    #[cfg(windows)]
    Built(crate::ear::WhisperState),
    #[cfg(unix)]
    Helper(Helper),
}

impl Ear {
    /// Whisper on `card`, from setup's pieces.
    pub fn new(card: Card, found: &BTreeMap<String, PathBuf>) -> Result<Ear> {
        let model = found
            .get("whisper-model")
            .context("whisper-model is not set up")?;
        #[cfg(windows)]
        return match card {
            Card::Nvidia => {
                let dir = found
                    .get("whisper-server")
                    .context("whisper-server is not set up")?;
                start(&dir.join("whisper-server.exe"), model).map(Ear::Server)
            }
            Card::Other => crate::ear::built(model).map(Ear::Built),
        };
        #[cfg(unix)]
        {
            anyhow::ensure!(card == Card::Nvidia, "on Linux Whisper runs on an NVIDIA card only");
            Helper::start(model).map(Ear::Helper)
        }
    }

    /// What Whisper hears in `wav` (Kokoro's samples, 24 kHz), `hint` its prompt.
    pub fn hear(&mut self, wav: &[f32], hint: &str) -> Result<String> {
        match self {
            #[cfg(windows)]
            Ear::Server(_) => {
                let body = crate::http::post_form(
                    &format!("http://127.0.0.1:{PORT}/inference"),
                    &[
                        ("response_format", "json"),
                        ("temperature", "0"),
                        ("temperature_inc", "0"), // no retry at a higher temperature: it repeats itself
                        ("prompt", hint),
                    ],
                    (
                        "file",
                        "c.wav",
                        &crate::kokoro::wav(wav, crate::kokoro::RATE),
                    ),
                    Duration::from_secs(120),
                )?;
                let v: serde_json::Value =
                    serde_json::from_str(&body).context("Whisper's answer is no JSON")?;
                Ok(v["text"]
                    .as_str()
                    .context("Whisper's answer has no text")?
                    .to_string())
            }
            #[cfg(windows)]
            Ear::Built(state) => crate::ear::hear(state, wav, crate::kokoro::RATE, hint),
            #[cfg(unix)]
            Ear::Helper(h) => h.hear(wav, hint),
        }
    }
}

#[cfg(windows)]
/// whisper.cpp's server on PORT, once it answers (http::serve, at most 60 s).
fn start(exe: &Path, model: &Path) -> Result<crate::http::Served> {
    let args = ["-m", &model.to_string_lossy(), "--port", &PORT.to_string()]
        .into_iter()
        .chain(["-t", "4", "-l", "en"])
        .map(String::from)
        .collect::<Vec<_>>();
    let url = format!("http://127.0.0.1:{PORT}/");
    crate::http::serve(exe, &args, &url, Duration::from_secs(60))
}

/// Linux: the program ebook2m4b-whisper, loading the model once and answering each question in turn.
#[cfg(unix)]
pub struct Helper(std::process::Child, crate::frames::Asker);

#[cfg(unix)]
impl Helper {
    fn start(model: &std::path::Path) -> Result<Helper> {
        use std::process::{Command, Stdio};
        let exe = std::env::current_exe()?.with_file_name("ebook2m4b-whisper");
        let mut child = Command::new(&exe)
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("{} could not be started", exe.display()))?;
        let pipes = child.stdin.take().zip(child.stdout.take());
        let (input, output) = pipes.context("ebook2m4b-whisper has no pipes")?;
        Ok(Helper(child, crate::frames::Asker::new(input, output)))
    }

    fn hear(&mut self, wav: &[f32], hint: &str) -> Result<String> {
        // the Windows server's 120 s for an answer and 60 s to start: the first answer waits for the model
        let wait = Duration::from_secs(180);
        let gone = "ebook2m4b-whisper stopped or hangs (its messages are above)";
        self.1.ask(hint, wav, wait).context(gone)
    }
}

#[cfg(unix)]
impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.0.kill(); // one already stopped is fine
        let _ = self.0.wait(); // nothing to tell: it was stopped
    }
}
