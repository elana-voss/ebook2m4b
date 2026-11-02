//! Whisper hears each spoken chunk back (Whisper large-v3-turbo q8_0, setup's "whisper-model"), in the book's
//! language, told the book's names as its prompt. An NVIDIA card, on either system: whisper.cpp's CUDA server
//! (setup's "whisper-server" folder), started for the book on PORT and killed on drop; another card, on
//! Windows only: whisper.cpp built in on Vulkan (ear.rs).
//! Traps: a server already answering on PORT is refused (one left running would hear for us with another
//! model); building on Windows needs VULKAN_SDK and LIBCLANG_PATH (CLAUDE.md, Rust).
use crate::machine::Card;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

const PORT: u16 = 8931;

pub enum Ear {
    /// held only to stop the server when the Ear is dropped
    Server(#[allow(dead_code)] crate::setup::process::Served),
    #[cfg(windows)]
    Built(crate::speech::ear::WhisperState, &'static str),
}

impl Ear {
    /// Whisper on `card`, from setup's pieces, hearing `language` ("en", "de").
    pub fn new(card: Card, found: &BTreeMap<String, PathBuf>, language: &'static str) -> Result<Ear> {
        let model = found
            .get("whisper-model")
            .context("whisper-model is not set up")?;
        match card {
            Card::Nvidia => {
                let dir = found
                    .get("whisper-server")
                    .context("whisper-server is not set up")?;
                let exe = crate::machine::exe(crate::machine::THIS, "whisper-server");
                start(&dir.join(exe), model, language).map(Ear::Server)
            }
            #[cfg(windows)]
            Card::Other => crate::speech::ear::built(model).map(|s| Ear::Built(s, language)),
            #[cfg(not(windows))]
            Card::Other => anyhow::bail!("on Linux Whisper runs on an NVIDIA card only"),
        }
    }

    /// What Whisper hears in `wav` (Kokoro's samples, 24 kHz), `hint` its prompt.
    pub fn hear(&mut self, wav: &[f32], hint: &str) -> Result<String> {
        match self {
            Ear::Server(_) => {
                let body = crate::setup::http::post_form(
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
                        &crate::speech::wav::wav(wav, crate::speech::wav::RATE),
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
            Ear::Built(state, language) => {
                crate::speech::ear::hear(state, wav, crate::speech::wav::RATE, hint, language)
            }
        }
    }
}

/// whisper.cpp's server on PORT, once it answers (process::serve, at most 60 s).
fn start(exe: &Path, model: &Path, language: &str) -> Result<crate::setup::process::Served> {
    let url = format!("http://127.0.0.1:{PORT}/");
    crate::setup::process::serve(exe, &server_args(model, language), &url, Duration::from_secs(60))
}

fn server_args(model: &Path, language: &str) -> Vec<String> {
    ["-m", &model.to_string_lossy(), "--port", &PORT.to_string(), "-t", "4", "-l", language]
        .map(String::from)
        .to_vec()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_server_hears_the_books_language() {
        let args = super::server_args(std::path::Path::new("m"), "de");
        assert_eq!(args[args.len() - 2..], ["-l", "de"]);
    }
}
