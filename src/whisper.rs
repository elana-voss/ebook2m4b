//! Whisper hears each spoken chunk back (Whisper large-v3-turbo q8_0, setup's "whisper-model"), in English,
//! told the book's names as its prompt. An NVIDIA card, on either system: whisper.cpp's CUDA server
//! (setup's "whisper-server" folder), started for the book on PORT and killed on drop; another card, on
//! Windows only: whisper.cpp built in on Vulkan (ear.rs).
//! Traps: a server already answering on PORT is refused (one left running would hear for us with another
//! model); building on Windows needs VULKAN_SDK and LIBCLANG_PATH (CLAUDE.md, Rust).
use crate::gpu::Card;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

const PORT: u16 = 8931;

pub enum Ear {
    /// held only to stop the server when the Ear is dropped
    Server(#[allow(dead_code)] crate::setup::http::Served),
    #[cfg(windows)]
    Built(crate::ear::WhisperState),
}

impl Ear {
    /// Whisper on `card`, from setup's pieces.
    pub fn new(card: Card, found: &BTreeMap<String, PathBuf>) -> Result<Ear> {
        let model = found
            .get("whisper-model")
            .context("whisper-model is not set up")?;
        match card {
            Card::Nvidia => {
                let dir = found
                    .get("whisper-server")
                    .context("whisper-server is not set up")?;
                let exe = crate::os::exe(crate::os::THIS, "whisper-server");
                start(&dir.join(exe), model).map(Ear::Server)
            }
            #[cfg(windows)]
            Card::Other => crate::ear::built(model).map(Ear::Built),
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
        }
    }
}

/// whisper.cpp's server on PORT, once it answers (http::serve, at most 60 s).
fn start(exe: &Path, model: &Path) -> Result<crate::setup::http::Served> {
    let args = ["-m", &model.to_string_lossy(), "--port", &PORT.to_string()]
        .into_iter()
        .chain(["-t", "4", "-l", "en"])
        .map(String::from)
        .collect::<Vec<_>>();
    let url = format!("http://127.0.0.1:{PORT}/");
    crate::setup::http::serve(exe, &args, &url, Duration::from_secs(60))
}
