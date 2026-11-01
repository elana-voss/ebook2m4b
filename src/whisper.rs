//! Whisper hears each spoken chunk back (Whisper large-v3-turbo q8_0, setup's "whisper-model"), in English,
//! told the book's names as its prompt, as t2a.py's Ear. An NVIDIA card: whisper.cpp's CUDA server (setup's
//! "whisper-server" folder), started for the book on PORT and killed on drop. Another card: whisper.cpp built
//! in on Vulkan (whisper-rs, the one module naming it), on the first discrete card ggml lists (the integrated
//! AMD one may come first; without a discrete one, the integrated), the 24 kHz voice resampled to Whisper's 16 kHz.
//! Traps: a server already answering on PORT is refused (one left running would hear for us with another
//! model); building needs VULKAN_SDK and LIBCLANG_PATH (CLAUDE.md, Rust).
use crate::gpu::Card;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::ffi::c_int;
use std::path::{Path, PathBuf};
use std::time::Duration;
use whisper_rs::whisper_rs_sys as ggml;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

const PORT: u16 = 8931;

pub enum Ear {
    /// held only to stop the server when the Ear is dropped
    Server(#[allow(dead_code)] crate::http::Served),
    Built(WhisperState),
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
                start(&dir.join("whisper-server.exe"), model).map(Ear::Server)
            }
            Card::Other => built(model).map(Ear::Built),
        }
    }

    /// What Whisper hears in `wav` (Kokoro's samples, 24 kHz), `hint` its prompt.
    pub fn hear(&mut self, wav: &[f32], hint: &str) -> Result<String> {
        match self {
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
            Ear::Built(state) => {
                let wav = resample(wav, crate::kokoro::RATE, 16_000);
                let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                p.set_language(Some("en"));
                p.set_temperature_inc(0.0);
                p.set_n_threads(4);
                p.set_initial_prompt(hint);
                p.set_no_timestamps(true);
                p.set_print_special(false);
                p.set_print_progress(false);
                p.set_print_realtime(false);
                p.set_print_timestamps(false);
                state.full(p, &wav)?;
                let mut text = String::new();
                for seg in state.as_iter() {
                    text += &seg.to_str_lossy()?;
                }
                Ok(text)
            }
        }
    }
}

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

/// Of Vulkan's cards (each discrete or not), the first discrete one, else the first.
fn pick(discrete: &[bool]) -> Option<usize> {
    let first = discrete.iter().position(|&d| d);
    first.or((!discrete.is_empty()).then_some(0))
}

/// whisper.cpp built in, on the first discrete card Vulkan lists, else its integrated one.
fn built(model: &Path) -> Result<WhisperState> {
    // SAFETY: ggml's registry of its built-in backends; each device lives as long as the program.
    let kinds: Vec<_> = unsafe {
        (0..ggml::ggml_backend_dev_count())
            .map(|i| ggml::ggml_backend_dev_type(ggml::ggml_backend_dev_get(i)))
            .filter(|&t| {
                t == ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                    || t == ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            })
            .collect()
    };
    let discrete: Vec<bool> = kinds
        .iter()
        .map(|&t| t == ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU)
        .collect();
    let Some(first) = pick(&discrete) else {
        bail!("Vulkan sees no graphics card; Whisper runs on the GPU only");
    };
    let mut params = WhisperContextParameters::default();
    params.use_gpu(true).gpu_device(c_int::try_from(first)?);
    let ctx = WhisperContext::new_with_params(model, params)?;
    Ok(ctx.create_state()?)
}

/// `samples` at `from` a second made `to` a second: each new sample the old ones near it weighed by a sinc
/// in a Hann window of TAPS on each side, cut off below the lower rate's half.
fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    const TAPS: f64 = 32.0;
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let step = f64::from(from) / f64::from(to);
    let cut = (f64::from(to) / f64::from(from)).min(1.0);
    let reach = TAPS / cut;
    let count = (samples.len() as f64 / step).floor() as usize;
    (0..count)
        .map(|n| {
            let at = n as f64 * step;
            let first = (at - reach).ceil().max(0.0) as usize;
            let last = ((at + reach).floor() as usize).min(samples.len() - 1);
            let mut sum = 0.0;
            for (k, sample) in samples.iter().enumerate().take(last + 1).skip(first) {
                let x = (k as f64 - at) * cut;
                let sinc = if x == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * x / TAPS).cos();
                sum += f64::from(*sample) * sinc * window * cut;
            }
            sum as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn whisper_takes_the_discrete_card_else_the_integrated() {
        assert_eq!(super::pick(&[false, true]), Some(1));
        assert_eq!(super::pick(&[false]), Some(0));
        assert_eq!(super::pick(&[]), None);
    }
}
