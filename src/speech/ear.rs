//! whisper.cpp built in (whisper-rs, the one module naming it), on Windows for a card other than NVIDIA's
//! (Vulkan, whisper.rs). It takes the first discrete card ggml lists (the integrated AMD one may come first; without a discrete one,
//! the integrated) and hears Kokoro's 24 kHz voice resampled to Whisper's 16 kHz.
use anyhow::{Result, bail};
use std::ffi::c_int;
use std::path::Path;
use whisper_rs::whisper_rs_sys as ggml;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub use whisper_rs::WhisperState;

/// What Whisper hears in `wav` (samples at `rate` a second), `hint` its prompt, in `language`.
pub fn hear(state: &mut WhisperState, wav: &[f32], rate: u32, hint: &str, language: &str) -> Result<String> {
    let wav = resample(wav, rate, 16_000);
    let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    p.set_language(Some(language));
    p.set_temperature_inc(0.0);
    p.set_n_threads(crate::machine::CORES as _);
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

/// Of ggml's cards (each discrete or not), the first discrete one, else the first.
fn pick(discrete: &[bool]) -> Option<usize> {
    let first = discrete.iter().position(|&d| d);
    first.or((!discrete.is_empty()).then_some(0))
}

/// whisper.cpp built in, on the first discrete card ggml lists, else its integrated one.
pub fn built(model: &Path) -> Result<WhisperState> {
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
        bail!("ggml sees no graphics card; Whisper runs on the GPU only");
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
