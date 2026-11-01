//! Kokoro v1.0 on ONNX Runtime (the export patched for the GPU, onnx.rs): CUDA on an NVIDIA card, DirectML on
//! the primary adapter on another. ONNX Runtime is loaded from the folder setup names for it ("ort" for CUDA, "ort-dml" for DirectML).
//! Spoken as KPipeline does: the chunk's phonemes (pronounce.rs) in pieces of at most 510, ids padded with 0 at both ends,
//! the voice's style row at phoneme count - 1, the speed asked for; phonemes Kokoro's vocabulary lacks are
//! dropped. A chunk without phonemes gives a tenth of a second of silence, as speak.py.
//! Trap: the CUDA provider's DLLs find cuDNN and cuBLAS through PATH, so new() puts the "ort" folder first on
//! it; it runs before any other thread reads the environment.
use anyhow::{Context, Result, bail};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;

pub const RATE: u32 = 24_000;
pub const MAX_PHONEMES: usize = 510;

pub struct Kokoro {
    session: Session,
    vocab: HashMap<char, i64>,
    style: Vec<f32>,
    speed: f32,
}

impl Kokoro {
    /// `ort` holds onnxruntime.dll and its provider's DLLs; `model` the patched export; `voice` a voice's .bin.
    pub fn new(
        ort: &Path,
        model: &Path,
        tokenizer: &Path,
        voice: &Path,
        cuda: bool,
        speed: f32,
    ) -> Result<Kokoro> {
        // once a run: each book makes its own voice, ONNX Runtime is loaded once
        static LOADED: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
        let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        if !*loaded {
            let path = std::env::var_os("PATH").unwrap_or_default();
            let mut dirs = vec![ort.to_path_buf()];
            dirs.extend(std::env::split_paths(&path));
            // SAFETY: called before any other thread of this program reads the environment
            unsafe { std::env::set_var("PATH", std::env::join_paths(dirs)?) };
            ort::init_from(ort.join("onnxruntime.dll"))
                .context("ONNX Runtime could not be loaded")?
                .commit();
            *loaded = true;
        }
        let ep = if cuda {
            ort::ep::CUDA::default().build().error_on_failure()
        } else {
            ort::ep::DirectML::default()
                .with_device_id(0)
                .build()
                .error_on_failure()
        };
        // a builder's error holds the builder, which is not Send: anyhow takes its message
        let e = |e: ort::Error<_>| anyhow::anyhow!("{e}");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(e)?
            .with_memory_pattern(false)
            .map_err(e)?
            .with_parallel_execution(false)
            .map_err(e)?
            .with_intra_threads(4) // half the physical cores
            .map_err(e)?
            .with_execution_providers([ep])
            .map_err(e)?
            .commit_from_file(model)
            .with_context(|| format!("Kokoro's model {}", model.display()))?;
        let tok: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tokenizer)?)?;
        let vocab = tok["model"]["vocab"]
            .as_object()
            .context("Kokoro's tokenizer has no vocabulary")?
            .iter()
            .filter_map(|(k, v)| Some((k.chars().next()?, v.as_i64()?)))
            .collect();
        let style: Vec<f32> = std::fs::read(voice)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if style.len() < (MAX_PHONEMES) * 256 {
            bail!("{} is no Kokoro voice", voice.display());
        }
        Ok(Kokoro {
            session,
            vocab,
            style,
            speed,
        })
    }

    /// The samples (24 kHz) of `phonemes` spoken.
    pub fn say(&mut self, phonemes: &str) -> Result<Vec<f32>> {
        let mut all = Vec::new();
        for p in pieces(phonemes) {
            all.extend(self.say_one(&p)?);
        }
        Ok(all)
    }

    /// The samples of at most 510 phonemes.
    fn say_one(&mut self, phonemes: &str) -> Result<Vec<f32>> {
        let ps: Vec<char> = phonemes.chars().collect(); // pieces() cut it to MAX_PHONEMES
        if ps.is_empty() {
            return Ok(vec![0.0; RATE as usize / 10]);
        }
        let mut ids = vec![0i64];
        ids.extend(ps.iter().filter_map(|c| self.vocab.get(c).copied()));
        ids.push(0);
        let row = (ps.len() - 1) * 256;
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, ids.len()], ids))?,
            "style" => Tensor::from_array(([1usize, 256], self.style[row..row + 256].to_vec()))?,
            "speed" => Tensor::from_array(([1usize], vec![self.speed]))?,
        ])?;
        let (_, wav) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(wav.to_vec())
    }
}

/// `phonemes` cut into pieces of at most 510, each at the last space before the limit (a word longer than
/// that at the limit), as KPipeline speaks a long passage in several runs.
fn pieces(phonemes: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest: Vec<char> = phonemes.chars().collect();
    while rest.len() > MAX_PHONEMES {
        let cut = rest[..=MAX_PHONEMES]
            .iter()
            .rposition(|c| *c == ' ')
            .filter(|&i| i > 0);
        let (take, skip) = cut.map_or((MAX_PHONEMES, 0), |i| (i, 1));
        out.push(rest[..take].iter().collect());
        rest.drain(..take + skip);
    }
    out.push(rest.into_iter().collect());
    out
}

/// A 32-bit float WAV of `samples` at `rate`, one channel.
pub fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 4) as u32;
    let mut b = Vec::with_capacity(44 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}

#[cfg(test)]
mod tests {
    #[test]
    fn long_phonemes_are_spoken_in_pieces_of_at_most_510() {
        let words: Vec<String> = (0..300).map(|i| format!("w{}", i % 10)).collect();
        let long = words.join(" ");
        let ps = super::pieces(&long);
        assert!(ps.len() > 1);
        assert!(ps.iter().all(|p| p.chars().count() <= 510 && !p.is_empty()));
        assert_eq!(ps.join(" "), long);
        let word = "a".repeat(600);
        assert_eq!(
            super::pieces(&word).iter().map(|p| p.len()).sum::<usize>(),
            600
        );
        assert_eq!(super::pieces("ab"), vec!["ab".to_string()]);
    }

    #[test]
    fn a_wav_holds_its_samples_after_a_44_byte_head() {
        let b = super::wav(&[0.5, -1.0], 24_000);
        assert_eq!(b.len(), 52);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 44);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 24_000);
        assert_eq!(f32::from_le_bytes(b[48..52].try_into().unwrap()), -1.0);
    }
}
