//! Kokoro v1.0 on ONNX Runtime (the export patched for the GPU, onnx.rs): CUDA on an NVIDIA card, DirectML on
//! the primary adapter on another. ONNX Runtime is loaded from the folder setup names for it ("ort" for CUDA, "ort-dml" for DirectML).
//! Spoken as KPipeline does: the chunk's phonemes (pronounce.rs) in pieces of at most 510, ids padded with 0 at both ends,
//! the voice's style row at phoneme count - 1, the speed asked for; phonemes Kokoro's vocabulary lacks are
//! dropped. A chunk without phonemes gives a tenth of a second of silence.
//! Trap: the CUDA provider's DLLs find cuDNN and cuBLAS through PATH, so new() puts the "ort" folder first on
//! it (on Windows, whose environment calls are thread-safe; the heat watch's thread may be running). On Linux the loader reads LD_LIBRARY_PATH only
//! when a program starts, so new() loads every library of the folder itself first (preload), each visible to
//! those loaded after it, and ONNX Runtime's CUDA provider then finds them loaded; that provider itself is
//! left to ONNX Runtime: loaded first, it crashes.
use crate::setup::KokoroFiles;
use crate::speech::wav::RATE;
use anyhow::{Context, Result, bail};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::collections::HashMap;

pub const MAX_PHONEMES: usize = 510;

pub struct Kokoro {
    session: Session,
    vocab: HashMap<char, i64>,
    style: Vec<f32>,
    speed: f32,
}

impl Kokoro {
    pub fn new(files: KokoroFiles, speed: f32) -> Result<Kokoro> {
        let KokoroFiles { ort, model, tokenizer, voice, cuda } = files;
        // once a run: each book makes its own voice, ONNX Runtime is loaded once
        static LOADED: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
        let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        if !*loaded {
            #[cfg(windows)]
            {
                let path = std::env::var_os("PATH").unwrap_or_default();
                let mut dirs = vec![ort.to_path_buf()];
                dirs.extend(std::env::split_paths(&path));
                // SAFETY: Windows' environment calls are thread-safe, and the heat watch's thread may be spawning
                unsafe { std::env::set_var("PATH", std::env::join_paths(dirs)?) };
            }
            #[cfg(unix)]
            preload(ort)?;
            ort::init_from(ort.join(crate::machine::onnxruntime(crate::machine::THIS)))
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

/// Linux: every library in `dir` but ONNX Runtime's providers (preloaded) loaded, globally visible, each kept for the program's life; one whose own
/// libraries are not loaded yet fails and is tried again in the next round, until a round loads none.
#[cfg(unix)]
fn preload(dir: &std::path::Path) -> Result<()> {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    let mut left: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| preloaded(&p.file_name().unwrap_or_default().to_string_lossy()))
        .collect();
    loop {
        let before = left.len();
        // SAFETY: NVIDIA's and ONNX Runtime's libraries, whose initialisers only set up themselves
        left.retain(|p| match unsafe { Library::open(Some(p), RTLD_NOW | RTLD_GLOBAL) } {
            Ok(lib) => {
                std::mem::forget(lib); // kept loaded: the CUDA provider uses it
                false
            }
            Err(_) => true, // its own libraries not loaded yet
        });
        if left.len() == before {
            break;
        }
    }
    Ok(())
}

/// Whether preload loads this file: a library, but not one of ONNX Runtime's providers, whose initialisers
/// need ONNX Runtime to have loaded them (the CUDA provider's crashes on a null host otherwise).
#[cfg_attr(not(unix), allow(dead_code))] // tested on both systems
fn preloaded(name: &str) -> bool {
    name.contains(".so") && !name.starts_with("libonnxruntime_providers_")
}

#[cfg(test)]
mod tests {
    #[test]
    fn onnx_runtime_loads_its_own_providers() {
        let names = ["libcublas.so.13", "libonnxruntime.so.1", "libonnxruntime_providers_cuda.so", "x.txt"];
        let kept: Vec<_> = names.into_iter().filter(|n| super::preloaded(n)).collect();
        assert_eq!(kept, ["libcublas.so.13", "libonnxruntime.so.1"]);
    }

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
}
