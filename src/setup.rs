//! First start: what the program needs is fetched once into its home folder, %LOCALAPPDATA%\ebook2m4b,
//! never system-wide; home\setup.json names where each piece is, so a later start fetches nothing.
//!
//! An NVIDIA card gets ONNX Runtime's CUDA build with CUDA 13's libraries (NVIDIA's own wheels on PyPI)
//! and whisper.cpp's CUDA server; another card ONNX Runtime with DirectML from Windows ML's NuGet package (Whisper then runs built in,
//! on Vulkan). Every card: llama.cpp's server (Vulkan) and Gemma 4 E4B for what the text rules cannot
//! settle, Whisper large-v3-turbo, Kokoro v1.0 (onnx-community's export) and the voice asked for,
//! espeak-ng's data (for words the dictionary lacks), and ffmpeg.
//! Each download is checked by its pinned SHA-256 (not a voice: the 54 are fetched when asked for).
//! Kokoro's model is then patched for the GPU once per patch version (onnx.rs), as "kokoro-gpu" ("kokoro-gpu-dml" on DirectML).
//! Each card's ONNX Runtime has its own folder (ort, ort-dml),
//! so a card changed later fetches its own. ONNX Runtime's DLLs nothing loads
//! (TensorRT, cuFFTW, NVBLAS) are not unpacked.
use crate::gpu::Card;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// What one download gives: a file, or what is taken out of an archive.
enum Take {
    /// the download itself, at this path in home
    File(String),
    /// every .dll in the archive, into this folder of home
    Dlls(&'static str),
    /// what lies under the prefix, its folders kept, into this folder of home
    Tree(&'static str, &'static str),
    /// one member, to this path in home
    Member(&'static str, &'static str),
}

struct Item {
    key: String,
    url: String,
    sha256: Option<&'static str>,
    take: Take,
}

const HF: &str = "https://huggingface.co";
const PYPI: &str = "https://files.pythonhosted.org/packages";

fn item(key: &str, url: &str, sha256: Option<&'static str>, take: Take) -> Item {
    Item {
        key: key.into(),
        url: url.into(),
        sha256,
        take,
    }
}

fn items(card: Card, voice: &str) -> Vec<Item> {
    let mut list = match card {
        Card::Nvidia => vec![
            item(
                "onnxruntime",
                &format!(
                    "{PYPI}/05/ec/fe99397c6d6582fe26502b608f132a1c7748381fa72523c8b7a0686b9b14/onnxruntime_gpu-1.31.1-cp312-cp312-win_amd64.whl"
                ),
                Some("2bb1e9373baba88f6415257f5b6a589d23e23c16004f7d89adf4344eada5749e"),
                Take::Dlls("ort"),
            ),
            item(
                "cuda-runtime",
                &format!(
                    "{PYPI}/b7/94/6b867483bec07da24ffa32736c79fabb94ef3a7af4d787a9d4a974868576/nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl"
                ),
                Some("f79298c8a098cec150a597c8eba58ecdab96e3bdc4b9bc4f9983635031740492"),
                Take::Dlls("ort"),
            ),
            item(
                "cublas",
                &format!(
                    "{PYPI}/bc/0f/8982a363ab246d22876f3eb8baccce81f0da6304d39f86d63b67389c5457/nvidia_cublas-13.4.1.3-py3-none-win_amd64.whl"
                ),
                Some("7e7f780caf07bde7b36b0d25bdad29e8ccbacfc4b75066907bdb601af1e6d4aa"),
                Take::Dlls("ort"),
            ),
            item(
                "cufft",
                &format!(
                    "{PYPI}/85/b2/f8af21a2ed1beed337a6a02c5a28aeb85441f4d578ec3d529543c775ea4b/nvidia_cufft-12.0.0.61-py3-none-win_amd64.whl"
                ),
                Some("2abce5b39d2f5ae12730fb7e5db6696533e36c26e2d3e8fd1750bdd2853364eb"),
                Take::Dlls("ort"),
            ),
            item(
                "cudnn",
                &format!(
                    "{PYPI}/87/6a/e55ff0ac26a5c6e2b21f41c9d04ad096b4ed6da593fba7e25845c61b0532/nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl"
                ),
                Some("7d96f634adafd55c72231eb0500ca77ab109ec8ebff7b33000b76e081bc4558e"),
                Take::Dlls("ort"),
            ),
            item(
                "whisper-server",
                "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-win-cuda-12.4.0-x64.zip",
                Some("afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a"),
                Take::Tree("Release/", "whisper"),
            ),
        ],
        Card::Other => vec![item(
            "onnxruntime",
            // Windows ML: ONNX Runtime 1.28.3 with DirectML (ort needs 1.27 or later; the
            // onnxruntime-directml wheel and NuGet's Microsoft.ML.OnnxRuntime.DirectML stop at 1.24.4)
            "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/2.5.96/microsoft.windows.ai.machinelearning.2.5.96.nupkg",
            Some("35d6e84feb4142e8cecaf36704fa3ab0fc8251d892850fb80e8341f7d09653a3"),
            Take::Dlls("ort-dml"),
        )],
    };
    list.extend([
        item("llama-server", "https://github.com/ggml-org/llama.cpp/releases/download/b11528/llama-b11528-bin-win-vulkan-x64.zip",
            Some("9f4f8222ac7e97531632c991e8e8b9d7532c8df5e527b3b4b1f3165f61eaa0ee"), Take::Tree("", "llama")),
        item("llm", &format!("{HF}/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf"),
            Some("676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"), Take::File("models/gemma-4-E4B_q4_0-it.gguf".into())),
        item("whisper-model", &format!("{HF}/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin"),
            Some("317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1"), Take::File("models/ggml-large-v3-turbo-q8_0.bin".into())),
        item("kokoro", &format!("{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"),
            Some("8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"), Take::File("models/kokoro/model.onnx".into())),
        item("kokoro-tokenizer", &format!("{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json"),
            Some("77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34"), Take::File("models/kokoro/tokenizer.json".into())),
        item("espeak-ng-data", &format!("{PYPI}/9d/ed/a3d872fbad4f3a3f3db0e8c31768ab14e77cd77306de16b8b20b1e1df7ea/espeakng_loader-0.2.4-py3-none-win_amd64.whl"),
            Some("41f1e08ac9deda2efd1ea9de0b81dab9f5ae3c4b24284f76533d0a7b1dd7abd7"), Take::Tree("espeakng_loader/espeak-ng-data/", "espeak-ng-data")),
        item("ffmpeg", "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip",
            Some("60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"), Take::Member("ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe", "ffmpeg/ffmpeg.exe")),
    ]);
    // a voice of the 54 is fetched when first asked for
    list.push(item(
        &format!("voice-{voice}"),
        &format!("{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/{voice}.bin"),
        None,
        Take::File(format!("models/kokoro/voices/{voice}.bin")),
    ));
    list
}

/// Kokoro's model patched for this card: its key in setup.json and its path in home (DirectML's has the
/// ConvTranspose pads changed too, so a card changed later gets its own). The patch's version is in the name,
/// so a model made by an older patch is made again.
pub fn gpu_model(card: Card) -> (&'static str, String) {
    let name = match card {
        Card::Nvidia => "model-gpu",
        Card::Other => "model-gpu-dml",
    };
    let key = if card == Card::Nvidia {
        "kokoro-gpu"
    } else {
        "kokoro-gpu-dml"
    };
    (
        key,
        format!("models/kokoro/{name}-v{}.onnx", crate::onnx::VERSION),
    )
}

/// Whether setup.json's patched model is there and made by this patch.
fn patched_is_current(found: &BTreeMap<String, PathBuf>, key: &str, want: &Path) -> bool {
    found.get(key).is_some_and(|p| p == want && p.exists())
}

/// Where each piece is, after `ensure`.
pub struct Tools {
    pub home: PathBuf,
    pub card: Card,
    /// each piece's key (as in setup.json) and where it is
    pub found: BTreeMap<String, PathBuf>,
}

pub fn home() -> Result<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set")?;
    Ok(PathBuf::from(local).join("ebook2m4b"))
}

/// Fetches what is missing for this card and voice; says what it fetches and how far it is.
pub fn ensure(card: Card, voice: &str) -> Result<Tools> {
    let home = home()?;
    fs::create_dir_all(&home)?;
    let state = home.join("setup.json");
    let mut found: BTreeMap<String, PathBuf> = match fs::read_to_string(&state) {
        Ok(s) => serde_json::from_str(&s).context("setup.json")?,
        Err(_) => BTreeMap::new(), // first start
    };
    for it in items(card, voice) {
        if found
            .get(&it.key)
            .is_some_and(|p| p.exists() && in_place(&it.take, p, &home))
        {
            continue;
        }
        let at = match &it.take {
            Take::File(rel) => {
                let dest = home.join(rel);
                download(&it, &dest)?;
                dest
            }
            _ => {
                let archive = home
                    .join("downloads")
                    .join(it.url.rsplit('/').next().unwrap_or("download"));
                download(&it, &archive)?;
                let at = unpack(&it.take, &archive, &home)?;
                fs::remove_file(&archive)?;
                at
            }
        };
        found.insert(it.key.clone(), at);
        fs::write(&state, serde_json::to_string_pretty(&found)?)?;
    }
    let (key, rel) = gpu_model(card);
    let dst = home.join(rel);
    if !patched_is_current(&found, key, &dst) {
        let src = found
            .get("kokoro")
            .context("Kokoro's model is not set up")?;
        println!("Kokoro's model is made ready for the GPU");
        crate::onnx::patch(src, &dst, card == Card::Other)?;
        if let Some(old) = found.get(key).filter(|p| **p != dst) {
            let _ = fs::remove_file(old); // an older patch's file; one already gone is fine
        }
        found.insert(key.into(), dst);
        fs::write(&state, serde_json::to_string_pretty(&found)?)?;
    }
    Ok(Tools { home, card, found })
}

/// Whether a piece found at `at` lies where this card's own would: each card's ONNX Runtime has its own
/// folder, so a card changed later fetches its own instead of loading the other's.
fn in_place(take: &Take, at: &Path, home: &Path) -> bool {
    match take {
        Take::Dlls(dir) => at == home.join(dir),
        _ => true,
    }
}

fn download(it: &Item, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest.parent().context("no folder")?)?;
    let (size, mut reader) = crate::http::get(&it.url)?;
    let name = it.url.rsplit('/').next().unwrap_or(&it.key);
    println!("Downloading {name} ({} MB)", size / 1_000_000);
    let part = dest.with_extension("part");
    let mut file = fs::File::create(&part)?;
    let mut hash = crate::hash::Sha256::default();
    let (mut got, mut shown) = (0u64, 0u64);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hash.write_all(&buf[..n])?;
        got += n as u64;
        if size > 0 && got * 20 / size > shown {
            shown = got * 20 / size;
            print!("\r  {}%", shown * 5);
            std::io::stdout().flush()?;
        }
    }
    println!();
    drop(file);
    let digest = hash.hex();
    if it.sha256.is_some_and(|want| digest != want) {
        fs::remove_file(&part)?;
        bail!("{name}: the download is not the expected file (SHA-256 {digest})");
    }
    fs::rename(&part, dest)?;
    Ok(())
}

/// DLLs of the wheels that nothing loads: TensorRT's provider, cuFFTW, NVBLAS.
const UNUSED_DLLS: [&str; 3] = [
    "/onnxruntime_providers_tensorrt.dll",
    "/cufftw64_12.dll",
    "/nvblas64_13.dll",
];

/// Whether a DLL of a wheel or NuGet package is one an x64 program loads: a NuGet package also holds
/// arm64 builds of the same names and .NET assemblies (lib/), which would overwrite or clutter the x64 ones.
fn x64_native(name: &str) -> bool {
    !name.starts_with("lib/")
        && (!name.starts_with("runtimes/") || name.starts_with("runtimes/win-x64/"))
}

/// Takes what `take` names out of the archive into home; returns where it went.
fn unpack(take: &Take, archive: &Path, home: &Path) -> Result<PathBuf> {
    let at = match take {
        Take::Dlls(dir) | Take::Tree(_, dir) => home.join(dir),
        Take::Member(_, path) => home.join(path),
        Take::File(_) => unreachable!("not an archive"),
    };
    crate::archive::Archive::open(archive)?.each(|name, is_dir, bytes| {
        let dest = match take {
            Take::Dlls(dir)
                if name.ends_with(".dll")
                    && !UNUSED_DLLS.iter().any(|u| name.ends_with(u))
                    && x64_native(name) =>
            {
                home.join(dir).join(name.rsplit('/').next().unwrap_or(name))
            }
            Take::Tree(prefix, dir) if name.starts_with(prefix) && !is_dir => {
                home.join(dir).join(&name[prefix.len()..])
            }
            Take::Member(member, path) if name == *member => home.join(path),
            _ => return Ok(()),
        };
        fs::create_dir_all(dest.parent().context("no folder")?)?;
        std::io::copy(bytes, &mut fs::File::create(&dest)?)?;
        Ok(())
    })?;
    if !at.exists() {
        bail!("{} did not hold what was expected", archive.display());
    }
    Ok(at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_runtime_in_another_cards_folder_is_fetched_again() {
        let home = Path::new("h");
        assert!(!in_place(&Take::Dlls("ort-dml"), &home.join("ort"), home));
        assert!(in_place(&Take::Dlls("ort"), &home.join("ort"), home));
    }

    #[test]
    fn another_card_gets_directml_and_no_cuda() {
        let list = items(Card::Other, "bm_george");
        let keys: Vec<&str> = list.iter().map(|i| i.key.as_str()).collect();
        let ort = &list[0];
        assert!(
            ort.key == "onnxruntime" && ort.url.contains("microsoft.windows.ai.machinelearning")
        );
        for cuda in ["cuda-runtime", "cublas", "cufft", "cudnn", "whisper-server"] {
            assert!(!keys.contains(&cuda), "{cuda} on the DirectML route");
        }
        assert!(list.iter().all(|i| !i.url.contains("cuda")));
        for shared in ["llama-server", "llm", "whisper-model", "kokoro", "ffmpeg"] {
            assert!(keys.contains(&shared), "{shared} missing");
        }
        assert!(!keys.contains(&"calibre"), "mobi is read without Calibre");
        assert!(keys.contains(&"voice-bm_george"));
        // every piece but the voice is checked by its SHA-256
        assert!(
            list.iter()
                .all(|i| i.sha256.is_some() || i.key.starts_with("voice-"))
        );
    }

    #[test]
    fn an_archive_gives_only_what_is_asked_for() {
        let dir = std::env::temp_dir().join(format!("t2a-unpack-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.zip");
        let names = [
            "pkg/capi/a.dll",
            "pkg/b.txt",
            "data/x/y.bin",
            "bin/tool.exe",
        ];
        let files: Vec<_> = names.iter().map(|n| (*n, n.as_bytes())).collect();
        crate::archive::write(&archive, &files);
        unpack(&Take::Dlls("ort"), &archive, &dir).unwrap();
        unpack(&Take::Tree("data/", "tree"), &archive, &dir).unwrap();
        unpack(&Take::Member("bin/tool.exe", "t/tool.exe"), &archive, &dir).unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("ort/a.dll")).unwrap(),
            "pkg/capi/a.dll"
        );
        assert!(!dir.join("ort/b.txt").exists());
        assert_eq!(
            fs::read_to_string(dir.join("tree/x/y.bin")).unwrap(),
            "data/x/y.bin"
        );
        assert_eq!(
            fs::read_to_string(dir.join("t/tool.exe")).unwrap(),
            "bin/tool.exe"
        );
        assert!(unpack(&Take::Member("missing", "m/missing"), &archive, &dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn each_card_has_its_own_patched_model() {
        let (nk, np) = gpu_model(Card::Nvidia);
        let (ok, op) = gpu_model(Card::Other);
        assert!(nk != ok && np != op);
    }

    #[test]
    fn a_model_patched_by_an_older_patch_is_made_again() {
        let dir = std::env::temp_dir().join(format!("t2a-stale-{}", std::process::id()));
        let (key, rel) = gpu_model(Card::Nvidia);
        let old = dir.join("models/kokoro/model-gpu.onnx");
        let now = dir.join(&rel);
        for p in [&old, &now] {
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, b"x").unwrap();
        }
        let mut found = BTreeMap::new();
        found.insert(key.to_string(), old);
        assert!(!patched_is_current(&found, key, &now));
        found.insert(key.to_string(), now.clone());
        assert!(patched_is_current(&found, key, &now));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_dlls_nothing_loads_are_not_unpacked() {
        let dir = std::env::temp_dir().join(format!("t2a-unused-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.zip");
        let unused = [
            "onnxruntime_providers_tensorrt.dll",
            "cufftw64_12.dll",
            "nvblas64_13.dll",
        ];
        let names: Vec<String> = unused
            .iter()
            .chain(&["onnxruntime.dll"])
            .map(|n| format!("pkg/{n}"))
            .collect();
        let files: Vec<_> = names.iter().map(|n| (n.as_str(), &b"x"[..])).collect();
        crate::archive::write(&archive, &files);
        unpack(&Take::Dlls("ort"), &archive, &dir).unwrap();
        assert!(dir.join("ort/onnxruntime.dll").exists());
        for name in unused {
            assert!(!dir.join("ort").join(name).exists(), "{name} unpacked");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_the_x64_dlls_of_a_nuget_package_are_unpacked() {
        let dir = std::env::temp_dir().join(format!("t2a-nupkg-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.nupkg");
        let x64 = &b"x64"[..];
        crate::archive::write(
            &archive,
            &[
                ("runtimes/win-x64/native/onnxruntime.dll", x64),
                ("runtimes/win-arm64/native/onnxruntime.dll", &b"arm"[..]),
                ("runtimes/win-arm64ec/native/onnxruntime.dll", &b"arm"[..]),
                ("lib/net8.0/Microsoft.ML.OnnxRuntime.dll", &b"net"[..]),
            ],
        );
        unpack(&Take::Dlls("ort-dml"), &archive, &dir).unwrap();
        assert_eq!(fs::read(dir.join("ort-dml/onnxruntime.dll")).unwrap(), x64);
        assert!(!dir.join("ort-dml/Microsoft.ML.OnnxRuntime.dll").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
