//! What setup fetches, per system and card, each by its pinned SHA-256 (not a voice: the 54 are fetched
//! when asked for). An NVIDIA card: ONNX Runtime's CUDA build with CUDA 13's libraries (NVIDIA's own wheels
//! on PyPI, on Linux cuRAND too: its provider links it) and whisper.cpp's CUDA server (upstream's on Windows;
//! upstream has none for Linux, so there our own release file, tools/whisper_server_build.sh).
//! Another card, Windows only: ONNX Runtime with
//! DirectML from Windows ML's NuGet package (Whisper then runs built in, on Vulkan); Linux refuses it.
//! Every card: llama.cpp's server (Vulkan on Windows, CUDA on Linux) and Gemma 4 E4B, Whisper large-v3-turbo,
//! Kokoro v1.0 (onnx-community's export) and the voice asked for, espeak-ng's data (the same files on
//! both systems, taken from the Windows wheel), and ffmpeg. The German voice (german()) comes only when a
//! German book is first read; its tokenizer is the same file as the English model's.
use crate::machine::{Card, Os};
use anyhow::{Result, bail};

/// What one download gives: a file, or what is taken out of an archive.
pub enum Take {
    /// the download itself, at this path in home
    File(String),
    /// every library of the system in the archive (setup::native_lib), flat into this folder of home
    Libs(&'static str),
    /// what lies under the prefix, its folders kept, into this folder of home
    Tree(&'static str, &'static str),
    /// one member, to this path in home
    Member(&'static str, &'static str),
}

pub struct Item {
    pub key: String,
    pub url: String,
    pub sha256: Option<&'static str>,
    pub take: Take,
}

const HF: &str = "https://huggingface.co";
const PYPI: &str = "https://files.pythonhosted.org/packages";
const GH: &str = "https://github.com";

fn item(key: &str, url: &str, sha256: &'static str, take: Take) -> Item {
    Item {
        key: key.into(),
        url: url.into(),
        sha256: Some(sha256),
        take,
    }
}

fn wheel(key: &str, path: &str, sha256: &'static str) -> Item {
    item(key, &format!("{PYPI}/{path}"), sha256, Take::Libs("ort"))
}

/// ONNX Runtime with CUDA 13's libraries and, on Windows, whisper.cpp's CUDA server.
fn cuda(os: Os) -> Vec<Item> {
    match os {
        Os::Windows => vec![
            wheel("onnxruntime", "05/ec/fe99397c6d6582fe26502b608f132a1c7748381fa72523c8b7a0686b9b14/onnxruntime_gpu-1.31.1-cp312-cp312-win_amd64.whl",
                "2bb1e9373baba88f6415257f5b6a589d23e23c16004f7d89adf4344eada5749e"),
            wheel("cuda-runtime", "b7/94/6b867483bec07da24ffa32736c79fabb94ef3a7af4d787a9d4a974868576/nvidia_cuda_runtime-13.0.96-py3-none-win_amd64.whl",
                "f79298c8a098cec150a597c8eba58ecdab96e3bdc4b9bc4f9983635031740492"),
            wheel("cublas", "bc/0f/8982a363ab246d22876f3eb8baccce81f0da6304d39f86d63b67389c5457/nvidia_cublas-13.4.1.3-py3-none-win_amd64.whl",
                "7e7f780caf07bde7b36b0d25bdad29e8ccbacfc4b75066907bdb601af1e6d4aa"),
            wheel("cufft", "85/b2/f8af21a2ed1beed337a6a02c5a28aeb85441f4d578ec3d529543c775ea4b/nvidia_cufft-12.0.0.61-py3-none-win_amd64.whl",
                "2abce5b39d2f5ae12730fb7e5db6696533e36c26e2d3e8fd1750bdd2853364eb"),
            wheel("cudnn", "87/6a/e55ff0ac26a5c6e2b21f41c9d04ad096b4ed6da593fba7e25845c61b0532/nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
                "7d96f634adafd55c72231eb0500ca77ab109ec8ebff7b33000b76e081bc4558e"),
            item("whisper-server", &format!("{GH}/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-win-cuda-12.4.0-x64.zip"),
                "afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a", Take::Tree("Release/", "whisper")),
        ],
        Os::Linux => vec![
            item("onnxruntime", &format!("{GH}/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-gpu_cuda13-1.31.1.tgz"),
                "b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095", Take::Libs("ort")),
            wheel("cuda-runtime", "98/8a/3431271f6344874b8f1ac03f16b3d679c91493f8da63f716160403e6d0a0/nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5"),
            wheel("cublas", "e1/97/d483a41da34df5c88727c57319576967f6d7befba5028d70928d2999635e/nvidia_cublas-13.4.1.3-py3-none-manylinux_2_27_x86_64.whl",
                "a5917cb642cfb07f4a0231ee16aec7e2e0357d8cb8b1b78a0c9c68dba3e6afc7"),
            wheel("cufft", "a8/2f/7b57e29836ea8714f81e9898409196f47d772d5ddedddf1592eadb8ab743/nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "6c44f692dce8fd5ffd3e3df134b6cdb9c2f72d99cf40b62c32dde45eea9ddad3"),
            wheel("cudnn", "af/75/96ea5c5368eb595c39d629cde08a66227a864e71ccb0e593add2612bb952/nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl",
                "9677e76f21862eb5da7ee5ed69d544738b2d8b5c3ce7e5ec125c5592e6cdbdc8"),
            wheel("curand", "07/73/3ee8e5b4cb891401e603ffd3a59b35c6afe785fd2de123afe7c7029603dc/nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl",
                "25c3457ae7a224fdd484dab90b0fc5dc0e842fab5db3012afa4a5bd2af4eb7e5"),
            item("whisper-server", &format!("{GH}/elana-voss/ebook2m4b/releases/download/whisper-server-b5454/whisper-server-b5454-linux-x64.tar.gz"),
                "c07986c261b7327d50339fb526a619836ae97e9ba5919a06186c5327812f6146", Take::Tree("whisper-server-b5454-linux-x64/", "whisper")),
        ],
    }
}

/// llama.cpp's server and ffmpeg, built for each system.
fn programs(os: Os) -> [Item; 2] {
    match os {
        Os::Windows => [
            item("llama-server", &format!("{GH}/ggml-org/llama.cpp/releases/download/b11528/llama-b11528-bin-win-vulkan-x64.zip"),
                "9f4f8222ac7e97531632c991e8e8b9d7532c8df5e527b3b4b1f3165f61eaa0ee", Take::Tree("", "llama")),
            item("ffmpeg", &format!("{GH}/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"),
                "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba",
                Take::Member("ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe", "ffmpeg/ffmpeg.exe")),
        ],
        Os::Linux => [
            item("llama-server", &format!("{GH}/ggml-org/llama.cpp/releases/download/b11528/llama-b11528-bin-ubuntu-cuda-13.4-x64.tar.gz"),
                "7fd907d5710d8f95772e73cc71ca2450d5e1ae78c41563656253bb99ab81e3a4", Take::Tree("llama-b11528/", "llama")),
            item("ffmpeg", &format!("{PYPI}/04/96/15fe7ef562f7494c358cbd9b521d2da82d7a331b5513703756166478bd5f/imageio_ffmpeg-0.7.0-py3-none-manylinux2014_x86_64.whl"),
                "4e1f36bef7bedd4bc5dc25f60a8345d88f57133fa1e754f09b411ac61b854408",
                Take::Member("imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2", "ffmpeg/ffmpeg")),
        ],
    }
}

/// Kerstin, the German voice, and the German model she needs.
pub fn german() -> Vec<Item> {
    const AT: &str = "https://huggingface.co/crane-local-ai/Kokoro-82M-v1.0-German-ONNX/resolve/eeccb15741604cdc894053e1f4c72bb56c1a59be";
    vec![
        item("kerstin-model", &format!("{AT}/onnx/model.onnx"),
            "f541f868499a8186803c72166a09469821e971dbb1e4cd8eb7dcf593f49da0b1", Take::File("models/kerstin/model.onnx".into())),
        item("voice-df_kerstin", &format!("{AT}/voices/df_kerstin.bin"),
            "f96576f656c58221b42ce46c74398dc2a6ffc03b8408abe530d85ebd717ca062", Take::File("models/kerstin/df_kerstin.bin".into())),
    ]
}

/// Everything setup fetches on `os` for `card` and `voice`; Linux refuses a card other than NVIDIA's.
pub fn items(os: Os, card: Card, voice: &str) -> Result<Vec<Item>> {
    let mut list = match (card, os) {
        (Card::Nvidia, _) => cuda(os),
        (Card::Other, Os::Windows) => vec![item(
            "onnxruntime",
            // Windows ML: ONNX Runtime 1.28.3 with DirectML (ort needs 1.27 or later; the
            // onnxruntime-directml wheel and NuGet's Microsoft.ML.OnnxRuntime.DirectML stop at 1.24.4)
            "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/2.5.96/microsoft.windows.ai.machinelearning.2.5.96.nupkg",
            "35d6e84feb4142e8cecaf36704fa3ab0fc8251d892850fb80e8341f7d09653a3",
            Take::Libs("ort-dml"),
        )],
        (Card::Other, Os::Linux) => bail!(
            "No NVIDIA card found (nvidia-smi lists none): on Linux the voice, Whisper and the language model run on an NVIDIA card with its driver only"
        ),
    };
    list.extend(programs(os));
    list.extend([
        item("llm", &format!("{HF}/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf"),
            "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee", Take::File("models/gemma-4-E4B_q4_0-it.gguf".into())),
        item("whisper-model", &format!("{HF}/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin"),
            "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1", Take::File("models/ggml-large-v3-turbo-q8_0.bin".into())),
        item("kokoro", &format!("{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"),
            "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb", Take::File("models/kokoro/model.onnx".into())),
        item("kokoro-tokenizer", &format!("{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json"),
            "77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34", Take::File("models/kokoro/tokenizer.json".into())),
        item("espeak-ng-data", &format!("{PYPI}/9d/ed/a3d872fbad4f3a3f3db0e8c31768ab14e77cd77306de16b8b20b1e1df7ea/espeakng_loader-0.2.4-py3-none-win_amd64.whl"),
            "41f1e08ac9deda2efd1ea9de0b81dab9f5ae3c4b24284f76533d0a7b1dd7abd7", Take::Tree("espeakng_loader/espeak-ng-data/", "espeak-ng-data")),
    ]);
    // a voice of the 54 is fetched when first asked for; first, as a mistyped one fails on its 404
    list.insert(
        0,
        Item {
            key: format!("voice-{voice}"),
            url: format!("{HF}/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/{voice}.bin"),
            sha256: None,
            take: Take::File(format!("models/kokoro/voices/{voice}.bin")),
        },
    );
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(list: &[Item]) -> Vec<&str> {
        list.iter().map(|i| i.key.as_str()).collect()
    }

    #[test]
    fn a_mistyped_voice_fails_before_the_big_downloads() {
        let list = items(Os::Windows, Card::Nvidia, "af_zzz").unwrap();
        assert_eq!(list[0].key, "voice-af_zzz");
    }

    #[test]
    fn another_card_gets_directml_and_no_cuda() {
        let list = items(Os::Windows, Card::Other, "bm_george").unwrap();
        let keys = keys(&list);
        let ort = list.iter().find(|i| i.key == "onnxruntime").unwrap();
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
    fn linux_gets_its_own_builds_with_a_whisper_server() {
        let list = items(Os::Linux, Card::Nvidia, "af_bella").unwrap();
        let win = items(Os::Windows, Card::Nvidia, "af_bella").unwrap();
        // Linux's CUDA provider links cuRAND (libcurand.so.10); Windows' loads it never
        let mut want = keys(&win);
        let at = want.iter().position(|k| *k == "cudnn").unwrap() + 1;
        want.insert(at, "curand");
        assert_eq!(keys(&list), want);
        for it in &list {
            assert!(!it.url.contains("win") || it.key == "espeak-ng-data", "{}", it.url);
            assert!(it.sha256.is_some() || it.key.starts_with("voice-"));
        }
        assert!(items(Os::Linux, Card::Other, "af_bella").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn the_server_build_refuses_a_folder_holding_another_checkout() {
        use std::os::unix::fs::PermissionsExt;
        let work = std::env::temp_dir().join(format!("ebook2m4b-server-{}", std::process::id()));
        std::fs::create_dir_all(work.join("src")).unwrap();
        // no network even if the script went on
        let curl = work.join("curl");
        std::fs::write(&curl, "#!/bin/sh
exit 7
").unwrap();
        std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = format!("{}:{}", work.display(), std::env::var("PATH").unwrap());
        let out = std::process::Command::new("bash")
            .args(["tools/whisper_server_build.sh", "b1"])
            .arg(&work)
            .env("PATH", path)
            .output()
            .unwrap();
        let said = String::from_utf8_lossy(&out.stderr).into_owned();
        let _ = std::fs::remove_dir_all(&work); // a temp folder; left behind it harms nothing
        assert!(!out.status.success() && said.contains("another checkout"), "{said}");
    }
}
