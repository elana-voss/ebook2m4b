//! First start: what the program needs (downloads.rs, per system and card) is fetched once into its home
//! folder (machine::home: %LOCALAPPDATA%\ebook2m4b, on Linux ~/.local/share/ebook2m4b), never system-wide;
//! home/setup.json names where each piece is, so a later start fetches nothing.
//! Each download is checked by its pinned SHA-256 before it is unpacked.
//! Kokoro's model is then patched once per patch version (onnx.rs: for the GPU, and the silence between words
//! limited), as "kokoro-gpu" ("kokoro-gpu-dml" on DirectML); Kerstin's likewise as "kerstin-gpu[-dml]".
//! An older patch version's copies, for either card, are removed.
//! Each card's ONNX Runtime has its own folder (ort, ort-dml), so a card changed later fetches its own.
//! Kerstin, the German voice, and her model are fetched (and patched) the first time a German book is read
//! (Tools::kerstin).
//! Libraries nothing loads (TensorRT's provider, cuFFTW, NVBLAS) are not unpacked.
//! Linux: links in the archives are made again, every file taken may be run, and the libraries' folders
//! are put on LD_LIBRARY_PATH for the programs started later (llama-server, whisper-server).
pub mod archive;
pub mod downloads;
pub mod hash;
pub mod http;
pub mod process;

use crate::setup::downloads::{Item, Take, german, items};
use crate::machine::{Card, Os, THIS};
use crate::setup::archive::Kind;
use crate::speech::onnx::Model;
use anyhow::{Context, Result, bail};
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// A speech model patched for this card: its key in setup.json and its path in home (DirectML's has the
/// ConvTranspose pads changed too, so a card changed later gets its own). The patch's version is in the name,
/// so a model made by an older patch is made again.
pub fn patched_model(model: Model, card: Card) -> (&'static str, String) {
    let (key, folder) = match (model, card) {
        (Model::Kokoro, Card::Nvidia) => ("kokoro-gpu", "kokoro"),
        (Model::Kokoro, Card::Other) => ("kokoro-gpu-dml", "kokoro"),
        (Model::Kerstin, Card::Nvidia) => ("kerstin-gpu", "kerstin"),
        (Model::Kerstin, Card::Other) => ("kerstin-gpu-dml", "kerstin"),
    };
    let name = if card == Card::Nvidia { "model-gpu" } else { "model-gpu-dml" };
    (
        key,
        format!("models/{folder}/{name}-v{}.onnx", crate::speech::onnx::VERSION),
    )
}

/// Makes `model`'s patched copy for `card` from the downloaded one (`source`, its key) unless this patch
/// made it already; an older patch's files, for either card, are removed.
fn make_patched(
    found: &mut BTreeMap<String, PathBuf>,
    home: &Path,
    state: &Path,
    model: Model,
    card: Card,
) -> Result<()> {
    let (key, rel) = patched_model(model, card);
    let dst = home.join(rel);
    let known = found.len();
    if let Some(folder) = dst.parent() {
        remove_older_patches(found, folder);
    }
    if patched_is_current(found, key, &dst) {
        if found.len() != known {
            fs::write(state, serde_json::to_string_pretty(&found)?)?;
        }
        return Ok(());
    }
    let source = if model == Model::Kokoro { "kokoro" } else { "kerstin-model" };
    let src = found.get(source).with_context(|| format!("{source} is not set up"))?;
    println!("{model:?}'s model is made ready (once)");
    crate::speech::onnx::patch(src, &dst, model, card == Card::Other)?;
    if let Some(old) = found.get(key).filter(|p| **p != dst) {
        let _ = fs::remove_file(old); // an older patch's file; one already gone is fine
    }
    found.insert(key.into(), dst);
    if let Some(old) = found.remove("kerstin-model-dml") {
        let _ = fs::remove_file(old); // Kerstin's model with only DirectML's pads, before version 3; gone is fine
    }
    fs::write(state, serde_json::to_string_pretty(&found)?)?;
    Ok(())
}

/// Removes from `folder` every patched model (model-gpu[-dml]-v<n>.onnx) of an older patch version, for
/// either card, and the setup.json keys naming one.
fn remove_older_patches(found: &mut BTreeMap<String, PathBuf>, folder: &Path) {
    let older = |name: &str| {
        let Some(rest) = name.strip_prefix("model-gpu").and_then(|r| r.strip_suffix(".onnx")) else {
            return false;
        };
        let rest = rest.strip_prefix("-dml").unwrap_or(rest);
        rest.strip_prefix("-v")
            .and_then(|n| n.parse::<u32>().ok())
            .is_some_and(|n| n < crate::speech::onnx::VERSION)
    };
    let Ok(entries) = fs::read_dir(folder) else {
        return; // no folder, nothing to remove
    };
    for path in entries.flatten().map(|e| e.path()) {
        if path.file_name().and_then(|n| n.to_str()).is_some_and(older) {
            let _ = fs::remove_file(&path); // a copy that stays only costs disk; the next start tries again
            found.retain(|_, p| *p != path);
        }
    }
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
    /// the German voice's pieces, once the first German book has fetched them
    german: OnceCell<BTreeMap<String, PathBuf>>,
}

/// What Kokoro is made from: `ort` holds ONNX Runtime and its provider's libraries; `model` the export
/// patched for this card; `voice` a voice's .bin; `cuda` whether the card is NVIDIA's.
#[derive(Clone, Copy)]
pub struct KokoroFiles<'a> {
    pub ort: &'a Path,
    pub model: &'a Path,
    pub tokenizer: &'a Path,
    pub voice: &'a Path,
    pub cuda: bool,
}

impl Tools {
    fn piece(&self, key: &str) -> Result<&Path> {
        self.found
            .get(key)
            .map(PathBuf::as_path)
            .with_context(|| format!("{key} is not set up"))
    }

    /// eSpeak's data folder.
    pub fn espeak(&self) -> Result<&Path> {
        self.piece("espeak-ng-data")
    }

    pub fn ffmpeg(&self) -> Result<&Path> {
        self.piece("ffmpeg")
    }

    /// Kokoro's files for this card and `voice` (a voice's name, as af_bella).
    pub fn kokoro(&self, voice: &str) -> Result<KokoroFiles<'_>> {
        Ok(KokoroFiles {
            ort: self.piece("onnxruntime")?,
            model: self.piece(patched_model(Model::Kokoro, self.card).0)?,
            tokenizer: self.piece("kokoro-tokenizer")?,
            voice: self.piece(&format!("voice-{voice}"))?,
            cuda: self.card == Card::Nvidia,
        })
    }

    /// Kokoro's files for a German book, fetched the first time one is read: Kerstin's model patched once
    /// beside it (the silence between words limited; on another card also the pads DirectML takes).
    pub fn kerstin(&self) -> Result<KokoroFiles<'_>> {
        if self.german.get().is_none() {
            let mut found = self.found.clone();
            let state = self.home.join(STATE);
            fetch(german(), &mut found, &self.home, &state)?;
            make_patched(&mut found, &self.home, &state, Model::Kerstin, self.card)?;
            // set just above if empty; a run is one thread, nothing else sets it
            let _ = self.german.set(found);
        }
        let german = self.german.get().context("the German voice is not set up")?;
        kerstin_files(german, self.piece("onnxruntime")?, self.card)
    }

    /// The route this card takes, as said to the reader.
    pub fn route(&self) -> &'static str {
        if self.card == Card::Nvidia { "CUDA" } else { "DirectML" }
    }
}

fn kerstin_files<'a>(german: &'a BTreeMap<String, PathBuf>, ort: &'a Path, card: Card) -> Result<KokoroFiles<'a>> {
    let piece = |key: &str| {
        german
            .get(key)
            .map(PathBuf::as_path)
            .with_context(|| format!("{key} is not set up"))
    };
    Ok(KokoroFiles {
        ort,
        model: piece(patched_model(Model::Kerstin, card).0)?,
        tokenizer: piece("kokoro-tokenizer")?,
        voice: piece("voice-df_kerstin")?,
        cuda: card == Card::Nvidia,
    })
}

pub fn home() -> Result<PathBuf> {
    crate::machine::home(crate::machine::THIS, |k| std::env::var(k).ok())
}

/// What setup.json holds, under home.
const STATE: &str = "setup.json";

/// Fetches the items not in place yet into `found`, written to `state` after each.
fn fetch(list: Vec<Item>, found: &mut BTreeMap<String, PathBuf>, home: &Path, state: &Path) -> Result<()> {
    for it in list {
        if found
            .get(&it.key)
            .is_some_and(|p| p.exists() && in_place(&it.take, p, home))
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
                let at = unpack(THIS, &it.take, &archive, home)?;
                fs::remove_file(&archive)?;
                at
            }
        };
        found.insert(it.key.clone(), at);
        fs::write(state, serde_json::to_string_pretty(&found)?)?;
    }
    Ok(())
}

/// Fetches what is missing for this card and voice; says what it fetches and how far it is.
pub fn ensure(card: Card, voice: &str) -> Result<Tools> {
    let home = home()?;
    fs::create_dir_all(&home)?;
    let state = home.join(STATE);
    let mut found: BTreeMap<String, PathBuf> = match fs::read_to_string(&state) {
        Ok(s) => serde_json::from_str(&s).context("setup.json")?,
        Err(_) => BTreeMap::new(), // first start
    };
    fetch(items(THIS, card, voice)?, &mut found, &home, &state)?;
    make_patched(&mut found, &home, &state, Model::Kokoro, card)?;
    #[cfg(unix)]
    {
        let old = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let path = library_path(vec![home.join("ort"), home.join("llama")], &old)?;
        // SAFETY: setup runs before this program starts any other thread
        unsafe { std::env::set_var("LD_LIBRARY_PATH", path) };
    }
    Ok(Tools { home, card, found, german: OnceCell::new() })
}

/// The library search path: `front` first, then `old`'s entries. An empty `old` adds no entry (one empty
/// entry would make the loader search the current folder).
#[cfg_attr(not(unix), allow(dead_code))]
fn library_path(front: Vec<PathBuf>, old: &std::ffi::OsStr) -> Result<std::ffi::OsString> {
    let mut dirs = front;
    if !old.is_empty() {
        dirs.extend(std::env::split_paths(old));
    }
    Ok(std::env::join_paths(dirs)?)
}

/// Whether a piece found at `at` lies where this card's own would: each card's ONNX Runtime has its own
/// folder, so a card changed later fetches its own instead of loading the other's.
fn in_place(take: &Take, at: &Path, home: &Path) -> bool {
    match take {
        Take::Libs(dir) => at == home.join(dir),
        _ => true,
    }
}

fn download(it: &Item, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest.parent().context("no folder")?)?;
    let (size, mut reader) = crate::setup::http::get(&it.url)?;
    let name = it.url.rsplit('/').next().unwrap_or(&it.key);
    println!("Downloading {name} ({} MB)", size / 1_000_000);
    let part = dest.with_extension("part");
    let mut file = fs::File::create(&part)?;
    let mut hash = crate::setup::hash::Sha256::default();
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

/// Libraries of the downloads that nothing loads: TensorRT's provider, cuFFTW, NVBLAS.
const UNUSED_LIBS: [&str; 3] = ["onnxruntime_providers_tensorrt", "cufftw", "nvblas"];

/// Whether an archive's member is a library `os` loads, one setup takes: a DLL on Windows (a NuGet package
/// also holds arm64 builds of the same names and .NET assemblies, lib/, which would overwrite or clutter the
/// x64 ones); on Linux a shared object (name.so or name.so.N) in a lib/ folder.
fn native_lib(os: Os, name: &str) -> bool {
    let file = name.rsplit('/').next().unwrap_or(name);
    if UNUSED_LIBS.iter().any(|u| file.contains(u)) {
        return false;
    }
    match os {
        Os::Windows => {
            file.ends_with(".dll")
                && !name.starts_with("lib/")
                && (!name.starts_with("runtimes/") || name.starts_with("runtimes/win-x64/"))
        }
        Os::Linux => {
            name.contains("lib/") && (file.ends_with(".so") || file.contains(".so."))
        }
    }
}

/// Takes what `take` names out of the archive into home; returns where it went. Links are made again
/// (on Linux; Windows' downloads hold none), and on Linux every file taken may be run.
fn unpack(os: Os, take: &Take, archive: &Path, home: &Path) -> Result<PathBuf> {
    let at = match take {
        Take::Libs(dir) | Take::Tree(_, dir) => home.join(dir),
        Take::Member(_, path) => home.join(path),
        Take::File(_) => unreachable!("not an archive"),
    };
    crate::setup::archive::Archive::open(archive)?.each(|name, kind, bytes| {
        let dest = match take {
            Take::Libs(dir) if kind != Kind::Dir && native_lib(os, name) => {
                home.join(dir).join(name.rsplit('/').next().unwrap_or(name))
            }
            Take::Tree(prefix, dir) if name.starts_with(prefix) && kind != Kind::Dir => {
                home.join(dir).join(&name[prefix.len()..])
            }
            Take::Member(member, path) if name == *member => home.join(path),
            _ => return Ok(()),
        };
        fs::create_dir_all(dest.parent().context("no folder")?)?;
        match kind {
            Kind::Link(to) => link(&to, &dest)?,
            _ => {
                std::io::copy(bytes, &mut fs::File::create(&dest)?)?;
                runnable(&dest)?;
            }
        }
        Ok(())
    })?;
    if !at.exists() {
        bail!("{} did not hold what was expected", archive.display());
    }
    Ok(at)
}

/// A symbolic link at `dest` to `to` (a name beside it), one already there replaced.
#[cfg(unix)]
fn link(to: &str, dest: &Path) -> Result<()> {
    let _ = fs::remove_file(dest); // none there is the usual case
    std::os::unix::fs::symlink(to, dest).with_context(|| dest.display().to_string())
}

/// Windows' downloads hold no links; a Linux archive unpacked here (tests) gets its link as a file naming the target.
#[cfg(not(unix))]
fn link(to: &str, dest: &Path) -> Result<()> {
    Ok(fs::write(dest, to)?)
}

#[cfg(unix)]
fn runnable(file: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    Ok(fs::set_permissions(file, fs::Permissions::from_mode(0o755))?)
}

#[cfg(not(unix))]
fn runnable(_: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_path_has_no_empty_entry() {
        let front = vec![PathBuf::from("ort"), PathBuf::from("llama")];
        let none = library_path(front.clone(), std::ffi::OsStr::new("")).unwrap();
        assert_eq!(std::env::split_paths(&none).collect::<Vec<_>>(), front);
        let old = std::env::join_paths(["a"]).unwrap();
        let some = library_path(front, &old).unwrap();
        assert_eq!(std::env::split_paths(&some).count(), 3);
    }

    #[test]
    fn a_runtime_in_another_cards_folder_is_fetched_again() {
        let home = Path::new("h");
        assert!(!in_place(&Take::Libs("ort-dml"), &home.join("ort"), home));
        assert!(in_place(&Take::Libs("ort"), &home.join("ort"), home));
    }

    #[test]
    fn an_archive_gives_only_what_is_asked_for() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-unpack-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.zip");
        let names = [
            "pkg/capi/a.dll",
            "pkg/b.txt",
            "data/x/y.bin",
            "bin/tool.exe",
        ];
        let files: Vec<_> = names.iter().map(|n| (*n, n.as_bytes())).collect();
        crate::setup::archive::write(&archive, &files);
        unpack(Os::Windows, &Take::Libs("ort"), &archive, &dir).unwrap();
        unpack(Os::Windows, &Take::Tree("data/", "tree"), &archive, &dir).unwrap();
        unpack(Os::Windows, &Take::Member("bin/tool.exe", "t/tool.exe"), &archive, &dir).unwrap();
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
        assert!(unpack(Os::Windows, &Take::Member("missing", "m/missing"), &archive, &dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn each_card_has_its_own_patched_model() {
        let all = [Model::Kokoro, Model::Kerstin].map(|m| [Card::Nvidia, Card::Other].map(|c| patched_model(m, c)));
        let all: Vec<_> = all.iter().flatten().collect();
        for (i, a) in all.iter().enumerate() {
            assert!(all[i + 1..].iter().all(|b| a.0 != b.0 && a.1 != b.1), "{all:?}");
        }
    }

    #[test]
    fn a_model_patched_by_an_older_patch_is_made_again() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-stale-{}", std::process::id()));
        let (key, rel) = patched_model(Model::Kokoro, Card::Nvidia);
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
    fn older_patches_for_either_card_are_removed() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-older-{}", std::process::id()));
        let v = crate::speech::onnx::VERSION;
        let names = [
            format!("model-gpu-dml-v{}.onnx", v - 1),
            format!("model-gpu-v{}.onnx", v - 1),
            format!("model-gpu-v{v}.onnx"),
            format!("model-gpu-dml-v{v}.onnx"),
            "model.onnx".to_string(),
        ];
        fs::create_dir_all(&dir).unwrap();
        for n in &names {
            fs::write(dir.join(n), b"x").unwrap();
        }
        let mut found = BTreeMap::new();
        found.insert("kokoro-gpu-dml".to_string(), dir.join(&names[0]));
        found.insert("kokoro".to_string(), dir.join(&names[4]));
        remove_older_patches(&mut found, &dir);
        let left: Vec<bool> = names.iter().map(|n| dir.join(n).exists()).collect();
        assert_eq!(left, [false, false, true, true, true]);
        assert_eq!(found.keys().collect::<Vec<_>>(), ["kokoro"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_dlls_nothing_loads_are_not_unpacked() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-unused-{}", std::process::id()));
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
        crate::setup::archive::write(&archive, &files);
        unpack(Os::Windows, &Take::Libs("ort"), &archive, &dir).unwrap();
        assert!(dir.join("ort/onnxruntime.dll").exists());
        for name in unused {
            assert!(!dir.join("ort").join(name).exists(), "{name} unpacked");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_the_x64_dlls_of_a_nuget_package_are_unpacked() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-nupkg-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.nupkg");
        let x64 = &b"x64"[..];
        crate::setup::archive::write(
            &archive,
            &[
                ("runtimes/win-x64/native/onnxruntime.dll", x64),
                ("runtimes/win-arm64/native/onnxruntime.dll", &b"arm"[..]),
                ("runtimes/win-arm64ec/native/onnxruntime.dll", &b"arm"[..]),
                ("lib/net8.0/Microsoft.ML.OnnxRuntime.dll", &b"net"[..]),
            ],
        );
        unpack(Os::Windows, &Take::Libs("ort-dml"), &archive, &dir).unwrap();
        assert_eq!(fs::read(dir.join("ort-dml/onnxruntime.dll")).unwrap(), x64);
        assert!(!dir.join("ort-dml/Microsoft.ML.OnnxRuntime.dll").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn linux_libraries_and_programs_are_taken_with_their_links() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-linux-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let archive = dir.join("a.tgz");
        crate::setup::archive::write_tgz(
            &archive,
            &[
                ("ort-1/lib/libonnxruntime.so.1.31.1", b"o"),
                ("ort-1/lib/libonnxruntime_providers_tensorrt.so", b"t"),
                ("ort-1/include/onnxruntime_c_api.h", b"h"),
                ("llama-1/llama-server", b"l"),
            ],
            &[("ort-1/lib/libonnxruntime.so.1", "libonnxruntime.so.1.31.1")],
        );
        unpack(Os::Linux, &Take::Libs("ort"), &archive, &dir).unwrap();
        unpack(Os::Linux, &Take::Tree("llama-1/", "llama"), &archive, &dir).unwrap();
        assert!(dir.join("ort/libonnxruntime.so.1.31.1").exists());
        assert!(dir.join("ort/libonnxruntime.so.1").exists());
        assert!(!dir.join("ort/libonnxruntime_providers_tensorrt.so").exists());
        assert!(!dir.join("ort/onnxruntime_c_api.h").exists());
        assert!(dir.join("llama/llama-server").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn each_system_takes_its_own_libraries() {
        assert!(native_lib(Os::Linux, "nvidia/cu13/lib/libcudart.so.13"));
        assert!(native_lib(Os::Linux, "nvidia/cudnn/lib/libcudnn.so.9"));
        assert!(!native_lib(Os::Linux, "nvidia/cu13/lib/libcufftw.so.12"));
        assert!(!native_lib(Os::Linux, "nvidia/cu13/bin/cudart64_13.dll"));
        assert!(!native_lib(Os::Windows, "nvidia/cu13/lib/libcudart.so.13"));
        assert!(native_lib(Os::Windows, "nvidia/cu13/bin/x64/cudart64_13.dll"));
    }

    #[test]
    fn kerstin_has_her_own_model_and_voice_and_the_shared_tokenizer() {
        let mut found: BTreeMap<String, PathBuf> = ["kerstin-model", "kerstin-gpu", "kokoro-tokenizer"]
            .map(|k| (k.to_string(), PathBuf::from(format!("/{k}"))))
            .into_iter()
            .collect();
        let ort = Path::new("/ort");
        assert!(kerstin_files(&found, ort, Card::Nvidia).is_err()); // no voice yet
        found.insert("voice-df_kerstin".into(), "/voice".into());
        let k = kerstin_files(&found, ort, Card::Nvidia).unwrap();
        assert_eq!(k.model, Path::new("/kerstin-gpu")); // patched: the silence between words limited
        assert_eq!(k.tokenizer, Path::new("/kokoro-tokenizer"));
        assert_eq!(k.voice, Path::new("/voice"));
        assert!(k.cuda);
    }

    #[test]
    fn kerstin_on_another_card_has_the_model_with_directmls_pads() {
        let keys = ["kerstin-model", "kerstin-gpu", "kerstin-gpu-dml", "kokoro-tokenizer", "voice-df_kerstin"];
        let found: BTreeMap<String, PathBuf> = keys
            .map(|k| (k.to_string(), PathBuf::from(format!("/{k}"))))
            .into_iter()
            .collect();
        let k = kerstin_files(&found, Path::new("/ort"), Card::Other).unwrap();
        assert_eq!(k.model, Path::new("/kerstin-gpu-dml"));
        assert!(!k.cuda);
    }

    #[test]
    fn tools_hand_out_their_pieces_by_what_they_are() {
        let found = ["onnxruntime", "kokoro-gpu", "kokoro-tokenizer", "voice-af_zzz", "espeak-ng-data"]
            .map(|k| (k.to_string(), PathBuf::from(format!("/{k}"))));
        let tools = Tools {
            home: PathBuf::new(),
            card: Card::Nvidia,
            found: found.into_iter().collect(),
            german: OnceCell::new(),
        };
        assert_eq!(tools.espeak().unwrap(), Path::new("/espeak-ng-data"));
        assert_eq!(tools.route(), "CUDA");
        let k = tools.kokoro("af_zzz").unwrap();
        assert!(k.cuda);
        assert_eq!(k.ort, Path::new("/onnxruntime"));
        assert_eq!(k.model, Path::new("/kokoro-gpu"));
        assert_eq!(k.tokenizer, Path::new("/kokoro-tokenizer"));
        assert_eq!(k.voice, Path::new("/voice-af_zzz"));
        let missing = tools.ffmpeg().err().unwrap();
        assert_eq!(missing.to_string(), "ffmpeg is not set up");
        assert!(tools.kokoro("bm_none").is_err());
    }
}
