//! First start: what the program needs (downloads.rs, per system and card) is fetched once into its home
//! folder (os::home: %LOCALAPPDATA%\ebook2m4b, on Linux ~/.local/share/ebook2m4b), never system-wide;
//! home/setup.json names where each piece is, so a later start fetches nothing.
//! Each download is checked by its pinned SHA-256 before it is unpacked.
//! Kokoro's model is then patched for the GPU once per patch version (onnx.rs), as "kokoro-gpu" ("kokoro-gpu-dml" on DirectML).
//! Each card's ONNX Runtime has its own folder (ort, ort-dml), so a card changed later fetches its own.
//! Libraries nothing loads (TensorRT's provider, cuFFTW, NVBLAS) are not unpacked.
//! Linux: links in the archives are made again, every file taken may be run, and the libraries' folders
//! are put on LD_LIBRARY_PATH for the programs started later (llama-server, whisper-server).
use crate::downloads::{Item, Take, items};
use crate::gpu::Card;
use crate::os::{Os, THIS};
use crate::archive::Kind;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

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
    crate::os::home(crate::os::THIS, |k| std::env::var(k).ok())
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
    for it in items(THIS, card, voice)? {
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
                let at = unpack(THIS, &it.take, &archive, &home)?;
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
    #[cfg(unix)]
    {
        let old = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let path = library_path(vec![home.join("ort"), home.join("llama")], &old)?;
        // SAFETY: setup runs before this program starts any other thread
        unsafe { std::env::set_var("LD_LIBRARY_PATH", path) };
    }
    Ok(Tools { home, card, found })
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
    crate::archive::Archive::open(archive)?.each(|name, kind, bytes| {
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
        crate::archive::write(&archive, &files);
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
        let (nk, np) = gpu_model(Card::Nvidia);
        let (ok, op) = gpu_model(Card::Other);
        assert!(nk != ok && np != op);
    }

    #[test]
    fn a_model_patched_by_an_older_patch_is_made_again() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-stale-{}", std::process::id()));
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
        crate::archive::write(&archive, &files);
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
        crate::archive::write(
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
        crate::archive::write_tgz(
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
}
