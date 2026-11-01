//! Archives read: zip (an EPUB in epub.rs, a DOCX, setup's wheels and zips) and gzipped tar (setup's Linux
//! downloads), told apart by their first bytes; the one module naming the zip, tar and flate2 crates.
//! Trap: each() skips a member whose path would leave the archive's folder, and a tar's hard link,
//! and a link whose target is not a file in its own folder.
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub enum Archive {
    Zip(zip::ZipArchive<File>),
    TarGz(PathBuf),
}

/// What a member is: a file, a folder, or a symbolic link to the path it holds.
#[derive(Debug, PartialEq)]
pub enum Kind {
    File,
    Dir,
    Link(String),
}

impl Archive {
    pub fn open(path: &Path) -> Result<Archive> {
        let mut file = File::open(path).with_context(|| path.display().to_string())?;
        let mut magic = [0u8; 2];
        let gz = file.read_exact(&mut magic).is_ok() && magic == [0x1f, 0x8b];
        if gz {
            return Ok(Archive::TarGz(path.to_path_buf()));
        }
        let file = File::open(path).with_context(|| path.display().to_string())?;
        let zip = zip::ZipArchive::new(file)
            .with_context(|| format!("{} is no zip archive", path.display()))?;
        Ok(Archive::Zip(zip))
    }

    /// A zip member's text (bytes that are no UTF-8 replaced); an error when there is no such member.
    pub fn text(&mut self, name: &str) -> Result<String> {
        let Archive::Zip(zip) = self else {
            bail!("{name}: not a zip archive");
        };
        let mut f = zip
            .by_name(name)
            .with_context(|| format!("{name} is missing"))?;
        let mut b = Vec::new();
        f.read_to_end(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }

    /// `f` given each member in order: its name ('/' between folders, no leading "./"), its kind, its bytes.
    pub fn each(&mut self, mut f: impl FnMut(&str, Kind, &mut dyn Read) -> Result<()>) -> Result<()> {
        match self {
            Archive::Zip(zip) => {
                for i in 0..zip.len() {
                    let mut entry = zip.by_index(i)?;
                    let Some(name) = entry.enclosed_name() else {
                        continue; // a path leaving the folder
                    };
                    let name = name.to_string_lossy().replace('\\', "/");
                    let kind = if entry.is_dir() { Kind::Dir } else { Kind::File };
                    f(&name, kind, &mut entry)?;
                }
            }
            Archive::TarGz(path) => {
                let file = File::open(&*path).with_context(|| path.display().to_string())?;
                let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
                for entry in tar.entries()? {
                    let mut entry = entry?;
                    let Some(name) = enclosed(&entry.path()?) else {
                        continue; // a path leaving the folder
                    };
                    let kind = match entry.header().entry_type() {
                        tar::EntryType::Directory => Kind::Dir,
                        tar::EntryType::Symlink => match entry.link_name()?.as_deref().and_then(beside) {
                            Some(to) => Kind::Link(to),
                            None => continue, // a link to nothing or out of its folder
                        },
                        tar::EntryType::Regular | tar::EntryType::Continuous => Kind::File,
                        _ => continue, // hard links, devices: none in the downloads
                    };
                    f(&name, kind, &mut entry)?;
                }
            }
        }
        Ok(())
    }
}

/// A link's target when it is a file in the link's own folder (the downloads' links are such), else None.
fn beside(to: &Path) -> Option<String> {
    let mut parts = to.components();
    match (parts.next(), parts.next()) {
        (Some(Component::Normal(name)), None) => Some(name.to_string_lossy().into_owned()),
        _ => None,
    }
}

/// A tar member's path as '/'-joined name, None when it is absolute or climbs out with "..".
fn enclosed(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for c in path.components() {
        match c {
            Component::Normal(p) => parts.push(p.to_string_lossy().into_owned()),
            Component::CurDir => {}
            _ => return None,
        }
    }
    let name = parts.join("/");
    (!name.is_empty()).then_some(name)
}

/// A zip at `path` holding `files` (name, bytes), for tests.
#[cfg(test)]
pub fn write(path: &Path, files: &[(&str, &[u8])]) {
    use std::io::Write;
    let mut z = zip::ZipWriter::new(File::create(path).unwrap());
    for (name, bytes) in files {
        z.start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        z.write_all(bytes).unwrap();
    }
    z.finish().unwrap();
}

/// A gzipped tar at `path` holding `files` (name, bytes) and `links` (name, target), for tests.
#[cfg(test)]
pub fn write_tgz(path: &Path, files: &[(&str, &[u8])], links: &[(&str, &str)]) {
    let gz = flate2::write::GzEncoder::new(File::create(path).unwrap(), Default::default());
    let mut t = tar::Builder::new(gz);
    for (name, bytes) in files {
        let mut h = tar::Header::new_gnu();
        h.set_size(bytes.len() as u64);
        h.set_mode(0o755);
        t.append_data(&mut h, name, *bytes).unwrap();
    }
    for (name, to) in links {
        let mut h = tar::Header::new_gnu();
        h.set_entry_type(tar::EntryType::Symlink);
        h.set_size(0);
        t.append_link(&mut h, name, to).unwrap();
    }
    t.into_inner().unwrap().finish().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gzipped_tar_gives_its_files_and_links() {
        let dir = std::env::temp_dir().join(format!("t2a-tgz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.tgz");
        write_tgz(
            &path,
            &[("./top/lib/libx.so.1.2", b"x"), ("top/bin/tool", b"t")],
            &[("top/lib/libx.so.1", "libx.so.1.2")],
        );
        let mut got = Vec::new();
        Archive::open(&path)
            .unwrap()
            .each(|name, kind, bytes| {
                let mut b = String::new();
                bytes.read_to_string(&mut b)?;
                got.push((name.to_string(), kind, b));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            got,
            [
                ("top/lib/libx.so.1.2".into(), Kind::File, "x".into()),
                ("top/bin/tool".into(), Kind::File, "t".into()),
                ("top/lib/libx.so.1".into(), Kind::Link("libx.so.1.2".into()), String::new()),
            ]
        );
        assert_eq!(enclosed(Path::new("../x")), None);
        assert_eq!(beside(Path::new("libx.so.1.2")).as_deref(), Some("libx.so.1.2"));
        assert_eq!(beside(Path::new("../lib64/x.so")), None);
        assert_eq!(beside(Path::new("/usr/lib/x.so")), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
