//! Zip archives read (an EPUB in epub.rs, setup's downloads in setup.rs), the one module naming the zip crate.
//! Trap: each() skips a member whose path would leave the archive's folder.
use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;

pub struct Archive(zip::ZipArchive<std::fs::File>);

impl Archive {
    pub fn open(path: &Path) -> Result<Archive> {
        let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
        let zip = zip::ZipArchive::new(file)
            .with_context(|| format!("{} is no zip archive", path.display()))?;
        Ok(Archive(zip))
    }

    /// A member's text (bytes that are no UTF-8 replaced); an error when there is no such member.
    pub fn text(&mut self, name: &str) -> Result<String> {
        let mut f = self
            .0
            .by_name(name)
            .with_context(|| format!("{name} is missing"))?;
        let mut b = Vec::new();
        f.read_to_end(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }

    /// `f` given each member in order: its name ('/' between folders), whether it is a folder, its bytes.
    pub fn each(
        &mut self,
        mut f: impl FnMut(&str, bool, &mut dyn Read) -> Result<()>,
    ) -> Result<()> {
        for i in 0..self.0.len() {
            let mut entry = self.0.by_index(i)?;
            let Some(name) = entry.enclosed_name() else {
                continue; // a path leaving the folder
            };
            let name = name.to_string_lossy().replace('\\', "/");
            let dir = entry.is_dir();
            f(&name, dir, &mut entry)?;
        }
        Ok(())
    }
}

/// A zip at `path` holding `files` (name, bytes), for tests.
#[cfg(test)]
pub fn write(path: &Path, files: &[(&str, &[u8])]) {
    use std::io::Write;
    let mut z = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, bytes) in files {
        z.start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        z.write_all(bytes).unwrap();
    }
    z.finish().unwrap();
}
