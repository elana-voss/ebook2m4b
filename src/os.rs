//! The two systems the program is built for, Windows and Linux (macOS not), and what differs between them
//! by name alone: the home folder, a program's file name, ONNX Runtime's library. Each is a function of the
//! system, so both are tested on either; THIS is the one built for.
use anyhow::{Context, Result};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Os {
    Windows,
    Linux,
}

pub const THIS: Os = if cfg!(windows) { Os::Windows } else { Os::Linux };

/// The home folder: %LOCALAPPDATA%\ebook2m4b on Windows; on Linux $XDG_DATA_HOME/ebook2m4b, else
/// ~/.local/share/ebook2m4b. `var` reads an environment variable (one set but empty counts as unset).
pub fn home(os: Os, var: impl Fn(&str) -> Option<String>) -> Result<PathBuf> {
    let var = |k: &str| var(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = match os {
        Os::Windows => var("LOCALAPPDATA").context("LOCALAPPDATA is not set")?,
        Os::Linux => match var("XDG_DATA_HOME") {
            Some(d) => d,
            None => var("HOME")
                .context("neither XDG_DATA_HOME nor HOME is set")?
                .join(".local/share"),
        },
    };
    Ok(base.join("ebook2m4b"))
}

/// A program's file name: `name`.exe on Windows, `name` on Linux.
pub fn exe(os: Os, name: &str) -> String {
    match os {
        Os::Windows => format!("{name}.exe"),
        Os::Linux => name.to_string(),
    }
}

/// ONNX Runtime's library as setup unpacks it.
pub fn onnxruntime(os: Os) -> &'static str {
    match os {
        Os::Windows => "onnxruntime.dll",
        Os::Linux => "libonnxruntime.so.1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn each_system_has_its_home_folder() {
        let w = home(Os::Windows, env(&[("LOCALAPPDATA", r"C:\L")])).unwrap();
        assert_eq!(w, PathBuf::from(r"C:\L").join("ebook2m4b"));
        let x = home(Os::Linux, env(&[("XDG_DATA_HOME", "/d"), ("HOME", "/h")])).unwrap();
        assert_eq!(x, PathBuf::from("/d").join("ebook2m4b"));
        let h = home(Os::Linux, env(&[("XDG_DATA_HOME", ""), ("HOME", "/h")])).unwrap();
        assert_eq!(h, PathBuf::from("/h").join(".local/share").join("ebook2m4b"));
        assert!(home(Os::Linux, env(&[])).is_err());
        assert!(home(Os::Windows, env(&[("HOME", "/h")])).is_err());
    }

    #[test]
    fn programs_and_libraries_are_named_per_system() {
        assert_eq!(exe(Os::Windows, "llama-server"), "llama-server.exe");
        assert_eq!(exe(Os::Linux, "llama-server"), "llama-server");
        assert_ne!(onnxruntime(Os::Windows), onnxruntime(Os::Linux));
    }
}
