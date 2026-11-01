//! The layer rules a search can check (section 5 of the organising work): reads `src/` as text and
//! fails the build on any breach, each printed with its file and line. Comment lines do not count.
//!
//! - A library is named only in the module that wraps it. `anyhow`, `regex` and `serde_json` are the
//!   program's vocabulary (errors, patterns and JSON values that cross every layer), not wrapped.
//! - A layer uses only the layers below it: machine, setup, book, model, text, speech, run, main.
//! - The readers are `book`'s own: outside it only `book::read` and the shared types are named.
//! - Printing is `main`'s and `run`'s, with setup's, the heat wait's and the spoken progress.
//! - `main` touches no file.
use regex::Regex;
use std::path::Path;

const LAYERS: [&str; 8] = ["machine", "setup", "book", "model", "text", "speech", "run", "main"];

/// Each library and the files that may name it.
const LIBRARIES: [(&str, &[&str]); 13] = [
    ("clap", &["main.rs"]),
    ("flate2", &["setup/archive.rs"]),
    ("html_escape", &["book/soup.rs"]),
    ("libloading", &["speech/kokoro.rs"]),
    ("misaki_rs", &["speech/pronounce.rs"]),
    ("ort", &["speech/kokoro.rs"]),
    ("percent_encoding", &["book/epub.rs"]),
    ("roxmltree", &["book/mod.rs", "book/epub.rs"]),
    ("sha2", &["setup/hash.rs"]),
    ("tar", &["setup/archive.rs"]),
    ("ureq", &["setup/http.rs"]),
    ("whisper_rs", &["speech/ear.rs"]),
    ("zip", &["setup/archive.rs"]),
];

/// A file and a text its code never holds.
const NEVER: [(&str, &str); 8] = [
    ("main.rs", "std::fs"),
    ("speech/speak.rs", "crate::text"),
    ("speech/assemble.rs", "crate::speech::speak"),
    ("speech/assemble.rs", "crate::speech::kokoro"),
    ("speech/whisper.rs", "crate::speech::kokoro"),
    ("speech/pronounce.rs", "crate::speech::sounds"),
    ("book/text.rs", "crate::book::mobi"),
    ("text/extract.rs", "crate::model::llm"),
];

/// The one crossing upward: setup patches Kokoro's model as it fetches it (speech/onnx.rs).
const UPWARD: [(&str, &str); 1] = [("setup/mod.rs", "speech::onnx")];

/// The files that may print.
const PRINTS: [&str; 5] = ["main.rs", "run.rs", "machine.rs", "setup/mod.rs", "speech/speak.rs"];

/// Each file under `dir`, its path from `src/` with forward slashes, and its code lines with numbers.
fn sources(dir: &Path, from: &Path, out: &mut Vec<(String, Vec<(usize, String)>)>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            sources(&p, from, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let name = p.strip_prefix(from).unwrap().to_string_lossy().replace('\\', "/");
            let code = std::fs::read_to_string(&p)
                .unwrap()
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim_start().starts_with("//"))
                .map(|(i, l)| (i + 1, l.split(" // ").next().unwrap_or("").to_string()))
                .collect();
            out.push((name, code));
        }
    }
}

/// The layer a file under `src/` belongs to: its first folder, or its stem.
fn layer(file: &str) -> &str {
    file.split('/').next().unwrap().trim_end_matches(".rs")
}

#[test]
fn the_layer_rules_hold() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&src, &src, &mut files);
    let mut wrong = Vec::new();
    let rank = |l: &str| LAYERS.iter().position(|x| *x == l);
    for (file, _) in &files {
        if rank(layer(file)).is_none() {
            wrong.push(format!("{file}: in no layer"));
        }
    }
    for (_, allowed) in LIBRARIES {
        for f in allowed {
            if !src.join(f).exists() {
                wrong.push(format!("{f}: the rules name it, it is not there"));
            }
        }
    }
    let up = Regex::new(r"\bcrate::(\w+)(?:::(\w+))?").unwrap();
    let reader =
        Regex::new(r"\bbook::(epub|mobi|kf8|soup|htmlbook|txt|html|fb2|docx|text)\b").unwrap();
    let print = Regex::new(r"\b(e?print(ln)?)!").unwrap();
    let libraries: Vec<(&str, &[&str], Regex)> = LIBRARIES
        .iter()
        .map(|(lib, allowed)| (*lib, *allowed, Regex::new(&format!(r"\b{lib}::")).unwrap()))
        .collect();
    for (file, code) in &files {
        let own = rank(layer(file)).unwrap_or(0);
        for (n, line) in code {
            for (lib, allowed, named) in &libraries {
                if named.is_match(line) && !allowed.contains(&file.as_str()) {
                    wrong.push(format!("{file}:{n}: names {lib}"));
                }
            }
            for c in up.captures_iter(line) {
                let path = format!("{}::{}", &c[1], c.get(2).map_or("", |m| m.as_str()));
                let above = rank(&c[1]).is_none_or(|r| r > own);
                if above && !UPWARD.contains(&(file.as_str(), path.as_str())) {
                    wrong.push(format!("{file}:{n}: reaches up to crate::{path}"));
                }
            }
            if layer(file) != "book" && reader.is_match(line) {
                wrong.push(format!("{file}:{n}: names a reader"));
            }
            if print.is_match(line) && !PRINTS.contains(&file.as_str()) {
                wrong.push(format!("{file}:{n}: prints"));
            }
            for (f, text) in NEVER {
                if f == file && line.contains(text) {
                    wrong.push(format!("{file}:{n}: holds {text}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "layer rules broken:\n{}", wrong.join("\n"));
}
