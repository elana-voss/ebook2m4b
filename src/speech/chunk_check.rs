//! The hidden comparison of chunks::chunks with each joined text counted whole, on real books (chapter
//! headers and the program's own length, no language model). Not run by default:
//! CHECK_BOOKS=<folder> CHECK_OUT=<file> CHECK_MODE=counted|sum cargo test --release chunk_check -- --ignored
//! writes one line per book: hashes of the chunks before and after the respellings and of the words found,
//! chunks over LIMIT, the characters length was given over the book's, seconds. counted: 4 books at a
//! time, one thread each, and the largest difference between a sum and its true count on a last line.
//! sum: one book at a time, on the threads of chunks().
use crate::machine::in_order;
use crate::setup::hash::json_key;
use crate::speech::chunks::{self, Chunk, DRIFT, LIMIT};
use crate::speech::pronounce::Pronounce;
use crate::speech::sounds;
use crate::text::{extract, language::Language};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn key(cs: &[Chunk]) -> String {
    let all: Vec<_> = cs
        .iter()
        .map(|c| serde_json::json!([c.chapter, c.text, c.pause, c.cut]))
        .collect();
    json_key(&serde_json::Value::Array(all))
}

fn one(path: &Path, counted: bool, store: &sounds::Store) -> String {
    let book = crate::book::read(path).unwrap();
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    let Some(language) = extract::language(&book) else {
        return format!("{name}\tno language");
    };
    let mut notes = Vec::new();
    let mut all = vec![extract::opening(&book, &name, language)];
    all.extend(extract::chapters(&book, None, &HashSet::new(), &mut notes, language).unwrap());
    let size: usize = all.iter().flat_map(|c| &c.paragraphs).map(|p| p.chars().count()).sum();
    let english = Pronounce::new(&crate::setup::home().unwrap().join("espeak-ng-data")).unwrap();
    let german = english.german();
    let given = AtomicUsize::new(0);
    let cut = |len: &(dyn Fn(&str) -> usize + Sync), l: Language| {
        let counting = |t: &str| {
            given.fetch_add(t.chars().count(), Ordering::Relaxed);
            len(t)
        };
        if counted {
            chunks::chunks_counted(&all, &counting, l)
        } else {
            chunks::chunks(&all, &counting, l)
        }
    };
    let start = Instant::now();
    let (first, words, p) = match language {
        Language::English => {
            let first = cut(&|t| english.unsettled_length(t), Language::English);
            let (known, invented) = sounds::found(&first, &english).unwrap();
            sounds::apply(&invented, &english, store).unwrap();
            let words = json_key(&serde_json::json!([known, invented]));
            (key(&first), words, &english)
        }
        Language::German => (String::new(), String::new(), &german),
    };
    let last = cut(&|t| p.length(t), language);
    let seconds = start.elapsed().as_secs_f32();
    let over = last.iter().filter(|c| p.length(&c.text) > LIMIT).count();
    let times = given.load(Ordering::Relaxed) as f32 / size.max(1) as f32;
    format!("{name}\t{first}\t{}\t{words}\t{} chunks\t{over} over\t{times:.2} times\t{seconds:.1} s", key(&last), last.len())
}

#[test]
#[ignore = "reads the books named by CHECK_BOOKS"]
fn chunk_check() {
    let folder = std::env::var("CHECK_BOOKS").unwrap();
    let counted = std::env::var("CHECK_MODE").unwrap() == "counted";
    let mut books: Vec<PathBuf> = std::fs::read_dir(folder).unwrap().map(|e| e.unwrap().path()).collect();
    books.retain(|p| p.is_file());
    books.sort();
    let store = sounds::load(Path::new(sounds::FILE)).unwrap();
    let mut lines = if counted {
        in_order(&books, |_, b| one(b, true, &store))
    } else {
        books.iter().map(|b| one(b, false, &store)).collect()
    };
    if counted {
        lines.push(format!("largest difference {}", DRIFT.load(Ordering::Relaxed)));
    }
    std::fs::write(std::env::var("CHECK_OUT").unwrap(), lines.join("\n") + "\n").unwrap();
}
