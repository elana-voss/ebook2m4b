//! What the rules cannot decide with confidence, asked of a language model (llm.rs; judge.py's port):
//! the intended word for a misprint, whether a piece of a book is read aloud, whether a line is a
//! chapter heading, an invented word's English respelling (sounds.rs). Questions go in batches of 20; an item the model leaves out keeps the rules'
//! choice, and extract.rs and misprints.rs check each answer before using it.
//! Trap: the prompts, the items' JSON and the schemas are judge.py's to the byte, so the cache's keys
//! stay the same (llm.rs's head); a change here asks every question again.
use crate::extract::Judge;
use crate::llm::Llm;
use anyhow::Result;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::HashMap;

const BATCH: usize = 20;

const MISPRINT: &str = "You proofread an English novel that was scanned with OCR. Each item is a word that is \
not in the dictionary, with the text around it, and maybe a guess. Answer with what the author \
wrote: the word spelt right, or two words if two were run together. If the word is right as it \
stands (a name, a word the author invented, a rare, foreign or dialect word), answer it unchanged. \
Words this book uses that are right as they stand: ";

const READ: &str = "You prepare an ebook for an audiobook. Each numbered item is a piece of the ebook: its \
title and how it starts. Say whether it is read aloud. Read: the story and the author's own parts \
(prologue, foreword, introduction, dedication, epilogue, afterword, appendix). Not read: title page, \
copyright, contents, praise and blurbs, lists of other books, publisher's adverts, previews of \
other books, notes by whoever scanned or converted the ebook.";

const HEADING: &str = "You prepare an ebook for an audiobook. Each numbered item is a short line from the \
book, with the start of the text after it. Say whether the line is a heading that starts a new \
chapter or part, or a sentence of the text itself.";

/// The answer's schema: {"answers": [item, ...]}, every field of the item required.
fn answers(item: Value) -> Value {
    let required: Vec<&String> = item
        .as_object()
        .map(|m| m.keys().collect())
        .unwrap_or_default();
    json!({"type": "object", "additionalProperties": false, "required": ["answers"],
        "properties": {"answers": {"type": "array", "items": {
            "type": "object", "additionalProperties": false, "required": required,
            "properties": item}}}})
}

const RESPELL: &str = "You prepare an English novel for an audiobook. Each item is a word no dictionary has (a name or a word the author invented), with the text around it. Answer how an English reader says it, as an English respelling: its syllables in plain English letters, joined by hyphens, the stressed one in capitals, e.g. Chooir: CHOO-eer, Vasena: vuh-SAY-nuh.";

pub struct ModelJudge(pub RefCell<Llm>);

impl ModelJudge {
    /// An English respelling ("CHOO-eer") of each (word, passage); a word the model leaves out has none
    /// (sounds.rs).
    pub fn respell(&self, items: &[(String, String)]) -> Result<HashMap<String, String>> {
        let asked = items
            .iter()
            .map(|(w, t)| json!({"word": w, "text": t}))
            .collect();
        let got = self.batches(
            RESPELL,
            asked,
            json!({"word": {"type": "string"}, "say": {"type": "string"}}),
        )?;
        Ok(got
            .iter()
            .filter_map(|a| Some((a["word"].as_str()?.to_string(), a["say"].as_str()?.to_string())))
            .collect())
    }

    fn batches(&self, system: &str, items: Vec<Value>, item: Value) -> Result<Vec<Value>> {
        let schema = answers(item);
        let mut out = Vec::new();
        for batch in items.chunks(BATCH) {
            let user = Value::Array(batch.to_vec()).to_string();
            let got = self.0.borrow_mut().ask(system, &user, &schema)?;
            out.extend(got["answers"].as_array().cloned().unwrap_or_default());
        }
        Ok(out)
    }

    /// For each numbered item, its answer's `field`; true where the model is silent.
    fn yes_no(
        &self,
        system: &str,
        items: &[(String, String)],
        keys: [&str; 2],
        field: &str,
    ) -> Result<Vec<bool>> {
        let asked = items
            .iter()
            .enumerate()
            .map(|(n, (a, b))| json!({"n": n, keys[0]: a, keys[1]: b}))
            .collect();
        let got = self.batches(
            system,
            asked,
            json!({"n": {"type": "integer"}, field: {"type": "boolean"}}),
        )?;
        let no: Vec<u64> = got
            .iter()
            .filter(|a| a[field] == false)
            .filter_map(|a| a["n"].as_u64())
            .collect();
        Ok((0..items.len() as u64).map(|n| !no.contains(&n)).collect())
    }
}

impl Judge for ModelJudge {
    fn headings(&self, items: &[(String, String)]) -> Result<Vec<bool>> {
        self.yes_no(HEADING, items, ["line", "after"], "heading")
    }

    fn read(&self, items: &[(String, String)]) -> Result<Vec<bool>> {
        self.yes_no(READ, items, ["title", "start"], "read")
    }

    fn misprints(
        &self,
        cases: &[(String, String, Option<String>)],
        keep: &[String],
    ) -> Result<HashMap<String, String>> {
        let items = cases
            .iter()
            .map(|(w, t, g)| match g.as_deref().filter(|g| !g.is_empty()) {
                Some(g) => json!({"word": w, "text": t, "guess": g}),
                None => json!({"word": w, "text": t}),
            })
            .collect();
        let keep = if keep.is_empty() {
            "none".into()
        } else {
            keep.join(", ")
        };
        let got = self.batches(
            &format!("{MISPRINT}{keep}."),
            items,
            json!({"word": {"type": "string"}, "fix": {"type": "string"}}),
        )?;
        Ok(got
            .iter()
            .filter_map(|a| {
                Some((
                    a["word"].as_str()?.to_string(),
                    a["fix"].as_str()?.to_string(),
                ))
            })
            .collect())
    }

    fn close(&self) {
        self.0.borrow_mut().close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Place, fake};

    fn judge(url: String, name: &str) -> (ModelJudge, std::path::PathBuf) {
        let cache =
            std::env::temp_dir().join(format!("t2a-judge-{name}-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&cache); // a leftover of an earlier run
        let place = Place::Remote {
            url,
            model: "m".into(),
            key_env: "PATH".into(),
        };
        (
            ModelJudge(RefCell::new(Llm::new(place, cache.clone()).unwrap())),
            cache,
        )
    }

    #[test]
    fn questions_go_in_twenties_and_silence_keeps_the_rules_choice() {
        // the model says no to item 3 and 21 and answers nothing about 22
        let (url, seen) = fake::server(|body| {
            let items: Vec<Value> =
                serde_json::from_str(body["messages"][1]["content"].as_str().unwrap()).unwrap();
            let answers: Vec<Value> = items
                .iter()
                .filter(|i| i["n"] != 22)
                .map(|i| json!({"n": i["n"], "read": i["n"] != 3 && i["n"] != 21}))
                .collect();
            json!({ "answers": answers })
        });
        let (j, cache) = judge(url, "read");
        let items: Vec<(String, String)> = (0..25).map(|n| (format!("T{n}"), "s".into())).collect();
        let read = j.read(&items).unwrap();
        assert_eq!(read.iter().filter(|r| !**r).count(), 2);
        assert!(!read[3] && !read[21] && read[22]);
        assert_eq!(seen.lock().unwrap().len(), 2);
        std::fs::remove_file(cache).unwrap();
    }

    #[test]
    fn a_word_is_respelt_and_one_left_out_has_none() {
        let (url, _) = fake::server(|_| json!({"answers": [{"word": "Chooir", "say": "CHOO-eer"}]}));
        let (j, cache) = judge(url, "respell");
        let got = j
            .respell(&[
                ("Chooir".into(), "Chooir came.".into()),
                ("Zarl".into(), "Zarl went.".into()),
            ])
            .unwrap();
        assert_eq!(got, HashMap::from([("Chooir".to_string(), "CHOO-eer".to_string())]));
        std::fs::remove_file(cache).unwrap();
    }

    #[test]
    fn the_questions_stay_the_same_to_the_byte() {
        // the cache keys of the questions, model "m"
        let (url, _) = fake::server(|_| json!({"answers": [{"word": "thier", "fix": "their"}]}));
        let (j, cache) = judge(url, "keys");
        let got = j
            .misprints(
                &[
                    ("thier".into(), "said thier café — “so”".into(), None),
                    ("barl".into(), "a barl".into(), Some("bard".into())),
                ],
                &["zarl".into()],
            )
            .unwrap();
        assert_eq!(got["thier"], "their");
        j.read(&[("Title".into(), "start".into())]).unwrap();
        j.headings(&[("Chapter One".into(), "It was".into())])
            .unwrap();
        let kept: serde_json::Map<String, Value> =
            serde_json::from_str(&std::fs::read_to_string(&cache).unwrap()).unwrap();
        let keys: Vec<&str> = kept.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "d3d0a0b1e0615bd50c4c6d133c5104e4e95ed44c3deb1d7ae77de75dcf50f200",
                "eeaa4cf2406bab33ede9c4757a62eb99541aacd643a4adb6627e056ff38d2007",
                "5f1e66b6f0201ca6697b3e8b21b1172093dee5f38404020e8ef431f015cbf57f"
            ]
        );
        std::fs::remove_file(cache).unwrap();
    }
}
