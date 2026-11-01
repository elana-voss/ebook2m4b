//! Kokoro's phonemes of a piece of text as Python's KPipeline (lang "a") gives them today, and words.py's
//! sounds() for the check against Whisper. Our fork of misaki-rs (elana-voss/misaki-rs, by git rev in
//! Cargo.toml) does the work, as misaki's G2P is set up there: US English, eSpeak for words its
//! dictionaries lack, nothing left unknown (unk ""). KPipeline makes one piece of up to 510 phonemes, a
//! chunk's limit (chunks.rs, by unmarked()), so a chunk's phonemes are the whole chunk's.
//! Traps: espeak-ng finds its data by PIPER_ESPEAKNG_DATA_DIRECTORY, which new() sets, so it is made
//! before other threads start; eSpeak is initialised once per process, so its folder cannot change.
use anyhow::{Context, Result, bail};
use misaki_rs::fallback::FallbackError;
use misaki_rs::{EspeakFallback, Fallback, G2P, Language};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The version of the fork's word rules (stems, variants), raised with each change to them: a book's
/// work_id holds it, so a change speaks the books again.
pub const RULES: u32 = 5;

thread_local! {
    /// the words eSpeak was sent on this thread (misaki calls it on the caller's)
    static NOTED: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
}

/// eSpeak, noting each word misaki sends it (the words its dictionaries do not answer).
struct Noted(Arc<EspeakFallback>);

impl Fallback for Noted {
    fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
        NOTED.with_borrow_mut(|n| n.insert(word.to_string()));
        self.0.phonemize(word)
    }
}

pub struct Pronounce {
    g2p: G2P,
    espeak: Arc<EspeakFallback>,
    /// word: the phonemes it is said with instead of misaki's (sounds.rs's respellings)
    said: Mutex<BTreeMap<String, String>>,
}

impl Pronounce {
    /// `data` is the espeak-ng-data folder (setup's piece "espeak-ng-data").
    pub fn new(data: &Path) -> Result<Self> {
        if !data.join("phontab").is_file() {
            bail!("eSpeak's data is missing at {}", data.display());
        }
        let parent = data
            .parent()
            .context("eSpeak's data folder has no parent")?;
        // SAFETY: called at start, before any other thread reads the environment
        unsafe { std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", parent) };
        let espeak = Arc::new(EspeakFallback::new(false).map_err(anyhow::Error::msg)?);
        let fallback = Noted(espeak.clone());
        let mut g2p = G2P::with_fallback(Language::EnglishUS, Some(Box::new(fallback)));
        g2p.unk = String::new();
        Ok(Self {
            g2p,
            espeak,
            said: Mutex::new(BTreeMap::new()),
        })
    }

    /// The phonemes Kokoro speaks for `text`, a word given its own (say_as) said with them.
    pub fn phonemes(&self, text: &str) -> Result<String> {
        self.raw(&self.mark(text))
    }

    /// The phonemes of `text` without the words given their own (say_as).
    pub fn unmarked(&self, text: &str) -> Result<String> {
        self.raw(text)
    }

    fn mark(&self, text: &str) -> String {
        let mut text = text.to_string();
        for (word, ps) in self.said.lock().expect("no thread panics holding it").iter() {
            text = marked(&text, word, ps);
        }
        text
    }

    fn raw(&self, text: &str) -> Result<String> {
        Ok(self
            .g2p
            .g2p(text)
            .with_context(|| format!("no phonemes for {text:?}"))?
            .0)
    }

    /// The words of `text` misaki's dictionaries do not answer, so eSpeak speaks them (say_as ignored);
    /// "Ivok's", tried whole first, is "Ivok".
    pub fn espeak_words(&self, text: &str) -> Result<BTreeSet<String>> {
        NOTED.with_borrow_mut(BTreeSet::clear);
        self.raw(text)?;
        Ok(NOTED.with_borrow_mut(std::mem::take)
            .into_iter()
            .map(|w| {
                let w = w.strip_suffix("'s").or(w.strip_suffix("’s")).unwrap_or(&w);
                w.to_string()
            })
            .collect())
    }

    /// eSpeak's phonemes for an English respelling ("CHOO-eer"), read as one word in lower case.
    pub fn respelt(&self, respelling: &str) -> Result<String> {
        let w: String = respelling
            .chars()
            .filter(|c| c.is_alphabetic())
            .flat_map(char::to_lowercase)
            .collect();
        self.espeak
            .phonemize(&w)
            .map_err(|e| anyhow::anyhow!("eSpeak cannot say {respelling:?}: {e}"))
    }

    /// `word` said with `phonemes` from now on (none: misaki's again).
    pub fn say_as(&self, word: &str, phonemes: Option<String>) {
        let mut said = self.said.lock().expect("no thread panics holding it");
        match phonemes {
            Some(ps) => said.insert(word.to_string(), ps),
            None => said.remove(word),
        };
    }

    /// The sounds of text without stress marks, spaces or punctuation (words.py's sounds()).
    pub fn sounds(&self, text: &str) -> Result<String> {
        // Hyphens part words as words.py's sounds() does, after a word said as given is marked.
        let p = self.raw(&self.mark(text).replace('-', " "))?;
        Ok(p.chars()
            .filter(|&c| c.is_alphanumeric() && c != 'ˈ' && c != 'ˌ')
            .collect())
    }
}

/// `text` with each whole `word` (no letter nor hyphen beside it: not "Ro" of "Ro-Tan") given
/// `phonemes` by misaki's link: [word](/ps/).
fn marked(text: &str, word: &str, phonemes: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find(word) {
        let (before, after) = (&rest[..i], &rest[i + word.len()..]);
        let alone = !before.chars().next_back().is_some_and(crate::sounds::inner)
            && !after.chars().next().is_some_and(crate::sounds::inner);
        out += before;
        if alone {
            // misaki takes no hyphen in a link's text: "[Zo-ri-ba](/ps/)" would be eSpeak's
            out += &format!("[{}](/{phonemes}/)", word.replace('-', ""));
        } else {
            out += word;
        }
        rest = after;
    }
    out + rest
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::LazyLock;

    pub(crate) static P: LazyLock<Pronounce> = LazyLock::new(|| {
        Pronounce::new(&crate::setup::home().unwrap().join("espeak-ng-data")).unwrap()
    });

    #[test]
    fn phonemes_are_kpipelines() {
        // KPipeline's output; "Zo-ri-ba", "Vasena" and "Ivok's", eSpeak's, are the program's
        let cases = [
            (
                "“I ask your favour,” she said, “for the 3 barls of Zo-ri-ba.”",
                "“ˌI ˈæsk jʊɹ fˈAvəɹ,” ʃi sˈɛd, “fɔɹ ðə θɹˈi bˈɑɹlz ʌv zˈOɹˈIbˈɑ.”",
            ),
            (
                "Vasena, the Ivok's daughter, smiled.",
                "vˈAsnə, ði ˈIvɑks dˈɔTəɹ, smˈIld.",
            ),
        ];
        for (text, want) in cases {
            assert_eq!(P.phonemes(text).unwrap(), want);
        }
    }

    #[test]
    fn sounds_are_words_pys() {
        assert_eq!(
            P.sounds("The Ivok's well-known daughter.").unwrap(),
            "ðiIvɑkswɛlnOndɔTəɹ"
        );
    }

    #[test]
    fn a_word_is_said_as_given_and_only_whole() {
        P.say_as("Zzyqat", Some("kˈuɹus".into()));
        let got = P.phonemes("Zzyqat smiled at Zzyqatin.").unwrap();
        P.say_as("Zzyqat", None);
        assert!(got.starts_with("kˈuɹus smˈIld"), "{got}");
        assert!(!got[6..].contains("kˈuɹus"), "{got}");
        assert!(!P.phonemes("Zzyqat").unwrap().contains("kˈuɹus"));
        P.say_as("Zzyqat", Some("kˈuɹus".into()));
        let joined = P.phonemes("Zzyqat-Var ran.").unwrap();
        P.say_as("Zzyqat", None);
        assert!(!joined.contains("kˈuɹus"), "{joined}");
        P.say_as("Zzy-qat", Some("kˈuɹus".into()));
        let heard = P.sounds("Zzy-qat ran.").unwrap() + "|" + &P.phonemes("Zzy-qat ran.").unwrap();
        P.say_as("Zzy-qat", None);
        assert!(heard.starts_with("kuɹusɹæn|kˈuɹus"), "{heard}");
    }

    #[test]
    fn the_words_espeak_speaks_are_noted() {
        let got = P.espeak_words("Vasena, the Ivok's daughter, smiled.").unwrap();
        assert_eq!(got.into_iter().collect::<Vec<_>>(), ["Ivok", "Vasena"]);
    }

    #[test]
    fn dashes_and_contractions_reach_no_espeak() {
        let text = "The moral is—‘Be good.’ Fetch it, lad!—Here. You can't—I won’t; she couldn’t.";
        let got = P.espeak_words(text).unwrap();
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn a_respelling_is_espeaks() {
        assert_eq!(P.respelt("CHOO-eer").unwrap(), P.respelt("chooeer").unwrap());
        assert!(P.respelt("CHOO-eer").unwrap().starts_with('ʧ'));
    }

    #[test]
    fn missing_espeak_data_is_an_error() {
        assert!(Pronounce::new(Path::new("no/such/espeak-ng-data")).is_err());
    }
}
