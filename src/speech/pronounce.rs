//! Kokoro's phonemes of a piece of text as Kokoro's own KPipeline (lang "a") gives them (a German book:
//! german(), eSpeak's alone, as misaki's EspeakG2P, as the German voice was trained), and a text's
//! sounds for the check against Whisper. Our fork of misaki-rs (elana-voss/misaki-rs, by git rev in
//! Cargo.toml) does the work, as misaki's G2P is set up there: US English, eSpeak for words its
//! dictionaries lack, nothing left unknown (unk ""). KPipeline makes one piece of up to 510 phonemes, a
//! chunk's limit (chunks.rs, by length()), so a chunk's phonemes are the whole chunk's.
//! Traps: espeak-ng finds its data by PIPER_ESPEAKNG_DATA_DIRECTORY, which new() sets, so it is made
//! before other threads start; eSpeak is initialised once per process, so its folder cannot change.
use anyhow::{Context, Result, bail};
use misaki_rs::fallback::{FallbackError, espeak_g2p};
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
    /// misaki's, for English; none: German, eSpeak alone
    g2p: Option<G2P>,
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
            g2p: Some(g2p),
            espeak,
            said: Mutex::new(BTreeMap::new()),
        })
    }

    /// The same eSpeak in German: for a book's German voice, its words all eSpeak's.
    pub fn german(&self) -> Pronounce {
        Self {
            g2p: None,
            espeak: self.espeak.clone(),
            said: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn is_german(&self) -> bool {
        self.g2p.is_none()
    }

    /// The phonemes Kokoro speaks for `text`, a word given its own (say_as) said with them.
    pub fn phonemes(&self, text: &str) -> Result<String> {
        self.raw(&self.mark(text))
    }

    /// `text`'s length as Kokoro counts it, as it will be spoken; text without phonemes counts its
    /// characters (about one phoneme each), as chunks() needs a length, not an error.
    pub fn length(&self, text: &str) -> usize {
        self.phonemes(text).map_or(text.chars().count(), |p| p.chars().count())
    }

    /// The same before the book's respellings are settled: what chunks are cut by to ask about them.
    pub fn unsettled_length(&self, text: &str) -> usize {
        self.raw(text).map_or(text.chars().count(), |p| p.chars().count())
    }

    fn mark(&self, text: &str) -> String {
        let mut text = text.to_string();
        for (word, ps) in self.said.lock().expect("no thread panics holding it").iter() {
            text = marked(&text, word, ps);
        }
        text
    }

    fn raw(&self, text: &str) -> Result<String> {
        match &self.g2p {
            Some(g2p) => Ok(g2p
                .g2p(text)
                .with_context(|| format!("no phonemes for {text:?}"))?
                .0),
            None => espeak_g2p(text, "de").with_context(|| format!("no phonemes for {text:?}")),
        }
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

    /// The sounds of text without stress marks, spaces or punctuation.
    pub fn sounds(&self, text: &str) -> Result<String> {
        // Hyphens part words, after a word said as given is marked.
        let p = self.raw(&self.mark(text).replace('-', " "))?;
        Ok(p.chars()
            .filter(|&c| c.is_alphanumeric() && c != 'ˈ' && c != 'ˌ')
            .collect())
    }
}

/// `text` with each whole `word` (no letter nor hyphen beside it: not "Ro" of "Ro-Tan") given
/// `phonemes` by misaki's link: [word](/ps/).
fn marked(text: &str, word: &str, phonemes: &str) -> String {
    if word.is_empty() {
        return text.to_string(); // a hand-edited sounds.json; find("") would never advance
    }
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find(word) {
        let (before, after) = (&rest[..i], &rest[i + word.len()..]);
        let alone = !before.chars().next_back().is_some_and(inner)
            && !after.chars().next().is_some_and(inner);
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

/// A letter or a hyphen: what joins a word to its neighbour ("Ro-Tan" is one word, as misaki reads it).
pub fn inner(c: char) -> bool {
    c.is_alphabetic() || c == '-'
}

/// `text` holds `word` whole, no letter nor hyphen beside it.
pub fn holds(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(i, _)| {
        !text[..i].chars().next_back().is_some_and(inner)
            && !text[i + word.len()..].chars().next().is_some_and(inner)
    })
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
    fn an_empty_word_changes_nothing() {
        assert_eq!(marked("some text", "", "x"), "some text");
    }

    #[test]
    fn a_chunk_stays_in_the_limit_with_its_respellings() {
        P.say_as("Zzyqot", Some("kˈuɹus".repeat(40)));
        let text = "Zzyqot ran on the hill. ".repeat(40);
        let chs = [crate::text::extract::Chapter { title: "T".into(), paragraphs: vec![text] }];
        let cs = crate::speech::chunks::chunks(&chs, &|t| P.length(t), crate::text::language::Language::English);
        let longest = cs.iter().map(|c| P.phonemes(&c.text).unwrap().chars().count()).max();
        P.say_as("Zzyqot", None);
        assert!(longest <= Some(510), "{longest:?}");
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
    fn german_is_espeaks_with_the_voices_symbols() {
        // the program's own eSpeak against the system's 1.52-dev through the plan's table:
        // `espeak-ng -q -v de --ipa --tie=^`
        let got = P.german().phonemes("Die Zeit heißt Euro").unwrap();
        assert_eq!(got, "diː ʦˈIt hˈIst ˈɔøroː");
    }

    #[test]
    fn missing_espeak_data_is_an_error() {
        assert!(Pronounce::new(Path::new("no/such/espeak-ng-data")).is_err());
    }
}
