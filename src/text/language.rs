//! Which language a book is in: its text's very common words are counted, the larger count decides, and a
//! book where neither list makes up a share of the words a language's own text reaches is in neither.
//! The lists leave out what both languages write often (in, an, so, man, des, on).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    English,
    German,
}

impl Language {
    /// The code Whisper and eSpeak know it by.
    pub fn code(self) -> &'static str {
        match self {
            Language::English => "en",
            Language::German => "de",
        }
    }
}

const ENGLISH: [&str; 15] = [
    "the", "and", "of", "to", "was", "that", "he", "it", "his", "with", "had", "for", "but", "not", "she",
];
const GERMAN: [&str; 15] = [
    "und", "der", "die", "das", "nicht", "ist", "ich", "sie", "er", "es", "ein", "zu", "den", "mit", "von",
];

/// The share of a book's words its language's list must reach, in percent.
const SHARE: usize = 8;

/// The language of these pieces of text; None when they are in neither.
pub fn detect<'a>(pieces: impl Iterator<Item = &'a str>) -> Option<Language> {
    let (mut english, mut german, mut words) = (0, 0, 0);
    for word in pieces.flat_map(|p| p.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty())) {
        let word = word.to_lowercase();
        words += 1;
        english += usize::from(ENGLISH.contains(&word.as_str()));
        german += usize::from(GERMAN.contains(&word.as_str()));
    }
    let (language, count) = if german > english {
        (Language::German, german)
    } else {
        (Language::English, english)
    };
    (count > 0 && count * 100 >= words * SHARE).then_some(language)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EN: &str = "It was the best of times and he said that the old man had gone to the sea with his dog, but she did not know it.";
    const DE: &str = "Er sagte, dass die Frau nicht zu Hause ist und der Hund mit dem Mann in den Wald lief, und ich sah es von der Tür aus, denn sie war ein Kind.";
    const FR: &str = "Il était une fois une vieille femme qui vivait dans une petite maison au bord de la mer avec son chien et ses deux enfants.";

    #[test]
    fn the_larger_count_decides() {
        assert_eq!(detect([EN].into_iter()), Some(Language::English));
        assert_eq!(detect([DE].into_iter()), Some(Language::German));
    }

    #[test]
    fn a_book_that_mixes_the_two_goes_by_the_one_with_more_text() {
        assert_eq!(detect([DE, EN, DE].into_iter()), Some(Language::German));
        assert_eq!(detect([EN, DE, EN].into_iter()), Some(Language::English));
    }

    #[test]
    fn a_book_of_neither_or_no_words_is_in_neither() {
        assert_eq!(detect([FR].into_iter()), None);
        assert_eq!(detect([""].into_iter()), None);
    }
}
