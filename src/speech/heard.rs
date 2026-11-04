//! How far what Whisper heard is from a chunk's text: heard_error(), in sounds, with Whisper's spelling
//! of the text's invented names forgiven. speak.rs asks it of each chunk.
use crate::speech::pronounce::Pronounce;
use crate::text::language::Language;
use anyhow::Result;

/// a line of fewer sounds counts as this many: Whisper hears a short line without context, so one or two
/// sounds off there ("Be sad" for "be said") are the ear's, and a short line is unsure only from three
const SHORT_LINE: usize = 20;

/// Edit distance of the sounds of what was heard against the text's, over the text's sounds' length
/// (numbers as words, as the voice was given them), at least SHORT_LINE. A word
/// Whisper writes in capitals ("MIP") is a word it heard, not letters, so it is lowered before its sounds
/// are asked. What Whisper hears beyond the text's end is free: it goes on after the voice stopped
/// (repeats the line, finishes the sentence), the voice reads its phonemes and adds none.
pub fn heard_error(p: &Pronounce, expected: &str, heard: &str) -> Result<f32> {
    let sounds = |t: &str| -> Result<String> {
        let language = if p.is_german() { Language::German } else { Language::English };
        let t = crate::text::clean::clean_in(t, language);
        if t.trim().is_empty() {
            return Ok(String::new()); // nothing heard: no phonemes to ask for
        }
        p.sounds(&t)
    };
    let heard: String = heard
        .split(' ')
        .map(|w| {
            if w.chars().filter(|c| c.is_alphabetic()).count() > 1
                && !w.chars().any(|c| c.is_lowercase())
            {
                w.to_lowercase()
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    // the dictionary knows only English: in German every word would be a name, and forgiven
    let heard = if p.is_german() { heard } else { names_as_heard(p, expected, &heard)? };
    let (a, b) = (sounds(expected)?, sounds(&heard)?);
    let wrong = crate::text::edit::distance_to_prefix(&a, &b);
    Ok(wrong as f32 / a.chars().count().max(SHORT_LINE) as f32)
}

/// A word's English letters: a heard word and a name are compared by these.
fn ascii_letters(w: &str) -> String {
    w.chars()
        .filter(char::is_ascii_alphabetic)
        .flat_map(char::to_lowercase)
        .collect()
}

/// `heard` with Whisper's spelling of the text's invented names put right: a word of `expected` outside the
/// dictionary (words::known), of four letters or more, replaces the one to three heard words in a row whose
/// sounds differ from its own in at most half ("hair of Ollie" for "Haravoli", "B-Day Toshik" for
/// "Bidaitoshik"), or the one heard word whose letters differ in at most a third ("Choir" for "Chooir",
/// "Malukta" for "Mel-Ok-Ta"); of those, one spelt near, else the nearest in sounds. A name heard as spelt stays. Whisper writes words it knows for a name it does not, and
/// eSpeak's sounds for its spelling are far from the name's, though what was heard was the name.
fn names_as_heard(p: &Pronounce, expected: &str, heard: &str) -> Result<String> {
    let mut out: Vec<String> = heard.split(' ').map(String::from).collect();
    let said: Vec<String> = out.iter().map(|w| p.sounds(w)).collect::<Result<_>>()?;
    let names: Vec<&str> = expected
        .split(|c: char| c.is_whitespace())
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.chars().filter(char::is_ascii_alphabetic).count() >= 4 && !crate::text::words::known(w))
        .collect();
    // a name heard as spelt is that name: not another's nearest sound
    let heard: Vec<String> = out.iter().map(|w| ascii_letters(w)).collect();
    let mut taken: Vec<bool> = heard.iter().map(|w| names.iter().any(|n| ascii_letters(n) == *w)).collect();
    for name in names.into_iter().filter(|n| !heard.contains(&ascii_letters(n))) {
        let want = p.sounds(name)?;
        let spelt = ascii_letters(name);
        let mut near = None;
        for k in 0..out.len() {
            for len in 1..=3.min(out.len() - k) {
                if taken[k..k + len].iter().any(|&t| t) {
                    break;
                }
                let got: String = said[k..k + len].concat();
                let d = crate::text::edit::distance(&want, &got);
                let by_letters = len == 1 && crate::text::edit::distance(&spelt, &heard[k]) * 3 <= spelt.len();
                if (by_letters || d * 2 <= want.chars().count())
                    && near.is_none_or(|n| (!by_letters, d, len, k) < n)
                {
                    near = Some((!by_letters, d, len, k));
                }
            }
        }
        if let Some((_, _, len, k)) = near {
            let tail: String = out[k + len - 1].chars().rev().take_while(|c| !c.is_alphanumeric()).collect();
            out[k] = format!("{name}{}", tail.chars().rev().collect::<String>());
            out[k + 1..k + len].iter_mut().for_each(String::clear);
            taken[k..k + len].iter_mut().for_each(|t| *t = true);
        }
    }
    Ok(out.into_iter().filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::pronounce::tests::P;
    use crate::speech::speak::MAX_ERROR;

    #[test]
    fn german_words_are_no_names_forgiven() {
        let g = P.german();
        let err = heard_error(&g, "Sie sahen den Morgenstern", "Sie sahen den Morgensturm").unwrap();
        assert!(err > 0.0);
        let err = heard_error(&g, "Es waren zwölf Männer", "Es waren 12 Männer").unwrap();
        assert_eq!(err, 0.0); // what Whisper writes in digits is read in German too
    }

    #[test]
    fn heard_error_compares_sounds() {
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert_eq!(
            err(
                "The Ivok's daughter, 3 barls.",
                "the ivok's daughter three barls"
            ),
            0.0
        );
        assert!(err("The Ivok's daughter.", "") > 0.5); // nothing heard
        let one = err("The cat sat.", "The bat sat.");
        assert!(one > 0.0 && one < 0.2, "{one}");
        // Whisper writes a name in capitals ("MIP"); the voice would have spelt it, Whisper heard a word
        assert_eq!(err("Chapter thirteen Mip", "Chapter 13 MIP"), 0.0);
    }

    #[test]
    fn a_name_whisper_nearly_spells_is_the_name() {
        let names = |e, h| names_as_heard(&P, e, h).unwrap();
        // Kokoro's sounds of "Choir" are far from those of "Chooir"; the letters are not
        assert_eq!(
            names("“I shall go,” said Chooir. Mel-Ok-Ta left.", "I shall go, said Choir. Malukta left."),
            "I shall go, said Chooir. Mel-Ok-Ta left."
        );
        // one spelt far from it stays; a letter of four may differ ("Carl" for "Zarl")
        assert_eq!(names("said Chooir.", "said table."), "said table.");
        assert_eq!(names("said Zarl.", "said Carl."), "said Zarl.");
    }

    #[test]
    fn a_name_whisper_spells_by_its_sound_is_the_name() {
        // Whisper writes words it knows that sound like the name, one or several
        let names = |e, h| names_as_heard(&P, e, h).unwrap();
        assert_eq!(names("said the Haravoli.", "said the hair of Ollie."), "said the Haravoli.");
        assert!(names("Amblis jumped;", "Am bliss jumped.").ends_with("Amblis jumped."));
        assert_eq!(names("asked Bidaitoshik.", "asked B-Day Toshik."), "asked Bidaitoshik.");
        // a name heard as spelt is that name, not the nearest sound of another one
        assert_eq!(
            names("Outside I encountered Mel-Ok-Ta and Mel-Da-Sa.", "Outside I encountered Malukta and Mel-Da-Sa."),
            "Outside I encountered Mel-Ok-Ta and Mel-Da-Sa."
        );
        assert_eq!(names("“I hunt,” said Chooir. “Of course,” said Selmus.", "I hunt, said Choir. Of course, said Selmus."), "I hunt, said Chooir. Of course, said Selmus.");
        assert_eq!(names("the dreen ran", "the drain ran"), "the dreen ran");
    }

    #[test]
    fn a_tail_heard_beyond_the_text_is_whispers() {
        // Whisper goes on after the voice stopped (repeats, finishes the sentence); the voice adds no word
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert_eq!(
            err(
                "‘Go in peace,’ said the Initiate, stepping aside.",
                "Go in peace, said the initiate, stepping aside Go in peace, said the initiate, stepping aside"
            ),
            0.0
        );
        // a tail the voice dropped still counts
        assert!(err("The cat sat on the mat.", "The cat sat") > 0.2);
    }

    #[test]
    fn a_short_line_is_unsure_only_from_three_sounds_off() {
        // one or two sounds off in a line of a few words is the ear without context ("Be sad" for "be said")
        let err = |a, b| heard_error(&P, a, b).unwrap();
        assert!(err("be said.", "Be sad.") <= MAX_ERROR);
        assert!(err("Zarl!", "Mary!") > MAX_ERROR);
    }
}
