//! Levenshtein's edit distance.

/// Levenshtein distance in characters.
pub fn distance(a: &str, b: &str) -> usize {
    *distances(a, b).last().expect("a row has b's length + 1")
}

/// The least distance of `a` to a prefix of `b`: what `b` has beyond `a`'s end costs nothing.
pub fn distance_to_prefix(a: &str, b: &str) -> usize {
    distances(a, b).into_iter().min().expect("a row has b's length + 1")
}

/// Levenshtein's last row: the distance of `a` to each prefix of `b`.
fn distances(a: &str, b: &str) -> Vec<usize> {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for j in 1..=b.len() {
            let here = (row[j] + 1)
                .min(row[j - 1] + 1)
                .min(prev + usize::from(ca != b[j - 1]));
            prev = row[j];
            row[j] = here;
        }
    }
    row
}
