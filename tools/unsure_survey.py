"""The unsure passages (out/*.unsure.txt) by likely cause, one table; the words not in Kokoro's dictionary
(data/known-words.txt) they hold, most frequent first.
    python tools/unsure_survey.py [out]
A line is counted under the first cause that fits: title (a chapter heading), name (a word outside the
dictionary), number (digits), short (under 40 characters), other.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWN = set(
    (ROOT / "data" / "known-words.txt").read_text(encoding="utf-8").split()
)
ENDINGS = ["", "s", "es", "ed", "ing", "er", "ly", "'s", "’s"]


def known(word):
    w = word.lower()
    return any(w.endswith(e) and w[: len(w) - len(e)] in KNOWN for e in ENDINGS) or (
        w.endswith("ies") and w[:-3] + "y" in KNOWN
    )


def unknown_words(text):
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", text) if not known(w)]


def cause(text):
    if re.match(r"^Chapter \w+( .*)?$", text) and len(text.split()) <= 6:
        return "title"
    if unknown_words(text):
        return "name"
    if re.search(r"\d", text):
        return "number"
    if len(text) < 40:
        return "short"
    return "other"


def main(out):
    table = {}
    words = Counter()
    for f in sorted(Path(out).glob("*.unsure.txt")):
        book = f.name.replace(".unsure.txt", "")
        counts = Counter()
        for line in f.read_text(encoding="utf-8").splitlines():
            n, err, text = line.split("\t", 2)
            counts[cause(text)] += 1
            words.update(w.lower() for w in set(unknown_words(text)))
        table[book] = counts
    causes = ["title", "name", "number", "short", "other"]
    print("book".ljust(40), *(c.rjust(7) for c in causes), "  total")
    for book, counts in table.items():
        print(
            book[:40].ljust(40),
            *(str(counts[c]).rjust(7) for c in causes),
            str(sum(counts.values())).rjust(7),
        )
    total = sum(table.values(), Counter())
    print("all".ljust(40), *(str(total[c]).rjust(7) for c in causes), str(sum(total.values())).rjust(7))
    print("\nunknown words (lines holding them):")
    print(", ".join(f"{w} {n}" for w, n in words.most_common(60)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "out")
