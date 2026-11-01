"""The text of each mobi/azw/azw3 in books/ (or --books) by two routes, compared: Calibre's EPUB
(work/<book>/book.epub, else made now by tools/' ebook-convert) and another (the books read directly,
or a folder of EPUBs another converter made of them).

Usage: python tools/mobi_compare.py [--books DIR] [--epubs DIR] [--out DIR]
Runs target/release/ebook2m4b.exe --text --no-judge (no model asked: seconds a book) on both (out: %TEMP%/ebook2m4b/compare/{calibre,other})
and prints per book: the opening line (title, authors), the chapter titles, the text (similarity and the
first differing line) and changes.txt, where they differ; then how many books match.
"""
import argparse
import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXE = Path("target/release/ebook2m4b.exe")
EBOOK_CONVERT = Path("tools/calibre/PFiles64/Calibre2/ebook-convert.exe")


def run(src, out):
    if out.exists():
        shutil.rmtree(out)
    r = subprocess.run([str(EXE), str(src), "--text", str(out), "--no-judge"], capture_output=True, text=True)
    if r.returncode:
        print(f"--text {src} failed:\n{r.stderr[-2000:]}")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i + 1}:\n    calibre: {x[:160]!r}\n    other:   {y[:160]!r}"
    return f"lengths {len(a)} / {len(b)} lines"


def compare(c, o):
    """How the --text folders c and o differ: opening, chapter titles, text, changes.txt (empty: alike)."""
    ct, ot = [(p / "text.txt").read_text(encoding="utf-8").splitlines() for p in (c, o)]
    cc, oc = [(p / "changes.txt").read_text(encoding="utf-8").splitlines() for p in (c, o)]
    notes = []
    if ct[:1] != ot[:1]:
        notes.append(f"opening: {ct[:1]} / {ot[:1]}")
    ch, oh = [[l for l in t if l.startswith("# ")] for t in (ct, ot)]
    if ch != oh:
        n = sum(x != y for x, y in zip(ch, oh)) + abs(len(ch) - len(oh))
        notes.append(f"chapters {len(ch)} / {len(oh)}, {n} titles differ; {first_diff(ch, oh)}")
    if ct != ot:
        cw, ow = [sum(len(l.split()) for l in t if not l.startswith("# ")) for t in (ct, ot)]
        ops = difflib.SequenceMatcher(None, ct, ot, autojunk=False).get_opcodes()
        n = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
        notes.append(f"words {cw} / {ow}, {n} lines differ; {first_diff(ct, ot)}")
    if cc != oc:
        notes.append(f"changes.txt {len(cc)} / {len(oc)} lines; {first_diff(cc, oc)}")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", type=Path, default=Path("books"))
    ap.add_argument("--epubs", type=Path, help="EPUBs of books/ by another converter (default: books/ read directly)")
    ap.add_argument("--out", type=Path, default=Path(os.environ["TEMP"]) / "ebook2m4b" / "compare")
    a = ap.parse_args()
    books = sorted(b for b in a.books.iterdir() if b.suffix.lower() in (".mobi", ".azw", ".azw3"))
    cal_in = a.out / "calibre-in"
    cal_in.mkdir(parents=True, exist_ok=True)
    for b in books:
        epub = Path("work") / b.stem / "book.epub"
        if epub.exists():
            shutil.copyfile(epub, cal_in / f"{b.stem}.epub")
        elif not (cal_in / f"{b.stem}.epub").exists():
            r = subprocess.run([str(EBOOK_CONVERT), str(b), str(cal_in / f"{b.stem}.epub")], capture_output=True)
            if r.returncode:
                sys.exit(f"Calibre could not convert {b.name}")
    run(cal_in, a.out / "calibre")
    run(a.epubs or a.books, a.out / "other")
    same = 0
    for b in books:
        c, o = a.out / "calibre" / b.stem, a.out / "other" / b.stem
        if not (o / "text.txt").exists():
            print(f"{b.stem}: not read by the other route")
            continue
        notes = compare(c, o)
        if notes:
            print(f"{b.stem}:\n  " + "\n  ".join(notes))
        else:
            same += 1
    print(f"{same}/{len(books)} books the same")


if __name__ == "__main__":
    main()
