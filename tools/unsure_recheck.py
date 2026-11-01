"""Checks every listed unsure passage of each book again (spoken, heard, compared, as the product does:
the program's --recheck) and prints the count still unsure per book, old and new.
    python tools/unsure_recheck.py <results folder> [book name part ...]
Each book's lines go to <results folder>/<book>.tsv (chunk, error listed, error now, heard, text);
a book whose .tsv is there already is not run again (delete it to rerun). Needs target/release/ebook2m4b.exe
and the books in books/; the GPU, one book at a time. Passages fixed in the text but listed from the old
run are left out of the counts: one per line of <results folder>/fixed.txt, <book name>, a tab, <chunk>.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "target" / "release" / "ebook2m4b.exe"


def main(folder, parts):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    listed = folder / "fixed.txt"
    lines = listed.read_text(encoding="utf-8").splitlines() if listed.exists() else []
    fixed = {tuple(line.split("\t")[:2]) for line in lines if line}
    total = [0, 0, 0]
    print("book".ljust(40), " listed", "    was", "    now")
    for unsure in sorted((ROOT / "out").glob("*.unsure.txt")):
        book = unsure.name.replace(".unsure.txt", "")
        if parts and not any(p.lower() in book.lower() for p in parts):
            continue
        tsv = folder / f"{book}.tsv"
        if not tsv.exists():
            src = next(ROOT.joinpath("books").glob(book + ".*"))
            got = subprocess.run(
                [str(EXE), str(src), "--recheck", str(unsure)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            )
            lines = [l for l in got.stdout.splitlines() if l.count("\t") >= 4]
            if got.returncode or not lines:
                print(book, "failed:", got.stdout[-500:], got.stderr[-500:])
                continue
            tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows = [l.split("\t", 4) for l in tsv.read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if (book, r[0]) not in fixed]
        was = sum(float(r[1]) > 0.1 for r in rows)
        now = sum(float(r[2]) > 0.1 for r in rows)
        print(book[:40].ljust(40), str(len(rows)).rjust(7), str(was).rjust(7), str(now).rjust(7))
        total = [a + b for a, b in zip(total, (len(rows), was, now))]
    print("all".ljust(40), *(str(t).rjust(7) for t in total))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
