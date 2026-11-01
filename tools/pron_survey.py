"""Per book: the English words eSpeak speaks, the invented words, and of those how many pass the round trip
with eSpeak's guess and with the sound kept (work/sounds.json; plan "Pronunciation fixes").
    python tools/pron_survey.py <out file> [book name part ...]
The books are those with an out/<book>.unsure.txt; runs target/release/ebook2m4b.exe --words (no GPU).
sounds.json: {"word": {"say": respelling or null, "espeak": distance, "respelling": distance or null}};
a word passes at a distance of a third of its letters or less (sounds.rs PASS).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "target" / "release" / "ebook2m4b.exe"
PASS = 1 / 3


def main(out, parts):
    sounds_file = ROOT / "work" / "sounds.json"
    sounds = json.loads(sounds_file.read_text(encoding="utf-8")) if sounds_file.exists() else {}
    lines = []
    print("book".ljust(40), "english", "invented", "espeak-pass", "kept-pass")
    for unsure in sorted((ROOT / "out").glob("*.unsure.txt")):
        book = unsure.name.replace(".unsure.txt", "")
        if parts and not any(p.lower() in book.lower() for p in parts):
            continue
        src = next(ROOT.joinpath("books").glob(book + ".*"))
        got = subprocess.run([str(EXE), str(src), "--words"], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8")
        rows = {r[1]: r for r in (l.split("\t", 3) for l in got.stdout.splitlines()) if len(r) == 4}
        if got.returncode or len(rows) < 2:
            print(book, "failed:", got.stdout[-300:], got.stderr[-300:])
            continue
        lines += ["\t".join(r) for r in rows.values()]
        invented = rows["invented"][3].split()
        tried = [sounds[w] for w in invented if w in sounds]
        es = sum(e["espeak"] <= PASS for e in tried)
        kept = sum(min(e["espeak"], e["respelling"] if e["respelling"] is not None else 9) <= PASS
                   for e in tried)
        print(book[:40].ljust(40), rows["english"][2].rjust(7), rows["invented"][2].rjust(8),
              f"{es}/{len(tried)}".rjust(11), f"{kept}/{len(tried)}".rjust(9))
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
