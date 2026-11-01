"""Which words of the books words.rs known() calls known, before and after a change to it.
    python tools/known_flip.py save <name>     known() of every word of work/*/text.txt -> <scratch>/known-<name>.tsv
    python tools/known_flip.py compare <a> <b> the words known in <a> and unknown in <b>, and the reverse
Runs the ignored test words::tests::known_of_a_word_list (cargo, after `source env.sh`). <scratch> is
%TEMP%/ebook2m4b.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ["TEMP"]) / "ebook2m4b"
WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def save(name):
    words = set()
    for t in sorted((ROOT / "work").glob("*/text.txt")):
        words |= {w.lower() for w in WORD.findall(t.read_text(encoding="utf-8"))}
    lst = SCRATCH / "known-words-list.txt"
    lst.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    out = SCRATCH / f"known-{name}.tsv"
    env = dict(os.environ, T2A_WORDS=str(lst), T2A_KNOWN=str(out))
    cmd = ["cargo", "test", "--release", "-j", "4", "--bin", "ebook2m4b", "known_of_a_word_list", "--",
           "--ignored", "--quiet"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr[-2000:])
    print(f"{len(words)} words -> {out}")


def load(name):
    rows = (SCRATCH / f"known-{name}.tsv").read_text(encoding="utf-8").splitlines()
    return {w: k == "1" for w, k in (r.split("\t") for r in rows)}


def compare(a, b):
    ka, kb = load(a), load(b)
    lost = sorted(w for w in ka if ka[w] and not kb.get(w, False))
    won = sorted(w for w in kb if kb[w] and not ka.get(w, False))
    print(f"known in {a}, unknown in {b}: {len(lost)}")
    for w in lost:
        print("  -", w)
    print(f"unknown in {a}, known in {b}: {len(won)}")
    for w in won:
        print("  +", w)


if __name__ == "__main__":
    {"save": lambda: save(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
