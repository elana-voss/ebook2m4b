"""Scan for private terms before the repo is published.

Usage: python tools/private_scan.py [--log] [repo]   (repo: the folder scanned, by default this one)
Terms: private_terms.txt beside this script (one case-insensitive regex a line, # comments; the file stays untracked).
Scans every tracked file (git grep), and with --log also `git log -p --all` with authors and messages.
Prints each term's hit count and its hits; exit 1 when any term hits.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), Path(__file__).resolve().parent.parent))
# misaki's US dictionaries verbatim (src/text/words.rs says how it was made): ordinary English words only
SKIP = {"data/known-words.txt"}
terms = [t for t in (Path(__file__).resolve().parent / "private_terms.txt").read_text(encoding="utf-8").splitlines()
         if t.strip() and not t.startswith("#")]


def tracked_lines():
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    for f in files.decode().split("\0"):
        if not f:
            continue
        p = ROOT / f
        yield f, 0, f  # the path itself
        if f in SKIP:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary or deleted: only its path is scanned
        for i, line in enumerate(text.splitlines(), 1):
            yield f, i, line


def log_lines():
    out = subprocess.run(["git", "log", "-p", "--all", "--format=commit %H%nauthor %an <%ae>%ncommitter %cn <%ce>%n%B",
                          "--", "."] + [f":!{f}" for f in SKIP],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace")
    for i, line in enumerate(out.splitlines(), 1):
        yield "git log", i, line


lines = list(tracked_lines()) + (list(log_lines()) if "--log" in sys.argv else [])
total = 0
for t in terms:
    rx = re.compile(t, re.I)
    hits = [(f, i, l) for f, i, l in lines if rx.search(l)]
    total += len(hits)
    print(f"{len(hits):6}  {t}")
    for f, i, l in hits:
        print(f"        {f}:{i}: {l.strip()[:140]}")
sys.exit(1 if total else 0)
