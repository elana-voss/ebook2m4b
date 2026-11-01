"""The comparison of phase work: every book by each route, stored as a tree and diffed against a stored run.

Usage: python tools/baseline.py store work/baseline-a722a18   (the first run: converts with Calibre, long)
       python tools/baseline.py check work/baseline-a722a18 [--out work/compare-now] [--show 5]
A tree holds mobi/ (tools/mobi_compare.py: calibre-in, calibre/<book>, other/<book>) and formats/
(tools/format_compare.py: in, epub/<book>, fb2/<book>, ...), each <book> with text.txt and changes.txt,
and mobi-summary.txt, formats-summary.txt, what those tools printed. `check` reuses the baseline's Calibre
files, runs the present target/release/ebook2m4b.exe, writes the whole diff to <out>/diff.txt (private: the
books' text, under work/) and prints per route and file how many lines moved.
"""
import argparse
import difflib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def tool(name, args, summary):
    r = subprocess.run([sys.executable, str(HERE / name), *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    summary.write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        sys.exit(f"{name} failed:\n{r.stderr[-2000:]}")


def run_all(out, inputs):
    """Both tools into the tree out, with the Calibre files of the tree inputs."""
    out.mkdir(parents=True, exist_ok=True)
    cal = inputs / "mobi" / "calibre-in"
    tool("mobi_compare.py", ["--out", out / "mobi", "--calibre-in", cal], out / "mobi-summary.txt")
    tool("format_compare.py", ["--out", out / "formats", "--in", inputs / "formats" / "in", "--epubs", cal], out / "formats-summary.txt")


def outputs(tree):
    """Every text.txt and changes.txt under tree, by its path from there."""
    return {p.relative_to(tree).as_posix(): p for f in ("text.txt", "changes.txt") for p in tree.rglob(f)}


def diff(old, new, out, show):
    a, b = outputs(old), outputs(new)
    moved = 0
    with (out / "diff.txt").open("w", encoding="utf-8") as whole:
        for name in sorted(a.keys() | b.keys()):
            if name not in a or name not in b:
                print(f"{name}: only in {'the new run' if name in b else 'the baseline'}")
                moved += 1
                continue
            x, y = [p.read_text(encoding="utf-8").splitlines() for p in (a[name], b[name])]
            if x == y:
                continue
            moved += 1
            lines = list(difflib.unified_diff(x, y, "baseline/" + name, "new/" + name, lineterm="", n=0))
            whole.write("\n".join(lines) + "\n")
            body = [l for l in lines[2:] if l[0] in "+-"]
            print(f"{name}: -{sum(l[0] == '-' for l in body)} +{sum(l[0] == '+' for l in body)}")
            for l in body[:show]:
                print("   ", l[:200])
    print(f"{len(a)} baseline files, {moved} differ; the whole diff: {out / 'diff.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["store", "check"])
    ap.add_argument("baseline", type=Path)
    ap.add_argument("--out", type=Path, default=Path("work/compare-now"))
    ap.add_argument("--show", type=int, default=0, help="lines of each file's diff to print")
    a = ap.parse_args()
    if a.what == "store":
        run_all(a.baseline, a.baseline)
    else:
        run_all(a.out, a.baseline)
        diff(a.baseline, a.out, a.out, a.show)


if __name__ == "__main__":
    main()
