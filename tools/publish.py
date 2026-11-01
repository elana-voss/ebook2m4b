"""Bring the public repo up to date from this one, and commit there.

Usage: python tools/publish.py "<commit message>" [public folder]   (by default ..\\ebook2m4b beside this repo)
Copies every file tracked here except the private ones (PRIVATE below) into the public folder, removes there the
tracked files no longer tracked here, scans the result with tools/private_scan.py, and commits only on 0 hits,
as elana-voss with the noreply address; then scans the public history with --log. No remote, no push.
Exit 1 on any hit (nothing committed when the files hit), 0 when committed or nothing changed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PRIVATE = ("CLAUDE.md", "docs/", ".claude/")
AUTHOR = ["-c", "user.name=elana-voss", "-c", "user.email=1338567+elana-voss@users.noreply.github.com"]


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def tracked(repo):
    return {f for f in git(repo, "ls-files", "-z").split("\0") if f}


def public_files():
    return sorted(f for f in tracked(HERE) if not f.startswith(PRIVATE))


def scan(repo, *flags):
    return subprocess.run([sys.executable, str(HERE / "tools" / "private_scan.py"), *flags, str(repo)]).returncode


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    message = sys.argv[1]
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "ebook2m4b"
    files = public_files()
    for f in files:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / f, dest / f)
    gone = sorted(tracked(dest) - set(files))
    if gone:
        git(dest, "rm", "-q", "--", *gone)
    git(dest, "add", "-f", "--", *files)  # -f: a tracked file here may match the public .gitignore (tools/*.sh)
    if not git(dest, "status", "--porcelain"):
        print("public repo already up to date")
        return 0
    if scan(dest):
        print("private terms found: nothing committed (the changes stay staged in", dest, ")")
        return 1
    git(dest, *AUTHOR, "commit", "-q", "-m", message)
    print(git(dest, "log", "--oneline", "-1").strip(), f"({len(files)} files, {len(gone)} removed)")
    return scan(dest, "--log")


if __name__ == "__main__":
    sys.exit(main())
