"""The text of some books in FB2, DOCX, TXT and HTML (Calibre's conversions of their EPUB) against the EPUB route.

Usage: python tools/format_compare.py [--books NAME ...] [--formats fb2 docx txt html] [--out DIR]
The EPUB is work/<book>/book.epub (made earlier by Calibre of books/<book>.mobi). tools/' ebook-convert
makes <out>/in/<format>/<book>.<format> once (HTML: the index.html of an htmlz); each folder is then
read by target/release/ebook2m4b.exe --text --no-judge and compared as tools/mobi_compare.py does.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mobi_compare import EBOOK_CONVERT, compare, run  # noqa: E402

BOOKS = ["Alice in Wonderland"]


def convert(epub, fmt, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    made = dest.with_suffix(".htmlz") if fmt == "html" else dest
    r = subprocess.run([str(EBOOK_CONVERT), str(epub), str(made)], capture_output=True)
    if r.returncode:
        sys.exit(f"Calibre could not make {made.name}")
    if fmt == "html":
        with zipfile.ZipFile(made) as z:
            dest.write_bytes(z.read("index.html"))
        made.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=BOOKS)
    ap.add_argument("--formats", nargs="+", default=["fb2", "docx", "txt", "html"])
    ap.add_argument("--out", type=Path, default=Path(os.environ["TEMP"]) / "ebook2m4b" / "formats")
    a = ap.parse_args()
    ins = a.out / "in"
    (ins / "epub").mkdir(parents=True, exist_ok=True)
    for b in a.books:
        epub = ins / "epub" / f"{b}.epub"
        if not epub.exists():
            shutil.copyfile(Path("work") / b / "book.epub", epub)
        for f in a.formats:
            convert(epub, f, ins / f / f"{b}.{f}")
    for f in ["epub", *a.formats]:
        run(ins / f, a.out / f)
    for f in a.formats:
        same = 0
        for b in a.books:
            o = a.out / f / b
            if not (o / "text.txt").exists():
                print(f"{f} {b}: not read")
                continue
            notes = compare(a.out / "epub" / b, o)
            if notes:
                print(f"{f} {b}:\n  " + "\n  ".join(notes))
            else:
                same += 1
        print(f"{f}: {same}/{len(a.books)} books the same")


if __name__ == "__main__":
    main()
