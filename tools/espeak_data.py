"""Fetch only eSpeak's data into the settings folder, as setup would: the tests that speak need it (the release
workflow runs this before the tests on a fresh machine). The URL and SHA-256 are read from src/setup/downloads.rs.

Usage: python tools/espeak_data.py
The settings folder: %LOCALAPPDATA%\\ebook2m4b on Windows, else $XDG_DATA_HOME/ebook2m4b or ~/.local/share/ebook2m4b.
"""
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PREFIX = "espeakng_loader/espeak-ng-data/"


def home():
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "ebook2m4b"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "ebook2m4b"


def main():
    src = (HERE / "src" / "setup" / "downloads.rs").read_text(encoding="utf-8")
    m = re.search(r'item\("espeak-ng-data", &format!\("\{PYPI\}/([^"]+)"\),\s*"([0-9a-f]{64})"', src)
    if not m:
        sys.exit("espeak-ng-data not found in src/setup/downloads.rs")
    data = urllib.request.urlopen("https://files.pythonhosted.org/packages/" + m[1]).read()
    if hashlib.sha256(data).hexdigest() != m[2]:
        sys.exit("espeak-ng-data: SHA-256 differs")
    dest = home() / "espeak-ng-data"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.startswith(PREFIX) and not name.endswith("/") and ".." not in name:
                out = dest / name[len(PREFIX):]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    print(f"eSpeak's data in {dest}")


if __name__ == "__main__":
    main()
