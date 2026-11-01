"""Survey of mobi/azw/azw3 files: format (MOBI 6, KF8, both), compression, DRM, encoding.

Usage: python tools/mobi_survey.py [folder]   (default books/). Reads only the PDB and MOBI headers.
"""
import struct
import sys
from pathlib import Path

COMP = {1: "none", 2: "PalmDOC", 17480: "HUFF/CDIC"}


def headers(data, rec):
    """MOBI header facts of the record at offset rec (record 0 or the KF8 boundary's next)."""
    comp, _, text_len, nrec, rec_size, enc = struct.unpack(">HHIHHH", data[rec:rec + 14])
    mobi = data[rec + 16:]
    if mobi[:4] != b"MOBI":
        return None
    hlen, mtype, codepage, _, version = struct.unpack(">IIIII", mobi[4:24])
    exth = {}
    flags = struct.unpack(">I", mobi[0x70:0x74])[0] if hlen >= 0x74 else 0
    if flags & 0x40:
        e = rec + 16 + hlen
        if data[e:e + 4] == b"EXTH":
            n = struct.unpack(">I", data[e + 8:e + 12])[0]
            p = e + 12
            for _ in range(n):
                t, ln = struct.unpack(">II", data[p:p + 8])
                exth.setdefault(t, data[p + 8:p + ln])
                p += ln
    return dict(comp=COMP.get(comp, comp), drm=enc, version=version, codepage=codepage,
                nrec=nrec, exth=exth)


def survey(path):
    data = path.read_bytes()
    n = struct.unpack(">H", data[76:78])[0]
    offs = [struct.unpack(">I", data[78 + 8 * i:82 + 8 * i])[0] for i in range(n)]
    h = headers(data, offs[0])
    if h is None:
        return f"{path.name}: no MOBI header ({data[60:68]!r})"
    fmt = "KF8" if h["version"] == 8 else f"MOBI {h['version']}"
    if 121 in h["exth"]:
        b = struct.unpack(">I", h["exth"][121])[0]
        if b != 0xFFFFFFFF and b + 1 < n:
            k = headers(data, offs[b + 1])
            if k:
                fmt = f"both (MOBI {h['version']} + KF8, {k['comp']})"
    enc = "UTF-8" if h["codepage"] == 65001 else f"cp{h['codepage']}"
    drm = "none" if h["drm"] == 0 else f"DRM type {h['drm']}"
    return f"{path.name}: {fmt}; {h['comp']}; {drm}; {enc}; {h['nrec']} text records"


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "books")
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".mobi", ".azw", ".azw3"):
            print(survey(p))


main()
