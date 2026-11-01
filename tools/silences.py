"""Silences in a made audiobook, for the milestone "Silence inside a sentence".

  python tools/silences.py link <work/<book>/<voice>> <keep>   hard-links each chunk WAV into <keep> as it
      appears (the run removes its work folder at the end); stops when the work folder is gone.
  python tools/silences.py check <book.m4b> <keep>   ffmpeg silencedetect (-45 dB, over 1.0 s) over the
      whole book, and each chunk WAV's silences over 1.0 s inside it (not at its start or end): a silence
      between chunks is a sentence end, or 0.4 s at a cut (assemble.rs), so a long one inside a sentence
      can only be inside a chunk.
"""
import os, re, subprocess, sys, time

RATE = 24_000
LONG = 1.0


def link(src, keep):
    os.makedirs(keep, exist_ok=True)
    seen = False
    while True:
        if os.path.isdir(src):
            seen = True
            for f in os.listdir(src):
                if re.fullmatch(r"\d{6}\.wav", f) and not os.path.exists(os.path.join(keep, f)):
                    try:
                        os.link(os.path.join(src, f), os.path.join(keep, f))
                    except OSError as e:  # removed between listing and linking: the next pass tells
                        print("not linked", f, e, flush=True)
        elif seen:
            return
        time.sleep(10)


def detect(path):
    """(start, end) of each silence over LONG s that ffmpeg's silencedetect finds in path, in seconds."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                        f"silencedetect=noise=-45dB:d={LONG}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends + [float("inf")] * (len(starts) - len(ends))))


def check(m4b, keep):
    ds = [e - s for s, e in detect(m4b)]
    print(f"book: {len(ds)} silences over {LONG} s; shortest {min(ds, default=0):.2f}, "
          f"longest {max(ds, default=0):.2f}")
    for lo, hi in [(1.0, 1.2), (1.2, 1.4), (1.4, 2.0), (2.0, 99)]:
        print(f"  {lo}-{hi} s: {sum(lo <= d < hi for d in ds)}")
    files = sorted(f for f in os.listdir(keep) if f.endswith(".wav"))
    bad = 0
    for f in files:
        path = os.path.join(keep, f)
        seconds = (os.path.getsize(path) - 44) / 4 / RATE  # float WAV, kokoro.rs's 44-byte header
        for s, e in detect(path):
            if s > 0.01 and e < seconds - 0.01:
                bad += 1
                print(f"  inside chunk {f[:6]}: {e - s:.2f} s at {s:.2f} s")
    print(f"chunks: {len(files)}; silences over {LONG} s inside a chunk: {bad}")


if __name__ == "__main__":
    {"link": link, "check": check}[sys.argv[1]](*sys.argv[2:])
