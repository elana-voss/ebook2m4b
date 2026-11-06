"""Silences between two words with no punctuation between them, for milestone 24 ("Silence between words").

  python tools/word_gaps.py <sentences> --lang en|de --model <model.onnx> --voice <voice.bin>
      [--speed 1] [--split 0.16] [--hear] [--out <table.json>] [--label <name>]
  python tools/word_gaps.py --compare <before.json> <after.json> --limit <frames at speed 1> --top <seconds>
      [--out <table.json>]
  python tools/word_gaps.py --trace <table.json>... [--out <table.json>]

<sentences>: a text file, one sentence a line, or a JSON list of [before, after, sentence] whose pause sits
between "before" and "after" (those pairs are counted as tagged). The product's phonemes come from the
ignored test pronounce::tests::phonemes_of_lines (cargo, built as CLAUDE.md says). The model runs on the
GPU (onnxruntime-gpu, CUDA) with its per-token frames exposed as an extra output: Kokoro's
/encoder/Clip_output_0, Kerstin's `duration`; 600 samples a frame at 24 kHz.

A gap is the design's token pair: token i a space (id 16) whose neighbours both have id 17 or more; its
frames are those of token i-1 (the word's last sound) and token i; its silence is the longest run of 10 ms
windows under -45 dBFS from token i-1's start to the end of the first sound after the space (a stress mark
before it taken with it). A gap with silence at or over --split is "wrong", the rest "normal".

--hear: Whisper (setup's whisper-server and whisper-model, as the product, no prompt) hears each sentence;
its word match is the share of the sentence's words heard in order (difflib on lower-cased words).

--compare (24b): two tables of the same sentences and speed, the unpatched model's and the patched one's.
Counts every gap's last sound + gap over Round(limit/speed) frames in each, the pairs the patch should have
left alone that changed, every other token whose frames changed (punctuation, pauses, sounds), the wrong
gaps and the Whisper match in each. Prints a few lines; the per-pair table goes to --out.

--trace (24b): where the silence of each gap at or over --split sits. Every gap of a table carries its tokens from the word's
last sound to the first sound after the space, each as [symbol, frames, seconds of the silent run inside it],
and "run": the run's start and end and the window's end, in seconds from the last sound's start.
"""
import argparse, difflib, io, json, os, re, subprocess, sys, tempfile, time, urllib.request, uuid, wave

import numpy as np

RATE, FRAME, WIN = 24_000, 600, 240
QUIET = 10 ** (-45 / 20)
SPACE, FIRST_SOUND, ARROWS = 16, 17, range(169, 174)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ebook2m4b", "models", "kokoro", "tokenizer.json")
FRAMES_OUT = {"/encoder/Clip_output_0", "duration"}
SETUP = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ebook2m4b")
PORT = 8932  # not the product's 8931: a book being spoken keeps its own server


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)


def sentences(path):
    """The sentences and, for a JSON input, each one's tagged pause as (before, after)."""
    if path.endswith(".json"):
        items = load(path)
        return [s for _, _, s in items], [(a, b) for a, b, _ in items]
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines, [None] * len(lines)


def phonemes(lines, lang):
    """The product's phonemes of each line (pronounce.rs through its ignored test)."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
        with open(src, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        env = dict(os.environ, WORD_GAPS_IN=src, WORD_GAPS_OUT=dst, WORD_GAPS_LANG=lang)
        r = subprocess.run(["cargo", "test", "--release", "-j", "4", "phonemes_of_lines", "--", "--ignored"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode:
            sys.exit("phonemes failed:\n" + r.stderr[-2000:])
        with open(dst, encoding="utf-8") as f:
            out = f.read().split("\n")[: len(lines)]
    assert len(out) == len(lines), "a line lost its phonemes"
    return out


def session(model):
    """The model on CUDA with its per-token frames as one more output."""
    import torch  # noqa: F401  (loads the CUDA and cuDNN DLLs onnxruntime-gpu needs)
    import onnx, onnxruntime as ort
    m = onnx.load(model)
    names = {o.name for o in m.graph.output}
    frames = next(iter(FRAMES_OUT & {o for n in m.graph.node for o in n.output}))
    if frames not in names:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(frames))
    ort.set_default_logger_severity(3)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    s = ort.InferenceSession(m.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
    if s.get_providers()[0] != "CUDAExecutionProvider":
        sys.exit("CUDA is not available to onnxruntime: the model runs on the GPU only")
    return s, frames


def silence(audio, lo, hi, cut):
    """The longest run of samples whose 10 ms RMS is under -45 dBFS in audio[lo:hi] that starts before `cut`
    (the first sound after the space: a run starting inside it is that word's, such as a pause at the
    punctuation after it): (seconds, first, end)."""
    rms = np.sqrt(np.convolve(audio.astype(np.float64) ** 2, np.ones(WIN) / WIN, "same"))
    best = run = 0
    at = lo
    for k, q in enumerate(rms[lo:hi] < QUIET):
        run = run + 1 if q else 0
        if run > best and lo + k + 1 - run < cut:
            best, at = run, lo + k + 1 - run
    return best / RATE, at, at + best


class Ear:
    """whisper.cpp's server as the product starts it (whisper.rs server_args), stopped by close()."""

    def __init__(self, lang):
        setup = load(os.path.join(SETUP, "setup.json"))
        exe = os.path.join(setup["whisper-server"], "whisper-server.exe")
        args = [exe, "-m", setup["whisper-model"], "--port", str(PORT), "-t", "4", "-l", lang]
        self.proc = subprocess.Popen(args,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2)
                return
            except OSError:
                if self.proc.poll() is not None:
                    sys.exit("whisper-server ended at its start")
                time.sleep(0.5)
        self.proc.kill()
        sys.exit("whisper-server did not answer in 60 s")

    def hear(self, audio):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        b = uuid.uuid4().hex
        fields = [("response_format", "json"), ("temperature", "0"), ("temperature_inc", "0")]
        body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                        for k, v in fields)
        body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="s.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/inference", body,
                                     {"Content-Type": f"multipart/form-data; boundary={b}"})
        return json.load(urllib.request.urlopen(req, timeout=120))["text"].strip()

    def close(self):
        self.proc.kill()


def word_match(said, heard):
    """The share of the said words that Whisper heard, in order."""
    a, b = (re.findall(r"\w+", t.lower()) for t in (said, heard))
    matched = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(m.size for m in matched) / max(len(a), 1)


def tagged_word(sentence, pause):
    """The index of the word after the tagged pause, counted in whitespace-parted words."""
    a, b = pause
    at = sentence.find(a + " " + b)
    return None if at < 0 else len((sentence[: at + len(a)]).split())


def measure(args):
    vocab = load(TOKENIZER)["model"]["vocab"]
    stress = {vocab["ˈ"], vocab["ˌ"]}
    style = np.fromfile(args.voice, dtype=np.float32).reshape(-1, 256)
    lines, pauses = sentences(args.sentences)
    ps = phonemes(lines, args.lang)
    sess, frames_name = session(args.model)
    ear = Ear(args.lang) if args.hear else None
    rows, per_sentence, arrows_seen = [], [], 0
    for n, (line, p, pause) in enumerate(zip(lines, ps, pauses)):
        if len(p) > 510:
            sys.exit(f"sentence {n + 1} has {len(p)} phonemes, over Kokoro's 510: shorten it")
        chars = ["_"] + [c for c in p if c in vocab] + ["_"]
        ids = [vocab[c] if c != "_" else 0 for c in chars]
        arrows_seen += sum(i in ARROWS for i in ids)
        wav, dur = sess.run(["waveform", frames_name], {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": style[len(p) - 1 : len(p)],
            "speed": np.array([args.speed], dtype=np.float32)})
        audio, dur = wav.reshape(-1), np.rint(np.asarray(dur, dtype=np.float64).reshape(-1)).astype(int).tolist()
        start = np.cumsum([0] + dur) * FRAME
        per_sentence.append({"phonemes": p, "ids": ids, "frames": dur})
        if ear:
            heard = ear.hear(audio)
            per_sentence[-1].update(heard=heard, match=round(word_match(line, heard), 4))
        words = p.split(" ")
        tag = tagged_word(line, pause) if pause else None
        if tag is not None and len(words) != len(line.split()):
            tag = None  # the phoneme words do not line up with the text's: the tag cannot be placed
        word = 0
        for i in range(1, len(ids) - 1):
            if ids[i] != SPACE:
                continue
            word += 1
            if ids[i - 1] < FIRST_SOUND or ids[i + 1] < FIRST_SOUND:
                continue
            end = i + 3 if ids[i + 1] in stress else i + 2
            end = min(end, len(ids))
            sil, lo, hi = silence(audio, int(start[i - 1]), int(start[end]), int(start[i + 1]))
            # where the silence sits: each token's symbol, frames and seconds of the run inside its frames
            trace = [[chars[k], dur[k], round(max(0, min(hi, start[k + 1]) - max(lo, start[k])) / RATE, 4)]
                     for k in range(i - 1, end)]
            rows.append({"sentence": n + 1, "words": words[word - 1] + " " + words[word],
                         "last": dur[i - 1], "gap": dur[i], "frames": dur[i - 1] + dur[i],
                         "silence": round(sil, 4), "wrong": sil >= args.split,
                         "arrow": ids[i - 1] in ARROWS or ids[i + 1] in ARROWS,
                         "tagged": tag == word, "trace": trace,
                         "run": [round((lo - start[i - 1]) / RATE, 4), round((hi - start[i - 1]) / RATE, 4),
                                 round((start[end] - start[i - 1]) / RATE, 4)]})
        if pause and tag is None:
            print(f"sentence {n + 1}: its tagged pause could not be placed", file=sys.stderr)
    if ear:
        ear.close()
    save({"model": args.model, "voice": args.voice, "speed": args.speed, "split": args.split,
          "gaps": rows, "sentences": per_sentence}, args.out)
    summary(args.label or os.path.basename(os.path.dirname(args.model)), rows, arrows_seen, pauses)


def summary(label, rows, arrows_seen, pauses):
    ok = [r for r in rows if not r["wrong"]]
    bad = [r for r in rows if r["wrong"]]
    top = max(r["frames"] for r in ok)
    print(f"{label}: {len(rows)} gaps, normal {len(ok)}, wrong {len(bad)}; arrows in the phonemes "
          f"{arrows_seen}, at a gap {sum(r['arrow'] for r in rows)}")
    print(f"  normal: frames (last sound + gap) at most {top}, silence at most "
          f"{max(r['silence'] for r in ok):.3f} s; {sum(r['silence'] < 0.025 for r in ok)} under 0.025 s")
    if bad:
        lo = min(r["frames"] for r in bad)
        print(f"  wrong: frames {lo}-{max(r['frames'] for r in bad)}, silence "
              f"{min(r['silence'] for r in bad):.3f}-{max(r['silence'] for r in bad):.3f} s; "
              f"wrong at or under {top} frames: {sum(r['frames'] <= top for r in bad)}; "
              f"normal at or over {lo}: {sum(r['frames'] >= lo for r in ok)}")
    if any(pauses):
        t = [r for r in rows if r["tagged"]]
        print(f"  tagged pauses placed {len(t)} of {len(pauses)}, of them wrong {sum(r['wrong'] for r in t)}")


def pairs(ids):
    """The token pairs (i-1, i) the patch limits: i a space, both neighbours sounds."""
    return [(i - 1, i) for i in range(1, len(ids) - 1)
            if ids[i] == SPACE and ids[i - 1] >= FIRST_SOUND and ids[i + 1] >= FIRST_SOUND]


def compare(args):
    before, after = (load(f) for f in args.compare)
    if before["speed"] != after["speed"] or len(before["sentences"]) != len(after["sentences"]):
        sys.exit("the two tables are not of the same sentences and speed")
    top = round(args.limit / before["speed"])  # Python's round, as ONNX's Round, halves to even
    over_b = over_a = changed_pairs = other_changed = other_tokens = 0
    rows = []
    gaps = iter(zip(before["gaps"], after["gaps"]))  # in the order of pairs(), sentence by sentence
    for n, (b, a) in enumerate(zip(before["sentences"], after["sentences"])):
        if b["ids"] != a["ids"]:
            sys.exit(f"sentence {n + 1}: the tokens differ")
        inside = set()
        for j, i in pairs(b["ids"]):
            inside |= {j, i}
            fb, fa = b["frames"][j] + b["frames"][i], a["frames"][j] + a["frames"][i]
            over_b += fb > top
            over_a += fa > top
            changed_pairs += fb <= top and (b["frames"][j], b["frames"][i]) != (a["frames"][j], a["frames"][i])
            gb, ga = next(gaps)
            rows.append({"sentence": n + 1, "token": i, "words": gb["words"], "before": fb, "after": fa,
                         "silence_before": gb["silence"], "silence_after": ga["silence"]})
        for k in range(len(b["ids"])):
            if k not in inside:
                other_tokens += 1
                other_changed += b["frames"][k] != a["frames"][k]
    print(f"speed {before['speed']}, limit {top} frames: {len(rows)} gaps; over the limit before {over_b}, "
          f"after {over_a}; pairs under it changed {changed_pairs}")
    print(f"  other tokens {other_tokens}, changed {other_changed}")
    set_ = [r for r in rows if r["before"] > top]
    if set_:
        print(f"  pairs the rule set ({len(set_)}): silence after {min(r['silence_after'] for r in set_):.3f}-"
              f"{max(r['silence_after'] for r in set_):.3f} s, over the top {args.top} s "
              f"{sum(r['silence_after'] > args.top for r in set_)}")
    kept = [r for r in rows if r["before"] <= top]
    print(f"  pairs left ({len(kept)}): over the top {args.top} s before {sum(r['silence_before'] > args.top for r in kept)}, "
          f"after {sum(r['silence_after'] > args.top for r in kept)} (frames unchanged: not the rule's)")
    wb, wa = (sum(r["wrong"] for r in t["gaps"]) for t in (before, after))
    print(f"  wrong gaps (silence >= {before['split']} s) before {wb}, after {wa}")
    if "match" in before["sentences"][0] and "match" in after["sentences"][0]:
        mb = [s["match"] for s in before["sentences"]]
        ma = [s["match"] for s in after["sentences"]]
        print(f"  Whisper word match before {sum(mb) / len(mb):.4f}, after {sum(ma) / len(ma):.4f}; sentences "
              f"lower after {sum(y < x for x, y in zip(mb, ma))}, higher {sum(y > x for x, y in zip(mb, ma))}")
    save(rows, args.out)


def trace(args):
    """Where the silence of each gap at or over --split sits: per table one line, the seconds of the run in the
    word's last sound, the space and the sounds after it (a stress mark and the first sound); each gap to --out."""
    rows = []
    for f in args.trace:
        t = load(f)
        bad = [g for g in t["gaps"] if g["silence"] >= args.split]
        part = [sum(g["trace"][0][2] for g in bad), sum(g["trace"][1][2] for g in bad),
                sum(x[2] for g in bad for x in g["trace"][2:])]
        whole = max(sum(part), 1e-9)
        if not bad:
            print(f"{os.path.basename(f)}: no gap at or over {args.split} s")
            continue
        print(f"{os.path.basename(f)}: at or over {args.split} s {len(bad)} (frames {min(g['frames'] for g in bad)}-"
              f"{max(g['frames'] for g in bad)}), silence {whole:.2f} s; in the last sound {part[0] / whole:.0%}, "
              f"the space {part[1] / whole:.0%}, after it {part[2] / whole:.0%}; gaps mostly after the space "
              f"{sum(sum(x[2] for x in g['trace'][2:]) > g['silence'] / 2 for g in bad)}")
        rows += [dict(table=os.path.basename(f), **{k: g[k] for k in ("sentence", "words", "silence", "trace")})
                 for g in bad]
    save(rows, args.out)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("sentences", nargs="?")
    a.add_argument("--lang", choices=["en", "de"])
    a.add_argument("--model")
    a.add_argument("--voice")
    a.add_argument("--speed", type=float, default=1.0)
    a.add_argument("--split", type=float, default=0.16, help="seconds of silence from which a gap is wrong")
    a.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "word_gaps.json"))
    a.add_argument("--label")
    a.add_argument("--hear", action="store_true", help="Whisper's word match of each sentence")
    a.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a.add_argument("--limit", type=int, help="with --compare: the limit at speed 1, in frames")
    a.add_argument("--top", type=float, help="with --compare: the top of the normal silence, in seconds")
    a.add_argument("--trace", nargs="+", metavar="TABLE", help="where the wrong gaps' silence sits")
    args = a.parse_args()
    if args.trace:
        trace(args)
    elif args.compare:
        if args.limit is None or args.top is None:
            a.error("--compare needs --limit and --top")
        compare(args)
    elif not (args.sentences and args.lang and args.model and args.voice):
        a.error("measuring needs <sentences>, --lang, --model and --voice")
    else:
        measure(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
