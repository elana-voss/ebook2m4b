"""Silences between two words with no punctuation between them, for milestone 24 ("Silence between words").

  python tools/word_gaps.py <sentences> --lang en|de --model <model.onnx> --voice <voice.bin>
      [--speed 1] [--split 0.16] [--out <table.json>] [--label <name>]

<sentences>: a text file, one sentence a line, or a JSON list of [before, after, sentence] whose pause sits
between "before" and "after" (those pairs are counted as tagged). The product's phonemes come from the
ignored test pronounce::tests::phonemes_of_lines (cargo, built as CLAUDE.md says). The model runs on the
GPU (onnxruntime-gpu, CUDA) with its per-token frames exposed as an extra output: Kokoro's
/encoder/Clip_output_0, Kerstin's `duration`; 600 samples a frame at 24 kHz.

A gap is the design's token pair: token i a space (id 16) whose neighbours both have id 17 or more; its
frames are those of token i-1 (the word's last sound) and token i; its silence is the longest run of 10 ms
windows under -45 dBFS from token i-1's start to the end of the first sound after the space (a stress mark
before it taken with it). A gap with silence at or over --split is "wrong", the rest "normal".

24b runs it on the patched and the unpatched model: the JSON holds each sentence's frames per token, so
every other sound's length can be compared too. Prints a few summary lines; the table goes to --out.
"""
import argparse, json, os, re, subprocess, sys, tempfile

import numpy as np

RATE, FRAME, WIN = 24_000, 600, 240
QUIET = 10 ** (-45 / 20)
SPACE, FIRST_SOUND, ARROWS = 16, 17, range(169, 174)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ebook2m4b", "models", "kokoro", "tokenizer.json")
FRAMES_OUT = {"/encoder/Clip_output_0", "duration"}


def sentences(path):
    """The sentences and, for a JSON input, each one's tagged pause as (before, after)."""
    if path.endswith(".json"):
        items = json.load(open(path, encoding="utf-8"))
        return [s for _, _, s in items], [(a, b) for a, b, _ in items]
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    return lines, [None] * len(lines)


def phonemes(lines, lang):
    """The product's phonemes of each line (pronounce.rs through its ignored test)."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
        open(src, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        env = dict(os.environ, WORD_GAPS_IN=src, WORD_GAPS_OUT=dst, WORD_GAPS_LANG=lang)
        r = subprocess.run(["cargo", "test", "--release", "-j", "4", "phonemes_of_lines", "--", "--ignored"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode:
            sys.exit("phonemes failed:\n" + r.stderr[-2000:])
        out = open(dst, encoding="utf-8").read().split("\n")[: len(lines)]
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


def silence(audio, lo, hi):
    """The longest run of samples whose 10 ms RMS is under -45 dBFS in audio[lo:hi], in seconds."""
    rms = np.sqrt(np.convolve(audio.astype(np.float64) ** 2, np.ones(WIN) / WIN, "same"))
    best = run = 0
    for q in rms[lo:hi] < QUIET:
        run = run + 1 if q else 0
        best = max(best, run)
    return best / RATE


def tagged_word(sentence, pause):
    """The index of the word after the tagged pause, counted in whitespace-parted words."""
    a, b = pause
    at = sentence.find(a + " " + b)
    return None if at < 0 else len((sentence[: at + len(a)]).split())


def measure(args):
    vocab = {k: v for k, v in json.load(open(TOKENIZER, encoding="utf-8"))["model"]["vocab"].items()}
    stress = {vocab["ˈ"], vocab["ˌ"]}
    style = np.fromfile(args.voice, dtype=np.float32).reshape(-1, 256)
    lines, pauses = sentences(args.sentences)
    ps = phonemes(lines, args.lang)
    sess, frames_name = session(args.model)
    rows, per_sentence, arrows_seen = [], [], 0
    for n, (line, p, pause) in enumerate(zip(lines, ps, pauses)):
        if len(p) > 510:
            sys.exit(f"sentence {n + 1} has {len(p)} phonemes, over Kokoro's 510: shorten it")
        ids = [0] + [vocab[c] for c in p if c in vocab] + [0]
        arrows_seen += sum(i in ARROWS for i in ids)
        wav, dur = sess.run(["waveform", frames_name], {
            "input_ids": np.array([ids], dtype=np.int64),
            "style": style[len(p) - 1 : len(p)],
            "speed": np.array([args.speed], dtype=np.float32)})
        audio, dur = wav.reshape(-1), np.rint(np.asarray(dur, dtype=np.float64).reshape(-1)).astype(int).tolist()
        start = np.cumsum([0] + dur) * FRAME
        per_sentence.append({"phonemes": p, "ids": ids, "frames": dur})
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
            sil = silence(audio, int(start[i - 1]), int(start[min(end, len(ids))]))
            rows.append({"sentence": n + 1, "words": words[word - 1] + " " + words[word],
                         "last": dur[i - 1], "gap": dur[i], "frames": dur[i - 1] + dur[i],
                         "silence": round(sil, 4), "wrong": sil >= args.split,
                         "arrow": ids[i - 1] in ARROWS or ids[i + 1] in ARROWS,
                         "tagged": tag == word})
        if pause and tag is None:
            print(f"sentence {n + 1}: its tagged pause could not be placed", file=sys.stderr)
    json.dump({"model": args.model, "voice": args.voice, "speed": args.speed, "split": args.split,
               "gaps": rows, "sentences": per_sentence},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
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


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("sentences")
    a.add_argument("--lang", choices=["en", "de"], required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--voice", required=True)
    a.add_argument("--speed", type=float, default=1.0)
    a.add_argument("--split", type=float, default=0.16, help="seconds of silence from which a gap is wrong")
    a.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "word_gaps.json"))
    a.add_argument("--label")
    measure(a.parse_args())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
