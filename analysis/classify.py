"""Dot/dash classifier + Morse decoder.

Key finding (see amp_study.py): absolute peak height does NOT separate dots from
dashes across sensor batches, but WITHIN one recording a dash is ~2.5x a dot.
So we classify RELATIVELY: per file, split press amplitudes into two classes
(Otsu threshold) -> low=dot, high=dash. A bimodality check flags single-type
letters (all dots / all dashes) where no within-file contrast exists.
"""
import glob, os, json
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE, MORSE_INV

EXCLUDE = {"K-30"}          # failed recording (open circuit)


def otsu_threshold(v):
    """1-D Otsu threshold that maximizes between-class variance."""
    v = np.sort(v)
    if len(v) < 2 or v[-1] == v[0]:
        return v.mean(), 0.0
    best_t, best_var = v[0], -1
    for i in range(1, len(v)):
        lo, hi = v[:i], v[i:]
        w0, w1 = len(lo) / len(v), len(hi) / len(v)
        var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, (lo.max() + hi.min()) / 2
    # separation score: gap between class means over pooled spread
    lo = v[v < best_t]; hi = v[v >= best_t]
    if len(lo) and len(hi):
        sep = (hi.mean() - lo.mean()) / (np.std(v) + 1e-9)
        ratio = hi.mean() / (lo.mean() + 1e-9)
    else:
        sep = ratio = 0.0
    return best_t, ratio


def classify_file(path):
    a = analyze_file(path)
    if a["token"] in EXCLUDE:
        return None
    amps = np.array([p[2] for p in a["presses"]])
    if len(amps) == 0:
        return None
    thr, ratio = otsu_threshold(amps)
    bimodal = ratio >= 1.6          # dash/dot mean ratio; <1.6 -> single-type file
    letter = a["letters"][0]
    code = MORSE[letter]
    n_elem = len(code)

    # decode each repetition using the per-file threshold
    decoded, correct_reps = [], 0
    for rep in a["reps"]:
        seq = "".join("-" if p[2] >= thr else "." for p in rep)
        decoded.append(seq)
        if seq == code:
            correct_reps += 1

    # if the file is single-type, the "self-decode" is meaningless (no contrast);
    # element-accuracy is what matters there and is trivially 100% once we know type
    reps_full = [r for r in a["reps"] if len(r) == n_elem]
    rep_acc = correct_reps / len(a["reps"]) if a["reps"] else 0

    # element-level accuracy on repetitions with the right element count
    el_tot = el_ok = 0
    for rep in reps_full:
        pred = ["-" if p[2] >= thr else "." for p in rep]
        for pc, tc in zip(pred, code):
            el_tot += 1
            el_ok += (pc == tc)
    el_acc = el_ok / el_tot if el_tot else float("nan")

    return dict(token=a["token"], letter=letter, code=code, n_elem=n_elem,
                R0=a["R0"], n_press=len(amps), n_reps=len(a["reps"]),
                reps_full=len(reps_full), thr=round(float(thr), 2),
                ratio=round(float(ratio), 2), bimodal=bool(bimodal),
                rep_acc=round(rep_acc, 3), el_acc=round(float(el_acc), 3))


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    rows = [r for f in files if (r := classify_file(f))]
    hdr = f"{'file':8} {'code':6} {'bim':4} {'ratio':>5} {'reps':>4} {'full':>4} {'rep_acc':>7} {'el_acc':>7}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (not x["bimodal"], -x["el_acc"] if x["el_acc"]==x["el_acc"] else 0)):
        print(f"{r['token']:8} {r['code']:6} {'Y' if r['bimodal'] else 'n':4} "
              f"{r['ratio']:5.2f} {r['n_reps']:4d} {r['reps_full']:4d} "
              f"{r['rep_acc']*100:6.0f}% {r['el_acc']*100:6.0f}%")

    mixed = [r for r in rows if r["bimodal"]]
    single = [r for r in rows if not r["bimodal"]]
    def wavg(rs, key):
        w = sum(r["reps_full"] for r in rs)
        return sum(r[key]*r["reps_full"] for r in rs)/w if w else float("nan")
    print("\n=== mixed-code letters (within-file contrast exists) ===")
    print(f"  files={len(mixed)}  weighted element-acc={wavg(mixed,'el_acc')*100:.1f}%  "
          f"weighted rep-decode-acc={wavg(mixed,'rep_acc')*100:.1f}%")
    print(f"  single-type files (no contrast, need message context): "
          f"{sorted(set(r['letter'] for r in single))}")

    with open(os.path.join(os.path.dirname(__file__), "out", "classify.json"), "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
