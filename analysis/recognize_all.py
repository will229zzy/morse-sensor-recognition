"""Full A-Z recognizer with a hybrid dot/dash rule.

A dash is a BIGGER tap than a dot -- bigger in height AND longer in duration.
  * If a recording contains both sizes (dash/dot height ratio >= 1.6), the contrast
    is obvious -> classify each tap by RELATIVE height (Otsu split).
  * If a recording is all one type (a pure letter: E H I S T M O ...), there is no
    internal contrast, but a dash press still lasts longer -> classify the whole file
    by its median press DURATION (learned threshold ~3.6 s). Height fails here because
    press force varies between sessions; duration is a human action and stays stable.
"""
import glob, os, json
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE, MORSE_INV
from classify import otsu_threshold, EXCLUDE

WIDTH_THR = 3.6      # seconds; dot press < WIDTH_THR <= dash press (learned)
RATIO_MIN = 1.6      # height ratio above which a file is treated as having contrast


def label_taps(reps, presses):
    """Return a function-free list: for each rep, the decoded dot/dash string."""
    heights = np.array([p[2] for p in presses])
    thr_h, ratio = otsu_threshold(heights)
    if ratio >= RATIO_MIN:
        mode = "height (relative)"
        rule = lambda p: "-" if p[2] >= thr_h else "."
    else:
        mode = "width (absolute)"
        med_w = np.median([p[3] for p in presses])
        file_is_dash = med_w >= WIDTH_THR
        rule = lambda p, d=file_is_dash: "-" if d else "."
    return [ "".join(rule(p) for p in rep) for rep in reps ], mode


def recognize(path):
    a = analyze_file(path)
    if a["token"] in EXCLUDE or not a["presses"]:
        return None
    L = a["letters"][0]; code = MORSE[L]; k = len(code)
    seqs, mode = label_taps(a["reps"], a["presses"])
    clean = [s for s, rep in zip(seqs, a["reps"]) if len(rep) == k]
    # majority-voted prediction for the whole file (what letter does it read as?)
    from collections import Counter
    votes = Counter(MORSE_INV.get(s, "?") for s in clean)
    pred = votes.most_common(1)[0][0] if clean else "?"
    letter_ok = sum(s == code for s in clean)
    return dict(token=a["token"], L=L, code=code, k=k, mode=mode,
                n_reps=len(a["reps"]), clean=len(clean),
                letter_ok=letter_ok,
                acc=letter_ok / len(clean) if clean else None,
                pred=pred, pred_ok=(pred == L))


rows = [r for f in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))) if (r := recognize(f))]

# merge files of the same letter
per = {}
for r in rows:
    d = per.setdefault(r["L"], dict(L=r["L"], code=r["code"], mode=r["mode"],
                                    clean=0, letter_ok=0, preds=[]))
    d["clean"] += r["clean"]; d["letter_ok"] += r["letter_ok"]; d["preds"].append(r["pred"])
for d in per.values():
    d["acc"] = d["letter_ok"] / d["clean"] if d["clean"] else None
    from collections import Counter
    d["pred"] = Counter(d["preds"]).most_common(1)[0][0]

print(f"{'L':2} {'code':5} {'rule':18} {'clean':>5} {'letter-acc':>10} {'reads as':>9}")
print("-" * 56)
n_ok = 0
for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    d = per[L]
    ok = "✓" if d["pred"] == L else "✗ ("+d["pred"]+")"
    n_ok += (d["pred"] == L)
    acc = "  n/a" if d["acc"] is None else f"{d['acc']*100:5.0f}%"
    print(f"{L:2} {d['code']:5} {d['mode']:18} {d['clean']:5d} {acc:>10} {ok:>9}")

tot_clean = sum(d["clean"] for d in per.values())
tot_ok = sum(d["letter_ok"] for d in per.values())
print("-" * 56)
print(f"letters read correctly (majority vote): {n_ok}/26")
print(f"per-repetition whole-letter accuracy   : {tot_ok/tot_clean*100:.1f}%  "
      f"({tot_ok}/{tot_clean} clean repetitions)")

json.dump([{k: v for k, v in d.items() if k != "preds"} for d in per.values()],
          open(os.path.join(os.path.dirname(__file__), "out", "recognize_all.json"), "w"),
          indent=2, ensure_ascii=False)
