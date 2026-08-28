"""Final evaluation of the letter recognizer, with honest, audience-ready numbers.

Method (plain language): find each finger-tap in the resistance trace, measure how
BIG each tap is, call small taps a dot and big taps a dash (the size threshold is set
per recording, because a 'big' tap on one sensor equals a 'small' tap on another),
read the dot/dash pattern off the Morse table, and output the letter.

We report:
  * dot/dash element accuracy on cleanly-segmented repetitions
  * whole-letter decode accuracy
  * a per-letter breakdown, split into 'has size contrast' vs 'single-type' letters
"""
import glob, os, json
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE
from classify import otsu_threshold, EXCLUDE

# letters whose Morse code contains BOTH dots and dashes -> size contrast exists
MIXED = {L for L, c in MORSE.items() if "." in c and "-" in c}
PURE = set(MORSE) - MIXED  # E,H,I,S (all dots) / T,M,O (all dashes) — no contrast alone


def eval_file(path):
    a = analyze_file(path)
    if a["token"] in EXCLUDE:
        return None
    amps = np.array([p[2] for p in a["presses"]])
    if len(amps) == 0:
        return None
    thr, ratio = otsu_threshold(amps)
    L = a["letters"][0]; code = MORSE[L]; k = len(code)

    clean = [r for r in a["reps"] if len(r) == k]     # correctly-segmented repetitions
    el_tot = el_ok = letter_ok = 0
    for rep in clean:
        pred = "".join("-" if p[2] >= thr else "." for p in rep)
        for pc, tc in zip(pred, code):
            el_tot += 1; el_ok += (pc == tc)
        letter_ok += (pred == code)
    return dict(token=a["token"], letter=L, code=code, k=k,
                mixed=(L in MIXED), n_reps=len(a["reps"]), clean=len(clean),
                seg_rate=round(len(clean) / len(a["reps"]), 3) if a["reps"] else 0,
                el_tot=el_tot, el_ok=el_ok,
                el_acc=round(el_ok / el_tot, 3) if el_tot else None,
                letter_ok=letter_ok,
                letter_acc=round(letter_ok / len(clean), 3) if clean else None)


files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
rows = [r for f in files if (r := eval_file(f))]

# aggregate per letter (merge multiple files of the same letter)
per_letter = {}
for r in rows:
    d = per_letter.setdefault(r["letter"], dict(letter=r["letter"], code=r["code"],
        mixed=r["mixed"], el_tot=0, el_ok=0, clean=0, letter_ok=0, n_reps=0))
    for key in ("el_tot", "el_ok", "clean", "letter_ok", "n_reps"):
        d[key] += r[key]
for d in per_letter.values():
    d["el_acc"] = round(d["el_ok"]/d["el_tot"], 3) if d["el_tot"] else None
    d["letter_acc"] = round(d["letter_ok"]/d["clean"], 3) if d["clean"] else None

def agg(letters):
    et = sum(per_letter[L]["el_tot"] for L in letters if L in per_letter)
    eo = sum(per_letter[L]["el_ok"] for L in letters if L in per_letter)
    cl = sum(per_letter[L]["clean"] for L in letters if L in per_letter)
    lo = sum(per_letter[L]["letter_ok"] for L in letters if L in per_letter)
    return (eo/et if et else 0, lo/cl if cl else 0, et, cl)

print("per-letter (merged across files):")
print(f"{'L':2} {'code':5} {'type':6} {'reps':>5} {'clean':>5} {'elem':>5} {'el_acc':>7} {'let_acc':>7}")
for L in sorted(per_letter):
    d = per_letter[L]
    print(f"{L:2} {d['code']:5} {'mixed' if d['mixed'] else 'pure':6} "
          f"{d['n_reps']:5d} {d['clean']:5d} {d['el_tot']:5d} "
          f"{(d['el_acc'] or 0)*100:6.0f}% {(d['letter_acc'] or 0)*100:6.0f}%")

ma, ml, met, mcl = agg(MIXED)
pa, pl, pet, pcl = agg(PURE)
print(f"\nMIXED letters (18): dot/dash element acc = {ma*100:.1f}%  "
      f"(n={met})   whole-letter decode = {ml*100:.1f}% (n={mcl})")
print(f"PURE  letters ( 8): element acc = {pa*100:.1f}% (n={pet}) "
      f"-- expected ~chance: a lone letter has no size reference")

out = dict(per_letter=list(per_letter.values()),
           summary=dict(mixed_el_acc=round(ma, 4), mixed_letter_acc=round(ml, 4),
                        mixed_n=met, pure_el_acc=round(pa, 4), pure_n=pet))
with open(os.path.join(os.path.dirname(__file__), "out", "evaluate.json"), "w") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
print("\nwrote out/evaluate.json")
