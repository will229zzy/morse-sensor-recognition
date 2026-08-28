"""Consolidated final metrics under the hybrid rule (height-contrast OR duration).
Outputs out/final.json for the report and prints the headline numbers."""
import glob, os, json
from collections import Counter
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE, MORSE_INV
from recognize_all import label_taps, WIDTH_THR, RATIO_MIN
from classify import otsu_threshold, EXCLUDE

per = {}
el_tot = el_ok = 0
for f in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    a = analyze_file(f)
    if a["token"] in EXCLUDE or not a["presses"]:
        continue
    L = a["letters"][0]; code = MORSE[L]; k = len(code)
    seqs, mode = label_taps(a["reps"], a["presses"])
    d = per.setdefault(L, dict(L=L, code=code, marks=code.replace('.', '·').replace('-', '—'),
                               mode=mode, clean=0, letter_ok=0, el_tot=0, el_ok=0, preds=[]))
    d["mode"] = mode
    for s, rep in zip(seqs, a["reps"]):
        if len(rep) != k:
            continue
        d["clean"] += 1
        d["letter_ok"] += (s == code)
        for pc, tc in zip(s, code):
            d["el_tot"] += 1; d["el_ok"] += (pc == tc); el_tot += 1; el_ok += (pc == tc)
        d["preds"].append(MORSE_INV.get(s, "?"))

letters_ok = 0
for d in per.values():
    d["letter_acc"] = d["letter_ok"] / d["clean"] if d["clean"] else None
    d["el_acc"] = d["el_ok"] / d["el_tot"] if d["el_tot"] else None
    d["pred"] = Counter(d["preds"]).most_common(1)[0][0] if d["preds"] else "?"
    d["ok"] = (d["pred"] == d["L"])
    letters_ok += d["ok"]

tot_clean = sum(d["clean"] for d in per.values())
tot_lok = sum(d["letter_ok"] for d in per.values())
summary = dict(letters_ok=letters_ok, letters_total=26,
               element_acc=round(el_ok / el_tot, 4),
               per_rep_letter_acc=round(tot_lok / tot_clean, 4),
               clean_reps=tot_clean, width_thr=WIDTH_THR)
print("HEADLINE")
print(f"  letters read correctly : {letters_ok}/26")
print(f"  dot/dash element acc   : {el_ok/el_tot*100:.1f}%  ({el_ok}/{el_tot})")
print(f"  whole-letter (per rep) : {tot_lok/tot_clean*100:.1f}%  ({tot_lok}/{tot_clean})")
fails = [d["L"] for d in per.values() if not d["ok"]]
print(f"  still wrong            : {sorted(fails)}  (ambiguous recordings: dot/dash not distinct)")

out = dict(summary=summary,
           letters=[{k: v for k, v in d.items() if k != "preds"}
                    for d in sorted(per.values(), key=lambda x: x["L"])])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "final.json"), "w"),
          indent=2, ensure_ascii=False)
print("\nwrote out/final.json")
