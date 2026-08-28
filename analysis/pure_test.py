"""Can we tell a 'small' tap (dot) from a 'big' tap (dash) for SINGLE-TYPE letters,
where there is no within-recording contrast? Test three notions of tap size:
  height  = peak ΔR/R0 (%)
  width   = press duration (s)
  area    = integral of ΔR/R0 over the press (%·s)  -> captures 'longer AND harder'
Report per file so we can see whether pure-dot files sit below pure-dash files."""
import glob, os
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE
from classify import EXCLUDE

PURE_DOT = {"E", "H", "I", "S"}
PURE_DASH = {"T", "M", "O"}

def tap_area(a, p):
    m = (a["sec"] >= p[0]) & (a["sec"] <= p[1])
    if m.sum() < 2:
        return p[2] * p[3]
    return float(np.trapz(a["x"][m], a["sec"][m]))

rows = []
for f in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    a = analyze_file(f)
    if a["token"] in EXCLUDE or not a["presses"]:
        continue
    L = a["letters"][0]
    h = np.array([p[2] for p in a["presses"]])
    w = np.array([p[3] for p in a["presses"]])
    ar = np.array([tap_area(a, p) for p in a["presses"]])
    typ = "DOT " if L in PURE_DOT else "DASH" if L in PURE_DASH else "mix"
    rows.append((a["token"], L, typ, a["R0"], np.median(h), np.median(w), np.median(ar)))

print(f"{'file':8} {'L':2} {'type':4} {'R0':>7} {'height%':>7} {'width_s':>7} {'area%s':>7}")
for r in sorted(rows, key=lambda x: (x[2], x[1])):
    print(f"{r[0]:8} {r[1]:2} {r[2]:4} {r[3]:7.0f} {r[4]:7.2f} {r[5]:7.2f} {r[6]:7.2f}")

for feat, idx in [("height", 4), ("width", 5), ("area", 6)]:
    dot = [r[idx] for r in rows if r[2] == "DOT "]
    dash = [r[idx] for r in rows if r[2] == "DASH"]
    print(f"\n{feat:6}: pure-DOT files range {min(dot):.2f}-{max(dot):.2f} | "
          f"pure-DASH files range {min(dash):.2f}-{max(dash):.2f} | "
          f"{'SEPARABLE' if max(dot) < min(dash) else 'OVERLAP'}")
