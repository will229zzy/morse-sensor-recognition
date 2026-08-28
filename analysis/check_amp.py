"""Verify the reference rule: dot = small peak, dash = large peak (by AMPLITUDE).
For letters with a known mix of dots and dashes, measure each element's peak height,
group into repetitions, and check whether amplitude order matches the Morse pattern."""
import glob, os
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE

# letters whose code mixes dots and dashes, with clean recordings
tests = ["A", "N", "U", "D-20", "K70", "R", "W", "G-55", "F42"]

for nm in tests:
    f = (glob.glob(os.path.join(RAW_DIR, f"{nm} *.csv")) or
         glob.glob(os.path.join(RAW_DIR, f"{nm}*.csv")))
    if not f:
        continue
    a = analyze_file(f[0])
    letter = a["letters"][0]
    code = MORSE[letter]
    # collect per-element amplitude, aligned to reps that have the full element count
    amps_by_pos = [[] for _ in code]
    good_reps = 0
    for rep in a["reps"]:
        if len(rep) != len(code):
            continue
        good_reps += 1
        for i, p in enumerate(rep):
            amps_by_pos[i].append(p[2])   # p[2] = peak height (ΔR/R0 %)
    means = [np.median(v) if v else float("nan") for v in amps_by_pos]
    marks = "".join("·" if c == "." else "—" for c in code)
    txt = "  ".join(f"{marks[i]}:{means[i]:5.1f}" for i in range(len(code)))
    print(f"{letter} ({code:5}) full-reps={good_reps:3d} | median peak% per position: {txt}")
