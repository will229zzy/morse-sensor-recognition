#!/usr/bin/env python3
"""Recognize the letter(s) in one raw resistance recording.

Usage:
    python recognize.py "../raw data/D-20 2026-08-19 13-50-36 0.csv"

Prints the taps it found, whether each is a dot or a dash (small tap vs big tap),
and the decoded letter(s), then compares against the true letter in the filename.
"""
import sys, os
import numpy as np
from morse_pipeline import analyze_file, MORSE, MORSE_INV
from classify import otsu_threshold


def recognize(path):
    a = analyze_file(path)
    amps = np.array([p[2] for p in a["presses"]])
    if len(amps) == 0:
        print("no taps detected"); return
    thr, ratio = otsu_threshold(amps)
    true_letter = a["letters"][0]
    code = MORSE[true_letter]
    print(f"file            : {a['base']}")
    print(f"true letter     : {true_letter}  ({code})")
    print(f"taps detected   : {len(amps)}   size threshold: {thr:.1f}%  "
          f"(dash/dot ratio {ratio:.1f})")
    print(f"contrast        : {'yes (dot/dash separable)' if ratio >= 1.6 else 'no (single-type letter, needs message context)'}")

    reps = [r for r in a["reps"] if len(r) == len(code)]
    correct = 0
    for rep in reps[:8]:
        seq = "".join("-" if p[2] >= thr else "." for p in rep)
        pred = MORSE_INV.get(seq, "?")
        ok = "OK" if pred == true_letter else "x"
        sizes = " ".join(f"{p[2]:.1f}%" for p in rep)
        print(f"   rep -> [{sizes}]  =>  {seq:5} => {pred}  {ok}")
    for rep in reps:
        seq = "".join("-" if p[2] >= thr else "." for p in rep)
        correct += (MORSE_INV.get(seq) == true_letter)
    if reps:
        print(f"decode accuracy : {correct}/{len(reps)} clean repetitions = "
              f"{correct/len(reps)*100:.0f}% recognized as '{true_letter}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    recognize(sys.argv[1])
