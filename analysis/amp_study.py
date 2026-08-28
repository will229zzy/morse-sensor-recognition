"""Establish the dot/dash amplitude boundary.
Single-type letters give ground-truth labels without any grouping:
  dots  = E(.) H(....) I(..) S(...)
  dashes= T(-) M(--) O(---)
We compare raw peak height and several per-file normalizations to find a rule
that separates dots from dashes across sensor batches."""
import glob, os
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR

DOT_FILES = ["E", "H", "I", "S"]
DASH_FILES = ["T", "M", "O"]

def peaks(nm):
    f = (glob.glob(os.path.join(RAW_DIR, f"{nm} *.csv")) or
         glob.glob(os.path.join(RAW_DIR, f"{nm}*.csv")))
    a = analyze_file(f[0])
    amp = np.array([p[2] for p in a["presses"]])
    return a, amp

print(f"{'file':4} {'type':4} {'R0':>7} {'n':>4} {'amp_med':>7} {'amp_p25':>7} {'amp_p75':>7}")
rows = []
for nm in DOT_FILES + DASH_FILES:
    a, amp = peaks(nm)
    typ = "dot" if nm in DOT_FILES else "dash"
    print(f"{nm:4} {typ:4} {a['R0']:7.0f} {len(amp):4d} "
          f"{np.median(amp):7.2f} {np.percentile(amp,25):7.2f} {np.percentile(amp,75):7.2f}")
    for v in amp:
        rows.append((nm, typ, a["R0"], v))

# Global separability of RAW amplitude
dots = np.array([r[3] for r in rows if r[1] == "dot"])
dashes = np.array([r[3] for r in rows if r[1] == "dash"])
print("\n--- RAW peak height (%) ---")
print(f"dots : med={np.median(dots):.2f}  p90={np.percentile(dots,90):.2f}")
print(f"dash : med={np.median(dashes):.2f}  p10={np.percentile(dashes,10):.2f}")
# best single global threshold
allv = np.r_[dots, dashes]; y = np.r_[np.zeros(len(dots)), np.ones(len(dashes))]
best_t, best_acc = 0, 0
for t in np.linspace(allv.min(), allv.max(), 400):
    acc = ((allv >= t) == (y == 1)).mean()
    if acc > best_acc:
        best_acc, best_t = acc, t
print(f"best global RAW threshold = {best_t:.2f}%  -> acc={best_acc*100:.1f}%")
