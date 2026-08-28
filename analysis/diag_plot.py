import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, os, sys
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR

names = sys.argv[1:] or ["E", "C-30", "K70", "B-64", "K-30", "D-20", "G-55"]
fig, axes = plt.subplots(len(names), 1, figsize=(14, 2.2 * len(names)))
if len(names) == 1:
    axes = [axes]
for ax, nm in zip(axes, names):
    f = glob.glob(os.path.join(RAW_DIR, f"{nm} *.csv")) or glob.glob(os.path.join(RAW_DIR, f"{nm}*.csv"))
    a = analyze_file(f[0])
    ax.plot(a["sec"], a["x"], lw=0.5, color="crimson")
    for p in a["presses"]:
        ax.axvspan(p[0], p[1], color="orange", alpha=0.25)
    # rep boundaries
    for rep in a["reps"]:
        if rep:
            ax.axvline(rep[0][0], color="navy", lw=0.4, alpha=0.5)
    ax.set_title(f"{a['token']}  Ek={a['n_elem']}  #press={len(a['presses'])}  "
                 f"reps={len(a['reps'])}  num={'+'.join(str(n) for n in a['numbers'])}",
                 fontsize=9, loc="left")
    ax.set_ylabel("ΔR/R₀ %", fontsize=7)
    ax.margins(x=0.005)
axes[-1].set_xlabel("time (s)")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "out", "diag.png")
plt.savefig(out, dpi=90)
print("saved", out)
