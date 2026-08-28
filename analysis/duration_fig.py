"""Illustrate the duration rule: I (··, two short taps) vs M (——, two long taps).
Heights are similar; only the press DURATION tells dot from dash for pure letters."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob, os
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR

zh = None
for p in ["/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf",
          "/System/Library/Fonts/STHeiti Medium.ttc"]:
    if os.path.exists(p):
        zh = fm.FontProperties(fname=p); break
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

fig, axes = plt.subplots(2, 1, figsize=(9, 4.4))
for ax, nm, col, name, marks in [(axes[0], "I", "#27ae60", "I", "··"),
                                  (axes[1], "M", "#8e44ad", "M", "——")]:
    f = glob.glob(os.path.join(RAW_DIR, f"{nm} *.csv"))[0]
    a = analyze_file(f)
    reps = [r for r in a["reps"] if len(r) == a["n_elem"]]
    chosen = reps[3:3 + 5]
    t0 = chosen[0][0][0] - 2; t1 = chosen[-1][-1][1] + 2
    m = (a["sec"] >= t0) & (a["sec"] <= t1)
    ax.plot(a["sec"][m] - t0, a["x"][m], color=col, lw=1.5)
    ws = [p[3] for rep in chosen for p in rep]
    ax.set_title(f"字母 {name} = {marks}    每次按压时长中位数 ≈ {np.median(ws):.1f}s",
                 fontsize=11, loc="left", fontproperties=zh)
    ax.set_ylabel("ΔR/R₀ (%)", fontsize=9)
    ax.margins(x=0.01); ax.set_ylim(bottom=-1)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("时间 (s)", fontsize=9, fontproperties=zh)
fig.suptitle("纯字母:峰高相近,靠「按得多久」区分点(短)和划(长)",
             fontsize=12, fontproperties=zh, y=1.0)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "out", "duration.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
