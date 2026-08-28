"""标注示例图:D 和 F 的几次重复,标出点/划(中文标签)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob, os
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE
from classify import otsu_threshold

# 找一个可用的中文字体
zh = None
for path in ["/System/Library/Fonts/PingFang.ttc",
             "/Library/Fonts/Arial Unicode.ttf",
             "/System/Library/Fonts/STHeiti Medium.ttc",
             "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(path):
        zh = fm.FontProperties(fname=path); break
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

fig, axes = plt.subplots(2, 1, figsize=(9, 4.6))
for ax, nm, nrep in [(axes[0], "D-20", 4), (axes[1], "F42", 4)]:
    f = glob.glob(os.path.join(RAW_DIR, f"{nm}*.csv"))[0]
    a = analyze_file(f)
    L = a["letters"][0]; code = MORSE[L]
    marks = "".join("—" if c == "-" else "·" for c in code)
    thr, _ = otsu_threshold(np.array([p[2] for p in a["presses"]]))
    reps = [r for r in a["reps"] if len(r) == a["n_elem"]]
    chosen = reps[2:2 + nrep]
    t0 = chosen[0][0][0] - 2; t1 = chosen[-1][-1][1] + 2
    m = (a["sec"] >= t0) & (a["sec"] <= t1)
    ax.plot(a["sec"][m] - t0, a["x"][m], color="#c0392b", lw=1.4)
    ax.axhline(thr, color="#2980b9", ls="--", lw=1, alpha=0.8)
    ax.text(0.2, thr + 0.3, f"大小分界 {thr:.0f}%", color="#2980b9", fontsize=8, fontproperties=zh)
    for rep in chosen:
        for p in rep:
            lab = "—" if p[2] >= thr else "·"
            col = "#8e44ad" if p[2] >= thr else "#27ae60"
            ax.annotate(lab, (p[0] + (p[1]-p[0])/2 - t0, p[2] + 0.4),
                        ha="center", color=col, fontsize=13, fontweight="bold")
    ax.set_ylabel("敲击大小 ΔR/R₀ (%)", fontsize=9, fontproperties=zh)
    ax.set_title(f"字母 {L}  =  {marks}    (大敲 = 划，小敲 = 点)",
                 fontsize=11, loc="left", fontproperties=zh)
    ax.margins(x=0.01); ax.set_ylim(bottom=-1)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("时间 (s)", fontsize=9, fontproperties=zh)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "out", "example.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
