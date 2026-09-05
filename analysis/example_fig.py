"""峰高示例图(D / F):用 morse_sensor 的检测,标注每次按压的点/划。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import glob, os
import numpy as np
import morse_sensor as ms

zh = None
for p in ["/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf",
          "/System/Library/Fonts/STHeiti Medium.ttc"]:
    if os.path.exists(p):
        zh = fm.FontProperties(fname=p); break
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})


RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")

def reps_of(nm):
    f = glob.glob(os.path.join(RAW, f"{nm} *.csv")) or glob.glob(os.path.join(RAW, f"{nm}*.csv"))
    sec, R = ms.load_keysight_csv(f[0]); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
    taps = ms.detect_taps(sec, rel)
    L = ms.letter_from_filename(f[0]); code = ms.MORSE[L]; k = len(code)
    reps = [r for r in ms._regroup_by_count(taps, k) if len(r) == k]
    for r in reps:
        ms.classify_group(r)
    return sec, rel, reps, L, code


fig, axes = plt.subplots(2, 1, figsize=(9, 4.6))
for ax, nm in [(axes[0], "D-20"), (axes[1], "F42")]:
    sec, rel, reps, L, code = reps_of(nm)
    marks = "".join("—" if c == "-" else "·" for c in code)
    chosen = reps[2:6]
    t0 = chosen[0][0].t_start - 2; t1 = chosen[-1][-1].t_end + 2
    m = (sec >= t0) & (sec <= t1)
    ax.plot(sec[m] - t0, rel[m], color="#c0392b", lw=1.4)
    for r in chosen:
        for t in r:
            lab = "—" if t.symbol == "-" else "·"
            col = "#8e44ad" if t.symbol == "-" else "#27ae60"
            ax.annotate(lab, (t.t_center - t0, t.height + 0.4), ha="center",
                        color=col, fontsize=13, fontweight="bold")
    ax.set_ylabel("敲击大小 ΔR/R₀ (%)", fontsize=9, fontproperties=zh)
    ax.set_title(f"字母 {L}  =  {marks}    (大敲 = 划，小敲 = 点)",
                 fontsize=11, loc="left", fontproperties=zh)
    ax.margins(x=0.01); ax.set_ylim(bottom=-1)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("时间 (s)", fontsize=9, fontproperties=zh)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "out", "example.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
