"""时长示例图:I(··,两短) vs M(——,两长)。峰高相近,靠半高宽(时长)区分。"""
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
    f = glob.glob(os.path.join(RAW, f"{nm} *.csv"))[0]
    sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
    taps = ms.detect_taps(sec, rel); code = ms.MORSE[ms.letter_from_filename(f)]
    reps = [r for r in ms._regroup_by_count(taps, len(code)) if len(r) == len(code)]
    return sec, rel, reps


fig, axes = plt.subplots(2, 1, figsize=(9, 4.4))
for ax, nm, col, name, marks in [(axes[0], "I", "#27ae60", "I", "··"),
                                 (axes[1], "M", "#8e44ad", "M", "——")]:
    sec, rel, reps = reps_of(nm)
    chosen = reps[3:8]
    t0 = chosen[0][0].t_start - 2; t1 = chosen[-1][-1].t_end + 2
    m = (sec >= t0) & (sec <= t1)
    ax.plot(sec[m] - t0, rel[m], color=col, lw=1.5)
    ws = [t.width for r in chosen for t in r]
    ax.set_title(f"字母 {name} = {marks}    每次按压半高宽中位 ≈ {np.median(ws):.1f}s",
                 fontsize=10, loc="left", fontproperties=zh, color=col)
    ax.set_ylabel("ΔR/R₀ (%)", fontsize=8, fontproperties=zh)
    ax.margins(x=0.01); ax.set_ylim(bottom=-1)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("时间 (s)", fontsize=9, fontproperties=zh)
fig.suptitle("纯字母:峰高相近,靠「按得多久」区分点(短)和划(长)",
             fontsize=12, fontproperties=zh, y=1.0)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "out", "duration.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
