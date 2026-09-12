#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面板(c):26 个字母的 ΔR/R₀ 波形叠加图(每字母多条重复叠加),风格仿参考论文 Fig.5c。"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import morse_sensor as ms
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))
import dataset as ds          # 统一前端:跳过仪器故障段 + 拆开两字母混录文件

RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")
N_OVER = 12            # 每字母叠加多少条重复
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 26 个区分色
CMAP = plt.cm.hsv(np.linspace(0, 1, 27))[:26]

_BEST = {}
for _f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    for _L, _sec, _rel, _reps, _ in ds.letter_blocks(_f):
        if _L not in _BEST or len(_reps) > len(_BEST[_L][2]):
            _BEST[_L] = (_sec, _rel, _reps)


def best_reps(L):
    """取该字母干净重复最多的一段录制,返回若干条重复的(时间,波形)。"""
    if L not in _BEST:
        return []
    sec, rel, reps = _BEST[L]
    out = []
    for r in reps:
        i0 = int(np.searchsorted(sec, r[0].t_start - 1.5))
        i1 = int(np.searchsorted(sec, r[-1].t_end + 1.5))
        seg = rel[max(0, i0):i1]
        t = np.arange(len(seg)) * (sec[1] - sec[0] if len(sec) > 1 else 0.42)
        if len(seg) > 3:
            out.append((t, seg - np.percentile(seg, 10)))
    return out


fig, axes = plt.subplots(2, 13, figsize=(18, 4.2))
for idx, L in enumerate(LETTERS):
    ax = axes[idx // 13, idx % 13]
    reps = best_reps(L)
    col = CMAP[idx]
    for t, seg in reps[3:3 + N_OVER]:
        ax.plot(t, seg, color=col, lw=0.7, alpha=0.55)
    ax.set_title(L, fontsize=13, fontweight="bold", pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color="#ccc", lw=0.4)
axes[0, 0].set_ylabel("ΔR/R₀ (%)", fontsize=10)
axes[1, 0].set_ylabel("ΔR/R₀ (%)", fontsize=10)
plt.tight_layout(pad=0.4)
out = os.path.join(os.path.dirname(__file__), "out", "fig_c_waveforms.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
