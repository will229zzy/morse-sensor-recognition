#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面板(e):用我们自己的规则法(可解释)得到 26×26 混淆矩阵 + 总准确率,仿参考论文 Fig.5e。
每个干净重复独立解码,预测字母 vs 真实字母。无需训练、无需黑箱。"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import morse_sensor as ms

RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
idx = {c: i for i, c in enumerate(LETTERS)}

cm = np.zeros((26, 26), int)
for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    L = ms.letter_from_filename(f)
    base = os.path.basename(f)
    if L is None or "K-30" in base or "D13+F58" in base:   # 排除坏文件与混合文件
        continue
    sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
    taps = ms.detect_taps(sec, rel); k = len(ms.MORSE[L])
    wt = ms.message_width_threshold(taps)
    reps = [r for r in ms._regroup_by_count(taps, k) if len(r) == k]
    for r in reps:
        ms.classify_group(r, wt)
        pred = ms.MORSE_INV.get("".join(t.symbol for t in r), None)
        if pred in idx:
            cm[idx[L], idx[pred]] += 1
        # 无效码计入自身行的"漏判"——这里跳过(极少)

total = cm.sum(); correct = np.trace(cm); acc = correct / total * 100
print(f"总样本 {total},正确 {correct},准确率 {acc:.1f}%")
offdiag = [(LETTERS[i], LETTERS[j], cm[i, j]) for i in range(26) for j in range(26)
           if i != j and cm[i, j] > 0]
print("非对角(真→判,次数):", offdiag if offdiag else "无(完全对角)")

fig, ax = plt.subplots(figsize=(7.2, 6.6))
im = ax.imshow(cm, cmap="Purples", vmin=0)
ax.set_xticks(range(26)); ax.set_xticklabels(list(LETTERS), fontsize=7)
ax.set_yticks(range(26)); ax.set_yticklabels(list(LETTERS), fontsize=7)
ax.set_xlabel("Predicted labels", fontsize=10); ax.set_ylabel("True labels", fontsize=10)
ax.set_title(f"Accuracy: {acc:.1f}%", fontsize=13)
import matplotlib.patches as mpatches
for i in range(26):
    for j in range(26):
        if cm[i, j]:
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5.5,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "#444")
        if i != j and cm[i, j]:                       # 高亮识别错误的格子
            ax.add_patch(mpatches.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                            edgecolor="#d81e05", lw=1.8))
            ax.annotate(f"{LETTERS[i]}→{LETTERS[j]} ({int(cm[i,j])})",
                        xy=(j, i), xytext=(j + 3.5, i - 1.5), fontsize=8, color="#d81e05",
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color="#d81e05", lw=1))
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "out", "fig_e_confusion.png")
plt.savefig(out, dpi=150, facecolor="white"); print("saved", out)
