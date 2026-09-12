#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出可直接用 Origin 画图的数据(CSV):
  out/origin/confusion_matrix.csv     26×26 混淆矩阵(行=真实,列=预测)→ Origin 热图/矩阵图
  out/origin/waveforms/<L>.csv        每字母:第1列 time_s,其后各列为若干次重复的 ΔR/R₀(%)
                                      → Origin 选中所有列画叠加线图(面板 c)
  out/origin/per_letter_accuracy.csv  逐字母准确率(条形图用)
  out/origin/README.txt               Origin 画图说明
"""
import glob, os, sys
import numpy as np
import pandas as pd
import morse_sensor as ms
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))
import dataset as ds          # 统一前端:跳过仪器故障段 + 拆开两字母混录文件

RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")
OUT = os.path.join(os.path.dirname(__file__), "out", "origin")
WAVE = os.path.join(OUT, "waveforms")
os.makedirs(WAVE, exist_ok=True)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N_REPS = 15                        # 每字母导出多少条重复(叠加曲线)
DT = 1 / 2.37

BLOCKS = {}                        # 字母 -> 干净重复最多的那段录制
for _f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    for _L, _sec, _rel, _reps, _wt in ds.letter_blocks(_f):
        if _L not in BLOCKS or len(_reps) > len(BLOCKS[_L][2]):
            BLOCKS[_L] = (_sec, _rel, _reps, _wt)


def best_file(L):
    return BLOCKS.get(L)


# ---------- (c) 每字母波形 CSV ----------
for L in LETTERS:
    got = best_file(L)
    if not got:
        continue
    sec, rel, reps, _ = got
    clips = []
    for r in reps[3:3 + N_REPS]:
        i0 = int(np.searchsorted(sec, r[0].t_start - 1.5))
        i1 = int(np.searchsorted(sec, r[-1].t_end + 1.5))
        seg = rel[max(0, i0):i1].astype(float)
        seg = seg - np.percentile(seg, 10)
        if len(seg) > 3:
            clips.append(seg)
    if not clips:
        continue
    Lmax = max(len(c) for c in clips)
    grid = np.arange(Lmax) * DT                       # 公共时间轴
    data = {"time_s": np.round(grid, 3)}
    for i, c in enumerate(clips, 1):
        t = np.arange(len(c)) * DT
        data[f"rep_{i:02d}"] = np.round(np.interp(grid, t, c, right=0.0), 4)
    pd.DataFrame(data).to_csv(os.path.join(WAVE, f"{L}.csv"), index=False)

# ---------- (e) 混淆矩阵 CSV + 逐字母准确率 ----------
idx = {c: i for i, c in enumerate(LETTERS)}
cm = np.zeros((26, 26), int)
for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    for L, sec, rel, reps, wt in ds.letter_blocks(f):
        for r in reps:
            ms.classify_group(r, wt)
            p = ms.MORSE_INV.get("".join(t.symbol for t in r))
            if p in idx:
                cm[idx[L], idx[p]] += 1

dfcm = pd.DataFrame(cm, index=list(LETTERS), columns=list(LETTERS))
dfcm.index.name = "True\\Pred"
dfcm.to_csv(os.path.join(OUT, "confusion_matrix.csv"))
# 归一化(百分比,每行)版本,画热图更直观
dfpct = (dfcm.T / dfcm.sum(1).replace(0, 1)).T * 100
dfpct.round(2).to_csv(os.path.join(OUT, "confusion_matrix_percent.csv"))

acc = np.diag(cm) / cm.sum(1).clip(min=1) * 100
pd.DataFrame({"letter": list(LETTERS), "n_samples": cm.sum(1),
             "accuracy_%": np.round(acc, 2)}).to_csv(
    os.path.join(OUT, "per_letter_accuracy.csv"), index=False)

total = cm.sum(); correct = int(np.trace(cm))
errs = [(LETTERS[i], LETTERS[j], int(cm[i, j])) for i in range(26) for j in range(26)
        if i != j and cm[i, j]]

with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
    fh.write(f"""Origin 画图数据说明
================================
总样本 {total},正确 {correct},总准确率 {correct/total*100:.2f}%
识别错误(真->判,次数): {errs}

1) 面板(c) 波形叠加图  ——  waveforms/<字母>.csv
   - 第 1 列 time_s = 时间(秒);其余每列 rep_XX = 一次重复的 ΔR/R0 (%)
   - Origin: 导入某字母的 csv -> 选中 time_s 作 X、所有 rep_* 作 Y -> Plot: Line
   - 26 个字母各画一个小图,排成 2x13,即得面板(c)

2) 面板(e) 混淆矩阵  ——  confusion_matrix.csv(计数) / confusion_matrix_percent.csv(每行百分比)
   - 行 = 真实字母,列 = 预测字母
   - Origin: 导入 -> 设为矩阵(Matrix)-> Plot: Heatmap/Image;
     想让少量错误可见,可用对数色标,或在对应格子单独标注数值
   - 标题写 Accuracy: {correct/total*100:.1f}%

3) 逐字母准确率  ——  per_letter_accuracy.csv
   - letter / n_samples / accuracy_%  -> Origin 条形图

坐标轴:纵轴 ΔR/R0 (%)(相对电阻变化),横轴 时间 (s),与参考论文 Fig.5 一致。
""")

print(f"导出完成 -> {OUT}")
print(f"  waveforms/ : {len(os.listdir(WAVE))} 个字母 CSV")
print(f"  confusion_matrix.csv (+_percent)  总准确率 {correct/total*100:.2f}%  错误 {errs}")
print(f"  per_letter_accuracy.csv, README.txt")
