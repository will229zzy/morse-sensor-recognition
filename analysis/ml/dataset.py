#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py — 把 raw data/ 里的电阻曲线整理成三种方法共用的统一数据集。

一个样本 = 一次完整的字母敲击(一个 repetition)。三种方法共用同一个前端
(去毛刺 → 去漂移 → 峰检测),只有"决策环节"不同,这样对比是受控的。

每个样本提供三种表示:
  X_seq  (N, 4, 4)  按压序列:每次按压一个 token = [相对峰高, 时长, 相对时长, 前间隔]
                    → 给 Transformer / BiLSTM
  X_feat (N, 21)    定长手工特征                        → 给 SVM / 随机森林
  X_raw  (N, 128)   重采样后的原始波形段                → 给 t-SNE 的"原始数据"面板

划分:每份文件内的重复按时间顺序切成 5 个连续块 → 5 折交叉验证。
      同一块内的重复不会跨训练/测试,并且测试块边界处各清除 1 个重复(purge),
      避免"相邻的两次按压一个在训练一个在测试"这种泄漏。
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import morse_sensor as ms  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "raw data")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
L2I = {c: i for i, c in enumerate(LETTERS)}
EXCLUDE = ("K-30", "D13+F58")        # 坏文件(接触故障)与两字母混合文件
MAX_TAPS = 4                          # A-Z 最长 4 个元素
RAW_LEN = 128                         # 原始波形重采样长度
N_FOLDS = 5
PURGE = 1                             # 测试块边界两侧各清除的重复数

FEAT_NAMES = (
    ["n_elements"]
    + [f"h_rel_{i}" for i in range(1, MAX_TAPS + 1)]
    + [f"dur_{i}" for i in range(1, MAX_TAPS + 1)]
    + [f"dur_rel_{i}" for i in range(1, MAX_TAPS + 1)]
    + [f"gap_{i}" for i in range(1, MAX_TAPS + 1)]
    + ["h_max/h_min", "dur_max/dur_min", "dur_median", "h_mean"]
)


def _token_seq(rep):
    """一次重复 → (4,4) 的 token 序列 + 真实长度。token = [h_rel, dur, dur_rel, gap_before]"""
    h = np.array([t.height for t in rep], float)
    w = np.array([t.width for t in rep], float)
    gap = np.array([0.0] + [rep[i].t_start - rep[i - 1].t_end for i in range(1, len(rep))])
    hr = h / max(h.max(), 1e-6)                # 相对峰高:组内归一(核心物理量)
    wr = w / max(w.max(), 1e-6)
    seq = np.zeros((MAX_TAPS, 4), np.float32)
    n = min(len(rep), MAX_TAPS)
    seq[:n, 0], seq[:n, 1], seq[:n, 2], seq[:n, 3] = hr[:n], w[:n], wr[:n], gap[:n]
    return seq, n


def _features(rep):
    """一次重复 → 21 维定长手工特征。"""
    h = np.array([t.height for t in rep], float)
    w = np.array([t.width for t in rep], float)
    gap = np.array([0.0] + [rep[i].t_start - rep[i - 1].t_end for i in range(1, len(rep))])
    hr = h / max(h.max(), 1e-6)
    wr = w / max(w.max(), 1e-6)

    def pad(a):
        out = np.zeros(MAX_TAPS, float)
        out[:min(len(a), MAX_TAPS)] = a[:MAX_TAPS]
        return out

    return np.concatenate([
        [len(rep)], pad(hr), pad(w), pad(wr), pad(gap),
        [h.max() / max(h.min(), 1e-6), w.max() / max(w.min(), 1e-6),
         float(np.median(w)), float(h.mean())],
    ]).astype(np.float32)


def _raw_clip(sec, rel, rep, pad_s=1.5):
    """一次重复 → 重采样到 RAW_LEN 的波形段(峰值归一,去掉基线)。"""
    i0 = int(np.searchsorted(sec, rep[0].t_start - pad_s))
    i1 = int(np.searchsorted(sec, rep[-1].t_end + pad_s))
    seg = rel[max(0, i0):max(i1, i0 + 2)].astype(float)
    if len(seg) < 3:
        return np.zeros(RAW_LEN, np.float32)
    seg = seg - np.percentile(seg, 10)
    g = np.interp(np.linspace(0, len(seg) - 1, RAW_LEN), np.arange(len(seg)), seg)
    return (g / max(g.max(), 1e-6)).astype(np.float32)


def build(verbose=True):
    """扫描 raw data/,返回统一数据集字典。"""
    X_seq, lens, X_feat, X_raw, y = [], [], [], [], []
    file_id, rep_pos, n_in_file, rule_pred = [], [], [], []
    files = []

    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        base = os.path.basename(f)
        L = ms.letter_from_filename(f)
        if L is None or any(x in base for x in EXCLUDE):
            continue
        sec, R = ms.load_keysight_csv(f)
        rel, _ = ms.detrend(ms.deglitch(R), sec)
        taps = ms.detect_taps(sec, rel)
        k = len(ms.MORSE[L])
        reps = [r for r in ms._regroup_by_count(taps, k) if len(r) == k]
        if not reps:
            continue
        wt = ms.message_width_threshold(taps)       # 无监督,不用标签
        fid = len(files)
        files.append(base)
        for j, rep in enumerate(reps):
            seq, n = _token_seq(rep)
            X_seq.append(seq); lens.append(n)
            X_feat.append(_features(rep))
            X_raw.append(_raw_clip(sec, rel, rep))
            y.append(L2I[L])
            file_id.append(fid); rep_pos.append(j); n_in_file.append(len(reps))
            # 方法① 规则解码的预测(无需训练,这里一并算好)
            ms.classify_group(rep, wt)
            sym = "".join(t.symbol for t in rep)
            rule_pred.append(L2I.get(ms.MORSE_INV.get(sym, "?"), -1))
        if verbose:
            print(f"  {base[:34]:34s} {L}  reps={len(reps):3d}  width_thr={wt:.2f}s")

    d = dict(
        X_seq=np.asarray(X_seq, np.float32), lens=np.asarray(lens, np.int64),
        X_feat=np.asarray(X_feat, np.float32), X_raw=np.asarray(X_raw, np.float32),
        y=np.asarray(y, np.int64), file_id=np.asarray(file_id, np.int64),
        rep_pos=np.asarray(rep_pos, np.int64), n_in_file=np.asarray(n_in_file, np.int64),
        rule_pred=np.asarray(rule_pred, np.int64), files=np.asarray(files),
        feat_names=np.asarray(FEAT_NAMES),
    )
    d["fold"] = _make_folds(d)
    return d


def _make_folds(d):
    """每份文件内按时间顺序切 N_FOLDS 个连续块,块号即折号。"""
    fold = np.zeros(len(d["y"]), np.int64)
    for fid in np.unique(d["file_id"]):
        m = d["file_id"] == fid
        pos, n = d["rep_pos"][m], int(m.sum())
        fold[m] = np.minimum((pos * N_FOLDS) // max(n, 1), N_FOLDS - 1)
    return fold


def split(d, fold):
    """第 fold 折的(训练索引, 测试索引)。训练集会清除紧贴测试块边界的 PURGE 个重复。"""
    test = np.where(d["fold"] == fold)[0]
    train = np.where(d["fold"] != fold)[0]
    if PURGE:
        drop = set()
        for fid in np.unique(d["file_id"]):
            tp = d["rep_pos"][(d["file_id"] == fid) & (d["fold"] == fold)]
            if len(tp) == 0:
                continue
            lo, hi = tp.min(), tp.max()
            for i in train:
                if d["file_id"][i] == fid and (lo - PURGE <= d["rep_pos"][i] < lo
                                               or hi < d["rep_pos"][i] <= hi + PURGE):
                    drop.add(i)
        train = np.array([i for i in train if i not in drop])
    return train, test


if __name__ == "__main__":
    d = build()
    n = len(d["y"])
    print(f"\n样本(每次敲击一次重复): {n}    字母: {len(np.unique(d['y']))}    文件: {len(d['files'])}")
    cnt = np.bincount(d["y"], minlength=26)
    print("每字母样本数:", "  ".join(f"{LETTERS[i]}:{cnt[i]}" for i in range(26)))
    print("每折样本数  :", np.bincount(d["fold"]))
    tr, te = split(d, 0)
    print(f"第0折: 训练 {len(tr)}  测试 {len(te)}  (purge 掉 {n - len(tr) - len(te)} 个边界重复)")
    ok = (d["rule_pred"] == d["y"]).mean()
    print(f"方法① 规则解码 全量准确率(无需训练): {ok*100:.2f}%")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "ml", "dataset.npz")
    np.savez_compressed(out, **d)
    print("saved", out)
