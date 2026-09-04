#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morse_sensor.py — 把柔性电阻传感器上的「摩尔斯敲击」解码成字母。

场景
----
在一块可拉伸的电阻传感器上按压输入摩尔斯电码:轻/短敲 = 点(·),重/长敲 = 划(—)。
用数字万用表(如 Keysight 34461A)记录「电阻(Ω)—时间」曲线,本模块从这条曲线里
自动找出每一次敲击、判断点还是划、再按标准摩尔斯码表还原出字母。

核心思路(一条规则,两种情况)
------------------------------
一次敲击有多"大",体现在两方面:峰更高(ΔR/R₀ 更大)、或按得更久(时长更长)。
但峰高的绝对值会随传感器状态和按力变化,所以:
  * 一段录制里如果同时有明显的高峰和矮峰(有对比)——按【相对峰高】判点/划;
  * 如果所有敲击高矮相近(纯点或纯划,无对比)——按【绝对时长】判(划按得更久)。
时长是人的动作,跨录制稳定,正好补上峰高不稳的短板。

直接使用
--------
    from morse_sensor import decode_csv
    r = decode_csv("recording.csv")
    print(r.text)                 # 解码出的字母,如 "D"
    for lg in r.letters:          # 每个字母的点划与波形位置
        print(lg.symbol, lg.letter)

命令行
------
    python morse_sensor.py recording.csv            # 解码单个文件
    python morse_sensor.py --batch "raw data"       # 批量:按文件名首字母评测准确率

依赖:numpy, pandas。(仅画波形时才需要 matplotlib。)
"""
from __future__ import annotations
import os
import glob
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 可调参数(对新数据一般无需改动;若采样率或按压节奏差别很大可微调)
# --------------------------------------------------------------------------- #
DETREND_WINDOW_S = 40.0    # 估计缓慢静息基线的时间窗(秒);去掉慢漂移
GLITCH_RATIO     = 3.0     # 电阻超过局部中位数这么多倍即判为接触故障,剔除
TAP_MIN_WIDTH_S  = 0.6     # 一次有效按压的最短持续时间(秒)
TAP_MERGE_GAP_S  = 0.9     # 间隔小于此的相邻鼓包视为同一次按压(去抖)
HI_FRAC, LO_FRAC = 0.35, 0.15   # 检测阈值:相对该录制峰值分位数的高/低门限
HEIGHT_RATIO_MIN = 1.6     # 高峰/矮峰均值之比 ≥ 此值 → 认为"有高矮对比"
DASH_WIDTH_S     = 3.6     # 纯字母兜底:按压时长 ≥ 此值判为划,否则判为点

# 标准摩尔斯码表
MORSE = {
    'A': '.-',   'B': '-...', 'C': '-.-.', 'D': '-..',  'E': '.',
    'F': '..-.', 'G': '--.',  'H': '....', 'I': '..',   'J': '.---',
    'K': '-.-',  'L': '.-..', 'M': '--',   'N': '-.',   'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.',  'S': '...',  'T': '-',
    'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
}
MORSE_INV = {v: k for k, v in MORSE.items()}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Tap:
    """一次按压。"""
    t_start: float
    t_end: float
    height: float          # 峰高,ΔR/R₀ 的百分数
    width: float           # 按压时长(秒)
    symbol: str = "?"      # '.' 或 '-'(分类后填入)

    @property
    def t_center(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


@dataclass
class LetterGroup:
    """一个字母 = 若干次按压。"""
    taps: List[Tap]
    symbol: str = ""       # 点划串,如 '-..'
    letter: str = "?"      # 解码出的字符


@dataclass
class DecodeResult:
    sec: np.ndarray                       # 时间轴(秒)
    rel: np.ndarray                       # 相对电阻变化 ΔR/R₀(%)
    baseline_ohm: float                   # 估计的静息电阻(Ω)
    taps: List[Tap]
    letters: List[LetterGroup]
    rule: str = ""                        # 实际采用的判据:'相对峰高' 或 '绝对时长'
    text: str = ""                        # 拼接后的解码结果


# --------------------------------------------------------------------------- #
# 1. 读取与预处理
# --------------------------------------------------------------------------- #
def load_keysight_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 Keysight 34461A 导出的 CSV,返回(相对秒, 电阻Ω)。

    文件头 8 行是仪器信息,第 8 行为表头 `Time (s), Channel 1 (Ω)`,数据从第 9 行起;
    时间列是绝对时间戳,这里换算成从 0 开始的秒。若你的仪器格式不同,只需改此函数。
    """
    df = pd.read_csv(path, skiprows=7, names=["t", "R"], header=0)
    t = pd.to_datetime(df["t"], errors="coerce", format="mixed")
    ok = t.notna()
    df, t = df[ok], t[ok]
    sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    R = pd.to_numeric(df["R"], errors="coerce").to_numpy()
    good = ~np.isnan(R)
    return sec[good], R[good]


def deglitch(R: np.ndarray, ratio: float = GLITCH_RATIO) -> np.ndarray:
    """只剔除物理上不可能的跳变(接触故障，如电阻突然跳到几百万欧),不碰正常按压。"""
    R = R.astype(float).copy()
    med = pd.Series(R).rolling(9, center=True, min_periods=1).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = R / med
    bad = (rel > ratio) | (rel < 1.0 / ratio)
    R[bad] = med[bad]
    return R


def detrend(R: np.ndarray, sec: np.ndarray,
            win_s: float = DETREND_WINDOW_S) -> Tuple[np.ndarray, float]:
    """减去缓慢移动的静息基线,得到相对变化 ΔR/R₀(%)。

    基线取一个大时间窗内的低分位数(跟随按压之间的静息水平),这样即便传感器有慢漂移,
    按压仍然是干净的正向鼓包。返回(相对变化%, 基线中位数Ω)。
    """
    if len(sec) < 20:
        R0 = float(np.percentile(R, 20))
        return (R - R0) / R0 * 100.0, R0
    fs = len(sec) / (sec[-1] - sec[0])
    win = max(11, int(round(fs * win_s)) | 1)
    base = pd.Series(R).rolling(win, center=True,
                                min_periods=max(5, win // 4)).quantile(0.10)
    base = base.bfill().ffill().to_numpy()
    base = np.maximum(base, 1e-6)
    return (R - base) / base * 100.0, float(np.median(base))


# --------------------------------------------------------------------------- #
# 2. 找按压
# --------------------------------------------------------------------------- #
def detect_taps(sec: np.ndarray, rel: np.ndarray) -> List[Tap]:
    """用带滞回的阈值把每一次按压(高出静息的鼓包)找出来。"""
    top = np.percentile(rel, 98)
    hi = max(0.8, HI_FRAC * top)      # 必须超过它才算一次按压
    lo = max(0.4, LO_FRAC * top)      # 用它界定按压的起止
    above = rel > lo
    d = np.diff(above.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(above)]

    raw: List[Tap] = []
    for s, e in zip(starts, ends):
        seg = rel[s:e]
        if seg.size == 0 or seg.max() < hi:
            continue
        w = sec[e - 1] - sec[s]
        if w < TAP_MIN_WIDTH_S:
            continue
        raw.append(Tap(sec[s], sec[e - 1], float(seg.max()), float(w)))

    # 合并因短暂掉落而被拆开的同一次按压
    merged: List[Tap] = []
    for t in raw:
        if merged and t.t_start - merged[-1].t_end < TAP_MERGE_GAP_S:
            last = merged[-1]
            last.t_end = t.t_end
            last.height = max(last.height, t.height)
            last.width = last.t_end - last.t_start
        else:
            merged.append(t)
    return merged


# --------------------------------------------------------------------------- #
# 3. 分组成字母(无监督:按敲击之间的间隔)
# --------------------------------------------------------------------------- #
def _otsu(values: np.ndarray) -> Tuple[float, float]:
    """一维 Otsu 阈值 + 两类均值之比(用于判断是否存在两个尺度)。"""
    v = np.sort(values)
    if len(v) < 2 or v[-1] == v[0]:
        return float(v.mean()), 1.0
    best_t, best_var = v[0], -1.0
    for i in range(1, len(v)):
        lo, hi = v[:i], v[i:]
        var = (len(lo) * len(hi)) * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, 0.5 * (lo.max() + hi.min())
    lo, hi = v[v < best_t], v[v >= best_t]
    ratio = (hi.mean() / lo.mean()) if len(lo) and len(hi) and lo.mean() > 0 else 1.0
    return float(best_t), float(ratio)


def group_into_letters(taps: List[Tap]) -> List[List[Tap]]:
    """按相邻按压之间的间隔把 taps 切成一个个字母。

    间隔天然有两个尺度:字母内部(点划之间)的短间隔,和字母之间的长间隔。用 Otsu 在
    间隔上找分界即可,无需事先知道字母是什么。"""
    if len(taps) <= 1:
        return [taps] if taps else []
    gaps = np.array([taps[i + 1].t_start - taps[i].t_end for i in range(len(taps) - 1)])
    thr, ratio = _otsu(gaps)
    if ratio < 1.8:          # 间隔没有明显两个尺度 → 视作同一个字母(或单字母重复由调用方处理)
        thr = np.inf
    groups, cur = [], [taps[0]]
    for i, g in enumerate(gaps):
        if g > thr:
            groups.append(cur); cur = []
        cur.append(taps[i + 1])
    groups.append(cur)
    return groups


# --------------------------------------------------------------------------- #
# 4. 判点/划 —— 有对比用峰高,无对比用时长
# --------------------------------------------------------------------------- #
def classify_taps(taps: List[Tap]) -> str:
    """就地给每个 tap 填上 '.'或'-',返回采用的判据名称。

    在整段录制的所有按压上决定用哪条判据:峰高明显分两档就按相对峰高,否则按绝对时长。"""
    heights = np.array([t.height for t in taps])
    thr_h, ratio = _otsu(heights)
    if ratio >= HEIGHT_RATIO_MIN:
        for t in taps:
            t.symbol = "-" if t.height >= thr_h else "."
        return "相对峰高"
    else:
        for t in taps:
            t.symbol = "-" if t.width >= DASH_WIDTH_S else "."
        return "绝对时长"


# --------------------------------------------------------------------------- #
# 5. 顶层:解码一整段录制
# --------------------------------------------------------------------------- #
def decode(sec: np.ndarray, R: np.ndarray) -> DecodeResult:
    """从(时间, 电阻)解码出字母。"""
    R = deglitch(R)
    rel, base = detrend(R, sec)
    taps = detect_taps(sec, rel)
    rule = classify_taps(taps) if taps else ""
    groups = group_into_letters(taps)

    letters: List[LetterGroup] = []
    for g in groups:
        sym = "".join(t.symbol for t in g)
        letters.append(LetterGroup(taps=g, symbol=sym, letter=MORSE_INV.get(sym, "?")))
    text = "".join(lg.letter for lg in letters)
    return DecodeResult(sec=sec, rel=rel, baseline_ohm=base,
                        taps=taps, letters=letters, rule=rule, text=text)


def decode_csv(path: str) -> DecodeResult:
    """便捷入口:直接给 CSV 路径。"""
    sec, R = load_keysight_csv(path)
    return decode(sec, R)


# --------------------------------------------------------------------------- #
# 单文件重复评测(当一份录制是同一个字母重复很多次时)
# --------------------------------------------------------------------------- #
def letter_from_filename(path: str) -> Optional[str]:
    """从文件名首字符取真值字母(如 'D-20 ....csv' → 'D');取不到返回 None。"""
    name = os.path.basename(path)
    c = name[:1].upper()
    return c if c in MORSE else None


def evaluate_repeated_letter(path: str, expected_len: Optional[int] = None) -> dict:
    """一份录制 = 同一字母重复多次时,统计识别情况。

    这里的重复切分借助已知真值的点划个数(评测用);实际部署解码用 decode() 即可。
    返回准确率等指标。
    """
    sec, R = load_keysight_csv(path)
    res = decode(sec, R)
    true_letter = letter_from_filename(path)
    k = expected_len if expected_len else (len(MORSE.get(true_letter, "")) or None)

    # 用已知长度把按压切成一次次重复(仅评测用)
    taps = res.taps
    if not taps:
        return dict(file=os.path.basename(path), letter=true_letter, taps=0)
    reps = _regroup_by_count(taps, k) if k else group_into_letters(taps)
    thr_h, ratio = _otsu(np.array([t.height for t in taps]))
    use_height = ratio >= HEIGHT_RATIO_MIN
    ok = 0
    for rep in reps:
        if len(rep) != (k or len(rep)):
            continue
        sym = "".join(("-" if (t.height >= thr_h if use_height else t.width >= DASH_WIDTH_S)
                       else ".") for t in rep)
        ok += (MORSE_INV.get(sym) == true_letter)
    n = sum(1 for rep in reps if not k or len(rep) == k)
    return dict(file=os.path.basename(path), letter=true_letter,
                rule=("相对峰高" if use_height else "绝对时长"),
                taps=len(taps), clean_reps=n,
                acc=(ok / n if n else None))


def _regroup_by_count(taps: List[Tap], k: int) -> List[List[Tap]]:
    """已知每个字母 k 个元素时,选一个间隔阈值让「恰好 k 个」的重复最多(评测用)。"""
    if len(taps) <= 1 or not k:
        return [taps]
    gaps = np.array([taps[i + 1].t_start - taps[i].t_end for i in range(len(taps) - 1)])
    sg = np.unique(gaps)
    cands = (sg[:-1] + sg[1:]) / 2 if len(sg) > 1 else sg
    best_thr, best = float(np.median(gaps)), -1
    for thr in cands:
        reps, cur = [], [taps[0]]
        for i, g in enumerate(gaps):
            if g > thr:
                reps.append(cur); cur = []
            cur.append(taps[i + 1])
        reps.append(cur)
        score = sum(len(r) == k for r in reps) - 0.001 * len(reps)
        if score > best:
            best, best_thr = score, thr
    reps, cur = [], [taps[0]]
    for i, g in enumerate(gaps):
        if g > best_thr:
            reps.append(cur); cur = []
        cur.append(taps[i + 1])
    reps.append(cur)
    return reps


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def _main():
    ap = argparse.ArgumentParser(description="从电阻曲线解码摩尔斯字母")
    ap.add_argument("path", help="CSV 文件,或用 --batch 时的文件夹")
    ap.add_argument("--batch", action="store_true",
                    help="批量:把文件夹里每个 CSV 当作『首字母重复多次』评测准确率")
    args = ap.parse_args()

    if not args.batch:
        r = decode_csv(args.path)
        print(f"文件      : {os.path.basename(args.path)}")
        print(f"静息电阻  : {r.baseline_ohm:.0f} Ω")
        print(f"检测到按压: {len(r.taps)}   判据: {r.rule}")
        print(f"解码结果  : {r.text}")
        for i, lg in enumerate(r.letters[:20], 1):
            sizes = " ".join(f"{t.height:.1f}%/{t.width:.1f}s" for t in lg.taps)
            print(f"  字母{i}: {lg.symbol:6} → {lg.letter}    [{sizes}]")
        return

    files = sorted(glob.glob(os.path.join(args.path, "*.csv")))
    print(f"{'file':30} {'letter':6} {'rule':8} {'reps':>5} {'acc':>6}")
    accs = []
    for f in files:
        r = evaluate_repeated_letter(f)
        if r.get("acc") is not None:
            accs.append(r["acc"])
        a = "  -  " if r.get("acc") is None else f"{r['acc']*100:4.0f}%"
        print(f"{r['file'][:30]:30} {str(r.get('letter')):6} "
              f"{r.get('rule',''):8} {str(r.get('clean_reps','')):>5} {a:>6}")
    if accs:
        print(f"\n平均每次重复识别率: {np.mean(accs)*100:.1f}%  (共 {len(accs)} 个文件)")


if __name__ == "__main__":
    _main()
