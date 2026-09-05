#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_test.py — 用真实单字母片段拼出随机「消息」,测试现有算法的端到端识别率。

做法:
  1) 从每个字母的真实录制里切出一段段「干净重复」的波形片段(clip);
  2) 随机选若干字母,各取一个 clip,中间插入字母间隔,拼成一条连续的 ΔR/R₀ 曲线;
  3) 用 morse_sensor 里*现成、未改动*的算法(检测→按间隔切分字母→逐组判点划→查码表)解码;
  4) 和原始字母序列比对,统计准确率。

同时给两个口径:
  * 端到端:算法自己按间隔切分字母(真实场景)
  * 给定边界:直接用拼接时的真字母边界(只考验"点划分类",隔离切分误差)

注意:片段来自不同录制(不同传感器状态/按力),因此这是偏难的跨场景测试。
"""
from __future__ import annotations
import glob, os, random
import numpy as np
import morse_sensor as ms

RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")
DT = 1.0 / 2.37                 # 统一时间栅格(真实采样约 2.37 Hz)
PAD_S = 4.0                     # 每个字母片段前后留白(须足够长以保住按压拖尾,否则半高宽被低估)
GAP_RANGE_S = (6.0, 9.0)       # 字母之间的间隔(明显大于字母内部点划间隔)
EXCLUDE = {"K-30", "D13+F58"}   # 坏文件 / 混合文件不作素材


def build_library(seed=0):
    """letter -> [clip, ...];clip 是一段以 ~0 为基线、含该字母一次完整敲击的 ΔR/R₀ 波形。"""
    lib = {}
    best = {}   # 每个字母选干净重复最多的一份文件
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        tok = os.path.basename(f).split(" ")[0]
        L = ms.letter_from_filename(f)
        if L is None or tok in EXCLUDE:
            continue
        sec, R = ms.load_keysight_csv(f)
        R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
        taps = ms.detect_taps(sec, rel); code = ms.MORSE[L]; k = len(code)
        reps = [r for r in ms._regroup_by_count(taps, k) if len(r) == k]
        if L not in best or len(reps) > best[L][0]:
            best[L] = (len(reps), sec, rel, reps, k)
    for L, (n, sec, rel, reps, k) in best.items():
        clips = []
        for r in reps:
            i0 = int(np.searchsorted(sec, r[0].t_start - PAD_S))
            i1 = int(np.searchsorted(sec, r[-1].t_end + PAD_S))
            clip = rel[max(0, i0):i1].astype(float).copy()
            if clip.size < 3:
                continue
            clip = clip - np.percentile(clip, 10)      # 基线归零
            clip = np.clip(clip, -0.5, None)
            # 峰值归一化到统一尺度(模拟"同一次、同一传感器",保留字母内部点/划比例)
            pk = clip.max()
            if pk <= 0:
                continue
            clip = clip / pk * 10.0
            clips.append(clip)
        if clips:
            lib[L] = clips
    return lib


def make_message(lib, letters, rng):
    """把给定字母序列拼成一条曲线;返回(sec, rel, 边界列表[(t0,t1)/字母])。"""
    parts, bounds, t = [], [], 0.0
    for i, L in enumerate(letters):
        clip = rng.choice(lib[L]) * rng.uniform(0.85, 1.2)   # 同一会话内的按力小抖动
        if i > 0:
            g = int(round(rng.uniform(*GAP_RANGE_S) / DT))
            parts.append(np.random.normal(0, 0.12, g).clip(-0.4, 0.4))
            t += g * DT
        t0 = t
        parts.append(clip)
        t += len(clip) * DT
        bounds.append((t0, t))
    rel = np.concatenate(parts)
    sec = np.arange(len(rel)) * DT
    return sec, rel, bounds


def decode_end_to_end(sec, rel):
    taps = ms.detect_taps(sec, rel)
    groups = ms.group_into_letters(taps)
    out = []
    for g in groups:
        ms.classify_group(g)
        out.append(ms.MORSE_INV.get("".join(t.symbol for t in g), "?"))
    return "".join(out)


def decode_with_boundaries(sec, rel, bounds):
    """给定真字母边界,只测点划分类。"""
    taps = ms.detect_taps(sec, rel)
    out = []
    for (t0, t1) in bounds:
        g = [t for t in taps if t0 <= t.t_center <= t1]
        if not g:
            out.append("?"); continue
        ms.classify_group(g)
        out.append(ms.MORSE_INV.get("".join(t.symbol for t in g), "?"))
    return "".join(out)


def build_shape_library():
    """收集真实的「元素形状」:每个点/划的(归一化峰高, 半高宽)。
    峰高按各文件的划高度归一,去掉跨录制的振幅差异,只保留点/划比例与形状。"""
    dots, dashes = [], []
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        tok = os.path.basename(f).split(" ")[0]
        L = ms.letter_from_filename(f)
        if L is None or tok in EXCLUDE:
            continue
        sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
        taps = ms.detect_taps(sec, rel); code = ms.MORSE[L]; k = len(code)
        reps = [r for r in ms._regroup_by_count(taps, k) if len(r) == k]
        if not reps:
            continue
        dash_h = np.median([t.height for r in reps for t, c in zip(r, code) if c == "-"]) \
            if "-" in code else np.median([t.height for r in reps for t in r])
        if dash_h <= 0:
            continue
        for r in reps:
            for t, c in zip(r, code):
                (dashes if c == "-" else dots).append((t.height / dash_h, t.width))
    return dots, dashes


def synth_cadence_message(letters, shapes, elem_gap, letter_gap, rng):
    """按统一节奏(元素间隔 elem_gap、字母间隔 letter_gap)合成消息,元素形状取自真实数据。"""
    dots, dashes = shapes
    centers, syms, t = [], [], 6.0
    for L in letters:
        for c in ms.MORSE[L]:
            centers.append(t); syms.append(c)
            t += elem_gap * rng.uniform(0.9, 1.1)
        t += (letter_gap - elem_gap) * rng.uniform(0.9, 1.1)
    T = t + 6
    sec = np.arange(0, T, DT); rel = np.zeros_like(sec)
    # 每条消息一套一致的点/划画像(模拟"一个人一口气按",同真实数据:会话内点/划各自稳定)
    scale = rng.uniform(8, 13)
    max_w = elem_gap * 0.55
    dash_h = scale
    dot_h = scale * rng.uniform(0.34, 0.46)                    # 点约为划的 0.4(真实比例)
    dash_w = min(rng.uniform(2.3, 2.7), max_w)
    dot_w = min(rng.uniform(1.6, 2.1), max_w)
    for cc, c in zip(centers, syms):
        h = (dash_h if c == "-" else dot_h) * rng.uniform(0.92, 1.08)
        w = (dash_w if c == "-" else dot_w) * rng.uniform(0.9, 1.1)
        sig = max(0.4, w) / 2.355
        rel += h * np.exp(-((sec - cc) ** 2) / (2 * sig * sig))
    rel += np.random.normal(0, 0.15, len(sec))
    return sec, rel


def run_cadence(n_messages=400, len_range=(3, 6), elem_gap=5.0, letter_gap=12.0, seed=5):
    """统一节奏(她说的 ~5 秒)下的端到端测试。"""
    rng = random.Random(seed); np.random.seed(seed)
    shapes = build_shape_library()
    alpha = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    tot = edit = exact = 0; examples = []
    for _ in range(n_messages):
        letters = [rng.choice(alpha) for _ in range(rng.randint(*len_range))]
        msg = "".join(letters)
        sec, rel = synth_cadence_message(letters, shapes, elem_gap, letter_gap, rng)
        d = decode_end_to_end(sec, rel)
        tot += len(msg); edit += levenshtein(msg, d); exact += (d == msg)
        if len(examples) < 30:
            examples.append((msg, d, sec, rel))
    return dict(n_messages=n_messages, tot_chars=tot, elem_gap=elem_gap, letter_gap=letter_gap,
                cer=edit / tot, exact=exact / n_messages, examples=examples)


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def run(n_messages=400, len_range=(3, 6), seed=1):
    rng = random.Random(seed)
    np.random.seed(seed)
    lib = build_library()
    alphabet = sorted(lib.keys())
    tot_chars = e2e_edit = e2e_exact = 0
    orc_chars = orc_ok = 0
    conf = {}            # (intended, got) -> count  (给定边界下的字母混淆)
    orc_by_letter = {}   # letter -> [ok, total]
    seg_correct = 0      # 端到端切分出的字母数 == 真实
    examples = []
    for _ in range(n_messages):
        L = rng.randint(*len_range)
        letters = [rng.choice(alphabet) for _ in range(L)]
        msg = "".join(letters)
        sec, rel, bounds = make_message(lib, letters, rng)
        d_e2e = decode_end_to_end(sec, rel)
        d_orc = decode_with_boundaries(sec, rel, bounds)
        tot_chars += len(msg)
        e2e_edit += levenshtein(msg, d_e2e)
        e2e_exact += (d_e2e == msg)
        seg_correct += (len(d_e2e) == len(msg))
        for a, b in zip(msg, d_orc):
            orc_chars += 1; orc_ok += (a == b)
            st_ = orc_by_letter.setdefault(a, [0, 0])
            st_[1] += 1; st_[0] += (a == b)
            if a != b:
                conf[(a, b)] = conf.get((a, b), 0) + 1
        if len(examples) < 60:
            examples.append((msg, d_e2e, d_orc, sec, rel, bounds))
    return dict(
        n_messages=n_messages, tot_chars=tot_chars,
        e2e_cer=e2e_edit / tot_chars, e2e_exact=e2e_exact / n_messages,
        orc_acc=orc_ok / orc_chars, orc_chars=orc_chars,
        seg_correct=seg_correct / n_messages,
        orc_by_letter=orc_by_letter,
        conf=conf, examples=examples, alphabet=alphabet, lib=lib)


if __name__ == "__main__":
    r = run()
    print(f"随机消息 {r['n_messages']} 条,共 {r['tot_chars']} 个字母")
    print(f"端到端(含自动切分): 整条全对 {r['e2e_exact']*100:.1f}%   "
          f"字符错误率 CER {r['e2e_cer']*100:.1f}%  → 字符准确率 {(1-r['e2e_cer'])*100:.1f}%")
    print(f"给定字母边界(只测点划分类): 字母准确率 {r['orc_acc']*100:.1f}%")
    if r["conf"]:
        top = sorted(r["conf"].items(), key=lambda x: -x[1])[:8]
        print("主要混淆(真→判, 次数):", ", ".join(f"{a}→{b}:{n}" for (a, b), n in top))
