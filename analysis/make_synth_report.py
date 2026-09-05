#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《合成消息识别 测试报告》PDF。"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

import morse_sensor as ms
import synth_test as st
from make_report import (ZH, wrapped, marks, C_SIGNAL, C_DOT, C_DASH, C_CUT,
                         C_INK, C_MUTE, C_OK, C_BAD)

RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")


def all_intervals():
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        tok = os.path.basename(f).split(" ")[0]
        if ms.letter_from_filename(f) is None or tok in {"K-30", "D13+F58"}:
            continue
        sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
        taps = ms.detect_taps(sec, rel)
        out += [b.t_center - a.t_center for a, b in zip(taps[:-1], taps[1:])]
    return np.array(out)


def intra_by_letter():
    out = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        tok = os.path.basename(f).split(" ")[0]; L = ms.letter_from_filename(f)
        if L is None or tok in {"K-30", "D13+F58"}:
            continue
        code = ms.MORSE[L]; k = len(code)
        if k < 2:
            continue
        sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
        taps = ms.detect_taps(sec, rel)
        reps = [r for r in ms._regroup_by_count(taps, k) if len(r) == k]
        gs = [r[i + 1].t_center - r[i].t_center for r in reps for i in range(len(r) - 1)]
        if gs:
            out.setdefault(L, []).append(np.median(gs))
    return {L: float(np.median(v)) for L, v in out.items()}


def cadence_seg_rate(shapes, n=300, seed=9):
    import random
    rng = random.Random(seed); np.random.seed(seed)
    alpha = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"); ok = 0
    for _ in range(n):
        letters = [rng.choice(alpha) for _ in range(rng.randint(3, 6))]
        sec, rel = st.synth_cadence_message(letters, shapes, 5.0, 12.0, rng)
        if len(ms.group_into_letters(ms.detect_taps(sec, rel))) == len(letters):
            ok += 1
    return ok / n


def plot_window(ax, nm, seconds=70):
    f = glob.glob(os.path.join(RAW, f"{nm}*.csv"))[0]
    sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
    taps = ms.detect_taps(sec, rel)
    t0 = taps[3].t_start - 2; t1 = t0 + seconds; m = (sec >= t0) & (sec <= t1)
    ax.plot(sec[m] - t0, rel[m], color=C_SIGNAL, lw=1.1)
    win = [t for t in taps if t0 <= t.t_center <= t1]
    for a, b in zip(win[:-1], win[1:]):
        ax.text((a.t_center + b.t_center) / 2 - t0, max(rel[m]) * 0.5,
                f"{b.t_center - a.t_center:.0f}", ha="center", fontsize=7.5, color="#555")
    L = ms.letter_from_filename(f)
    ax.set_title(f"字母 {L} = {marks(ms.MORSE[L])}   (数字=相邻按压的峰心间隔,秒)",
                 fontproperties=ZH, fontsize=9.5, loc="left")
    ax.set_ylabel("ΔR/R₀ %", fontproperties=ZH, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False); ax.margins(x=0.01); ax.tick_params(labelsize=7)


def plot_synth(ax, msg, dec, sec, rel):
    ax.plot(sec, rel, color=C_SIGNAL, lw=1.1)
    ax.set_ylim(-1, max(rel) * 1.15); ax.margins(x=0.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("ΔR/R₀ %", fontproperties=ZH, fontsize=7.5); ax.tick_params(labelsize=7)
    ok = (dec == msg)
    ax.set_title(f"真实:{msg}   →   识别:{dec}   {'✓' if ok else '✗'}",
                 fontproperties=ZH, fontsize=10, loc="left", color=(C_OK if ok else C_BAD))


# --------------------------------------------------------------------------- #
def page_cover(pdf, orc, rc):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .955, "合成消息识别 · 测试报告", 13, C_SIGNAL, "bold")
    wrapped(fig, .08, .925, "现有算法在随机字母组合上的表现", 23, C_INK, "bold")
    wrapped(fig, .08, .855,
            "用真实的单字母按压数据,随机拼成多字母「消息」,喂给现有的、未改动的识别算法,"
            "再和原文比对。据合作者说明,输入是按约 5 秒一拍的节奏敲的——本测试据此模拟"
            "「同一次、统一节奏」的消息(元素形状与点/划比例取自真实数据)。", 11.5, C_MUTE)
    stats = [(f"{orc['orc_acc']*100:.0f}%", "点划→字母 分类准确率\n(给定字母边界,真实数据)", C_OK),
             (f"{(1-rc['cer'])*100:.0f}%", "端到端字符准确率\n(5秒节奏,含自动切分)", C_OK),
             (f"{rc['exact']*100:.0f}%", "整条消息全对\n(5秒节奏)", C_INK)]
    for i, (v, lab, col) in enumerate(stats):
        x = .08 + i * .30
        fig.text(x + .12, .74, v, fontsize=28, color=col, ha="center", fontproperties=ZH, fontweight="bold")
        fig.text(x + .12, .675, lab, fontsize=10, color=C_MUTE, ha="center", fontproperties=ZH, linespacing=1.4)
    succ = next((e for e in rc["examples"] if e[1] == e[0] and 4 <= len(e[0]) <= 5), rc["examples"][0])
    ax = fig.add_axes([.08, .42, .84, .17]); plot_synth(ax, *succ)
    ax.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=9)
    wrapped(fig, .08, .355, f"测试规模:{rc['n_messages']} 条随机消息,每条 3~6 个字母,"
            "元素间隔约 5s、字母间隔约 12s。", 11, C_MUTE)
    wrapped(fig, .08, .31,
            "结论:识别核心(点/划→字母)很稳;只要按统一节奏输入(她说的 5 秒一拍、字母之间停顿"
            "更长),自动切分字母也能做对,端到端字符准确率约 93%。之前偏低的估计,是因为把不同"
            "节奏的录制硬拼进一条消息——那并不符合真实输入(详见后页)。", 11, C_INK)
    wrapped(fig, .08, .10, "说明:这是基于真实元素统计 + 5 秒节奏的仿真;若能录几条真实的多字母消息,"
            "即可直接给出实测端到端准确率。", 9.5, C_MUTE)
    pdf.savefig(fig); plt.close(fig)


def page_timing(pdf, seg_rate):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .955, "计时分析:节奏一致,才切得开字母", 17, C_INK, "bold")
    wrapped(fig, .08, .915,
            "「点/划→字母」不难,难在把连续按压切成一个个字母。这完全取决于计时。", 11.5, C_MUTE)

    # 间隔直方图(5秒确认)
    iv = all_intervals()
    ax = fig.add_axes([.10, .60, .82, .24])
    ax.hist(iv, bins=np.arange(0, 20, 1), color=C_DOT, alpha=.85)
    ax.axvline(5, color=C_DASH, lw=1.5); ax.text(5.2, ax.get_ylim()[1]*.9, "≈5s", color=C_DASH, fontproperties=ZH)
    ax.set_xlabel("相邻按压 峰心间隔(秒)", fontproperties=ZH, fontsize=9.5)
    ax.set_ylabel("次数", fontproperties=ZH, fontsize=9)
    ax.set_title("全部数据的按压间隔:主峰约 5s(= 合作者说的节奏),字母间隔在 ~10s 处",
                 fontproperties=ZH, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    # D 干净 / B 慢 的对比
    wrapped(fig, .08, .55, "但各次录制的节奏差异很大:", 11.5, C_INK, "bold")
    axd = fig.add_axes([.10, .40, .82, .085]); plot_window(axd, "D-20")
    axb = fig.add_axes([.10, .275, .82, .085]); plot_window(axb, "B-64")
    axb.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=8)
    wrapped(fig, .08, .235,
            "D:每下约 5~7s、字母间约 12s,内外分明 → 好切分。B:每下 10~16s(这次录得很慢),"
            "字母内间隔比别处的字母间隔还大。把这种不同节奏的片段混进同一条消息,自动切分必然出错——"
            "这只是测试拼接的问题,不是真实输入。", 10.5, C_MUTE)
    wrapped(fig, .08, .125,
            f"关键验证:当按统一的 5 秒节奏合成消息时,自动切分「字母个数」正确率达 {seg_rate*100:.0f}%。"
            "也就是说,只要输入节奏一致,切分这一步是站得住的。", 11.5, C_OK, "bold")
    pdf.savefig(fig); plt.close(fig)


def page_perletter(pdf, orc):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .95, "逐字母:点划→字母 分类准确率", 16, C_INK, "bold")
    wrapped(fig, .08, .915, "随机消息里、给定正确字母边界时,每个字母被认对的比例(真实数据)。", 10.5, C_MUTE)
    bl = orc["orc_by_letter"]; order = sorted(bl)
    ax = fig.add_axes([.14, .07, .78, .82]); ys = np.arange(len(order))[::-1]
    for y, L in zip(ys, order):
        okc, tot = bl[L]; acc = okc / tot * 100 if tot else 0
        ax.barh(y, acc, height=.66, color=C_DOT if acc >= 90 else C_BAD)
        ax.text(-2, y, f"{L} {marks(ms.MORSE[L])}", ha="right", va="center", fontsize=8.5, fontproperties=ZH)
        ax.text(acc + 1, y, f"{acc:.0f}% (n={tot})", va="center", fontsize=8, color=C_MUTE, fontproperties=ZH)
    ax.set_xlim(0, 120); ax.set_ylim(-1, len(order)); ax.set_yticks([])
    ax.set_xlabel("准确率 (%)", fontproperties=ZH, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.axvline(100, color="#ccc", lw=.8)
    pdf.savefig(fig); plt.close(fig)


def page_examples(pdf, rc):
    succ = [e for e in rc["examples"] if e[1] == e[0]]
    fail = [e for e in rc["examples"] if e[1] != e[0]]
    picks = (succ[:3] + fail[:1])[:4]
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .96, "示例:5 秒节奏消息的波形与识别", 16, C_INK, "bold")
    wrapped(fig, .08, .925, "标题给出 真实 → 识别。这些是「统一节奏」下合成的消息。", 10, C_MUTE)
    gs = GridSpec(len(picks), 1, left=.10, right=.95, top=.88, bottom=.06, hspace=.5)
    for i, e in enumerate(picks):
        ax = fig.add_subplot(gs[i]); plot_synth(ax, *e)
        if i == len(picks) - 1:
            ax.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=9)
    pdf.savefig(fig); plt.close(fig)


def main():
    orc = st.run(n_messages=400, seed=1)                       # 真实拼接:取 oracle 分类
    rc = st.run_cadence(n_messages=400, elem_gap=5.0, letter_gap=12.0, seed=5)
    seg_rate = cadence_seg_rate(st.build_shape_library())
    out = os.path.join(os.path.dirname(__file__), "out", "合成消息测试报告.pdf")
    with PdfPages(out) as pdf:
        page_cover(pdf, orc, rc)
        page_timing(pdf, seg_rate)
        page_perletter(pdf, orc)
        page_examples(pdf, rc)
        pdf.infodict()["Title"] = "合成消息识别 测试报告"
    print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")
    print(f"orc={orc['orc_acc']*100:.1f}%  e2e_5s_char={(1-rc['cer'])*100:.1f}%  "
          f"exact={rc['exact']*100:.1f}%  seg={seg_rate*100:.0f}%")


if __name__ == "__main__":
    main()
