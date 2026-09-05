#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synth_report.py — 生成「合成消息识别」测试报告 PDF。

用真实单字母片段拼成随机消息,测现有算法端到端识别率,单独成一份报告。
    python make_synth_report.py            # 输出 out/合成消息测试报告.pdf
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import morse_sensor as ms
import synth_test as st
from make_report import (ZH, wrapped, marks, C_SIGNAL, C_DOT, C_DASH, C_CUT,
                         C_INK, C_MUTE, C_OK, C_BAD)

RAW = os.path.join(os.path.dirname(__file__), "..", "raw data")


def intra_gaps_by_letter():
    """每个多元素字母的内部点划间隔中位(秒)。"""
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
        gs = [r[i + 1].t_start - r[i].t_end for r in reps for i in range(len(r) - 1)]
        if gs:
            out.setdefault(L, []).extend(gs)
    return {L: float(np.median(v)) for L, v in out.items()}


def plot_message(ax, msg, dec, sec, rel, bounds):
    ax.plot(sec, rel, color=C_SIGNAL, lw=1.2)
    for (t0, t1), ch in zip(bounds, msg):
        ax.axvspan(t0, t1, color=C_DOT, alpha=0.07)
        ax.text((t0 + t1) / 2, ax.get_ylim()[1] if False else max(rel) * 1.02, ch,
                ha="center", va="bottom", fontsize=11, fontproperties=ZH, color=C_INK)
    ax.set_ylim(-1, max(rel) * 1.18)
    ax.margins(x=0.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("ΔR/R₀ (%)", fontproperties=ZH, fontsize=8)
    ok = (dec == msg)
    ax.set_title(f"真实:{msg}    →    识别:{dec}    {'✓' if ok else '✗'}",
                 fontproperties=ZH, fontsize=11, loc="left",
                 color=(C_OK if ok else C_BAD))


def page_cover(pdf, r):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .95, "合成消息识别 · 测试报告", 13, C_SIGNAL, "bold")
    wrapped(fig, .08, .92, "现有算法在随机字母组合上的表现", 24, C_INK, "bold")
    wrapped(fig, .08, .85,
            "把每个字母的真实按压片段随机拼接成「消息」(中间加字母间隔),喂给现有的、"
            "未改动的识别算法,再和原始字母序列比对。这样可以检验算法在没见过的随机组合上"
            "还准不准。片段取自不同录制,已统一到相近的振幅尺度以模拟同一次输入。", 11.5, C_MUTE)
    stats = [(f"{r['orc_acc']*100:.1f}%", "点划分类准确率\n(给定字母边界)", C_OK),
             (f"{(1-r['e2e_cer'])*100:.0f}%", "端到端字符准确率\n(含自动切分字母)", C_BAD),
             (f"{r['seg_correct']*100:.0f}%", "整条消息字母数\n切分正确的比例", C_BAD)]
    for i, (v, lab, col) in enumerate(stats):
        x = .08 + i * .30
        fig.text(x + .12, .72, v, fontsize=27, color=col, ha="center",
                 fontproperties=ZH, fontweight="bold")
        fig.text(x + .12, .655, lab, fontsize=10.5, color=C_MUTE, ha="center",
                 fontproperties=ZH, linespacing=1.4)
    # 一条成功示例
    succ = next((e for e in r["examples"] if e[1] == e[0] and 4 <= len(e[0]) <= 5), r["examples"][0])
    ax = fig.add_axes([.08, .40, .84, .17])
    plot_message(ax, succ[0], succ[1], succ[3], succ[4], succ[5])
    ax.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=9)
    wrapped(fig, .08, .335,
            f"测试规模:{r['n_messages']} 条随机消息、共 {r['tot_chars']} 个字母,每条 3~6 个字母。", 11, C_MUTE)
    wrapped(fig, .08, .29,
            "一句话结论:识别「点/划→字母」这一核心能力在随机组合上依然很强(给定字母边界时 "
            f"{r['orc_acc']*100:.0f}%);端到端的短板在于「把消息切成一个个字母」——原始数据"
            "字母内部的敲击间隔忽大忽小,和字母之间的间隔重叠,导致自动切分不稳(详见后页)。",
            11, C_INK)
    pdf.savefig(fig); plt.close(fig)


def _plot_window(ax, nm, seconds=72):
    """画某字母录制的一段原始波形,标出检测到的按压与相邻间隔(秒),作为计时证据。"""
    f = glob.glob(os.path.join(RAW, f"{nm}*.csv"))[0]
    sec, R = ms.load_keysight_csv(f); R = ms.deglitch(R); rel, _ = ms.detrend(R, sec)
    taps = ms.detect_taps(sec, rel)
    t0 = taps[3].t_start - 2; t1 = t0 + seconds; m = (sec >= t0) & (sec <= t1)
    ax.plot(sec[m] - t0, rel[m], color=C_SIGNAL, lw=1.1)
    win = [t for t in taps if t0 <= t.t_center <= t1]
    for t in win:
        ax.axvline(t.t_center - t0, color=C_CUT, lw=0.5, alpha=0.5)
    for a, b in zip(win[:-1], win[1:]):
        g = b.t_start - a.t_end
        ax.text((a.t_center + b.t_center) / 2 - t0, max(rel[m]) * 0.5, f"{g:.0f}",
                ha="center", fontsize=7.5, color="#555")
    L = ms.letter_from_filename(f)
    ax.set_title(f"字母 {L} = {marks(ms.MORSE[L])}   (蓝线=按压,数字=间隔秒数)",
                 fontproperties=ZH, fontsize=9.5, loc="left")
    ax.set_ylabel("ΔR/R₀ %", fontproperties=ZH, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False); ax.margins(x=0.01)
    ax.tick_params(labelsize=7)


def page_finding(pdf, r):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .955, "关键结论:瓶颈在「切分」,不在「识别」", 17, C_INK, "bold")
    wrapped(fig, .08, .915, f"• 只测点划分类(给定正确字母边界):{r['orc_acc']*100:.1f}% —— 核心识别很稳。",
            11.5, C_OK)
    wrapped(fig, .08, .888, f"• 端到端(算法自己切分字母):字符准确率 {(1-r['e2e_cer'])*100:.0f}%、"
            f"整条全对 {r['e2e_exact']*100:.0f}% —— 差距几乎全来自切分。", 11.5, C_BAD)

    # 证据:两条真实时间轴(D 干净 / B 混乱)
    wrapped(fig, .08, .845, "证据 · 看真实时间轴(不是所有字母都一样):", 11.5, C_INK, "bold")
    axd = fig.add_axes([.10, .70, .82, .09]); _plot_window(axd, "D-20")
    axb = fig.add_axes([.10, .565, .82, .09]); _plot_window(axb, "B-64")
    axb.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=8)
    wrapped(fig, .08, .525,
            "D:字母内间隔约 2~4s、字母间约 9s,分得开 → 能切分。"
            "B:间隔全在 7~15s,内外没区别 → 一定切错。同一个人,不同字母节奏差很多。", 10.5, C_MUTE)

    # 各字母内部间隔中位
    intra = intra_gaps_by_letter()
    order = sorted(intra.items(), key=lambda kv: kv[1])
    ax = fig.add_axes([.14, .20, .78, .26])
    for y, (L, g) in enumerate(order):
        ax.barh(y, g, color=(C_BAD if g > 5.5 else C_DOT), height=.72)
        ax.text(-0.25, y, f"{L}", ha="right", va="center", fontsize=8, fontproperties=ZH)
    ax.axvspan(6, 9, color=C_DASH, alpha=0.15)
    ax.text(7.5, 1.2, "字母间间隔", ha="center", color=C_DASH, fontsize=8.5, fontproperties=ZH)
    ax.set_yticks([]); ax.set_xlim(0, 10)
    ax.set_xlabel("各字母 · 内部点划间隔(秒,中位)", fontproperties=ZH, fontsize=9)
    ax.set_title("多数字母内部间隔约 3s,但 B/G 偏大、与字母间间隔重叠",
                 fontproperties=ZH, fontsize=10.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    wrapped(fig, .08, .115, "结论:识别核心(点/划→字母)已在随机组合上验证可用;端到端要跑通,"
            "需采集时保持规范计时——点划间隔短而一致、字母间隔明显更长。也可用「固定摩尔斯计时」"
            "重录一小批消息来直接验证端到端上限。", 11.5, C_INK, "bold")
    pdf.savefig(fig); plt.close(fig)


def page_perletter(pdf, r):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .95, "逐字母:点划分类准确率(给定边界)", 16, C_INK, "bold")
    wrapped(fig, .08, .915, "在随机消息里、给定正确字母边界时,每个字母被认对的比例。", 10.5, C_MUTE)
    bl = r["orc_by_letter"]
    order = sorted(bl.keys())
    ax = fig.add_axes([.14, .07, .78, .82])
    ys = np.arange(len(order))[::-1]
    for y, L in zip(ys, order):
        ok, tot = bl[L]; acc = ok / tot * 100 if tot else 0
        ax.barh(y, acc, height=.66, color=C_DOT if acc >= 90 else C_BAD)
        ax.text(-2, y, f"{L} {marks(ms.MORSE[L])}", ha="right", va="center",
                fontsize=8.5, fontproperties=ZH)
        ax.text(acc + 1, y, f"{acc:.0f}% (n={tot})", va="center", fontsize=8,
                color=C_MUTE, fontproperties=ZH)
    ax.set_xlim(0, 118); ax.set_ylim(-1, len(order)); ax.set_yticks([])
    ax.set_xlabel("准确率 (%)", fontproperties=ZH, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(100, color="#ccc", lw=.8)
    pdf.savefig(fig); plt.close(fig)


def page_examples(pdf, r):
    succ = [e for e in r["examples"] if e[1] == e[0]]
    fail = [e for e in r["examples"] if e[1] != e[0] and len(e[1]) > len(e[0])]
    picks = (succ[:2] + fail[:2])[:4]
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .96, "示例:合成消息的波形与识别结果", 16, C_INK, "bold")
    wrapped(fig, .08, .925, "绿色底纹标出每个真实字母的范围;标题给出 真实 → 识别。"
            "失败多为一个字母被切成几段(过度切分)。", 10, C_MUTE)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(len(picks), 1, left=.10, right=.95, top=.88, bottom=.06, hspace=.5)
    for i, e in enumerate(picks):
        ax = fig.add_subplot(gs[i])
        plot_message(ax, e[0], e[1], e[3], e[4], e[5])
        if i == len(picks) - 1:
            ax.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=9)
    pdf.savefig(fig); plt.close(fig)


def main():
    r = st.run(n_messages=400, len_range=(3, 6), seed=7)
    out = os.path.join(os.path.dirname(__file__), "out", "合成消息测试报告.pdf")
    with PdfPages(out) as pdf:
        page_cover(pdf, r)
        page_finding(pdf, r)
        page_perletter(pdf, r)
        page_examples(pdf, r)
        pdf.infodict()["Title"] = "合成消息识别 测试报告"
    print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")
    print(f"orc={r['orc_acc']*100:.1f}%  e2e_char={(1-r['e2e_cer'])*100:.1f}%  "
          f"exact={r['e2e_exact']*100:.1f}%  seg_ok={r['seg_correct']*100:.1f}%")


if __name__ == "__main__":
    main()
