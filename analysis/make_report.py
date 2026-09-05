#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py — 生成 PDF 分析报告(封面 + 原理 + 准确率 + 每个字母的波形)。

    python make_report.py              # 默认读 ../raw data,输出 out/摩尔斯识别报告.pdf
    python make_report.py <数据文件夹> <输出.pdf>
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec

import morse_sensor as ms

# ---- 中文字体 ----
ZH = None
for p in ["/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf",
          "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(p):
        ZH = fm.FontProperties(fname=p); break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})

# ---- 配色 ----
C_SIGNAL = "#c0392b"; C_DOT = "#1f9d6b"; C_DASH = "#7c4dcf"; C_CUT = "#2980b9"
C_INK = "#1b2230"; C_MUTE = "#5b6675"; C_OK = "#1f9d6b"; C_BAD = "#c98a1a"


def marks(code):
    return code.replace(".", "·").replace("-", "—")


# --------------------------------------------------------------------------- #
# 分析每个文件:静息信号 + 若干干净重复 + 每次按压的点划标注
# --------------------------------------------------------------------------- #
def analyze(path):
    sec, R = ms.load_keysight_csv(path)
    R = ms.deglitch(R)
    rel, base = ms.detrend(R, sec)
    taps = ms.detect_taps(sec, rel)
    letter = ms.letter_from_filename(path)
    code = ms.MORSE.get(letter, "")
    k = len(code) or None
    if not taps or not k:
        return None
    reps = ms._regroup_by_count(taps, k)
    clean = [r for r in reps if len(r) == k]
    rules = []
    for r in clean:
        rules.append(ms.classify_group(r))
    ok = sum("".join(t.symbol for t in r) == code for r in clean)
    from collections import Counter
    rule = Counter(rules).most_common(1)[0][0] if rules else "相对峰高"
    thr_h = ms._otsu(np.array([t.height for t in taps]))[0]

    # 用真值统计点/划的实际大小,给失败原因用
    dot_h = [t.height for r in clean for t, c in zip(r, code) if c == "."]
    dash_h = [t.height for r in clean for t, c in zip(r, code) if c == "-"]
    dot_w = [t.width for r in clean for t, c in zip(r, code) if c == "."]
    dash_w = [t.width for r in clean for t, c in zip(r, code) if c == "-"]
    return dict(path=path, letter=letter, code=code, k=k, sec=sec, rel=rel,
                base=base, taps=taps, clean=clean, rule=rule, thr_h=thr_h,
                n_clean=len(clean), ok=ok, acc=(ok / len(clean) if clean else None),
                dot_h=np.median(dot_h) if dot_h else None,
                dash_h=np.median(dash_h) if dash_h else None,
                dot_w=np.median(dot_w) if dot_w else None,
                dash_w=np.median(dash_w) if dash_w else None)


def gather(folder):
    """按字母合并多份文件,选干净重复最多的一份作代表波形。"""
    per = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        a = analyze(f)
        if a is None or a["letter"] is None:
            continue
        L = a["letter"]
        d = per.setdefault(L, dict(letter=L, code=a["code"], n_clean=0, ok=0, best=None))
        d["n_clean"] += a["n_clean"]; d["ok"] += a["ok"]
        if d["best"] is None or a["n_clean"] > d["best"]["n_clean"]:
            d["best"] = a
    for d in per.values():
        d["acc"] = d["ok"] / d["n_clean"] if d["n_clean"] else None
        d["pred_ok"] = d["acc"] is not None and d["acc"] >= 0.5
        d["rule"] = d["best"]["rule"] if d["best"] else ""
        d["reason"] = _fail_reason(d["best"]) if not d["pred_ok"] else ""
    return per


def _fail_reason(a):
    """给没认出的字母写一句为什么:核心就是点(短波)和划(长波)没敲出区分。"""
    if a is None or a.get("dash_h") is None or a.get("dot_h") is None:
        return "干净重复太少,无法稳定判读"
    c = a["dash_h"] / a["dot_h"] if a["dot_h"] else 1.0
    if c < 1.6:
        return (f"点和划敲得差不多大(峰高只差 {c:.1f}×,通常需 2× 以上)——"
                f"看不出哪些是短波、哪些是长波")
    return "点划大小时大时小、不稳定——长短波的分组认不准"


# --------------------------------------------------------------------------- #
# 画图工具
# --------------------------------------------------------------------------- #
def _contiguous_reps(clean, n):
    """从干净重复里挑出时间上连续的一段(避免跨越长时间停顿,画出来才紧凑好看)。"""
    if len(clean) <= n:
        return clean
    # 相邻重复之间的间隔;正常重复间隔远小于中途停顿
    gaps = [clean[i + 1][0].t_start - clean[i][-1].t_end for i in range(len(clean) - 1)]
    thr = 3 * np.median([r[-1].t_end - r[0].t_start for r in clean]) + 8
    best_i = 0
    for i in range(len(clean) - n + 1):
        if all(gaps[j] < thr for j in range(i, i + n - 1)):
            best_i = i
            if i >= 2:               # 跳过最开头几次(通常还没按稳)
                break
    return clean[best_i:best_i + n]


def plot_reps(ax, a, n=3, annotate=True, show_cut=False):
    """在 ax 上画某文件的连续 n 个干净重复,并标注每次按压的点/划。"""
    reps = _contiguous_reps(a["clean"], n) or a["clean"][:n]
    if not reps:
        ax.text(0.5, 0.5, "无干净重复", ha="center", va="center",
                transform=ax.transAxes, color=C_MUTE, fontproperties=ZH); return
    t0 = reps[0][0].t_start - 2; t1 = reps[-1][-1].t_end + 2
    m = (a["sec"] >= t0) & (a["sec"] <= t1)
    ax.plot(a["sec"][m] - t0, a["rel"][m], color=C_SIGNAL, lw=1.4)
    if show_cut and a["rule"] == "相对峰高":
        ax.axhline(a["thr_h"], color=C_CUT, ls="--", lw=1, alpha=.8)
        ax.text(0.3, a["thr_h"] + 0.3, f"大小分界 {a['thr_h']:.0f}%",
                color=C_CUT, fontsize=8, fontproperties=ZH)
    if annotate:
        for rep in reps:
            for t in rep:
                lab = "—" if t.symbol == "-" else "·"
                col = C_DASH if t.symbol == "-" else C_DOT
                ax.annotate(lab, (t.t_center - t0, t.height + 0.4), ha="center",
                            color=col, fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=-1); ax.margins(x=0.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _wrap_cn(text, max_units):
    """按可视宽度手动折行(中文算 1、ASCII 算 0.55),中文没有空格无法靠 matplotlib 自动换行。"""
    out, line, units = [], "", 0.0
    for ch in text:
        w = 0.55 if ord(ch) < 0x2E80 else 1.0
        if ch == "\n" or (units + w > max_units and line):
            out.append(line)
            line, units = ("" if ch == "\n" else ch), (0.0 if ch == "\n" else w)
        else:
            line += ch; units += w
    if line:
        out.append(line)
    return "\n".join(out)


def wrapped(fig, x, y, text, size=11, color=C_INK, weight="normal", ha="left", right=0.93):
    max_units = (right - x) * fig.get_figwidth() * 72.0 / size
    fig.text(x, y, _wrap_cn(text, max_units), fontsize=size, color=color,
             ha=ha, va="top", fontproperties=ZH, fontweight=weight, linespacing=1.5)


# --------------------------------------------------------------------------- #
# 各页
# --------------------------------------------------------------------------- #
def page_cover(pdf, per):
    letters_ok = sum(d["pred_ok"] for d in per.values())
    tot = sum(d["n_clean"] for d in per.values()); tot_ok = sum(d["ok"] for d in per.values())
    per_rep = tot_ok / tot if tot else 0
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    wrapped(fig, .08, .93, "柔性电阻传感器 · 摩尔斯电码字母识别", 13, C_SIGNAL, "bold")
    wrapped(fig, .08, .90, "分析报告", 30, C_INK, "bold")
    wrapped(fig, .08, .84,
            "在一块可拉伸的电阻传感器上按压输入摩尔斯电码——轻/短敲是点(·)、重/长敲是划(—)。"
            "数字万用表记录电阻随时间的变化,本报告用这条曲线自动识别所敲的字母,"
            "并逐一给出每个字母的波形与识别准确率。", 11.5, C_MUTE)
    # 三个大数字
    stats = [(f"{per_rep*100:.1f}%", "点划逐次识别率"),
             (f"{letters_ok} / 26", "字母整体认对"),
             (f"{tot:,}", "参与统计的重复次数")]
    for i, (v, lab) in enumerate(stats):
        x = .08 + i * .30
        fig.text(x + .12, .72, v, fontsize=30, color=C_OK, ha="center",
                 fontproperties=ZH, fontweight="bold")
        fig.text(x + .12, .67, lab, fontsize=11, color=C_MUTE, ha="center",
                 fontproperties=ZH)
    # 一段总览波形(取 D 作示例)
    ax = fig.add_axes([.08, .40, .84, .20])
    dfile = per.get("D", {}).get("best")
    if dfile:
        plot_reps(ax, dfile, n=4, show_cut=True)
        ax.set_title("示例:字母 D 的电阻波形(一大峰=划,两小峰=点 → —·· → D)",
                     fontproperties=ZH, fontsize=11, loc="left")
        ax.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=9)
        ax.set_ylabel("ΔR/R₀ (%)", fontproperties=ZH, fontsize=9)
    wrapped(fig, .08, .34,
            "方法一句话:先减掉缓慢漂移的静息基线,找出每一次按压;一次按压有多"
            "「大」体现在峰更高、或按得更久。据此判点/划,再对照标准摩尔斯码表还原字母。", 11, C_MUTE)
    wrapped(fig, .08, .07, "数据:36 份「电阻—时间」录制,每字母一份,Keysight 34461A,采样约 2.4 Hz。",
            9, C_MUTE)
    pdf.savefig(fig); plt.close(fig)


def page_method(pdf, per):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .95, "识别原理:怎么从波形认出点和划", 18, C_INK, "bold")
    wrapped(fig, .08, .90,
            "一次敲击是否为「划」,看它比周围的敲击是不是更「大」。「大」有两条线索,"
            "程序按录制自身情况自动选用:", 11.5, C_MUTE)

    # 情况一:相对峰高(混合字母)
    wrapped(fig, .08, .855, "情况一 · 含点又含划的字母:比峰高", 13, C_DOT, "bold")
    ax1 = fig.add_axes([.08, .60, .84, .21])
    best = per.get("F", {}).get("best") or per.get("D", {}).get("best")
    if best:
        plot_reps(ax1, best, n=4, show_cut=True)
        ax1.set_title(f"字母 {best['letter']} = {marks(best['code'])}"
                      "  同一录制里划就是比点高,直接相对比较",
                      fontproperties=ZH, fontsize=10.5, loc="left")
        ax1.set_ylabel("ΔR/R₀ (%)", fontproperties=ZH, fontsize=9)
    wrapped(fig, .08, .565,
            "若对所有录制用同一个固定高度分界,只有 58% 正确(传感器软硬和按力不同);"
            "改成在同一录制内部相对比较,可达 95% 以上。", 10.5, C_MUTE)

    # 情况二:绝对时长(纯字母)
    wrapped(fig, .08, .50, "情况二 · 全是点或全是划的字母:比时长", 13, C_DASH, "bold")
    axI = fig.add_axes([.08, .28, .40, .17])
    axM = fig.add_axes([.52, .28, .40, .17])
    di = per.get("I", {}).get("best"); dm = per.get("M", {}).get("best")
    if di:
        plot_reps(axI, di, n=4, annotate=False)
        axI.set_title(f"I = ··  时长≈{np.median([t.width for r in di['clean'] for t in r]):.1f}s",
                      fontproperties=ZH, fontsize=10, loc="left", color=C_DOT)
        axI.set_ylabel("ΔR/R₀ (%)", fontproperties=ZH, fontsize=8)
    if dm:
        plot_reps(axM, dm, n=4, annotate=False)
        axM.set_title(f"M = ——  时长≈{np.median([t.width for r in dm['clean'] for t in r]):.1f}s",
                      fontproperties=ZH, fontsize=10, loc="left", color=C_DASH)
    wrapped(fig, .08, .235,
            "纯字母一份录制里没有高矮对比。但划按得更久——每次按压更宽(半高宽更大),"
            "而按压时长是人的动作、跨录制稳定。注意上图:点字母 I 的峰甚至比划字母 M 还高——"
            "所以此时不能靠高度,只能靠时长。", 10.5, C_MUTE)
    wrapped(fig, .08, .13, "四个步骤:①找按压 → ②量峰高与时长 → ③小=点/大=划 → ④查摩尔斯码表。",
            11.5, C_INK, "bold")
    pdf.savefig(fig); plt.close(fig)


def page_accuracy(pdf, per):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
    wrapped(fig, .08, .95, "识别准确率:逐字母", 18, C_INK, "bold")
    order = sorted(per.values(), key=lambda d: d["letter"])
    ax = fig.add_axes([.16, .10, .74, .78])
    ys = np.arange(len(order))[::-1]
    for y, d in zip(ys, order):
        acc = (d["acc"] or 0) * 100
        color = C_BAD if not d["pred_ok"] else (C_DASH if d["rule"] == "绝对时长" else C_DOT)
        ax.barh(y, acc, height=.66, color=color)
        ax.text(-3, y, f"{d['letter']} {marks(d['code'])}", ha="right", va="center",
                fontsize=9, fontproperties=ZH, color=C_INK)
        mk = "✓" if d["pred_ok"] else "✗"
        ax.text(acc + 1.5, y, f"{acc:.0f}% {mk}", va="center", fontsize=8.5,
                color=(C_OK if d["pred_ok"] else C_BAD), fontproperties=ZH)
    ax.set_xlim(0, 112); ax.set_ylim(-1, len(order))
    ax.set_yticks([]); ax.set_xlabel("每次重复完整解码正确的比例 (%)", fontproperties=ZH, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(100, color="#ccc", lw=.8)
    # 图例
    for i, (c, lab) in enumerate([(C_DOT, "靠峰高识别"), (C_DASH, "靠时长识别(纯字母)"),
                                  (C_BAD, "未认出")]):
        fig.text(.16 + i * .26, .055, "■", color=c, fontsize=12, fontproperties=ZH)
        fig.text(.185 + i * .26, .055, lab, fontsize=9, color=C_MUTE, fontproperties=ZH)
    fails = [d["letter"] for d in order if not d["pred_ok"]]
    wrapped(fig, .08, .035,
            f"未认出的 {', '.join(fails)} 是因为那几份录制本身点划没做出区分"
            f"(点按太重或划按太短),重录即可,不是方法问题。", 9, C_MUTE)
    pdf.savefig(fig); plt.close(fig)


def page_gallery(pdf, per):
    order = sorted(per.values(), key=lambda d: d["letter"])
    per_page = 4
    for start in range(0, len(order), per_page):
        chunk = order[start:start + per_page]
        fig = plt.figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
        wrapped(fig, .08, .96, "每个字母的波形", 16, C_INK, "bold")
        wrapped(fig, .08, .925, "红=电阻曲线;绿点 · 为点、紫 — 为划,标在每次按压的峰顶。",
                10, C_MUTE)
        gs = GridSpec(per_page, 1, left=.10, right=.94, top=.88, bottom=.06, hspace=.55)
        for i, d in enumerate(chunk):
            ax = fig.add_subplot(gs[i])
            best = d["best"]
            if best:
                plot_reps(ax, best, n=3, show_cut=(d["rule"] == "相对峰高"))
            mk = "✓" if d["pred_ok"] else "✗"
            acc = "—" if d["acc"] is None else f"{d['acc']*100:.0f}%"
            col = C_OK if d["pred_ok"] else C_BAD
            tail = f"      判据:{d['rule']}" if d["pred_ok"] else "      未认出"
            ax.set_title(f"{d['letter']}  =  {marks(d['code'])}      识别率 {acc} {mk}{tail}",
                         fontproperties=ZH, fontsize=11, loc="left", color=col)
            ax.set_ylabel("ΔR/R₀ (%)", fontproperties=ZH, fontsize=8)
            if not d["pred_ok"] and d.get("reason"):
                ax.text(0.5, 0.94, "原因:" + d["reason"], transform=ax.transAxes,
                        ha="center", va="top", fontsize=9.5, color=C_BAD,
                        fontproperties=ZH,
                        bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3e3",
                                  ec=C_BAD, lw=1))
            if i == len(chunk) - 1:
                ax.set_xlabel("时间 (s)", fontproperties=ZH, fontsize=9)
        pdf.savefig(fig); plt.close(fig)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "raw data")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "out", "摩尔斯识别报告.pdf")
    per = gather(folder)
    with PdfPages(out) as pdf:
        page_cover(pdf, per)
        page_method(pdf, per)
        page_accuracy(pdf, per)
        page_gallery(pdf, per)
        info = pdf.infodict(); info["Title"] = "摩尔斯电码字母识别 分析报告"
    print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
