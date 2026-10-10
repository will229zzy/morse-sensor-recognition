#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_word_report.py — 《单词测试报告》中文 PDF + 论文用英文图 + Origin 数据。

先运行:  python word_test.py   与   python ml/cross_batch.py
再运行:  python make_word_report.py
输出:    out/单词测试报告.pdf
         out/words/figures/*.png     (英文,投稿用)
         out/words/origin/*.csv      (每张图对应的数据)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ml"))
import morse_sensor as ms                             # noqa: E402
import dataset as D                                   # noqa: E402
import word_test as WT                                # noqa: E402
from make_comparison_report import (ZH, A4, C_INK, C_MUTE, C_RED, C_BLUE, C_GREY,  # noqa: E402
                                    C_AMBER, txt, page, ax_cn, close)

OUTD = os.path.join(HERE, "out")
WD = os.path.join(OUTD, "words")
ML = os.path.join(OUTD, "ml")
FIG = os.path.join(WD, "figures")
ORI = os.path.join(WD, "origin")
PDF = os.path.join(OUTD, "单词测试报告.pdf")
L26 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SEG_COLOR = {"silence": C_GREY, "relative": C_AMBER, "consensus": C_RED}
SEG_EN = {"silence": "Silence (s)", "relative": "Relative gap", "consensus": "Consensus (ours)"}
M_ORDER = ["rule", "cnn_raw", "rf", "transformer", "svm", "bilstm"]
M_CN = {"rule": "规则解码", "rf": "特征+随机森林", "svm": "特征+SVM", "transformer": "Transformer",
        "bilstm": "BiLSTM", "cnn_raw": "1D-CNN(原始波形)"}
M_EN = {"rule": "Rule-based", "rf": "Features+RF", "svm": "Features+SVM",
        "transformer": "Transformer", "bilstm": "BiLSTM", "cnn_raw": "1D-CNN (raw)"}
M_COLOR = {"rule": C_RED, "rf": "#2e86c1", "svm": "#5dade2", "transformer": C_BLUE,
           "bilstm": "#7fb3d5", "cnn_raw": C_GREY}


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load():
    d = dict(W=pd.read_csv(os.path.join(WD, "per_word.csv")),
             S=pd.read_csv(os.path.join(WD, "short_messages.csv")),
             Y=pd.read_csv(os.path.join(WD, "synthetic_check.csv")),
             XB=pd.read_csv(os.path.join(ML, "cross_batch_summary.csv")),
             C=dict(np.load(os.path.join(ML, "cross_batch.npz"), allow_pickle=True)),
             a=dict(np.load(os.path.join(ML, "dataset.npz"), allow_pickle=True)),
             w=dict(np.load(os.path.join(ML, "words_dataset.npz"), allow_pickle=True)))
    d["gaps"] = labelled_gaps()
    return d


def labelled_gaps():
    """全部 20 份录制里每个间隔的两种量法数值 + 标准答案层级(0 字母内 / 1 字母间 / 2 单词间)。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(D.WORD_DIR, "*.csv"))):
        W, sec, rel, reps, wt, taps = D.word_reps(f)
        lab = WT.true_cuts(taps, reps, W)
        gs, gr = ms.silence_gaps(taps), ms.relative_gaps(taps)
        for i in np.where(lab >= 0)[0]:
            rows.append((W, int(lab[i]), gs[i], gr[i]))
    return pd.DataFrame(rows, columns=["word", "level", "silence_s", "relative"])


def sos_example():
    f = glob.glob(os.path.join(D.WORD_DIR, "SOS *.csv"))[0]
    W, sec, rel, reps, wt, taps = D.word_reps(f)
    return sec, rel, reps[30]


def short_table(S):
    return (S.groupby(["method", "n_words"]).exact.mean().unstack(0) * 100)


def zero_few(XB):
    z = XB[XB.setting == "zero"].set_index("method")
    f = XB[XB.setting == "few"].set_index("method")
    return z, f


def save_png(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 英文投稿图 + Origin 数据
# --------------------------------------------------------------------------- #
def paper_figures(d):
    os.makedirs(ORI, exist_ok=True)
    G = d["gaps"]
    # 1) 间隔归一化
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, xl, lg in ((axes[0], "silence_s", "Silence between presses (s)", False),
                            (axes[1], "relative", "Silence / mean press duration", True)):
        x = G[col].values
        bins = np.logspace(np.log10(max(x.min(), 0.05)), np.log10(x.max()), 70)
        for lv, nm, c in ((0, "within letter", C_BLUE), (1, "between letters", C_AMBER),
                          (2, "between words", C_RED)):
            ax.hist(G[G.level == lv][col], bins=bins, color=c, alpha=.75, label=nm)
        ax.set_xscale("log"); ax.set_xlabel(xl, fontsize=11); ax.set_ylabel("Count", fontsize=11)
        ax.grid(alpha=.25); ax.set_axisbelow(True)
        if lg:
            ax.legend(frameon=False, fontsize=9)
    axes[0].set_title("(a) Raw silence: within-letter and between-letter overlap", fontsize=11)
    axes[1].set_title("(b) Relative gap: three clean levels", fontsize=11)
    plt.tight_layout(); save_png(fig, "gap_normalization.png")
    G.to_csv(os.path.join(ORI, "gap_distributions.csv"), index=False)

    # 2) 短消息
    T = short_table(d["S"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ["silence", "relative", "consensus"]:
        ax.plot(T.index, T[m], "-o", color=SEG_COLOR[m], lw=2 if m == "consensus" else 1.5,
                ms=5, label=SEG_EN[m])
    ax.set_xscale("log"); ax.set_xticks(T.index); ax.set_xticklabels(T.index)
    ax.set_xlabel("Message length (words)", fontsize=11)
    ax.set_ylabel("Exact-message accuracy (%)", fontsize=11)
    ax.set_ylim(60, 101.5); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Short-message decoding (real recordings, sliced)", fontsize=11)
    save_png(fig, "short_message_accuracy.png")
    T.reset_index().to_csv(os.path.join(ORI, "short_message_accuracy.csv"), index=False)

    # 3) 跨批次
    z, f = zero_few(d["XB"])
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(M_ORDER)); wd = .38
    zv = [z.loc[m, "acc_mean"] * 100 for m in M_ORDER]
    ze = [z.loc[m, "acc_std"] * 100 for m in M_ORDER]
    fv = [f.loc[m, "acc_mean"] * 100 for m in M_ORDER]
    b1 = ax.bar(x - wd / 2, zv, wd, yerr=ze, capsize=3, color=[M_COLOR[m] for m in M_ORDER],
                edgecolor="#2c3e50", lw=.5, label="Trained on Aug letters only")
    b2 = ax.bar(x + wd / 2, fv, wd, color=[M_COLOR[m] for m in M_ORDER], alpha=.45,
                edgecolor="#2c3e50", lw=.5, hatch="//", label="+ 16 Sep words (held-out words)")
    for b, v in list(zip(b1, zv)) + list(zip(b2, fv)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([M_EN[m] for m in M_ORDER], fontsize=9)
    ax.set_ylabel("Letter accuracy on Sep recordings (%)", fontsize=10)
    ax.set_ylim(0, 112); ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Cross-session generalisation (5,583 letters, 22 classes)", fontsize=11)
    save_png(fig, "cross_session_generalisation.png")
    pd.DataFrame({"method": [M_EN[m] for m in M_ORDER], "zero_shot_mean_%": zv,
                  "zero_shot_std_%": ze, "few_shot_%": fv}).to_csv(
        os.path.join(ORI, "cross_session_generalisation.csv"), index=False)

    d["W"].to_csv(os.path.join(ORI, "per_word_results.csv"), index=False)
    d["Y"].to_csv(os.path.join(ORI, "synthetic_check.csv"), index=False)


# --------------------------------------------------------------------------- #
# PDF 各页
# --------------------------------------------------------------------------- #
def page_cover(pdf, d):
    W, z = d["W"], zero_few(d["XB"])[0]
    n_w = int(W.n_reps.sum()); n_l = len(d["w"]["y"])
    ex = W.consensus_exact.sum() / n_w * 100
    cer = W.consensus_edit.sum() / W.consensus_chars.sum() * 100
    ex0 = W.silence_exact.sum() / n_w * 100
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    txt(fig, .08, .93, "柔性电阻传感器 · 摩尔斯电码识别", 12.5, C_RED, "bold")
    txt(fig, .08, .895, "单词测试报告", 30, C_INK, "bold")
    txt(fig, .08, .825, "第一次用真实的连续输入(整个单词)检验解码器,并检验在旧数据上"
        "训练的机器学习模型能否直接用在新一批录制上。", 12.5, C_MUTE)
    fig.lines.append(plt.Line2D([.08, .93], [.775, .775], color="#d8dee8", lw=1,
                                transform=fig.transFigure))
    rows = [
        ("数据", "20 个常用词 × 100 遍 = %d 个单词、%d 个字母、16,200 次按压" % (n_w, n_l)),
        ("按压检测", "检测到 %d / 应有 %d,%.2f%%" % (W.taps.sum(), W.expected.sum(),
                                            W.taps.sum() / W.expected.sum() * 100)),
        ("点划判别", "%d 个字母全部正确(100%%),该部分 8 月就已定死,属于盲测" % n_l),
        ("整词识别", "%.2f%%,字错率 %.2f%%(验证集成绩,见第 9 页)" % (ex, cer)),
        ("改进前", "%.2f%%(只看静默秒数,这一列相当于盲测)" % ex0),
        ("换批次", "规则解码 100%%;旧数据训练的模型 %.0f%%–%.0f%%"
         % (z.loc[["rf", "svm", "transformer", "bilstm", "cnn_raw"], "acc_mean"].min() * 100,
            z.loc[["rf", "svm", "transformer", "bilstm", "cnn_raw"], "acc_mean"].max() * 100)),
    ]
    y = .715
    for k, v in rows:
        txt(fig, .08, y, k, 11, C_RED, "bold"); txt(fig, .25, y, v, 11, C_INK); y -= .052

    txt(fig, .08, .375, "一句话总结", 13, C_INK, "bold")
    txt(fig, .08, .345,
        "解码器能直接从电阻曲线读出整个单词,只按一个词的短消息也有 99% 的整条正确率。"
        "更重要的是:换了一批录制(传感器幅度翻倍、按法也变了)之后,规则解码不用任何调整"
        "仍然全对,而在旧数据上训练的机器学习模型掉到 49%–87%。", 11.5, C_INK)
    txt(fig, .08, .215, "报告回答五个问题", 12, C_INK, "bold")
    for i, q in enumerate(["1. 新数据和 8 月的有什么不同?", "2. 原来的解码器卡在哪里,怎么改的?",
                           "3. 整个单词、短消息分别能认多准?",
                           "4. 旧数据上训练的模型能不能直接用?为什么?",
                           "5. 哪些数字可以当测试成绩报,哪些不行?"]):
        txt(fig, .10, .180 - i * .031, q, 11, C_MUTE)
    close(pdf, fig)


def page_data(pdf, d):
    fig = page("这批数据,和 8 月有什么不同", "数据")
    txt(fig, .08, .872,
        "20 份录制,每份是同一个词连续按 100 遍。词有两类:日常词(SOS、STOP、SAFE、DOWN、"
        "UP、LOW、YES、NO、OK、AIR)和无线电通信的常用缩写(CQ、QTH、QSL、QRM、QRP、TNX、"
        "AGN、AR、AS、SK)。三层间隔(点划之间 < 字母之间 < 单词之间)都在。", 11)
    sec, rel, r = sos_example()
    i0 = int(np.searchsorted(sec, r[0].t_start - 3)); i1 = int(np.searchsorted(sec, r[-1].t_end + 3))
    ax = fig.add_axes([.10, .555, .83, .215])
    t0 = sec[i0]
    ax.plot(sec[i0:i1] - t0, rel[i0:i1], color=C_RED, lw=1.5)
    code = "".join(ms.MORSE[c] for c in "SOS")
    for t, c in zip(r, code):
        ax.text(t.t_center - t0, t.height * 1.08, "点" if c == "." else "划", ha="center",
                fontproperties=ZH, fontsize=9, color=C_BLUE)
    for k, (p, q) in enumerate(zip(r[:-1], r[1:])):
        g = q.t_start - p.t_end
        y0 = max(p.height, q.height) * .18
        ax.annotate("", xy=(q.t_start - t0, y0), xytext=(p.t_end - t0, y0),
                    arrowprops=dict(arrowstyle="<->", color=C_MUTE, lw=.8))
        ax.text(0.5 * (p.t_end + q.t_start) - t0, y0 * 1.6, "%.1fs" % g, ha="center", fontsize=7.5,
                color=C_AMBER if k in (2, 5) else C_MUTE, fontweight="bold" if k in (3, 4) else None)
    ax.set_ylim(0, max(t.height for t in r) * 1.3)
    ax_cn(ax, "一遍 SOS:注意 O 里两个划之间停了约 5 秒(加粗),和字母之间(橙色)几乎一样长",
          "时间(秒)", "ΔR/R₀ (%)")

    a, w = d["a"], d["w"]
    fn = list(a["feat_names"]); ih = fn.index("h_mean")

    def med_w(dd, sym):
        v = [s[k, 1] for s, n, yy in zip(dd["X_seq"], dd["lens"], dd["y"])
             for k, c in enumerate(ms.MORSE[L26[yy]]) if c == sym]
        return np.median(v)
    txt(fig, .08, .48, "两批录制的关键差别", 13, C_INK, "bold")
    rows = [("", "8 月单字母", "9 月单词"),
            ("平均峰高 ΔR/R₀", "%.1f%%" % np.median(a["X_feat"][:, ih]),
             "%.1f%%  (翻了一倍多)" % np.median(w["X_feat"][:, ih])),
            ("点的按压时长", "%.2f 秒" % med_w(a, "."), "%.2f 秒" % med_w(w, ".")),
            ("划的按压时长", "%.2f 秒" % med_w(a, "-"), "%.2f 秒" % med_w(w, "-")),
            ("停顿的规律", "按压中心间隔固定", "按得越久,后面停得越久")]
    y = .445
    for i, (k, v1, v2) in enumerate(rows):
        wgt = "bold" if i == 0 else "normal"
        txt(fig, .10, y, k, 10.5, C_MUTE if i == 0 else C_INK, wgt)
        txt(fig, .38, y, v1, 10.5, C_MUTE if i == 0 else C_INK, wgt)
        txt(fig, .62, y, v2, 10.5, C_MUTE if i == 0 else C_RED if i in (1, 4) else C_INK, wgt)
        y -= .036
    txt(fig, .08, .245, "这意味着什么", 13, C_INK, "bold")
    txt(fig, .08, .213,
        "幅度翻倍:任何依赖「多高算划」这种绝对标准的方法都会失效 —— 这正是后面机器学习"
        "迁移失败的主因。停顿规律改变:原来只看「停了几秒」来切字母的办法,会把划后面的长停顿"
        "误当成字母边界 —— 这是解码器在 SOS、QSL 上失败的原因。", 11)
    txt(fig, .08, .095,
        "两批都是同一位实验者、同一个传感器方案录的。这种批次之间的变化在实际使用中一定会出现,"
        "所以「换了批次还能不能用」是比「同一批数据上的准确率」更有意义的指标。", 10.5, C_MUTE)
    close(pdf, fig)


def page_problem(pdf, d):
    fig = page("原来的解码器卡在哪里,怎么改的", "方法")
    G = d["gaps"]
    for j, (col, ttl, xl) in enumerate((("silence_s", "只看停了几秒:字母内(蓝)和字母间(橙)重叠",
                                         "相邻按压之间的静默(秒,对数刻度)"),
                                        ("relative", "除以按压时长后:三层清楚分开",
                                         "静默 ÷ 两侧按压的平均时长(对数刻度)"))):
        ax = fig.add_axes([.10 + j * .44, .64, .37, .19])
        x = G[col].values
        bins = np.logspace(np.log10(max(x.min(), .05)), np.log10(x.max()), 60)
        for lv, nm, c in ((0, "字母内", C_BLUE), (1, "字母间", C_AMBER), (2, "单词间", C_RED)):
            ax.hist(G[G.level == lv][col], bins=bins, color=c, alpha=.75, label=nm)
        ax.set_xscale("log")
        ax_cn(ax, ttl, xl, "间隔个数", 8.5)
        if j:
            ax.legend(prop=ZH, fontsize=8, frameon=False)
    txt(fig, .08, .575, "问题", 13, C_INK, "bold")
    txt(fig, .08, .543,
        "原解码器按「停了几秒」切分字母。9 月这批是按得越久停得越久,划后面的停顿和两个字母之间"
        "的停顿差不多长(左图蓝橙重叠),于是 O 被切成三个 T,SOS 读成 STTTS,QSL 读成 TKSL。"
        "这两个词 100 遍全错,整词正确率只有 %.2f%%。" % (d["W"].silence_exact.sum() / 20), 11)
    txt(fig, .08, .445, "改进一:停顿也要相对地看", 13, C_INK, "bold")
    txt(fig, .08, .413,
        "把停顿时长除以两侧按压的平均时长(右图)。字母内约 1.1、字母间约 2.3、单词间约 5.4,"
        "三层分得很开。这和判点划时「要跟同一个字母里的邻居比」是同一个道理:"
        "停顿长短也要放在上下文里看。", 11)
    txt(fig, .08, .315, "改进二:两种量法取共识", 13, C_INK, "bold")
    txt(fig, .08, .283,
        "但「相对间隔」在 8 月那种「中心间隔固定」的节奏下反而会出错:点后面的停顿相对显长,"
        "被误切。两种量法出错的方式都是「多切一刀」,而且多切的位置不同。所以每条消息里两种量法"
        "各自独立分层,两者都认为该断的地方才断 —— 刚好互相挡掉对方的误切。", 11)
    txt(fig, .08, .165, "没有新增任何参数", 13, C_INK, "bold")
    txt(fig, .08, .133,
        "两种量法的分界都在每条消息内部自动找(三类 Otsu),不写死任何秒数,"
        "也不需要事先知道这条消息是哪种节奏。", 11)
    close(pdf, fig)


def page_consensus(pdf, d):
    fig = page("为什么要取共识:两种节奏都要扛得住", "方法")
    W, Y, T = d["W"], d["Y"], short_table(d["S"])
    txt(fig, .08, .872,
        "下表把三种切分方式放在两类数据上比较。9 月真实单词是「按得越久停得越久」的节奏;"
        "合成消息是用 8 月真实单字母波形、按「中心间隔固定」的节奏拼出来的。", 11)
    cols = [(.08, "切分方式"), (.30, "9 月单词\n整份录制"), (.46, "9 月单词\n只按 1 个词"),
            (.62, "合成消息\n5 秒节奏"), (.78, "合成消息\n英文单词")]
    for x, h in cols:
        txt(fig, x, .80, h, 10, C_MUTE, "bold", lh=1.3)
    fig.lines.append(plt.Line2D([.08, .93], [.755, .755], color="#d8dee8", lw=1,
                                transform=fig.transFigure))
    y = .74
    for m in ["silence", "relative", "consensus"]:
        ex = W[f"{m}_exact"].sum() / W.n_reps.sum() * 100
        r = Y[Y.segmenter == m].set_index("test")
        vals = ["%.2f%%" % ex, "%.2f%%" % T.loc[1, m], "%.1f%%" % (r.loc["cad_5", "exact"] * 100),
                "%.1f%%" % (r.loc["words", "exact"] * 100)]
        col = C_RED if m == "consensus" else C_INK
        txt(fig, .08, y, WT.SEG_CN[m], 10.5, col, "bold" if m == "consensus" else "normal")
        for (x, _), v in zip(cols[1:], vals):
            txt(fig, x, y, v, 10.5, col, "bold" if m == "consensus" else "normal")
        y -= .038
    txt(fig, .08, .60, "怎么读这张表", 13, C_INK, "bold")
    txt(fig, .08, .568,
        "只看秒数:在 8 月节奏下好,在 9 月节奏下差(只按一个词时只有 70%)。"
        "只看相对间隔:正好反过来。两者共识:两边都和各自最合适的那种持平。", 11)
    txt(fig, .08, .48, "为什么这一点重要", 13, C_INK, "bold")
    txt(fig, .08, .448,
        "同一个人在两次录制里就用了两种节奏,换一个人只会差别更大。解码器不能假设"
        "使用者按某一种节奏来按 —— 取共识让它不需要知道,也不需要调参数。", 11)
    txt(fig, .08, .36, "关于合成消息", 13, C_INK, "bold")
    txt(fig, .08, .328,
        "合成消息的节奏是我们在 9 月之前、还没有真实连续数据时,根据 8 月单字母录制推断的。"
        "现在有了真实单词,真实数据的结论优先;合成消息在这里只用来检查「换成另一种节奏时会不会退化」。"
        "共识规则在合成消息上比改动前只差 0.3–0.7 个百分点(400 条里 1–2 条)。", 11)
    close(pdf, fig)


def page_full(pdf, d):
    fig = page("结果一:整份录制直接读出单词", "结果")
    W = d["W"].sort_values("silence_exact")
    ax = fig.add_axes([.10, .565, .83, .27])
    x = np.arange(len(W)); wd = .4
    ax.bar(x - wd / 2, W.silence_exact, wd, color=C_GREY, label="只看静默秒数(改动前)")
    ax.bar(x + wd / 2, W.consensus_exact, wd, color=C_RED, label="两者共识(现行)")
    ax.set_xticks(x); ax.set_xticklabels(W.word, fontsize=8, rotation=45)
    ax.set_ylim(0, 128); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax_cn(ax, "每个词 100 遍里读对了几遍", None, "读对的遍数")
    ax.legend(prop=ZH, fontsize=8.5, frameon=False, loc="upper center", ncol=2)
    D_ = d["W"]
    ex = D_.consensus_exact.sum() / 20; cer = D_.consensus_edit.sum() / D_.consensus_chars.sum() * 100
    txt(fig, .08, .495, "总体", 13, C_INK, "bold")
    txt(fig, .08, .463,
        "2000 遍单词里读对 %d 遍,整词正确率 %.2f%%,字错率 %.2f%%。改动前是 %.2f%% / %.2f%%。"
        % (D_.consensus_exact.sum(), ex, cer, D_.silence_exact.sum() / 20,
           D_.silence_edit.sum() / D_.silence_chars.sum() * 100), 11)
    txt(fig, .08, .395, "错误出在哪一环", 13, C_INK, "bold")
    n_bad = int(D_.n_reps.sum() - D_.consensus_exact.sum())
    n_inc = int(D_.n_reps.sum() - D_.n_complete.sum())
    errs = "、".join(f"{r.word} 读成 {e}" for r in D_.itertuples()
                    for e in str(r.consensus_errors).split("|") if e and e != "nan")
    tail = ("全部来自那几遍漏检或多检了按压的单词(能完整切出的 %d 遍全对):%s"
            % (int(D_.n_complete.sum()), errs)) if n_bad == n_inc else "明细:%s" % errs
    rows = [("按压检测", "检测到 %d / 应有 %d(%.2f%%),%d 份里 %d 份一次不差"
             % (D_.taps.sum(), D_.expected.sum(), D_.taps.sum() / D_.expected.sum() * 100,
                len(D_), int((D_.taps == D_.expected).sum()))),
            ("切分", "字母边界、单词边界的查准率和查全率都是 100%"),
            ("点划判别", "给定正确的字母边界,%d 个字母全部判对" % len(d["w"]["y"])),
            ("剩下的错误", "%d 遍读错,%s" % (n_bad, tail))]
    y = .363
    for k, v in rows:
        txt(fig, .10, y, k, 10.5, C_RED, "bold"); txt(fig, .26, y, v, 10.5, C_INK); y -= .052
    txt(fig, .08, .135,
        "也就是说,只要按压被正确检测出来,后面的切分和判别在这 2000 遍上没有出过错。"
        "剩下的误差都在最前面的「找按压」这一步,通常是一次按得太轻、或两次按得太近被并成一次。",
        10.5, C_MUTE)
    close(pdf, fig)


def page_short(pdf, d):
    fig = page("结果二:只按一个词也能认", "结果")
    T = short_table(d["S"])
    txt(fig, .08, .872,
        "实际使用时没人会把一个词按 100 遍。所以我们从录制里截出 1、2、3、5、10 个词长的"
        "原始信号片段,每段单独送进解码器(从找按压开始重新做一遍),看整条消息全对的比例。", 11)
    ax = fig.add_axes([.12, .55, .78, .25])
    for m in ["silence", "relative", "consensus"]:
        ax.plot(T.index, T[m], "--s" if m == "relative" else "-o", color=SEG_COLOR[m],
                lw=2.2 if m == "consensus" else 1.5, ms=7 if m == "relative" else 5,
                mfc="none" if m == "relative" else SEG_COLOR[m], zorder=3 if m == "relative" else 2,
                label=WT.SEG_CN[m] + ("(与共识几乎重合)" if m == "relative" else ""))
    ax.set_xscale("log"); ax.set_xticks(T.index); ax.set_xticklabels(T.index)
    ax.set_ylim(60, 101.5)
    ax_cn(ax, None, "一条消息里有几个词", "整条全对(%)")
    ax.legend(prop=ZH, fontsize=8.5, frameon=False, loc="lower right")
    n1 = int((d["S"].n_words == 1).sum() / 3)
    txt(fig, .08, .48, "结果", 13, C_INK, "bold")
    txt(fig, .08, .448,
        "只按一个词(共 %d 条)时,现行方法整条全对 %.1f%%,只看秒数的旧思路只有 %.1f%%。"
        "消息越短,自适应分界能参考的停顿越少,旧思路就越吃亏;现行方法在各种长度下都稳定在"
        " 98%%–99%%。" % (n1, T.loc[1, "consensus"], T.loc[1, "silence"]), 11)
    F = d["S"][(d["S"].method == "consensus") & (d["S"].exact == 0)]
    top = F.groupby("word").size().sort_values(ascending=False)
    txt(fig, .08, .35, "剩下的失败,机理很清楚", 13, C_INK, "bold")
    txt(fig, .08, .318,
        "共 %d 条失败,集中在 %s。典型的两种:QTH 单独按一遍读成「?」,CQ 连按两遍读成「CQCQ」。"
        "原因相同:Q→T、Q→C 这种「划接划」的字母间隔,换成相对值只有 1.6 左右,和字母内部的 1.1 "
        "拉不太开;消息很短时,「相对间隔」这一路判断「分不出层次」,而共识规则要求两路都同意才切,"
        "于是该切的地方没切。" % (len(F), "、".join(f"{k}({v})" for k, v in top.head(4).items())),
        11)
    txt(fig, .08, .175, "可以修,但我们没有修", 13, C_INK, "bold")
    txt(fig, .08, .143,
        "修法是让「分不出层次」的那一路不参与投票。但这个漏洞是看着这批数据的失败才发现的,"
        "改了之后在这批数据上的成绩就更难算数。所以规则法在单词测试之前就冻结,"
        "这里如实报告;如果以后有新录制,可以在新数据上验证这个改动。", 11)
    close(pdf, fig)


def page_cross(pdf, d):
    fig = page("结果三:换一批录制,旧模型还能用吗", "结果")
    z, f = zero_few(d["XB"])
    txt(fig, .08, .872,
        "用 8 月的 2661 个单字母训练随机森林、SVM、Transformer、BiLSTM、1D-CNN(超参数与之前的"
        "三方法对比完全相同),直接去认 9 月单词里切出来的 5583 个字母。字母边界都用标准答案,"
        "所以比的纯粹是「判字母」这一步。", 11)
    ax = fig.add_axes([.10, .555, .83, .235])
    x = np.arange(len(M_ORDER)); wd = .38
    zv = [z.loc[m, "acc_mean"] * 100 for m in M_ORDER]; ze = [z.loc[m, "acc_std"] * 100 for m in M_ORDER]
    fv = [f.loc[m, "acc_mean"] * 100 for m in M_ORDER]
    b1 = ax.bar(x - wd / 2, zv, wd, yerr=ze, capsize=3, color=[M_COLOR[m] for m in M_ORDER],
                edgecolor="#2c3e50", lw=.5, label="只用 8 月训练")
    b2 = ax.bar(x + wd / 2, fv, wd, color=[M_COLOR[m] for m in M_ORDER], alpha=.4, hatch="//",
                edgecolor="#2c3e50", lw=.5, label="再加 16 个 9 月单词训练(测另外 4 个)")
    for b, v in list(zip(b1, zv)) + list(zip(b2, fv)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, "%.0f" % v, ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([M_CN[m] for m in M_ORDER], fontproperties=ZH, fontsize=8.5)
    ax.set_ylim(0, 132); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax_cn(ax, None, None, "9 月字母的识别准确率(%)")
    ax.legend(prop=ZH, fontsize=8.5, frameon=False, loc="upper center", ncol=2)
    txt(fig, .08, .48, "结果", 13, C_INK, "bold")
    txt(fig, .08, .448,
        "规则解码不做任何调整,%d 个字母全部正确。在 8 月数据上训练的五种模型降到 %.0f%%–%.0f%%;"
        "Transformer 五次训练之间差了 ±%.0f 个百分点,很不稳定。即使给它们看了 16 个新单词"
        "(按单词切分,测试的 4 个词从没见过),也只恢复到 %.0f%%–%.0f%%。"
        % (len(d["w"]["y"]), min(zv[1:]), max(zv[1:]), z.loc["transformer", "acc_std"] * 100,
           min(fv[1:]), max(fv[1:])), 11)
    txt(fig, .08, .335, "对照:同一批数据内部", 13, C_INK, "bold")
    sm = pd.read_csv(os.path.join(ML, "summary_metrics.csv")).set_index("method")
    cv = sm.loc[["rf", "svm", "transformer", "bilstm", "cnn_raw"], "accuracy"] * 100
    txt(fig, .08, .303,
        ("在 8 月数据内部做交叉验证时,这几种模型都是 %.1f%%–%.2f%%。也就是说,"
         % (cv.min(), cv.max())) +
        "「同一批数据上的高准确率」并不能保证「换一批还能用」—— 这正是只报交叉验证准确率的风险。",
        11)
    txt(fig, .08, .205, "为什么按单词切分", 13, C_INK, "bold")
    txt(fig, .08, .173,
        "同一个词的 100 遍来自同一次录制,节奏和幅度几乎一样。如果按遍数随机切,"
        "测试集里的每个词在训练时都见过,成绩会虚高。按单词切分才是在考「没见过的词」。", 11)
    close(pdf, fig)


def page_why(pdf, d):
    fig = page("为什么机器学习模型迁移不过去", "分析")
    a, w = d["a"], d["w"]
    fn = list(a["feat_names"]); ih = fn.index("h_mean")
    ax = fig.add_axes([.10, .63, .37, .19])
    bins = np.linspace(0, 28, 50)
    ax.hist(a["X_feat"][:, ih], bins=bins, color=C_BLUE, alpha=.6, density=True, label="8 月(训练)")
    ax.hist(w["X_feat"][:, ih], bins=bins, color=C_AMBER, alpha=.6, density=True, label="9 月(测试)")
    ax_cn(ax, "平均峰高:整体右移一倍多", "ΔR/R₀ (%)", "占比", 8.5)
    ax.legend(prop=ZH, fontsize=8, frameon=False)
    ax = fig.add_axes([.56, .63, .37, .19])

    def durs(dd, sym):
        return np.array([s[k, 1] for s, n, yy in zip(dd["X_seq"], dd["lens"], dd["y"])
                         for k, c in enumerate(ms.MORSE[L26[yy]]) if c == sym])
    bins = np.linspace(0, 6, 50)
    for dd, nm, c in ((a, "8 月", C_BLUE), (w, "9 月", C_AMBER)):
        ax.hist(durs(dd, "."), bins=bins, color=c, alpha=.55, density=True, label=nm + " 点")
        ax.hist(durs(dd, "-"), bins=bins, color=c, alpha=.55, density=True, histtype="step",
                lw=1.8, label=nm + " 划")
    ax_cn(ax, "按压时长:9 月的点落在 8 月点和划之间", "半高宽(秒)", "占比", 8.5)
    ax.legend(prop=ZH, fontsize=7.5, frameon=False)

    C = d["C"]; y = C["y"]
    from collections import Counter
    top = {}
    for m in ["rf", "transformer", "cnn_raw"]:
        c = Counter()
        for r in range(C[f"zero_{m}"].shape[0]):
            for t, p in zip(y, C[f"zero_{m}"][r]):
                if t != p:
                    c[(L26[t], L26[p])] += 1
        top[m] = "、".join(f"{a_}→{b_}" for (a_, b_), _ in c.most_common(4))
    txt(fig, .08, .555, "模型学到的是「绝对标准」", 13, C_INK, "bold")
    txt(fig, .08, .523,
        "8 月训练出来的模型,实际上记住了「峰高多少、按多久算划」。9 月的峰高翻了一倍多,"
        "点也变长了,于是大量的点被当成划。错误高度集中:随机森林最常见的是 %s;Transformer 是 %s。"
        "S→O 就是三个点全被当成了划。" % (top["rf"], top["transformer"]), 11)
    txt(fig, .08, .42, "规则解码看的是「相对标准」", 13, C_INK, "bold")
    txt(fig, .08, .388,
        "它只问「同一个字母里,谁比谁高、谁比谁按得久」,判断纯点、纯划字母时的时长分界也是在每条"
        "消息里重新找的。幅度整体翻倍、按压整体变长,这些比较的结果都不变。", 11)
    txt(fig, .08, .30, "一个旁证:1D-CNN 在模型里迁移得最好", 13, C_INK, "bold")
    txt(fig, .08, .268,
        "它吃的波形事先做过「峰值归一化」和「拉伸成固定长度」,幅度和整体快慢的变化都被抹掉了,"
        "所以只降到 %.0f%%。它剩下的错误主要是 %s —— 拉伸抹掉的,正是区分这些字母要用的长短信息。"
        "这说明:能跨批次迁移的,恰恰是做了相对化处理的表示。"
        % (zero_few(d["XB"])[0].loc["cnn_raw", "acc_mean"] * 100, top["cnn_raw"]), 11)
    txt(fig, .08, .135, "对论文的意义", 13, C_RED, "bold")
    txt(fig, .08, .103,
        "柔性传感器的幅度会随器件状态、佩戴、按压力度漂移。基于相对比较的解码不需要重新标定,"
        "这比「同一批数据上 99.9%」更能说明这套材料—算法方案在实际中可用。", 11)
    close(pdf, fig)


def page_summary(pdf, d):
    fig = page("哪些数字可以当测试成绩报", "小结")
    W = d["W"]; n_l = len(d["w"]["y"])
    z = zero_few(d["XB"])[0].loc[["rf", "svm", "transformer", "bilstm", "cnn_raw"], "acc_mean"] * 100
    lo, hi = z.min(), z.max()
    txt(fig, .08, .872,
        "这批单词数据在测试过程中被我们看过,并据此改了切分方法。所以要分清哪些是盲测、"
        "哪些是验证集上的成绩。", 11)
    rows = [("盲测(可直接报)", C_RED, [
                "点划判别:给定字母边界,%d 个字母 100%% 正确。这部分 8 月就定死了,单词数据没影响过它。" % n_l,
                "跨批次泛化:规则解码 100%%,旧数据训练的模型 %.0f%%–%.0f%%。模型的超参数从没在单词数据上调过。"
                % (lo, hi),
                "改动前的切分(只看静默秒数):整词 %.2f%%。" % (W.silence_exact.sum() / 20)]),
            ("验证集(要注明)", C_AMBER, [
                "现行切分(两者共识):整词 %.2f%%,字错率 %.2f%%;只按一个词时整条 %.1f%%。"
                % (W.consensus_exact.sum() / 20, W.consensus_edit.sum() / W.consensus_chars.sum() * 100,
                   short_table(d["S"]).loc[1, "consensus"]),
                "这个切分方法是看了 SOS、QSL 的失败才设计的。我们在没参与设计的 8 月数据上做了独立检查"
                "(35 份录制的字母内、外间隔都能完全分开),并确认它在 8 月节奏的合成消息上没有退化。"])]
    y = .80
    for title, col, items in rows:
        txt(fig, .08, y, title, 13, col, "bold"); y -= .036
        for it in items:
            txt(fig, .10, y, "· " + it, 10.5, C_INK); y -= .062
        y -= .015
    txt(fig, .08, y - .005, "建议写法", 13, C_INK, "bold")
    txt(fig, .08, y - .037,
        "论文里主打两个盲测结果:点划判别在新批次 %d 个字母上 100%%;跨批次对比中规则解码 100%%,"
        "机器学习 %.0f%%–%.0f%%。整词识别 %.1f%% 作为应用演示,注明切分规则是在这批数据上确定的。"
        % (n_l, lo, hi, W.consensus_exact.sum() / 20), 11)
    txt(fig, .08, y - .145, "已准备好的材料", 13, C_INK, "bold")
    txt(fig, .08, y - .177,
        "英文投稿图 3 张(间隔归一化、短消息准确率、跨批次泛化),每张配 Origin 数据;"
        "逐词结果表、短消息逐条结果、合成消息对照表;全部脚本可一键复现。", 11)
    close(pdf, fig)


def main():
    d = load()
    paper_figures(d)
    with PdfPages(PDF) as pdf:
        for fn in (page_cover, page_data, page_problem, page_consensus, page_full,
                   page_short, page_cross, page_why, page_summary):
            fn(pdf, d)
    print("saved", PDF)
    print("figures →", FIG, sorted(os.listdir(FIG)))
    print("origin  →", ORI, sorted(os.listdir(ORI)))


if __name__ == "__main__":
    main()
