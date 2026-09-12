#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_comparison_report.py — 生成《三种方法对比》中文 PDF 报告(写给非算法读者)。

    python make_comparison_report.py        # → out/三方法对比报告.pdf

图表全部用中文重画,不复用投稿用的英文图。数据来自 out/ml/results.npz。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager as fm                          # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ml"))
import morse_sensor as ms                                          # noqa: E402
import dataset as ds                                               # noqa: E402
import dataset as D                                                # noqa: E402

RAW = os.path.join(HERE, "..", "raw data")
ML = os.path.join(HERE, "out", "ml")
OUT = os.path.join(HERE, "out", "三方法对比报告.pdf")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

ZH = None
for _p in ["/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(_p):
        ZH = fm.FontProperties(fname=_p); break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})

C_INK, C_MUTE, C_RED = "#1b2230", "#5b6675", "#c0392b"
C_BLUE, C_GREY, C_AMBER = "#1a5276", "#95a5a6", "#e0a02a"
A4 = (8.27, 11.69)

METHODS = ["rule", "rf", "svm", "transformer", "bilstm", "cnn_raw"]
CN = {"rule": "规则解码\n(不需训练)", "rf": "特征+随机森林", "svm": "特征+SVM",
      "transformer": "Transformer", "bilstm": "BiLSTM", "cnn_raw": "1D-CNN\n(原始波形)"}
COLOR = {"rule": C_RED, "rf": "#2e86c1", "svm": "#5dade2", "transformer": C_BLUE,
         "bilstm": "#7fb3d5", "cnn_raw": C_GREY}
FEAT_CN = {"n_elements": "按压次数", "h_max/h_min": "最高/最矮峰之比",
           "dur_max/dur_min": "最长/最短按压之比", "dur_median": "按压时长(中位)",
           "h_mean": "平均峰高"}
for _i in range(1, 5):
    FEAT_CN["h_rel_%d" % _i] = "第%d次相对峰高" % _i
    FEAT_CN["dur_%d" % _i] = "第%d次按压时长" % _i
    FEAT_CN["dur_rel_%d" % _i] = "第%d次相对时长" % _i
    FEAT_CN["gap_%d" % _i] = "第%d次前的间隔" % _i


# --------------------------------------------------------------------------- #
# 排版助手
# --------------------------------------------------------------------------- #
def _wrap_cn(text, max_units):
    """按可视宽度手动折行(中文算 1、ASCII 算 0.55)。中文没有空格,matplotlib 无法自动换行。"""
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


def txt(fig, x, y, s, size=10.5, color=C_INK, weight="normal", right=0.93, lh=1.55):
    fig.text(x, y, _wrap_cn(s, (right - x) * fig.get_figwidth() * 72.0 / size),
             fontsize=size, color=color, ha="left", va="top",
             fontproperties=ZH, fontweight=weight, linespacing=lh)


def page(title, kicker=""):
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    if kicker:
        txt(fig, .08, .958, kicker, 9.5, C_RED, "bold")
    txt(fig, .08, .937, title, 19, C_INK, "bold")
    fig.lines.append(plt.Line2D([.08, .93], [.905, .905], color="#d8dee8",
                                lw=1, transform=fig.transFigure))
    return fig


def ax_cn(ax, title=None, xlabel=None, ylabel=None, size=9):
    if title:
        ax.set_title(title, fontproperties=ZH, fontsize=size + 1.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=ZH, fontsize=size)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=ZH, fontsize=size)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(ZH)
    ax.grid(alpha=.25); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def close(pdf, fig):
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load():
    R = dict(np.load(os.path.join(ML, "results.npz"), allow_pickle=True))
    S = pd.read_csv(os.path.join(ML, "summary_metrics.csv")).set_index("method")
    return R, S


def one_block(letter, min_reps=1):
    """取某字母的一段录制(时间轴, 波形, 重复列表, 点划时长分界)。"""
    for f in sorted(glob.glob(os.path.join(RAW, "%s*.csv" % letter))):
        for LL, sec, rel, reps, wt in ds.letter_blocks(f):
            if LL == letter and len(reps) >= min_reps:
                return sec, rel, reps, wt
    return None


def et_waveforms():
    """取 E 与 T 的真实波形若干条,以及各自的中位按压时长。"""
    got = {}
    for L in ("E", "T"):
        b = one_block(L, 18)
        if not b:
            continue
        sec, rel, reps, _ = b
        clips, durs = [], []
        for r in reps[5:17]:
            i0 = int(np.searchsorted(sec, r[0].t_start - 1.2))
            i1 = int(np.searchsorted(sec, r[-1].t_end + 1.2))
            seg = rel[max(0, i0):i1].astype(float)
            if len(seg) > 4:
                seg = seg - np.percentile(seg, 10)
                clips.append(seg / max(seg.max(), 1e-6))
                durs.append(r[0].width)
        got[L] = (clips, float(np.median(durs)) if durs else 0.0)
    return got


# --------------------------------------------------------------------------- #
# 各页
# --------------------------------------------------------------------------- #
def page_cover(pdf, R, S):
    n = int(R["cm_rule"].sum())
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    txt(fig, .08, .93, "柔性电阻传感器 · 摩尔斯电码识别", 12.5, C_RED, "bold")
    txt(fig, .08, .895, "三种方法对比报告", 30, C_INK, "bold")
    txt(fig, .08, .825,
        "我们用三种原理完全不同的方法,把同一批传感器信号各识别了一遍,"
        "看看结论是否一致,以及深度学习到底有没有帮助。", 12.5, C_MUTE)
    fig.lines.append(plt.Line2D([.08, .93], [.775, .775], color="#d8dee8", lw=1,
                                transform=fig.transFigure))

    rows = [
        ("数据规模", "%d 次敲击 · 26 个字母 · 36 份录制(全部可用,未丢弃任何一份)" % n),
        ("验证方式", "5 折交叉验证,每次敲击都恰好被测试一次"),
        ("最好成绩", "%.2f%%(手工特征 + 随机森林,错 1 次)" % (S.loc["rf", "accuracy"] * 100)),
        ("规则解码", "%.2f%%(错 2 次,且完全不需要训练数据)" % (S.loc["rule", "accuracy"] * 100)),
        ("深度模型", "%.2f%%(Transformer,错 7 次)" % (S.loc["transformer", "accuracy"] * 100)),
        ("主要结论", "三种方法结论一致,深度学习没有带来提升"),
    ]
    y = .715
    for k, v in rows:
        txt(fig, .08, y, k, 11, C_RED, "bold")
        txt(fig, .25, y, v, 11, C_INK)
        y -= .052

    txt(fig, .08, .375, "一句话总结", 13, C_INK, "bold")
    txt(fig, .08, .345,
        "识别准确率高,并不是因为算法特别聪明,而是因为传感器信号本身足够干净、"
        "点和划本来就分得开。三种互不相干的方法都得到几乎相同的结果,正好证明了这一点 —— "
        "这比只报一个模型的高准确率更有说服力。", 11.5, C_INK)

    txt(fig, .08, .215, "报告回答四个问题", 12, C_INK, "bold")
    qs = ["1. 三种方法分别怎么做?(不懂算法也能看明白)",
          "2. 它们的结果一致吗?",
          "3. 为什么把波形直接丢给神经网络反而更容易出错?",
          "4. 数据本身有没有问题?"]
    for i, q in enumerate(qs):
        txt(fig, .10, .180 - i * .033, q, 11, C_MUTE)
    close(pdf, fig)


def page_what(pdf):
    fig = page("我们做了什么", "背景")
    txt(fig, .08, .872,
        "传感器记录下来的是一条「电阻随时间变化」的曲线。手指按一下,电阻就鼓起一个包;"
        "按得轻而短是「点」,按得重而久是「划」。把一串点划按摩尔斯码表翻译过来,就是一个字母。", 11)
    txt(fig, .08, .800,
        "要把曲线还原成字母,中间必须有一套判断规则。问题是:这套规则应该由人写出来,"
        "还是交给机器去学?这正是「要不要用机器学习」的争议所在。", 11)
    txt(fig, .08, .728,
        "所以我们没有二选一,而是把三种做法都实现了一遍,让它们在完全相同的数据、"
        "完全相同的划分上比较。只有这样,比出来的差异才真的是方法本身的差异。", 11)

    b = one_block("H", 8)
    if b:
        sec, rel, reps, wt = b
        r = reps[6]
        i0 = int(np.searchsorted(sec, r[0].t_start - 2))
        i1 = int(np.searchsorted(sec, r[-1].t_end + 2))
        ax = fig.add_axes([.11, .415, .80, .21])
        seg = rel[i0:i1]
        ax.plot(sec[i0:i1] - sec[i0], seg, color=C_RED, lw=1.7)
        ax.set_ylim(min(0, seg.min()), seg.max() * 1.42)   # 给标注留出空间,免得顶到标题
        ms.classify_group(r, wt)
        for tp in r:
            ax.annotate("点" if tp.symbol == "." else "划",
                        xy=(tp.t_center - sec[i0], tp.height),
                        xytext=(tp.t_center - sec[i0], tp.height * 1.28),
                        ha="center", va="bottom", fontproperties=ZH, fontsize=10,
                        color=C_BLUE,
                        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1,
                                        shrinkB=3))
        ax_cn(ax, "一次字母 H 的敲击:四个矮包 = 四个点 = H",
              "时间(秒)", "电阻变化 ΔR/R₀ (%)")

    txt(fig, .08, .335, "为什么要比三种,而不是只做一种", 13, C_INK, "bold")
    txt(fig, .08, .303,
        "如果只用一种方法得到很高的准确率,读者没法判断这个结果是传感器信号好,"
        "还是算法凑巧调得好。三种原理完全不同的方法同时给出接近的结果,"
        "才能说明可分性来自材料与器件本身。对材料类论文来说,这是更强的论据。", 11)

    txt(fig, .08, .195, "一个重要前提:公平", 13, C_INK, "bold")
    txt(fig, .08, .163,
        "三种方法共用同一套信号前端 —— 去噪、去漂移、找出每一次按压 —— 只有「怎么判断」"
        "这一步不同。这样比出来的才是判断方式本身,而不是前处理的差别。", 11)
    txt(fig, .08, .075,
        "另外,训练和测试是按录制的时间顺序切分的,不是随机打乱。相邻两次敲击的传感器"
        "状态几乎一样,随机打乱会让成绩虚高 —— 这一点在审稿时经常被追问。", 10.5, C_MUTE)
    close(pdf, fig)


def page_methods(pdf):
    fig = page("三种方法分别是什么", "方法")
    blocks = [
        ("① 规则解码 —— 像人一样看波形", C_RED,
         "把肉眼判断的经验直接写成规则:在同一个字母里,鼓包更高或按得更久的那个就是划,"
         "另一个就是点。如果整个字母的几次按压高矮差不多(比如 S 是三个点),"
         "就改看按压持续了多久。\n"
         "特点:不需要任何训练数据,拿来就能用,而且每一步都能讲清楚为什么。"),
        ("② 手工特征 + 机器学习 —— 人给尺子,机器学怎么切", "#2e86c1",
         "先把每次按压量成几个数字:峰有多高、按了多久、离上一次多远。"
         "再把这些数字交给常见的机器学习模型(随机森林、SVM),让它自己找分界线。\n"
         "特点:需要一些训练数据,但用的还是我们指定的物理量,因此能看出哪个量最重要。"),
        ("③ Transformer —— 让机器自己找规律", C_BLUE,
         "Transformer 是目前最新的一类模型,它的机制正好是把序列里的各个元素互相比较。"
         "而我们的核心规律恰恰是:点和划要跟同一个字母里的邻居比,不能看绝对值。\n"
         "特点:我们用规则显式写下来的东西,它是自己学出来的 —— 两条路殊途同归。"),
        ("对照组:1D-CNN 直接吃原始波形 —— 不给任何物理量", C_GREY,
         "这一路刻意不做任何物理处理:把波形拉伸成固定长度,直接喂给神经网络,"
         "也就是最常见的做法。用它来回答一个问题:那些物理量到底有没有用?\n"
         "结果:它的表现最差,而且错得很有规律 —— 下一页专门讲这件事。"),
    ]
    y = .868
    for title, col, body in blocks:
        txt(fig, .08, y, title, 12.5, col, "bold")
        txt(fig, .08, y - .030, body, 10.8, C_INK)
        y -= .172

    # 一眼看完的速览表
    txt(fig, .08, .215, "一眼看完", 12.5, C_INK, "bold")
    cols = [(.08, "方法"), (.35, "要不要训练数据"), (.58, "能不能解释"), (.83, "本次错误")]
    for cx, head in cols:
        txt(fig, cx, .185, head, 10, C_MUTE, "bold")
    fig.lines.append(plt.Line2D([.08, .93], [.176, .176], color="#d8dee8", lw=1,
                                transform=fig.transFigure))
    table = [("规则解码", "不需要", "每一步都能讲清楚", "2", C_RED),
             ("手工特征 + 机器学习", "少量", "能看出哪个量最重要", "1", "#2e86c1"),
             ("Transformer", "较多", "黑箱", "7", C_BLUE),
             ("1D-CNN(原始波形)", "较多", "黑箱", "31", C_GREY)]
    yy = .162
    for name, need, explain, err, col in table:
        txt(fig, .08, yy, name, 10.5, col, "bold")
        txt(fig, .35, yy, need, 10.5, C_INK)
        txt(fig, .58, yy, explain, 10.5, C_INK)
        txt(fig, .83, yy, err, 10.5, C_INK, "bold")
        yy -= .034

    txt(fig, .08, .020,
        "四种做法用的是同一批数据、同一套划分,成绩可以直接放在一起比较。", 10, C_MUTE)
    close(pdf, fig)


def page_results(pdf, R, S):
    fig = page("结果:三种方法结论一致", "结果一")
    n = int(R["cm_rule"].sum())
    txt(fig, .08, .872,
        "在全部 %d 次敲击上,六种做法(三种主方法,外加 SVM、BiLSTM 两个旁证和一个对照组)"
        "的准确率都在 98.8%% 以上。差别不在准不准,而在错几次 —— "
        "所以右图用错误次数来看,更清楚。" % n, 11)

    x = np.arange(len(METHODS))
    ax = fig.add_axes([.10, .615, .38, .195])
    acc = [S.loc[m, "accuracy"] * 100 for m in METHODS]
    ax.bar(x, acc, .62, color=[COLOR[m] for m in METHODS], edgecolor="#2c3e50", lw=.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CN[m].replace("\n", " ") for m in METHODS],
                       fontproperties=ZH, fontsize=7, rotation=32, ha="right")
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    for xi, a in zip(x, acc):
        ax.text(xi, a + 2.5, "%.2f" % a, ha="center", fontsize=7)
    ax_cn(ax, "准确率(%)")

    ax = fig.add_axes([.58, .615, .35, .195])
    err = [int(S.loc[m, "n_errors"]) for m in METHODS]
    ax.bar(x, err, .62, color=[COLOR[m] for m in METHODS], edgecolor="#2c3e50", lw=.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CN[m].replace("\n", " ") for m in METHODS],
                       fontproperties=ZH, fontsize=7, rotation=32, ha="right")
    for xi, e in zip(x, err):
        ax.text(xi, e + max(err) * .035, str(e), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(err) * 1.28)
    ax_cn(ax, "错误次数(共 %d 次敲击)" % n)

    txt(fig, .08, .505, "怎么读这两张图", 13, C_INK, "bold")
    txt(fig, .08, .473,
        "左图六根柱子几乎一样高,说明所有方法都能把字母认出来;右图才看得出差别:"
        "规则解码错 2 次,加上机器学习错 1 次,Transformer 错 7 次,"
        "而完全不用物理量的 1D-CNN 错了 31 次。", 11)

    txt(fig, .08, .395, "三点结论", 13, C_INK, "bold")
    concl = [
        "① 深度模型没有带来提升。它们和规则法看的是同样的输入,只是把一条本来就确定的"
        "规律重新学了一遍,还学得没那么准。",
        "② 规则解码与随机森林只差 1 次。在 %d 次里,这属于统计上分不出高下,"
        "不能说机器学习更好。" % n,
        "③ 真正拉开差距的是有没有用物理量,而不是模型够不够深 —— "
        "不用物理量的那一路,错误多了十几倍。",
    ]
    for i, s in enumerate(concl):
        txt(fig, .09, .363 - i * .075, s, 10.8, C_INK)

    txt(fig, .08, .135, "这对论文意味着什么", 12.5, C_RED, "bold")
    txt(fig, .08, .103,
        "可以写成「我们用三种独立方法交叉验证,结果一致」。这是在证明器件性能,"
        "而不是在比拼算法 —— 审稿人更难挑毛病,同时机器学习相关的图表也一应俱全。", 11)
    close(pdf, fig)


def page_et(pdf, R):
    fig = page("E 和 T:为什么直接把波形丢给神经网络会出错", "结果二")
    txt(fig, .08, .872,
        "字母 E 是一个点,字母 T 是一个划。它们都只有一次按压,鼓包的形状一模一样,"
        "唯一的区别是 —— 按住的时间不同。", 11)

    got = et_waveforms()
    ax = fig.add_axes([.10, .615, .36, .195])
    for L, col in (("E", C_BLUE), ("T", C_AMBER)):
        clips, dur = got.get(L, ([], 0))
        for c in clips[:8]:
            ax.plot(np.arange(len(c)) / 2.37, c, color=col, lw=1, alpha=.65)
        ax.plot([], [], color=col, lw=2, label="%s:按住约 %.1f 秒" % (L, dur))
    ax_cn(ax, "真实录制:宽度明显不同", "时间(秒)", "归一化鼓包")
    ax.legend(prop=ZH, fontsize=8, frameon=False, loc="upper right")

    ax = fig.add_axes([.57, .615, .36, .195])
    y = R["y"]
    for L, col in (("E", C_BLUE), ("T", C_AMBER)):
        m = y == LETTERS.index(L)
        ax.plot(np.arange(R["X_raw"].shape[1]), R["X_raw"][m].mean(0), color=col, lw=2,
                label="%s(拉伸后)" % L)
    ax_cn(ax, "拉成固定长度后:几乎重合", "重采样点", "归一化鼓包")
    ax.legend(prop=ZH, fontsize=8, frameon=False, loc="upper right")

    txt(fig, .08, .555, "问题出在哪", 13, C_INK, "bold")
    txt(fig, .08, .523,
        "神经网络要求输入长度固定,所以常规做法是把每段波形拉伸成同样长度。"
        "可是一旦拉伸,「按了多久」这个信息就被抹掉了(见右上图,两条线几乎重合)—— "
        "而这恰恰是区分 E 和 T 的唯一线索。", 11)

    cm = R["cm_cnn_raw"]
    iE, iT = LETTERS.index("E"), LETTERS.index("T")
    n_et = int(cm[iE, iT] + cm[iT, iE])
    tot_err = int(cm.sum() - np.trace(cm))
    txt(fig, .08, .425, "结果是可以预料的", 13, C_INK, "bold")
    txt(fig, .08, .393,
        "直接吃原始波形的 1D-CNN 一共错了 %d 次,其中 %d 次就是把 E 和 T 弄混,占 %.0f%%。"
        "而保留了按压时长的其它几种方法,在 E 和 T 上一次都没错。"
        % (tot_err, n_et, n_et / tot_err * 100), 11)

    dfr = pd.read_csv(os.path.join(ML, "origin", "tsne_2d_raw.csv"))
    dff = pd.read_csv(os.path.join(ML, "origin", "tsne_2d_feature.csv"))
    for j, (df, name) in enumerate(((dfr, "只看波形形状"), (dff, "用上物理量"))):
        a = fig.add_axes([.10 + j * .44, .155, .36, .185])
        oth = ~df["label"].isin(["E", "T"])
        a.scatter(df["tsne_1"][oth], df["tsne_2"][oth], s=3, color="#d7dde6", lw=0)
        for L, col in (("E", C_BLUE), ("T", C_AMBER)):
            m = df["label"] == L
            a.scatter(df["tsne_1"][m], df["tsne_2"][m], s=7, color=col, lw=0, label=L)
        a.set_xticks([]); a.set_yticks([])
        ax_cn(a, name)
        a.grid(False)
        a.legend(prop=ZH, fontsize=8, frameon=False, loc="best")

    txt(fig, .08, .105,
        "上面两张图把每一次敲击画成一个点,相似的挨在一起,灰色是其余 24 个字母。"
        "左图只看波形形状,E(蓝)和 T(黄)混成一团;右图用上按压时长后,两者完全分开。",
        10.5, C_MUTE)
    close(pdf, fig)


def page_feature(pdf, R):
    fig = page("机器学习自己选出了我们用的那个物理量", "结果三")
    txt(fig, .08, .872,
        "随机森林在学习的时候,会顺便算出哪个测量值对判断最有用。"
        "这一步完全由数据决定,我们没有干预。结果如下:", 11)

    imp, names = R["importance"], list(R["feat_names"])
    o = np.argsort(imp)[::-1][:12][::-1]
    ax = fig.add_axes([.32, .455, .60, .35])
    cols = [C_RED if names[i].startswith("dur") else "#2e86c1" for i in o]
    ax.barh(range(len(o)), imp[o], color=cols, edgecolor="#2c3e50", lw=.5)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([FEAT_CN.get(names[i], names[i]) for i in o],
                       fontproperties=ZH, fontsize=9)
    for i, v in enumerate(imp[o]):
        ax.text(v + .0015, i, "%.3f" % v, va="center", fontsize=7.5)
    ax.set_xlim(0, imp[o].max() * 1.20)
    ax_cn(ax, "各测量值的重要程度(越长越重要,红色 = 和时长有关)")

    top = [FEAT_CN.get(names[i], names[i]) for i in np.argsort(imp)[::-1][:4]]
    txt(fig, .08, .395, "读出来的信息", 13, C_INK, "bold")
    txt(fig, .08, .363,
        "排在最前面的四项是:%s。其中三项都和按压持续多久有关(图中红色的条)。"
        % "、".join(top), 11)
    txt(fig, .08, .295,
        "这正是我们在规则里用来区分点和划的那个量。也就是说,机器学习独立地、"
        "完全从数据出发,得出了和我们相同的结论。", 11)

    txt(fig, .08, .205, "为什么这张图值得放进论文", 13, C_INK, "bold")
    txt(fig, .08, .173,
        "它把「我们选的物理量是对的」从一句主观判断,变成了一个可验证的结果:"
        "不是我们猜对了,而是数据本身就这么说。对于强调材料与算法协同设计的论文,"
        "这是一条很自然的证据链。", 11)
    txt(fig, .08, .075,
        "顺带一提,峰高也排得很靠前 —— 说明「按得重」和「按得久」这两个物理量都在起作用,"
        "和我们规则里的两条判据正好对应。", 10.5, C_MUTE)
    close(pdf, fig)


def page_learning(pdf, R):
    fig = page("规则解码不需要训练数据", "结果四")
    txt(fig, .08, .872,
        "机器学习方法都要先喂一批标好答案的数据才能开始工作,规则解码不需要 —— "
        "它从第一次敲击起就能用。下图横轴是喂进去的训练样本数。", 11)

    ax = fig.add_axes([.12, .545, .80, .265])
    d = {k: R[k] for k in ("y", "fold", "file_id", "rep_pos")}
    ns = [(R["fold"][D.split(d, k)[0]] != (k + 1) % D.N_FOLDS).sum()
          for k in range(D.N_FOLDS)]
    xs = R["train_sizes"] * float(np.mean(ns))
    ra = float(np.trace(R["cm_rule"])) / R["cm_rule"].sum()
    ax.axhline(ra, color=C_RED, ls="--", lw=2)
    ax.scatter([0], [ra], s=240, marker="*", color=C_RED, zorder=5,
               edgecolor="white", lw=.6)
    ax.text(30, ra - .011, "规则解码:0 个训练样本", fontproperties=ZH, fontsize=10,
            color=C_RED, va="top")
    for m in ["rf", "svm", "transformer", "bilstm", "cnn_raw"]:
        ax.plot(xs, R["lc_%s" % m], "-o", color=COLOR[m], lw=1.7, ms=4,
                label=CN[m].replace("\n", ""))
    ax_cn(ax, None, "喂进去的训练样本数", "测试准确率")
    ax.legend(prop=ZH, fontsize=8.5, frameon=False, loc="lower right")

    lc_b, lc_c = R["lc_bilstm"], R["lc_cnn_raw"]
    txt(fig, .08, .478, "图里说了什么", 13, C_INK, "bold")
    txt(fig, .08, .446,
        "红色虚线是规则解码,它在横轴 0 处就已经是 %.2f%%。BiLSTM 只喂 %d 个样本时是 %.1f%%,"
        "要喂到 %d 个左右才追上;完全不用物理量的 1D-CNN 即使喂满 %d 个,也只有 %.1f%%,"
        "始终没追上。" % (ra * 100, int(xs[0]), lc_b[0] * 100, int(xs[1]),
                      int(xs[-1]), lc_c[-1] * 100), 11)

    txt(fig, .08, .345, "为什么这一点在实际中重要", 13, C_INK, "bold")
    txt(fig, .08, .313,
        "换一块传感器、换一个人按,或者要增加数字和标点,机器学习方法都得重新采集、"
        "重新标注、重新训练。规则解码不用 —— 它读的是物理量,换了场景直接就能跑。", 11)

    txt(fig, .08, .225, "还有一个能力上的差别", 13, C_INK, "bold")
    txt(fig, .08, .193,
        "26 类分类模型只能对一个切好的片段给出一个字母。而规则解码可以直接读一整段连续消息 —— "
        "我们已经验证过 SOS、HELP、WATER、ESCAPE 这类词能整句读出。"
        "「用一片柔性传感器实时解码任意消息」,比「分成 26 个类」更有应用价值,"
        "也更适合作为论文的应用落点。", 11)
    close(pdf, fig)


def page_data(pdf, R):
    fig = page("数据情况", "数据")
    cnt = np.bincount(R["y"], minlength=26)
    ax = fig.add_axes([.10, .625, .83, .185])
    ax.bar(range(26), cnt, .68, color=[C_RED if c < 90 else "#2e86c1" for c in cnt],
           edgecolor="#2c3e50", lw=.4)
    ax.axhline(100, color=C_MUTE, ls="--", lw=1)
    ax.set_xticks(range(26)); ax.set_xticklabels(LETTERS, fontsize=8)
    for i, c in enumerate(cnt):
        if c < 90:
            ax.text(i, c + 4, str(c), ha="center", fontsize=8, color=C_RED,
                    fontweight="bold")
    ax_cn(ax, "每个字母可用的敲击次数(虚线 = 100)", None, "次数")

    txt(fig, .08, .570, "总体情况很好", 13, C_INK, "bold")
    txt(fig, .08, .538,
        "26 个字母共 %d 次敲击,36 份录制全部可用,没有任何一份需要丢弃。"
        "绝大多数字母都在 100 次左右。" % int(cnt.sum()), 11)

    txt(fig, .08, .465, "有两件小事想跟你确认", 13, C_RED, "bold")
    txt(fig, .08, .433,
        "① 字母 X 只有 79 次。从波形看你是一直在按的,应该是录制在 36 分钟时结束了"
        "(按当时的节奏,录满 100 次大约需要 45 分钟)。这不影响现在的结论,"
        "如果方便的话可以补录约 20 次。", 11)
    txt(fig, .08, .345,
        "② 有一份文件叫 D13+F58,里面顺序录了两个字母:前面约 1000 秒是 D,之后是 F。"
        "我们按这个分界拆开后得到 33 次 D 和 68 次 F,解码全部正确。"
        "只是文件名里的数字(13、58)和实际条数对不上,想问一下命名的含义。", 11)

    txt(fig, .08, .245, "另外修好了一个问题", 13, C_INK, "bold")
    txt(fig, .08, .213,
        "文件 K-30 早先被我们当成坏文件。重新检查发现:传感器是在第 544 秒才接触不良"
        "(电阻从 9.9 千欧跳到 23 兆欧),而 30 次 K 全部发生在那之前,一次都没丢。"
        "程序现在会自动识别并跳过这种故障时段,所以这 30 次已经全部加了回来。", 11)
    txt(fig, .08, .115,
        "这个修正挺重要:原来的程序遇到探头脱开时不会报错,只会安静地「看不到信号」,"
        "很容易被当成数据本身有问题。现在不会了。", 10.5, C_MUTE)
    close(pdf, fig)


def page_summary(pdf, R):
    fig = page("小结与可提供的材料", "结论")
    n = int(R["cm_rule"].sum())
    txt(fig, .08, .872, "四条结论", 13, C_INK, "bold")
    concl = [
        "① 三种原理完全不同的方法都达到 98.8%~100%,结论一致。这说明字母之间的可分性"
        "来自传感器与材料本身,不依赖某个特定模型。",
        "② 深度学习没有带来提升。Transformer 和 BiLSTM 的错误次数反而是规则解码的 3~7 倍。",
        "③ 真正起作用的是物理量,不是模型深度。不用物理量、直接把波形喂进 CNN,"
        "错误多了十几倍,而且近一半是 E 和 T 混淆。",
        "④ 随机森林独立地选出了按压时长作为最重要的测量值,与我们规则里用的量一致。",
    ]
    for i, s in enumerate(concl):
        txt(fig, .09, .838 - i * .072, s, 10.8, C_INK)

    txt(fig, .08, .545, "已经准备好的材料", 13, C_INK, "bold")
    items = [
        ("结果图,15 张",
         "混淆矩阵、方法对比、特征重要性、学习曲线、训练曲线、PR 与 ROC 曲线、"
         "t-SNE(二维与三维)。图内文字为英文,可直接投稿。"),
        ("Origin 数据,17 个 CSV",
         "每张图都配一份对应数据,可以在 Origin 里按自己的风格重画。"),
        ("波形数据",
         "26 个字母各 15 条重复,已对齐到统一时间轴,可直接画叠加曲线图。"),
        ("完整代码",
         "可直接用在新数据上,换传感器或换采样率都不需要改;遇到接触不良会自动跳过。"),
    ]
    yy = .510
    for k, v in items:
        txt(fig, .09, yy, k, 11, C_RED, "bold")
        txt(fig, .09, yy - .027, v, 10.5, C_INK)
        yy -= .082

    txt(fig, .08, .155, "如果要写进论文", 13, C_INK, "bold")
    txt(fig, .08, .123,
        "建议把这组对比写成「多方法交叉验证」,落点放在器件性能而不是算法优劣,"
        "同时保留 Transformer 与 CNN 的完整结果 —— 机器学习相关的章节和图表都是齐的。"
        "共 %d 次敲击的逐次结果也都在,需要哪种写法我可以再整理。" % n, 11)
    close(pdf, fig)


def main():
    R, S = load()
    with PdfPages(OUT) as pdf:
        page_cover(pdf, R, S)
        page_what(pdf)
        page_methods(pdf)
        page_results(pdf, R, S)
        page_et(pdf, R)
        page_feature(pdf, R)
        page_learning(pdf, R)
        page_data(pdf, R)
        page_summary(pdf, R)
    print("saved", OUT)


if __name__ == "__main__":
    main()
