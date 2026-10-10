#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_package.py — 把所有产出整理成一个可直接发给协作者的交付包。

    python make_package.py            # → out/交付包/  以及 out/交付包_<日期>.zip

会先校验数据是否为最新(混淆矩阵总数必须等于 results.npz 里的样本数),
避免把早先的过期结果打包出去。
"""
import datetime as dt
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ML = os.path.join(OUT, "ml")
PKG = os.path.join(OUT, "交付包")

# 论文用图:文件名 → 中文说明(文件名保持英文,投稿时不用改)
FIGURES = [
    ("model_comparison.png", "六种做法的总体对比(准确率 + 错误次数)"),
    ("confusion_matrix_rule.png", "混淆矩阵 — 规则解码"),
    ("confusion_matrix_rf.png", "混淆矩阵 — 手工特征 + 随机森林"),
    ("confusion_matrix_svm.png", "混淆矩阵 — 手工特征 + SVM"),
    ("confusion_matrix_transformer.png", "混淆矩阵 — Transformer"),
    ("confusion_matrix_bilstm.png", "混淆矩阵 — BiLSTM"),
    ("confusion_matrix_cnn_raw.png", "混淆矩阵 — 1D-CNN(原始波形,对照组)"),
    ("per_letter_accuracy.png", "逐字母准确率(三种主方法并排)"),
    ("feature_importance.png", "随机森林特征重要性"),
    ("learning_curve.png", "学习曲线:准确率随训练样本数的变化"),
    ("training_curves.png", "训练过程曲线(逐 epoch 的准确率与损失)"),
    ("precision_recall_curves.png", "PR 曲线(26 个字母 + micro 平均)"),
    ("roc_curves.png", "ROC 曲线(26 个字母 + micro 平均)"),
    ("tsne_2d.png", "t-SNE 二维:原始波形空间 vs 物理特征空间"),
    ("tsne_3d.png", "t-SNE 三维:同上"),
]
WORD_FIGURES = [
    ("cross_session_generalisation.png", "跨批次泛化:8 月训练 → 9 月测试(规则 100%,模型 49%–87%)"),
    ("short_message_accuracy.png", "短消息:一条消息 1–10 个词时的整条正确率"),
    ("gap_normalization.png", "间隔归一化:静默秒数 vs 静默 ÷ 按压时长"),
]
PANELS = [("fig_c_waveforms.png", "26 个字母的波形叠加图(仿参考论文 Fig.5c)"),
          ("fig_e_confusion.png", "26×26 混淆矩阵,红框标出识别错误(仿 Fig.5e)")]
REPORTS = [("单词测试报告.pdf", "★ 最新:真实单词识别 + 换批次后模型还能不能用"),
           ("三方法对比报告.pdf", "★ 三种方法对比,通俗版"),
           ("摩尔斯识别报告.pdf", "识别方法与逐字母波形"),
           ("合成消息测试报告.pdf", "连续消息(整句)解码的测试")]
# (源路径, 基准目录, 包内文件名, 说明)—— 包内文件名必须唯一,否则会互相覆盖
CODE = [("morse_sensor.py", HERE, "morse_sensor.py",
         "核心模块:把电阻曲线解码成字母,可直接用于新数据"),
        ("ml/dataset.py", HERE, "dataset.py", "把录制整理成统一数据集"),
        ("ml/benchmark.py", HERE, "benchmark.py", "三方法 5 折交叉验证"),
        ("ml/figures.py", HERE, "figures.py", "生成全部结果图"),
        ("word_test.py", HERE, "word_test.py", "真实单词测试(检测/切分/整词/短消息)"),
        ("ml/cross_batch.py", HERE, "cross_batch.py", "跨批次泛化实验"),
        ("ml/README.md", HERE, "实验协议说明.md", "可直接用于论文 Methods 章节"),
        ("ml/结果说明.md", HERE, "结果与解读.md", "各项结果的详细解读"),
        ("README.md", os.path.dirname(HERE), "项目总说明.md", "整个项目的说明")]


def check_fresh():
    """确认要打包的是最新结果,而不是早先的过期版本。"""
    R = np.load(os.path.join(ML, "results.npz"), allow_pickle=True)
    n_npz = int(len(R["y"]))
    n_cm = int(R["cm_rule"].sum())
    cm_csv = os.path.join(ML, "origin", "confusion_matrix_rule.csv")
    n_csv = int(pd.read_csv(cm_csv, index_col=0).values.sum())
    if not (n_npz == n_cm == n_csv):
        sys.exit("结果不一致,拒绝打包:results.npz=%d 混淆矩阵=%d CSV=%d\n"
                 "请先重新运行 benchmark.py 与 figures.py。" % (n_npz, n_cm, n_csv))
    return n_npz


def pad(s, width):
    """按等宽字体的显示宽度补空格(中文占 2 格),%-Ns 按字符数补会对不齐。"""
    w = sum(2 if ord(c) >= 0x2E80 else 1 for c in s)
    return s + " " * max(1, width - w)


def copy(src, dst_dir, name=None):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, name or os.path.basename(src)))


def main():
    n = check_fresh()
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    d_rep = os.path.join(PKG, "01_报告")
    d_fig = os.path.join(PKG, "02_论文用图")
    d_dat = os.path.join(PKG, "03_Origin数据")
    d_wav = os.path.join(d_dat, "波形数据_每字母")
    d_met = os.path.join(PKG, "04_指标表")
    d_code = os.path.join(PKG, "05_代码")

    for name, _ in REPORTS:
        copy(os.path.join(OUT, name), d_rep)
    for name, _ in FIGURES:
        copy(os.path.join(ML, "figures", name), d_fig)
    for name, _ in PANELS:
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            copy(p, d_fig)
    for name, _ in WORD_FIGURES:
        copy(os.path.join(OUT, "words", "figures", name), d_fig)

    n_csv = 0
    for f in sorted(os.listdir(os.path.join(ML, "origin"))):
        if f.endswith(".csv"):
            copy(os.path.join(ML, "origin", f), d_dat); n_csv += 1
    wav_src = os.path.join(OUT, "origin", "waveforms")
    n_wav = 0
    if os.path.isdir(wav_src):
        for f in sorted(os.listdir(wav_src)):
            if f.endswith(".csv"):
                copy(os.path.join(wav_src, f), d_wav); n_wav += 1

    n_wcsv = 0
    wo = os.path.join(OUT, "words", "origin")
    for f in sorted(os.listdir(wo)):
        if f.endswith(".csv"):
            copy(os.path.join(wo, f), os.path.join(d_dat, "单词测试")); n_wcsv += 1

    for f in ("summary_metrics.csv", "fold_metrics.csv", "cross_batch_summary.csv"):
        copy(os.path.join(ML, f), d_met)
    for rel, base, name, _ in CODE:
        src = os.path.join(base, rel)
        if os.path.exists(src):
            copy(src, d_code, name)

    # ---- 说明文件 ----
    S = pd.read_csv(os.path.join(ML, "summary_metrics.csv")).set_index("method")
    lines = [
        "摩尔斯电码识别 — 交付包",
        "生成日期:%s" % dt.date.today().isoformat(),
        "=" * 58, "",
        "【一句话】三种原理完全不同的方法在同一批数据上都达到 98.8%~100%,",
        "结论一致 —— 说明字母的可分性来自传感器与材料本身,不依赖某个特定模型。", "",
        "8 月单字母:%d 次敲击 · 26 个字母 · 36 份录制(全部可用,未丢弃任何一份)" % n,
        "验证方式:5 折交叉验证,折按录制的时间顺序切分(不是随机打乱),",
        "          测试块边界还清除了相邻重复,以免相邻敲击造成成绩虚高。", "",
        "总体成绩(错误次数 / 共 %d 次):" % n,
    ]
    cn = {"rule": "规则解码(不需训练数据)", "rf": "手工特征 + 随机森林",
          "svm": "手工特征 + SVM", "transformer": "Transformer",
          "bilstm": "BiLSTM", "cnn_raw": "1D-CNN(原始波形,对照组)"}
    for m in ["rule", "rf", "svm", "transformer", "bilstm", "cnn_raw"]:
        lines.append("    %s%6.2f%%    错 %d 次"
                     % (pad(cn[m], 30), S.loc[m, "accuracy"] * 100,
                        int(S.loc[m, "n_errors"])))
    W = pd.read_csv(os.path.join(OUT, "words", "per_word.csv"))
    XB = pd.read_csv(os.path.join(ML, "cross_batch_summary.csv"))
    z = XB[(XB.setting == "zero") & (XB.method != "rule")].acc_mean * 100
    lines += ["", "9 月真实单词测试(20 个词 × 100 遍):",
              "    整词识别 %.2f%%,字错率 %.2f%%(切分方法是看过这批数据后改的,属验证集成绩)"
              % (W.consensus_exact.sum() / W.n_reps.sum() * 100,
                 W.consensus_edit.sum() / W.consensus_chars.sum() * 100),
              "    点划判别:给定字母边界,5583 个字母全部正确(这部分 8 月已定死,属盲测)",
              "    换批次:用 8 月数据训练的模型在 9 月数据上只有 %.0f%%–%.0f%%,规则解码 100%%"
              % (z.min(), z.max())]
    lines += ["", "-" * 58, "", "【目录】", ""]
    lines += ["01_报告/   中文 PDF,给人读的"]
    for name, desc in REPORTS:
        lines.append("    %s%s" % (pad(name, 26), desc))
    lines += ["", "02_论文用图/   共 %d 张,图内文字为英文,可直接投稿"
              % (len(FIGURES) + len(PANELS) + len(WORD_FIGURES))]
    for name, desc in WORD_FIGURES + FIGURES + PANELS:
        lines.append("    %s%s" % (pad(name, 36), desc))
    lines += ["", "03_Origin数据/   每张图对应一份 CSV,可在 Origin 里按自己的风格重画",
              "    %d 个图表数据 CSV(文件名与上面的图一一对应)" % n_csv,
              "    单词测试/   %d 个 CSV(间隔分布、短消息、跨批次、逐词结果、合成对照)" % n_wcsv,
              "    波形数据_每字母/   %d 个字母,每个含 15 条重复,已对齐到统一时间轴" % n_wav,
              "        → Origin:导入后选 time_s 作 X、所有 rep_* 作 Y,画 Line 图",
              "    confusion_matrix_*.csv  行 = 真实字母,列 = 预测字母",
              "        → Origin:导入后设为 Matrix,画 Heatmap/Image",
              "", "04_指标表/",
              "    summary_metrics.csv   六种做法的汇总指标",
              "    fold_metrics.csv      每一折的指标(共 5 折 × 6 种)",
              "    cross_batch_summary.csv  跨批次泛化(不给新数据 / 给 16 个词)",
              "", "05_代码/   可直接用在新数据上"]
    for _rel, _b, name, desc in CODE:
        lines.append("    %s%s" % (pad(name, 26), desc))
    lines += [
        "", "-" * 58, "",
        "【两件想跟你确认的事】", "",
        "1) 字母 X 只有 79 次。从波形看你是一直在按的,应该是录制在 36 分钟时",
        "   结束了(按当时的节奏,录满 100 次约需 45 分钟)。不影响任何结论,",
        "   不需要补录,只是告知一下。", "",
        "2) 文件 D13+F58 里顺序录了两个字母:前约 1000 秒是 D,之后是 F。",
        "   我们拆开后得到 33 次 D 和 68 次 F,解码全部正确;只是文件名里的",
        "   数字(13、58)和实际条数对不上,想问一下命名的含义。", "",
        "另:文件 K-30 早先被误判为坏文件。实际是传感器在第 544 秒才接触不良,",
        "30 次 K 全部在那之前,一次没丢,现已全部纳入。",
    ]
    with open(os.path.join(PKG, "00_先读我.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- 打包 ----
    zip_path = os.path.join(OUT, "摩尔斯识别_交付包_%s.zip"
                            % dt.date.today().strftime("%Y%m%d"))
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(PKG):
            for f in sorted(files):
                if f == ".DS_Store":
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.join("摩尔斯识别_交付包",
                                        os.path.relpath(p, PKG)))
                n_files += 1
    print("交付包 →", PKG)
    print("压缩包 →", zip_path,
          "(%d 个文件, %.1f MB)" % (n_files, os.path.getsize(zip_path) / 1e6))


if __name__ == "__main__":
    main()
