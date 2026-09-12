#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figures.py — 按柔性/可穿戴传感领域常见图式输出三方法对比的全套结果图。

图式对齐她上一篇论文(ACS AMI 2026)的图谱:
  混淆矩阵 seaborn Blues 热图、Model comparison 条形图、feature importance、
  learning curve(准确率 vs 训练集大小)、训练曲线(epoch)、t-SNE(原始 vs 特征)、
  PR 曲线、ROC 曲线。

图内文字用英文(投稿用);每张图同时导出同名 CSV,她可以自己在 Origin 里重画。
输出:out/ml/figures/*.png  +  out/ml/origin/*.csv
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
import seaborn as sns                     # noqa: E402
from sklearn.manifold import TSNE         # noqa: E402
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_curve)   # noqa: E402
from sklearn.preprocessing import label_binarize, StandardScaler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "ml")
FIG = os.path.join(OUT, "figures")
ORI = os.path.join(OUT, "origin")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
N_CLS = 26

# 主图三方法 + 消融对照
MAIN = ["rule", "rf", "transformer"]
ALL = ["rule", "rf", "svm", "transformer", "bilstm", "cnn_raw"]
NAME = {
    "rule": "Rule-based\n(no training)",
    "rf": "Features\n+ Random Forest",
    "svm": "Features\n+ SVM",
    "transformer": "Transformer\n(press sequence)",
    "bilstm": "BiLSTM\n(press sequence)",
    "cnn_raw": "1D-CNN\n(raw waveform)",
}
FLAT = {k: v.replace("\n", " ") for k, v in NAME.items()}
SHORT = {"rule": "Rule-based", "rf": "RF", "svm": "SVM", "transformer": "Transformer",
         "bilstm": "BiLSTM", "cnn_raw": "1D-CNN\n(raw)"}
COLOR = {"rule": "#c0392b", "rf": "#2e86c1", "svm": "#5dade2",
         "transformer": "#1a5276", "bilstm": "#7fb3d5", "cnn_raw": "#95a5a6"}
CLS_COLORS = plt.cm.nipy_spectral(np.linspace(0.03, 0.97, N_CLS))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", os.path.relpath(p, OUT))


def csv(df, name):
    os.makedirs(ORI, exist_ok=True)
    df.to_csv(os.path.join(ORI, name), index=False)


# --------------------------------------------------------------------------- #
def fig_confusion(R):
    """混淆矩阵热图(seaborn Blues,和她上篇同款样式)。"""
    for m in ALL:
        cm = R[f"cm_{m}"]
        acc = np.trace(cm) / cm.sum() * 100
        fig, ax = plt.subplots(figsize=(10.5, 8.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                    xticklabels=LETTERS, yticklabels=LETTERS, ax=ax,
                    annot_kws={"size": 5.2}, cbar_kws={"shrink": .82},
                    linewidths=.3, linecolor="#eef3f8")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {FLAT[m]}   (Accuracy: {acc:.2f}%)", fontsize=13)
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        save(fig, f"confusion_matrix_{m}.png")
        df = pd.DataFrame(cm, index=LETTERS, columns=LETTERS)
        df.index.name = "True\\Pred"
        df.to_csv(os.path.join(ORI, f"confusion_matrix_{m}.csv"))


def fig_model_comparison(R):
    """Model comparison:左=四个指标(全尺度,不截断纵轴);右=错误样本数(差异在这里才看得清)。"""
    S = pd.read_csv(os.path.join(OUT, "summary_metrics.csv")).set_index("method")
    mets = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labs = ["Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"]
    x = np.arange(len(ALL)); w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    ax = axes[0]
    for i, (mt, lb) in enumerate(zip(mets, labs)):
        v = [S.loc[m, mt] for m in ALL]
        b = ax.bar(x + (i - 1.5) * w, v, w, label=lb,
                   color=plt.cm.Blues(0.35 + 0.18 * i), edgecolor="#2c3e50", lw=.5)
        ax.bar_label(b, fmt="%.4f", fontsize=6, padding=2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in ALL], fontsize=8.5)
    ax.set_ylabel("Score", fontsize=12); ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("Model Performance Comparison\n(5-fold cross-validation, n = 2530)",
                 fontsize=12)
    ax.legend(ncol=2, fontsize=8.5, frameon=False, loc="lower center")
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)

    ax = axes[1]
    err = [int(S.loc[m, "n_errors"]) for m in ALL]
    b = ax.bar(x, err, 0.62, color=[COLOR[m] for m in ALL], edgecolor="#2c3e50", lw=.5)
    ax.bar_label(b, fmt="%d", fontsize=10, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in ALL], fontsize=9)
    ax.set_ylabel("Misclassified samples (of 2530)", fontsize=11)
    ax.set_ylim(0, max(err) * 1.28)
    ax.set_title("Number of Errors\n(same axis for all methods)", fontsize=12)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    save(fig, "model_comparison.png")
    csv(S.loc[ALL, mets + ["cohen_kappa", "n_errors"]].reset_index()
        .assign(method=[FLAT[m] for m in ALL]), "model_comparison.csv")


def fig_feature_importance(R):
    """随机森林特征重要性(横向条形,她上篇有同名图)。"""
    imp, names = R["importance"], list(R["feat_names"])
    o = np.argsort(imp)[::-1][:16][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 6))
    ax.barh(range(len(o)), imp[o], color=plt.cm.Blues(np.linspace(.45, .9, len(o))),
            edgecolor="#2c3e50", lw=.5)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([names[i] for i in o], fontsize=9)
    ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=11)
    ax.set_title("Feature Importance — Random Forest", fontsize=13)
    for i, v in enumerate(imp[o]):
        ax.text(v + .002, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    save(fig, "feature_importance.png")
    csv(pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False), "feature_importance.csv")


def _mean_train_size(R):
    """各折实际训练样本数(扣掉验证折与 purge)的平均值。"""
    import dataset as D
    d = {k: R[k] for k in ("y", "fold", "file_id", "rep_pos")}
    ns = []
    for k in range(D.N_FOLDS):
        tr_all, _ = D.split(d, k)
        ns.append((R["fold"][tr_all] != (k + 1) % D.N_FOLDS).sum())
    return float(np.mean(ns))


def fig_learning_curve(R):
    """准确率 vs 训练集大小 —— 规则法是 0 训练数据的一条水平线。"""
    fr = R["train_sizes"]
    n_tr = _mean_train_size(R)                      # 各折训练样本数的平均
    xs = fr * n_tr
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    rule_acc = float(np.trace(R["cm_rule"])) / R["cm_rule"].sum()
    ax.axhline(rule_acc, color=COLOR["rule"], ls="--", lw=2,
               label=f"{FLAT['rule']} — {rule_acc*100:.2f}%")
    ax.scatter([0], [rule_acc], s=220, marker="*", color=COLOR["rule"], zorder=5,
               edgecolor="white", lw=.6)
    ax.text(40, rule_acc - .011, "0 training samples needed", fontsize=9.5,
            color=COLOR["rule"], va="top")
    rows = [{"train_samples": 0, "method": FLAT["rule"], "accuracy": rule_acc}]
    for m in ["rf", "svm", "transformer", "bilstm", "cnn_raw"]:
        v = R[f"lc_{m}"]
        ax.plot(xs, v, "-o", color=COLOR[m], lw=1.8, ms=5, label=FLAT[m])
        rows += [{"train_samples": int(s), "method": FLAT[m], "accuracy": float(a)}
                 for s, a in zip(xs, v)]
    ax.set_xlabel("Number of training samples", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Learning Curve — accuracy vs. training set size", fontsize=13)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.grid(alpha=.25); ax.set_axisbelow(True)
    save(fig, "learning_curve.png")
    csv(pd.DataFrame(rows).pivot(index="train_samples", columns="method",
                                 values="accuracy").reset_index(), "learning_curve.csv")


def fig_training_curves():
    """训练过程曲线(epoch):左准确率、右损失 —— 她方向最常见的双面板。"""
    with open(os.path.join(OUT, "histories.json")) as fh:
        H = json.load(fh)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    rows = []
    for m, h in H.items():
        if not h:
            continue
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_acc"] for r in h], color=COLOR[m], lw=1.6,
                     label=f"{FLAT[m]} — train")
        axes[0].plot(ep, [r["val_acc"] for r in h], color=COLOR[m], lw=1.4, ls="--",
                     label=f"{FLAT[m]} — val")
        axes[1].plot(ep, [r["train_loss"] for r in h], color=COLOR[m], lw=1.6)
        axes[1].plot(ep, [r["val_loss"] for r in h], color=COLOR[m], lw=1.4, ls="--")
        rows += [dict(method=FLAT[m], **r) for r in h]
    axes[0].set_xlabel("Epoch", fontsize=11); axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Training / Validation Accuracy", fontsize=12)
    axes[0].legend(fontsize=7.2, frameon=False, loc="lower right")
    axes[1].set_xlabel("Epoch", fontsize=11); axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Training / Validation Loss  (solid = train, dashed = val)",
                      fontsize=12)
    axes[1].set_yscale("log")
    for a in axes:
        a.grid(alpha=.25); a.set_axisbelow(True)
    save(fig, "training_curves.png")
    csv(pd.DataFrame(rows), "training_curves.csv")


def _mark_et(ax, Z, y):
    """在 t-SNE 图上圈出 E(·) 与 T(—) —— 二者只靠"按压时长"区分。

    定长重采样会把绝对时长归一化掉,所以在原始波形空间里这一对会挤成一团;
    在物理特征空间里则完全分开。图上给出"簇心距 / 簇内散布"这个无量纲分离度。
    """
    iE, iT = LETTERS.index("E"), LETTERS.index("T")
    A, B = Z[y == iE], Z[y == iT]
    if not len(A) or not len(B):
        return
    cA, cB = A.mean(0), B.mean(0)
    sep = np.linalg.norm(cA - cB) / max(0.5 * (A.std(0).mean() + B.std(0).mean()), 1e-9)
    for P, c in ((A, cA), (B, cB)):                    # E、T 各圈一个
        r = np.percentile(np.linalg.norm(P - c, axis=1), 92) * 1.35
        ax.add_patch(plt.Circle(c, r, fill=False, ls="--", lw=1.6, ec="#c0392b", zorder=4))
    ax.annotate("", xy=cA, xytext=cB, zorder=4,
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4,
                                shrinkA=6, shrinkB=6))
    ax.text(0.022, 0.978, f"E (·) vs T (—)\ncluster separation = {sep:.2f}",
            transform=ax.transAxes, ha="left", va="top", zorder=6,
            fontsize=10, color="#c0392b", fontweight="bold",
            bbox=dict(fc="white", ec="#c0392b", lw=.9, alpha=.92, pad=3.2))


def fig_tsne(R):
    """t-SNE:原始波形空间 vs 手工特征空间(对应她上篇 raw / processed 两张)。"""
    y = R["y"]
    sets = {
        "raw": ("Raw waveform space (128-point resampled)", R["X_raw"]),
        "feature": ("Physical feature space (21-D)",
                    StandardScaler().fit_transform(R["X_feat"])),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, (key, (title, X)) in zip(axes, sets.items()):
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
                 learning_rate="auto").fit_transform(X)
        for i in range(N_CLS):
            m = y == i
            ax.scatter(Z[m, 0], Z[m, 1], s=7, color=CLS_COLORS[i], alpha=.75, lw=0)
            ax.text(Z[m, 0].mean(), Z[m, 1].mean(), LETTERS[i], fontsize=9,
                    fontweight="bold", ha="center", va="center",
                    color="black", path_effects=None)
        _mark_et(ax, Z, y)                    # 高亮 E / T:固定窗口下这一对会重叠
        ax.set_title(f"t-SNE — {title}", fontsize=12)
        ax.set_xlabel("t-SNE 1", fontsize=11); ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        csv(pd.DataFrame({"tsne_1": Z[:, 0], "tsne_2": Z[:, 1],
                          "label": [LETTERS[i] for i in y]}), f"tsne_2d_{key}.csv")
    save(fig, "tsne_2d.png")

    # 3D 版(她上篇也有)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(13.5, 6.2))
    for j, (key, (title, X)) in enumerate(sets.items(), 1):
        Z = TSNE(n_components=3, perplexity=30, init="pca", random_state=0,
                 learning_rate="auto").fit_transform(X)
        ax = fig.add_subplot(1, 2, j, projection="3d")
        for i in range(N_CLS):
            m = y == i
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=6, color=CLS_COLORS[i], alpha=.7, lw=0)
        ax.set_title(f"t-SNE 3D — {title}", fontsize=11)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_zlabel("t-SNE 3")
        csv(pd.DataFrame({"tsne_1": Z[:, 0], "tsne_2": Z[:, 1], "tsne_3": Z[:, 2],
                          "label": [LETTERS[i] for i in y]}), f"tsne_3d_{key}.csv")
    save(fig, "tsne_3d.png")


def fig_pr_roc(R):
    """PR 曲线与 ROC 曲线(26 类细线 + micro-average 粗线)。"""
    y = R["y"]
    Y = label_binarize(y, classes=range(N_CLS))
    for kind in ("pr", "roc"):
        fig, axes = plt.subplots(1, len(MAIN), figsize=(5 * len(MAIN), 4.6),
                                 sharey=True)
        rows = []
        for ax, m in zip(np.atleast_1d(axes), MAIN):
            P = R[f"prob_{m}"]
            for i in range(N_CLS):
                if Y[:, i].sum() == 0:
                    continue
                if kind == "pr":
                    pr, rc, _ = precision_recall_curve(Y[:, i], P[:, i])
                    ax.plot(rc, pr, lw=.8, color=CLS_COLORS[i], alpha=.6)
                else:
                    fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
                    ax.plot(fpr, tpr, lw=.8, color=CLS_COLORS[i], alpha=.6)
            if kind == "pr":
                pr, rc, _ = precision_recall_curve(Y.ravel(), P.ravel())
                ap = average_precision_score(Y, P, average="micro")
                ax.plot(rc, pr, lw=2.4, color="#c0392b",
                        label=f"micro-average (AP = {ap:.4f})")
                ax.set_xlabel("Recall", fontsize=11)
                rows.append(dict(method=FLAT[m], metric="micro_AP", value=float(ap)))
            else:
                fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
                a = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2.4, color="#c0392b",
                        label=f"micro-average (AUC = {a:.4f})")
                ax.plot([0, 1], [0, 1], "k--", lw=.7)
                ax.set_xlabel("False Positive Rate", fontsize=11)
                rows.append(dict(method=FLAT[m], metric="micro_AUC", value=float(a)))
            ax.set_title(FLAT[m], fontsize=11)
            if m == "rule":      # 规则法没有概率输出,得分由判据余量折算,需标注清楚
                ax.text(0.5, 0.06, "score = margin to decision threshold\n(not a"
                        " calibrated probability)", transform=ax.transAxes,
                        ha="center", fontsize=7.6, color="#555", style="italic")
            ax.legend(fontsize=8, frameon=False, loc="lower left")
            ax.grid(alpha=.22); ax.set_axisbelow(True)
        np.atleast_1d(axes)[0].set_ylabel("Precision" if kind == "pr"
                                          else "True Positive Rate", fontsize=11)
        fig.suptitle("Precision-Recall Curves (26 letters)" if kind == "pr"
                     else "ROC Curves (26 letters)", fontsize=13)
        save(fig, f"{'precision_recall' if kind=='pr' else 'roc'}_curves.png")
        csv(pd.DataFrame(rows), f"{'pr' if kind=='pr' else 'roc'}_summary.csv")


def fig_per_letter(R):
    """逐字母准确率对比(三方法并排),看错误集中在哪些字母。"""
    y = R["y"]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x = np.arange(N_CLS); w = 0.26
    rows = {}
    for i, m in enumerate(MAIN):
        cm = R[f"cm_{m}"]
        acc = np.diag(cm) / cm.sum(1).clip(min=1) * 100
        ax.bar(x + (i - 1) * w, acc, w, label=FLAT[m], color=COLOR[m],
               edgecolor="#2c3e50", lw=.4)
        rows[FLAT[m]] = acc
    ax.set_xticks(x); ax.set_xticklabels(LETTERS, fontsize=9)
    ax.set_ylim(80, 101.5); ax.set_ylabel("Per-letter accuracy (%)", fontsize=11)
    ax.set_xlabel("Letter", fontsize=11)
    ax.set_title("Per-letter Accuracy", fontsize=13)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    save(fig, "per_letter_accuracy.png")
    csv(pd.DataFrame({"letter": LETTERS, "n_samples": np.bincount(y, minlength=N_CLS),
                      **{k: np.round(v, 2) for k, v in rows.items()}}),
        "per_letter_accuracy.csv")


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(ORI, exist_ok=True)
    R = dict(np.load(os.path.join(OUT, "results.npz"), allow_pickle=True))
    print("出图 →", os.path.relpath(FIG))
    fig_confusion(R)
    fig_model_comparison(R)
    fig_feature_importance(R)
    fig_learning_curve(R)
    fig_training_curves()
    fig_per_letter(R)
    fig_pr_roc(R)
    fig_tsne(R)
    print("Origin CSV →", os.path.relpath(ORI))


if __name__ == "__main__":
    main()
