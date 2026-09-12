#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py — 三种方法在完全相同的划分上对比。

  ① Rule-based decoding      规则解码(人显式写出"组内相对比较"),不需要训练数据
  ② Features + Random Forest 手工特征 + 随机森林(人给特征,机器学决策边界)
  ③ Transformer encoder      按压序列 + 自注意力(机器自己学"元素之间的关系")
  (附) Features + SVM / BiLSTM  作为 sanity check

协议:5 折交叉验证,折 = 每份录制内按时间顺序切的连续块,测试块边界 purge 1 个重复。
      对于需要训练的方法,取另一折做验证集(早停),其余 3 折训练。
      每个样本恰好被测试一次 → 三种方法都得到覆盖全部 2530 个样本的混淆矩阵。
输出:out/ml/results.npz + 逐折指标 CSV
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "ml")
SEED = 0
N_CLS = 26
torch.set_num_threads(4)          # 小张量下线程过多反而变慢
TRAIN_SIZES = [0.1, 0.325, 0.55, 0.775, 1.0]      # 与她上篇 learning_curve 一致
EPOCHS = 200
PATIENCE = 30


def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)


# --------------------------------------------------------------------------- #
# ① 规则解码 —— 无需训练,预测已在 dataset 里算好;这里只补一个置信度用于 PR 曲线
# --------------------------------------------------------------------------- #
def rule_scores(d, idx):
    """把规则解码的硬判决转成得分向量:预测类给置信度,其余均分。

    置信度 = 该字母内所有按压里"离判据阈值最近"的那次的相对余量(越大越可信)。
    """
    import morse_sensor as ms
    sc = np.zeros((len(idx), N_CLS), float)
    for r, i in enumerate(idx):
        seq, n = d["X_seq"][i], d["lens"][i]
        hr, dur, wr = seq[:n, 0], seq[:n, 1], seq[:n, 2]
        if n >= 2 and hr.max() / max(hr.min(), 1e-6) >= ms.HEIGHT_RATIO_MIN:
            v, thr = hr, ms._otsu(hr)[0]                       # 组内相对峰高
        elif n >= 2 and wr.max() / max(wr.min(), 1e-6) >= ms.HEIGHT_RATIO_MIN:
            v, thr = wr, ms._otsu(wr)[0]                       # 组内相对时长
        else:
            v, thr = dur, ms.DASH_WIDTH_S                      # 纯字母:绝对时长
        conf = float(np.clip(np.min(np.abs(v - thr)) / max(abs(thr), 1e-6), 0, 1))
        p = d["rule_pred"][i]
        if p < 0:
            sc[r] = 1.0 / N_CLS
        else:
            sc[r] = (1 - conf) / (N_CLS - 1)
            sc[r, p] = conf
    return sc


# --------------------------------------------------------------------------- #
# ③ Transformer encoder(按压序列)
# --------------------------------------------------------------------------- #
class PressTransformer(nn.Module):
    """每次按压是一个 token;self-attention 让模型自己学会"跟同组其它按压比较"。"""

    def __init__(self, d_in=4, d_model=64, nhead=4, nlayers=2, n_cls=N_CLS, p=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, D.MAX_TAPS, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           dropout=p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(p),
                                  nn.Linear(d_model, n_cls))

    def forward(self, x, lens):
        mask = torch.arange(D.MAX_TAPS, device=x.device)[None, :] >= lens[:, None]
        h = self.enc(self.proj(x) + self.pos, src_key_padding_mask=mask)
        keep = (~mask).float().unsqueeze(-1)
        return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1))   # masked mean-pool


class PressBiLSTM(nn.Module):
    """对照用:双向 LSTM(她方向里同样常见)。"""

    def __init__(self, d_in=4, hid=64, n_cls=N_CLS, p=0.1):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hid, n_cls))

    def forward(self, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=D.MAX_TAPS)
        mask = (torch.arange(D.MAX_TAPS, device=x.device)[None, :] < lens[:, None]).float()
        return self.head((out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True))


class RawCNN(nn.Module):
    """对照(消融):不用物理前端,直接把定长重采样的原始波形喂进 1D-CNN。

    架构照她上篇论文的 model_2:Conv 32/64/128 (k=5) + MaxPool → Dense 256 → Dropout 0.5。
    定长重采样会把"按压持续多久"这个绝对信息归一化掉,而 E(·) 与 T(—) 恰好只靠时长区分,
    所以这一路预期会在纯点/纯划字母上失手——这正是固定窗口 CNN 在本任务上的真实短板。
    """

    def __init__(self, n_in=D.RAW_LEN, n_cls=N_CLS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5), nn.ReLU(), nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, n_in)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(256, n_cls))

    def forward(self, x, lens=None):
        return self.head(self.conv(x.unsqueeze(1)))


def train_raw_model(Xtr, ytr, Xva, yva, Xte, epochs=EPOCHS, record_history=False):
    """训练原始波形 CNN,返回(测试集概率, 训练历史)。"""
    set_seed()
    Xtr_, Xva_, Xte_ = (torch.tensor(a) for a in (Xtr, Xva, Xte))
    ytr_, yva_ = torch.tensor(ytr), torch.tensor(yva)
    net = RawCNN()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    n, bs = len(ytr_), 64
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr_[b]), ytr_[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            lo_va = net(Xva_)
            acc_va = float((lo_va.argmax(1) == yva_).float().mean())
            if record_history:                    # 整训练集前向很贵,只在要画曲线时算
                l_va = float(lossf(lo_va, yva_))
                acc_tr = float((net(Xtr_).argmax(1) == ytr_).float().mean())
                hist.append(dict(epoch=ep + 1, train_loss=tot / n, train_acc=acc_tr,
                                 val_loss=l_va, val_acc=acc_va))
        if acc_va > best:
            best, bad = acc_va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(Xte_), 1).numpy()
    return prob, hist


def _norm_stats(X, lens):
    """按训练集统计 token 里"有量纲"的两列(时长、间隔)的均值/标准差。"""
    m = np.arange(D.MAX_TAPS)[None, :] < lens[:, None]
    mu = np.zeros(4, np.float32); sd = np.ones(4, np.float32)
    for c in (1, 3):
        v = X[:, :, c][m]
        mu[c], sd[c] = v.mean(), max(v.std(), 1e-6)
    return mu, sd


def train_seq_model(kind, Xtr, ltr, ytr, Xva, lva, yva, Xte, lte,
                    epochs=EPOCHS, record_history=False):
    """训练序列模型,返回(测试集概率, 训练历史)。用验证集早停。"""
    set_seed()
    mu, sd = _norm_stats(Xtr, ltr)
    t = lambda X: torch.tensor((X - mu) / sd)                       # noqa: E731
    Xtr_, Xva_, Xte_ = t(Xtr), t(Xva), t(Xte)
    ltr_, lva_, lte_ = (torch.tensor(a) for a in (ltr, lva, lte))
    ytr_, yva_ = torch.tensor(ytr), torch.tensor(yva)

    net = (PressTransformer() if kind == "transformer" else PressBiLSTM())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    n, bs = len(ytr_), 64
    best, best_state, bad, hist = -1.0, None, 0, []

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr_[b], ltr_[b]), ytr_[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            lo_va = net(Xva_, lva_)
            acc_va = float((lo_va.argmax(1) == yva_).float().mean())
            if record_history:                    # 整训练集前向很贵,只在要画曲线时算
                l_va = float(lossf(lo_va, yva_))
                acc_tr = float((net(Xtr_, ltr_).argmax(1) == ytr_).float().mean())
                hist.append(dict(epoch=ep + 1, train_loss=tot / n, train_acc=acc_tr,
                                 val_loss=l_va, val_acc=acc_va))
        if acc_va > best:
            best, bad = acc_va, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(Xte_, lte_), 1).numpy()
    return prob, hist


# --------------------------------------------------------------------------- #
# 指标(与她上篇同一套)
# --------------------------------------------------------------------------- #
def metrics(y, pred):
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro",
                                                    zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, pred, average="weighted",
                                                    zero_division=0)
    return dict(accuracy=accuracy_score(y, pred),
                macro_precision=mp, macro_recall=mr, macro_f1=mf,
                weighted_precision=wp, weighted_recall=wr, weighted_f1=wf,
                cohen_kappa=cohen_kappa_score(y, pred))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    npz = os.path.join(OUT, "dataset.npz")
    d = dict(np.load(npz, allow_pickle=True)) if os.path.exists(npz) else D.build(False)
    y = d["y"]; N = len(y)

    METHODS = ["rule", "rf", "transformer", "svm", "bilstm", "cnn_raw"]
    pred = {m: -np.ones(N, np.int64) for m in METHODS}
    prob = {m: np.zeros((N, N_CLS), np.float32) for m in METHODS}
    fold_rows, histories, importances = [], {}, []

    for k in range(D.N_FOLDS):
        va_fold = (k + 1) % D.N_FOLDS
        tr_all, te = D.split(d, k)
        va = tr_all[d["fold"][tr_all] == va_fold]
        tr = tr_all[d["fold"][tr_all] != va_fold]
        print(f"\n── 折 {k}: 训练 {len(tr)}  验证 {len(va)}  测试 {len(te)}")

        # ① 规则解码:无需训练
        pred["rule"][te] = d["rule_pred"][te]
        prob["rule"][te] = rule_scores(d, te)

        # ② 手工特征 + 随机森林 / SVM(训练集 = 训练折 + 验证折,它们不需要早停)
        fit_idx = np.concatenate([tr, va])
        rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
        rf.fit(d["X_feat"][fit_idx], y[fit_idx])
        pred["rf"][te] = rf.predict(d["X_feat"][te])
        prob["rf"][te] = rf.predict_proba(d["X_feat"][te])
        importances.append(rf.feature_importances_)

        # SVM 是附带项,不做 Platt 概率校准(26 类下极慢);得分用 decision_function 折算
        sv = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED))
        sv.fit(d["X_feat"][fit_idx], y[fit_idx])
        pred["svm"][te] = sv.predict(d["X_feat"][te])
        df = sv.decision_function(d["X_feat"][te])
        prob["svm"][te] = np.exp(df - df.max(1, keepdims=True)) / \
            np.exp(df - df.max(1, keepdims=True)).sum(1, keepdims=True)

        # ③ Transformer(+ BiLSTM 对照)
        for kind in ("transformer", "bilstm"):
            p, hist = train_seq_model(
                kind, d["X_seq"][tr], d["lens"][tr], y[tr],
                d["X_seq"][va], d["lens"][va], y[va],
                d["X_seq"][te], d["lens"][te], record_history=(k == 0))
            prob[kind][te] = p
            pred[kind][te] = p.argmax(1)
            if k == 0:
                histories[kind] = hist

        # (消融) 无物理前端:原始波形 + 1D-CNN(她上篇架构)
        p, hist = train_raw_model(d["X_raw"][tr], y[tr], d["X_raw"][va], y[va],
                                  d["X_raw"][te], record_history=(k == 0))
        prob["cnn_raw"][te] = p
        pred["cnn_raw"][te] = p.argmax(1)
        if k == 0:
            histories["cnn_raw"] = hist

        for m in METHODS:
            r = metrics(y[te], pred[m][te]); r.update(fold=k, method=m)
            fold_rows.append(r)
            print(f"   {m:12s} acc={r['accuracy']*100:6.2f}%  macroF1={r['macro_f1']:.4f}"
                  f"   [{time.time()-t0:.0f}s]")

    # ---- 汇总 ----
    print("\n" + "=" * 62)
    print(f"{'方法':<14}{'准确率':>9}{'macro-F1':>11}{'kappa':>9}{'错误数':>8}")
    summary = {}
    for m in METHODS:
        r = metrics(y, pred[m])
        r["n_errors"] = int((pred[m] != y).sum())
        summary[m] = r
        print(f"{m:<14}{r['accuracy']*100:8.2f}%{r['macro_f1']:11.4f}"
              f"{r['cohen_kappa']:9.4f}{r['n_errors']:8d}")

    cms = {m: confusion_matrix(y, pred[m], labels=range(N_CLS)) for m in METHODS}

    # ---- 学习曲线:准确率 vs 训练集大小(与她上篇同款) ----
    print("\n── 学习曲线(准确率 vs 训练集比例)")
    lc = {m: [] for m in ["rf", "transformer", "svm", "bilstm", "cnn_raw"]}
    rng = np.random.RandomState(SEED)
    for frac in TRAIN_SIZES:
        accs = {m: [] for m in lc}
        for k in range(D.N_FOLDS):
            va_fold = (k + 1) % D.N_FOLDS
            tr_all, te = D.split(d, k)
            va = tr_all[d["fold"][tr_all] == va_fold]
            tr = tr_all[d["fold"][tr_all] != va_fold]
            sub = tr if frac >= 1.0 else rng.choice(tr, max(N_CLS, int(len(tr) * frac)),
                                                    replace=False)
            fit_idx = np.concatenate([sub, va])
            m2 = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
            m2.fit(d["X_feat"][fit_idx], y[fit_idx])
            accs["rf"].append(accuracy_score(y[te], m2.predict(d["X_feat"][te])))
            s2 = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED))
            s2.fit(d["X_feat"][fit_idx], y[fit_idx])
            accs["svm"].append(accuracy_score(y[te], s2.predict(d["X_feat"][te])))
            for kind in ("transformer", "bilstm"):
                p, _ = train_seq_model(kind, d["X_seq"][sub], d["lens"][sub], y[sub],
                                       d["X_seq"][va], d["lens"][va], y[va],
                                       d["X_seq"][te], d["lens"][te], epochs=150)
                accs[kind].append(accuracy_score(y[te], p.argmax(1)))
            p, _ = train_raw_model(d["X_raw"][sub], y[sub], d["X_raw"][va], y[va],
                                   d["X_raw"][te], epochs=150)
            accs["cnn_raw"].append(accuracy_score(y[te], p.argmax(1)))
        for m in lc:
            lc[m].append(float(np.mean(accs[m])))
        print(f"   frac={frac:<6} " + "  ".join(f"{m}={np.mean(accs[m])*100:.2f}%" for m in lc)
              + f"   [{time.time()-t0:.0f}s]")

    # ---- 保存 ----
    import pandas as pd
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "fold_metrics.csv"), index=False)
    pd.DataFrame([dict(method=m, **summary[m]) for m in METHODS]).to_csv(
        os.path.join(OUT, "summary_metrics.csv"), index=False)
    np.savez_compressed(
        os.path.join(OUT, "results.npz"),
        y=y, files=d["files"], feat_names=d["feat_names"], fold=d["fold"],
        file_id=d["file_id"], rep_pos=d["rep_pos"], lens=d["lens"],
        X_feat=d["X_feat"], X_raw=d["X_raw"], X_seq=d["X_seq"],
        methods=np.array(METHODS),
        importance=np.mean(importances, 0),
        train_sizes=np.array(TRAIN_SIZES),
        **{f"pred_{m}": pred[m] for m in METHODS},
        **{f"prob_{m}": prob[m] for m in METHODS},
        **{f"cm_{m}": cms[m] for m in METHODS},
        **{f"lc_{m}": np.array(lc[m]) for m in lc},
    )
    with open(os.path.join(OUT, "histories.json"), "w") as fh:
        json.dump(histories, fh)
    print(f"\n完成,用时 {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
