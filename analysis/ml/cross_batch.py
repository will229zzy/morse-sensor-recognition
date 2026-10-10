#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_batch.py — 跨批次泛化:8 月单字母训练 → 9 月单词里的字母测试。

两批录制的按法不同(9 月"划"的半高宽约是 8 月的 1.3 倍,且划后停顿随之变长),
正好检验"换了节奏,不重新训练还能不能用"。

实验 ④  不给新数据:只用 8 月 2661 个字母训练,测 9 月全部 5583 个字母。
         每种模型训练 5 次(每次用 8 月的不同一折做早停验证),报均值 ± 标准差。
         超参数全部沿用 benchmark.py,没有在 9 月数据上调过任何东西。
实验 ④b 给少量新数据:9 月按"单词"分 5 折(每折 4 个词),
         训练 = 8 月全部 + 其余 16 个词,测试 = 留出的 4 个词。
         必须按词切分:同一个词的 100 遍来自同一次录制,按遍数切等于提前看到答案。

规则法不训练:沿用 8 月就定死的点划判别,字母边界取标准答案(与模型一致)。
输出:out/ml/cross_batch.npz + cross_batch_summary.csv
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B          # noqa: E402  复用同一套模型与训练函数
import dataset as D            # noqa: E402

OUT = B.OUT
MODELS = ["rf", "svm", "transformer", "bilstm", "cnn_raw"]
N_RUNS = 5
WORD_FOLDS = 5


def _load():
    a = dict(np.load(os.path.join(OUT, "dataset.npz"), allow_pickle=True))
    w = dict(np.load(os.path.join(OUT, "words_dataset.npz"), allow_pickle=True))
    return a, w


def _fit_predict(model, Xtr, ytr, Xva, yva, Xte, seed):
    """在 (Xtr, ytr) 上训练、(Xva, yva) 上早停,返回对 Xte 的预测。X 是 dict(seq/lens/feat/raw)。"""
    import torch
    torch.set_num_threads(B.THREADS_PER_WORKER)
    B.set_seed(seed)
    if model in ("rf", "svm"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        Xf = np.concatenate([Xtr["feat"], Xva["feat"]]); yf = np.concatenate([ytr, yva])
        clf = (RandomForestClassifier(n_estimators=500, random_state=seed,
                                      n_jobs=B.THREADS_PER_WORKER) if model == "rf"
               else make_pipeline(StandardScaler(), SVC(C=10, gamma="scale",
                                                        random_state=seed)))
        clf.fit(Xf, yf)
        return clf.predict(Xte["feat"])
    if model == "cnn_raw":
        p, _ = B.train_raw_model(Xtr["raw"], ytr, Xva["raw"], yva, Xte["raw"])
    else:
        p, _ = B.train_seq_model(model, Xtr["seq"], Xtr["lens"], ytr,
                                 Xva["seq"], Xva["lens"], yva, Xte["seq"], Xte["lens"])
    return p.argmax(1)


def _pack(d, idx):
    return dict(seq=d["X_seq"][idx], lens=d["lens"][idx],
                feat=d["X_feat"][idx], raw=d["X_raw"][idx])


def job_zero_shot(model, run):
    """④:只用 8 月训练。第 run 折做早停验证,其余训练。"""
    a, w = _load()
    va = np.where(a["fold"] == run)[0]; tr = np.where(a["fold"] != run)[0]
    pred = _fit_predict(model, _pack(a, tr), a["y"][tr], _pack(a, va), a["y"][va],
                        _pack(w, np.arange(len(w["y"]))), seed=B.SEED + run)
    return "zero", model, run, np.arange(len(w["y"])), pred


def job_few_shot(model, k):
    """④b:8 月全部 + 9 月 16 个词训练,留出 4 个词测试(按词切分)。"""
    a, w = _load()
    wid = w["word_id"]
    test = np.where(wid % WORD_FOLDS == k)[0]
    rest = np.where(wid % WORD_FOLDS != k)[0]
    rng = np.random.RandomState(B.SEED + k)
    # 早停验证:8 月第 0 折 + 训练用单词里随机 15% 的字母(都不属于留出的测试词)
    va_w = rng.choice(rest, int(0.15 * len(rest)), replace=False)
    tr_w = np.setdiff1d(rest, va_w)
    va_a = np.where(a["fold"] == 0)[0]; tr_a = np.where(a["fold"] != 0)[0]
    cat = lambda d1, i1, d2, i2: {k_: np.concatenate([_pack(d1, i1)[k_], _pack(d2, i2)[k_]])
                                  for k_ in ("seq", "lens", "feat", "raw")}
    Xtr = cat(a, tr_a, w, tr_w); ytr = np.concatenate([a["y"][tr_a], w["y"][tr_w]])
    Xva = cat(a, va_a, w, va_w); yva = np.concatenate([a["y"][va_a], w["y"][va_w]])
    pred = _fit_predict(model, Xtr, ytr, Xva, yva, _pack(w, test), seed=B.SEED + k)
    return "few", model, k, test, pred


def main():
    t0 = time.time()
    a, w = _load()
    yw = w["y"]; N = len(yw)
    print(f"训练(8 月单字母) {len(a['y'])} 个  →  测试(9 月单词里的字母) {N} 个,"
          f"{len(np.unique(yw))} 种字母,{len(w['words'])} 个词")
    jobs = ([delayed(job_zero_shot)(m, r) for m in ["cnn_raw", "transformer", "bilstm", "rf", "svm"]
             for r in range(N_RUNS)]
            + [delayed(job_few_shot)(m, k) for m in ["cnn_raw", "transformer", "bilstm", "rf", "svm"]
               for k in range(WORD_FOLDS)])
    out = Parallel(n_jobs=B.N_WORKERS, backend="loky", verbose=5)(jobs)

    zero = {m: np.full((N_RUNS, N), -1, np.int64) for m in MODELS}
    few = {m: np.full(N, -1, np.int64) for m in MODELS}
    for kind, m, r, idx, pred in out:
        if kind == "zero":
            zero[m][r, idx] = pred
        else:
            few[m][idx] = pred

    rows = [dict(method="rule", setting="zero", acc_mean=(w["rule_pred"] == yw).mean(),
                 acc_std=0.0, n_err=int((w["rule_pred"] != yw).sum())),
            dict(method="rule", setting="few", acc_mean=(w["rule_pred"] == yw).mean(),
                 acc_std=0.0, n_err=int((w["rule_pred"] != yw).sum()))]
    print(f"\n{'方法':12s}{'④ 不给新数据(5 次)':>24s}{'④b 给 16 个词':>18s}")
    print(f"{'rule':12s}{(w['rule_pred']==yw).mean()*100:>17.2f}%          "
          f"{(w['rule_pred']==yw).mean()*100:>10.2f}%")
    for m in MODELS:
        accs = (zero[m] == yw[None, :]).mean(1)
        fa = (few[m] == yw).mean()
        rows.append(dict(method=m, setting="zero", acc_mean=accs.mean(), acc_std=accs.std(),
                         n_err=float((zero[m] != yw[None, :]).sum(1).mean())))
        rows.append(dict(method=m, setting="few", acc_mean=fa, acc_std=0.0,
                         n_err=int((few[m] != yw).sum())))
        print(f"{m:12s}{accs.mean()*100:>12.2f}% ± {accs.std()*100:4.2f}"
              f"{fa*100:>16.2f}%")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_batch_summary.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "cross_batch.npz"), y=yw, word=w["word"],
                        word_id=w["word_id"], pos=w["pos"], rule=w["rule_pred"],
                        **{f"zero_{m}": zero[m] for m in MODELS},
                        **{f"few_{m}": few[m] for m in MODELS})
    print(f"\n完成,用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
