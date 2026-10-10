#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
word_test.py — 规则解码器在 9 月真实单词录制上的测试(每份 = 同一个词连续按约 100 遍)。

  ① 按压检测   检测到的按压数 vs 应有数(遍数 × 该词点划总数)
  ② 切分       解码器切出的字母边界、单词边界,对照标准答案的查准率 / 查全率
  ③ 端到端     整份录制直接解码 → 整词正确率、字错率(CER)
  ④ 短消息     从录制里截出 1/2/3/5/10 个词的原始信号,单独解码(模拟实际只按一次 SOS)

②③④ 都对比两种切分:"只看静默秒数"(原思路)与 morse_sensor 现行的"两种量法取共识"。
注意:现行切分是看过这批数据的失败之后改的,这里的成绩是验证集成绩;"只看静默秒数"
那一列才相当于盲测。点划判别部分在 8 月就已定死,未因这批数据改动。

    python word_test.py         # → out/words/*.csv
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ml"))
import morse_sensor as ms      # noqa: E402
import dataset as D            # noqa: E402

OUT = os.path.join(HERE, "out", "words")
SHORT_LENS = [1, 2, 3, 5, 10]
PAD_S = 3.0                    # 截短消息时首尾各留的静息时长
REPS_PER_WORD = 100            # 录制协议:每个词按 100 遍(20 份里 16 份的按压数恰为 100×点划数)


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def segment_silence_only(taps):
    """对照:只看静默秒数(原思路),同样自适应分三层。"""
    if not taps:
        return []
    g = ms.silence_gaps(taps)
    t1, t2 = ms.gap_levels(g)
    words, cw, cl = [], [], [taps[0]]
    for i, gg in enumerate(g):
        if gg > t1 or gg > t2:
            cw.append(cl); cl = []
        if gg > t2:
            words.append(cw); cw = []
        cl.append(taps[i + 1])
    cw.append(cl); words.append(cw)
    return words


def segment_relative_only(taps):
    """对照:只看相对间隔(静默 ÷ 两侧按压平均时长),同样自适应分三层。"""
    if not taps:
        return []
    g = ms.relative_gaps(taps)
    t1, t2 = ms.gap_levels(g)
    words, cw, cl = [], [], [taps[0]]
    for i, gg in enumerate(g):
        if gg > t1 or gg > t2:
            cw.append(cl); cl = []
        if gg > t2:
            words.append(cw); cw = []
        cl.append(taps[i + 1])
    cw.append(cl); words.append(cw)
    return words


SEGMENTERS = {"silence": segment_silence_only, "relative": segment_relative_only,
              "consensus": ms.segment}
SEG_CN = {"silence": "只看静默秒数", "relative": "只看相对间隔", "consensus": "两者共识(现行)"}

# 合成消息:用 8 月单字母波形按"中心间隔固定"的节奏拼出来,检验切分在另一种节奏下是否退化
SYNTH = {"cad_4": "4 秒节奏", "cad_5": "5 秒节奏", "cad_7": "7 秒节奏", "words": "英文单词"}


def synth_job(seg_name, test):
    import synth_test as st
    ms.segment = SEGMENTERS[seg_name]          # loky 子进程里各自打补丁
    if test == "words":
        r = st.run_words()
    else:
        g = float(test.split("_")[1])
        r = st.run_cadence(elem_gap=g, letter_gap=g * 2.4)
    return dict(segmenter=seg_name, test=test, exact=r["exact"], cer=r["cer"])


def decode_with(seg_fn, sec, R):
    """用指定的切分函数跑完整解码,返回文本(单词以空格分隔)。"""
    orig = ms.segment
    ms.segment = seg_fn
    try:
        return ms.decode(sec, R).text
    finally:
        ms.segment = orig


def true_cuts(taps, reps, word):
    """标准答案:按压序号 i 与 i+1 之间是 0=不切 / 1=字母边界 / 2=单词边界;不确定处为 -1。"""
    pos = {id(t): i for i, t in enumerate(taps)}
    lab = np.full(len(taps) - 1, -1)
    lens = [len(ms.MORSE[c]) for c in word]
    bnd = set(np.cumsum(lens)[:-1] - 1)
    for r in reps:
        idx = [pos[id(t)] for t in r]
        for k in range(len(idx) - 1):
            if idx[k + 1] == idx[k] + 1:
                lab[idx[k]] = 1 if k in bnd else 0
    for a, b in zip(reps[:-1], reps[1:]):
        ia, ib = pos[id(a[-1])], pos[id(b[0])]
        if ib == ia + 1:
            lab[ia] = 2
    return lab


def pred_cuts(words):
    lab = []
    for wi, wd in enumerate(words):
        for li, lt in enumerate(wd):
            lab += [0] * (len(lt) - 1)
            if li < len(wd) - 1:
                lab.append(1)
        if wi < len(words) - 1:
            lab.append(2)
    return np.array(lab)


def one_file(path):
    W, sec, rel, reps, wt, taps = D.word_reps(path)
    sec_full, R_full = ms.load_keysight_csv(path)
    K = sum(len(ms.MORSE[c]) for c in W)
    # 分母按她实际按的遍数(100),而不是能完整切出来的遍数——
    # 漏检/多检了按压的那几遍也要算进来,否则会把检测失误藏掉
    res = dict(word=W, n_elem=K, n_reps=REPS_PER_WORD, n_complete=len(reps),
               taps=len(taps), expected=REPS_PER_WORD * K)
    truth_full = " ".join([W] * REPS_PER_WORD)
    lab = true_cuts(taps, reps, W)
    short = []
    for name, fn in SEGMENTERS.items():
        # ③ 端到端
        text = decode_with(fn, sec_full, R_full)
        dec = text.split(" ")
        res[f"{name}_n_words"] = len(dec)
        res[f"{name}_exact"] = sum(d == W for d in dec)
        res[f"{name}_edit"] = levenshtein(text, truth_full)
        res[f"{name}_chars"] = len(truth_full)
        res[f"{name}_errors"] = "|".join(sorted({d for d in dec if d != W}))
        # ② 切分(在同一组按压上比较,和解码器内部一致)
        p = pred_cuts(fn(taps))
        ok = lab >= 0
        for lvl, nm in ((1, "letter"), (2, "word")):   # 单词边界同时也是字母边界
            t, q = lab[ok] >= lvl, p[ok] >= lvl
            tp = int((t & q).sum())
            res[f"{name}_{nm}_prec"] = tp / max(int(q.sum()), 1)
            res[f"{name}_{nm}_rec"] = tp / max(int(t.sum()), 1)
        # ④ 短消息:截原始信号单独解码
        for n in SHORT_LENS:
            for s in range(0, len(reps) - n + 1, n):
                chunk = reps[s:s + n]
                t0, t1 = chunk[0][0].t_start - PAD_S, chunk[-1][-1].t_end + PAD_S
                i0, i1 = np.searchsorted(sec_full, t0), np.searchsorted(sec_full, t1)
                truth = " ".join([W] * n)
                got = decode_with(fn, sec_full[i0:i1] - sec_full[i0], R_full[i0:i1])
                short.append(dict(word=W, method=name, n_words=n, start=s,
                                  exact=int(got == truth), edit=levenshtein(got, truth),
                                  chars=len(truth), got=got))
    return res, short


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(D.WORD_DIR, "*.csv")))
    out = Parallel(n_jobs=min(len(files), 11), backend="loky", verbose=0)(
        delayed(one_file)(f) for f in files)
    W = pd.DataFrame([r for r, _ in out])
    S = pd.DataFrame([x for _, s in out for x in s])
    W.to_csv(os.path.join(OUT, "per_word.csv"), index=False)
    S.to_csv(os.path.join(OUT, "short_messages.csv"), index=False)
    Y = pd.DataFrame(Parallel(n_jobs=11, backend="loky")(
        delayed(synth_job)(sg, t) for sg in SEGMENTERS for t in SYNTH))
    Y.to_csv(os.path.join(OUT, "synthetic_check.csv"), index=False)

    print("① 按压检测:检测到 %d / 应有 %d = %.2f%%;一次不差的录制 %d/%d"
          % (W.taps.sum(), W.expected.sum(), W.taps.sum() / W.expected.sum() * 100,
             int((W.taps == W.expected).sum()), len(W)))
    print("\n②③ 整份录制(%d 个单词实例,其中 %d 个能被完整切出)"
          % (W.n_reps.sum(), W.n_complete.sum()))
    print(f"{"切分方式":14s}{"整词全对":>10s}{'字错率':>9s}{'字母边界 查准/查全':>22s}{'单词边界 查准/查全':>22s}")
    for nm, cn in SEG_CN.items():
        ex = W[f"{nm}_exact"].sum() / W.n_reps.sum() * 100
        cer = W[f"{nm}_edit"].sum() / W[f"{nm}_chars"].sum() * 100
        lp, lr = W[f"{nm}_letter_prec"].mean() * 100, W[f"{nm}_letter_rec"].mean() * 100
        wp, wr = W[f"{nm}_word_prec"].mean() * 100, W[f"{nm}_word_rec"].mean() * 100
        print(f"{cn:10s}{ex:>9.2f}%{cer:>8.2f}%{lp:>12.2f}% / {lr:5.2f}%{wp:>12.2f}% / {wr:5.2f}%")
    print("\n④ 短消息(截原始信号单独解码):整条全对率")
    print(f"{'切分方式':10s}" + "".join(f"{n:>3d} 个词(n={int((S.method=='consensus')[S.n_words==n].sum()):4d})"
                                         .rjust(16) for n in SHORT_LENS))
    for nm, cn in SEG_CN.items():
        row = [S[(S.method == nm) & (S.n_words == n)].exact.mean() * 100 for n in SHORT_LENS]
        print(f"{cn:10s}" + "".join(f"{v:>15.2f}%" for v in row))
    print("\n⑤ 合成消息(8 月节奏):整条全对 / 字错率")
    print(f"{'切分方式':14s}" + "".join(f"{v:>18s}" for v in SYNTH.values()))
    for nm, cn in SEG_CN.items():
        r = Y[Y.segmenter == nm].set_index("test")
        print(f"{cn:14s}" + "".join(f"{r.loc[t,'exact']*100:>9.1f}% / {r.loc[t,'cer']*100:5.2f}%"
                                     for t in SYNTH))
    print(f"\n完成,用时 {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
