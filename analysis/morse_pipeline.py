"""
Morse-code resistive-sensor raw-data pipeline.

Each CSV in ../raw data/ is a Keysight 34461A resistance-vs-time recording of one
letter's Morse code pressed many times. This module provides the shared core:
  load -> per-file baseline / relative change -> deglitch -> press-event detection
  -> two-level grouping (elements within a letter; repetitions between letters)
  -> per-element dot/dash width, decode, and compare against the filename label.

Run directly to print a QC table over every file.
"""
import os, re, glob, json
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw data")

MORSE = {
    'A': '.-',   'B': '-...', 'C': '-.-.', 'D': '-..',  'E': '.',
    'F': '..-.', 'G': '--.',  'H': '....', 'I': '..',   'J': '.---',
    'K': '-.-',  'L': '.-..', 'M': '--',   'N': '-.',   'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.',  'S': '...',  'T': '-',
    'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-', 'Y': '-.--',
    'Z': '--..',
}
MORSE_INV = {v: k for k, v in MORSE.items()}


def parse_name(basename):
    """'B-20 2026-08-18 20-01-25 0.csv' -> label token 'B-20', letter 'B', number 20.
    'D13+F58 ...' -> letter list ['D','F'], numbers [13,58]."""
    token = basename.split(" ")[0]
    letters, numbers = [], []
    for part in token.split("+"):
        m = re.match(r"([A-Za-z])[-]?([0-9]*)", part)
        if m:
            letters.append(m.group(1).upper())
            numbers.append(int(m.group(2)) if m.group(2) else None)
    return token, letters, numbers


def load(path):
    """Return (t_sec, R_ohm) with absolute timestamps converted to relative seconds."""
    df = pd.read_csv(path, skiprows=7, names=["t", "R"], header=0)
    t = pd.to_datetime(df["t"], errors="coerce", format="mixed")
    m = t.notna()
    df, t = df[m], t[m]
    sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    R = pd.to_numeric(df["R"], errors="coerce").to_numpy()
    good = ~np.isnan(R)
    return sec[good], R[good]


def deglitch(R, ratio=3.0):
    """Replace only physically implausible spikes (contact faults like K-30's 23 MOhm).
    A sample is bad only if it is >ratio x or <1/ratio x the local rolling median, so
    real 3-30% presses are never touched."""
    R = R.astype(float).copy()
    med = pd.Series(R).rolling(9, center=True, min_periods=1).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = R / med
    bad = (rel > ratio) | (rel < 1.0 / ratio)
    R[bad] = med[bad]
    return R, int(bad.sum())


def to_relative(R):
    """Per-file baseline R0 (robust low-percentile) -> (R-R0)/R0 in %."""
    R0 = np.percentile(R, 20)
    return (R - R0) / R0 * 100.0, R0


def detect_presses(sec, x, min_width=0.6, min_sep=0.9):
    """Hysteresis event detection. Returns list of (t_start, t_end, peak, width)."""
    fs = len(sec) / (sec[-1] - sec[0])
    top = np.percentile(x, 98)
    hi = max(0.8, 0.35 * top)   # must exceed this to count as a press
    lo = max(0.4, 0.15 * top)   # extent boundary
    above_lo = x > lo
    d = np.diff(above_lo.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if above_lo[0]:
        starts = np.r_[0, starts]
    if above_lo[-1]:
        ends = np.r_[ends, len(above_lo)]
    presses = []
    for s, e in zip(starts, ends):
        seg = x[s:e]
        if seg.size == 0 or seg.max() < hi:
            continue
        w = sec[e - 1] - sec[s]
        if w < min_width:
            continue
        presses.append([sec[s], sec[e - 1], float(seg.max()), float(w)])
    # merge presses separated by less than min_sep (dropout during one press)
    merged = []
    for p in presses:
        if merged and p[0] - merged[-1][1] < min_sep:
            merged[-1][1] = p[1]
            merged[-1][2] = max(merged[-1][2], p[2])
            merged[-1][3] = merged[-1][1] - merged[-1][0]
        else:
            merged.append(p)
    return merged


def _split_by_thr(presses, gaps, thr):
    reps, cur = [], [presses[0]]
    for i, g in enumerate(gaps):
        if g > thr:
            reps.append(cur); cur = []
        cur.append(presses[i + 1])
    reps.append(cur)
    return reps


def group_repetitions(presses, n_elements):
    """Split presses into repetitions. Because we know the letter, each repetition
    should contain exactly n_elements dots/dashes. We sweep the gap threshold that
    separates short 'element gaps' from long 'letter gaps' and pick the one that
    yields the most repetitions with exactly n_elements elements (for single-element
    letters, the most singletons)."""
    if len(presses) <= 1:
        return [presses]
    gaps = np.array([presses[i + 1][0] - presses[i][1] for i in range(len(presses) - 1)])
    if n_elements <= 1:
        return _split_by_thr(presses, gaps, max(1.5, 0.4 * np.median(gaps)))
    # candidate thresholds = midpoints between consecutive sorted gap values
    sg = np.unique(gaps)
    cands = (sg[:-1] + sg[1:]) / 2 if len(sg) > 1 else sg
    best_thr, best_score = np.median(gaps), -1
    for thr in cands:
        reps = _split_by_thr(presses, gaps, thr)
        exact = sum(len(r) == n_elements for r in reps)
        # prefer thresholds giving many exact-count reps; tie-break toward fewer total
        score = exact - 0.001 * len(reps)
        if score > best_score:
            best_score, best_thr = score, thr
    return _split_by_thr(presses, gaps, best_thr)


def classify_elements(reps, dot_dash_thr):
    """Turn each repetition's element widths into a morse string using a width threshold."""
    out = []
    for rep in reps:
        code = "".join("-" if p[3] >= dot_dash_thr else "." for p in rep)
        out.append(code)
    return out


def analyze_file(path):
    base = os.path.basename(path)
    token, letters, numbers = parse_name(base)
    sec, R = load(path)
    R, n_glitch = deglitch(R)
    x, R0 = to_relative(R)
    presses = detect_presses(sec, x)
    n_elem = len(MORSE[letters[0]]) if letters else 1
    reps = group_repetitions(presses, n_elem)
    widths = np.array([p[3] for p in presses]) if presses else np.array([])
    return dict(base=base, token=token, letters=letters, numbers=numbers,
                sec=sec, x=x, R0=R0, n_glitch=n_glitch, n_elem=n_elem,
                presses=presses, reps=reps, widths=widths)


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"{'file':10} {'let':4} {'Ek':>3} {'num':5} {'R0':>7} {'glit':>4} "
          f"{'#press':>6} {'reps':>4} {'N/Ek':>5} {'elem/rep':>8} {'wmed':>5}")
    rows = []
    for f in files:
        a = analyze_file(f)
        nrep = len(a["reps"])
        npress = len(a["presses"])
        elem_per = npress / nrep if nrep else 0
        implied = npress / a["n_elem"] if a["n_elem"] else 0   # reps implied by press count
        w = a["widths"]
        wmed = np.median(w) if w.size else 0
        num = "+".join(str(n) for n in a["numbers"])
        print(f"{a['token']:10} {'/'.join(a['letters']):4} {a['n_elem']:3d} {num:5} "
              f"{a['R0']:7.0f} {a['n_glitch']:4d} {npress:6d} {nrep:4d} {implied:5.0f} "
              f"{elem_per:8.2f} {wmed:5.2f}")
        rows.append(dict(token=a["token"], letters="/".join(a["letters"]), n_elem=a["n_elem"],
                         num=num, R0=round(a["R0"], 1), glitches=a["n_glitch"],
                         n_press=npress, n_reps=nrep, reps_implied=round(implied, 1),
                         elem_per_rep=round(elem_per, 2), w_med=round(float(wmed), 2)))
    with open(os.path.join(os.path.dirname(__file__), "out", "qc_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
