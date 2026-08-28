"""Which tap feature best separates dot vs dash GLOBALLY (across all recordings)?
Use every cleanly-segmented repetition (element count matches the known letter) to
label each tap dot/dash from ground truth, then score global thresholds on
height, width, and a normalized width."""
import glob, os
import numpy as np
from morse_pipeline import analyze_file, RAW_DIR, MORSE
from classify import EXCLUDE

def tap_area(a, p):
    m = (a["sec"] >= p[0]) & (a["sec"] <= p[1])
    return float(np.trapz(a["x"][m], a["sec"][m])) if m.sum() >= 2 else p[2]*p[3]

H, W, AR, Y, FILEMAXW = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    a = analyze_file(f)
    if a["token"] in EXCLUDE or not a["presses"]:
        continue
    code = MORSE[a["letters"][0]]; k = len(code)
    fw = np.median([p[3] for p in a["presses"]])
    for rep in a["reps"]:
        if len(rep) != k:
            continue
        for p, c in zip(rep, code):
            H.append(p[2]); W.append(p[3]); AR.append(tap_area(a, p))
            Y.append(1 if c == "-" else 0); FILEMAXW.append(fw)
H, W, AR, Y = map(np.array, (H, W, AR, Y))
nd = int((Y == 0).sum()); nh = int((Y == 1).sum())
print(f"labeled taps: {len(Y)}  (dots={nd}, dashes={nh})")

def best_threshold(x, y):
    ts = np.unique(np.quantile(x, np.linspace(0, 1, 300)))
    best_t, best_acc = ts[0], 0
    for t in ts:
        acc = max(((x >= t) == y).mean(), ((x < t) == y).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc

for name, x in [("height %", H), ("width  s", W), ("area %s", AR),
                ("width/filemedwidth", W/np.array(FILEMAXW))]:
    t, acc = best_threshold(x, Y)
    # separate accuracy on dots vs dashes at that threshold (dash = x>=t)
    pred = (x >= t).astype(int)
    dot_acc = (pred[Y == 0] == 0).mean(); dash_acc = (pred[Y == 1] == 1).mean()
    print(f"{name:20} best t={t:7.2f}  overall={acc*100:5.1f}%  "
          f"dot={dot_acc*100:4.0f}%  dash={dash_acc*100:4.0f}%")

# height mostly works within mixed files (relative). Test: for pure-type files width only.
print("\n-- width threshold sweep (for the pure/no-contrast case) --")
for t in [3.2, 3.4, 3.5, 3.55, 3.6, 3.8]:
    pred = (W >= t).astype(int)
    print(f"  t={t:4.2f}s  overall={((pred==Y).mean())*100:5.1f}%  "
          f"dot={((pred[Y==0]==0).mean())*100:4.0f}%  dash={((pred[Y==1]==1).mean())*100:4.0f}%")
