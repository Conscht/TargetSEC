#!/usr/bin/env python3
"""Paired significance test, TargetSEC vs the HiFiGAN baseline, on full Test1.

Both systems converted the same 16,903 utterances to the same 7 targets, so
every one of the 118,321 conversions is a matched pair. That pairing is what
makes a per-utterance test possible, and it is the substitute for multi-seed
training: it answers "is this gap real?" without another 3-day run, though it
does not answer "is it stable across initialisations".

Three levels of aggregation, because they make different independence claims:

  conversion (n=118,321)  -- 7 conversions of one utterance are NOT independent.
                             Reported for completeness; do not quote its p.
  utterance  (n=16,903)   -- average the 7 targets per utterance first. Sound
                             unless the same speaker recurs, which they do.
  speaker    (n=~1,400)   -- average per speaker. The conservative test, and
                             the one to quote.

    python3 stat_test_paired.py
"""
import os, json, math, ast
from collections import defaultdict
import numpy as np

# the repo root, one level up now that this lives in eval/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/eval_outputs/FULL_baseline_ep99_test1/metadata.jsonl"
TSEC = f"{ROOT}/eval_outputs/FULL_targetsec_ep341_test1/metadata.jsonl"
MEL  = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Test1"
LAB  = (f"{ROOT}/labels_consensus.csv")

def fmt_p(p):
    """Never print p=0. Doubles underflow below ~1e-308; say so instead."""
    if p != p: return "   nan"
    return "<1e-300" if p <= 0 else f"{p:.3g}"


try:
    from scipy import stats as _st
except ImportError:
    _st = None


def load(path):
    """(batch_idx, class) -> absolute arousal error."""
    err = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            err[(d["batch_idx"], d["class"])] = abs(d["pred_arousal"] - d["target_arousal"])
    return err


def paired(d, label, unit):
    """Paired t (+ Wilcoxon and a sign test) on d = baseline - targetsec.
    Positive mean => TargetSEC has the smaller error, i.e. is better."""
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    if _st is not None:
        p_t = 2 * _st.t.sf(abs(t), n - 1)
        # Wilcoxon is O(n log n) but scipy caps the exact form; normal approx is
        # what it uses at this n anyway.
        p_w = _st.wilcoxon(d, alternative="two-sided", method="approx").pvalue
        pos = int((d > 0).sum()); neg = int((d < 0).sum())
        p_s = _st.binomtest(pos, pos + neg, 0.5).pvalue
    else:
        p_t = math.erfc(abs(t) / math.sqrt(2))
        p_w = float("nan")
        pos = int((d > 0).sum()); neg = int((d < 0).sum())
        z = (pos - (pos + neg) / 2) / math.sqrt((pos + neg) / 4)
        p_s = math.erfc(abs(z) / math.sqrt(2))
    ci = 1.96 * se
    dz = m / d.std(ddof=1)          # Cohen's d_z for paired data
    print(f"  {label:<26} n={n:>7,}  mean diff {m:+.5f} +/- {ci:.5f}"
          f"  d_z={dz:+.3f}")
    print(f"  {'':<26} TargetSEC better on {100*pos/(pos+neg):5.1f}% of pairs"
          f"   t p={fmt_p(p_t)}   wilcoxon p={fmt_p(p_w)}   sign p={fmt_p(p_s)}")


print(__doc__.split("\n")[0])
print(f"scipy: {'yes' if _st else 'no (normal approximations)'}\n")

eb, et = load(BASE), load(TSEC)
keys = sorted(set(eb) & set(et))
assert len(keys) == len(eb) == len(et) == 118321, (len(keys), len(eb), len(et))

b = np.array([eb[k] for k in keys])
t_ = np.array([et[k] for k in keys])
idx = np.array([k[0] for k in keys])
cls = np.array([k[1] for k in keys])

print("Headline L_abs (mean absolute arousal error, [0,1] scale)")
print(f"  baseline   {b.mean():.4f}")
print(f"  TargetSEC  {t_.mean():.4f}")
print(f"  difference {b.mean()-t_.mean():+.4f}\n")

print("Paired tests on d = |err_baseline| - |err_TargetSEC|  (positive favours TargetSEC)")
paired(b - t_, "conversion level", "conversion")

# ---- utterance level ----
n_utt = idx.max() + 1
bu = np.bincount(idx, weights=b, minlength=n_utt) / 7.0
tu = np.bincount(idx, weights=t_, minlength=n_utt) / 7.0
paired(bu - tu, "utterance level", "utterance")

# ---- speaker level ----
entries = [f.replace("_mel.pt", ".wav")
           for f in sorted(x for x in os.listdir(MEL) if x.endswith("_mel.pt"))]
assert len(entries) == n_utt, (len(entries), n_utt)

spk_of_file = {}
with open(LAB) as f:
    hdr = f.readline().strip().split(",")
    ci_name, ci_spk = hdr.index("FileName"), hdr.index("SpkrID")
    for line in f:
        p = line.rstrip("\n").split(",")
        if len(p) > max(ci_name, ci_spk):
            spk_of_file[p[ci_name]] = p[ci_spk]

groups = defaultdict(list)
unknown = 0
for i, fn in enumerate(entries):
    s = spk_of_file.get(fn)
    if s is None or s in ("Unknown", ""):
        unknown += 1
        continue
    groups[s].append(i)

bs = np.array([bu[v].mean() for v in groups.values()])
ts = np.array([tu[v].mean() for v in groups.values()])
paired(bs - ts, "speaker level  <- quote this", "speaker")
print(f"  {'':<26} {len(groups)} speakers, {unknown} utterances with no speaker ID dropped\n")

# ---- per class, at utterance level ----
print("Per target level (utterance-level pairs within each class, n=16,903)")
print("  class   baseline  TargetSEC      diff        t p       better on")
for c in range(1, 8):
    m = cls == c
    bc, tc = b[m], t_[m]
    d = bc - tc
    se = d.std(ddof=1) / math.sqrt(len(d))
    tt = d.mean() / se
    p = 2 * _st.t.sf(abs(tt), len(d) - 1) if _st else math.erfc(abs(tt) / math.sqrt(2))
    win = 100 * (d > 0).mean()
    flag = "" if d.mean() > 0 else "   <- baseline wins"
    print(f"    {c}     {bc.mean():.4f}    {tc.mean():.4f}   {d.mean():+.4f}"
          f"   {fmt_p(p):>9}   {win:5.1f}%{flag}")
