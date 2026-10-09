#!/usr/bin/env python3
"""Regenerate ser_wvmos_by_arousal.pdf from the post-leak-fix Test1 results.

Two panels, never one with two y-axes: L_mse and WVMOS are different scales and
a dual axis lets the reader infer any relationship the placement suggests.

L_mse is recomputed here from each run's metadata.jsonl, so the figure cannot
drift from the tables. WVMOS is per-class and only exists in the benchmark
stdout, so it is transcribed with the source log named beside it.

Colours are the dataviz reference categorical palette in fixed slot order, and
that assignment follows the SYSTEM, not its rank -- adding Uncert below must not
repaint TargetSEC. Validated --pairs all on a white surface: worst CVD dE 9.2,
worst normal-vision dE 16.3. Aqua sits at 2.82:1 on white, so every series also
carries a distinct marker and dash pattern; that doubles as the grayscale-print
and colourblind fallback.

    python3 plot_arousal_figure.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LEVELS = np.arange(1, 8)

# Ground-truth Test1 WVMOS. Worth drawing: every system here scores at or above
# it, because WVMOS penalises the recording noise in in-the-wild podcast audio.
GT_WVMOS = 3.451

SYSTEMS = [
    # key            label                       colour     marker  dash    metadata.jsonl
    ("targetsec",   "TargetSEC (ours)",         "#2a78d6", "o", (None, None),
     "eval_outputs/FULL_targetsec_ep341_test1/metadata.jsonl"),
    ("hifigan",     "HiFiGAN [10]",             "#eb6834", "s", (4, 2),
     "eval_outputs/FULL_baseline_ep99_test1/metadata.jsonl"),
    # To add the cited baselines, fill in per-level values below and append here:
    #   ("uncert",  "Uncert [12]",   "#1baf7a", "^", (1, 2), None),
    #   ("emoconv", "EmoConv-Diff [9]", "#4a3aa7", "D", (6, 2, 1, 2), None),
]

# Per-class WVMOS -- from the benchmark stdout, which is the only place it exists.
#   targetsec: logs/tsec_full-2476703.out
#   hifigan  : logs/full_base_eval-2473554.out
WVMOS = {
    "targetsec": [3.4639, 3.5123, 3.6275, 3.6622, 3.6007, 3.3742, 3.4180],
    "hifigan":   [3.4102, 3.3880, 3.3836, 3.3628, 3.2909, 3.1278, 2.9094],
}

# Per-level values for systems we did not run (cited from their papers).
LMSE_CITED = {}


def lmse_per_level(path):
    """Mean squared arousal error per target level, straight from the manifest."""
    acc = {c: [] for c in range(1, 8)}
    with open(os.path.join(ROOT, path)) as f:
        for line in f:
            d = json.loads(line)
            acc[d["class"]].append((d["pred_arousal"] - d["target_arousal"]) ** 2)
        return [float(np.mean(acc[c])) for c in range(1, 8)]


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.45))

INK, MUTED = "#0b0b0b", "#52514e"
for ax in (axL, axR):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=3)
    ax.grid(axis="y", color="#e6e5e2", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(LEVELS)
    ax.set_xlabel("Target arousal level")

handles = []
for key, label, colour, marker, dash, meta in SYSTEMS:
    y = lmse_per_level(meta) if meta else LMSE_CITED[key]
    ln, = axL.plot(LEVELS, y, color=colour, marker=marker, markersize=4.2,
                   linewidth=1.6, dashes=dash, zorder=3,
                   markeredgecolor="white", markeredgewidth=0.6, label=label)
    handles.append(ln)
    axR.plot(LEVELS, WVMOS[key], color=colour, marker=marker, markersize=4.2,
             linewidth=1.6, dashes=dash, zorder=3,
             markeredgecolor="white", markeredgewidth=0.6)

axL.set_ylabel(r"SER error $\mathcal{L}_{mse}$ $\downarrow$")
axL.set_ylim(0, None)

axR.set_ylabel(r"WVMOS $\uparrow$")
gt = axR.axhline(GT_WVMOS, color=MUTED, linewidth=0.8, dashes=(2, 2), zorder=1,
                 label="ground-truth Test1")
handles.append(gt)

fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015),
           ncol=len(handles), frameon=False, handlelength=2.6,
           columnspacing=1.8, labelcolor=INK)

fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("pdf", "png"):
    p = os.path.join(ROOT, f"ser_wvmos_by_arousal.{ext}")
    fig.savefig(p, dpi=220, bbox_inches="tight")
    print(f"wrote {p}")
