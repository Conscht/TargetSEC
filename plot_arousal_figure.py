import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 8)

# Data
models = ["HiFiGAN", "EmoConv-Diff", "Uncert", "TargetSEC"]
# HiFiGAN and TargetSEC recomputed on the leak-free split: full MSP-Podcast
# Test1, 16,903 utterances per level, from each run's metadata.jsonl.
#   TargetSEC = LDM epoch 341   (eval_outputs/FULL_targetsec_ep341_test1)
#   HiFiGAN   = our reimplementation, epoch 99, L_SER detached
#               (eval_outputs/FULL_baseline_ep99_test1)
# EmoConv-Diff and Uncert are unchanged -- cited from their papers.
ser_mse_mean = {
    "HiFiGAN":      [0.2342, 0.1116, 0.0422, 0.0151, 0.0220, 0.0574, 0.1233],
    "EmoConv-Diff": [0.1644, 0.0905, 0.0469, 0.0129, 0.0253, 0.0638, 0.1102],
    "Uncert":       [0.1698, 0.0714, 0.0223, 0.0123, 0.0286, 0.0720, 0.1138],
    "TargetSEC":    [0.1691, 0.0604, 0.0258, 0.0087, 0.0115, 0.0335, 0.1475],
}

# Same two runs; per-level WVMOS from the benchmark stdout
#   TargetSEC: logs/tsec_full-2476703.out
#   HiFiGAN  : logs/full_base_eval-2473554.out
wvmos_mean = {
    "HiFiGAN":      [3.41, 3.39, 3.38, 3.36, 3.29, 3.13, 2.91],
    "EmoConv-Diff": [2.45, 2.27, 2.59, 2.59, 2.79, 2.57, 2.56],
    "Uncert" :      [3.37, 3.29, 3.32, 3.29, 3.27, 3.20, 3.25],
    "TargetSEC":    [3.46, 3.51, 3.63, 3.66, 3.60, 3.37, 3.42],
}

# --- COLOR DEFINITIONS ---
model_colors = {
    "HiFiGAN": "#1f77b4",       # Blue
    "EmoConv-Diff": "#ff7f0e",  # Orange
    "Uncert": "#2ca02c",        # Green
    "TargetSEC": "#7030a0"      # Professional Purple
}

M = len(models)
bar_w = 0.8 / M
offsets = (np.arange(M) - (M - 1) / 2) * bar_w

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2), dpi=220)

# Left: SER MSE
for i, m in enumerate(models):
    ax1.bar(x + offsets[i], ser_mse_mean[m], width=bar_w, label=m, color=model_colors[m])
ax1.set_xticks(x)
ax1.set_xticklabels([f"{v:.1f}" for v in x])
ax1.set_xlabel("Arousal")
ax1.set_ylabel(r"$\mathcal{L}_{mse}$")
ax1.set_ylim(0.0, 0.25)
ax1.grid(True, axis="y", alpha=0.25)

# Right: WVMOS
for i, m in enumerate(models):
    ax2.bar(x + offsets[i], wvmos_mean[m], width=bar_w, label=m, color=model_colors[m])
ax2.set_xticks(x)
ax2.set_xticklabels([f"{v:.1f}" for v in x])
ax2.set_xlabel("Arousal")
ax2.set_ylabel("WVMOS")
# WVMOS spans 1-5, so 1.0 is the scale's true floor and the bars must start
# there -- cutting the baseline to 2.0 would stretch a 3.66-vs-2.91 gap into a
# visual ~1.8x. All the dead space is at the TOP (max value 3.66), so trimming
# the ceiling buys ~25% more resolution and distorts nothing.
ax2.set_ylim(1.0, 4.0)
ax2.grid(True, axis="y", alpha=0.25)

# Legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), 
           ncol=4, frameon=True, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("ser_wvmos_by_arousal.pdf")
plt.savefig("ser_wvmos_by_arousal_means_only.png", dpi=300)
plt.show()