#!/usr/bin/env python3
"""Build the repository banner from real model output.

The strip across the banner is one Test1 utterance converted by TargetSEC to
all seven arousal levels, drawn as amplitude envelopes side by side and
coloured along the arousal ramp. So the banner shows the thing it advertises
-- the same words and the same speaker, carried from calm to activated --
rather than being decoration.

    python3 tools/make_banner.py          # writes docs/banner.png
"""
import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = "src01_spk035_M"
BG, FG, MUTED = "#131218", "#f3f1f7", "#8b8599"
RAMP = ["#3c3a47", "#4e4463", "#614f80", "#75599d", "#8a66b8", "#9f74d1", "#b490e8"]
W, H, DPI = 2400, 620, 200

# the page's display face, so banner and site share a wordmark
_f = os.path.join(ROOT, "tools", "fonts", "Newsreader-Medium.ttf")
if os.path.exists(_f):
    fm.fontManager.addfont(_f)
    DISPLAY = fm.FontProperties(fname=_f).get_name()
else:
    DISPLAY = "DejaVu Serif"


def envelope(path, n=460):
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.abs(x)
    pad = (-len(x)) % n
    e = np.pad(x, (0, pad)).reshape(n, -1).max(1)
    return e / (e.max() + 1e-9)


fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.text(0.042, 0.80, "TargetSEC", color=FG, fontsize=54,
        family=DISPLAY, va="center", ha="left")
ax.text(0.042, 0.635, "arousal-conditioned latent style diffusion  ·  in-the-wild speech emotion conversion",
        color=MUTED, fontsize=13.5, family="DejaVu Sans", va="center", ha="left")

# the strip: seven conversions of one utterance, left calm, right activated
x0, x1, ymid, amp = 0.042, 0.958, 0.30, 0.175
seg = (x1 - x0) / 7
for c in range(1, 8):
    p = os.path.join(ROOT, "docs", "audio", f"{SRC}__target{c}__targetsec.ogg")
    env = envelope(p)
    xs = np.linspace(x0 + (c - 1) * seg, x0 + c * seg - 0.006, len(env))
    ax.fill_between(xs, ymid - env * amp, ymid + env * amp,
                    color=RAMP[c - 1], linewidth=0)
    lab = {1: "1  ·  calm", 7: "7  ·  activated"}.get(c, str(c))
    ax.text(xs.mean(), 0.075, lab, color=MUTED, fontsize=11,
            family="DejaVu Sans", ha="center", va="center")
ax.add_patch(Rectangle((x0, 0.137), x1 - x0, 0.0035, color="#2b2834", lw=0))

out = os.path.join(ROOT, "docs", "banner.png")
fig.savefig(out, dpi=DPI, facecolor=BG)
print(f"wrote {out}  ({os.path.getsize(out)/1e3:.0f} kB, {W}x{H})")
