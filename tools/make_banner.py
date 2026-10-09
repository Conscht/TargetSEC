#!/usr/bin/env python3
"""Build the repository banner, mirroring the project page's header.

Strip, caption, centred wordmark, subtitle -- the same order as the page, so
the repo and the site read as one thing. The strip is one Test1 utterance
converted by TargetSEC to all seven arousal levels: real amplitude envelopes,
so the energy visibly grows from calm to activated.

The end faces are drawn rather than set as emoji: the available font has a
sleeping face but no star-struck one, its emoji are monochrome, and drawing
them keeps both on the project's palette.

    python3 tools/make_banner.py          # writes docs/banner.png
"""
import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "src01_spk035_M"
BG, FG, MUTED = "#131218", "#f3f1f7", "#8b8599"
# the page's --w1..--w7, the strip ramp chosen to stay visible on either ground
W_RAMP = ["#403a52", "#4e4366", "#5e4f7e", "#705d96", "#836caf", "#977cc6", "#ac8fdd"]
W, H, DPI = 2400, 800, 200

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
    e = np.pad(x, (0, (-len(x)) % n)).reshape(n, -1).max(1)
    return e / (e.max() + 1e-9)


def face(ax, x, y, t, fill, px=30):
    """A face whose expression opens with t, matching the page's ramp faces."""
    rx, ry = px / W, px / H
    ink = "#e9e4f2" if t < 0.75 else "#ffffff"
    ax.add_patch(Ellipse((x, y), 2 * rx, 2 * ry, facecolor=fill, edgecolor="none", zorder=4))
    er = 0.115 + 0.022 * t
    for dx in (-0.37, 0.37):
        ax.add_patch(Ellipse((x + rx * dx, y + ry * (0.27 + 0.11 * t)),
                             2 * rx * er, 2 * ry * er, facecolor=ink, edgecolor="none", zorder=5))
    m0 = y - ry * (0.33 - 0.12 * t)
    ax.add_patch(PathPatch(
        Path([(x - rx * .47, m0), (x, m0 - ry * 1.0 * t), (x + rx * .47, m0)],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
        fc="none", ec=ink, lw=2.0, capstyle="round", zorder=5))


fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── the strip ───────────────────────────────────────────────────────────────
x0, x1, ymid, amp = 0.045, 0.955, 0.775, 0.145
seg = (x1 - x0) / 7
for c in range(1, 8):
    e = envelope(os.path.join(ROOT, "docs", "audio", f"{SRC}__target{c}__targetsec.ogg"))
    xs = np.linspace(x0 + (c - 1) * seg, x0 + c * seg - 0.006, len(e))
    ax.fill_between(xs, ymid - e * amp, ymid + e * amp, color=W_RAMP[c - 1], linewidth=0)

# ── caption, faces at the ends ──────────────────────────────────────────────
cy = 0.565
face(ax, x0 + 0.013, cy, 0.0, W_RAMP[0])
face(ax, x1 - 0.013, cy, 1.0, W_RAMP[6])
ax.text(x0 + 0.033, cy, "1  ·  CALM", color=MUTED, fontsize=11.5, family="DejaVu Sans",
        ha="left", va="center")
ax.text(0.5, cy, "ONE UTTERANCE, SEVEN LEVELS", color=MUTED, fontsize=11.5,
        family="DejaVu Sans", ha="center", va="center")
ax.text(x1 - 0.033, cy, "7  ·  ACTIVATED", color=MUTED, fontsize=11.5,
        family="DejaVu Sans", ha="right", va="center")

# ── wordmark ────────────────────────────────────────────────────────────────
ax.text(0.5, 0.345, "TargetSEC", color=FG, fontsize=62, family=DISPLAY,
        ha="center", va="center")
ax.text(0.5, 0.135,
        "arousal-conditioned latent style diffusion  ·  in-the-wild speech emotion conversion",
        color=MUTED, fontsize=14, family="DejaVu Sans", ha="center", va="center")

out = os.path.join(ROOT, "docs", "banner.png")
fig.savefig(out, dpi=DPI, facecolor=BG)
print(f"wrote {out}  ({os.path.getsize(out)/1e3:.0f} kB, {W}x{H})")
