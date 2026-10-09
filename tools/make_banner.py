#!/usr/bin/env python3
"""Build the repository banner, mirroring the project page's header.

Strip, caption, centred wordmark, subtitle -- the same order as the page, so
the repo and the site read as one thing. The strip is one Test1 utterance
converted by TargetSEC to all seven arousal levels: real amplitude envelopes,
so the energy visibly grows from calm to activated.

The end emoji are the same codepoints the page uses (U+1F634, U+1F929),
composited from Noto Color Emoji. A PNG bakes the glyph in, so unlike the
page -- where the reader's own OS draws them and Windows shows Segoe -- one
style has to be chosen for everyone. Noto is the closest redistributable
match; Segoe UI Emoji is proprietary and cannot be shipped.

    python3 tools/make_banner.py          # writes docs/banner.png
"""
import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

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


def emoji(ax, x, y, cp, px=46):
    """Composite a Noto Color Emoji PNG at (x, y), px wide in device pixels."""
    f = os.path.join(ROOT, "tools", "emoji", f"emoji_u{cp}.png")
    img = mpimg.imread(f)
    # OffsetImage zoom is image-pixels per point, and a point is DPI/72 device px
    zoom = px / (img.shape[0] * DPI / 72)
    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y),
                                 frameon=False, box_alignment=(0.5, 0.5), zorder=6))


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
emoji(ax, x0 + 0.010, cy, "1f634")
emoji(ax, x1 - 0.010, cy, "1f929")
ax.text(x0 + 0.036, cy, "1  ·  CALM", color=MUTED, fontsize=11.5, family="DejaVu Sans",
        ha="left", va="center")
ax.text(0.5, cy, "ONE UTTERANCE, SEVEN LEVELS", color=MUTED, fontsize=11.5,
        family="DejaVu Sans", ha="center", va="center")
ax.text(x1 - 0.036, cy, "7  ·  ACTIVATED", color=MUTED, fontsize=11.5,
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
