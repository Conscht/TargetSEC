#!/usr/bin/env python3
"""Regenerate the page header's waveform strip, in place.

The strip is one Test1 utterance converted by TargetSEC to all seven arousal
levels: real amplitude envelopes, so the energy visibly grows left to right.
It is written as inline SVG rather than an image so it scales, stays crisp,
and takes its colours from the page's theme tokens.

    python3 tools/make_wave_header.py      # rewrites the block in docs/index.html
"""
import os, re
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, N, W, H = "src01_spk035_M", 96, 1200, 132
BEGIN, END = "<!-- wave:begin -->", "<!-- wave:end -->"


def envelope(path, n=N):
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.abs(x)
    e = np.pad(x, (0, (-len(x)) % n)).reshape(n, -1).max(1)
    return e / (e.max() + 1e-9)


seg, mid, amp, gap = W / 7, H / 2, H / 2 - 9, 7
paths = []
for c in range(1, 8):
    e = envelope(os.path.join(ROOT, "docs", "audio", f"{SRC}__target{c}__targetsec.ogg"))
    x0, x1 = (c - 1) * seg, c * seg - gap
    xs = np.linspace(x0, x1, N)
    top = " ".join(f"{x:.1f},{mid - v * amp:.1f}" for x, v in zip(xs, e))
    bot = " ".join(f"{x:.1f},{mid + v * amp:.1f}" for x, v in zip(xs[::-1], e[::-1]))
    paths.append(f'<polygon class="w{c}" points="{top} {bot}"/>')

svg = (f'{BEGIN}\n  <svg class="wave" viewBox="0 0 {W} {H}" role="img" '
       f'xmlns="http://www.w3.org/2000/svg"\n'
       f'       aria-label="One utterance converted to all seven arousal levels, '
       f'drawn as amplitude envelopes growing from calm on the left to activated '
       f'on the right.">\n    '
       + "\n    ".join(paths) + f'\n  </svg>\n  {END}')

p = os.path.join(ROOT, "docs", "index.html")
s = open(p).read()
s = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), svg, s, flags=re.S)
open(p, "w").write(s)
print(f"wrote the strip into {p}  ({len(svg)/1024:.1f} kB of SVG)")
