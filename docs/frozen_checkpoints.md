# Checkpoints frozen for the paper

Copies live here so they are never evicted by ModelCheckpoint's save_top_k and
never overwritten by `-latest`. Do not train into this directory.

## baseline_noser_seed1235_ep39.ckpt
HiFiGAN baseline reimplementation (Prabhu et al., arXiv 2306.01916).
  - trained on MSP-Podcast Train (63,076), validated on Development
  - Test1 held out; all reported numbers are Test1
  - L_SER DETACHED (logged, no gradient) -- avoids the closed loop with the
    audeering SER that also scores the benchmark
  - style: n/a (baseline has no style encoder; emo_proj on the annotated
    (EmoAct-1)/6 scalar)

Measured on 2000 random Test1 utterances x 7 target arousals, from the
epoch-41 `latest` checkpoint (since overwritten; this epoch-39 copy is the
nearest surviving equivalent and should be re-measured before publication):
  WVMOS  3.316   L_mse 0.093   L_abs 25.7%
Reference: paper z_l+z_s+z_e = 2.64 / 0.0971 / 26.4%
           paper +L_SER      = 3.26 / 0.0843 / 24.4%
           no conditioning   =  --  / 0.111  / 28.6%
           ground-truth Test1 WVMOS = 3.451
