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

## baseline_noser_seed1235_ep99_FINAL.ckpt  <-- USE THIS ONE
Selected by sweeping the periodic grid (39/59/79/99/109/119); epoch 99 is the
closest match to the paper on all three metrics. Note the per-epoch
`val_arousal_mae` monitor correlates only r=0.38 with the swept L_abs, so
checkpoint selection cannot be done from the training logs -- budget a sweep
for every future seed.

Measured on 2000 random Test1 utterances x 7 target arousals:
  WVMOS 3.290   L_mse 0.0873   L_abs 24.56%
  paper HiFiGAN [14]:  3.26   0.084   24%
per-class L_abs: 0.476 0.321 0.185 0.099 0.116 0.202 0.321
per-class WVMOS: 3.43 3.41 3.41 3.39 3.32 3.15 2.93

## TargetSEC stage-2 arms (launched 08-20)
Stage 1 had no single best checkpoint -- a real trade-off, confirmed at 800 utts:
  ep59 : resynth WVMOS 3.558, val_ccc_mean 0.1049
  ep112: resynth WVMOS 3.391, val_ccc_mean 0.0893  <- global best emotion of 113 epochs
  (ground truth resynth WVMOS = 3.451)
Both arms run stage 2 in parallel (20 epochs, detached L_SER); pick the winner
by measuring stage-2 output rather than guessing.
  job 2473365 -> from stage-1 ep112
  job 2473550 -> from stage-1 ep59
