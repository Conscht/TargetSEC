# TargetSEC — Target-Arousal Speech Emotion Conversion via Latent Diffusion

Encoder–decoder speech synthesis system for **target arousal control** on MSP-Podcast v1.10.  
The style slot of a HiFi-GAN-based decoder is replaced at inference time by a **Latent Diffusion Model (LDM)** conditioned on emotion embeddings and speaker identity.

---

## System Overview

```
Content  : HuBERT tokens  (128-dim after projection)
Speaker  : WavLM ECAPA   (512-dim, broadcast)
Style    : MelStyleEncoder (128-dim)
             ↑ replaced at inference by LDM output
Decoder  : HiFi-GAN (upsampling vocoder)
```

**At inference**, the style encoder is replaced by the LDM:

```
Emotion embedding (1024-dim, audeering wav2vec2)
Speaker embedding (512-dim)
        ↓ LDM (DDPM, 1000 steps / DDIM 50 steps)
Style vector (128-dim)  →  HiFi-GAN decoder  →  waveform
```

Emotion embeddings at inference are **top-20% class averages** over Test1 (7 arousal classes, `avgclass_emo_embeds/Test1/{1..7}.npy`).

---

## Style Encoder Fine-Tuning

The MelStyleEncoder (Meta-StyleSpeech, ~500 K params) was originally pretrained on LibriTTS.  
It is **fine-tuned end-to-end** on MSP-Podcast together with the decoder using a 10× lower learning rate, adapting to spontaneous emotional speech while preserving the pretrained structure.

### Style Embedding Space — Before vs. After Fine-Tuning

Points are coloured by arousal class (1 = calm/blue → 7 = activated/red).

**UMAP**

![Style shift UMAP](style_shift_umap.png)

**PCA**

![Style shift PCA](style_shift_pca.png)

Arrows in the right panel show the per-utterance shift from the original (○) to the fine-tuned (△) encoder.  
Both projections are computed on all 16 903 Test1 utterances; arrows are subsampled to 2 000 for clarity.

---

## Components

| Module | File | Notes |
|--------|------|-------|
| Style encoder | `StyleSpeech/models/StyleSpeech.py` | MelStyleEncoder, fine-tuned |
| Synthesizer (decoder) | `src/synthesizer_style_module.py` | HiFi-GAN + style slot |
| LDM | `src/diffusion_module_fixed.py` | DDPM conditioned on emo + speaker |
| HiFiGAN baseline | `src/synthesizer_hifigan_module.py` | Scalar arousal → Linear(1,128), no style encoder |
| Emotion encoder | `src/emotion/emotion_encoder.py` | audeering wav2vec2-large, frozen |

---

## Training

### Synthesizer fine-tune
```bash
sbatch slurm_jobs/training_synthesizer_finetune.slurm
# checkpoint: checkpoints_synthesizer_finetune/synthesizer_finetune-*-epoch=48-val_loss=16.24.ckpt
```

### LDM fine-tune
```bash
sbatch slurm_jobs/train_ldm_finetune.slurm
# checkpoint: checkpoints_ldm_finetune/ldm_finetune-*-epoch=596-val_loss=0.4642.ckpt
# style stats: style_stats_finetune.pt  (computed by compute_style_stats_finetune.py)
```

### HiFiGAN baseline (scalar arousal, faithful [7] reimplementation)
```bash
sbatch slurm_jobs/training_hifigan_baseline.slurm
# checkpoint: checkpoints_hifigan_baseline_scalar/hifigan_baseline_scalar-*-epoch=137-val_loss=19.69.ckpt
```

---

## Evaluation

```bash
# TargetSEC full pipeline (synthesizer + LDM)
sbatch slurm_jobs/benchmark_finetune.slurm
# → eval_outputs/ldm_finetune_epoch596/

# HiFiGAN scalar baseline
sbatch slurm_jobs/benchmark_hifigan_baseline.slurm
# → eval_outputs/hifigan_baseline_scalar_epoch137/

# WVMOS is run automatically after each benchmark via vmos_calc_*.py
```

Metrics computed per arousal class: **MAE**, **MSE** (audeering arousal), **WVMOS**.  
Independent intelligibility: Whisper-medium WER (`wer_eval.py`).  
Independent emotion ranking: SpeechBrain IEMOCAP Spearman ρ (`second_emotion_eval.py`).

---

## Dataset

**MSP-Podcast v1.10** — spontaneous conversational speech, 16 kHz  
`/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/`

| Split | Utterances |
|-------|-----------|
| Train | ~60 k |
| Val   | ~6 k |
| Test1 | 16 903 |

HuBERT tokens + WavLM speaker embeddings preprocessed to:  
`hubert-km100/parsed_with_spkrEmbeds/{train,val,test1}.txt`

---

## Environment

Python 3.10, PyTorch 2.3.1, PyTorch-Lightning 2.3.0  
Key packages: `transformers 4.41`, `diffusers 0.18`, `librosa 0.9`, `wvmos 1.0`

```bash
conda activate emoldm
```
