"""
Benchmark the fine-tuned TargetSEC model (epoch=48, val_loss=16.24).

Runs the full pipeline: fine-tuned synthesizer + fine-tuned LDM (epoch=596,
val_loss=0.4642) → saves WAVs per arousal class → WVMOS computed separately
via vmos_calc.py.
"""
import os
import json
from collections import defaultdict

import torch
import torchaudio
import pytorch_lightning as pl

from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.synthesizer_style_module import SynthesizerLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
from src.diffusion_module_fixed import DiffusionLightningModule
from src.decoder.decoder_modules import broadcast_embeddings
from src.emotion.emotion_encoder import process_func

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT_SYNTH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_synthesizer_finetune/"
    "synthesizer_finetune-06-13_18-22-18-epoch=48-val_loss=16.24.ckpt"
)
CHECKPOINT_LDM = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_ldm_finetune/ldm_finetune-06-14_13-02-09-epoch=596-val_loss=0.4642.ckpt"
)
PRETRAINED_STYLE_PATH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
)
EMOTION_EMBEDDING_DIR = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/avgclass_emo_embeds"
)
SAVE_ROOT = "eval_outputs/ldm_finetune_epoch596"
SAVE_AUDIO_LIMIT = None  # None = all utterances

# ── Configs ────────────────────────────────────────────────────────────────
config_synth = {
    "cross_attention_dim": 768,
    "generator": {
        "input_dim": 768,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        "upsample_rates": [5, 4, 4, 2, 2],
        "upsample_initial_channel": 1024,
        "upsample_kernel_sizes": [11, 8, 8, 4, 4],
        "gin_channels": 0,
        "resblock": "1",
    },
    "data": {
        "sampling_rate": 16000,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
    },
    "training": {"learning_rate": 5e-5, "batch_size": 8},
}

config_ldm = {
    "training": {"learning_rate": 3e-5, "batch_size": 32, "cfg_prob": 0.3, "warmup_steps": 55_000},
    "inference": {"guidance_scale": 4.0, "guidance_rescale": 0.7},
    "cross_attention_dim": 768,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_emotion_embeddings(embedding_dir, split="Test1"):
    import numpy as np
    out = {}
    split_path = os.path.join(embedding_dir, split)
    for c in range(1, 8):
        out[c] = np.load(os.path.join(split_path, f"{c}.npy"))
    return out


if __name__ == "__main__":
    pl.seed_everything(1234)

    # ── Output dirs ──────────────────────────────────────────────────────────
    os.makedirs(SAVE_ROOT, exist_ok=True)
    meta_path = os.path.join(SAVE_ROOT, "metadata.jsonl")
    wav_root = os.path.join(SAVE_ROOT, "wav")
    for c in range(1, 8):
        os.makedirs(os.path.join(wav_root, f"class_{c}"), exist_ok=True)

    sr = config_synth["data"]["sampling_rate"]

    # ── Emotion embeddings ───────────────────────────────────────────────────
    emotion_embeddings = load_emotion_embeddings(EMOTION_EMBEDDING_DIR, split="Test1")
    emo_bank = torch.stack(
        [torch.tensor(emotion_embeddings[c]) for c in range(1, 8)], dim=0
    ).to(device)
    if emo_bank.ndim == 2:
        emo_bank = emo_bank.unsqueeze(1)  # (7, 1, D)
    targets = (torch.arange(1, 8, device=device, dtype=torch.float32) - 1.0) / 6.0

    # ── Load models ──────────────────────────────────────────────────────────
    # Style encoder: load pretrained weights; checkpoint will override with fine-tuned weights
    style_encoder = MelStyleEncoder(style_config)
    style_encoder.load_state_dict(torch.load(PRETRAINED_STYLE_PATH, map_location="cpu"))

    gen = Generator(config_synth)
    discrim = MultiPeriodDiscriminator()

    print(f"Loading synthesizer from: {CHECKPOINT_SYNTH}")
    synthesizer = SynthesizerLightningModule.load_from_checkpoint(
        CHECKPOINT_SYNTH,
        style_encoder=style_encoder,
        decoder=gen,
        discriminator=discrim,
        config=config_synth,
    ).to(device).eval()

    print(f"Loading LDM from: {CHECKPOINT_LDM}")
    ldm = DiffusionLightningModule.load_from_checkpoint(
        CHECKPOINT_LDM,
        style_encoder=style_encoder,
        config=config_ldm,
    ).to(device).eval()

    for p in synthesizer.parameters():
        p.requires_grad = False
    for p in ldm.parameters():
        p.requires_grad = False

    decoder = synthesizer.decoder.to(device).eval()
    dict_proj = synthesizer.dict.to(device).eval()

    # ── Metrics ──────────────────────────────────────────────────────────────
    mse_sum, mae_sum, n_sum = defaultdict(float), defaultdict(float), defaultdict(int)
    mse_global, mae_global, n_global = 0.0, 0.0, 0

    # ── Eval loop ────────────────────────────────────────────────────────────
    test_loader = test_create_data_loader(batch_size=1)

    for batch_idx, batch in enumerate(test_loader):
        if SAVE_AUDIO_LIMIT is not None and batch_idx >= SAVE_AUDIO_LIMIT:
            break

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        linguistic, _ = torch.nn.utils.rnn.pad_packed_sequence(
            batch["hubert"], batch_first=True
        )
        linguistic = linguistic.to(device)
        linguistic = dict_proj(linguistic).transpose(1, 2)

        K = 7
        linguistic_k = linguistic.repeat(K, 1, 1)
        speaker_k = batch["speaker_emb"].repeat(K, 1)

        with torch.inference_mode():
            style = ldm(emo_bank.float(), speaker_k.float())
            if style.ndim == 3:
                style = style.squeeze(1)

            emb = broadcast_embeddings(linguistic_k, speaker_k, style)
            y_hat = decoder(emb).squeeze(1).clamp(-1.0, 1.0)

            emo_pred = process_func(
                y_hat.detach().cpu().numpy(), device=device, embeddings=False
            )
            pred_ar = torch.tensor(emo_pred[:, 0], dtype=torch.float32, device=device)

        err = pred_ar - targets
        for i, c in enumerate(range(1, 8)):
            mse_sum[c] += float(err[i].pow(2).item())
            mae_sum[c] += float(err[i].abs().item())
            n_sum[c] += 1
            mse_global += float(err[i].pow(2).item())
            mae_global += float(err[i].abs().item())
            n_global += 1

            out_path = os.path.join(wav_root, f"class_{c}", f"utt_{batch_idx:06d}.wav")
            torchaudio.save(out_path, y_hat[i].detach().cpu().unsqueeze(0), sample_rate=sr)

            with open(meta_path, "a") as f:
                f.write(json.dumps({
                    "batch_idx": batch_idx, "class": c,
                    "wav_path": out_path,
                    "target_arousal": float(targets[i].item()),
                    "pred_arousal": float(pred_ar[i].item()),
                    "checkpoint_synth": CHECKPOINT_SYNTH,
                    "checkpoint_ldm": CHECKPOINT_LDM,
                }) + "\n")

        if batch_idx % 200 == 0 and batch_idx > 0:
            print(f"[{batch_idx}] utterances processed")

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n=== Arousal benchmark (Test1) ===")
    per_class_mae, per_class_mse = {}, {}
    for c in range(1, 8):
        n = max(1, n_sum[c])
        per_class_mae[c] = mae_sum[c] / n
        per_class_mse[c] = mse_sum[c] / n
        print(f"  Class {c}: MAE={per_class_mae[c]:.4f}  MSE={per_class_mse[c]:.4f}  N={n_sum[c]}")

    print(f"\n  Global:    MAE={mae_global/max(1,n_global):.4f}  MSE={mse_global/max(1,n_global):.4f}")
    print(f"  Macro avg: MAE={sum(per_class_mae.values())/7:.4f}  MSE={sum(per_class_mse.values())/7:.4f}")
    print(f"\nWAVs saved to: {wav_root}")
