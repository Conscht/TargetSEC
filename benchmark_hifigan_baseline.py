"""
Benchmark the HiFiGAN baseline synthesizer.

Full pipeline: scalar arousal class → emo_proj(1→128) → decoder → audio.
Matches [7]: emotion encoder = simple trainable linear layers on scalar arousal label.
No style encoder, no LDM. Generates 7 arousal variants per test utterance.

Run after training_hifigan_baseline.slurm has a good checkpoint.
"""
import os
import json
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import pytorch_lightning as pl

from src.synthesizer_hifigan_module import HiFiGANBaselineLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
from src.decoder.decoder_modules import broadcast_embeddings
from src.emotion.emotion_encoder import process_func

CHECKPOINT = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_hifigan_baseline_scalar/"
    "hifigan_baseline_scalar-06-14_14-52-53-epoch=137-val_loss=19.69.ckpt"
)
EMOTION_EMBEDDING_DIR = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/avgclass_emo_embeds"
)
SAVE_ROOT = "eval_outputs/hifigan_baseline_scalar_epoch137"

config = {
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
    "training": {"learning_rate": 1e-4, "batch_size": 8},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_emotion_embeddings(embedding_dir, split="Test1"):
    out = {}
    split_path = os.path.join(embedding_dir, split)
    for c in range(1, 8):
        out[c] = np.load(os.path.join(split_path, f"{c}.npy"))
    return out


if __name__ == "__main__":
    pl.seed_everything(1234)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    meta_path = os.path.join(SAVE_ROOT, "metadata.jsonl")
    wav_root = os.path.join(SAVE_ROOT, "wav")
    for c in range(1, 8):
        os.makedirs(os.path.join(wav_root, f"class_{c}"), exist_ok=True)

    sr = config["data"]["sampling_rate"]

    # Scalar class values normalised to [0,1] — same range used during training
    targets = (torch.arange(1, 8, device=device, dtype=torch.float32) - 1.0) / 6.0  # (7,)

    gen = Generator(config)
    discrim = MultiPeriodDiscriminator()

    print(f"Loading HiFiGAN baseline from: {CHECKPOINT}")
    model = HiFiGANBaselineLightningModule.load_from_checkpoint(
        CHECKPOINT,
        decoder=gen,
        discriminator=discrim,
        config=config,
    ).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    decoder   = model.decoder.to(device).eval()
    dict_proj = model.dict.to(device).eval()
    emo_proj  = model.emo_proj.to(device).eval()

    mse_sum, mae_sum, n_sum = defaultdict(float), defaultdict(float), defaultdict(int)
    mse_global, mae_global, n_global = 0.0, 0.0, 0

    test_loader = test_create_data_loader(batch_size=1)

    for batch_idx, batch in enumerate(test_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        linguistic, _ = torch.nn.utils.rnn.pad_packed_sequence(
            batch["hubert"], batch_first=True
        )
        linguistic = dict_proj(linguistic.to(device)).transpose(1, 2)

        K = 7
        linguistic_k = linguistic.repeat(K, 1, 1)
        speaker_k    = batch["speaker_emb"].repeat(K, 1)

        with torch.inference_mode():
            style_k = emo_proj(targets.unsqueeze(1))   # (7, 1) → (7, 128)
            emb   = broadcast_embeddings(linguistic_k, speaker_k, style_k)
            y_hat = decoder(emb).squeeze(1).clamp(-1.0, 1.0)

            emo_pred = process_func(
                y_hat.detach().cpu().numpy(), device=device, embeddings=False
            )
            pred_ar = torch.tensor(emo_pred[:, 0], dtype=torch.float32, device=device)

        err = pred_ar - targets
        for i, c in enumerate(range(1, 8)):
            mse_sum[c] += float(err[i].pow(2).item())
            mae_sum[c] += float(err[i].abs().item())
            n_sum[c]   += 1
            mse_global += float(err[i].pow(2).item())
            mae_global += float(err[i].abs().item())
            n_global   += 1

            out_path = os.path.join(wav_root, f"class_{c}", f"utt_{batch_idx:06d}.wav")
            torchaudio.save(out_path, y_hat[i].detach().cpu().unsqueeze(0), sample_rate=sr)

            with open(meta_path, "a") as f:
                f.write(json.dumps({
                    "batch_idx": batch_idx, "class": c,
                    "wav_path": out_path,
                    "target_arousal": float(targets[i].item()),
                    "pred_arousal": float(pred_ar[i].item()),
                    "checkpoint": CHECKPOINT,
                }) + "\n")

        if batch_idx % 200 == 0 and batch_idx > 0:
            print(f"[{batch_idx}] utterances processed")

    print("\n=== Arousal benchmark — HiFiGAN baseline (Test1) ===")
    per_class_mae, per_class_mse = {}, {}
    for c in range(1, 8):
        n = max(1, n_sum[c])
        per_class_mae[c] = mae_sum[c] / n
        per_class_mse[c] = mse_sum[c] / n
        print(f"  Class {c}: MAE={per_class_mae[c]:.4f}  MSE={per_class_mse[c]:.4f}  N={n_sum[c]}")

    print(f"\n  Global:    MAE={mae_global/max(1,n_global):.4f}  MSE={mse_global/max(1,n_global):.4f}")
    print(f"  Macro avg: MAE={sum(per_class_mae.values())/7:.4f}  MSE={sum(per_class_mse.values())/7:.4f}")
    print(f"\nWAVs saved to: {wav_root}")
