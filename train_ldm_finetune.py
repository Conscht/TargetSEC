"""
Retrain the LDM (speaker + emotion conditioned) using the fine-tuned
MelStyleEncoder as the frozen style teacher.

Run AFTER compute_style_stats_finetune.py has produced style_stats_finetune.pt.

Key differences from train2_diffusion.py:
  - Style encoder loaded from fine-tuned synthesizer checkpoint (not pretrained LibriTTS)
  - style_stats_path points to style_stats_finetune.pt
  - USE_SPEAKER_COND = True (matches the production LDM architecture)
  - Saves to checkpoints_ldm_finetune/

Usage:
    python train_ldm_finetune.py [--resume <ckpt>]
"""
import argparse
import os
import sys
import torch
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from processing.dataset_diffusion import create_dataloaders
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.diffusion_module_fixed import DiffusionLightningModule

sys.path.append(os.getcwd())

FINETUNE_CHECKPOINT = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_synthesizer_finetune/"
    "synthesizer_finetune-06-13_18-22-18-epoch=48-val_loss=16.24.ckpt"
)
STYLE_STATS_PATH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "style_stats_finetune.pt"
)


def load_finetuned_style_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    style_keys = {k[len("style_encoder."):]: v
                  for k, v in state.items() if k.startswith("style_encoder.")}
    if not style_keys:
        raise RuntimeError("No style_encoder.* keys found in checkpoint.")
    encoder = MelStyleEncoder(style_config)
    encoder.load_state_dict(style_keys, strict=True)
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="LDM checkpoint to resume from")
    args = parser.parse_args()

    if not os.path.exists(STYLE_STATS_PATH):
        raise FileNotFoundError(
            f"Style stats not found: {STYLE_STATS_PATH}\n"
            "Run compute_style_stats_finetune.py first."
        )

    pl.seed_everything(1234, workers=True)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"GPUs: {num_gpus}")
    print(f"Style stats: {STYLE_STATS_PATH}")
    print(f"Style encoder: {FINETUNE_CHECKPOINT}")

    style_encoder = load_finetuned_style_encoder(FINETUNE_CHECKPOINT)
    style_encoder.eval()

    config = {
        "training": {
            "learning_rate": 3e-5,
            "batch_size": 32,
            "cfg_prob": 0.3,
            "warmup_steps": 55_000,
        },
        "inference": {"guidance_scale": 4.0, "guidance_rescale": 0.7},
        "cross_attention_dim": 768,
    }

    train_loader, val_loader = create_dataloaders(batch_size=config["training"]["batch_size"])

    model = DiffusionLightningModule(
        style_encoder=style_encoder,
        config=config,
        use_speaker_cond=True,
        style_stats_path=STYLE_STATS_PATH,
    )

    base_name = f"ldm_finetune-{datetime.now().strftime('%m-%d_%H-%M-%S')}"
    os.makedirs("checkpoints_ldm_finetune", exist_ok=True)
    os.makedirs("logs_ldm_finetune", exist_ok=True)

    logger = TensorBoardLogger(save_dir="logs_ldm_finetune", name=base_name)

    checkpoint_best = ModelCheckpoint(
        dirpath="checkpoints_ldm_finetune",
        filename=f"{base_name}-{{epoch:03d}}-{{val_loss:.4f}}",
        save_top_k=4,
        monitor="val_loss",
        mode="min",
        verbose=True,
    )
    checkpoint_latest = ModelCheckpoint(
        dirpath="checkpoints_ldm_finetune",
        filename=f"{base_name}-latest",
        save_top_k=1,
        monitor="epoch",
        mode="max",
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=200,
        mode="min",
        verbose=True,
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=600,
        accelerator="gpu",
        devices=num_gpus,
        precision=32,
        strategy="auto",
        gradient_clip_val=0.5,
        callbacks=[checkpoint_best, checkpoint_latest, early_stop],
    )

    ckpt_path = args.resume if args.resume else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
