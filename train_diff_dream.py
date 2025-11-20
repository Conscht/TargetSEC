import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from processing.dataset_diffusion import create_dataloaders
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config

# >>> use the new Lightning module <<<
from src.diffusion_module_dream import DiffusionLightningModule

# optional, if you actually use it somewhere
# from src.DiffusionCallback import AudioSampleCallback


# -------------------------------------------------------------------------
# Paths / checkpoints
# -------------------------------------------------------------------------

STYLE_ENCODER_CKPT = (
    "/Users/Conscht/Documents/New folder/Audio/MSP-Podcast-1.10/"
    "pre-trained_models/pre-trained_style"
)

# path to the diffusion UNet config (same as before)
UNET_CONFIG_PATH = "config/diffusion_model_config2.json"

# path to style stats (mean, std) you computed earlier
STYLE_STATS_PATH = "style_stats.pt"

# If you want to resume:
CHECKPOINT_PATH = None
# CHECKPOINT_PATH = r"C:\Users\Conscht\Documents\New folder\Code\EmoConv-LDM\checkpoints\diffusion_model_training-11-14_02-28-59-latest.ckpt"


# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------

def generate_base_name(log_name: str) -> str:
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"{log_name}-{timestamp}"


def setup_logger(log_folder: str, base_name: str) -> TensorBoardLogger:
    return TensorBoardLogger(save_dir=log_folder, name=base_name)


def setup_callbacks(checkpoint_folder: str, base_name: str):
    checkpoint_filename = f"{base_name}-{{epoch}}-{{val_loss:.4f}}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=checkpoint_filename,
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-latest",
        save_top_k=1,
        verbose=True,
        monitor="epoch",
        mode="max",
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=200,
        verbose=True,
        mode="min",
    )

    # you can add AudioSampleCallback here if you want
    # audio_callback = AudioSampleCallback(...)

    return [checkpoint_callback, latest_checkpoint_callback, early_stopping_callback]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    seed = 1234
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    base_name = generate_base_name("diffusion_model_training")

    # DreamVG-style config (same spirit as before)
    config = {
        "training": {
            "learning_rate": 3e-5,
            "batch_size": 32,
            "cfg_prob": 0.3,  # train-time CFG probability (like DreamVoice)
            "warmup_steps": 55_000,
        },
        "inference": {
            "guidance_scale": 3.0,
            "guidance_rescale": 0.7,
        },
        "cross_attention_dim": 256,  # emo(1024) + spk(512) -> 128+128
    }

    # ----------------------- Data -----------------------
    train_loader, val_loader = create_dataloaders(
        batch_size=config["training"]["batch_size"]
    )

    # ------------------ Style encoder -------------------
    pretrained_style_encoder = MelStyleEncoder(style_config)
    pretrained_style_encoder.load_state_dict(
        torch.load(STYLE_ENCODER_CKPT, map_location="cpu")
    )
    pretrained_style_encoder.eval()

    # ---------------------- Model -----------------------
    model = DiffusionLightningModule(
        style_encoder=pretrained_style_encoder,
        config=config,
        unet_model_config_path=UNET_CONFIG_PATH,
        style_stats_path=STYLE_STATS_PATH,
    )

    # ---------------------- Logging / callbacks ---------
    logger = setup_logger("logs", base_name)
    callbacks = setup_callbacks("checkpoints", base_name)

    # ---------------------- Trainer ---------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=150,
        min_epochs=10,
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else None,
        precision=32,            # keep 32bit since 16 caused NaNs in style encoder
        gradient_clip_val=0.5,
        callbacks=callbacks,
    )

    # ---------------------- Train -----------------------
    if CHECKPOINT_PATH is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=CHECKPOINT_PATH)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
