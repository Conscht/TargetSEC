import torch
import os
import sys

# --- PATH HACK (Ensures Python finds your src folder) ---
sys.path.append(os.getcwd())

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Imports
from processing.dataset_diffusion import create_dataloaders
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config

# ---------------------------------------------------------
# CRITICAL CHANGE: Import from your NEW Ablation Module
# ---------------------------------------------------------
from src.diffusion_module_fixed import DiffusionLightningModule

# =========================================================
# ABLATION CONTROL
# Set False to train the "Emo Only" version (Zero-Masking)
# Set True to train the "Speaker + Emo" version
# =========================================================
USE_SPEAKER_COND = False 

# Paths
STYLE_ENCODER_PATH = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
CHECKPOINT_PATH = None  # Add path if resuming training

def main():
    seed = 1234
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    print(f"Training Mode: {'SPEAKER + EMO' if USE_SPEAKER_COND else 'EMO ONLY (Zero Masked)'}")

    # Generate unique name based on ablation setting
    base_name = generate_base_name(USE_SPEAKER_COND)

    # Initialize the pretrained style encoder (Frozen Teacher)
    pretrained_style_encoder = MelStyleEncoder(style_config)
    pretrained_style_encoder.load_state_dict(torch.load(STYLE_ENCODER_PATH))
    pretrained_style_encoder.eval()

    config = {
        "training": {
            "learning_rate": 3e-5,
            "batch_size": 32,
            "cfg_prob": 0.3,
        },
        "inference": {"guidance_scale": 3.0, "guidance_rescale": 0.7},
        
        # -------------------------------------------------------
        # NOTE: This must match the ablation module logic 
        # (512 emo + 256 spk = 768)
        # -------------------------------------------------------
        "cross_attention_dim": 768, 
    }

    train_loader, val_loader = create_dataloaders(batch_size=config['training']['batch_size'])

    # Initialize Model with Ablation Flag
    model = DiffusionLightningModule(
        style_encoder=pretrained_style_encoder, 
        config=config,
        use_speaker_cond=USE_SPEAKER_COND, # <--- Passes the flag here
        style_stats_path="/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/style_stats_new.pt"
    )

    logger = setup_logger("logs_ablation_ldm", base_name)
    callbacks = setup_callbacks("checkpoints_ablation_ldm", base_name, val_loader)

    trainer = Trainer(
        logger=logger,
        max_epochs=600,
        accelerator="gpu",
        devices=num_gpus,
        precision=32,
        callbacks=callbacks,
        gradient_clip_val=0.5,
    )

    if CHECKPOINT_PATH is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=CHECKPOINT_PATH)
    else:
        trainer.fit(model, train_loader, val_loader)

def generate_base_name(use_spk):
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    cond_str = "wSpk" if use_spk else "noSpk"
    return f"LDM_Ablation_{cond_str}_{timestamp}"

def setup_logger(log_folder, base_name):
    return TensorBoardLogger(save_dir=log_folder, name=base_name)

def setup_callbacks(checkpoint_folder, base_name, val_loader):
    checkpoint_filename = f"{base_name}-{{epoch:02d}}-{{val_loss:.2f}}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=checkpoint_filename,
        save_top_k=4,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-latest",
        save_top_k=1,
        verbose=True,
        monitor='epoch',
        mode='max',
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=200,
        verbose=True,
        mode='min'
    )

    return [checkpoint_callback, latest_checkpoint_callback, early_stopping_callback]

if __name__ == "__main__":
    main()