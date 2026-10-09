import torch
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# IMPORTS
from processing.dataset_diffusion import create_dataloaders
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from Ablation.style_emo_mlp_module import StyleEmoLightningModule  # Your new MLP class

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--style_stats", required=True,
                help="Stats for the CURRENT stage-2 style encoder. Was hardcoded to "
                     "style_stats_new.pt, which belongs to the December (Test1-trained) "
                     "encoder and would silently reproduce the old ablation numbers.")
_p.add_argument("--use_speaker", type=int, choices=[0,1], default=None)
_p.add_argument("--max_epochs", type=int, default=300)
_p.add_argument("--synth_checkpoint", required=True,
                help="Stage-2 checkpoint whose style encoder produced --style_stats. "
                     "Must match, or the MLP regresses toward the wrong latent space.")
_a, _ = _p.parse_known_args()


# =========================================================
# Change this to False for "Emo-Only MLP", True for "Emo+Spk MLP"
# =========================================================
USE_SPEAKER_COND = bool(_a.use_speaker) if _a.use_speaker is not None else True 

# Paths (Adjust if needed)
STYLE_ENCODER_PATH = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
STYLE_STATS_PATH = _a.style_stats

def main():
    seed = 1234
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Starting MLP Ablation | Speaker Cond: {USE_SPEAKER_COND}")
    print(f"Number of GPUs: {num_gpus}")

    # Load Frozen Teacher
    pretrained_style_encoder = MelStyleEncoder(style_config)
    _ck = torch.load(_a.synth_checkpoint, map_location="cpu")["state_dict"]
    _se = {k[len("style_encoder."):]: v for k, v in _ck.items()
           if k.startswith("style_encoder.")}
    assert _se, f"no style_encoder.* keys in {_a.synth_checkpoint}"
    pretrained_style_encoder.load_state_dict(_se, strict=True)
    pretrained_style_encoder.eval()
    print(f"Style encoder from: {_a.synth_checkpoint} ({len(_se)} tensors)")

    # Config
    config = {
        "training": {
            "learning_rate": 1e-4, 
            "batch_size": 32,
        }
    }

    # Data
    train_loader, val_loader = create_dataloaders(batch_size=config['training']['batch_size'])

    # Initialize MLP Model
    model = StyleEmoLightningModule(
        style_encoder=pretrained_style_encoder,
        config=config,
        use_speaker_cond=USE_SPEAKER_COND,
        style_stats_path=STYLE_STATS_PATH
    )

    # Logging
    base_name = generate_base_name(USE_SPEAKER_COND)
    logger = TensorBoardLogger(save_dir="logs_ablation_mlp", name=base_name)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_ablation_mlp",
        filename=f"{base_name}-{{epoch:02d}}-{{val_loss:.3f}}",
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,  
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=_a.max_epochs, 
        accelerator="gpu",
        devices=num_gpus,
        precision=32,
        callbacks=[checkpoint_callback, early_stop],
        gradient_clip_val=0.5,
    )

    trainer.fit(model, train_loader, val_loader)


def generate_base_name(use_spk):
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    cond_str = "wSpk" if use_spk else "noSpk"
    return f"MLP_Baseline_{cond_str}_{timestamp}"

if __name__ == "__main__":
    main()