"""
Fine-tuning script: loads the best frozen-style-encoder checkpoint and
jointly fine-tunes the decoder + style encoder with lower LRs.

  decoder LR:       5e-5  (was 1e-4 during initial training)
  style encoder LR: 5e-6  (10x lower than decoder)
  discriminator LR: 5e-5

Weights are loaded from the checkpoint; optimizer state is NOT restored so
the new LRs take effect immediately.

Max 50 epochs, checkpoint saved every epoch (top-10 by val_loss kept).
"""
import argparse
import torch
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import MelSpectrogramDataset, collate_fn, create_dataloaders
from src.synthesizer_style_module import SynthesizerLightningModule
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.decoder.decoder import Generator, MultiPeriodDiscriminator

PRETRAINED_STYLE_PATH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
)
BEST_CHECKPOINT = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_synthesizer/"
    "synthesizer_training_speakr-12-14_15-51-55-epoch=122-val_loss=17.55.ckpt"
)

config = {
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
    "training": {
        "learning_rate": 5e-5,   # decoder LR; style encoder gets 5e-6 (x0.1)
        "batch_size": 8,
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Finetune checkpoint to resume from (restores full trainer state)")
    args = parser.parse_args()

    pl.seed_everything(1234)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Build model
    style_encoder = MelStyleEncoder(style_config)
    style_encoder.load_state_dict(torch.load(PRETRAINED_STYLE_PATH))
    style_encoder.train()

    gen = Generator(config)
    discrim = MultiPeriodDiscriminator()

    model = SynthesizerLightningModule(
        style_encoder=style_encoder,
        decoder=gen,
        discriminator=discrim,
        config=config,
    )

    ckpt_path = None
    if args.resume:
        # Resume: PL restores model weights + optimizer + epoch counter
        ckpt_path = args.resume
        print(f"Resuming from: {ckpt_path}")
    else:
        # Fresh finetune start: load weights only, fresh optimizer with new LRs
        print(f"Loading weights from: {BEST_CHECKPOINT}")
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
        print("Weights loaded successfully.")

    train_loader, val_loader = create_dataloaders(batch_size=config["training"]["batch_size"])

    base_name = f"synthesizer_finetune-{datetime.now().strftime('%m-%d_%H-%M-%S')}"

    logger = TensorBoardLogger(save_dir="logs_synthesizer_finetune", name=base_name)

    checkpoint_best = ModelCheckpoint(
        dirpath="checkpoints_synthesizer_finetune",
        filename=f"{base_name}-{{epoch}}-{{val_loss:.2f}}",
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        verbose=True,
    )
    checkpoint_latest = ModelCheckpoint(
        dirpath="checkpoints_synthesizer_finetune",
        filename=f"{base_name}-latest",
        save_top_k=1,
        monitor="epoch",
        mode="max",
        every_n_epochs=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=30,
        mode="min",
        verbose=True,
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=150,   # enough headroom for both fresh and resumed runs
        min_epochs=5,
        accelerator="gpu",
        devices=num_gpus,
        precision="32",
        strategy="auto",
        callbacks=[checkpoint_best, checkpoint_latest, early_stop],
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
