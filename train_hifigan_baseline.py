"""
Training script for the HiFiGAN baseline synthesizer.

This trains a synthesizer that injects emotion embeddings directly (no style
encoder), serving as the HiFiGAN [14] baseline for the human evaluation study.
Checkpoints are saved to checkpoints_hifigan_baseline/ to avoid overwriting
the TargetSEC synthesizer.
"""
import torch
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import MelSpectrogramDataset, collate_fn, create_dataloaders
from src.synthesizer_hifigan_module import HiFiGANBaselineLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator

checkpoint = None

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
        "learning_rate": 1e-4,
        "batch_size": 8,
    }
}

gen = Generator(config)
discrim = MultiPeriodDiscriminator()


def main():
    pl.seed_everything(1234)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    base_name = generate_base_name("hifigan_baseline")
    train_loader, val_loader = create_dataloaders(batch_size=config['training']['batch_size'])

    model = HiFiGANBaselineLightningModule(decoder=gen, discriminator=discrim, config=config)

    logger = setup_logger("logs_hifigan_baseline", base_name)
    callbacks = setup_callbacks("checkpoints_hifigan_baseline", base_name)

    trainer = Trainer(
        logger=logger,
        max_epochs=400,
        min_epochs=10,
        accelerator="gpu",
        devices=num_gpus,
        precision="32",
        strategy="auto",
        callbacks=callbacks,
    )

    if checkpoint is not None:
        trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, EarlyStopping)]
        trainer.should_stop = False
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)


def generate_base_name(log_name):
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"{log_name}-{timestamp}"

def setup_logger(log_folder, base_name):
    return TensorBoardLogger(save_dir=log_folder, name=base_name)

def setup_callbacks(checkpoint_folder, base_name):
    checkpoint_filename = f"{base_name}-{{epoch}}-{{val_loss:.2f}}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=checkpoint_filename,
        save_top_k=3,
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
        patience=300,
        verbose=True,
        mode='min'
    )
    return [checkpoint_callback, latest_checkpoint_callback, early_stopping_callback]


if __name__ == "__main__":
    main()
