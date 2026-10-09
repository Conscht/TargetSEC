"""TargetSEC base synthesizer.

Trains on MSP-Podcast Train, validates on Development (see src/dataset.py).
Test1 is held out for benchmarking.

Optimization settings match train_hifigan_baseline.py so the two systems differ
by method, not by training budget:
  lr 2e-4, batch 16, MPD(2,3,4,5,7,11) + MSD(3 scales), lambda_fm 2.
Pass --learning_rate 1e-4 --batch_size 8 to restore the previous settings.

    python train_synthesizer.py --seed 1234
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
# from src.train_synthesizer_distributed import SynthesizerLightningModule
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.decoder.decoder import Generator, CombinedDiscriminator

config = {
    "generator": {
        "input_dim": 768,  
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        "upsample_rates": [5,4,4,2,2],
        "upsample_initial_channel": 1024,   # increase the channels for feature extraction
        "upsample_kernel_sizes": [11,8,8,4,4],#"upsample_kernel_sizes": [16, 10, 8, 4] ,
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
            "learning_rate": 2e-4,
            "batch_size": 16,
        }
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234,
                   help="Seed; also names the run, checkpoint files and log dir.")
    p.add_argument("--resume", default=None,
                   help="Checkpoint to resume optimizer + epoch state from.")
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=config["training"]["batch_size"])
    p.add_argument("--learning_rate", type=float, default=config["training"]["learning_rate"])
    p.add_argument("--limit_val_batches", type=int, default=1000,
                   help="Fixed-size validation subset (val loader is unshuffled).")
    p.add_argument("--ser_loss", choices=["differentiable", "detached"],
                   default="detached",
                   help="Default 'detached': the CCC term is logged but carries no "
                        "gradient, so the decoder is never optimised through the "
                        "audeering model that also scores the benchmark. A live "
                        "gradient drives L_abs below the metric's own noise floor "
                        "(0.098 for SER vs human labels on real speech).")
    p.add_argument("--unfreeze_style", action="store_true",
                   help="Train the style encoder in stage 1 too, collapsing the "
                        "frozen/unfrozen two-stage recipe into a single stage.")
    return p.parse_args()



def main():
    args = parse_args()
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    print(f"Seed: {args.seed}  batch_size: {args.batch_size}  lr: {args.learning_rate}")

    # Stage 1: the LibriTTS-pretrained style encoder is held FROZEN and only the
    # decoder trains against it. train_synthesizer_finetune.py then unfreezes it
    # at 0.1x LR. Pass --unfreeze_style to collapse both stages into one.
    pretrained_style_encoder = MelStyleEncoder(style_config)
    pretrained_style_encoder.load_state_dict(torch.load(
        "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/"
        "MSP-Podcast-1.10/pre-trained_models/pre-trained_style"))

    gen = Generator(config)
    # MPD(2,3,4,5,7,11) + MSD(3 scales) -- same discriminator as the baseline.
    discrim = CombinedDiscriminator()

    tag = "ser" if args.ser_loss == "differentiable" else "noser"
    base_name = generate_base_name(f"synthesizer_training_speakr-{tag}-seed{args.seed}")

    train_loader, val_loader = create_dataloaders(batch_size=config['training']['batch_size'])

    model = SynthesizerLightningModule(
        style_encoder=pretrained_style_encoder, decoder=gen, discriminator=discrim,
        config=config, freeze_style_encoder=not args.unfreeze_style,
        ser_loss_mode=args.ser_loss)
    print(f"Style encoder frozen: {model.freeze_style_encoder}")
    print(f"L_SER mode: {args.ser_loss}")


    logger = setup_logger("logs_synthesizer", base_name)
    callbacks = setup_callbacks("checkpoints_synthesizer", base_name)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        min_epochs=10,
        accelerator="gpu",
        devices=num_gpus,
        precision="32",  # => when on 16, style encoder gives Nan values
        strategy="auto",
        callbacks=callbacks,
        # Development has 8376 cached mels and the val loader runs at batch
        # size 1, so a full pass would cost more than a training epoch. The val
        # loader is unshuffled, so this is a fixed, deterministic subset.
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


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

    # Dense periodic grid -- see train_hifigan_baseline.py: WVMOS is not logged
    # during training and no monitored metric tracks it, so the best-sounding
    # checkpoint can be evicted before it is ever measured.
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-periodic-{{epoch}}",
        every_n_epochs=10,
        save_top_k=-1,
        verbose=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=300,
        verbose=True,
        mode='min'
    )

    return [checkpoint_callback, latest_checkpoint_callback, periodic_checkpoint, early_stopping_callback]

# def save_model(model, model_folder, base_name):
#     if not os.path.exists(model_folder):
#         os.makedirs(model_folder)
#     model_filename = f"{base_name}.pth"
#     model_filepath = os.path.join(model_folder, model_filename)
#     torch.save(model.state_dict(), model_filepath)
#     print(f"Model saved to {model_filepath}.")

if __name__ == "__main__":
    main()

