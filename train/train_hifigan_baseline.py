"""
Training script for the HiFiGAN baseline synthesizer — annotated arousal label version.

Reimplementation of Prabhu, Lehmann-Willenbrock & Gerkmann, "In-the-wild Speech
Emotion Conversion Using Disentangled Self-Supervised Representations and Neural
Vocoder-based Resynthesis" (arXiv 2306.01916), Table 1 row `z_l + z_s + z_e + L_SER`.

Emotion encoder = small MLP on the annotated arousal label (EmoAct-1)/6 from
labels_consensus.csv, NOT SER-model regression outputs. This means inference with
(c-1)/6 is perfectly in-distribution.

Trains on MSP-Podcast Train (63,076 utts), validates on Development.
Test1 is held out for benchmark_hifigan_baseline.py.

Checkpoints saved to checkpoints_hifigan_baseline_annotated/.

Run one seed:
    python train_hifigan_baseline.py --seed 1234
"""
import argparse
import torch
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import MelSpectrogramDataset, collate_fn, create_dataloaders_with_arousal
from src.synthesizer_hifigan_module import HiFiGANBaselineLightningModule
from src.decoder.decoder import Generator, CombinedDiscriminator

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
        # HiFi-GAN / Polyak et al. reference values. The previous 1e-4 / 8 were
        # inherited from the TargetSEC synthesizer, not from the paper.
        # Batch size also matters for correctness: L_SER is a CCC, i.e. a batch
        # statistic, and is very noisy below ~16.
        "learning_rate": 2e-4,
        "batch_size": 16,
    },
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
                   default="differentiable",
                   help="'detached' logs L_SER without a gradient (conditioning "
                        "still comes from z_e + L_recon). Use it to avoid the "
                        "closed loop with the audeering model that also scores "
                        "the benchmark.")
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

    tag = "ser" if args.ser_loss == "differentiable" else "noser"
    base_name = generate_base_name(f"hifigan_baseline_annotated-{tag}-seed{args.seed}")
    train_loader, val_loader = create_dataloaders_with_arousal(
        batch_size=config["training"]["batch_size"]
    )

    gen = Generator(config)
    # MPD (periods 2,3,4,5,7,11) + MSD (3 scales), per sec. 3.2 of the paper.
    discrim = CombinedDiscriminator()

    model = HiFiGANBaselineLightningModule(decoder=gen, discriminator=discrim,
                                           config=config, ser_loss_mode=args.ser_loss)
    print(f"L_SER mode: {args.ser_loss}")

    logger = setup_logger("logs_hifigan_baseline_annotated", base_name)
    callbacks = setup_callbacks("checkpoints_hifigan_baseline_annotated", base_name)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        min_epochs=10,
        accelerator="gpu",
        devices=num_gpus,
        precision="32",
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
    # Two selection criteria on purpose. val_loss is 45 * mel L1, and across the
    # previous nine benchmarks it had no relationship to WVMOS (ep268 val 19.41
    # -> WVMOS 2.77, ep283 val 19.56 -> 2.05), so selecting on it alone samples
    # the reported metric close to at random. val_arousal_mae tracks the emotion
    # conversion objective directly.
    mel_ckpt = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-{{epoch}}-{{val_loss:.2f}}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    arousal_ckpt = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-bestmae-{{epoch}}-{{val_arousal_mae:.4f}}",
        save_top_k=3,
        verbose=True,
        monitor="val_arousal_mae",
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
    # Dense periodic grid, kept regardless of any monitored metric.
    #
    # WVMOS is not logged during training (it would mean a third large model
    # resident on the GPU every epoch), and neither val_loss nor
    # val_arousal_mae tracks it -- seed 1234 peaked at WVMOS 3.098 around epoch
    # 64 and had fallen to 2.944 by epoch 100, while both monitors stayed flat.
    # Selecting on them evicted the best-sounding checkpoint before it could be
    # measured. This keeps every 10th epoch so the peak can be found afterwards.
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-periodic-{{epoch}}",
        every_n_epochs=10,
        save_top_k=-1,   # keep them all
        verbose=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=300,
        verbose=True,
        mode="min",
    )
    return [mel_ckpt, arousal_ckpt, latest_checkpoint_callback, periodic_checkpoint,
            early_stopping_callback]


if __name__ == "__main__":
    main()
