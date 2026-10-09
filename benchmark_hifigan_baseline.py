"""
Benchmark the HiFiGAN baseline synthesizer.

Full pipeline: scalar arousal class → emo_proj(1→128) → decoder → audio.
Matches [7]: emotion encoder = simple trainable linear layers on scalar arousal label.
No style encoder, no LDM. Generates 7 arousal variants per test utterance.

Trained on annotated EmoAct labels via (EmoAct-1)/6, so inference with the
same (c-1)/6 scale is in-distribution by construction — no regression-head
workaround needed (unlike the earlier SER-output-trained checkpoint).

Run after training_hifigan_baseline.slurm has a good checkpoint.
"""
import os
import json
import argparse
from collections import defaultdict

import torch
import torchaudio
import pytorch_lightning as pl

from src.synthesizer_hifigan_module import HiFiGANBaselineLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
from src.decoder.decoder_modules import broadcast_embeddings
from src.emotion.emotion_encoder import process_func

_DEFAULT_CHECKPOINT = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_hifigan_baseline_annotated/"
    "hifigan_baseline_annotated-06-21_23-17-41-epoch=104-val_loss=20.00.ckpt"
)
_DEFAULT_SAVE_ROOT = "eval_outputs/hifigan_mlp_epoch104"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
parser.add_argument("--save_root", default=_DEFAULT_SAVE_ROOT)
parser.add_argument("--n_utts", type=int, default=0,
                    help="Evaluate only the first N utterances (0 = all of Test1). "
                         "NOTE: the loader is sorted by filename, so a truncated run "
                         "is NOT a random sample -- it is the earliest podcasts only. "
                         "Use for smoke tests, never for reported numbers.")
parser.add_argument("--split", choices=["test1", "dev"], default="test1",
                    help="'dev' selects checkpoints on Development so that "
                         "reporting on Test1 is not model selection on the test set.")
parser.add_argument("--sample_n", type=int, default=0,
                    help="Evaluate a RANDOM sample of N utterances (0 = all). Unlike "
                         "--n_utts this is unbiased, so it is safe for reported numbers. "
                         "Use it to get WVMOS without writing 20 GB of WAVs.")
parser.add_argument("--sample_seed", type=int, default=0,
                    help="Seed for --sample_n, so the subset is reproducible.")
parser.add_argument("--no_wav", action="store_true",
                    help="Skip writing WAVs. Arousal L_abs/L_mse only, ~an order of "
                         "magnitude faster and no 20+ GB of output. WVMOS needs the "
                         "WAVs, so omit this when you also want naturalness.")
args, _ = parser.parse_known_args()
CHECKPOINT = args.checkpoint
SAVE_ROOT = args.save_root

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


if __name__ == "__main__":
    pl.seed_everything(1234)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    meta_path = os.path.join(SAVE_ROOT, "metadata.jsonl")
    # Opened in append mode below, so a re-run into an existing save_root would
    # silently double every record and corrupt stat_test.py's per-utterance
    # grouping. Start clean.
    if os.path.exists(meta_path):
        os.remove(meta_path)
    wav_root = os.path.join(SAVE_ROOT, "wav")
    for c in range(1, 8):
        os.makedirs(os.path.join(wav_root, f"class_{c}"), exist_ok=True)

    sr = config["data"]["sampling_rate"]

    # Trained directly on (EmoAct-1)/6 annotated labels, so (c-1)/6 at inference
    # is in-distribution by construction — same scale used for evaluation targets.
    targets = (torch.arange(1, 8, device=device, dtype=torch.float32) - 1.0) / 6.0  # (7,)
    inference_scalars = targets.unsqueeze(1)  # (7, 1)
    print("Inference scalars ((c-1)/6):", inference_scalars.squeeze().tolist())

    gen = Generator(config)

    # Inference only needs decoder / dict / emo_proj. Loading with
    # discriminator=None and strict=False keeps the benchmark working across
    # discriminator changes (MultiPeriodDiscriminator vs CombinedDiscriminator)
    # instead of failing on unexpected discriminator.* keys.
    print(f"Loading HiFiGAN baseline from: {CHECKPOINT}")
    model = HiFiGANBaselineLightningModule.load_from_checkpoint(
        CHECKPOINT,
        decoder=gen,
        discriminator=None,
        config=config,
        strict=False,
    ).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    decoder   = model.decoder.to(device).eval()
    dict_proj = model.dict.to(device).eval()
    emo_proj  = model.emo_proj.to(device).eval()

    mse_sum, mae_sum, n_sum = defaultdict(float), defaultdict(float), defaultdict(int)
    mse_global, mae_global, n_global = 0.0, 0.0, 0

    test_loader = test_create_data_loader(batch_size=1, split=args.split)
    print(f'Evaluating on split: {args.split}')

    # Random subset. The loader is sorted by filename and unshuffled, so
    # batch_idx is the dataset index -- picking indices up front gives an
    # unbiased sample, unlike --n_utts which takes the earliest podcasts.
    keep = None
    if args.sample_n:
        n_total = len(test_loader)
        rng = __import__("random").Random(args.sample_seed)
        keep = set(rng.sample(range(n_total), min(args.sample_n, n_total)))
        print(f"Random sample: {len(keep)} of {n_total} utterances (seed {args.sample_seed})")

    for batch_idx, batch in enumerate(test_loader):
        if args.n_utts and batch_idx >= args.n_utts:
            break
        if keep is not None and batch_idx not in keep:
            continue
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
            style_k = emo_proj(inference_scalars)   # (7, 1) → (7, 128)
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
            if not args.no_wav:
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
    if args.no_wav:
        print("\nNo WAVs written (--no_wav); metrics only. Re-run without it for WVMOS.")
    else:
        print(f"\nWAVs saved to: {wav_root}")
