"""
Recompute style statistics using the fine-tuned MelStyleEncoder.

Loads style encoder weights from the fine-tuned synthesizer checkpoint,
runs it over the MSP-Podcast training set (K_SLICES random windows per
utterance, same segment_size as DiffusionLightningModule), and saves
mean/std to style_stats_finetune.pt.

Run this BEFORE training the new LDM:
    python compute_style_stats_finetune.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MelSpectrogramDataset, collate_fn
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TENSOR_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Train"
META_TRAIN = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/train.txt"
)
import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--finetune_checkpoint", required=True,
                help="Stage-2 checkpoint whose style encoder defines the LDM's "
                     "target latents. Was hardcoded to a June (Test1-trained) "
                     "checkpoint -- passing it explicitly stops the leak walking back in.")
_p.add_argument("--out", default=None)
_a, _ = _p.parse_known_args()
FINETUNE_CHECKPOINT = _a.finetune_checkpoint
OUT_PATH = _a.out or "style_stats_finetune.pt"

BATCH_SIZE  = 16
NUM_WORKERS = 7
SEGMENT_SIZE = 125   # must match DiffusionLightningModule.segment_size
K_SLICES    = 8
STD_MIN     = 1e-3
SEED        = 42


def collate_fn_stats(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    mels = [b["mel_spectrogram"] for b in batch]
    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)
    max_len = int(mel_lengths.max().item())
    n_mels = mels[0].size(1)
    padded = [
        torch.cat([m, m.new_zeros(max_len - m.size(0), n_mels)], dim=0)
        if m.size(0) < max_len else m
        for m in mels
    ]
    return {"mel_spectrogram": torch.stack(padded), "mel_lengths": mel_lengths}


def rand_mel_slice(mel, mel_len, segment_size):
    slices = []
    for i in range(mel.size(0)):
        valid = int(mel_len[i].item())
        seg = min(segment_size, valid)
        s = 0 if valid <= seg else np.random.randint(0, valid - seg + 1)
        slices.append(mel[i, s:s+seg, :])
    tgt = min(x.size(0) for x in slices)
    return torch.stack([x[:tgt] for x in slices], dim=0)


class WelfordStats:
    def __init__(self, dim, device):
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2   = torch.zeros(dim, device=device)

    def update_batch(self, x):
        for i in range(x.size(0)):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (x[i] - self.mean)

    def finalize(self):
        var = self.M2 / (self.n - 1) if self.n >= 2 else torch.zeros_like(self.mean)
        std = torch.sqrt(torch.clamp(var, min=1e-8)).clamp(min=STD_MIN)
        return self.mean, std, self.n


def load_finetuned_style_encoder(ckpt_path):
    """Extract style encoder weights from SynthesizerLightningModule checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    style_keys = {k[len("style_encoder."):]: v
                  for k, v in state.items() if k.startswith("style_encoder.")}
    if not style_keys:
        raise RuntimeError("No style_encoder.* keys found in checkpoint.")
    print(f"Extracted {len(style_keys)} style encoder weight tensors from checkpoint.")
    encoder = MelStyleEncoder(style_config)
    missing, unexpected = encoder.load_state_dict(style_keys, strict=True)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    return encoder


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading fine-tuned style encoder from:\n  {FINETUNE_CHECKPOINT}")
    style_encoder = load_finetuned_style_encoder(FINETUNE_CHECKPOINT)
    style_encoder.eval().to(DEVICE)
    for p in style_encoder.parameters():
        p.requires_grad = False

    ds = MelSpectrogramDataset(tensor_directory=TENSOR_DIR, embedding_file=META_TRAIN)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_stats,
        pin_memory=True,
        drop_last=False,
    )

    stats = None
    pbar = tqdm(loader, desc="Computing style stats (fine-tuned encoder)", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            mel     = batch["mel_spectrogram"].to(DEVICE, non_blocking=True)
            mel_len = batch["mel_lengths"].to(DEVICE, non_blocking=True)

            for _ in range(K_SLICES):
                mel_slice = rand_mel_slice(mel, mel_len, SEGMENT_SIZE)
                style = style_encoder(mel_slice)  # (B, D)
                if style.dim() != 2:
                    style = style.view(style.size(0), -1)
                if stats is None:
                    stats = WelfordStats(dim=style.size(1), device=DEVICE)
                stats.update_batch(style)

            if stats:
                pbar.set_postfix(samples=stats.n)

    if stats is None:
        raise RuntimeError("No valid batches — check dataset paths.")

    mean, std, n = stats.finalize()
    print(f"\n[style_stats_finetune] n={n}")
    print(f"  mean[:10] = {mean[:10].cpu().numpy()}")
    print(f"  std[:10]  = {std[:10].cpu().numpy()}")
    print(f"  std min/med/max = {std.min():.6f} / {std.median():.6f} / {std.max():.6f}")

    torch.save(
        {"mean": mean.cpu(), "std": std.cpu(), "n": n,
         "segment_size": SEGMENT_SIZE, "k_slices": K_SLICES,
         "source_checkpoint": FINETUNE_CHECKPOINT},
        OUT_PATH,
    )
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
