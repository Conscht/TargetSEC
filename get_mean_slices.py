
# compute_style_stats.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MelSpectrogramDataset, collate_fn
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== adjust paths ====
TENSOR_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Train"
AUDIO_DIR    = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/Audio"
META_TRAIN   = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/train.txt"
STYLE_CKPT   = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
# ======================

BATCH_SIZE   = 16
NUM_WORKERS  = 7

SEGMENT_SIZE = 125          # must match DiffusionLightningModule.segment_size
K_SLICES     = 8           # recommended: 4
STD_MIN      = 1e-3         # clamp to prevent exploding normalization
SEED         = 42

def collate_fn_stats(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    mels = [b["mel_spectrogram"] for b in batch]   # each is (T, 80), unpadded
    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)  # ✅ true mel lengths

    max_len_mel = int(mel_lengths.max().item())
    n_mels = mels[0].size(1)

    padded_mels = [
        torch.cat([m, m.new_zeros(max_len_mel - m.size(0), n_mels)], dim=0)
        if m.size(0) < max_len_mel else m
        for m in mels
    ]

    return {
        "mel_spectrogram": torch.stack(padded_mels),   # (B, Tm, 80)
        "mel_lengths": mel_lengths,                    # (B,)
    }


def rand_mel_slice(mel: torch.Tensor, mel_len: torch.Tensor, segment_size: int) -> torch.Tensor:
    """
    mel: (B, Tm, n_mels)
    mel_len: (B,)
    returns mel_slice: (B, seg_T, n_mels) with same seg_T for all in batch
    """
    B, Tm, C = mel.shape
    slices = []
    for i in range(B):
        valid = int(mel_len[i].item())
        seg_len = min(segment_size, valid)
        if valid <= seg_len:
            s = 0
        else:
            s = np.random.randint(0, valid - seg_len + 1)
        e = s + seg_len
        slices.append(mel[i, s:e, :])

    tgt = min(x.size(0) for x in slices)
    return torch.stack([x[:tgt] for x in slices], dim=0)

class WelfordStats:
    """Streaming per-dimension mean/std for vectors of shape (D,)"""
    def __init__(self, dim: int, device: str):
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, device=device)

    def update_batch(self, x: torch.Tensor):
        """x: (B, D)"""
        for i in range(x.size(0)):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            var = torch.zeros_like(self.mean)
        else:
            var = self.M2 / (self.n - 1)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        std = torch.clamp(std, min=STD_MIN)
        return self.mean, std, self.n

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = MelSpectrogramDataset(
        tensor_directory=TENSOR_DIR,
        embedding_file=META_TRAIN,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,               # important: randomizes utterance order
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_stats,
        pin_memory=True,
        drop_last=False,
    )

    style_encoder = MelStyleEncoder(style_config)
    style_encoder.load_state_dict(torch.load(STYLE_CKPT, map_location="cpu"))
    style_encoder.eval().to(DEVICE)

    stats = None

    pbar = tqdm(loader, desc="Computing style stats", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue

            mel = batch["mel_spectrogram"].to(DEVICE, non_blocking=True)
            mel_len = batch["mel_lengths"].to(DEVICE, non_blocking=True) 

            for _ in range(K_SLICES):
                mel_slice = rand_mel_slice(mel, mel_len, SEGMENT_SIZE)
                print("mel_slice shape:", mel_slice.shape)

                style = style_encoder(mel_slice)  # (B, D)

                if style.dim() != 2:
                    style = style.view(style.size(0), -1)

                if stats is None:
                    stats = WelfordStats(dim=style.size(1), device=DEVICE)

                stats.update_batch(style)

            pbar.set_postfix(samples=stats.n if stats else 0)

    if stats is None:
        raise RuntimeError("No valid batches processed; dataset/collate likely filtered everything out.")

    mean, std, n = stats.finalize()

    print(f"\n[style_stats] n={n}")
    print(f"mean[:10]={mean[:10].detach().cpu().numpy()}")
    print(f"std[:10] ={std[:10].detach().cpu().numpy()}")
    print(f"std(min/median/max)=({std.min().item():.6f}, {std.median().item():.6f}, {std.max().item():.6f})")

    torch.save(
        {"mean": mean.cpu(), "std": std.cpu(), "n": n, "segment_size": SEGMENT_SIZE, "k_slices": K_SLICES},
        "style_stats_k8.pt"
    )
    print("Saved style_stats.pt")

if __name__ == "__main__":
    main()
