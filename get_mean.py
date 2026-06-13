# compute_style_stats.py
import torch
from torch.utils.data import DataLoader
import numpy as np

from processing.dataset_diffusion import MelSpectrogramDataset, collate_fn
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pfade anpassen an dein Setup
TENSOR_DIR   = r"C:\Users\Conscht\Documents\New folder\mel_spectograms\Train"
AUDIO_DIR    = r"C:\Users\Conscht\Documents\New folder\Audio\Audio"
META_TRAIN   = r"C:\Users\Conscht\Documents\New folder\Audio\MSP-Podcast-1.10\hubert-km100\parsed_with_spkrEmbeds\train.txt"
EMO_DIR      = r"C:\Users\Conscht\Documents\New folder\emotion_embeddings"
BATCH_SIZE   = 8

def main():
    # 1) Dataset + Loader
    ds = MelSpectrogramDataset(
        tensor_directory=TENSOR_DIR,
        embedding_file=META_TRAIN,
        emo_dir=EMO_DIR,
        audio_dir=AUDIO_DIR,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2) Style Encoder laden
    pretrained_style_encoder = MelStyleEncoder(style_config)
    style_encoder = pretrained_style_encoder
    style_encoder.load_state_dict(torch.load("/Users/Conscht/Documents/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"))
    style_encoder.eval()
    style_encoder.to(DEVICE)
    style_encoder.eval()

    sum_vec    = None
    sum_sq_vec = None
    n_total    = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            mel = batch["mel_spectrogram"].to(DEVICE)   # (B, Tm, n_mels)

            # falls dein style_encoder (B, T, n_mels) erwartet, ist das schon OK
            style = style_encoder(mel)                  # (B, 128) – wie bei dir im Training

            if style.dim() > 2:
                style = style.view(style.size(0), -1)   # safety

            if sum_vec is None:
                dim = style.size(1)
                sum_vec    = torch.zeros(dim, device=DEVICE)
                sum_sq_vec = torch.zeros(dim, device=DEVICE)

            sum_vec    += style.sum(dim=0)
            sum_sq_vec += (style ** 2).sum(dim=0)
            n_total    += style.size(0)

    mean = (sum_vec / n_total).cpu()
    var  = (sum_sq_vec / n_total).cpu() - mean**2
    std  = torch.sqrt(torch.clamp(var, min=1e-8))

    print("mean:", mean[:10])
    print("std:", std[:10])

    torch.save({"mean": mean, "std": std, "n": n_total}, "style_stats.pt")
    print("saved to style_stats.pt")

if __name__ == "__main__":
    main()
