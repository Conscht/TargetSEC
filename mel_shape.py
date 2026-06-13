import os, torch
from collections import Counter

mel_dir = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Train"  # adjust if needed
shapes = Counter()

for fn in os.listdir(mel_dir):
    if not fn.endswith("_mel.pt"):
        continue
    path = os.path.join(mel_dir, fn)
    mel = torch.load(path)
    if mel.ndim == 3 and mel.size(0) == 1:
        mel = mel.squeeze(0)
    shapes[(mel.shape[0], mel.shape[1])] += 1

print("Found shapes (n_mels, T) or (T, n_mels):")
for s, c in shapes.items():
    print(f"{s}: {c} files")
