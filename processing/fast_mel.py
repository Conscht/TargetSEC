import torchaudio
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0
).to(device)

db_transform = torchaudio.transforms.AmplitudeToDB(stype='power').to(device)
