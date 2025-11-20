import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from processing.preprocessor import _stft  # Import your initialized STFT
import argparse

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mel_from_wav(audio, stft):
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)
    audio = torch.clip(audio.to(device), -1.0, 1.0)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    
    mel, _ = stft.mel_spectrogram(audio)
    mel = mel.squeeze(0).cpu()  # (n_mels, T)
    return mel


def process_and_save(audio_path, save_path, stft):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != stft.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=stft.sampling_rate)
        waveform = resampler(waveform)

    mel = get_mel_from_wav(waveform, stft)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mel, save_path)


def process_directory(audio_dir, output_dir, stft):
    audio_paths = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                relative = os.path.relpath(full_path, audio_dir)
                out_path = os.path.join(output_dir, os.path.splitext(relative)[0] + "_mel.pt")
                audio_paths.append((full_path, out_path))

    print(f"Found {len(audio_paths)} .wav files.")
    for audio_path, save_path in tqdm(audio_paths, desc="Processing audio"):
        process_and_save(audio_path, save_path, stft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Input directory with .wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save .pt files")
    args = parser.parse_args()

    process_directory(args.audio_dir, args.output_dir, _stft)
