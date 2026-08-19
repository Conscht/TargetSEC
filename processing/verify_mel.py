"""Verify the differentiable mel matches TacotronSTFT, and that it carries gradients.

Run:  python -m processing.verify_mel
Requires a GPU, because processing/stft.py hard-codes .cuda() in STFT.transform.
"""
import torch

from processing.mel_differentiable import mel_spectrogram_torch
from processing.preprocessor import get_mel_from_wav, _stft


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2.5 s at 16 kHz -- the training segment length.
    y = (torch.rand(4, 40000, device=device) * 2 - 1) * 0.5

    ref = torch.from_numpy(get_mel_from_wav(y, _stft.to(device))).to(device)
    new = mel_spectrogram_torch(y)

    assert ref.shape == new.shape, f"shape mismatch: {ref.shape} vs {new.shape}"
    max_abs = (ref - new).abs().max().item()
    print(f"shape           : {tuple(new.shape)}")
    print(f"max |ref - new| : {max_abs:.3e}")
    assert max_abs < 1e-3, f"mel mismatch too large: {max_abs}"

    # The whole point: gradients must reach the waveform.
    y_grad = y.clone().requires_grad_(True)
    mel_spectrogram_torch(y_grad).sum().backward()
    assert y_grad.grad is not None and torch.isfinite(y_grad.grad).all()
    assert y_grad.grad.abs().sum() > 0, "mel produced zero gradient"
    print(f"grad |dmel/dy|  : {y_grad.grad.abs().mean().item():.3e}  (non-zero, finite)")

    # And confirm the old path does NOT -- documents why this module exists.
    y_old = y.clone().requires_grad_(True)
    old = torch.from_numpy(get_mel_from_wav(y_old, _stft.to(device)))
    print(f"old path requires_grad: {old.requires_grad}  (expected False)")

    print("\nOK")


if __name__ == "__main__":
    main()
