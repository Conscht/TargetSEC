"""The checks that would have caught the two bugs that invalidated an earlier
revision: a reconstruction loss and an emotion loss that were computed, logged,
and never backpropagated.

These need no GPU, no corpus and no network. Run from the repository root:

    PYTHONPATH="$PWD" pytest tests -q
"""
import torch
import pytest

from processing.mel_differentiable import mel_spectrogram_torch
from src.emotion.emotion_encoder import concordance_cc


def test_mel_is_differentiable():
    """The original bug: the mel path went through numpy, so 45 * L_recon --
    HiFi-GAN's dominant fidelity term -- contributed no gradient at all."""
    wav = torch.randn(2, 16000, requires_grad=True)
    mel = mel_spectrogram_torch(wav)
    assert mel.requires_grad, "mel lost the autograd graph"
    mel.sum().backward()
    assert wav.grad is not None and wav.grad.abs().sum() > 0


def test_mel_matches_the_cached_pipeline():
    """Cached *_mel.pt files were written by TacotronSTFT. If the torch mel
    disagrees with it, val_loss stops being comparable to every earlier run."""
    pytest.importorskip("librosa")
    from processing.preprocessor import get_mel_from_wav, _stft  # noqa: F401

    wav = torch.randn(1, 16000) * 0.1
    ours = mel_spectrogram_torch(wav)[0]                      # (80, frames)
    theirs = torch.as_tensor(get_mel_from_wav(wav, _stft))    # (80, frames)
    n = min(ours.shape[-1], theirs.shape[-1])
    diff = (ours[..., :n] - theirs[..., :n]).abs().max().item()
    assert diff < 1e-3, f"torch mel diverges from TacotronSTFT by {diff:.2e}" 


@pytest.mark.parametrize("n", [8, 64])
def test_ccc_is_one_for_identical_sequences(n):
    x = torch.randn(n)
    assert concordance_cc(x, x).item() == pytest.approx(1.0, abs=1e-5)


def test_ccc_matches_torchmetrics():
    tm = pytest.importorskip("torchmetrics")
    from torchmetrics import ConcordanceCorrCoef

    x, y = torch.randn(256), torch.randn(256)
    assert concordance_cc(x, y).item() == pytest.approx(
        ConcordanceCorrCoef()(x, y).item(), abs=1e-5
    )


def test_ccc_is_differentiable():
    x = torch.randn(32, requires_grad=True)
    concordance_cc(x, torch.randn(32)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
