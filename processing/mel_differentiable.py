"""Differentiable mel-spectrogram front-end.

`processing.stft.TacotronSTFT` cannot be used inside a loss: `STFT.transform`
hard-codes `.cuda()`/`.cpu()`, `TacotronSTFT.mel_spectrogram` calls
`magnitudes.data`, and `preprocessor.get_mel_from_wav` returns a NumPy array.
Any L1 built on it is a constant with respect to the generator.

This module reimplements exactly the same transform with `torch.stft`, keeping
the graph intact, so `L_recon` actually trains the decoder.

Numerically equivalent to `TacotronSTFT(1024, 256, 1024, 80, 16000, 0.0, 8000.0)`:
  * reflect padding of n_fft//2 on both sides (`center=True`, `pad_mode="reflect"`)
  * periodic Hann window of `win_length`, zero-centre padded to `n_fft`
  * magnitude (not power) spectrogram
  * the identical `librosa.filters.mel` basis (librosa 0.9 positional API)
  * `log10(clamp(x, min=1e-5))` compression -- log10, NOT ln

That last point matters: every historical `val_loss` in this repo is
`45 * L1` under log10 compression. Switching to ln would silently rescale the
metric by ~2.3x and make new runs incomparable to old logs.

Verify with `python -m processing.verify_mel`.
"""
import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center
from scipy.signal import get_window

from config.stylespeech_model_config import style_config as _cfg


class DifferentiableMelSpectrogram(nn.Module):
    """Mel front-end that keeps gradients flowing back to the waveform."""

    def __init__(self,
                 n_fft=None, hop_length=None, win_length=None,
                 n_mels=None, sampling_rate=None, fmin=None, fmax=None,
                 clip_val=1e-5):
        super().__init__()
        # Default to the single front-end the rest of the repo uses, so the
        # mel loss is defined on the same features as the cached *_mel.pt files.
        self.n_fft = n_fft if n_fft is not None else _cfg.filter_length
        self.hop_length = hop_length if hop_length is not None else _cfg.hop_length
        self.win_length = win_length if win_length is not None else _cfg.win_length
        self.n_mels = n_mels if n_mels is not None else _cfg.n_mel_channels
        self.sampling_rate = sampling_rate if sampling_rate is not None else _cfg.sampling_rate
        self.fmin = fmin if fmin is not None else _cfg.mel_fmin
        self.fmax = fmax if fmax is not None else _cfg.mel_fmax
        self.clip_val = clip_val

        # Same positional call as processing/stft.py -> identical basis.
        mel_basis = librosa_mel_fn(
            self.sampling_rate, self.n_fft, self.n_mels, self.fmin, self.fmax)
        self.register_buffer(
            "mel_basis", torch.from_numpy(mel_basis).float(), persistent=False)

        # TacotronSTFT windows the Fourier basis with a win_length Hann window
        # zero-centre padded to n_fft; reproduce that padding explicitly rather
        # than relying on torch.stft's own centring.
        window = get_window("hann", self.win_length, fftbins=True)
        window = pad_center(window, self.n_fft)
        self.register_buffer(
            "window", torch.from_numpy(window).float(), persistent=False)

    def forward(self, y):
        """y: (B, T) or (T,) waveform in [-1, 1]  ->  (B, n_mels, frames)."""
        if y.dim() == 1:
            y = y.unsqueeze(0)

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,     # window is already padded to n_fft
            window=self.window.to(y.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # Magnitude spectrogram. The +1e-9 keeps the sqrt gradient finite at
        # exactly-zero bins, which do occur on silence-padded segments.
        magnitude = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

        mel = torch.matmul(self.mel_basis.to(y.device), magnitude)
        return torch.log10(torch.clamp(mel, min=self.clip_val))


# Module-level instance mirroring `processing.preprocessor._stft`, so callers
# can `from processing.mel_differentiable import mel_spectrogram_torch`.
_mel_fn = DifferentiableMelSpectrogram()


def mel_spectrogram_torch(y):
    """Differentiable drop-in for `get_mel_from_wav(y, _stft)`.

    Unlike `get_mel_from_wav` this returns a torch tensor on `y`'s device with
    the graph attached, and does not clip the input to [-1, 1] (the generator's
    tanh already bounds it, and clipping would zero the gradient at saturation).
    """
    return _mel_fn(y)
