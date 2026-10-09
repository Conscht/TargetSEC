"""Gate for the HiFiGAN baseline fixes. Run before spending GPU-hours.

    python verify_hifigan_losses.py

Checks, in order:
  1. concordance_cc agrees with torchmetrics and is differentiable
  2. the frozen SER passes gradients through to its input waveform
  3. mel_loss and emo_loss both carry gradients into the generator
  4. the discriminator is MPD(6 periods) + MSD(3 scales) = 9 sub-discriminators
  5. the SER is absent from the checkpoint state_dict
  6. the dataloaders refuse to touch Test1

Item 3 is the one that matters: it is exactly the check that would have caught
`45 * mel_loss` and `L_SER` being dead constants.
"""
import torch
import torch.nn.functional as F

from src.decoder.decoder import Generator, CombinedDiscriminator, MultiPeriodDiscriminator
from src.decoder.decoder_modules import broadcast_embeddings
from src.emotion.emotion_encoder import DifferentiableSER, concordance_cc
from src.synthesizer_hifigan_module import (
    HiFiGANBaselineLightningModule, LAMBDA_RECON, LAMBDA_SER, LAMBDA_FM,
)
from processing.mel_differentiable import mel_spectrogram_torch

CONFIG = {
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
    "data": {"sampling_rate": 16000, "filter_length": 1024, "hop_length": 256,
             "win_length": 1024, "n_mel_channels": 80, "mel_fmin": 0.0, "mel_fmax": 8000.0},
    "training": {"learning_rate": 2e-4, "batch_size": 4},
}


def check_ccc():
    from torchmetrics.functional.regression import concordance_corrcoef
    torch.manual_seed(0)
    a = torch.rand(32)
    b = torch.rand(32)

    assert torch.isclose(concordance_cc(a, a), torch.tensor(1.0), atol=1e-5), \
        f"CCC(a, a) = {concordance_cc(a, a).item()}, expected 1.0"

    ours = concordance_cc(a, b)
    theirs = concordance_corrcoef(a, b)
    assert torch.isclose(ours, theirs, atol=1e-5), f"{ours.item()} vs {theirs.item()}"

    b_grad = b.clone().requires_grad_(True)
    concordance_cc(a, b_grad).backward()
    assert b_grad.grad is not None and b_grad.grad.abs().sum() > 0

    print(f"  CCC(a,a)={concordance_cc(a, a).item():.6f}  "
          f"CCC(a,b)={ours.item():.6f} (torchmetrics {theirs.item():.6f})  grad OK")


def check_ser(device):
    ser = DifferentiableSER().to(device)
    wav = ((torch.rand(2, 16000, device=device) * 2 - 1) * 0.3).requires_grad_(True)
    out = ser(wav)
    assert out.shape == (2, 3), f"expected (2,3) got {tuple(out.shape)}"
    out[:, 0].sum().backward()
    assert wav.grad is not None and wav.grad.abs().sum() > 0, "SER blocked the gradient"
    assert all(not p.requires_grad for p in ser.parameters()), "SER weights not frozen"
    assert not ser.training, "SER left training mode (dropout would perturb the loss)"

    ser.train()  # must be a no-op
    assert not ser.training, "SER.train() escaped eval mode"
    print(f"  SER out {tuple(out.shape)}  arousal={out[0, 0].item():.4f}  "
          f"grad |dA/dwav|={wav.grad.abs().mean().item():.3e}  frozen+eval OK")


def check_gradients(device):
    torch.manual_seed(0)
    gen = Generator(CONFIG)
    disc = CombinedDiscriminator()
    model = HiFiGANBaselineLightningModule(decoder=gen, discriminator=disc, config=CONFIG).to(device)

    B, T = 4, 125
    units = torch.randint(0, 100, (B, T), device=device)
    speaker = torch.randn(B, 512, device=device)
    arousal = torch.rand(B, device=device)

    x = model.dict(units).transpose(1, 2)
    style = model.emo_proj(arousal.unsqueeze(1))
    y_hat = model.decoder(broadcast_embeddings(x, speaker, style)).squeeze(1)
    y = (torch.rand(B, T * 320, device=device) * 2 - 1) * 0.3

    mel_loss = F.l1_loss(mel_spectrogram_torch(y_hat), mel_spectrogram_torch(y))
    pred_arousal = model.ser(y_hat)[:, 0]
    emo_loss = 1.0 - concordance_cc(arousal, pred_arousal)

    assert mel_loss.requires_grad, "mel_loss is a constant -- 45 * L_recon trains nothing"
    assert emo_loss.requires_grad, "emo_loss is a constant -- L_SER trains nothing"

    # Backprop ONLY the two terms under test, so a non-zero grad cannot be
    # attributed to the adversarial or feature-matching path.
    model.zero_grad(set_to_none=True)
    (LAMBDA_RECON * mel_loss + LAMBDA_SER * emo_loss).backward()

    probes = {
        "decoder.conv_pre": model.decoder.conv_pre.weight,
        "decoder.conv_post": model.decoder.conv_post.weight,
        "emo_proj[0]": model.emo_proj[0].weight,
        "dict": model.dict.weight,
    }
    for name, p in probes.items():
        assert p.grad is not None, f"{name}: grad is None"
        g = p.grad.abs().sum().item()
        assert g > 0, f"{name}: zero gradient"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"
        print(f"  {name:22s} |grad| = {g:.4e}")

    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in model._ser_holder[0].parameters()), "SER accumulated gradients"

    print(f"  mel_loss={mel_loss.item():.4f} (x{LAMBDA_RECON})  "
          f"emo_loss={emo_loss.item():.4f} (x{LAMBDA_SER})  lambda_fm={LAMBDA_FM}")
    return model


def check_discriminator(device):
    disc = CombinedDiscriminator().to(device)
    y = (torch.rand(2, 1, 8000, device=device) * 2 - 1) * 0.3
    y_hat = (torch.rand(2, 1, 8000, device=device) * 2 - 1) * 0.3
    y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)

    n_mpd = len(disc.mpd.discriminators)
    n_msd = len(disc.msd.discriminators)
    assert n_mpd == 6, f"expected 6 period sub-discriminators, got {n_mpd}"
    assert n_msd == 3, f"expected 3 scale sub-discriminators, got {n_msd}"
    assert len(y_d_r) == len(y_d_g) == len(fmap_r) == len(fmap_g) == 9

    old = MultiPeriodDiscriminator()
    print(f"  MPD periods={[d.period for d in disc.mpd.discriminators]}  MSD scales={n_msd}  "
          f"total={len(y_d_r)} (was {len(old.discriminators)})")


def check_checkpoint_size(model):
    sd = model.state_dict()
    ser_keys = [k for k in sd if k.startswith("_ser") or ".wav2vec2." in k]
    assert not ser_keys, f"SER leaked into state_dict ({len(ser_keys)} keys)"
    n = sum(v.numel() for v in sd.values())
    print(f"  state_dict: {len(sd)} tensors, {n/1e6:.1f}M params, no SER keys")


def check_no_test_leak():
    from src.dataset import (create_dataloaders_with_arousal, DEFAULT_TENSOR_DIR,
                             DEFAULT_VAL_TENSOR_DIR, DEFAULT_TEST_TENSOR_DIR)
    assert DEFAULT_TENSOR_DIR.endswith("/Train"), DEFAULT_TENSOR_DIR
    assert DEFAULT_VAL_TENSOR_DIR.endswith("/Development"), DEFAULT_VAL_TENSOR_DIR
    try:
        create_dataloaders_with_arousal(batch_size=2, tensor_dir=DEFAULT_TEST_TENSOR_DIR)
    except ValueError as e:
        print(f"  guard fired: {str(e)[:70]}...")
        return
    raise AssertionError("training on Test1 was NOT rejected")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    print("[1/6] concordance_cc");        check_ccc()
    print("[2/6] DifferentiableSER");     check_ser(device)
    print("[3/6] loss gradients");        model = check_gradients(device)
    print("[4/6] discriminator");         check_discriminator(device)
    print("[5/6] checkpoint contents");   check_checkpoint_size(model)
    print("[6/6] split guard");           check_no_test_leak()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
