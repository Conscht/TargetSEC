import torch
import torch.nn as nn

# --------- helpers ---------
class FixedEmbedding(nn.Module):
    """A single learnable 'null' token for classifier-free guidance during training."""
    def __init__(self, features: int):
        super().__init__()
        self.emb = nn.Embedding(1, features)
        nn.init.zeros_(self.emb.weight)

    def forward(self, B: int, T: int, device, dtype):
        # (B, T, C)
        v = self.emb.weight[0].to(device=device, dtype=dtype)
        return v.view(1, 1, -1).expand(B, T, -1)


def rescale_noise_cfg(noise_cfg: torch.Tensor, noise_text: torch.Tensor, k: float = 0.7):
    """
    Guidance-rescale: match the std of guided prediction to the conditional branch,
    then blend with factor k (0.7 by default).
    """
    dims = tuple(range(1, noise_cfg.ndim))
    std_text = noise_text.std(dim=dims, keepdim=True)
    std_cfg = noise_cfg.std(dim=dims, keepdim=True).clamp_min(1e-6)
    noise_rs = noise_cfg * (std_text / std_cfg)
    return k * noise_rs + (1.0 - k) * noise_cfg