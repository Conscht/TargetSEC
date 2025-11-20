# dreamvg_like.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DConditionModel, DDIMScheduler
from einops import repeat
from copy import deepcopy
from typing import Any, Tuple
from torch import Tensor


# -------------------------------------------------------------------------
# Helpers (DreamVoice-style)
# -------------------------------------------------------------------------

def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


class FixedEmbedding(nn.Module):
    """
    Same idea as DreamVoice:
    Learn a single unconditional embedding vector and repeat it over (L) time.
    """

    def __init__(self, features: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(1, features)

    def forward(self, y: Tensor) -> Tensor:
        """
        y: (B, L, C)  -> we only use B and L to shape the unconditional token
        returns (B, L, C)
        """
        B, L, C = y.shape[0], y.shape[-2], y.shape[-1]
        device = y.device

        base = torch.zeros(B, device=device, dtype=torch.long)
        embed = self.embedding(base)              # (B, C)
        fixed_embedding = repeat(embed, "b c -> b l c", l=L)  # (B, L, C)
        return fixed_embedding


# -------------------------------------------------------------------------
# UNet wrapper (P2E_Cross-like) – but agnostic to the meaning of "prompt"
# -------------------------------------------------------------------------

class P2E_Cross_My(nn.Module):
    """
    DreamVoice-style P2E_Cross, adapted to your conditioning.

    - target: (B, C)  normalized style latents
    - prompt: (B, L, D_cond)  your emo+spk token(s)
    """

    def __init__(self, unet_config_path: str):
        super().__init__()

        # Load UNet config the same way diffusers does
        unet_config = UNet2DConditionModel.load_config(unet_config_path)
        self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")

        self.cfg_embedding = FixedEmbedding(self.unet.config.cross_attention_dim)

        # Map cond dim to cross_attention_dim if needed (assumes cond dim == cross_att_dim)
        self.context_embedding = nn.Sequential(
            nn.Linear(self.unet.config.cross_attention_dim,
                      self.unet.config.cross_attention_dim),
            nn.SiLU(),
            nn.Linear(self.unet.config.cross_attention_dim,
                      self.unet.config.cross_attention_dim),
        )

    def forward(
        self,
        target: Tensor,                # (B, C)
        t: Tensor,                     # (B,)
        prompt: Tensor,                # (B, L, D_cond == cross_att_dim)
        prompt_mask: Tensor = None,
        train_cfg: bool = False,
        cfg_prob: float = 0.0,
    ) -> Tensor:
        B, C = target.shape
        target_4d = target.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        if train_cfg and cfg_prob > 0.0:
            # Randomly replace the prompt by the unconditional embedding
            batch_mask = rand_bool((B, 1, 1), proba=cfg_prob, device=target.device)
            fixed_embedding = self.cfg_embedding(prompt).to(target_4d.dtype)
            prompt = torch.where(batch_mask, fixed_embedding, prompt)

        prompt = self.context_embedding(prompt)  # (B, L, cross_att_dim)

        # Make sure sample and encoder_hidden_states have same dtype
        target_4d = target_4d.to(prompt.dtype)

        out = self.unet(
            sample=target_4d,
            timestep=t,
            encoder_hidden_states=prompt,
            encoder_attention_mask=prompt_mask,
        )["sample"]           # (B, C, 1, 1)

        return out.squeeze(-1).squeeze(-1)      # (B, C)


# -------------------------------------------------------------------------
# DreamVG-style diffusion wrapper (for your style latents)
# -------------------------------------------------------------------------

class DreamVG_My(nn.Module):
    """
    DreamVoice / DreamVG-style DDIM diffusion, adapted to your setting.

    - Diffuses in a normalized style-latent space
    - Conditions on emo+spk embeddings
    """

    def __init__(
        self,
        unet_config_path: str,
        style_mean: Tensor,    # (C,)
        style_std: Tensor,     # (C,)
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cfg_prob: float = 0.1,
    ):
        super().__init__()

        self.model = P2E_Cross_My(unet_config_path)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            prediction_type="v_prediction",
            clip_sample=False,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )
        self.inference_scheduler = deepcopy(self.noise_scheduler)

        self.cfg_prob = cfg_prob

        # Store style normalization as buffers
        style_mean = style_mean.view(1, -1)  # (1, C)
        style_std = style_std.view(1, -1)    # (1, C)
        self.register_buffer("style_mean", style_mean)
        self.register_buffer("style_std", style_std)

    # ------------------------  Normalization helpers  ---------------------

    def _norm_style(self, x: Tensor) -> Tensor:
        # x: (B, C)
        return (x - self.style_mean) / (self.style_std + 1e-6)

    def _denorm_style(self, x: Tensor) -> Tensor:
        # x: (B, C)
        return x * self.style_std + self.style_mean

    # ------------------------  Training forward  --------------------------

    def forward(
        self,
        style_latents: Tensor,      # (B, C) unnormalized style from your encoder
        cond: Tensor,               # (B, L, D_cond) emo+spk token(s)
        validation_mode: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Training forward pass: returns (loss, cosine_similarity)
        """

        device = style_latents.device
        B, C = style_latents.shape

        # Normalize to diffusion space
        x0 = self._norm_style(style_latents)  # (B, C)

        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        if validation_mode:
            timesteps = torch.full(
                (B,), num_train_timesteps // 2, device=device, dtype=torch.long
            )
        else:
            timesteps = torch.randint(
                0, num_train_timesteps, (B,), device=device, dtype=torch.long
            )

        # Forward diffusion: add noise
        noise = torch.randn_like(x0)                      # (B, C)
        x0_4d = x0.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        noise_4d = noise.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)

        noisy_4d = self.noise_scheduler.add_noise(
            x0_4d, noise_4d, timesteps
        )                                                 # (B, C, 1, 1)
        noisy = noisy_4d.squeeze(-1).squeeze(-1)          # (B, C)

        # Target for v-prediction
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target_4d = self.noise_scheduler.get_velocity(
                x0_4d, noise_4d, timesteps
            )                                             # (B, C, 1, 1)
            target = target_4d.squeeze(-1).squeeze(-1)   # (B, C)
        else:
            target = noise

        # CFG training logic inside the model
        model_pred = self.model(
            target=noisy,
            t=timesteps,
            prompt=cond,
            prompt_mask=None,
            train_cfg=not validation_mode,
            cfg_prob=self.cfg_prob if not validation_mode else 0.0,
        )                                                 # (B, C)

        # Simple MSE loss like DreamVoice
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Cosine similarity for logging
        cosine_similarity = F.cosine_similarity(
            model_pred.float(), target.float(), dim=1
        ).mean()

        return loss, cosine_similarity

    # ------------------------  CFG rescale helper  ------------------------

    @staticmethod
    def _rescale_noise_cfg(
        cfg_pred: Tensor,
        text_pred: Tensor,
        guidance_rescale: float = 0.7,
    ) -> Tensor:
        """
        DreamVoice-style rescale_noise_cfg:
        - match std of cfg_pred to text_pred
        - then blend with text_pred
        """
        std_text = text_pred.std(dim=1, keepdim=True)
        std_cfg = cfg_pred.std(dim=1, keepdim=True) + 1e-6
        cfg_pred = cfg_pred * (std_text / std_cfg)
        return guidance_rescale * cfg_pred + (1.0 - guidance_rescale) * text_pred

    # ------------------------  Inference / Sampling  ----------------------

    @torch.no_grad()
    def inference(
        self,
        cond: Tensor,               # (B, L, D_cond)
        guidance_scale: float = 2.0,
        guidance_rescale: float = 0.7,
        ddim_steps: int = 50,
        eta: float = 1.0,
        random_seed: int = 2023,
    ) -> Tensor:
        """
        DreamVG-like sampling:
        returns denormalized style latents (B, C)
        """

        device = cond.device
        B, L, D = cond.shape

        self.model.eval()
        self.inference_scheduler.set_timesteps(ddim_steps, device=device)
        timesteps = self.inference_scheduler.timesteps

        # Random generator
        generator = torch.Generator(device=device)
        if random_seed is not None:
            generator = generator.manual_seed(random_seed)
        else:
            generator.seed()

        # Initialize latents in normalized style space
        C = self.style_mean.shape[1]
        latents = torch.randn((B, C), generator=generator, device=device)  # (B, C)

        for t in timesteps:
            t_batch = t.repeat(B)  # (B,)

            # Scale input
            latents_4d = latents.unsqueeze(-1).unsqueeze(-1)
            latents_scaled_4d = self.inference_scheduler.scale_model_input(
                latents_4d, t
            )
            latents_in = latents_scaled_4d.squeeze(-1).squeeze(-1)  # (B, C)

            # Conditional / unconditional predictions
            if guidance_scale:
                # conditional
                out_text = self.model(
                    target=latents_in,
                    t=t_batch,
                    prompt=cond,
                    prompt_mask=None,
                    train_cfg=False,
                )  # (B, C)

                # unconditional: use cfg_embedding only
                out_uncond = self.model(
                    target=latents_in,
                    t=t_batch,
                    prompt=cond,  # content ignored because train_cfg=True + cfg_prob=1
                    prompt_mask=None,
                    train_cfg=True,
                    cfg_prob=1.0,
                )  # (B, C)

                pred = out_uncond + guidance_scale * (out_text - out_uncond)

                if guidance_rescale > 0.0:
                    pred = self._rescale_noise_cfg(pred, out_text, guidance_rescale)
            else:
                pred = self.model(
                    target=latents_in,
                    t=t_batch,
                    prompt=cond,
                    prompt_mask=None,
                    train_cfg=False,
                )

            # Diffusion step
            pred_4d = pred.unsqueeze(-1).unsqueeze(-1)
            latents_4d = self.inference_scheduler.step(
                model_output=pred_4d,
                timestep=t,
                sample=latents_4d,
                eta=eta,
                generator=generator,
            ).prev_sample

            latents = latents_4d.squeeze(-1).squeeze(-1)  # (B, C)

        # Map back to your original style space
        style = self._denorm_style(latents)  # (B, C)
        return style
