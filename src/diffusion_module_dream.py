# diffusion_lightning_module.py
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.Diffusion.diffusion_dream import DreamVG_My


class DiffusionLightningModule(pl.LightningModule):
    """
    Lightning wrapper for DreamVG_My:
    - Takes your style encoder and dataset batch
    - Trains diffusion on normalized style latents, conditioned on emo+spk embeddings
    """

    def __init__(
        self,
        style_encoder: nn.Module,
        config: dict,
        unet_model_config_path: str = "config/diffusion_model_config2.json",
        style_stats_path: str = "style_stats.pt",
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # -----------------------------------------------------------------
        # Load style stats for normalization
        # -----------------------------------------------------------------
        stats = torch.load(style_stats_path, map_location="cpu")
        style_mean = stats["mean"]  # (C,)
        style_std = stats["std"]    # (C,)

        # -----------------------------------------------------------------
        # DreamVG-style diffusion core
        # -----------------------------------------------------------------
        self.diffusion_model = DreamVG_My(
            unet_config_path=unet_model_config_path,
            style_mean=style_mean,
            style_std=style_std,
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            cfg_prob=self.config["training"].get("cfg_prob", 0.1),
        )

        # -----------------------------------------------------------------
        # Embedding MLPs (as you had them)
        # -----------------------------------------------------------------
        self.emo_mlp = nn.Sequential(
            nn.Linear(1024, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        self.spk_mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        # -----------------------------------------------------------------
        # Style encoder (teacher) – frozen
        # -----------------------------------------------------------------
        self.style_encoder = style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False

        # Audio / mel params (if you need them elsewhere)
        self.sr = 16000
        self.mel_hop = 256
        self.segment_size = 125  # number of mel frames per segment

    # ---------------------------------------------------------------------
    # Helper: random aligned slices of mel & audio
    # ---------------------------------------------------------------------

    def forward(self, emotion_embedding: torch.Tensor,
                        speaker_embedding: torch.Tensor) -> torch.Tensor:
            """
            Map (emo_emb, spk_emb) -> style embedding for SSL:
            input:  emo  (B, 1024) or (B, 1, 1024)
                    spk  (B, 512)
            output: style_emb (B, 1, C)
            """

            # squeeze possible extra dims (B,1,D) -> (B,D)
            emotion_embedding = emotion_embedding.squeeze()
            speaker_embedding = speaker_embedding.squeeze()

            if emotion_embedding.dim() == 1:
                emotion_embedding = emotion_embedding.unsqueeze(0)
            if speaker_embedding.dim() == 1:
                speaker_embedding = speaker_embedding.unsqueeze(0)

            # Project to bottleneck
            emo_token = self.emo_mlp(emotion_embedding)   # (B, C)
            spk_token = self.spk_mlp(speaker_embedding)   # (B, C)

            # Build DreamVG-style cond: (B, 1, D_cond)
            cond = torch.cat([emo_token, spk_token], dim=1).unsqueeze(1)

            # Run diffusion **inference**
            style = self.diffusion_model.inference(
                cond,
                guidance_scale=self.config["inference"].get("guidance_scale", 3.0),
                guidance_rescale=self.config["inference"].get("guidance_rescale", 0.7),
                ddim_steps=self.config["inference"].get("num_steps", 50),
                eta=1.0,
                random_seed=None,
            )  # (B, C)

            # SSL expects (B, 1, D)
            return style.unsqueeze(1)

    def _rand_mel_audio_slice(self, mel, mel_len, audio):
        """
        mel: (B, Tm, n_mels)
        mel_len: (B,) or list of valid Tm per example
        audio: (B, Ts)
        returns:
          mel_slice: (B, Tm_slice, n_mels)
          audio_slice: (B, Ts_slice)
        """
        B, Tm, n_mels = mel.shape
        mel_slices, audio_slices = [], []

        for i in range(B):
            valid = int(mel_len[i].item())
            seg_len = min(self.segment_size, valid)

            if valid <= seg_len:
                s_m = 0
                e_m = valid
            else:
                s_m = np.random.randint(0, valid - seg_len + 1)
                e_m = s_m + seg_len

            mel_seg = mel[i, s_m:e_m, :]  # (seg_len, n_mels)

            s_s = s_m * self.mel_hop
            e_s = min(e_m * self.mel_hop, audio.size(1))
            audio_seg = audio[i, s_s:e_s]  # (Ts_segment,)

            mel_slices.append(mel_seg)
            audio_slices.append(audio_seg)

        tgt_m = min(m.size(0) for m in mel_slices)
        mel_slice = torch.stack([m[:tgt_m] for m in mel_slices], dim=0)  # (B, tgt_m, n_mels)

        tgt_s = min(a.size(0) for a in audio_slices)
        audio_slice = torch.stack([a[:tgt_s] for a in audio_slices], dim=0)  # (B, tgt_s)

        return mel_slice, audio_slice

    # ---------------------------------------------------------------------
    # Helper: extract teacher style latents + conditioner
    # ---------------------------------------------------------------------

    def extract_latents_and_emotion(self, batch):
        """
        batch keys expected:
          'mel_spectrogram': (B, Tm, n_mels)
          'mel_original_lengths': (B,)
          'audio': (B, Ts)
          'speaker_emb': (B, 512)
          'emotion_emb': (B, 1, 1024) or (B, 1024)
        returns:
          style_latents: (B, C)  flat style latents (before normalization)
          cond: (B, 1, 256)      emo+spk condition
        """

        mel = batch["mel_spectrogram"]       # (B, Tm, n_mels)
        audio = batch["audio"]               # (B, Ts)
        mel_len = batch["mel_original_lengths"]
        speaker = batch["speaker_emb"]       # (B, 512)
        emo = batch["emotion_emb"]           # (B, 1, 1024) or (B, 1024)

        # 1) random aligned slices
        mel_slice, audio_slice = self._rand_mel_audio_slice(mel, mel_len, audio)

        # 2) style latents from style encoder (teacher)
        with torch.no_grad():
            style = self.style_encoder(mel_slice)  # (B, C=128)
        style_latents = style  # (B, C)

        # 3) emotion + speaker condition
        emo = emo.squeeze(1) if emo.dim() == 3 else emo  # (B, 1024)
        emo_token = self.emo_mlp(emo)                    # (B, 128)
        spk_token = self.spk_mlp(speaker)                # (B, 128)

        cond = torch.cat([emo_token, spk_token], dim=1).unsqueeze(1)  # (B, 1, 256)

        return style_latents.to(cond.device), cond

    # ---------------------------------------------------------------------
    # Training / validation steps
    # ---------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log("train_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None

        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        style_latents, cond = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(style_latents, cond, validation_mode=False)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cosine_sim", cosine_sim, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            self.log("val_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None

        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        style_latents, cond = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(style_latents, cond, validation_mode=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cosine_sim", cosine_sim, on_step=False, on_epoch=True, prog_bar=True)

        # optional: log norm of target style latents
        with torch.no_grad():
            self.log(
                "val_style_norm_target",
                style_latents.norm(dim=1).mean(),
                prog_bar=False,
            )

        return loss

    # ---------------------------------------------------------------------
    # Inference helper (generate style latents from emo+spk)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def sample_styles(
        self,
        emo_emb: torch.Tensor,    # (B, 1024)
        spk_emb: torch.Tensor,    # (B, 512)
        guidance_scale: float = 5.0,
        ddim_steps: int = 50,
        random_seed: int = 2023,
    ) -> torch.Tensor:
        """
        Convenience function:
        - Build cond from emo+spk
        - Run diffusion_model.inference
        returns style samples: (B, C)
        """

        emo_token = self.emo_mlp(emo_emb)     # (B, 128)
        spk_token = self.spk_mlp(spk_emb)     # (B, 128)
        cond = torch.cat([emo_token, spk_token], dim=1).unsqueeze(1)  # (B,1,256)

        style_samples = self.diffusion_model.inference(
            cond,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
            ddim_steps=ddim_steps,
            eta=1.0,
            random_seed=random_seed,
        )
        return style_samples  # (B, C)

    # ---------------------------------------------------------------------
    # Optimizer / scheduler
    # ---------------------------------------------------------------------

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]

        params = list(self.diffusion_model.parameters()) + \
                 list(self.emo_mlp.parameters()) + \
                 list(self.spk_mlp.parameters())

        optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=0.01,
        )

        warmup_steps = self.config["training"].get("warmup_steps", 55000)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
