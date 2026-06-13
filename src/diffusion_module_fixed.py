import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from src.Diffusion.diffusion import AudioDiffusion
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import numpy as np


class DiffusionLightningModule(pl.LightningModule):
    def __init__(
        self,
        style_encoder,
        config,
        use_speaker_cond: bool = True,  # <-- NEW FLAG
        unet_model_config_path="config/diffusion_model_config2.json",
        pretrained_unet=None,
        style_stats_path="/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/style_stats_new.pt",
    ):
        super().__init__()

        self.config = config
        self.use_speaker_cond = use_speaker_cond

        self.diffusion_model = AudioDiffusion(
            unet_model_config_path=unet_model_config_path,
            unet_model_name=pretrained_unet,
            snr_gamma=None,
            cfg_prob=self.config["training"].get("cfg_prob", 0.3),
        )

        self.sr = 16000
        self.mel_hop = 256
        self.segment_size = 125

        # ------------------------------------------------------------------
        # FIXED ARCHITECTURE STRATEGY:
        # We keep the MLP sizes constant even if speaker is disabled.
        # This ensures the UNet always receives (B, 1, 768) context.
        # ------------------------------------------------------------------
        self.emo_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )
        self.spk_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        self.style_encoder = style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False

        stats = torch.load(style_stats_path, map_location="cpu")
        self.register_buffer("style_mean", stats["mean"].view(1, -1))  # (1,128)
        self.register_buffer("style_std",  stats["std"].view(1, -1))   # (1,128)

        self.save_hyperparameters(ignore=["style_encoder"])

    def _rand_mel_audio_slice(self, mel, mel_len, audio):
        B, Tm, n_mels = mel.shape
        mel_slices, audio_slices = [], []

        for i in range(B):
            valid = int(mel_len[i].item())
            seg_len = min(self.segment_size, valid)

            if valid <= seg_len:
                s_m, e_m = 0, valid
            else:
                s_m = np.random.randint(0, valid - seg_len + 1)
                e_m = s_m + seg_len

            mel_seg = mel[i, s_m:e_m, :]
            s_s = s_m * self.mel_hop
            e_s = min(e_m * self.mel_hop, audio.size(1))
            audio_seg = audio[i, s_s:e_s]

            mel_slices.append(mel_seg)
            audio_slices.append(audio_seg)

        tgt_m = min(m.size(0) for m in mel_slices)
        mel_slice = torch.stack([m[:tgt_m] for m in mel_slices], dim=0)

        tgt_s = min(a.size(0) for a in audio_slices)
        audio_slice = torch.stack([a[:tgt_s] for a in audio_slices], dim=0)
        return mel_slice, audio_slice

    def _build_condition(self, emo: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
        """
        Returns cond with FIXED shape (B,1,768) regardless of flag.
        Strategy: Project both, then ZERO OUT the speaker token if disabled.
        """
        emo = emo.squeeze()
        if emo.dim() == 1:
            emo = emo.unsqueeze(0)

        speaker = speaker.squeeze()
        if speaker.dim() == 1:
            speaker = speaker.unsqueeze(0)

        emo_token = self.emo_mlp(emo)           # (B,512)
        spk_token = self.spk_mlp(speaker)       # (B,256)

        # ABLATION LOGIC: Mask information without changing dimensions
        if not self.use_speaker_cond:
            spk_token = torch.zeros_like(spk_token)

        cond = torch.cat([emo_token, spk_token], dim=1).unsqueeze(1)  # (B,1,768)
        return cond

    def forward(self, emotion_embedding, speaker_embedding, seed=None) -> torch.Tensor:
        # Check for missing speaker input during inference
        if speaker_embedding is None:
             if self.use_speaker_cond:
                 raise ValueError("Model requires speaker embedding!")
             else:
                 # Create dummy if we are in 'No Speaker' mode anyway
                 speaker_embedding = torch.zeros((emotion_embedding.shape[0], 512), device=self.device)

        ldm_condition = self._build_condition(emotion_embedding, speaker_embedding)

        latents_norm = self.diffusion_model.inference(
            ldm_condition,
            num_steps=self.config["inference"].get("num_steps", 100),
            guidance_scale=self.config["inference"].get("guidance_scale", 3.0),
            guidance_rescale_k=self.config["inference"].get("guidance_rescale_k", 0.7),
            seed=seed,
            disable_progress=True,
        )

        B = ldm_condition.shape[0]
        latents_norm = latents_norm.view(B, -1)  # (B,128)

        # denormalize back to original style space
        style = latents_norm * self.style_std + self.style_mean  # (B,128)
        return style

    def extract_latents_and_emotion(self, batch):
        mel     = batch["mel_spectrogram"]
        audio   = batch["audio"]
        mel_len = batch["mel_original_lengths"]
        speaker = batch["speaker_emb"]  # (B,512)

        # 1) random aligned slices
        mel_slice, _ = self._rand_mel_audio_slice(mel, mel_len, audio)

        # 2) style target from mel slice (teacher)
        with torch.no_grad():
            style = self.style_encoder(mel_slice)  # (B,128)

        # normalize into diffusion space
        style_norm = (style - self.style_mean) / (self.style_std + 1e-6)  # (B,128)
        true_latents = style_norm.unsqueeze(-1).unsqueeze(-1)             # (B,128,1,1)

        # 3) emotion embedding
        emo = batch["emotion_emb"].detach().squeeze(1)  # (B,1024)

        # 4) build conditioning
        cond = self._build_condition(emo, speaker)

        return true_latents.to(cond.device), cond

    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log("train_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None

        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        true_latents, cond = self.extract_latents_and_emotion(batch)

        loss, cosine_sim = self.diffusion_model(true_latents, cond)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cosine_sim", cosine_sim, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            self.log("val_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None

        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        true_latents, cond = self.extract_latents_and_emotion(batch)

        loss, cosine_sim = self.diffusion_model(true_latents, cond, validation_mode=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cosine_sim", cosine_sim, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            self.log(
                "val_style_norm_target",
                true_latents.view(true_latents.size(0), -1).norm(dim=1).mean(),
                prog_bar=False,
            )
        return loss

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay, no_decay = [], []
        for n, p in self.diffusion_model.named_parameters():
            if "null_token" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        
        # Add MLPs to optimizer
        decay.extend(self.emo_mlp.parameters())
        decay.extend(self.spk_mlp.parameters())

        optimizer = AdamW(
            [{"params": decay, "weight_decay": 0.01, "lr": lr},
             {"params": no_decay, "weight_decay": 0.0,  "lr": lr}]
        )

        warmup_steps = 55000
        scheduler = LambdaLR(
            optimizer,
            lambda s: s / max(1, warmup_steps) if s < warmup_steps else 1.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }