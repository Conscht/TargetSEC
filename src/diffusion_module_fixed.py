import torch
import os
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from src.Diffusion.diffusion import AudioDiffusion
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import numpy as np

class DiffusionLightningModule(pl.LightningModule):
    def __init__(self, style_encoder, config,
                 unet_model_config_path="config/diffusion_model_config2.json",
                 pretrained_unet=None,
                 style_stats_path="style_stats.pt"):
        super(DiffusionLightningModule, self).__init__()

        self.config = config

        self.diffusion_model = AudioDiffusion(
            unet_model_config_path=unet_model_config_path,
            unet_model_name=pretrained_unet,
            snr_gamma=None,
            cfg_prob=self.config["training"].get("cfg_prob", 0.1)
        )

        self.sr      = 16000
        self.mel_hop = 256

        self.segment_size = 125
        self.emo_mlp = nn.Sequential(
            nn.Linear(1024, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.spk_mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.style_encoder = style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False

        # 🔹 NEW: load style mean / std and register as buffers
        stats = torch.load(style_stats_path, map_location="cpu")
        style_mean = stats["mean"].view(1, -1)   # (1, C)
        style_std  = stats["std"].view(1, -1)    # (1, C)
        self.register_buffer("style_mean", style_mean)
        self.register_buffer("style_std", style_std)

        self.save_hyperparameters(config)

    def _rand_mel_audio_slice(self, mel, mel_len, audio):
        B, Tm, n_mels = mel.shape
        mel_slices, audio_slices = [], []

        for i in range(B):
            valid = int(mel_len[i].item())

            # desired segment length in mel frames
            seg_len = min(self.segment_size, valid)  # self.segment_size ≈ 125

            if valid <= seg_len:
                s_m = 0
                e_m = valid
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



    def forward(self, emotion_embedding, speaker_embedding):
        emotion_embedding = emotion_embedding.squeeze()
        speaker_embedding = speaker_embedding.squeeze()

        if emotion_embedding.dim() == 1:
            emotion_embedding = emotion_embedding.unsqueeze(0)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        emo_token = self.emo_mlp(emotion_embedding)   # (B,128)
        spk_token = self.spk_mlp(speaker_embedding)   # (B,128)
        ldm_condition = torch.cat([emo_token, spk_token], dim=1).unsqueeze(1)  # (B,1,256)

        latents_norm = self.diffusion_model.inference(ldm_condition)  # (B, C, 1, 1) or (B,C)
        batchsize = ldm_condition.shape[0]
        latents_norm = latents_norm.view(batchsize, -1)               # (B, C)

        # 🔹 denormalize back to original style space
        style = latents_norm * self.style_std + self.style_mean       # (B, C)

        return style
    
    def check_for_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} contains NaN or Inf values.")

    def extract_latents_and_emotion(self, batch):
        mel     = batch['mel_spectrogram']              # (B, Tm, n_mels)
        audio   = batch['audio']                        # (B, Ts)
        mel_len = batch['mel_original_lengths']         # list[int]
        speaker = batch['speaker_emb']                  # (B,512)

        # 1) random, aligned slices
        mel_slice, audio_slice = self._rand_mel_audio_slice(mel, mel_len, audio)

        # 2) style target from mel slice (teacher)
        with torch.no_grad():
            style = self.style_encoder(mel_slice)        # (B,128)

        # 🔹 normalize into diffusion space
        style_norm = (style - self.style_mean) / (self.style_std + 1e-6)  # (B,128)

        true_latents = style_norm.unsqueeze(-1).unsqueeze(-1) 

        # 3) emotion condition (B,1024)
        emo = batch['emotion_emb'].detach().squeeze(1)  # (B,1024)
        # 4) build condition token (B,1,1536)
        emo_token = self.emo_mlp(emo)       # [B, C]
        spk_token = self.spk_mlp(speaker)       # [B, C]
        cond = torch.cat([emo_token, spk_token], dim=1).unsqueeze(1)

        return true_latents.to(cond.device), cond


    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log('train_loss', 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None 
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        true_latents, emotion = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(true_latents, emotion)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_cosine_sim', cosine_sim, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            self.log('val_loss', 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None  # Skip bad batch
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        
        true_latents, emotion = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(true_latents, emotion, validation_mode=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cosine_sim', cosine_sim, on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            # get UNet prediction at midpoint t for logging (already computed as part of loss if you want)
            self.log("val_style_norm_target", true_latents.view(true_latents.size(0), -1).norm(dim=1).mean(), prog_bar=False)

        return loss

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay, no_decay = [], []
        for n, p in self.diffusion_model.named_parameters():
            if "null_token" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = AdamW(
            [{"params": decay, "weight_decay": 0.01, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0,  "lr": lr}]
        )
        warmup_steps = 55000
        scheduler = LambdaLR(optimizer, lambda s: s / max(1, warmup_steps) if s < warmup_steps else 1.0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
