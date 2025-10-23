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
    def __init__(self, style_encoder, config, unet_model_config_path="config/diffusion_model_config2.json", pretrained_unet=None):
        super(DiffusionLightningModule, self).__init__()

        self.config = config

        self.diffusion_model = AudioDiffusion(
            unet_model_config_path=unet_model_config_path,
            unet_model_name=pretrained_unet,
            snr_gamma=1,
            cfg_prob=self.config["training"].get("cfg_prob", 0.1)
        )
        

        self.sr      = 16000
        self.mel_hop = 256

        self.segment_size = 100    # 0.5 secs in Mel Spec Units: 32 * 256 = 8.192 (16kHz)
        self.embeddingTransform = nn.Sequential(
            nn.Linear(self.config['cross_attention_dim'],
                    self.config['cross_attention_dim']),
            nn.SiLU(),
            nn.Linear(self.config['cross_attention_dim'],
                    self.config['cross_attention_dim'])
        )

        self.style_encoder = style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False

        self.save_hyperparameters(config) 

    def _rand_mel_audio_slice(self, mel, mel_len, audio):
        B, Tm, n_mels = mel.shape
        mel_slices, audio_slices = [], []

        for i in range(B):
            valid = int(mel_len[i].item())            # <— wichtig
            if valid < 4:
                s_m, e_m = 0, valid
            else:
                seg_len = max(32, int(np.random.uniform(0.4, 0.8) * valid))
                seg_len = min(seg_len, valid)
                s_m = np.random.randint(0, max(1, valid - seg_len + 1))
                e_m = s_m + seg_len

            mel_seg = mel[i, s_m:e_m, :]              # (t, n_mels)
            s_s = s_m * self.mel_hop
            e_s = min(e_m * self.mel_hop, audio.size(1))
            audio_seg = audio[i, s_s:e_s]             # (t_samps,)

            mel_slices.append(mel_seg)
            audio_slices.append(audio_seg)

        tgt_m = min(m.size(0) for m in mel_slices)
        mel_slice = torch.stack([m[:tgt_m] for m in mel_slices], dim=0)

        tgt_s = min(a.size(0) for a in audio_slices)
        audio_slice = torch.stack([a[:tgt_s] for a in audio_slices], dim=0)
        return mel_slice, audio_slice


    def forward(self, emotion_embedding, speaker_embedding):
        # emotion_embedding = F.normalize(emotion_embedding, p=2, dim=-1)
        # speaker_embedding = F.normalize(speaker_embedding, p=2, dim=-1)

        emotion_embedding = emotion_embedding.squeeze() 
        speaker_embedding = speaker_embedding.squeeze()

        # Now make sure both are 2D: [batch, features]
        if emotion_embedding.dim() == 1:
            emotion_embedding = emotion_embedding.unsqueeze(0)
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

        ldm_condition = torch.cat([emotion_embedding, speaker_embedding], dim=1)
        ldm_condition = self.embeddingTransform(ldm_condition.unsqueeze(1))

        latents = self.diffusion_model.inference(ldm_condition)
        batchsize = ldm_condition.shape[0]
        latents = latents.view(batchsize, -1)

        return latents
    
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
        true_latents = style.unsqueeze(-1).unsqueeze(-1)  # (B,128,1,1)

        # 3) emotion condition (B,1024)
        emo = batch['emotion_emb'].detach().squeeze(1)  # (B,1024)
        # 4) build condition token (B,1,1536)
        cond = torch.cat([emo, speaker], dim=1)
        cond = self.embeddingTransform(cond).unsqueeze(1)

        return true_latents.to(cond.device), cond


    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log('train_loss', 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return None 
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        torch.manual_seed(self.current_epoch + 42) 

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

        torch.manual_seed(42)
        np.random.seed(42)
        
        true_latents, emotion = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(true_latents, emotion, validation_mode=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cosine_sim', cosine_sim, on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            pred_v, cond = self.extract_latents_and_emotion(batch)  # pred_v is teacher target here
            # get UNet prediction at midpoint t for logging (already computed as part of loss if you want)
            self.log("val_style_norm_target", pred_v.view(pred_v.size(0), -1).norm(dim=1).mean(), prog_bar=False)

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
        warmup_steps = 50_000
        scheduler = LambdaLR(optimizer, lambda s: s / max(1, warmup_steps) if s < warmup_steps else 1.0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
