import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from Ablation.architectures.style_mlps import StyleEmoMLP

class StyleEmoLightningModule(pl.LightningModule):
    def __init__(
        self,
        style_encoder,
        config,
        use_speaker_cond=False,  # <--- NEW FLAG
        style_stats_path="/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/style_stats_new.pt",
    ):
        super().__init__()
        self.config = config
        self.use_speaker_cond = use_speaker_cond

        # 1. Determine Input Dimension based on setup
        # Emotion (1024) + Optional Speaker (512)
        input_dim = 1024 + 512 if self.use_speaker_cond else 1024
        
        # 2. Initialize MLP with dynamic input size
        self.style_emo_mlp = StyleEmoMLP(in_dim=input_dim, hidden_dim=512, out_dim=128)

        self.sr = 16000
        self.mel_hop = 256
        self.segment_size = 125

        # Freeze Teacher
        self.style_encoder = style_encoder.eval()
        for p in self.style_encoder.parameters():
            p.requires_grad = False

        stats = torch.load(style_stats_path, map_location="cpu")
        self.register_buffer("style_mean", stats["mean"].view(1, -1))
        self.register_buffer("style_std", stats["std"].view(1, -1))

        self.save_hyperparameters(config)

    def _rand_mel_audio_slice(self, mel, mel_len, audio):
        # [Keep this method exactly as it was in your code]
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

    def extract_target_and_input(self, batch):
        mel = batch["mel_spectrogram"]
        audio = batch["audio"]
        mel_len = batch["mel_original_lengths"]

        mel_slice, _ = self._rand_mel_audio_slice(mel, mel_len, audio)

        with torch.no_grad():
            style = self.style_encoder(mel_slice)

        target_style_norm = (style - self.style_mean) / (self.style_std + 1e-6)

        # 3. Handle Conditioning Logic
        emo = batch["emotion_emb"].detach().squeeze(1)  # (B, 1024)
        
        if self.use_speaker_cond:
            spk = batch["speaker_emb"].detach().squeeze(1) # (B, 512)
            # Concatenate features for the MLP
            mlp_input = torch.cat([emo, spk], dim=1) # (B, 1536)
        else:
            mlp_input = emo # (B, 1024)

        return target_style_norm, mlp_input

    @torch.no_grad()
    def forward(self, emotion_embedding, speaker_embedding=None):
        emo = emotion_embedding.squeeze()
        if emo.dim() == 1: emo = emo.unsqueeze(0)

        if self.use_speaker_cond:
            if speaker_embedding is None:
                raise ValueError("Model trained with speaker conditioning, but none provided.")
            spk = speaker_embedding.squeeze()
            if spk.dim() == 1: spk = spk.unsqueeze(0)
            mlp_input = torch.cat([emo, spk], dim=1)
        else:
            mlp_input = emo

        pred_norm = self.style_emo_mlp(mlp_input)
        style = pred_norm * self.style_std + self.style_mean
        return style

    def training_step(self, batch, batch_idx):
        if batch is None: return None
        target_style_norm, mlp_input = self.extract_target_and_input(batch)
        pred_style_norm = self.style_emo_mlp(mlp_input)
        loss = F.mse_loss(pred_style_norm, target_style_norm)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None: return None
        target_style_norm, mlp_input = self.extract_target_and_input(batch)
        pred_style_norm = self.style_emo_mlp(mlp_input)
        loss = F.mse_loss(pred_style_norm, target_style_norm)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # [Same as before]
        lr = self.config["training"]["learning_rate"]
        optimizer = AdamW(self.style_emo_mlp.parameters(), lr=lr, weight_decay=0.01)
        scheduler = LambdaLR(optimizer, lambda s: s / max(1, 55000) if s < 55000 else 1.0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}