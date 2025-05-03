import torch
import os
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from src.Diffusion.diffusion import AudioDiffusion
from src.decoder.decoder_modules import rand_slice_segments
from src.emotion.emotion_encoder import process_func
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW



class DiffusionLightningModule(pl.LightningModule):
    def __init__(self, style_encoder, config, unet_model_config_path="config/diffusion_model_config2.json", pretrained_unet=None):
        super(DiffusionLightningModule, self).__init__()

        self.diffusion_model = AudioDiffusion(
            scheduler_name="stabilityai/stable-diffusion-2-1", 
            unet_model_config_path=unet_model_config_path,
            unet_model_name=pretrained_unet,
            snr_gamma=None,
        )

        self.style_encoder = style_encoder
        self.config = config
        self.segment_size = 32 # 0.5 secs in Mel Spec Units: 32 * 256 = 8.192 (16kHz)

        self.save_hyperparameters(config) 

    

    def forward(self, emotion_embedding):
        latents = self.diffusion_model.inference(emotion_embedding)
        batchsize = emotion_embedding.shape[0]

        latents = latents.view(batchsize, -1)

        return latents
    
    def check_for_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} contains NaN or Inf values.")

    def extract_latents_and_emotion(self, batch):
        raw_audio = batch['audio']
        speaker_emb = batch['speaker_emb']
        mel_spec = batch['mel_spectrogram'].transpose(1, 2)
        mel_length = batch['mel_original_lengths']

        x, mel_start_id = rand_slice_segments(mel_spec, mel_length, segment_size=self.segment_size)
        x = x.transpose(1, 2)

        y_audio = []
        expected_len = self.segment_size * 256

        for i in range(raw_audio.size(0)):
            start = mel_start_id[i].item() * 256
            end = start + expected_len
            segment = raw_audio[i, start:end]
            if segment.size(0) < expected_len:
                raise ValueError(f"Audio segment too short: {segment.size(0)}")
            y_audio.append(segment[:expected_len])

        y_audio = torch.stack(y_audio)

        with torch.no_grad():
            emotion_embedding = process_func(y_audio.cpu().numpy(), device=speaker_emb.device, sampling_rate=16000, embeddings=True)
            true_latents = self.style_encoder(x)

        self.check_for_nan_inf(true_latents, "true_latents")

        return true_latents.view(true_latents.size(0), 128, 1, 1), emotion_embedding.unsqueeze(1).to(x.device)


    def training_step(self, batch, batch_idx):
        torch.manual_seed(self.current_epoch + 42) 

        true_latents, emotion = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(true_latents, emotion)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_cosine_sim', cosine_sim, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        true_latents, emotion = self.extract_latents_and_emotion(batch)
        loss, cosine_sim = self.diffusion_model(true_latents, emotion)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cosine_sim', cosine_sim, on_step=False, on_epoch=True, prog_bar=True)
        return loss



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config["training"]["learning_rate"])
        
        # Define warmup steps (e.g., 10% of total steps)
        warmup_steps = 12000
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # Maintain constant learning rate after warmup

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step
                'frequency': 1       # Check every batch
            }
        }
    
