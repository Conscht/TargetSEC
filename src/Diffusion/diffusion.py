import random
import inspect
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from diffusers import DDPMScheduler, UNet2DConditionModel
from processing.diffusion_utils import FixedEmbedding, rescale_noise_cfg

class AudioDiffusion(nn.Module):
    '''Creates an Diffusion Model.
    '''
    
    def __init__(
        self,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        cfg_prob=0.1

    ):
        '''Initialize the Diffusion Model, by setting up all components.

        Args:
        (int) num_emotions: Number of emotion labels.
        (int) embedding_dim: Dimension of the embeddings.
        (int) encoder_dim: Dim of encoder output.
        (Strg) scheduler_name: Scheduler name.
        (Strg) unet_model_name: Unet model name.
        (Strg) unet_model_config_path: Path to existing model.
        (float) snr_gamma: Gamma value for imrpoved training.
        '''
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="v_prediction",
            clip_sample=False,
            timestep_spacing="trailing",
            # rescale_betas_zero_snr=True
        )
        self.inference_scheduler = deepcopy(self.noise_scheduler)

        if unet_model_config_path is not None:
            unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
            self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
            self.set_from = "random"
            print("UNet initialized randomly.")
        else:
            print("[Info] UNet has not been initialized, missing path.")

        self.null_token = FixedEmbedding(self.unet.config.cross_attention_dim)
        self.cfg_prob = cfg_prob


    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


    def forward(self, latents, emotion_embeddings, validation_mode=False):
        '''Forward pass of the LDM.
        
        Args:
        (int) latents = Dimension of latent.
        (vec) emotion_embeddings = Guidance vector.
        (bool) validation_mode = Bool if validation mode is true.
        '''

        device = emotion_embeddings.device
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        bsz = latents.shape[0]

        # Decide if we use fixed size of timesteps(val), or random size(training, better for generalization)
        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps // 2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        # Forward Diffusion, gradually add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target noise
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        attention_mask = None

        # Predict the noise with the unet
        if not validation_mode and self.cfg_prob > 0.0:
            B, T, D = emotion_embeddings.shape  # T=1 in your setup
            keep = (torch.rand(B, device=device) >= self.cfg_prob).view(B, 1, 1)  # True = keep cond
            null = self.null_token(B, T, device=device, dtype=emotion_embeddings.dtype)  # (B, T, D)
            emotion_embeddings = torch.where(keep, emotion_embeddings, null)

        # Predict the noise with the unet
        if self.set_from == "random":
            model_pred = self.unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=emotion_embeddings, 
                encoder_attention_mask=attention_mask,
            ).sample


        # Calculate the loss, the loss is the predicted MSE of predicted noise and actual noise
        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        # Flatten to (batch, 128)
        model_pred = model_pred.squeeze(-1).squeeze(-1)
        target = target.squeeze(-1).squeeze(-1)

        # Normalize the vectors to ensure they are unit vectors
        # model_pred = F.normalize(model_pred.float(), p=2, dim=1)
        # target = F.normalize(target.float(), p=2, dim=1)

        # Calculate cosine similarity
        cosine_similarity = F.cosine_similarity(model_pred.float(), target.float(), dim=1) # targett and prediction => (batch, 128, 1, 1)
        cosine_similarity = cosine_similarity.mean()
        
        return loss, cosine_similarity
    
    def forward_fine_tuning(self, emotion_embeddings, num_steps=10, guidance_scale=3.0):
        device = emotion_embeddings.device
        batch_size = emotion_embeddings.size(0)

        self.noise_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # Expand embeddings for classifier-free guidance if needed
        do_guidance = guidance_scale > 1.0
        if do_guidance:
            emotion_embeddings = emotion_embeddings.repeat_interleave(2, dim=0)

        # Random latent
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, 1, 1),
            device=device,
            dtype=emotion_embeddings.dtype,
            requires_grad=True  # crucial for backprop
        ) * self.noise_scheduler.init_noise_sigma

        for t in timesteps:
            latent_model_input = latents
            if do_guidance:
                latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=emotion_embeddings,
                encoder_attention_mask=None,
            ).sample

            if do_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        return latents.squeeze(-1).squeeze(-1)  # shape: (B, D)


    @torch.no_grad()
    def inference(self, embeddings, inference_scheduler=None, num_steps=100, guidance_scale=3, num_samples_per_prompt=1, disable_progress=True):
        device = embeddings.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = embeddings.size(0) * num_samples_per_prompt


        
        cond = embeddings.repeat_interleave(num_samples_per_prompt, 0).to(torch.float32)  # (B,1,1536)

        num_channels_latents = self.unet.config.in_channels
        
        # Get random latent

        # Match the size of latent size inside the loop
        if classifier_free_guidance:
            B, T, D = cond.shape
            uncond = self.null_token(B, T, device=cond.device, dtype=cond.dtype)
            emotion_embeddings = torch.cat([uncond, cond], dim=0)
        else:
            emotion_embeddings = cond

        inference_scheduler = self.inference_scheduler
        inference_scheduler.set_timesteps(num_steps, device=device)

        timesteps = inference_scheduler.timesteps

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, emotion_embeddings.dtype, device)

        
        progress_bar = tqdm(range(num_steps), disable=disable_progress)


        # Gradualyl denoise sample
        for i, t in enumerate(timesteps):
            x = latents
            if classifier_free_guidance:
                x = torch.cat([x, x], dim=0)
            x = inference_scheduler.scale_model_input(x, t)
            

            # Predict noise
            noise_pred = self.unet(x,
                                   t, 
                                   encoder_hidden_states=emotion_embeddings,
                                   encoder_attention_mask=None,
                                   ).sample

            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = rescale_noise_cfg(noise_pred_cfg, noise_pred_cond, k=0.7)
                
            # remove noise, get sample with less noise
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)


        return latents


    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 1, 1)  # Add an extra dimension for broadcasting to 2D
        latents = torch.randn(shape, generator=None, device=device, dtype=dtype)
        latents = latents * inference_scheduler.init_noise_sigma
        return latents
