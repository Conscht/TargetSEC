import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AudioSampleCallback(pl.Callback):
    def __init__(self, decoder, val_dataloader: DataLoader, sample_rate=16000, every_n_epochs=5):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.sample_rate = sample_rate
        self.every_n_epochs = every_n_epochs
        self.decoder = decoder

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        device = pl_module.device


        self.decoder = self.decoder.to(device).eval()

        # Sample a single batch from val dataloader
        batch = next(iter(self.val_dataloader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            # Extract conditioning and latents
            _, condition = pl_module.extract_latents_and_emotion(batch)
            latents = pl_module.diffusion_model.inference(condition)

            # Generate audio
            generated_audio = self.decoder(latents).squeeze(1).cpu()

        # Log audio sample to TensorBoard
        trainer.logger.experiment.add_audio(
            f"val_audio_sample_epoch_{trainer.current_epoch}",
            generated_audio[0],
            global_step=trainer.current_epoch,
            sample_rate=self.sample_rate,
        )
