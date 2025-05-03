import torch
from src.Style.style_encoder import MelStyleEncoder
import json
import src.Style.utils as utils
import argparse
# import StyleSpeech.models.StyleSpeech as testStyle
from src.synthesizer_module import SynthesizerLightningModule
from config.stylespeech_model_config import style_config
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
import pytorch_lightning as pl


checkpoint = "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/checkpoints_synthesizer/synthesizer_training_style_encoder-09-06_15-21-37-latest.ckpt"
# checkpoint = None

config = {
    "generator": {
        "input_dim": 256,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        "upsample_rates": [5,4,4,2,2],
        "upsample_initial_channel": 1024,   # increase the channels for feature extraction
        "upsample_kernel_sizes": [11,8,8,4,4],#"upsample_kernel_sizes": [16, 10, 8, 4] ,
        "gin_channels": 0,
        "resblock": 1,

    },
      "data": {
        "sampling_rate": 16000,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": None,
    },
    "training": {
            "learning_rate": 2e-4,
            "batch_size": 16,
        }
}





# Initialize the pretrained style encoder
pretrained_style_encoder = MelStyleEncoder(style_config)
pretrained_style_encoder.load_state_dict(torch.load("pre-trained_models/pre-trained_style"))
pretrained_style_encoder.eval()

gen = Generator(config)
discrim = MultiPeriodDiscriminator()

model = SynthesizerLightningModule(style_encoder=pretrained_style_encoder, decoder=gen, discriminator=discrim, config=config)



# Create the test data loader
test_loader = test_create_data_loader(batch_size=1)

# Initialize the PyTorch Lightning Trainer
trainer = pl.Trainer(accelerator="gpu", devices=1)

# Run the test procedure using the trainer and model with the specified checkpoint
trainer.test(model=model, dataloaders=test_loader, ckpt_path=checkpoint)


