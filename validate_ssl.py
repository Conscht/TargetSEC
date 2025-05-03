
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
from src.diffusion_module import DiffusionLightningModule
from src.SSL import EmoSSL, load_emotion_embeddings

# Load emotion embeddings from the directory
emotion_embedding_dir = "/data/rajprabhu/dataset/MSP-Podcast-1.10/avgclass_emo_embeds"
emotion_embeddings = load_emotion_embeddings(emotion_embedding_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_synth = "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/checkpoints_synthesizer/synthesizer_training_style_encoder-09-01_20-53-59-latest.ckpt"

checkpoint_ldm =   "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/checkpoints/diffusion_model_training-10-08_15-59-43-epoch=40-val_loss=0.45.ckpt"


# diffusion_model_training-09-27_16-01-55-epoch=39-val_loss=0.07.ckpt bestes
# diffusion_model_training-09-29_02-31-56-latest.ckpt best loss

config = {
    "generator": {
        "input_dim": 768,
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

# Load synthesizer from checkpoint
synthesizer = SynthesizerLightningModule.load_from_checkpoint(checkpoint_synth, style_encoder=pretrained_style_encoder, decoder=gen, discriminator=discrim, config=config).eval()

# Load LDM from checkpoint
ldm = DiffusionLightningModule.load_from_checkpoint(checkpoint_ldm, style_encoder=pretrained_style_encoder, config=config).eval()




# Example of running evaluation with test data
test_data_loader = test_create_data_loader(batch_size=1)

synthesis_with_ldm = EmoSSL(synthesizer, ldm)


# Iterate over the emotion classes (1-7)
for emotion_class in range(4, 8):
    print(f"Validating with emotion class {emotion_class}")

    # Get the corresponding emotion embedding
    emo_embedding = emotion_embeddings[emotion_class]

    # Ensure the emotion embedding is on the same device as the model
    emo_embedding = torch.tensor(emo_embedding).to(device)

    # Loop through all batches using the same emotion embedding
    for batch_idx, batch in enumerate(test_data_loader):
        # Move batch tensors to the same device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            else:
                # Skip non-tensor elements
                batch[key] = value

        # Perform inference and evaluation for each batch using the current emotion embedding
        synthesis_with_ldm.test_step(batch, batch_idx, emo_embedding, emotion_class)
