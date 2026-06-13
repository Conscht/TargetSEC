
import torch
from src.Style.style_encoder import MelStyleEncoder
import json
import src.Style.utils as utils
import argparse
# import StyleSpeech.models.StyleSpeech as testStyle
from src.synthesizer_style_module import SynthesizerLightningModule
from config.stylespeech_model_config import style_config
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
import pytorch_lightning as pl
from src.diffusion_module_fixed import DiffusionLightningModule
from src.SSL import EmoSSL, load_emotion_embeddings

# Load emotion embeddings from the directory
emotion_embedding_dir = r"/sc/home/constantin.auga/New folder/Audio/MSP-Podcast-1.10/avgclass_emo_embeds"

emotion_embeddings = load_emotion_embeddings(emotion_embedding_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_synth = r"/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/processing/synthesizer_training_wvmos_3.05-12-11_04-33-22-latest.ckpt"

checkpoint_ldm =   r"/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/checkpoints/diffusion_model_training-11-19_11-32-04-epoch=149-val_loss=0.12.ckpt"
# checkpoint_ldm = r"C:\Users\Conscht\Documents\New folder\Code\EmoConv-LDM\checkpoints\diffusion_model_training-11-18_23-50-04-epoch=145-val_loss=0.1088_dream_diff.ckpt"
# checkpoint_ldm = r"C:\Users\Conscht\Documents\New folder\Code\EmoConv-LDM\checkpoints\diffusion_model_training-11-16_14-36-04-epoch=105-val_loss=0.12.ckpt"


# diffusion_model_training-09-27_16-01-55-epoch=39-val_loss=0.07.ckpt bestes
# diffusion_model_training-09-29_02-31-56-latest.ckpt best loss

config = {
    "cross_attention_dim": 512,

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
            "learning_rate": 1e-4,
            "batch_size": 8,
        } 
}
config_2 = {
        "training": {
            "learning_rate": 3e-5,
            "batch_size": 32,
            "cfg_prob": 0.3,  # train-time CFG probability (like DreamVoice)
            "warmup_steps": 55_000,
        },
        "inference": {
            "guidance_scale": 3.0,
            "guidance_rescale": 0.9,
        },
        "cross_attention_dim": 512,  # emo(1024) + spk(512) -> 128+128
    }



# Initialize the pretrained style encoder
pretrained_style_encoder = MelStyleEncoder(style_config)
pretrained_style_encoder.load_state_dict(torch.load("/sc/home/constantin.auga/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"))
pretrained_style_encoder.eval()

gen = Generator(config)
discrim = MultiPeriodDiscriminator()

# Load synthesizer from checkpoint
synthesizer = SynthesizerLightningModule.load_from_checkpoint(checkpoint_synth, style_encoder=pretrained_style_encoder, decoder=gen, discriminator=discrim, config=config).eval()

# Load LDM from checkpoint
ldm = DiffusionLightningModule.load_from_checkpoint(checkpoint_ldm,  style_encoder=pretrained_style_encoder, config=config_2).eval()




# Example of running evaluation with test data
if __name__ == "__main__":
    import torch.multiprocessing
    pl.seed_everything(1234)
    torch.multiprocessing.freeze_support()  # Optional on Windows but good practice

    # Your existing logic
    # ...
    test_data_loader = test_create_data_loader(batch_size=1)
    synthesis_with_ldm = EmoSSL(synthesizer, ldm, config=config)

    from collections import defaultdict

    mse_sum = defaultdict(float)
    mae_sum = defaultdict(float)
    n_sum   = defaultdict(int)
    for emotion_class in range(1, 8):
        print(f"Validating with emotion class {emotion_class}")
        emo_embedding = torch.tensor(emotion_embeddings[emotion_class]).to(device)

        for batch_idx, batch in enumerate(test_data_loader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            synthesis_with_ldm.test_step(batch, batch_idx, emo_embedding, emotion_class)
            out = synthesis_with_ldm.test_outputs[-1]  # last result dict appended in test_step
            mse_sum[emotion_class] += float(out["mse_arousal"])
            mae_sum[emotion_class] += float(out["mae_arousal"])
            n_sum[emotion_class]   += 1

    print("\n=== Arousal goodness on Test2 ===")
    per_class_mse = {}
    per_class_mae = {}

    for c in range(1, 8):
        n = max(1, n_sum[c])
        per_class_mse[c] = mse_sum[c] / n
        per_class_mae[c] = mae_sum[c] / n
        print(f"Class {c}: MSE={per_class_mse[c]:.4f}  MAE={per_class_mae[c]:.4f}  N={n_sum[c]}")

    macro_mse = sum(per_class_mse.values()) / 7.0
    macro_mae = sum(per_class_mae.values()) / 7.0
    print(f"Macro: MSE={macro_mse:.4f}  MAE={macro_mae:.4f}")
    print(f"Worst-class: MSE={max(per_class_mse.values()):.4f}  MAE={max(per_class_mae.values()):.4f}")


 