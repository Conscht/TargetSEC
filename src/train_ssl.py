import torch
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import MelSpectrogramDataset, collate_fn, create_dataloaders
from src.synthesizer_module import SynthesizerLightningModule
# from src.train_synthesizer_distributed import SynthesizerLightningModule
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.decoder.decoder import Generator, DiscriminatorS, MultiPeriodDiscriminator

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
from src.SSL import EmoSSL

checkpoint_synth = "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/checkpoints_synthesizer/synthesizer_training_style_encoder-09-01_20-53-59-latest.ckpt"

checkpoint_ldm =   "/data/rajprabhu/dataset/1auga/ba-constantin-ldm/checkpoints/diffusion_model_training-10-08_15-59-43-epoch=40-val_loss=0.45.ckpt"
checkpoint=None
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




def main():
    pl.seed_everything(1234)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    base_name = generate_base_name("synthesizer_diffusion")


    train_loader, val_loader = create_dataloaders(batch_size=config['training']['batch_size'])

    model = EmoSSL(synthesizer=synthesizer, ldm=ldm, config=config)


    logger = setup_logger("logs_synthesizer", base_name)
    callbacks = setup_callbacks("checkpoints_synthesizer", base_name)

    trainer = Trainer(
        logger=logger,
        max_epochs=400,
        min_epochs=10,
        accelerator="gpu",  
        devices=num_gpus,  
        precision="32",  # => when on 16, style encoder gives Nan values
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
    )

    if checkpoint is not None:
        # Ensure that EarlyStopping is not in the list of callbacks
        trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, EarlyStopping)]
        
        trainer.should_stop = False
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)

        if trainer.should_stop:
            print("Training should stop: trainer.should_stop is True.")
        else:
            print("Training should continue: trainer.should_stop is False.")
        
        # save_model(model, "models_synthesizer", base_name)

    else:
        trainer.fit(model, train_loader, val_loader)                                                               
        # save_model(model, "models_synthesizer", base_name)
        

def generate_base_name(log_name):
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"{log_name}-{timestamp}"

def setup_logger(log_folder, base_name):
    return TensorBoardLogger(save_dir=log_folder, name=base_name)

def setup_callbacks(checkpoint_folder, base_name):
    checkpoint_filename = f"{base_name}-{{epoch}}-{{val_loss:.2f}}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=checkpoint_filename,
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename=f"{base_name}-latest",
        save_top_k=1,
        verbose=True,
        monitor='epoch',
        mode='max',
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=300,
        verbose=True,
        mode='min'
    )

    return [checkpoint_callback, latest_checkpoint_callback, early_stopping_callback]

# def save_model(model, model_folder, base_name):
#     if not os.path.exists(model_folder):
#         os.makedirs(model_folder)
#     model_filename = f"{base_name}.pth"
#     model_filepath = os.path.join(model_folder, model_filename)
#     torch.save(model.state_dict(), model_filepath)
#     print(f"Model saved to {model_filepath}.")

if __name__ == "__main__":
    main()

