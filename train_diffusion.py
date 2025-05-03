import torch
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import MelSpectrogramDataset, collate_fn, create_dataloaders
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.diffusion_module_fixed import DiffusionLightningModule


# Initialize the pretrained style encoder
pretrained_style_encoder = MelStyleEncoder(style_config)
pretrained_style_encoder.load_state_dict(torch.load("/Users/Conscht/Documents/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"))
pretrained_style_encoder.eval()

#checkpoint = "/Users/Conscht/Documents/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/diffusion_model_training-10-08_15-59-43-latest.ckpt"
checkpoint = None


def main():
    pl.seed_everything(1234)
    torch.set_float32_matmul_precision("high")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    base_name = generate_base_name("diffusion_model_training")



    config = {
        "training": {
            "learning_rate": 1e-4,  # Increased learning rate for faster convergence
            "batch_size": 16        # Keep batch size at 32 for now, but consider reducing if needed
        }
    }


    train_loader, val_loader = create_dataloaders(batch_size=config['training']['batch_size'])
    model = DiffusionLightningModule(style_encoder=pretrained_style_encoder, config=config)


    logger = setup_logger("logs", base_name)
    callbacks = setup_callbacks("checkpoints", base_name)

    trainer = Trainer(
        logger=logger,
        max_epochs=150,
        min_epochs=10,
        accelerator="gpu", 
        gradient_clip_val=0.5, 
        devices=num_gpus,  
        precision=32,  # => when on 16, style encoder gives Nan values
        callbacks=callbacks,
    )

    if checkpoint is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)

    # model.save_unet(os.path.join("models_unet", base_name)) ## save the unet

    # save_model(model, "models", base_name)

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
        patience=60,
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

