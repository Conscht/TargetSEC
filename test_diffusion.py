# Assume 'Model' is the class where the entire model is defined which includes the Style Encoder
from StyleSpeech.models.StyleSpeech import StyleSpeech
import torch
import librosa

from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.emotion.emotion_encoder import EmotionModel

from src.Diffusion.diffusion import AudioDiffusion
from src.Diffusion.diffusion_module import DiffusionLightningModule


# Prepare Audio
tensor_path = "audio/train/MSP-PODCAST_1495_0163_0018_mel.pt"
audio_path = "/data/rajprabhu/dataset/MSP-Podcast-1.10/Audio/MSP-PODCAST_1495_0163_0018.wav"


audio, _ = librosa.load(audio_path, sr=None)  
waveform = torch.tensor(audio, dtype=torch.float32) 

max_length = 16000
if waveform.shape[0] < max_length:
    padded_waveform = torch.cat((waveform, torch.zeros(max_length - waveform.shape[0])), dim=0)
else:
    padded_waveform = waveform[:max_length]

# Ensure the waveform has the correct shape [batch_size, num_channels, sequence_length]
padded_waveform = padded_waveform.unsqueeze(0)  # Add batch dimension
input_data = torch.load(tensor_path)
input_data = input_data.transpose(2,1)


# Initialize the pretrained style encoder
pretrained_style_encoder = MelStyleEncoder(style_config)
pretrained_style_encoder.load_state_dict(torch.load("pre-trained_models/pre-trained_style"))
pretrained_style_encoder.eval()
# Initialize the emotion encoder
emotion_encoder = EmotionModel.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')
emotion_encoder.eval()

with torch.no_grad():
    emo_embedding, rec_emo = emotion_encoder(padded_waveform)
emo_embedding = emo_embedding.unsqueeze(1)  # (1, 1024) => Expected(1, 1, 1024)


config = {
    "training": {
    "learning_rate": 1e-4,
    "batch_size": 16
    }
}



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = DiffusionLightningModule.load_from_checkpoint("checkpoints/diffusion_model_training-07-22_11-17-20-epoch=59-val_loss=626.46.ckpt", 
                                                      style_encoder=pretrained_style_encoder,  
                                                      emotional_encoder=emotion_encoder, 
                                                      config=config).to(device)

model.eval()

emo_embedding = emo_embedding.to(device)
input_data = input_data.to(device)
scheduler=(model.diffusion_model.inference_scheduler)
with torch.no_grad():
    model_pred = model.diffusion_model.inference(emo_embedding, inference_scheduler=scheduler, num_steps=40)
    target = pretrained_style_encoder(input_data)
    print("Trained Diff model", model_pred)
    print("Trained Style model", target)

print("MSE", torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean"))


import audeer
import audonnx
import numpy as np
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

sampling_rate = 16000
signal = np.random.normal(size=sampling_rate).astype(np.float32)
prediction_audi  = model(signal, sampling_rate)["hidden_states"]
reconginition  = model(signal, sampling_rate)["logits"]


reconginition= torch.from_numpy(reconginition).float()
prediction_audi= torch.from_numpy(prediction_audi).float()
prediction_audi = prediction_audi.to(device)
reconginition = reconginition.to(device)
rec_emo = rec_emo.to(device)

print("prediction_audi", prediction_audi)
print("Pred Emo Model", emo_embedding)

print("prediction_audi dimension", reconginition)
print("Emo Model dimension", rec_emo)

emo_embedding = emo_embedding.squeeze(1)
print("MSE", torch.nn.functional.mse_loss(prediction_audi, emo_embedding.float(), reduction="mean"))
print("MSE Recognition", torch.nn.functional.mse_loss(rec_emo, reconginition.float(), reduction="mean"))