# Assume 'Model' is the class where the entire model is defined which includes the Style Encoder
from StyleSpeech.models.StyleSpeech import StyleSpeech
import torch

from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config



# instantiate the StyleSpeech model with this configuration object
model_config = style_config

tensor_path = "mel_spectrograms/Test1/MSP-PODCAST_0001_0063_mel.pt"
mel_spectrogram = torch.load(tensor_path)

pretrained_style_encoder = MelStyleEncoder(style_config)
pretrained_style_encoder.load_state_dict(torch.load("pre-trained_models/pre-trained_style"))
pretrained_style_encoder.eval()

print("Shape of tensor", mel_spectrogram.shape)
# if len(mel_spectrogram.shape) == 3 and mel_spectrogram.shape[0] == 1:
#     mel_spectrogram = mel_spectrogram.squeeze(0)

mel_spectrogram = mel_spectrogram.transpose(2, 1)  # (80, 128) =>(128, 80)
print("shape", mel_spectrogram.shape)

style_embedding3 = pretrained_style_encoder(mel_spectrogram)
print(style_embedding3, style_embedding3)
print(torch.equal(style_embedding3, style_embedding3))
if torch.isnan(style_embedding3).any() or torch.isinf(style_embedding3).any():
    raise ValueError("style_embedding3 contains BECAUSE OF  style_embedding3.view(batch_size, channels, height, width)  NaN or Inf values.")




