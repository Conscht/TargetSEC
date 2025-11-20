# Assume 'Model' is the class where the entire model is defined which includes the Style Encoder
from StyleSpeech.models.StyleSpeech import StyleSpeech
import torch

from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config


config = style_config

# Initialize the model
model = StyleSpeech(config)

tensor_path = "audio/train/MSP-PODCAST_0001_0019_mel.pt"
input_data = torch.load(tensor_path)
input_data = input_data.transpose(2,1)

# Load the checkpoint
checkpoint_path = '/export/home/1auga/infhome/Documents/ba-constantin-ldm/stylespeech.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Load the model state_dict
model.load_state_dict(checkpoint['model'])

print(input_data.shape)

# If the optimizer state_dict is also stored in the checkpoint
if 'optimizer' in checkpoint:
    optimizer = torch.optim.Adam(model.parameters())  # Initialize the optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

# Print the step to verify
print("Checkpoint step:", checkpoint['step'])

# Set the model to evaluation mode
model = model.style_encoder
model.eval()

# torch.save(model.state_dict(), "pre-trained_models/pre-trained_style")

# Using torch.no_grad() for inference
with torch.no_grad():
    output_trained = model(input_data)
    print("Output from the trained model:", output_trained)

# checkpoint_path = 'models/diffusion_model_training-06-26_00-19-53.pth'
checkpoint = torch.load('pre-trained_models/pre-trained_style')

checkpoint.eval()

# Print all keys in the checkpoint dictionary
print(checkpoint.keys())


# Initialize an untrained model for comparison
untrained_model = MelStyleEncoder(config)
untrained_model.eval()
print("shape", input_data.shape)
# Using torch.no_grad() for inference on untrained model
with torch.no_grad():
    output_untrained = untrained_model(input_data)
    print("Output from the untrained model:", output_untrained)
    print(torch.equal(output_trained, output_untrained))
if torch.isnan(output_trained).any() or torch.isinf(output_trained).any():
    raise ValueError("style_embedding3 contains BECAUSE OF  style_embedding3.view(batch_size, channels, height, width)  NaN or Inf values.")



