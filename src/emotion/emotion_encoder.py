import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits
    
    # def get_emotion_embeddings(self, audio):
    #     """Get the emotional embeddings.
        
    #     Args:
    #     (wav) audio: Audio file"""
    #     input_values = self.processor(audio, return_tensors="pt").input_values
    #     with torch.no_grad():
    #         hidden_states, logits = self.emotion_model(input_values)
    #     return hidden_states

# Load model from hub
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
emotion_model = EmotionModel.from_pretrained(model_name)



def process_func(x: np.ndarray, device, sampling_rate: int = 16000, embeddings: bool = True) -> torch.Tensor:
    # Process the input audio using the Wav2Vec2Processor

    emotion_model.to(device)



    inputs = processor(x, sampling_rate=sampling_rate, return_tensors="pt")


    y = inputs['input_values']  # Extract the input values tensor from the processor
    y = y.to(device)


    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(device)
    # Run the model inference on the batch
    with torch.no_grad():
        outputs = emotion_model(y)
        y = outputs[0 if embeddings else 1]  # Extract either the embeddings or logits tensor based on the `embeddings` flag


    
    return y



def get_emotion_model():
    return processor
