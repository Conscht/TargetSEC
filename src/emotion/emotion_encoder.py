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


class DifferentiableSER(nn.Module):
    """Frozen audeering SER usable *inside* a loss.

    `process_func` above cannot be used for training: it runs under
    `torch.no_grad()` and round-trips through NumPy, so any loss built on it is
    a constant with respect to the generator. This wrapper keeps the graph, so
    gradients flow through the (frozen) SER back into the waveform -- which is
    what L_SER in the paper requires.

    Weights are frozen and the module is pinned to eval mode: `RegressionHead`
    contains dropout, and letting it activate would inject noise into the loss.
    """

    def __init__(self, pretrained_name: str = model_name):
        super().__init__()
        self.model = EmotionModel.from_pretrained(pretrained_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # nn.Module.__init__ leaves self.training True; the child .eval() alone
        # would not cover the wrapper, and a later .train() on the parent would
        # re-enable RegressionHead's dropout inside the loss.
        self.eval()

    def train(self, mode: bool = True):
        # Ignore `mode`: this module must never leave eval.
        return super().train(False)

    @staticmethod
    def normalize(wav: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Torch port of Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm.

        The processor uses population variance (numpy ddof=0), so unbiased=False
        here rather than torch's default.
        """
        mean = wav.mean(dim=-1, keepdim=True)
        var = wav.var(dim=-1, unbiased=False, keepdim=True)
        return (wav - mean) / torch.sqrt(var + eps)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) raw 16 kHz audio -> (B, 3) = arousal, dominance, valence."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return self.model(self.normalize(wav))[1]


def concordance_cc(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Lin's concordance correlation coefficient, differentiable.

    CCC is a *batch-level* statistic -- undefined for a single pair -- so its
    value depends on batch size. Matches torchmetrics.ConcordanceCorrCoef
    (biased / population moments).
    """
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    mx, my = x.mean(), y.mean()
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    return 2.0 * cov / (vx + vy + (mx - my) ** 2 + eps)
