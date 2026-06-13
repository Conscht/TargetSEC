
import torch 
import torch.nn as nn

class StyleEmoMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()

        self.mel_hop = 256
        self.segment_size = 125
        self.emo_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.emo_mlp(x)
        return x


