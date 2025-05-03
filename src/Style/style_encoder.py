import torch
import torch.nn as nn
import numpy as np

from .blocks import Mish, LinearNorm,  Conv1dGLU, \
                    MultiHeadAttention


class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder.
    '''
    
    def __init__(self, config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = config.n_mel_channels 
        self.hidden_dim = config.style_hidden
        self.out_dim = config.style_vector_dim
        self.kernel_size = config.style_kernel_size
        self.n_head = config.style_head
        self.dropout = config.dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        #Input does step by step into this
        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        # Input shape (batch_size, time_steps, mel_channels)
        print("Shape in forward", x.shape)   
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        print("Shape after mask", x.shape)  
        
        # spectral
        print("Shape before spectral", x.shape)       
        x = self.spectral(x)   # Expects [batch_size, time_steps, mel_channels]

        # temporal
        print("Shape before transpse", x.shape)
        x = x.transpose(1,2)
        print("Shape befor Temporal", x.shape)
        x = self.temporal(x)  # Expects [batch_size, channels, time_steps] = > [batch_size, hidden_dim, time_steps]
        print("Shape after Temporal", x.shape)
        x = x.transpose(1,2)

        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w

