# models/decoder.py

import torch.nn as nn
from models.autoVC.blocks import ConvNorm, LinearNorm


class Decoder(nn.Module):
    """
    AutoVC-style decoder:
    - LSTM → Conv stack → LSTM → Linear
    - Takes bottleneck as input (Content + Speaker + Emotion)
    - Reconstructs mel-spectrogram
    """

    # input_dim = 64 + 256 + 768 = 1088:
    # dim_neck = 32 → content_dim = 2 * dim_neck = 64
    # speaker_emb_dim = 256 (from pretrained speaker encoder)
    # emotion_emb_dim = 768 (from emo2vec)
    def __init__(self, input_dim: int = 1088, dim_pre: int = 512, output_dim: int = 80):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, dim_pre, num_layers=1, batch_first=True)

        self.conv_stack = nn.Sequential(
            ConvNorm(dim_pre, dim_pre, kernel_size=5, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(dim_pre),
            nn.ReLU(),

            ConvNorm(dim_pre, dim_pre, kernel_size=5, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(dim_pre),
            nn.ReLU(),

            ConvNorm(dim_pre, dim_pre, kernel_size=5, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(dim_pre),
            nn.ReLU()
        )

        self.lstm2 = nn.LSTM(dim_pre, 1024, num_layers=2, batch_first=True)
        self.linear = LinearNorm(1024, output_dim)

    def forward(self, x):
        """
        x: (B, T, input_dim=content+speaker+emotion)
        """
        x, _ = self.lstm1(x)  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T) → for Conv1D

        x = self.conv_stack(x)  # (B, H, T)
        x = x.transpose(1, 2)  # back to (B, T, H)

        x, _ = self.lstm2(x)  # (B, T, 1024)
        mel_out = self.linear(x)  # (B, T, 80)
        return mel_out
