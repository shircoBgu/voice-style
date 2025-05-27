# models/decoder.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    AutoVC-style decoder:
    - LSTM → Conv stack → LSTM → Linear
    - Takes bottleneck as input (Content + Speaker + Emotion)
    - Reconstructs mel-spectrogram
    """
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=80):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

        self.conv_stack = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.lstm2 = nn.LSTM(hidden_dim, 1024, num_layers=2, batch_first=True)

        self.proj = nn.Linear(1024, output_dim)

    def forward(self, x):
        """
        x: (B, T, input_dim=content+speaker+emotion)
        """
        x, _ = self.lstm1(x)            # (B, T, H)
        x = x.transpose(1, 2)           # (B, H, T) → for Conv1D

        x = self.conv_stack(x)          # (B, H, T)
        x = x.transpose(1, 2)           # back to (B, T, H)

        x, _ = self.lstm2(x)            # (B, T, 1024)
        mel_out = self.proj(x)          # (B, T, 80)
        return mel_out
