# models/content_encoder.py
# Content encoder: bidirectional LSTM → FC

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    """
    Based on AutoVC encoder:
    - 3 Conv1D layers with InstanceNorm
    - 1 Bidirectional LSTM
    - Final Linear projection
    """

    def __init__(self, input_dim=80, hidden_dim=256, output_dim=128):
        super(ContentEncoder, self).__init__()

        # This is a stack of 3 Conv1D layers, each followed by:
        # - InstanceNorm1d: normalizes across frequency channels (unlike BatchNorm)
        # - ReLU: non-linearity
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU()
        )

        # A bidirectional LSTM processes each time step, both forward and backward in time.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)

        # This projects the LSTM output to a lower-dimensional content embedding
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):  # x is a mel-spectrogram tensor of shape (B, T, 80)
        x = x.transpose(1, 2)  # (B, T, 80) → (B, 80, T) Prep for Conv1D
        x = self.conv(x)  # (B, 80, T) → (B, H, T) Extract local time-frequency patterns
        x = x.transpose(1, 2)  # (B, H, T) → (B, T, H) Prep for LSTM

        out, _ = self.lstm(x)  # → (B, T, 2*H) Learn temporal dependencies (bi-dir)
        out = self.linear(out)  # → (B, T, output_dim) Compress to bottleneck dimension

        return out
