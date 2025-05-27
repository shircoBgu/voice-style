# models/speaker_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    """
    Speaker Encoder based on AutoVC's Encoder:
    - 3 Conv1D layers with BatchNorm and ReLU
    - 2-layer LSTM
    - Global average pooling
    - Final Linear projection
    """
    def __init__(self, input_dim=80, hidden_dim=512, output_dim=128):
        super(SpeakerEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, 80)
        x = x.transpose(1, 2)  # (B, 80, T)
        x = self.conv_layers(x)  # (B, H, T)
        x = x.transpose(1, 2)  # (B, T, H)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # (B, T, H)

        # Global average pooling over time
        out = torch.mean(outputs, dim=1)  # (B, H)

        out = self.linear(out)  # (B, output_dim)
        return out
