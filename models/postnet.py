# models/postnet.py

import torch
import torch.nn as nn

class Postnet(nn.Module):
    """
    Postnet module:
    - 5 Conv1D layers
    - First 4 use tanh activation
    - Final layer is linear
    - Residual: output = decoder_output + postnet_output
    """
    def __init__(self, mel_dim=80, hidden_dim=512, kernel_size=5):
        super(Postnet, self).__init__()

        layers = []

        # First 4 layers: 80 → 512, 512 → 512, ...

        # first layer 80 -> 512
        layers.append(nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        ))

        # other 3 layers 512 -> 512
        for _ in range(3):
            layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh()
            ))

        # Final layer: 512 → 80, no activation
        layers.append(nn.Sequential(
            nn.Conv1d(hidden_dim, mel_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(mel_dim)
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        x: (B, T, 80) — decoder output
        """
        x = x.transpose(1, 2)  # (B, 80, T)
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        x = x.transpose(1, 2)  # (B, T, 80)
        return x
