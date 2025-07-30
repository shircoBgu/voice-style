# models/content_encoder.py
# Content encoder: bidirectional LSTM → FC
from models.autoVC.blocks import ConvNorm  # uses Xavier init with gain

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    """
    Based on AutoVC encoder:
        - 3 Conv1D layers with BatchNorm1d
        - 1 Bidirectional LSTM
    """

    def __init__(self, speaker_dim, dim_neck=32, freq=32):
        super(ContentEncoder, self).__init__()
        self.freq = freq
        self.dim_neck = dim_neck

        # Input: (80 + speaker_dim) channels
        in_channels = 80 + speaker_dim

        self.convs = nn.Sequential(
            ConvNorm(in_channels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ConvNorm(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ConvNorm(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=dim_neck,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, mel, speaker_emb):
        """
        mel: (B, T, 80)
        speaker_emb: (B, speaker_dim)
        """
        B, T, _ = mel.shape

        # Expand speaker embedding to (B, T, speaker_dim)
        spk_expanded = speaker_emb.unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([mel, spk_expanded], dim=-1)  # (B, T, 80 + speaker_dim)
        x = x.transpose(1, 2)  # (B, 80+spk, T)
        x = self.convs(x)  # (B, 512, T)
        x = x.transpose(1, 2)  # (B, T, 512)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # (B, T, 2*dim_neck)

        # Split forward and backward
        out_forward = outputs[:, :, :self.dim_neck]  # (B, T, dim_neck)
        out_backward = outputs[:, :, self.dim_neck:]  # (B, T, dim_neck)

        codes = []
        for i in range(0, T, self.freq):
            if i + self.freq <= T:
                code = torch.cat(
                    (out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]),
                    dim=-1
                )  # (B, 2*dim_neck)
                codes.append(code)

        return codes  # list of (B, 2*dim_neck)
