# models/pretrained_speaker_encoder.py

import torch
import torch.nn as nn

class PretrainedSpeakerEncoder(nn.Module):
    def __init__(self, ckpt_path, device='cpu'):
        super().__init__()

        # Load the full model checkpoint (wrapped in DataParallel)
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Extract the actual model
        full_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = full_model if isinstance(full_model, dict) else full_model.state_dict()

        # Remove 'module.' prefix (DataParallel wrappers)
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            clean_state_dict[new_k] = v

        # Build matching architecture (IBM-style)
        self.conv = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=3, batch_first=True)
        self.proj = nn.Linear(256, 256)

        # Load weights
        self.load_state_dict(clean_state_dict, strict=False)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 80, T)
        x = self.conv(x)       # (B, 512, T)
        x = x.transpose(1, 2)  # (B, T, 512)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)    # (B, T, 256)
        x = self.proj(x)       # (B, T, 256)
        x = torch.mean(x, dim=1)  # (B, 256)
        return x
