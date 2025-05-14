# models/autovc.py

import torch
import torch.nn as nn

from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoder
from models.decoder import Decoder

class AutoVC(nn.Module):
    def __init__(self,
                 content_dim=80,
                 speaker_dim=80,
                 content_emb_dim=128,
                 speaker_emb_dim=128,
                 emotion_emb_dim=128,
                 bottleneck_dim=384,   # 128 + 128 + 128
                 mel_dim=80):
        super(AutoVC, self).__init__()

        self.content_encoder = ContentEncoder(input_dim=content_dim,
                                              hidden_dim=256,
                                              output_dim=content_emb_dim)

        self.speaker_encoder = SpeakerEncoder(input_dim=speaker_dim,
                                              hidden_dim=256,
                                              output_dim=speaker_emb_dim)

        self.decoder = Decoder(input_dim=bottleneck_dim,
                               hidden_dim=256,
                               output_dim=mel_dim)

    def forward(self, source_mel, target_mel, emotion_emb):
        """
        source_mel: (B, T, 80) - source speech input
        target_mel: (B, T', 80) - target speaker reference
        emotion_emb: (B, 128) - emotion embedding from label
        """
        # Content and speaker embeddings
        content_emb = self.content_encoder(source_mel)           # (B, T, 128)
        speaker_emb = self.speaker_encoder(target_mel)           # (B, 128)

        # Repeat speaker and emotion to match time dimension
        T = content_emb.size(1)
        speaker_emb = speaker_emb.unsqueeze(1).repeat(1, T, 1)   # (B, T, 128)
        emotion_emb = emotion_emb.unsqueeze(1).repeat(1, T, 1)   # (B, T, 128)

        # Concatenate: (content | speaker | emotion)
        bottleneck = torch.cat([content_emb, speaker_emb, emotion_emb], dim=-1)  # (B, T, 384)

        # Reconstruct mel spectrogram
        mel_out = self.decoder(bottleneck)  # (B, T, 80)
        return mel_out
