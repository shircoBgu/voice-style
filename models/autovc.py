import torch
import torch.nn as nn
from models.content_encoder import ContentEncoder
from models.postnet import Postnet
from models.speaker_encoder import SpeakerEncoder
from models.decoder import Decoder


class AutoVC(nn.Module):
    def __init__(self,
                 content_dim=80,
                 speaker_dim=80,
                 content_emb_dim=128,
                 speaker_emb_dim=128,
                 emotion_emb_dim=128,
                 bottleneck_dim=384,  # C + S + E
                 mel_dim=80):
        super(AutoVC, self).__init__()

        self.content_encoder = ContentEncoder(input_dim=content_dim,
                                              hidden_dim=256,
                                              output_dim=content_emb_dim)

        self.speaker_encoder = SpeakerEncoder(input_dim=speaker_dim,
                                              hidden_dim=512,
                                              output_dim=speaker_emb_dim)

        self.decoder = Decoder(input_dim=bottleneck_dim,
                               hidden_dim=256,
                               output_dim=mel_dim)
        self.postnet = Postnet()
        self.use_postnet = False

    def forward(self, source_mel, target_mel, emotion_embedding):
        """
        source_mel: (B, T, 80) - source audio
        target_mel: (B, T', 80) - reference audio for speaker identity
        emotion_embedding: (B, 128) - vector representing desired emotion
        """

        # 1. Encode source content
        content_emb = self.content_encoder(source_mel)  # (B, T, 128)

        # 2. Encode speaker identity
        speaker_emb = self.speaker_encoder(target_mel)  # (B, 128)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, content_emb.size(1), -1)  # (B, T, 128)

        # 3. Broadcast emotion vector over time
        emotion_emb = emotion_embedding.unsqueeze(1).expand(-1, content_emb.size(1), -1)  # (B, T, 128)

        # 4. Fuse all embeddings (C + S + E)
        bottleneck = torch.cat([content_emb, speaker_emb, emotion_emb], dim=-1)  # (B, T, 384)

        # 5. Decode to mel-spectrogram
        mel_out = self.decoder(bottleneck)  # (B, T, 80)

        if self.use_postnet:
            mel_pred = mel_out + self.postnet(mel_out)  # residual
        else:
            mel_pred = mel_out

        return mel_pred
