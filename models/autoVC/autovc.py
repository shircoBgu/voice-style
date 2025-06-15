import torch
import torch.nn as nn
from models.autoVC.content_encoder import ContentEncoder
from models.autoVC.postnet import Postnet
from models.autoVC.speaker_encoder import SpeakerEncoder
from models.autoVC.decoder import Decoder


class AutoVC(nn.Module):
    def __init__(self, num_emotions,
                 num_speakers,
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

        self.emotion_embedding = nn.Embedding(num_emotions, emotion_emb_dim)
        self.speaker_classifier = nn.Linear(speaker_emb_dim, num_speakers)
        self.postnet = Postnet()
        self.use_postnet = True

    def forward(self, source_mel, target_mel, emotion_label):
        """
        Forward pass of the AutoVC model.

        Args:
            source_mel (Tensor): Tensor of shape (B, T, 80)
                The mel-spectrogram of the source speaker's utterance.
            target_mel (Tensor): Tensor of shape (B, T', 80)
                A reference mel-spectrogram of the target speaker (for identity).
            emotion_label (Tensor): Tensor of shape (B,)
                Categorical emotion label index (e.g., 0 = 'neutral', 1 = 'happy', etc.).

        Returns:
            mel_pred (Tensor): Reconstructed mel-spectrogram of shape (B, T, 80)
                in the target speaker's voice and intended emotional style.
        """

        # 1. Encode source content
        content_emb = self.content_encoder(source_mel)  # (B, T, 128)

        # 2. Encode speaker identity
        speaker_emb = self.speaker_encoder(target_mel)  # (B, 128)
        spk_logits = self.speaker_classifier(speaker_emb)  # (B, num_speakers)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, content_emb.size(1), -1)  # (B, T, 128)

        # 3. Broadcast emotion vector over time
        emotion_vec = self.emotion_embedding(emotion_label)  # (B, 128)
        emotion_vec = emotion_vec.unsqueeze(1).expand(-1, content_emb.size(1), -1)  # (B, T, 128)

        # 4. Fuse all embeddings (C + S + E)
        bottleneck = torch.cat([content_emb, speaker_emb, emotion_vec], dim=-1)  # (B, T, 384)

        # 5. Decode to mel-spectrogram
        mel_out = self.decoder(bottleneck)  # (B, T, 80)
        # print("Decoder raw output std:", mel_out.std().item())

        if self.use_postnet:
            mel_post = mel_out + self.postnet(mel_out)
            # print("Postnet-enhanced output std:", mel_post.std().item())
            mel_pred = mel_post
        else:
            mel_pred = mel_out

        # print("Bottleneck stats:", bottleneck.mean().item(), bottleneck.std().item())
        # print("content_emb std:", content_emb.std().item())
        # print("speaker_emb std:", speaker_emb.std().item())
        # print("emotion_vec std:", emotion_vec.std().item())
        return mel_pred, spk_logits
