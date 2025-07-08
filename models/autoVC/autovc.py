import torch
import torch.nn as nn
from models.autoVC.content_encoder import ContentEncoder
from models.autoVC.postnet import Postnet
from models.autoVC.speaker_encoder import SpeakerEncoder
from models.autoVC.decoder import Decoder
from models.autoVC.pretrained_speaker_encoder import PretrainedSpeakerEncoder


class AutoVC(nn.Module):
    def __init__(self, num_emotions,
                 num_speakers,
                 dim_neck=32,
                 freq=32,
                 speaker_emb_dim=256,
                 emotion_emb_dim=128,
                 mel_dim=80):
        super(AutoVC, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ← hardcoded with fallback

        self.content_encoder = ContentEncoder(speaker_dim=speaker_emb_dim,
                                              dim_neck=dim_neck,
                                              freq=freq)

        # self.speaker_encoder = SpeakerEncoder(input_dim=speaker_dim,
        #                                       hidden_dim=512,
        #                                       output_dim=speaker_emb_dim)

        self.speaker_encoder = PretrainedSpeakerEncoder(ckpt_path="3000000-BL.ckpt", device=device)
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        self.decoder_input_dim = 2 * dim_neck + speaker_emb_dim + emotion_emb_dim

        self.decoder = Decoder(input_dim=self.decoder_input_dim,
                               dim_pre=512,
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
            spk_logits: (B, num_speakers) — for speaker classification loss
        """

        B, T, _ = source_mel.shape

        # 1. Encode src speaker identity
        src_speaker_emb = self.speaker_encoder(source_mel)  # (B, 256)

        # 2. Extract target speaker embedding for decoder + classification
        tgt_speaker_emb = self.speaker_encoder(target_mel)  # (B, 256)
        spk_logits = self.speaker_classifier(tgt_speaker_emb)  # (B, num_speakers)

        # 2. Encode content using src speaker embedding
        codes = self.content_encoder(source_mel, src_speaker_emb)  # list of (B, 64)

        segment_len = T // len(codes)
        content_emb = torch.cat([
            code.unsqueeze(1).expand(-1, segment_len, -1) for code in codes
        ], dim=1)

        # Crop or pad to match T
        if content_emb.size(1) > T:
            content_emb = content_emb[:, :T, :]
        elif content_emb.size(1) < T:
            pad_len = T - content_emb.size(1)
            pad = content_emb[:, -1:, :].expand(-1, pad_len, -1)
            content_emb = torch.cat([content_emb, pad], dim=1)

        # 3. 4. Expand target speaker + emotion embeddings
        speaker_exp = tgt_speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 256)
        emotion_vec = self.emotion_embedding(emotion_label)  # (B, 128)
        emotion_exp = emotion_vec.unsqueeze(1).expand(-1, T, -1)  # (B, T, 128)

        # 4. Fuse all embeddings (C + S + E)
        bottleneck = torch.cat([content_emb, speaker_exp, emotion_exp], dim=-1)  # (B, T, 448)

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
        return mel_pred, mel_out, torch.cat(codes, dim=-1), spk_logits
