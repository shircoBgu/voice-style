import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.autoVC.autovc import AutoVC
from models.emotion_classifier import EmotionClassifier
from scripts.utils.mel_dataset import MelDataset
from scripts.utils.converter import VoiceConverter


def run_sanity_check():
    # Config
    csv_path = "/content/drive/MyDrive/voice_style_project/processed_dataset/iemocap_with_mels_fixedlen_FIXED.csv"
    batch_size = 4
    num_emotions = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load one batch
    dataset = MelDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mels, speaker_ids, emotion_labels = next(iter(dataloader))  # Take first batch
    mels, emotion_labels = mels.to(device), emotion_labels.to(device)

    # Create dummy target mel (just reuse mels for now)
    source_mel = mels
    target_mel = mels

    # Initialize models
    autovc = AutoVC().to(device)
    emotion_cls = EmotionClassifier().to(device)

    # Forward through AutoVC
    with torch.no_grad():
        mel_pred = autovc(source_mel, target_mel, emotion_labels)  # (B, T, 80)

    print(f"✅ AutoVC output shape: {mel_pred.shape}")

    # Forward through Emotion Classifier
    with torch.no_grad():
        logits = emotion_cls(mel_pred)  # (B, num_emotions)
        print(f"✅ EmotionClassifier logits shape: {logits.shape}")

    # Compute dummy loss
    loss = F.cross_entropy(logits, emotion_labels)
    print(f"✅ Dummy cross-entropy loss: {loss.item():.4f}")

    # Check speaker similarity loss
    with torch.no_grad():
        target_spk_emb = autovc.speaker_encoder(target_mel)
        pred_spk_emb = autovc.speaker_encoder(mel_pred)
        cos_sim = F.cosine_similarity(pred_spk_emb, target_spk_emb, dim=-1)
        speaker_loss = 1.0 - cos_sim.mean()
        print(f"✅ Dummy speaker similarity loss: {speaker_loss.item():.4f}")

    converter = VoiceConverter("/content/drive/MyDrive/voice_style_project/voice_style_project/config.json")
    converter.load_hifigan()

    with torch.no_grad():
        audio_waveform = converter.hifigan_model(mel_pred.transpose(1, 2))  # (B, 1, T_wave)
        print(f"✅ HiFi-GAN output shape: {audio_waveform.shape}")

    print("All sanity checks passed!")

if __name__ == "__main__":
    run_sanity_check()
