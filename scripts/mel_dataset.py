import os
import torch
import torchaudio
from torch.utils.data import Dataset
import json

class MelDataset(Dataset):
    def __init__(self, metadata_path, emotion_map=None, sample_rate=16000, n_mels=80):
        """
        metadata_path: path to JSON or CSV with info on audio paths, speaker and emotion
        emotion_map: dict to normalize emotion labels (e.g., {"happiness": "happy"})
        """
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.emotion_map = emotion_map if emotion_map else {}
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(item['path'])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to Mel
        mel = self.mel_transform(waveform).squeeze(0)  # shape: [n_mels, time]

        # Speaker ID and emotion
        speaker = item['speaker']
        emotion = item['emotion'].lower()
        emotion = self.emotion_map.get(emotion, emotion)  # map to unified label

        return mel, speaker, emotion
