import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class MelDataset(Dataset):
    def __init__(self, csv_path, speakers_map=None, emotions_map=None, dataset_filter=None):
        # Load full dataset
        full_df = pd.read_csv(csv_path)

        # Add global speaker identifier
        full_df["global_speaker"] = full_df["dataset_id"].astype(str) + "_" + full_df["speaker_id"].astype(str)

        # Build speaker mapping over ALL data
        if speakers_map is None:
            unique_speakers = sorted(full_df["global_speaker"].unique())
            self.speaker2idx = {spk: i for i, spk in enumerate(unique_speakers)}
        else:
            self.speaker2idx = speakers_map

        # Build emotion mapping over ALL data
        if emotions_map is None:
            unique_emotions = sorted(full_df["emotion_label"].unique())
            self.emo2idx = {emo: i for i, emo in enumerate(unique_emotions)}
        else:
            self.emo2idx = emotions_map

        # Filter dataset for training
        if dataset_filter is not None:
            full_df = full_df[full_df["dataset_id"] == dataset_filter].reset_index(drop=True)

        self.df = full_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # phase 1
        # row = self.df.iloc[idx]
        # if not os.path.exists(row['mel_path']):
        #     raise FileNotFoundError(f"Missing mel file: {row['mel_path']}")
        # mel = np.load(row['mel_path'])    # (80, 251)
        # mel = torch.tensor(mel.T, dtype=torch.float32) #(251,80) as train.py expected
        # speaker_id = self.speaker2idx[row['speaker_id']]
        # emotion_label = self.emo2idx[row['emotion_label']]
        # # return mel, speaker_id, emotion_label
        # return mel, mel, emotion_label  # source_mel, target_mel, emotion_label for training phase


        # phase 2
        row = self.df.iloc[idx]
        if not os.path.exists(row['mel_path']):
            raise FileNotFoundError(f"Missing mel file: {row['mel_path']}")
        # Source
        src_mel = np.load(row['mel_path'])
        src_mel = torch.tensor(src_mel.T, dtype=torch.float32)
        src_speaker = row['global_speaker']

        # Target mel (random, different sample)
        while True:
            tgt_idx = random.randint(0, len(self.df) - 1)
            tgt_row = self.df.iloc[tgt_idx]
            if src_speaker != tgt_row['global_speaker']:
                break

        if not os.path.exists(tgt_row['mel_path']):
            raise FileNotFoundError(f"Missing mel file: {tgt_row['mel_path']}")
        # Target
        tgt_mel = np.load(tgt_row['mel_path'])
        tgt_mel = torch.tensor(tgt_mel.T, dtype=torch.float32)

        tgt_emotion = self.emo2idx[tgt_row['emotion_label']]
        speaker_label = self.speaker2idx[tgt_row['global_speaker']]

        return src_mel, tgt_mel, tgt_emotion , speaker_label