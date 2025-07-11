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

        # Filter out speakers with only one utterance
        counts = full_df["global_speaker"].value_counts()
        valid_speakers = counts[counts > 1].index
        full_df = full_df[full_df["global_speaker"].isin(valid_speakers)]

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
        src_row = self.df.iloc[idx]
        if not os.path.exists(src_row['mel_path']):
            raise FileNotFoundError(f"Missing mel file: {src_row['mel_path']}")
        # Source
        src_mel = np.load(src_row['mel_path'])
        src_mel = torch.tensor(src_mel.T, dtype=torch.float32)
        src_speaker = src_row['global_speaker']
        src_utt = src_row['utterance_id']

        # Get rows for same speaker
        same_speaker_df = self.df[self.df['global_speaker'] == src_speaker]

        # Ensure at least two different utterances exist
        if len(same_speaker_df) < 2:
            raise ValueError(f"Not enough utterances for speaker {src_speaker}")

        # Select a different utterance from the same speaker
        while True:
            tgt_row = same_speaker_df.sample(n=1).iloc[0]
            tgt_utt = tgt_row['utterance_id']
            if tgt_utt != src_utt:
                break

        if not os.path.exists(tgt_row['mel_path']):
            raise FileNotFoundError(f"Missing mel file: {tgt_row['mel_path']}")

        # Target mel (random, same speaker & different utterance)
        tgt_mel = np.load(tgt_row['mel_path'])
        tgt_mel = torch.tensor(tgt_mel.T, dtype=torch.float32)
        speaker_label = self.speaker2idx[src_speaker]  # same speaker label

        tgt_emotion = self.emo2idx[tgt_row['emotion_label']]

        src_wav_path = src_row['wav_path']  # for emotion embedding
        tgt_wav_path = tgt_row['wav_path']  # for emotion embedding

        return src_mel, tgt_mel, tgt_emotion, speaker_label, src_wav_path, tgt_wav_path
