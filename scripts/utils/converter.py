import os
import glob
import json
import torch
import torchaudio
import numpy as np
from scipy.io.wavfile import write
from hifigan.models import Generator
from hifigan.env import AttrDict


class VoiceConverter:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emo2idx = None
        self.speaker2idx = None
        self.autovc_model = None
        self.hifigan_model = None

    def load_autovc(self, model_class, checkpoint_path=None):
        if checkpoint_path is None:
            ckpt_dir = self.config["training"].get("checkpoint_dir", "checkpoints")
            candidates = glob.glob(os.path.join(ckpt_dir, "*.pt"))
            if not candidates:
                raise FileNotFoundError("No AutoVC checkpoint found")
            checkpoint_path = sorted(candidates)[-1]

        print(f"Loading AutoVC from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            num_emotions = len(state.get("emo2idx", {}))
            num_speakers = len(state.get("speaker2idx", {}))
            model = model_class(num_emotions=num_emotions, num_speakers=num_speakers).to(self.device)
            model.load_state_dict(state["model_state"])
            model.eval()
            self.emo2idx = state.get("emo2idx", {})
            self.speaker2idx = state.get("speaker2idx", {})
            self.autovc_model = model
        else:
            raise ValueError("Checkpoint missing 'model_state'")

    def load_hifigan(self):
        hifigan_dir = self.config["paths"]["pretrained_hifigan"]
        config_file = os.path.join(hifigan_dir, "config.json")
        checkpoint_file = os.path.join(hifigan_dir, "generator_v1")

        # Ensure config and model file are loaded from the same directory
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Missing config.json in {hifigan_dir}")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Missing generator_v1 in {hifigan_dir}")

        with open(config_file) as f:
            config = AttrDict(json.load(f))

        model = Generator(config).to(self.device)
        ckpt = torch.load(checkpoint_file, map_location=self.device)
        model.load_state_dict(ckpt['generator'] if 'generator' in ckpt else ckpt)

        model.eval()
        model.remove_weight_norm()
        self.hifigan_model = model

    def load_audio_as_mel(self, path, target_len=None):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = self.config['dataset']['sample_rate']
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.config['dataset']['n_mels']
        )
        mel = mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5)).squeeze(0).transpose(0, 1)

        if target_len:
            mel = mel[:target_len] if mel.size(0) > target_len else torch.cat(
                [mel, mel[-1:].repeat(target_len - mel.size(0), 1)]
            )
        return mel.unsqueeze(0).to(self.device)

    def load_mel_from_npy(self, path, target_len=None):
        """
        Load a precomputed mel spectrogram saved as (80, T) .npy file and prepare it for inference.
        Returns: Tensor of shape (1, T, 80)
        """
        mel = np.load(path)  # shape: (80, T)
        mel = torch.tensor(mel.T, dtype=torch.float32)  # Transpose to (T, 80)

        if target_len:
            mel = mel[:target_len] if mel.size(0) > target_len else torch.cat(
                [mel, mel[-1:].repeat(target_len - mel.size(0), 1)]
            )

        return mel.unsqueeze(0).to(self.device)  # shape: (1, T, 80)

    def convert(self, source_path, target_path, emotion_label, output_path, use_npy=False):
        if not all([self.autovc_model, self.hifigan_model]):
            raise RuntimeError("Models not loaded")

        load_mel = self.load_mel_from_npy if use_npy else self.load_audio_as_mel
        source_mel = load_mel(source_path)
        target_mel = load_mel(target_path, target_len=source_mel.shape[1])

        emotion_tensor = torch.tensor([emotion_label], dtype=torch.long).to(self.device)

        with torch.no_grad():
            mel_out = self.autovc_model(source_mel, target_mel, emotion_tensor)
            audio = self.hifigan_model(mel_out.transpose(1, 2)).squeeze().cpu().numpy()
            audio = audio / np.max(np.abs(audio))
            audio = np.int16(audio * 32767)

            write(output_path, self.config["dataset"]["sample_rate"], audio)
            print(f"Audio saved to {output_path}")
